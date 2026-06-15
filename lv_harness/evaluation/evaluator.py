"""
StreamingEvaluator: a streaming evaluator that collects and computes multi-dimensional metrics over the timeline.

Reuses the eval_answer logic from control_api_high_system.py.
"""
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..data.types import TemporalQuestion, AgentAnswer, EvalRecord

logger = logging.getLogger(__name__)


class StreamingEvaluator:
    """Streaming evaluator.

    Args:
        metrics: list of active metrics
        eval_model: name of the model used for evaluation
        api_config_path: path to the API config file
        string_match_first: whether to prefer string matching
    """

    def __init__(self, metrics: List[str] = None,
                 eval_model: str = "gemini-2.5-flash",
                 api_config_path: str = "configs/api_config.json",
                 string_match_first: bool = True,
                 enable_thinking: bool = False):
        self.active_metrics = metrics or ["accuracy"]
        self.eval_model = eval_model
        self.string_match_first = string_match_first
        self._enable_thinking = enable_thinking
        self.records: List[EvalRecord] = []

        # Initialize the API client used for evaluation
        self._eval_client = None
        self._api_config_path = api_config_path
        self._init_eval_client()

    def _init_eval_client(self):
        """Initialize the API client used for evaluation."""
        try:
            import openai
            with open(self._api_config_path) as f:
                api_cfg = json.load(f)
            if self.eval_model in api_cfg:
                self._eval_client = openai.OpenAI(
                    base_url=api_cfg[self.eval_model].get("base_url", None),
                    api_key=api_cfg[self.eval_model]["api_key"],
                )
        except Exception as e:
            logger.warning(f"Evaluation API client init failed: {e}; will use string matching only")

    def record(self, question: TemporalQuestion, answer: AgentAnswer,
               clip_id: int, memory_stats: Dict) -> EvalRecord:
        """Record one question-answer result."""
        is_correct = self._evaluate_correctness(question, answer)

        record = EvalRecord(
            question_id=question.question_id,
            clip_id=clip_id,
            answer=answer.content,
            ground_truth=question.answer,
            is_correct=is_correct,
            confidence=answer.confidence,
            num_rounds=answer.num_rounds,
            tokens_used=answer.tokens_used,
            memory_stats=memory_stats,
            timestamp=time.time(),
            question=question.question,
            category=question.category,
            video_name=question.video_name,
            search_queries=answer.search_queries,
        )
        self.records.append(record)
        return record

    def _evaluate_correctness(self, question: TemporalQuestion,
                              answer: AgentAnswer) -> bool:
        """Evaluate answer correctness. Reuses the logic from control_api_high_system.py."""
        predict = answer.content
        ground_truth = question.answer

        if not predict or not predict.strip():
            return False

        # String matching
        str_result = self._string_match_eval(predict, ground_truth)
        if str_result is True:
            logger.debug(f"[StringMatch] CORRECT")
            return True

        # API evaluation
        if self._eval_client and str_result is not True:
            try:
                return self._api_eval(question.question, predict, ground_truth)
            except Exception as e:
                logger.warning(f"API evaluation failed: {e}")
                if str_result is not None:
                    return str_result
                return False

        return str_result if str_result is not None else False

    @staticmethod
    def _extract_option_letter(text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        m = re.match(r"^\s*([A-Da-d])\b", text.strip())
        if m:
            return m.group(1).upper()
        return None

    def _string_match_eval(self, predict: str, ground_truth: str):
        """String-matching evaluation. Returns True/False/None."""
        gt_letter = self._extract_option_letter(ground_truth)
        if not gt_letter:
            return None

        pred_text = predict.strip()
        pred_letter = self._extract_option_letter(pred_text)
        if pred_letter:
            all_letters = re.findall(r'\b([A-Da-d])\.\s', pred_text)
            if len(all_letters) == 1:
                return pred_letter == gt_letter
            elif len(all_letters) > 1:
                return None

        gt_content = ground_truth.strip()
        gt_content_only = re.sub(r'^[A-Da-d]\.\s*', '', gt_content).strip().rstrip('.')
        if gt_content_only and len(gt_content_only) > 3:
            if gt_content_only.lower() in pred_text.lower():
                return True

        if pred_text.upper() in ('A', 'B', 'C', 'D'):
            return pred_text.upper() == gt_letter

        return None

    def _api_eval(self, question: str, predict: str, ground_truth: str) -> bool:
        """Evaluate answer correctness using the API. Aligns with the prompt in control_api_high_system_v2.py."""
        eval_prompt = (
            "You are provided with a question, a ground truth answer, and an answer "
            "from an agent model. Your task is to determine whether the ground truth "
            "answer can be logically inferred from the agent's answer, in the context "
            "of the question.\n\n"
            "Do not directly compare the surface forms of the agent answer and the "
            "ground truth answer. Instead, assess whether the meaning expressed by the "
            "agent answer supports or implies the ground truth answer. If the ground "
            "truth can be reasonably derived from the agent answer, return \"Yes\". "
            "If it cannot, return \"No\".\n\n"
            "Important notes:\n"
            "\t• Do not require exact wording or matching structure.\n"
            "\t• Semantic inference is sufficient, as long as the agent answer entails "
            "or implies the meaning of the ground truth answer, given the question.\n"
            "\t• Only return \"Yes\" or \"No\", with no additional explanation or formatting.\n\n"
            "Input fields:\n"
            "\t• question: the question asked\n"
            "\t• ground_truth_answer: the correct answer\n"
            "\t• agent_answer: the model's answer to be evaluated\n\n"
            "Now evaluate the following input:\n\n"
            "Input:\n"
            f"\t• question: {question}\n"
            f"\t• ground_truth_answer: {ground_truth}\n"
            f"\t• agent_answer: {predict}\n\n"
            "Output ('Yes' or 'No'):"
        )
        messages = [{"role": "user", "content": eval_prompt}]

        # DeepSeek-V4 family: control thinking mode via self._enable_thinking (off by default)
        extra_kwargs = {}
        if "deepseek-v4" in self.eval_model:
            extra_kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}

        for retry in range(10):
            try:
                resp = self._eval_client.chat.completions.create(
                    model=self.eval_model, messages=messages,
                    temperature=0, timeout=30, max_tokens=2048,
                    top_p=0.95,
                    **extra_kwargs,
                )
                result = resp.choices[0].message.content.lower()
                return "yes" in result
            except Exception as e:
                time.sleep(min(20, 2 * (retry + 1)))
                logger.warning(f"[Eval Retry {retry+1}] {e}")
        return False

    def compute_all(self) -> Dict[str, Any]:
        """Compute all active metrics."""
        results = {}
        for metric_name in self.active_metrics:
            results[metric_name] = self._compute_metric(metric_name)
        return results

    def _compute_metric(self, name: str) -> Any:
        """Compute a single metric."""
        if not self.records:
            return {}

        if name == "accuracy":
            correct = sum(1 for r in self.records if r.is_correct)
            total = len(self.records)
            return {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0}

        elif name == "accuracy_at_t":
            by_clip = defaultdict(list)
            for r in self.records:
                by_clip[r.clip_id].append(r.is_correct)
            return {f"clip_{k}": sum(v) / len(v) for k, v in sorted(by_clip.items())}

        elif name == "accuracy_by_category":
            by_cat = defaultdict(list)
            for r in self.records:
                cat = r.category or "unknown"
                by_cat[cat].append(r.is_correct)
            return {k: sum(v) / len(v) for k, v in sorted(by_cat.items())}

        elif name == "answer_latency":
            latencies = []
            for r in self.records:
                latencies.append(r.num_rounds)
            import numpy as np
            return {
                "mean_rounds": float(np.mean(latencies)) if latencies else 0,
                "median_rounds": float(np.median(latencies)) if latencies else 0,
            }

        elif name == "token_cost":
            total_tokens = sum(r.tokens_used for r in self.records)
            return {"total_tokens": total_tokens, "avg_tokens": total_tokens / len(self.records)}

        else:
            logger.warning(f"Unknown metric: {name}")
            return {}

    def summary(self) -> str:
        """Produce an evaluation summary."""
        results = self.compute_all()
        lines = ["=" * 60, "LV-Harness Evaluation Results", "=" * 60]
        for metric, value in results.items():
            if isinstance(value, dict):
                lines.append(f"\n[{metric}]")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {metric}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)
