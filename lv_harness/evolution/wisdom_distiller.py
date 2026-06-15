"""
WisdomDistiller: distills strategy-level wisdom from Skills and a large set of Learnings.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from .learning_types import Learning, LearningType
from .skill_types import Skill

logger = logging.getLogger(__name__)


class WisdomDistiller:
    """Distills strategy-level wisdom from Skills and a large set of Learnings.

    WISDOM.md is the Agent's "long-term memory", containing:
    1. Optimal strategies for different video types
    2. Handling experience for different question types
    3. Known systemic weaknesses and ways to avoid them

    Two distillation modes are supported:
    - Statistical mode (default): rule-based aggregation, zero additional cost
    - LLM mode: calls an LLM to produce strategy-level insights, able to see "why it went wrong" in the prompt

    Args:
        wisdom_path: path to the WISDOM.md file
        reflections_dir: directory where reflection reports are stored
        use_llm: whether to use LLM distillation (default False, kept for backward compatibility)
        llm_model: LLM model name, effective when use_llm=True
        llm_max_failures: upper limit on the number of failure cases fed to the LLM, to avoid an overly long prompt
        reflection_llm_max_tokens: maximum output tokens for the LLM reflection. A limit too low will cause the Markdown to be truncated
    """

    def __init__(self, wisdom_path: str, reflections_dir: str = "",
                 use_llm: bool = False,
                 llm_model: str = "gemini-2.5-flash",
                 llm_max_failures: int = 20,
                 reflection_llm_max_tokens: int = 8192,
                 agent_mode: str = "multi_round_search"):
        self.wisdom_path = Path(wisdom_path)
        self.reflections_dir = Path(reflections_dir) if reflections_dir else None
        if self.reflections_dir:
            self.reflections_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_max_failures = llm_max_failures
        self.reflection_llm_max_tokens = reflection_llm_max_tokens
        self.agent_mode = agent_mode
        self._is_control_mode = (agent_mode == "control_api_harness")
        # Incremental reflection state: records the number of learnings at the last "reflection write", to avoid duplicate output
        self._last_reflection_learning_count = 0

    def _retrieval_prefixes_text(self) -> str:
        """Generate the retrieval-prefix description in the prompt based on agent_mode."""
        if self._is_control_mode:
            return (
                "plain text queries for semantic search over the memory bank, "
                "and character ID queries (e.g. 'What is the name of <character_0>') "
                "for resolving character mappings."
            )
        return (
            "`VIDEO:<event_id>:<query>` for fine-grained drilldown, "
            "`NEIGHBOR:<prev|next>:<query>` for adjacent events, and "
            "bare text for top-level semantic search."
        )

    def _reflection_lever_example(self) -> str:
        """Generate the lever example in the reflection prompt based on agent_mode."""
        if self._is_control_mode:
            return "add character ID resolution hint after 3 empty searches"
        return "add NEIGHBOR: hint after 3 empty searches"

    def distill_after_batch(self, batch_results: Dict,
                            skills: List[Skill],
                            learnings: List[Learning]) -> str:
        """After a batch of videos is evaluated, distill strategy-level wisdom.

        Choose the distillation method based on the use_llm switch:
        - False: statistical aggregation
        - True: LLM distillation (falls back to statistical aggregation on failure)
        """
        wisdom = None

        if self.use_llm:
            try:
                wisdom = self._generate_llm_wisdom(batch_results, skills, learnings)
                logger.info(f"[WisdomDistiller] LLM distillation succeeded (model={self.llm_model})")
            except Exception as e:
                logger.warning(
                    f"[WisdomDistiller] LLM distillation failed, falling back to statistical mode: {e}"
                )
                wisdom = None

        if wisdom is None:
            wisdom = self._generate_statistical_wisdom(batch_results, skills, learnings)

        # Save WISDOM.md
        self.wisdom_path.parent.mkdir(parents=True, exist_ok=True)
        self.wisdom_path.write_text(wisdom, encoding="utf-8")
        logger.info(f"WISDOM.md updated: {self.wisdom_path}")

        # Save the reflection report (full mode is used here)
        if self.reflections_dir:
            self._save_reflection(batch_results, skills, learnings, incremental=False)
            self._last_reflection_learning_count = len(learnings)

        return wisdom

    def distill_incremental(self, batch_results: Dict,
                            skills: List[Skill],
                            learnings: List[Learning],
                            batch_idx: int,
                            force: bool = False,
                            min_new_learnings: int = 16) -> None:
        """Incremental reflection: called at the end of each batch, writing only the reflection without updating WISDOM.md.

        Design goal: leave staged per-batch reflection artifacts even if the run is interrupted.

        Args:
            force: force the write (ignore the min_new_learnings threshold).
            min_new_learnings: skip when fewer than this many are added, to avoid frequently writing duplicate content.
        """
        if self.reflections_dir is None:
            return
        new_count = len(learnings) - self._last_reflection_learning_count
        if not force and new_count < min_new_learnings:
            return
        self._save_reflection(
            batch_results, skills, learnings,
            incremental=True, batch_idx=batch_idx,
        )
        self._last_reflection_learning_count = len(learnings)

    def _generate_statistical_wisdom(self, batch_results: Dict,
                                     skills: List[Skill],
                                     learnings: List[Learning]) -> str:
        """Generate a wisdom report based on statistical analysis."""
        total = len(learnings)
        correct = sum(1 for l in learnings if l.is_correct)
        accuracy = correct / total if total > 0 else 0

        # Statistics by error type
        error_dist = Counter(l.learning_type.value for l in learnings if not l.is_correct)

        # Statistics by video
        video_stats = {}
        for l in learnings:
            if l.video_name not in video_stats:
                video_stats[l.video_name] = {"correct": 0, "total": 0}
            video_stats[l.video_name]["total"] += 1
            if l.is_correct:
                video_stats[l.video_name]["correct"] += 1

        # Skill summary: strictly use the downstream metric (the true "effect after being injected")
        skill_lines = []
        verified_good, verified_bad, untested = [], [], []
        for s in skills:
            down_total = s.downstream_success + s.downstream_failure
            if down_total == 0:
                down_rate_str = "N/A"
                note = "not injected yet"
                untested.append(s)
            else:
                down_rate_str = f"{s.downstream_success_rate:.1%}"
                delta = s.downstream_success_rate - accuracy
                if down_total >= 5 and delta >= 0.05:
                    note = f"✅ above baseline (+{delta:+.1%})"
                    verified_good.append((s, delta))
                elif down_total >= 5 and delta <= -0.05:
                    note = f"⚠️ below baseline ({delta:+.1%})"
                    verified_bad.append((s, delta))
                else:
                    note = "insufficient samples / near baseline"

            src_total = s.success_count + s.failure_count
            strategy_cell = (
                s.recommended_search_strategy
                if s.recommended_search_strategy
                else "N/A"
            )
            skill_lines.append(
                f"| {s.name} | {strategy_cell} | "
                f"{down_rate_str} | {down_total} | "
                f"{s.success_rate:.0%} ({src_total}) | {note} |"
            )

        # Verified effective / ineffective comparison section
        verified_good.sort(key=lambda x: x[1], reverse=True)
        verified_bad.sort(key=lambda x: x[1])
        verified_section_lines = ["### 4.1 Verified Effective Skills (injected >=5 times, win rate above baseline by 5%+)"]
        if verified_good:
            for s, delta in verified_good:
                strat_repr = (
                    f"`{s.recommended_search_strategy}`"
                    if s.recommended_search_strategy
                    else "_N/A_"
                )
                verified_section_lines.append(
                    f"- **{s.name}** -> {strat_repr}, "
                    f"downstream {s.downstream_success_rate:.1%} "
                    f"({s.downstream_success}/{s.downstream_success + s.downstream_failure}), "
                    f"delta vs baseline {delta:+.1%}"
                )
        else:
            verified_section_lines.append("- No skill with sufficient samples significantly above baseline yet")

        verified_section_lines.append("")
        verified_section_lines.append("### 4.2 Verified Ineffective Skills (injected >=5 times, win rate below baseline by 5%+)")
        if verified_bad:
            for s, delta in verified_bad:
                strat_repr = (
                    f"`{s.recommended_search_strategy}`"
                    if s.recommended_search_strategy
                    else "_N/A_"
                )
                verified_section_lines.append(
                    f"- **{s.name}** -> {strat_repr}, "
                    f"downstream {s.downstream_success_rate:.1%} "
                    f"({s.downstream_success}/{s.downstream_success + s.downstream_failure}), "
                    f"delta vs baseline {delta:+.1%}. **Recommend disabling via Router or rewriting special instructions.**"
                )
        else:
            verified_section_lines.append("- No skill significantly underperforming baseline yet")

        verified_section = "\n".join(verified_section_lines)

        wisdom = f"""# LV-Harness Agent Long-Term Wisdom

> Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
> Based on {len(video_stats)} videos, {total} questions
> Distillation mode: statistical

## 1. Overall Performance

- **Baseline accuracy**: {accuracy:.1%} ({correct}/{total})
- **Error type distribution**: {dict(error_dist)}

## 2. Skill Library Status (ranked by downstream win rate)

| Skill Name | Strategy | Downstream Win Rate | Injections | Source Signal (samples) | Note |
|------------|----------|---------------------|------------|------------------------|------|
{chr(10).join(skill_lines) if skill_lines else '| (no skills yet) | - | - | - | - | - |'}

> Downstream win rate = proportion of correct answers after the skill was injected. Source signal only reflects the learning cluster that produced this skill, not its real effectiveness.

## 3. Per-Video Performance

| Video | Accuracy | Questions |
|-------|----------|-----------|
{chr(10).join(f"| {v[:30]} | {s['correct']}/{s['total']} ({s['correct']/s['total']:.0%}) | {s['total']} |" for v, s in sorted(video_stats.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1))[:20])}

## 4. Skill Effect Verification (vs baseline {accuracy:.1%})

{verified_section}

## 5. Known Weaknesses

{self._identify_weaknesses(learnings)}
"""
        return wisdom

    # ------------------------------------------------------------------
    # LLM distillation mode
    # ------------------------------------------------------------------
    def _generate_llm_wisdom(self, batch_results: Dict,
                             skills: List[Skill],
                             learnings: List[Learning]) -> str:
        """Produce strategy-level insights via an LLM.

        Uses a two-stage analyze-then-format approach:
        1. Aggregate statistical facts (cheap)
        2. Have the LLM produce the three sections strategy / weaknesses / suggestions based on failure cases + statistical facts
        """
        # Lazy import, to avoid pulling the dependency when LLM mode is not enabled
        from .llm_utils import call_llm

        total = len(learnings)
        correct = sum(1 for l in learnings if l.is_correct)
        accuracy = correct / total if total > 0 else 0

        # Two-dimensional statistics by error type and question_type (if question_type exists)
        error_type_dist = Counter(l.learning_type.value for l in learnings if not l.is_correct)
        qtype_err_dist = Counter()
        for l in learnings:
            qtype = getattr(l, "question_type", None) or "unknown"
            if not l.is_correct:
                qtype_err_dist[qtype] += 1

        # Select typical failure cases to feed the LLM (sorted by confidence descending; high-confidence errors are most valuable)
        failures = [l for l in learnings if not l.is_correct]
        failures.sort(key=lambda l: l.confidence, reverse=True)
        sample_failures = failures[: self.llm_max_failures]

        # Select typical success cases
        successes = [l for l in learnings if l.is_correct]
        sample_successes = successes[: max(3, self.llm_max_failures // 4)]

        # Construct the LLM prompt
        failure_items = []
        for l in sample_failures:
            failure_items.append({
                "question_type": getattr(l, "question_type", "unknown"),
                "learning_type": l.learning_type.value,
                "question": l.question[:200],
                "agent_answer": l.agent_answer[:120],
                "ground_truth": l.ground_truth[:120],
                "confidence": round(l.confidence, 2),
                "num_rounds": l.num_rounds,
                "search_queries": l.search_queries_used[:6],
            })
        success_items = []
        for l in sample_successes:
            success_items.append({
                "question_type": getattr(l, "question_type", "unknown"),
                "question": l.question[:200],
                "num_rounds": l.num_rounds,
                "search_queries": l.search_queries_used[:6],
            })

        # Skill snapshot: expose both the source-signal win rate and the downstream win rate, so the LLM
        # can distinguish "just the cluster win rate that produced this skill" from "the actual effect after being injected"
        skill_snapshot = []
        for s in skills:
            down_total = s.downstream_success + s.downstream_failure
            skill_snapshot.append({
                "name": s.name,
                "question_type": getattr(s, "question_type", "general"),
                "strategy": s.recommended_search_strategy,
                "source_signal_rate": round(s.success_rate, 3),
                "source_samples": s.success_count + s.failure_count,
                "downstream_success_rate": (
                    round(s.downstream_success_rate, 3) if down_total > 0 else None
                ),
                "downstream_samples": down_total,
            })

        # Compute the error rate for each question_type, to help the LLM locate weak qtypes
        qtype_total = Counter()
        qtype_correct = Counter()
        for l in learnings:
            qt = getattr(l, "question_type", None) or "unknown"
            qtype_total[qt] += 1
            if l.is_correct:
                qtype_correct[qt] += 1
        qtype_breakdown = {
            qt: {
                "total": qtype_total[qt],
                "correct": qtype_correct[qt],
                "accuracy": round(qtype_correct[qt] / qtype_total[qt], 3)
                if qtype_total[qt] > 0 else 0.0,
            }
            for qt in qtype_total
        }

        prompt = f"""You are an "experience distillation" researcher analysing a batch of runs from a
video-QA agent. Your job is to turn the raw logs below into **strategy-level, long-term wisdom**
that will be injected into the agent's system prompt for the next run.

Hard requirements for your output:
- Produce **actionable, specific** guidance. Every rule must tell the agent *when* it
  applies and *exactly what* to do (which search prefix to use, how many rounds, what
  evidence to double-check, etc.).
- **Do not merely restate the statistics**. Derive non-obvious patterns from the
  failure/success examples.
- Explicitly exploit the `downstream_success_rate` field: strategies with high downstream
  rates are proven winners; strategies with low downstream rates are proven losers and
  must be flagged as such.
- Prefer concrete prefixes used by this system: {self._retrieval_prefixes_text()}
- Be concise, rigorous, and write in English.

## Batch statistics
- Samples: {total} | Correct: {correct} | Overall accuracy: {accuracy:.1%}
- Error distribution by learning_type: {dict(error_type_dist)}
- Error distribution by question_type: {dict(qtype_err_dist)}
- Per-question-type accuracy: {json.dumps(qtype_breakdown, ensure_ascii=False)}

## Current skill library (with downstream effect)
`downstream_success_rate` = win rate observed *after* this skill was injected.
`null` = the skill has never been injected yet and its real value is unknown.
{json.dumps(skill_snapshot, ensure_ascii=False, indent=2)}

## High-confidence failure samples (sorted by agent confidence, up to {self.llm_max_failures})
Each item shows what the agent searched, what it answered, and what the correct answer was.
{json.dumps(failure_items, ensure_ascii=False, indent=2)}

## Representative successful samples
{json.dumps(success_items, ensure_ascii=False, indent=2)}

## Output format (use this exact Markdown skeleton; no extra prose)
```
## Strategy Insights
- <Insight 1: For question type X, combine searches in order A -> B -> C because ...>
- <Insight 2 ...>
(3-6 bullets, each citing a question_type or scenario)

## Proven-Effective Patterns
- <Patterns that the `downstream_success_rate` evidence supports. Cite the skill or
  search prefix that worked, and under what condition.>
(2-4 bullets; if no skill has reliable positive downstream evidence yet, write
"No skill has accumulated enough positive downstream evidence in this batch." and stop this section.)

## Proven-Ineffective Patterns
- <Patterns that clearly hurt performance, with downstream evidence. Tell the agent
  what to avoid or replace.>
(1-3 bullets; if none, write "No skill has been shown to underperform the baseline yet." and stop.)

## Known Weaknesses
- <Systemic failure mode 1: condition that triggers it, why it happens>
- <Systemic failure mode 2 ...>
(2-5 bullets, grounded in the failure samples)

## Actionable Rules
- <Rule 1: When you see X, do Y before Z. Stop after N rounds if evidence is still thin.>
- <Rule 2 ...>
(3-6 bullets; each must be directly executable at inference time)
```

Output only the Markdown above. Do not add any explanation before or after."""

        response = call_llm(
            model=self.llm_model,
            prompt=prompt,
            timeout=60,
            max_retries=3,
        )

        if not response or not response.strip():
            raise RuntimeError("LLM returned empty response")

        # Assemble the final WISDOM.md: statistical summary + LLM insights
        header = f"""# LV-Harness Agent Long-Term Wisdom

> Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
> Distilled from {total} questions | Mode: llm ({self.llm_model})
> Baseline accuracy: {accuracy:.1%} ({correct}/{total})

"""
        return header + response.strip() + "\n"

    def _identify_weaknesses(self, learnings: List[Learning]) -> str:
        """Identify systemic weaknesses from accumulated learnings."""
        weaknesses = []

        # High-confidence errors
        high_conf_errors = [l for l in learnings if not l.is_correct and l.confidence >= 0.8]
        if high_conf_errors:
            weaknesses.append(
                f"- **High-confidence errors**: {len(high_conf_errors)} cases. "
                f"The agent was confident but wrong. Confidence calibration needed."
            )

        # Multi-round search failures
        search_failures = [l for l in learnings if not l.is_correct and l.num_rounds >= 4]
        if search_failures:
            weaknesses.append(
                f"- **Search failures**: {len(search_failures)} cases. "
                f"Multiple retrieval rounds still failed to find the answer. Search strategy needs improvement."
            )

        return "\n".join(weaknesses) if weaknesses else "- No systemic weaknesses identified yet"

    def _save_reflection(self, batch_results: Dict,
                         skills: List[Skill],
                         learnings: List[Learning],
                         incremental: bool = False,
                         batch_idx: Optional[int] = None) -> None:
        """Save the reflection report (by default using the LLM to generate genuinely useful content). Falls back to a statistical template on failure.

        Args:
            incremental: incremental mode (triggered by on_batch_complete). When True, the filename uses batch_idx;
                        when False, it uses a timestamp (at harness end).
        """
        if self.reflections_dir is None:
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if incremental and batch_idx is not None:
            filename = f"reflection_batch_{batch_idx:04d}_{timestamp}.md"
        else:
            filename = f"reflection_final_{timestamp}.md"
        filepath = self.reflections_dir / filename

        content = None
        if self.use_llm:
            try:
                content = self._generate_llm_reflection(
                    batch_results, skills, learnings,
                    incremental=incremental, batch_idx=batch_idx,
                )
            except Exception as exc:
                logger.warning(
                    f"[WisdomDistiller] LLM reflection generation failed, falling back to the statistical template: {exc}"
                )
                content = None

        if content is None:
            content = self._statistical_reflection(
                batch_results, skills, learnings,
                incremental=incremental, batch_idx=batch_idx,
            )

        filepath.write_text(content, encoding="utf-8")
        logger.info(
            f"[WisdomDistiller] reflection report saved: {filepath.name} "
            f"(mode={'LLM' if self.use_llm and 'LLM' not in filename else 'statistical'}, "
            f"learnings={len(learnings)})"
        )

    def _statistical_reflection(self, batch_results: Dict,
                                skills: List[Skill],
                                learnings: List[Learning],
                                incremental: bool,
                                batch_idx: Optional[int]) -> str:
        """The statistical-mode reflection (as the LLM fallback).

        Relative to the old version that only wrote 5 failures, this adds:
          - qtype-level cumulative accuracy
          - an overview of the current pros and cons of skill downstream
          - the TOP-5 high-confidence failure samples
          - the count of samples with 0 retrieval but correct/incorrect answers (a counter-indicator)
        """
        from collections import Counter
        total = len(learnings)
        correct = sum(1 for l in learnings if l.is_correct)
        acc = correct / total if total else 0.0

        qt_total: Counter = Counter()
        qt_correct: Counter = Counter()
        for l in learnings:
            qt = getattr(l, "question_type", None) or "unknown"
            qt_total[qt] += 1
            if l.is_correct:
                qt_correct[qt] += 1
        qt_lines = []
        for qt in sorted(qt_total, key=lambda x: -qt_total[x]):
            qt_lines.append(
                f"| {qt} | {qt_correct[qt]}/{qt_total[qt]} "
                f"({qt_correct[qt] / qt_total[qt]:.0%}) |"
            )

        skill_lines = []
        for s in skills:
            dtot = s.downstream_success + s.downstream_failure
            if dtot == 0:
                skill_lines.append(f"- **{s.name}**: never injected")
            else:
                skill_lines.append(
                    f"- **{s.name}**: downstream "
                    f"{s.downstream_success_rate:.0%} ({s.downstream_success}/{dtot})"
                )

        hc_failures = sorted(
            (l for l in learnings if not l.is_correct),
            key=lambda l: l.confidence,
            reverse=True,
        )[:5]
        fail_blocks = []
        for l in hc_failures:
            fail_blocks.append(
                f"### {l.learning_id}\n"
                f"- **question** ({getattr(l, 'question_type', 'unknown')}): "
                f"{(l.question or '')[:160]}\n"
                f"- **agent answer** (conf={l.confidence:.2f}, rounds={l.num_rounds}): "
                f"{(l.agent_answer or '')[:160]}\n"
                f"- **ground truth**: {(l.ground_truth or '')[:160]}\n"
                f"- **search queries**: {(l.search_queries_used or [])[:6]}\n"
            )

        zero_retrieval_total = sum(
            1 for l in learnings if not (l.search_queries_used or [])
        )
        zero_retrieval_correct = sum(
            1 for l in learnings
            if not (l.search_queries_used or []) and l.is_correct
        )

        header = (
            f"# Batch Reflection (statistical), batch_idx={batch_idx}"
            if incremental else "# Final Reflection (statistical)"
        )
        return f"""{header}

> learnings={total} | correct={correct} | accuracy={acc:.1%}

## Per-qtype accuracy

| question_type | correct/total |
|---|---|
{chr(10).join(qt_lines) if qt_lines else '| - | - |'}

## Skill downstream snapshot

{chr(10).join(skill_lines) if skill_lines else '- (no skills yet)'}

## Zero-retrieval questions (no search issued at all)

- Total: {zero_retrieval_total}
- Correct: {zero_retrieval_correct}

## Top-5 high-confidence failures

{chr(10).join(fail_blocks) if fail_blocks else '_no failure samples_'}
"""

    def _generate_llm_reflection(self, batch_results: Dict,
                                 skills: List[Skill],
                                 learnings: List[Learning],
                                 incremental: bool,
                                 batch_idx: Optional[int]) -> str:
        """Use an LLM to generate a "useful per-batch reflection". Unlike WISDOM.md, it focuses on
        whether per-batch patterns were confirmed or not, and which strategies should be revised recently.
        """
        from .llm_utils import call_llm
        from collections import Counter

        total = len(learnings)
        correct = sum(1 for l in learnings if l.is_correct)
        acc = correct / total if total else 0.0

        # qtype accuracy
        qt_total: Counter = Counter()
        qt_correct: Counter = Counter()
        for l in learnings:
            qt = getattr(l, "question_type", None) or "unknown"
            qt_total[qt] += 1
            if l.is_correct:
                qt_correct[qt] += 1
        qtype_breakdown = {
            qt: {
                "total": qt_total[qt],
                "correct": qt_correct[qt],
                "accuracy": round(qt_correct[qt] / qt_total[qt], 3)
                if qt_total[qt] else 0.0,
            }
            for qt in qt_total
        }

        # skill snapshot
        skill_snapshot = []
        for s in skills:
            dtot = s.downstream_success + s.downstream_failure
            skill_snapshot.append({
                "name": s.name,
                "question_type": getattr(s, "question_type", "general"),
                "downstream_success_rate": (
                    round(s.downstream_success_rate, 3) if dtot > 0 else None
                ),
                "downstream_samples": dtot,
            })

        # typical failures
        failures = sorted(
            (l for l in learnings if not l.is_correct),
            key=lambda l: l.confidence,
            reverse=True,
        )[: self.llm_max_failures]
        failure_items = []
        for l in failures:
            failure_items.append({
                "question_type": getattr(l, "question_type", "unknown"),
                "question": (l.question or "")[:200],
                "agent_answer": (l.agent_answer or "")[:120],
                "ground_truth": (l.ground_truth or "")[:120],
                "confidence": round(l.confidence, 2),
                "num_rounds": l.num_rounds,
                "search_queries": (l.search_queries_used or [])[:6],
            })

        # zero-retrieval signal (the "answered without retrieval" characteristic)
        zero_retrieval_total = sum(
            1 for l in learnings if not (l.search_queries_used or [])
        )
        zero_retrieval_correct = sum(
            1 for l in learnings
            if not (l.search_queries_used or []) and l.is_correct
        )

        title = (
            f"Batch {batch_idx} Reflection" if incremental
            else "Final Reflection"
        )
        prompt = f"""You are writing a **short, actionable reflection** for a video-QA agent\u2019s
self-improvement loop. Your readers are (a) engineers who will tune prompts and routing,
(b) the agent itself on the next run. Your job is NOT to restate the statistics below,
but to derive **non-obvious, specific patterns** and **next-run levers**.

## Basic stats (this batch)
- title: {title}
- samples: {total} | correct: {correct} | accuracy: {acc:.1%}
- qtype accuracy: {json.dumps(qtype_breakdown, ensure_ascii=False)}
- zero-retrieval questions: total={zero_retrieval_total}, correct={zero_retrieval_correct}
  (these are cases where the agent answered without issuing any search \u2014 usually
  hallucination if wrong, or trivial if correct)

## Skills currently known (with downstream effect)
`downstream_success_rate` = win rate *after* this skill was injected. `null` = never injected.
{json.dumps(skill_snapshot, ensure_ascii=False, indent=2)}

## High-confidence failures (agent was sure but wrong)
{json.dumps(failure_items, ensure_ascii=False, indent=2)}

## Output format (Markdown only, strictly follow the sections below)
```
# {title}

## Key Patterns (3-5 bullets)
- Each bullet cites a concrete qtype or failure mode observed above and explains *why* it fails.

## What Worked / Didn\u2019t Work This Batch
- Back each claim with downstream_success_rate or specific case IDs.

## Concrete Next-Run Levers (3-5 bullets)
- Each bullet is an actionable change: e.g. \"raise route_threshold for skill X\", \"{self._reflection_lever_example()}\", \"disable keyword:<name> trigger because it\u2019s an entity not a pattern\".
- If a skill\u2019s downstream is clearly below baseline, say so explicitly.

## Open Questions (0-3 bullets, optional)
- Issues needing human review (e.g. \"bedroom_01_Q05 always returns no results \u2014 is EventGraph miss or query phrasing?\").
```

Be concise. Write in English. No fluff, no generic advice."""

        response = call_llm(
            model=self.llm_model,
            prompt=prompt,
            timeout=60,
            max_retries=3,
            max_tokens=self.reflection_llm_max_tokens,
        )
        if not response or not response.strip():
            raise RuntimeError("LLM returned empty reflection")
        response = response.strip()
        if self._looks_truncated_reflection(response):
            logger.warning(
                "[WisdomDistiller] the reflection may have been truncated, retrying with a larger max_tokens "
                f"({self.reflection_llm_max_tokens} -> {self.reflection_llm_max_tokens * 2})"
            )
            response = call_llm(
                model=self.llm_model,
                prompt=prompt + "\n\nImportant: produce the complete Markdown with all required sections. Do not stop mid-sentence.",
                timeout=90,
                max_retries=2,
                max_tokens=self.reflection_llm_max_tokens * 2,
            ).strip()
            if not response:
                raise RuntimeError("LLM returned empty reflection after retry")
        header = (
            f"> Auto-generated by WisdomDistiller (LLM={self.llm_model}) | "
            f"learnings={total} | accuracy={acc:.1%}\n\n"
        )
        return header + response + "\n"

    @staticmethod
    def _looks_truncated_reflection(text: str) -> bool:
        """Roughly judge whether the reflection appears to have been truncated by max_tokens.

        Mainly covers two cases:
        1. A required section is missing.
        2. The end stops in an obviously unfinished state, such as a comma, conjunction, or unclosed parenthesis.
        """
        stripped = (text or "").strip()
        if not stripped:
            return True

        required_sections = [
            "## Key Patterns",
            "## What Worked / Didn",
            "## Concrete Next-Run Levers",
        ]
        if any(section not in stripped for section in required_sections):
            return True

        last_line = stripped.splitlines()[-1].strip()
        if not last_line:
            return True
        safe_endings = (".", "!", "?", ")", "]", "`")
        if last_line.endswith(safe_endings):
            return False
        risky_endings = (",", ":", ";", "-", "—", "(", "[", "`")
        if last_line.endswith(risky_endings):
            return True
        tail_words = last_line.lower().split()
        if tail_words and tail_words[-1] in {"and", "or", "but", "because", "with", "to", "on", "for", "of", "the", "a", "an"}:
            return True
        return False

    # ==================================================================
    # P1: decomposition strategy guidance (aggregated from DECOMPOSE_WIN / DECOMPOSE_FAIL / SUBTASK_STALL)
    # ==================================================================
    def generate_decompose_guidance(self, learnings: List[Learning], question=None) -> str:
        """Generate dynamic decomposition strategy guidance usable for the current question from historical experience.

        Design principles:
        - Use structured Ledger learnings rather than parsing Markdown text.
        - Preferentially match the tags of the current question; when there is no current question, fall back to global statistics.
        - Address both P1 goals: reduce ineffective decomposition, and mitigate cases that should be decomposed but are not.
        """
        from collections import defaultdict

        try:
            from .learning_capture import infer_question_type_tags
        except Exception:
            infer_question_type_tags = None

        current_tags = []
        current_question = ""
        if question is not None:
            current_question = getattr(question, "question", "") or ""
            if infer_question_type_tags is not None:
                try:
                    current_tags = infer_question_type_tags(question)
                except Exception:
                    current_tags = []
        current_tag_set = set(current_tags)

        def _learning_tags(l: Learning) -> List[str]:
            tags = getattr(l, "question_type_tags", None) or []
            if tags:
                return list(tags)
            qt = getattr(l, "question_type", "") or "general"
            return [qt]

        def _matches_current(l: Learning) -> bool:
            if not current_tag_set:
                return True
            return bool(current_tag_set & set(_learning_tags(l)))

        # Aggregate by tag, to avoid diluting the signal of multi-label samples by using only the primary question_type.
        stats: dict = defaultdict(lambda: {
            "win": 0,
            "fail": 0,
            "miss": 0,
            "stall": 0,
            "resolved_sum": 0,
            "subtask_sum": 0,
            "abandoned_sum": 0,
            "examples_win": [],
            "examples_fail": [],
            "stall_questions": [],
            "miss_questions": [],
        })

        for l in learnings:
            if not _matches_current(l):
                continue
            lt = l.learning_type
            if lt not in {
                LearningType.DECOMPOSE_WIN,
                LearningType.DECOMPOSE_FAIL,
                LearningType.DECOMPOSE_MISS,
                LearningType.SUBTASK_STALL,
            }:
                continue
            for tag in _learning_tags(l):
                bucket = stats[tag]
                if lt == LearningType.DECOMPOSE_WIN:
                    bucket["win"] += 1
                    bucket["examples_win"].append((l.question or "")[:120])
                elif lt == LearningType.DECOMPOSE_FAIL:
                    bucket["fail"] += 1
                    bucket["examples_fail"].append((l.question or "")[:120])
                elif lt == LearningType.DECOMPOSE_MISS:
                    bucket["miss"] += 1
                    bucket["miss_questions"].append((l.question or "")[:120])
                elif lt == LearningType.SUBTASK_STALL:
                    bucket["stall"] += 1
                    stall_q = getattr(l, "subtask_question", "") or ""
                    if stall_q:
                        bucket["stall_questions"].append(stall_q[:100])
                bucket["resolved_sum"] += int(getattr(l, "resolved_count", 0) or 0)
                bucket["subtask_sum"] += int(getattr(l, "subtask_count", 0) or 0)
                bucket["abandoned_sum"] += int(getattr(l, "abandoned_count", 0) or 0)

        if not stats:
            return self._cold_start_decompose_guidance(current_question, current_tags)

        lines = ["Decomposition guidance (dynamic, from Ledger learnings):"]
        useful_lines = 0
        target_tags = current_tags or sorted(stats.keys())
        for tag in target_tags:
            if tag not in stats:
                continue
            s = stats[tag]
            total_decomposed = s["win"] + s["fail"]
            total_signal = total_decomposed + s["miss"] + s["stall"]
            if total_signal < 2:
                continue

            win_rate = s["win"] / total_decomposed if total_decomposed else 0.0
            avg_resolved = (
                s["resolved_sum"] / s["subtask_sum"]
                if s["subtask_sum"] else 0.0
            )
            useful_lines += 1

            if s["miss"] >= max(1, total_decomposed):
                lines.append(
                    f"- {tag}: DO NOT keep atomic by default. Historical misses={s['miss']} "
                    f"show that under-decomposition caused wrong answers; require at least 2 subtasks "
                    f"when the question asks for multiple facts, people, time steps, or evidence sources."
                )
            elif total_decomposed >= 2 and win_rate >= 0.55:
                lines.append(
                    f"- {tag}: DECOMPOSE ACTIVELY ({s['win']}/{total_decomposed} wins, "
                    f"avg_resolved={avg_resolved:.0%}). Use 2-4 focused subtasks with explicit evidence targets."
                )
            elif total_decomposed >= 2 and win_rate <= 0.30 and s["miss"] == 0:
                lines.append(
                    f"- {tag}: AVOID OVER-DECOMPOSITION ({s['win']}/{total_decomposed} wins). "
                    f"Prefer 1 concise subtask unless the current question contains clear multi-hop/multi-detail cues."
                )
            else:
                lines.append(
                    f"- {tag}: USE A SMALL SPLIT ({s['win']}/{max(total_decomposed, 1)} decomposed wins, "
                    f"misses={s['miss']}). Start with 2 subtasks max and make each independently searchable."
                )

            if s["abandoned_sum"] or s["stall"]:
                lines.append(
                    f"  - Guardrail: avoid vague subtasks; past runs had stalls={s['stall']} "
                    f"and abandoned_subtasks={s['abandoned_sum']}."
                )
            for stall_q in s["stall_questions"][:2]:
                lines.append(f"  - Avoid stalled subtask pattern: {stall_q}")
            for miss_q in s["miss_questions"][:1]:
                lines.append(f"  - Under-decomposition example to avoid: {miss_q}")

        if useful_lines == 0:
            return self._cold_start_decompose_guidance(current_question, current_tags)

        lines.append(
            "- Output guardrail: if you create only one subtask for a multi-signal question, "
            "the subtask must explicitly cover every required entity/time/fact; otherwise split it."
        )
        return "\n".join(lines)

    @staticmethod
    def _cold_start_decompose_guidance(question_text: str, question_tags: List[str]) -> str:
        """Lightweight dynamic guidance when there are no historical samples, avoiding hardcoding a large decision tree into the static prompt."""
        q = (question_text or "").strip().lower()
        tags = set(question_tags or [])
        multi_tags = {
            "multi_hop_reasoning",
            "multi_detail_reasoning",
            "cross_modal_reasoning",
            "person_understanding",
            "temporal_order",
            "comparison",
            "counting",
            "reason_purpose",
        }
        multi_patterns = [
            " and ", " both ", " each ", " respectively ",
            "before", "after", "then", "first", "last", "why", "how many",
            "number of", "count", "compare", "different", "same",
        ]
        if not ((tags & multi_tags) or any(p in q for p in multi_patterns)):
            return ""
        return (
            "Decomposition guidance (dynamic cold-start guardrail):\n"
            "- This question has multi-step or multi-detail cues. Do not skip decomposition just because it is short.\n"
            "- Create 2-3 independently searchable subtasks that cover the required entities, time steps, or evidence sources.\n"
            "- Keep the split minimal; avoid vague subtasks that cannot be answered by retrieval."
        )

    def load_wisdom(self) -> str:
        """Load the current WISDOM.md content."""
        if self.wisdom_path.exists():
            return self.wisdom_path.read_text(encoding="utf-8")
        return ""
