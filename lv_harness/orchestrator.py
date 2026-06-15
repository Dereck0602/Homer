"""
HarnessOrchestrator: the core scheduler of LV-Harness.

Responsibilities:
1. Initialize all components based on the YAML configuration
2. Drive timeline advancement (a two-level loop: per video, then per clip)
3. Coordinate memory ingestion, QA scheduling, and evaluation recording
4. Manage lifecycle hooks
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from .config import load_config
from .hooks import HookSystem, LoggingHook, ProgressHook
from .reasoning.verification_hooks import (
    AnswerVerificationHook, SearchResultVerificationHook, RuntimeHealthHook
)
from .data.types import TemporalQuestion, AgentAnswer, EvalRecord
from .data.dataset import TemporalQADataset
from .memory.base import MemoryStrategy
from .memory.snapshot import SnapshotManager
from .reasoning.base import ReasoningAgent
from .reasoning.answer_policy import AnswerPolicy, AlwaysAnswer, ConfidentAnswer, DeferredAnswer
from .reasoning.task_ledger import strip_ledger_ops_from_text
from .evaluation.evaluator import StreamingEvaluator

# Self-evolution components (imported on demand)
from .evolution.learning_capture import LearningCapture
from .evolution.skill_promoter import SkillPromoter
from .evolution.skill_router import SkillRouter
from .evolution.wisdom_distiller import WisdomDistiller

logger = logging.getLogger(__name__)


class HarnessOrchestrator:
    """The core scheduler of LV-Harness.

    Supports two usage modes:
    1. Launch from a YAML configuration file
    2. Assemble components programmatically

    Args:
        config_path: path to the YAML configuration file
        overrides: command-line override parameters
    """

    def __init__(self, config_path: str = None, overrides: Dict[str, Any] = None):
        self.config = load_config(config_path, overrides)

        # Initialize components for each layer
        self.qa_dataset = self._build_qa_dataset()
        self.memory = self._build_memory()
        self.snapshot_mgr = self._build_snapshot_manager()
        self.agent = self._build_agent()
        self.answer_policy = self._build_answer_policy()
        self.evaluator = self._build_evaluator()
        self.hooks = HookSystem()

        # Register default hooks
        self.hooks.register_hook_object(LoggingHook())

        # Register deterministic verification hooks (Harness Engineering: the Verify pillar)
        self.hooks.register_hook_object(AnswerVerificationHook())
        self.hooks.register_hook_object(SearchResultVerificationHook())
        self.hooks.register_hook_object(RuntimeHealthHook())

        # Queue of deferred questions
        self.deferred_questions: List[Tuple[TemporalQuestion, int]] = []

        # Output configuration
        self._output_dir = self.config.get("output", {}).get("dir", "data/results")
        os.makedirs(self._output_dir, exist_ok=True)

        # Trajectory saving configuration
        self._save_trajectory = self.config.get("output", {}).get("save_trajectory", False)
        self._trajectory_dir = None
        if self._save_trajectory:
            self._trajectory_dir = os.path.join(self._output_dir, "trajectories")
            os.makedirs(self._trajectory_dir, exist_ok=True)

        # Self-evolution components (controlled via the --evolution argument)
        self._evolution_enabled = self.config.get("evolution", {}).get("enabled", False)
        # Read-only mode: load existing skills for reasoning injection, but do not write any learning/skill/wisdom
        self._evolution_readonly = self.config.get("evolution", {}).get("readonly", False)
        self._learning_capture = None
        self._skill_promoter = None
        self._skill_router = None
        self._wisdom_distiller = None
        # The skill id injected for the current question, used to write back the downstream win rate after answering
        self._current_injected_skill_id = None
        # The deferred skill ids actually triggered for the current question. A negative-sample skill is only counted toward usage/downstream after its hook fires.
        self._current_fired_skill_ids = []
        if self._evolution_enabled:
            self._build_evolution_components()

    @classmethod
    def from_components(cls, qa_dataset: TemporalQADataset,
                        memory: MemoryStrategy,
                        agent: ReasoningAgent,
                        evaluator: StreamingEvaluator,
                        answer_policy: AnswerPolicy = None,
                        snapshot_mgr: SnapshotManager = None,
                        enable_evolution: bool = False) -> "HarnessOrchestrator":
        """Assemble components programmatically."""
        instance = cls.__new__(cls)
        instance.config = {"evolution": {"enabled": enable_evolution}}
        instance.qa_dataset = qa_dataset
        instance.memory = memory
        instance.snapshot_mgr = snapshot_mgr or SnapshotManager(interval=0)
        instance.agent = agent
        instance.answer_policy = answer_policy or AlwaysAnswer()
        instance.evaluator = evaluator
        instance.hooks = HookSystem()
        instance.hooks.register_hook_object(LoggingHook())
        instance.deferred_questions = []
        instance._output_dir = "data/results"
        instance._evolution_enabled = enable_evolution
        instance._evolution_readonly = False
        instance._learning_capture = None
        instance._skill_promoter = None
        instance._skill_router = None
        instance._wisdom_distiller = None
        instance._current_injected_skill_id = None
        instance._current_fired_skill_ids = []
        if enable_evolution:
            instance._build_evolution_components()
        return instance

    def run(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline.

        Processing grouped by video:
        1. For each video, load the pre-built memory (if any)
        2. Answer all questions for that video
        3. Record the evaluation results
        """
        self.hooks.trigger("on_harness_start", config=self.config)

        # Processing grouped by video
        videos = self._group_questions_by_video()
        total_questions = len(self.qa_dataset.questions)

        logger.info(f"{len(videos)} videos, {total_questions} questions in total")

        output_path = self._get_output_path()
        done_ids = self._load_done_ids(output_path)

        with open(output_path, "a", encoding="utf-8") as f_out:
            for video_idx, (video_name, questions) in enumerate(videos.items()):
                # Filter out already completed questions
                pending = [q for q in questions if q.question_id not in done_ids]
                if not pending:
                    continue

                self.hooks.trigger("on_video_start", video_name=video_name)

                # Initialize memory
                mem_path = pending[0].mem_path if pending else ""
                self.memory.on_video_start(video_name, mem_path=mem_path)

                # Process all questions for this video
                for q in pending:
                    # Note: q.before_clip=0 is valid and common (the opening question in M3-Bench);
                    # we cannot use `or -1` short-circuiting, otherwise it would be wrongly replaced with -1.
                    clip_id_for_q = q.before_clip if q.before_clip is not None else -1
                    result = self._handle_question(q, clip_id_for_q)
                    if result:
                        record, answer = result
                        # Write the result
                        result_obj = self._build_result_obj(q, record, answer)
                        f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                        f_out.flush()
                        done_ids.add(q.question_id)

                self.hooks.trigger("on_video_complete", video_name=video_name)
                self.memory.on_video_end(video_name)

                # Incremental reflection: leave staged artifacts even if the run is interrupted after each video is processed
                self._run_incremental_reflection(batch_idx=video_idx)

        # Compute metrics and generate the report
        results = self.evaluator.compute_all()

        # Batch-level self-evolution
        self._run_post_batch_evolution(results)

        self.hooks.trigger("on_harness_end", results=results)

        # Save the evaluation summary
        self._save_results(results)

        return results

    def run_batch(self, batch_size: int = 64) -> Dict[str, Any]:
        """Batch processing mode: compatible with the existing batch processing logic.

        Group all questions by batch_size and process each batch concurrently.
        """
        self.hooks.trigger("on_harness_start", config=self.config)

        output_path = self._get_output_path()
        done_ids = self._load_done_ids(output_path)

        # Collect all pending questions
        all_questions = [q for q in self.qa_dataset.questions
                         if q.question_id not in done_ids]

        logger.info(f"{len(all_questions)} pending questions in total ({len(done_ids)} already skipped)")

        # Split into batches
        batches = [all_questions[i:i+batch_size]
                   for i in range(0, len(all_questions), batch_size)]

        with open(output_path, "a", encoding="utf-8") as f_out:
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} questions)")

                for q in batch:
                    # Load the memory of the corresponding video
                    video_name = q.video_name
                    self.memory.on_video_start(video_name, mem_path=q.mem_path)

                    # Note: q.before_clip=0 is valid and common (the opening question in M3-Bench);
                    # we cannot use `or -1` short-circuiting, otherwise it would be wrongly replaced with -1.
                    clip_id_for_q = q.before_clip if q.before_clip is not None else -1
                    result = self._handle_question(q, clip_id_for_q)
                    if result:
                        record, answer = result
                        result_obj = self._build_result_obj(q, record, answer)
                        f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                        f_out.flush()

                self.hooks.trigger("on_batch_complete", batch_idx=batch_idx)

                # Incremental reflection: leave artifacts even if the run is interrupted after each batch is processed
                self._run_incremental_reflection(batch_idx=batch_idx)

        results = self.evaluator.compute_all()

        # Batch-level self-evolution
        self._run_post_batch_evolution(results)

        self.hooks.trigger("on_harness_end", results=results)
        self._save_results(results)
        return results

    def _handle_question(self, q: TemporalQuestion, clip_id: int) -> Optional[tuple]:
        """Process a single question: answer and evaluate.

        Returns:
            an (EvalRecord, AgentAnswer) tuple, or None (when the answer is deferred).

        Self-evolution closed loop (when --evolution is enabled):
        1. Before answering: match a skill via SkillRouter and inject special_instructions into the Agent
        2. After answering: capture experience via LearningCapture, then SkillPromoter attempts to distill a skill
        """
        self.hooks.trigger("on_question_received", question=q, clip_id=clip_id)

        # ---- Self-evolution: inject a skill before answering ----
        if self._evolution_enabled:
            self._inject_skill_for_question(q)

        try:
            answer = self.agent.answer(q, self.memory)
            self.hooks.trigger("on_answer_generated", answer=answer, question=q)

            if self.answer_policy.should_answer(answer, clip_id, q):
                record = self._record_answer(q, answer, clip_id)

                # ---- Self-evolution: capture experience after answering and attempt to distill a skill ----
                if self._evolution_enabled:
                    self._evolve_after_answer(q, answer, record, clip_id)

                return (record, answer)
            else:
                self.deferred_questions.append((q, clip_id))
                return None
        except Exception as e:
            logger.error(f"Failed to process question {q.question_id}: {e}")
            # Self-healing: retry once (Harness Engineering: Verify + Feedback Loop)
            try:
                logger.info(f"[SelfHeal] Retrying question {q.question_id}")
                answer = self.agent.answer(q, self.memory)
                self.hooks.trigger("on_answer_generated", answer=answer, question=q)
                record = self._record_answer(q, answer, clip_id)
                return (record, answer)
            except Exception as e2:
                logger.error(f"[SelfHeal] Retry still failed: {e2}")
                failed_answer = AgentAnswer(content="", confidence=0.0, is_final=True)
                record = self._record_answer(q, failed_answer, clip_id)
                return (record, failed_answer)

    def _record_answer(self, q: TemporalQuestion, answer: AgentAnswer,
                       clip_id: int) -> EvalRecord:
        """Record and evaluate the answer."""
        memory_stats = self.memory.stats()
        record = self.evaluator.record(q, answer, clip_id, memory_stats)
        self.hooks.trigger("on_answer_evaluated", record=record, question=q, answer=answer)
        self.qa_dataset.mark_answered(q.question_id)
        return record

    def _build_result_obj(self, q: TemporalQuestion, record: EvalRecord,
                          answer: AgentAnswer = None) -> Dict:
        """Build the output result object (compatible with the existing format).

        When save_trajectory is enabled, the full trajectory is saved to a separate JSON file,
        and a trajectory_file field pointing to that file is added to the result object.
        """
        result = {
            "id": q.question_id,
            "question": q.question,
            "options": q.options,
            "answer": q.answer,
            "response": record.answer,
            "gpt_eval": record.is_correct,
            "num_rounds": record.num_rounds,
            "video_name": q.video_name,
            "category": q.category,
        }

        # Save the full trajectory
        if self._save_trajectory and answer is not None:
            trajectory = self._build_trajectory(q, answer, record)
            traj_filename = f"{q.video_name}_{q.question_id}.json"
            traj_path = os.path.join(self._trajectory_dir, traj_filename)
            try:
                with open(traj_path, "w", encoding="utf-8") as f:
                    json.dump(trajectory, f, ensure_ascii=False, indent=2)
                result["trajectory_file"] = traj_filename
            except Exception as e:
                logger.warning(f"Failed to save trajectory {q.question_id}: {e}")

        return result

    def _build_trajectory(self, q: TemporalQuestion, answer: AgentAnswer,
                          record: EvalRecord) -> Dict:
        """Build the complete reasoning trajectory object.

        Includes:
        - Question metadata
        - The full conversation of each round (system prompt, user, assistant)
        - The search queries and search results of each round
        - The final answer and evaluation result
        - Resource consumption statistics
        """
        # Parse conversations into structured rounds
        rounds = self._parse_rounds(answer.conversations)

        trajectory = {
            # ---- Metadata ----
            "meta": {
                "question_id": q.question_id,
                "video_name": q.video_name,
                "category": q.category,
                "question": q.question,
                "options": q.options,
                "ground_truth": q.answer,
            },
            # ---- Reasoning process ----
            "reasoning": {
                "total_rounds": answer.num_rounds,
                "search_queries": answer.search_queries,
                "rounds": rounds,
                "reasoning_trace": answer.reasoning_trace,
            },
            # ---- Full conversation history (raw format, convenient for replay) ----
            "conversations": answer.conversations,
            # ---- Final result ----
            "result": {
                "response": answer.content,
                "confidence": answer.confidence,
                "is_correct": record.is_correct,
                "is_final": answer.is_final,
            },
            # ---- Resource consumption ----
            "stats": {
                "tokens_used": answer.tokens_used,
                "num_rounds": answer.num_rounds,
                "memory_stats": record.memory_stats,
                "timestamp": record.timestamp,
            },
        }
        return trajectory

    @staticmethod
    def _parse_rounds(conversations: List[Dict]) -> List[Dict]:
        """Parse the raw conversation list into structured rounds.

        Each round contains:
        - round_idx: round number
        - user_input: user input (search results or the initial prompt)
        - assistant_output: model output
        - action: the parsed action (Answer/Search)
        - search_query: the search query (if it is a Search)
        - search_type: the search type (plain/VIDEO/NEIGHBOR)
        """
        if not conversations:
            return []

        rounds = []
        round_idx = 0

        # Skip the system prompt and start from the first user message
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            if msg.get("role") == "system":
                i += 1
                continue

            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                assistant_content = ""

                # Find the next assistant message
                if i + 1 < len(conversations) and conversations[i + 1].get("role") == "assistant":
                    assistant_content = conversations[i + 1].get("content", "")
                    i += 2
                else:
                    i += 1

                # Parse the action
                action = None
                search_query = None
                search_type = None
                import re
                # Aligned with guardrails.ResponseValidator.ACTION_PATTERN:
                #   - Square brackets are optional (the model often omits "Action: Search")
                #   - Case-insensitive ("action: search" is also accepted)
                #   - Only the Answer / Search whitelist is accepted
                #   - Compatible with a "</think>" prefix (the model occasionally includes a think tag)
                parse_src = (
                    assistant_content.split("</think>")[-1]
                    if "</think>" in assistant_content
                    else assistant_content
                )
                action_match = re.search(
                    r"Action:\s*\[?\s*(Answer|Search)\s*\]?\s*.*?Content:\s*(.*)",
                    parse_src,
                    re.DOTALL | re.IGNORECASE,
                )
                if action_match:
                    # capitalize normalizes "search" / "SEARCH" into "Search"
                    action = action_match.group(1).strip().capitalize()
                    content = action_match.group(2).strip()
                    # ---- Clean content: strip the trailing ```ledger``` / ```json``` code blocks
                    #      and the `LedgerOps: {...}` fragment, so that the search_query in the
                    #      trajectory exactly matches the query actually sent to the retriever
                    #      (ledger_agent performs the same strip before executing the search).
                    content = strip_ledger_ops_from_text(content).strip()
                    if action == "Search" and content:
                        search_query = content
                        if content.upper().startswith("VIDEO:"):
                            search_type = "VIDEO_drilldown"
                        elif content.upper().startswith("NEIGHBOR:"):
                            search_type = "NEIGHBOR_walk"
                        elif content.upper().startswith("KEYFRAME:"):
                            search_type = "keyframe_inspect"
                        else:
                            search_type = "plain_query"
                    elif action == "Answer":
                        search_type = None

                round_info = {
                    "round_idx": round_idx,
                    "user_input": user_content[:500] + ("..." if len(user_content) > 500 else ""),
                    "user_input_length": len(user_content),
                    "assistant_output": assistant_content,
                    "action": action,
                    "search_query": search_query,
                    "search_type": search_type,
                }
                rounds.append(round_info)
                round_idx += 1
            else:
                i += 1

        return rounds

    # ---- Self-evolution closed-loop methods ----

    def _build_evolution_components(self):
        """Build all components of the self-evolution system."""
        evo_cfg = self.config.get("evolution", {})
        base_dir = evo_cfg.get("dir", ".lv_harness")

        learnings_dir = evo_cfg.get("learnings_dir", f"{base_dir}/learnings")
        skills_dir = evo_cfg.get("skills_dir", f"{base_dir}/skills")
        wisdom_path = evo_cfg.get("wisdom_path", f"{base_dir}/WISDOM.md")
        reflections_dir = evo_cfg.get("reflections_dir", f"{base_dir}/reflections")
        mem_visual = self.config.get("memory", {}).get("visual_layer", {}) or {}
        visual_layer_enabled = bool(
            evo_cfg.get(
                "visual_layer_enabled",
                mem_visual.get("enabled", False),
            )
        )

        self._learning_capture = LearningCapture(
            learnings_dir=learnings_dir,
            capture_successes=evo_cfg.get("capture_successes", True),
            load_prior_learnings=evo_cfg.get("load_prior_learnings", False),
        )
        self._skill_promoter = SkillPromoter(
            skills_dir=skills_dir,
            promote_threshold=evo_cfg.get("promote_threshold", 3),
            use_llm_instructions=evo_cfg.get("skill_use_llm_instructions", False),
            instructions_llm_model=evo_cfg.get(
                "skill_instructions_llm_model", "gemini-2.5-flash"
            ),
            visual_layer_enabled=visual_layer_enabled,
            agent_mode=self.config.get("reasoning", {}).get("agent", "ledger_multi_round"),
        )
        self._skill_router = SkillRouter(
            skills_dir=skills_dir,
            route_threshold=evo_cfg.get("route_threshold", 0.3),
            min_downstream_samples=evo_cfg.get("router_min_downstream_samples", 5),
            disable_below_baseline_margin=evo_cfg.get(
                "router_disable_margin", 0.05
            ),
            hard_fail_min_samples=evo_cfg.get("router_hard_fail_min_samples", 3),
            visual_layer_enabled=visual_layer_enabled,
            agent_mode=self.config.get("reasoning", {}).get("agent", "ledger_multi_round"),
        )
        # P0: Inject SkillRouter's blacklist check into SkillPromoter, to prevent blacklisted skills
        #     from being revived by maybe_promote via re-update / re-creation
        self._skill_promoter.set_blacklist_checker(self._skill_router.is_blacklisted)
        self._wisdom_distiller = WisdomDistiller(
            wisdom_path=wisdom_path,
            reflections_dir=reflections_dir,
            use_llm=evo_cfg.get("wisdom_use_llm", False),
            llm_model=evo_cfg.get("wisdom_llm_model", "gemini-2.5-flash"),
            llm_max_failures=evo_cfg.get("wisdom_llm_max_failures", 20),
            reflection_llm_max_tokens=evo_cfg.get("reflection_llm_max_tokens", 8192),
            agent_mode=self.config.get("reasoning", {}).get("agent", "ledger_multi_round"),
        )

        # Load historical skills (cross-run skill accumulation)
        # Controlled by the evolution.load_prior_skills switch, defaulting to True.
        # During experiments it can be set to False to ensure a "cold start", making it easier to control variables.
        if evo_cfg.get("load_prior_skills", True):
            prior_skills = self._skill_router.load_prior_skills()
            if prior_skills:
                # Sync to SkillPromoter to avoid recreating skills of the same cluster
                self._skill_promoter.load_existing_skills(prior_skills)
                logger.info(
                    f"[Self-evolution] Cross-run skill accumulation enabled: loaded "
                    f"{len(prior_skills)} historical skills from {skills_dir}"
                )
        else:
            logger.info("[Self-evolution] load_prior_skills=False; this run accumulates skills from scratch")

        # Load the existing WISDOM and inject it as global instructions
        if evo_cfg.get("inject_wisdom", True):
            self._inject_wisdom_instructions()

        logger.info(
            f"[Self-evolution] Enabled | learnings={learnings_dir} | skills={skills_dir} | "
            f"wisdom={wisdom_path} | promote_threshold={evo_cfg.get('promote_threshold', 3)} | "
            f"load_prior_learnings={evo_cfg.get('load_prior_learnings', False)}"
        )
        if self._evolution_readonly:
            logger.info(
                "[Self-evolution] readonly mode enabled: skill injection works normally, "
                "but no learning is written, no skill is promoted, and wisdom is not updated"
            )

    def _inject_skill_for_question(self, q: TemporalQuestion):
        """Before answering a question, match a skill via SkillRouter and inject it into the Agent.

        Closed-loop node: Skill -> Agent

        P2 phased injection strategy:
          - Positive-sample skills (hard_win / search_win, etc.) -> injected once into the system prompt at the entry point
          - Negative-sample skills (calibration / search_fail) -> not injected all at once;
            instead, an on_round_start_hook is registered, which pops up a mid-turn hint only when a runtime condition is met (consecutive empty searches / nearing the last round).
        """
        # Clear the skill marker injected in the previous round, to prevent downstream miscounting
        self._current_injected_skill_id = None
        self._current_fired_skill_ids = []
        # Clear the on_round_start_hook of the previous question
        if hasattr(self.agent, "_on_round_start_hook"):
            try:
                delattr(self.agent, "_on_round_start_hook")
            except Exception:
                pass

        # P1: Inject decomposition strategy guidance (only the Ledger agent has this method)
        if (
            hasattr(self.agent, "set_decompose_guidance")
            and self._wisdom_distiller
            and self._learning_capture
        ):
            try:
                guidance = self._wisdom_distiller.generate_decompose_guidance(
                    self._learning_capture.all_learnings,
                    question=q,
                )
                self.agent.set_decompose_guidance(guidance)
            except Exception as exc:
                logger.warning(f"[Self-evolution] generate_decompose_guidance error: {exc}")

        if not self._skill_router:
            return

        matched_skill = self._skill_router.route(q)
        if not matched_skill:
            # When no skill matches, clear the previously injected instructions
            self.agent.inject_instructions("")
            return

        skill_id = matched_skill.skill_id
        is_negative_cluster = (
            skill_id.startswith("SKILL-calibration__")
            or skill_id.startswith("SKILL-search_fail__")
        )

        # Build the standard skill text (context kept consistent with the original logic)
        down_total = (
            matched_skill.downstream_success
            + matched_skill.downstream_failure
        )
        if down_total > 0:
            perf_tag = (
                f"downstream_win_rate={matched_skill.downstream_success_rate:.0%} "
                f"({matched_skill.downstream_success}/{down_total})"
            )
        else:
            perf_tag = "downstream_win_rate=untested"

        instructions_parts = []

        # Sub-skill routing: first try to match an instruction slot
        slot_instructions = None
        if matched_skill.instruction_slots and self._skill_router:
            slot_instructions = self._skill_router.route_instruction_slot(
                q, matched_skill
            )

        if slot_instructions:
            # Use slot-level precise instructions
            slot_route = getattr(self._skill_router, "last_slot_route", {}) or {}
            slot_tag = (
                f", slot={slot_route.get('slot_id')}, slot_score={slot_route.get('score', 0):.2f}"
                if slot_route.get("matched") else ", slot=unknown"
            )
            instructions_parts.append(
                f"## Skill Guidance ({matched_skill.name}, {perf_tag}{slot_tag})\n"
                f"{slot_instructions}"
            )
        elif matched_skill.special_instructions:
            # Fallback: use skill-level general instructions
            clean_instr = matched_skill.special_instructions
            if clean_instr.startswith("[template]"):
                clean_instr = clean_instr[len("[template]"):].lstrip()
            slot_route = getattr(self._skill_router, "last_slot_route", {}) or {}
            fallback_tag = ""
            if matched_skill.instruction_slots:
                fallback_reason = slot_route.get("reason", "slot_not_matched")
                fallback_tag = f", fallback=skill, slot_reason={fallback_reason}"
            instructions_parts.append(
                f"## Skill Guidance ({matched_skill.name}, {perf_tag}{fallback_tag})\n"
                f"{clean_instr}"
            )
        # Note: examples (few-shot) are no longer injected into the agent prompt.
        # Reason: examples are bound to the specific QA pairs of the source video, and during cross-video transfer
        # they tend to mislead the agent into surface-level pattern matching rather than understanding the strategy rules,
        # while also consuming extra tokens.
        # The examples field is still kept in the skill .md file for manual debugging reference.

        full_instructions = "\n\n".join(instructions_parts) if instructions_parts else ""

        if is_negative_cluster:
            # P2 phased: do not write the system prompt; instead register a hook that pops up only when a runtime condition is met
            self.agent.inject_instructions("")  # Clear the system prompt to avoid noise
            self._register_deferred_skill_hook(matched_skill, full_instructions)
            logger.info(
                f"[Self-evolution] Deferred injection (negative-sample skill): {matched_skill.name} -> "
                f"question {q.question_id[:20]}..."
            )
        else:
            # Positive-sample skill: injected once into the system prompt at the entry point
            if full_instructions:
                self.agent.inject_instructions(full_instructions)
                self._current_injected_skill_id = matched_skill.skill_id
                self._skill_router.mark_injected(matched_skill.skill_id)
                logger.info(
                    f"[Self-evolution] Skill injection (positive-sample skill): {matched_skill.name} -> "
                    f"question {q.question_id[:20]}..."
                )
            else:
                self.agent.inject_instructions("")

        # Apply the recommended maximum number of rounds. recommended_max_rounds is provided by the skill
        # generation stage; its current default value is 10, so 5 is no longer used as the "do not override" sentinel value.
        if matched_skill.recommended_max_rounds and matched_skill.recommended_max_rounds > 0:
            self.agent.set_max_rounds(matched_skill.recommended_max_rounds)

    def _register_deferred_skill_hook(self, skill, full_instructions: str) -> None:
        """Register the hint of a negative-sample skill as an on_round_start_hook that fires only when a runtime condition is met.

        Trigger conditions:
          - search_fail / calibration cluster: consecutive low-increment searches >= 2, or the same focus stalls >= 2 times in a row.
            The low-increment and same-focus signals are updated by the agent's sufficiency assessor after each round of search.
        """
        if not full_instructions:
            return

        skill_id = skill.skill_id
        is_search_fail = skill_id.startswith("SKILL-search_fail__")
        is_calibration = skill_id.startswith("SKILL-calibration__")
        is_negative_skill = is_search_fail or is_calibration
        agent = self.agent

        # Use a closure to maintain the "already fired" state, to avoid injecting the same hint multiple times for the same question
        fired = {"value": False}

        def hook(round_idx: int, runtime_state: dict):
            if fired["value"]:
                return
            consecutive_low = runtime_state.get("consecutive_low_increment_searches", 0)
            same_focus_stall = runtime_state.get("same_focus_stall_count", 0)
            should_fire = bool(
                is_negative_skill and (
                    consecutive_low >= 2 or same_focus_stall >= 2
                )
            )
            if should_fire and hasattr(agent, "inject_deferred_hint"):
                logger.info(
                    f"[Self-evolution] Negative-sample skill fired: {skill.name} | "
                    f"round={round_idx} consecutive_low={consecutive_low} "
                    f"same_focus_stall={same_focus_stall}"
                )
                agent.inject_deferred_hint(full_instructions)
                fired["value"] = True
                fired_ids = getattr(self, "_current_fired_skill_ids", None)
                if fired_ids is None:
                    self._current_fired_skill_ids = []
                    fired_ids = self._current_fired_skill_ids
                if skill_id not in fired_ids:
                    fired_ids.append(skill_id)

        # Bind it to the agent; answer() calls it at the start of each round; it is delattr-ed for the next question in _inject_skill_for_question.
        try:
            setattr(agent, "_on_round_start_hook", hook)
        except Exception as exc:
            logger.warning(f"[Self-evolution] Failed to register on_round_start_hook: {exc}")

    def _evolve_after_answer(self, q: TemporalQuestion, answer: AgentAnswer,
                             record: EvalRecord, clip_id: int):
        """After answering and evaluating, run the self-evolution pipeline.

        Closed-loop node: Eval -> Learning -> Skill
        """
        if not self._learning_capture:
            return

        # Read-only mode: skip all write operations (learning capture / skill promote / downstream write-back)
        if self._evolution_readonly:
            return

        # Step 0: If a skill was injected this round, first write back the downstream metric (the truly meaningful win rate)
        skill_ids = []
        if getattr(self, "_current_injected_skill_id", None):
            skill_ids.append(self._current_injected_skill_id)
        for fired_skill_id in getattr(self, "_current_fired_skill_ids", []) or []:
            if fired_skill_id not in skill_ids:
                skill_ids.append(fired_skill_id)

        if skill_ids and self._skill_router:
            for skill_id in skill_ids:
                if skill_id in getattr(self, "_current_fired_skill_ids", []):
                    self._skill_router.mark_injected(skill_id)
                self._skill_router.record_outcome(
                    skill_id,
                    record.is_correct,
                )
                logger.debug(
                    f"[Self-evolution] downstream write-back: skill={skill_id} "
                    f"is_correct={record.is_correct}"
                )
            self._current_injected_skill_id = None
            self._current_fired_skill_ids = []

        # Step 1: Capture experience
        memory_stats = self.memory.stats()
        learning = self._learning_capture.capture_from_eval(
            question=q,
            answer=answer,
            is_correct=record.is_correct,
            memory_stats=memory_stats,
            clip_id=clip_id,
        )

        if learning:
            self.hooks.trigger(
                "on_learning_captured",
                learning=learning,
                question=q,
            )

        # Step 1.5: P0 subtask-level experience capture (only Ledger mode has _ledger_snapshot)
        ledger_snapshot = None
        if answer.reasoning_trace:
            for item in answer.reasoning_trace:
                if isinstance(item, dict) and "_ledger_snapshot" in item:
                    ledger_snapshot = item["_ledger_snapshot"]
                    break
        if ledger_snapshot:
            try:
                ledger_learnings = self._learning_capture.capture_from_ledger(
                    question=q,
                    answer=answer,
                    is_correct=record.is_correct,
                    ledger_snapshot=ledger_snapshot,
                    memory_stats=memory_stats,
                )
                for ll in ledger_learnings:
                    self.hooks.trigger(
                        "on_learning_captured",
                        learning=ll,
                        question=q,
                    )
            except Exception as exc:
                logger.warning(f"[Self-evolution] capture_from_ledger error: {exc}")

        # Step 2: Attempt to distill a skill (checked after every answer; SkillPromoter has an internal threshold control)
        if self._skill_promoter:
            new_skills = self._skill_promoter.maybe_promote(
                self._learning_capture.all_learnings
            )
            for skill in new_skills:
                # Register with the router
                self._skill_router.register_skill(skill)
                self.hooks.trigger("on_skill_promoted", skill=skill)
                logger.info(
                    f"[Self-evolution] New skill distilled: {skill.name} "
                    f"(success_rate={skill.success_rate:.0%}, "
                    f"from {len(skill.created_from)} learnings)"
                )

    def _run_incremental_reflection(self, batch_idx: int) -> None:
        """Trigger one incremental reflection after each video/batch completes.

        - Only writes `reflection_batch_*.md` under reflections_dir, does not update WISDOM.md.
        - Protected by the min_new_learnings threshold built into distill_incremental, so it does not write frequently.
        - Purpose: even if a run is interrupted midway, it leaves staged reflection artifacts for manual iteration.
        """
        if not self._evolution_enabled or not self._wisdom_distiller:
            return
        if self._evolution_readonly:
            return
        all_skills = (
            list(self._skill_router.skills.values())
            if self._skill_router else []
        )
        all_learnings = (
            self._learning_capture.all_learnings
            if self._learning_capture else []
        )
        if not all_learnings:
            return
        try:
            self._wisdom_distiller.distill_incremental(
                batch_results={},
                skills=all_skills,
                learnings=all_learnings,
                batch_idx=batch_idx,
            )
        except Exception as exc:
            logger.warning(f"[Self-evolution] Incremental reflection failed (does not affect the main pipeline): {exc}")

    def _run_post_batch_evolution(self, results: Dict[str, Any]):
        """After a batch of evaluations completes, run batch-level self-evolution.

        Closed-loop node: Batch Results -> Wisdom -> Agent (takes effect on the next run)
        """
        if not self._evolution_enabled:
            return
        if self._evolution_readonly:
            logger.info("[Self-evolution] readonly mode; skipping post-batch self-evolution writes")
            return

        if not self._wisdom_distiller:
            return

        # Collect all skills
        all_skills = list(self._skill_router.skills.values()) if self._skill_router else []
        all_learnings = self._learning_capture.all_learnings if self._learning_capture else []

        if not all_learnings:
            logger.info("[Self-evolution] No experience records; skipping wisdom distillation")
            return

        # First sync the current batch's baseline accuracy to SkillRouter,
        # used for prior smoothing of effective_rate and for the "verified negative effect" skill deactivation decision
        if self._skill_router is not None:
            correct = sum(1 for l in all_learnings if l.is_correct)
            baseline = correct / len(all_learnings)
            self._skill_router.update_baseline(baseline)
            logger.info(
                f"[Self-evolution] Synced baseline={baseline:.1%} to SkillRouter; "
                f"routing will now use downstream metrics + cold-start protection"
            )

        # Distill wisdom
        wisdom = self._wisdom_distiller.distill_after_batch(
            batch_results=results,
            skills=all_skills,
            learnings=all_learnings,
        )

        self.hooks.trigger("on_wisdom_distilled", wisdom=wisdom)

        # Statistics summary
        total_learnings = len(all_learnings)
        total_skills = len(all_skills)
        correct_count = sum(1 for l in all_learnings if l.is_correct)
        logger.info(
            f"[Self-evolution] Batch evolution complete | "
            f"learnings: {total_learnings} | "
            f"skills: {total_skills} | "
            f"accuracy: {correct_count}/{total_learnings} "
            f"({correct_count/total_learnings:.1%})"
        )

    def _inject_wisdom_instructions(self):
        """Extract known weaknesses from WISDOM.md and inject them as global instructions.

        Closed-loop node: Wisdom -> Agent (persisted across runs)
        """
        if not self._wisdom_distiller:
            return

        wisdom_text = self._wisdom_distiller.load_wisdom()
        if not wisdom_text:
            return

        # Extract the "Known Weaknesses" section as global instructions
        weakness_section = ""
        in_weakness = False
        for line in wisdom_text.split("\n"):
            if "Known Weakness" in line.lower():
                in_weakness = True
                continue
            if in_weakness:
                if line.startswith("## "):
                    break
                if line.strip():
                    weakness_section += line + "\n"

        if weakness_section.strip():
            wisdom_instructions = (
                "## Known Weaknesses from Past Experience\n"
                "Be aware of the following known issues and try to avoid them:\n"
                f"{weakness_section}"
            )
            self.agent.inject_instructions(wisdom_instructions)
            logger.info(f"[Self-evolution] Injected known-weakness instructions from WISDOM.md")

    # ---- Component construction methods ----

    def _build_qa_dataset(self) -> TemporalQADataset:
        data_cfg = self.config.get("data", {})
        return TemporalQADataset(
            data_file=data_cfg.get("annotation_file", "data/annotations/videomme.json"),
            temporal_mode=data_cfg.get("temporal_mode", "end_of_video"),
        )

    def _build_memory(self) -> MemoryStrategy:
        mem_cfg = self.config.get("memory", {})
        strategy = mem_cfg.get("strategy", "hierarchical")

        if strategy == "hierarchical":
            from .memory.hierarchical import HierarchicalMemory
            return HierarchicalMemory(mem_cfg)
        elif strategy == "no_graph_walk":
            # No-Graph-Walk ablation: the memory layer still uses HierarchicalMemory (which requires EventGraph + VideoGraph),
            # but the prompt layer does not provide the NEIGHBOR/KEYFRAME modes, restricting the model to only plain query + VIDEO:
            from .memory.hierarchical import HierarchicalMemory
            return HierarchicalMemory(mem_cfg)
        elif strategy == "videograph_only":
            from .memory.videograph_only import VideoGraphOnlyMemory
            return VideoGraphOnlyMemory(mem_cfg)
        elif strategy == "eventgraph_only":
            from .memory.eventgraph_only import EventGraphOnlyMemory
            return EventGraphOnlyMemory(mem_cfg)
        elif strategy == "sliding_window":
            from .memory.sliding_window import SlidingWindowMemory
            return SlidingWindowMemory(mem_cfg)
        elif strategy == "compressed":
            from .memory.compressed import CompressedMemory
            return CompressedMemory(mem_cfg)
        else:
            raise ValueError(
                f"Unknown memory strategy: {strategy}. "
                f"Options: hierarchical, no_graph_walk, videograph_only, eventgraph_only, sliding_window, compressed"
            )

    def _build_snapshot_manager(self) -> SnapshotManager:
        snap_cfg = self.config.get("memory", {}).get("snapshot", {})
        if not snap_cfg.get("enabled", False):
            return SnapshotManager(interval=0)
        return SnapshotManager(
            interval=snap_cfg.get("interval", 10),
            max_snapshots=snap_cfg.get("max_snapshots", 50),
            persist_dir=snap_cfg.get("persist_dir", None),
        )

    def _build_agent(self) -> ReasoningAgent:
        reason_cfg = self.config.get("reasoning", {})
        agent_type = reason_cfg.get("agent", "ledger_multi_round")

        # Project memory.visual_layer.enabled onto reasoning.visual_layer_enabled to
        # avoid duplicated settings. Only applied when the user has not explicitly set
        # reasoning.visual_layer_enabled, so advanced usage can still control it separately.
        if "visual_layer_enabled" not in reason_cfg:
            mem_visual = self.config.get("memory", {}).get("visual_layer", {}) or {}
            reason_cfg = dict(reason_cfg)  # do not mutate the original config
            reason_cfg["visual_layer_enabled"] = bool(mem_visual.get("enabled", False))

        # Project memory.strategy onto reasoning.memory_strategy so the agent can pick the matching prompt.
        if "memory_strategy" not in reason_cfg:
            mem_strategy = self.config.get("memory", {}).get("strategy", "hierarchical")
            reason_cfg = dict(reason_cfg) if not isinstance(reason_cfg, dict) else dict(reason_cfg)
            reason_cfg["memory_strategy"] = mem_strategy

        # enable_thinking: pass through from reasoning config to the agent.
        if "enable_thinking" not in reason_cfg:
            reason_cfg = dict(reason_cfg) if not isinstance(reason_cfg, dict) else dict(reason_cfg)
            reason_cfg["enable_thinking"] = bool(reason_cfg.get("enable_thinking", False))

        # enable_reason: defaults to True, set to False when --no_reason is passed.
        if "enable_reason" not in reason_cfg:
            reason_cfg = dict(reason_cfg) if not isinstance(reason_cfg, dict) else dict(reason_cfg)
            reason_cfg["enable_reason"] = True

        if agent_type == "multi_round_search":
            from .reasoning.multi_round import MultiRoundSearchAgent
            return MultiRoundSearchAgent(reason_cfg)
        elif agent_type == "ledger_multi_round":
            # TaskLedger-driven variant: a non-invasive extension of MultiRoundSearchAgent.
            # Implemented via subclassing + overriding answer(), reusing the original
            # _generate/_execute_search/guardrails/sufficiency/context_engineering/visual_layer.
            from .reasoning.ledger_agent import LedgerAwareMultiRoundAgent
            return LedgerAwareMultiRoundAgent(reason_cfg)
        elif agent_type == "decompose_only":
            # DecomposeOnly variant: keeps task decomposition + focus scheduling, but does
            # not inject any ledger content into the context (no TaskLedger snapshot /
            # evidence / notes). Used as a control experiment for LedgerAwareMultiRoundAgent
            # to verify whether ledger injection truly helps multi-round retrieval.
            # Reuses the ledger Planner and SubTask data structures.
            from .reasoning.decompose_only_agent import DecomposeOnlyMultiRoundAgent
            return DecomposeOnlyMultiRoundAgent(reason_cfg)
        elif agent_type == "control_api_harness":
            # ControlApiHarness variant: wraps the control_api.py reasoning logic as an
            # lv_harness Agent, keeping the original prompt and raw VideoGraph retrieval,
            # while adding the full harness machinery (guardrails self-repair / sufficiency
            # runtime state / budget / evolution compatibility). Used to verify the gain of
            # the harness machinery over the baseline reasoning flow.
            from .reasoning.control_api_harness_enhanced_agent import ControlApiHarnessEnhancedAgent
            return ControlApiHarnessEnhancedAgent(reason_cfg)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _build_answer_policy(self) -> AnswerPolicy:
        policy = self.config.get("reasoning", {}).get("answer_policy", "always")
        if policy == "always":
            return AlwaysAnswer()
        elif policy == "confident":
            return ConfidentAnswer()
        elif policy == "deferred":
            return DeferredAnswer()
        else:
            return AlwaysAnswer()

    def _build_evaluator(self) -> StreamingEvaluator:
        eval_cfg = self.config.get("evaluation", {})
        return StreamingEvaluator(
            metrics=eval_cfg.get("metrics", ["accuracy"]),
            eval_model=eval_cfg.get("eval_model", "gemini-2.5-flash"),
            api_config_path=self.config.get("reasoning", {}).get("api_config_path", "configs/api_config.json"),
            string_match_first=eval_cfg.get("string_match_first", True),
            enable_thinking=bool(eval_cfg.get("enable_thinking", False)),
        )

    # ---- Helper methods ----

    def _group_questions_by_video(self) -> Dict[str, List[TemporalQuestion]]:
        """Group questions by video."""
        from collections import OrderedDict
        groups = OrderedDict()
        for q in self.qa_dataset.questions:
            if q.video_name not in groups:
                groups[q.video_name] = []
            groups[q.video_name].append(q)
        return groups

    def _get_output_path(self) -> str:
        """Generate the output file path."""
        task_name = self.config.get("task", "eval")
        model_name = self.config.get("reasoning", {}).get("model", "unknown")
        filename = f"{task_name}-{model_name}-lv_harness.jsonl"
        return os.path.join(self._output_dir, filename)

    def _load_done_ids(self, output_path: str) -> set:
        """Load the IDs of already completed questions."""
        done_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "id" in obj:
                            done_ids.add(obj["id"])
                    except Exception:
                        pass
        if done_ids:
            logger.info(f"Skipped {len(done_ids)} already completed questions")
        return done_ids

    def _save_results(self, results: Dict[str, Any]):
        """Save the evaluation result summary."""
        summary_path = os.path.join(self._output_dir, "eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation summary saved: {summary_path}")

        # Print the summary
        print(self.evaluator.summary())

    def run_streaming(self) -> Dict[str, Any]:
        """Streaming inference mode: build memory while watching the video, and answer questions at checkpoints.

        Core pipeline:
        1. Group by video
        2. For each video, ingest memory clip by clip in order
        3. After ingesting each clip, check whether there are questions to answer
        4. Periodically trigger an EventGraph incremental update
        5. Periodically save memory snapshots

        Difference from run():
        - run() loads pre-built memory and answers questions directly (offline mode)
        - run_streaming() builds memory clip by clip from scratch, answering while watching (streaming mode)
        """
        from .data.stream import VideoStreamSimulator

        self.hooks.trigger("on_harness_start", config=self.config)

        data_cfg = self.config.get("data", {})
        mem_cfg = self.config.get("memory", {})
        clip_dir_template = data_cfg.get("clip_dir_template", "data/video_clips/{video_name}")
        eventgraph_update_interval = mem_cfg.get("eventgraph_update_interval", 20)
        eventgraph_model = mem_cfg.get("eventgraph_model", "DeepSeek-V3.1T")

        # Group by video
        videos = self._group_questions_by_video()
        total_questions = len(self.qa_dataset.questions)
        logger.info(f"[Streaming mode] {len(videos)} videos, {total_questions} questions in total")

        output_path = self._get_output_path()
        done_ids = self._load_done_ids(output_path)

        # EventGraph incremental update state
        eg_updater = self._build_eventgraph_updater(
            update_interval=eventgraph_update_interval,
            model=eventgraph_model,
        )

        with open(output_path, "a", encoding="utf-8") as f_out:
            for video_name, questions in videos.items():
                pending = [q for q in questions if q.question_id not in done_ids]
                if not pending:
                    continue

                self.hooks.trigger("on_video_start", video_name=video_name)

                # Determine the clip directory
                clip_dir = clip_dir_template.format(video_name=video_name)
                if not os.path.isdir(clip_dir):
                    logger.warning(f"[Streaming mode] clip directory does not exist: {clip_dir}; skipping video {video_name}")
                    continue

                # Initialize memory (prefer loading from cache, otherwise start from scratch)
                self.memory.on_video_start(video_name)

                # Check whether it was loaded from cache (skip clip ingestion)
                cache_hit = getattr(self.memory, 'loaded_from_cache', False)

                # Create the streaming clip simulator
                stream = VideoStreamSimulator(clip_dir=clip_dir)
                total_clips = stream.total_clips

                # Set the last_clip of the QA dataset
                if total_clips > 0:
                    self.qa_dataset.set_last_clip(total_clips - 1)

                # Reset the EventGraph incremental updater
                if eg_updater:
                    eg_updater.reset()

                if cache_hit:
                    # Cache hit: skip clip ingestion and answer all questions directly
                    logger.info(
                        f"[Streaming mode] Video {video_name}: cache hit, skipping ingestion of {total_clips} clips, "
                        f"answering {len(pending)} questions directly"
                    )
                    for q in pending:
                        if q.question_id in done_ids:
                            continue
                        result = self._handle_question(q, total_clips - 1)
                        if result:
                            record, answer = result
                            result_obj = self._build_result_obj(q, record, answer)
                            f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                            f_out.flush()
                            done_ids.add(q.question_id)

                    self.hooks.trigger("on_video_complete", video_name=video_name)
                    self.memory.on_video_end(video_name)
                    continue

                logger.info(f"[Streaming mode] Video {video_name}: {total_clips} clips, {len(pending)} questions")

                # Ingest clip by clip
                for clip_id, clip_data in stream:
                    self.hooks.trigger("on_clip_loaded", clip_id=clip_id)

                    # Step 1: Ingest memory
                    self.memory.ingest(clip_id, clip_data)
                    self.hooks.trigger("on_clip_ingested", clip_id=clip_id)

                    # Step 2: EventGraph incremental update
                    if eg_updater:
                        eg_updater.on_clip_ingested(clip_id, self.memory)

                    # Step 3: Save snapshot
                    saved = self.snapshot_mgr.maybe_save(clip_id, self.memory)
                    if saved:
                        self.hooks.trigger("on_snapshot_saved", clip_id=clip_id)

                    # Step 4: Check whether there are questions to answer
                    questions_at_clip = self.qa_dataset.get_questions_at(clip_id)
                    for q in questions_at_clip:
                        if q.question_id in done_ids:
                            continue
                        result = self._handle_question(q, clip_id)
                        if result:
                            record, answer = result
                            result_obj = self._build_result_obj(q, record, answer)
                            f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                            f_out.flush()
                            done_ids.add(q.question_id)

                    # Step 5: Process deferred questions
                    still_deferred = []
                    for dq, d_clip in self.deferred_questions:
                        if dq.question_id in done_ids:
                            continue
                        result = self._handle_question(dq, clip_id)
                        if result:
                            record, answer = result
                            result_obj = self._build_result_obj(dq, record, answer)
                            f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                            f_out.flush()
                            done_ids.add(dq.question_id)
                        else:
                            still_deferred.append((dq, d_clip))
                    self.deferred_questions = still_deferred

                # After the video ends, force-trigger an EventGraph incremental update (to process the final batch of clips that is smaller than update_interval)
                if eg_updater:
                    eg_updater.force_update(self.memory)

                # After the video ends, perform a final refresh of the equivalence relations
                if hasattr(self.memory, 'video_graph'):
                    self.memory.video_graph.refresh_equivalences()

                # After the video ends, force-answer all remaining deferred questions
                for dq, d_clip in self.deferred_questions:
                    if dq.question_id in done_ids:
                        continue
                    # Force answer
                    answer = self.agent.answer(dq, self.memory)
                    record = self._record_answer(dq, answer, total_clips - 1)
                    if record:
                        result_obj = self._build_result_obj(dq, record, answer)
                        f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                        f_out.flush()
                        done_ids.add(dq.question_id)
                self.deferred_questions = []

                self.hooks.trigger("on_video_complete", video_name=video_name)
                self.memory.on_video_end(video_name)

        # Compute metrics and generate the report
        results = self.evaluator.compute_all()

        # Batch-level self-evolution
        self._run_post_batch_evolution(results)

        self.hooks.trigger("on_harness_end", results=results)
        self._save_results(results)
        return results

    def _build_eventgraph_updater(self, update_interval: int = 20,
                                   model: str = "DeepSeek-V3.1T"):
        """Build the EventGraph incremental updater (used only in streaming mode)."""
        mem_cfg = self.config.get("memory", {})
        if not mem_cfg.get("eventgraph_incremental", False):
            return None

        try:
            from .memory.eventgraph_updater import EventGraphIncrementalUpdater
            api_config_path = self.config.get("reasoning", {}).get(
                "api_config_path", "configs/api_config.json"
            )
            return EventGraphIncrementalUpdater(
                update_interval=update_interval,
                model=model,
                api_config_path=api_config_path,
            )
        except ImportError:
            logger.warning("EventGraphIncrementalUpdater is unavailable; skipping incremental update")
            return None

    # ---- Time-travel API ----

    def replay_from(self, clip_id: int, question: TemporalQuestion) -> AgentAnswer:
        """Re-answer a question from a specified time point (for debugging and analysis)."""
        actual_clip = self.snapshot_mgr.restore_to(clip_id, self.memory)
        return self.agent.answer(question, self.memory)

    def compare_at_timepoints(self, question: TemporalQuestion,
                              clip_ids: List[int]) -> List[AgentAnswer]:
        """Answer the same question at multiple time points, used to analyze temporal robustness."""
        answers = []
        for clip_id in clip_ids:
            self.snapshot_mgr.restore_to(clip_id, self.memory)
            answer = self.agent.answer(question, self.memory)
            answers.append(answer)
        return answers
