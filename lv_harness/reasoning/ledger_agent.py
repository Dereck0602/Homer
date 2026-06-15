"""
LedgerAwareMultiRoundAgent: a multi-round retrieval reasoning agent based on TaskLedger.

Zero intrusion into the existing MultiRoundSearchAgent:
  - Fully implemented via inheritance plus overriding `answer()`
  - Does not touch a single byte of multi_round.py
  - Reuses all of the parent class infrastructure: _generate / _execute_search /
    _guardrails / _result_formatter / _conversation_mgr / _sufficiency / the visual layer, and so on
  - The orchestrator selects this implementation via the `agent == "ledger_multi_round"` switch; by default it still uses the parent class

Core flow differences (relative to the parent answer):
  1. Before answering begins, call QuestionPlanner.decompose to produce a TaskLedger (one cold-start LLM call)
  2. Each loop iteration:
       a. The code-level Scheduler picks the next subtask from the Ledger (rather than letting the LLM choose)
       b. Dynamically assemble the user message: `[TaskLedger Snapshot] + [Current Focus] + retrieval_text`
       c. The LLM outputs a Reason/Action/Content triple (LedgerOps output is no longer required)
       d. The code level automatically binds evidence to the subtask (without relying on LLM output)
       e. If all subtasks reach a terminal state, enter Synthesis early
  3. Synthesis stage: use ledger.to_synthesis_block() in place of the original all_search_results JSON
     so that the final prompt contains only structured evidence, avoiding long-context distractors

Backward compatibility:
  - The AgentAnswer signature is unchanged; the outer orchestrator and evaluator are fully reused
  - Additionally attach a `_ledger_snapshot` field (a dict) to reasoning_trace for trajectory tracing
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from .multi_round import (
    MultiRoundSearchAgent,
    _mem_is_visual_query,
)
from .task_ledger import TaskLedger, SubTask, strip_ledger_ops_from_text
from .ledger_prompts import (
    INITIAL_USER_WITH_LEDGER,
    INITIAL_USER_WITH_LEDGER_NO_REASON,
    SYNTHESIS_PROMPT_MCQ,
    SYNTHESIS_PROMPT_MCQ_NO_REASON,
    SYNTHESIS_PROMPT_OPEN,
    SYNTHESIS_PROMPT_OPEN_NO_REASON,
    format_ledger_injection,
    build_ledger_system_prompt,
)
from .planner import QuestionPlanner, PlannerConfig
from .progress_ledger import ProgressLedger, ProgressSignal
from ..data.types import TemporalQuestion, AgentAnswer
from ..memory.base import MemoryStrategy

logger = logging.getLogger(__name__)


class LedgerAwareMultiRoundAgent(MultiRoundSearchAgent):
    """A TaskLedger-driven multi-round retrieval reasoning agent.

    Configuration options (added on top of the original MultiRoundSearchAgent config, all with defaults):
        ledger:
          enabled: true                # master switch; when false, directly use the parent answer()
          planner:
            enabled: true
            max_subtasks: 5
            min_subtasks: 1
            temperature: 0.2
    max_tokens: 8192
            on_failure: "atomic"       # atomic | fail
          scheduler:
            max_attempts_per_subtask: 3
            early_stop_when_all_resolved: true
          synthesis:
            use_ledger_only: true      # true: feed only the ledger to synthesis; false: also feed the original all_search_results
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ledger_cfg = config.get("ledger", {}) or {}
        self._ledger_enabled = bool(ledger_cfg.get("enabled", True))

        planner_raw = ledger_cfg.get("planner", {}) or {}
        self._planner_config = PlannerConfig(
            enabled=bool(planner_raw.get("enabled", True)),
            max_subtasks=int(planner_raw.get("max_subtasks", 5)),
            min_subtasks=int(planner_raw.get("min_subtasks", 1)),
            temperature=float(planner_raw.get("temperature", 0.2)),
max_tokens=int(planner_raw.get("max_tokens", 8192)),
            on_failure=str(planner_raw.get("on_failure", "atomic")),
            skip_decompose_short_chars=int(
                planner_raw.get("skip_decompose_short_chars", 12)
            ),
            allow_short_multisignal_decompose=bool(
                planner_raw.get("allow_short_multisignal_decompose", True)
            ),
        )

        scheduler_raw = ledger_cfg.get("scheduler", {}) or {}
        self._max_attempts_per_subtask = int(
            scheduler_raw.get("max_attempts_per_subtask", 3)
        )
        self._early_stop_when_all_resolved = bool(
            scheduler_raw.get("early_stop_when_all_resolved", True)
        )

        synthesis_raw = ledger_cfg.get("synthesis", {}) or {}
        self._synthesis_ledger_only = bool(
            synthesis_raw.get("use_ledger_only", True)
        )

        # ---- A2: ProgressLedger / replan configuration ----
        progress_raw = ledger_cfg.get("progress", {}) or {}
        self._stall_low_inc_th = int(progress_raw.get("stall_low_increment_threshold", 2))
        self._same_focus_th = int(progress_raw.get("same_focus_replan_threshold", 3))
        self._max_replans = int(progress_raw.get("max_replans", 2))
        self._auto_abandon_stalled = bool(progress_raw.get("auto_abandon_stalled", True))

        # The ledger for the current question; reset at the start of every answer()
        self._current_ledger: Optional[TaskLedger] = None

        # P1: decomposition strategy guidance (dynamically injected by the orchestrator's self-evolution component)
        self._decompose_guidance: str = ""

        # Reason output mode switch (default True)
        self._enable_reason: bool = bool(config.get("enable_reason", True))

        # Memory strategy (projected by the orchestrator from memory.strategy)
        self._memory_strategy: str = config.get("memory_strategy", "hierarchical")

    def set_decompose_guidance(self, guidance: str) -> None:
        """P1: injected by the orchestrator, decomposition strategy guidance driven by historical experience."""
        self._decompose_guidance = guidance or ""

    # ==================================================================
    # A thin wrapper around the parent _generate so it accepts the kwargs used by the Planner
    # ------------------------------------------------------------------
    # The parent _generate signature is (messages, timeout=120); the Planner needs to pass
    # temperature / max_tokens. This wraps a layer that temporarily overrides the agent's sampling parameters.
    # ==================================================================
    def _planner_generate(self, messages, temperature=0.2, max_tokens=8192,
                          timeout=60, **_ignored):
        """LLM call adapter for use by QuestionPlanner."""
        saved_t, saved_m = self.temperature, self.max_tokens
        try:
            self.temperature = temperature
            self.max_tokens = max_tokens
            self._json_mode = True  # Planner output must be JSON
            return self._generate(messages, timeout=timeout)
        finally:
            self.temperature = saved_t
            self.max_tokens = saved_m
            self._json_mode = False

    # ==================================================================
    # Main entry point: override the parent answer()
    # ==================================================================
    def answer(self, question: TemporalQuestion,
               memory: MemoryStrategy) -> AgentAnswer:
        """Ledger-driven multi-round reasoning."""
        # Globally disabled: fall back to parent behavior (preserving compatibility as much as possible)
        if not self._ledger_enabled:
            return super().answer(question, memory)

        is_open_ended = not question.options
        option_text = "\n".join(question.options) if question.options else ""

        # ---- Step 0: reset the state of each subsystem ----
        # Reset the clip_wise dedup state (each new question starts from scratch)
        if hasattr(memory, "reset_current_clips"):
            memory.reset_current_clips()
        self._guardrails.reset()
        self._guardrails.set_question_text(question.question)
        self._sufficiency.reset()
        self._pending_hints = []
        self._runtime_state = {
            "consecutive_empty_searches": 0,
            "total_empty_searches": 0,
            "consecutive_low_increment_searches": 0,
            "same_focus_stall_count": 0,
            "last_focus_label": "",
            "rounds_so_far": 0,
        }
        self._pending_keyframe_paths = []
        self._keyframe_paths_by_msg_idx: Dict[int, List[Dict[str, Any]]] = {}
        keyframe_available = self._refresh_keyframe_preflight(memory)

        # ---- Step 1: Planner decomposition ----
        planner = QuestionPlanner(
            generate_fn=self._planner_generate, config=self._planner_config
        )
        try:
            ledger = planner.decompose(
                question.question, question.options,
                decompose_guidance=self._decompose_guidance,
            )
        except Exception as exc:
            logger.error(f"[LedgerAgent] Planner fatal exception, falling back to parent answer: {exc}")
            return super().answer(question, memory)
        self._current_ledger = ledger

        subtask_count = len(ledger.subtasks)
        effective_max_attempts_per_subtask = self._max_attempts_per_subtask
        if subtask_count <= 2:
            effective_max_attempts_per_subtask = max(
                effective_max_attempts_per_subtask, 5
            )

        logger.info(
            f"[LedgerAgent] ledger initialized: {subtask_count} subtasks, "
            f"ids={list(ledger.subtasks.keys())}, "
            f"max_attempts={effective_max_attempts_per_subtask} "
            f"(configured={self._max_attempts_per_subtask})"
        )

        # ---- A2: ProgressLedger is used for per-round deadlock/loop detection -> trigger replan ----
        progress = ProgressLedger(
            max_attempts_per_subtask=effective_max_attempts_per_subtask,
            stall_low_increment_threshold=self._stall_low_inc_th,
            same_focus_replan_threshold=self._same_focus_th,
            max_replans=self._max_replans,
        )
        # B1: FocusSummaryCache - records the set of focus_label values whose summary has
        # already been fully injected into retrieval_text, used by to_prompt_block for dedup;
        # also used to determine whether this round's retrieval hit "the same event" so a
        # diff-only version can be emitted.
        seen_focus_labels: Set[str] = set()
        # P1: records the videograph_hits items (set of str) already shown under each focus_label,
        #     so when the focus is unchanged only the diff (newly added items) is attached, avoiding
        #     swallowing key evidence.
        seen_vg_items_by_focus: Dict[str, Set[str]] = {}

        # Improvement 3: Ledger Snapshot injection frequency control
        # Record the ledger signature from the previous injection; only inject a full snapshot when the state changes
        _last_ledger_signature: str = ""

        # P0: Focus lock-on detection state machine
        #   N consecutive rounds on the same focus + zero new evidence -> first force a VIDEO drilldown;
        #   if VIDEO has already been done on that focus and there is still no increment, then force NEIGHBOR.
        _FOCUS_LOCKON_THRESHOLD = 2  # triggers after 2 consecutive rounds on the same focus with no increment
        focus_lockon_counter: int = 0
        focus_lockon_last_label: str = ""
        force_video_next: bool = False
        force_neighbor_next: bool = False
        video_drilled_focus_labels: Set[str] = set()

        # ---- Step 2: build the system prompt (V2 slim version, not depending on multi_round.py's SYSTEM_PROMPT) ----
        # Note: ledger mode no longer reuses the baseline SYSTEM_PROMPT / _KEYFRAME_CAPABILITY_*,
        #      to avoid double-defining the Output format with LEDGER_CAPABILITY_BLOCK, and at the same time
        #      cuts the long lectures on Query quality / Identity resolution (already covered by the Planner + subquestion driving).
        #      The baseline multi_round_search system prompt is completely unaffected.
        sys_prompt = build_ledger_system_prompt(
            question=question.question,
            options_text=option_text,
            is_open_ended=is_open_ended,
            visual_layer_enabled=keyframe_available,
            extra_instructions=self._extra_instructions or "",
            enable_reason=self._enable_reason,
            strategy=self._memory_strategy,
        )

        # ---- Step 3: build the initial user message (including the Ledger snapshot) ----
        first_subtask = ledger.next_pending(
            max_attempts=effective_max_attempts_per_subtask
        )
        first_focus_id = first_subtask.id if first_subtask else None
        first_focus_q = first_subtask.question if first_subtask else None
        ledger_injection = format_ledger_injection(
            ledger_block=ledger.to_prompt_block(
                highlight_id=first_focus_id,
                suppress_quote_labels=seen_focus_labels,
            ),
            current_subtask_id=first_focus_id,
            current_subtask_question=first_focus_q,
        )
        initial_user_template = INITIAL_USER_WITH_LEDGER if self._enable_reason else INITIAL_USER_WITH_LEDGER_NO_REASON
        initial_user = initial_user_template.format(
            ledger_injection=ledger_injection
        )
        if keyframe_available and _mem_is_visual_query(question.question):
            initial_user += (
                "\n\n[Visual cues detected] After you locate a focus event, "
                "strongly consider `KEYFRAME: <query>` to inspect visual details."
            )

        conversations = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": initial_user},
        ]

        all_search_results: List[Dict[str, Any]] = []
        search_queries: List[str] = []
        focus_label: Optional[str] = None

        # ==================================================================
        # Main loop
        # ==================================================================
        for round_idx in range(self.max_rounds):
            is_last_round = (round_idx == self.max_rounds - 1)

            # runtime state + hook (preserving the parent class mechanism)
            self._runtime_state["rounds_so_far"] = round_idx
            on_round_hook = getattr(self, "_on_round_start_hook", None)
            if callable(on_round_hook):
                try:
                    on_round_hook(round_idx, self._runtime_state)
                except Exception as exc:
                    logger.warning(f"[LedgerAgent] on_round_start_hook exception: {exc}")

            # Budget check (Constrain: reuse the parent class Guardrail)
            budget_exceeded, budget_reason = self._guardrails.check_budget()
            if budget_exceeded and not is_last_round:
                logger.info(f"[LedgerAgent] budget exceeded, forcing final round: {budget_reason}")
                is_last_round = True

            # ---- A1+A2: round-start Progress Check (reordered: replan takes priority over abandon/early-stop) ----
            # Key: previously calling mark_stalled_terminals first would immediately abandon a 1-subtask atomic plan,
            # causing schedulable_exhausted=True and early_synthesis to jump ahead, leaving replan no chance to rescue. Now:
            #   (1) first let progress.check evaluate whether to replan based on structural signals such as
            #       stall/dep_broken/low_inc/same_focus
            #   (2) only when replan is not triggered (or is a no-op) and the signal suggests early_synthesis,
            #       consider mark_stalled_terminals + schedulable_exhausted early-stop
            signal: ProgressSignal = progress.check(ledger)
            did_replan = False
            if signal.should_replan and not is_last_round:
                replan_result = planner.replan(ledger, signal.replan_reasons)
                if replan_result.get("applied"):
                    progress.mark_replan_done()
                    did_replan = True
                    logger.info(
                        f"[LedgerAgent] replan v{replan_result.get('plan_version')} "
                        f"upd={replan_result.get('updated')} "
                        f"abd={replan_result.get('abandoned')} "
                        f"add={replan_result.get('added')}"
                    )
                else:
                    # A no-op replan is still counted, to avoid repeated triggering for the same reason
                    progress.mark_replan_done()
                    logger.info(
                        f"[LedgerAgent] replan no-op (reasons={signal.replan_reasons})"
                    )

            # Only perform stall abandon + early stop when no replan happened,
            # giving the subtasks added/rewritten/abandoned by replan at least one round of execution.
            if not did_replan:
                if self._auto_abandon_stalled:
                    killed = ledger.mark_stalled_terminals(effective_max_attempts_per_subtask)
                    if killed:
                        logger.info(
                            f"[LedgerAgent] round#{round_idx} automatically abandoned stalled subtasks: {killed}"
                        )

                # Early stop: all subtasks have reached a terminal state, or the scheduler can no longer pick a subtask to advance
                schedulable_exhausted = ledger.next_pending(
                    max_attempts=effective_max_attempts_per_subtask
                ) is None
                if (self._early_stop_when_all_resolved
                        and (ledger.all_done() or schedulable_exhausted)
                        and not is_last_round):
                    logger.info(
                        f"[LedgerAgent] early synthesis: all_done={ledger.all_done()} "
                        f"schedulable_exhausted={schedulable_exhausted} "
                        f"resolved={ledger.resolved_count()}/{len(ledger.subtasks)}"
                    )
                    is_last_round = True
                elif signal.should_early_synthesis and not is_last_round:
                    # The early_synthesis fallback produced by ProgressLedger after max_replans is exhausted
                    logger.info(
                        f"[LedgerAgent] ProgressLedger triggered early synthesis "
                        f"reasons={signal.replan_reasons}"
                    )
                    is_last_round = True

            # --------------------------------------------------------------
            # Final round: Synthesis mode
            # --------------------------------------------------------------
            if is_last_round:
                synth_content = self._build_synthesis_prompt(
                    question=question,
                    ledger=ledger,
                    option_text=option_text,
                    is_open_ended=is_open_ended,
                    all_search_results=all_search_results,
                )
                # synthesis is a single-round forced answer, without conversation history
                force_messages = [{"role": "user", "content": synth_content}]
                response = self._generate(force_messages)
                conversations.append({"role": "assistant", "content": response})

                action, content = self._validated_parse(
                    response, question.options, conversations,
                    is_last_round=True,
                )
                return self._make_final_answer(
                    content=content or "",
                    confidence=0.6,
                    all_search_results=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    conversations=conversations,
                    ledger=ledger,
                    progress=progress,
                )

            # --------------------------------------------------------------
            # Regular round
            # --------------------------------------------------------------
            # 0. Code-level scheduler: fix this round's focus before generation, for attempts/evidence attribution.
            #    This is the key to "dynamic prompt injection + a hard code-level Guardrail": even if the model
            #    declares another target in LedgerOps, attempts are still attributed by scheduled_id;
            #    the model-declared target is only allowed an opportunistic bind.
            scheduled_st = ledger.next_pending(
                max_attempts=effective_max_attempts_per_subtask
            )
            scheduled_id = scheduled_st.id if scheduled_st else None
            # Advance the scheduled subtask from pending to searching (forced by code)
            if scheduled_st is not None and scheduled_st.status == "pending":
                scheduled_st.transition("searching", reason="scheduler_dispatch")

            # First do one conversation compaction (the compacted-away original user message does not affect the ledger,
            # because the ledger is independent state, which is one of the core values of the ledger design)
            conversations = self._conversation_mgr.manage(conversations)

            # Generate
            response = self._generate(conversations)
            conversations.append({"role": "assistant", "content": response})

            # ---- Strip any residual LedgerOps fence (safety net) ----
            # Since the prompt no longer requires the LLM to output LedgerOps, there should normally be no
            # ```ledger``` block. But as a defensive measure we still strip once, to avoid a
            # greedy Action/Content regex swallowing residual JSON into content.
            sanitized_response = strip_ledger_ops_from_text(response)

            # ---- Parse Action/Content (based on the sanitized response) ----
            action, content = self._validated_parse(
                sanitized_response, question.options, conversations,
                require_prior_search=is_open_ended,
            )
            # One more layer of insurance: even if the sanitizer missed it (very rare), forcibly clean content
            if content:
                content = strip_ledger_ops_from_text(content)

            # ---- [Answer]: return directly ----
            if action == "Answer":
                return self._make_final_answer(
                    content=content or "",
                    confidence=1.0,
                    all_search_results=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    conversations=conversations,
                    ledger=ledger,
                    progress=progress,
                )

            # ---- [Search]: execute retrieval ----
            retrieval_payload: Dict[str, Any] = {}
            is_empty = True
            if content:
                search_queries.append(content)
                retrieval_payload = self._execute_search(
                    content, memory, question, focus_label
                )
                if retrieval_payload.get("focus_label"):
                    focus_label = retrieval_payload["focus_label"]
                if content.strip().upper().startswith("VIDEO:") and focus_label:
                    video_drilled_focus_labels.add(focus_label)
                all_search_results.append({
                    "query": content,
                    "retrieval": {"event_retrieval": retrieval_payload},
                })
                is_empty = not retrieval_payload.get("focus")

                # runtime state
                if is_empty:
                    self._runtime_state["consecutive_empty_searches"] += 1
                    self._runtime_state["total_empty_searches"] += 1
                else:
                    self._runtime_state["consecutive_empty_searches"] = 0

                try:
                    retrieval_text = json.dumps(
                        retrieval_payload, ensure_ascii=False, default=str
                    )
                except Exception:
                    retrieval_text = str(retrieval_payload)
                self._guardrails.on_search_result(
                    is_empty, retrieval_text=retrieval_text
                )

                # sufficiency (reuse the parent class signal)
                sufficiency = self._sufficiency.assess_after_search(
                    retrieval_payload, content, question.options,
                    round_idx, self.max_rounds,
                    question_text=question.question,
                    keyframe_available=self._current_keyframe_available,
                )
                increment_state = sufficiency.get("increment", {}) or {}
                self._runtime_state["consecutive_low_increment_searches"] = int(
                    increment_state.get("consecutive_low", 0) or 0
                )
                same_focus_stalled = bool(
                    increment_state.get("focus_label_repeated", False)
                    and increment_state.get("is_low_increment", False)
                )
                self._runtime_state["same_focus_stall_count"] = (
                    self._runtime_state.get("same_focus_stall_count", 0) + 1
                    if same_focus_stalled
                    else 0
                )
                self._runtime_state["last_focus_label"] = retrieval_payload.get("focus_label", "") or ""
                sufficiency_hint = sufficiency.get("hint_message", "")
                if sufficiency.get("should_stop_searching", False):
                    self._guardrails.state.consecutive_empty_searches = (
                        self._guardrails.budget_config.max_consecutive_empty_searches
                    )
                    self._guardrails.state.grace_round_given = True

                # Record attempts on the scheduled subtask
                # The code level uses scheduled_id as the source of truth, no longer relying on the LLM-declared target.
                target_st = (
                    ledger.subtasks.get(scheduled_id) if scheduled_id
                    else None
                )
                if target_st is not None:
                    target_st.attempts += 1
                    if is_empty:
                        target_st.record_failed_query(content)

            # ---- Code-level automatic evidence binding (without relying on LLM-output LedgerOps) ----
            # Core change: after each retrieval, the code level automatically extracts key information
            # from the scheduled subtask + retrieval_payload and binds it to the subtask, ensuring the
            # ledger evidence is always maintained.
            if content and not is_empty and target_st is not None:
                self._auto_bind_evidence(
                    target_st, retrieval_payload, round_idx, content
                )

            # ---- LedgerOps removed: code-level automatic binding fully takes over evidence management ----
            # The LLM is no longer required to output LedgerOps; state management is fully driven by the code level.
            # If the LLM still unexpectedly outputs a ```ledger``` block (very rare),
            # strip_ledger_ops_from_text has already stripped it from content above,
            # so it is no longer parsed or applied here.

            # ---- A2: record this round's accounting to ProgressLedger (used by the next round's start-of-round check) ----
            progress.record_round(
                ledger=ledger,
                scheduled_id=scheduled_id,
                focus_label=retrieval_payload.get("focus_label") if content else None,
                is_empty=is_empty,
            )

            # ---- Build the next round's user message: retrieval text + ledger injection ----
            # B1: FocusSummaryCache - the same focus has its summary fully injected only once across rounds
            focus_label_new = retrieval_payload.get("focus_label") or "" if content else ""
            focus_is_fresh = bool(focus_label_new) and (focus_label_new not in seen_focus_labels)

            if content:
                if is_empty:
                    search_result_text = self._result_formatter.format_empty_result(
                        content,
                        strategy_hint=(
                            "Try a different search strategy: different keywords, "
                            "VIDEO:/NEIGHBOR: prefix, or mark the current subtask "
                            "'abandoned' if truly irrelevant."
                        ),
                    )
                elif not focus_is_fresh and focus_label_new:
                    # Hit an already-seen focus: do not re-inject the summary, but keep the videograph_hits diff
                    score = retrieval_payload.get("event_score")
                    try:
                        score_str = f" (event_score={float(score):.2f})" if score is not None else ""
                    except Exception:
                        score_str = ""
                    search_result_text = (
                        f"Searched knowledge: focus unchanged, still at "
                        f"\"{focus_label_new}\"{score_str}. The event summary has "
                        f"already been shown earlier in this conversation; no new "
                        f"textual evidence was added this round."
                    )
                    # ---- P1: differential retention of videograph_hits ----
                    # Different queries for the same focus may return different videograph_hits subsets,
                    # and this fine-grained information (such as "bowls of green grapes") is often key to the answer.
                    # Attach only the items newly added this round, avoiding duplication without swallowing new evidence.
                    vg_hits = retrieval_payload.get("videograph_hits", {})
                    if vg_hits:
                        prev_items = seen_vg_items_by_focus.get(focus_label_new, set())
                        new_vg_lines: list = []
                        for clip_key, items in vg_hits.items():
                            if isinstance(items, list):
                                for item in items:
                                    item_str = str(item).strip()
                                    if item_str and item_str not in prev_items:
                                        new_vg_lines.append(f"    {clip_key}: {item_str}")
                                        prev_items.add(item_str)
                        seen_vg_items_by_focus[focus_label_new] = prev_items
                        if new_vg_lines:
                            search_result_text += (
                                "\n\n🔍 New fine-grained details (from VideoGraph):\n"
                                + "\n".join(new_vg_lines[:20])  # at most 20 to avoid blowing up context
                            )
                    # ---- P1 END ----

                    # ---- P0: Focus lock-on counting ----
                    # Same focus + no new videograph items -> count += 1
                    if not (vg_hits and new_vg_lines):
                        if focus_label_new == focus_lockon_last_label:
                            focus_lockon_counter += 1
                        else:
                            focus_lockon_counter = 1
                            focus_lockon_last_label = focus_label_new
                    else:
                        # There is new evidence, reset the count
                        focus_lockon_counter = 0
                        focus_lockon_last_label = focus_label_new

                    # P0: threshold reached -> if VIDEO has not been done on the current focus, force VIDEO first;
                    #     if VIDEO has already been done and there is still nothing new, then force NEIGHBOR.
                    if focus_lockon_counter >= _FOCUS_LOCKON_THRESHOLD:
                        if focus_label_new and focus_label_new not in video_drilled_focus_labels:
                            force_video_next = True
                            force_neighbor_next = False
                        else:
                            force_neighbor_next = True
                        focus_lockon_counter = 0  # reset, to avoid consecutive triggering
                    # ---- P0 END ----

                    if sufficiency_hint:
                        search_result_text += f"\n{sufficiency_hint}"
                else:
                    search_result_text = self._result_formatter.format_search_result(
                        retrieval_payload, content,
                        sufficiency_hint=sufficiency_hint,
                    )
                    # Only a new focus is registered into seen; empty and same-name avoid pollution
                    if focus_label_new:
                        seen_focus_labels.add(focus_label_new)
                    # P1: when seeing this focus for the first time, record all videograph_hits into seen
                    vg_hits_first = retrieval_payload.get("videograph_hits", {})
                    if vg_hits_first and focus_label_new:
                        items_set: Set[str] = set()
                        for _items in vg_hits_first.values():
                            if isinstance(_items, list):
                                for _it in _items:
                                    items_set.add(str(_it).strip())
                        seen_vg_items_by_focus[focus_label_new] = items_set
                    # P0: new focus -> reset the lock-on count
                    focus_lockon_counter = 0
                    focus_lockon_last_label = focus_label_new or ""
                    force_video_next = False
                    force_neighbor_next = False
            else:
                search_result_text = (
                    "Searched knowledge: (No valid query provided.)"
                )

            # Consume the grace hint
            grace_hint = self._guardrails.state.pending_grace_hint
            if grace_hint:
                search_result_text += f"\n\n{grace_hint}"
                self._guardrails.state.pending_grace_hint = ""

            # Consume deferred hints (skill, etc.)
            deferred = self._consume_pending_hints()
            if deferred:
                search_result_text = (search_result_text or "") + deferred

            # Pick the next scheduled subtask (note: this is the focus for the *next* round, not this one)
            next_st = ledger.next_pending(
                max_attempts=effective_max_attempts_per_subtask
            )
            next_focus_id = next_st.id if next_st else None
            next_focus_q = next_st.question if next_st else None

            # Extra hint: if some subtask has exhausted its attempts but is still not resolved
            scheduler_hint = ""
            if next_st is not None and next_st.attempts >= effective_max_attempts_per_subtask - 1:
                scheduler_hint = (
                    f"Subtask {next_st.id} has been attempted {next_st.attempts} "
                    f"times. Consider marking it 'partial' with best-effort answer "
                    f"or 'abandoned' if clearly unreachable, then move on."
                )

            # B4: hint dedup. If a sufficiency SATURATION WARNING is already appended in search_result_text,
            # adding another scheduler_hint of similar tone would only be negatively repetitive. Do a crude but robust dedup.
            srt_lower = (search_result_text or "").lower()
            if scheduler_hint and (
                "saturation warning" in srt_lower
                or "diminishing new information" in srt_lower
            ):
                scheduler_hint = ""

            # ---- P0: Focus lock-on forced VIDEO / NEIGHBOR injection ----
            # When there are N consecutive rounds on the same focus + zero new evidence, prefer requiring the next
            # round to use VIDEO: to enter the detailed memory of the current event; if VIDEO has already been done
            # on that focus and there is still nothing new, then force NEIGHBOR: to switch the focus event.
            if force_video_next:
                forced_query = next_focus_q or question.question
                search_result_text += (
                    "\n\n⚠️ [FORCED VIDEO] You have been stuck on the same "
                    "focus event without new evidence, and this focus has not "
                    "been inspected with VideoGraph details yet. Your next "
                    "search MUST use `VIDEO: <query>` to drill down within the "
                    "current focus event before answering or moving to a neighbor. "
                    f"Suggested query: VIDEO: {forced_query}"
                )
                force_video_next = False

            if force_neighbor_next:
                neighbor_labels: list = []
                for nb in (retrieval_payload.get("neighbors") or []):
                    if isinstance(nb, dict):
                        lbl = nb.get("segment_label") or nb.get("label") or ""
                        if lbl:
                            neighbor_labels.append(lbl)
                    elif isinstance(nb, str):
                        neighbor_labels.append(nb)
                if neighbor_labels:
                    nb_list_str = "\n".join(f"  - {lbl}" for lbl in neighbor_labels[:6])
                    search_result_text += (
                        "\n\n⚠️ [FORCED NEIGHBOR] You have been stuck on the same "
                        "focus event for multiple rounds with no new evidence. "
                        "You MUST use `NEIGHBOR: <segment_label>` as your next "
                        "search to shift focus to a different event. "
                        "Available neighbors:\n" + nb_list_str
                    )
                else:
                    # When no neighbor is available, prompt using a brand-new query to switch events
                    search_result_text += (
                        "\n\n⚠️ [FORCED DIVERSIFY] You have been stuck on the same "
                        "focus event for multiple rounds with no new evidence. "
                        "Your next search MUST use completely different keywords "
                        "to find a different event. Do NOT repeat similar queries."
                    )
                force_neighbor_next = False  # consume it, to avoid consecutive injection

            ledger_injection = ""
            current_sig = ledger.snapshot_signature()
            if current_sig != _last_ledger_signature or round_idx == 0:
                # State changed (or first round): inject a full snapshot
                ledger_injection = format_ledger_injection(
                    ledger_block=ledger.to_prompt_block(
                        highlight_id=next_focus_id,
                        suppress_quote_labels=seen_focus_labels,
                    ),
                    current_subtask_id=next_focus_id,
                    current_subtask_question=next_focus_q,
                    hint=scheduler_hint,
                )
                _last_ledger_signature = current_sig
            else:
                # No state change: inject only a one-line compact summary + current focus hint
                compact_line = ledger.to_compact_status_line()
                focus_line = ""
                if next_focus_id and next_focus_q:
                    focus_line = (
                        f"\n[Current Focus] **{next_focus_id}**: {next_focus_q}"
                    )
                hint_line = f"\n[Hint] {scheduler_hint}" if scheduler_hint else ""
                ledger_injection = f"\n{compact_line}{focus_line}{hint_line}"
            search_result_text = (search_result_text or "") + "\n" + ledger_injection

            # keyframe marker (reuse the parent class visual layer)
            if self._pending_keyframe_paths:
                kf_summary = self._format_keyframe_marker(
                    self._pending_keyframe_paths
                )
                search_result_text = (search_result_text or "") + kf_summary

            conversations.append({"role": "user", "content": search_result_text})
            if self._pending_keyframe_paths:
                msg_idx = len(conversations) - 1
                self._keyframe_paths_by_msg_idx[msg_idx] = list(
                    self._pending_keyframe_paths
                )
                self._pending_keyframe_paths = []

        # In theory this should not be reached
        return self._make_final_answer(
            content="", confidence=0.0,
            all_search_results=all_search_results,
            search_queries=search_queries,
            num_rounds=self.max_rounds,
            conversations=conversations,
            ledger=ledger,
            progress=progress,
        )

    # ==================================================================
    # Code-level automatic Evidence Binding
    # ==================================================================
    def _auto_bind_evidence(self, target_st: SubTask,
                            retrieval_payload: Dict[str, Any],
                            round_idx: int, query: str) -> None:
        """The code level proactively binds the retrieval result to the scheduled subtask.

        It does not rely on LLM-output LedgerOps, ensuring every valid retrieval per round is recorded in the ledger.
        This is the core mechanism of the ledger as a "read-write context manager".

        Binding strategy:
          - Extract focus_label, summary, event_score, mode from retrieval_payload
          - Take the first 200 characters of the summary as the quote
          - Automatically advance the subtask state: pending->searching, searching stays
          - Do not overwrite the best_answer already set by the LLM via LedgerOps
        """
        from .task_ledger import Evidence, _safe_float, _safe_int_list

        focus = retrieval_payload.get("focus") or {}
        focus_label = str(retrieval_payload.get("focus_label") or "")
        event_score = _safe_float(retrieval_payload.get("event_score"), 0.0)
        mode = str(retrieval_payload.get("mode") or "event_first")

        # Extract the quote: prefer focus.summary
        quote = ""
        if isinstance(focus, dict):
            raw_summary = focus.get("summary") or focus.get("description") or ""
            quote = str(raw_summary).strip()[:200]

        # Add key details from videograph_hits to the quote
        vg_hits = retrieval_payload.get("videograph_hits", {})
        if vg_hits and isinstance(vg_hits, dict):
            vg_snippets = []
            for clip_key, items in vg_hits.items():
                if isinstance(items, list):
                    for item in items[:3]:  # take at most 3 items per clip
                        item_str = str(item).strip()
                        if item_str:
                            vg_snippets.append(item_str)
            if vg_snippets:
                vg_text = " | ".join(vg_snippets[:5])  # at most 5 items
                if quote:
                    remaining = 400 - len(quote)
                    if remaining > 50:
                        quote += " [VG: " + vg_text[:remaining - 10] + "]"
                else:
                    quote = "[VG: " + vg_text[:300] + "]"

        # Extract clip_ids
        clip_ids = []
        if isinstance(focus, dict):
            clip_ids = _safe_int_list(focus.get("clip_ids"))

        # Only bind when there is substantive content
        if not focus_label and not quote:
            return

        # Check whether evidence with the same focus_label + same query already exists (avoid duplicate binding)
        for existing_ev in target_st.evidence:
            if existing_ev.focus_label == focus_label and existing_ev.query == query:
                return  # already bound, skip

        ev = Evidence(
            round_idx=round_idx,
            query=query,
            focus_label=focus_label,
            quote=quote,
            score=event_score,
            clip_ids=clip_ids,
            mode=mode,
        )
        target_st.add_evidence(ev)

        # Automatically advance the state: pending -> searching (if not already advanced)
        if target_st.status == "pending":
            target_st.transition("searching", reason="auto_bind_evidence")

        # If event_score is relatively high and the subtask has no best_answer yet,
        # use the quote as a temporary best_answer (the LLM can later override it via LedgerOps)
        if event_score >= 0.6 and not target_st.best_answer and quote:
            short_answer = quote.replace("\n", " ").strip()
            if len(short_answer) > 260:
                short_answer = short_answer[:260].rstrip() + "..."
            target_st.best_answer = short_answer
            # High-score evidence automatically advances to partial
            if target_st.status == "searching":
                target_st.transition("partial", reason="auto_bind_high_score_evidence")

        logger.debug(
            f"[LedgerAgent] auto bind evidence: {target_st.id} <- "
            f"focus='{focus_label}' score={event_score:.2f} mode={mode}"
        )

    # ==================================================================
    # Synthesis prompt construction
    # ==================================================================
    def _build_synthesis_prompt(self, question: TemporalQuestion,
                                ledger: TaskLedger,
                                option_text: str,
                                is_open_ended: bool,
                                all_search_results: List[Dict[str, Any]]) -> str:
        """Construct the user prompt for the final synthesis stage.

        Always provides both:
          1. The ledger structured final state (subtask decomposition + best_answer + top evidence)
          2. All deduplicated full retrieval text (deduplicated by focus_label, keeping the richest version)

        Deduplication strategy:
          - Group by focus_label, keeping only one full summary per event
          - Merge (union) the videograph_hits of different queries for the same event
          - Keep neighbors/edges of different events (they may provide cross-event reasoning clues)
          - Finally format with RetrievalResultFormatter into structured natural language, rather than raw JSON
        """
        ledger_block = ledger.to_synthesis_block()

        # Deduplicate and format all retrieval results
        deduped_knowledge = self._deduplicate_and_format_retrievals(all_search_results)

        if is_open_ended:
            binary_hint = self._build_binary_choice_hint(question.question)
            prompt_template = SYNTHESIS_PROMPT_OPEN if self._enable_reason else SYNTHESIS_PROMPT_OPEN_NO_REASON
            return prompt_template.format(
                question=question.question,
                ledger_final_state=ledger_block,
                all_retrieved_knowledge=deduped_knowledge,
                binary_choice_hint=binary_hint,
            )
        else:
            prompt_template = SYNTHESIS_PROMPT_MCQ if self._enable_reason else SYNTHESIS_PROMPT_MCQ_NO_REASON
            return prompt_template.format(
                question=question.question,
                options=option_text,
                ledger_final_state=ledger_block,
                all_retrieved_knowledge=deduped_knowledge,
            )

    def _deduplicate_and_format_retrievals(
        self, all_search_results: List[Dict[str, Any]]
    ) -> str:
        """Deduplicate the retrieval results of all rounds by focus_label, merge videograph_hits,
        then format with RetrievalResultFormatter into structured natural language.

        Deduplication rules:
          - For the same focus_label keep only one focus summary (take the full version of its first occurrence)
          - Take the union of videograph_hits across rounds for the same focus
          - Take the union of neighbors and edges (deduplicated by segment_label)
          - Skip empty retrieval results (no focus)
          - Finally order by first occurrence

        Returns:
            The formatted deduplicated retrieval text, for use by the synthesis prompt
        """
        if not all_search_results:
            return "(No retrieval results available.)"

        # Group and aggregate by focus_label
        # key: focus_label, value: aggregated retrieval_payload
        seen_labels: Dict[str, Dict[str, Any]] = {}  # use dict to preserve order
        label_order: List[str] = []  # first-occurrence order
        query_by_label: Dict[str, List[str]] = {}  # records the queries corresponding to each label

        for result in all_search_results:
            query = result.get("query", "")
            retrieval = result.get("retrieval", {})
            payload = retrieval.get("event_retrieval", retrieval)

            focus = payload.get("focus") or {}
            focus_label = str(payload.get("focus_label") or "")

            # Skip empty retrievals
            if not focus_label and not focus:
                continue

            # Use focus_label as the dedup key; when there is no label, use the first 50 characters of the summary
            dedup_key = focus_label
            if not dedup_key:
                dedup_key = str(focus.get("summary", ""))[:50] or f"_unnamed_{len(seen_labels)}"

            if dedup_key not in seen_labels:
                # First occurrence: keep in full
                seen_labels[dedup_key] = {
                    "focus": focus,
                    "focus_label": focus_label,
                    "event_score": payload.get("event_score", 0),
                    "mode": payload.get("mode", "event_first"),
                    "neighbors": list(payload.get("neighbors") or []),
                    "edges": list(payload.get("edges") or []),
                    "videograph_hits": dict(payload.get("videograph_hits") or {}),
                }
                label_order.append(dedup_key)
                query_by_label[dedup_key] = [query] if query else []
            else:
                # Subsequent occurrences: merge videograph_hits, neighbors, edges
                existing = seen_labels[dedup_key]

                # Merge videograph_hits (merge by clip_key, dedup items)
                new_vg = payload.get("videograph_hits") or {}
                for clip_key, items in new_vg.items():
                    if clip_key not in existing["videograph_hits"]:
                        existing["videograph_hits"][clip_key] = list(items) if isinstance(items, list) else [items]
                    else:
                        existing_items = set(str(x) for x in existing["videograph_hits"][clip_key])
                        if isinstance(items, list):
                            for item in items:
                                if str(item) not in existing_items:
                                    existing["videograph_hits"][clip_key].append(item)
                                    existing_items.add(str(item))

                # Merge neighbors (deduplicated by segment_label)
                existing_nb_labels = set()
                for nb in existing["neighbors"]:
                    if isinstance(nb, dict):
                        existing_nb_labels.add(nb.get("segment_label") or nb.get("label") or "")
                    elif isinstance(nb, str):
                        existing_nb_labels.add(nb)
                for nb in (payload.get("neighbors") or []):
                    if isinstance(nb, dict):
                        lbl = nb.get("segment_label") or nb.get("label") or ""
                        if lbl and lbl not in existing_nb_labels:
                            existing["neighbors"].append(nb)
                            existing_nb_labels.add(lbl)
                    elif isinstance(nb, str) and nb not in existing_nb_labels:
                        existing["neighbors"].append(nb)
                        existing_nb_labels.add(nb)

                # Merge edges (deduplicated by from+to)
                existing_edge_keys = set()
                for e in existing["edges"]:
                    if isinstance(e, dict):
                        existing_edge_keys.add(
                            (e.get("from", ""), e.get("to", ""))
                        )
                for e in (payload.get("edges") or []):
                    if isinstance(e, dict):
                        key = (e.get("from", ""), e.get("to", ""))
                        if key not in existing_edge_keys:
                            existing["edges"].append(e)
                            existing_edge_keys.add(key)

                # Update event_score (take the highest score)
                new_score = payload.get("event_score", 0)
                if new_score and new_score > (existing.get("event_score") or 0):
                    existing["event_score"] = new_score

                # Record the query
                if query:
                    query_by_label[dedup_key].append(query)

        # Format the output
        if not seen_labels:
            return "(No valid retrieval results after deduplication.)"

        parts: List[str] = []
        for idx, dedup_key in enumerate(label_order):
            payload = seen_labels[dedup_key]
            queries = query_by_label.get(dedup_key, [])

            # Use RetrievalResultFormatter to format each deduplicated event
            formatted = self._result_formatter.format_search_result(
                payload, queries[0] if queries else "",
            )
            # Add an annotation for the source queries
            if len(queries) > 1:
                queries_note = f"\n  (Retrieved by {len(queries)} queries: {queries[:4]})"
                formatted += queries_note

            parts.append(formatted)

        return "\n\n---\n\n".join(parts)

    # ==================================================================
    # Result packaging (attach the ledger snapshot to reasoning_trace for trajectory review)
    # ==================================================================
    def _make_final_answer(self, content: str, confidence: float,
                           all_search_results: List[Dict[str, Any]],
                           search_queries: List[str],
                           num_rounds: int,
                           conversations: List[Dict[str, Any]],
                           ledger: TaskLedger,
                           progress: Optional[ProgressLedger] = None) -> AgentAnswer:
        """Construct an AgentAnswer, attaching the ledger snapshot + progress snapshot (if provided)."""
        trace = list(all_search_results)
        try:
            trace.append({"_ledger_snapshot": ledger.to_dict()})
        except Exception as exc:
            logger.debug(f"[LedgerAgent] ledger snapshot serialization failed: {exc}")
        if progress is not None:
            try:
                trace.append({"_progress_snapshot": progress.to_dict()})
            except Exception as exc:
                logger.debug(f"[LedgerAgent] progress snapshot serialization failed: {exc}")

        return AgentAnswer(
            content=content,
            confidence=confidence,
            is_final=True,
            reasoning_trace=trace,
            search_queries=search_queries,
            num_rounds=num_rounds,
            tokens_used=self._guardrails.state.tokens_used,
            conversations=conversations,
        )
