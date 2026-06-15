"""
DecomposeOnlyMultiRoundAgent: a reasoning Agent that only performs task decomposition but does not inject the Ledger into the context.

Purpose: serves as a control experiment for LedgerAwareMultiRoundAgent.
  - Test the hypothesis: whether only doing task decomposition (subtasks) + scheduling the focus, without injecting
    the TaskLedger Snapshot / evidence / notes into every round's user message, works better than full ledger injection.

Differences from LedgerAwareMultiRoundAgent:
  1. Still calls QuestionPlanner to decompose subtasks, and still lets the code-layer scheduler choose the current focus
  2. But each round's user message **only appends**: the retrieval result + "[Current Focus] sub-question"
     (it no longer injects the ledger_block (task list / status / evidence / notes / global_notes))
  3. The system prompt removes the "Ledger awareness" section and no longer hints to the LLM that a ledger exists
  4. No more automatic evidence binding / automatic status advancement / replan / early_stop / auto_abandon
     (the ledger internal state degrades into a "progress counter for scheduling", only recording the attempts of each subtask)
  5. The Synthesis stage directly feeds all deduplicated retrieval results (consistent with LedgerAwareMultiRoundAgent's
     `_deduplicate_and_format_retrievals` output), without ledger_final_state

Does not affect the existing ledger pipeline:
  - A completely independent class, with zero modifications to ledger_agent.py / multi_round.py
  - A new SLIM system prompt variant is created to avoid referencing LEDGER_CAPABILITY_BLOCK
  - The orchestrator selects this implementation via the `agent == "decompose_only"` switch
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from .multi_round import (
    MultiRoundSearchAgent,
    _mem_is_visual_query,
)
from .task_ledger import TaskLedger, SubTask
from .ledger_prompts import (
    SLIM_SYSTEM_MCQ,
    SLIM_SYSTEM_OPEN,
    SLIM_SYSTEM_MCQ_WITH_KEYFRAME,
    SLIM_SYSTEM_OPEN_WITH_KEYFRAME,
)
from .planner import QuestionPlanner, PlannerConfig
from ..data.types import TemporalQuestion, AgentAnswer
from ..memory.base import MemoryStrategy

logger = logging.getLogger(__name__)


# ==================================================================
# 1) Block appended at the end of the system prompt: a counterpart to LEDGER_CAPABILITY_BLOCK
#    - Keep the Output format
#    - Remove all "Ledger awareness" hints
#    - Add a sentence "the system tells you each round which sub-question to focus on", without exposing the ledger concept
# ==================================================================
DECOMPOSE_ONLY_OUTPUT_BLOCK = """

Output format (every turn):
    Reason: <your reasoning>
    Action: [Answer] or [Search]
    Content: <answer or search query>

Sub-question scheduling:
- The original question has been pre-decomposed into atomic sub-questions before search.
- Each round, the system tells you which sub-question to focus on via `[Current Focus]`.
- Direct your search query to serve the current sub-question, but you may use any retrieval mode.
- If you discover a cross-subtask fact (e.g., identity binding like `<character_0> = Bob`), mention it naturally in your Reason.
"""

DECOMPOSE_ONLY_OUTPUT_BLOCK_MCQ = DECOMPOSE_ONLY_OUTPUT_BLOCK + """

MCQ additional rules:
- Before outputting [Answer], your Reason must compare the candidate against at least one plausible distractor.
- If you notice evidence supporting or contradicting specific options, mention it in your Reason.
"""


def build_decompose_only_system_prompt(
    question: str,
    options_text: str,
    is_open_ended: bool,
    visual_layer_enabled: bool = False,
    extra_instructions: str = "",
) -> str:
    """Symmetric to build_ledger_system_prompt, but the end of the system prompt no longer contains Ledger awareness.

    Based on visual_layer_enabled, directly select the full prompt version with/without KEYFRAME,
    instead of assembling it by concatenating a separate KEYFRAME block.
    """
    if is_open_ended:
        if visual_layer_enabled:
            base = SLIM_SYSTEM_OPEN_WITH_KEYFRAME.format(question=question)
        else:
            base = SLIM_SYSTEM_OPEN.format(question=question)
        parts: List[str] = [base, DECOMPOSE_ONLY_OUTPUT_BLOCK]
    else:
        if visual_layer_enabled:
            base = SLIM_SYSTEM_MCQ_WITH_KEYFRAME.format(question=question, options=options_text)
        else:
            base = SLIM_SYSTEM_MCQ.format(question=question, options=options_text)
        parts = [base, DECOMPOSE_ONLY_OUTPUT_BLOCK_MCQ]
    if extra_instructions:
        parts.append(extra_instructions)
    return "\n".join(parts)


# ==================================================================
# 2) Initial user message: keep the same "(no retrieval yet)" semantics as the ledger mode
#    but only attach the Current Focus, not the task list / status.
# ==================================================================
INITIAL_USER_DECOMPOSE_ONLY = """(no retrieval yet) You have not issued any search. Please start with [Search]. Do NOT output [Answer] in this turn.

[Current Focus] {focus_id}: {focus_question}
Your search this round should primarily serve this sub-question.

Output your Reason, Action, and Content.
"""


# ==================================================================
# 3) Agent body
# ==================================================================
class DecomposeOnlyMultiRoundAgent(MultiRoundSearchAgent):
    """An Agent that only decomposes and does not inject the Ledger.

    Reuses the parent class's infrastructure such as _generate / _execute_search / guardrails / sufficiency /
    _result_formatter / the visual layer / conversation_mgr.

    Key state (reset on each answer()):
      - self._current_ledger: serves only as the progress counter for the code-layer scheduler;
        its evidence / notes / global_notes and other fields are not injected into the prompt.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ledger_cfg = config.get("ledger", {}) or {}
        self._do_enabled = bool(ledger_cfg.get("enabled", True))

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

        self._current_ledger: Optional[TaskLedger] = None

        # P1: decomposition strategy guidance (dynamically injected by the orchestrator's self-evolving component),
        # aligned with the ledger agent to facilitate reusing the same self-evolution mechanism.
        self._decompose_guidance: str = ""

    def set_decompose_guidance(self, guidance: str) -> None:
        self._decompose_guidance = guidance or ""

    # ------------------------------------------------------------------
    # Planner adapter: completely identical to LedgerAwareMultiRoundAgent
    # ------------------------------------------------------------------
    def _planner_generate(self, messages, temperature=0.2, max_tokens=8192,
                          timeout=60, **_ignored):
        saved_t, saved_m = self.temperature, self.max_tokens
        try:
            self.temperature = temperature
            self.max_tokens = max_tokens
            self._json_mode = True
            return self._generate(messages, timeout=timeout)
        finally:
            self.temperature = saved_t
            self.max_tokens = saved_m
            self._json_mode = False

    # ==================================================================
    # Main entry point
    # ==================================================================
    def answer(self, question: TemporalQuestion,
               memory: MemoryStrategy) -> AgentAnswer:
        if not self._do_enabled:
            return super().answer(question, memory)

        is_open_ended = not question.options
        option_text = "\n".join(question.options) if question.options else ""

        # ---- Step 0: reset subsystems ----
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
            logger.error(f"[DecomposeOnly] Planner fatal exception; falling back to the parent answer: {exc}")
            return super().answer(question, memory)
        self._current_ledger = ledger

        logger.info(
            f"[DecomposeOnly] subtasks={list(ledger.subtasks.keys())} "
            f"(decomposed only, no ledger injection)"
        )

        # ---- Step 2: build the system prompt (without Ledger awareness) ----
        sys_prompt = build_decompose_only_system_prompt(
            question=question.question,
            options_text=option_text,
            is_open_ended=is_open_ended,
            visual_layer_enabled=keyframe_available,
            extra_instructions=self._extra_instructions or "",
        )

        # ---- Step 3: build the initial user message (carrying only the Current Focus) ----
        first_subtask = ledger.next_pending(
            max_attempts=self._max_attempts_per_subtask
        )
        if first_subtask is None:
            # Extreme case: decomposition failed, fall back to a single sub-question
            first_focus_id = "t1"
            first_focus_q = question.question
        else:
            first_focus_id = first_subtask.id
            first_focus_q = first_subtask.question
            if first_subtask.status == "pending":
                first_subtask.transition("searching", reason="scheduler_dispatch")

        initial_user = INITIAL_USER_DECOMPOSE_ONLY.format(
            focus_id=first_focus_id,
            focus_question=first_focus_q,
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
        active_subtask_id: Optional[str] = first_focus_id
        active_subtask_question: str = first_focus_q
        # Record focus_labels whose summary has already been registered, so repeated focuses take the diff-only path,
        # consistent with the ledger agent's B1 cache; this is pure deduplication and does not involve ledger injection.
        seen_focus_labels: Set[str] = set()
        seen_vg_items_by_focus: Dict[str, Set[str]] = {}

        # ==================================================================
        # Main loop
        # ==================================================================
        for round_idx in range(self.max_rounds):
            is_last_round = (round_idx == self.max_rounds - 1)

            self._runtime_state["rounds_so_far"] = round_idx
            on_round_hook = getattr(self, "_on_round_start_hook", None)
            if callable(on_round_hook):
                try:
                    on_round_hook(round_idx, self._runtime_state)
                except Exception as exc:
                    logger.warning(f"[DecomposeOnly] on_round_start_hook exception: {exc}")

            budget_exceeded, budget_reason = self._guardrails.check_budget()
            if budget_exceeded and not is_last_round:
                logger.info(f"[DecomposeOnly] budget exceeded; force final round: {budget_reason}")
                is_last_round = True

            # --------------------------------------------------------------
            # Last round: Synthesis (without ledger_final_state)
            # --------------------------------------------------------------
            if is_last_round:
                synth_content = self._build_synthesis_prompt(
                    question=question,
                    option_text=option_text,
                    is_open_ended=is_open_ended,
                    all_search_results=all_search_results,
                )
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
                )

            # --------------------------------------------------------------
            # Regular round
            # --------------------------------------------------------------
            # 0. This round's focus must stay consistent with the [Current Focus] exposed to the model
            #    in the previous user message, and attempts are also attributed to the same subtask.
            #    Otherwise, when multiple subtasks have no dependencies, next_pending would prefer a new pending
            #    subtask, causing the prompt focus to be misaligned with the code-side bookkeeping.
            scheduled_st = (
                ledger.subtasks.get(active_subtask_id)
                if active_subtask_id else None
            )
            scheduled_id = scheduled_st.id if scheduled_st else None

            conversations = self._conversation_mgr.manage(conversations)

            response = self._generate(conversations)
            conversations.append({"role": "assistant", "content": response})

            action, content = self._validated_parse(
                response, question.options, conversations,
                require_prior_search=is_open_ended,
            )

            # ---- [Answer]: return directly ----
            if action == "Answer":
                return self._make_final_answer(
                    content=content or "",
                    confidence=1.0,
                    all_search_results=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    conversations=conversations,
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
                all_search_results.append({
                    "query": content,
                    "retrieval": {"event_retrieval": retrieval_payload},
                })
                is_empty = not retrieval_payload.get("focus")

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
                self._runtime_state["last_focus_label"] = (
                    retrieval_payload.get("focus_label", "") or ""
                )
                sufficiency_hint = sufficiency.get("hint_message", "")
                if sufficiency.get("should_stop_searching", False):
                    self._guardrails.state.consecutive_empty_searches = (
                        self._guardrails.budget_config.max_consecutive_empty_searches
                    )
                    self._guardrails.state.grace_round_given = True

                # attempts count (used only for scheduler advancement, not written into the prompt)
                target_st = (
                    ledger.subtasks.get(scheduled_id) if scheduled_id
                    else None
                )
                if target_st is not None:
                    target_st.attempts += 1
                    if is_empty:
                        target_st.record_failed_query(content)
                    # When attempts exceed the limit, abandon directly so the scheduler switches to the next subtask;
                    # no replan / cascade is introduced, keeping the simple "decompose only" semantics as much as possible.
                    if target_st.attempts >= self._max_attempts_per_subtask:
                        if target_st.status not in ("resolved", "partial", "abandoned"):
                            target_st.transition(
                                "abandoned",
                                reason="max_attempts_reached",
                            )
            else:
                sufficiency_hint = ""

            # ---- Build the next round's user message: retrieval text + only the Current Focus ----
            focus_label_new = (
                retrieval_payload.get("focus_label") or "" if content else ""
            )
            focus_is_fresh = bool(focus_label_new) and (
                focus_label_new not in seen_focus_labels
            )

            if content:
                if is_empty:
                    search_result_text = self._result_formatter.format_empty_result(
                        content,
                        strategy_hint=(
                            "Try a different search strategy: different keywords, "
                            "VIDEO:/NEIGHBOR: prefix, or rephrase the query."
                        ),
                    )
                elif not focus_is_fresh and focus_label_new:
                    score = retrieval_payload.get("event_score")
                    try:
                        score_str = (
                            f" (event_score={float(score):.2f})"
                            if score is not None else ""
                        )
                    except Exception:
                        score_str = ""
                    search_result_text = (
                        f"Searched knowledge: focus unchanged, still at "
                        f"\"{focus_label_new}\"{score_str}. The event summary has "
                        f"already been shown earlier in this conversation; no new "
                        f"textual evidence was added this round."
                    )
                    # Retain the videograph_hits differences for the same focus
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
                                + "\n".join(new_vg_lines[:20])
                            )
                    if sufficiency_hint:
                        search_result_text += f"\n{sufficiency_hint}"
                else:
                    search_result_text = self._result_formatter.format_search_result(
                        retrieval_payload, content,
                        sufficiency_hint=sufficiency_hint,
                    )
                    if focus_label_new:
                        seen_focus_labels.add(focus_label_new)
                    vg_hits_first = retrieval_payload.get("videograph_hits", {})
                    if vg_hits_first and focus_label_new:
                        items_set: Set[str] = set()
                        for _items in vg_hits_first.values():
                            if isinstance(_items, list):
                                for _it in _items:
                                    items_set.add(str(_it).strip())
                        seen_vg_items_by_focus[focus_label_new] = items_set
            else:
                search_result_text = (
                    "Searched knowledge: (No valid query provided.)"
                )

            grace_hint = self._guardrails.state.pending_grace_hint
            if grace_hint:
                search_result_text += f"\n\n{grace_hint}"
                self._guardrails.state.pending_grace_hint = ""

            deferred = self._consume_pending_hints()
            if deferred:
                search_result_text = (search_result_text or "") + deferred

            # ---- Next round's focus (expose only the id + question text, not any ledger state) ----
            next_st = ledger.next_pending(
                max_attempts=self._max_attempts_per_subtask
            )
            if next_st is not None:
                next_focus_id = next_st.id
                next_focus_q = next_st.question
                if next_st.status == "pending":
                    next_st.transition("searching", reason="scheduler_dispatch")
            else:
                # All subtasks have reached a terminal state: still point to the previous focus's text,
                # letting the LLM know it can wrap up next round (the system does not force synthesis, leaving the closure to the budget/last round).
                next_focus_id = scheduled_id or first_focus_id
                next_focus_q = (
                    ledger.subtasks[scheduled_id].question
                    if scheduled_id and scheduled_id in ledger.subtasks
                    else active_subtask_question or first_focus_q
                )
            active_subtask_id = next_focus_id
            active_subtask_question = next_focus_q

            focus_block = (
                f"\n\n[Current Focus] {next_focus_id}: {next_focus_q}\n"
                "Your search this round should primarily serve this sub-question."
            )
            search_result_text = (search_result_text or "") + focus_block

            # keyframe marker (reuse the parent class's visual layer)
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

        # Theoretically should not be reached
        return self._make_final_answer(
            content="", confidence=0.0,
            all_search_results=all_search_results,
            search_queries=search_queries,
            num_rounds=self.max_rounds,
            conversations=conversations,
        )

    # ==================================================================
    # Synthesis prompt: directly feed all deduplicated retrieval results, without ledger_final_state
    # ==================================================================
    def _build_synthesis_prompt(self, question: TemporalQuestion,
                                option_text: str,
                                is_open_ended: bool,
                                all_search_results: List[Dict[str, Any]]) -> str:
        deduped_knowledge = self._deduplicate_and_format_retrievals(all_search_results)

        if is_open_ended:
            binary_hint = self._build_binary_choice_hint(question.question)
            return (
                "You are given an open-ended question and the COMPLETE deduplicated "
                "retrieved knowledge from all search rounds.\n\n"
                f"Question: {question.question}\n\n"
                "All Retrieved Knowledge (deduplicated by event):\n"
                f"{deduped_knowledge}\n\n"
                "Evidence utilisation rules (CRITICAL):\n"
                "  - If ANY event summary, videograph hit, or keyframe description "
                "in the retrieved knowledge contains an entity (a name, drink, fruit, "
                "location, time, job title, emotional state, action, object, etc.) "
                "that could plausibly answer the question, use THAT entity as the answer.\n"
                "  - When multiple candidates appear, answer with the most contextually "
                "relevant one, typically the first mentioned or the one most tied to "
                "the named person.\n"
                "  - For yes/no questions, answer \"Yes\" if ANY positive evidence "
                "exists; answer \"No\" only when evidence explicitly contradicts it.\n"
                "  - Do NOT reply with \"no evidence found\" or \"cannot be determined\" "
                "when relevant information exists in the retrieval.\n\n"
                "Output:\n"
                "    Reason: <your reasoning>\n"
                "    Action: [Answer]\n"
                "    Content: <concise free-form answer, one short phrase or a single sentence>\n"
                "  - Your answer MUST be definite. NEVER output \"unknown\", "
                "\"not specified\", or similar refusals.\n"
                "  - Use real character names (never `<character_0>`).\n"
                f"{binary_hint}"
            )
        else:
            return (
                "You are given a multiple-choice question and the COMPLETE deduplicated "
                "retrieved knowledge from all search rounds.\n\n"
                f"Question: {question.question}\nOptions:\n{option_text}\n\n"
                "All Retrieved Knowledge (deduplicated by event):\n"
                f"{deduped_knowledge}\n\n"
                "MCQ decision rules:\n"
                "  - Compare ALL options before choosing.\n"
                "  - Prefer direct evidence from event summaries, videograph hits, "
                "and keyframe descriptions.\n"
                "  - You MUST pick exactly one option with the strongest direct support "
                "after eliminating distractors.\n\n"
                "Output:\n"
                "    Reason: <your reasoning>\n"
                "    Action: [Answer]\n"
                "    Content: <option letter and content>\n"
                "  - Use real character names (never `<character_0>`).\n"
            )

    def _deduplicate_and_format_retrievals(
        self, all_search_results: List[Dict[str, Any]]
    ) -> str:
        """Deduplication and formatting logic consistent with LedgerAwareMultiRoundAgent.

        Group by focus_label:
          - For the same focus_label, keep only one focus summary
          - Take the union of videograph_hits
          - Deduplicate and merge neighbors / edges by label
          - Format into structured natural language using _result_formatter
        """
        if not all_search_results:
            return "(No retrieval results available.)"

        seen_labels: Dict[str, Dict[str, Any]] = {}
        label_order: List[str] = []
        query_by_label: Dict[str, List[str]] = {}

        for result in all_search_results:
            query = result.get("query", "")
            retrieval = result.get("retrieval", {})
            payload = retrieval.get("event_retrieval", retrieval)

            focus = payload.get("focus") or {}
            focus_label = str(payload.get("focus_label") or "")

            if not focus_label and not focus:
                continue

            dedup_key = focus_label
            if not dedup_key:
                dedup_key = (
                    str(focus.get("summary", ""))[:50]
                    or f"_unnamed_{len(seen_labels)}"
                )

            if dedup_key not in seen_labels:
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
                existing = seen_labels[dedup_key]

                new_vg = payload.get("videograph_hits") or {}
                for clip_key, items in new_vg.items():
                    if clip_key not in existing["videograph_hits"]:
                        existing["videograph_hits"][clip_key] = (
                            list(items) if isinstance(items, list) else [items]
                        )
                    else:
                        existing_items = set(
                            str(x) for x in existing["videograph_hits"][clip_key]
                        )
                        if isinstance(items, list):
                            for item in items:
                                if str(item) not in existing_items:
                                    existing["videograph_hits"][clip_key].append(item)
                                    existing_items.add(str(item))

                existing_nb_labels = set()
                for nb in existing["neighbors"]:
                    if isinstance(nb, dict):
                        existing_nb_labels.add(
                            nb.get("segment_label") or nb.get("label") or ""
                        )
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

                new_score = payload.get("event_score", 0)
                if new_score and new_score > (existing.get("event_score") or 0):
                    existing["event_score"] = new_score

                if query:
                    query_by_label[dedup_key].append(query)

        if not seen_labels:
            return "(No valid retrieval results after deduplication.)"

        parts: List[str] = []
        for dedup_key in label_order:
            payload = seen_labels[dedup_key]
            queries = query_by_label.get(dedup_key, [])
            formatted = self._result_formatter.format_search_result(
                payload, queries[0] if queries else "",
            )
            if len(queries) > 1:
                formatted += (
                    f"\n  (Retrieved by {len(queries)} queries: {queries[:4]})"
                )
            parts.append(formatted)

        return "\n\n---\n\n".join(parts)

    # ==================================================================
    # Result packaging
    # ==================================================================
    def _make_final_answer(self, content: str, confidence: float,
                           all_search_results: List[Dict[str, Any]],
                           search_queries: List[str],
                           num_rounds: int,
                           conversations: List[Dict[str, Any]]) -> AgentAnswer:
        """Construct an AgentAnswer.

        Unlike LedgerAwareMultiRoundAgent: attach only a lightweight plan_snapshot,
        not a ledger / progress snapshot, to avoid confusing the trajectory review.
        """
        trace = list(all_search_results)
        if self._current_ledger is not None:
            try:
                plan_snapshot = {
                    "_decompose_only_plan": {
                        "subtasks": [
                            {
                                "id": st.id,
                                "question": st.question,
                                "depends_on": list(st.depends_on or []),
                                "status": st.status,
                                "attempts": st.attempts,
                            }
                            for st in self._current_ledger.subtasks.values()
                        ],
                        "global_notes": list(
                            self._current_ledger.global_notes or []
                        ),
                    }
                }
                trace.append(plan_snapshot)
            except Exception as exc:
                logger.debug(f"[DecomposeOnly] plan snapshot serialization failed: {exc}")

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
