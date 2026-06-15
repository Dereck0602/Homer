"""
ControlApiHarnessAgent: wraps the reasoning logic of control_api.py into an lv_harness Agent,
keeping the original prompt and retrieval approach unchanged while adding the harness mechanism.

Design principles:
  1. **Prompt unchanged**: use control_api.py's original system_prompt + instruction concatenation approach
  2. **Retrieval unchanged**: character id queries use mem_wise=True, others use clip_wise deduplication
  3. **Parsing unchanged**: use control_api.py's Action: [X] Content: Y regex parsing
  4. **Add the Harness mechanism**:
     - Guardrails: format validation + self-repair loop (duplicate query detection, format correction)
     - Sufficiency: information sufficiency assessment + early-stop signal
     - ConversationManager: context window management (overflow protection)
     - Budget: consecutive empty retrievals / token budget exceeded -> force the final round
     - Evolution compatibility: inject_instructions / inject_deferred_hint / _on_round_start_hook
  5. **Does not affect other agent modes**: standalone file, routed via the orchestrator's agent_type

Key differences from MultiRoundSearchAgent:
  - system_prompt uses control_api.py's original format (Question + Options in the system message)
  - the instruction is appended to the end of each round's user message (rather than given all at once in the system prompt)
  - the retrieval result format is "Searched knowledge: {json}" (rather than the structured format_search_result)
  - the forced-answer approach in the final round appends a hint to the end of the user message (rather than a separate LAST_ROUND_PROMPT)
"""
from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List, Optional, Set

from .multi_round import MultiRoundSearchAgent
from ..data.types import TemporalQuestion, AgentAnswer
from ..memory.base import MemoryStrategy

logger = logging.getLogger(__name__)


# ==================================================================
# control_api.py original prompt (aligned with the open-source original version)
# ==================================================================

# ---- Open-ended QA version (used by open-ended QA datasets such as M3-Bench) ----
CONTROL_SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason "
    "about whether the provided knowledge is sufficient to answer the question. "
    "If it is sufficient, output [Answer] followed by the answer. "
    "If it is not sufficient, output [Search] and generate a query that will be "
    "encoded into embeddings for a vector similarity search. The query will help "
    "retrieve additional information from a memory bank.\n\n"
    "Question: {question}"
)

# Keep the OPEN alias for backward compatibility with old references
CONTROL_SYSTEM_PROMPT_OPEN = CONTROL_SYSTEM_PROMPT

CONTROL_INSTRUCTION = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{i}>" or "What is the character id of {name}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {content} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{i}>."""

# Keep the OPEN alias for backward compatibility with old references
CONTROL_INSTRUCTION_OPEN = CONTROL_INSTRUCTION

# ---- Multiple-choice version (used by MCQ datasets such as Video-MME, aligned with control_api.py's original system_prompt) ----
CONTROL_SYSTEM_PROMPT_MCQ = (
    "You are given a question with multiple-choice options and some relevant knowledge. "
    "Your task is to reason about whether the provided knowledge is sufficient to answer "
    "the question. If it is sufficient, choose the correct option(s) and output [Answer] "
    "followed by the selected option letter(s) and their content. If it is not sufficient, "
    "output [Search] and generate a query that will be encoded into embeddings for a "
    "vector similarity search. The query will help retrieve additional information from "
    "a memory bank.\n\n"
    "Question: {question}\n"
    "Options: {options}"
)

CONTROL_INSTRUCTION_MCQ = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.

If the answer can be derived from the provided knowledge, the {content} should be the selected option letter(s) (e.g. A, B, C, or D) and the corresponding option content.

You can get the mapping relationship between character ID and name by using search query such as: 
"What is the name of <character_{i}>" or 
"What is the character id of {name}".

After obtaining the mapping, it is best to use character ID instead of name for searching.
Only name can appear in the final answer content, not character ID like <character_{i}>."""

CONTROL_LAST_ROUND_SUFFIX = (
    "\n(The Action of this round must be [Answer]. "
    "If there is insufficient information, you can make reasonable guesses.)"
)

CONTROL_LAST_ROUND_SUFFIX_MCQ = (
    "\n(The Action of this round must be [Answer]. "
    "If there is insufficient information, you can make reasonable guesses. "
    "Output the option letter and content.)"
)

# Parsing pattern (exactly the same as control_api.py)
CONTROL_ACTION_PATTERN = re.compile(
    r"Action:\s*\[?\s*(Answer|Search)\s*\]?\s*.*?Content:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


class ControlApiHarnessAgent(MultiRoundSearchAgent):
    """Wraps the reasoning logic of control_api.py into an lv_harness Agent.

    Inherits from MultiRoundSearchAgent to reuse:
      - _generate (LLM call + retry)
      - _guardrails (format validation + self-repair)
      - _sufficiency (information sufficiency assessment)
      - _conversation_mgr (context management)
      - inject_instructions / inject_deferred_hint / _on_round_start_hook (self-evolution compatibility)
      - _refresh_keyframe_preflight / visual layer (if enabled)

    Override answer() to implement control_api.py's reasoning flow.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # control_api.py uses topk=2 as the default retrieval count
        self._control_topk = int(config.get("control_topk", 2))

    # ------------------------------------------------------------------
    # Parsing method: aligned with control_api.py's consumer function
    # ------------------------------------------------------------------
    def _parse_control_response(self, response: str):
        """Parse a control_api.py-format response.

        Compatible with thinking mode (strips the content before </think>).
        Returns (action, content), where action is "Answer"/"Search"/None.
        """
        text = response.split("</think>")[-1] if "</think>" in response else response
        match = CONTROL_ACTION_PATTERN.search(text)
        if match:
            action = match.group(1).strip().capitalize()
            content = match.group(2).strip()
            if action not in ("Answer", "Search"):
                return "Search", None
            return action, content or None
        return "Search", None

    # ------------------------------------------------------------------
    # Retrieval method: aligned with control_api.py's consumer function
    # ------------------------------------------------------------------
    def _execute_control_search(
        self,
        query: str,
        memory: MemoryStrategy,
        question: TemporalQuestion,
    ) -> Dict[str, Any]:
        """Execute control_api.py-style retrieval.

        Logically identical to control_api.py's consumer function:
          - "character id" in query -> mem_wise=True, topk=20
          - others -> clip_wise (mem_wise=False), topk=self._control_topk

        Returns a memories dict (consistent with control_api.py's new_memories format).
        """
        before_clip = question.before_clip

        # Use the memory layer's retrieve method (video_drilldown mode)
        # This automatically goes through the character check + clip_wise deduplication logic in hierarchical.py
        result = memory.retrieve(
            query=query,
            before_clip=before_clip,
            mode="video_drilldown",
            focus_label=None,  # do not restrict focus, search globally
        )

        # Merge memories
        memories = result.memories or {}
        return memories

    # ==================================================================
    # Main entry: answer()
    # ==================================================================
    def answer(self, question: TemporalQuestion,
               memory: MemoryStrategy) -> AgentAnswer:
        """control_api.py-style multi-round reasoning with the harness mechanism added.

        Flow:
        1. Build the system prompt (Question + Options in the system message)
        2. Initial user message: "Searched knowledge: {}"
        3. Each round:
           a. Append the instruction to the end of the user message
           b. On the final round, append the forced-answer hint
           c. Generate the response -> parse Action/Content
           d. [Answer] -> return
           e. [Search] -> execute retrieval -> build the next round's user message
        4. The Harness mechanism is interleaved within each round:
           - guardrails: format validation + duplicate query detection + self-repair
           - sufficiency: information-increment assessment + early-stop signal
           - budget: consecutive empty retrievals / budget exceeded -> force the final round
           - conversation_mgr: context window management
        """
        is_open_ended = not question.options

        # ---- Reset subsystems ----
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

        # ---- Build the system prompt (route to a different prompt depending on whether it is multiple-choice) ----
        if is_open_ended:
            sys_content = CONTROL_SYSTEM_PROMPT.format(
                question=question.question
            )
            instruction = CONTROL_INSTRUCTION
            last_round_suffix = CONTROL_LAST_ROUND_SUFFIX
        else:
            option_text = "\n".join(question.options) if question.options else ""
            sys_content = CONTROL_SYSTEM_PROMPT_MCQ.format(
                question=question.question,
                options=option_text,
            )
            instruction = CONTROL_INSTRUCTION_MCQ
            last_round_suffix = CONTROL_LAST_ROUND_SUFFIX_MCQ
        # Extra instructions injected by self-evolution
        if self._extra_instructions:
            sys_content += f"\n\n{self._extra_instructions}"

        # ---- Initial user message (consistent with control_api.py) ----
        conversations: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "Searched knowledge: {}"},
        ]

        all_search_results: List[Dict[str, Any]] = []
        search_queries: List[str] = []

        # ==================================================================
        # Main loop
        # ==================================================================
        for round_idx in range(self.max_rounds):
            is_last_round = (round_idx == self.max_rounds - 1)

            # ---- Harness: runtime hook (staged injection for self-evolution) ----
            self._runtime_state["rounds_so_far"] = round_idx
            on_round_hook = getattr(self, "_on_round_start_hook", None)
            if callable(on_round_hook):
                try:
                    on_round_hook(round_idx, self._runtime_state)
                except Exception as exc:
                    logger.warning(
                        f"[ControlHarness] on_round_start_hook exception: {exc}"
                    )

            # ---- Harness: budget check ----
            budget_exceeded, budget_reason = self._guardrails.check_budget()
            if budget_exceeded and not is_last_round:
                logger.info(
                    f"[ControlHarness] Budget exceeded, forcing final round: {budget_reason}"
                )
                is_last_round = True

            # ---- Harness: context management (overflow protection) ----
            conversations = self._conversation_mgr.manage(conversations)

            # ---- Append the instruction to the last user message ----
            # Note: control_api.py's approach is to append the instruction to the end of the user message every round.
            # Here we append it temporarily before generation and restore it afterward (to avoid the instruction accumulating repeatedly in the conversation history)
            last_user_idx = None
            for i in range(len(conversations) - 1, -1, -1):
                if conversations[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                original_content = conversations[last_user_idx]["content"]
                augmented_content = original_content + instruction
                if is_last_round:
                    augmented_content += last_round_suffix
                # P2 staged injection: append the skill hint
                deferred = self._consume_pending_hints()
                if deferred:
                    augmented_content += deferred
                conversations[last_user_idx]["content"] = augmented_content

            # ---- Generate the response ----
            response = self._generate(conversations)
            conversations.append({"role": "assistant", "content": response})

            # ---- Restore the user message (remove the temporarily appended instruction) ----
            if last_user_idx is not None:
                conversations[last_user_idx]["content"] = original_content

            # ---- Parse the response ----
            action, content = self._parse_control_response(response)

            # ---- Harness: format validation + self-repair ----
            # Validate using guardrails (duplicate query detection, format checking)
            _, _, feedback = self._guardrails.validate_response(
                response, question.options,
                require_prior_search=(is_open_ended and round_idx == 0),
            )

            if feedback and action == "Search":
                # Strategy/format error: attempt self-repair
                is_strategy_error = "already searched similar" in feedback.lower()
                if is_strategy_error and not is_last_round:
                    # Duplicate query: prompt to change strategy
                    retry_msg = (
                        f"STRATEGY ISSUE: {feedback}\n"
                        "Please use a COMPLETELY DIFFERENT search query, "
                        "or output [Answer] if you have enough information."
                    )
                    retry_convs = conversations.copy()
                    retry_convs.append({"role": "user", "content": retry_msg})
                    retry_response = self._generate(retry_convs)
                    retry_action, retry_content = self._parse_control_response(
                        retry_response
                    )
                    if retry_action and retry_content:
                        action, content = retry_action, retry_content
                        # Replace with the repaired response
                        conversations[-1]["content"] = retry_response

            # ---- Final round: force Answer ----
            if is_last_round and action == "Search":
                logger.info(
                    "[ControlHarness] Still Search on the final round, forcing it to Answer"
                )
                # Try to extract the answer from the response
                fallback = self._extract_answer_from_response(
                    response, question.options
                )
                if fallback:
                    action, content = "Answer", fallback
                else:
                    action, content = "Answer", content or ""

            # ---- [Answer]: return the final answer ----
            if action == "Answer":
                return AgentAnswer(
                    content=content or "",
                    confidence=1.0 if not is_last_round else 0.6,
                    is_final=True,
                    reasoning_trace=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    tokens_used=self._guardrails.state.tokens_used,
                    conversations=conversations,
                )

            # ---- [Search]: execute retrieval ----
            memories: Dict[str, Any] = {}
            is_empty = True

            if content:
                search_queries.append(content)
                memories = self._execute_control_search(
                    content, memory, question
                )
                is_empty = len(memories) == 0

                all_search_results.append({
                    "query": content,
                    "retrieval": {"memories": memories},
                })

                # ---- Harness: update runtime state ----
                if is_empty:
                    self._runtime_state["consecutive_empty_searches"] += 1
                    self._runtime_state["total_empty_searches"] += 1
                else:
                    self._runtime_state["consecutive_empty_searches"] = 0

                self._guardrails.on_search_result(
                    is_empty,
                    retrieval_text=json.dumps(
                        memories, ensure_ascii=False, default=str
                    )[:2000],
                )

                # ---- Harness: information sufficiency assessment ----
                # Construct a payload compatible with the sufficiency interface
                sufficiency_payload = {
                    "focus": memories if memories else None,
                    "focus_label": f"round_{round_idx}",
                    "event_score": 0.8 if memories else 0.0,
                }
                sufficiency = self._sufficiency.assess_after_search(
                    sufficiency_payload, content, question.options,
                    round_idx, self.max_rounds,
                    question_text=question.question,
                    keyframe_available=False,
                )
                sufficiency_hint = sufficiency.get("hint_message", "")

                # ---- Harness: early-stop signal ----
                if sufficiency.get("should_stop_searching", False):
                    self._guardrails.state.consecutive_empty_searches = (
                        self._guardrails.budget_config.max_consecutive_empty_searches
                    )
                    self._guardrails.state.grace_round_given = True
                    logger.info(
                        "[ControlHarness] Sufficiency triggered the early-stop signal"
                    )

            # ---- Build the next round's user message (control_api.py format) ----
            search_result_text = (
                "Searched knowledge: "
                + json.dumps(memories, ensure_ascii=False)
                    .encode("utf-8", "ignore")
                    .decode("utf-8")
            )
            if is_empty:
                search_result_text += (
                    "\n(The search result is empty. "
                    "Please try searching from another perspective.)"
                )
            elif sufficiency_hint:
                search_result_text += f"\n{sufficiency_hint}"

            # Harness: grace-round hint
            grace_hint = self._guardrails.state.pending_grace_hint
            if grace_hint:
                search_result_text += f"\n\n{grace_hint}"
                self._guardrails.state.pending_grace_hint = ""

            conversations.append({"role": "user", "content": search_result_text})

        # In theory this should not be reached
        return AgentAnswer(
            content="", confidence=0.0, is_final=True,
            reasoning_trace=all_search_results,
            search_queries=search_queries,
            num_rounds=self.max_rounds,
            conversations=conversations,
        )
