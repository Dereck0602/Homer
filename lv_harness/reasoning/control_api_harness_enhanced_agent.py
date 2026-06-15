"""
ControlApiHarnessEnhancedAgent: an enhanced harness wrapper around the control_api.py reasoning flow.

This file is used to replace the shortcomings of the first version control_api_harness_agent.py, without
changing run_homer_batch_evo.py. Core goals:
  1. Keep the main prompt of control_api.py and the Action/Search/Answer reasoning shape unchanged.
  2. Strictly align retrieval with the raw VideoGraph search semantics of control_api.py.
  3. Complete the full guardrails self-repair loop.
  4. Let the sufficiency increment signal genuinely drive runtime_state, supporting self-evolution deferred hints.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from mmagent.retrieve import search

from .control_api_harness_agent import (
    CONTROL_INSTRUCTION,
    CONTROL_INSTRUCTION_MCQ,
    CONTROL_INSTRUCTION_OPEN,
    CONTROL_LAST_ROUND_SUFFIX,
    CONTROL_LAST_ROUND_SUFFIX_MCQ,
    CONTROL_SYSTEM_PROMPT,
    CONTROL_SYSTEM_PROMPT_MCQ,
    CONTROL_SYSTEM_PROMPT_OPEN,
    ControlApiHarnessAgent,
)
from ..data.types import AgentAnswer, TemporalQuestion
from ..memory.base import MemoryStrategy

logger = logging.getLogger(__name__)


class ControlApiHarnessEnhancedAgent(ControlApiHarnessAgent):
    """An enhanced control_api harness agent.

    Compared with the first version:
      - The self-repair logic fully covers format errors, duplicate queries, premature answers, and final-round Search.
      - Retrieval preferentially uses memory.video_graph + mmagent.retrieve.search directly, strictly replicating control_api.py.
      - Raw memories are wrapped into a sufficiency payload, and the increment signal is written back into runtime_state.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._current_control_focus_label = ""

    # ------------------------------------------------------------------
    # Guardrails: full self-repair dedicated to the control_api format
    # ------------------------------------------------------------------
    def _validated_parse_control(
        self,
        response: str,
        options: List[str],
        conversations: List[Dict[str, Any]],
        is_last_round: bool = False,
        require_prior_search: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """Parse and validate a control_api format response, performing one self-repair on failure.

        It cannot directly call the parent _validated_parse, because the parent's retry prompt would introduce
        multi_round-style hints such as Reason/VIDEO/NEIGHBOR. This keeps the control_api
        Action/Content output constraints, but reuses the same guardrails state machine.
        """
        action, content, feedback = self._guardrails.validate_response(
            response,
            options,
            require_prior_search=require_prior_search,
        )

        if is_last_round and action == "Search":
            logger.info("[ControlHarnessEnhanced] final round output Search, forcing conversion to Answer")
            fallback = self._extract_answer_from_response(response, options)
            return "Answer", fallback or content or ""

        if not feedback:
            return action, content

        is_strategy_error = "already searched similar" in feedback.lower()
        logger.info(
            "[ControlHarnessEnhanced] guardrail triggered self-repair: %s | %s",
            "strategy" if is_strategy_error else "format",
            feedback[:120],
        )

        if is_last_round:
            retry_msg = (
                "Your previous response had an issue:\n"
                f"{feedback}\n\n"
                "This is the FINAL round. You MUST output exactly:\n"
                "Action: [Answer]\n"
                "Content: <the best option letter and option content, or concise answer>\n"
                "Do not output [Search]."
            )
        elif is_strategy_error:
            retry_msg = (
                "STRATEGY ISSUE:\n"
                f"{feedback}\n\n"
                "Please choose one:\n"
                "1. If the searched knowledge is enough, output Action: [Answer].\n"
                "2. Otherwise output Action: [Search] with a completely different query.\n\n"
                "Use exactly this format:\n"
                "Action: [Answer] or [Search]\n"
                "Content: <content>"
            )
        else:
            retry_msg = (
                "Your previous response had a format or policy issue:\n"
                f"{feedback}\n\n"
                "Please correct it using exactly this format:\n"
                "Action: [Answer] or [Search]\n"
                "Content: <content>"
            )

        retry_conversations = copy.copy(conversations)
        retry_conversations.append({"role": "user", "content": retry_msg})
        retry_response = self._generate(retry_conversations)

        action2, content2, feedback2 = self._guardrails.validate_response(
            retry_response,
            options,
            require_prior_search=require_prior_search,
        )

        if not feedback2:
            if is_last_round and action2 == "Search":
                fallback = self._extract_answer_from_response(retry_response, options)
                return "Answer", fallback or content2 or ""
            if conversations and conversations[-1].get("role") == "assistant":
                conversations[-1]["content"] = retry_response
            return action2, content2

        logger.warning(
            "[ControlHarnessEnhanced] self-repair failed, using fallback parsing: %s",
            feedback2[:120],
        )

        extracted = self._extract_answer_from_response(retry_response, options)
        if extracted:
            if conversations and conversations[-1].get("role") == "assistant":
                conversations[-1]["content"] = retry_response
            return "Answer", extracted

        extracted = self._extract_answer_from_response(response, options)
        if extracted:
            return "Answer", extracted

        fallback_action, fallback_content = self._parse_control_response(response)
        if is_last_round:
            return "Answer", fallback_content or ""

        if fallback_action == "Search" and fallback_content:
            # validate_response may already have rejected due to a duplicate query. If self-repair also fails, let the
            # original query through, but do not write it into guardrails.search_queries again, to avoid duplicate pollution.
            logger.info("[ControlHarnessEnhanced] fallback letting the original Search query through")
            return "Search", fallback_content

        return fallback_action or "Search", fallback_content

    # ------------------------------------------------------------------
    # Retrieval: strictly replicate the control_api.py raw VideoGraph search
    # ------------------------------------------------------------------
    @staticmethod
    def _is_control_character_query(query: str) -> bool:
        """Consistent with control_api.py, only the `character id` branch triggers mem_wise."""
        return "character id" in (query or "")

    @staticmethod
    def _strip_search_prefix(query: str) -> str:
        """Strip the VIDEO:/NEIGHBOR:/KEYFRAME: prefix.

        control_api.py does not support these prefixes; they come from the guardrails EMPTY RESULTS WARNING
        hint. If the model outputs these prefixes, they must be stripped before passing to search().
        """
        if not query:
            return query
        for prefix in ("VIDEO:", "NEIGHBOR:", "KEYFRAME:"):
            if query.upper().startswith(prefix):
                return query[len(prefix):].strip()
        return query

    @staticmethod
    def _sanitize_grace_hint(hint: str) -> str:
        """Replace the VIDEO:/NEIGHBOR: suggestions in the guardrails EMPTY RESULTS WARNING with generic suggestions.

        The control_api.py mode does not support the VIDEO:/NEIGHBOR:/KEYFRAME: prefixes, but the guardrails
        BudgetGuardrail will suggest using these prefixes in pending_grace_hint.
        Here they are replaced with generic suggestions applicable to the control_api mode, to avoid misleading the model.
        It does not modify guardrails.py itself, and does not affect other agent modes.
        """
        # Replace the suggestion lines related to VIDEO/NEIGHBOR/KEYFRAME
        import re
        # Remove the suggestion lines containing VIDEO:/NEIGHBOR:/KEYFRAME:
        lines = hint.split("\n")
        filtered = []
        for line in lines:
            if any(kw in line for kw in ("'VIDEO:", "'NEIGHBOR:", "'KEYFRAME:",
                                          "VIDEO:", "NEIGHBOR:", "KEYFRAME:")):
                continue
            filtered.append(line)
        result = "\n".join(filtered)
        # If after removal there is no concrete suggestion following "switch to a different approach:", add a generic one
        if "switch to a different approach" in result and result.strip().endswith(":"):
            result += "\n     - Use completely different keywords or rephrase your query"
        return result

    def _get_control_mem_node(self, memory: MemoryStrategy):
        """Get the current VideoGraph node, preferring the video_graph already loaded by memory.

        If video_graph exists but is empty (text_nodes is an empty list), it means on_video_start
        failed to successfully load the prebuilt VideoGraph; return None to trigger the fallback path.
        """
        mem_node = getattr(memory, "video_graph", None)
        if mem_node is not None:
            # Detect an empty VideoGraph: an empty text_nodes indicates load failure
            text_nodes = getattr(mem_node, "text_nodes", None)
            if text_nodes is not None and len(text_nodes) > 0:
                return mem_node
            # text_nodes is empty, try to reload
            logger.warning(
                "[ControlHarnessEnhanced] video_graph.text_nodes is empty, trying to reload"
            )

        mem_path = getattr(memory, "_mem_path", "") or ""
        load_fn = getattr(memory, "_load_video_graph", None)
        if mem_path and callable(load_fn):
            import os
            resolved = self._resolve_mem_path(mem_path)
            if resolved and os.path.exists(resolved):
                loaded = load_fn(resolved)
                # After a successful load, write back to memory to avoid repeated loading in later rounds
                if loaded is not None and hasattr(memory, "video_graph"):
                    memory.video_graph = loaded
                # Also correct _mem_path, to avoid later stats() and similar running into path problems again
                if resolved != mem_path and hasattr(memory, "_mem_path"):
                    memory._mem_path = resolved
                return loaded
        return None

    @staticmethod
    def _resolve_mem_path(path: str) -> str:
        """Return mem_path itself (placeholder implementation; can be replaced with actual path resolution logic as needed)."""
        return path

    def _execute_control_search(
        self,
        query: str,
        memory: MemoryStrategy,
        question: TemporalQuestion,
    ) -> Dict[str, Any]:
        """Execute retrieval equivalent to the control_api.py consumer.

        control_api.py logic:
          - mem_node = load_video_graph(data["mem_path"])
          - when before_clip is non-empty, truncate_memory_by_clip(before_clip, False)
          - refresh_equivalences()
          - "character id" in query -> search(..., [], mem_wise=True, topk=20)
          - otherwise -> search(..., data["currenr_clips"], threshold=0.5, topk=topk)
        """
        # Strip the VIDEO:/NEIGHBOR:/KEYFRAME: prefix (control_api.py does not support these prefixes)
        query = self._strip_search_prefix(query)
        if not query:
            return {}

        before_clip = question.before_clip
        mem_node = self._get_control_mem_node(memory)
        if mem_node is None:
            logger.warning(
                "[ControlHarnessEnhanced] no usable video_graph found, falling back to memory.retrieve"
            )
            result = memory.retrieve(
                query=query,
                top_k=self._control_topk,
                before_clip=before_clip,
            )
            return result.memories or {}

        # control_api.py truncates mem_node in place. lv_harness's search itself supports
        # before_clip filtering; to avoid polluting later questions on the same worker, here we only try to replicate on a clone.
        search_node = mem_node
        if before_clip is not None and hasattr(mem_node, "truncate_memory_by_clip"):
            try:
                search_node = copy.deepcopy(mem_node)
                search_node.truncate_memory_by_clip(before_clip, False)
            except Exception as exc:
                logger.debug(
                    "[ControlHarnessEnhanced] deepcopy/truncate failed, switching to before_clip parameter filtering: %s",
                    exc,
                )
                search_node = mem_node

        if hasattr(search_node, "refresh_equivalences"):
            search_node.refresh_equivalences()

        if self._is_control_character_query(query):
            memories, _, _ = search(
                search_node,
                query,
                [],
                mem_wise=True,
                topk=20,
                before_clip=before_clip,
            )
        else:
            current_clips = getattr(memory, "_current_clips", []) or []
            memories, new_current_clips, _ = search(
                search_node,
                query,
                current_clips,
                threshold=0.5,
                topk=self._control_topk,
                before_clip=before_clip,
            )
            if hasattr(memory, "_current_clips"):
                memory._current_clips = new_current_clips

        return memories or {}

    # ------------------------------------------------------------------
    # Sufficiency payload: wrap raw memories into a structure whose increment can be computed
    # ------------------------------------------------------------------
    @staticmethod
    def _memory_item_to_text(item: Any) -> str:
        if isinstance(item, dict):
            parts: List[str] = []
            for key in (
                "content", "text", "description", "summary", "label",
                "caption", "memory", "value",
            ):
                val = item.get(key)
                if val:
                    parts.append(str(val))
            if parts:
                return " | ".join(parts)
            return json.dumps(item, ensure_ascii=False, default=str)
        return str(item)

    def _build_control_sufficiency_payload(
        self,
        memories: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """Convert control_api raw memories into a payload consumable by SufficiencySignal."""
        if not memories:
            return {
                "mode": "control_api_raw",
                "focus": None,
                "focus_label": "",
                "event_score": 0.0,
                "videograph_hits": {},
            }

        clip_keys = [str(k) for k in memories.keys()]
        focus_label = "control_raw:" + ",".join(clip_keys[:8])
        self._current_control_focus_label = focus_label

        vg_hits: Dict[str, List[str]] = {}
        summary_parts: List[str] = []
        for key, value in memories.items():
            items = value if isinstance(value, list) else [value]
            text_items = [self._memory_item_to_text(item) for item in items]
            vg_hits[str(key)] = text_items
            summary_parts.extend(text_items[:5])

        summary = " ".join(summary_parts)[:1200]
        return {
            "mode": "control_api_raw",
            "focus": {
                "summary": summary,
                "segment_label": focus_label,
                "label": focus_label,
            },
            "focus_label": focus_label,
            "event_score": 0.8,
            "videograph_hits": vg_hits,
        }

    def _update_runtime_from_sufficiency(
        self,
        sufficiency: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> None:
        """Write the sufficiency increment signal back into runtime_state."""
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
            payload.get("focus_label", "") or ""
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def answer(
        self,
        question: TemporalQuestion,
        memory: MemoryStrategy,
    ) -> AgentAnswer:
        is_open_ended = not question.options

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
        self._current_control_focus_label = ""

        # Route to different prompts and instructions depending on whether it is a multiple-choice question
        if is_open_ended:
            # Open-ended QA (M3-Bench, etc.): use the original open-ended prompt
            sys_content = CONTROL_SYSTEM_PROMPT.format(
                question=question.question
            )
            instruction = CONTROL_INSTRUCTION
            last_round_suffix = CONTROL_LAST_ROUND_SUFFIX
        else:
            # Multiple-choice (Video-MME, etc.): use the MCQ prompt with Options
            option_text = "\n".join(question.options) if question.options else ""
            sys_content = CONTROL_SYSTEM_PROMPT_MCQ.format(
                question=question.question,
                options=option_text,
            )
            instruction = CONTROL_INSTRUCTION_MCQ
            last_round_suffix = CONTROL_LAST_ROUND_SUFFIX_MCQ
        if self._extra_instructions:
            sys_content += f"\n\n{self._extra_instructions}"

        conversations: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "Searched knowledge: {}"},
        ]

        all_search_results: List[Dict[str, Any]] = []
        search_queries: List[str] = []

        for round_idx in range(self.max_rounds):
            is_last_round = round_idx == self.max_rounds - 1

            self._runtime_state["rounds_so_far"] = round_idx
            on_round_hook = getattr(self, "_on_round_start_hook", None)
            if callable(on_round_hook):
                try:
                    on_round_hook(round_idx, self._runtime_state)
                except Exception as exc:
                    logger.warning(
                        "[ControlHarnessEnhanced] on_round_start_hook exception: %s",
                        exc,
                    )

            budget_exceeded, budget_reason = self._guardrails.check_budget()
            if budget_exceeded and not is_last_round:
                logger.info(
                    "[ControlHarnessEnhanced] budget/early-stop signal triggered the final round: %s",
                    budget_reason,
                )
                is_last_round = True

            conversations = self._conversation_mgr.manage(conversations)

            # Align with the open-source original: the instruction is permanently appended to the user message, with no rollback
            last_user_idx = None
            for idx in range(len(conversations) - 1, -1, -1):
                if conversations[idx].get("role") == "user":
                    last_user_idx = idx
                    break

            if last_user_idx is not None:
                conversations[last_user_idx]["content"] += instruction
                if is_last_round:
                    conversations[last_user_idx]["content"] += last_round_suffix
                deferred = self._consume_pending_hints()
                if deferred:
                    conversations[last_user_idx]["content"] += deferred

            response = self._generate(conversations)
            conversations.append({"role": "assistant", "content": response})

            action, content = self._validated_parse_control(
                response,
                question.options,
                conversations,
                is_last_round=is_last_round,
                require_prior_search=(is_open_ended and round_idx == 0),
            )

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

            memories: Dict[str, Any] = {}
            is_empty = True
            sufficiency_hint = ""

            if content:
                search_queries.append(content)
                memories = self._execute_control_search(content, memory, question)
                is_empty = len(memories) == 0

                payload = self._build_control_sufficiency_payload(memories, content)
                all_search_results.append({
                    "query": content,
                    "retrieval": {
                        "memories": memories,
                        "control_sufficiency_payload": payload,
                    },
                })

                if is_empty:
                    self._runtime_state["consecutive_empty_searches"] += 1
                    self._runtime_state["total_empty_searches"] += 1
                else:
                    self._runtime_state["consecutive_empty_searches"] = 0

                retrieval_text = json.dumps(
                    memories,
                    ensure_ascii=False,
                    default=str,
                )
                self._guardrails.on_search_result(
                    is_empty,
                    retrieval_text=retrieval_text[:2000],
                )

                sufficiency = self._sufficiency.assess_after_search(
                    payload,
                    content,
                    question.options,
                    round_idx,
                    self.max_rounds,
                    question_text=question.question,
                    keyframe_available=False,
                )
                self._update_runtime_from_sufficiency(sufficiency, payload)
                sufficiency_hint = sufficiency.get("hint_message", "")

                if sufficiency.get("should_stop_searching", False):
                    self._guardrails.state.consecutive_empty_searches = (
                        self._guardrails.budget_config.max_consecutive_empty_searches
                    )
                    self._guardrails.state.grace_round_given = True
                    logger.info(
                        "[ControlHarnessEnhanced] Sufficiency triggered an early-stop signal: consecutive_low=%s",
                        sufficiency.get("increment", {}).get("consecutive_low", 0),
                    )

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

            grace_hint = self._guardrails.state.pending_grace_hint
            if grace_hint:
                # The control_api mode does not support the VIDEO:/NEIGHBOR: prefixes, replace with generic suggestions
                grace_hint = self._sanitize_grace_hint(grace_hint)
                search_result_text += f"\n\n{grace_hint}"
                self._guardrails.state.pending_grace_hint = ""

            conversations.append({"role": "user", "content": search_result_text})

        return AgentAnswer(
            content="",
            confidence=0.0,
            is_final=True,
            reasoning_trace=all_search_results,
            search_queries=search_queries,
            num_rounds=self.max_rounds,
            tokens_used=self._guardrails.state.tokens_used,
            conversations=conversations,
        )
