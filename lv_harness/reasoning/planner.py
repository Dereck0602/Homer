"""
QuestionPlanner: the decomposer for the TaskLedger (the Plan stage of Plan-and-Solve).

Responsibilities:
  - Input: the original question + options (may be empty for MCQ) + LLM client (reuses the agent's client)
  - Output: an initialized TaskLedger (with 1-5 SubTasks, dependencies, and rationale injected into global_notes)

Design principles (for zero-intrusion integration with MultiRoundSearchAgent):
  1. This module does not import any agent implementation; it only depends on task_ledger + ledger_prompts + the standard library.
     => It can be unit-tested independently and reused under any LLM backend.
  2. The Planner is a one-time cold-start call. On failure it must gracefully degrade to "single SubTask = original question",
     and must never let a decomposition failure block the main flow (echoing the user's preference: hard code-level Guardrail > relying on model reliability).
  3. Supports injecting an external `generate_fn` ((messages, **kwargs) -> str), letting the caller fully control
     temperature / max_tokens / retry, avoiding coupling with the agent's _generate.
  4. Lenient JSON output parsing: supports code-fence ```json```, bare JSON, and JSON fragments wrapped in prose.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .task_ledger import SubTask, TaskLedger
from .ledger_prompts import DECOMPOSE_PROMPT_MCQ, DECOMPOSE_PROMPT_OPEN, REPLAN_PROMPT

logger = logging.getLogger(__name__)

# ---- json-repair: soft dependency (the industry-standard LLM JSON repair library, specialized for truncation/trailing commas/missing closing braces) ----
# Once imported successfully, the lowest-priority 4th fallback path is enabled; if the import fails, the original 3 paths are kept.
try:
    from json_repair import repair_json as _repair_json  # type: ignore
    _HAS_JSON_REPAIR = True
except Exception:  # pragma: no cover - fall back to the pure standard library when not installed
    _repair_json = None
    _HAS_JSON_REPAIR = False
    logger.info(
        "[Planner] json-repair not installed, using only the standard JSON parsing path; "
        "consider `pip install json-repair` to repair truncated/malformed LLM output."
    )


# ---- JSON extraction: compatible with multiple LLM output formats ----
# Note: both _JSON_BLOCK_RE and _BARE_JSON_RE require a closing '}', so they fail once the output is truncated by max_tokens;
# hence the added 4th json-repair fallback to handle the Gemini/Qwen truncation case.
_JSON_BLOCK_RE = re.compile(
    r"```(?:json|ledger)?\s*\n(\{.*?\})\s*\n```",
    re.DOTALL | re.IGNORECASE,
)
_BARE_JSON_RE = re.compile(r"(\{[\s\S]*\})", re.DOTALL)
# unclosed code fence: only a leading ```json/```ledger, no trailing ```; used for slicing in the truncation case
_UNCLOSED_FENCE_RE = re.compile(
    r"```(?:json|ledger)?\s*\n([\s\S]*)",
    re.IGNORECASE,
)


def _try_json_repair(candidate: str) -> Optional[Dict[str, Any]]:
    """Use json-repair to fix an incomplete fragment; return a dict or None.

    - The input is expected to be a fragment that looks like a JSON object (containing a leading '{').
    - On failure all exceptions are swallowed, ensuring the caller gets None and does not crash.
    """
    if not _HAS_JSON_REPAIR or not candidate:
        return None
    try:
        fixed = _repair_json(candidate, return_objects=True)
    except Exception as exc:
        logger.debug(f"[Planner] json-repair fix exception: {exc}")
        return None
    if isinstance(fixed, dict):
        return fixed
    # json-repair may also return list/str/empty dict; only a non-empty dict is accepted here
    return None


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first valid JSON object from the LLM response.

    Parsing paths (by priority):
      1) JSON wrapped in a strict code-fence (```json ... ```)
      2) the largest fragment from the first '{' to the last '}'
      3) greedy regex fallback (complete JSON)
      4) **json-repair truncation fix** (recoverable even if the output was truncated by max_tokens)
    """
    if not text:
        return None
    # 1) code fence first
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception as exc:
            logger.debug(f"[Planner] code-fence JSON parse failed: {exc}")
    # 2) bare JSON: the largest fragment from the first '{' to the last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 3) greedy fallback
    m = _BARE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception as exc:
            logger.debug(f"[Planner] bare JSON parse failed: {exc}")

    # 4) json-repair fallback: first try the fragment inside an unclosed fence, then try the whole segment from the first '{'
    if _HAS_JSON_REPAIR:
        # 4a) unclosed ```json / ```ledger fence (a typical truncation case)
        m = _UNCLOSED_FENCE_RE.search(text)
        if m:
            inner = m.group(1)
            # cut off any reversed ``` fence that may remain at the tail, leaving json-repair a pure JSON-ish string
            inner = inner.split("```", 1)[0]
            obj = _try_json_repair(inner)
            if obj:
                logger.info("[Planner] recovered JSON via json-repair from an unclosed code-fence")
                return obj
        # 4b) slice directly from the first '{' to the end and let json-repair handle it
        if first >= 0:
            obj = _try_json_repair(text[first:])
            if obj:
                logger.info("[Planner] recovered JSON via json-repair from a bare fragment")
                return obj

    return None


@dataclass
class PlannerConfig:
    """Planner behavior configuration (all with conservative defaults)."""
    enabled: bool = True                 # when off, decompose goes straight to fallback (single SubTask)
    max_subtasks: int = 5                # upper bound (hard constraint, truncated if exceeded)
    min_subtasks: int = 1                # lower bound (at least 1, i.e. the original question)
    temperature: float = 0.2             # the decomposition stage should be stable, low temperature
    # Note: Gemini 2.5 flash outputs 3-4 subtasks + rationale, often >1k tokens, easily truncated.
    # Set to 4096 to lower the truncation probability; the retry path relaxes it once more by x1.5.
    max_tokens: int = 8192
    timeout: int = 60
    # fallback behavior: when the LLM fails / returns empty
    #   "atomic": use the original question as a single SubTask
    #   "fail":   raise an exception (for debug only)
    on_failure: str = "atomic"
    # question length threshold: questions that are too short (< N chars) skip decomposition and go straight to atomic
    skip_decompose_short_chars: int = 12
    # ---- retry on parse failure (introduced in F4) ----
    # Whether to retry the LLM call once when _extract_json_obj returns None.
    # The retry attaches "the first 400 chars of the previous output" as reflection, guiding the model to output more compact JSON.
    retry_on_parse_fail: bool = True
    # the multiplier (rounded up) for max_tokens on retry, giving the model more room to complete the JSON
    retry_max_tokens_multiplier: float = 1.5
    # P1: short questions with multi-step/multi-detail signals no longer go straight to atomic; the code-level guardrail forces them into the Planner.
    allow_short_multisignal_decompose: bool = True


# user-defined LLM call signature: (messages, temperature, max_tokens, timeout) -> str
GenerateFn = Callable[..., str]


class QuestionPlanner:
    """One-time question decomposer.

    Typical usage (called once at the start of LedgerAwareMultiRoundAgent.answer):

        planner = QuestionPlanner(generate_fn=self._generate, config=PlannerConfig())
        ledger = planner.decompose(question.question, question.options)

    Args:
        generate_fn:  (messages, temperature=..., max_tokens=..., timeout=...) -> str
                       compatible with the signature of MultiRoundSearchAgent._generate.
        config:       PlannerConfig; uses the default when None.
    """

    def __init__(self, generate_fn: GenerateFn, config: Optional[PlannerConfig] = None):
        self._generate = generate_fn
        self.config = config or PlannerConfig()
        # store the rationale of the most recent decomposition, avoiding the unreliable hack of setattr on a list object
        self._last_rationale: str = ""

    # ------------------------------------------------------------------
    # main external interface
    # ------------------------------------------------------------------
    def decompose(self, question: str, options: Optional[List[str]] = None,
                  decompose_guidance: str = ""
                  ) -> TaskLedger:
        """Decompose the original question into a TaskLedger. Guarantees a non-empty return (on failure, falls back to atomic).

        Args:
            question: the original question text
            options: the MCQ options list (pass empty for open-ended QA)
            decompose_guidance: P1 dynamically injected decomposition-strategy guidance (from historical experience)
        """
        options = options or []
        ledger = TaskLedger(question=question, options=list(options))
        # reset rationale on each decompose, to avoid leftover from the first question leaking into later questions
        self._last_rationale = ""
        # P1: save guidance for _call_llm_decompose to use
        self._decompose_guidance = decompose_guidance
        must_consider_decompose = self._should_force_decompose(
            question, options, decompose_guidance
        )

        # short-circuit 1: Planner is disabled
        if not self.config.enabled:
            self._fallback_atomic(ledger, reason="planner_disabled")
            return ledger

        # short-circuit 2: the question is too short, no decomposition needed
        if (
            len(question.strip()) < self.config.skip_decompose_short_chars
            and not must_consider_decompose
        ):
            self._fallback_atomic(ledger, reason="question_too_short")
            return ledger

        # normal path: call the LLM to decompose
        try:
            subtasks = self._call_llm_decompose(question, options, self._decompose_guidance)
        except Exception as exc:
            logger.warning(f"[Planner] LLM decomposition exception, falling back to atomic: {exc}")
            subtasks = []

        if not subtasks:
            if self.config.on_failure == "fail":
                raise RuntimeError("[Planner] decomposition failed and on_failure='fail'")
            self._fallback_atomic(ledger, reason="llm_returned_empty")
            return ledger

        subtasks = self._apply_decomposition_guardrails(
            subtasks, question, must_consider_decompose
        )
        if not subtasks:
            if self.config.on_failure == "fail":
                raise RuntimeError("[Planner] decomposition was judged empty by the guardrail and on_failure='fail'")
            self._fallback_atomic(ledger, reason="guardrail_rejected_decompose")
            return ledger

        # truncate to max_subtasks
        subtasks = subtasks[: self.config.max_subtasks]

        # build SubTasks and append to the ledger
        added_ids: List[str] = []
        for st_dict in subtasks:
            sid = str(st_dict.get("id") or "").strip()
            q = str(st_dict.get("question") or "").strip()
            if not sid or not q:
                continue
            # filter illegal dependencies: keep only existing ids (declaration order guarantees forward dependencies)
            deps_raw = st_dict.get("depends_on") or []
            deps = [str(d) for d in deps_raw if str(d) in added_ids]
            ledger.add_subtask(SubTask(id=sid, question=q, depends_on=deps))
            added_ids.append(sid)

        # fallback: if nothing was added after filtering, fall back to atomic
        if not ledger.subtasks:
            self._fallback_atomic(ledger, reason="all_subtasks_invalid")

        # put rationale into global_notes, for later synthesis / debug tracing
        rationale = self._last_rationale
        if rationale:
            ledger.add_global_note(f"[plan] {rationale}")

        logger.info(
            f"[Planner] decomposition complete: {len(ledger.subtasks)} subtasks "
            f"(ids={list(ledger.subtasks.keys())})"
        )
        return ledger

    # ------------------------------------------------------------------
    # code-level decomposition-decision Guardrails
    # ------------------------------------------------------------------
    def _should_force_decompose(
        self,
        question: str,
        options: List[str],
        decompose_guidance: str,
    ) -> bool:
        """Recognize cases of "short but should be decomposed / historical experience requires decomposition", avoiding a short-circuit before the prompt."""
        if not self.config.allow_short_multisignal_decompose:
            return False
        guidance = (decompose_guidance or "").lower()
        if any(
            marker in guidance
            for marker in (
                "do not keep atomic",
                "decompose actively",
                "under-decomposition",
                "multi-step or multi-detail",
                "require at least 2 subtasks",
            )
        ):
            return True
        q = (question or "").strip().lower()
        multi_patterns = [
            r"\b(before|after|then|next|first|last)\b",
            r"\bhow many\b|\bnumber of\b|\bcount\b",
            r"\bwhy\b",
            r"\b(compare|different|same|more than|less than)\b",
            r"\b(and|both|each|respectively)\b",
            r"\bwho\b.*\b(what|where|when|why|how)\b",
        ]
        if any(re.search(p, q) for p in multi_patterns):
            return True
        # MCQ options themselves are not a forced-decomposition signal; they are only passed to the Planner as prompt content.
        return False

    def _apply_decomposition_guardrails(
        self,
        subtasks: List[Dict[str, Any]],
        question: str,
        must_consider_decompose: bool,
    ) -> List[Dict[str, Any]]:
        """Apply lightweight hard constraints to the LLM decomposition result, reducing invalid decomposition and missed decomposition."""
        cleaned: List[Dict[str, Any]] = []
        seen_questions = set()
        original = (question or "").strip()
        for st in subtasks:
            if not isinstance(st, dict):
                continue
            sid = str(st.get("id") or "").strip()
            q = str(st.get("question") or "").strip()
            if not sid or not q:
                continue
            # prevent truncated/meaningless sub-questions from polluting the ledger, e.g. "The number of ... in the"
            if len(q.split()) < 3 and q.lower() != original.lower():
                continue
            norm = re.sub(r"\s+", " ", q.lower()).strip(" ?.!。")
            if norm in seen_questions:
                continue
            seen_questions.add(norm)
            cleaned.append(st)

        if not cleaned:
            return []

        # If the code/historical experience forces the view that decomposition should be considered, but the LLM still only gave the atomic original question,
        # generate a minimal binary split plan to prevent "should decompose but does not" from continuing to slip through.
        if must_consider_decompose and len(cleaned) == 1:
            only_q = str(cleaned[0].get("question") or "").strip()
            if self._is_atomic_echo(only_q, original):
                return self._fallback_minimal_split(original)

        # When over-decomposed and every subtask is just a rewrite of the original question, collapse back to atomic to reduce invalid decomposition.
        if len(cleaned) > 2:
            echo_count = sum(
                1 for st in cleaned
                if self._is_atomic_echo(str(st.get("question") or ""), original)
            )
            if echo_count >= len(cleaned) - 1:
                return [{"id": "t1", "question": original, "depends_on": []}]

        return cleaned

    @staticmethod
    def _is_atomic_echo(subtask_question: str, original_question: str) -> bool:
        """Determine whether the sub-question is just a repeat of the original question."""
        sq = re.sub(r"\W+", " ", (subtask_question or "").lower()).strip()
        oq = re.sub(r"\W+", " ", (original_question or "").lower()).strip()
        if not sq or not oq:
            return False
        if sq == oq:
            return True
        s_words = set(sq.split())
        o_words = set(oq.split())
        if not s_words or not o_words:
            return False
        overlap = len(s_words & o_words) / max(1, len(s_words))
        return overlap >= 0.85 and abs(len(s_words) - len(o_words)) <= 3

    @staticmethod
    def _fallback_minimal_split(question: str) -> List[Dict[str, Any]]:
        """A minimal retrievable binary split for when history/rules force decomposition but the LLM outputs atomic."""
        q = (question or "").strip()
        return [
            {
                "id": "t1",
                "question": f"Find the direct evidence needed to answer: {q}",
                "depends_on": [],
            },
            {
                "id": "t2",
                "question": f"Verify any missing entity, count, time, or relation before final answering: {q}",
                "depends_on": ["t1"],
            },
        ]

    # ------------------------------------------------------------------
    # internal: LLM call + JSON parsing
    # ------------------------------------------------------------------
    def _call_llm_decompose(self, question: str, options: List[str],
                            decompose_guidance: str = ""
                            ) -> List[Dict[str, Any]]:
        """Call the LLM (retrying once if necessary), returning a subtasks list (possibly empty).

        Retry strategy (enabled only when retry_on_parse_fail=True):
          - 1st: use the full prompt + the configured max_tokens
          - if _extract_json_obj returns None:
              2nd: attach "the first 400 chars of the previous output" + a compactness hint, max_tokens x1.5
          - still failing => return [] (decompose falls back to atomic)
        """
        is_mcq = bool(options)
        if is_mcq:
            option_text = "\n".join(options)
            base_prompt = DECOMPOSE_PROMPT_MCQ.format(
                question=question, options=option_text
            )
        else:
            base_prompt = DECOMPOSE_PROMPT_OPEN.format(question=question)

        # P1: inject the historical-experience-driven decomposition-strategy guidance
        if decompose_guidance:
            base_prompt += f"\n\n{decompose_guidance}"

        # 1st call
        raw = self._invoke_llm(
            system="You are a careful task planner.",
            user=base_prompt,
            max_tokens=self.config.max_tokens,
        )
        obj = _extract_json_obj(raw) if raw else None

        # 2nd (retry): only when the 1st parse failed and retry is enabled
        if obj is None and self.config.retry_on_parse_fail:
            prev_snippet = (raw or "").strip()[:400]
            retry_prompt = (
                base_prompt
                + "\n\n---\nIMPORTANT: your previous response failed to parse as JSON."
                + " Output ONLY a valid, COMPACT JSON object (no prose, no markdown fences,"
                + " no trailing commas). Keep each `question` <= 20 words and omit `rationale`"
                + " if it would exceed the token budget."
                + (f"\nYour previous (failed) output was:\n{prev_snippet}\n" if prev_snippet else "")
            )
            retry_tokens = int(self.config.max_tokens * self.config.retry_max_tokens_multiplier)
            logger.info(
                f"[Planner] decompose first parse failed, triggering retry (max_tokens={retry_tokens})"
            )
            raw2 = self._invoke_llm(
                system="You are a careful task planner. Output strict JSON only.",
                user=retry_prompt,
                max_tokens=retry_tokens,
            )
            obj = _extract_json_obj(raw2) if raw2 else None
            if obj is None:
                logger.warning(
                    f"[Planner] could not extract JSON from response (incl. retry), first 200 chars: {(raw or '')[:200]!r}"
                )
                return []

        if obj is None:
            if raw:
                logger.warning(
                    f"[Planner] could not extract JSON from response, first 200 chars: {raw[:200]!r}"
                )
            return []

        subtasks = obj.get("subtasks") or obj.get("sub_tasks") or []
        if not isinstance(subtasks, list):
            logger.warning(f"[Planner] 'subtasks' field is not a list: {type(subtasks)}")
            return []

        # Store rationale on an instance attribute, to avoid setattr on a builtin list (not supported by Python).
        self._last_rationale = str(obj.get("rationale") or "").strip()
        return subtasks

    # ------------------------------------------------------------------
    # Internal: unified LLM call wrapper (reused by decompose/replan, swallows exceptions and returns "")
    # ------------------------------------------------------------------
    def _invoke_llm(self, system: str, user: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw = self._generate(
                messages,
                # allow generate_fn to ignore these kwargs; most implementations use them
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                timeout=self.config.timeout,
            )
        except Exception as exc:
            logger.warning(f"[Planner] _invoke_llm exception: {exc}")
            return ""
        return raw or ""

    @staticmethod
    def _extract_rationale(subtasks: List[Dict[str, Any]]) -> str:
        """Backward compatibility: old tests or external callers may still call this; rationale now lives on an instance attribute."""
        return str(getattr(subtasks, "_rationale", "") or "").strip()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_atomic(ledger: TaskLedger, reason: str = "") -> None:
        """Failure fallback: add the original question as the sole SubTask to the ledger.

        This keeps LedgerAwareMultiRoundAgent's scheduling/synthesis logic fully generic,
        with no need to additionally check "whether a decomposition exists". The reason is
        written to global_notes to ease trajectory review.

        F3 refinement: distinguish a "semantically reasonable single subtask" from a "true LLM failure":
          - Semantically reasonable (the question is inherently atomic / too short / Planner disabled) -> use a neutral tag,
            to avoid the model being nudged toward negative reasoning every round by the negative anchor
            "atomic fallback (llm_returned_empty)".
          - True failure (LLM returned empty / JSON parse failed / all subtasks invalid) -> keep
            the original tag, to ease offline debugging and localization.
        """
        if ledger.subtasks:
            return
        ledger.add_subtask(
            SubTask(id="t1", question=ledger.question, depends_on=[])
        )
        # neutral reasons: even if the model sees them, it is not negatively influenced
        _NEUTRAL_REASONS = {
            "planner_disabled",
            "question_too_short",
        }
        if reason in _NEUTRAL_REASONS:
            ledger.add_global_note(
                "[plan] original question is atomic; using it as a single subtask"
            )
        elif reason:
            # true failure: keep the original tag to ease localization
            ledger.add_global_note(f"[plan] atomic fallback ({reason})")

    # ==================================================================
    # Replan: outer-loop replanning (A2)
    # ------------------------------------------------------------------
    # Trigger: ledger_agent's main loop calls this when ProgressLedger.check(...)
    # yields should_replan=True; reasons explains why a replan is needed.
    # This function directly modifies the passed-in ledger (update_subtask_question /
    # abandon_with_cascade / add_subtask); the caller does not need to apply again.
    # ------------------------------------------------------------------
    def replan(self, ledger: TaskLedger, reasons: List[str]) -> Dict[str, Any]:
        """Based on the current ledger state, ask the LLM to do an incremental repair of the plan.

        Returns:
            dict with keys:
              - applied: bool whether any modification actually happened
              - updated: [id, ...]
              - abandoned: [id, ...]   (already includes all cascade-propagated ids)
              - added: [id, ...]
              - rationale: LLM explanation
              - warnings: [str, ...]
              - plan_version: int
        """
        result: Dict[str, Any] = {
            "applied": False,
            "updated": [],
            "abandoned": [],
            "added": [],
            "rationale": "",
            "warnings": [],
            "plan_version": ledger.plan_version,
        }

        if not self.config.enabled:
            result["warnings"].append("planner_disabled")
            return result

        # Build the ledger snapshot (take only fields useful for replan, to reduce tokens)
        snapshot_lines: List[str] = []
        for st in ledger.iter_in_order():
            line = f"- {st.id} [{st.status}]: {st.question}"
            if st.best_answer:
                line += f" | answer={st.best_answer.strip()[:120]}"
            if st.evidence:
                last_ev = st.evidence[-1]
                ev_quote = (last_ev.quote or "").strip().replace("\n", " ")[:120]
                if ev_quote:
                    line += f" | last_ev={ev_quote}"
            if st.depends_on:
                line += f" | deps={st.depends_on}"
            if st.attempts:
                line += f" | attempts={st.attempts}"
            snapshot_lines.append(line)
        snapshot = "\n".join(snapshot_lines) if snapshot_lines else "(empty)"

        reasons_str = "\n".join(f"- {r}" for r in reasons) if reasons else "- (unspecified)"

        prompt = REPLAN_PROMPT.format(
            question=ledger.question,
            ledger_snapshot=snapshot,
            reasons=reasons_str,
        )
        # First call (via the unified _invoke_llm, which swallows exceptions automatically)
        raw = self._invoke_llm(
            system="You are a careful task re-planner.",
            user=prompt,
            max_tokens=self.config.max_tokens,
        )
        obj = _extract_json_obj(raw) if raw else None

        # Second call (retry): triggered on parse failure, attaching the previous output as reflection
        if obj is None and self.config.retry_on_parse_fail:
            prev_snippet = (raw or "").strip()[:400]
            retry_prompt = (
                prompt
                + "\n\n---\nIMPORTANT: your previous response failed to parse as JSON."
                + " Output ONLY a valid, COMPACT JSON object (no prose, no markdown fences,"
                + " no trailing commas). Omit `rationale` if needed to fit the token budget."
                + (f"\nYour previous (failed) output was:\n{prev_snippet}\n" if prev_snippet else "")
            )
            retry_tokens = int(self.config.max_tokens * self.config.retry_max_tokens_multiplier)
            logger.info(
                f"[Planner.replan] first parse failed, triggering retry (max_tokens={retry_tokens})"
            )
            raw2 = self._invoke_llm(
                system="You are a careful task re-planner. Output strict JSON only.",
                user=retry_prompt,
                max_tokens=retry_tokens,
            )
            obj = _extract_json_obj(raw2) if raw2 else None

        if not obj:
            result["warnings"].append("llm_no_json")
            return result

        result["rationale"] = str(obj.get("rationale") or "").strip()[:240]

        # ---- 1. abandon (done first: subsequent update/add are based on the post-abandon state) ----
        for ab in (obj.get("abandon_subtasks") or [])[:5]:
            if isinstance(ab, str):
                aid, reason = ab.strip(), ""
            elif isinstance(ab, dict):
                aid = str(ab.get("id") or "").strip()
                reason = str(ab.get("reason") or "").strip()
            else:
                continue
            if not aid or aid not in ledger.subtasks:
                result["warnings"].append(f"abandon invalid id={aid}")
                continue
            if ledger.subtasks[aid].status == "resolved":
                result["warnings"].append(f"abandon rejected: {aid} already resolved")
                continue
            affected = ledger.abandon_with_cascade(aid, reason=reason or "replan")
            if affected:
                result["abandoned"].extend(affected)
                result["applied"] = True

        # ---- 2. update ----
        for up in (obj.get("update_subtasks") or [])[:5]:
            if not isinstance(up, dict):
                continue
            uid = str(up.get("id") or "").strip()
            new_q = str(up.get("new_question") or up.get("question") or "").strip()
            reason = str(up.get("reason") or "").strip()
            if not uid or not new_q or uid not in ledger.subtasks:
                result["warnings"].append(f"update invalid id={uid}")
                continue
            if ledger.subtasks[uid].status == "resolved":
                result["warnings"].append(f"update rejected: {uid} already resolved")
                continue
            if ledger.update_subtask_question(uid, new_q, reason=reason or "replan"):
                result["updated"].append(uid)
                result["applied"] = True

        # ---- 3. add ----
        existing_ids = set(ledger.subtasks.keys())
        for ns in (obj.get("new_subtasks") or [])[:3]:
            if not isinstance(ns, dict):
                continue
            sid = str(ns.get("id") or "").strip()
            q = str(ns.get("question") or "").strip()
            if not sid or not q or sid in existing_ids:
                result["warnings"].append(f"add invalid id={sid}")
                continue
            deps_raw = ns.get("depends_on") or []
            deps = [str(d) for d in deps_raw if str(d) in ledger.subtasks]
            ledger.add_subtask(SubTask(id=sid, question=q, depends_on=deps))
            existing_ids.add(sid)
            result["added"].append(sid)
            result["applied"] = True

        if result["applied"]:
            ledger.plan_version += 1
            note = (
                f"[replan v{ledger.plan_version}] "
                f"updated={result['updated']} abandoned={result['abandoned']} "
                f"added={result['added']}"
            )
            if result["rationale"]:
                note += f" :: {result['rationale']}"
            ledger.add_global_note(note)
        result["plan_version"] = ledger.plan_version
        return result
