"""
TaskLedger: dynamic task ledger for multi-round retrieval reasoning.

Design goals (works with LedgerAwareMultiRoundAgent, zero intrusion into the
existing MultiRoundSearchAgent):
1. Upgrade "compound question decomposition + multi-round retrieval + evidence
   archiving + final synthesis" from verbal constraints in SYSTEM_PROMPT into
   explicit state that can be enforced by hard code-level Guardrails.
2. Each sub-question is a SubTask with the lifecycle pending -> searching ->
   partial / resolved / abandoned. State transitions are validated by code, so
   the model cannot "verbally claim completion".
3. Each retrieval produces several Evidence items attributed to specific
   SubTasks (possibly across SubTasks), avoiding cross-round information
   dilution.
4. Provides `to_prompt_block()` to render the current ledger as a compact prompt
   fragment, dynamically injected into each round's user message (echoing the
   prompt-engineering preference: dynamic prompt injection + hard code-level
   constraints).

This module does not import any other in-project modules, ensuring it is
independently testable with zero side effects.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- json-repair soft dependency: repair JSON when LLM output is truncated by max_tokens ----
try:
    from json_repair import repair_json as _repair_json  # type: ignore
    _HAS_JSON_REPAIR = True
except Exception:  # pragma: no cover
    _repair_json = None
    _HAS_JSON_REPAIR = False


def _try_json_repair_dict(candidate: str) -> Optional[Dict[str, Any]]:
    """Use json-repair to fix a fragment into a dict; return None on failure."""
    if not _HAS_JSON_REPAIR or not candidate:
        return None
    try:
        fixed = _repair_json(candidate, return_objects=True)
    except Exception as exc:
        logger.debug(f"[task_ledger] json-repair fix exception: {exc}")
        return None
    return fixed if isinstance(fixed, dict) else None


# ---- type aliases ----
SubTaskStatus = Literal["pending", "searching", "partial", "resolved", "abandoned"]
_VALID_STATUSES = ("pending", "searching", "partial", "resolved", "abandoned")


# ---- internal utilities: lenient parsing ----
def _safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert any input to float; return default on failure."""
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return default


def _safe_int_list(x: Any) -> List[int]:
    """Safely convert any input to List[int]; non-numeric elements are dropped."""
    if not isinstance(x, (list, tuple)):
        return []
    out: List[int] = []
    for c in x:
        if isinstance(c, bool):
            continue  # avoid True/False being treated as 1/0
        if isinstance(c, (int, float)):
            out.append(int(c))
            continue
        try:
            out.append(int(str(c).strip()))
        except Exception:
            continue
    return out


# ---- module-level utility: strip LedgerOps code blocks, for Agent use ----
# The match scope is a bit wider than LedgerUpdater's internal regex: any ```...```
# containing JSON will be stripped, to prevent LedgerOps from being swallowed into
# the Action Content by the greedy parser.
_STRIP_FENCE_RE = re.compile(
    r"```(?:ledger|json)?\s*\n?\{.*?\}\s*\n?```",
    re.DOTALL | re.IGNORECASE,
)
_STRIP_INLINE_RE = re.compile(
    r"LedgerOps\s*:\s*\{.*?\}\s*(?:\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def strip_ledger_ops_from_text(text: str) -> str:
    """Strip ```ledger``` / ```json``` code blocks and `LedgerOps: {...}` fragments from arbitrary text.

    Used to clean LedgerOps that got swallowed into the Action Content by the
    greedy parser, preserving the real search query or answer content. After
    stripping, leading/trailing whitespace is stripped; if the result is empty,
    the original text is returned.
    """
    if not text:
        return text
    cleaned = _STRIP_FENCE_RE.sub("", text)
    cleaned = _STRIP_INLINE_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text

# Legal state transitions (hard code-level constraints)
# Rules:
#   - pending may move to searching / partial / resolved / abandoned
#   - searching may move to partial / resolved / abandoned / pending (rollback retry)
#   - partial may stay partial / move up to resolved / move down to abandoned
#   - resolved may roll back to partial (allowing the model to correct a wrong judgment in later rounds)
#   - abandoned is terminal
_ALLOWED_TRANSITIONS: Dict[str, set] = {
    "pending":   {"searching", "partial", "resolved", "abandoned"},
    "searching": {"partial", "resolved", "abandoned", "pending"},
    "partial":   {"partial", "resolved", "abandoned", "searching"},
    "resolved":  {"partial"},   # allow rollback to partial (correction mechanism)
    "abandoned": set(),   # terminal
}


@dataclass
class Evidence:
    """An evidence item that one retrieval provides for a given SubTask."""
    round_idx: int                       # which round it comes from (0-based)
    query: str                           # the actual query string submitted to memory
    focus_label: str = ""                # corresponds to retrieval_payload["focus_label"]
    quote: str = ""                      # key fragment excerpted from focus.summary (truncated)
    score: float = 0.0                   # event_score / relevance
    clip_ids: List[int] = field(default_factory=list)
    mode: str = "event_first"            # event_first / video_drilldown / neighbor / keyframe_inspect

    def short_repr(self, include_quote: bool = True, max_quote_len: int = 140) -> str:
        """Compact display.

        Args:
            include_quote: if the quote already appeared in the same round's focus
                summary, the caller may pass False to avoid duplicate injection.
            max_quote_len: quote truncation length (default 140; the synthesis stage
                may pass a larger value)
        """
        sc = _safe_float(self.score, 0.0)
        head = (
            f"[R{self.round_idx} {self.mode} score={sc:.2f}] "
            f"{self.focus_label or '(no-focus)'}"
        )
        if not include_quote:
            return head
        q = (self.quote or "").strip().replace("\n", " ")
        if not q:
            return head
        if len(q) > max_quote_len:
            q = q[:max_quote_len] + "..."
        return f"{head}: {q}"

    def full_repr(self) -> str:
        """Full display (for the synthesis stage), without truncating the quote."""
        sc = _safe_float(self.score, 0.0)
        head = (
            f"[R{self.round_idx} {self.mode} score={sc:.2f}] "
            f"{self.focus_label or '(no-focus)'}"
        )
        q = (self.quote or "").strip().replace("\n", " ")
        if not q:
            return head
        return f"{head}: {q}"


@dataclass
class SubTask:
    """The full lifecycle ledger of one atomic sub-question."""
    id: str                                      # stable id, e.g. "t1", "t2"
    question: str                                # sub-question text
    depends_on: List[str] = field(default_factory=list)
    status: SubTaskStatus = "pending"
    evidence: List[Evidence] = field(default_factory=list)
    best_answer: Optional[str] = None            # current best-answer draft
    confidence: float = 0.0                      # [0,1]
    attempts: int = 0                            # number of retrieval attempts so far
    failed_queries: List[str] = field(default_factory=list)   # historically attempted queries
    notes: List[str] = field(default_factory=list)            # the model's free-form notes
    rationale: str = ""                          # reasoning fragment attached when resolved

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def can_transition(self, new_status: SubTaskStatus) -> bool:
        if new_status not in _VALID_STATUSES:
            return False
        if new_status == self.status:
            # same status is allowed (e.g. partial -> partial only updates evidence)
            return True
        return new_status in _ALLOWED_TRANSITIONS.get(self.status, set())

    def transition(self, new_status: SubTaskStatus, reason: str = "") -> bool:
        """Attempt a state transition; an illegal transition silently keeps the original state and returns False. The caller should log it."""
        if not self.can_transition(new_status):
            return False
        old = self.status
        self.status = new_status
        if reason:
            self.notes.append(f"[{old}->{new_status}] {reason}")
        return True

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        # Auto-maintain confidence: weighted by evidence count + max score (capped at 0.95, leaving room for the model's answer)
        if self.evidence:
            max_score = max(e.score for e in self.evidence)
            # More items means more confidence, but with diminishing returns (1->0.4 / 2->0.6 / 3->0.73 / 4->0.83 / 5->0.9)
            count_factor = 1.0 - (0.6 ** len(self.evidence))
            self.confidence = min(0.95, 0.5 * max_score + 0.5 * count_factor)

    def record_failed_query(self, q: str) -> None:
        if q and q not in self.failed_queries:
            self.failed_queries.append(q)

    def is_terminal(self) -> bool:
        """resolved or abandoned counts as terminal (used for all_done / deps checks).
        Note: resolved may roll back to partial, but it is still treated as "done" in the all_done check.
        """
        return self.status in ("resolved", "abandoned")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "evidence": [asdict(e) for e in self.evidence],
            "best_answer": self.best_answer,
            "confidence": round(self.confidence, 3),
            "attempts": self.attempts,
            "failed_queries": list(self.failed_queries),
            "notes": list(self.notes),
            "rationale": self.rationale,
        }


@dataclass
class TaskLedger:
    """The task ledger for the entire question.

    Conventions:
      - `subtasks` is an id -> SubTask mapping; dependencies are declared in SubTask.depends_on.
      - `global_notes` records cross-SubTask observations (e.g. "<character_0> bound to Bob").
      - `_order` at the top preserves the insertion order of SubTasks, used for rendering and scheduling.
      - `plan_version` records how many times the plan has been replanned/updated, for trajectory review.
    """
    question: str
    options: List[str] = field(default_factory=list)          # non-empty only for MCQ
    subtasks: Dict[str, SubTask] = field(default_factory=dict)
    global_notes: List[str] = field(default_factory=list)
    _order: List[str] = field(default_factory=list)
    plan_version: int = 1                                     # initial plan is v1; incremented on every replan/update

    # ---- initialization / add-remove ----
    def add_subtask(self, subtask: SubTask) -> None:
        if subtask.id in self.subtasks:
            logger.debug(f"[Ledger] ignoring duplicate subtask id: {subtask.id}")
            return
        # filter illegal dependencies (referencing unknown ids)
        subtask.depends_on = [d for d in subtask.depends_on if d != subtask.id]
        self.subtasks[subtask.id] = subtask
        self._order.append(subtask.id)

    def iter_in_order(self) -> List[SubTask]:
        return [self.subtasks[i] for i in self._order if i in self.subtasks]

    def get(self, sid: str) -> Optional[SubTask]:
        return self.subtasks.get(sid)

    def add_global_note(self, note: str) -> None:
        note = (note or "").strip()
        if note and note not in self.global_notes:
            self.global_notes.append(note)

    # ---- Plan maintenance (A1/A3 backtracking and replanning infrastructure) ----
    def update_subtask_question(self, sid: str, new_question: str,
                                reason: str = "") -> bool:
        """Rewrite a SubTask's question text (without resetting status/evidence).

        Triggered by: Planner.replan or LedgerOps.update_subtasks.
        Returns False if sid does not exist or new_question is empty.
        """
        st = self.subtasks.get(sid)
        nq = (new_question or "").strip()
        if st is None or not nq or nq == st.question:
            return False
        old_q = st.question
        st.question = nq
        st.notes.append(f"[rewrite] '{old_q}' -> '{nq}'" + (f" ({reason})" if reason else ""))
        self.plan_version += 1
        return True

    def abandon_with_cascade(self, sid: str, reason: str = "") -> List[str]:
        """Mark sid as abandoned, and recursively handle all downstream tasks that directly/indirectly depend on it.

        Used for A1 automatic deadlock escape: close the loop directly when a subtask's attempts reach the limit.
        Subtasks whose dependency dep is already terminal (abandoned/resolved) are not affected.
        Returns the list of ids actually set to abandoned (including sid itself; already-terminal ones are not counted).
        """
        affected: List[str] = []
        root = self.subtasks.get(sid)
        if root is None:
            return affected
        if not root.is_terminal():
            # everything other than resolved/abandoned should be able to enter abandoned; we do not go through transition() here because
            # searching->abandoned / partial->abandoned / pending->abandoned are all legal
            # (see _ALLOWED_TRANSITIONS), but we also tolerate state-machine drift.
            if root.transition("abandoned", reason=reason or "cascade_root"):
                affected.append(root.id)

        # BFS: a cycle is theoretically impossible (add_subtask already filters self-references), but we still use visited to avoid an accidental infinite loop
        frontier = [root.id]
        visited = {root.id}
        while frontier:
            nxt: List[str] = []
            for dead_id in frontier:
                for st in self.iter_in_order():
                    if st.id in visited:
                        continue
                    if dead_id in st.depends_on and not st.is_terminal():
                        if st.transition("abandoned", reason=f"cascade_from_{dead_id}"):
                            affected.append(st.id)
                            nxt.append(st.id)
                            visited.add(st.id)
            frontier = nxt

        if affected:
            self.plan_version += 1
            self.add_global_note(
                f"[cascade] abandoned {affected} (root={sid}, reason={reason or 'n/a'})"
            )
        return affected

    def mark_stalled_terminals(self, max_attempts: int) -> List[str]:
        """Code-side hard Guardrail:
        Automatically abandon (and cascade) subtasks whose attempts have reached the limit but are still stuck in searching/partial.
        Returns all ids abandoned this time (including cascaded ones).
        """
        killed: List[str] = []
        for st in list(self.iter_in_order()):
            if st.status in ("searching", "partial") and st.attempts >= max_attempts:
                killed.extend(
                    self.abandon_with_cascade(
                        st.id,
                        reason=f"attempts>={max_attempts} stalled",
                    )
                )
        # deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for i in killed:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq

    # ---- scheduling ----
    def _deps_ready(self, st: SubTask) -> bool:
        for dep in st.depends_on:
            d = self.subtasks.get(dep)
            if d is None:
                # a missing dependency does not block, only warns
                logger.warning(f"[Ledger] subtask {st.id} dependency does not exist: {dep}")
                continue
            if not d.is_terminal():
                return False
        return True

    def next_pending(self, max_attempts: int = 3) -> Optional[SubTask]:
        """Return the next SubTask that should be scheduled, by the following priority:
          1. status=pending with dependencies satisfied and attempts < max_attempts
          2. status=searching / partial with attempts < max_attempts
        The original order is stable (`_order`), avoiding meaningless jitter.

        The attempts limit applies uniformly to pending / searching / partial, preventing
        the same subtask from being scheduled indefinitely when the model forgets to declare
        a state transition (hard code-level Guardrail).
        """
        # 1. pending, dependencies satisfied, and attempts not exceeded
        for st in self.iter_in_order():
            if (st.status == "pending"
                    and self._deps_ready(st)
                    and st.attempts < max_attempts):
                return st
        # 2. searching / partial that can still be advanced
        for st in self.iter_in_order():
            if st.status in ("searching", "partial") and st.attempts < max_attempts:
                return st
        return None

    def all_done(self) -> bool:
        return all(st.is_terminal() for st in self.subtasks.values())

    def snapshot_signature(self) -> str:
        """Return a lightweight signature of the current ledger state, used to detect whether the state changed.

        The signature contains, for each subtask: (id, status, evidence_count, first 20 chars of best_answer).
        When the signature matches the previous round, the ledger is unchanged and the full snapshot injection can be skipped.
        """
        parts = []
        for st in self.iter_in_order():
            ba_prefix = (st.best_answer or "")[:20]
            parts.append(f"{st.id}:{st.status}:{len(st.evidence)}:{ba_prefix}")
        notes_sig = str(len(self.global_notes))
        return "|".join(parts) + f"||notes={notes_sig}"

    def to_compact_status_line(self) -> str:
        """A one-line compact status summary, used for lightweight injection when the ledger is unchanged.

        Format example: [Ledger] t1:resolved t2:partial(2ev) t3:pending
        """
        parts = []
        for st in self.iter_in_order():
            ev_count = len(st.evidence)
            if ev_count > 0 and st.status not in ("resolved", "abandoned"):
                parts.append(f"{st.id}:{st.status}({ev_count}ev)")
            else:
                parts.append(f"{st.id}:{st.status}")
        return "[Ledger status] " + " | ".join(parts)

    def any_unresolved(self) -> bool:
        """There are still SubTasks without consensus (excluding resolved)."""
        return any(st.status != "resolved" for st in self.subtasks.values())

    def resolved_count(self) -> int:
        return sum(1 for st in self.subtasks.values() if st.status == "resolved")

    # ---- rendering ----
    def to_prompt_block(self, max_evidence_per_task: int = 3,
                        highlight_id: Optional[str] = None,
                        suppress_quote_labels: Optional[set] = None) -> str:
        """Render as a compact Markdown-ish fragment, for prefix injection into each round's user message.

        Tiered rendering (B2):
          - Active (highlight_id or searching/partial): expand evidence + best_answer + failed_queries
          - Resolved: one-line summary (id + answer + top evidence)
          - Pending: one minimal line (id + question + depends_on)
          - Abandoned: one neutral marker line (to avoid the model retrying that direction)

        Evidence quote overlap removal (B3): if evidence.focus_label is in
        suppress_quote_labels (indicating the summary was already fully injected in
        this round's retrieval_text), the quote is no longer redundantly attached in
        the ledger, leaving only label+score.

        Note: since the full retrieval text of earlier rounds in the conversation
        history has been compressed away, the ledger's evidence is the LLM's main
        window into historical retrieval results. Therefore more evidence items are
        shown by default (max_evidence_per_task=3) to ensure key information is not
        lost.

        Args:
            max_evidence_per_task: how many evidence items to show at most per active subtask (default 3)
            highlight_id: the target for this round selected by the scheduler
            suppress_quote_labels: the set of focus_labels already injected into this round's retrieval_text
        """
        suppress = suppress_quote_labels or set()
        lines = ["[TaskLedger Snapshot]"]
        if not self.subtasks:
            lines.append("  (empty: no decomposition yet)")
            if self.global_notes:
                lines.append("  global notes: " + " | ".join(self.global_notes[-3:]))
            return "\n".join(lines)

        active, resolved, pending, abandoned = [], [], [], []
        for st in self.iter_in_order():
            if st.status == "abandoned":
                abandoned.append(st)
            elif st.status == "resolved":
                resolved.append(st)
            elif st.id == highlight_id or st.status in ("searching", "partial"):
                active.append(st)
            else:
                pending.append(st)

        # Active: fully expanded
        for st in active:
            marker = "▶ " if (highlight_id and st.id == highlight_id) else "  "
            dep_str = f" depends_on={st.depends_on}" if st.depends_on else ""
            conf_val = _safe_float(st.confidence, 0.0)
            conf_str = f" conf={conf_val:.2f}" if conf_val > 0 else ""
            lines.append(
                f"{marker}{st.id} [{st.status}{conf_str}]{dep_str}: {st.question}"
            )
            for ev in st.evidence[-max_evidence_per_task:]:
                keep_quote = ev.focus_label not in suppress
                lines.append(f"      · {ev.short_repr(include_quote=keep_quote)}")
            if st.best_answer:
                ba = st.best_answer.strip().replace("\n", " ")
                if len(ba) > 180:
                    ba = ba[:180] + "..."
                lines.append(f"      ► best_answer: {ba}")
            if st.failed_queries:
                fq = st.failed_queries[-2:]
                lines.append(f"      × failed_queries: {fq}")

        # Resolved: summary + top evidence (since the early conversation is compressed, resolved evidence also needs to be shown)
        for st in resolved:
            ba = (st.best_answer or "").strip().replace("\n", " ")
            if len(ba) > 160:
                ba = ba[:160] + "..."
            lines.append(f"  ✓ {st.id} [resolved]: {ba or '(done)'}")
            # show the top 1 evidence summary to help the LLM understand the source of the answer
            if st.evidence:
                top_ev = max(st.evidence, key=lambda e: e.score)
                keep_quote = top_ev.focus_label not in suppress
                lines.append(f"      · {top_ev.short_repr(include_quote=keep_quote)}")

        # Pending: one minimal line
        for st in pending:
            dep_str = f" <-{st.depends_on}" if st.depends_on else ""
            lines.append(f"  • {st.id} [pending]{dep_str}: {st.question}")

        # Abandoned: one neutral line
        for st in abandoned:
            lines.append(f"  ✗ {st.id} [abandoned]: {st.question}")

        if self.global_notes:
            lines.append("  [global notes] " + " | ".join(self.global_notes[-5:]))
        return "\n".join(lines)

    def to_synthesis_block(self) -> str:
        """The complete view for the final synthesis stage: lists each SubTask's best_answer and all evidence.

        Design intent: the synthesis stage needs information as complete as possible
        to make the final judgment. Since earlier rounds in the conversation history
        have been compressed, the ledger's evidence is an important supplement beyond
        the deduplicated full retrieval text. Here the full quote of all evidence is
        shown, ensuring the LLM does not miss key details during synthesis.
        """
        lines = ["[TaskLedger Final State]"]
        for st in self.iter_in_order():
            line = f"- {st.id} [{st.status}]: {st.question}"
            if st.best_answer:
                line += f"\n    answer: {st.best_answer.strip()}"
            # show all evidence (descending by score) to ensure complete information at the synthesis stage
            sorted_ev = sorted(st.evidence, key=lambda e: e.score, reverse=True)
            for ev in sorted_ev:
                line += f"\n    evidence: {ev.full_repr()}"
            if not st.evidence and st.status != "resolved":
                line += "\n    (no evidence collected)"
            lines.append(line)
        if self.global_notes:
            lines.append("[Global notes]")
            for note in self.global_notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "order": list(self._order),
            "subtasks": {sid: st.to_dict() for sid, st in self.subtasks.items()},
            "global_notes": list(self.global_notes),
            "plan_version": int(self.plan_version),
            "stats": {
                "total": len(self.subtasks),
                "resolved": self.resolved_count(),
                "abandoned": sum(1 for st in self.subtasks.values() if st.status == "abandoned"),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================
# LedgerUpdater: parses the LLM's EvidenceBinding / state-transition instructions
# ------------------------------------------------------------
# In a normal round, besides Action/Content, the LLM may also output a structured
# "LedgerOps" block before/after Reason. Parsing uses a lenient strategy: missing or
# out-of-order entries are allowed, illegal items are dropped with a warning, and the
# whole round's response is not rejected (tolerating occasional format drift in baseline models).
# ============================================================
class LedgerUpdater:
    """Parses ledger operations from the LLM response and applies them to the TaskLedger.

    Supports two kinds of input (either one or a mix):
      1. JSON block (recommended):
          ```ledger
          {
            "target": "t2",
            "bind": [
              {"id": "t2", "status": "partial",
               "quote": "Bob cooked pasta at 14:23", "answer": "pasta"}
            ],
            "notes": ["<character_0> seems to be Bob"],
            "new_subtasks": [{"id": "t4", "question": "...", "depends_on": ["t2"]}]
          }
          ```
      2. line-level KV (fallback):
          LedgerOps:
            Target: t2
            Bind t2 partial: Bob cooked pasta ...
            Note: <character_0> seems to be Bob
    """

    import re as _re
    # Priority 1: strict ```ledger fence (recommended protocol)
    _LEDGER_FENCE_RE = _re.compile(
        r"```ledger\s*\n(\{.*?\})\s*\n```",
        _re.DOTALL | _re.IGNORECASE,
    )
    # Priority 2: generic ```json / ``` fence (only accepted if it passes the schema check)
    _GENERIC_FENCE_RE = _re.compile(
        r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
        _re.DOTALL | _re.IGNORECASE,
    )
    # Priority 3: inline JSON (LedgerOps: {...})
    _INLINE_JSON_RE = _re.compile(
        r"LedgerOps\s*:\s*(\{.*?\})(?:\n\n|$)",
        _re.DOTALL | _re.IGNORECASE,
    )

    # LedgerOps legal fields; the generic-fence JSON is accepted only if it hits at least one of these
    _LEDGER_SCHEMA_KEYS = (
        "target", "Target", "bind", "bindings", "notes", "new_subtasks",
        "update_subtasks", "abandon_subtasks",
    )

    @classmethod
    def _looks_like_ledger_ops(cls, obj: Any) -> bool:
        """schema validation: avoid mistaking an ordinary JSON response for LedgerOps."""
        if not isinstance(obj, dict):
            return False
        return any(k in obj for k in cls._LEDGER_SCHEMA_KEYS)

    @classmethod
    def parse(cls, response_text: str) -> Dict[str, Any]:
        """Extract a ledger-ops dict from the response text; return {} if not found.

        Strategy (priority from strict to lenient, from fast to slow):
          1. Strict ```ledger fence (accepted immediately on match).
          2. Generic ```json fence: must pass the _looks_like_ledger_ops schema check,
             to avoid swallowing unrelated JSON the model occasionally outputs (e.g. echoed retrieval results).
          3. Try the inline `LedgerOps: {...}` form.
          4. **(new)** json-repair truncation fallback: even if the output was truncated by
             max_tokens (missing closing fence or closing brace), a best-effort usable ledger ops dict can be recovered.
             Note: the main loop calls parse every round, so no retry is added here (the next round naturally produces new output).
        """
        if not response_text:
            return {}
        # 1. strict ledger fence
        m = cls._LEDGER_FENCE_RE.search(response_text)
        if m:
            try:
                obj = json.loads(m.group(1))
                if cls._looks_like_ledger_ops(obj):
                    return obj
                logger.debug("[LedgerUpdater] ledger fence content did not pass schema validation")
            except Exception as exc:
                logger.debug(f"[LedgerUpdater] ledger fence parse failed: {exc}")
        # 2. generic fence (schema validation required)
        m = cls._GENERIC_FENCE_RE.search(response_text)
        if m:
            try:
                obj = json.loads(m.group(1))
                if cls._looks_like_ledger_ops(obj):
                    return obj
            except Exception as exc:
                logger.debug(f"[LedgerUpdater] generic fence parse failed: {exc}")
        # 3. inline JSON (LedgerOps: {...})
        m = cls._INLINE_JSON_RE.search(response_text)
        if m:
            try:
                obj = json.loads(m.group(1))
                if cls._looks_like_ledger_ops(obj):
                    return obj
            except Exception as exc:
                logger.debug(f"[LedgerUpdater] inline JSON parse failed: {exc}")

        # 4. json-repair truncation fallback (soft dependency): for the unclosed `​``ledger\n{...` case
        if _HAS_JSON_REPAIR:
            # 4a) unclosed ```ledger / ```json fence
            fence_match = re.search(
                r"```(?:ledger|json)?\s*\n([\s\S]*)",
                response_text,
                flags=re.IGNORECASE,
            )
            if fence_match:
                inner = fence_match.group(1).split("```", 1)[0]
                obj = _try_json_repair_dict(inner)
                if obj and cls._looks_like_ledger_ops(obj):
                    logger.info("[LedgerUpdater] recovered ops via json-repair from an unclosed fence")
                    return obj
            # 4b) repair directly starting from the first '{'
            first_brace = response_text.find("{")
            if first_brace >= 0:
                obj = _try_json_repair_dict(response_text[first_brace:])
                if obj and cls._looks_like_ledger_ops(obj):
                    logger.info("[LedgerUpdater] recovered ops via json-repair from a bare fragment")
                    return obj

        return {}

    @classmethod
    def apply(cls, ledger: TaskLedger, ops: Dict[str, Any],
              round_idx: int, last_query: str = "",
              last_retrieval: Optional[Dict[str, Any]] = None
              ) -> Tuple[Optional[str], List[str]]:
        """Apply the parsed ops to the ledger.

        Returns:
            (target_id_or_none, warnings)
        """
        warnings: List[str] = []
        if not ops:
            return None, warnings

        target_id = ops.get("target") or ops.get("Target") or None
        if target_id and target_id not in ledger.subtasks:
            warnings.append(f"target '{target_id}' does not exist, ignored")
            target_id = None

        # 1. new_subtasks: allow the model to dynamically append (strict cap: no more than 3 new per round)
        for ns in (ops.get("new_subtasks") or [])[:3]:
            if not isinstance(ns, dict):
                continue
            sid = str(ns.get("id") or "").strip()
            q = str(ns.get("question") or "").strip()
            if not sid or not q:
                warnings.append(f"new_subtask missing field: {ns}")
                continue
            if sid in ledger.subtasks:
                warnings.append(f"new_subtask duplicate id: {sid}")
                continue
            deps_raw = ns.get("depends_on") or []
            deps = [str(d) for d in deps_raw if str(d) in ledger.subtasks]
            ledger.add_subtask(SubTask(id=sid, question=q, depends_on=deps))

        # 2. bind: the core. Bind evidence to a SubTask + state transition + best_answer
        binds = ops.get("bind") or ops.get("bindings") or []
        if isinstance(binds, dict):
            binds = [binds]
        for b in binds:
            if not isinstance(b, dict):
                continue
            sid = str(b.get("id") or "").strip()
            st = ledger.subtasks.get(sid)
            if not st:
                warnings.append(f"bind unknown subtask: {sid}")
                continue

            # 2a. evidence: prefer the bind's own quote / score / focus_label,
            #     falling back to this round's last_retrieval focus_label/summary
            quote = str(b.get("quote") or "").strip()
            score = _safe_float(b.get("score"), 0.0)
            focus_label = str(b.get("focus_label") or "").strip()
            mode = str(b.get("mode") or "").strip() or "event_first"
            clip_ids = _safe_int_list(b.get("clip_ids"))

            if last_retrieval:
                if not focus_label:
                    focus_label = str(last_retrieval.get("focus_label") or "")
                if not score:
                    score = _safe_float(last_retrieval.get("event_score"), 0.0)
                if not quote:
                    focus = last_retrieval.get("focus") or {}
                    if isinstance(focus, dict):
                        q = focus.get("summary") or focus.get("description") or ""
                        quote = (q or "").strip()[:200]
                if not mode or mode == "event_first":
                    mode = last_retrieval.get("mode") or mode
                if not clip_ids:
                    focus = last_retrieval.get("focus") or {}
                    if isinstance(focus, dict):
                        clip_ids = _safe_int_list(focus.get("clip_ids"))

            if quote or focus_label:
                ev = Evidence(
                    round_idx=round_idx, query=last_query,
                    focus_label=focus_label, quote=quote,
                    score=score, clip_ids=clip_ids, mode=mode,
                )
                st.add_evidence(ev)

            # 2b. best_answer
            #   F1: auto-backfill. Models like Gemini often give only a quote and no answer,
            #   leaving best_answer empty at the synthesis stage -> refusal to answer. In the
            #   following two cases the quote prefix is promoted to serve as best_answer, only
            #   as a fallback, not overriding an answer the model already gave:
            #     (a) status is partial / resolved but answer is empty
            #     (b) no explicit status but there is a quote, and this subtask currently has no best_answer
            ans = b.get("answer") or b.get("best_answer")
            if ans:
                st.best_answer = str(ans).strip()[:400]
            else:
                status_hint = str(b.get("status") or "").strip().lower()
                filled_quote = quote  # the quote parsed/backfilled this round
                if filled_quote and (
                    (status_hint in ("partial", "resolved") and not st.best_answer)
                    or (not status_hint and not st.best_answer)
                ):
                    # light compression: take only the first 260 chars, newline -> space, to avoid being too long or breaking the prompt
                    short = filled_quote.replace("\n", " ").strip()
                    if len(short) > 260:
                        short = short[:260].rstrip() + "..."
                    st.best_answer = short
                    warnings.append(
                        f"[F1] {st.id} best_answer auto-backfilled from quote (model did not explicitly give an answer)"
                    )

            # 2c. state transition
            new_status = str(b.get("status") or "").strip().lower()
            if new_status and new_status in _VALID_STATUSES:
                if not st.transition(new_status, reason=str(b.get("reason") or "")):
                    warnings.append(
                        f"illegal state transition {st.id}: {st.status} -> {new_status} (ignored)"
                    )

        # 3. notes
        #   F5: notes dict tolerance. Gemini often outputs
        #     notes: [{"id": "<character_0>", "name": "Robert"}]
        #   Here we recognize the identity-binding format (id+name) and convert it into the
        #   readable `<character_0> = Robert` string, so downstream models actually use the
        #   identity binding. Other dict types fall back to JSON serialization; strings/other
        #   scalars are stringified as-is.
        for note in (ops.get("notes") or []):
            if isinstance(note, dict):
                nid = note.get("id") or note.get("placeholder") or note.get("tag")
                nname = note.get("name") or note.get("value") or note.get("binding")
                if nid and nname:
                    ledger.add_global_note(f"{nid} = {nname}")
                    continue
                try:
                    ledger.add_global_note(json.dumps(note, ensure_ascii=False))
                except Exception:
                    ledger.add_global_note(str(note))
            else:
                ledger.add_global_note(str(note))

        # 4. update_subtasks: allow the model to explicitly rewrite sub-question text (A3)
        for up in (ops.get("update_subtasks") or [])[:5]:
            if not isinstance(up, dict):
                continue
            uid = str(up.get("id") or "").strip()
            new_q = str(up.get("new_question") or up.get("question") or "").strip()
            reason = str(up.get("reason") or "").strip()
            if uid and new_q:
                ok = ledger.update_subtask_question(uid, new_q, reason=reason)
                if not ok:
                    warnings.append(f"update_subtask invalid: id={uid}")

        # 5. abandon_subtasks: allow the model to explicitly abandon a plan branch (with cascade)
        for ab in (ops.get("abandon_subtasks") or [])[:5]:
            # support two forms: a string id or {id, reason}
            if isinstance(ab, str):
                aid, reason = ab.strip(), ""
            elif isinstance(ab, dict):
                aid = str(ab.get("id") or "").strip()
                reason = str(ab.get("reason") or "").strip()
            else:
                continue
            if not aid:
                continue
            affected = ledger.abandon_with_cascade(aid, reason=reason or "model_abandon")
            if not affected:
                warnings.append(f"abandon_subtask invalid or already terminal: id={aid}")

        return target_id, warnings
