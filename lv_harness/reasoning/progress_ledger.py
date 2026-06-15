"""
ProgressLedger: a Magentic-One-style "progress ledger", a pure code-side Progress Check.

Division of labor with TaskLedger:
  - TaskLedger   : question decomposition, SubTask lifecycle, evidence archiving (static ledger)
  - ProgressLedger: checks each round "whether overall progress is being made" and decides whether a replan is needed (dynamic control flow)

Design principles (aligned with the long-standing preference for "hard code-level Guardrails > relying on the model's self-awareness"):
  1. All trigger conditions are pure code heuristics, fully testable, and do not depend on an LLM.
  2. Output a structured ProgressSignal (rather than directly editing the ledger); the agent main loop decides follow-up actions.
  3. Treat the ledger as read-only, make no state changes, keep a single responsibility.
  4. Zero-intrusion: do not import any agent-side module; no external dependency beyond task_ledger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .task_ledger import TaskLedger, SubTask

logger = logging.getLogger(__name__)


# =====================================================================
# Signal semantics convention
# ---------------------------------------------------------------------
# When should_replan = True, the agent main loop calls Planner.replan after this round ends.
# replan_reasons holds one or more structured reason strings, used by the replan prompt.
# =====================================================================
@dataclass
class ProgressSignal:
    should_replan: bool = False
    should_early_synthesis: bool = False
    replan_reasons: List[str] = field(default_factory=list)
    # auxiliary fields, to ease trajectory review
    stall_rounds: int = 0
    low_increment_rounds: int = 0
    same_focus_rounds: int = 0
    dep_broken_ids: List[str] = field(default_factory=list)
    stalled_ids: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.replan_reasons:
            self.replan_reasons.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_replan": self.should_replan,
            "should_early_synthesis": self.should_early_synthesis,
            "replan_reasons": list(self.replan_reasons),
            "stall_rounds": self.stall_rounds,
            "low_increment_rounds": self.low_increment_rounds,
            "same_focus_rounds": self.same_focus_rounds,
            "dep_broken_ids": list(self.dep_broken_ids),
            "stalled_ids": list(self.stalled_ids),
        }


# =====================================================================
# ProgressLedger: cumulative trajectory recording + Progress Check
# ---------------------------------------------------------------------
# The main loop calls .record_round(...) each round, then .check(...) to get the signal.
# =====================================================================
class ProgressLedger:
    """Track progress signals across rounds and make replan / early_synthesis decisions."""

    def __init__(self,
                 max_attempts_per_subtask: int = 3,
                 stall_low_increment_threshold: int = 2,
                 same_focus_replan_threshold: int = 3,
                 max_replans: int = 2) -> None:
        # the externally configured scheduler limit; reused here to detect "plan-level deadlock"
        self._max_attempts = max_attempts_per_subtask
        # N consecutive low-increment rounds => trigger replan
        self._low_inc_th = stall_low_increment_threshold
        # N consecutive rounds with focus on the same event => trigger replan (plain query drift)
        self._same_focus_th = same_focus_replan_threshold
        # the maximum number of replans for the whole question (defensive cap)
        self._max_replans = max_replans

        # runtime state
        self._history: List[Dict[str, Any]] = []  # one per round: focus_label / new_evidence_cnt / scheduled_id
        self._prev_total_evidence: int = 0
        self._replan_count: int = 0

    # ------------------------------------------------------------------
    # Record one round's result (called by the agent after retrieval + ledger apply)
    # ------------------------------------------------------------------
    def record_round(self, ledger: TaskLedger,
                     scheduled_id: Optional[str],
                     focus_label: Optional[str],
                     is_empty: bool) -> None:
        total_evidence = sum(len(st.evidence) for st in ledger.subtasks.values())
        new_ev = max(0, total_evidence - self._prev_total_evidence)
        self._prev_total_evidence = total_evidence

        self._history.append({
            "scheduled_id": scheduled_id,
            "focus_label": focus_label or "",
            "is_empty": bool(is_empty),
            "new_evidence": int(new_ev),
            "plan_version": int(ledger.plan_version),
        })
        logger.debug(
            f"[ProgressLedger] round#{len(self._history)-1} "
            f"sched={scheduled_id} focus='{focus_label}' new_ev={new_ev} "
            f"empty={is_empty} plan_v={ledger.plan_version}"
        )

    # ------------------------------------------------------------------
    # Progress Check: read-only inspection of ledger + own trajectory, producing a signal
    # ------------------------------------------------------------------
    def check(self, ledger: TaskLedger) -> ProgressSignal:
        sig = ProgressSignal()

        # ---- 1. All terminal -> early synthesis (not replan) ----
        if ledger.subtasks and ledger.all_done():
            sig.should_early_synthesis = True
            sig.add_reason("all_subtasks_terminal")
            return sig

        # ---- 2. Deadlock detection: subtasks whose attempts are exhausted but are still searching/partial ----
        stalled: List[SubTask] = [
            st for st in ledger.iter_in_order()
            if st.status in ("searching", "partial")
            and st.attempts >= self._max_attempts
        ]
        if stalled:
            sig.stalled_ids = [st.id for st in stalled]

        # ---- 3. Broken dependency: a pending subtask's depends_on contains an already-abandoned id ----
        #      This signal means the plan is already "inherently infeasible" and must be replanned to unblock
        dep_broken: List[str] = []
        for st in ledger.iter_in_order():
            if st.status != "pending":
                continue
            for dep in st.depends_on:
                dep_st = ledger.subtasks.get(dep)
                if dep_st is not None and dep_st.status == "abandoned":
                    dep_broken.append(st.id)
                    break
        if dep_broken:
            sig.dep_broken_ids = dep_broken

        # ---- 4. Consecutive low increment: new_evidence==0 for the last N rounds ----
        low_inc = 0
        for h in reversed(self._history):
            if h["new_evidence"] == 0:
                low_inc += 1
            else:
                break
        sig.low_increment_rounds = low_inc

        # ---- 5. Consecutive same focus: hitting the same focus_label for the last N rounds ----
        same_focus = 0
        if self._history:
            latest_focus = self._history[-1]["focus_label"]
            if latest_focus:
                for h in reversed(self._history):
                    if h["focus_label"] == latest_focus:
                        same_focus += 1
                    else:
                        break
        sig.same_focus_rounds = same_focus

        # ---- 6. schedulable_exhausted: the scheduler can no longer pick any advanceable task ----
        schedulable_exhausted = ledger.next_pending(
            max_attempts=self._max_attempts
        ) is None

        # ---- 7. Aggregate trigger conditions ----
        reasons_triggered = False
        if self._replan_count >= self._max_replans:
            # Already replanned max times, no longer trigger; just wrap up with early synthesis
            if stalled or dep_broken or schedulable_exhausted:
                sig.should_early_synthesis = True
                sig.add_reason(f"max_replans_reached({self._replan_count})")
            return sig

        if stalled:
            reasons_triggered = True
            sig.add_reason(
                f"stalled_subtasks={[st.id for st in stalled]} "
                f"(attempts>={self._max_attempts} but still not terminal)"
            )
        if dep_broken:
            reasons_triggered = True
            sig.add_reason(
                f"dep_broken_subtasks={dep_broken} "
                f"(their dependencies were abandoned)"
            )
        if low_inc >= self._low_inc_th and len(self._history) >= self._low_inc_th:
            reasons_triggered = True
            sig.add_reason(
                f"low_increment_streak={low_inc} "
                f"(no new evidence added in last {low_inc} rounds)"
            )
        if same_focus >= self._same_focus_th and len(self._history) >= self._same_focus_th:
            reasons_triggered = True
            sig.add_reason(
                f"same_focus_streak={same_focus} "
                f"(focus stuck at '{self._history[-1]['focus_label']}')"
            )
        if schedulable_exhausted and not reasons_triggered:
            # Purely scheduling-exhausted with no other signal (e.g. an all pending/abandoned mix); conservatively early synthesis
            sig.should_early_synthesis = True
            sig.add_reason("schedulable_exhausted")
            return sig

        if reasons_triggered:
            sig.should_replan = True
        sig.stall_rounds = max(low_inc, same_focus)
        return sig

    # ------------------------------------------------------------------
    # Bookkeeping after a replan completes (called by the agent)
    # ------------------------------------------------------------------
    def mark_replan_done(self) -> None:
        self._replan_count += 1
        logger.info(f"[ProgressLedger] replan #{self._replan_count} applied")

    @property
    def replan_count(self) -> int:
        return self._replan_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replan_count": self._replan_count,
            "history": list(self._history),
            "config": {
                "max_attempts_per_subtask": self._max_attempts,
                "stall_low_increment_threshold": self._low_inc_th,
                "same_focus_replan_threshold": self._same_focus_th,
                "max_replans": self._max_replans,
            },
        }
