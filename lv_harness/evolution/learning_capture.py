"""
LearningCapture: an automatic experience capturer.

Captures experience events during the reasoning process in real time and writes them to a Markdown file.
"""
import os
import re
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .learning_types import Learning, LearningType
from ..data.types import TemporalQuestion, AgentAnswer

logger = logging.getLogger(__name__)


# ---- Question-type inference rules (fallback, used when TemporalQuestion.category is empty) ----
# Lightweight rules based on interrogatives/keywords. Effective for English questions and covers common video-QA types.
_QUESTION_TYPE_PATTERNS = [
    # (question_type, [regex/keywords])
    ("temporal_order",    [r"\bbefore\b", r"\bafter\b", r"\bfirst\b", r"\blast\b", r"\bthen\b", r"\bnext\b", r"\bsequence\b", r"\border\b"]),
    ("counting",          [r"\bhow many\b", r"\bhow much\b", r"\bcount\b", r"\bnumber of\b"]),
    ("duration",          [r"\bhow long\b", r"\bduration\b", r"\btime does it take\b"]),
    ("action_recognition",[r"\bwhat (is|are|was|were) .* doing\b", r"\bwhat action\b", r"\bwhat activity\b"]),
    ("object_identification",[r"\bwhat (is|are) (this|that|these|those|it)\b", r"\bwhat object\b", r"\bwhat color\b", r"\bwhat kind\b"]),
    ("person_identification",[r"\bwho (is|are|was|were)\b", r"\bwhich (person|character|man|woman|boy|girl)\b"]),
    ("location",          [r"\bwhere\b", r"\bwhat (place|location|room)\b"]),
    ("reason_purpose",    [r"\bwhy\b", r"\bfor what reason\b", r"\bpurpose\b"]),
    ("manner_how",        [r"\bhow (does|do|did|is|are|was|were)\b"]),
    ("ocr_text",          [r"\bwhat (does|do) .* say\b", r"\bwhat (is|was) written\b", r"\btext\b", r"\bsign\b", r"\blabel\b"]),
    ("comparison",        [r"\bdifference\b", r"\bcompare\b", r"\bmore\b.*\bthan\b", r"\bless\b.*\bthan\b", r"\bsame\b"]),
]

_QUESTION_TYPE_REGEX = [
    (qtype, [re.compile(p, re.IGNORECASE) for p in patterns])
    for qtype, patterns in _QUESTION_TYPE_PATTERNS
]

# ---- P2: single-tag primary key priority ----
# Background: the sample distribution of M3-Bench robot is: Multi-Detail 66% > Person 43% > Cross-Modal 37%
# > General Knowledge 26% > Multi-Hop 7%. If Multi-Detail is used as the primary tag,
# it will "swallow" the other tags, making the samples in the other buckets even sparser. Prioritizing rare tags lets the same
# multi-tag question be assigned to its most distinctive bucket, making each bucket's samples denser and more homogeneous.
# This can be overridden via the environment variable QUESTION_TYPE_PRIORITY in comma-separated form (earlier means higher priority).
_DEFAULT_TAG_PRIORITY = (
    "multi_hop_reasoning",
    "general_knowledge_extraction",
    "cross_modal_reasoning",
    "person_understanding",
    "multi_detail_reasoning",
)


def _load_tag_priority() -> tuple:
    raw = os.environ.get("QUESTION_TYPE_PRIORITY", "").strip()
    if not raw:
        return _DEFAULT_TAG_PRIORITY
    return tuple(t.strip() for t in raw.split(",") if t.strip())


_TAG_PRIORITY = _load_tag_priority()


def _normalize_tag(tag: str) -> str:
    """Normalize a raw tag into the lowercase underscore format used for clustering."""
    tag = (tag or "").strip().lower()
    if not tag:
        return ""
    # Spaces/hyphens -> underscores, strip extra leading/trailing underscores
    tag = re.sub(r"[\s\-]+", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag


def infer_question_type_tags(question: TemporalQuestion) -> List[str]:
    """P2: return the full list of single tags for the question (normalized).

    Splitting rule: the category field is split into multiple tags by comma (M3-Bench / LVOmniBench format).
    If category is empty, fall back to the keyword rules, in which case the result is a single-element list.
    """
    category = (question.category or "").strip()
    if category:
        tags = [_normalize_tag(t) for t in category.split(",") if t.strip()]
        tags = [t for t in tags if t]
        if tags:
            return tags

    # Fallback: keyword rules
    q_text = (question.question or "").strip()
    for qtype, regexes in _QUESTION_TYPE_REGEX:
        if any(r.search(q_text) for r in regexes):
            return [qtype]
    return ["general"]


def _pick_primary_tag(tags: List[str]) -> str:
    """Select one tag from the single-tag list as the primary key, used for SkillPromoter clustering.

    Rule: first take the first one that hits in the order of _TAG_PRIORITY; if none hit, take the first one in alphabetical order.
    """
    if not tags:
        return "general"
    tag_set = set(tags)
    for t in _TAG_PRIORITY:
        if t in tag_set:
            return t
    return sorted(tags)[0]


def infer_question_type(question: TemporalQuestion) -> str:
    """Infer the "primary tag" (single tag) of the question.

    Historical behavior: previously it directly returned a comma-joined combined key, causing SkillPromoter clustering to produce
    42 highly sparse buckets. After P2, it was changed to split into multiple tags + select a primary tag, dropping the number of buckets to the single-tag
    dimension, making metric statistics and downstream win-rate judgments more reliable.
    """
    tags = infer_question_type_tags(question)
    return _pick_primary_tag(tags)


class LearningCapture:
    """Automatically captures experience events during the reasoning process.

    Design philosophy (inspired by OpenClaw):
    - Capture is automatic and requires no manual intervention
    - Stored as Markdown, human-readable and git-trackable
    - One file per day, to avoid a single file growing too large

    Args:
        learnings_dir: the directory for storing experience logs
        capture_successes: whether to also record successful cases
    """

    def __init__(self, learnings_dir: str, capture_successes: bool = True,
                 load_prior_learnings: bool = True):
        self.learnings_dir = Path(learnings_dir)
        self.learnings_dir.mkdir(parents=True, exist_ok=True)
        self.capture_successes = capture_successes
        self._counter = 0
        self._jsonl_path = self.learnings_dir / "learnings.jsonl"
        self._all_learnings = (
            self._load_existing_learnings() if load_prior_learnings else []
        )
        self._counter = self._infer_counter_from_existing(self._all_learnings)
        if not load_prior_learnings and self._jsonl_path.exists():
            logger.info(
                f"[LearningCapture] load_prior_learnings=False, ignoring historical structured experience: "
                f"{self._jsonl_path}"
            )

    def _persist_learning(self, learning: Learning) -> None:
        """Write both the human-readable Markdown and the machine-recoverable JSONL."""
        self._write_to_markdown(learning)
        self._write_to_jsonl(learning)

    def _write_to_jsonl(self, learning: Learning) -> None:
        """Write the structured sidecar, used to restore the experience loop across runs."""
        payload = asdict(learning)
        payload["learning_type"] = learning.learning_type.value
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_existing_learnings(self) -> List[Learning]:
        """Restore historical experience from the structured JSONL. Markdown is still kept for humans to read and does not participate in machine parsing."""
        if not self._jsonl_path.exists():
            return []
        loaded: List[Learning] = []
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lt_raw = obj.get("learning_type", LearningType.ERROR_CORRECTION.value)
                    obj["learning_type"] = LearningType(lt_raw)
                    valid_fields = Learning.__dataclass_fields__.keys()
                    kwargs = {k: v for k, v in obj.items() if k in valid_fields}
                    loaded.append(Learning(**kwargs))
                except Exception as exc:
                    logger.warning(
                        f"[LearningCapture] skipping unparseable historical learning "
                        f"{self._jsonl_path}:{line_no}: {exc}"
                    )
        if loaded:
            logger.info(
                f"[LearningCapture] loaded {len(loaded)} historical structured experience entries: "
                f"{self._jsonl_path}"
            )
        return loaded

    @staticmethod
    def _infer_counter_from_existing(learnings: List[Learning]) -> int:
        """Avoid learning_id repeating from 001 after a restart on the same day."""
        today = datetime.now().strftime("%Y%m%d")
        max_idx = 0
        prefix = f"LRN-{today}-"
        for l in learnings:
            lid = getattr(l, "learning_id", "") or ""
            if not lid.startswith(prefix):
                continue
            try:
                max_idx = max(max_idx, int(lid.rsplit("-", 1)[-1]))
            except Exception:
                continue
        return max_idx

    def capture_from_eval(self, question: TemporalQuestion,
                          answer: AgentAnswer,
                          is_correct: bool,
                          memory_stats: Dict[str, Any],
                          clip_id: int = -1) -> Optional[Learning]:
        """Automatically capture experience from a single QA evaluation.

        Capture rules:
        1. High confidence but wrong -> CONFIDENCE_CALIBRATION (most valuable)
        2. Still wrong after multiple retrieval rounds -> SEARCH_FAILURE
        3. Ordinary wrong answer -> ERROR_CORRECTION
        4. Correct in a single round -> SEARCH_SUCCESS (record the efficient strategy)
        """
        learning = None
        # P2: record both the single-tag list and the primary tag
        tags = infer_question_type_tags(question)
        question_type = _pick_primary_tag(tags)

        if not is_correct and answer.confidence >= 0.8:
            learning = self._create_learning(
                LearningType.CONFIDENCE_CALIBRATION,
                question, answer, is_correct, clip_id, question_type, tags,
                what_happened=f"The agent gave the wrong answer '{answer.content[:50]}' with {answer.confidence:.0%} confidence; the correct answer is '{question.answer[:50]}'",
            )
        elif not is_correct and answer.num_rounds >= 4:
            learning = self._create_learning(
                LearningType.SEARCH_FAILURE,
                question, answer, is_correct, clip_id, question_type, tags,
                what_happened=f"Still failed to find the correct information after {answer.num_rounds} retrieval rounds",
            )
        elif not is_correct:
            learning = self._create_learning(
                LearningType.ERROR_CORRECTION,
                question, answer, is_correct, clip_id, question_type, tags,
                what_happened=f"The agent answered '{answer.content[:50]}'; the correct answer is '{question.answer[:50]}'",
            )
        elif is_correct and answer.num_rounds <= 1 and self.capture_successes:
            learning = self._create_learning(
                LearningType.SEARCH_SUCCESS,
                question, answer, is_correct, clip_id, question_type, tags,
                what_happened=f"Answered correctly using only {answer.num_rounds} retrieval rounds",
            )
        elif (
            is_correct
            and answer.num_rounds >= 3
            and self.capture_successes
            # Exclude trivial "correct with zero retrieval" samples: the correctness of such samples comes from priors or the LLM's own knowledge,
            # and cannot reflect any "multi-step retrieval strategy", otherwise it would pollute the few-shot examples of the hard_win skill
            and self._has_meaningful_search(answer.search_queries)
        ):
            # HARD_WIN: multiple rounds + substantive retrieval actions + final correct answer -- a valuable multi-step reasoning demonstration
            learning = self._create_learning(
                LearningType.HARD_WIN,
                question, answer, is_correct, clip_id, question_type, tags,
                what_happened=f"Answered correctly after combining {answer.num_rounds} retrieval rounds, demonstrating an effective multi-step reasoning strategy",
            )

        if learning:
            self._all_learnings.append(learning)
            self._persist_learning(learning)

        return learning

    def _create_learning(self, learning_type: LearningType,
                         question: TemporalQuestion,
                         answer: AgentAnswer,
                         is_correct: bool,
                         clip_id: int,
                         question_type: str,
                         question_type_tags: list,
                         what_happened: str = "") -> Learning:
        """Create a single experience record."""
        self._counter += 1
        today = datetime.now().strftime("%Y%m%d")
        learning_id = f"LRN-{today}-{self._counter:03d}"

        return Learning(
            learning_id=learning_id,
            learning_type=learning_type,
            video_name=question.video_name,
            question_id=question.question_id,
            clip_id=clip_id,
            question=question.question,
            agent_answer=answer.content,
            ground_truth=question.answer,
            is_correct=is_correct,
            confidence=answer.confidence,
            question_type=question_type,
            question_type_tags=list(question_type_tags),
            what_happened=what_happened,
            search_queries_used=answer.search_queries,
            search_strategy_used=self._infer_strategy(answer.search_queries),
            num_rounds=answer.num_rounds,
            timestamp=datetime.now().isoformat(),
        )

    @staticmethod
    def _has_meaningful_search(search_queries) -> bool:
        """Determine whether an answer performed "substantive retrieval".

        Used to filter out trivial samples in the HARD_WIN scenario (zero retrieval or only 1 empty query); such samples:
          - derive their correctness from LLM priors/common sense rather than a multi-step retrieval strategy
          - have no few-shot reference value, and if mixed into skill examples would mislead the downstream agent

        Decision rule: at least 2 non-empty queries.
        """
        if not search_queries:
            return False
        valid = [q for q in search_queries if isinstance(q, str) and q.strip()]
        return len(valid) >= 2

    @staticmethod
    def _infer_strategy(search_queries) -> str:
        """Infer the retrieval strategy actually adopted by the agent from the prefixes of search_queries.

        Corresponds to the retrieval modes defined in SYSTEM_PROMPT in reasoning/multi_round.py:
          - KEYFRAME:<query>             -> keyframe_inspect (third visual layer, most critical)
          - VIDEO:<event_id>:<query>     -> video_drilldown
          - NEIGHBOR:<direction>:<query> -> neighbor_switch
          - others (direct query)        -> event_first

        Priority rule ("the most critical step"):
          KEYFRAME > VIDEO > NEIGHBOR > event_first
        This way, hard_win samples with vision naturally cluster in the keyframe_inspect bucket,
        and once the threshold is met, SkillPromoter will promote a sub-tag skill of the form
        `hard_win__<qt>__keyframe_inspect`.
        """
        if not search_queries:
            return ""
        has_keyframe = False
        has_video = False
        has_neighbor = False
        for q in search_queries:
            if not isinstance(q, str):
                continue
            qs = q.strip()
            if qs.upper().startswith("KEYFRAME:"):
                has_keyframe = True
            elif qs.upper().startswith("VIDEO:"):
                has_video = True
            elif qs.upper().startswith("NEIGHBOR:"):
                has_neighbor = True
        # Highest priority: any visual involvement is classified as keyframe_inspect (merged into hard_win as a sub-tag)
        if has_keyframe:
            return "keyframe_inspect"
        if has_video and has_neighbor:
            return "mixed_video_neighbor"
        if has_video:
            return "video_drilldown"
        if has_neighbor:
            return "neighbor_switch"
        return "event_first"

    def _write_to_markdown(self, learning: Learning):
        """Write the experience to the current day's Markdown file."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self.learnings_dir / f"{today}.md"

        entry = f"""
## [{learning.learning_id}] {learning.learning_type.value} | question_type={learning.question_type}

- **Video**: {learning.video_name}
- **Question**: {learning.question[:100]}
- **Agent answer**: {learning.agent_answer[:100]}
- **Correct answer**: {learning.ground_truth[:100]}
- **Confidence**: {learning.confidence:.2f}
- **Retrieval rounds**: {learning.num_rounds}

### What happened
{learning.what_happened}

---
"""
        with open(filepath, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(f"# Experience log: {today}\n\n")
            f.write(entry)

    @staticmethod
    def _normalize_ledger_subtasks(ledger_snapshot: dict) -> List[Dict[str, Any]]:
        """Handle both the dict structure of TaskLedger.to_dict() and the legacy list structure."""
        raw = ledger_snapshot.get("subtasks", []) if isinstance(ledger_snapshot, dict) else []
        if isinstance(raw, dict):
            order = ledger_snapshot.get("order") or []
            ordered = []
            seen = set()
            for sid in order:
                st = raw.get(sid)
                if isinstance(st, dict):
                    ordered.append(st)
                    seen.add(sid)
            for sid, st in raw.items():
                if sid not in seen and isinstance(st, dict):
                    ordered.append(st)
            return ordered
        if isinstance(raw, list):
            return [st for st in raw if isinstance(st, dict)]
        return []

    @staticmethod
    def _ledger_counts(subtasks: List[Dict[str, Any]], ledger_snapshot: dict) -> Dict[str, int]:
        """Extract stable counts from the snapshot; fall back to subtasks statistics when stats is missing."""
        stats = ledger_snapshot.get("stats") or {}
        resolved = stats.get("resolved")
        abandoned = stats.get("abandoned")
        if resolved is None:
            resolved = sum(1 for st in subtasks if st.get("status") == "resolved")
        if abandoned is None:
            abandoned = sum(1 for st in subtasks if st.get("status") == "abandoned")
        partial = sum(
            1 for st in subtasks
            if st.get("status") in ("partial", "searching", "pending")
        )
        return {
            "total": len(subtasks),
            "resolved": int(resolved or 0),
            "abandoned": int(abandoned or 0),
            "partial": int(partial or 0),
        }

    @staticmethod
    def _looks_like_should_decompose(question_text: str, question_tags: List[str]) -> bool:
        """Lightweight code-level guardrail: identify questions that are short but often require multi-evidence decomposition."""
        q = (question_text or "").strip().lower()
        tags = set(question_tags or [])
        multi_signal_tags = {
            "multi_hop_reasoning",
            "multi_detail_reasoning",
            "cross_modal_reasoning",
            "person_understanding",
            "temporal_order",
            "comparison",
            "counting",
            "reason_purpose",
        }
        if tags & multi_signal_tags:
            return True
        patterns = [
            r"\b(before|after|then|next|first|last)\b",
            r"\bhow many\b|\bnumber of\b|\bcount\b",
            r"\bwho\b.*\b(what|where|when|why|how)\b",
            r"\b(and|both|each|respectively)\b",
            r"\bwhy\b",
            r"\b(compare|same|different|more than|less than)\b",
        ]
        return any(re.search(p, q) for p in patterns)

    def _attach_ledger_metadata(
        self,
        learning: Learning,
        subtasks: List[Dict[str, Any]],
        ledger_snapshot: dict,
        counts: Dict[str, int],
        trigger_subtask: Optional[Dict[str, Any]] = None,
    ) -> Learning:
        """Attach the Ledger's internal signals to the Learning as structured fields, for P1 statistics."""
        learning.subtask_count = counts["total"]
        learning.resolved_count = counts["resolved"]
        learning.abandoned_count = counts["abandoned"]
        learning.partial_count = counts["partial"]
        learning.plan_version = int(ledger_snapshot.get("plan_version") or 0)
        learning.ledger_notes = list(ledger_snapshot.get("global_notes") or [])[-8:]
        learning.ledger_stats = dict(ledger_snapshot.get("stats") or {})
        if trigger_subtask:
            learning.subtask_question = trigger_subtask.get("question", "") or ""
            learning.subtask_status = trigger_subtask.get("status", "") or ""
            learning.subtask_attempts = int(trigger_subtask.get("attempts") or 0)
        return learning

    # ==================================================================
    # P0: subtask-level experience capture (extract fine-grained signals from the Ledger snapshot)
    # ==================================================================
    def capture_from_ledger(
        self,
        question: TemporalQuestion,
        answer: AgentAnswer,
        is_correct: bool,
        ledger_snapshot: dict,
        memory_stats: dict = None,
    ) -> list:
        """Extract subtask-level experience from the Ledger snapshot.

        Capture rules:
        1. SUBTASK_STALL: a subtask with attempts >= 3 but still not resolved (stall pattern)
        2. DECOMPOSE_WIN: multi-subtask decomposition + the whole question answered correctly (effective decomposition pattern)
        3. DECOMPOSE_FAIL: multi-subtask decomposition + the whole question answered wrong (ineffective decomposition pattern)

        Args:
            question: the original question
            answer: the agent's final answer
            is_correct: whether the whole question was answered correctly
            ledger_snapshot: the output of ledger.to_dict()
            memory_stats: memory statistics (optional)

        Returns:
            the list of captured Learnings
        """
        if not ledger_snapshot:
            return []

        captured = []
        subtasks = self._normalize_ledger_subtasks(ledger_snapshot)
        if not subtasks:
            return []

        tags = infer_question_type_tags(question)
        question_type = _pick_primary_tag(tags)
        counts = self._ledger_counts(subtasks, ledger_snapshot)
        subtask_count = counts["total"]
        resolved_count = counts["resolved"]
        abandoned_count = counts["abandoned"]
        partial_count = counts["partial"]
        plan_version = int(ledger_snapshot.get("plan_version") or 0)

        # ---- Rule 0: DECOMPOSE_MISS (should have decomposed but the Planner kept it atomic and then answered wrong) ----
        if (
            subtask_count == 1
            and not is_correct
            and self._looks_like_should_decompose(question.question, tags)
        ):
            learning = self._create_learning(
                LearningType.DECOMPOSE_MISS,
                question, answer, is_correct,
                clip_id=-1,
                question_type=question_type,
                question_type_tags=tags,
                what_happened=(
                    "Planner kept the question atomic, but this question has "
                    "multi-evidence/multi-step signals and the final answer was wrong."
                ),
            )
            self._attach_ledger_metadata(
                learning, subtasks, ledger_snapshot, counts, subtasks[0]
            )
            self._all_learnings.append(learning)
            self._persist_learning(learning)
            captured.append(learning)

        # ---- Rule 1: SUBTASK_STALL (subtask stalled) ----
        for st in subtasks:
            if (
                st.get("attempts", 0) >= 3
                and st.get("status") not in ("resolved",)
            ):
                learning = self._create_learning(
                    LearningType.SUBTASK_STALL,
                    question, answer, is_correct,
                    clip_id=-1,
                    question_type=question_type,
                    question_type_tags=tags,
                    what_happened=(
                        f"Subtask '{st.get('question', '')[:80]}' stalled after "
                        f"{st.get('attempts', 0)} attempts (status={st.get('status')}). "
                        f"Failed queries: {st.get('failed_queries', [])[:3]}"
                    ),
                )
                self._attach_ledger_metadata(
                    learning, subtasks, ledger_snapshot, counts, st
                )
                self._all_learnings.append(learning)
                self._persist_learning(learning)
                captured.append(learning)

        # ---- Rules 2/3: DECOMPOSE_WIN / DECOMPOSE_FAIL (effect of multi-subtask decomposition) ----
        # Only record when there are >= 2 subtasks (a single subtask = atomic, not counted as "decomposition")
        if subtask_count >= 2:
            if is_correct:
                learning = self._create_learning(
                    LearningType.DECOMPOSE_WIN,
                    question, answer, is_correct,
                    clip_id=-1,
                    question_type=question_type,
                    question_type_tags=tags,
                    what_happened=(
                        f"Decomposed into {subtask_count} subtasks, "
                        f"{resolved_count}/{subtask_count} resolved, "
                        f"abandoned={abandoned_count}, partial={partial_count}, "
                        f"plan_version={plan_version}. "
                        f"Subtasks: {[st.get('question','')[:50] for st in subtasks[:4]]}"
                    ),
                )
                self._attach_ledger_metadata(
                    learning, subtasks, ledger_snapshot, counts
                )
                self._all_learnings.append(learning)
                self._persist_learning(learning)
                captured.append(learning)
            else:
                learning = self._create_learning(
                    LearningType.DECOMPOSE_FAIL,
                    question, answer, is_correct,
                    clip_id=-1,
                    question_type=question_type,
                    question_type_tags=tags,
                    what_happened=(
                        f"Decomposed into {subtask_count} subtasks but FAILED. "
                        f"Only {resolved_count}/{subtask_count} resolved, "
                        f"abandoned={abandoned_count}, partial={partial_count}, "
                        f"plan_version={plan_version}. "
                        f"Subtasks: {[st.get('question','')[:50] for st in subtasks[:4]]}"
                    ),
                )
                self._attach_ledger_metadata(
                    learning, subtasks, ledger_snapshot, counts
                )
                self._all_learnings.append(learning)
                self._persist_learning(learning)
                captured.append(learning)

        return captured

    @property
    def all_learnings(self):
        return self._all_learnings
