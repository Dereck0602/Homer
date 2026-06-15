"""
TemporalQADataset: management of timestamped question-answer datasets.

Compatible with the existing videomme.json format, while extending support for temporal annotations.
"""
import json
import logging
from collections import defaultdict
from typing import List, Dict, Optional

from .types import TemporalQuestion

logger = logging.getLogger(__name__)


class TemporalQADataset:
    """Manage timestamped question-answer data.

    Compatible with the existing videomme.json format:
    - For data without temporal annotations, default ask_at_clips = [last_clip]
    - Degrades to the current offline evaluation mode

    Args:
        data_file: path to the data file (videomme.json format)
        temporal_mode: temporal mode
            - "end_of_video": all questions are asked at the last clip (default, backward compatible)
            - "periodic": ask once every N clips
            - "event_triggered": ask at event boundaries
        last_clip_id: ID of the last clip (used for end_of_video mode)
    """

    def __init__(self, data_file: str, temporal_mode: str = "end_of_video",
                 last_clip_id: int = -1):
        self.data_file = data_file
        self.temporal_mode = temporal_mode
        self.last_clip_id = last_clip_id
        self.questions: List[TemporalQuestion] = []
        self._by_clip: Dict[int, List[TemporalQuestion]] = defaultdict(list)
        self._by_video: Dict[str, List[TemporalQuestion]] = defaultdict(list)
        self._answered: set = set()

        self._load(data_file)

    def _load(self, data_file: str):
        """Load the data file, compatible with the videomme.json format."""
        with open(data_file, "r", encoding="utf-8") as f:
            datas = json.load(f)

        for video_key, v in datas.items():
            mem_path = v.get("mem_path", "")
            for qa in v.get("qa_list", []):
                qid = qa["question_id"]
                before_clip = qa.get("before_clip", None)

                # Determine ask_at_clips
                if before_clip is not None:
                    ask_at = [before_clip]
                    answerable_after = before_clip
                elif self.last_clip_id >= 0:
                    ask_at = [self.last_clip_id]
                    answerable_after = self.last_clip_id
                else:
                    # Deferred until set_last_clip is called
                    ask_at = []
                    answerable_after = -1

                # Compatible with multiple data formats:
                #   - lvomnibench/videomme: always has qa["option"] (A/B/C/D multiple choice), qa["answer"] like "A. xxx"
                #   - m3-bench (early robot/web/videomme versions): no option, qa["answer"] is free text
                # When options is an empty list, downstream reasoning/evaluator automatically switches to open-ended QA mode
                options = qa.get("option") or qa.get("options") or []
                # category is compatible with multiple fields: category / task_type / type (may be a list)
                category_raw = (
                    qa.get("category")
                    or qa.get("task_type")
                    or qa.get("type")
                    or ""
                )
                if isinstance(category_raw, list):
                    category = ",".join(str(c) for c in category_raw)
                else:
                    category = str(category_raw)

                tq = TemporalQuestion(
                    question_id=qid,
                    question=qa["question"],
                    options=options,
                    answer=qa["answer"],
                    answerable_after_clip=answerable_after,
                    ask_at_clips=ask_at,
                    category=category,
                    difficulty=qa.get("difficulty", ""),
                    video_name=video_key,
                    mem_path=mem_path,
                    before_clip=before_clip,
                )
                self.questions.append(tq)
                self._by_video[video_key].append(tq)

                for clip_id in ask_at:
                    self._by_clip[clip_id].append(tq)

        logger.info(f"Loaded {len(self.questions)} questions from {len(self._by_video)} videos")

    def set_last_clip(self, last_clip_id: int):
        """Set the ID of the last clip (for deferred setting in end_of_video mode)."""
        self.last_clip_id = last_clip_id
        # Rebuild the _by_clip index
        self._by_clip.clear()
        for q in self.questions:
            if not q.ask_at_clips:
                q.ask_at_clips = [last_clip_id]
                q.answerable_after_clip = last_clip_id
            for clip_id in q.ask_at_clips:
                self._by_clip[clip_id].append(q)

    def get_questions_at(self, clip_id: int) -> List[TemporalQuestion]:
        """Get the questions that need to be answered at the given clip."""
        return [q for q in self._by_clip.get(clip_id, [])
                if q.question_id not in self._answered]

    def get_questions_for_video(self, video_name: str) -> List[TemporalQuestion]:
        """Get all questions for the given video."""
        return self._by_video.get(video_name, [])

    def mark_answered(self, question_id: str):
        """Mark a question as answered."""
        self._answered.add(question_id)

    def get_all_pending(self, current_clip: int) -> List[TemporalQuestion]:
        """Get all questions answerable as of the current clip but not yet answered."""
        return [q for q in self.questions
                if q.answerable_after_clip <= current_clip
                and q.question_id not in self._answered]

    def __len__(self) -> int:
        return len(self.questions)
