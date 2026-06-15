"""
HookSystem: the lifecycle hook system.

Supports inserting custom logic at key points (logging, monitoring, visualization, etc.).
"""
import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)


class HookSystem:
    """The lifecycle hook system.

    Hook Points:
    - on_harness_start: Harness starts
    - on_video_start: start processing a video
    - on_clip_loaded: a clip finished loading
    - on_clip_ingested: a clip finished being written to memory
    - on_snapshot_saved: a snapshot finished saving
    - on_question_received: received a pending question
    - on_search_executed: a retrieval completed
    - on_answer_generated: the agent generated an answer
    - on_answer_evaluated: an answer finished being evaluated
    - on_video_complete: a video finished processing
    - on_batch_complete: a batch finished
    - on_harness_end: Harness ends
    """

    HOOK_POINTS = [
        "on_harness_start",
        "on_video_start",
        "on_clip_loaded",
        "on_clip_ingested",
        "on_snapshot_saved",
        "on_question_received",
        "on_search_executed",
        "on_answer_generated",
        "on_answer_evaluated",
        "on_video_complete",
        "on_batch_complete",
        "on_harness_end",
        # Self-evolution system hook points
        "on_learning_captured",
        "on_skill_promoted",
        "on_skill_activated",
        "on_wisdom_updated",
        "on_wisdom_distilled",
        # Verification system hook points (Harness Engineering: the Verify pillar)
        "on_format_validation_failed",
        "on_self_heal_triggered",
        "on_budget_exceeded",
        "on_duplicate_query_detected",
    ]

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {p: [] for p in self.HOOK_POINTS}

    def register(self, hook_point: str, callback: Callable) -> None:
        """Register a hook callback."""
        if hook_point not in self._hooks:
            logger.warning(f"Unknown hook point: {hook_point}, created automatically")
            self._hooks[hook_point] = []
        self._hooks[hook_point].append(callback)

    def trigger(self, hook_point: str, **kwargs) -> None:
        """Trigger all callbacks for the given hook point."""
        for callback in self._hooks.get(hook_point, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Hook {hook_point} callback execution failed: {e}")

    def register_hook_object(self, hook_obj: Any) -> None:
        """Register a hook object, automatically binding all its on_* methods."""
        for point in self.HOOK_POINTS:
            method = getattr(hook_obj, point, None)
            if method and callable(method):
                self.register(point, method)


class ProgressHook:
    """tqdm progress bar Hook."""

    def __init__(self, total_clips: int = 0, total_questions: int = 0):
        self._total_clips = total_clips
        self._total_questions = total_questions
        self._clip_pbar = None
        self._correct = 0
        self._total_answered = 0

    def on_harness_start(self, **kwargs):
        from tqdm import tqdm
        if self._total_clips > 0:
            self._clip_pbar = tqdm(total=self._total_clips, desc="Processing clips")

    def on_clip_ingested(self, **kwargs):
        if self._clip_pbar:
            self._clip_pbar.update(1)

    def on_answer_evaluated(self, **kwargs):
        record = kwargs.get("record")
        if record:
            self._total_answered += 1
            if record.is_correct:
                self._correct += 1
            acc = self._correct / self._total_answered if self._total_answered > 0 else 0
            logger.info(f"[Eval] {self._correct}/{self._total_answered} = {acc:.1%}")

    def on_harness_end(self, **kwargs):
        if self._clip_pbar:
            self._clip_pbar.close()


class LoggingHook:
    """Logging Hook."""

    def on_harness_start(self, **kwargs):
        logger.info("=" * 60)
        logger.info("LV-Harness started")
        logger.info("=" * 60)

    def on_video_start(self, **kwargs):
        video_name = kwargs.get("video_name", "unknown")
        logger.info(f"Start processing video: {video_name}")

    def on_answer_evaluated(self, **kwargs):
        record = kwargs.get("record")
        if record:
            status = "OK" if record.is_correct else "X"
            logger.info(
                f"  {status} Q: {record.question[:60]}... "
                f"A: {record.answer[:40]}... "
                f"GT: {record.ground_truth[:40]}..."
            )

    def on_harness_end(self, **kwargs):
        results = kwargs.get("results", {})
        logger.info("=" * 60)
        logger.info("LV-Harness complete")
        if "accuracy" in results:
            acc = results["accuracy"]
            logger.info(f"Accuracy: {acc.get('correct', 0)}/{acc.get('total', 0)} = {acc.get('accuracy', 0):.1%}")
        logger.info("=" * 60)
