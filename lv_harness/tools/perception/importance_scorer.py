"""
ClipImportanceScorer: assess clip importance to decide the memory generation granularity.

Design principles:
- Use rules / a lightweight model, consuming no LLM calls
- Output an importance score + directive (skip/brief/normal/critical)
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImportanceResult:
    """Importance assessment result."""
    score: float           # 0.0 ~ 1.0
    directive: str         # skip / brief / normal / critical
    reason: str = ""


class ClipImportanceScorer:
    """Assess clip importance.

    Level 1 (rule-driven) implementation:
    - Based on the clip's position in the video (beginning/end are more important)
    - Based on the clip's frame count / duration
    - Based on the visual difference from the previous clip (if available)

    Args:
        config: configuration dict
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.skip_threshold = config.get("skip_threshold", 0.2)
        self.critical_threshold = config.get("critical_threshold", 0.8)

    def score(self, clip_id: int, total_clips: int,
              num_frames: int = 0,
              visual_diff: float = 0.0,
              has_speech: bool = True) -> ImportanceResult:
        """Assess clip importance.

        Args:
            clip_id: the current clip ID
            total_clips: total number of clips
            num_frames: frame count
            visual_diff: visual difference from the previous clip [0, 1]
            has_speech: whether it contains speech

        Returns:
            ImportanceResult
        """
        score = 0.5  # base score

        # Position weighting: beginning and end are more important
        if total_clips > 0:
            position = clip_id / total_clips
            if position < 0.1 or position > 0.9:
                score += 0.15

        # Visual difference weighting: scene changes are more important
        if visual_diff > 0.5:
            score += 0.2 * visual_diff

        # Speech weighting: clips with speech are more important
        if has_speech:
            score += 0.1

        # Frame count weighting: few frames may indicate a transition segment
        if num_frames > 0 and num_frames < 3:
            score -= 0.15

        score = max(0.0, min(1.0, score))

        # Determine the directive
        if score < self.skip_threshold:
            directive = "skip"
        elif score < 0.4:
            directive = "brief"
        elif score < self.critical_threshold:
            directive = "normal"
        else:
            directive = "critical"

        return ImportanceResult(
            score=score,
            directive=directive,
            reason=f"pos={clip_id}/{total_clips}, visual_diff={visual_diff:.2f}, speech={has_speech}",
        )
