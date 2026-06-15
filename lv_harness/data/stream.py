"""
VideoStreamSimulator: turn an offline video into a time-ordered stream at clip granularity.
"""
import glob
import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional

from .types import ClipData

logger = logging.getLogger(__name__)

# Lazy import, to avoid errors in environments without mmagent
_process_video_clip = None

def _get_process_video_clip():
    global _process_video_clip
    if _process_video_clip is None:
        from mmagent.utils.video_processing import process_video_clip
        _process_video_clip = process_video_clip
    return _process_video_clip


class VideoStreamSimulator:
    """Turn an offline video into a streaming clip sequence.

    Supports multiple modes:
    - sequential: yield clips one by one in clip_id order
    - batch: yield N clips at a time (simulating batch processing)

    Args:
        clip_dir: directory of clip files (containing 0.mp4, 1.mp4, ... etc.)
        mode: streaming mode, "sequential" or "batch"
        batch_size: number of clips yielded per step in batch mode
    """

    def __init__(self, clip_dir: str, mode: str = "sequential", batch_size: int = 1):
        self.clip_dir = clip_dir
        self.mode = mode
        self.batch_size = batch_size
        self._clips = self._discover_clips()

    def _discover_clips(self):
        """Discover and sort all clip files."""
        clips = sorted(
            glob.glob(f"{self.clip_dir}/*.mp4"),
            key=lambda x: int(Path(x).stem),
        )
        if not clips:
            logger.warning(f"No .mp4 files found in {self.clip_dir}")
        return clips

    @property
    def total_clips(self) -> int:
        return len(self._clips)

    def __iter__(self) -> Iterator[Tuple[int, ClipData]]:
        """Yield (clip_id, clip_data) pairs."""
        process_fn = _get_process_video_clip()
        for clip_path in self._clips:
            clip_id = int(Path(clip_path).stem)
            try:
                base64_video, base64_frames, base64_audio = process_fn(clip_path)
                if not base64_frames:
                    logger.warning(f"Clip {clip_id} has no valid frames, skipping")
                    continue
                clip_data = ClipData(
                    clip_id=clip_id,
                    base64_video=base64_video,
                    base64_frames=base64_frames,
                    base64_audio=base64_audio,
                    path=clip_path,
                )
                yield clip_id, clip_data
            except Exception as e:
                logger.error(f"Failed to load clip {clip_id}: {e}")
                continue

    def __len__(self) -> int:
        return self.total_clips
