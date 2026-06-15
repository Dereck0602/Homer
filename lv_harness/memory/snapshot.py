"""
SnapshotManager: manages point-in-time snapshots of the memory state and supports time travel.
"""
import os
import pickle
import logging
from typing import Dict, Optional

from .base import MemoryStrategy
from ..data.types import MemorySnapshot

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages point-in-time snapshots of the memory state.

    Args:
        interval: snapshot interval (save once every N clips)
        max_snapshots: maximum number of snapshots
        persist_dir: persistence directory (optional)
    """

    def __init__(self, interval: int = 10, max_snapshots: int = 50,
                 persist_dir: Optional[str] = None):
        self.interval = interval
        self.max_snapshots = max_snapshots
        self.persist_dir = persist_dir
        self._snapshots: Dict[int, MemorySnapshot] = {}

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    def maybe_save(self, clip_id: int, memory: MemoryStrategy) -> bool:
        """Save the current memory state if the snapshot interval is reached. Returns whether a snapshot was saved."""
        if self.interval <= 0:
            return False
        if clip_id % self.interval != 0:
            return False

        snapshot = memory.snapshot()
        self._snapshots[clip_id] = snapshot

        # Evict the oldest snapshot
        if len(self._snapshots) > self.max_snapshots:
            oldest = min(self._snapshots.keys())
            del self._snapshots[oldest]

        # Persist to disk
        if self.persist_dir:
            self._persist(clip_id, snapshot)

        logger.debug(f"Snapshot saved: clip_id={clip_id}")
        return True

    def restore_to(self, clip_id: int, memory: MemoryStrategy) -> int:
        """Restore to the nearest snapshot point. Returns the clip_id actually restored to."""
        available = sorted(k for k in self._snapshots if k <= clip_id)
        if not available:
            raise ValueError(f"No snapshot available (clip_id <= {clip_id})")
        target = available[-1]
        memory.restore(self._snapshots[target])
        logger.info(f"Memory restored to clip_id={target}")
        return target

    def _persist(self, clip_id: int, snapshot: MemorySnapshot):
        """Persist the snapshot to disk."""
        path = os.path.join(self.persist_dir, f"snapshot_{clip_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(snapshot, f)

    def has_snapshot(self, clip_id: int) -> bool:
        return clip_id in self._snapshots

    @property
    def snapshot_ids(self):
        return sorted(self._snapshots.keys())
