"""
MemoryStrategy: a unified abstract interface for memory strategies.

Every memory strategy must implement this interface so that the HarnessOrchestrator can invoke them uniformly.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..data.types import ClipData, RetrievalResult, MemorySnapshot


class MemoryStrategy(ABC):
    """A unified abstract interface for memory strategies."""

    @abstractmethod
    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Ingest a new clip and update the internal memory state."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5,
                 before_clip: Optional[int] = None,
                 **kwargs) -> RetrievalResult:
        """Retrieve relevant memories based on the query."""
        ...

    @abstractmethod
    def snapshot(self) -> MemorySnapshot:
        """Export a snapshot of the current memory state."""
        ...

    @abstractmethod
    def restore(self, snapshot: MemorySnapshot) -> None:
        """Restore the memory state from a snapshot."""
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return statistics about the memory system."""
        ...

    def on_video_start(self, video_name: str, **kwargs) -> None:
        """Callback invoked when a video starts being processed (optional implementation)."""
        pass

    def on_video_end(self, video_name: str, **kwargs) -> None:
        """Callback invoked when a video finishes being processed (optional implementation)."""
        pass
