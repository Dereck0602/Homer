"""
SlidingWindowMemory: a baseline strategy that keeps only the memory of the most recent N clips.

Used to verify the advantage of long-term memory (VideoGraph + EventGraph) over a simple sliding window.
"""
import os
import json
import pickle
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional

from .base import MemoryStrategy
from .clip_ingestion import IngestionConfig, process_clip
from ..data.types import ClipData, RetrievalResult, MemorySnapshot

logger = logging.getLogger(__name__)


class SlidingWindowMemory(MemoryStrategy):
    """Baseline strategy: keep only the memory of the most recent N clips.

    Simulates a fixed-size memory window:
    - When a new clip arrives, write it into the VideoGraph
    - When the number of clips exceeds the window size, truncate the earliest clip memories
    - During retrieval, search only within the memories inside the window

    Args:
        config: configuration dictionary containing:
            - window_size: sliding window size (keep the most recent N clips), default 20
            - topk: retrieval top_k
            - memory_config_path: path to the memory configuration file
    """

    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get("window_size", 20)
        self._topk = config.get("topk", 2)

        from mmagent.videograph import VideoGraph
        from mmagent.utils.general import load_video_graph

        memory_config_path = config.get("memory_config_path", "configs/memory_config.json")
        if os.path.exists(memory_config_path):
            with open(memory_config_path) as f:
                memory_config = json.load(f)
        else:
            memory_config = config.get("videograph", {})

        self.video_graph = VideoGraph(**memory_config)
        self._load_video_graph = load_video_graph
        self._VideoGraph = VideoGraph
        self._memory_config = memory_config

        # Ordered list of ingested clip_ids
        self._clip_ids: list = []
        self._total_clips = 0
        self._current_video_name = ""
        self._mem_path = ""

        # Internalized memory generation configuration
        self._ingestion_config = IngestionConfig.from_harness_config(config)

    def on_video_start(self, video_name: str, **kwargs):
        """Reset or load the VideoGraph when a video starts being processed."""
        self._current_video_name = video_name
        self._mem_path = kwargs.get("mem_path", "")
        self._total_clips = 0
        self._clip_ids = []

        if self._mem_path and os.path.exists(self._mem_path):
            self.video_graph = self._load_video_graph(self._mem_path)
            # Restore clip_ids from the loaded VideoGraph
            clip_ids = sorted(getattr(self.video_graph, 'text_nodes_by_clip', {}).keys())
            self._clip_ids = list(clip_ids)
            self._total_clips = len(self._clip_ids)
            # Truncate if it exceeds the window size
            self._enforce_window()
            logger.info(f"[SlidingWindow] Loaded prebuilt VideoGraph: {self._mem_path}, "
                        f"keeping the most recent {len(self._clip_ids)} clips")
        else:
            self.video_graph = self._VideoGraph(**self._memory_config)

    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Ingest a clip and maintain the sliding window."""
        process_clip(
            video_graph=self.video_graph,
            base64_video=clip_data.base64_video,
            base64_frames=clip_data.base64_frames,
            clip_id=clip_id,
            clip_path=clip_data.path,
            ingestion_config=self._ingestion_config,
        )
        self._clip_ids.append(clip_id)
        self._total_clips += 1

        # Maintain the sliding window
        self._enforce_window()

    def _enforce_window(self):
        """Truncate the earliest clip memories when the number of clips exceeds the window size."""
        if len(self._clip_ids) > self.window_size:
            # Number of clips that need to be truncated
            cutoff_count = len(self._clip_ids) - self.window_size
            cutoff_clip_id = self._clip_ids[cutoff_count]  # Keep memories starting from this clip
            self.video_graph.truncate_memory_by_clip(cutoff_clip_id, refresh=True)
            self._clip_ids = self._clip_ids[cutoff_count:]
            logger.debug(f"[SlidingWindow] Truncated to clip_id >= {cutoff_clip_id}, "
                         f"current window: {len(self._clip_ids)} clips")

    def retrieve(self, query: str, top_k: int = 5,
                 before_clip: Optional[int] = None,
                 **kwargs) -> RetrievalResult:
        """Retrieve within the memories inside the sliding window."""
        from mmagent.retrieve import search

        # Rely only on the internal timestamp filtering of search(before_clip=...); truncate has already
        # maintained the window boundary in _enforce_window, so it is not repeated here, to avoid the
        # shallow-copy side effects seen earlier and the issue where truncate silently fails when
        # before_clip=0 and there is no node with timestamp==0.
        mem_node = self.video_graph
        mem_node.refresh_equivalences()

        memories, _, _ = search(
            mem_node, query, [],
            threshold=0.5, mem_wise=True,
            topk=self._topk,
            before_clip=before_clip,
        )

        return RetrievalResult(
            event_info=None,
            memories=memories,
            source="sliding_window",
        )

    def snapshot(self) -> MemorySnapshot:
        data = pickle.dumps((self.video_graph, self._clip_ids))
        return MemorySnapshot(
            clip_id=self._total_clips,
            data=data,
            stats=self.stats(),
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        self.video_graph, self._clip_ids = pickle.loads(snapshot.data)
        self._total_clips = snapshot.clip_id

    def stats(self) -> Dict[str, Any]:
        total_nodes = len(getattr(self.video_graph, 'text_nodes', []))
        return {
            "total_nodes": total_nodes,
            "total_clips": self._total_clips,
            "window_size": self.window_size,
            "current_window_clips": len(self._clip_ids),
            "video_name": self._current_video_name,
            "has_eventgraph": False,
            "strategy": "sliding_window",
        }
