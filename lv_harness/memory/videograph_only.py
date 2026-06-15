"""
VideoGraphOnlyMemory: ablation experiment that uses only the VideoGraph, without the EventGraph.

Used to verify the contribution of the EventGraph to the final performance.
"""
import os
import json
import pickle
import logging
from typing import Dict, Any, Optional

from .base import MemoryStrategy
from .clip_ingestion import IngestionConfig, process_clip
from ..data.types import ClipData, RetrievalResult, MemorySnapshot

logger = logging.getLogger(__name__)


class VideoGraphOnlyMemory(MemoryStrategy):
    """Ablation experiment: perform memory and retrieval using only the VideoGraph.

    Differences from HierarchicalMemory:
    - Does not use the EventGraph
    - Retrieval performs semantic search directly on the VideoGraph, without event-level localization

    Args:
        config: configuration dictionary containing:
            - videograph: VideoGraph configuration
            - memory_config_path: path to the memory configuration file
            - topk: retrieval top_k
    """

    # Fields not accepted by VideoGraph; they must be filtered out before passing arguments
    _NON_VIDEOGRAPH_KEYS = {"visual_layer"}

    @classmethod
    def _filter_videograph_kwargs(cls, memory_config: dict) -> dict:
        """Strip non-VideoGraph fields from memory_config to avoid TypeError."""
        if not isinstance(memory_config, dict):
            return {}
        return {k: v for k, v in memory_config.items() if k not in cls._NON_VIDEOGRAPH_KEYS}

    def __init__(self, config: dict):
        self.config = config
        self._topk = config.get("topk", 2)

        from mmagent.videograph import VideoGraph
        from mmagent.utils.general import load_video_graph

        memory_config_path = config.get("memory_config_path", "configs/memory_config.json")
        if os.path.exists(memory_config_path):
            with open(memory_config_path) as f:
                memory_config = json.load(f)
        else:
            memory_config = config.get("videograph", {})

        self.video_graph = VideoGraph(**self._filter_videograph_kwargs(memory_config))
        self._load_video_graph = load_video_graph
        self._VideoGraph = VideoGraph
        self._memory_config = memory_config

        self._total_clips = 0
        self._current_video_name = ""
        self._mem_path = ""

        # Internalized memory generation configuration
        self._ingestion_config = IngestionConfig.from_harness_config(config)
        # Interval for periodically refreshing equivalence relations
        self._equivalence_refresh_interval = config.get("equivalence_refresh_interval", 5)

    def on_video_start(self, video_name: str, **kwargs):
        """Reset or load the VideoGraph when a video starts being processed."""
        self._current_video_name = video_name
        self._mem_path = kwargs.get("mem_path", "")
        self._total_clips = 0

        if self._mem_path and os.path.exists(self._mem_path):
            self.video_graph = self._load_video_graph(self._mem_path)
            logger.info(f"[VideoGraphOnly] Loaded prebuilt VideoGraph: {self._mem_path}")
        else:
            self.video_graph = self._VideoGraph(**self._filter_videograph_kwargs(self._memory_config))

    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Write the clip into the VideoGraph."""
        process_clip(
            video_graph=self.video_graph,
            base64_video=clip_data.base64_video,
            base64_frames=clip_data.base64_frames,
            clip_id=clip_id,
            clip_path=clip_data.path,
            ingestion_config=self._ingestion_config,
        )
        self._total_clips += 1

        # Periodically refresh equivalence relations
        if self._total_clips % self._equivalence_refresh_interval == 0:
            self.video_graph.refresh_equivalences()

    def retrieve(self, query: str, top_k: int = 5,
                 before_clip: Optional[int] = None,
                 **kwargs) -> RetrievalResult:
        """Retrieve directly on the VideoGraph, without going through the EventGraph."""
        from mmagent.retrieve import search

        # Rely only on the internal timestamp filtering of search(before_clip=...), to avoid the side
        # effects of truncate and the issue where truncate silently fails when before_clip=0 and there
        # is no node with timestamp==0.
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
            source="videograph_only",
        )

    def snapshot(self) -> MemorySnapshot:
        data = pickle.dumps(self.video_graph)
        return MemorySnapshot(
            clip_id=self._total_clips,
            data=data,
            stats=self.stats(),
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        self.video_graph = pickle.loads(snapshot.data)
        self._total_clips = snapshot.clip_id

    def stats(self) -> Dict[str, Any]:
        total_nodes = len(getattr(self.video_graph, 'text_nodes', []))
        return {
            "total_nodes": total_nodes,
            "total_clips": self._total_clips,
            "video_name": self._current_video_name,
            "has_eventgraph": False,
            "strategy": "videograph_only",
        }
