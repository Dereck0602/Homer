"""
CompressedMemory: an experimental strategy that periodically compresses old memories while keeping recent ones at full detail.

Inspiration: the human memory consolidation process, where recent memories retain details and distant memories keep only the key points.
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


class CompressedMemory(MemoryStrategy):
    """Experimental strategy: periodically compress old memories.

    Strategy:
    - The memories of the most recent recent_window clips are kept at full detail (episodic + semantic)
    - Old memories beyond recent_window are compressed (truncate fine-grained memories, remove semantic nodes)
    - Compression is triggered once every compress_interval clips

    Args:
        config: configuration dictionary containing:
            - recent_window: full-detail recent window size, default 20
            - compress_interval: compression trigger interval, default 10
            - topk: retrieval top_k
            - memory_config_path: path to the memory configuration file
    """

    def __init__(self, config: dict):
        self.config = config
        self.recent_window = config.get("recent_window", 20)
        self.compress_interval = config.get("compress_interval", 10)
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

        self._total_clips = 0
        self._clips_since_compress = 0
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
        self._clips_since_compress = 0

        if self._mem_path and os.path.exists(self._mem_path):
            self.video_graph = self._load_video_graph(self._mem_path)
            self._total_clips = len(getattr(self.video_graph, 'text_nodes_by_clip', {}))
            logger.info(f"[CompressedMemory] Loaded prebuilt VideoGraph: {self._mem_path}")
        else:
            self.video_graph = self._VideoGraph(**self._memory_config)

    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Ingest a clip and periodically trigger compression."""
        process_clip(
            video_graph=self.video_graph,
            base64_video=clip_data.base64_video,
            base64_frames=clip_data.base64_frames,
            clip_id=clip_id,
            clip_path=clip_data.path,
            ingestion_config=self._ingestion_config,
        )
        self._total_clips += 1
        self._clips_since_compress += 1

        # Periodically refresh equivalence relations
        if self._total_clips % self._equivalence_refresh_interval == 0:
            self.video_graph.refresh_equivalences()

        # Periodically trigger compression
        if (self._clips_since_compress >= self.compress_interval
                and self._total_clips > self.recent_window):
            self._compress(clip_id)
            self._clips_since_compress = 0

    def _compress(self, current_clip_id: int):
        """Compress old memories.

        Strategy:
        1. Truncate the fine-grained memory of clips before recent_window (truncate_memory_by_clip)
        2. Delete semantic-type nodes (prune_memory_by_node_type)
        """
        cutoff_clip = max(0, current_clip_id - self.recent_window)
        if cutoff_clip <= 0:
            return

        nodes_before = len(getattr(self.video_graph, 'text_nodes', []))

        # Step 1: truncate the fine-grained memory of old clips
        self.video_graph.truncate_memory_by_clip(cutoff_clip, refresh=True)

        # Step 2: delete semantic-type nodes (global deletion, frees more space)
        if hasattr(self.video_graph, 'prune_memory_by_node_type'):
            self.video_graph.prune_memory_by_node_type(node_type='semantic')

        nodes_after = len(getattr(self.video_graph, 'text_nodes', []))
        logger.info(f"[CompressedMemory] compression complete: cutoff_clip={cutoff_clip}, "
                     f"nodes: {nodes_before} -> {nodes_after}")

    def retrieve(self, query: str, top_k: int = 5,
                 before_clip: Optional[int] = None,
                 **kwargs) -> RetrievalResult:
        """Retrieve from the compressed memory."""
        from mmagent.retrieve import search

        # Rely only on search(before_clip=...) internal timestamp filtering; _compress already manages
        # historical compression logic, so we do not truncate again here, avoiding shallow-copy side effects
        # and the issue of before_clip=0 silently failing.
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
            source="compressed",
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
            "recent_window": self.recent_window,
            "compress_interval": self.compress_interval,
            "video_name": self._current_video_name,
            "has_eventgraph": False,
            "strategy": "compressed",
        }
