"""
EventGraphOnlyMemory: ablation experiment that uses only the EventGraph, without VideoGraph fine-grained retrieval.

Used to verify the contribution of VideoGraph fine-grained memory to the final performance.
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


class EventGraphOnlyMemory(MemoryStrategy):
    """Ablation experiment: retrieve using only the EventGraph.

    Differences from HierarchicalMemory:
    - Retrieval uses only the event-level descriptions from the EventGraph
    - No VideoGraph drilldown is performed
    - VideoGraph is still needed to ingest clips (because the EventGraph depends on the memory text from VideoGraph)

    Args:
        config: configuration dictionary containing:
            - eventgraph_dir: EventGraph data directory
            - memory_config_path: path to the memory configuration file
    """

    def __init__(self, config: dict):
        self.config = config

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

        # EventGraph Retriever
        self.event_retriever = None
        eventgraph_dir = config.get("eventgraph_dir", "")
        if eventgraph_dir and os.path.exists(eventgraph_dir):
            self._init_event_retriever(eventgraph_dir)

        self._total_clips = 0
        self._current_video_name = ""
        self._mem_path = ""

        # Internalized memory generation configuration
        self._ingestion_config = IngestionConfig.from_harness_config(config)
        # Interval for periodically refreshing equivalence relations
        self._equivalence_refresh_interval = config.get("equivalence_refresh_interval", 5)

    def _init_event_retriever(self, eventgraph_dir: str):
        """Initialize the EventGraph retriever."""
        from mmagent.event_retrieve import EventGraphRetriever
        emb_cache_dir = os.path.join(eventgraph_dir, "emb_cache")
        os.makedirs(emb_cache_dir, exist_ok=True)
        self.event_retriever = EventGraphRetriever(
            eventgraph_dir=eventgraph_dir,
            emb_cache_dir=emb_cache_dir,
            embedding_model="text-embedding-v4",
            neighbors_topk=3,
        )
        logger.info(f"[EventGraphOnly] EventGraph initialization complete: {eventgraph_dir}")

    def on_video_start(self, video_name: str, **kwargs):
        """Reset or load the VideoGraph when a video starts being processed."""
        self._current_video_name = video_name
        self._mem_path = kwargs.get("mem_path", "")
        self._total_clips = 0

        if self._mem_path and os.path.exists(self._mem_path):
            self.video_graph = self._load_video_graph(self._mem_path)
            logger.info(f"[EventGraphOnly] Loaded prebuilt VideoGraph: {self._mem_path}")
        else:
            self.video_graph = self._VideoGraph(**self._memory_config)

    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Write the clip into the VideoGraph (the EventGraph depends on the memory text from VideoGraph)."""
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
        """Retrieve using only the EventGraph, without VideoGraph drilldown.

        Returns event-level descriptions (summary + neighbors), without fine-grained memories.
        """
        event_info = None

        if self.event_retriever and self._current_video_name:
            video_name = self.event_retriever.get_video_name(
                self._mem_path or self._current_video_name
            )
            mode = kwargs.get("mode", "event_first")
            focus_label = kwargs.get("focus_label", None)

            if mode == "neighbor" and focus_label:
                last_query = kwargs.get("last_query", query)
                focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                    video_name, focus_label, query=last_query, neighbors_topk=3,
                    before_clip=before_clip,
                )
                event_info = {
                    "mode": "event_neighbor",
                    "focus_label": focus_label,
                    "focus": focus_seg,
                    "neighbors": neighbors,
                    "edges": rel_edges,
                }
            else:
                best_seg, score = self.event_retriever.best_match(
                    video_name, query, before_clip=before_clip,
                )
                if best_seg:
                    focus_label = best_seg.get("segment_label")
                    focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                        video_name, focus_label, query=query, neighbors_topk=3,
                        before_clip=before_clip,
                    )
                    event_info = {
                        "mode": "event_first",
                        "query": query,
                        "event_score": score,
                        "focus_label": focus_label,
                        "focus": focus_seg,
                        "neighbors": neighbors,
                        "edges": rel_edges,
                    }

        return RetrievalResult(
            event_info=event_info,
            memories=None,  # Do not return VideoGraph fine-grained memories
            source="eventgraph_only",
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
            "has_eventgraph": self.event_retriever is not None,
            "strategy": "eventgraph_only",
        }
