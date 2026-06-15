"""
MemoryCompressor: memory compression tool.

Uses only existing VideoGraph interfaces:
- truncate_memory_by_clip: truncate the memory of early clips
- prune_memory_by_node_type: delete nodes of a given type

Supports two compression levels: light / medium
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class CompressionAdvice:
    """Compression advice."""
    should_compress: bool
    urgency: float                  # 0.0 - 1.0
    reason: str
    suggested_level: Optional[str]  # light / medium


@dataclass
class CompressionResult:
    """Compression result."""
    level: str
    clips_affected: List[int] = field(default_factory=list)
    nodes_removed: int = 0
    nodes_merged: int = 0
    nodes_summarized: int = 0
    nodes_before: int = 0
    nodes_after: int = 0
    compression_ratio: float = 0.0


class MemoryCompressor:
    """Memory compression tool.

    Compression strategy (only light and medium levels are kept):
    1. light: truncate the fine-grained memory of early clips (reuses the existing truncate_memory_by_clip)
    2. medium: on top of light, additionally delete semantic-type nodes (reuses the existing prune_memory_by_node_type)

    Feasibility note:
    - Uses only the existing VideoGraph interfaces truncate_memory_by_clip and prune_memory_by_node_type
    - Does not rely on non-existent interfaces such as remove_node, get_nodes_by_clip(type=...), downcast_embeddings

    Args:
        config: configuration dict
    """

    TOOL_NAME = "compress_memory"
    TOOL_DESCRIPTION = """Compress old memories to free up space. The agent can specify:
    - compression range: compress_before_clip (compress memories before this clip)
    - compression level: light (truncate early fine-grained memory) / medium (truncate + delete semantic nodes)"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.node_threshold = config.get("node_threshold", 500)
        self.auto_compress = config.get("auto_compress", True)

    def should_compress(self, memory_stats: Dict[str, Any],
                        current_clip: int) -> CompressionAdvice:
        """Decide whether compression is needed."""
        total_nodes = memory_stats.get("total_nodes", 0)

        if total_nodes < self.node_threshold:
            return CompressionAdvice(
                should_compress=False, urgency=0.0,
                reason="node count has not exceeded the threshold", suggested_level=None,
            )

        urgency = min(1.0, (total_nodes - self.node_threshold) / self.node_threshold)

        return CompressionAdvice(
            should_compress=True,
            urgency=urgency,
            reason=f"node count {total_nodes} exceeds threshold {self.node_threshold}",
            suggested_level="medium",  # only light/medium levels are supported
        )

    def compress(self, video_graph, compress_before_clip: int,
                 level: str = "medium") -> CompressionResult:
        """Perform memory compression.

        Args:
            video_graph: the VideoGraph instance
            compress_before_clip: compress memories before this clip
            level: compression level (light / medium)
        """
        stats_before = {"total_nodes": len(getattr(video_graph, 'text_nodes', []))}

        if level == "light":
            result = self._compress_light(video_graph, compress_before_clip)
        elif level == "medium":
            result = self._compress_medium(video_graph, compress_before_clip)
        else:
            logger.warning(f"unsupported compression level '{level}', falling back to medium")
            result = self._compress_medium(video_graph, compress_before_clip)

        stats_after = {"total_nodes": len(getattr(video_graph, 'text_nodes', []))}
        result.nodes_before = stats_before["total_nodes"]
        result.nodes_after = stats_after["total_nodes"]
        result.compression_ratio = 1 - (stats_after["total_nodes"] / max(stats_before["total_nodes"], 1))

        logger.info(
            f"[MemoryCompressor] level={level}, "
            f"nodes: {result.nodes_before} -> {result.nodes_after} "
            f"(ratio={result.compression_ratio:.1%})"
        )
        return result

    def _compress_light(self, video_graph, before_clip: int) -> CompressionResult:
        """Light compression: truncate the fine-grained memory of early clips.

        Reuses the existing interface: VideoGraph.truncate_memory_by_clip(clip_id)
        """
        video_graph.truncate_memory_by_clip(before_clip, refresh=True)
        return CompressionResult(
            level="light",
            clips_affected=list(range(0, before_clip)),
            nodes_removed=-1,  # truncate does not return the exact number removed
        )

    def _compress_medium(self, video_graph, before_clip: int) -> CompressionResult:
        """Medium compression: on top of light, additionally delete semantic-type nodes.

        Reuses the existing interfaces:
        - VideoGraph.truncate_memory_by_clip(clip_id)
        - VideoGraph.prune_memory_by_node_type('semantic')
        """
        result = self._compress_light(video_graph, before_clip)
        video_graph.prune_memory_by_node_type(node_type='semantic')
        result.level = "medium"
        return result
