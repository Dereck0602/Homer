"""
MemoryInspector: let the agent inspect the current state of the memory system.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryInspector:
    """Let the agent inspect the current state of the memory system.

    Wraps the MemoryStrategy.stats() interface and provides a human-readable summary.
    """

    TOOL_NAME = "inspect_memory"
    TOOL_DESCRIPTION = """Inspect the current state of the memory system, including:
    - total number of nodes
    - number of processed clips
    - distribution of node types
    - whether an EventGraph exists"""

    def inspect(self, memory_stats: Dict[str, Any]) -> str:
        """Produce a human-readable summary of the memory state."""
        lines = [
            "=== Memory System State ===",
            f"Total nodes: {memory_stats.get('total_nodes', 'N/A')}",
            f"Processed clips: {memory_stats.get('total_clips', 'N/A')}",
            f"Current video: {memory_stats.get('video_name', 'N/A')}",
            f"EventGraph: {'loaded' if memory_stats.get('has_eventgraph') else 'not loaded'}",
        ]
        return "\n".join(lines)
