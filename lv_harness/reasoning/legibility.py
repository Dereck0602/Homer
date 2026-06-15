"""
ApplicationLegibility: the application legibility layer.

An implementation of Harness Engineering's Application Legibility: it lets the
agent "see" and "understand" the internal state of the memory system, so it can
make better retrieval decisions.

Core idea (OpenAI):
"The system must be perceivable, understandable, and debuggable by the agent."
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MemoryStateDescriptor:
    """Turn the memory system state into a natural-language description the agent can understand.

    This is not a plain stats() dump, but structured information that helps the agent's decisions.
    """

    @staticmethod
    def describe(memory_stats: Dict[str, Any],
                 search_history: List[Dict] = None) -> str:
        """Produce an agent-readable description of the memory system state."""
        parts = []

        total_nodes = memory_stats.get("total_nodes", 0)
        total_clips = memory_stats.get("total_clips", 0)
        has_eventgraph = memory_stats.get("has_eventgraph", False)
        strategy = memory_stats.get("strategy", "unknown")

        parts.append(f"Memory: {total_nodes} nodes across {total_clips} clips")

        if has_eventgraph:
            parts.append("EventGraph: available (supports event-level search)")
        else:
            parts.append("EventGraph: not available (only VideoGraph fine-grained search)")

        if strategy == "sliding_window":
            window = memory_stats.get("current_window_clips", 0)
            parts.append(f"Strategy: sliding window (recent {window} clips)")
        elif strategy == "compressed":
            parts.append("Strategy: compressed (old memories are summarized)")

        return " | ".join(parts)


class SearchCapabilityDescriptor:
    """Describe the currently available retrieval capabilities to help the agent pick the right strategy."""

    @staticmethod
    def describe(has_eventgraph: bool,
                 focus_label: Optional[str] = None,
                 available_neighbors: List[str] = None) -> str:
        """Produce a description of the retrieval capabilities."""
        lines = ["Available search capabilities:"]

        lines.append("  1. Plain query → search EventGraph for relevant events"
                     if has_eventgraph else
                     "  1. Plain query → search VideoGraph for relevant memories")

        if has_eventgraph:
            lines.append("  2. VIDEO: <query> → drill down into fine-grained memories within current event")
            lines.append("  3. NEIGHBOR: <label> → explore adjacent events in EventGraph")

        if focus_label:
            lines.append(f"\nCurrent focus event: '{focus_label}'")

        if available_neighbors:
            neighbor_list = ", ".join(f"'{n}'" for n in available_neighbors[:5])
            lines.append(f"Available neighbors: {neighbor_list}")

        return "\n".join(lines)


class SystemContextBuilder:
    """Build the system context that is injected into the system prompt.

    Packages information such as memory state and retrieval capabilities into an agent-readable context block.
    """

    def __init__(self, enable_memory_status: bool = True,
                 enable_search_capabilities: bool = True):
        self._enable_memory_status = enable_memory_status
        self._enable_search_capabilities = enable_search_capabilities

    def build_context(self, memory_stats: Dict[str, Any],
                      has_eventgraph: bool,
                      focus_label: Optional[str] = None,
                      available_neighbors: List[str] = None,
                      round_idx: int = 0,
                      max_rounds: int = 5) -> str:
        """Build the system context string."""
        parts = []

        if self._enable_memory_status:
            status = MemoryStateDescriptor.describe(memory_stats)
            parts.append(f"[System Status] {status}")

        if self._enable_search_capabilities:
            caps = SearchCapabilityDescriptor.describe(
                has_eventgraph, focus_label, available_neighbors
            )
            parts.append(caps)

        parts.append(f"[Progress] Round {round_idx + 1}/{max_rounds}")

        return "\n".join(parts)
