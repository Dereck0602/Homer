"""
FeedbackEnhancer: enhanced feedback loop.

Implementation of the Feedback Loop enhancement for Harness Engineering:
- Retrieval result quality assessment (without relying on an LLM)
- Repeated query detection and strategy-switching suggestions
- Smart feedback on empty results (guiding the Agent to change strategy)
- Reasoning process deduplication
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class SearchFeedbackEnhancer:
    """Search feedback enhancer. Provides the Agent with smarter retrieval feedback.

    Core idea: rather than simply passing the retrieval results back as-is, attach
    structured feedback information to help the Agent make a better next decision.
    """

    def __init__(self):
        self._search_history: List[Dict] = []
        self._strategy_usage: Counter = Counter()

    def reset(self):
        """Reset state (called at the start of each new video)."""
        self._search_history = []
        self._strategy_usage = Counter()

    def enhance_feedback(self, query: str, retrieval_payload: Dict,
                         round_idx: int, max_rounds: int) -> str:
        """Enhance the feedback information for retrieval results.

        On top of the raw retrieval results, attach:
        1. Result quality assessment
        2. Remaining rounds reminder
        3. Strategy-switching suggestions
        """
        # Record search history
        strategy = self._detect_strategy(query)
        self._strategy_usage[strategy] += 1
        self._search_history.append({
            "query": query,
            "strategy": strategy,
            "has_focus": bool(retrieval_payload.get("focus")),
            "round": round_idx,
        })

        # Build base feedback
        feedback_parts = []

        # Remaining rounds reminder
        remaining = max_rounds - round_idx - 1
        if remaining <= 2:
            feedback_parts.append(
                f"⚠️ Only {remaining} search round(s) remaining. "
                f"Consider answering if you have enough information."
            )

        # Result quality assessment
        is_empty = not retrieval_payload.get("focus")
        if is_empty:
            suggestion = self._suggest_on_empty(strategy)
            feedback_parts.append(suggestion)

        # Strategy usage statistics
        if len(self._search_history) >= 3:
            dominant = self._strategy_usage.most_common(1)[0]
            if dominant[1] >= 3:
                feedback_parts.append(
                    f"Note: You have used '{dominant[0]}' strategy {dominant[1]} times. "
                    f"Consider trying a different approach."
                )

        return "\n".join(feedback_parts)

    def _detect_strategy(self, query: str) -> str:
        """Detect the strategy type used by the query."""
        q = query.strip().upper()
        if q.startswith("VIDEO:"):
            return "video_drilldown"
        elif q.startswith("NEIGHBOR:"):
            return "neighbor_walk"
        elif q.startswith("KEYFRAME:"):
            return "keyframe_inspect"
        else:
            return "event_search"

    def _suggest_on_empty(self, current_strategy: str) -> str:
        """Strategy-switching suggestions when results are empty."""
        suggestions = {
            "event_search": (
                "The event search returned no results. Suggestions:\n"
                "- Try rephrasing your query with different keywords\n"
                "- Use more general terms to match event summaries\n"
                "- If you previously found a relevant event, try VIDEO: to drill down"
            ),
            "video_drilldown": (
                "The video drilldown returned no fine-grained memories. Suggestions:\n"
                "- The current event may not contain the details you need\n"
                "- Try NEIGHBOR: to explore adjacent events\n"
                "- Or reformulate with a plain query to search different events"
            ),
            "neighbor_walk": (
                "The neighbor walk found no relevant adjacent events. Suggestions:\n"
                "- Try a plain query to search for a completely different event\n"
                "- Or consider answering with the information you already have"
            ),
            "keyframe_inspect": (
                "No keyframe images could be loaded for the current event's clips. Suggestions:\n"
                "- Use VIDEO: to get textual fine-grained details first\n"
                "- Try NEIGHBOR: to shift focus to an adjacent event whose clips may have keyframes\n"
                "- Or reformulate with a plain query to locate a different event"
            ),
        }
        return suggestions.get(current_strategy, "Try a different search strategy.")


class ConversationDeduplicator:
    """Conversation deduplicator. Prevents excessive redundant information from accumulating during reasoning.

    The reasoning-level manifestation of Harness Engineering's Entropy Management.
    """

    @staticmethod
    def deduplicate_search_results(all_results: List[Dict]) -> List[Dict]:
        """Remove duplicate retrieval results (deduplicated by focus_label)."""
        seen_labels = set()
        deduped = []
        for result in all_results:
            retrieval = result.get("retrieval", {}).get("event_retrieval", {})
            label = retrieval.get("focus_label", "")
            if label and label in seen_labels:
                continue
            if label:
                seen_labels.add(label)
            deduped.append(result)
        return deduped

    @staticmethod
    def summarize_search_history(all_results: List[Dict], max_entries: int = 5) -> List[Dict]:
        """When the search history is too long, keep only the most recent N entries."""
        if len(all_results) <= max_entries:
            return all_results
        # Keep the first entry (initial context) and the most recent N-1 entries
        return [all_results[0]] + all_results[-(max_entries - 1):]
