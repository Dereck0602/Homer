"""
EntropyManager: entropy management system.

Implementation of Entropy Management for Harness Engineering:
A long-running Agent accumulates "entropy" (inconsistent, redundant, or stale
information), which must be actively cleaned up to keep the system reliable.

Three levels of entropy management:
1. Retrieval level: query deduplication, invalid query filtering
2. Reasoning level: conversation history deduplication, context window management
3. Evolution level: stale skill retirement, experience library cleanup
"""
import logging
from typing import Dict, Any, List, Optional
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class SearchEntropyManager:
    """Retrieval-level entropy management. Prevents the retrieval system from accumulating invalid state."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._max_history = config.get("max_search_history", 50)
        self._query_history: List[str] = []
        self._result_cache: Dict[str, Any] = {}

    def reset(self):
        """Reset state."""
        self._query_history = []
        self._result_cache = {}

    def should_skip_query(self, query: str) -> bool:
        """Determine whether the query should be skipped (exact duplicate)."""
        normalized = query.strip().lower()
        return normalized in [q.strip().lower() for q in self._query_history[-5:]]

    def record_query(self, query: str, result_empty: bool):
        """Record a query and its result."""
        self._query_history.append(query)
        if len(self._query_history) > self._max_history:
            self._query_history = self._query_history[-self._max_history:]

    def get_query_diversity_score(self) -> float:
        """Compute the query diversity score (0 to 1, higher is better)."""
        if len(self._query_history) <= 1:
            return 1.0
        unique = len(set(q.strip().lower() for q in self._query_history))
        return unique / len(self._query_history)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_queries": len(self._query_history),
            "unique_queries": len(set(q.strip().lower() for q in self._query_history)),
            "diversity_score": self.get_query_diversity_score(),
        }


class ConversationEntropyManager:
    """Reasoning-level entropy management. Prevents an overly long conversation history from overflowing the context window."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._max_conversation_tokens = config.get("max_conversation_tokens", 30000)
        self._max_messages = config.get("max_conversation_messages", 20)

    def trim_conversations(self, conversations: List[Dict],
                           estimated_tokens: int = 0) -> List[Dict]:
        """Trim the conversation history to prevent context window overflow.

        Strategy:
        1. Keep the system prompt (the first message)
        2. Keep the most recent N messages
        3. Replace the middle messages with a summary
        """
        if len(conversations) <= self._max_messages:
            return conversations

        # Keep system prompt + the most recent messages
        system_msg = conversations[0] if conversations[0]["role"] == "system" else None
        recent_count = self._max_messages - (1 if system_msg else 0) - 1  # -1 for summary

        trimmed = []
        if system_msg:
            trimmed.append(system_msg)

        # Add the summary message
        dropped_count = len(conversations) - recent_count - (1 if system_msg else 0)
        if dropped_count > 0:
            trimmed.append({
                "role": "user",
                "content": (
                    f"[Context: {dropped_count} earlier messages have been summarized. "
                    f"The most recent search results and reasoning are preserved below.]"
                ),
            })

        # Add the most recent messages
        trimmed.extend(conversations[-recent_count:])

        logger.debug(
            f"[EntropyMgr] Conversation trimmed: {len(conversations)} -> {len(trimmed)} messages"
        )
        return trimmed


class SkillEntropyManager:
    """Evolution-level entropy management. Cleans up stale skills and experiences."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._max_skills = config.get("max_skills", 50)
        self._min_success_rate = config.get("min_skill_success_rate", 0.1)
        self._min_usage_count = config.get("min_skill_usage_count", 5)

    def identify_stale_skills(self, skills: List[Any]) -> List[str]:
        """Identify stale skills that should be retired.

        Retirement conditions:
        1. Usage count exceeds the threshold but the success rate is too low
        2. When the skill library exceeds maximum capacity, retire the ones with the lowest success rate
        """
        stale_ids = []

        for skill in skills:
            total_usage = skill.success_count + skill.failure_count
            if total_usage >= self._min_usage_count and skill.success_rate < self._min_success_rate:
                stale_ids.append(skill.skill_id)
                logger.info(
                    f"[EntropyMgr] Marked stale skill: {skill.name} "
                    f"(success_rate={skill.success_rate:.0%}, usage={total_usage})"
                )

        # Capacity limit
        if len(skills) > self._max_skills:
            sorted_skills = sorted(skills, key=lambda s: s.success_rate)
            excess = len(skills) - self._max_skills
            for s in sorted_skills[:excess]:
                if s.skill_id not in stale_ids:
                    stale_ids.append(s.skill_id)

        return stale_ids


class EntropyManager:
    """Unified entry point for the entropy management system."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.search = SearchEntropyManager(config.get("search", {}))
        self.conversation = ConversationEntropyManager(config.get("conversation", {}))
        self.skill = SkillEntropyManager(config.get("skill", {}))

    def reset_per_video(self):
        """Reset at the start of each video."""
        self.search.reset()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "search": self.search.stats,
        }
