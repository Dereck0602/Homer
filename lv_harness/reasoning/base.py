"""
ReasoningAgent: the unified abstract interface for reasoning agents.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ..data.types import TemporalQuestion, AgentAnswer
from ..memory.base import MemoryStrategy


class ReasoningAgent(ABC):
    """Unified interface for reasoning agents."""

    @abstractmethod
    def answer(self, question: TemporalQuestion,
               memory: MemoryStrategy) -> AgentAnswer:
        """Given a question and a memory system, return an answer."""
        ...

    def inject_instructions(self, instructions: str):
        """Inject extra instructions into the system prompt (optional)."""
        pass

    def set_search_strategy(self, strategy: str):
        """Switch the retrieval strategy (optional)."""
        pass

    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of reasoning rounds (optional)."""
        pass
