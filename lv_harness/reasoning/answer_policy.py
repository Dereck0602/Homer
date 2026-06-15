"""
AnswerPolicy: decides when the agent should produce an answer.
"""
from abc import ABC, abstractmethod
from ..data.types import AgentAnswer, TemporalQuestion


class AnswerPolicy(ABC):
    """Decides when the agent should produce an answer."""

    @abstractmethod
    def should_answer(self, agent_answer: AgentAnswer,
                      current_clip: int, question: TemporalQuestion) -> bool:
        ...


class AlwaysAnswer(AnswerPolicy):
    """Always answer immediately (current default behavior)."""
    def should_answer(self, *args) -> bool:
        return True


class ConfidentAnswer(AnswerPolicy):
    """Answer only when confidence exceeds the threshold, otherwise defer."""
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def should_answer(self, agent_answer: AgentAnswer, *args) -> bool:
        return agent_answer.confidence >= self.threshold


class DeferredAnswer(AnswerPolicy):
    """Deferred answering: if the current information is insufficient, wait for more clips and retry.

    This is the core innovation for streaming scenarios: the agent can say
    "I'm not sure yet, let me see more content".
    """
    def __init__(self, max_defer_clips: int = 10):
        self.max_defer_clips = max_defer_clips

    def should_answer(self, agent_answer: AgentAnswer,
                      current_clip: int, question: TemporalQuestion) -> bool:
        if agent_answer.confidence >= 0.8:
            return True
        if question.answerable_after_clip >= 0:
            if current_clip - question.answerable_after_clip >= self.max_defer_clips:
                return True  # waited too long, force an answer
        return False
