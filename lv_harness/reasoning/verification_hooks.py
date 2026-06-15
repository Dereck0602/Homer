"""
VerificationHooks: deterministic verification hooks.

Implementation of the Verify pillar of Harness Engineering:
- Answer format verification (deterministic checks that do not rely on an LLM)
- Retrieval result quality assessment
- Triggering of the self-repair loop
- Runtime exception monitoring
"""
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class AnswerVerificationHook:
    """Answer verification hook. Performs deterministic checks after the Agent generates an answer.

    Checks:
    1. Whether the answer contains a valid option letter (A/B/C/D)
    2. Whether the answer is empty
    3. Whether the answer contains a placeholder that should not appear (e.g., <character_0>)
    """

    PLACEHOLDER_PATTERN = re.compile(r"<(?:character|face|person)_\d+>")
    OPTION_LETTER_PATTERN = re.compile(r"\b([A-Da-d])\b")

    def on_answer_generated(self, **kwargs):
        """Triggered after the Agent generates an answer."""
        answer = kwargs.get("answer")
        question = kwargs.get("question")
        if not answer or not question:
            return

        content = answer.content
        issues = []

        # Check for an empty answer
        if not content or not content.strip():
            issues.append("EMPTY_ANSWER: the Agent returned an empty answer")

        # Check for placeholders
        if content and self.PLACEHOLDER_PATTERN.search(content):
            placeholders = self.PLACEHOLDER_PATTERN.findall(content)
            issues.append(
                f"PLACEHOLDER_LEAK: the answer contains unresolved placeholders: {placeholders}"
            )

        # Only check option letters for multiple-choice questions (open-ended questions have empty question.options, skip)
        is_multiple_choice = bool(getattr(question, "options", None))
        if (
            is_multiple_choice
            and content
            and not self.OPTION_LETTER_PATTERN.search(content)
        ):
            issues.append("NO_OPTION_LETTER: no option letter (A/B/C/D) found in the answer")

        if issues:
            for issue in issues:
                logger.warning(f"[AnswerVerification] {issue} | Q: {question.question_id}")


class SearchResultVerificationHook:
    """Search result verification hook. Performs a quality assessment after each retrieval.

    Checks:
    1. Whether the retrieval result is empty
    2. Whether the focus node of the retrieval result exists
    3. The count of consecutive empty retrievals
    """

    def __init__(self):
        self._consecutive_empty = 0
        self._total_searches = 0
        self._empty_searches = 0

    def on_search_executed(self, **kwargs):
        """Triggered after a retrieval completes."""
        result = kwargs.get("result")
        query = kwargs.get("query", "")
        self._total_searches += 1

        if not result:
            self._consecutive_empty += 1
            self._empty_searches += 1
            logger.warning(
                f"[SearchVerification] Empty retrieval result | query='{query[:60]}' | "
                f"consecutive empty retrievals: {self._consecutive_empty}"
            )
            return

        # Check the focus node
        event_info = result.event_info if hasattr(result, 'event_info') else result
        if isinstance(event_info, dict):
            if not event_info.get("focus"):
                self._consecutive_empty += 1
                self._empty_searches += 1
                logger.debug(
                    f"[SearchVerification] Retrieval has no focus node | query='{query[:60]}'"
                )
            else:
                self._consecutive_empty = 0
        else:
            self._consecutive_empty = 0

    def on_video_start(self, **kwargs):
        """Reset the count when switching videos."""
        self._consecutive_empty = 0

    def on_harness_end(self, **kwargs):
        """Output statistics when the harness ends."""
        if self._total_searches > 0:
            empty_rate = self._empty_searches / self._total_searches
            logger.info(
                f"[SearchVerification] Retrieval statistics: "
                f"{self._total_searches} total, "
                f"{self._empty_searches} empty results ({empty_rate:.1%})"
            )


class RuntimeHealthHook:
    """Runtime health monitoring hook.

    Monitored items:
    1. Question processing failure rate
    2. LLM call latency
    3. Abnormal memory system state
    """

    def __init__(self):
        self._total_questions = 0
        self._failed_questions = 0
        self._total_correct = 0

    def on_answer_evaluated(self, **kwargs):
        """Triggered after answer evaluation."""
        record = kwargs.get("record")
        if not record:
            return

        self._total_questions += 1
        if record.is_correct:
            self._total_correct += 1

        # Check for an empty answer (treated as a failure)
        if not record.answer or not record.answer.strip():
            self._failed_questions += 1
            logger.warning(
                f"[RuntimeHealth] Empty answer | Q: {record.question_id} | "
                f"failure rate: {self._failed_questions}/{self._total_questions}"
            )

    def on_harness_end(self, **kwargs):
        """Output a health report when the harness ends."""
        if self._total_questions > 0:
            fail_rate = self._failed_questions / self._total_questions
            acc = self._total_correct / self._total_questions
            logger.info(
                f"[RuntimeHealth] Health report: "
                f"{self._total_questions} total questions, "
                f"{self._failed_questions} failed ({fail_rate:.1%}), "
                f"accuracy {self._total_correct}/{self._total_questions} ({acc:.1%})"
            )
            if fail_rate > 0.1:
                logger.warning(
                    f"[RuntimeHealth] WARNING: failure rate too high ({fail_rate:.1%} > 10%), "
                    f"please check the LLM connection and prompt configuration"
                )
