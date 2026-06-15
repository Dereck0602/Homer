"""
Guardrails: Agent behavior constraint system.

Implementation of the Constrain pillar of Harness Engineering:
- Mandatory output format validation (the answer must be a valid option)
- Search query guardrail (prevent invalid/duplicate queries)
- Token/time budget control
- Self-repair loop (validation failure -> retry with the error message)
"""
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """Resource budget configuration."""
    max_tokens_per_question: int = 50000      # Maximum token consumption for a single question
    max_time_per_question: float = 300.0      # Maximum time spent on a single question (seconds)
    max_retries_on_format_error: int = 2      # Maximum number of retries on a format error
    max_consecutive_empty_searches: int = 2   # Maximum number of consecutive empty retrieval results
    max_similar_queries: int = 2              # Maximum number of similar queries allowed


@dataclass
class GuardrailState:
    """Guardrail state during a single question-answering process."""
    tokens_used: int = 0
    start_time: float = 0.0
    search_queries: List[str] = field(default_factory=list)
    consecutive_empty_searches: int = 0
    format_retries: int = 0
    # Grace round mechanism: give the Agent one chance to adjust the first time an early-stop condition is triggered
    grace_round_given: bool = False
    # Hint pending injection when the grace round is triggered (set by BudgetGuardrail, consumed by multi_round)
    pending_grace_hint: str = ""
    # Information saturation soft hint has been issued (set by SufficiencySignal)
    sufficiency_soft_warning: bool = False
    # Context for the identity resolution guardrail
    question_text: str = ""                              # Current question text (used to extract person names)
    retrieval_log: List[str] = field(default_factory=list)  # Text summary of each round's retrieval result
    identity_retry_given: bool = False                   # The retry for identity-confusion refusals has been used

    def reset(self):
        self.tokens_used = 0
        self.start_time = time.time()
        self.search_queries = []
        self.consecutive_empty_searches = 0
        self.format_retries = 0
        self.grace_round_given = False
        self.pending_grace_hint = ""
        self.sufficiency_soft_warning = False
        self.question_text = ""
        self.retrieval_log = []
        self.identity_retry_given = False


class OutputFormatValidator:
    """Output format validator: a deterministic hook that does not depend on the LLM.

    Validates whether the Agent's output conforms to the expected format:
    - Answer action: the content must contain a valid option letter (A/B/C/D)
    - Search action: the query must not be empty or too short
    """

    VALID_OPTIONS = {"A", "B", "C", "D"}
    # Compatible with the following formats:
    #   - Action: [Search]  Content: ...
    #   - Action: Search    Content: ...          <- the model often omits the brackets
    #   - action: search    content: ...          <- case drift
    ACTION_PATTERN = re.compile(
        r"Action:\s*\[?\s*(Answer|Search)\s*\]?\s*.*?Content:\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def validate_answer(cls, content: str, options: List[str]) -> Tuple[bool, str]:
        """Validate whether the answer content contains a valid option.

        Returns:
            (is_valid, error_message)
        """
        if not content or not content.strip():
            return False, "Answer content is empty."

        text = content.strip()

        # Open-ended QA mode (no options): the answer is valid as long as it is non-empty
        if not options:
            return True, ""

        # Extract the option letter
        letter_match = re.match(r"^\s*([A-Da-d])\b", text)
        if letter_match:
            letter = letter_match.group(1).upper()
            if letter in cls.VALID_OPTIONS:
                return True, ""
            return False, f"Option letter '{letter}' is not valid. Must be one of A, B, C, D."

        # Check whether it contains option content
        for opt in options:
            opt_clean = re.sub(r'^[A-Da-d]\.\s*', '', opt).strip()
            if opt_clean and len(opt_clean) > 3 and opt_clean.lower() in text.lower():
                return True, ""

        return False, (
            "Answer does not contain a valid option letter (A/B/C/D) or recognizable option content. "
            "Please output your answer starting with the option letter."
        )

    @classmethod
    def validate_search_query(cls, content: str) -> Tuple[bool, str]:
        """Validate whether the search query is valid."""
        if not content or not content.strip():
            return False, "Search query is empty. Please provide a specific search query."

        query = content.strip()
        # Strip the prefix
        for prefix in ("VIDEO:", "NEIGHBOR:", "KEYFRAME:"):
            if query.upper().startswith(prefix):
                query = query[len(prefix):].strip()

        if len(query) < 3:
            return False, f"Search query '{query}' is too short. Please provide a more specific query."

        return True, ""

    @classmethod
    def validate_action_format(cls, response: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """Validate whether the response contains a valid Action format.

        Returns:
            (is_valid, error_message, action, content)
        """
        text = response.split("</think>")[-1] if "</think>" in response else response
        match = cls.ACTION_PATTERN.search(text)
        if match:
            action = match.group(1).strip().capitalize()  # "search"/"SEARCH" -> "Search"
            content = match.group(2).strip()
            return True, "", action, content

        return False, (
            "Response does not contain a valid Action format. "
            "Please output in the format:\n"
            "Reason: <your reasoning>\n"
            "Action: [Answer] or [Search]\n"
            "Content: <content>"
        ), None, None


class IdentityResolutionGuardrail:
    """Identity resolution guardrail: prevents the Agent from refusing to answer due to struggling over character_N <-> name.

    Scenario: in M3-Bench open-ended QA, events returned by EventGraph often use placeholders
    like `<character_0>` to refer to characters. The Agent often falls into a loop of "cannot explicitly bind
    <character_0> to Nancy", and finally outputs "The information does not state ...".

    This guardrail detects two failure modes:
      1. The Agent is about to [Answer] but the content is a typical "insufficient evidence refusal", and the
         retrieval history already contains a placeholder plus the person named in the question -> inject guidance that
         "a default binding can be made" and require a retry.
      2. The Agent consecutively searches identity-binding queries such as "name of <character_N>" / "is character_N -> Name"
         >= 2 times -> inject guidance to "stop struggling, assume the unification and answer".
    """

    # "Refusal" keyword patterns
    _REFUSAL_PATTERNS = [
        r"does not (?:explicitly )?(?:state|mention|indicate|specify|contain)",
        r"not (?:mentioned|specified|indicated|stated) in the",
        r"no information (?:about|on|regarding)",
        r"the (?:provided|retrieved) (?:information|knowledge|evidence) (?:does not|doesn't)",
        r"cannot (?:confirm|determine|identify|verify)",
        r"it is (?:not (?:clear|known|possible)|unclear|unknown)",
        r"insufficient (?:information|evidence)",
    ]
    _REFUSAL_REGEX = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

    # "Identity-binding query" patterns (search content style)
    _IDENTITY_QUERY_REGEX = re.compile(
        r"(?:name of\s*<?character_?\d+>?|is\s*<?character_?\d+>?\s*(?:named|is)|"
        r"who is\s*<?character_?\d+>?|identif\w*\s*<?character_?\d+>?)",
        re.IGNORECASE,
    )

    # Placeholder detection
    _PLACEHOLDER_REGEX = re.compile(r"<?character_?\d+>?|<?face_?\d+>?", re.IGNORECASE)

    # Simple heuristic for the subject person name (a Token starting with an uppercase letter)
    _NAME_TOKEN_REGEX = re.compile(r"\b([A-Z][a-z]{2,})\b")

    @classmethod
    def _looks_like_refusal(cls, content: str) -> bool:
        if not content:
            return False
        return bool(cls._REFUSAL_REGEX.search(content))

    @classmethod
    def _extract_question_name(cls, question: str) -> Optional[str]:
        """Roughly extract a person name from the question (the first word starting with an uppercase letter, excluding common leading words)."""
        if not question:
            return None
        STOP = {"Does", "Do", "Is", "Are", "What", "Where", "When", "Why",
                "How", "Who", "Which", "Will", "Can", "The", "A", "An",
                "Has", "Have", "Had"}
        for m in cls._NAME_TOKEN_REGEX.finditer(question):
            tok = m.group(1)
            if tok not in STOP:
                return tok
        return None

    @classmethod
    def check_premature_refusal(cls, answer_content: str, question: str,
                                search_history: List[str],
                                retrieval_log: List[str]) -> Tuple[bool, str]:
        """The Agent wants to [Answer] but the output is an identity-confusion type refusal.

        Trigger conditions:
          - answer_content matches the refusal pattern
          - a person name can be extracted from the question
          - a `<character_N>` placeholder has appeared in the retrieval history/log
        """
        if not cls._looks_like_refusal(answer_content):
            return False, ""
        name = cls._extract_question_name(question)
        if not name:
            return False, ""
        blob = " ".join(retrieval_log or []) + " " + " ".join(search_history or [])
        if not cls._PLACEHOLDER_REGEX.search(blob):
            return False, ""

        return True, (
            f"IDENTITY_RESOLUTION: You are about to refuse because you could not "
            f"explicitly bind a `<character_N>` placeholder to the name '{name}'. "
            f"This is usually unnecessary. Follow this rule:\n"
            f"  - If the retrieved event is clearly about the scenario asked and "
            f"only ONE named person ('{name}') is involved per the question, "
            f"treat the placeholder(s) in that event as referring to '{name}' "
            f"and proceed to answer, stating the assumption briefly in Reason.\n"
            f"  - Only refuse if the retrieved evidence truly contains no relevant "
            f"information about the asked fact (not just about the ID mapping).\n"
            f"Please rewrite your response: give a concrete Answer grounded in the "
            f"event you already retrieved, using real names (not placeholders)."
        )

    @classmethod
    def check_identity_query_loop(cls, new_query: str,
                                  search_history: List[str]) -> Tuple[bool, str]:
        """Detect a loop of identity-binding searches."""
        if not new_query or not cls._IDENTITY_QUERY_REGEX.search(new_query):
            return False, ""
        identity_count = sum(
            1 for q in search_history if cls._IDENTITY_QUERY_REGEX.search(q or "")
        )
        if identity_count >= 1:  # Adding the current query makes this the 2nd time
            return True, (
                "IDENTITY_RESOLUTION_LOOP: You have already issued an identity-"
                "resolution search (binding <character_N> to a name) before. "
                "Stop searching for identity mappings. Instead, assume the named "
                "person in the question corresponds to the main character in the "
                "relevant event you already retrieved, and proceed to either:\n"
                "  1. [Answer] with that assumption (state it in Reason), or\n"
                "  2. Search for the ACTUAL fact being asked (not the identity), "
                "using `VIDEO: <query about the fact>` to drill down."
            )
        return False, ""


class SearchQueryGuardrail:
    """Search query guardrail: prevents invalid and duplicate queries."""

    def __init__(self, max_similar: int = 2):
        self._max_similar = max_similar

    def check_duplicate(self, new_query: str, history: List[str]) -> Tuple[bool, str]:
        """Check whether the new query is too similar to historical queries.

        Returns:
            (is_duplicate, suggestion_message)
        """
        if not history:
            return False, ""

        new_lower = new_query.lower().strip()
        # Strip the prefix
        for prefix in ("video:", "neighbor:", "keyframe:"):
            if new_lower.startswith(prefix):
                new_lower = new_lower[len(prefix):].strip()

        similar_count = 0
        for old_query in history:
            old_lower = old_query.lower().strip()
            for prefix in ("video:", "neighbor:", "keyframe:"):
                if old_lower.startswith(prefix):
                    old_lower = old_lower[len(prefix):].strip()

            # Exactly identical
            if new_lower == old_lower:
                similar_count += 1
            # Highly overlapping (Jaccard similarity > 0.7)
            elif self._jaccard_similarity(new_lower, old_lower) > 0.7:
                similar_count += 1

        if similar_count >= self._max_similar:
            return True, (
                f"You have already searched similar queries {similar_count} times. "
                "Please try a completely different search strategy:\n"
                "- Use different keywords\n"
                "- Try VIDEO: prefix to drill down into fine-grained memories\n"
                "- Try NEIGHBOR: prefix to explore adjacent events\n"
                "- Or consider answering with the information you already have"
            )
        return False, ""

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Compute the Jaccard similarity of two strings (based on word sets)."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


class BudgetGuardrail:
    """Resource budget guardrail: controls token and time consumption."""

    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig()

    def check_budget(self, state: GuardrailState) -> Tuple[bool, str]:
        """Check whether the budget is exceeded.

        Grace Round mechanism:
        - Time/Token budget exceeded: hard stop (no grace)
        - Consecutive empty searches/information saturation: give the Agent one grace round the first time it is triggered;
          hard stop only when it is still triggered after the grace round

        Returns:
            (is_exceeded, reason)
        """
        # Time budget (hard stop, no grace)
        elapsed = time.time() - state.start_time
        if elapsed > self.config.max_time_per_question:
            return True, f"Time budget exceeded ({elapsed:.0f}s > {self.config.max_time_per_question:.0f}s)"

        # Token budget (hard stop, no grace)
        if state.tokens_used > self.config.max_tokens_per_question:
            return True, f"Token budget exceeded ({state.tokens_used} > {self.config.max_tokens_per_question})"

        # Consecutive empty searches / information saturation (with grace round)
        if state.consecutive_empty_searches >= self.config.max_consecutive_empty_searches:
            if not state.grace_round_given:
                # First trigger: grant a grace round, no hard stop
                state.grace_round_given = True
                # Reset the consecutive empty search count to give the Agent a chance to start over
                state.consecutive_empty_searches = 0
                # Set the pending hint so that multi_round appends it in the next round's retrieval result
                state.pending_grace_hint = (
                    "\u26a0\ufe0f EMPTY RESULTS WARNING: Your recent searches returned no useful results. "
                    "You have ONE more chance before being forced to answer. Consider:\n"
                    "  1. If you already have enough information, output [Answer] now.\n"
                    "  2. If you must search, switch to a different approach:\n"
                    "     - Use 'VIDEO: <query>' to search fine-grained details (character actions, dialogues)\n"
                    "     - Use 'NEIGHBOR: <segment_label>' to explore a different event segment\n"
                    "     - Use completely different keywords to find a different event\n"
                    "  If the next search still returns empty, you will be forced to answer."
                )
                logger.info(
                    f"[Guardrail] grace round: consecutive empty searches reached the threshold; give the Agent one round to adjust"
                )
                return False, ""
            else:
                # The grace round is used up; hard stop
                return True, (
                    f"Too many consecutive empty search results after grace round "
                    f"({state.consecutive_empty_searches} >= {self.config.max_consecutive_empty_searches})"
                )

        return False, ""


class AgentGuardrails:
    """Unified entry point for the Agent behavior constraint system.

    Integrates all guardrail components and provides a unified checking interface.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        budget_cfg = BudgetConfig(
            max_tokens_per_question=config.get("max_tokens_per_question", 50000),
            max_time_per_question=config.get("max_time_per_question", 300.0),
            max_retries_on_format_error=config.get("max_retries_on_format_error", 2),
            max_consecutive_empty_searches=config.get("max_consecutive_empty_searches", 2),
            max_similar_queries=config.get("max_similar_queries", 2),
        )
        self.budget_config = budget_cfg
        self.format_validator = OutputFormatValidator()
        self.query_guardrail = SearchQueryGuardrail(max_similar=budget_cfg.max_similar_queries)
        self.budget_guardrail = BudgetGuardrail(budget_cfg)
        self.identity_guardrail = IdentityResolutionGuardrail()
        self._state = GuardrailState()
        # Switch: whether to enable the identity resolution guardrail (only useful in open-ended QA scenarios)
        self._identity_check_enabled = config.get("enable_identity_guardrail", True)

    def reset(self):
        """Reset the state (called at the start of each new question)."""
        self._state.reset()

    @property
    def state(self) -> GuardrailState:
        return self._state

    def validate_response(self, response: str, options: List[str],
                          require_prior_search: bool = False
                          ) -> Tuple[str, Optional[str], str]:
        """Validate the Agent response and return the corrected action and content.

        Args:
            response: the raw LLM response
            options: the list of options (empty for open-ended QA)
            require_prior_search: if True, directly [Answer] is forbidden when no [Search]
                has been performed yet (used in open-ended QA to prevent hallucinated direct answers in the first round).

        Returns:
            (action, content, feedback_message)
            - an empty feedback_message means validation passed
            - a non-empty feedback_message means a retry is needed; the content is the feedback for the Agent
        """
        # Step 1: validate the Action format
        fmt_valid, fmt_err, action, content = self.format_validator.validate_action_format(response)
        if not fmt_valid:
            self._state.format_retries += 1
            if self._state.format_retries > self.budget_config.max_retries_on_format_error:
                # Exceeded the retry limit; force parsing
                return "Search", None, ""
            return "", None, fmt_err

        # Step 2: validate the Answer content
        if action == "Answer":
            # Open-ended QA defense: wants to answer directly before any search in the first round; force a Search first
            if require_prior_search and not self._state.search_queries:
                return "", None, (
                    "PREMATURE_ANSWER: You have not performed any [Search] yet. "
                    "The initial user message is only a placeholder and contains NO "
                    "retrieval result. You MUST issue at least one [Search] before "
                    "answering. Please output Action: [Search] with a concrete query."
                )
            ans_valid, ans_err = self.format_validator.validate_answer(content, options)
            if not ans_valid:
                self._state.format_retries += 1
                if self._state.format_retries > self.budget_config.max_retries_on_format_error:
                    return "Answer", content, ""  # Exceeded the retry limit; accept the original answer
                return "", None, ans_err

            # Identity resolution guardrail: detect refusal-type answers plus a placeholder that has appeared
            # Enabled only for open-ended QA (options is empty), and triggered at most once per question
            if (self._identity_check_enabled and not options
                    and not self._state.identity_retry_given
                    and self._state.search_queries):
                is_refusal, fb = self.identity_guardrail.check_premature_refusal(
                    content,
                    self._state.question_text,
                    self._state.search_queries,
                    self._state.retrieval_log,
                )
                if is_refusal:
                    self._state.identity_retry_given = True
                    logger.info(
                        "[IdentityGuardrail] detected an identity-confusion type refusal; injecting default-binding guidance"
                    )
                    return "", None, fb

            return "Answer", content, ""

        # Step 3: validate the Search query
        if action == "Search":
            q_valid, q_err = self.format_validator.validate_search_query(content)
            if not q_valid:
                return "", None, q_err

            # Identity resolution loop detection (open-ended QA only): the Agent consecutively searches identity-binding questions
            if (self._identity_check_enabled and not options):
                is_loop, loop_msg = self.identity_guardrail.check_identity_query_loop(
                    content, self._state.search_queries
                )
                if is_loop:
                    logger.info(
                        "[IdentityGuardrail] detected an identity-confusion query loop; interrupting"
                    )
                    return "", None, loop_msg

            # Check for duplicate queries
            is_dup, dup_msg = self.query_guardrail.check_duplicate(
                content, self._state.search_queries
            )
            if is_dup:
                return "", None, dup_msg

            self._state.search_queries.append(content)
            return "Search", content, ""

        return action, content, ""

    def check_budget(self) -> Tuple[bool, str]:
        """Check the resource budget."""
        return self.budget_guardrail.check_budget(self._state)

    def on_search_result(self, is_empty: bool, retrieval_text: str = ""):
        """Retrieval result callback.

        Args:
            is_empty: whether this round's retrieval is empty
            retrieval_text: the text summary of the retrieval result (used for subsequent identity resolution detection)
        """
        if is_empty:
            self._state.consecutive_empty_searches += 1
        else:
            self._state.consecutive_empty_searches = 0
        if retrieval_text:
            # Keep only the first 500 characters to avoid memory bloat
            self._state.retrieval_log.append(retrieval_text[:500])

    def set_question_text(self, question: str):
        """Set the current question text (for the identity resolution guardrail to extract person names)."""
        self._state.question_text = question or ""

    def on_tokens_used(self, tokens: int):
        """Token consumption callback."""
        self._state.tokens_used += tokens
