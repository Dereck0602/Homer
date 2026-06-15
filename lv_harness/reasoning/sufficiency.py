"""
InformationSufficiencySignal: information sufficiency signal.

A core enhancement of Harness Engineering:
Using a deterministic method (without relying on an LLM), it assesses "whether the
currently accumulated information is sufficient to answer the question", and attaches
the assessment as a structured signal to the feedback, helping the Agent decide whether
to keep searching or to answer.

Assessment along two dimensions:
1. Information Increment: the amount of new information added across two consecutive retrieval rounds
2. Saturation Detection: early-stop triggering that works in concert with the Guardrail
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class InformationIncrementAssessor:
    """Information increment assessor. Measures the amount of new information each retrieval round brings.

    Core idea:
    - Convert each round's retrieval results into an "information fingerprint" (a keyword set + focus_label)
    - Compare the information fingerprints of two adjacent rounds and compute the new-information ratio
    - When the new-information ratio stays below a threshold for N consecutive rounds, declare "information saturation"

    Coordination with the Guardrail:
    - On information saturation, notify the Guardrail to trigger a budget-exceeded condition via sufficiency_signal
    - This is not a hard block but an "advisory signal"; the final decision is left to the Agent
    """

    def __init__(self, config: dict = None):
        config = config or {}
        # A new-information ratio below this threshold is considered "low increment"
        self._low_increment_threshold = config.get("low_increment_threshold", 0.15)
        # Saturation is triggered once consecutive low-increment rounds reach this value
        self._saturation_rounds = config.get("saturation_rounds", 2)

        # State
        self._round_fingerprints: List[Set[str]] = []
        self._cumulative_keywords: Set[str] = set()
        self._consecutive_low_increments: int = 0
        self._focus_labels_seen: List[str] = []
        self._increment_history: List[float] = []
        self._has_used_videograph: bool = False  # Whether VIDEO drilldown has already been used
        self._has_used_keyframe: bool = False  # Third tier: whether KEYFRAME has been used

    def reset(self):
        """Reset state (called at the start of each new question)."""
        self._round_fingerprints = []
        self._cumulative_keywords = set()
        self._consecutive_low_increments = 0
        self._focus_labels_seen = []
        self._increment_history = []
        self._has_used_videograph = False
        self._has_used_keyframe = False

    def assess(self, retrieval_payload: Dict[str, Any],
               query: str) -> Dict[str, Any]:
        """Assess the information increment of the current retrieval round.

        Args:
            retrieval_payload: the raw payload returned by the current retrieval round
            query: the retrieval query for the current round

        Returns:
            An assessment result dictionary:
            {
                "increment_ratio": float,       # new-information ratio (0 to 1)
                "is_low_increment": bool,        # whether this is a low increment
                "is_saturated": bool,            # whether saturated (consecutive low increments)
                "consecutive_low": int,          # number of consecutive low-increment rounds
                "cumulative_keywords": int,      # total number of cumulative keywords
                "new_keywords_count": int,       # number of new keywords in this round
                "focus_label_repeated": bool,    # whether focus_label is repeated
                "has_videograph_hits": bool,     # whether it contains videograph fine-grained information
            }
        """
        # Extract this round's information fingerprint (excluding query keywords to avoid query contamination)
        current_fingerprint = self._extract_fingerprint(retrieval_payload, query)

        # Detect whether this is VIDEO drilldown mode (videograph_hits is non-empty)
        vg_hits = retrieval_payload.get("videograph_hits", {})
        has_videograph_hits = bool(
            isinstance(vg_hits, dict)
            and any(isinstance(v, list) and len(v) > 0 for v in vg_hits.values())
        )
        # Third tier: detect whether this is KEYFRAME inspect mode (keyframe_count > 0 or mode="keyframe_inspect")
        has_keyframes = bool(retrieval_payload.get("keyframe_count", 0)) or (
            retrieval_payload.get("mode") == "keyframe_inspect"
        )

        # Compute new information
        new_keywords = current_fingerprint - self._cumulative_keywords
        total_current = len(current_fingerprint)
        new_count = len(new_keywords)

        # Compute the increment ratio: new keywords / total keywords this round
        if total_current > 0:
            increment_ratio = new_count / total_current
        else:
            increment_ratio = 0.0

        # Check whether focus_label is repeated
        focus_label = retrieval_payload.get("focus_label", "")
        focus_repeated = focus_label in self._focus_labels_seen if focus_label else False

        # VIDEO drilldown first-time immunity: immunity applies only the first time the system
        # switches from EventGraph to VIDEO drilldown, because on first entry the fine-grained
        # information (character actions, dialogues, etc.) is a semantically brand-new layer of
        # information, yet its keywords may lexically overlap with the event summary and cause the
        # increment to be underestimated. If the Agent keeps searching within VIDEO drilldown, the
        # increment is judged normally, to avoid the Agent repeatedly searching in the same mode
        # without information saturation being detected.
        # Third tier: KEYFRAME first-time immunity (symmetric to VIDEO first-time immunity):
        # when the visual modality is first introduced, the information layer is completely renewed
        # and should not be treated as "low increment".
        is_first_videograph = has_videograph_hits and not self._has_used_videograph
        is_first_keyframe = has_keyframes and not self._has_used_keyframe
        if is_first_videograph or is_first_keyframe:
            is_low = False
            logger.debug(
                f"[Sufficiency] First-time modality immunity: vg_first={is_first_videograph}, "
                f"kf_first={is_first_keyframe}, ratio={increment_ratio:.2f}"
            )
        else:
            # Determine whether this is a low increment
            # Condition: new ratio below threshold, or focus_label repeated and increment not high
            is_low = (increment_ratio < self._low_increment_threshold) or (
                focus_repeated and increment_ratio < 0.3
            )

        # Update VIDEO drilldown / KEYFRAME usage state
        if has_videograph_hits:
            self._has_used_videograph = True
        if has_keyframes:
            self._has_used_keyframe = True

        # Update the consecutive low-increment count
        if is_low:
            self._consecutive_low_increments += 1
        else:
            self._consecutive_low_increments = 0

        # Determine whether saturated
        is_saturated = self._consecutive_low_increments >= self._saturation_rounds

        # Update state
        self._cumulative_keywords.update(current_fingerprint)
        self._round_fingerprints.append(current_fingerprint)
        self._increment_history.append(increment_ratio)
        if focus_label:
            self._focus_labels_seen.append(focus_label)

        result = {
            "increment_ratio": round(increment_ratio, 3),
            "is_low_increment": is_low,
            "is_saturated": is_saturated,
            "consecutive_low": self._consecutive_low_increments,
            "cumulative_keywords": len(self._cumulative_keywords),
            "new_keywords_count": new_count,
            "focus_label_repeated": focus_repeated,
            "has_videograph_hits": has_videograph_hits,
            "has_keyframes": has_keyframes,
        }

        if is_low:
            logger.debug(
                f"[Sufficiency] Low increment: ratio={increment_ratio:.2f}, "
                f"new={new_count}, consecutive_low={self._consecutive_low_increments}"
            )
        if is_saturated:
            logger.info(
                f"[Sufficiency] Information saturation: {self._consecutive_low_increments} consecutive low-increment rounds, "
                f"{len(self._cumulative_keywords)} cumulative keywords"
            )

        return result

    def _extract_fingerprint(self, payload: Dict[str, Any],
                             query: str) -> Set[str]:
        """Extract an information fingerprint (a keyword set) from the retrieval payload.

        Extraction sources (only the retrieval system's output, excluding the query itself):
        1. summary/description of the focus node
        2. label and summary of neighbors
        3. relation of edges
        4. key information in videograph_hits

        Note: query keywords are no longer included. The query is the Agent's input rather than
        the retrieval system's output. Counting it into the fingerprint would cause keyword overlap
        among similar queries to artificially lower the increment ratio (query contamination).
        """
        keywords = set()

        # Extract from the focus node
        focus = payload.get("focus", {})
        if isinstance(focus, dict):
            for key in ("summary", "description", "label", "segment_label"):
                val = focus.get(key, "")
                if val:
                    keywords.update(self._tokenize(str(val)))

        # Extract from focus_label
        focus_label = payload.get("focus_label", "")
        if focus_label:
            keywords.update(self._tokenize(focus_label))

        # Extract from neighbors
        neighbors = payload.get("neighbors", [])
        if isinstance(neighbors, list):
            for n in neighbors:
                if isinstance(n, dict):
                    for key in ("label", "summary"):
                        val = n.get(key, "")
                        if val:
                            keywords.update(self._tokenize(str(val)))
                elif isinstance(n, str):
                    keywords.update(self._tokenize(n))

        # Extract from edges
        edges = payload.get("edges", [])
        if isinstance(edges, list):
            for e in edges:
                if isinstance(e, dict):
                    for key in ("relation", "label", "description"):
                        val = e.get(key, "")
                        if val:
                            keywords.update(self._tokenize(str(val)))

        # Extract from videograph_hits
        vg_hits = payload.get("videograph_hits", {})
        if isinstance(vg_hits, dict):
            for category, items in vg_hits.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            for key in ("content", "text", "description", "label"):
                                val = item.get(key, "")
                                if val:
                                    keywords.update(self._tokenize(str(val)))
                        elif isinstance(item, str):
                            keywords.update(self._tokenize(item))

        # Third tier: inject fingerprints from the keyframe dimension (each newly seen clip brings new information)
        kf_clip_scope = payload.get("keyframe_clip_scope", [])
        if isinstance(kf_clip_scope, list):
            for cid in kf_clip_scope:
                keywords.add(f"__kf_clip_{cid}__")

        return keywords

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple tokenization: extract meaningful words (length >= 2, with stopwords removed)."""
        # Stopwords (high-frequency, low-meaning words)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or", "nor",
            "not", "no", "so", "if", "than", "too", "very", "just", "about",
            "this", "that", "these", "those", "it", "its", "he", "she", "they",
            "them", "his", "her", "their", "my", "your", "our", "what", "which",
            "who", "whom", "how", "when", "where", "why", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "only", "own",
            "same", "also", "here", "there",
        }
        # Extract alphanumeric words
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return {w for w in words if len(w) >= 2 and w not in stopwords}


class SufficiencySignal:
    """Unified entry point for the information sufficiency signal.

    Generates a structured sufficiency signal based on information increment assessment
    (consecutive low-increment detection) and works in concert with the Guardrail system.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self._increment_assessor = InformationIncrementAssessor(
            config.get("increment", {})
        )
        self._enabled = config.get("enabled", True)

    def reset(self):
        """Reset state (called at the start of each new question)."""
        self._increment_assessor.reset()
        self._soft_warning_given = False  # Whether a soft saturation warning has already been issued

    def assess_after_search(self, retrieval_payload: Dict[str, Any],
                            query: str,
                            options: List[str],
                            round_idx: int,
                            max_rounds: int,
                            question_text: str = "",
                            keyframe_available: bool = True) -> Dict[str, Any]:
        """Assess information sufficiency after each retrieval round.

        Args:
            retrieval_payload: the raw payload returned by the current retrieval round
            query: the retrieval query for the current round
            options: the option list (kept for caller compatibility, but no longer used for coverage detection)
            round_idx: the current round index
            max_rounds: the maximum number of rounds
            question_text: the original question text (added in the third tier, optional). When provided,
                it strengthens the recommendation of the KEYFRAME strategy for visual questions; when empty,
                behavior is the same as before, for backward compatibility.
            keyframe_available: whether the current video has usable keyframes. When unavailable, KEYFRAME is not recommended.

        Returns:
            {
                "increment": {...},             # information increment assessment result
                "should_stop_searching": bool,   # whether stopping the search is recommended
                "hint_message": str,             # hint message for the Agent
            }
        """
        if not self._enabled:
            return {
                "increment": {},
                "should_stop_searching": False,
                "hint_message": "",
            }

        # Information increment assessment
        increment = self._increment_assessor.assess(retrieval_payload, query)

        # Determine whether to stop searching based solely on consecutive low increments
        should_stop = self._should_stop(increment, round_idx, max_rounds)

        # Generate the hint message
        hint = self._build_hint(increment, should_stop, round_idx, max_rounds,
                                question_text=question_text,
                                keyframe_available=keyframe_available)

        return {
            "increment": increment,
            "should_stop_searching": should_stop,
            "hint_message": hint,
        }

    def _should_stop(self, increment: Dict,
                     round_idx: int, max_rounds: int) -> bool:
        """Determine whether to stop searching based on consecutive low increments.

        Grace Round mechanism:
        - When the saturation condition is first triggered, only a soft hint is issued (should_stop=False),
          giving the Agent one round to adjust its retrieval strategy
        - Only if the Agent still triggers the saturation condition after the grace round is stopping actually recommended

        Stop condition:
        - Information saturation (N consecutive low-increment rounds, where N is configured by saturation_rounds)
        """
        # The only condition: information saturation caused by consecutive low increments
        if not increment.get("is_saturated", False):
            return False

        # Grace round mechanism: on the first trigger, only issue a soft hint, do not hard-stop
        if not self._soft_warning_given:
            self._soft_warning_given = True
            logger.info(
                f"[Sufficiency] First saturation trigger -> issuing a soft hint, granting the Agent one grace round | "
                f"increment={increment.get('increment_ratio', 0):.2f}, "
                f"consecutive_low={increment.get('consecutive_low', 0)}"
            )
            return False  # Stopping is not recommended, but the hint will tell the Agent that information is nearing saturation

        # The grace round has been used up; actually recommend stopping
        logger.info(
            f"[Sufficiency] Saturated again after the grace round -> recommending stopping the search | "
            f"increment={increment.get('increment_ratio', 0):.2f}, "
            f"consecutive_low={increment.get('consecutive_low', 0)}"
        )
        return True

    def _build_hint(self, increment: Dict,
                    should_stop: bool, round_idx: int,
                    max_rounds: int,
                    question_text: str = "",
                    keyframe_available: bool = True) -> str:
        """Build the sufficiency hint message for the Agent.

        Distinguishes three states:
        1. should_stop=True: still saturated after the grace round, strongly recommend answering
        2. _soft_warning_given=True and should_stop=False: first saturation,
           give the Agent a chance to adjust its strategy
        3. Normal state: a lightweight hint
        """
        parts = []
        remaining = max_rounds - round_idx - 1

        if should_stop:
            # Still saturated after the grace round: strongly recommend answering
            parts.append(
                f"\U0001f4a1 SUFFICIENCY SIGNAL: Search results are no longer providing new information "
                f"(consecutive low-increment rounds: {increment.get('consecutive_low', 0)}). "
                f"You are strongly encouraged to output [Answer] now."
            )
        elif self._soft_warning_given and not should_stop:
            # First saturation trigger (grace round): give targeted suggestions based on the Agent's prior retrieval mode
            strategy_suggestions = self._build_strategy_suggestions(
                increment, round_idx, question_text=question_text,
                keyframe_available=keyframe_available
            )
            parts.append(
                f"\u26a0\ufe0f INFORMATION SATURATION WARNING: Recent searches are returning diminishing new information "
                f"(increment={increment.get('increment_ratio', 0):.0%}). "
                f"You have ONE more chance to find useful information. Consider:\n"
                f"  1. If you already have enough information, output [Answer] now.\n"
                f"  2. If you must search, try a COMPLETELY DIFFERENT strategy:\n"
                f"{strategy_suggestions}\n"
                f"  If the next search still yields little new information, you will be forced to answer."
            )
        else:
            # Normal state: lightweight hint
            if increment.get("is_low_increment"):
                parts.append(
                    f"\u2139\ufe0f The last search added limited new information "
                    f"(increment={increment.get('increment_ratio', 0):.0%}). "
                    f"Consider whether you have enough to answer."
                )

            if remaining <= 2:
                parts.append(
                    f"\u26a0\ufe0f Only {remaining} search round(s) remaining. "
                    f"Consider answering if you have enough information."
                )

        return "\n".join(parts)

    def _build_strategy_suggestions(self, increment: Dict,
                                     round_idx: int,
                                     question_text: str = "",
                                     keyframe_available: bool = True) -> str:
        """Based on the agent's prior retrieval patterns, generate targeted strategy-switch suggestions.

        Analyze the existing retrieval history (via focus_labels_seen and increment_history)
        to determine which retrieval mode the agent has mainly used, then suggest switching to others.

        Third layer: if question_text matches visual keywords and KEYFRAME has not been used,
        promote the KEYFRAME suggestion to the top with emphatic wording.
        """
        seen_labels = self._increment_assessor._focus_labels_seen
        queries_count = len(self._increment_assessor._round_fingerprints)

        # Analyze prior retrieval patterns
        has_used_video = False
        has_used_neighbor = False
        has_used_event = False
        has_used_keyframe = self._increment_assessor._has_used_keyframe
        for fp in self._increment_assessor._round_fingerprints:
            # Infer the retrieval mode from keywords in the fingerprint
            if "video" in fp or any("clip" in kw for kw in fp):
                has_used_video = True
            if len(seen_labels) > 1:
                has_used_neighbor = True
            has_used_event = True  # by default EventGraph has been used

        # Third layer: first determine whether this is a visual question (defensive import; fall back to False on failure)
        is_visual = False
        if keyframe_available and question_text and not has_used_keyframe and seen_labels:
            try:
                from ..memory.hierarchical import _is_visual_query
                is_visual = _is_visual_query(question_text)
            except Exception:
                is_visual = False

        suggestions = []
        priority = 1

        # Third layer: visual question and KEYFRAME never used -> promote to top + emphasize
        if is_visual:
            suggestions.append(
                f"     {priority}. ⭐ **STRONGLY RECOMMENDED** Use 'KEYFRAME: <query>' NOW "
                f"— this question asks about visual details (layout / appearance / spatial "
                f"position / on-going activity) that textual memories often miss. Up to 5 "
                f"keyframe images of the focus event will be attached."
            )
            priority += 1

        # If VIDEO: drilldown has never been used, strongly recommend it
        if not has_used_video:
            suggestions.append(
                f"     {priority}. \U0001f50d Use 'VIDEO: <query>' to drill down into fine-grained details "
                f"(character actions, dialogues, visual details) within the current event"
            )
            priority += 1

        # If NEIGHBOR: has never been used, suggest exploring adjacent events
        if not has_used_neighbor and seen_labels:
            suggestions.append(
                f"     {priority}. \U0001f517 Use 'NEIGHBOR: <segment_label>' to explore an adjacent event "
                f"(you've been focused on '{seen_labels[-1]}', try a neighboring segment)"
            )
            priority += 1

        # If focus_label keeps repeating, suggest entirely different keywords
        if seen_labels and len(set(seen_labels)) <= 1:
            suggestions.append(
                f"     {priority}. \U0001f504 Use entirely different keywords to search for a different event "
                f"(you've been searching within the same event '{seen_labels[0]}')"
            )
            priority += 1

        # If multiple modes have been tried but it is still saturated, suggest answering directly
        if has_used_video and has_used_neighbor:
            suggestions.append(
                f"     {priority}. \u2705 You've already tried multiple search strategies. "
                f"Consider answering with the information you have."
            )
            priority += 1

        # Third-layer general KEYFRAME suggestion (a regular suggestion for non-visual questions that still have not used it; weaker position and wording)
        # If the visual question already appeared at the top, do not inject again here
        if keyframe_available and (not is_visual) and (not has_used_keyframe) and seen_labels:
            suggestions.append(
                f"     {priority}. 🖼 Use 'KEYFRAME: <query>' to inspect the visual "
                f"keyframes of the current focus event — helpful for colors, on-screen text, "
                f"spatial layout, or appearance questions that text memories do not capture."
            )
            priority += 1

        # Fallback: if there are no specific suggestions, give a generic one
        if not suggestions:
            suggestions = [
                f"     1. \U0001f50d Use 'VIDEO: <query>' to get fine-grained details within the current event",
                f"     2. \U0001f517 Use 'NEIGHBOR: <segment_label>' to explore an adjacent event",
                f"     3. \U0001f504 Use entirely different keywords to search for a different event",
            ]

        return "\n".join(suggestions)
