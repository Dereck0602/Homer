"""
ContextEngineering: context engineering optimization.

A core enhancement of Harness Engineering:
optimize the quality of context received by the Agent, reduce attention dilution, and increase information density.

Optimizations at two levels:
1. Retrieval Formatting: convert raw JSON into structured natural language
2. Conversation Management: integrate entropy management to prevent context window overflow
"""
import re
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RetrievalResultFormatter:
    """Retrieval result formatter: converts raw JSON into Agent-friendly structured text.

    Core idea:
    - Raw JSON contains many nested structures and redundant fields, which require the Agent to spend a lot of attention parsing
    - After converting to structured natural language, the Agent can locate key information faster
    - At the same time, it filters out metadata fields that are useless for answering the question
    """

    # Metadata fields to filter out (useless for answering the question)
    _META_FIELDS = {
        "clip_ids", "embedding", "vector", "node_id", "node_type",
        "created_at", "updated_at", "source_file", "processing_config",
    }

    def format_search_result(self, retrieval_payload: Dict[str, Any],
                             query: str,
                             sufficiency_hint: str = "") -> str:
        """Format the retrieval result into Agent-friendly text.

        Args:
            retrieval_payload: raw retrieval payload
            query: search query
            sufficiency_hint: information sufficiency hint (from SufficiencySignal)

        Returns:
            the formatted text
        """
        parts = ["Searched knowledge:"]

        # Format the focus node
        focus = retrieval_payload.get("focus", {})
        focus_label = retrieval_payload.get("focus_label", "")

        if focus and isinstance(focus, dict):
            parts.append(self._format_focus_node(focus, focus_label))
        else:
            parts.append("(No matching event found for this query.)")

        # Format neighbors
        neighbors = retrieval_payload.get("neighbors", [])
        edges = retrieval_payload.get("edges", [])
        if neighbors:
            parts.append(self._format_neighbors(neighbors, edges))

        # Format VideoGraph fine-grained information
        vg_hits = retrieval_payload.get("videograph_hits", {})
        if vg_hits:
            parts.append(self._format_videograph_hits(vg_hits))

        # Third layer: keyframe content preview (plain text; images are spliced into the message in multimodal form at the multi_round layer)
        if retrieval_payload.get("mode") == "keyframe_inspect" or retrieval_payload.get("keyframe_count"):
            parts.append(self._format_keyframe_hits(retrieval_payload))

        # Append the sufficiency hint
        if sufficiency_hint:
            parts.append(f"\n{sufficiency_hint}")

        return "\n".join(parts)

    def format_empty_result(self, query: str,
                            strategy_hint: str = "") -> str:
        """Format an empty retrieval result."""
        text = (
            "Searched knowledge: (No results found.)\n"
            "The search did not return any matching events or memories."
        )
        if strategy_hint:
            text += f"\n{strategy_hint}"
        return text

    def _format_focus_node(self, focus: Dict, focus_label: str = "") -> str:
        """Format the focus node."""
        lines = []

        if focus_label:
            lines.append(f"\n📍 Focus Event: {focus_label}")
        else:
            label = focus.get("label", focus.get("segment_label", "Unknown"))
            lines.append(f"\n📍 Focus Event: {label}")

        # Extract key information
        summary = focus.get("summary", focus.get("description", ""))
        if summary:
            # Truncate an overly long summary
            if len(summary) > 800:
                summary = summary[:800] + "..."
            lines.append(f"  Summary: {summary}")

        # Time range
        time_range = focus.get("time_range", focus.get("temporal_range", ""))
        if time_range:
            lines.append(f"  Time: {time_range}")

        # Other useful fields (filter out metadata)
        for key, val in focus.items():
            if key in self._META_FIELDS:
                continue
            if key in ("summary", "description", "label", "segment_label",
                       "time_range", "temporal_range", "focus_label"):
                continue
            if val and isinstance(val, str) and len(val) > 2:
                # Truncate an overly long value
                display_val = val[:300] + "..." if len(val) > 300 else val
                lines.append(f"  {key}: {display_val}")

        return "\n".join(lines)

    def _format_neighbors(self, neighbors: List, edges: List = None) -> str:
        """Format neighbor nodes."""
        lines = ["\n🔗 Adjacent Events:"]

        # Build the edge map
        edge_map = {}
        if edges:
            for e in edges:
                if isinstance(e, dict):
                    target = e.get("target", e.get("to", ""))
                    relation = e.get("relation", e.get("label", ""))
                    if target and relation:
                        edge_map[target] = relation

        for i, n in enumerate(neighbors[:5]):  # Show at most 5 neighbors
            if isinstance(n, dict):
                label = n.get("label", n.get("segment_label", f"neighbor_{i}"))
                summary = n.get("summary", "")
                relation = edge_map.get(label, "")

                line = f"  - {label}"
                if relation:
                    line += f" ({relation})"
                if summary:
                    short_summary = summary[:150] + "..." if len(summary) > 150 else summary
                    line += f": {short_summary}"
                lines.append(line)
            elif isinstance(n, str):
                relation = edge_map.get(n, "")
                line = f"  - {n}"
                if relation:
                    line += f" ({relation})"
                lines.append(line)

        if len(neighbors) > 5:
            lines.append(f"  ... and {len(neighbors) - 5} more neighbors")

        return "\n".join(lines)

    def _format_videograph_hits(self, vg_hits: Dict) -> str:
        """Format VideoGraph fine-grained retrieval results."""
        lines = ["\n\U0001f50d Fine-grained Details:"]

        for category, items in vg_hits.items():
            if not items:
                continue

            category_label = category.replace("_", " ").title()
            lines.append(f"  [{category_label}]")

            if isinstance(items, list):
                for item in items[:10]:  # At most 10 entries per category
                    if isinstance(item, dict):
                        content = item.get("content", item.get("text",
                                    item.get("description", str(item))))
                        if isinstance(content, str):
                            short = content[:200] + "..." if len(content) > 200 else content
                            lines.append(f"    - {short}")
                    elif isinstance(item, str):
                        short = item[:200] + "..." if len(item) > 200 else item
                        lines.append(f"    - {short}")

                if len(items) > 10:
                    lines.append(f"    ... and {len(items) - 10} more items")
            elif isinstance(items, str):
                lines.append(f"    {items[:300]}")

        return "\n".join(lines)

    def _format_keyframe_hits(self, retrieval_payload: Dict) -> str:
        """Third layer: plain-text preview of keyframe retrieval (no base64 attached)."""
        kf_count = int(retrieval_payload.get("keyframe_count", 0) or 0)
        scope = retrieval_payload.get("keyframe_clip_scope", []) or []
        lines = ["\n\U0001f5bc Keyframes:"]
        if kf_count > 0:
            lines.append(
                f"  {kf_count} keyframe image(s) from clips {scope} are attached below."
            )
            lines.append(
                "  Examine them for colors, on-screen text, layout, appearance details "
                "that the textual memories may miss."
            )
        else:
            reason = retrieval_payload.get("keyframe_reason", "")
            lines.append("  No keyframe images could be loaded for this event.")
            if reason:
                lines.append(f"  (Reason: {reason})")
            lines.append(
                "  Consider using 'VIDEO:' for textual details or 'NEIGHBOR:' to shift focus."
            )
        if retrieval_payload.get("keyframe_downgrade"):
            lines.append(
                "  Note: no focus event was set before KEYFRAME, so the system "
                "auto-matched a best-match event. Next time, locate a focus via a "
                "plain query or VIDEO: first for more targeted keyframes."
            )
        return "\n".join(lines)

class ConversationManager:
    """Conversation history manager: integrates entropy management to prevent context window overflow.

    Automatically manages conversation history within the reasoning loop:
    - When the message count exceeds a threshold, compress early messages
    - Retain the system prompt and the most recent interactions
    - Replace compressed messages with concise summaries

    Ledger mode (ledger_mode=True):
    - Keep only the full retrieval text of the most recent 3 rounds (6 messages)
    - Earlier rounds are entirely replaced by the Ledger structured state (a one-line summary)
    - No full retrieval text is discarded; it is stored in _retrieval_archive
    - During the synthesis stage, the full deduplicated retrieval text can be obtained via get_deduplicated_retrieval_text()
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self._max_messages = config.get("max_conversation_messages", 14)
        self._compress_search_results = config.get("compress_search_results", True)
        # Ledger mode: keep only the full retrieval text of the most recent 3 rounds
        self._ledger_mode = config.get("ledger_mode", False)
        # Store the full retrieval text of all rounds (not discarded, for use by synthesis)
        self._retrieval_archive: List[Dict[str, str]] = []
        # Set of archived focus_labels (used for deduplication)
        self._archived_focus_labels: set = set()

    def archive_retrieval(self, round_idx: int, query: str,
                          retrieval_text: str, focus_label: str = "") -> None:
        """Archive the full text of one retrieval round (for use by the synthesis stage).

        Args:
            round_idx: round index
            query: search query
            retrieval_text: the full formatted retrieval result text
            focus_label: the focus event label hit in this round
        """
        self._retrieval_archive.append({
            "round_idx": round_idx,
            "query": query,
            "retrieval_text": retrieval_text,
            "focus_label": focus_label,
        })
        if focus_label:
            self._archived_focus_labels.add(focus_label)

    def get_deduplicated_retrieval_text(self) -> str:
        """Get all deduplicated full retrieval text, for use by the synthesis stage.

        Deduplication strategy:
        - For the same focus_label, keep only the full text from its first appearance (usually the richest in information)
        - Entries without a focus_label (empty results, etc.) are all retained
        - Results from VIDEO/NEIGHBOR drilldown are retained even if the focus is the same (because they contain different details)
        """
        seen_focus_full: set = set()  # Focuses whose full summary has already been output
        parts: List[str] = []

        for entry in self._retrieval_archive:
            text = entry["retrieval_text"]
            focus = entry["focus_label"]
            query = entry["query"]
            round_idx = entry["round_idx"]

            # Empty result or no focus: always retain
            if not focus or "No results found" in text or "No matching" in text:
                parts.append(f"[Round {round_idx}] Query: {query}\n{text}")
                continue

            # VIDEO/NEIGHBOR/KEYFRAME drilldown: retain even if the focus is the same (details differ)
            is_drilldown = any(
                query.strip().upper().startswith(prefix)
                for prefix in ("VIDEO:", "NEIGHBOR:", "KEYFRAME:")
            )
            if is_drilldown:
                parts.append(f"[Round {round_idx}] Query: {query}\n{text}")
                seen_focus_full.add(focus)
                continue

            # Ordinary query: for the same focus, keep only the first full text
            if focus not in seen_focus_full:
                parts.append(f"[Round {round_idx}] Query: {query}\n{text}")
                seen_focus_full.add(focus)
            else:
                # Subsequent rounds with the same focus: keep only the newly added videograph information
                # Check whether there is a "New fine-grained details" section
                vg_marker = "🔍 New fine-grained details"
                if vg_marker in text:
                    vg_start = text.index(vg_marker)
                    parts.append(
                        f"[Round {round_idx}] Query: {query} "
                        f"(focus unchanged: {focus})\n{text[vg_start:]}"
                    )

        return "\n\n---\n\n".join(parts) if parts else "(No retrieval results)"

    def manage(self, conversations: List[Dict]) -> List[Dict]:
        """Manage conversation history, compressing it when necessary.

        Compression is currently disabled: return the original conversation directly, without any truncation or summarization.
        Keep the full multi-round conversation history and let the model leverage the entire context on its own.

        Ledger mode strategy (disabled):
        - Always keep only the system prompt + the full content of the most recent 3 rounds (6 messages)
        - Compress earlier rounds into a one-line summary
        - Trigger compression starting from round 5 (do not wait until 14 messages)

        Baseline mode strategy (disabled):
        1. If the message count <= threshold, do nothing
        2. When it exceeds the threshold:
           a. Keep the system prompt (the 1st message)
           b. Compress the intermediate retrieval result messages
           c. Keep the most recent 4 messages (2 interaction rounds)
        """
        # Compression disabled: return the original conversation directly
        return conversations
        # if self._ledger_mode:
        #     return self._manage_ledger_mode(conversations)
        # return self._manage_baseline_mode(conversations)

    def _manage_ledger_mode(self, conversations: List[Dict]) -> List[Dict]:
        """Ledger mode: keep only the full retrieval text of the most recent 3 rounds.

        Conversation structure:
          [system] + [user(initial)] + [assistant(r0)] + [user(r0_result)] + [assistant(r1)] + ...

        After compression:
          [system] + [compressed_summary] + [the user + assistant of the most recent 3 rounds]
        """
        # Note: the system prompt never participates in compression and is always kept as-is at the front of the output.
        # Compression is worthwhile only with at least system + 8 non-system messages (4 interaction rounds)
        # Keep the most recent 6 non-system messages (3 rounds: each round has user_result + assistant_response)
        if len(conversations) <= 9:  # system(1) + 8 messages = 4 interaction rounds to trigger
            return conversations

        # Separate the system prompt (the system prompt is not counted in rounds and does not participate in compression)
        system_msg = None
        rest = conversations
        if conversations and conversations[0].get("role") == "system":
            system_msg = conversations[0]
            rest = conversations[1:]

        # Keep the most recent 6 messages (3 full interaction rounds)
        recent_count = 6
        if len(rest) <= recent_count:
            return conversations

        early = rest[:-recent_count]
        recent = rest[-recent_count:]

        # Compress early messages into a concise summary
        compressed = self._compress_early_messages_ledger(early)

        # Reassemble
        result = []
        if system_msg:
            result.append(system_msg)
        result.append({
            "role": "user",
            "content": compressed,
        })
        result.extend(recent)

        logger.debug(
            f"[ContextEng/Ledger] conversation compression: {len(conversations)} → {len(result)} messages"
        )
        return result

    def _manage_baseline_mode(self, conversations: List[Dict]) -> List[Dict]:
        """Baseline mode (backward compatible with the original logic)."""
        if len(conversations) <= self._max_messages:
            return conversations

        # Separate the system prompt
        system_msg = None
        rest = conversations
        if conversations and conversations[0].get("role") == "system":
            system_msg = conversations[0]
            rest = conversations[1:]

        # Keep the most recent 4 messages (2 full interaction rounds)
        recent_count = 4
        if len(rest) <= recent_count:
            return conversations

        early = rest[:-recent_count]
        recent = rest[-recent_count:]

        # Compress early messages
        compressed = self._compress_early_messages(early)

        # Reassemble
        result = []
        if system_msg:
            result.append(system_msg)
        result.append({
            "role": "user",
            "content": compressed,
        })
        result.extend(recent)

        logger.debug(
            f"[ContextEng] conversation compression: {len(conversations)} → {len(result)} messages"
        )
        return result

    def _compress_early_messages_ledger(self, messages: List[Dict]) -> str:
        """Compress early messages in Ledger mode: keep only each round's query + focus + action summary.

        Do not keep the full retrieval text (it is already archived in _retrieval_archive
        and obtained during the synthesis stage via get_deduplicated_retrieval_text()).
        """
        lines = ["[Earlier rounds summary: full retrieval available at synthesis]"]

        round_idx = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user" and ("Searched knowledge" in content or "no retrieval yet" in content):
                # Extract the focus label
                focus_match = re.search(r'Focus Event:\s*(.+?)(?:\n|$)', content)
                focus = focus_match.group(1).strip() if focus_match else ""

                # Extract whether there is a result
                has_result = (
                    "No results found" not in content
                    and "(No matching" not in content
                    and "no retrieval yet" not in content
                )

                # Extract the focus unchanged marker
                is_unchanged = "focus unchanged" in content

                round_idx += 1
                if not has_result or "no retrieval yet" in content:
                    status = "no results"
                elif is_unchanged:
                    status = f"focus unchanged: '{focus}'"
                else:
                    status = f"new focus: '{focus}'" if focus else "found event"
                lines.append(f"  R{round_idx}: {status}")

            elif role == "assistant":
                # Extract the Agent's action and query
                action_match = re.search(r'Action:\s*\[(Answer|Search)\]', content)
                if action_match:
                    action = action_match.group(1)
                    if action == "Search":
                        content_match = re.search(r'Content:\s*(.*?)(?:\n|$)', content)
                        query = content_match.group(1).strip()[:60] if content_match else "?"
                        if lines and lines[-1].startswith("  R"):
                            lines[-1] += f" → [{query}]"
                        else:
                            lines.append(f"  → Search: [{query}]")

            i += 1

        return "\n".join(lines)

    def _compress_early_messages(self, messages: List[Dict]) -> str:
        """Baseline mode: compress early messages into a concise summary."""
        lines = ["[Summary of earlier search rounds]"]

        round_idx = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user" and "Searched knowledge" in content:
                # This is a retrieval result message
                # Extract whether there is a result
                has_result = "No results found" not in content and "(No matching" not in content
                # Extract the focus label (if any)
                focus_match = re.search(r'"focus_label":\s*"([^"]+)"', content)
                focus = focus_match.group(1) if focus_match else "none"

                round_idx += 1
                status = f"found event '{focus}'" if has_result and focus != "none" else "no results"
                lines.append(f"  Round {round_idx}: {status}")

            elif role == "assistant":
                # Extract the Agent's action
                action_match = re.search(r'Action:\s*\[(Answer|Search)\]', content)
                if action_match:
                    action = action_match.group(1)
                    if action == "Search":
                        content_match = re.search(r'Content:\s*(.*?)(?:\n|$)', content)
                        query = content_match.group(1).strip()[:80] if content_match else "?"
                        lines[-1] += f" → searched: '{query}'"

            i += 1

        return "\n".join(lines)
