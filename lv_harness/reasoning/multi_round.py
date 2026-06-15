"""
MultiRoundSearchAgent: a multi-round retrieval reasoning Agent.

Directly wires up the core logic in control_api_high_system.py:
- system_prompt defines the Agent behavior
- supports three retrieval modes: EventGraph retrieval / VideoGraph drilldown / Neighbor walk
- forced answer on the last round (last_round_prompt)

Optional third layer: the visual layer KEYFRAME (zero impact on the main flow when enabled=false).
"""
import os
import re
import json
import copy
import base64
import time
import logging
import multiprocessing.pool
from functools import lru_cache
from typing import Dict, List, Optional, Any

from .base import ReasoningAgent
from .guardrails import AgentGuardrails
from .sufficiency import SufficiencySignal
from .context_engineering import RetrievalResultFormatter, ConversationManager
from ..data.types import TemporalQuestion, AgentAnswer, RetrievalResult
from ..memory.base import MemoryStrategy
from ..memory.hierarchical import _is_visual_query as _mem_is_visual_query  # third layer: visual question detection

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _load_image_b64(img_path: str) -> str:
    """LRU-cached base64 encoding of an image, avoiding repeated I/O for the same image across rounds.

    Loaded only in the reasoning/multi_round main thread, not via multiprocessing IPC,
    consistent with optimization #5 in control_api_triple_v2.py.
    """
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Streamlined system_prompt (context-engineering optimization)
# Core principle: keep only the format requirements and retrieval-mode descriptions, dropping the lengthy decision guide
# The decision guide is injected dynamically via the sufficiency signal and feedback
SYSTEM_PROMPT = """You are answering a multiple-choice question about a video. You can search a memory bank for relevant information before answering.

Question: {question}
Options: {options}

Output format:
Reason: <your reasoning>
Action: [Answer] or [Search]
Content: <answer option OR search query>

Search modes (use as Content):
1. <query>: search EventGraph for relevant events (returns focus event + neighbors)
2. VIDEO: <query>: drill down into fine-grained details within the current focus event
3. NEIGHBOR: <segment_label>: shift focus to an adjacent event (label must be from previous results)

Query quality (applies to every search round, not just the first):
- Every query MUST carry enough semantic signal to be embedded-matched against event/VideoGraph nodes. A single token (a bare name like "Bob", a bare noun like "dinner", a bare color) is NOT acceptable. The retrieval index stores full event summaries, so bare tokens almost always return weak/irrelevant hits.
- Build each query as: <subject> + <predicate or attribute> + <optional context>. Examples:
    BAD:  "Bob" | "dinner" | "red pot" | "Cary cooking ability"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
          "Cary preparing ingredients or serving dishes in the kitchen"
- If the question names a person (e.g., "Bob"), the memory bank usually indexes that person under a `<character_N>` placeholder, NOT under the literal name. So a plain-name search tends to match weakly. Two acceptable first moves:
    (a) issue a binding query: `What is the name of <character_0>?` (use any candidate placeholder you have seen), then reuse the ID in later queries like `VIDEO: <character_2> cooking dinner`;
    (b) go straight to a content-rich query combining the name with what the question really asks: `Bob's behavior when playing basketball indoors` rather than just `Bob`.
- If the previous round returned a weakly-matched event (e.g., the focus summary barely touches the question), do NOT just retry with a slight variation. REWRITE the query with more predicates or switch to VIDEO/NEIGHBOR.

Identity resolution:
- The EventGraph uses anonymized placeholders like `<character_0>`, `<face_1>`. If exactly one named person (from the question) appears in the retrieved event and no contradicting evidence exists, you MAY treat that placeholder as that person, state this assumption briefly in Reason.

Key rules:
- Analyze retrieved information carefully before deciding to search again.
- You MUST perform at least one [Search] before issuing any [Answer]. The first turn's user message is just a placeholder, NOT a retrieval result.
- When you have enough information, answer immediately, do not search unnecessarily.
- When answering, output the option letter and content (e.g., "B. The character is cooking").
- Use character names from retrieved information, not placeholder IDs like <character_0>."""

# Open-ended QA prompt (no options): used for free-text answer datasets such as m3-bench
# Design principles (following the established conventions of SYSTEM_PROMPT):
#   1. Keep the static prompt streamlined: include only format / retrieval modes / tone / hard format constraints
#   2. The decision guide (when to Answer / VIDEO / NEIGHBOR / rewrite the query) is injected dynamically into the
#      user message at runtime by SufficiencySignal and BudgetGuardrail based on the situation (see
#      _result_formatter.format_search_result / format_empty_result in multi_round.py)
#   3. Identity resolution (character_N -> name) is kept in the static prompt, because it is "domain knowledge"
#      rather than a "decision path", and the sufficiency module is not responsible for it
SYSTEM_PROMPT_OPEN = """You are answering an open-ended question about a video. You can search a memory bank (EventGraph + VideoGraph). Prefer evidence-based answers; when evidence is partial, you MAY cautiously infer from context, clearly distinguish direct quotation from inference in your Reason, and never fabricate specifics.

Question: {question}

Output format:
Reason: <your reasoning, citing which retrieved snippet supports the answer>
Action: [Answer] or [Search]
Content: <concise free-form answer OR search query>

Search modes (use as Content):
1. <query>: search EventGraph for the most relevant event/segment (returns focus event + 1-hop neighbors + edges)
2. VIDEO: <query>: drill down into fine-grained details (objects, actions, dialogues, attributes) within the current focus event's clips
3. NEIGHBOR: <segment_label>: shift focus to an adjacent event (segment_label must come from the previous EventGraph retrieval)

Query quality (applies to every search round, not just the first):
- Every query MUST carry enough semantic signal to be embedded-matched against event/VideoGraph nodes. A single token (a bare name like "Bob", a bare noun like "dinner", a bare color) is NOT acceptable. The retrieval index stores full event summaries, so bare tokens almost always return weak/irrelevant hits.
- Build each query as: <subject> + <predicate or attribute> + <optional context>. Examples:
    BAD:  "Bob" | "dinner" | "red pot" | "Cary cooking ability" | "Sophie"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
          "Cary preparing ingredients or serving dishes in the kitchen"
          "Sophie emotional state and daily routine expressions"
- If the question names a person (e.g., "Bob", "Sophie"), the memory bank usually indexes that person under a `<character_N>` placeholder, NOT the literal name. So a plain-name search tends to match weakly. Two acceptable first moves:
    (a) issue a binding query `What is the name of <character_0>?` (use any candidate placeholder you have seen), then reuse the ID in later queries, e.g., `VIDEO: <character_2> cooking dinner`;
    (b) go straight to a content-rich query combining the name with what the question really asks, e.g., `Bob's behavior when playing basketball indoors` rather than just `Bob`.
- If the previous round returned a low-scoring / weakly-matched event (the focus summary barely touches the question, or event_score is low), do NOT just retry with a near-duplicate query. REWRITE with more predicates, or switch to VIDEO: / NEIGHBOR: / binding query as appropriate.

Identity resolution:
- The EventGraph uses anonymized placeholders like `<character_0>`, `<face_1>`. If exactly one named person (from the question) appears in the retrieved event and no contradicting evidence exists, you MAY treat that placeholder as that person, state this assumption briefly in Reason.

Key rules:
- Analyze retrieved information carefully before deciding to search again.
- You MUST perform at least one [Search] before issuing any [Answer]. The first turn's user message is just a placeholder, NOT a retrieval result.
- When you have enough information, answer immediately, do not search unnecessarily.
- Keep the final answer concise (a short phrase or a single sentence); no option letters.
- Use real character names; NEVER output placeholder IDs like `<character_0>` or `<face_1>` in the final answer."""

LAST_ROUND_PROMPT = """You are given a question about a specific video, a set of answer options, and all the retrieved knowledge from previous search rounds. Each entry in the knowledge list contains a search query and its retrieval results from the EventGraph and/or VideoGraph.

Your task is to analyze the provided information, reason over it, and select the most reasonable and well-supported option as the answer to the question.

Output Requirements:
  - Your response must begin with a brief reasoning process that explains how you arrive at the answer.
  - Then, output the answer in the format:
    Reason: <your reasoning>
    Action: [Answer]
    Content: <option letter and content>
  - You must select exactly one option from the provided choices. Do not generate free-form answers.
  - Your selection must be definite — even if the information is partial or ambiguous, you must infer and select the most reasonable option based on the given evidence.
  - Do not refuse to answer or say that the answer is unknowable. Use reasoning to reach the best possible conclusion.
  - You MUST NOT output [Search] or any search query. This is the final round — no more searches are allowed.

Additional Guidelines:
  - When referring to a character, always use their specific name if it appears in the retrieved information.
  - Do not use placeholder tags like <character_1> or <face_1>.
  - Focus on reasoning and selecting the correct option.

Question: {question}
Options:
{options}

All Retrieved Knowledge:
{search_results}"""

# Forced-answer prompt for the last round of open-ended QA
# Following the baseline last_round_prompt: force a definite answer, refusals are not allowed
LAST_ROUND_PROMPT_OPEN = """You are given an open-ended question about a specific video and all the retrieved knowledge from previous search rounds. Each entry in the knowledge list contains a search query and its retrieval results from the EventGraph and/or VideoGraph.

Your task is to analyze the provided information, reason over it, and produce the best concise answer.

Output Requirements:
  - Your response must begin with a brief reasoning process that explains how you arrive at the answer.
  - Then, output the answer in the format:
    Reason: <your reasoning>
    Action: [Answer]
    Content: <concise free-form answer, a short phrase or single sentence>
  - Your answer MUST be definite — even if the information is partial or ambiguous, you MUST infer and give the most reasonable answer based on the given evidence. Do NOT refuse to answer and do NOT say the answer is unknown.
  - You MUST NOT output [Search] or any search query. This is the final round — no more searches are allowed.

Handling presupposed choices:
  - If the question presents a binary or limited choice (e.g., "X or Y?", "taller or shorter?", "A, B, or C?"), you MUST pick one of the presented options. Do NOT reply with "neither", "cannot be determined", "only one exists", or similar — the question assumes the choice is valid.
  - If the retrieved evidence does not directly support any option, use common sense + partial cues (e.g., scene context, character attributes, typical behavior) to pick the more likely option. Explicitly mark this as an inference in your Reason ("Inference: ...").

Identity resolution reminder:
  - The retrieved knowledge may refer to people with placeholders like `<character_0>`. In the final answer, use the real names mentioned in the question or retrieved information.
  - If only one named person in the question matches a placeholder in the relevant event, treat them as the same person and answer with the real name.

Additional Guidelines:
  - When referring to a character, always use their specific name if it appears in the retrieved information.
  - Do not use placeholder tags like <character_1> or <face_1>.
{binary_choice_hint}
Question: {question}

All Retrieved Knowledge:
{search_results}"""

ACTION_PATTERN = r"Action:\s*\[?\s*(Answer|Search)\s*\]?\s*.*?Content:\s*(.*)"


# ============================================================
# Visual-layer (visual_layer) prompt fragments
# ------------------------------------------------------------
# Design principles (consistent with the conventions of SYSTEM_PROMPT above):
#   1. The static prompt only describes capabilities and hard constraints, without a decision tree; when to use KEYFRAME
#      is described by the sufficiency runtime hint plus deferred skill injection.
#   2. Appended to the end of sys_prompt only when visual_layer.enabled=true;
#      when disabled, the original prompt is byte-for-byte unchanged, guaranteeing zero difference for old runs.
# ============================================================
_KEYFRAME_CAPABILITY_MCQ = """

Keyframe inspection (visual layer, IMPORTANT):
4. KEYFRAME: <query> — load keyframe images of the current focus event's clips into your view.
   You SHOULD actively use KEYFRAME (not just as a fallback) whenever the question asks about:
     - spatial layout / object location ("where is X", "which table/shelf is X on", "X is next to Y")
     - visual appearance ("what color", "what is X wearing", "what does the sign say")
     - on-going activity ("what is X doing right now", "what is X holding")
   Textual memories rarely capture these details reliably; keyframes do.
   Requirements:
     - Issue KEYFRAME AFTER at least one plain query, VIDEO:, or NEIGHBOR: has established a focus event.
       (If no focus exists yet, the system will pick a best-match event for you as a soft fallback.)
     - Up to 5 images will be attached below the next user message. Cite them explicitly in Reason."""

_KEYFRAME_CAPABILITY_OPEN = """

Keyframe inspection (visual layer, IMPORTANT):
4. KEYFRAME: <query> — load keyframe images of the current focus event's clips into your view.
   You SHOULD actively use KEYFRAME (not just as a fallback) whenever the question asks about:
     - spatial layout / object location ("where is X", "which table/shelf is X on", "X is next to Y")
     - visual appearance ("what color", "what is X wearing", "what does the sign say")
     - on-going activity ("what is X doing right now", "what is X holding")
   Textual memories rarely capture these details reliably; keyframes do.
   Requirements:
     - Issue KEYFRAME AFTER at least one plain query, VIDEO:, or NEIGHBOR: has established a focus event.
       (If no focus exists yet, the system will pick a best-match event for you as a soft fallback.)
     - Up to 5 images will be attached below the next user message. Cite them explicitly in Reason
       (e.g., "Reason: Keyframe from clip 3 shows the laptop on the low table in front of the couch ...")."""

_KEYFRAME_LAST_ROUND_HINT = (
    "\n  - If keyframe images were attached during previous rounds, recall and "
    "incorporate those visual observations (colors, on-screen text, spatial "
    "layout, appearances) into your reasoning."
)


# ============================================================
# VideoGraph-Only ablation prompt (baseline agent only)
# ------------------------------------------------------------
# Uses only VideoGraph for semantic retrieval, without EventGraph / NEIGHBOR / KEYFRAME.
# ============================================================
SYSTEM_PROMPT_VIDEOGRAPH_ONLY = """You are answering a multiple-choice question about a video. You can search a VideoGraph memory bank for relevant information before answering.

Question: {question}
Options: {options}

Output format:
Reason: <your reasoning>
Action: [Answer] or [Search]
Content: <answer option OR search query>

Search mode:
- <query>: search VideoGraph for relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.). Each search returns the top-k most similar memory nodes.

Query quality:
- Every query MUST carry enough semantic signal. A single token or bare noun is NOT acceptable.
- Build each query as: <subject> + <predicate or attribute> + <optional context>.
    BAD:  "Bob" | "dinner" | "red pot"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
- If the previous round returned weak results, do NOT retry with a near-duplicate. REWRITE with more predicates or try a completely different angle.

Identity resolution:
- The VideoGraph may use anonymized placeholders like `<character_0>`. If exactly one named person matches, you MAY treat that placeholder as that person.

Key rules:
- You MUST perform at least one [Search] before issuing any [Answer].
- When you have enough information, answer immediately.
- When answering, output the option letter and content.
- Use character names from retrieved information, not placeholder IDs."""

SYSTEM_PROMPT_OPEN_VIDEOGRAPH_ONLY = """You are answering an open-ended question about a video. You can search a VideoGraph memory bank for relevant information before answering. Prefer evidence-based answers; when evidence is partial, you MAY cautiously infer from context and never fabricate specifics.

Question: {question}

Output format:
Reason: <your reasoning, citing which retrieved snippet supports the answer>
Action: [Answer] or [Search]
Content: <concise free-form answer OR search query>

Search mode:
- <query>: search VideoGraph for relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.). Each search returns the top-k most similar memory nodes.

Query quality:
- Every query MUST carry enough semantic signal. A single token or bare noun is NOT acceptable.
- Build each query as: <subject> + <predicate or attribute> + <optional context>.
    BAD:  "Bob" | "dinner" | "red pot" | "Sophie"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
          "Sophie emotional state and daily routine expressions"
- If the previous round returned weak results, do NOT retry with a near-duplicate. REWRITE with more predicates or try a completely different angle.

Identity resolution:
- The VideoGraph may use anonymized placeholders like `<character_0>`. If exactly one named person matches, you MAY treat that placeholder as that person.

Key rules:
- You MUST perform at least one [Search] before issuing any [Answer].
- When you have enough information, answer immediately.
- Keep the final answer concise (a short phrase or a single sentence); no option letters.
- Use real character names; NEVER output placeholder IDs in the final answer."""


# ============================================================
# No-Graph-Walk ablation prompt (baseline agent only)
# ------------------------------------------------------------
# Keeps EventGraph event-node retrieval + VideoGraph fine-grained retrieval,
# but removes NEIGHBOR (graph-structure traversal) and KEYFRAME (keyframe visual inspection).
# ============================================================
SYSTEM_PROMPT_NO_GRAPH_WALK = """You are answering a multiple-choice question about a video. You can search a memory bank for relevant information before answering.

Question: {question}
Options: {options}

Output format:
Reason: <your reasoning>
Action: [Answer] or [Search]
Content: <answer option OR search query>

Search modes (use as Content):
1. <query>: search EventGraph for relevant events (returns focus event summary and clip information)
2. VIDEO: <query>: drill down into fine-grained details within the current focus event

Query quality:
- Every query MUST carry enough semantic signal. A single token or bare noun is NOT acceptable.
- Build each query as: <subject> + <predicate or attribute> + <optional context>.
    BAD:  "Bob" | "dinner" | "red pot"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
- If the previous round returned weak results, do NOT retry with a near-duplicate. REWRITE with more predicates or switch to VIDEO:.

Identity resolution:
- The EventGraph uses anonymized placeholders like `<character_0>`. If exactly one named person matches, you MAY treat that placeholder as that person.

Key rules:
- You MUST perform at least one [Search] before issuing any [Answer].
- When you have enough information, answer immediately.
- When answering, output the option letter and content.
- Use character names from retrieved information, not placeholder IDs."""

SYSTEM_PROMPT_OPEN_NO_GRAPH_WALK = """You are answering an open-ended question about a video. You can search a memory bank (EventGraph + VideoGraph). Prefer evidence-based answers; when evidence is partial, you MAY cautiously infer from context and never fabricate specifics.

Question: {question}

Output format:
Reason: <your reasoning, citing which retrieved snippet supports the answer>
Action: [Answer] or [Search]
Content: <concise free-form answer OR search query>

Search modes (use as Content):
1. <query>: search EventGraph for the most relevant event/segment (returns focus event summary and clip information)
2. VIDEO: <query>: drill down into fine-grained details (objects, actions, dialogues, attributes) within the current focus event's clips

Query quality:
- Every query MUST carry enough semantic signal. A single token or bare noun is NOT acceptable.
- Build each query as: <subject> + <predicate or attribute> + <optional context>.
    BAD:  "Bob" | "dinner" | "red pot" | "Sophie"
    GOOD: "Bob disrupts Alice homework with basketball"
          "what dish is being cooked in the red pot on the stove"
          "Sophie emotional state and daily routine expressions"
- If the previous round returned weak results, do NOT retry with a near-duplicate. REWRITE with more predicates or switch to VIDEO:.

Identity resolution:
- The EventGraph uses anonymized placeholders like `<character_0>`. If exactly one named person matches, you MAY treat that placeholder as that person.

Key rules:
- You MUST perform at least one [Search] before issuing any [Answer].
- When you have enough information, answer immediately.
- Keep the final answer concise (a short phrase or a single sentence); no option letters.
- Use real character names; NEVER output placeholder IDs in the final answer."""


class MultiRoundSearchAgent(ReasoningAgent):
    """Multi-round retrieval reasoning Agent.

    Reuses the core logic in control_api_high_system.py,
    wrapped as a standalone ReasoningAgent implementation.

    Args:
        config: configuration dict, containing:
            - backend: "openai" or "vllm"
            - model: model name
            - max_rounds: maximum number of reasoning rounds
            - temperature: sampling temperature
            - max_tokens: maximum number of generated tokens
            - api_config_path: path to the API config file
            - workers: concurrency (openai mode)
    """

    def __init__(self, config: dict):
        self.backend = config.get("backend", "openai")
        self.model = config.get("model", "gemini-2.5-flash")
        self.max_rounds = config.get("max_rounds", 5)
        self.temperature = config.get("temperature", 0.6)
        self.max_tokens = config.get("max_tokens", 8192)
        self.seed = config.get("seed", None)  # decoding seed, None means not fixed
        self.workers = config.get("workers", 4)
        self._extra_instructions = ""
        # P2 staged injection: queue of mid-turn skill hints pending flush
        # Does not modify the system prompt; instead appends a hint before the next user message,
        # injected together with the orchestrator based on runtime state (number of empty searches / high-confidence about to answer)
        self._pending_hints: list = []
        # Runtime state counters (visible to the orchestrator, used to decide whether to trigger a skill hint)
        self._runtime_state: dict = {
            "consecutive_empty_searches": 0,
            "total_empty_searches": 0,
            "rounds_so_far": 0,
        }

        # DeepSeek-V4 thinking mode switch (off by default; scripts can enable it via ENABLE_THINKING=true)
        self._enable_thinking = bool(config.get("enable_thinking", False))

        # Third layer: visual-layer switch (false by default; overridden by the orchestrator via reasoning.visual_layer_enabled
        # or the memory config). When enabled=false, sys_prompt is exactly the same as before.
        self._visual_layer_enabled = bool(config.get("visual_layer_enabled", False))
        self._current_keyframe_available = self._visual_layer_enabled
        self._current_keyframe_preflight: Dict[str, Any] = {}
        # List of unconsumed keyframe paths for the current question, flushed in multimodal form in the next user message
        self._pending_keyframe_paths: list = []

        # Memory strategy (projected by the orchestrator from memory.strategy)
        self._memory_strategy: str = config.get("memory_strategy", "hierarchical")

        # Constraint system (Harness Engineering: Constrain + Verify)
        guardrail_cfg = config.get("guardrails", {})
        self._guardrails = AgentGuardrails(guardrail_cfg)

        # Information sufficiency signal (Harness Engineering: information-increment evaluation + early stopping)
        sufficiency_cfg = config.get("sufficiency", {})
        self._sufficiency = SufficiencySignal(sufficiency_cfg)

        # Context engineering (Harness Engineering: retrieval result formatting + conversation management)
        context_cfg = config.get("context_engineering", {})
        self._result_formatter = RetrievalResultFormatter()
        self._conversation_mgr = ConversationManager(context_cfg)

        # Initialize the backend
        self._client = None
        api_config_path = config.get("api_config_path", "configs/api_config.json")
        if self.backend == "openai":
            self._init_openai(api_config_path)

    def _init_openai(self, api_config_path: str):
        """Initialize the OpenAI client."""
        import openai
        with open(api_config_path) as f:
            api_cfg = json.load(f)
        if self.model not in api_cfg:
            raise KeyError(f"Model '{self.model}' is not configured in {api_config_path}")
        self._client = openai.OpenAI(
            base_url=api_cfg[self.model].get("base_url", None),
            api_key=api_cfg[self.model]["api_key"],
        )

    def _generate(self, messages: List[Dict], timeout: int = 120) -> str:
        """Call the LLM to generate a response.

        Third-layer hook-in: if the user text in messages contains a keyframe marker and the agent has recorded
        the corresponding paths (_keyframe_paths_by_msg_idx), assemble them dynamically into a multimodal content list before sending.
        conversations itself stays pure text, so there is no burden during persistence.
        """
        if self._visual_layer_enabled and self._keyframe_paths_by_msg_idx:
            messages = self._build_multimodal_messages(
                messages, self._keyframe_paths_by_msg_idx
            )

        # DeepSeek-V4 series: control thinking mode via self._enable_thinking (off by default)
        extra_kwargs = {}
        if "deepseek-v4" in self.model:
            extra_kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}

        # JSON mode: set temporarily by _planner_generate and others to force the LLM to output valid JSON
        if getattr(self, "_json_mode", False):
            extra_kwargs["response_format"] = {"type": "json_object"}

        for retry in range(10):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    seed=self.seed,
                    timeout=timeout,
                    **extra_kwargs,
                )
                return resp.choices[0].message.content
            except Exception as e:
                sleep_s = min(20, 2 * (retry + 1))
                logger.warning(f"[LLM Retry {retry+1}/10] {e}")
                time.sleep(sleep_s)
        raise RuntimeError("LLM call failed (already retried 10 times)")

    # ------------------------------------------------------------------
    # Third-layer helper methods: keyframe text marker and multimodal message assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _format_keyframe_marker(kf_paths: List[Dict[str, Any]]) -> str:
        """Generate a pure-text semantic marker for the keyframes (without base64). As a suffix to the user message,
        it helps the LLM and trajectory readers identify which clips' keyframes are attached this round.
        """
        if not kf_paths:
            return ""
        clip_ids = sorted({kf.get("clip_id") for kf in kf_paths if kf.get("clip_id") is not None})
        return (
            f"\n(Visual keyframes attached: {len(kf_paths)} image(s) from clips "
            f"{clip_ids}. Examine them for visual details such as colors, "
            "on-screen text, spatial layout, appearances.)"
        )

    def _build_multimodal_messages(self, messages: List[Dict],
                                   paths_by_idx: Dict[int, List[Dict[str, Any]]]
                                   ) -> List[Dict]:
        """Convert pure-text messages into multimodal messages (assembling images by index).

        - Rule: replace only when the index i of messages[i] hits paths_by_idx and content is a str.
        - Compatible with force_messages / retry_conversations: these temporary conversations differ in length from conversations,
          so we must also verify that "the original message content corresponding to idx in paths_by_idx" is consistent with the current messages[idx];
          to be safe, we adopt a "simple version" strategy here: replace only when len(messages) > idx and messages[idx]["role"] == "user"
          and content is a str and contains the "Visual keyframes attached" marker; otherwise fall back silently.
        """
        if not paths_by_idx:
            return messages
        msgs_out = list(messages)
        for idx, kf_paths in paths_by_idx.items():
            if idx < 0 or idx >= len(msgs_out):
                continue
            msg = msgs_out[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or "Visual keyframes attached" not in content:
                continue
            parts: List[Dict[str, Any]] = [{"type": "text", "text": content}]
            for kf in kf_paths:
                p = kf.get("path")
                cid = kf.get("clip_id")
                if not p:
                    continue
                try:
                    b64 = _load_image_b64(p)
                except Exception as exc:
                    logger.warning(f"[Keyframe] failed to load image {p}: {exc}")
                    continue
                parts.append({"type": "text", "text": f"[Keyframe from clip {cid}]"})
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            msgs_out[idx] = {"role": "user", "content": parts}
        return msgs_out

    def _refresh_keyframe_preflight(self, memory: MemoryStrategy) -> bool:
        """Before each question begins, check whether the current video has a usable keyframe directory."""
        if not self._visual_layer_enabled:
            self._current_keyframe_available = False
            self._current_keyframe_preflight = {"available": False, "reason": "visual_layer_disabled"}
            return False

        preflight_fn = getattr(memory, "keyframe_preflight", None)
        if not callable(preflight_fn):
            self._current_keyframe_available = False
            self._current_keyframe_preflight = {"available": False, "reason": "memory_preflight_unavailable"}
            return False

        try:
            info = preflight_fn() or {}
        except Exception as exc:
            logger.warning(f"[Keyframe] preflight failed: {exc}")
            info = {"available": False, "reason": "preflight_exception"}
        self._current_keyframe_preflight = info
        self._current_keyframe_available = bool(info.get("available", False))
        if not self._current_keyframe_available:
            logger.info(
                "[Keyframe] disabled for current question: %s (%s)",
                info.get("reason", "unknown"), info.get("keyframe_dir", ""),
            )
        return self._current_keyframe_available

    def _parse_action(self, response: str):
        """Parse the Action and Content from the model response.

        Compatible with the following formats (by frequency):
          - Action: [Search]    Content: ...
          - Action: Search      Content: ...            <- the model often omits the brackets
          - action: search      content: ...            <- case drift
          - Action:[Answer]Content:...                  <- missing spaces
        """
        # Strip the think tag
        text = response.split("</think>")[-1] if "</think>" in response else response
        match = re.search(ACTION_PATTERN, text, re.DOTALL | re.IGNORECASE)
        if match:
            action = match.group(1).strip().capitalize()  # "search"/"SEARCH" -> "Search"
            content = match.group(2).strip()
            # Words other than Answer/Search are not accepted (the whitelist is already constrained in the regex; this is a fallback)
            if action not in ("Answer", "Search"):
                return "Search", None
            return action, content or None
        return "Search", None

    # Presupposed binary / limited-choice detection for "X or Y?" / "taller or shorter?"
    # Examples:
    #   "Which coat rack should Emma's coat be laced, taller one or shorter one?"
    #   "Is the car red or blue?"
    #   "Does Tom prefer tea, coffee or water?"
    # Matching forms:
    #   A) comma/colon-led enumeration: ", A or B" / ": A, B or C"
    #   B) verb-prefixed: Is/Was/Does/Did/Should/Could/Will... <tail containing "A or B?">
    _BINARY_SPLIT_REGEX = re.compile(
        r"(?:,|:)\s*([^,?]+?)\s+or\s+([^,?]+?)(?:[?.]|$)",
        re.IGNORECASE,
    )
    _MULTI_CHOICE_REGEX = re.compile(
        r"(?:,|:)\s*([^,?]+?)\s*,\s*([^,?]+?)\s+or\s+([^,?]+?)(?:[?.]|$)",
        re.IGNORECASE,
    )
    # Verb-prefix fallback: match the whole sentence, capture both sides of "or", then truncate in post-processing
    _VERB_LED_OR_REGEX = re.compile(
        r"^\s*(?:is|are|was|were|does|did|do|has|have|should|could|would|will|can)\b"
        r"\s*(.*?)\s+or\s+(.*?)\s*[?.]\s*$",
        re.IGNORECASE,
    )

    def _build_binary_choice_hint(self, question: str) -> str:
        """If the question contains presupposed options in the "X or Y?" style, extract the candidates and build a hint.

        Returns:
          - On a match: a hint string like "\n[Presupposed choices]..." (with surrounding newlines)
          - On no match: an empty string
        """
        if not question:
            return ""
        q = question.strip()

        # First try the three-way choice (form A)
        m3 = self._MULTI_CHOICE_REGEX.search(q)
        if m3:
            opts = [m3.group(i).strip().strip(".?! ") for i in (1, 2, 3)]
            opts = [o for o in opts if 1 <= len(o) <= 40]
            if len(opts) == 3:
                return (
                    "\n[Presupposed choices detected]\n"
                    f"  The question presupposes a choice among: {opts[0]} / {opts[1]} / {opts[2]}.\n"
                    "  You MUST pick exactly one. Do NOT say \"neither\", \"none\", or "
                    "\"cannot be determined\". If evidence is inconclusive, infer the "
                    "more likely option from context and mark it as an inference.\n"
                )

        # Form A: comma/colon-led binary
        m2 = self._BINARY_SPLIT_REGEX.search(q)
        if m2:
            a = m2.group(1).strip().strip(".?! ")
            b = m2.group(2).strip().strip(".?! ")
            if 1 <= len(a) <= 40 and 1 <= len(b) <= 40:
                return self._format_binary_hint(a, b)

        # Form B: verb-prefix fallback ("Is the car red or blue?")
        mv = self._VERB_LED_OR_REGEX.match(q)
        if mv:
            left_raw = mv.group(1).strip().strip(",.?! ")
            right_raw = mv.group(2).strip().strip(",.?! ")
            # Left side: take the last 1-3 tokens ("the car red" -> "red"; "the dog barking" -> "dog barking"; "Nancy leave before" -> "leave before")
            # Right side: take the first 1-3 tokens ("after Tom" -> "after Tom"; "blue" -> "blue"; "sleeping" -> "sleeping")
            def _tail(s, k=2):
                toks = s.split()
                return " ".join(toks[-k:]) if toks else ""
            def _head(s, k=2):
                toks = s.split()
                return " ".join(toks[:k]) if toks else ""
            a = _tail(left_raw, 2)
            b = _head(right_raw, 2)
            if 1 <= len(a) <= 30 and 1 <= len(b) <= 30:
                return self._format_binary_hint(a, b)

        return ""

    @staticmethod
    def _format_binary_hint(a: str, b: str) -> str:
        return (
            "\n[Presupposed choices detected]\n"
            f"  The question presupposes a binary choice: \"{a}\" vs \"{b}\".\n"
            "  You MUST pick exactly one. Do NOT say \"neither\", \"only one "
            "exists\", or \"cannot be determined\". If the evidence does not "
            "explicitly support either option, use common sense and partial "
            "cues to infer the more likely one, and mark it as an inference "
            "in your Reason.\n"
        )


    def _validated_parse(self, response: str, options: List[str],
                         conversations: List[Dict],
                         is_last_round: bool = False,
                         require_prior_search: bool = False) -> tuple:
        """Response parsing with constraint validation (self-repair loop).

        Harness Engineering core: Constrain + Verify + Feedback Loop
        1. Validate the Agent output format
        2. Validate the legality of the answer/query
        3. On validation failure, retry with the error message (self-repair)
        4. On the last-round retry, force [Answer] and disallow [Search]
        5. Distinguish format errors from strategy errors (such as duplicate queries) and give the Agent different feedback
        6. When require_prior_search=True, disallow [Answer] before any search (anti-hallucination for open-ended QA)
        """
        action, content, feedback = self._guardrails.validate_response(
            response, options, require_prior_search=require_prior_search,
        )

        # Last round: if the Agent outputs Search, force it to Answer
        if is_last_round and action == "Search":
            logger.info("[Guardrail] Last round: Agent output Search, forcing it to Answer")
            # Try to extract useful answer content from the original response
            fallback_answer = self._extract_answer_from_response(response, options)
            if fallback_answer:
                return "Answer", fallback_answer
            # Cannot extract; return the original content for the evaluator to handle
            return "Answer", content or ""

        if not feedback:
            return action, content

        # Determine feedback type: strategy error (duplicate query) vs format error
        is_strategy_error = "already searched similar" in feedback.lower()

        logger.info(f"[Guardrail] Validation failed ({'strategy error' if is_strategy_error else 'format error'}), "
                    f"triggering self-repair: {feedback[:80]}...")

        # Build different retry messages depending on the error type
        if is_last_round:
            retry_msg = (
                f"Your previous response had an issue:\n{feedback}\n\n"
                f"IMPORTANT: This is the FINAL round. You MUST output [Answer], NOT [Search].\n"
                f"Please select the most reasonable option from the choices and output:\n"
                f"Reason: <your reasoning>\n"
                f"Action: [Answer]\n"
                f"Content: <option letter and content>"
            )
        elif is_strategy_error:
            # Strategy error: guide the Agent to change strategy or answer directly
            retry_msg = (
                f"STRATEGY ISSUE: {feedback}\n\n"
                f"You have been searching for similar information repeatedly. "
                f"Please choose ONE of the following:\n"
                f"1. If you already have enough information, output [Answer] with your best choice.\n"
                f"2. If you must search, use a COMPLETELY DIFFERENT strategy:\n"
                f"   - Try 'VIDEO: <query>' to get fine-grained details within the current event\n"
                f"   - Try 'NEIGHBOR: <segment_label>' to explore an adjacent event\n"
                f"   - Use entirely different keywords to search for a different event\n\n"
                f"Output in the format:\n"
                f"Reason: <your reasoning>\n"
                f"Action: [Answer] or [Search]\n"
                f"Content: <content>"
            )
        else:
            # Format error: ask for a corrected format
            retry_msg = (
                f"Your previous response had a format issue:\n{feedback}\n\n"
                f"Please correct your response and output again in the required format."
            )

        # Key: retry on a copy of the conversation to avoid the correction message polluting the main conversation history
        # This way the Agent in later rounds will not see the irrelevant correction conversation
        retry_conversations = conversations.copy()
        retry_conversations.append({"role": "user", "content": retry_msg})

        retry_response = self._generate(retry_conversations)

        # Re-validate (no recursion, to avoid an infinite loop)
        action2, content2, feedback2 = self._guardrails.validate_response(
            retry_response, options, require_prior_search=require_prior_search,
        )

        if not feedback2:
            # Last round: even if the format is correct, Search is not allowed
            if is_last_round and action2 == "Search":
                logger.info("[Guardrail] Still Search after the last-round retry, forcing it to Answer")
                fallback = self._extract_answer_from_response(retry_response, options)
                return "Answer", fallback or content2 or ""
            # Retry succeeded: replace the original erroneous response in the main conversation with the valid retried response
            if conversations and conversations[-1]["role"] == "assistant":
                conversations[-1]["content"] = retry_response
            return action2, content2

        # Both attempts failed
        logger.warning(f"[Guardrail] Self-repair failed")

        # Special handling for strategy errors: the Agent gives duplicate queries twice, suggesting it may already have enough information
        # Try to extract an answer from the original response and guide it to answer directly
        if is_strategy_error:
            extracted = self._extract_answer_from_response(response, options)
            if extracted:
                logger.info(f"[Guardrail] Strategy-error fallback: extracted an answer from the original response: {extracted[:60]}")
                return "Answer", extracted
            # If no answer can be extracted, accept the original query but skip duplicate detection
            # (letting the Agent keep reasoning is better than getting stuck)
            original_action, original_content = self._parse_action(response)
            if original_action == "Search" and original_content:
                logger.info(f"[Guardrail] Strategy-error fallback: allow the original query (skip duplicate detection)")
                # Manually add the query to the history to avoid triggering it again next time
                self._guardrails.state.search_queries.append(original_content)
                return "Search", original_content

        # General fallback: try to extract an answer from the original response
        extracted = self._extract_answer_from_response(response, options)
        if extracted:
            logger.info(f"[Guardrail] Extracted an answer from the original response: {extracted[:60]}")
            return "Answer", extracted

        # Last-round fallback: return Answer no matter what
        if is_last_round:
            fallback_action, fallback_content = self._parse_action(response)
            return "Answer", fallback_content or ""

        # Normal round: fall back to the original parsing
        fallback_action, fallback_content = self._parse_action(response)
        return fallback_action, fallback_content

    @staticmethod
    def _extract_answer_from_response(response: str, options: List[str]) -> Optional[str]:
        """Try to extract an answer from a non-standard-format response.

        Handles cases where the Agent outputs an answer but in an irregular format, for example:
        - "The final answer is $\\boxed{C}$"
        - "I think the answer is B. ..."
        - directly outputs the option content
        """
        text = response.split("</think>")[-1] if "</think>" in response else response

        # Try to match the \boxed{X} format
        boxed_match = re.search(r'\\boxed\{([A-Da-d])\}', text)
        if boxed_match:
            letter = boxed_match.group(1).upper()
            # Find the corresponding full option
            for opt in options:
                if opt.strip().startswith(f"{letter}."):
                    return opt.strip()
            return letter

        # Try to match the "the answer is X" format
        answer_match = re.search(r'(?:the\s+)?answer\s+is\s+([A-Da-d])\b', text, re.IGNORECASE)
        if answer_match:
            letter = answer_match.group(1).upper()
            for opt in options:
                if opt.strip().startswith(f"{letter}."):
                    return opt.strip()
            return letter

        # Try to match a standalone option letter (such as "C. 8")
        option_match = re.search(r'\b([A-Da-d])\.\s', text)
        if option_match:
            letter = option_match.group(1).upper()
            for opt in options:
                if opt.strip().startswith(f"{letter}."):
                    return opt.strip()
            return letter

        # Try to match the option content
        for opt in options:
            opt_content = re.sub(r'^[A-Da-d]\.\s*', '', opt).strip()
            if opt_content and len(opt_content) > 1 and opt_content.lower() in text.lower():
                return opt.strip()

        return None

    def answer(self, question: TemporalQuestion,
               memory: MemoryStrategy) -> AgentAnswer:
        """Multi-round reasoning flow (integrating Harness Engineering constraints and validation).

        Compatible with two QA formats:
          - Multiple-choice (options non-empty): uses SYSTEM_PROMPT / LAST_ROUND_PROMPT
          - Open-ended QA (options empty): uses SYSTEM_PROMPT_OPEN / LAST_ROUND_PROMPT_OPEN
        """
        is_open_ended = not question.options
        option_text = "\n".join(question.options) if question.options else ""

        # Reset the clip_wise deduplication state (every new question starts from scratch)
        if hasattr(memory, "reset_current_clips"):
            memory.reset_current_clips()

        # Reset the constraint system state
        self._guardrails.reset()
        # Set the current question so the identity-resolution guardrail can extract person names and make context judgments
        self._guardrails.set_question_text(question.question)
        keyframe_available = self._refresh_keyframe_preflight(memory)

        # Build the initial conversation
        if self._memory_strategy == "videograph_only":
            # VideoGraph-Only ablation: use the simplified prompt, without EventGraph/NEIGHBOR/KEYFRAME
            if is_open_ended:
                sys_prompt = SYSTEM_PROMPT_OPEN_VIDEOGRAPH_ONLY.format(question=question.question)
            else:
                sys_prompt = SYSTEM_PROMPT_VIDEOGRAPH_ONLY.format(
                    question=question.question, options=option_text
                )
            # In videograph_only mode, do not append the keyframe capability block
            keyframe_available = False
        elif self._memory_strategy == "no_graph_walk":
            # No-Graph-Walk ablation: keep EventGraph node retrieval + VideoGraph, remove NEIGHBOR/KEYFRAME
            if is_open_ended:
                sys_prompt = SYSTEM_PROMPT_OPEN_NO_GRAPH_WALK.format(question=question.question)
            else:
                sys_prompt = SYSTEM_PROMPT_NO_GRAPH_WALK.format(
                    question=question.question, options=option_text
                )
            # In no_graph_walk mode, do not append the keyframe capability block
            keyframe_available = False
        elif is_open_ended:
            sys_prompt = SYSTEM_PROMPT_OPEN.format(question=question.question)
        else:
            sys_prompt = SYSTEM_PROMPT.format(
                question=question.question, options=option_text
            )
        # Optional third layer: visual-layer capability description (appended only when enabled=true; byte-for-byte unchanged when disabled)
        if keyframe_available:
            sys_prompt += (
                _KEYFRAME_CAPABILITY_OPEN if is_open_ended else _KEYFRAME_CAPABILITY_MCQ
            )
        if self._extra_instructions:
            sys_prompt += f"\n\n{self._extra_instructions}"

        # Initial user message: explicitly state "no retrieval yet" to avoid the model mistaking "{}" for a retrieval result
        # MCQ and OPEN use the same first-turn prompt, forcing search before answering
        initial_user = (
            "(no retrieval yet) You have not issued any search. "
            "Please start with [Search]. Do NOT output [Answer] in this turn."
        )

        # Third layer: if the visual layer is on and the question hits visual keywords, append a one-time hint to the first user message,
        # reminding the model to prioritize KEYFRAME after locating a focus.
        # Likewise this only takes effect when enabled=true; when disabled, initial_user is byte-for-byte unchanged.
        if keyframe_available and _mem_is_visual_query(question.question):
            initial_user += (
                "\n\n[Visual cues detected in the question] This question likely "
                "depends on visual details (layout / appearance / spatial position / "
                "on-going activity). After you locate a focus event (via a plain query, "
                "VIDEO:, or NEIGHBOR:), strongly consider issuing `KEYFRAME: <query>` "
                "to inspect the actual keyframes before answering."
            )
        conversations = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": initial_user},
        ]

        all_search_results = []
        search_queries = []
        focus_label = None

        # Reset the information sufficiency signal
        self._sufficiency.reset()

        # P2 staged injection: clear hints and runtime counters left over from the previous question
        self._pending_hints = []
        self._runtime_state = {
            "consecutive_empty_searches": 0,
            "total_empty_searches": 0,
            "consecutive_low_increment_searches": 0,
            "same_focus_stall_count": 0,
            "last_focus_label": "",
            "rounds_so_far": 0,
        }
        # Third layer: clear the keyframe injection paths left over from the previous question
        self._pending_keyframe_paths = []
        # Third layer: record the keyframe paths associated with each user message (key = index in conversations)
        # conversations itself still stores pure text; images are assembled dynamically by index only before sending to the LLM.
        # This way existing logic such as trajectory persistence / conversation compression is completely unaffected.
        self._keyframe_paths_by_msg_idx: Dict[int, List[Dict[str, Any]]] = {}

        for round_idx in range(self.max_rounds):
            is_last_round = (round_idx == self.max_rounds - 1)

            # P2 staged injection: expose the current round, allowing external code to inject a hint just before this round's user message
            self._runtime_state["rounds_so_far"] = round_idx
            on_round_hook = getattr(self, "_on_round_start_hook", None)
            if callable(on_round_hook):
                try:
                    on_round_hook(round_idx, self._runtime_state)
                except Exception as exc:
                    logger.warning(f"[MultiRound] on_round_start_hook exception ignored: {exc}")

            # Budget check (Constrain: resource budget)
            budget_exceeded, budget_reason = self._guardrails.check_budget()
            if budget_exceeded and not is_last_round:
                logger.info(f"[Guardrail] Budget exceeded, forcing the last round: {budget_reason}")
                is_last_round = True

            if is_last_round:
                # Last round: use last_round_prompt to force an answer
                search_results_json = json.dumps(all_search_results, ensure_ascii=False)
                if is_open_ended:
                    # Detect whether the question presupposes a binary / limited choice ("A or B?", "taller or shorter?", etc.)
                    # On a match, inject a reinforcing hint that explicitly lists the candidates and forces a single choice
                    binary_hint = self._build_binary_choice_hint(question.question)
                    force_content = LAST_ROUND_PROMPT_OPEN.format(
                        question=question.question,
                        search_results=search_results_json,
                        binary_choice_hint=binary_hint,
                    )
                else:
                    force_content = LAST_ROUND_PROMPT.format(
                        question=question.question,
                        options=option_text,
                        search_results=search_results_json,
                    )
                force_messages = [
                    {"role": "user", "content": force_content}
                ]
                response = self._generate(force_messages)
                conversations.append({"role": "assistant", "content": response})

                # Use validated parsing (Verify: deterministic checks + self-repair)
                action, content = self._validated_parse(
                    response, question.options, conversations,
                    is_last_round=True,
                )
                return AgentAnswer(
                    content=content or "",
                    confidence=0.6,
                    is_final=True,
                    reasoning_trace=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    tokens_used=self._guardrails.state.tokens_used,
                    conversations=conversations,
                )

            # Conversation history management (context engineering: prevent context-window overflow)
            conversations = self._conversation_mgr.manage(conversations)

            # Normal round: generate a response
            response = self._generate(conversations)
            conversations.append({"role": "assistant", "content": response})

            # Use validated parsing (Verify: deterministic checks + self-repair)
            # In the open-ended QA scenario, if the Agent wants to answer directly before searching, the guardrail rejects it and requires a Search first
            action, content = self._validated_parse(
                response, question.options, conversations,
                require_prior_search=is_open_ended,
            )

            if action == "Answer":
                return AgentAnswer(
                    content=content or "",
                    confidence=1.0,
                    is_final=True,
                    reasoning_trace=all_search_results,
                    search_queries=search_queries,
                    num_rounds=round_idx + 1,
                    tokens_used=self._guardrails.state.tokens_used,
                    conversations=conversations,
                )

            # Search: execute retrieval
            if content:
                search_queries.append(content)
                retrieval_payload = self._execute_search(
                    content, memory, question, focus_label
                )
                # Update focus_label
                if retrieval_payload.get("focus_label"):
                    focus_label = retrieval_payload["focus_label"]

                all_search_results.append({
                    "query": content,
                    "retrieval": {"event_retrieval": retrieval_payload},
                })

                is_empty = not retrieval_payload.get("focus")
                # Synchronously update runtime_state for use by external hooks
                if is_empty:
                    self._runtime_state["consecutive_empty_searches"] += 1
                    self._runtime_state["total_empty_searches"] += 1
                else:
                    self._runtime_state["consecutive_empty_searches"] = 0
                # Pass a text summary of the retrieval result to the guardrail (for identity resolution to detect whether placeholders appear)
                try:
                    retrieval_text = json.dumps(
                        retrieval_payload, ensure_ascii=False, default=str
                    )
                except Exception:
                    retrieval_text = str(retrieval_payload)
                self._guardrails.on_search_result(is_empty, retrieval_text=retrieval_text)

                # Information sufficiency evaluation (Harness Engineering: information-increment evaluation)
                sufficiency = self._sufficiency.assess_after_search(
                    retrieval_payload, content, question.options,
                    round_idx, self.max_rounds,
                    question_text=question.question,
                    keyframe_available=self._current_keyframe_available,
                )
                increment_state = sufficiency.get("increment", {}) or {}
                self._runtime_state["consecutive_low_increment_searches"] = int(
                    increment_state.get("consecutive_low", 0) or 0
                )
                same_focus_stalled = bool(
                    increment_state.get("focus_label_repeated", False)
                    and increment_state.get("is_low_increment", False)
                )
                self._runtime_state["same_focus_stall_count"] = (
                    self._runtime_state.get("same_focus_stall_count", 0) + 1
                    if same_focus_stalled
                    else 0
                )
                self._runtime_state["last_focus_label"] = retrieval_payload.get("focus_label", "") or ""
                sufficiency_hint = sufficiency.get("hint_message", "")

                # When information is saturated, trigger early stopping via the Guardrail
                # Grace-round mechanism: on the first saturation, only send a soft hint, giving the Agent one round to adjust
                # Only hard-stop if it is still saturated after the grace round
                if sufficiency.get("should_stop_searching", False):
                    # should_stop=True means the grace round has passed and a hard stop is truly needed
                    self._guardrails.state.consecutive_empty_searches = (
                        self._guardrails.budget_config.max_consecutive_empty_searches
                    )
                    # Mark the grace round as used up (in coordination with BudgetGuardrail's grace round)
                    self._guardrails.state.grace_round_given = True
                    logger.info(
                        f"[Sufficiency] Still saturated after the grace round -> Guardrail will trigger early stopping next round | "
                        f"increment={sufficiency['increment'].get('increment_ratio', 0):.2f}, "
                        f"consecutive_low={sufficiency['increment'].get('consecutive_low', 0)}"
                    )

                # Context engineering: format the retrieval result
                if is_empty:
                    search_result_text = self._result_formatter.format_empty_result(
                        content,
                        strategy_hint=(
                            "Try a different search strategy: "
                            "use different keywords, try VIDEO: prefix for fine-grained search, "
                            "or try NEIGHBOR: prefix to explore adjacent events."
                        ),
                    )
                else:
                    search_result_text = self._result_formatter.format_search_result(
                        retrieval_payload, content,
                        sufficiency_hint=sufficiency_hint,
                    )

                # Consume the Guardrail grace-round hint (if any)
                # When consecutive empty searches trigger a grace round, BudgetGuardrail sets pending_grace_hint
                grace_hint = self._guardrails.state.pending_grace_hint
                if grace_hint:
                    search_result_text += f"\n\n{grace_hint}"
                    self._guardrails.state.pending_grace_hint = ""  # clear after consuming
            else:
                search_result_text = (
                    "Searched knowledge: (No valid query provided. "
                    "Please provide a specific search query.)"
                )

            # P2 staged injection: append the queued skill hints to the end of the user message
            deferred = self._consume_pending_hints()
            if deferred:
                search_result_text = (search_result_text or "") + deferred

            # Third layer: if KEYFRAME retrieval was triggered this round, insert a semantic placeholder in the pure text (without base64),
            # and attach the actual path list to self._keyframe_paths_by_msg_idx for dynamic image assembly before _generate;
            # what trajectory persistence sees is still the text summary "viewed N KFs from clips [...]".
            if self._pending_keyframe_paths:
                kf_summary = self._format_keyframe_marker(self._pending_keyframe_paths)
                search_result_text = (search_result_text or "") + kf_summary

            conversations.append({"role": "user", "content": search_result_text})

            if self._pending_keyframe_paths:
                # Associate with the index of the user message just appended
                msg_idx = len(conversations) - 1
                self._keyframe_paths_by_msg_idx[msg_idx] = list(self._pending_keyframe_paths)
                self._pending_keyframe_paths = []

        # Should never reach here
        return AgentAnswer(
            content="", confidence=0.0, is_final=True,
            num_rounds=self.max_rounds, conversations=conversations,
        )

    def _execute_search(self, content: str, memory: MemoryStrategy,
                        question: TemporalQuestion,
                        focus_label: Optional[str]) -> Dict[str, Any]:
        """Execute retrieval and return the retrieval payload."""
        c = content.strip()
        before_clip = question.before_clip

        if c.upper().startswith("NEIGHBOR:"):
            new_focus = c.split(":", 1)[1].strip()
            result = memory.retrieve(
                query=c, before_clip=before_clip,
                mode="neighbor", focus_label=new_focus,
            )
            payload = result.event_info or {}
            payload["focus_label"] = new_focus
            return payload

        elif c.upper().startswith("VIDEO:"):
            q = c.split(":", 1)[1].strip()
            if not focus_label:
                # First do one event_first to obtain a focus
                result = memory.retrieve(query=q, before_clip=before_clip, mode="event_first")
                if result.event_info and result.event_info.get("focus_label"):
                    focus_label = result.event_info["focus_label"]

            result = memory.retrieve(
                query=q, before_clip=before_clip,
                mode="video_drilldown", focus_label=focus_label,
            )
            payload = result.event_info or {}
            payload["videograph_hits"] = result.memories or {}
            payload["focus_label"] = focus_label
            return payload

        elif c.upper().startswith("KEYFRAME:") and self._visual_layer_enabled:
            q = c.split(":", 1)[1].strip()
            if not self._current_keyframe_available:
                # When the current video has no keyframe directory/images, do not enter keyframe_inspect, to avoid wasting a round on an empty search.
                result = memory.retrieve(query=q or c, before_clip=before_clip, mode="event_first")
                payload = result.event_info or {}
                return payload
            # Third layer: explicit keyframe request, soft fallback (when there is no focus, the memory layer does best_match internally)
            result = memory.retrieve(
                query=q, before_clip=before_clip,
                mode="keyframe_inspect", focus_label=focus_label,
                question_text=question.question,
            )
            payload = result.event_info or {}
            # Update focus: write back either the soft-fallback focus or the existing focus, for the upper layer to sync
            if payload.get("focus_label"):
                pass  # already set
            elif focus_label:
                payload["focus_label"] = focus_label
            # Store the path list in the agent state: the next user message will flush it in multimodal form
            kf_paths = []
            if result.raw_payload:
                kf_paths = list(result.raw_payload.get("keyframe_paths", []) or [])
            if kf_paths:
                self._pending_keyframe_paths = kf_paths
            return payload

        else:
            # When the visual layer is off, even if the model mistakenly emits KEYFRAME:, treat it as a plain query via event_first (compatible with the old behavior)
            if c.upper().startswith("KEYFRAME:"):
                c = c.split(":", 1)[1].strip() or c
            result = memory.retrieve(query=c, before_clip=before_clip, mode="event_first")
            payload = result.event_info or {}
            return payload

    def inject_instructions(self, instructions: str):
        """Inject extra instructions into the system prompt."""
        self._extra_instructions = instructions

    def inject_deferred_hint(self, hint: str) -> None:
        """P2 staged injection: queue a mid-turn hint.

        Difference from inject_instructions:
          - inject_instructions: affects the system prompt of the entire request, injected at the cold-start stage.
          - inject_deferred_hint: appended only before the next user message, reactively responding to runtime state
            (such as consecutive empty searches, high confidence nearing the end, etc.).
        """
        if hint and isinstance(hint, str):
            self._pending_hints.append(hint.strip())

    def get_runtime_state(self) -> dict:
        """Return the agent's latest runtime state, for the orchestrator to select a staged skill."""
        return dict(self._runtime_state)

    def _consume_pending_hints(self) -> str:
        """Take the hints pending flush and join them into a piece of text (appended after the next user message)."""
        if not self._pending_hints:
            return ""
        hints = self._pending_hints
        self._pending_hints = []
        return "\n\n[Skill hint]\n" + "\n".join(f"- {h}" for h in hints)

    def set_max_rounds(self, max_rounds: int):
        self.max_rounds = max_rounds
