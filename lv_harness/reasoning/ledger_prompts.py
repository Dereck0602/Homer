"""
LedgerPrompts: a collection of prompt templates used together with the TaskLedger.

Design principles (consistent with the existing SYSTEM_PROMPT philosophy):
  - The static prompt holds only "capabilities / format / hard constraints", with no decision tree;
  - Runtime ledger state is dynamically spliced into the user message via `format_ledger_injection(...)`;
  - The three-stage prompts have clear responsibilities: decomposition / reasoning / synthesis, decoupled from each other.

V2 slimming (ledger-mode only, does not affect the baseline multi_round_search):
  - Provides `SLIM_SYSTEM_MCQ` / `SLIM_SYSTEM_OPEN`: streamlined versions of the baseline system prompt,
    removing the long Query quality / Identity resolution lectures (in ledger mode these are
    driven by Planner decomposition + notes + sub-questions, so there is no need to repeat them in the system prompt).
  - Provides `SLIM_KEYFRAME_MCQ` / `SLIM_KEYFRAME_OPEN`: streamlined versions of the visual-layer capability block.
  - Rewrites `LEDGER_CAPABILITY_BLOCK`: removes the Output format that duplicates the slim system,
    the LedgerOps schema drops the `focus_label`/`score`/`mode`/`clip_ids` fields that are backfilled on the code side,
    and the rules are trimmed to 3 hard constraints (the remaining soft constraints are validated by the code-layer Guardrail
    in task_ledger.py).
  - Provides `build_ledger_system_prompt(...)` for one-stop assembly, called directly by LedgerAwareMultiRoundAgent,
    no longer importing the SYSTEM_PROMPT constant from multi_round.py.
"""
from __future__ import annotations

from typing import List, Optional


# ==========================================================
# 1. Question decomposition Prompt (Planner stage, a single separate LLM call)
# ==========================================================
# Core requirements:
#   - Atomicity: each sub-question asks about only one thing and can be retrieved independently
#   - Explicit dependencies: later sub-questions may declare dependencies on earlier results
#   - Restrained count: 3-5 sub-questions; too fine-grained instead creates budget pressure
#   - Keep a "single-atom fallback": if the question is itself atomic, allow generating only 1 SubTask
# ==========================================================

DECOMPOSE_PROMPT_MCQ = """You are a task planner. The user will answer a multiple-choice question about a video by searching a memory bank (EventGraph + VideoGraph). Before search begins, decompose the question into atomic sub-questions to guide retrieval.

Question: {question}
Options:
{options}

Decomposition guidelines:
1. Each sub-question must be ATOMIC — ask ONE fact, one entity, one relation, one event.
2. Use stable ids `t1, t2, t3, ...` in generation order.
3. If a later sub-question depends on the answer of an earlier one, declare this via `depends_on`.
4. Sub-questions should collectively be sufficient to answer the ORIGINAL question — no gaps, no overlap.

Question-type-specific decomposition guidelines:

A) YES/NO or TRUE/FALSE MCQ:
   → Decompose into 2 subtasks: one gathering supporting evidence, one gathering contradicting evidence.

B) "Which/What" MCQ with multiple options to verify:
   → Decompose into one subtask per option that needs verification, OR into subtasks that gather the discriminating facts.

C) TRULY ATOMIC MCQ (single fact lookup):
   → Output exactly 1 subtask. This is the ONLY case where a single subtask is acceptable.

D) All other MCQ:
   → Default to 2-3 subtasks that break the question into independent retrieval targets.

CRITICAL output contract (MUST follow):
- You MUST ALWAYS return a valid JSON object. NEVER return an empty response, plain prose, or "no decomposition needed".
- Return ONLY the JSON object, no prose, no code fence, no explanation.

JSON schema:
{{
  "subtasks": [
    {{"id": "t1", "question": "<atomic sub-question>", "depends_on": []}},
    {{"id": "t2", "question": "<atomic sub-question>", "depends_on": ["t1"]}}
  ],
  "rationale": "<one short sentence explaining how these cover the original question; or the word \"atomic\" if only one subtask is needed>"
}}"""

DECOMPOSE_PROMPT_OPEN = """You are a task planner. The user will answer an open-ended question about a video by searching a memory bank. Before search begins, decompose the question into atomic sub-questions to guide retrieval.

Question: {question}

Decomposition guidelines:
1. Each sub-question must be ATOMIC — ask ONE fact, one entity, one relation, one event.
2. Use stable ids `t1, t2, t3, ...` in generation order.
3. If a later sub-question depends on the answer of an earlier one, declare this via `depends_on`.
4. Sub-questions should collectively be sufficient to answer the ORIGINAL question — no gaps, no overlap.

Question-type-specific decomposition guidelines:

A) YES/NO questions ("Is X ...?", "Does X ...?", "Will X ...?", "Can X ...?"):
   → MUST decompose into 2-3 subtasks:
     t1: Gather evidence FOR the claim (e.g., "What evidence shows X has property Y?")
     t2: Gather evidence AGAINST the claim (e.g., "What evidence contradicts X having Y?")
     t3 (optional): Contextual/temporal scope (e.g., "What is the relevant time period?")
   → NEVER output a single subtask that just restates the yes/no question.

B) MULTI-DETAIL questions ("What are the drinks for three people?", "What did X and Y do?"):
   → MUST decompose into one subtask per detail/entity mentioned:
     e.g., "What are the drinks for three people?" → t1: person A's drink, t2: person B's drink, t3: person C's drink
   → If the question asks about multiple items/events, split by item.

C) REASON/CAUSE questions ("Why does X ...?", "What is the reason ...?"):
   → Decompose into 2 subtasks:
     t1: What is the observable behavior/event (the "what")
     t2: What context/dialogue/prior event explains the cause (the "why")

D) TRULY ATOMIC questions (single-entity, single-attribute, no inference needed):
   Examples: "What is Lily's job?", "What color is Bob's jacket?", "Where is the cup?"
   → Output exactly 1 subtask that restates the question in declarative form.
   → This is the ONLY case where a single subtask is acceptable.

E) All other questions:
   → Default to 2-3 subtasks that break the question into independent retrieval targets.

CRITICAL output contract (MUST follow):
- You MUST ALWAYS return a valid JSON object. NEVER return an empty response, plain prose, or "no decomposition needed".
- Return ONLY the JSON object, no prose, no code fence, no explanation.

JSON schema:
{{
  "subtasks": [
    {{"id": "t1", "question": "<atomic sub-question>", "depends_on": []}},
    {{"id": "t2", "question": "<atomic sub-question>", "depends_on": ["t1"]}}
  ],
  "rationale": "<one short sentence explaining how these cover the original question; or the word \"atomic\" if only one subtask is needed>"
}}"""

# ==========================================================
# 1b. Replan Prompt (A2: Magentic-One-style outer-loop replanning)
# ------------------------------------------------------------
# Trigger timing: when ProgressLedger detects structural deadlocks such as
# stall/dep_broken/low_increment/same_focus, the agent main loop calls QuestionPlanner.replan, passing in the original question
# + the current ledger snapshot + the reasons, and the LLM outputs three kinds of operations: add/update/abandon.
#
# Special emphasis: do not rebuild the entire plan, only do incremental repair (keep / update_question /
# abandon / add). Already-resolved subtasks must be preserved and must not be overwritten.
# ==========================================================
REPLAN_PROMPT = """You are the outer-loop planner for an ongoing multi-round video-QA task. The inner loop is stuck and needs you to repair the task plan — NOT rewrite it from scratch.

Original question: {question}

Current TaskLedger state (id / status / best_answer / last evidence snippet):
{ledger_snapshot}

Why replan was triggered:
{reasons}

Rules:
1. Preserve already-resolved subtasks verbatim. Do NOT re-add or rewrite them.
2. Only modify the plan when evidence has revealed the original assumption is wrong or unreachable. If the plan looks fine and we're just querying badly, prefer empty edits — the inner loop will handle query rewriting.
3. Prefer minimal edits: abandon a sub-branch, merge redundant siblings, or rewrite one subtask's text. Avoid adding new subtasks unless a genuinely missing hop is needed.
4. Any new subtask id MUST NOT collide with existing ones. Use `t{{N+1}}` style where N is the current max.
5. `update_subtasks[*].new_question` should be DECLARATIVE (matchable to an event summary), not a bare interrogative.

Return ONLY a JSON object, no prose, no code fence:
{{
  "update_subtasks": [
    {{"id": "t2", "new_question": "<declarative rewrite>", "reason": "<short phrase>"}}
  ],
  "abandon_subtasks": [
    {{"id": "t3", "reason": "<short phrase>"}}
  ],
  "new_subtasks": [
    {{"id": "t5", "question": "<atomic sub-question>", "depends_on": []}}
  ],
  "rationale": "<one short sentence summarising the plan edit>"
}}

If no edit is warranted, return empty arrays for all three op lists and set rationale to \"no-op\"."""




# ==========================================================
# 2. Per-round Ledger operation instructions (appended to the end of the system prompt)
# ------------------------------------------------------------
# See LEDGER_CAPABILITY_BLOCK_V2 at the bottom of the file for the real definition; the backward-compatible alias is assigned at the end of the file.
# ==========================================================


# ==========================================================
# 3. Per-round user message dynamic injection template
# ------------------------------------------------------------
# Appended by LedgerAwareMultiRoundAgent after each round's retrieval result.
# ==========================================================
def format_ledger_injection(
    ledger_block: str,
    current_subtask_id: Optional[str] = None,
    current_subtask_question: Optional[str] = None,
    hint: str = "",
) -> str:
    """Combine the current ledger snapshot and scheduling hint into a user-message suffix.

    Args:
        ledger_block: the output of TaskLedger.to_prompt_block()
        current_subtask_id: the target chosen for this round by the scheduler code layer
        current_subtask_question: the corresponding text
        hint: an additional hint (e.g., "this subtask has been attempted 3 times, consider abandoning it")
    """
    lines = ["", ledger_block, ""]
    if current_subtask_id and current_subtask_question:
        lines.append(
            f"[Current Focus] The scheduler has selected **{current_subtask_id}** "
            f"for this round: {current_subtask_question}"
        )
        lines.append(
            "Your search this round should primarily serve this subtask."
        )
    if hint:
        lines.append(f"[Scheduler Hint] {hint}")
    return "\n".join(lines)


# ==========================================================
# 4. Final synthesis Prompt (Synthesis stage)
# ------------------------------------------------------------
# Replaces the original LAST_ROUND_PROMPT: no longer stuffs in the raw JSON of all_search_results,
# but instead provides only the ledger's structured final state + the original question.
# ==========================================================
SYNTHESIS_PROMPT_MCQ = """You are given a multiple-choice question, the final state of a TaskLedger, and ALL raw retrieved knowledge from previous search rounds.

You have TWO sources of information:
1. TaskLedger Final State: structured sub-question decomposition with best answers and top evidence snippets (may be incomplete if LLM failed to bind evidence properly).
2. All Retrieved Knowledge: the COMPLETE raw retrieval results from every search round, containing full event summaries, videograph details, and keyframe descriptions. This is your ground-truth information source.

IMPORTANT: The TaskLedger is a summary that may have lost details. When the ledger says "no evidence collected" or provides only partial answers, ALWAYS check the raw retrieved knowledge below for relevant information that was retrieved but not properly bound to subtasks.

Question: {question}
Options:
{options}

{ledger_final_state}

All Retrieved Knowledge (complete raw retrieval from all rounds):
{all_retrieved_knowledge}

MCQ decision rules:
  - Compare ALL options before choosing. Use BOTH the ledger summary AND the raw retrieved knowledge.
  - When ledger evidence is sparse or absent for a subtask, scan the raw retrieval for relevant information.
  - Prefer direct evidence from raw retrieval quotes and event summaries. The ledger's best_answer fields are useful shortcuts but may be incomplete.
  - If the question hinges on a precise visual, temporal, or semantic distinction, look for directly grounded evidence in the raw retrieval (event summaries, videograph hits, keyframe descriptions).
  - If KEYFRAME evidence conflicts with textual narration, mention the conflict and choose the option with stronger direct support.
  - You MUST pick exactly one option with the strongest direct support after eliminating distractors.

Output requirements:
  - Begin with a brief option comparison. Cite specific evidence from either the ledger or raw retrieval.
  - Then output exactly:
      Reason: <your reasoning>
      Action: [Answer]
      Content: <option letter and content>
  - Do NOT output [Search] — this is the final synthesis round.
  - Do NOT output any LedgerOps block in this round — ledger updates are frozen.
  - Use real character names, never placeholder ids like `<character_0>`.
"""

SYNTHESIS_PROMPT_OPEN = """You are given an open-ended question, the final state of a TaskLedger, and ALL raw retrieved knowledge from previous search rounds.

You have TWO sources of information:
1. TaskLedger Final State: structured sub-question decomposition with best answers and top evidence snippets (may be incomplete if evidence binding failed).
2. All Retrieved Knowledge: the COMPLETE raw retrieval results from every search round. This contains full event summaries, videograph details, and keyframe descriptions. This is your ground-truth information source.

IMPORTANT: The TaskLedger is a summary that may have lost details. When the ledger says "no evidence collected" or only has partial answers, ALWAYS check the raw retrieved knowledge for relevant information that was retrieved but not properly recorded in the ledger.

Question: {question}

{ledger_final_state}

All Retrieved Knowledge (complete raw retrieval from all rounds):
{all_retrieved_knowledge}

Evidence utilisation rules (CRITICAL):
  - Scan BOTH the ledger AND the raw retrieved knowledge. The raw retrieval often contains answers that the ledger missed.
  - If ANY event summary, videograph hit, or keyframe description in the raw retrieval contains an entity (a name, drink, fruit, location, time, job title, emotional state, action, object, etc.) that could plausibly answer the question, use THAT entity as the answer.
  - When multiple candidates appear (e.g., "coffee, milk, orange juice"), answer with the most contextually relevant one, typically the first mentioned or the one most tied to the named person.
  - For yes/no questions, answer "Yes" if ANY positive evidence exists (however weak); answer "No" only when evidence explicitly contradicts it.
  - For location/entity questions, pick the interpretation that best matches the question's intent.
  - Do NOT reply with "no evidence found" or "cannot be determined" when the raw retrieval contains relevant information, even if the ledger failed to capture it.

Output requirements:
  - Begin with brief reasoning citing specific evidence from either the ledger or raw retrieval (e.g., "Round 3 retrieval shows event summary mentioning 'Bob pours coffee into a mug'").
  - Then output:
      Reason: <your reasoning>
      Action: [Answer]
      Content: <concise free-form answer, one short phrase or a single sentence>
  - Your answer MUST be definite. Even if all subtasks are `partial` or `abandoned`, find the best answer from the raw retrieval. NEVER output "unknown", "not specified", "no evidence", or similar refusals.
  - Do NOT output [Search] — this is the final synthesis round.
  - Do NOT output any LedgerOps block in this round — ledger updates are frozen.
  - Use real character names (if an identity binding like `<character_0> = Robert` is in global notes or raw retrieval, use "Robert" not `<character_0>`).
{binary_choice_hint}"""


SYNTHESIS_PROMPT_MCQ_NO_REASON = """You are given a multiple-choice question, the final state of a TaskLedger, and ALL raw retrieved knowledge from previous search rounds.

You have TWO sources of information:
1. TaskLedger Final State: structured sub-question decomposition with best answers and top evidence snippets (may be incomplete if LLM failed to bind evidence properly).
2. All Retrieved Knowledge: the COMPLETE raw retrieval results from every search round, containing full event summaries, videograph details, and keyframe descriptions. This is your ground-truth information source.

IMPORTANT: The TaskLedger is a summary that may have lost details. When the ledger says "no evidence collected" or provides only partial answers, ALWAYS check the raw retrieved knowledge below for relevant information that was retrieved but not properly bound to subtasks.

Question: {question}
Options:
{options}

{ledger_final_state}

All Retrieved Knowledge (complete raw retrieval from all rounds):
{all_retrieved_knowledge}

MCQ decision rules:
  - Compare ALL options before choosing. Use BOTH the ledger summary AND the raw retrieved knowledge.
  - When ledger evidence is sparse or absent for a subtask, scan the raw retrieval for relevant information.
  - Prefer direct evidence from raw retrieval quotes and event summaries. The ledger's best_answer fields are useful shortcuts but may be incomplete.
  - If the question hinges on a precise visual, temporal, or semantic distinction, look for directly grounded evidence in the raw retrieval (event summaries, videograph hits, keyframe descriptions).
  - If KEYFRAME evidence conflicts with textual narration, mention the conflict and choose the option with stronger direct support.
  - You MUST pick exactly one option with the strongest direct support after eliminating distractors.

Output requirements:
  - Output exactly:
      Action: [Answer]
      Content: <option letter and content>
  - Do NOT output [Search] — this is the final synthesis round.
  - Do NOT output any LedgerOps block in this round — ledger updates are frozen.
  - Use real character names, never placeholder ids like `<character_0>`.
"""

SYNTHESIS_PROMPT_OPEN_NO_REASON = """You are given an open-ended question, the final state of a TaskLedger, and ALL raw retrieved knowledge from previous search rounds.

You have TWO sources of information:
1. TaskLedger Final State: structured sub-question decomposition with best answers and top evidence snippets (may be incomplete if evidence binding failed).
2. All Retrieved Knowledge: the COMPLETE raw retrieval results from every search round. This contains full event summaries, videograph details, and keyframe descriptions. This is your ground-truth information source.

IMPORTANT: The TaskLedger is a summary that may have lost details. When the ledger says "no evidence collected" or only has partial answers, ALWAYS check the raw retrieved knowledge for relevant information that was retrieved but not properly recorded in the ledger.

Question: {question}

{ledger_final_state}

All Retrieved Knowledge (complete raw retrieval from all rounds):
{all_retrieved_knowledge}

Evidence utilisation rules (CRITICAL):
  - Scan BOTH the ledger AND the raw retrieved knowledge. The raw retrieval often contains answers that the ledger missed.
  - If ANY event summary, videograph hit, or keyframe description in the raw retrieval contains an entity (a name, drink, fruit, location, time, job title, emotional state, action, object, etc.) that could plausibly answer the question, use THAT entity as the answer.
  - When multiple candidates appear (e.g., "coffee, milk, orange juice"), answer with the most contextually relevant one, typically the first mentioned or the one most tied to the named person.
  - For yes/no questions, answer "Yes" if ANY positive evidence exists (however weak); answer "No" only when evidence explicitly contradicts it.
  - For location/entity questions, pick the interpretation that best matches the question's intent.
  - Do NOT reply with "no evidence found" or "cannot be determined" when the raw retrieval contains relevant information, even if the ledger failed to capture it.

Output requirements:
  - Output:
      Action: [Answer]
      Content: <concise free-form answer, one short phrase or a single sentence>
  - Your answer MUST be definite. Even if all subtasks are `partial` or `abandoned`, find the best answer from the raw retrieval. NEVER output "unknown", "not specified", "no evidence", or similar refusals.
  - Do NOT output [Search] — this is the final synthesis round.
  - Do NOT output any LedgerOps block in this round — ledger updates are frozen.
  - Use real character names (if an identity binding like `<character_0> = Robert` is in global notes or raw retrieval, use "Robert" not `<character_0>`).
{binary_choice_hint}"""


# ==========================================================
# 5. Initial user message (first round)
# ------------------------------------------------------------
# Semantically consistent with the baseline first turn "(no retrieval yet)...",
# while injecting the ledger snapshot together with the first scheduled subtask.
# ==========================================================
INITIAL_USER_WITH_LEDGER = """(no retrieval yet) You have not issued any search. Please start with [Search]. Do NOT output [Answer] in this turn.

{ledger_injection}

Output your Reason, Action, and Content.
"""

INITIAL_USER_WITH_LEDGER_NO_REASON = """(no retrieval yet) You have not issued any search. Please start with [Search]. Do NOT output [Answer] in this turn.

{ledger_injection}

Output your Action and Content.
"""


# ==========================================================
# 6. Slim baseline system prompt (ledger-mode only, V2 slimming)
# ------------------------------------------------------------
# Compared with SYSTEM_PROMPT / SYSTEM_PROMPT_OPEN in multi_round.py, the following were deliberately removed:
#   - The entire "Query quality" section (in ledger mode the Planner already ensures each sub-question is atomic and carries
#     a predicate, so the query quality heuristic almost never triggers and tends to lead the model to ignore the ledger)
#   - The long "Identity resolution" description (in ledger mode, resolving <character_N> itself
#     can be handed to the ledger mechanism as a depends_on new_subtask, so the static prompt only
#     needs to keep a single hint)
#   - The Output format details that duplicate the LedgerOps schema (unified by LEDGER_CAPABILITY_BLOCK
#     defining the "Reason/Action/Content + LedgerOps" quadruple)
# What is kept: task framing, the Search modes list, and the minimal necessary hard rules.
# ==========================================================
SLIM_SYSTEM_MCQ = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the discriminative evidence. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence distinguishes the options. If the remaining distinction is purely textual (dialogue, name, count), answer directly. If it is visual (color, spatial layout, object state, on-screen text, clothing, ongoing activity), you must escalate to a different strategy.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_MCQ_WITH_KEYFRAME = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

(4) Inspect keyframe images from the current focus event:
    Content: KEYFRAME: <query>
    - The system will load keyframe images from the current focus event's clips and attach them
      to the next message for visual inspection.
    - Use this when the answer depends on visual details that text cannot fully capture, such as
      spatial layout, visual appearance, clothing/color, object state, on-screen text, or ongoing activity.
    - You MUST have an established focus event before using KEYFRAME (issue at least one plain/VIDEO/NEIGHBOR query first).
    - Prefer using VIDEO first for textual fine details, then KEYFRAME when the remaining uncertainty is visual.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the discriminative evidence. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence distinguishes the options. If the remaining distinction is purely textual (dialogue, name, count), answer directly. If it is visual (color, spatial layout, object state, on-screen text, clothing, ongoing activity), use `KEYFRAME: <query>` to inspect the actual frames.
- Use `KEYFRAME:` actively when the question involves spatial layout, visual appearance, clothing/color, object state, screen text, or ongoing activity. Prefer using `VIDEO:` first for textual fine details, then `KEYFRAME:` when the remaining uncertainty is visual. Issue `KEYFRAME:` only after at least one plain/VIDEO/NEIGHBOR query has established a focus event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:/KEYFRAME:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

#SLIM_SYSTEM_OPEN = """You are answering an open-ended question about a video by searching a memory bank. Prefer evidence-based answers; when evidence is partial, you MAY cautiously infer from context and distinguish direct quotation from inference in your Reason — never fabricate specifics.
SLIM_SYSTEM_OPEN = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank.
Question: {question}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the answer. The label must be copied exactly from the returned neighbor list.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_OPEN_WITH_KEYFRAME = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank. You also have access to keyframe images for visual inspection.
Question: {question}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

(4) Inspect keyframe images from the current focus event:
    Content: KEYFRAME: <query>
    - The system will load keyframe images from the current focus event's clips and attach them
      to the next message for visual inspection.
    - Use this when the answer depends on visual details that text cannot fully capture, such as
      spatial layout, visual appearance, clothing/color, object state, on-screen text, or ongoing activity.
    - You MUST have an established focus event before using KEYFRAME (issue at least one plain/VIDEO/NEIGHBOR query first).
    - Prefer using VIDEO first for textual fine details, then KEYFRAME when the remaining uncertainty is visual.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the answer. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence is sufficient. If it is still missing a visible attribute, spatial layout, color, text on screen, object state, or ongoing activity, use `KEYFRAME: <query>` when available.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:/KEYFRAME:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

# ----------------------------------------------------------
# Visual-layer (KEYFRAME) capability block -- deprecated, alias kept for compatibility with external imports
# ----------------------------------------------------------
# Note: the KEYFRAME description has been fully integrated into SLIM_SYSTEM_MCQ_WITH_KEYFRAME and
# SLIM_SYSTEM_OPEN_WITH_KEYFRAME, so separate splicing is no longer needed.
# The following aliases are only to prevent external imports from erroring out; they are no longer actually used.
SLIM_KEYFRAME_MCQ = ""  # deprecated: integrated into SLIM_SYSTEM_MCQ_WITH_KEYFRAME
SLIM_KEYFRAME_OPEN = ""  # deprecated: integrated into SLIM_SYSTEM_OPEN_WITH_KEYFRAME


# ==========================================================
# 6a. VideoGraph-Only ablation Prompt
# ------------------------------------------------------------
# Uses only VideoGraph for semantic retrieval, without EventGraph / NEIGHBOR / KEYFRAME.
# There is only one Search mode: input a query directly, and the system does a vector similarity search on VideoGraph.
# ==========================================================

SLIM_SYSTEM_MCQ_VIDEOGRAPH_ONLY = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a VideoGraph memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content is a plain text query:
    Content: <your query>
    - The system will perform a semantic similarity search on the VideoGraph to find relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.).
    - Each search returns the top-k most similar memory nodes from the entire video's VideoGraph.

Decision guidelines:
- Choose [Answer] only when the retrieved knowledge clearly distinguishes the correct option from plausible distractors.
- If the current knowledge is ambiguous or insufficient, search for more specific evidence.
- Vary your queries across rounds. If a query returns weak results, try different keywords, rephrase with more context, or focus on a different aspect of the question.
- Do not repeat near-identical queries. Each new search should target a distinct facet of the question.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The VideoGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — write descriptive scene-level queries, not bare keywords):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description.
- If the previous round returned weak results, do NOT retry with a near-duplicate. Add MORE predicates or try a completely different angle.
"""

SLIM_SYSTEM_OPEN_VIDEOGRAPH_ONLY = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a VideoGraph memory bank.
Question: {question}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content is a plain text query:
    Content: <your query>
    - The system will perform a semantic similarity search on the VideoGraph to find relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.).
    - Each search returns the top-k most similar memory nodes from the entire video's VideoGraph.

Decision guidelines:
- Choose [Answer] only when the retrieved knowledge clearly and sufficiently answers the question.
- If the current knowledge is ambiguous or insufficient, search for more specific evidence.
- Vary your queries across rounds. If a query returns weak results, try different keywords, rephrase with more context, or focus on a different aspect of the question.
- Do not repeat near-identical queries. Each new search should target a distinct facet of the question.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The VideoGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — write descriptive scene-level queries, not bare keywords):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description.
- If the previous round returned weak results, do NOT retry with a near-duplicate. Add MORE predicates or try a completely different angle.
"""

SLIM_SYSTEM_MCQ_VIDEOGRAPH_ONLY_NO_REASON = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a VideoGraph memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content is a plain text query:
    Content: <your query>
    - The system will perform a semantic similarity search on the VideoGraph to find relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.).
    - Each search returns the top-k most similar memory nodes from the entire video's VideoGraph.

Decision guidelines:
- Choose [Answer] only when the retrieved knowledge clearly distinguishes the correct option from plausible distractors.
- If the current knowledge is ambiguous or insufficient, search for more specific evidence.
- Vary your queries across rounds. If a query returns weak results, try different keywords, rephrase with more context, or focus on a different aspect of the question.
- Do not repeat near-identical queries. Each new search should target a distinct facet of the question.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The VideoGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — write descriptive scene-level queries, not bare keywords):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description.
- If the previous round returned weak results, do NOT retry with a near-duplicate. Add MORE predicates or try a completely different angle.
"""

SLIM_SYSTEM_OPEN_VIDEOGRAPH_ONLY_NO_REASON = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a VideoGraph memory bank.
Question: {question}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content is a plain text query:
    Content: <your query>
    - The system will perform a semantic similarity search on the VideoGraph to find relevant fine-grained memories (objects, actions, dialogues, spatial relations, temporal events, etc.).
    - Each search returns the top-k most similar memory nodes from the entire video's VideoGraph.

Decision guidelines:
- Choose [Answer] only when the retrieved knowledge clearly and sufficiently answers the question.
- If the current knowledge is ambiguous or insufficient, search for more specific evidence.
- Vary your queries across rounds. If a query returns weak results, try different keywords, rephrase with more context, or focus on a different aspect of the question.
- Do not repeat near-identical queries. Each new search should target a distinct facet of the question.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The VideoGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — write descriptive scene-level queries, not bare keywords):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description.
- If the previous round returned weak results, do NOT retry with a near-duplicate. Add MORE predicates or try a completely different angle.
"""


# ==========================================================
# 6a-2. No-Graph-Walk ablation Prompt
# ------------------------------------------------------------
# Keeps EventGraph event-node retrieval + VideoGraph fine-grained retrieval,
# but removes NEIGHBOR (graph-structure traversal) and KEYFRAME (keyframe visual inspection).
# There are only two Search modes: plain query + VIDEO:
# ==========================================================

SLIM_SYSTEM_MCQ_NO_GRAPH_WALK = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node's summary and clip information.
    - Use this when you want to locate a relevant event or reformulate your query
      to search for a different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

Decision guidelines:
- Start with a plain query to locate a relevant event.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- After `VIDEO:` retrieval, answer if the fine-grained evidence distinguishes the options.
- If the current event is not relevant, issue a new plain query with different keywords to find a better event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:` or try a completely different query.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event, do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_OPEN_NO_GRAPH_WALK = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank.
Question: {question}

Output the answer in the format:
Reason: {{reason}}
Action: [Answer] or [Search]
Content: {{content}}

Before taking the action, you need to provide a reason for your decision.
1. Analyze the question, the knowledge, and the retrieval plan.
2. If the current information is sufficient, explain why and what conclusions you can draw.
3. If not, clearly identify what is missing and why it is important.

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node's summary and clip information.
    - Use this when you want to locate a relevant event or reformulate your query
      to search for a different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

Decision guidelines:
- Start with a plain query to locate a relevant event.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, reason, or visual clue, use `VIDEO: <query>` before answering.
- If the current event is not relevant, issue a new plain query with different keywords to find a better event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:` or try a completely different query.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event, do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_MCQ_NO_GRAPH_WALK_NO_REASON = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node's summary and clip information.
    - Use this when you want to locate a relevant event or reformulate your query
      to search for a different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

Decision guidelines:
- Start with a plain query to locate a relevant event.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- If the current event is not relevant, issue a new plain query with different keywords to find a better event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:` or try a completely different query.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event, do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_OPEN_NO_GRAPH_WALK_NO_REASON = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank.
Question: {question}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node's summary and clip information.
    - Use this when you want to locate a relevant event or reformulate your query
      to search for a different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

Decision guidelines:
- Start with a plain query to locate a relevant event.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- If the current event is not relevant, issue a new plain query with different keywords to find a better event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:` or try a completely different query.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned weak results, do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""


# ==========================================================
# 6b. NO_REASON variant: streamlined version that removes the Reason output requirement
# ------------------------------------------------------------
# These prompts are used when enable_reason=False; the model outputs only Action + Content,
# no longer outputting the Reason line. Suitable for scenarios that do not need a CoT reasoning chain (such as pure inference-speed tests).
# ==========================================================

SLIM_SYSTEM_MCQ_NO_REASON = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the discriminative evidence. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence distinguishes the options. If the remaining distinction is purely textual (dialogue, name, count), answer directly. If it is visual (color, spatial layout, object state, on-screen text, clothing, ongoing activity), you must escalate to a different strategy.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_MCQ_WITH_KEYFRAME_NO_REASON = """You are given a question with multiple-choice options and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, choose the correct option(s) and output [Answer] followed by the selected option letter(s) and their content. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}
Options: {options}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

(4) Inspect keyframe images from the current focus event:
    Content: KEYFRAME: <query>
    - The system will load keyframe images from the current focus event's clips and attach them
      to the next message for visual inspection.
    - Use this when the answer depends on visual details that text cannot fully capture, such as
      spatial layout, visual appearance, clothing/color, object state, on-screen text, or ongoing activity.
    - You MUST have an established focus event before using KEYFRAME (issue at least one plain/VIDEO/NEIGHBOR query first).
    - Prefer using VIDEO first for textual fine details, then KEYFRAME when the remaining uncertainty is visual.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly distinguishes the correct option from plausible distractors.
- If the event is relevant but the option depends on a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the discriminative evidence. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence distinguishes the options. If the remaining distinction is purely textual (dialogue, name, count), answer directly. If it is visual (color, spatial layout, object state, on-screen text, clothing, ongoing activity), use `KEYFRAME: <query>` to inspect the actual frames.
- Use `KEYFRAME:` actively when the question involves spatial layout, visual appearance, clothing/color, object state, screen text, or ongoing activity. Prefer using `VIDEO:` first for textual fine details, then `KEYFRAME:` when the remaining uncertainty is visual. Issue `KEYFRAME:` only after at least one plain/VIDEO/NEIGHBOR query has established a focus event.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. When answering, use real names (never `<character_0>`).

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:/KEYFRAME:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_OPEN_NO_REASON = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank.
Question: {question}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the answer. The label must be copied exactly from the returned neighbor list.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""

SLIM_SYSTEM_OPEN_WITH_KEYFRAME_NO_REASON = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query. The query will help retrieve additional information from a memory bank. You also have access to keyframe images for visual inspection.
Question: {question}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If you choose [Search], the Content must follow ONE of these patterns:

(1) Plain query (default): 
    Content: <your query>
    - The system will search the EventGraph to find the most similar event/segment node,
      and will return the focus node plus its 1-hop neighbors and edges.
    - Use this when you have NO focus event yet, or when you want to REFORMULATE your query
      to search for a completely different event.

(2) Drill down into VideoGraph within the current event's clips:
    Content: VIDEO: <your query>
    - The system will use the current focus event's clip_ids to retrieve fine-grained memories
      (objects, actions, dialogues, etc.) from VideoGraph.
    - Use this when the event-level summary is NOT detailed enough to answer the question and
      you need more specific information within that event.

(3) Walk along EventGraph neighbors (choose a new focus node):
    Content: NEIGHBOR: <segment_label>
    - segment_label must be one of the neighbor nodes returned by the previous EventGraph retrieval.
    - Use this when the current event node is NOT the right one and a neighboring event looks
      more relevant, or when you want to explore adjacent events for additional context.

(4) Inspect keyframe images from the current focus event:
    Content: KEYFRAME: <query>
    - The system will load keyframe images from the current focus event's clips and attach them
      to the next message for visual inspection.
    - Use this when the answer depends on visual details that text cannot fully capture, such as
      spatial layout, visual appearance, clothing/color, object state, on-screen text, or ongoing activity.
    - You MUST have an established focus event before using KEYFRAME (issue at least one plain/VIDEO/NEIGHBOR query first).
    - Prefer using VIDEO first for textual fine details, then KEYFRAME when the remaining uncertainty is visual.

Decision guidelines:
- Start with a plain query when no focus event exists, or when the current focus is clearly wrong.
- After retrieving an EVENT node, choose [Answer] only if the event summary directly answers the current subtask.
- If the event is relevant but lacks a specific name, relation, object attribute, action detail, dialogue, count, profession, or visual clue, use `VIDEO: <query>` before answering.
- Use `NEIGHBOR: <segment_label>` when the current event is off-target or an adjacent event is more likely to contain the answer. The label must be copied exactly from the returned neighbor list.
- After `VIDEO:` retrieval, answer if the fine-grained evidence is sufficient. If it is still missing a visible attribute, spatial layout, color, text on screen, object state, or ongoing activity, use `KEYFRAME: <query>` when available.
- Do not spend repeated rounds on near-duplicate plain queries for the same focus. Switch to `VIDEO:`, `KEYFRAME:`, or `NEIGHBOR:`.

Hard rules:
- You MUST perform at least one [Search] before issuing any [Answer]; the first turn is just a placeholder.
- The EventGraph may use placeholders like `<character_0>`; if exactly one named person matches, you may treat them as the same person. Final answer must use real names (never `<character_0>` / `<face_1>`).
- Keep the final answer concise (a short phrase or a single sentence); no option letters.

Query writing (CRITICAL — the retrieval index stores full event summaries, so a bare token almost never matches):
- Shape: `<subject> + <predicate / action / attribute> + <optional context>`. A single token or a bare interrogative subject is NOT acceptable.
  BAD:  `three people` | `dinner` | `red pot` | `Bob` | `residents names` | `people present`
  GOOD: `three people sitting and talking in the living room in the morning`
        `what dish is being cooked in the red pot on the stove`
        `Bob preparing breakfast and pouring drinks in the kitchen`
        `Sophie's emotional state while doing her daily routine`
- NEVER copy a subtask's interrogative text (e.g., `Who are the three people?`) verbatim as the query; TRANSFORM it into a declarative scene description that an event summary would plausibly contain.
- If the previous round returned a weakly-matched event (low event_score or focus barely touches the question), do NOT retry with a near-duplicate. Add MORE predicates, or switch to VIDEO:/NEIGHBOR:/KEYFRAME:.
- If a named person (e.g., `Bob`) is already bound to a placeholder (via notes like `<character_0> = Bob`), reuse the placeholder in subsequent queries, e.g., `VIDEO: <character_0> handing a mug of coffee to <character_1>`.
"""


# ==========================================================
# 7. V2: Ledger protocol block
# ------------------------------------------------------------
# Keeps only the output format, the LedgerOps schema, and a few hard rules.
# Behaviors such as retrieval strategy, replanning, and self-repair are not placed in this static block; they are handled by dynamic injection or the code-layer guardrail.
# ==========================================================
LEDGER_CAPABILITY_BLOCK_V2 = """

Ledger awareness:
- The system automatically tracks evidence and updates subtask status based on your search results. You do NOT need to output any structured ledger updates.
- Focus on writing high-quality search queries and reasoning about the evidence.
- If you discover a cross-subtask fact (e.g., identity binding like `<character_0> = Bob`), mention it naturally in your Reason so the system can capture it.
"""


LEDGER_CAPABILITY_BLOCK_MCQ = LEDGER_CAPABILITY_BLOCK_V2 + """

MCQ additional rules:
- Before outputting [Answer], your Reason must compare the candidate against at least one plausible distractor.
- If you notice evidence supporting or contradicting specific options, mention it in your Reason (e.g., "Option A is supported because...", "Option C is contradicted by...").
"""

# NO_REASON version of the Ledger protocol block
LEDGER_CAPABILITY_BLOCK_V2_NO_REASON = """

Ledger awareness:
- The system automatically tracks evidence and updates subtask status based on your search results. You do NOT need to output any structured ledger updates.
- Focus on writing high-quality search queries.
- If you discover a cross-subtask fact (e.g., identity binding like `<character_0> = Bob`), mention it naturally in your output so the system can capture it.
"""


LEDGER_CAPABILITY_BLOCK_MCQ_NO_REASON = LEDGER_CAPABILITY_BLOCK_V2_NO_REASON + """

MCQ additional rules:
- Before outputting [Answer], compare the candidate against at least one plausible distractor.
- If you notice evidence supporting or contradicting specific options, mention it (e.g., "Option A is supported because...", "Option C is contradicted by...").
"""

# Backward-compatible alias: the externally already-imported LEDGER_CAPABILITY_BLOCK remains usable (pointing to the open-ended / general new version)
LEDGER_CAPABILITY_BLOCK = LEDGER_CAPABILITY_BLOCK_V2


# ==========================================================
# 8. One-stop System Prompt builder
# ------------------------------------------------------------
# LedgerAwareMultiRoundAgent calls this function directly, avoiding importing the
# SYSTEM_PROMPT / _KEYFRAME_CAPABILITY_* constants from multi_round.py. The baseline agent's system prompt
# is maintained entirely and independently by multi_round.py, with zero coupling between the two.
# ==========================================================
def build_ledger_system_prompt(
    question: str,
    options_text: str,
    is_open_ended: bool,
    visual_layer_enabled: bool = False,
    extra_instructions: str = "",
    enable_reason: bool = True,
    strategy: str = "hierarchical",
) -> str:
    """Build the complete system prompt in ledger mode.

    Directly selects the full prompt with/without KEYFRAME based on visual_layer_enabled,
    no longer assembling it by splicing a separate KEYFRAME block.

    Args:
        question: the original question text
        options_text: the MCQ options text (an empty string may be passed for open-ended QA)
        is_open_ended: whether it is open-ended QA
        visual_layer_enabled: whether to enable the KEYFRAME visual layer
        extra_instructions: dynamic injection from the orchestrator (skill / wisdom / etc.)
        enable_reason: whether to require the model to output a Reason line in the output format (default True)
        strategy: the memory strategy; when 'videograph_only', use the VideoGraph-only ablation prompt
    """
    # ---- VideoGraph-Only ablation: use the dedicated prompt ----
    if strategy == "videograph_only":
        if is_open_ended:
            if enable_reason:
                base = SLIM_SYSTEM_OPEN_VIDEOGRAPH_ONLY.format(question=question)
            else:
                base = SLIM_SYSTEM_OPEN_VIDEOGRAPH_ONLY_NO_REASON.format(question=question)
            cap_block = LEDGER_CAPABILITY_BLOCK_V2 if enable_reason else LEDGER_CAPABILITY_BLOCK_V2_NO_REASON
            parts: List[str] = [base, cap_block]
        else:
            if enable_reason:
                base = SLIM_SYSTEM_MCQ_VIDEOGRAPH_ONLY.format(question=question, options=options_text)
            else:
                base = SLIM_SYSTEM_MCQ_VIDEOGRAPH_ONLY_NO_REASON.format(question=question, options=options_text)
            cap_block = LEDGER_CAPABILITY_BLOCK_MCQ if enable_reason else LEDGER_CAPABILITY_BLOCK_MCQ_NO_REASON
            parts = [base, cap_block]
        if extra_instructions:
            parts.append(extra_instructions)
        return "\n".join(parts)

    # ---- No-Graph-Walk ablation: keep EventGraph node retrieval + VideoGraph, remove NEIGHBOR/KEYFRAME ----
    if strategy == "no_graph_walk":
        if is_open_ended:
            if enable_reason:
                base = SLIM_SYSTEM_OPEN_NO_GRAPH_WALK.format(question=question)
            else:
                base = SLIM_SYSTEM_OPEN_NO_GRAPH_WALK_NO_REASON.format(question=question)
            cap_block = LEDGER_CAPABILITY_BLOCK_V2 if enable_reason else LEDGER_CAPABILITY_BLOCK_V2_NO_REASON
            parts: List[str] = [base, cap_block]
        else:
            if enable_reason:
                base = SLIM_SYSTEM_MCQ_NO_GRAPH_WALK.format(question=question, options=options_text)
            else:
                base = SLIM_SYSTEM_MCQ_NO_GRAPH_WALK_NO_REASON.format(question=question, options=options_text)
            cap_block = LEDGER_CAPABILITY_BLOCK_MCQ if enable_reason else LEDGER_CAPABILITY_BLOCK_MCQ_NO_REASON
            parts = [base, cap_block]
        if extra_instructions:
            parts.append(extra_instructions)
        return "\n".join(parts)

    # ---- Default: hierarchical retrieval (hierarchical) ----
    if is_open_ended:
        if visual_layer_enabled:
            if enable_reason:
                base = SLIM_SYSTEM_OPEN_WITH_KEYFRAME.format(question=question)
            else:
                base = SLIM_SYSTEM_OPEN_WITH_KEYFRAME_NO_REASON.format(question=question)
        else:
            if enable_reason:
                base = SLIM_SYSTEM_OPEN.format(question=question)
            else:
                base = SLIM_SYSTEM_OPEN_NO_REASON.format(question=question)
        cap_block = LEDGER_CAPABILITY_BLOCK_V2 if enable_reason else LEDGER_CAPABILITY_BLOCK_V2_NO_REASON
        parts: List[str] = [base, cap_block]
    else:
        if visual_layer_enabled:
            if enable_reason:
                base = SLIM_SYSTEM_MCQ_WITH_KEYFRAME.format(question=question, options=options_text)
            else:
                base = SLIM_SYSTEM_MCQ_WITH_KEYFRAME_NO_REASON.format(question=question, options=options_text)
        else:
            if enable_reason:
                base = SLIM_SYSTEM_MCQ.format(question=question, options=options_text)
            else:
                base = SLIM_SYSTEM_MCQ_NO_REASON.format(question=question, options=options_text)
        cap_block = LEDGER_CAPABILITY_BLOCK_MCQ if enable_reason else LEDGER_CAPABILITY_BLOCK_MCQ_NO_REASON
        parts = [base, cap_block]
    if extra_instructions:
        parts.append(extra_instructions)
    return "\n".join(parts)
