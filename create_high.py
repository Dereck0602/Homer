import os
import sys
import json
import base64
import mimetypes
import subprocess
import re
import tempfile
import argparse
from pathlib import Path
from openai import OpenAI
from typing import Literal, Optional
import pandas as pd
from tqdm import tqdm
import time
import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Bootstrap: register the script directory and lv_harness/vendor on sys.path so
# that `from mmagent.xxx import ...` resolves to the self-contained vendor copy
# under lv_harness/vendor/mmagent/.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_THIS_DIR, "lv_harness", "vendor")
for _p in (_THIS_DIR, _VENDOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmagent.retrieve import translate
from mmagent.eventgraph import EventGraph

MEMORY_PROMPT_TEMPLATE = """
You are a video narrative analyst. Your job is to consolidate fragmented clip-level memories into higher-level temporal and causal structures that can be used for retrieval and reasoning.

INPUT
You will receive:
1) PREVIOUS_GRAPH: an existing segment-level graph built from earlier clips.
2) NEW_MEMORIES: a list of memories extracted every ~30s from a long video.
Each item is either:
- [episodic] clip_id=<int> | <list of strings describing visible actions / spoken content>
- [semantic] clip_id=<int> | <list of strings describing high-level meanings / themes>

GOAL
Maintain a compact directed graph:
- Nodes are higher-level SEGMENTS (each segment groups multiple clips).
- Edges connect SEGMENTS with explicit TEMPORAL and CAUSAL relations.
- Each segment stores ONLY: segment_label, summary, clip_ids.
- Each edge stores ONLY: from, to, relation_type (temporal or causal), relation, evidence_clips, explicitness, rationale (short).
- This is a sliding-window refinement task: use BOTH the new memories AND (if non-empty) the existing graph.

IMPORTANT: Long-range dependencies are allowed.
Temporal/causal edges do NOT need to connect adjacent segments. New evidence may introduce links between long-range segments.

CRITICAL OUTPUT MODE (PATCH ONLY)
- DO NOT output the full graph.
- Output ONLY the CHANGES needed to update PREVIOUS_GRAPH using NEW_MEMORIES.
- "segments" in the output must contain ONLY:
  (a) newly created segments, and/or
  (b) existing segments that you MODIFY (typically by appending new clip_ids and updating the summary accordingly).
- "edges" in the output must contain ONLY:
  (a) newly created edges, and/or
  (b) existing edges that you MODIFY (e.g., updated rationale/explicitness due to new evidence), and/or
  (c) edges you want to REMOVE (see EDGE REMOVAL encoding below).
- If nothing changes, output {{"segments": [], "edges": []}}.

NON-AGGRESSIVE EDITING RULES (VERY IMPORTANT)
- For any existing segment in PREVIOUS_GRAPH: if you are NOT adding any new clip_id to it, do NOT modify it (do not rewrite its summary, do not reorder clip_ids, do not change the label).
- Prefer extending an existing segment over creating a near-duplicate.
- Only merge/split segments when the new evidence makes it clearly necessary; otherwise avoid structural churn.
- When segmenting events, maintain sufficient granularity. Do not merge distinct events into one due to over-abstraction.

COVERAGE REQUIREMENT (VERY IMPORTANT)
- Ensure EVERY clip_id appearing in NEW_MEMORIES is included in some segment after the update.
- Therefore, in this PATCH output, you must either:
  (1) create a new segment containing those new clip_ids, OR
  (2) output an updated existing segment where you append those new clip_ids.
- Do not leave any NEW_MEMORIES clip_id uncovered.

WHAT YOU MUST DO (IN ONE PASS)
If PREVIOUS_GRAPH is empty:
- Initialize the graph from NEW_MEMORIES.
- Even in this case, you still output in PATCH format; since there is no prior graph, your patch will effectively be the full initial set of segments and edges.
Otherwise:
- Update the existing graph using NEW_MEMORIES with minimal changes:
  1) Assign every new clip_id to a segment (extend or create).
  2) Update summaries ONLY for segments that receive new clip_ids.
  3) Add temporal/causal edges supported by NEW_MEMORIES (including long-range links).
  4) Revise or remove old edges only if contradicted or clearly improved by new evidence; keep stable otherwise.

SEGMENTATION RULES
- Create segments by grouping adjacent clips that share the same main topic/scene/goal.
- Prefer 3-8 segments for the given input (do not create too many).
- Each segment label must be short (<= 8 words), unique, and natural language.
- Each segment must include the list of clip ids it covers (sorted).
- Reuse and extend existing segments whenever possible; do not create duplicates.

SUMMARY REQUIREMENTS (IMPORTANT)
For each segment you OUTPUT in this PATCH, write a DETAILED summary (4-8 sentences) that is useful for reasoning.
The summary must include:
1) What happens in this segment (major actions and events, merged from micro steps)
2) Who is involved (speaker/cook/people) and what they do
3) A high-level description (summary, not low-level step-by-step details)
Do NOT add any information not present in the memories.
NOTE: Do NOT rewrite summaries for segments you are not modifying.

EDGE RULES
- Add a TEMPORAL edge when one segment clearly happens before/after/overlap/during another.
  * relation must be one of: "before", "overlap", "concurrent", "contains".
  * "before": from-segment happens before to-segment (do NOT use "after"; 
    if B happens after A, write from=A, to=B, relation="before").
  * "overlap": the two segments partially overlap in time.
  * "concurrent": the two segments happen at roughly the same time.
  * "contains": from-segment temporally contains to-segment (to is a sub-phase of from).
- Add a CAUSAL edge only when there is a clear because/therefore/enables/requires relationship.
  * relation must be one of: "causes", "enables", "requires", "prevents".
  * "causes": from-segment directly causes to-segment to happen.
  * "enables": from-segment makes to-segment possible (but does not directly cause it).
  * "requires": to-segment requires from-segment as a precondition.
  * "prevents": from-segment prevents to-segment from happening.
  * explicitness: "explicit" if directly stated/shown; otherwise "inferred".
  * confidence: a number in [0,1].
  * rationale: 1 short sentence grounded ONLY in the given memories.
- Long-range edges are allowed and encouraged when justified.
- Do NOT assume temporal adjacency implies causality.
- Prefer fewer, stronger edges over many weak ones.

EDGE REVISION RULES (when PREVIOUS_GRAPH is non-empty)
- Keep old edges unchanged if still supported; do not output unchanged edges.
- If new memories strengthen an inferred causal edge, you may output a MODIFIED version of that edge (same from/to/relation_type/relation) with updated evidence/rationale/explicitness/confidence.
- If new memories weaken an edge, you may remove it.

EDGE REMOVAL ENCODING (PATCH FORMAT)
- To remove an existing edge, output an edge object with the same identifiers (from, to, relation_type, relation) and add:
  "remove": true
- Do not include removed edges in normal form unless remove=true.

CLIP EVIDENCE RULES
- Every segment you output must list all clip_ids it covers (including previously covered ones if it is an updated segment).
- Every edge you output (added/modified/removed) must include evidence_clips that justify the decision.
- For edges spanning old and new segments, include both old and new clip ids in evidence_clips when possible.
- Each segment should cover no more than 40 clip_ids.

CONSTRAINTS
- Use ONLY the provided memories; do not invent new entities/events/places/dates.
- Output STRICT JSON ONLY (no markdown, no extra text).
- Segment references in edges must match segment labels exactly.

OUTPUT SCHEMA (STRICT) - PATCH ONLY
{{
  "segments": [
    {{
      "segment_label": "...",
      "summary": "...",
      "clip_ids": [ ... ]
    }}
  ],
  "edges": [
    {{
      "from": "...(segment_label)",
      "to": "...(segment_label)",
      "relation_type": "temporal|causal",
      "relation": "before|after|overlap|during|causes|enables|explains|motivates|prevents",
      "evidence_clips": [ ... ],
      "explicitness": "explicit|inferred",
      "confidence": 0.0,
      "rationale": "...",
      "remove": false
    }}
  ]
}}

Now process the following inputs and produce the PATCH JSON.

PREVIOUS_GRAPH
{graph}

MEMORIES
{memory}
"""

# default config file path (configs/api_config.json next to this script)
DEFAULT_API_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "api_config.json")


def build_client(model: str, api_config_path: str = DEFAULT_API_CONFIG) -> OpenAI:
    """Read the base_url and api_key for the given model from api_config.json and build an OpenAI client."""
    with open(api_config_path, "r", encoding="utf-8") as f:
        api_config = json.load(f)

    if model not in api_config:
        available = ", ".join(sorted(api_config.keys()))
        raise ValueError(
            f"Model '{model}' not found in {api_config_path}. "
            f"Available models: {available}"
        )

    cfg = api_config[model]
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")

    if not base_url or not api_key:
        raise ValueError(f"base_url or api_key for model '{model}' is empty; please check {api_config_path}")

    print(f"[build_client] model={model}, base_url={base_url}")
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_text(s: str) -> str:
    if s is None:
        raise ValueError("empty content")
    s = s.strip()

    # 1) strip <think>...</think> reasoning traces (emitted by some models)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip()

    # 2) strip ```json ... ``` or ``` ... ``` fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()

    # 3) after unwrapping, try parsing directly; if it succeeds, do not over-trim
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # 4) if non-JSON text is still mixed in, extract the outermost JSON object/array.
    #    NOTE: match {} (object) first, then [] (array),
    #    because the top level is {}, and an inner [] could be mis-extracted and break the JSON.
    lbc, rbc = s.find("{"), s.rfind("}")
    if lbc != -1 and rbc != -1 and rbc > lbc:
        return s[lbc:rbc+1].strip()

    lbr, rbr = s.find("["), s.rfind("]")
    if lbr != -1 and rbr != -1 and rbr > lbr:
        return s[lbr:rbr+1].strip()

    # 5) if nothing is found, return as-is and let the caller raise
    return s

def robust_json_loads(content: str):
    cleaned = extract_json_text(content)
    obj = json.loads(cleaned)
    # if an array is required: when the model returns a dict, wrap it in a list
    if isinstance(obj, dict):
        obj = [obj]
    return obj

def parse_sample_from_messages(messages):
    """
    Extract from the parquet `messages` structure:
    - video_rel_or_url: e.g. '5nKz1hzvSqs.mp4'
    - query: e.g. 'What is ... ?'
    - steps: the assistant's '<think>...<answer>...'
    """
    video_rel_or_url = None
    user_text = None
    assistant_text = None

    for msg in list(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            # user content: [video_url, text]
            for item in list(content):
                if isinstance(item, dict) and item.get("type") == "video_url":
                    vu = item.get("video_url") or {}
                    if isinstance(vu, dict) and vu.get("url"):
                        video_rel_or_url = vu["url"]
                if isinstance(item, dict) and item.get("type") == "text":
                    user_text = item.get("text")

        elif role == "assistant":
            # assistant content: [text]
            for item in list(content):
                if isinstance(item, dict) and item.get("type") == "text":
                    assistant_text = item.get("text")

    # extract the final question from user_text as the query (a robust, minimal approach)
    query = None
    if user_text:
        lines = [ln.strip() for ln in str(user_text).splitlines() if ln.strip()]
        if lines:
            query = lines[-1].split("Question: ")[-1]  # usually the last line is the question

    return video_rel_or_url, query, assistant_text

def call_text_llm(
    client: OpenAI,
    prompt_text: str,
    model: str = "qwen3vl32b",
    timeout_s: int = 120,
    max_retries: int = 5,
    retry_sleep_s: float = 1.0,
) -> Optional[str]:
    """Call the chat-completions API with a pure-text prompt.
    Returns None if empty result after retries.
    """

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                timeout=timeout_s,
            )

            msg = resp.choices[0].message
            content = getattr(msg, "content", None)

            if content:
                return content

            raise ValueError("Empty content")

        except Exception as e:
            # on the last attempt, raise so the caller can handle it
            if attempt == max_retries - 1:
                print(f"[Error] {e}")
                raise
            time.sleep(retry_sleep_s)

    return None

def truncate(text: str, max_len: int | None) -> str:
    if not max_len or len(text) <= max_len:
        return text
    return text[:max_len] + "..."

def extract_clip(vg, clip_id: int,
                    only: str | None = None,
                    max_len: int | None = None,
                    show_faces: bool = True) -> None:
    # 1) ---------- Text Memory ----------
    
    node_ids = vg.text_nodes_by_clip.get(clip_id)
    if not node_ids:
        print(f"[Warning] clip_id={clip_id} does not exist or the clip does not have a text node")
        return

    texts = []
    for nid in node_ids:
        node = vg.nodes[nid]
        if only and node.type != only:
            continue
        contents = node.metadata.get("contents", [])
        contents = translate(vg, contents)
        contents = [truncate(c, max_len) for c in contents]
        if contents:
            texts.append(f"[{node.type:^8}] id={clip_id:<4} | " + contents[0])
    
    return texts



MAX_WORKERS = 1
BATCH_SIZE = 30

def process_one_video(file: str, folder_path: str, out_dir: str, client, model: str) -> str:
    filename = os.path.splitext(file)[0]

    PKL_PATH = os.path.join(folder_path, f'{filename}.pkl')
    OUT_JSON = os.path.join(out_dir, f'{filename}.json')

    if os.path.exists(OUT_JSON) and os.path.getsize(OUT_JSON) > 0:
        return f"SKIP {filename}"

    global_graph = EventGraph()

    vg = pd.read_pickle(PKL_PATH)
    clip_ids = sorted(vg.event_sequence_by_clip.keys())

    # the patch (newly added subgraph) generated by the LLM in the previous round
    prev_patch = {"segments": [], "edges": []}

    batch_texts = []
    for i, clip_id in enumerate(clip_ids):
        texts = extract_clip(vg, clip_id=clip_id)
        if texts:
            batch_texts.extend(texts)

        is_batch_end = ((i + 1) % BATCH_SIZE == 0)
        is_last = (i == len(clip_ids) - 1)
        if not (is_batch_end or is_last):
            continue

        memory_text = "\n".join(batch_texts)

        # pass in the previous round's patch instead of the full graph
        graph_input = json.dumps(prev_patch, ensure_ascii=False)
        prompt_text = MEMORY_PROMPT_TEMPLATE.format(graph=graph_input, memory=memory_text)

        content = call_text_llm(client=client, prompt_text=prompt_text, model=model)
        
        # more robust: clean then parse; on parse failure treat as an empty patch
        # (to avoid feeding a dirty string into the next round)
        try:
            cleaned = extract_json_text(content)
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {"segments": [], "edges": [], "raw": content}

        # merge into the full graph (the locally maintained global_graph)
        global_graph.update_from_llm_output(parsed)

        # update prev_patch: only feed the "previous round's newly added subgraph" next time.
        # NOTE: if parsed carries `raw` (parse failure), keep only the valid fields to avoid bloat.
        prev_patch = {"segments": parsed.get("segments", []), "edges": parsed.get("edges", [])}

        batch_texts = []

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(global_graph.to_dict(), f, ensure_ascii=False, indent=2)

    return f"DONE {filename}"


def get_target_videos_from_annotation(annotation_path: str, last_n: int = 300) -> set:
    """Take the video file names of the last last_n questions from videomme.json.

    Logic: iterate over all questions in ascending video-id (key) order, take the last
    last_n, and return the file names of the videos those questions belong to
    (e.g. 'TGom0uiW130.mp4').
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # collect all (video_key, video_path) pairs in ascending video-id (numeric) order
    all_questions = []  # [(video_key, video_path, question_dict), ...]
    for vkey in sorted(annotations.keys(), key=lambda x: int(x)):
        item = annotations[vkey]
        video_path = item.get("video_path", "")
        for qa in item.get("qa_list", []):
            all_questions.append((vkey, video_path, qa))

    # take the last last_n questions
    tail_questions = all_questions[-last_n:]

    # extract the corresponding set of video file names
    target_videos = set()
    for _, vpath, _ in tail_questions:
        basename = os.path.basename(vpath)  # e.g. 'TGom0uiW130.mp4'
        if basename:
            target_videos.add(basename)

    print(f"[filter] the last {last_n} questions cover {len(target_videos)} videos")
    return target_videos


def parse_args():
    parser = argparse.ArgumentParser(description="Build the event-level high-level graph from memory graphs")
    parser.add_argument("--folder_path", type=str, required=True,
                        help="directory containing the .pkl memory graph files")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="directory for the output JSON files")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.1T",
                        help="model name to call (default: DeepSeek-V3.1T); must match a key in api_config.json")
    parser.add_argument("--api_config", type=str, default=DEFAULT_API_CONFIG,
                        help="API config file path (default: configs/api_config.json)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="number of concurrent threads (default: 1)")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="number of clips per batch (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    global MAX_WORKERS, BATCH_SIZE
    MAX_WORKERS = args.max_workers
    BATCH_SIZE = args.batch_size

    folder_path = args.folder_path
    out_dir = args.out_dir
    model = args.model

    # NOTE: whether the client is thread-safe depends on build_client()'s implementation:
    # - a stateless HTTP client (requests/httpx) is usually fine
    # - if unsure, build_client() per thread (see the "thread-safe approach" below)
    client = build_client(model=model, api_config_path=args.api_config)

    pkl_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(".pkl")]

    results = {"DONE": 0, "SKIP": 0, "FAIL": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_video, f, folder_path, out_dir, client, model): f
            for f in pkl_files
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos (threads)"):
            file = futures[fut]
            try:
                status = fut.result()
                if status.startswith("DONE"):
                    results["DONE"] += 1
                elif status.startswith("SKIP"):
                    results["SKIP"] += 1
                else:
                    # fallback
                    results["FAIL"] += 1
                tqdm.write(status)
            except Exception as e:
                results["FAIL"] += 1
                tqdm.write(f"FAIL {os.path.splitext(file)[0]}: {repr(e)}")

    print("Summary:", results)

if __name__ == "__main__":
    main()
