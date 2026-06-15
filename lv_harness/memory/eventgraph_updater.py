"""
EventGraphIncrementalUpdater: the EventGraph incremental updater.

In streaming inference mode, it automatically triggers an EventGraph incremental update every N clips.
It reuses the core logic from create_high.py:
- MEMORY_PROMPT_TEMPLATE: the LLM prompt template
- extract_clip: extract clip memory text from VideoGraph
- EventGraph.update_from_llm_output: merge the patch into the global graph
"""
import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _extract_json_text(s: str) -> str:
    """Extract JSON text from the LLM output. Reuses the logic of create_high.py."""
    if s is None:
        raise ValueError("empty content")
    s = s.strip()

    # Remove the <think>...</think> reasoning process
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip()

    # Remove the ```json ... ``` wrapper
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()

    try:
        json.loads(s)
        return s
    except Exception:
        pass

    lbc, rbc = s.find("{"), s.rfind("}")
    if lbc != -1 and rbc != -1 and rbc > lbc:
        return s[lbc:rbc + 1].strip()

    lbr, rbr = s.find("["), s.rfind("]")
    if lbr != -1 and rbr != -1 and rbr > lbr:
        return s[lbr:rbr + 1].strip()

    return s


def _extract_clip_texts(video_graph, clip_id: int) -> List[str]:
    """Extract the memory text of the specified clip from VideoGraph.

    Reuses the core logic of extract_clip in create_high.py.
    """
    from mmagent.retrieve import translate

    node_ids = getattr(video_graph, 'text_nodes_by_clip', {}).get(clip_id)
    if not node_ids:
        return []

    texts = []
    nodes = getattr(video_graph, 'nodes', {})
    for nid in node_ids:
        node = nodes.get(nid)
        if node is None:
            continue
        contents = node.metadata.get("contents", [])
        contents = translate(video_graph, contents)
        if contents:
            node_type = getattr(node, 'type', 'unknown')
            texts.append(f"[{node_type:^8}] clip_id={clip_id:<4} | " + contents[0])

    return texts


# Reuse the MEMORY_PROMPT_TEMPLATE from create_high.py
# To avoid redefining the very long prompt, import it directly from create_high.py
_MEMORY_PROMPT_TEMPLATE = None


def _get_memory_prompt_template() -> str:
    """Lazily load MEMORY_PROMPT_TEMPLATE."""
    global _MEMORY_PROMPT_TEMPLATE
    if _MEMORY_PROMPT_TEMPLATE is None:
        try:
            # Try to import from create_high.py
            import sys
            import importlib.util
            # Locate create_high.py under the m3-agent directory
            m3_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            create_high_path = os.path.join(m3_agent_dir, "create_high.py")
            if not os.path.exists(create_high_path):
                # Try to find it in PYTHONPATH
                for p in sys.path:
                    candidate = os.path.join(p, "create_high.py")
                    if os.path.exists(candidate):
                        create_high_path = candidate
                        break

            if os.path.exists(create_high_path):
                spec = importlib.util.spec_from_file_location("create_high", create_high_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _MEMORY_PROMPT_TEMPLATE = mod.MEMORY_PROMPT_TEMPLATE
                logger.info(f"[EventGraphUpdater] loaded MEMORY_PROMPT_TEMPLATE from {create_high_path}")
            else:
                raise FileNotFoundError("create_high.py not found")
        except Exception as e:
            logger.warning(f"[EventGraphUpdater] unable to import MEMORY_PROMPT_TEMPLATE: {e}, using the built-in simplified version")
            _MEMORY_PROMPT_TEMPLATE = _FALLBACK_PROMPT_TEMPLATE

    return _MEMORY_PROMPT_TEMPLATE


# Simplified prompt (used when create_high.py cannot be imported)
_FALLBACK_PROMPT_TEMPLATE = """
You are a video narrative analyst. Consolidate clip-level memories into higher-level temporal and causal structures.

INPUT:
1) PREVIOUS_GRAPH: existing segment-level graph from earlier clips.
2) NEW_MEMORIES: memories extracted from new video clips.

GOAL: Output a PATCH (only changes) to update the graph.
- "segments": only new or modified segments
- "edges": only new, modified, or removed edges

Output STRICT JSON ONLY:
{{
  "segments": [
    {{"segment_label": "...", "summary": "...", "clip_ids": [...]}}
  ],
  "edges": [
    {{"from": "...", "to": "...", "relation_type": "temporal|causal", "relation": "...", "evidence_clips": [...], "explicitness": "explicit|inferred", "confidence": 0.0, "rationale": "..."}}
  ]
}}

PREVIOUS_GRAPH
{graph}

MEMORIES
{memory}
"""


class EventGraphIncrementalUpdater:
    """The EventGraph incremental updater.

    In streaming inference mode, it automatically triggers an EventGraph incremental update every update_interval clips.

    Flow:
    1. Buffer the ID of each clip
    2. When the buffer reaches update_interval, extract memory text from VideoGraph
    3. Call the LLM to generate an EventGraph patch
    4. Merge the patch into the global EventGraph
    5. Rebuild the embedding cache of EventGraphRetriever

    Args:
        update_interval: trigger an update every N clips
        model: the LLM model used to generate the EventGraph patch
        api_config_path: path to the API configuration file
        max_retries: maximum number of retries for LLM calls
    """

    def __init__(self, update_interval: int = 20,
                 model: str = "DeepSeek-V3.1T",
                 api_config_path: str = "configs/api_config.json",
                 max_retries: int = 5):
        self.update_interval = update_interval
        self.model = model
        self.api_config_path = api_config_path
        self.max_retries = max_retries

        self._clip_buffer: List[int] = []
        self._prev_patch: Dict[str, list] = {"segments": [], "edges": []}
        self._client = None
        self._event_graph = None
        self._update_count = 0
        self._tmp_dirs: List[str] = []  # track temporary directories for easy cleanup

    def reset(self):
        """Reset the updater state (called when switching videos)."""
        self._clip_buffer = []
        self._prev_patch = {"segments": [], "edges": []}
        self._event_graph = None
        self._update_count = 0
        self._cleanup_tmp_dirs()

    def _cleanup_tmp_dirs(self):
        """Clean up all temporary directories to avoid disk leaks."""
        import shutil
        for tmp_dir in self._tmp_dirs:
            try:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.debug(f"[EventGraphUpdater] failed to clean up temporary directory: {tmp_dir}: {e}")
        self._tmp_dirs = []

    def _ensure_client(self):
        """Lazily initialize the LLM client."""
        if self._client is None:
            from openai import OpenAI
            with open(self.api_config_path) as f:
                api_cfg = json.load(f)
            if self.model not in api_cfg:
                raise KeyError(f"model '{self.model}' is not configured in {self.api_config_path}")
            cfg = api_cfg[self.model]
            self._client = OpenAI(
                base_url=cfg.get("base_url", ""),
                api_key=cfg.get("api_key", ""),
            )

    def _ensure_event_graph(self):
        """Lazily initialize the EventGraph."""
        if self._event_graph is None:
            from mmagent.eventgraph import EventGraph
            self._event_graph = EventGraph()

    def on_clip_ingested(self, clip_id: int, memory) -> bool:
        """Callback after a clip is ingested.

        Args:
            clip_id: the ID of the just-ingested clip
            memory: a MemoryStrategy instance (must have a video_graph attribute)

        Returns:
            whether an EventGraph update was triggered
        """
        self._clip_buffer.append(clip_id)

        if len(self._clip_buffer) >= self.update_interval:
            self._do_update(memory)
            return True
        return False

    def force_update(self, memory) -> bool:
        """Force a single update (called at the end of the video)."""
        if self._clip_buffer:
            self._do_update(memory)
            return True
        return False

    def _do_update(self, memory):
        """Perform an EventGraph incremental update."""
        self._ensure_client()
        self._ensure_event_graph()

        video_graph = getattr(memory, 'video_graph', None)
        if video_graph is None:
            logger.warning("[EventGraphUpdater] memory has no video_graph attribute, skipping update")
            self._clip_buffer = []
            return

        # Step 1: extract the memory text of the clips in the buffer from VideoGraph
        batch_texts = []
        for cid in self._clip_buffer:
            texts = _extract_clip_texts(video_graph, cid)
            if texts:
                batch_texts.extend(texts)

        if not batch_texts:
            logger.debug(f"[EventGraphUpdater] no valid memory text in the buffer, skipping update")
            self._clip_buffer = []
            return

        memory_text = "\n".join(batch_texts)
        graph_input = json.dumps(self._prev_patch, ensure_ascii=False)

        # Step 2: call the LLM to generate the patch
        prompt_template = _get_memory_prompt_template()
        prompt_text = prompt_template.format(graph=graph_input, memory=memory_text)

        content = self._call_llm(prompt_text)
        if content is None:
            logger.error("[EventGraphUpdater] LLM call failed, skipping this update")
            self._clip_buffer = []
            return

        # Step 3: parse the patch
        try:
            cleaned = _extract_json_text(content)
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"[EventGraphUpdater] JSON parsing failed: {e}, skipping this update")
            parsed = {"segments": [], "edges": []}

        # Step 4: merge into the global EventGraph
        self._event_graph.update_from_llm_output(parsed)

        # Step 5: update prev_patch
        self._prev_patch = {
            "segments": parsed.get("segments", []),
            "edges": parsed.get("edges", []),
        }

        # Step 6: rebuild the retriever (if memory has an event_retriever)
        self._rebuild_retriever(memory)

        self._update_count += 1
        clips_processed = list(self._clip_buffer)
        self._clip_buffer = []

        logger.info(
            f"[EventGraphUpdater] update #{self._update_count} complete: "
            f"processed {len(clips_processed)} clips "
            f"(clip_ids: {clips_processed[0]}~{clips_processed[-1]}), "
            f"new segments: {len(parsed.get('segments', []))}, "
            f"new edges: {len(parsed.get('edges', []))}"
        )

    def _call_llm(self, prompt_text: str) -> Optional[str]:
        """Call the LLM to generate an EventGraph patch."""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_text}],
                    timeout=120,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
                raise ValueError("Empty content")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"[EventGraphUpdater] LLM call failed (already retried {self.max_retries} times): {e}")
                    return None
                time.sleep(min(20, 2 * (attempt + 1)))
        return None

    def _rebuild_retriever(self, memory):
        """Rebuild the embedding cache of EventGraphRetriever."""
        try:
            from mmagent.event_retrieve import EventGraphRetriever
            import tempfile

            # Save the EventGraph to a temporary directory
            tmp_dir = tempfile.mkdtemp(prefix="lv_harness_eg_")
            self._tmp_dirs.append(tmp_dir)
            eg_json_path = os.path.join(tmp_dir, "eventgraph.json")
            with open(eg_json_path, "w", encoding="utf-8") as f:
                json.dump(self._event_graph.to_dict(), f, ensure_ascii=False)

            emb_cache_dir = os.path.join(tmp_dir, "emb_cache")
            os.makedirs(emb_cache_dir, exist_ok=True)

            memory.event_retriever = EventGraphRetriever(
                eventgraph_dir=tmp_dir,
                emb_cache_dir=emb_cache_dir,
                embedding_model="text-embedding-v4",
                neighbors_topk=3,
            )
            logger.info("[EventGraphUpdater] EventGraphRetriever rebuilt")
        except Exception as e:
            logger.warning(f"[EventGraphUpdater] failed to rebuild EventGraphRetriever: {e}")

    @property
    def event_graph(self):
        """Get the current EventGraph instance."""
        return self._event_graph

    @property
    def stats(self) -> Dict[str, Any]:
        """Return statistics about the updater."""
        return {
            "update_count": self._update_count,
            "buffer_size": len(self._clip_buffer),
            "update_interval": self.update_interval,
            "model": self.model,
        }
