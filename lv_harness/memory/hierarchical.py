"""
HierarchicalMemory: the two-layer memory strategy of M3-Agent (VideoGraph + EventGraph).

This is the current default strategy of M3-Agent and connects directly to the existing code:
- mmagent.videograph.VideoGraph
- mmagent.event_retrieve.EventGraphRetriever
- mmagent.retrieve.search
- lv_harness.memory.clip_ingestion.process_clip (internalized memory generation)

Optional third layer: the visual layer (visual_layer)
- When enabled=false, all original logic and interfaces in this file remain bit-for-bit identical (strict backward compatibility).
- When enabled=true, a new `retrieve(mode="keyframe_inspect")` branch is added, which loads
  keyframe paths from {clips_root}/{video_name}/{keyframe_dir_name}/ (without loading base64;
  image loading is delegated to the reasoning/multi_round main thread).
"""
import os
import json
import glob
import pickle
import logging
from typing import Dict, Any, Optional, List

from .base import MemoryStrategy
from .clip_ingestion import IngestionConfig, process_clip
from ..data.types import ClipData, RetrievalResult, MemorySnapshot

logger = logging.getLogger(__name__)

# Visual-question recognition keywords (lv_harness extended version, based on control_api_triple_v2.py with added spatial/situational words).
# Used only as a hint for sufficiency / feedback to "recommend using KEYFRAME"; it does not trigger anything automatically.
# Reason for extension: inspecting the failure samples of m3bench/robot (Q09 "where is the yoga mat", Q11 "which table is the laptop on",
# Q12 "what is Jack getting ready to do") shows they are all visually oriented, but the baseline vocabulary lacks spatial/situational words
# such as where/which/on/doing, leading to a low recognition rate for visual questions. These are filled in here.
VISUAL_KEYWORDS = {
    # Appearance / visual attributes
    "color", "colour", "shape", "logo", "text", "appear", "appearance",
    "look", "looks", "looking", "wear", "wearing", "worn", "dress", "dressed",
    "visual", "scene", "layout", "background", "screen", "display",
    "image", "picture", "photo", "sign", "symbol", "icon",
    "written", "writing", "read", "show", "shown", "showing",
    # Colors
    "red", "blue", "green", "yellow", "white", "black", "pink", "purple",
    "orange", "brown",
    # Spatial orientation
    "left", "right", "top", "bottom", "middle", "center",
    # Visible attributes of people
    "face", "expression", "gesture", "posture", "position",
    # ---- lv_harness extension: spatial / container / localization questions ----
    # Spatial interrogatives such as "where / which X"
    "where", "which",
    # Spatial prepositions (matching "on the table / in the kitchen / at the door / under / behind", etc.)
    "on", "in", "at", "under", "over", "above", "below", "behind", "beside",
    "next", "inside", "outside", "between", "near", "beneath", "atop",
    # Location / container nouns
    "located", "location", "placed", "place", "table", "shelf", "wall",
    "floor", "chair", "sofa", "couch", "desk", "counter", "drawer", "door",
    "window", "room", "corner",
    # Action / state (situational visual questions: what is X doing)
    "doing", "holding", "holds", "carrying", "carries", "sitting", "standing",
    "lying", "wearing", "eating", "drinking", "reading", "writing", "using",
    "opening", "closing", "pointing", "touching", "typing",
    # Situational visual words: "What is X getting ready to do" / "X is about to"
    "getting", "ready",
}


_CAUSAL_TOKENS = {"why", "because", "reason", "motivation", "purpose"}
_ACTION_TOKENS = {
    "doing", "holding", "holds", "carrying", "carries", "sitting", "standing",
    "lying", "wearing", "eating", "drinking", "reading", "writing", "using",
    "opening", "closing", "pointing", "touching", "typing",
    # "getting"/"ready"/"do" cover situational visual questions like "What is X getting ready to do",
    # and work together with the identically named words in VISUAL_KEYWORDS: they are stripped out in why/causal questions to avoid misclassification.
    "getting", "ready", "do",
}


def _is_visual_query(question: str) -> bool:
    """Determine whether the question is visually oriented (token intersection matching VISUAL_KEYWORDS).

    Refinement: if the question contains "motivation/reason" marker words such as why/because/reason, the action words
    (doing/holding/eating...) are removed from the tokens used for the decision, to avoid misclassifying
    a motivation question like "Why does Lily like doing yoga" as a visual question.
    Other visual words (color/where/on/table/...) still participate in the decision.
    """
    if not question:
        return False
    tokens = set(question.lower().split())
    if tokens & _CAUSAL_TOKENS:
        # Motivation/reason questions: action words no longer count as visual signals
        tokens = tokens - _ACTION_TOKENS
    return bool(tokens & VISUAL_KEYWORDS)


# The "harness-level" fields allowed at the top level of memory_config.json (these are not VideoGraph constructor parameters).
# When constructing VideoGraph, these fields must be stripped, otherwise VideoGraph.__init__ will raise
# TypeError: unexpected keyword argument.
_NON_VIDEOGRAPH_KEYS = {"visual_layer"}


def _filter_videograph_kwargs(memory_config: dict) -> dict:
    """Strip non-VideoGraph fields from the dictionary loaded from memory_config.json."""
    if not isinstance(memory_config, dict):
        return {}
    return {k: v for k, v in memory_config.items() if k not in _NON_VIDEOGRAPH_KEYS}


class HierarchicalMemory(MemoryStrategy):
    """The two-layer memory strategy of M3-Agent: VideoGraph + EventGraph.

    Connects to existing modules:
    - VideoGraph: fine-grained memory (episodic + semantic)
    - EventGraph: event-level memory (retrieved via EventGraphRetriever)

    Args:
        config: configuration dictionary, containing:
            - videograph: VideoGraph configuration
            - eventgraph_dir: EventGraph data directory
            - update_interval: EventGraph incremental update interval (number of clips)
            - topk: VideoGraph retrieval top_k
    """

    def __init__(self, config: dict):
        self.config = config
        self._topk = config.get("topk", 2)
        self._mem_wise_topk = config.get("mem_wise_topk", 20)

        # Visual-layer configuration (third layer)
        # - Read first from the visual_layer section at the top level of memory_config.json
        # - Also allow reading from the top level of the harness config (the passed-in config),
        #   to make it convenient for orchestrator/run_m3bench_test.sh to override via the CLI
        self._visual_cfg = self._resolve_visual_layer_config(config)
        self._visual_enabled = bool(self._visual_cfg.get("enabled", False))

        # Deferred import to avoid circular dependencies
        from mmagent.videograph import VideoGraph
        from mmagent.utils.general import load_video_graph

        # VideoGraph
        vg_config = config.get("videograph", {})
        memory_config_path = config.get("memory_config_path", "configs/memory_config.json")
        if os.path.exists(memory_config_path):
            with open(memory_config_path) as f:
                memory_config = json.load(f)
        else:
            memory_config = vg_config

        # Strip harness-level fields (such as visual_layer) to avoid passing them to VideoGraph
        self.video_graph = VideoGraph(**_filter_videograph_kwargs(memory_config))
        self._load_video_graph = load_video_graph

        # Internalized memory-generation configuration
        self._ingestion_config = IngestionConfig.from_harness_config(config)

        # EventGraph Retriever
        self.event_retriever = None
        eventgraph_dir = config.get("eventgraph_dir", "")
        if eventgraph_dir and os.path.exists(eventgraph_dir):
            self._init_event_retriever(eventgraph_dir)

        # Memory cache directory (in streaming mode, save/reuse the already-built VideoGraph)
        self._memory_cache_dir = config.get("memory_cache_dir", "")
        if self._memory_cache_dir:
            os.makedirs(self._memory_cache_dir, exist_ok=True)
            logger.info(f"[MemoryCache] cache directory: {self._memory_cache_dir}")

        # State
        self._total_clips = 0
        self._current_video_name = ""
        self._mem_path = ""
        self._loaded_from_cache = False  # marks whether this run was loaded from cache
        # Interval for periodically refreshing equivalence relations (refresh once every N clips)
        self._equivalence_refresh_interval = config.get("equivalence_refresh_interval", 5)
        # clip_wise deduplication state: records clips that have already been returned, to avoid multi-round VIDEO queries returning duplicate content
        # Aligned with data["currenr_clips"] in control_api.py
        self._current_clips: List[int] = []

    def _init_event_retriever(self, eventgraph_dir: str):
        """Initialize the EventGraph retriever."""
        from mmagent.event_retrieve import EventGraphRetriever
        emb_cache_dir = os.path.join(eventgraph_dir, "emb_cache")
        os.makedirs(emb_cache_dir, exist_ok=True)
        self.event_retriever = EventGraphRetriever(
            eventgraph_dir=eventgraph_dir,
            emb_cache_dir=emb_cache_dir,
            embedding_model="text-embedding-v4",
            neighbors_topk=3,
        )
        logger.info(f"[EventGraph] initialization complete: {eventgraph_dir}")

    # ------------------------------------------------------------------
    # Third layer: visual-layer (visual_layer) utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_visual_layer_config(config: dict) -> dict:
        """Resolve the visual_layer configuration.

        Priority: harness config top level > memory_config.json top level > empty dictionary.
        Defaults: enabled=false, keyframe_dir_name="keyframe_hybridv2", max_images_per_call=5.
        """
        # 1) passed in at the top level of the harness config
        top_level = config.get("visual_layer", {})
        # 2) read from the top level of memory_config.json (if memory_config has been loaded)
        memory_config_path = config.get("memory_config_path", "configs/memory_config.json")
        mc_level = {}
        if os.path.exists(memory_config_path):
            try:
                with open(memory_config_path) as f:
                    mc = json.load(f)
                mc_level = mc.get("visual_layer", {}) or {}
            except Exception:
                mc_level = {}
        merged = {
            "enabled": False,
            "clips_root": "",
            "keyframe_dir_name": "keyframe_hybridv2",
            "max_images_per_call": 5,
        }
        merged.update(mc_level)
        merged.update(top_level or {})
        return merged

    def _get_keyframe_dir(self, video_name: str) -> str:
        """Return the keyframe directory path corresponding to video_name (existence not guaranteed)."""
        clips_root = self._visual_cfg.get("clips_root", "") or ""
        kf_dir_name = self._visual_cfg.get("keyframe_dir_name", "keyframe_hybridv2")
        if not clips_root or not video_name:
            return ""
        return os.path.join(clips_root, video_name, kf_dir_name)

    def keyframe_preflight(self) -> Dict[str, Any]:
        """Check whether the current video has an available keyframe directory, for per-question downgrade at the reasoning layer."""
        status = {
            "enabled": self._visual_enabled,
            "available": False,
            "video_name": "",
            "keyframe_dir": "",
            "reason": "visual_layer_disabled",
        }
        if not self._visual_enabled:
            return status

        if not self._current_video_name:
            status["reason"] = "video_missing"
            return status
        if not self.event_retriever:
            status["reason"] = "event_retriever_missing"
            return status

        video_name = self._current_video_name
        try:
            video_name = self.event_retriever.get_video_name(
                self._mem_path or self._current_video_name
            )
        except Exception as exc:
            logger.warning(f"[Keyframe] preflight video name resolve failed: {exc}")
        status["video_name"] = video_name

        kf_dir = self._get_keyframe_dir(video_name)
        status["keyframe_dir"] = kf_dir
        if not kf_dir:
            status["reason"] = "keyframe_dir_unconfigured"
            return status
        if not os.path.isdir(kf_dir):
            status["reason"] = "keyframe_dir_missing"
            return status

        has_images = bool(glob.glob(os.path.join(kf_dir, "*.jpg")))
        if not has_images:
            status["reason"] = "keyframe_images_missing"
            return status

        status["available"] = True
        status["reason"] = "ok"
        return status

    def _resolve_clip_keyframe_paths(self, video_name: str, clip_id: int) -> List[str]:
        """Resolve all keyframe file paths for a given clip (sorted).

        The three globs exactly replicate the implementation in control_api_triple_v2.py:
          1) {kf_dir}/*_{clip_id}_*.jpg
          2) {kf_dir}/{clip_id}_*.jpg
          3) Fallback: sort {clips_root}/{video_name}/*.mp4 to get the basename, then match
             {kf_dir}/{clip_basename}_*.jpg
        Returns an empty list when all matches fail (the upper layer will note this in the feedback).
        """
        kf_dir = self._get_keyframe_dir(video_name)
        if not kf_dir or not os.path.isdir(kf_dir):
            return []

        pattern_candidates = (
            glob.glob(os.path.join(kf_dir, f"*_{clip_id}_*.jpg"))
            + glob.glob(os.path.join(kf_dir, f"{clip_id}_*.jpg"))
        )
        if not pattern_candidates:
            clips_root = self._visual_cfg.get("clips_root", "") or ""
            clips_dir = os.path.join(clips_root, video_name) if clips_root else ""
            if clips_dir and os.path.isdir(clips_dir):
                mp4_files = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
                if 0 <= clip_id < len(mp4_files):
                    clip_basename = os.path.splitext(
                        os.path.basename(mp4_files[clip_id])
                    )[0]
                    pattern_candidates = sorted(glob.glob(
                        os.path.join(kf_dir, f"{clip_basename}_*.jpg")
                    ))

        return sorted(set(pattern_candidates))

    def _load_keyframes_for_clips(self, video_name: str, clip_ids: List[int],
                                  max_images: int) -> List[Dict[str, Any]]:
        """Load keyframe paths with quota distributed evenly across clips (does **not** load base64).

        Returns [{"clip_id": int, "path": str}, ...], with a total count not exceeding max_images.
        Design considerations:
          - lv_harness runs single-process inference (unlike control_api_triple_v2, which uses
            multiprocessing), but it already holds large objects such as the memory cache and the
            pre-built VideoGraph; caching base64 at the memory layer would inflate the pickle size. This is delegated to the upper-layer LRU cache.
          - When the number of clips exceeds max_images, take only 1 image from each of the first max_images clips
            (more stable than "fewer than 1 image per clip").
        """
        if not clip_ids or max_images <= 0:
            return []
        sorted_cids = sorted(set(int(c) for c in clip_ids))
        if len(sorted_cids) >= max_images:
            # 1 image per clip, taking only the first max_images clips
            quotas = {cid: 1 for cid in sorted_cids[:max_images]}
        else:
            per_clip = max(1, max_images // len(sorted_cids))
            quotas = {cid: per_clip for cid in sorted_cids}

        results: List[Dict[str, Any]] = []
        for cid in sorted_cids:
            if cid not in quotas:
                break
            if len(results) >= max_images:
                break
            quota = quotas[cid]
            loaded = 0
            for img_path in self._resolve_clip_keyframe_paths(video_name, cid):
                if loaded >= quota or len(results) >= max_images:
                    break
                results.append({"clip_id": cid, "path": img_path})
                loaded += 1
        return results

    def _get_cache_path(self, video_name: str) -> str:
        """Get the cache file path for the specified video."""
        safe_name = video_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self._memory_cache_dir, f"{safe_name}.pkl")

    def reset_current_clips(self) -> None:
        """Reset the clip_wise deduplication state. Called by the agent at the start of each new question."""
        self._current_clips = []

    def on_video_start(self, video_name: str, **kwargs):
        """When a video starts being processed, load memory by priority: cache > pre-built > built from scratch."""
        self._current_video_name = video_name
        self._mem_path = kwargs.get("mem_path", "")
        self._total_clips = 0
        self._loaded_from_cache = False
        self._current_clips = []  # reset clip deduplication state for a new video

        # Priority 1: load from the cache directory (reuse in streaming mode)
        if self._memory_cache_dir:
            cache_path = self._get_cache_path(video_name)
            if os.path.exists(cache_path):
                try:
                    self.video_graph = self._load_video_graph(cache_path)
                    self._loaded_from_cache = True
                    logger.info(f"[MemoryCache] cache hit, skipping clip ingestion: {cache_path}")
                    return
                except Exception as e:
                    logger.warning(f"[MemoryCache] cache load failed, will rebuild: {e}")

        # Priority 2: load the pre-built VideoGraph (offline mode)
        if self._mem_path and os.path.exists(self._mem_path):
            self.video_graph = self._load_video_graph(self._mem_path)
            logger.info(f"loading the pre-built VideoGraph: {self._mem_path}")
        else:
            # Priority 3: create an empty VideoGraph from scratch (first build in streaming mode)
            from mmagent.videograph import VideoGraph
            memory_config_path = self.config.get("memory_config_path", "configs/memory_config.json")
            if os.path.exists(memory_config_path):
                with open(memory_config_path) as f:
                    memory_config = json.load(f)
            else:
                memory_config = self.config.get("videograph", {})
            # Strip harness-level fields (such as visual_layer) to avoid passing them to VideoGraph
            self.video_graph = VideoGraph(**_filter_videograph_kwargs(memory_config))

    def on_video_end(self, video_name: str, **kwargs):
        """When video processing ends, save the built VideoGraph to the cache directory."""
        if not self._memory_cache_dir:
            return
        if self._loaded_from_cache:
            logger.debug(f"[MemoryCache] loaded from cache this time, no need to re-save: {video_name}")
            return

        cache_path = self._get_cache_path(video_name)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.video_graph, f)
            logger.info(f"[MemoryCache] saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"[MemoryCache] failed to save cache: {e}")

    @property
    def loaded_from_cache(self) -> bool:
        """Whether this video was loaded from cache (used by the orchestrator to decide whether to skip clip ingestion)."""
        return self._loaded_from_cache

    def ingest(self, clip_id: int, clip_data: ClipData) -> None:
        """Incremental ingestion: write the clip into VideoGraph.

        Uses the internalized clip_ingestion.process_clip and no longer depends on the external m3_agent module-level configuration.
        """
        process_clip(
            video_graph=self.video_graph,
            base64_video=clip_data.base64_video,
            base64_frames=clip_data.base64_frames,
            clip_id=clip_id,
            clip_path=clip_data.path,
            ingestion_config=self._ingestion_config,
        )
        self._total_clips += 1

        # Periodically refresh equivalence relations (cross-clip face cluster merging)
        if self._total_clips % self._equivalence_refresh_interval == 0:
            self.video_graph.refresh_equivalences()

    def retrieve(self, query: str, top_k: int = 5,
                 before_clip: Optional[int] = None,
                 **kwargs) -> RetrievalResult:
        """Two-layer retrieval: first locate the event with EventGraph, then perform fine-grained retrieval with VideoGraph.

        Optional third layer (visual_layer.enabled=true):
          The mode="keyframe_inspect" branch returns keyframe paths and does not go through VideoGraph retrieval.
          This branch is written as a sibling alongside the original three modes and has no effect whatsoever on the original flow.
        """
        mode = kwargs.get("mode", "event_first")

        # ---- Third layer: keyframe_inspect (handled separately before the original branches, ensuring the original logic is unchanged) ----
        if mode == "keyframe_inspect" and self._visual_enabled:
            return self._retrieve_keyframe(
                query=query,
                before_clip=before_clip,
                focus_label=kwargs.get("focus_label"),
                question_text=kwargs.get("question_text", ""),
            )

        from mmagent.retrieve import search

        event_info = None
        focus_clips = None

        # EventGraph retrieval
        if self.event_retriever and self._current_video_name:
            video_name = self.event_retriever.get_video_name(
                self._mem_path or self._current_video_name
            )
            mode = kwargs.get("mode", "event_first")
            focus_label = kwargs.get("focus_label", None)

            if mode == "neighbor" and focus_label:
                last_query = kwargs.get("last_query", query)
                focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                    video_name, focus_label, query=last_query, neighbors_topk=3,
                    before_clip=before_clip,
                )
                event_info = {
                    "mode": "event_neighbor",
                    "focus_label": focus_label,
                    "focus": focus_seg,
                    "neighbors": neighbors,
                    "edges": rel_edges,
                }
            elif mode == "video_drilldown" and focus_label:
                focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                    video_name, focus_label, query=query, neighbors_topk=3,
                    before_clip=before_clip,
                )
                focus_clips = list(focus_seg.get("clip_ids", [])) if focus_seg else []
                event_info = {
                    "mode": "video_drilldown",
                    "query": query,
                    "focus_label": focus_label,
                    "focus": focus_seg,
                    "neighbors": neighbors,
                    "edges": rel_edges,
                }
            else:
                best_seg, score = self.event_retriever.best_match(
                    video_name, query, before_clip=before_clip,
                )
                if best_seg:
                    focus_label = best_seg.get("segment_label")
                    focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                        video_name, focus_label, query=query, neighbors_topk=3,
                        before_clip=before_clip,
                    )
                    event_info = {
                        "mode": "event_first",
                        "query": query,
                        "event_score": score,
                        "focus_label": focus_label,
                        "focus": focus_seg,
                        "neighbors": neighbors,
                        "edges": rel_edges,
                    }

        # VideoGraph retrieval
        # Rely only on the internal filtering of search(before_clip=...) (pruning nodes by clip_id > before_clip),
        # and no longer call truncate_memory_by_clip, to avoid:
        #   1) the side effect that modifying nodes/edges after a shallow copy still propagates to the original graph;
        #   2) when before_clip=0 and there is no node with timestamp==0, truncate silently fails, duplicating the search filtering.
        mem_node = self.video_graph
        mem_node.refresh_equivalences()

        is_drilldown = focus_clips is not None

        # Retrieval mode selection (aligned with the strategy in control_api.py):
        #   - queries involving a character -> mem_wise=True (by node similarity, precisely matching character information)
        #   - other queries -> mem_wise=False (clip_wise, returns full clip content + deduplication)
        use_mem_wise = self._is_character_query(query)

        if use_mem_wise:
            # character query: mem_wise mode, sorted by node similarity.
            # Key point: character identity is global information (a character may be named in any earlier clip),
            # so allowed_clips is not restricted even in VIDEO drilldown mode,
            # and only the before_clip temporal constraint is kept (no leaking of future information).
            effective_topk = self._mem_wise_topk if is_drilldown else (top_k or self._topk)
            memories, _, _ = search(
                mem_node, query, [],
                threshold=0.5, mem_wise=True,
                topk=effective_topk,
                before_clip=before_clip,
                allowed_clips=None,
            )
        else:
            # non-character query: clip_wise mode, returns full clip content + deduplication
            # Aligned with control_api.py: always use self._topk (default 2), unaffected by the caller's top_k parameter
            effective_topk = self._topk
            memories, self._current_clips, _ = search(
                mem_node, query, self._current_clips,
                threshold=0.5, mem_wise=False,
                topk=effective_topk,
                before_clip=before_clip,
                allowed_clips=focus_clips,
            )

        return RetrievalResult(
            event_info=event_info,
            memories=memories,
            source="hierarchical",
        )

    # ------------------------------------------------------------------
    # Helper method: determine whether the query involves a character
    # ------------------------------------------------------------------
    @staticmethod
    def _is_character_query(query: str) -> bool:
        """Determine whether the query involves a character (character ID mapping).

        Aligned with the logic in control_api.py:
          - contains "character id" -> True (querying the character ID mapping)
          - contains "<character_" -> True (using a character placeholder)
          - contains "character name" / "name of character" -> True
          - otherwise -> False (use clip_wise retrieval)
        """
        q_lower = query.lower()
        if "character id" in q_lower:
            return True
        if "<character_" in q_lower:
            return True
        if "character name" in q_lower or "name of character" in q_lower:
            return True
        if "name of <character" in q_lower:
            return True
        return False

    # ------------------------------------------------------------------
    # Third-layer retrieval: keyframe_inspect (soft downgrade)
    # ------------------------------------------------------------------
    def _retrieve_keyframe(self, query: str,
                           before_clip: Optional[int] = None,
                           focus_label: Optional[str] = None,
                           question_text: str = "") -> RetrievalResult:
        """KEYFRAME branch: soft downgrade + path-only return.

        Soft downgrade: if the caller does not pass focus_label, internally run event_retriever.best_match once to locate
        the focus (consistent with control_api_triple_v2), rather than hard-rejecting.

        The returned event_info structure is aligned with VIDEO drilldown but adds keyframe_count / keyframe_clip_scope;
        the actual list of paths is placed in RetrievalResult.raw_payload["keyframe_paths"]
        and handed to the reasoning/multi_round main thread to load base64.
        """
        # When the EventGraph retriever is not enabled, return an empty result directly (let guardrail/feedback lower the priority)
        if not self.event_retriever or not self._current_video_name:
            return RetrievalResult(
                event_info={
                    "mode": "keyframe_inspect",
                    "query": query,
                    "focus_label": focus_label,
                    "focus": None,
                    "neighbors": [],
                    "edges": [],
                    "keyframe_clip_scope": [],
                    "keyframe_count": 0,
                    "keyframe_reason": "event_retriever_or_video_missing",
                },
                memories=None,
                source="hierarchical",
                raw_payload={"keyframe_paths": []},
            )

        video_name = self.event_retriever.get_video_name(
            self._mem_path or self._current_video_name
        )

        # Soft downgrade: when there is no focus, run best_match once first
        downgrade_used = False
        if not focus_label:
            try:
                best_seg, _score = self.event_retriever.best_match(
                    video_name, query, before_clip=before_clip,
                )
                if best_seg:
                    focus_label = best_seg.get("segment_label")
                    downgrade_used = True
            except Exception as exc:
                logger.warning(f"[Keyframe] best_match downgrade failed: {exc}")

        focus_seg = None
        neighbors: List[Any] = []
        rel_edges: List[Any] = []
        focus_clips: List[int] = []
        if focus_label:
            try:
                focus_seg, neighbors, rel_edges = self.event_retriever.one_hop(
                    video_name, focus_label, query=query, neighbors_topk=3,
                    before_clip=before_clip,
                )
                if focus_seg:
                    focus_clips = list(focus_seg.get("clip_ids", []))
            except Exception as exc:
                logger.warning(f"[Keyframe] one_hop failed: {exc}")

        # Load path-only: capped at max_images_per_call
        max_images = int(self._visual_cfg.get("max_images_per_call", 5) or 5)
        kf_paths = self._load_keyframes_for_clips(
            video_name, focus_clips, max_images=max_images,
        )

        event_info = {
            "mode": "keyframe_inspect",
            "query": query,
            "focus_label": focus_label,
            "focus": focus_seg,
            "neighbors": neighbors,
            "edges": rel_edges,
            "keyframe_clip_scope": focus_clips,
            "keyframe_count": len(kf_paths),
        }
        if downgrade_used:
            event_info["keyframe_downgrade"] = True

        return RetrievalResult(
            event_info=event_info,
            memories=None,
            source="hierarchical",
            raw_payload={"keyframe_paths": kf_paths},
        )

    def snapshot(self) -> MemorySnapshot:
        """Export a snapshot of the current memory state."""
        data = pickle.dumps(self.video_graph)
        return MemorySnapshot(
            clip_id=self._total_clips,
            data=data,
            stats=self.stats(),
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """Restore the memory state from a snapshot."""
        self.video_graph = pickle.loads(snapshot.data)
        self._total_clips = snapshot.clip_id

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the memory system."""
        vg = self.video_graph
        total_nodes = len(getattr(vg, 'text_nodes', []))
        return {
            "total_nodes": total_nodes,
            "total_clips": self._total_clips,
            "video_name": self._current_video_name,
            "has_eventgraph": self.event_retriever is not None,
        }
