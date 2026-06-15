import json
import re
import logging
import random
from .utils.chat_qwen_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
import numpy as np
import os

class EventGraphRetriever:
    def __init__(
        self,
        eventgraph_dir: str,
        emb_cache_dir: str,
        embedding_model: str = "text-embedding-v4",
        neighbors_topk: int = 3,
    ):
        self.eventgraph_dir = eventgraph_dir
        self.emb_cache_dir = emb_cache_dir
        os.makedirs(self.emb_cache_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.neighbors_topk = neighbors_topk

        # per-video cache: video_name -> bundle
        # bundle = {
        #   "eventgraph": dict,
        #   "segs": list[dict],
        #   "labels": list[str],
        #   "label2idx": dict[str,int],
        #   "emb": np.ndarray (N,D)
        # }
        self._cache = {}

        # small per-process query embedding cache (optional)
        # key: (video_name, query) -> np.ndarray
        self._q_cache = {}

    # ---------- io ----------
    def _video_name_from_path(self, path: str) -> str:
        """Extract the video name from an arbitrary path (drop directory and extension)."""
        base = os.path.basename(path)
        return os.path.splitext(base)[0]

    # backward-compatible alias
    _video_name_from_mem_path = _video_name_from_path

    def _load_eventgraph_json(self, video_name: str, event_path: str | None = None) -> dict:
        """Load the EventGraph JSON.
        If event_path is provided, use that path directly first;
        otherwise fall back to looking up by video_name under eventgraph_dir.
        """
        if event_path:
            # event_path points directly to the EventGraph json file
            if os.path.exists(event_path):
                with open(event_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            # also try event_path + ".json"
            if not event_path.endswith(".json") and os.path.exists(event_path + ".json"):
                with open(event_path + ".json", "r", encoding="utf-8") as f:
                    return json.load(f)
        # fallback: look up by video_name under the global eventgraph_dir
        p1 = os.path.join(self.eventgraph_dir, video_name)
        p2 = p1 + ".json"
        path = p1 if os.path.exists(p1) else p2
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"EventGraph file not found: tried event_path={event_path}, "
                f"fallback {p1} or {p2}"
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _emb_cache_path(self, video_name: str) -> str:
        # you may also include embedding_model here to avoid conflicts between different models
        return os.path.join(self.emb_cache_dir, f"{video_name}.npz")

    # ---------- embeddings ----------
    def _embed_query(self, query: str) -> np.ndarray:
        # use the existing get_embedding_with_retry
        qv = get_embedding_with_retry(self.embedding_model, query)[0]
        return np.asarray(qv, dtype=np.float32)

    def _build_or_load_embeddings(self, video_name: str, eventgraph: dict):
        """
        Return a bundle: labels, segs, emb, label2idx
        """
        cache_path = self._emb_cache_path(video_name)

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            labels = data["labels"].tolist()
            emb = data["emb"].astype(np.float32)

            seg_map = {s.get("segment_label"): s for s in eventgraph.get("segments", [])}
            segs = [seg_map.get(l) for l in labels]

            # lightweight filtering: graph updates may make a label no longer exist
            keep = [(l, s, i) for i, (l, s) in enumerate(zip(labels, segs)) if s is not None]
            if len(keep) != len(labels):
                labels = [x[0] for x in keep]
                segs = [x[1] for x in keep]
                emb = emb[[x[2] for x in keep], :]

            label2idx = {l: i for i, l in enumerate(labels)}
            return labels, segs, emb, label2idx

        segs = eventgraph.get("segments", [])
        labels = [s.get("segment_label", "") for s in segs]
        texts = [(s.get("segment_label", "") + "\n" + s.get("summary", "")).strip() for s in segs]

        # use the existing parallel_get_embedding to encode all segments at once
        emb = parallel_get_embedding(self.embedding_model, texts)[0]
        emb = np.asarray(emb, dtype=np.float32)

        np.savez(cache_path, labels=np.array(labels, dtype=object), emb=emb)
        label2idx = {l: i for i, l in enumerate(labels)}
        return labels, segs, emb, label2idx

    # ---------- cache ----------
    def _ensure_video_loaded(self, video_name: str, event_path: str | None = None):
        if video_name in self._cache:
            return

        eventgraph = self._load_eventgraph_json(video_name, event_path=event_path)
        labels, segs, emb, label2idx = self._build_or_load_embeddings(video_name, eventgraph)

        self._cache[video_name] = {
            "eventgraph": eventgraph,
            "labels": labels,
            "segs": segs,
            "emb": emb,
            "label2idx": label2idx,
        }

    def get_video_name(self, path: str) -> str:
        """Extract the video name from a path. Accepts mem_path or event_path."""
        return self._video_name_from_path(path)

    # ---------- before_clip helpers ----------
    @staticmethod
    def _segment_visible(seg: dict, before_clip: int | None) -> bool:
        """Decide whether a segment has already occurred by the `before_clip` cutoff (at least one clip_id <= before_clip).

        Semantics: M3-Bench's `before_clip` means "the question should be answerable after seeing clip number before_clip",
        so a segment is considered "visible" as long as any one of its clip_ids is no later than before_clip.
        When before_clip is None or negative, there is no limit and all are visible.
        """
        if before_clip is None or before_clip < 0:
            return True
        if not seg:
            return False
        clip_ids = seg.get("clip_ids") or []
        if not clip_ids:
            # compatible with old data that lacks the clip_ids field: conservatively treat as visible to avoid accidental deletion
            return True
        try:
            return any(int(c) <= before_clip for c in clip_ids)
        except (TypeError, ValueError):
            return True

    # ---------- retrieval ----------
    def best_match(self, video_name: str, query: str, event_path: str | None = None,
                   before_clip: int | None = None):
        """
        Return (best_seg: dict|None, score: float)

        If before_clip is given, select the most similar item only among segments whose clip_ids include one <= before_clip.
        """
        self._ensure_video_loaded(video_name, event_path=event_path)
        bundle = self._cache[video_name]
        segs = bundle["segs"]
        emb = bundle["emb"]

        if emb is None or len(segs) == 0:
            return None, -1.0

        qv = self._embed_query(query)
        sims = emb @ qv

        if before_clip is None or before_clip < 0:
            idx = int(np.argmax(sims))
            return segs[idx], float(sims[idx])

        # select the most similar item only among visible segments
        visible_mask = np.array(
            [self._segment_visible(s, before_clip) for s in segs], dtype=bool
        )
        if not visible_mask.any():
            return None, -1.0
        masked = np.where(visible_mask, sims, -np.inf)
        idx = int(np.argmax(masked))
        return segs[idx], float(sims[idx])

    def _rank_neighbors_by_query(self, video_name: str, query: str, neighbors: list[dict],
                                 before_clip: int | None = None):
        """
        Rank neighbors directly using the global emb looked up by label, without re-embedding the neighbors.
        If before_clip is given, neighbors that only appear after the cutoff are filtered out first.
        """
        if not neighbors:
            return neighbors

        # temporal filtering: keep only neighbors that appeared before the before_clip cutoff
        if before_clip is not None and before_clip >= 0:
            neighbors = [n for n in neighbors if self._segment_visible(n, before_clip)]
            if not neighbors:
                return neighbors

        self._ensure_video_loaded(video_name)
        bundle = self._cache[video_name]
        emb = bundle["emb"]
        label2idx = bundle["label2idx"]

        if emb is None or emb.size == 0:
            return neighbors

        # optional: query embedding cache (reused within the same round or across rounds)
        qk = (video_name, query)
        if qk in self._q_cache:
            qv = self._q_cache[qk]
        else:
            qv = self._embed_query(query)
            self._q_cache[qk] = qv

        scored = []
        for seg in neighbors:
            lab = seg.get("segment_label")
            if not lab:
                continue
            idx = label2idx.get(lab)
            if idx is None:
                continue
            score = float(emb[idx] @ qv)
            scored.append((score, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [seg for _, seg in scored]

    def one_hop(self, video_name: str, focus_label: str, query: str = "", neighbors_topk: int | None = None,
                event_path: str | None = None, before_clip: int | None = None):
        """
        Return (focus_seg, neighbors_topk_sorted, filtered_edges).

        If before_clip is given, remove all neighbor segments whose clip_ids are all later than before_clip,
        and also drop edges that touch an invisible segment on either end (to prevent leaking future events to the agent).
        """
        self._ensure_video_loaded(video_name, event_path=event_path)
        eventgraph = self._cache[video_name]["eventgraph"]

        seg_map = {s.get("segment_label"): s for s in eventgraph.get("segments", [])}
        edges_all = eventgraph.get("edges", [])
        focus_seg = seg_map.get(focus_label)
        if focus_seg is None:
            return None, [], []

        nbr_labels, rel_edges = set(), []
        for e in edges_all:
            src, dst = e.get("from"), e.get("to")
            if src == focus_label or dst == focus_label:
                rel_edges.append(e)
                if src and src != focus_label:
                    nbr_labels.add(src)
                if dst and dst != focus_label:
                    nbr_labels.add(dst)

        neighbors_all = [seg_map[l] for l in nbr_labels if l in seg_map]

        # temporal filtering: remove neighbors that appear only after before_clip
        if before_clip is not None and before_clip >= 0:
            neighbors_all = [n for n in neighbors_all if self._segment_visible(n, before_clip)]

        k = self.neighbors_topk if neighbors_topk is None else neighbors_topk
        neighbors = neighbors_all
        if query and k is not None and len(neighbors_all) > k:
            ranked = self._rank_neighbors_by_query(
                video_name, query, neighbors_all, before_clip=before_clip,
            )
            neighbors = ranked[:k]

        keep = set([focus_label] + [n.get("segment_label") for n in neighbors if n.get("segment_label")])
        filtered_edges = [e for e in rel_edges if e.get("from") in keep and e.get("to") in keep]
        return focus_seg, neighbors, filtered_edges
