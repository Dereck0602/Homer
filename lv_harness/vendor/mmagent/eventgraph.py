from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
import json


@dataclass
class SegmentNode:
    segment_label: str
    summary: str
    clip_ids: List[int]


@dataclass(frozen=True)
class RelationEdge:
    src: str
    dst: str
    relation_type: str
    relation: str
    evidence_clips: Tuple[int, ...]
    explicitness: str
    rationale: str
    confidence: Optional[float] = None  # new: allow None
    remove: bool = False                # new: used by patch to delete an edge

    def key(self) -> Tuple[str, str, str, str]:
        return (self.src, self.dst, self.relation_type, self.relation)


class EventGraph:
    def __init__(self):
        self.segments: Dict[str, SegmentNode] = {}            # label -> node
        self.edges: Dict[Tuple[str, str, str, str], RelationEdge] = {}  # edge_key -> edge

        # index: clip_id -> set(labels)
        self.segment_labels_by_clip: Dict[int, Set[str]] = {}

        # optional: keep each LLM raw output for traceback
        self.llm_history: List[Dict[str, Any]] = []

    # ------------------ internal helpers ------------------

    def _index_add_segment(self, node: SegmentNode) -> None:
        for cid in node.clip_ids:
            self.segment_labels_by_clip.setdefault(cid, set()).add(node.segment_label)

    def _index_remove_segment(self, label: str) -> None:
        node = self.segments.get(label)
        if not node:
            return
        for cid in node.clip_ids:
            s = self.segment_labels_by_clip.get(cid)
            if not s:
                continue
            s.discard(label)
            if not s:
                self.segment_labels_by_clip.pop(cid, None)

    def _remove_edges_touching(self, label: str) -> None:
        # delete all edges with src=label or dst=label
        to_del = [k for k, e in self.edges.items() if e.src == label or e.dst == label]
        for k in to_del:
            self.edges.pop(k, None)

    def delete_segment(self, label: str) -> None:
        if label not in self.segments:
            return
        self._remove_edges_touching(label)
        self._index_remove_segment(label)
        self.segments.pop(label, None)

    def add_segment(self, node: SegmentNode) -> None:
        self.segments[node.segment_label] = node
        self._index_add_segment(node)

    def add_edge_if_valid(self, edge: RelationEdge) -> None:
        # only add when both endpoints exist
        if edge.src not in self.segments or edge.dst not in self.segments:
            return
        self.edges.setdefault(edge.key(), edge)

    # ------------------ public api ------------------

    def update_from_llm_output(self, llm_out: Dict[str, Any]) -> None:
        """
        Merge the LLM PATCH output and perform global subset replacement.
        """
        self.llm_history.append(llm_out)

        # ------------------ 1) merge segments ------------------
        for s in llm_out.get("segments", []):
            label = s["segment_label"]
            new_clip_ids = set(s.get("clip_ids", []))
            new_summary = s.get("summary", "")

            if label not in self.segments:
                # add
                node = SegmentNode(label, new_summary, sorted(new_clip_ids))
                self.add_segment(node)
            else:
                # update: union the clips; use the new summary
                old_node = self.segments[label]
                merged_clips = sorted(set(old_node.clip_ids) | new_clip_ids)

                self._index_remove_segment(label)
                self.segments[label] = SegmentNode(
                    label,
                    new_summary if new_summary else old_node.summary,
                    merged_clips,
                )
                self._index_add_segment(self.segments[label])

        # ------------------ 2) global subset replacement ------------------
        labels = list(self.segments.keys())
        to_remove = set()

        for a in labels:
            set_a = set(self.segments[a].clip_ids)
            for b in labels:
                if a == b:
                    continue
                set_b = set(self.segments[b].clip_ids)
                if set_a and set_a < set_b:  # strict subset
                    to_remove.add(a)
                    break

        for label in to_remove:
            self.delete_segment(label)

        # ------------------ 3) merge edges ------------------
        for e in llm_out.get("edges", []):
            key = (e["from"], e["to"], e.get("relation_type", ""), e.get("relation", ""))

            if e.get("remove", False):
                self.edges.pop(key, None)
                continue

            if e["from"] not in self.segments or e["to"] not in self.segments:
                continue

            edge = RelationEdge(
                src=e["from"],
                dst=e["to"],
                relation_type=e.get("relation_type", ""),
                relation=e.get("relation", ""),
                evidence_clips=tuple(e.get("evidence_clips", [])),
                explicitness=e.get("explicitness", ""),
                rationale=e.get("rationale", ""),
            )

            self.edges[key] = edge
    # ------------------ retrieval ------------------

    def get_segment(self, label: str) -> Optional[SegmentNode]:
        return self.segments.get(label)

    def find_segments_by_clip(self, clip_id: int) -> List[SegmentNode]:
        labels = self.segment_labels_by_clip.get(clip_id, set())
        return [self.segments[l] for l in labels if l in self.segments]

    def neighbors(self, label: str, relation_type: Optional[str] = None, direction: str = "out"):
        out = []
        for edge in self.edges.values():
            if direction == "out" and edge.src == label:
                if relation_type is None or edge.relation_type == relation_type:
                    out.append((self.segments[edge.dst], edge))
            if direction == "in" and edge.dst == label:
                if relation_type is None or edge.relation_type == relation_type:
                    out.append((self.segments[edge.src], edge))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [asdict(s) for s in self.segments.values()],
            "edges": [
                {
                    "from": e.src,
                    "to": e.dst,
                    "relation_type": e.relation_type,
                    "relation": e.relation,
                    "evidence_clips": list(e.evidence_clips),
                    "explicitness": e.explicitness,
                    "rationale": e.rationale,
                }
                for e in self.edges.values()
            ],
        }

    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)
