#!/usr/bin/env python3
"""
Compute accuracy by type.

Usage:
    python scripts/eval_accuracy_by_type.py \
        --result_file <path_to_result.jsonl> \
        --annotation_file <path_to_annotation.json>

Notes:
    - result_file: jsonl format, each line contains the "id" and "gpt_eval" (bool) fields
    - annotation_file: json format, structured as dict[video_id] -> {qa_list: [{question_id, type: [...], ...}]}
    - For questions belonging to multiple types, they are counted under each type
"""

import argparse
import json
from collections import defaultdict


def load_results(result_file: str) -> dict:
    """Load the jsonl result file, returning a {question_id: gpt_eval} mapping."""
    results = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("id") or obj.get("question_id")
                if qid is None:
                    continue
                # Compatible with both correctness fields
                is_correct = obj.get("gpt_eval", obj.get("is_correct", False))
                results[qid] = bool(is_correct)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def load_annotations(annotation_file: str) -> dict:
    """Load the annotation file, returning a {question_id: [type1, type2, ...]} mapping."""
    qid_to_types = {}
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video_id, video_info in data.items():
        for qa in video_info.get("qa_list", []):
            qid = qa.get("question_id")
            types = qa.get("type", [])
            if qid:
                qid_to_types[qid] = types if isinstance(types, list) else [types]

    return qid_to_types


def compute_accuracy_by_type(results: dict, qid_to_types: dict):
    """Compute accuracy by type."""
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    matched = 0
    for qid, is_correct in results.items():
        types = qid_to_types.get(qid, [])
        if not types:
            # Questions without type information go to "Unknown"
            types = ["Unknown"]
        matched += 1
        for t in types:
            type_total[t] += 1
            if is_correct:
                type_correct[t] += 1

    return type_correct, type_total, matched


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy by type")
    parser.add_argument("--result_file", type=str, required=True,
                        help="path to the jsonl result file")
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="path to the json annotation file")
    args = parser.parse_args()

    # Load data
    results = load_results(args.result_file)
    qid_to_types = load_annotations(args.annotation_file)

    # Compute
    type_correct, type_total, matched = compute_accuracy_by_type(results, qid_to_types)

    # Output
    print("=" * 70)
    print(f"Result file: {args.result_file}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Total results: {len(results)}, questions matched to a type: {matched}")
    print("=" * 70)
    print(f"{'Type':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 70)

    # Sort by accuracy descending
    all_types = sorted(type_total.keys(), key=lambda t: type_correct[t] / type_total[t] if type_total[t] > 0 else 0, reverse=True)

    total_correct = 0
    total_count = 0
    for t in all_types:
        correct = type_correct[t]
        total = type_total[t]
        acc = correct / total if total > 0 else 0
        print(f"{t:<35} {correct:>8} {total:>8} {acc:>9.2%}")
        total_correct += correct
        total_count += total

    print("-" * 70)
    # Overall accuracy (computed deduplicated per question)
    overall_correct = sum(1 for v in results.values() if v)
    overall_total = len(results)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"{'Overall (per question)':<35} {overall_correct:>8} {overall_total:>8} {overall_acc:>9.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
