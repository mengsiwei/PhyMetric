#!/usr/bin/env python3
"""
Analyze human annotation results.

Outputs:
  - Per-annotator accuracy vs ground truth
  - Inter-annotator agreement (Cohen's kappa, pairwise)
  - Human ensemble accuracy (majority vote)
  - Comparison vs VLM eval_summary.json (if provided)

Usage:
    python analyze.py \
        --annotations_dir annotations \
        --question_file   ../data/question_v2_with_kb.json \
        [--vlm_summary    ../output100/Wan2.1-T2V-14B/baseline/eval_summary.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_questions(question_file: str) -> dict[str, dict]:
    """Returns {answer_key: {ground_truth, difficulty, dimensions, phenomenon}}"""
    import re
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = {}
    for phenomenon, scenes in data.get("main_categories", {}).items():
        for scene_key, scene_data in scenes.items():
            scene_id = re.sub(r"[^\w\-.]", "_", f"{phenomenon}_{scene_key}")
            for q in scene_data.get("questions", []):
                key = f"{scene_id}__{q['id']}"
                records[key] = {
                    "ground_truth": "yes" if q.get("answer", 1) == 1 else "no",
                    "difficulty":   q.get("difficulty", "easy"),
                    "dimensions":   q.get("dimensions", []),
                    "phenomenon":   phenomenon,
                }
    return records


def load_all_annotations(annotations_dir: str) -> dict[str, dict[str, str]]:
    """Returns {annotator_id: {answer_key: answer}}"""
    ann_dir = Path(annotations_dir)
    result = {}
    for f in sorted(ann_dir.glob("*.json")):
        with open(f) as fp:
            ann = json.load(fp)
        annotator = ann["annotator_id"]
        result[annotator] = {
            k: v["answer"] for k, v in ann.get("answers", {}).items()
        }
    return result


def cohen_kappa(a1: list[str], a2: list[str]) -> float:
    """Compute Cohen's kappa for two annotators over shared questions."""
    assert len(a1) == len(a2)
    n = len(a1)
    if n == 0:
        return float("nan")
    agree = sum(x == y for x, y in zip(a1, a2)) / n
    labels = set(a1) | set(a2)
    p_e = sum(
        (a1.count(l) / n) * (a2.count(l) / n)
        for l in labels
    )
    if p_e == 1.0:
        return 1.0
    return (agree - p_e) / (1 - p_e)


def majority_vote(votes: list[str]) -> str:
    return "yes" if votes.count("yes") >= votes.count("no") else "no"


def accuracy(preds: list[str], gts: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, gts)) / len(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", default="annotations")
    parser.add_argument("--question_file",   default="../data/question_v2_with_kb.json")
    parser.add_argument("--vlm_summary",     default=None,
                        help="Path to VLM eval_summary.json for comparison")
    args = parser.parse_args()

    questions  = load_questions(args.question_file)
    annotators = load_all_annotations(args.annotations_dir)

    if not annotators:
        print("No annotation files found.")
        return

    print(f"\n{'='*60}")
    print(f"  PhyMetric Human Annotation Analysis")
    print(f"  Questions: {len(questions)} total")
    print(f"  Annotators: {', '.join(annotators)}")
    print(f"{'='*60}")

    # ── per-annotator accuracy ────────────────────────────────────────────────
    print("\n── Per-annotator accuracy vs ground truth ──")
    print(f"  {'Annotator':<15} {'Answered':>8} {'Overall':>8} {'Hard':>8} {'Easy':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    per_ann: dict[str, dict] = {}
    for ann_id, answers in annotators.items():
        keys = [k for k in answers if k in questions]
        gts  = [questions[k]["ground_truth"] for k in keys]
        preds = [answers[k] for k in keys]

        hard_keys = [k for k in keys if questions[k]["difficulty"] == "hard"]
        easy_keys = [k for k in keys if questions[k]["difficulty"] == "easy"]
        hard_acc = accuracy([answers[k] for k in hard_keys], [questions[k]["ground_truth"] for k in hard_keys])
        easy_acc = accuracy([answers[k] for k in easy_keys], [questions[k]["ground_truth"] for k in easy_keys])
        oa = accuracy(preds, gts)
        per_ann[ann_id] = {"keys": keys, "preds": preds, "gts": gts, "overall": oa}
        print(f"  {ann_id:<15} {len(keys):>8} {oa:>8.3f} {hard_acc:>8.3f} {easy_acc:>8.3f}")

    # ── pairwise agreement ────────────────────────────────────────────────────
    ann_ids = list(annotators.keys())
    if len(ann_ids) >= 2:
        print("\n── Pairwise Cohen's Kappa ──")
        for i in range(len(ann_ids)):
            for j in range(i + 1, len(ann_ids)):
                id1, id2 = ann_ids[i], ann_ids[j]
                shared = [k for k in annotators[id1] if k in annotators[id2] and k in questions]
                a1 = [annotators[id1][k] for k in shared]
                a2 = [annotators[id2][k] for k in shared]
                kappa = cohen_kappa(a1, a2)
                agree_pct = sum(x == y for x, y in zip(a1, a2)) / len(a1) * 100 if a1 else 0
                print(f"  {id1} vs {id2}: κ={kappa:.3f}  agree={agree_pct:.1f}%  shared={len(shared)}")

    # ── majority vote ensemble ────────────────────────────────────────────────
    print("\n── Human Ensemble (majority vote) ──")
    all_keys = sorted(questions.keys())
    ensemble_preds, ensemble_gts, ensemble_keys = [], [], []
    for k in all_keys:
        votes = [annotators[a][k] for a in ann_ids if k in annotators[a]]
        if votes:
            ensemble_preds.append(majority_vote(votes))
            ensemble_gts.append(questions[k]["ground_truth"])
            ensemble_keys.append(k)

    ens_acc = accuracy(ensemble_preds, ensemble_gts)
    print(f"  Coverage : {len(ensemble_keys)}/{len(all_keys)} questions")
    print(f"  Accuracy : {ens_acc:.4f}  ({sum(p==g for p,g in zip(ensemble_preds,ensemble_gts))}/{len(ensemble_preds)})")

    # per-difficulty
    for diff in ["easy", "hard"]:
        idxs = [i for i, k in enumerate(ensemble_keys) if questions[k]["difficulty"] == diff]
        if idxs:
            acc = sum(ensemble_preds[i] == ensemble_gts[i] for i in idxs) / len(idxs)
            print(f"  {diff:<6}   : {acc:.4f}  ({sum(ensemble_preds[i]==ensemble_gts[i] for i in idxs)}/{len(idxs)})")

    # per-dimension
    dim_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, k in enumerate(ensemble_keys):
        for d in questions[k]["dimensions"]:
            dim_stats[d]["total"] += 1
            if ensemble_preds[i] == ensemble_gts[i]:
                dim_stats[d]["correct"] += 1
    print("\n  Per-dimension:")
    for d in sorted(dim_stats):
        s = dim_stats[d]
        print(f"    {d:<22} {s['correct']/s['total']:.4f}  ({s['correct']}/{s['total']})")

    # ── compare with VLM ─────────────────────────────────────────────────────
    if args.vlm_summary and Path(args.vlm_summary).exists():
        with open(args.vlm_summary) as f:
            vlm = json.load(f)
        print("\n── Human Ensemble vs VLM ──")
        print(f"  {'Metric':<35} {'Human':>8} {'VLM':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8}")
        metrics = [
            ("Overall accuracy",         ens_acc,                          vlm.get("overall_accuracy", 0)),
            ("Difficulty-weighted acc",  None,                             vlm.get("difficulty_weighted_accuracy", 0)),
            ("Dimension-weighted acc",   None,                             vlm.get("dimension_weighted_accuracy", 0)),
            ("Combined-weighted acc",    None,                             vlm.get("combined_weighted_accuracy", 0)),
        ]
        for name, h_val, v_val in metrics:
            h_str = f"{h_val:.4f}" if h_val is not None else "  —   "
            print(f"  {name:<35} {h_str:>8} {v_val:>8.4f}")

    # ── save summary ──────────────────────────────────────────────────────────
    out_path = Path(args.annotations_dir) / "human_summary.json"
    summary = {
        "per_annotator": {
            ann_id: {"overall_accuracy": round(d["overall"], 4), "n_answered": len(d["keys"])}
            for ann_id, d in per_ann.items()
        },
        "ensemble": {
            "n_questions":       len(ensemble_keys),
            "overall_accuracy":  round(ens_acc, 4),
            "per_dimension":     {
                d: round(s["correct"] / s["total"], 4)
                for d, s in dim_stats.items()
            },
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
