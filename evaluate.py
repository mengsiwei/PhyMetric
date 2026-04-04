#!/usr/bin/env python3
"""
PhyMetric VLM Evaluation Pipeline (3 steps)

Step a — Frame sampling
    For each video, sample frames at each question's frame_indices.
    Save to:  {frames_dir}/{video_stem}/frame_{idx:04d}.jpg

Step b — VLM QA
    For each question, pass sampled frames + knowledge_base + question to QwenVL.
    VLM outputs: ANSWER: YES/NO | CONFIDENCE: 0.0-1.0 | REASON: ...
    Save all results to:  {output_dir}/qa_results.json

Step c — Scoring
    Compute: overall accuracy, difficulty-weighted accuracy,
             per-dimension accuracy, per-phenomenon accuracy.
    Save to:  {output_dir}/eval_summary.json  +  print table.

Usage:
    # Full pipeline (a + b + c)
    python evaluate.py \\
        --question_file data/question_v2_with_kb.json \\
        --video_dir     /path/to/videos \\
        --output_dir    output100/my_run \\
        --vlm_model_id  Qwen/Qwen2.5-VL-7B-Instruct \\
        --gpu 0

    # Step a only (frame extraction, no GPU needed)
    python evaluate.py --steps a \\
        --question_file data/question_v2_with_kb.json \\
        --video_dir     /path/to/videos \\
        --output_dir    output100/my_run

    # Steps b+c only (frames already extracted)
    python evaluate.py --steps bc \\
        --question_file data/question_v2_with_kb.json \\
        --output_dir    output100/my_run \\
        --vlm_model_id  Qwen/Qwen2.5-VL-7B-Instruct \\
        --gpu 0

    # Step c only (re-score existing qa_results.json)
    python evaluate.py --steps c --output_dir output100/my_run

Video naming convention (matches the phymetric run scripts):
    prompt-{idx}.mp4   where idx is the 0-based scene order in question_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _safe_scene_id(category: str, scene_key: str) -> str:
    return re.sub(r"[^\w\-.]", "_", f"{category}_{scene_key}")


def load_scenes(question_file: str) -> tuple[list[dict], float, int]:
    """
    Returns (scenes, q_fps, q_total_frames).
    q_fps and q_total_frames come from metadata and define the time-space
    in which frame_indices are expressed.
    """
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    q_fps   = float(meta.get("fps", 30))
    q_total = int(meta.get("total_frames", 60))
    scenes = []
    for phenomenon, scene_dict in data.get("main_categories", {}).items():
        for scene_key, scene_data in scene_dict.items():
            scenes.append({
                "idx": len(scenes),
                "phenomenon": phenomenon,
                "scene_id": _safe_scene_id(phenomenon, scene_key),
                "scene_name": scene_data.get("scene_name", scene_key),
                "scene_caption": scene_data.get("scene_caption", ""),
                "questions": scene_data.get("questions", []),
                "knowledge_base": scene_data.get("knowledge_base", {}),
            })
    return scenes, q_fps, q_total


# =============================================================================
# Step a — Frame sampling
# =============================================================================

def map_question_indices(
    q_indices: list[int],
    q_fps: float,
    q_total: int,
    v_fps: float,
    v_total: int,
) -> dict[int, int]:
    """
    Map question frame indices (defined for a q_total-frame benchmark clip)
    to actual frame indices in a video with v_total frames.

    Mapping is proportional by frame position (not time-based), so that
    the last benchmark frame always maps to the last actual frame:
        actual = round( q_idx / (q_total - 1) * (v_total - 1) )

    Returns {q_idx: actual_frame_idx}.
    """
    result = {}
    denom = max(q_total - 1, 1)
    for q_idx in q_indices:
        actual = int(round(q_idx / denom * (v_total - 1)))
        actual = max(0, min(actual, v_total - 1))
        result[q_idx] = actual
    return result


def extract_frames(
    video_path: str,
    q_index_to_actual: dict[int, int],
    out_dir: Path,
) -> dict[int, list[str]]:
    """
    For each q_idx → center frame, extract (center-1, center, center+1).
    Saves as frame_{actual_idx:04d}.jpg.
    Also writes frame_map.json: {q_idx_str: [path, path, path]}.
    Returns {q_idx: [paths_list]}.
    """
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # collect unique actual indices to extract (prev / center / next)
    actual_set: set[int] = set()
    for center in q_index_to_actual.values():
        for a in (center - 1, center, center + 1):
            if 0 <= a < v_total:
                actual_set.add(a)

    # extract each unique frame once
    extracted: set[int] = set()
    for a in sorted(actual_set):
        out_path = out_dir / f"frame_{a:04d}.jpg"
        if out_path.exists():
            extracted.add(a)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.add(a)
    cap.release()

    # build per-q_idx path lists and frame_map.json
    result: dict[int, list[str]] = {}
    for q_idx, center in q_index_to_actual.items():
        neighbors = [a for a in (center - 1, center, center + 1)
                     if 0 <= a < v_total and a in extracted]
        result[q_idx] = [str(out_dir / f"frame_{a:04d}.jpg") for a in neighbors]

    frame_map = {str(q_idx): paths for q_idx, paths in result.items()}
    with open(out_dir / "frame_map.json", "w", encoding="utf-8") as f:
        json.dump(frame_map, f, indent=2)

    return result


def get_video_info(video_path: str) -> tuple[float, int]:
    """Returns (fps, total_frames) of a video file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def step_a(
    scenes: list[dict],
    video_dir: Path,
    frames_dir: Path,
    q_fps: float = 30.0,
    q_total: int = 60,
):
    """
    For each scene, map question frame_indices from benchmark time-space
    (q_fps / q_total) to the actual video's time-space, then extract.

    frame_{q_idx:04d}.jpg uses the benchmark index as filename so that
    step_b can look up frames by the original question index.
    """
    print("\n" + "=" * 60)
    print("Step a: Frame extraction  "
          f"(benchmark: {q_fps}fps/{q_total}frames → mapped to actual video fps)")
    print("=" * 60)

    skipped, done = 0, 0
    for scene in scenes:
        video_path = video_dir / f"prompt-{scene['idx']}.mp4"
        if not video_path.exists():
            print(f"  [skip] {video_path.name} not found")
            skipped += 1
            continue

        v_fps, v_total = get_video_info(str(video_path))

        # collect all unique benchmark indices needed for this scene
        all_q_indices: set[int] = set()
        for q in scene["questions"]:
            all_q_indices.update(q.get("frame_indices", [10, 30, 50]))

        scene_frames_dir = frames_dir / scene["scene_id"]

        # skip if frame_map.json already covers all needed q_indices
        map_path = scene_frames_dir / "frame_map.json"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as _f:
                existing_map = json.load(_f)
            if all(str(i) in existing_map for i in all_q_indices):
                skipped += 1
                continue

        mapping = map_question_indices(
            sorted(all_q_indices), q_fps, q_total, v_fps, v_total
        )
        saved = extract_frames(str(video_path), mapping, scene_frames_dir)

        # log the mapping once per scene for transparency
        mapping_str = "  ".join(
            f"q{qi}→v{mapping[qi]}(±1)" for qi in sorted(mapping)
        )
        n_files = sum(len(v) for v in saved.values())
        print(
            f"  [{scene['idx']:3d}] {scene['scene_id']}  "
            f"video={v_fps:.0f}fps/{v_total}f  [{mapping_str}]  "
            f"saved={n_files} frames"
        )
        done += 1

    print(f"\n  Done: {done} scenes extracted, {skipped} skipped")


# =============================================================================
# Step b — VLM QA
# =============================================================================

VLM_SYSTEM_PROMPT = (
    "You are a physics-aware video evaluation assistant. "
    "You will be shown a sequence of frames from a short AI-generated video, "
    "a knowledge base describing what the video SHOULD depict physically, "
    "and a yes/no question about the video content. "
    "Reason carefully using the physical knowledge, then give your answer.\n\n"
    "Your response MUST follow this exact format (one line each):\n"
    "ANSWER: YES\n"
    "CONFIDENCE: 0.92\n"
    "REASON: <one sentence explaining your judgment>"
)


def build_vlm_prompt(scene_caption: str, knowledge_base: dict, question: dict) -> str:
    kb_text = ""
    if knowledge_base:
        seq = knowledge_base.get("expected_sequence", {})
        seq_lines = "\n".join(
            f"  • {k.replace('_', ' ').capitalize()}: {v}"
            for k, v in seq.items()
        )
        indicators = "\n".join(
            f"  • {v}" for v in knowledge_base.get("visual_indicators", [])
        )
        failures = "\n".join(
            f"  • {v}" for v in knowledge_base.get("common_failures", [])
        )
        phys_law = knowledge_base.get("physical_law", "")
        dim_hints = knowledge_base.get("evaluation_hints", {})
        q_dims = question.get("dimensions", [])
        hints = "\n".join(
            f"  • [{d}] {dim_hints[d]}"
            for d in q_dims if d in dim_hints
        )

        kb_text = f"""
--- KNOWLEDGE BASE ---
Physical law: {phys_law}

Expected sequence:
{seq_lines}

Visual indicators of correct physics:
{indicators}

Common generation failures:
{failures}

Evaluation hints for this question's dimensions:
{hints if hints else '  (see general indicators above)'}
--- END KNOWLEDGE BASE ---
"""

    frame_indices_str = ", ".join(str(i) for i in question.get("frame_indices", [10, 30, 50]))
    return (
        f"Scene caption: \"{scene_caption}\"\n"
        f"Frames shown: indices [{frame_indices_str}] out of 60 total frames (0.33s / 1.0s / 1.67s)\n"
        f"{kb_text}\n"
        f"Question (answer YES or NO): {question['question_text']}"
    )


def parse_vlm_response(response: str) -> tuple[str, float, str]:
    """Returns (answer: 'yes'/'no', confidence: 0-1, reason: str)."""
    answer = "no"
    confidence = 0.5
    reason = response.strip()

    ans_match = re.search(r"ANSWER\s*:\s*(YES|NO)", response, re.IGNORECASE)
    if ans_match:
        answer = ans_match.group(1).lower()

    conf_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", response)
    if conf_match:
        confidence = max(0.0, min(1.0, float(conf_match.group(1))))

    reason_match = re.search(r"REASON\s*:\s*(.+)", response, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return answer, confidence, reason


class QwenVLEvaluator:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor

        print(f"Loading VLM: {model_id}")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device
            ).eval()
        except Exception:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device,
                trust_remote_code=True,
            ).eval()
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._device = device

    def query(self, frame_paths: list[str], user_text: str) -> str:
        import torch
        from PIL import Image

        # Qwen2.5-VL encodes each image as visual tokens proportional to pixel count.
        # With 9 frames (3 indices × ±1 neighbors) at full resolution this easily
        # exceeds GPU memory.  Limit each image to 448×448 max and cap at 6 images.
        MAX_SIDE   = 448
        MAX_IMAGES = 6

        pil_images = []
        for p in frame_paths[:MAX_IMAGES]:
            if not Path(p).exists():
                continue
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            pil_images.append(img)

        image_content = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[1]:]
        result = self._processor.decode(generated, skip_special_tokens=True).strip()

        # free activation memory between queries
        del inputs, out
        torch.cuda.empty_cache()

        return result


def step_b(
    scenes: list[dict],
    frames_dir: Path,
    output_dir: Path,
    vlm: QwenVLEvaluator,
    resume: bool = True,
):
    """Run VLM QA for all questions. Saves qa_results.json."""
    print("\n" + "=" * 60)
    print("Step b: VLM QA evaluation")
    print("=" * 60)

    qa_path = output_dir / "qa_results.json"
    results: dict = {}

    # resume from existing
    if resume and qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Resuming from {qa_path} ({len(results)} scenes already done)")

    total_q = sum(len(s["questions"]) for s in scenes)
    done_q = 0

    for scene in scenes:
        sid = scene["scene_id"]
        scene_frames_dir = frames_dir / sid

        if sid in results and len(results[sid]) == len(scene["questions"]):
            done_q += len(scene["questions"])
            continue

        # load frame_map.json (written by step_a); fallback to legacy naming
        map_path = scene_frames_dir / "frame_map.json"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as _f:
                frame_map: dict[str, list[str]] = json.load(_f)
        else:
            frame_map = {}

        scene_results = results.get(sid, {})

        for q in scene["questions"]:
            qid = q["id"]
            if qid in scene_results:
                done_q += 1
                continue

            # collect frame paths: use frame_map (3 frames each), fallback to single frame
            frame_indices = q.get("frame_indices", [10, 30, 50])
            frame_paths: list[str] = []
            for i in frame_indices:
                mapped = frame_map.get(str(i))
                if mapped:
                    frame_paths.extend(mapped)
                else:
                    frame_paths.append(str(scene_frames_dir / f"frame_{i:04d}.jpg"))
            available = [p for p in frame_paths if Path(p).exists()]

            user_text = build_vlm_prompt(
                scene["scene_caption"],
                scene["knowledge_base"],
                q,
            )

            try:
                raw_response = vlm.query(available, user_text)
                pred_str, confidence, reason = parse_vlm_response(raw_response)
            except Exception as e:
                print(f"  ERROR {sid} q{qid}: {e}")
                pred_str, confidence, reason = "no", 0.0, str(e)

            gt = int(q.get("answer", 1))
            gt_str = "yes" if gt == 1 else "no"
            correct = (pred_str == gt_str)

            scene_results[qid] = {
                "question_text": q["question_text"],
                "dimensions": q.get("dimensions", []),
                "difficulty": q.get("difficulty", "easy"),
                "frame_indices": frame_indices,
                "ground_truth": gt_str,
                "pred": pred_str,
                "confidence": confidence,
                "correct": correct,
                "reason": reason,
            }
            done_q += 1

            status = "✓" if correct else "✗"
            print(
                f"  {status} [{sid}] q{qid} | GT={gt_str} PRED={pred_str} "
                f"conf={confidence:.2f} | {q['question_text'][:60]}"
            )

        results[sid] = scene_results

        # checkpoint after each scene
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Done: {done_q}/{total_q} questions evaluated")
    print(f"  Results saved to: {qa_path}")
    return results


# =============================================================================
# Step c — Scoring
# =============================================================================

# Dimension weights for weighted accuracy
DIMENSION_WEIGHTS = {
    "motion":           2.0,
    "direction":        2.0,
    "interaction":      2.0,
    "spatial":          1.5,
    "material":         1.5,
    "appearance/texture": 1.0,
    "shape":            1.0,
    "existence":        1.0,
    "size":             1.0,
    "number":           1.0,
    "object":           1.0,
}
DIFFICULTY_WEIGHTS = {"hard": 2.0, "easy": 1.0}


def step_c(scenes: list[dict], qa_results: dict, output_dir: Path):
    """Compute and print scoring summary."""
    print("\n" + "=" * 60)
    print("Step c: Scoring")
    print("=" * 60)

    all_q: list[dict] = []  # flat list of question result dicts + metadata

    for scene in scenes:
        sid = scene["scene_id"]
        scene_qa = qa_results.get(sid, {})
        for q in scene["questions"]:
            qid = q["id"]
            if qid not in scene_qa:
                continue
            r = scene_qa[qid]
            all_q.append({
                "scene_id": sid,
                "phenomenon": scene["phenomenon"],
                "qid": qid,
                **r,
            })

    if not all_q:
        print("  No evaluated questions found.")
        return

    n = len(all_q)
    correct_flags = [q["correct"] for q in all_q]

    # ---- overall ----
    overall_acc = float(np.mean(correct_flags))

    # ---- difficulty-weighted ----
    w_diff = [DIFFICULTY_WEIGHTS.get(q["difficulty"], 1.0) for q in all_q]
    w_diff_sum = sum(w_diff)
    weighted_diff_acc = sum(
        w * q["correct"] for w, q in zip(w_diff, all_q)
    ) / w_diff_sum if w_diff_sum > 0 else 0.0

    # ---- dimension-weighted (each question may span multiple dims) ----
    w_dim_total = 0.0
    w_dim_correct = 0.0
    for q in all_q:
        dims = q.get("dimensions", [])
        w = max(DIMENSION_WEIGHTS.get(d, 1.0) for d in dims) if dims else 1.0
        w_dim_total   += w
        w_dim_correct += w * q["correct"]
    weighted_dim_acc = w_dim_correct / w_dim_total if w_dim_total > 0 else 0.0

    # ---- combined weighted (difficulty × dimension) ----
    w_combined_total = 0.0
    w_combined_correct = 0.0
    for q in all_q:
        dims = q.get("dimensions", [])
        w_dim  = max(DIMENSION_WEIGHTS.get(d, 1.0) for d in dims) if dims else 1.0
        w_diff_i = DIFFICULTY_WEIGHTS.get(q["difficulty"], 1.0)
        w = w_dim * w_diff_i
        w_combined_total   += w
        w_combined_correct += w * q["correct"]
    weighted_combined_acc = w_combined_correct / w_combined_total if w_combined_total > 0 else 0.0

    # ---- per-dimension ----
    dim_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for q in all_q:
        for dim in q.get("dimensions", []):
            dim_stats[dim]["total"] += 1
            if q["correct"]:
                dim_stats[dim]["correct"] += 1

    # ---- per-difficulty ----
    diff_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for q in all_q:
        d = q.get("difficulty", "easy")
        diff_stats[d]["total"] += 1
        if q["correct"]:
            diff_stats[d]["correct"] += 1

    # ---- per-phenomenon ----
    phen_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for q in all_q:
        p = q["phenomenon"]
        phen_stats[p]["total"] += 1
        if q["correct"]:
            phen_stats[p]["correct"] += 1

    # ---- confidence calibration ----
    confidences = [q["confidence"] for q in all_q]
    avg_conf = float(np.mean(confidences))
    avg_conf_correct   = float(np.mean([q["confidence"] for q in all_q if     q["correct"]]))
    avg_conf_incorrect = float(np.mean([q["confidence"] for q in all_q if not q["correct"]])) \
        if any(not q["correct"] for q in all_q) else 0.0

    # ---- print ----
    print(f"\n  Total questions evaluated : {n}")
    print(f"  Overall accuracy          : {overall_acc:.4f}  ({sum(correct_flags)}/{n})")
    print(f"  Difficulty-weighted acc   : {weighted_diff_acc:.4f}")
    print(f"  Dimension-weighted acc    : {weighted_dim_acc:.4f}")
    print(f"  Combined-weighted acc     : {weighted_combined_acc:.4f}  ← primary metric")

    print(f"\n  Confidence: avg={avg_conf:.3f}  "
          f"(correct={avg_conf_correct:.3f}, wrong={avg_conf_incorrect:.3f})")

    print("\n  Per-difficulty:")
    for diff in ["easy", "hard"]:
        s = diff_stats.get(diff, {})
        if s:
            acc = s["correct"] / s["total"]
            print(f"    {diff:<6}  {acc:.4f}  ({s['correct']}/{s['total']})")

    print("\n  Per-dimension:")
    for dim in sorted(dim_stats):
        s = dim_stats[dim]
        acc = s["correct"] / s["total"]
        w = DIMENSION_WEIGHTS.get(dim, 1.0)
        print(f"    {dim:<22}  {acc:.4f}  ({s['correct']}/{s['total']})  weight={w}")

    print("\n  Per-phenomenon:")
    for phen in sorted(phen_stats):
        s = phen_stats[phen]
        acc = s["correct"] / s["total"]
        print(f"    {phen:<35}  {acc:.4f}  ({s['correct']}/{s['total']})")

    # ---- save ----
    summary = {
        "n_questions": n,
        "overall_accuracy": round(overall_acc, 4),
        "difficulty_weighted_accuracy": round(weighted_diff_acc, 4),
        "dimension_weighted_accuracy": round(weighted_dim_acc, 4),
        "combined_weighted_accuracy": round(weighted_combined_acc, 4),
        "confidence": {
            "avg": round(avg_conf, 4),
            "avg_correct": round(avg_conf_correct, 4),
            "avg_incorrect": round(avg_conf_incorrect, 4),
        },
        "per_difficulty": {
            d: {
                "accuracy": round(s["correct"] / s["total"], 4),
                "correct": s["correct"],
                "total": s["total"],
            }
            for d, s in diff_stats.items()
        },
        "per_dimension": {
            d: {
                "accuracy": round(s["correct"] / s["total"], 4),
                "correct": s["correct"],
                "total": s["total"],
                "weight": DIMENSION_WEIGHTS.get(d, 1.0),
            }
            for d, s in sorted(dim_stats.items())
        },
        "per_phenomenon": {
            p: {
                "accuracy": round(s["correct"] / s["total"], 4),
                "correct": s["correct"],
                "total": s["total"],
            }
            for p, s in sorted(phen_stats.items())
        },
    }

    out_path = output_dir / "eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary saved to: {out_path}")
    return summary


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="PhyMetric VLM Evaluation (steps a / b / c)")
    p.add_argument("--question_file", type=str, default="data/question_v2_with_kb.json",
                   help="Path to question_v2_with_kb.json (output of build_knowledge_base.py). "
                        "If knowledge_base field is absent, evaluation runs without KB.")
    p.add_argument("--video_dir",  type=str, default=None,
                   help="Directory containing prompt-{idx}.mp4 files (required for steps a/b)")
    p.add_argument("--output_dir", type=str, default="output100/eval_run",
                   help="Root output directory for this evaluation run")
    p.add_argument("--steps", type=str, default="abc",
                   help="Which steps to run: any combination of 'a', 'b', 'c' (default: abc)")
    p.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-done frames/questions (default: True)")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--max_scenes", type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    steps = set(args.steps.lower())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    print(f"Loading scenes from: {args.question_file}")
    scenes, q_fps, q_total = load_scenes(args.question_file)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    print(f"  {len(scenes)} scenes loaded  (benchmark: {q_fps}fps / {q_total} frames)")

    # --- Step a ---
    if "a" in steps:
        if not args.video_dir:
            raise ValueError("--video_dir is required for step a")
        step_a(scenes, Path(args.video_dir), frames_dir, q_fps=q_fps, q_total=q_total)

    # --- Step b ---
    if "b" in steps:
        if not args.video_dir and "a" not in steps:
            print("  Note: running step b without step a; frames must already exist in frames_dir")
        vlm = QwenVLEvaluator(args.vlm_model_id, device="cuda")
        qa_results = step_b(scenes, frames_dir, output_dir, vlm, resume=args.resume)
    else:
        # load existing qa_results if step c needs them
        qa_path = output_dir / "qa_results.json"
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_results = json.load(f)
        else:
            qa_results = {}

    # --- Step c ---
    if "c" in steps:
        step_c(scenes, qa_results, output_dir)


if __name__ == "__main__":
    main()
