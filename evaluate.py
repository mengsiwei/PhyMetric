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
    q_total = int(meta.get("total_frames", 120))
    scenes = []
    for phenomenon, cat_data in data.get("categories", {}).items():
        for scene_key, scene_data in cat_data.get("scenes", {}).items():
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
    v_total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_fps_raw = cap.get(cv2.CAP_PROP_FPS) or 0.0

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

    frame_map: dict = {str(q_idx): paths for q_idx, paths in result.items()}
    # store actual video spec so step_b can show accurate context to the VLM
    frame_map["_video_info"] = {
        "fps": round(v_fps_raw, 3),
        "total_frames": v_total,
        "duration_s": round(v_total / v_fps_raw, 2) if v_fps_raw > 0 else None,
    }
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
        video_path = video_dir / f"{scene['scene_id']}.mp4"
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


GENERIC_QUESTION_TEXT = (
    "Does this video correctly simulate the physical behavior described "
    "in the scene caption? Answer YES or NO."
)


def build_vlm_prompt(
    scene_caption: str,
    knowledge_base: dict,
    question: dict,
    video_fps: float | None = None,
    video_frames: int | None = None,
    video_duration: float | None = None,
    include_kb: bool = True,
    generic_question: bool = False,
) -> str:
    kb_text = ""
    if include_kb and knowledge_base:
        seq = knowledge_base.get("expected_sequence", {})
        seq_lines = "\n".join(
            f"  • {k.replace('_', ' ').capitalize()}: {v}"
            for k, v in seq.items()
        )
        # support both field names for backwards compatibility with v2 KBs
        signatures = knowledge_base.get("physics_signatures") or knowledge_base.get("visual_indicators", [])
        sig_lines = "\n".join(f"  • {v}" for v in signatures)
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

Expected temporal sequence:
{seq_lines}

Physical law signatures (what MUST be observable in a correct video):
{sig_lines}

Common generation failures (what to watch for):
{failures}

Evaluation hints for this question's dimensions:
{hints if hints else '  (apply the physical law signatures above)'}
--- END KNOWLEDGE BASE ---
"""

    frame_indices_str = ", ".join(str(i) for i in question.get("frame_indices", [30, 60, 100]))

    # Build video spec line: use actual video info if available, else benchmark reference
    if video_fps and video_frames:
        dur_str = f"{video_duration:.1f}s" if video_duration else f"{video_frames/video_fps:.1f}s"
        video_spec = (
            f"Video spec: {video_fps:.0f}fps, {video_frames} frames total, ~{dur_str}. "
            f"Shown frames are proportionally sampled from this video."
        )
    else:
        video_spec = "Video spec: benchmark reference (30fps, 120 frames, 4s)."

    question_text = GENERIC_QUESTION_TEXT if generic_question else question["question_text"]

    return (
        f"Scene caption: \"{scene_caption}\"\n"
        f"{video_spec}\n"
        f"Frames shown correspond to benchmark time-points: [{frame_indices_str}] / 120\n"
        f"{kb_text}\n"
        f"Question (answer YES or NO): {question_text}"
    )


def parse_vlm_response(response: str) -> tuple[str, float, str]:
    """Returns (answer: 'yes'/'no', confidence: 0-1, reason: str).

    Strips <think>...</think> blocks produced by thinking models before parsing.
    """
    # Remove thinking chain if present (Qwen3 thinking models)
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not clean:
        clean = response.strip()

    answer = "no"
    confidence = 0.5
    reason = clean

    ans_match = re.search(r"ANSWER\s*:\s*(YES|NO)", clean, re.IGNORECASE)
    if ans_match:
        answer = ans_match.group(1).lower()

    conf_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", clean)
    if conf_match:
        confidence = max(0.0, min(1.0, float(conf_match.group(1))))

    reason_match = re.search(r"REASON\s*:\s*(.+)", clean, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return answer, confidence, reason


class QwenVLEvaluator:
    """Vision-language evaluator supporting Qwen VL and GLM-4V model families."""

    def __init__(self, model_id: str, device: str = "cuda", enable_thinking: bool = False):
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        print(f"Loading VLM: {model_id}  (thinking={'on' if enable_thinking else 'off'})")

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch = getattr(cfg, "model_type", "").lower()
        self._is_thinking_model = "qwen3" in arch
        # glm4v_moe uses AutoProcessor (like Qwen); plain glm4v uses AutoTokenizer
        self._is_glm = (arch.startswith("glm4v") or arch.startswith("chatglm")
                        or arch.startswith("glm")) and arch != "glm4v_moe"
        self._enable_thinking = enable_thinking

        _load_kwargs = dict(device_map=device, trust_remote_code=True)
        try:
            self._model = _load_vl(model_id, torch.bfloat16, _load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{model_id}' as a vision-language model: {e}\n"
                "Ensure --vlm_model_id points to a VL model (e.g. Qwen/Qwen2.5-VL-7B-Instruct, "
                "Qwen/Qwen3.5-9B, or zai-org/GLM-4.6V-Flash)."
            ) from e

        # GLM-4V uses a tokenizer; Qwen VL uses a processor that wraps a tokenizer.
        if self._is_glm:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._processor = None
        else:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._tokenizer = None
        self._device = device

    def _load_images(self, frame_paths: list[str]):
        from PIL import Image
        MAX_SIDE, MAX_IMAGES = 448, 6
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
        return pil_images

    def _query_qwen(self, pil_images: list, user_text: str) -> str:
        import torch
        image_content = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        ]
        tmpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if self._is_thinking_model and not self._enable_thinking:
            tmpl_kwargs["enable_thinking"] = False
        try:
            text = self._processor.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop("enable_thinking", None)
            text = self._processor.apply_chat_template(messages, **tmpl_kwargs)

        inputs = self._processor(
            text=[text], images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self._device)
        gen_kwargs: dict = dict(
            max_new_tokens=128, do_sample=False,
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def _query_glm(self, pil_images: list, user_text: str) -> str:
        import torch
        # GLM-4V: system prompt is merged into user turn (no dedicated system role).
        combined = f"{VLM_SYSTEM_PROMPT}\n\n{user_text}"
        image_parts = [{"type": "image", "image": img} for img in pil_images]
        messages = [{"role": "user", "content": image_parts + [{"type": "text", "text": combined}]}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def query(self, frame_paths: list[str], user_text: str) -> str:
        import torch
        pil_images = self._load_images(frame_paths)
        result = self._query_glm(pil_images, user_text) if self._is_glm \
                 else self._query_qwen(pil_images, user_text)
        torch.cuda.empty_cache()
        return result


_VL_ARCH_PREFIXES = ("qwen3_5", "qwen2_5_vl", "qwen2_vl", "llava", "internvl", "idefics",
                     "glm4v", "chatglm", "glm")

def _is_vl_model(model_id: str) -> bool:
    """Return True if the model config declares a known VL architecture."""
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch = getattr(cfg, "model_type", "").lower()
        return any(arch.startswith(p) for p in _VL_ARCH_PREFIXES)
    except Exception:
        return False


def _load_vl(model_id: str, dtype, kwargs: dict):
    """
    Load a VL model by routing on model_type from config — avoids silent
    wrong-class loading (e.g. Qwen3_5 silently accepting GLM weights).
    """
    from transformers import AutoConfig
    cfg  = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = getattr(cfg, "model_type", "").lower()

    def _try(cls):
        try:
            return cls.from_pretrained(model_id, dtype=dtype, **kwargs).eval()
        except TypeError:
            return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs).eval()

    if arch.startswith("glm4v"):
        if arch == "glm4v_moe":
            from transformers import Glm4vMoeForConditionalGeneration
            return _try(Glm4vMoeForConditionalGeneration)
        else:
            from transformers import Glm4vForConditionalGeneration
            return _try(Glm4vForConditionalGeneration)

    if arch == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        return _try(Qwen3_5ForConditionalGeneration)

    if arch == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return _try(Qwen2_5_VLForConditionalGeneration)

    if arch == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return _try(Qwen2VLForConditionalGeneration)

    raise RuntimeError(
        f"Unsupported VL model_type '{arch}' for '{model_id}'. "
        "Supported: glm4v, glm4v_moe, qwen3_5, qwen2_5_vl, qwen2_vl."
    )


class TextLLMEvaluator:
    """Text-only evaluator for models that cannot process images (e.g. Mistral-7B).

    Sends only the scene caption + KB + question text (no frames).  Useful for
    measuring whether KB alone (without visual input) improves yes/no accuracy.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading text-only evaluator: {model_id}  (no video frames)")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
            ).eval()
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
            ).eval()
        self._device = device

    def query(self, frame_paths: list[str], user_text: str) -> str:
        """Ignore frame_paths entirely; answer based on text prompt only."""
        import torch

        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        import torch as _t
        del inputs, out
        _t.cuda.empty_cache()
        return result


def step_b(
    scenes: list[dict],
    frames_dir: Path,
    output_dir: Path,
    vlm: QwenVLEvaluator,
    resume: bool = True,
    no_kb: bool = False,
    center_frame_only: bool = False,
    generic_question: bool = False,
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

        # load frame_map.json (written by step_a); fallback to empty
        map_path = scene_frames_dir / "frame_map.json"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as _f:
                _raw_map: dict = json.load(_f)
            video_info: dict = _raw_map.pop("_video_info", {})
            frame_map: dict[str, list[str]] = _raw_map
        else:
            frame_map = {}
            video_info = {}

        scene_results = results.get(sid, {})

        for q in scene["questions"]:
            qid = q["id"]
            if qid in scene_results:
                done_q += 1
                continue

            # collect frame paths
            frame_indices = q.get("frame_indices", [10, 30, 50])
            frame_paths: list[str] = []
            for i in frame_indices:
                mapped = frame_map.get(str(i))
                if mapped:
                    if center_frame_only:
                        # take only the center frame (middle of sorted neighbor list)
                        frame_paths.append(mapped[len(mapped) // 2])
                    else:
                        frame_paths.extend(mapped)
                else:
                    frame_paths.append(str(scene_frames_dir / f"frame_{i:04d}.jpg"))
            available = [p for p in frame_paths if Path(p).exists()]

            user_text = build_vlm_prompt(
                scene["scene_caption"],
                scene["knowledge_base"],
                q,
                video_fps=video_info.get("fps"),
                video_frames=video_info.get("total_frames"),
                video_duration=video_info.get("duration_s"),
                include_kb=not no_kb,
                generic_question=generic_question,
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
    "motion":             2.0,
    "direction":          2.0,
    "interaction":        2.0,
    "causal":             2.0,
    "spatial":            1.5,
    "material":           1.5,
    "appearance/texture": 1.0,
    "shape":              1.0,
    "existence":          1.0,
    "size":               1.0,
    "number":             1.0,
    "object":             1.0,
}
DIFFICULTY_WEIGHTS = {"hard": 2.0, "medium": 1.5, "easy": 1.0}


def step_c(scenes: list[dict], qa_results: dict, output_dir: Path, run_meta: dict | None = None):
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
    for diff in ["easy", "medium", "hard"]:
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
        "run_metadata": run_meta or {},
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
    p.add_argument("--question_file", type=str, default="data/question_v3_with_kb.json",
                   help="Path to question file with KB (output of build_knowledge_base.py). "
                        "If knowledge_base field is absent, evaluation runs without KB.")
    p.add_argument("--video_dir",  type=str, default=None,
                   help="Directory containing prompt-{idx}.mp4 files (required for steps a/b)")
    p.add_argument("--output_dir", type=str, default="output100/eval_run",
                   help="Root output directory for this evaluation run")
    p.add_argument("--steps", type=str, default="abc",
                   help="Which steps to run: any combination of 'a', 'b', 'c' (default: abc)")
    p.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace model ID of the VLM judge (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--video_model", type=str, default=None,
                   help="Name of the video generator being evaluated (e.g. CogVideoX-5B). "
                        "Stored in eval_summary.json for record-keeping.")
    p.add_argument("--video_fps", type=float, default=None,
                   help="FPS of generated videos (overrides auto-detection). Stored in eval_summary.")
    p.add_argument("--video_frames", type=int, default=None,
                   help="Total frames of generated videos (overrides auto-detection). Stored in eval_summary.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-done frames/questions (default: True)")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--max_scenes", type=int, default=None)

    # ---- ablation flags ----
    p.add_argument("--no_kb", action="store_true", default=False,
                   help="[Ablation] Skip KB block in VLM prompt (tests KB contribution)")
    p.add_argument("--thinking", action="store_true", default=False,
                   help="[Ablation] Enable chain-of-thought for thinking models (default: disabled)")
    p.add_argument("--center_frame_only", action="store_true", default=False,
                   help="[Ablation] Use only center frame per question index, no ±1 neighbors")
    p.add_argument("--generic_question", action="store_true", default=False,
                   help="[Ablation] Replace dimension-specific questions with a single generic "
                        "physics plausibility question (tests structured QA contribution)")
    p.add_argument("--scene_ids_file", type=str, default=None,
                   help="[Ablation] Text file with one scene_id per line; restricts evaluation "
                        "to those scenes. Generate with generate_ablation_subset.py.")
    p.add_argument("--frames_dir", type=str, default=None,
                   help="Override frames directory (default: output_dir/frames). "
                        "Share pre-extracted frames across ablation conditions.")

    return p.parse_args()


def main():
    args = _parse_args()
    # Only set CUDA_VISIBLE_DEVICES if not already set by the caller.
    # ablation.sh sets it externally (CUDA_VISIBLE_DEVICES="1" python evaluate.py --gpu 0)
    # so that batch-parallel jobs land on different physical GPUs.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    steps = set(args.steps.lower())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir / "frames"

    print(f"Loading scenes from: {args.question_file}")
    scenes, q_fps, q_total = load_scenes(args.question_file)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    # restrict to ablation subset if a scene_ids_file is provided
    if args.scene_ids_file:
        with open(args.scene_ids_file) as _f:
            allowed = {line.strip() for line in _f if line.strip()}
        scenes = [s for s in scenes if s["scene_id"] in allowed]
        print(f"  Restricted to {len(scenes)} scenes from {args.scene_ids_file}")

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
        if _is_vl_model(args.vlm_model_id):
            vlm = QwenVLEvaluator(args.vlm_model_id, device="cuda",
                                  enable_thinking=args.thinking)
        else:
            print(f"  Note: {args.vlm_model_id} is not a VL model — using TextLLMEvaluator (no frames)")
            vlm = TextLLMEvaluator(args.vlm_model_id, device="cuda")
        qa_results = step_b(
            scenes, frames_dir, output_dir, vlm, resume=args.resume,
            no_kb=args.no_kb,
            center_frame_only=args.center_frame_only,
            generic_question=args.generic_question,
        )
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
        run_meta = {
            "video_model": args.video_model or (Path(args.video_dir).parent.name if args.video_dir else "unknown"),
            "video_dir": args.video_dir,
            "video_fps": args.video_fps,
            "video_frames": args.video_frames,
            "vlm_model_id": args.vlm_model_id,
            "question_file": args.question_file,
            # ablation flags stored for reproducibility
            "ablation": {
                "no_kb": args.no_kb,
                "thinking": args.thinking,
                "center_frame_only": args.center_frame_only,
                "generic_question": args.generic_question,
                "scene_ids_file": args.scene_ids_file,
            },
        }
        step_c(scenes, qa_results, output_dir, run_meta=run_meta)


if __name__ == "__main__":
    main()
