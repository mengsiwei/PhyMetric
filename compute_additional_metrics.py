#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal/motion/aesthetic/image-quality proxy metrics for video variants."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root dir containing variant folders (e.g. output20/Wan2.1-T2V-1.3B).",
    )
    parser.add_argument(
        "--sampled_frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames per video.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional cap of scenes per variant for debugging.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit variant names. Defaults to all subdirs with prompt-*.mp4.",
    )
    return parser.parse_args()


def safe_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def list_variants(output_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [output_root / name for name in requested]

    variants: list[Path] = []
    for item in sorted(output_root.iterdir()):
        if not item.is_dir():
            continue
        if any(item.glob("prompt-*.mp4")):
            variants.append(item)
    return variants


def list_scene_indices(variant_dir: Path, max_scenes: int | None) -> list[int]:
    indices: list[int] = []
    for video_path in sorted(variant_dir.glob("prompt-*.mp4")):
        stem = video_path.stem
        try:
            indices.append(int(stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    if max_scenes is not None:
        return indices[:max_scenes]
    return indices


def load_prompts(variant_dir: Path) -> dict[int, str]:
    csv_path = variant_dir / "metrics" / "clip_similarity.csv"
    if not csv_path.is_file():
        return {}

    prompts: dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["scene_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            prompts[idx] = row.get("prompt", "")
    return prompts


def sample_indices(num_frames: int, sampled_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    n = min(sampled_frames, num_frames)
    if n == 1:
        return [0]
    raw = [round(i * (num_frames - 1) / (n - 1)) for i in range(n)]
    return sorted(set(raw))


def to_gray_small(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def frame_quality_features(frame_rgb: np.ndarray) -> tuple[float, float, float, float]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(gray.std())
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = float(hsv[..., 1].mean() / 255.0)
    mean_luma = float(gray.mean() / 255.0)
    exposure = max(0.0, 1.0 - abs(mean_luma - 0.5) * 2.0)
    return sharpness, contrast, saturation, exposure


def quality_proxy_from_features(sharpness: float, contrast: float) -> float:
    sharpness_norm = 1.0 - math.exp(-sharpness / 150.0)
    contrast_norm = min(1.0, contrast / 64.0)
    return max(0.0, min(1.0, 0.7 * sharpness_norm + 0.3 * contrast_norm))


def temporal_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32) / 255.0
    b = frame_b.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return float(math.exp(-5.0 * mse))


def calc_video_metrics(video_path: Path, sampled_frames: int) -> dict[str, float | int | None]:
    reader = imageio.get_reader(str(video_path))
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()

    if not frames:
        return {
            "sampled_frame_count": 0,
            "temporal_consistency": None,
            "motion_consistency": None,
            "motion_strength": None,
            "image_quality": None,
            "aesthetic_score": None,
        }

    indices = sample_indices(len(frames), sampled_frames)
    sampled = [frames[i] for i in indices]

    sharpness_vals: list[float] = []
    contrast_vals: list[float] = []
    saturation_vals: list[float] = []
    exposure_vals: list[float] = []
    quality_vals: list[float] = []
    temporal_vals: list[float] = []
    flow_magnitudes: list[float] = []

    gray_small = [to_gray_small(frame) for frame in sampled]
    for frame in sampled:
        sharpness, contrast, saturation, exposure = frame_quality_features(frame)
        sharpness_vals.append(sharpness)
        contrast_vals.append(contrast)
        saturation_vals.append(saturation)
        exposure_vals.append(exposure)
        quality_vals.append(quality_proxy_from_features(sharpness, contrast))

    for i in range(len(sampled) - 1):
        temporal_vals.append(temporal_similarity(sampled[i], sampled[i + 1]))

        flow = cv2.calcOpticalFlowFarneback(
            gray_small[i],
            gray_small[i + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(float(mag.mean()))

    temporal_consistency = float(np.mean(temporal_vals)) if temporal_vals else None
    motion_strength = float(np.mean(flow_magnitudes)) if flow_magnitudes else None

    motion_consistency: float | None
    if flow_magnitudes:
        mean_mag = float(np.mean(flow_magnitudes))
        std_mag = float(np.std(flow_magnitudes))
        motion_consistency = 1.0 - (std_mag / (mean_mag + 1e-6))
        motion_consistency = float(max(0.0, min(1.0, motion_consistency)))
    else:
        motion_consistency = None

    image_quality = float(np.mean(quality_vals)) if quality_vals else None
    aesthetic_score = None
    if quality_vals and saturation_vals and exposure_vals:
        aesthetic_score = float(
            np.mean(
                0.5 * np.array(quality_vals)
                + 0.25 * np.array(saturation_vals)
                + 0.25 * np.array(exposure_vals)
            )
        )
        aesthetic_score = float(max(0.0, min(1.0, aesthetic_score)))

    return {
        "sampled_frame_count": len(indices),
        "temporal_consistency": safe_float(temporal_consistency),
        "motion_consistency": safe_float(motion_consistency),
        "motion_strength": safe_float(motion_strength),
        "image_quality": safe_float(image_quality),
        "aesthetic_score": safe_float(aesthetic_score),
    }


def metric_summary(rows: list[dict], variant: str, metric_key: str) -> dict:
    vals = [row[metric_key] for row in rows if row.get(metric_key) is not None]
    return {
        "variant": variant,
        "num_scenes": len(rows),
        "num_scored": len(vals),
        f"{metric_key}_mean": float(np.mean(vals)) if vals else None,
    }


def write_variant_outputs(variant_dir: Path, rows: list[dict]) -> dict[str, dict]:
    metrics_dir = variant_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "additional_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_idx",
                "prompt",
                "video_path",
                "exists",
                "sampled_frame_count",
                "temporal_consistency",
                "motion_consistency",
                "motion_strength",
                "image_quality",
                "aesthetic_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summaries = {
        "temporal_consistency": metric_summary(rows, variant_dir.name, "temporal_consistency"),
        "motion_consistency": metric_summary(rows, variant_dir.name, "motion_consistency"),
        "image_quality": metric_summary(rows, variant_dir.name, "image_quality"),
        "aesthetic_score": metric_summary(rows, variant_dir.name, "aesthetic_score"),
    }
    for key, data in summaries.items():
        with open(metrics_dir / f"{key}_summary.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return summaries


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    variants = list_variants(output_root, args.variants)
    if not variants:
        raise SystemExit(f"No variant dirs found under: {output_root}")

    overview_temporal: dict[str, dict] = {}
    overview_motion: dict[str, dict] = {}
    overview_quality: dict[str, dict] = {}
    overview_aesthetic: dict[str, dict] = {}

    for variant_dir in variants:
        prompts = load_prompts(variant_dir)
        scene_indices = list_scene_indices(variant_dir, args.max_scenes)
        rows: list[dict] = []
        for scene_idx in scene_indices:
            video_path = variant_dir / f"prompt-{scene_idx}.mp4"
            row = {
                "scene_idx": scene_idx,
                "prompt": prompts.get(scene_idx, ""),
                "video_path": str(video_path),
                "exists": video_path.is_file(),
            }
            if video_path.is_file():
                row.update(calc_video_metrics(video_path, args.sampled_frames))
            else:
                row.update(
                    {
                        "sampled_frame_count": 0,
                        "temporal_consistency": None,
                        "motion_consistency": None,
                        "motion_strength": None,
                        "image_quality": None,
                        "aesthetic_score": None,
                    }
                )
            rows.append(row)

        summaries = write_variant_outputs(variant_dir, rows)
        overview_temporal[variant_dir.name] = summaries["temporal_consistency"]
        overview_motion[variant_dir.name] = summaries["motion_consistency"]
        overview_quality[variant_dir.name] = summaries["image_quality"]
        overview_aesthetic[variant_dir.name] = summaries["aesthetic_score"]
        print(f"[done] {variant_dir.name} scenes={len(rows)}")

    with open(output_root / "temporal_consistency_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview_temporal, f, ensure_ascii=False, indent=2)
    with open(output_root / "motion_consistency_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview_motion, f, ensure_ascii=False, indent=2)
    with open(output_root / "image_quality_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview_quality, f, ensure_ascii=False, indent=2)
    with open(output_root / "aesthetic_score_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview_aesthetic, f, ensure_ascii=False, indent=2)

    print(f"[saved] {output_root}")


if __name__ == "__main__":
    main()
