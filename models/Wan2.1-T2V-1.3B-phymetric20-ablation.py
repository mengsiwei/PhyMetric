#!/usr/bin/env python3
"""
Run baseline-vs-latent ablations for Wan2.1-T2V-1.3B on the first N PhyMetric prompts.

Each experiment is:
- baseline
- or one latent configuration varying optimize_steps / optimize_start / optimize_end / optimize_latent_window

Experiments are distributed across the provided GPUs, and each GPU processes
its assigned experiments sequentially.

Outputs:
    <output_root>/<experiment>/prompt-{idx}.mp4

After generation, the parent process can automatically compute CLIP similarity
and optionally invoke the PhyMetric evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_EXPERIMENTS = (
    "baseline",
    "latent_w3_k1_early10",
    "latent_w5_k1_early10",
    "latent_w7_k1_early10",
    "latent_w3_k1_early20",
    "latent_w3_k1_mid20_40",
    "latent_w3_k2_early10",
)

EXPERIMENT_SPECS = {
    "baseline": {
        "memory_preset": "baseline",
    },
    "latent_w3_k1_early10": {
        "memory_preset": "none",
        "optimize_steps": 1,
        "optimize_start": 0.0,
        "optimize_end": 0.1,
        "optimize_latent_window": 3,
    },
    "latent_w5_k1_early10": {
        "memory_preset": "none",
        "optimize_steps": 1,
        "optimize_start": 0.0,
        "optimize_end": 0.1,
        "optimize_latent_window": 5,
    },
    "latent_w7_k1_early10": {
        "memory_preset": "none",
        "optimize_steps": 1,
        "optimize_start": 0.0,
        "optimize_end": 0.1,
        "optimize_latent_window": 7,
    },
    "latent_w3_k1_early20": {
        "memory_preset": "none",
        "optimize_steps": 1,
        "optimize_start": 0.0,
        "optimize_end": 0.2,
        "optimize_latent_window": 3,
    },
    "latent_w3_k1_mid20_40": {
        "memory_preset": "none",
        "optimize_steps": 1,
        "optimize_start": 0.2,
        "optimize_end": 0.4,
        "optimize_latent_window": 3,
    },
    "latent_w3_k2_early10": {
        "memory_preset": "none",
        "optimize_steps": 2,
        "optimize_start": 0.0,
        "optimize_end": 0.1,
        "optimize_latent_window": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B ablation on first PhyMetric prompts")
    parser.add_argument("--worker_gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_experiments", nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metric_worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--question_file", type=str, default="/home/siweim/code/PhyMetric/data/question_v2.json")
    parser.add_argument("--question_file_with_kb", type=str, default="/home/siweim/code/PhyMetric/data/question_v2_with_kb.json")
    parser.add_argument("--wan_root", type=str, default="/home/siweim/code/Wan2.1")
    parser.add_argument("--download_dir", type=str, default="/home/siweim/code/models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--latent_video_root", type=str, default="/home/siweim/code/latent-video")
    parser.add_argument(
        "--python_bin",
        type=str,
        default="/home/siweim/miniconda3/envs/wan21/bin/python",
        help="Python used to run Wan2.1 latent-video inference",
    )
    parser.add_argument("--output_root", type=str, default="/home/siweim/code/PhyMetric/output20/Wan2.1-T2V-1.3B")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3], help="GPU ids used for the experiment queue")
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS), choices=list(EXPERIMENT_SPECS))
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=8.0)
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--sample_solver", type=str, default="dpm++", choices=["unipc", "dpm++"])
    parser.add_argument("--offload_model", type=str, default="True")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--geometry_mode", type=str, default="first_order", choices=["first_order", "fisher_diag", "hessian_diag"])
    parser.add_argument("--hessian_samples", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=20)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--compute_clip", action="store_true", default=True)
    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_num_frames", type=int, default=8)
    parser.add_argument("--run_phymetric_eval", action="store_true", default=False)
    parser.add_argument("--eval_steps", type=str, default="abc", help="PhyMetric evaluate.py --steps")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def load_scenes(question_file: str) -> list[tuple[int, str]]:
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes: list[tuple[int, str]] = []
    for _, scene_dict in data.get("main_categories", {}).items():
        for _, scene_data in scene_dict.items():
            caption = (scene_data or {}).get("scene_caption", "").strip()
            if caption:
                scenes.append((len(scenes), caption))
    return scenes


def build_experiment_command(args, experiment: str) -> list[str]:
    spec = EXPERIMENT_SPECS[experiment]
    cmd = [
        args.python_bin,
        "run_wan21_1_3b_latent_opt.py",
        "--memory_preset",
        spec["memory_preset"],
        "--wan_root",
        args.wan_root,
        "--download_dir",
        args.download_dir,
        "--frame_num",
        str(args.frame_num),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_solver",
        args.sample_solver,
        "--sample_shift",
        str(args.sample_shift),
        "--sample_guide_scale",
        str(args.sample_guide_scale),
        "--base_seed",
        str(args.base_seed),
        "--offload_model",
        args.offload_model,
        "--geometry_mode",
        args.geometry_mode,
        "--hessian_samples",
        str(args.hessian_samples),
    ]
    if args.t5_cpu:
        cmd.append("--t5_cpu")

    if spec["memory_preset"] == "none":
        cmd.extend(
            [
                "--optimize_steps",
                str(spec["optimize_steps"]),
                "--optimize_start",
                str(spec["optimize_start"]),
                "--optimize_end",
                str(spec["optimize_end"]),
                "--optimize_latent_window",
                str(spec["optimize_latent_window"]),
            ]
        )
    return cmd


def worker_main(args, experiments: list[str], gpu: int):
    latent_video_root = Path(args.latent_video_root)
    scenes = load_scenes(args.question_file)
    scenes = scenes[args.start_index : args.start_index + args.max_scenes]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[worker] GPU={gpu} experiments={experiments} scenes={len(scenes)}", flush=True)
    for experiment in experiments:
        out_dir = Path(args.output_root) / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd_prefix = build_experiment_command(args, experiment)
        print(f"[{experiment}] GPU={gpu} out={out_dir}", flush=True)
        for scene_idx, caption in scenes:
            save_file = out_dir / f"prompt-{scene_idx}.mp4"
            if args.skip_existing and save_file.is_file():
                print(f"[{experiment}] skip existing prompt-{scene_idx}.mp4", flush=True)
                continue

            cmd = cmd_prefix + ["--prompt", caption, "--save_file", str(save_file)]
            print(f"[{experiment}] scene={scene_idx}", flush=True)
            print(" ".join(shlex.quote(part) for part in cmd), flush=True)
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, cwd=str(latent_video_root), env=env, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[{experiment}] failed on prompt-{scene_idx}.mp4: {exc}", flush=True)


def compute_clip_scores_for_variant(
    variant_dir: Path,
    scenes: list[tuple[int, str]],
    clip_model_id: str,
    clip_num_frames: int,
) -> tuple[list[dict], dict]:
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()

    rows = []
    for scene_idx, caption in scenes:
        video_path = variant_dir / f"prompt-{scene_idx}.mp4"
        row = {
            "scene_idx": scene_idx,
            "prompt": caption,
            "video_path": str(video_path),
            "exists": video_path.is_file(),
        }
        if not video_path.is_file():
            row["clip_similarity_mean"] = None
            rows.append(row)
            continue

        reader = imageio.get_reader(str(video_path))
        try:
            frames = [frame for frame in reader]
        finally:
            reader.close()

        if not frames:
            row["clip_similarity_mean"] = None
            rows.append(row)
            continue

        sample_count = min(clip_num_frames, len(frames))
        indices = sorted({round(i * (len(frames) - 1) / max(sample_count - 1, 1)) for i in range(sample_count)})
        sampled_frames = [Image.fromarray(frames[i]).convert("RGB") for i in indices]

        with torch.no_grad():
            text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
            text_embeds = model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            image_inputs = processor(images=sampled_frames, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            scores = torch.matmul(image_embeds, text_embeds.T).squeeze(-1)
            row["clip_similarity_mean"] = float(scores.mean().item())
            row["clip_similarity_max"] = float(scores.max().item())
            row["sampled_frame_count"] = len(indices)

        rows.append(row)

    valid_scores = [row["clip_similarity_mean"] for row in rows if row.get("clip_similarity_mean") is not None]
    summary = {
        "variant": variant_dir.name,
        "num_scenes": len(rows),
        "num_scored": len(valid_scores),
        "clip_similarity_mean": float(sum(valid_scores) / len(valid_scores)) if valid_scores else None,
    }
    return rows, summary


def save_clip_metrics(output_root: Path, experiment: str, rows: list[dict], summary: dict):
    metrics_dir = output_root / experiment / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "clip_similarity.csv"
    json_path = metrics_dir / "clip_similarity_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_idx",
                "prompt",
                "video_path",
                "exists",
                "clip_similarity_mean",
                "clip_similarity_max",
                "sampled_frame_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_phymetric_eval(args, experiment: str):
    variant_dir = Path(args.output_root) / experiment
    eval_out = variant_dir / "phymetric_eval"
    cmd = [
        args.python_bin,
        "evaluate.py",
        "--steps",
        args.eval_steps,
        "--question_file",
        args.question_file_with_kb,
        "--video_dir",
        str(variant_dir),
        "--output_dir",
        str(eval_out),
    ]
    print(f"[eval] {experiment}", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd="/home/siweim/code/PhyMetric", check=True)


def metric_worker_main(args, experiment: str):
    scenes = load_scenes(args.question_file)[args.start_index : args.start_index + args.max_scenes]
    rows, summary = compute_clip_scores_for_variant(
        Path(args.output_root) / experiment,
        scenes,
        args.clip_model_id,
        args.clip_num_frames,
    )
    save_clip_metrics(Path(args.output_root), experiment, rows, summary)
    print(f"[clip] {experiment}: {summary}", flush=True)


def launch_parent(args):
    assignments = {gpu: [] for gpu in args.gpus}
    for idx, experiment in enumerate(args.experiments):
        gpu = args.gpus[idx % len(args.gpus)]
        assignments[gpu].append(experiment)
    procs = []
    for gpu, experiments in assignments.items():
        if not experiments:
            continue
        cmd = [
            sys.executable,
            __file__,
            "--worker_gpu",
            str(gpu),
            "--worker_experiments",
            *experiments,
            "--question_file",
            args.question_file,
            "--question_file_with_kb",
            args.question_file_with_kb,
            "--wan_root",
            args.wan_root,
            "--download_dir",
            args.download_dir,
            "--latent_video_root",
            args.latent_video_root,
            "--python_bin",
            args.python_bin,
            "--output_root",
            args.output_root,
            "--frame_num",
            str(args.frame_num),
            "--sample_steps",
            str(args.sample_steps),
            "--sample_solver",
            args.sample_solver,
            "--sample_shift",
            str(args.sample_shift),
            "--sample_guide_scale",
            str(args.sample_guide_scale),
            "--base_seed",
            str(args.base_seed),
            "--offload_model",
            args.offload_model,
            "--geometry_mode",
            args.geometry_mode,
            "--hessian_samples",
            str(args.hessian_samples),
            "--start_index",
            str(args.start_index),
            "--max_scenes",
            str(args.max_scenes),
        ]
        if args.t5_cpu:
            cmd.append("--t5_cpu")
        if args.skip_existing:
            cmd.append("--skip_existing")
        if args.dry_run:
            cmd.append("--dry_run")
        print(f"[launch] gpu {gpu} experiments={experiments}", flush=True)
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent)))

    exit_code = 0
    for proc in procs:
        code = proc.wait()
        if code != 0:
            exit_code = code
    if exit_code != 0:
        raise SystemExit(exit_code)

    if args.dry_run:
        return

    aggregate = {}
    if args.compute_clip:
        for experiment in args.experiments:
            cmd = [
                args.python_bin,
                __file__,
                "--metric_worker",
                experiment,
                "--question_file",
                args.question_file,
                "--question_file_with_kb",
                args.question_file_with_kb,
                "--output_root",
                args.output_root,
                "--clip_model_id",
                args.clip_model_id,
                "--clip_num_frames",
                str(args.clip_num_frames),
                "--start_index",
                str(args.start_index),
                "--max_scenes",
                str(args.max_scenes),
            ]
            print(f"[clip-launch] {experiment}", flush=True)
            print(" ".join(shlex.quote(part) for part in cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))
            summary_path = Path(args.output_root) / experiment / "metrics" / "clip_similarity_summary.json"
            if summary_path.is_file():
                with open(summary_path, "r", encoding="utf-8") as f:
                    aggregate[experiment] = json.load(f)

    for experiment in args.experiments:
        if args.run_phymetric_eval:
            run_phymetric_eval(args, experiment)

    if aggregate:
        out_path = Path(args.output_root) / "clip_similarity_overview.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)
        print(f"[clip] wrote {out_path}", flush=True)


def main():
    args = parse_args()
    if args.worker_experiments is not None:
        worker_main(args, args.worker_experiments, args.worker_gpu)
    elif args.metric_worker is not None:
        metric_worker_main(args, args.metric_worker)
    else:
        launch_parent(args)


if __name__ == "__main__":
    main()
