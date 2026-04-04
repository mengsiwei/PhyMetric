#!/usr/bin/env python3
"""
Run three Wan2.2-T2V-A14B variants on PhyMetric scenes in parallel:

- baseline
- latent
- latent_hessian

Each variant is assigned to one GPU and writes videos as:
    output100/Wan2.2-T2V-A14B/<variant>/prompt-{idx}.mp4

The output naming matches PhyMetric's evaluator expectations.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANTS = ("baseline", "latent", "latent_hessian")


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2-T2V-A14B ablation on PhyMetric100")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--question_file", type=str, default="../data/question_v2.json")
    parser.add_argument("--wan_root", type=str, default="/home/siweim/code/Wan2.2")
    parser.add_argument("--ckpt_dir", type=str, default="/home/siweim/code/Wan2.2/Wan2.2-T2V-A14B")
    parser.add_argument("--latent_video_root", type=str, default="/home/siweim/code/latent-video")
    parser.add_argument(
        "--python_bin",
        type=str,
        default="/home/siweim/miniconda3/envs/wan21/bin/python",
        help="Python used to run latent-video inference",
    )
    parser.add_argument("--output_root", type=str, default="../output100/Wan2.2-T2V-A14B")
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 1, 2], help="GPU ids for baseline / latent / latent_hessian")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=list(DEFAULT_VARIANTS))
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=8, help="Low default for fast/OOM-safe testing")
    parser.add_argument("--sample_shift", type=float, default=12.0)
    parser.add_argument("--sample_guide_scale", type=float, nargs="+", default=[3.0, 4.0])
    parser.add_argument("--offload_model", type=str, default="True")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--optimize_steps", type=int, default=3)
    parser.add_argument("--optimize_start", type=float, default=0)
    parser.add_argument("--optimize_end", type=float, default=0.5)
    parser.add_argument("--step_size", type=float, default=0.08)
    parser.add_argument("--lambda_cfg", type=float, default=1.0)
    parser.add_argument("--lambda_temp", type=float, default=0.05)
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--hessian_samples", type=int, default=1)
    parser.add_argument("--hessian_damping", type=float, default=1e-4)
    parser.add_argument("--max_update_norm", type=float, default=2.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
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


def build_variant_args(args, variant: str) -> list[str]:
    common = [
        args.python_bin,
        "run_wan_latent_opt.py",
        "--wan_root",
        args.wan_root,
        "--ckpt_dir",
        args.ckpt_dir,
        "--size",
        args.size,
        "--frame_num",
        str(args.frame_num),
        "--base_seed",
        str(args.base_seed),
        "--sample_solver",
        args.sample_solver,
        "--sample_steps",
        str(args.sample_steps),
        "--sample_shift",
        str(args.sample_shift),
        "--sample_guide_scale",
        *[str(x) for x in args.sample_guide_scale],
        "--offload_model",
        args.offload_model,
        "--optimize_start",
        str(args.optimize_start),
        "--optimize_end",
        str(args.optimize_end),
        "--step_size",
        str(args.step_size),
        "--lambda_cfg",
        str(args.lambda_cfg),
        "--lambda_temp",
        str(args.lambda_temp),
        "--lambda_anchor",
        str(args.lambda_anchor),
        "--hessian_damping",
        str(args.hessian_damping),
        "--max_update_norm",
        str(args.max_update_norm),
    ]
    if args.t5_cpu:
        common.append("--t5_cpu")
    if args.convert_model_dtype:
        common.append("--convert_model_dtype")

    if variant == "baseline":
        common.extend(["--optimize_steps", "0", "--hessian_samples", "0"])
    elif variant == "latent":
        common.extend(["--optimize_steps", str(args.optimize_steps), "--hessian_samples", "0"])
    elif variant == "latent_hessian":
        common.extend(["--optimize_steps", str(args.optimize_steps), "--hessian_samples", str(args.hessian_samples)])
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return common


def worker_main(args, variant: str, gpu: int):
    latent_video_root = Path(args.latent_video_root)
    out_dir = Path(args.output_root) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args.question_file)
    scenes = scenes[args.start_index:]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    cmd_prefix = build_variant_args(args, variant)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[{variant}] GPU={gpu} scenes={len(scenes)} out={out_dir}")
    for scene_idx, caption in scenes:
        save_file = out_dir / f"prompt-{scene_idx}.mp4"
        if args.skip_existing and save_file.is_file():
            print(f"[{variant}] skip existing prompt-{scene_idx}.mp4")
            continue

        cmd = cmd_prefix + ["--prompt", caption, "--save_file", str(save_file)]
        print(f"[{variant}] scene={scene_idx}")
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, cwd=str(latent_video_root), env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[{variant}] failed on prompt-{scene_idx}.mp4: {exc}")


def launch_parent(args):
    procs = []
    for variant, gpu in zip(args.variants, args.gpus):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            variant,
            "--gpu",
            str(gpu),
            "--question_file",
            args.question_file,
            "--wan_root",
            args.wan_root,
            "--ckpt_dir",
            args.ckpt_dir,
            "--latent_video_root",
            args.latent_video_root,
            "--python_bin",
            args.python_bin,
            "--output_root",
            args.output_root,
            "--size",
            args.size,
            "--frame_num",
            str(args.frame_num),
            "--base_seed",
            str(args.base_seed),
            "--sample_solver",
            args.sample_solver,
            "--sample_steps",
            str(args.sample_steps),
            "--sample_shift",
            str(args.sample_shift),
            "--offload_model",
            args.offload_model,
            "--optimize_steps",
            str(args.optimize_steps),
            "--optimize_start",
            str(args.optimize_start),
            "--optimize_end",
            str(args.optimize_end),
            "--step_size",
            str(args.step_size),
            "--lambda_cfg",
            str(args.lambda_cfg),
            "--lambda_temp",
            str(args.lambda_temp),
            "--lambda_anchor",
            str(args.lambda_anchor),
            "--hessian_samples",
            str(args.hessian_samples),
            "--hessian_damping",
            str(args.hessian_damping),
            "--max_update_norm",
            str(args.max_update_norm),
            "--start_index",
            str(args.start_index),
        ]
        if args.max_scenes is not None:
            cmd.extend(["--max_scenes", str(args.max_scenes)])
        if args.t5_cpu:
            cmd.append("--t5_cpu")
        if args.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        if args.skip_existing:
            cmd.append("--skip_existing")
        if args.dry_run:
            cmd.append("--dry_run")
        cmd.extend(["--sample_guide_scale", *[str(x) for x in args.sample_guide_scale]])
        print(f"[launch] {variant} on gpu {gpu}")
        print(" ".join(shlex.quote(part) for part in cmd))
        procs.append(subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent)))

    exit_code = 0
    for proc in procs:
        code = proc.wait()
        if code != 0:
            exit_code = code
    raise SystemExit(exit_code)


def build_worker_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main():
    worker_parser = build_worker_parser()
    worker_args, _ = worker_parser.parse_known_args()
    args = parse_args()
    if worker_args.worker is not None:
        worker_main(args, worker_args.worker, worker_args.gpu)
    else:
        launch_parent(args)


if __name__ == "__main__":
    main()
