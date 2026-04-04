#!/usr/bin/env python3
"""
Run Cosmos-1.0-Diffusion-14B-Video2World inference on data/question_v2.json.

Conditions on a blank (black) first frame since no input video is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Cosmos-1.0-Diffusion-14B-Video2World on phymetric100")
    parser.add_argument("--question_file", type=str, default="../data/question_v2.json")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output100/Cosmos-1.0-Diffusion-14B-Video2World",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--model_id", type=str, default="nvidia/Cosmos-1.0-Diffusion-14B-Video2World")
    parser.add_argument("--negative_prompt", type=str, default="The video captures a game playing, animated, or cinematic visuals.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=36)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_scenes", type=int, default=None, help="Limit number of scenes")
    parser.add_argument("--start_index", type=int, default=0, help="Skip first N scenes")
    return parser.parse_args()


def _safe_scene_id(category: str, scene_key: str) -> str:
    raw = f"{category}_{scene_key}"
    return re.sub(r"[^\w\-.]", "_", raw)


def load_scenes(question_file: str) -> list[tuple[str, str]]:
    path = Path(question_file)
    if not path.is_file():
        raise FileNotFoundError(f"question_file not found: {question_file}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    main = data.get("main_categories") or {}
    scenes = []
    for category, scenes_dict in main.items():
        if not isinstance(scenes_dict, dict):
            continue
        for scene_key, scene_data in scenes_dict.items():
            if not isinstance(scene_data, dict):
                continue
            caption = scene_data.get("scene_caption")
            if not caption or not caption.strip():
                continue
            scene_id = _safe_scene_id(category, scene_key)
            scenes.append((scene_id, caption.strip()))
    return scenes


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from PIL import Image
    from diffusers import CosmosVideoToWorldPipeline
    from diffusers.utils import export_to_video
    from tqdm import tqdm

    scenes = load_scenes(args.question_file)
    if args.start_index > 0:
        scenes = scenes[args.start_index:]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        print("No scenes found in question file.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {args.gpu} | Scenes: {len(scenes)} | Out: {out_dir}")

    print(f"Loading model: {args.model_id}")
    pipe = CosmosVideoToWorldPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Black conditioning frame (no input video available)
    cond_image = Image.fromarray(
        __import__("numpy").zeros((args.height, args.width, 3), dtype="uint8")
    )

    for idx, (_, scene_caption) in enumerate(tqdm(scenes, desc="Cosmos-14B-V2W")):
        out_path = out_dir / f"prompt-{idx}.mp4"
        if out_path.is_file():
            tqdm.write(f"Skip (exists): prompt-{idx}.mp4")
            continue
        try:
            # Restore safety_checker to CUDA; its custom .to() makes pipe.device
            # report CPU after the first call, which breaks _execution_device.
            pipe.safety_checker.to("cuda")
            generator = torch.Generator("cuda").manual_seed(args.seed)
            output = pipe(
                image=cond_image,
                prompt=scene_caption,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                fps=args.fps,
                generator=generator,
            )
            frames = output.frames[0]
            export_to_video(frames, str(out_path), fps=args.fps)
        except Exception as e:
            tqdm.write(f"Error prompt-{idx}: {e}")
        finally:
            torch.cuda.empty_cache()

    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
