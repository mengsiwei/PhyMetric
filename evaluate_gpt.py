#!/usr/bin/env python3
"""
PhyMetric GPT Evaluation Pipeline  (steps b + c)

Replaces local VLM inference with OpenAI API calls (vision).
Step a (frame extraction) must be run first via evaluate.py --steps a;
the pre-extracted frames are reused here.

Images are sent with detail="low" (85 tokens/frame) to control cost.

Usage:
    python evaluate_gpt.py \\
        --question_file  data/question_v3_with_kb.json \\
        --frames_dir     outputs/Wan2.2-14B/frames \\
        --output_dir     outputs/ablation/Wan2.2-14B/C01_gpt_full \\
        --video_model    Wan2.2-14B \\
        --model          gpt-5.4 \\
        --workers        8

    # Step c only (re-score existing qa_results.json)
    python evaluate_gpt.py --steps c \\
        --question_file data/question_v3_with_kb.json \\
        --output_dir    outputs/ablation/Wan2.2-14B/C01_gpt_full

Environment:
    OPENAI_API_KEY  — required (or pass --api_key)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Shared prompt/scoring utilities from the open-source pipeline.
# evaluate.py is expected to be on sys.path (same directory).
from evaluate import (
    VLM_SYSTEM_PROMPT,
    build_vlm_prompt,
    load_scenes,
    parse_vlm_response,
    step_c,
)

try:
    from openai import APIStatusError, OpenAI, RateLimitError
except ImportError:
    raise SystemExit("openai package required:  pip install openai>=1.0")


# =============================================================================
# GPT Evaluator
# =============================================================================

def _b64_jpeg(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class GPTEvaluator:
    """OpenAI chat-completions with retry and per-process RPM cap."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        max_retries: int = 6,
        rpm: int = 60,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.max_retries = max_retries
        self._rpm = rpm
        self._lock = threading.Lock()
        self._call_ts: list[float] = []

    def _throttle(self):
        if self._rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._call_ts = [t for t in self._call_ts if now - t < 60.0]
            if len(self._call_ts) >= self._rpm:
                wait = 60.0 - (now - self._call_ts[0]) + 0.2
                time.sleep(max(0.0, wait))
                now = time.monotonic()
                self._call_ts = [t for t in self._call_ts if now - t < 60.0]
            self._call_ts.append(time.monotonic())

    def query(self, frame_paths: list[str], user_text: str) -> str:
        # Build multimodal content: images first, then text
        content: list[dict] = []
        for p in frame_paths:
            if not Path(p).exists():
                continue
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64_jpeg(p)}",
                    "detail": "low",
                },
            })
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=200,
                    temperature=0,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                else:
                    raise
        raise RuntimeError("Max retries exhausted")


# =============================================================================
# Step b — API QA
# =============================================================================

def step_b_gpt(
    scenes: list[dict],
    frames_dir: Path,
    output_dir: Path,
    evaluator: GPTEvaluator,
    resume: bool = True,
    no_kb: bool = False,
    generic_question: bool = False,
    workers: int = 4,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Step b: GPT QA  model={evaluator.model}  workers={workers}")
    print("=" * 60)

    qa_path = output_dir / "qa_results.json"
    results: dict = {}
    _lock = threading.Lock()

    if resume and qa_path.exists():
        with open(qa_path) as f:
            results = json.load(f)
        print(f"  Resuming — {len(results)} scenes already in {qa_path}")

    # ── Build flat task list ──────────────────────────────────────────────────
    tasks: list[tuple] = []
    for scene in scenes:
        sid = scene["scene_id"]
        scene_frames_dir = frames_dir / sid

        map_path = scene_frames_dir / "frame_map.json"
        if map_path.exists():
            with open(map_path) as f:
                raw = json.load(f)
            video_info: dict = raw.pop("_video_info", {})
            frame_map: dict = raw
        else:
            frame_map, video_info = {}, {}

        done = results.get(sid, {})
        if len(done) == len(scene["questions"]):
            continue

        for q in scene["questions"]:
            qid = q["id"]
            if qid in done:
                continue

            fi = q.get("frame_indices", [10, 30, 50])
            paths: list[str] = []
            for i in fi:
                mapped = frame_map.get(str(i))
                if mapped:
                    paths.extend(mapped)
                else:
                    paths.append(str(scene_frames_dir / f"frame_{i:04d}.jpg"))
            available = [p for p in paths if Path(p).exists()]

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
            gt_str = "yes" if int(q.get("answer", 1)) == 1 else "no"
            tasks.append((sid, qid, q, available, user_text, gt_str))

    total_q = sum(len(s["questions"]) for s in scenes)
    print(f"  {total_q - len(tasks)}/{total_q} already done — {len(tasks)} remaining")

    # ── Worker ───────────────────────────────────────────────────────────────
    def run_one(task: tuple):
        sid, qid, q, available, user_text, gt_str = task
        try:
            raw = evaluator.query(available, user_text)
            pred, conf, reason = parse_vlm_response(raw)
        except Exception as e:
            print(f"  ERROR {sid} q{qid}: {e}")
            pred, conf, reason = "no", 0.0, str(e)
        correct = pred == gt_str
        return sid, qid, {
            "question_text": q["question_text"],
            "dimensions":    q.get("dimensions", []),
            "difficulty":    q.get("difficulty", "easy"),
            "frame_indices": q.get("frame_indices", []),
            "ground_truth":  gt_str,
            "pred":          pred,
            "confidence":    conf,
            "correct":       correct,
            "reason":        reason,
        }

    def flush():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            sid, qid, entry = fut.result()
            mark = "✓" if entry["correct"] else "✗"
            print(f"  {mark} [{sid}] q{qid} | GT={entry['ground_truth']} "
                  f"PRED={entry['pred']} conf={entry['confidence']:.2f} | "
                  f"{entry['question_text'][:55]}")
            with _lock:
                results.setdefault(sid, {})[qid] = entry
                completed += 1
                if completed % 50 == 0:
                    flush()

    flush()
    print(f"\n  Done: {total_q} questions  |  saved → {qa_path}")
    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhyMetric GPT Evaluation (steps b + c)"
    )
    p.add_argument("--question_file", default="data/question_v3_with_kb.json",
                   help="Question file with KB (same as evaluate.py)")
    p.add_argument("--frames_dir", default=None,
                   help="Pre-extracted frames dir (required for step b; not needed for step c only)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", default="bc",
                   help="Steps to run: any of 'b', 'c' (default: bc)")
    p.add_argument("--model", default="gpt-5.4",
                   help="OpenAI model ID (default: gpt-5.4)")
    p.add_argument("--api_key", default=None,
                   help="OpenAI API key (fallback: OPENAI_API_KEY env var)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel API request threads (default: 4)")
    p.add_argument("--rpm", type=int, default=60,
                   help="Requests-per-minute cap for this process (default: 60)")
    p.add_argument("--video_model", default=None,
                   help="Video generator name stored in eval_summary.json")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--scene_ids_file", default=None,
                   help="Restrict evaluation to scene IDs listed in this file (one per line)")
    # Ablation flags — mirror evaluate.py
    p.add_argument("--no_kb", action="store_true", default=False,
                   help="[Ablation] Skip KB block in prompt")
    p.add_argument("--generic_question", action="store_true", default=False,
                   help="[Ablation] Use a single generic plausibility question")
    return p.parse_args()


def main():
    args = _parse_args()
    steps = set(args.steps.lower())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "b" in steps and args.frames_dir is None:
        raise SystemExit("--frames_dir is required for step b")
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(".")

    print(f"Loading scenes from: {args.question_file}")
    scenes, q_fps, q_total = load_scenes(args.question_file)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    if args.scene_ids_file:
        with open(args.scene_ids_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        scenes = [s for s in scenes if s["scene_id"] in allowed]
        print(f"  Restricted to {len(scenes)} scenes from {args.scene_ids_file}")
    print(f"  {len(scenes)} scenes  (benchmark: {q_fps}fps / {q_total} frames)")

    qa_results: dict = {}

    if "b" in steps:
        evaluator = GPTEvaluator(
            model=args.model,
            api_key=args.api_key,
            rpm=args.rpm,
        )
        qa_results = step_b_gpt(
            scenes, frames_dir, output_dir, evaluator,
            resume=args.resume,
            no_kb=args.no_kb,
            generic_question=args.generic_question,
            workers=args.workers,
        )
    else:
        qa_path = output_dir / "qa_results.json"
        if qa_path.exists():
            with open(qa_path) as f:
                qa_results = json.load(f)

    if "c" in steps:
        run_meta = {
            "video_model":    args.video_model or "unknown",
            "frames_dir":     str(frames_dir),
            "gpt_model":      args.model,
            "question_file":  args.question_file,
            "ablation": {
                "no_kb":            args.no_kb,
                "generic_question": args.generic_question,
                "scene_ids_file":   args.scene_ids_file,
            },
        }
        step_c(scenes, qa_results, output_dir, run_meta)


if __name__ == "__main__":
    main()
