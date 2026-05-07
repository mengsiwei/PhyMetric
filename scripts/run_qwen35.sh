#!/usr/bin/env bash
# Reviewer reproduction: Qwen3.5-9B (text-only judge over sampled frames).
# Single-scene smoke test using the bundled sample.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="${1:-Wan2.2-14B}"   # which T2V model's video to judge
GPU="${GPU:-0}"

OUT="$REPO/runs/qwen35/$MODEL_NAME"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/evaluate.py" \
    --question_file "$REPO/data/sample_question_with_kb.json" \
    --video_dir     "$REPO/sample_videos/$MODEL_NAME" \
    --output_dir    "$OUT" \
    --vlm_model_id  "Qwen/Qwen3.5-9B" \
    --video_model   "$MODEL_NAME" \
    --steps         abc \
    --gpu           0
