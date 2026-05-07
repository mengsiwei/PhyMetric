#!/usr/bin/env bash
# Reviewer reproduction: Qwen2.5-VL-7B-Instruct (vision-language judge).
# Single-scene smoke test using the bundled sample.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="${1:-Wan2.2-14B}"
GPU="${GPU:-0}"

OUT="$REPO/runs/qwen25vl/$MODEL_NAME"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/evaluate.py" \
    --question_file "$REPO/data/sample_question_with_kb.json" \
    --video_dir     "$REPO/sample_videos/$MODEL_NAME" \
    --output_dir    "$OUT" \
    --vlm_model_id  "Qwen/Qwen2.5-VL-7B-Instruct" \
    --video_model   "$MODEL_NAME" \
    --steps         abc \
    --gpu           0
