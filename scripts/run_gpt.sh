#!/usr/bin/env bash
# Reviewer reproduction: GPT-5.4 vision API as judge.
#   1) Step a: extract frames locally (no GPU needed)
#   2) Step b+c: stream frames to OpenAI for QA + scoring
# Requires OPENAI_API_KEY in the environment.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="${1:-Wan2.2-14B}"
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running}"

OUT="$REPO/runs/gpt/$MODEL_NAME"
mkdir -p "$OUT"

# 1) Frame extraction — reuses evaluate.py step a only.
python "$REPO/evaluate.py" \
    --steps         a \
    --question_file "$REPO/data/sample_question_with_kb.json" \
    --video_dir     "$REPO/sample_videos/$MODEL_NAME" \
    --output_dir    "$OUT"

# 2) GPT QA + scoring.
python "$REPO/evaluate_gpt.py" \
    --steps         bc \
    --question_file "$REPO/data/sample_question_with_kb.json" \
    --frames_dir    "$OUT/frames" \
    --output_dir    "$OUT" \
    --video_model   "$MODEL_NAME" \
    --model         "gpt-5.4" \
    --workers       4 \
    --rpm           60
