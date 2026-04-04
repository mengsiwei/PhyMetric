#!/usr/bin/env bash
# Start the PhyMetric human annotation server.
# Edit the paths and label to match the run you want annotated.

cd "$(dirname "$0")"

FRAMES_DIR="../output100/Wan2.1-T2V-14B/baseline/frames"
VIDEO_DIR="../output100/Wan2.1-T2V-14B/baseline"   # directory containing prompt-{idx}.mp4
RUN_LABEL="Wan2.1-T2V-14B baseline"
PORT=7860

python app.py \
  --question_file ../data/question_v2_with_kb.json \
  --frames_dir    "${FRAMES_DIR}" \
  --video_dir     "${VIDEO_DIR}" \
  --run_label     "${RUN_LABEL}" \
  --annotations_dir annotations \
  --port          ${PORT}
