# PhyMetric Human Evaluation Web (`human_eval`)

Flask-based annotation UI: after login, annotators answer physics yes/no questions in shuffled order while viewing sampled frames and optional full video. Results are saved as local JSON files.

## Environment

- **Python**: **3.10+** recommended (the code uses modern typing such as `str | None`).
- **Install dependencies** (from the `human_eval` directory):

```bash
cd human_eval
python -m pip install -r requirements.txt
```

`requirements.txt` lists only what **`app.py` / `run_server.sh`** need (currently **Flask**). **`analyze.py`** uses the standard library only.

---

## Paths and layout

Paths below are described relative to the **PhyMetric repo root**; scripts that use relative paths assume the working directory is **`human_eval/`**.

| Role | Default / example (relative to `human_eval`) | Notes |
|------|-----------------------------------------------|--------|
| Question JSON | `../data/question_v2_with_kb.json` | Same schema as `evaluate.py` |
| Frames root `frames_dir` | `../output100/.../frames` | One subdirectory per `scene_id` (see below) |
| Video dir `video_dir` | `../output100/.../baseline` | Directory containing `prompt-{idx}.mp4` (or parent layout as in examples) |
| Annotations `annotations_dir` | `annotations/` (default) | One `{annotator_id}.json` per annotator |

Edit the variables at the top of `run_server.sh` before launch.

---

## Configuration

### 1. `run_server.sh` (recommended)

Edit `human_eval/run_server.sh`:

- **`FRAMES_DIR`**: Root `frames` directory containing one folder per `scene_id`.
- **`VIDEO_DIR`**: Directory with `prompt-{idx}.mp4` files (`idx` is the 0-based scene index in the question file).
- **`RUN_LABEL`**: Label shown on the login page (any string).
- **`PORT`**: Listen port (default `7860`).

The script invokes:

```text
python app.py \
  --question_file ../data/question_v2_with_kb.json \
  --frames_dir    "${FRAMES_DIR}" \
  --video_dir     "${VIDEO_DIR}" \
  --run_label     "${RUN_LABEL}" \
  --annotations_dir annotations \
  --port          ${PORT}
```

### 2. Run `app.py` directly

```bash
python app.py \
  --question_file ../data/question_v2_with_kb.json \
  --frames_dir    /path/to/frames \
  --video_dir     /path/to/videos \
  --run_label     "My run label" \
  --annotations_dir annotations \
  --port 7860 \
  --host 0.0.0.0
```

- **`--frames_dir` and `--video_dir`**: At least one is required; both may be set to show frames and video.
- **`--debug`**: OK for local development; avoid in production.

---

## Data preparation

### 1. Question file (`--question_file`)

Use the same **`question_v2_with_kb.json`** structure as the main repo. Top level includes:

- **`main_categories`**: phenomenon → scene key → scene payload.
- Each scene has **`questions`** (`id`, `question_text`, `frame_indices`, `answer`, etc.), **`knowledge_base`**, **`scene_caption`**, etc.
- Optional **`metadata`**: `fps`, `total_frames` for the benchmark timeline of `frame_indices` (aligned with `evaluate.py`).

`scene_id` matches `evaluate.py`: derived from category name + scene key with unsafe characters replaced by `_`.

### 2. Video naming (`--video_dir`)

Must match PhyMetric generation scripts:

```text
prompt-0.mp4, prompt-1.mp4, ...
```

**`idx` follows the iteration order of scenes in the question file**: categories in `main_categories` order, scenes in sorted key order within each category; the first scene is `0`, then `1`, … Same as `load_scenes()` in repo-root `evaluate.py`.

### 3. Frames layout (`--frames_dir`)

For each **`scene_id`**, create `<frames_dir>/<scene_id>/`.

Two layouts are supported (same as `app.py` / `evaluate.py`):

1. **Recommended (matches automated eval)**: **`frame_map.json`** whose keys are stringified frame indices used in questions (e.g. `"10"`, `"30"`, `"50"`), values are lists of **absolute or relative** image paths (files must exist).
2. **Simple fallback**: Without `frame_map.json`, use single files **`frame_{idx:04d}.jpg`** (e.g. `frame_0010.jpg`) where `idx` is the benchmark frame index from the question.

If you already ran **`evaluate.py` Step a** only, the output usually already has per-`scene_id` folders and `frame_map.json`; point **`--frames_dir`** at that `frames` root.

Example (aligned with `run_server.sh` comments):

```text
../output100/Wan2.1-T2V-14B/baseline/
  prompt-0.mp4
  prompt-1.mp4
  ...
  frames/
    <scene_id_0>/
      frame_map.json
      frame_*.jpg
    <scene_id_1>/
      ...
```

Set **`VIDEO_DIR`** to `../output100/Wan2.1-T2V-14B/baseline` and **`FRAMES_DIR`** to `../output100/Wan2.1-T2V-14B/baseline/frames`.

### 4. Frames-only or video-only

- Frames only: set **`--frames_dir`** only; video panel may be empty.
- Video only: set **`--video_dir`** only; frame carousel may be empty if no frames are found.

---

## Launch

```bash
cd human_eval
chmod +x run_server.sh   # once
./run_server.sh
```

Open **`http://<host>:<PORT>/`** (local: `http://127.0.0.1:7860`).

---

## Outputs and analysis

- **Per annotator**: `annotations/<annotator_id>.json`.
- **Progress overview**: `/progress` (listed in `app.py` as exempt from login; see `before_request` for current behavior).
- **Aggregate stats** (stdlib only):

```bash
python analyze.py \
  --annotations_dir annotations \
  --question_file   ../data/question_v2_with_kb.json \
  --vlm_summary     ../output100/.../baseline/eval_summary.json   # optional: compare to VLM
```

Writes a short summary to `annotations/human_summary.json`.

---
