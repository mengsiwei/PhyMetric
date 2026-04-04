#!/usr/bin/env python3
"""
PhyMetric Human Evaluation Web App.

Annotators log in with a name, then answer YES/NO for each physics question
while viewing sampled frames from the generated video.

Usage:
    cd /home/siweim/code/PhyMetric/human_eval
    python app.py \
        --question_file ../data/question_v2_with_kb.json \
        --frames_dir    ../output100/Wan2.1-T2V-14B/baseline/frames \
        --run_label     "Wan2.1-T2V-14B baseline" \
        --port          7860
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_file, abort,
)

app = Flask(__name__)
app.secret_key = "phymetric-human-eval-secret-2024"

# ── global config (populated at startup) ─────────────────────────────────────
CFG: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═════════════════════════════════════════════════════════════════════════════

def _safe_id(category: str, scene_key: str) -> str:
    import re
    return re.sub(r"[^\w\-.]", "_", f"{category}_{scene_key}")


def load_all_questions() -> list[dict]:
    """Flatten question_v2_with_kb.json into a list of question records."""
    with open(CFG["question_file"], "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    scene_idx = 0
    for phenomenon, scenes in data.get("main_categories", {}).items():
        for scene_key, scene_data in scenes.items():
            scene_id = _safe_id(phenomenon, scene_key)
            kb = scene_data.get("knowledge_base", {})
            for q in scene_data.get("questions", []):
                records.append({
                    "global_idx":    len(records),
                    "scene_idx":     scene_idx,
                    "scene_id":      scene_id,
                    "phenomenon":    phenomenon,
                    "scene_name":    scene_data.get("scene_name", scene_key),
                    "scene_caption": scene_data.get("scene_caption", ""),
                    "q_id":          q["id"],
                    "question_text": q["question_text"],
                    "dimensions":    q.get("dimensions", []),
                    "difficulty":    q.get("difficulty", "easy"),
                    "frame_indices": q.get("frame_indices", [10, 30, 50]),
                    "ground_truth":  q.get("answer", 1),  # 1=yes, 0=no
                    "knowledge_base": kb,
                })
            scene_idx += 1
    return records


def get_question_order(annotator_id: str, total: int) -> list[int]:
    """Deterministic shuffle per annotator (same order on resume)."""
    import random
    seed = int(hashlib.md5(annotator_id.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)
    return order


def get_frame_groups(scene_id: str, frame_indices: list[int]) -> dict[int, list[str]]:
    """Return {q_idx: [path, ...]} grouped by benchmark frame index."""
    frames_root = CFG.get("frames_dir", "")
    if not frames_root:
        return {}
    frames_dir = Path(frames_root) / scene_id
    map_path = frames_dir / "frame_map.json"

    groups: dict[int, list[str]] = {}
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            frame_map = json.load(f)
        for idx in frame_indices:
            paths = [p for p in frame_map.get(str(idx), []) if Path(p).exists()]
            if paths:
                groups[idx] = paths
    else:
        for idx in frame_indices:
            p = frames_dir / f"frame_{idx:04d}.jpg"
            if p.exists():
                groups[idx] = [str(p)]
    return groups


def get_video_token(scene_idx: int) -> str | None:
    """Return base64 token for the scene's video file, or None if not found."""
    video_dir = CFG.get("video_dir", "")
    if not video_dir:
        return None
    video_path = Path(video_dir) / f"prompt-{scene_idx}.mp4"
    if not video_path.exists():
        return None
    return base64.urlsafe_b64encode(str(video_path.resolve()).encode()).decode()


def load_annotations(annotator_id: str) -> dict:
    ann_path = Path(CFG["annotations_dir"]) / f"{annotator_id}.json"
    if ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"annotator_id": annotator_id, "run_label": CFG.get("run_label", ""), "answers": {}}


def save_annotations(annotator_id: str, annotations: dict):
    Path(CFG["annotations_dir"]).mkdir(parents=True, exist_ok=True)
    ann_path = Path(CFG["annotations_dir"]) / f"{annotator_id}.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def answer_key(scene_id: str, q_id: str) -> str:
    return f"{scene_id}__{q_id}"


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.before_request
def require_login():
    allowed = {"login", "static", "serve_frame", "serve_video", "progress", "export_all"}
    if request.endpoint and request.endpoint not in allowed and "annotator_id" not in session:
        return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            error = "Please enter your name."
        elif not name.replace("-", "").replace("_", "").isalnum():
            error = "Name should contain only letters, numbers, hyphens, underscores."
        else:
            session["annotator_id"] = name
            return redirect(url_for("annotate"))
    return render_template("login.html", error=error, run_label=CFG.get("run_label", ""))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/annotate")
def annotate():
    annotator_id = session["annotator_id"]
    questions = app.config["QUESTIONS"]
    order = get_question_order(annotator_id, len(questions))
    annotations = load_annotations(annotator_id)
    answers = annotations["answers"]

    # find next unanswered question in shuffled order
    current_pos = None
    for pos, global_idx in enumerate(order):
        q = questions[global_idx]
        key = answer_key(q["scene_id"], q["q_id"])
        if key not in answers:
            current_pos = pos
            break

    if current_pos is None:
        return redirect(url_for("done"))

    return redirect(url_for("annotate_question", pos=current_pos))


@app.route("/annotate/<int:pos>", methods=["GET"])
def annotate_question(pos: int):
    annotator_id = session["annotator_id"]
    questions = app.config["QUESTIONS"]
    order = get_question_order(annotator_id, len(questions))
    annotations = load_annotations(annotator_id)

    if pos < 0 or pos >= len(order):
        return redirect(url_for("annotate"))

    global_idx = order[pos]
    q = questions[global_idx]

    # frames grouped by benchmark index: {idx: [token, ...]}
    frame_groups_raw = get_frame_groups(q["scene_id"], q["frame_indices"])
    frame_groups = {
        idx: [base64.urlsafe_b64encode(p.encode()).decode() for p in paths]
        for idx, paths in frame_groups_raw.items()
    }
    # slide list: [{label, tokens}] ordered by frame index
    slides = []
    index_labels = {10: "Early (~0.33s)", 30: "Mid (~1.0s)", 50: "Late (~1.67s)"}
    for idx in q["frame_indices"]:
        tokens = frame_groups.get(idx, [])
        label = index_labels.get(idx, f"Frame {idx}")
        slides.append({"label": label, "idx": idx, "tokens": tokens})

    video_token = get_video_token(q["scene_idx"])

    answered_count = len(annotations["answers"])
    total = len(questions)
    key = answer_key(q["scene_id"], q["q_id"])
    existing_answer = annotations["answers"].get(key, {}).get("answer")

    prev_pos = pos - 1 if pos > 0 else None
    next_pos = pos + 1 if pos + 1 < len(order) else None

    return render_template(
        "annotate.html",
        q=q,
        pos=pos,
        total=total,
        answered=answered_count,
        slides=slides,
        video_token=video_token,
        existing_answer=existing_answer,
        prev_pos=prev_pos,
        next_pos=next_pos,
        run_label=CFG.get("run_label", ""),
        annotator_id=annotator_id,
    )


@app.route("/answer", methods=["POST"])
def submit_answer():
    annotator_id = session["annotator_id"]
    questions = app.config["QUESTIONS"]
    order = get_question_order(annotator_id, len(questions))
    annotations = load_annotations(annotator_id)

    pos = int(request.form["pos"])
    answer = request.form.get("answer", "").lower()
    if answer not in ("yes", "no"):
        abort(400)

    global_idx = order[pos]
    q = questions[global_idx]
    key = answer_key(q["scene_id"], q["q_id"])

    annotations["answers"][key] = {
        "scene_id":      q["scene_id"],
        "q_id":          q["q_id"],
        "question_text": q["question_text"],
        "answer":        answer,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_annotations(annotator_id, annotations)

    # go to next unanswered
    return redirect(url_for("annotate"))


@app.route("/done")
def done():
    annotator_id = session["annotator_id"]
    annotations = load_annotations(annotator_id)
    total = len(app.config["QUESTIONS"])
    answered = len(annotations["answers"])
    return render_template(
        "done.html",
        annotator_id=annotator_id,
        answered=answered,
        total=total,
        run_label=CFG.get("run_label", ""),
    )


@app.route("/progress")
def progress():
    """Admin overview of all annotators' progress."""
    questions = app.config["QUESTIONS"]
    total = len(questions)
    ann_dir = Path(CFG["annotations_dir"])
    annotators = []
    if ann_dir.exists():
        for f in sorted(ann_dir.glob("*.json")):
            with open(f) as fp:
                ann = json.load(fp)
            answered = len(ann.get("answers", {}))
            correct = sum(
                1 for key, v in ann["answers"].items()
                if _check_correct(key, v["answer"], questions)
            )
            annotators.append({
                "id":       ann["annotator_id"],
                "answered": answered,
                "total":    total,
                "pct":      round(answered / total * 100) if total else 0,
                "accuracy": round(correct / answered * 100, 1) if answered else 0,
                "correct":  correct,
            })
    return render_template(
        "progress.html",
        annotators=annotators,
        total=total,
        run_label=CFG.get("run_label", ""),
    )


def _check_correct(key: str, answer: str, questions: list[dict]) -> bool:
    """Check if annotator's answer matches ground truth."""
    scene_id, q_id = key.split("__", 1)
    for q in questions:
        if q["scene_id"] == scene_id and q["q_id"] == q_id:
            gt = "yes" if q["ground_truth"] == 1 else "no"
            return answer == gt
    return False


@app.route("/export/<annotator_id>")
def export_annotator(annotator_id: str):
    """Download a single annotator's JSON."""
    ann_path = Path(CFG["annotations_dir"]) / f"{annotator_id}.json"
    if not ann_path.exists():
        abort(404)
    return send_file(ann_path, as_attachment=True)


@app.route("/export_all")
def export_all():
    """Return merged annotations from all annotators as JSON."""
    ann_dir = Path(CFG["annotations_dir"])
    merged = {}
    if ann_dir.exists():
        for f in sorted(ann_dir.glob("*.json")):
            with open(f) as fp:
                ann = json.load(fp)
            merged[ann["annotator_id"]] = ann
    return jsonify(merged)


def _decode_token_path(token: str, allowed_root: str):
    try:
        path = base64.urlsafe_b64decode(token.encode()).decode()
    except Exception:
        abort(400)
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(Path(allowed_root).resolve())):
        abort(403)
    if not resolved.exists():
        abort(404)
    return resolved


@app.route("/frame/<token>")
def serve_frame(token: str):
    path = _decode_token_path(token, CFG.get("frames_dir", "/"))
    return send_file(path, mimetype="image/jpeg")


@app.route("/video/<token>")
def serve_video(token: str):
    path = _decode_token_path(token, CFG.get("video_dir", "/"))
    return send_file(path, mimetype="video/mp4", conditional=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PhyMetric Human Evaluation Server")
    parser.add_argument("--question_file", default="../data/question_v2_with_kb.json")
    parser.add_argument("--frames_dir",    default="",
                        help="Path to frames/ directory (optional if --video_dir given)")
    parser.add_argument("--video_dir",     default="",
                        help="Directory containing prompt-{idx}.mp4 files (optional)")
    parser.add_argument("--run_label",     default="",
                        help="Display label for this run (e.g. 'Wan2.2 baseline')")
    parser.add_argument("--annotations_dir", default="annotations")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.frames_dir and not args.video_dir:
        parser.error("At least one of --frames_dir or --video_dir is required.")

    CFG["question_file"]    = args.question_file
    CFG["frames_dir"]       = str(Path(args.frames_dir).resolve()) if args.frames_dir else ""
    CFG["video_dir"]        = str(Path(args.video_dir).resolve())  if args.video_dir  else ""
    CFG["run_label"]        = args.run_label
    CFG["annotations_dir"]  = args.annotations_dir

    questions = load_all_questions()
    app.config["QUESTIONS"] = questions
    print(f"Loaded {len(questions)} questions from {args.question_file}")
    if CFG["frames_dir"]: print(f"Frames dir : {CFG['frames_dir']}")
    if CFG["video_dir"]:  print(f"Video dir  : {CFG['video_dir']}")
    print(f"Run label  : {args.run_label or '(none)'}")
    print(f"Starting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
