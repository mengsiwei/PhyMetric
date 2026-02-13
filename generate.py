#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate ~N physically plausible captions for PhyMetric by template+slot sampling,
rewrite corresponding questions, and export JSONL + CSV + basic stats + charts.

Run:
  python generate.py \
    --input data/question_v2.json \
    --output_dir data/generated \
    --num_captions 2000 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Question:
    question_id: str
    question_text: str
    dimensions: List[str]
    difficulty: str
    frame_indices: List[int]
    answer: int

@dataclass
class Scene:
    phenomenon: str
    base_scene_id: str
    scene_name: str
    scene_caption: str
    questions: List[Question]

@dataclass
class CaptionItem:
    caption_id: str
    phenomenon: str
    base_scene_id: str
    caption: str
    slots: Dict[str, Any]
    questions: List[Question]
    provenance: Dict[str, Any]


# -----------------------------
# Utilities
# -----------------------------

_DIM_MAP = {
    "appearance/texture": "appearance",
    "texture": "appearance",
    "color": "appearance",
    "orientation": "spatial",
}

BAD_TOKENS = {
    "teleport", "teleportation", "hover", "floating in the air", "defy gravity",
    "magically", "vanish", "disappear instantly",
}

def norm_dim(d: str) -> str:
    d0 = d.strip().lower()
    return _DIM_MAP.get(d0, d0)

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_id(prefix: str, seed_str: str) -> str:
    return f"{prefix}_{sha1(seed_str)[:12]}"

def safe_article(noun_phrase: str) -> str:
    # simple heuristic for "a"/"an"
    if not noun_phrase:
        return "a"
    return "an" if noun_phrase[0].lower() in "aeiou" else "a"

def replace_object_refs(text: str, base_obj: str, new_obj: str) -> str:
    # Replace common variants. Keep it conservative.
    base = base_obj.strip()
    new = new_obj.strip()

    # full phrase replacements
    patterns = [
        (rf"\bthe\s+{re.escape(base)}\b", f"the {new}"),
        (rf"\ba\s+{re.escape(base)}\b", f"{safe_article(new)} {new}"),
        (rf"\ban\s+{re.escape(base)}\b", f"{safe_article(new)} {new}"),
        (rf"\b{re.escape(base)}\b", new),
    ]

    out = text
    for p, r in patterns:
        out = re.sub(p, r, out, flags=re.IGNORECASE)
    return out

def contains_bad_token(caption: str) -> bool:
    low = caption.lower()
    return any(t in low for t in BAD_TOKENS)


# -----------------------------
# Parsing PhyMetric JSON
# -----------------------------

def load_scenes(path: str) -> Tuple[Dict[str, Any], List[Scene]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    cats = data.get("main_categories", {})

    scenes: List[Scene] = []
    for phenomenon, scene_dict in cats.items():
        for scene_id, scene_data in scene_dict.items():
            q_list = []
            for q in scene_data.get("questions", []):
                dims = [norm_dim(x) for x in q.get("dimensions", [])]
                q_list.append(
                    Question(
                        question_id=str(q.get("id")),
                        question_text=str(q.get("question_text", "")),
                        dimensions=dims,
                        difficulty=str(q.get("difficulty", "")),
                        frame_indices=list(q.get("frame_indices", [])),
                        answer=int(q.get("answer", 0)),
                    )
                )
            scenes.append(
                Scene(
                    phenomenon=phenomenon,
                    base_scene_id=scene_id,
                    scene_name=str(scene_data.get("scene_name", "")),
                    scene_caption=str(scene_data.get("scene_caption", "")),
                    questions=q_list,
                )
            )
    return meta, scenes


# -----------------------------
# Phenomenon specs
# -----------------------------

def default_specs() -> Dict[str, Dict[str, Any]]:
    """
    Minimal, conservative slot pools.
    You should extend these pools over time.
    """
    return {
        "Buoyancy_Float": {
            "object_pool": [
                "rubber duck", "cork", "ping pong ball", "foam block", "toy boat",
                "empty plastic bottle", "apple", "tennis ball",
            ],
            "medium_pool": ["water"],
            "container_pool": ["a clear glass", "a bowl", "a bathtub", "a small tank"],
            "templates": [
                "{art} {object} is gently placed into {container} of {medium}. It displaces the {medium} and floats at the surface. The motion settles into a stable position.",
                "{art} {object} is released onto the {medium} in {container}. It bobs briefly and then floats with part of it above the surface.",
            ],
        },
        "Buoyancy_Sink": {
            "object_pool": [
                "metal spoon", "iron key", "coin", "small rock", "steel bolt",
                "ceramic cup", "metal washer",
            ],
            "medium_pool": ["water"],
            "container_pool": ["a clear glass", "a deep bowl", "a tank"],
            "templates": [
                "{art} {object} is dropped into {container} of {medium}. It sinks downward under gravity until it reaches the bottom.",
                "{art} {object} enters the {medium} and moves straight down. Small bubbles may trail behind it before it rests on the bottom.",
            ],
        },
        "Friction_High": {
            "object_pool": ["wooden block", "book", "cardboard box", "rubber eraser"],
            "surface_pool": ["rough wood", "thick grass", "rough concrete", "paper"],
            "templates": [
                "{art} {object} is pushed across {surface}. It slows quickly due to strong friction and comes to rest within a short distance.",
                "{art} {object} slides on {surface}. It decelerates steadily and stops without speeding up on its own.",
            ],
        },
        "Friction_Low": {
            "object_pool": ["curling stone", "lego block", "flat stone", "hockey puck"],
            "surface_pool": ["ice", "polished glass", "polished marble", "smooth plastic"],
            "templates": [
                "{art} {object} is released on {surface}. It glides smoothly forward with minimal resistance and gradually slows down.",
                "{art} {object} slides on {surface} in a steady direction. The motion is continuous and does not jump between positions.",
            ],
        },
        "Collision_Elastic": {
            "object_pool": ["rubber ball", "tennis ball", "basketball", "football"],
            "target_pool": ["a rigid wall", "the floor", "a racket"],
            "templates": [
                "{art} {object} strikes {target} and rebounds backward after impact. Any deformation at contact quickly recovers.",
                "{art} {object} moves toward {target}, makes contact, and bounces away with visible speed. The trajectory stays continuous.",
            ],
        },
        "Collision_Inelastic": {
            "object_pool": ["wet clay", "soft dough", "mud", "ripe tomato"],
            "target_pool": ["a wall", "a shirt", "a hard floor"],
            "templates": [
                "{art} {object} hits {target}, deforms strongly, and sticks or stays without bouncing back.",
                "{art} {object} collides with {target}. It flattens on impact and remains near the contact area instead of rebounding.",
            ],
        },
        "Refraction": {
            "object_pool": ["spoon", "straw", "pencil"],
            "container_pool": ["a clear glass of water", "a transparent cup of water"],
            "templates": [
                "{art} {object} is partially submerged in {container}. It appears bent at the water surface due to refraction.",
                "{art} {object} sits in {container}. The part seen through water looks shifted compared with the part in air.",
            ],
        },
        "Reflection": {
            "object_pool": ["person", "car", "dancer"],
            "mirror_pool": ["a flat mirror", "a glass window", "a wet road surface", "a calm lake"],
            "templates": [
                "{art} {object} moves in front of {mirror}. The reflection stays aligned and changes consistently with the motion.",
                "{art} {mirror} shows a clear reflection as {object} moves. The reflection position updates smoothly over time.",
            ],
        },
        "Viscous Flow": {
            "fluid_pool": ["honey", "maple syrup", "molasses", "thick oil", "jam"],
            "surface_pool": ["toast", "bread", "a plate", "pancakes"],
            "templates": [
                "{art} {fluid} is poured onto {surface}. It flows slowly and forms a thick, cohesive stream as it spreads.",
                "{art} {fluid} drips and stretches into a thick thread. The puddle grows and spreads gradually due to high viscosity.",
            ],
        },
        "Gravity_Earth": {
            "object_pool": ["stone", "book", "apple", "metal key", "tennis ball"],
            "templates": [
                "{art} {object} is released above the ground on Earth. It falls downward under gravity and hits the ground within the clip.",
                "{art} {object} drops from a hand and moves down in a continuous path. After impact it slows and comes to rest.",
            ],
        },
        "Gravity_Space": {
            "object_pool": ["stone", "wrench", "feather", "apple", "ping pong ball"],
            "templates": [
                "{art} {object} is released above the Moon surface. It falls downward more slowly in low gravity and reaches the ground.",
                "{art} {object} drops in a lunar environment. The descent is smooth and continuous before it contacts the surface.",
            ],
        },
        # For remaining phenomena, the script will fallback to auto-templating.
    }


# -----------------------------
# Constraint checks
# -----------------------------

def check_constraints(phenomenon: str, slots: Dict[str, Any], caption: str) -> Tuple[bool, str]:
    # Global checks
    if contains_bad_token(caption):
        return False, "contains_bad_token"

    # Phenomenon-specific light checks
    if phenomenon == "Buoyancy_Float":
        if slots.get("medium") != "water":
            return False, "float_medium_not_water"
        if "float" not in caption.lower():
            return False, "float_missing_keyword"

    if phenomenon == "Buoyancy_Sink":
        if slots.get("medium") != "water":
            return False, "sink_medium_not_water"
        if "sink" not in caption.lower() and "bottom" not in caption.lower():
            return False, "sink_missing_keyword"

    if phenomenon == "Refraction":
        if "water" not in caption.lower():
            return False, "refraction_missing_water"
        if "refraction" not in caption.lower() and "bent" not in caption.lower() and "shift" not in caption.lower():
            return False, "refraction_missing_cue"

    if phenomenon == "Reflection":
        if "reflection" not in caption.lower():
            return False, "reflection_missing_cue"

    if phenomenon == "Viscous Flow":
        if "slow" not in caption.lower() and "thick" not in caption.lower():
            return False, "viscous_missing_cue"

    if phenomenon == "Gravity_Space":
        if "moon" not in caption.lower() and "lunar" not in caption.lower():
            return False, "space_missing_moon"

    return True, "ok"


# -----------------------------
# Generation
# -----------------------------

CAMERA_PREFIX = [
    "A static camera shot shows",
    "A close-up shot shows",
    "From a side view,",
    "In a single continuous shot,",
]
LIGHTING_SUFFIX = [
    "The lighting is natural daylight.",
    "The scene is lit by indoor warm light.",
    "Soft light reduces harsh shadows.",
    "",
]

def auto_template_from_scene(scene_caption: str) -> str:
    # Keep original caption as a fallback template.
    # Add minimal slots by replacing leading article object phrase.
    # If extraction fails, return original caption.
    m = re.match(r"^\s*(A|An|The)\s+(.+?)\s+is\s+", scene_caption)
    if not m:
        return scene_caption
    obj_phrase = m.group(2).strip()
    out = scene_caption
    out = replace_object_refs(out, obj_phrase, "{object}")
    return out

def sample_caption_for_scene(
    rng: random.Random,
    scene: Scene,
    spec: Dict[str, Any],
    variant_idx: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    phenomenon = scene.phenomenon

    # Determine base object phrase from caption or scene_name
    # Prefer caption: the "is" boundary reliably captures multi-word noun phrases
    base_obj = None
    m_cap = re.match(r"^\s*(A|An|The)\s+(.+?)\s+is\s+", scene.scene_caption)
    if m_cap:
        base_obj = m_cap.group(2).strip()
    if not base_obj:
        m_name = re.match(r"^\s*(A|An|The)\s+(.+?)\s+(is|in|on|at|from|with|floating|sinking|sliding|moving|falling|hitting|striking)\s+", scene.scene_name, re.IGNORECASE)
        if m_name:
            base_obj = m_name.group(2).strip()
    if not base_obj:
        base_obj = "object"

    # Choose template
    templates = spec.get("templates") or [auto_template_from_scene(scene.scene_caption)]
    template_id = rng.randrange(len(templates))
    template = templates[template_id]

    # Build slots
    slots: Dict[str, Any] = {}

    if phenomenon in ("Buoyancy_Float", "Buoyancy_Sink"):
        obj = rng.choice(spec["object_pool"])
        medium = rng.choice(spec["medium_pool"])
        container = rng.choice(spec["container_pool"])
        slots.update({
            "object": obj,
            "medium": medium,
            "container": container,
            "art": safe_article(obj),
        })
    elif phenomenon in ("Friction_High", "Friction_Low"):
        obj = rng.choice(spec["object_pool"])
        surface = rng.choice(spec["surface_pool"])
        slots.update({
            "object": obj,
            "surface": surface,
            "art": safe_article(obj),
        })
    elif phenomenon in ("Collision_Elastic", "Collision_Inelastic"):
        obj = rng.choice(spec["object_pool"])
        target = rng.choice(spec["target_pool"])
        slots.update({
            "object": obj,
            "target": target,
            "art": safe_article(obj),
        })
    elif phenomenon == "Refraction":
        obj = rng.choice(spec["object_pool"])
        container = rng.choice(spec["container_pool"])
        slots.update({
            "object": obj,
            "container": container,
            "art": safe_article(obj),
        })
    elif phenomenon == "Reflection":
        obj = rng.choice(spec["object_pool"])
        mirror = rng.choice(spec["mirror_pool"])
        slots.update({
            "object": obj,
            "mirror": mirror,
            "art": safe_article(obj),
        })
    elif phenomenon == "Viscous Flow":
        fluid = rng.choice(spec["fluid_pool"])
        surface = rng.choice(spec["surface_pool"])
        slots.update({
            "fluid": fluid,
            "surface": surface,
            "art": safe_article(fluid),
        })
    elif phenomenon in ("Gravity_Earth", "Gravity_Space"):
        obj = rng.choice(spec["object_pool"])
        slots.update({
            "object": obj,
            "art": safe_article(obj),
        })
    else:
        # generic fallback
        obj = rng.choice(spec.get("object_pool", [base_obj]))
        slots.update({"object": obj, "art": safe_article(obj)})

    # Compose caption with camera + lighting variation
    prefix = rng.choice(CAMERA_PREFIX)
    suffix = rng.choice(LIGHTING_SUFFIX)
    body = template.format(**slots)

    # If template is already a full sentence, optionally keep it plain
    if rng.random() < 0.4:
        caption = body
    else:
        caption = f"{prefix} {body} {suffix}".strip()

    caption = re.sub(r"\s+", " ", caption).strip()

    provenance = {
        "base_scene_caption_sha1": sha1(scene.scene_caption),
        "base_object_phrase": base_obj,
        "template_id": template_id,
        "variant_idx": variant_idx,
        "ops": ["camera_prefix", "lighting_suffix", "slot_sampling"],
    }

    return caption, slots, provenance, base_obj


def rewrite_questions(scene: Scene, base_obj: str, new_obj: str) -> List[Question]:
    out: List[Question] = []
    for q in scene.questions:
        qt = q.question_text

        # Replace base object phrase occurrences
        qt2 = replace_object_refs(qt, base_obj, new_obj)

        # Also try to replace last token of base_obj if it is a single noun like "duck"
        base_last = base_obj.split()[-1].strip()
        new_last = new_obj.split()[-1].strip()
        if base_last and new_last and base_last.lower() != new_last.lower():
            qt2 = re.sub(rf"\b{re.escape(base_last)}\b", new_last, qt2, flags=re.IGNORECASE)

        out.append(
            Question(
                question_id=q.question_id,
                question_text=qt2,
                dimensions=q.dimensions,
                difficulty=q.difficulty,
                frame_indices=q.frame_indices,
                answer=q.answer,
            )
        )
    return out


def generate_captions(
    scenes: List[Scene],
    specs: Dict[str, Dict[str, Any]],
    num_captions: int,
    seed: int,
) -> List[CaptionItem]:
    rng = random.Random(seed)

    # Group scenes by phenomenon
    by_ph: Dict[str, List[Scene]] = {}
    for s in scenes:
        by_ph.setdefault(s.phenomenon, []).append(s)

    phenomena = sorted(by_ph.keys())
    if not phenomena:
        raise ValueError("No scenes found.")

    # Uniform quota by phenomenon
    per_ph = max(1, num_captions // len(phenomena))
    remainder = num_captions - per_ph * len(phenomena)

    items: List[CaptionItem] = []
    seen = set()

    for i, ph in enumerate(phenomena):
        target = per_ph + (1 if i < remainder else 0)
        ph_scenes = by_ph[ph]
        ph_scenes = sorted(ph_scenes, key=lambda x: x.base_scene_id)

        # per-scene quota
        per_scene = max(1, target // len(ph_scenes))
        extra = target - per_scene * len(ph_scenes)

        ph_spec = specs.get(ph, None)
        if ph_spec is None:
            # fallback spec from extracted objects
            ph_spec = {
                "object_pool": ["object"],
                "templates": [auto_template_from_scene(ph_scenes[0].scene_caption)],
            }

        for j, scene in enumerate(ph_scenes):
            scene_target = per_scene + (1 if j < extra else 0)
            tries = 0
            made = 0
            while made < scene_target and tries < scene_target * 80:
                tries += 1
                caption, slots, prov, base_obj = sample_caption_for_scene(
                    rng=rng, scene=scene, spec=ph_spec, variant_idx=tries
                )

                ok, reason = check_constraints(ph, slots, caption)
                if not ok:
                    continue

                key = normalize_text(caption)
                if key in seen:
                    continue

                # rewrite questions if object exists
                new_obj = slots.get("object", base_obj)
                new_questions = rewrite_questions(scene, base_obj, new_obj)

                caption_id = make_id("cap", f"{ph}|{scene.base_scene_id}|{caption}|{seed}")
                prov2 = dict(prov)
                prov2["constraint_reason"] = reason
                prov2["seed"] = seed

                item = CaptionItem(
                    caption_id=caption_id,
                    phenomenon=ph,
                    base_scene_id=scene.base_scene_id,
                    caption=caption,
                    slots=slots,
                    questions=new_questions,
                    provenance=prov2,
                )
                items.append(item)
                seen.add(key)
                made += 1

    # If due to constraints we are short, do a second pass with relaxed sampling
    if len(items) < num_captions:
        # Simple top-up: random pick scenes without strict per-phenomenon quota
        all_scenes = scenes[:]
        while len(items) < num_captions:
            scene = rng.choice(all_scenes)
            ph = scene.phenomenon
            ph_spec = specs.get(ph, {"object_pool": ["object"], "templates": [scene.scene_caption]})
            caption, slots, prov, base_obj = sample_caption_for_scene(rng, scene, ph_spec, len(items))
            ok, reason = check_constraints(ph, slots, caption)
            if not ok:
                continue
            key = normalize_text(caption)
            if key in seen:
                continue
            new_obj = slots.get("object", base_obj)
            new_questions = rewrite_questions(scene, base_obj, new_obj)
            caption_id = make_id("cap", f"{ph}|{scene.base_scene_id}|{caption}|{seed}")
            prov["constraint_reason"] = reason
            prov["seed"] = seed
            items.append(
                CaptionItem(
                    caption_id=caption_id,
                    phenomenon=ph,
                    base_scene_id=scene.base_scene_id,
                    caption=caption,
                    slots=slots,
                    questions=new_questions,
                    provenance=prov,
                )
            )
            seen.add(key)

    return items[:num_captions]


# -----------------------------
# Export + metrics
# -----------------------------

def export_jsonl(items: List[CaptionItem], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            row = {
                "caption_id": it.caption_id,
                "phenomenon": it.phenomenon,
                "base_scene_id": it.base_scene_id,
                "caption": it.caption,
                "slots": it.slots,
                "questions": [asdict(q) for q in it.questions],
                "provenance": it.provenance,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def export_csv(items: List[CaptionItem], out_path: str) -> None:
    if pd is None:
        return
    rows = []
    for it in items:
        for q in it.questions:
            rows.append({
                "caption_id": it.caption_id,
                "phenomenon": it.phenomenon,
                "base_scene_id": it.base_scene_id,
                "caption": it.caption,
                "question_id": q.question_id,
                "question_text": q.question_text,
                "dimensions": "|".join(q.dimensions),
                "difficulty": q.difficulty,
                "frame_indices": "|".join(str(x) for x in q.frame_indices),
                "answer": q.answer,
                "provenance": json.dumps(it.provenance, ensure_ascii=False),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

def tokenize_simple(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in s.split() if t]

def distinct_n(tokens_list: List[List[str]], n: int) -> float:
    total = 0
    uniq = set()
    for toks in tokens_list:
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            uniq.add(tuple(toks[i:i+n]))
            total += 1
    return (len(uniq) / total) if total > 0 else 0.0

def compute_stats(items: List[CaptionItem]) -> Dict[str, Any]:
    cap_texts = [it.caption for it in items]
    toks_list = [tokenize_simple(t) for t in cap_texts]
    lengths = [len(toks) for toks in toks_list]

    by_ph = {}
    for it in items:
        by_ph[it.phenomenon] = by_ph.get(it.phenomenon, 0) + 1

    stats = {
        "num_captions": len(items),
        "phenomenon_counts": dict(sorted(by_ph.items(), key=lambda x: (-x[1], x[0]))),
        "avg_len_tokens": sum(lengths) / max(1, len(lengths)),
        "min_len_tokens": min(lengths) if lengths else 0,
        "max_len_tokens": max(lengths) if lengths else 0,
        "distinct_1": distinct_n(toks_list, 1),
        "distinct_2": distinct_n(toks_list, 2),
    }
    return stats

def save_stats(stats: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def plot_basic(items: List[CaptionItem], out_dir: str) -> None:
    if plt is None or pd is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    # Phenomenon bar chart
    by_ph = {}
    for it in items:
        by_ph[it.phenomenon] = by_ph.get(it.phenomenon, 0) + 1
    df = pd.DataFrame({"phenomenon": list(by_ph.keys()), "count": list(by_ph.values())})
    df = df.sort_values("count", ascending=False)

    plt.figure()
    plt.bar(df["phenomenon"], df["count"])
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phenomenon_counts.png"))
    plt.close()

    # Caption length histogram
    lens = [len(tokenize_simple(it.caption)) for it in items]
    plt.figure()
    plt.hist(lens, bins=30)
    plt.xlabel("caption length (tokens)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "caption_length_hist.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data/question_v2.json")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--num_captions", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta, scenes = load_scenes(args.input)
    specs = default_specs()

    items = generate_captions(
        scenes=scenes,
        specs=specs,
        num_captions=args.num_captions,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"captions_{args.num_captions}.jsonl")
    out_csv = os.path.join(args.output_dir, f"captions_{args.num_captions}_flat.csv")
    out_stats = os.path.join(args.output_dir, f"captions_{args.num_captions}_stats.json")
    out_fig_dir = os.path.join(args.output_dir, "figures")

    export_jsonl(items, out_jsonl)
    export_csv(items, out_csv)
    stats = compute_stats(items)
    stats["input_metadata"] = meta
    save_stats(stats, out_stats)
    plot_basic(items, out_fig_dir)

    print(json.dumps({
        "jsonl": out_jsonl,
        "csv": out_csv if pd is not None else None,
        "stats": out_stats,
        "figures": out_fig_dir if plt is not None else None,
        "num_captions": len(items),
    }, indent=2))

if __name__ == "__main__":
    main()
