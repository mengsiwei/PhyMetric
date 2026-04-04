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
    return {
        "Buoyancy_Float": {
            "object_pool": [
                "rubber duck", "cork", "ping pong ball", "foam block", "toy boat",
                "empty plastic bottle", "apple", "tennis ball", "wooden log", "plastic lid",
                "inflatable ring", "walnut shell", "hollow gourd", "balsa wood plank", "styrofoam cup",
                "empty tin can", "coconut shell", "bar of soap", "foam peanut", "rubber stopper",
                "toy sailboat", "cork board", "bamboo stick", "plastic spoon", "leaf",
                "foam dart", "paper boat", "plastic egg", "foam wedge", "rubber ring",
                "wine cork", "plastic float", "hollow ball", "empty gourd", "foam disc",
                "toy buoy", "wooden dowel", "inflatable cube", "plastic cup", "cork tile",
                "foam sphere", "rubber mat", "plastic tube", "hollow cylinder", "foam pad",
                "toy kayak", "wax block", "plastic shell", "foam brick", "rubber duck",
                "wooden spool", "foam ring",
            ],
            "medium_pool": ["water"],
            "container_pool": ["a clear glass", "a bowl", "a bathtub", "a small tank", "a bucket", "a large tub"],
            "templates": [
                "{art} {object} is gently placed into {container} of {medium}. It displaces the {medium} and floats at the surface. The motion settles into a stable position.",
                "{art} {object} is released onto the {medium} in {container}. It bobs briefly and then floats with part of it above the surface.",
            ],
        },
        "Buoyancy_Sink": {
            "object_pool": [
                "metal spoon", "iron key", "coin", "small rock", "steel bolt",
                "ceramic cup", "metal washer", "lead weight", "steel ball bearing", "iron nail",
                "copper pipe", "zinc plate", "stainless spoon", "metal ring", "cast iron weight",
                "glass marble", "stone pebble", "ceramic tile", "metal hook", "steel rod",
                "iron screw", "copper coin", "metal clip", "steel chain", "iron plug",
                "brass knob", "metal cap", "steel disc", "iron bar", "copper weight",
                "metal token", "steel pin", "iron bracket", "copper bolt", "metal gear",
                "steel cube", "iron sphere", "copper ring", "metal button", "steel link",
                "iron flange", "copper sheet", "metal block", "steel nut", "iron cylinder",
                "copper tube", "metal wedge", "steel plate", "iron chip", "brass weight",
                "metal anchor", "copper disc",
            ],
            "medium_pool": ["water"],
            "container_pool": ["a clear glass", "a deep bowl", "a tank", "a bucket", "a large container"],
            "templates": [
                "{art} {object} is dropped into {container} of {medium}. It sinks downward under gravity until it reaches the bottom.",
                "{art} {object} enters the {medium} and moves straight down. Small bubbles may trail behind it before it rests on the bottom.",
            ],
        },
        "Friction_High": {
            "object_pool": [
                "wooden block", "book", "cardboard box", "rubber eraser", "brick",
                "ceramic mug", "heavy backpack", "thick rug", "rubber mat", "wooden crate",
                "foam block", "canvas bag", "rubber glove", "cork board", "sandpaper sheet",
                "felt pad", "rubber boot", "burlap sack", "rubber tile", "woolen cloth",
                "leather shoe", "rubber wheel", "carpet square", "wooden pallet", "cork pad",
                "rubber stopper", "fabric pouch", "rubber block", "wooden drawer", "cork mat",
                "rubber seal", "fabric roll", "rubber plate", "wooden panel", "cork tile",
                "rubber cube", "fabric bundle", "rubber disc", "wooden slab", "cork wedge",
                "rubber strip", "fabric square", "rubber wedge", "wooden box", "cork block",
                "rubber sheet", "fabric pad", "rubber ring", "wooden beam", "cork ring",
                "rubber cylinder", "fabric piece",
            ],
            "surface_pool": ["rough wood", "thick grass", "rough concrete", "paper", "sandpaper", "asphalt", "rough brick"],
            "templates": [
                "{art} {object} is pushed across {surface}. It slows quickly due to strong friction and comes to rest within a short distance.",
                "{art} {object} slides on {surface}. It decelerates steadily and stops without speeding up on its own.",
            ],
        },
        "Friction_Low": {
            "object_pool": [
                "curling stone", "lego block", "flat stone", "hockey puck", "metal disc",
                "glass bead", "steel ball", "polished rock", "ceramic disc", "metal plate",
                "glass tile", "polished coin", "steel washer", "ceramic ball", "metal ring",
                "glass sphere", "polished token", "steel disc", "ceramic tile", "metal slab",
                "glass rod", "polished gem", "steel cylinder", "ceramic rod", "metal slug",
                "glass block", "polished pebble", "steel puck", "ceramic puck", "metal pebble",
                "glass chip", "polished cube", "steel slug", "ceramic slug", "metal cube",
                "glass wedge", "polished block", "steel token", "ceramic token", "metal token",
                "glass pad", "polished pad", "steel block", "ceramic block", "metal block",
                "glass square", "polished square", "steel square", "ceramic square", "metal square",
                "glass cylinder", "polished cylinder",
            ],
            "surface_pool": ["ice", "polished glass", "polished marble", "smooth plastic", "waxed wood", "smooth tile"],
            "templates": [
                "{art} {object} is released on {surface}. It glides smoothly forward with minimal resistance and gradually slows down.",
                "{art} {object} slides on {surface} in a steady direction. The motion is continuous and does not jump between positions.",
            ],
        },
        "Collision_Elastic": {
            "object_pool": [
                "rubber ball", "tennis ball", "basketball", "football", "ping pong ball",
                "bouncy ball", "squash ball", "racquetball", "handball", "rubber sphere",
                "elastic ball", "super ball", "golf ball", "billiard ball", "marble",
                "steel ball bearing", "rubber pellet", "foam ball", "air ball", "spring ball",
                "rubber cube", "elastic sphere", "bounce pad", "rubber disc", "spring disc",
                "elastic ring", "rubber cylinder", "bounce ring", "rubber token", "spring token",
                "elastic cube", "rubber wedge", "bounce wedge", "rubber block", "spring block",
                "elastic disc", "rubber slug", "bounce slug", "rubber chip", "spring chip",
                "elastic wedge", "rubber puck", "bounce puck", "rubber cap", "spring cap",
                "elastic puck", "rubber pad", "bounce pad", "rubber tile", "spring tile",
                "elastic tile", "rubber slab",
            ],
            "target_pool": ["a rigid wall", "the floor", "a racket", "a hard surface", "a steel plate", "a wooden wall"],
            "templates": [
                "{art} {object} strikes {target} and rebounds backward after impact. Any deformation at contact quickly recovers.",
                "{art} {object} moves toward {target}, makes contact, and bounces away with visible speed. The trajectory stays continuous.",
            ],
        },
        "Collision_Inelastic": {
            "object_pool": [
                "wet clay", "soft dough", "mud", "ripe tomato", "putty",
                "wet sand", "soft wax", "ripe banana", "soft cheese", "wet cement",
                "play dough", "soft butter", "ripe peach", "soft paste", "wet plaster",
                "modeling clay", "soft gel", "ripe plum", "soft foam", "wet pulp",
                "sculptor clay", "soft resin", "ripe fig", "soft mortar", "wet mud",
                "pottery clay", "soft mastic", "ripe grape", "soft putty", "wet clay ball",
                "art clay", "soft compound", "ripe berry", "soft wax ball", "wet earth",
                "ceramic clay", "soft material", "ripe fruit", "soft blob", "wet paste",
                "terracotta clay", "soft mass", "ripe mango", "soft chunk", "wet mix",
                "slip clay", "soft lump", "ripe kiwi", "soft piece", "wet aggregate",
                "fire clay", "soft block",
            ],
            "target_pool": ["a wall", "a shirt", "a hard floor", "a flat surface", "a concrete block"],
            "templates": [
                "{art} {object} hits {target}, deforms strongly, and sticks or stays without bouncing back.",
                "{art} {object} collides with {target}. It flattens on impact and remains near the contact area instead of rebounding.",
            ],
        },
        "Refraction": {
            "object_pool": [
                "spoon", "straw", "pencil", "chopstick", "ruler",
                "wooden stick", "metal rod", "glass rod", "plastic tube", "bamboo skewer",
                "knitting needle", "paint brush", "dowel rod", "drinking straw", "toothpick",
                "thin candle", "pen", "marker", "drumstick", "baton",
                "thin branch", "wire", "thin pipe", "thin rod", "wand",
                "pointer stick", "thin bar", "narrow tube", "thin pole", "slim rod",
                "slim tube", "thin stem", "narrow stick", "thin shaft", "slim straw",
                "narrow rod", "thin wire", "slim pencil", "narrow pen", "thin twig",
                "slim bar", "narrow bar", "thin strip", "slim strip", "narrow strip",
                "thin dowel", "slim dowel", "narrow dowel", "thin skewer", "slim skewer",
                "narrow skewer", "thin brush",
            ],
            "container_pool": ["a clear glass of water", "a transparent cup of water", "a glass bowl of water", "a fish tank"],
            "templates": [
                "{art} {object} is partially submerged in {container}. It appears bent at the water surface due to refraction.",
                "{art} {object} sits in {container}. The part seen through water looks shifted compared with the part in air.",
            ],
        },
        "Reflection": {
            "object_pool": [
                "person", "car", "dancer", "cyclist", "runner",
                "dog", "bird", "cat", "horse", "child",
                "pedestrian", "athlete", "skater", "swimmer", "jogger",
                "robot", "figure", "model", "actor", "performer",
                "basketball player", "tennis player", "soccer player", "gymnast", "acrobat",
                "musician", "painter", "photographer", "teacher", "engineer",
                "firefighter", "police officer", "doctor", "chef", "pilot",
                "driver", "passenger", "tourist", "visitor", "spectator",
                "balloon", "kite", "umbrella", "flag", "lantern",
                "statue", "sculpture", "mannequin", "puppet", "toy figure",
                "bird in flight", "butterfly", "leaf", "flower", "tree branch",
            ],
            "mirror_pool": ["a flat mirror", "a glass window", "a wet road surface", "a calm lake", "a polished metal surface", "a still pond"],
            "templates": [
                "{art} {object} moves in front of {mirror}. The reflection stays aligned and changes consistently with the motion.",
                "{art} {mirror} shows a clear reflection as {object} moves. The reflection position updates smoothly over time.",
            ],
        },
        "Viscous Flow": {
            "fluid_pool": [
                "honey", "maple syrup", "molasses", "thick oil", "jam",
                "caramel sauce", "chocolate sauce", "condensed milk", "peanut butter", "tahini",
                "fruit preserve", "corn syrup", "golden syrup", "treacle", "agave nectar",
                "thick gravy", "custard", "cream sauce", "yogurt", "sour cream",
                "mayonnaise", "ketchup", "mustard", "hummus", "salsa",
                "tomato paste", "apple butter", "date syrup", "rice syrup", "barley malt",
                "fig jam", "plum jam", "apricot jam", "peach preserve", "berry compote",
                "mango chutney", "tamarind paste", "coconut cream", "almond butter", "cashew butter",
                "sunflower butter", "walnut paste", "hazelnut spread", "pistachio cream", "macadamia paste",
                "thick soup", "bean paste", "lentil puree", "chickpea spread", "black bean paste",
                "sweet potato puree", "pumpkin puree",
            ],
            "surface_pool": ["toast", "bread", "a plate", "pancakes", "a cutting board", "a wooden spoon", "a flat surface"],
            "templates": [
                "{art} {fluid} is poured onto {surface}. It flows slowly and forms a thick, cohesive stream as it spreads.",
                "{art} {fluid} drips and stretches into a thick thread. The puddle grows and spreads gradually due to high viscosity.",
            ],
        },
        "Gravity_Earth": {
            "object_pool": [
                "stone", "book", "apple", "metal key", "tennis ball",
                "orange", "pear", "walnut", "coin", "marble",
                "hammer", "wrench", "nail", "bolt", "washer",
                "pen", "pencil", "ruler", "eraser", "stapler",
                "mug", "cup", "glass", "bottle", "jar",
                "phone", "remote", "calculator", "battery", "charger",
                "sock", "glove", "hat", "scarf", "belt",
                "spoon", "fork", "knife", "plate", "bowl",
                "ball", "toy car", "dice", "block", "bead",
                "clip", "pin", "button", "ring", "badge",
                "seed", "nut", "bean", "pebble", "chip",
                "lemon", "lime", "mango",
            ],
            "templates": [
                "{art} {object} is released above the ground on Earth. It falls downward under gravity and hits the ground within the clip.",
                "{art} {object} drops from a hand and moves down in a continuous path. After impact it slows and comes to rest.",
            ],
        },
        "Gravity_Space": {
            "object_pool": [
                "stone", "wrench", "feather", "apple", "ping pong ball",
                "bolt", "washer", "coin", "pen", "pencil",
                "marble", "bead", "clip", "pin", "button",
                "toy car", "dice", "block", "chip", "ring",
                "seed", "nut", "bean", "pebble", "key",
                "rubber ball", "foam block", "plastic disc", "metal slug", "glass bead",
                "paper ball", "cotton ball", "clay ball", "wax ball", "resin bead",
                "plastic bead", "wooden bead", "ceramic bead", "metal bead", "rubber bead",
                "foam bead", "silicone bead", "nylon bead", "acrylic bead", "vinyl bead",
                "polystyrene ball", "polyurethane ball", "latex ball", "silicone ball", "neoprene ball",
                "cork ball", "balsa ball", "bamboo bead",
            ],
            "templates": [
                "{art} {object} is released above the Moon surface. It falls downward more slowly in low gravity and reaches the ground.",
                "{art} {object} drops in a lunar environment. The descent is smooth and continuous before it contacts the surface.",
            ],
        },
        "Cloth Simulation": {
            "object_pool": [
                "cotton shirt", "linen scarf", "silk handkerchief", "woolen blanket", "polyester flag",
                "cotton towel", "linen tablecloth", "silk ribbon", "woolen shawl", "polyester banner",
                "denim jacket", "velvet curtain", "satin sheet", "flannel cloth", "canvas tarp",
                "cotton dress", "linen napkin", "silk veil", "woolen sweater", "polyester cape",
                "cotton sheet", "linen cloth", "silk sash", "woolen cloak", "nylon parachute",
                "cotton curtain", "linen drape", "silk wrap", "woolen mat", "nylon flag",
                "cotton bandana", "linen square", "silk piece", "woolen fabric", "nylon sheet",
                "cotton cloth", "linen strip", "silk strip", "woolen strip", "nylon strip",
                "cotton fabric", "linen fabric", "silk fabric", "woolen cloth", "nylon cloth",
                "cotton panel", "linen panel", "silk panel", "woolen panel", "nylon panel",
                "cotton drape", "linen wrap",
            ],
            "templates": [
                "{art} {object} is draped over a surface. It falls under gravity and settles into natural folds and wrinkles.",
                "{art} {object} hangs from a support and sways gently. The fabric folds flow and deform in a realistic manner.",
            ],
        },
        "Elastic Deformation": {
            "object_pool": [
                "rubber band", "spring", "elastic cord", "rubber strip", "bungee cord",
                "silicone tube", "latex sheet", "rubber mat", "elastic band", "foam pad",
                "rubber glove", "silicone ring", "latex balloon", "rubber ball", "elastic rope",
                "rubber seal", "silicone pad", "latex tube", "rubber washer", "elastic strap",
                "rubber hose", "silicone strip", "latex strip", "rubber ring", "elastic wire",
                "rubber sheet", "silicone sheet", "latex sheet", "rubber block", "elastic mesh",
                "rubber pad", "silicone cord", "latex cord", "rubber cord", "elastic film",
                "rubber film", "silicone film", "latex film", "rubber membrane", "elastic membrane",
                "silicone membrane", "latex membrane", "rubber disc", "silicone disc", "latex disc",
                "rubber cylinder", "silicone cylinder", "latex cylinder", "rubber cube", "silicone cube",
                "latex cube", "rubber slab",
            ],
            "templates": [
                "{art} {object} is stretched and then released. It snaps back toward its original shape without permanent deformation.",
                "{art} {object} is compressed under force. When the load is removed, it recovers its original form elastically.",
            ],
        },
        "Explosion": {
            "object_pool": [
                "firecracker", "balloon", "paper bag", "foam block", "cardboard box",
                "plastic container", "glass bottle", "ceramic pot", "wooden crate", "metal can",
                "rubber ball", "inflated bag", "pressurized tube", "sealed jar", "toy rocket",
                "party popper", "confetti cannon", "smoke bomb", "foam capsule", "plastic bubble",
                "toy grenade", "foam dart", "rubber capsule", "plastic pod", "foam pod",
                "rubber pod", "plastic sphere", "foam sphere", "rubber sphere", "plastic shell",
                "foam shell", "rubber shell", "plastic capsule", "foam capsule", "rubber capsule",
                "plastic bomb", "foam bomb", "rubber bomb", "plastic burst", "foam burst",
                "rubber burst", "plastic pop", "foam pop", "rubber pop", "plastic snap",
                "foam snap", "rubber snap", "plastic crack", "foam crack", "rubber crack",
                "plastic bang", "foam bang",
            ],
            "templates": [
                "{art} {object} detonates and sends fragments outward in all directions. The debris travels radially from the blast center.",
                "{art} {object} bursts apart suddenly. The resulting pieces scatter outward and then fall under gravity.",
            ],
        },
        "Fracture": {
            "object_pool": [
                "ceramic plate", "glass vase", "clay pot", "porcelain cup", "stone tablet",
                "glass panel", "ceramic tile", "clay brick", "porcelain bowl", "stone slab",
                "glass rod", "ceramic mug", "clay figure", "porcelain dish", "stone block",
                "glass sphere", "ceramic jar", "clay model", "porcelain tile", "stone ball",
                "glass cube", "ceramic bowl", "clay block", "porcelain jar", "stone disc",
                "glass tile", "ceramic disc", "clay disc", "porcelain disc", "stone tile",
                "glass block", "ceramic block", "clay cylinder", "porcelain cylinder", "stone cylinder",
                "glass cylinder", "ceramic cylinder", "clay sphere", "porcelain sphere", "stone sphere",
                "glass disc", "ceramic sphere", "clay tile", "porcelain block", "stone cube",
                "glass plate", "ceramic plate", "clay plate", "porcelain plate", "stone plate",
                "glass cup", "ceramic cup",
            ],
            "templates": [
                "{art} {object} is struck by a hard impact. It cracks and splits into fragments that scatter from the contact point.",
                "{art} {object} breaks apart under stress. The fracture lines propagate outward and separate the object into pieces.",
            ],
        },
        "Granular": {
            "object_pool": [
                "dry sand", "table salt", "white sugar", "rice grains", "dried lentils",
                "coffee grounds", "cornmeal", "wheat flour", "black pepper", "sesame seeds",
                "poppy seeds", "sunflower seeds", "pumpkin seeds", "flax seeds", "chia seeds",
                "quinoa grains", "millet grains", "barley grains", "oat flakes", "bran flakes",
                "coarse salt", "fine gravel", "coarse sand", "fine sand", "sea salt",
                "rock salt", "himalayan salt", "kosher salt", "pink salt", "black salt",
                "brown sugar", "raw sugar", "cane sugar", "powdered sugar", "coconut sugar",
                "dried beans", "dried peas", "dried chickpeas", "dried corn", "dried rice",
                "broken glass beads", "plastic pellets", "rubber pellets", "metal filings", "wood chips",
                "sawdust", "talcum powder", "baking soda", "baking powder", "cornstarch",
                "activated charcoal", "diatomite",
            ],
            "templates": [
                "{art} {object} is poured from a container. It flows downward and forms a natural pile with an angled slope.",
                "{art} {object} is released onto a surface. The grains scatter and settle into a stable heap under gravity.",
            ],
        },
        "Melting": {
            "object_pool": [
                "ice cube", "candle", "stick of butter", "chocolate bar", "block of wax",
                "ice block", "taper candle", "pat of butter", "dark chocolate", "paraffin block",
                "ice sculpture", "pillar candle", "cube of butter", "milk chocolate", "soy wax block",
                "ice sphere", "tea candle", "wedge of butter", "white chocolate", "beeswax block",
                "ice cylinder", "votive candle", "slice of butter", "chocolate chunk", "carnauba wax",
                "ice cone", "jar candle", "strip of butter", "chocolate square", "crayons",
                "ice disc", "birthday candle", "knob of butter", "chocolate coin", "lip balm",
                "ice tile", "dinner candle", "butter pat", "chocolate drop", "wax crayon",
                "ice block", "scented candle", "chilled butter", "chocolate nugget", "wax tablet",
                "snowball", "taper light", "cold butter", "chocolate pellet", "wax bead",
                "frost cube", "floating candle", "frozen butter",
            ],
            "templates": [
                "{art} {object} is placed on a warm surface. It gradually softens and loses its shape as it melts.",
                "{art} {object} is exposed to heat. The edges begin to soften first and the whole piece slowly liquefies.",
            ],
        },
        "Shadow": {
            "object_pool": [
                "tree branch", "hand", "bicycle wheel", "fence post", "lamp post",
                "potted plant", "outstretched arm", "spinning fan", "window frame", "chair leg",
                "cactus", "umbrella", "flagpole", "ladder", "antenna",
                "bottle", "cup", "vase", "toy figure", "statuette",
                "cube", "sphere", "pyramid", "cone", "cylinder",
                "cross", "star shape", "arrow shape", "letter cutout", "number cutout",
                "leaf", "flower", "stick", "rod", "bar",
                "book", "phone", "tablet", "frame", "sign",
                "key", "scissors", "comb", "fork", "spoon",
                "candle holder", "clock", "compass", "ruler", "protractor",
                "geometric shape", "paper cutout", "cardboard cutout", "wire sculpture",
            ],
            "templates": [
                "{art} {object} is lit by a directional light source. A clear shadow is cast on the ground and moves consistently with the object.",
                "{art} {object} rotates under steady light. The shadow changes shape smoothly as the angle between light and object shifts.",
            ],
        },
        "Smoke": {
            "object_pool": [
                "lit candle", "incense stick", "cigarette", "smoldering wood", "extinguished match",
                "campfire ember", "smoke machine nozzle", "dry ice block", "fog machine outlet", "burning rope",
                "smoldering paper", "cooling charcoal", "heated oil pan", "steaming kettle", "hot iron",
                "burning leaf", "glowing coal", "smoking torch", "cooling lava", "steaming grate",
                "burning twig", "smoldering cloth", "heated pipe", "steaming vent", "smoking chimney",
                "burning bark", "smoldering fiber", "heated coil", "steaming drain", "smoking funnel",
                "burning straw", "smoldering reed", "heated wire", "steaming pot", "smoking tube",
                "burning grass", "smoldering moss", "heated plate", "steaming bowl", "smoking jar",
                "burning hay", "smoldering peat", "heated stone", "steaming cup", "smoking flask",
                "burning log", "smoldering wick", "heated brick", "steaming mug", "smoking vessel",
                "burning chip", "smoldering twig",
            ],
            "templates": [
                "{art} {object} produces a thin stream of smoke that rises upward. The smoke disperses gradually as it moves higher.",
                "{art} {object} emits steady smoke that drifts in the ambient air. The plume expands and dissipates over time.",
            ],
        },
        "Splash": {
            "object_pool": [
                "stone", "rubber ball", "tennis ball", "metal key", "coin",
                "glass marble", "wooden block", "plastic toy", "steel bolt", "ceramic pebble",
                "lead sinker", "iron weight", "copper disc", "brass token", "zinc slug",
                "golf ball", "ping pong ball", "billiard ball", "squash ball", "racquetball",
                "apple", "orange", "lemon", "lime", "grape",
                "water balloon", "ice cube", "wax ball", "clay ball", "foam ball",
                "rubber duck", "toy boat", "plastic egg", "hollow ball", "cork",
                "acorn", "walnut", "pebble", "chip", "fragment",
                "bolt", "nut", "washer", "nail", "screw",
                "bead", "button", "ring", "disc", "slug",
                "small rock", "smooth stone", "rough pebble", "flat chip",
            ],
            "templates": [
                "{art} {object} is dropped into a pool of water from above. It hits the surface and creates a visible splash crown.",
                "{art} {object} enters the water at speed. Water sprays outward and upward, forming a splash pattern around the impact point.",
            ],
        },
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
    object_counts: Dict[str, Dict[str, int]] = {}  # {phenomenon: {obj: count}}
    MAX_OBJECT_USES = 2

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

                obj_key = slots.get("object") or slots.get("fluid") or base_obj
                ph_counts = object_counts.setdefault(ph, {})
                if ph_counts.get(obj_key, 0) >= MAX_OBJECT_USES:
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
                ph_counts[obj_key] = ph_counts.get(obj_key, 0) + 1
                made += 1

    # If due to constraints we are short, do a second pass with relaxed sampling
    if len(items) < num_captions:
        # Simple top-up: random pick scenes without strict per-phenomenon quota
        all_scenes = scenes[:]
        topup_tries = 0
        topup_max = (num_captions - len(items)) * 200
        while len(items) < num_captions and topup_tries < topup_max:
            topup_tries += 1
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
            obj_key = slots.get("object") or slots.get("fluid") or base_obj
            ph_counts = object_counts.setdefault(ph, {})
            if ph_counts.get(obj_key, 0) >= MAX_OBJECT_USES:
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
            ph_counts[obj_key] = ph_counts.get(obj_key, 0) + 1

    return items[:num_captions]


# -----------------------------
# Original scenes → CaptionItem
# -----------------------------

def scenes_to_original_items(scenes: List[Scene]) -> List[CaptionItem]:
    items: List[CaptionItem] = []
    for scene in scenes:
        caption_id = make_id("orig", f"{scene.phenomenon}|{scene.base_scene_id}")
        items.append(
            CaptionItem(
                caption_id=caption_id,
                phenomenon=scene.phenomenon,
                base_scene_id=scene.base_scene_id,
                caption=scene.scene_caption,
                slots={},
                questions=scene.questions,
                provenance={"source": "original", "scene_name": scene.scene_name},
            )
        )
    return items


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

    original_items = scenes_to_original_items(scenes)
    generated_items = generate_captions(
        scenes=scenes,
        specs=specs,
        num_captions=args.num_captions,
        seed=args.seed,
    )
    items = original_items + generated_items

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
