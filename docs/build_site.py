#!/usr/bin/env python3
"""
Regenerate docs/index.html with the taxonomy embedded from the benchmark source.

The taxonomy on the project page is not hand-maintained: it is derived from
data/question_v3.json in the main PhyMetric repo so the counts on the site can
never drift from the released dataset. Re-run after any dataset change.

    python docs/build_site.py --source ../PhyMetric/data/question_v3.json
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The dataset tags seven fine-grained domains; Table 6 of the paper reports six,
# merging optics into electromagnetism. The site follows the paper.
PAPER_DOMAIN = {
    "mechanics": "Mechanics", "fluid": "Fluid Dynamics", "material": "Material",
    "thermal": "Thermal", "chemistry": "Chemistry",
    "optics": "Electromagnetism", "electromagnetic": "Electromagnetism",
}
DOMAIN_ORDER = ["Mechanics", "Fluid Dynamics", "Material", "Thermal",
                "Chemistry", "Electromagnetism"]
DOMAIN_KEY = {d: d.split()[0].lower() for d in DOMAIN_ORDER}

DIM12 = ["motion", "interaction", "appearance/texture", "causal", "existence",
         "spatial", "material", "shape", "object", "direction", "size", "number"]


def build_taxonomy(src: Path) -> dict:
    data = json.loads(src.read_text())
    cats, dims, diff = [], collections.Counter(), collections.Counter()
    for name, node in data["categories"].items():
        md = node.get("category_metadata", {})
        scenes = node["scenes"]
        nq = 0
        for s in scenes.values():
            for q in s.get("questions", []):
                nq += 1
                diff[q.get("difficulty", "?")] += 1
                for d in q.get("dimensions", []):
                    dims[d] += 1
        cats.append({
            "n": name,
            "d": PAPER_DOMAIN[md.get("physics_domain", "mechanics")],
            "desc": md.get("description", ""),
            "law": md.get("physical_principle", ""),
            "obj": md.get("object_pool", [])[:12],
            "env": md.get("environment_pool", [])[:8],
            "kd": md.get("key_dimensions", []),
            "s": len(scenes),
            "q": nq,
        })
    cats.sort(key=lambda c: (DOMAIN_ORDER.index(c["d"]), -c["s"], c["n"]))
    return {
        "categories": cats,
        "dimensions": [{"n": d, "c": dims[d]} for d in DIM12],
        "difficulty": {k: diff[k] for k in ("easy", "medium", "hard")},
        "totals": {
            "scenes": sum(c["s"] for c in cats),
            "questions": sum(c["q"] for c in cats),
            "categories": len(cats),
            "domains": len(DOMAIN_ORDER),
        },
    }


# Abstract, results and case studies are transcribed from NeurIPS2026/ so the page
# and the paper cannot disagree. PAPER_DIR points at that source tree.
# Two paragraphs, broken at the abstract's own pivot: the gap, then the contribution.
# The text is verbatim from the paper; only the paragraph break is ours.
ABSTRACT = [
    "Text-to-video (T2V) generation models can produce visually realistic videos that "
    "nevertheless violate the physical consequences implied by the prompt and scene. "
    "Existing benchmarks largely rely on holistic plausibility scores or broad quality "
    "dimensions, leaving it unclear which physical behavior is being tested and why a "
    "model fails.",

    "We introduce PhyMetric, a scene-level diagnostic benchmark that recasts "
    "physical plausibility evaluation as verifiable, scene-grounded question answering. "
    "PhyMetric contains 1,115 scenes spanning 50 physical categories across six domains, "
    "paired with 8,920 answer-balanced binary questions over 12 evaluation dimensions. "
    "Each scene is associated with eight targeted yes/no questions and a structured "
    "physical prior specifying the governing law, expected temporal sequence, observable "
    "signatures, and common failure modes. At evaluation time, a vision-language model "
    "(VLM) judge answers each question from question-aware sampled frames conditioned on "
    "the prior, and correctness is hierarchically aggregated into scene, dimension, "
    "physical category, and model scores.",
]

# Table 2 (full benchmark) and Table 3 (100-scene controlled subset).
RESULTS = {
    "full": {
        "caption": "Full benchmark \u00b7 1,115 scenes \u00b7 open-source VLM judges",
        "judges": ["Qwen3.5-9B", "Qwen2.5-VL-7B", "Average"],
        "rows": [
            ["Wan2.2-14B",   [76.74, 78.07], [68.82, 70.82], [72.78, 74.45]],
            ["Cosmos-7B",    [76.28, 77.60], [68.89, 70.92], [72.59, 74.26]],
            ["Wan2.1-1.3B",  [73.86, 75.50], [67.26, 69.42], [70.56, 72.46]],
            ["CogVideoX-5B", [71.36, 73.18], [65.12, 67.62], [68.24, 70.40]],
        ],
    },
    "subset": {
        "caption": "100-scene controlled subset \u00b7 closed and open-source judges",
        "judges": ["Qwen3.5-9B", "Qwen2.5-VL-7B", "GPT-5.4", "Average"],
        "rows": [
            ["Wan2.2-14B",   [77.88, 79.23], [66.75, 69.54], [73.12, 74.26], [72.58, 74.34]],
            ["Cosmos-7B",    [76.50, 77.54], [67.37, 70.37], [74.00, 75.54], [72.62, 74.48]],
            ["Wan2.1-1.3B",  [74.62, 76.85], [65.87, 68.94], [72.12, 73.88], [70.87, 73.22]],
            ["CogVideoX-5B", [70.63, 72.45], [64.38, 67.13], [69.75, 71.67], [68.25, 70.42]],
        ],
    },
}

# Appendix A.4.4 case studies, with the real sampled frames.
CASES = [
    {
        "id": "gravity", "cat": "Gravity_Earth", "dom": "Mechanics",
        "title": "Apple released from a tree branch",
        "gen": "CogVideoX-5B", "anchors": [20, 60, 100],
        "caption": "A ripe apple is released from a tree branch above a grassy lawn and "
                   "falls under Earth\u2019s gravity until it lands on the ground.",
        "law": "Constant downward acceleration g \u2248 9.8\u2009m/s\u00b2; vertical "
               "velocity grows linearly, distance as \u00bdgt\u00b2.",
        "signature": "Downward speed increases monotonically until contact; the trajectory "
                     "is straight-vertical, not floating, hovering or rising.",
        "q": "Does the apple\u2019s downward speed visibly increase between the early and "
             "middle frames?",
        "qmeta": "id 02 \u00b7 motion \u00b7 easy", "truth": 1,
        "ans": 0, "conf": 0.83,
        "reason": "The apple\u2019s vertical position is essentially unchanged across the "
                  "three frames; it remains hovering just below the branch instead of "
                  "accelerating downward toward the ground.",
        "verdict": "miss",
        "note": "A typical gravity failure: the generator produces a near-static apple "
                "instead of free fall. The prior\u2019s monotonic-speed signature is what "
                "lets the judge flag the missing acceleration.",
    },
    {
        "id": "buoyancy", "cat": "Buoyancy_Float", "dom": "Fluid Dynamics",
        "title": "Rubber duck placed in water",
        "gen": "Wan2.1-1.3B", "anchors": [25, 65, 105],
        "caption": "A rubber duck is placed into water, displaces the water, and floats at "
                   "the surface in a stable position.",
        "law": "Archimedes\u2019 principle; the duck\u2019s average density is below that "
               "of water, so the buoyant force balances gravity.",
        "signature": "The duck never fully submerges; the waterline cuts through its lower "
                     "body for the entire clip.",
        "q": "Does the rubber duck remain partly above the water surface across all the "
             "input frames?",
        "qmeta": "id 04 \u00b7 spatial \u00b7 easy", "truth": 1,
        "ans": 1, "conf": 0.94,
        "reason": "In every frame the duck\u2019s body sits clearly above the waterline "
                  "with only its lower half submerged, consistent with stable floating.",
        "verdict": "hit",
        "note": "The mirror-image reflection below the waterline confirms the duck\u2013"
                "surface contact relation the prior\u2019s signature requires.",
    },
    {
        "id": "melting", "cat": "Melting", "dom": "Thermal",
        "title": "Ice cube on a warm metal plate",
        "gen": "Wan2.2-A14B", "anchors": [15, 55, 100],
        "caption": "An ice cube is placed on a warm metal plate at room temperature; over "
                   "four seconds it gradually melts into a small puddle of clear water "
                   "around its base.",
        "law": "Phase transition at the melting point absorbs latent heat; solid volume "
               "shrinks while a liquid pool grows around the contact region.",
        "signature": "Ice volume decreases monotonically, water volume increases; the cube "
                     "does not regrow or stay constant.",
        "q": "Does the ice cube\u2019s volume visibly shrink between the early and late "
             "frames?",
        "qmeta": "id 02 \u00b7 shape \u00b7 hard", "truth": 1,
        "ans": 1, "conf": 0.84,
        "reason": "The cube spans a noticeably smaller footprint on the plate in the late "
                  "frame and is partly surrounded by a thin liquid film not present early on.",
        "verdict": "hit",
        "note": "The prior\u2019s signature is what lets the judge turn a small visual "
                "change into a confident YES instead of treating it as ambiguous.",
    },
]

FRAME_FILES = {
    "gravity": ["case1_gravity_f020", "case1_gravity_f060", "case1_gravity_f100"],
    "buoyancy": ["case2_buoyancy_f025", "case2_buoyancy_f065", "case2_buoyancy_f105"],
    "melting": ["case3_melting_f015", "case3_melting_f055", "case3_melting_f100"],
}


def embed_frames(paper_dir: Path) -> dict:
    """Inline the case frames as data URIs so the page is self-contained."""
    out, missing = {}, []
    for case, names in FRAME_FILES.items():
        uris = []
        for n in names:
            p = paper_dir / "figures_neurips/eval_cases" / f"{n}.jpg"
            if not p.exists():
                missing.append(str(p)); continue
            uris.append("data:image/jpeg;base64," +
                        base64.b64encode(p.read_bytes()).decode())
        out[case] = uris
    if missing:
        raise SystemExit("missing case frames:\n  " + "\n  ".join(missing))
    return out


# The worked example on the page is the scene bundled with PhyMetric-Bench, so a
# visitor can run exactly what the page shows.
EXAMPLE = {
    "scene": "Gravity_Earth / scene_21",
    "caption": "A pen slips from the top rung of a ladder and accelerates "
               "downward under gravity until it strikes the ground.",
    "law": "Newton's second law with constant downward acceleration "
           "g ≈ 9.8 m/s²; velocity grows linearly with time.",
    "questions": [
        ("01", "easy", "motion · causal", 1,
         "Does the pen fall straight downward after slipping off the ladder?", "20, 60, 100"),
        ("02", "medium", "motion · interaction", 0,
         "Does the pen bounce upwards after hitting the ground?", "100, 110"),
        ("05", "hard", "motion · causal", 1,
         "Does the speed of the pen increase as it falls?", "20, 40, 60"),
        ("08", "hard", "existence · causal", 0,
         "Does the pen vanish before reaching the ground?", "100, 115"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    # docs/ -> PhyMetric-Bench/ -> code/ , where the main repo sits alongside
    ap.add_argument("--source", type=Path,
                    default=HERE.parents[1] / "PhyMetric/data/question_v3.json")
    ap.add_argument("--paper", type=Path, default=HERE.parents[1] / "PhyMetric/NeurIPS2026",
                    help="Paper source tree, for the case-study frames")
    ap.add_argument("--out", type=Path, default=HERE / "index.html")
    args = ap.parse_args()

    tax = build_taxonomy(args.source)
    t = tax["totals"]
    by_dom = collections.Counter()
    sc_dom = collections.Counter()
    for c in tax["categories"]:
        by_dom[c["d"]] += 1
        sc_dom[c["d"]] += c["s"]
    domains = [{"n": d, "k": DOMAIN_KEY[d], "c": by_dom[d], "s": sc_dom[d]}
               for d in DOMAIN_ORDER]

    html = (HERE / "template.html").read_text()
    # The template keeps a `null` after each token so it is valid JS on its own;
    # the substitution must consume that null, or the page ships `{...}null;`.
    frames = embed_frames(args.paper)
    cases = [dict(c, frames=frames[c["id"]]) for c in CASES]
    for token, payload in [("TAXONOMY", tax), ("DOMAINS", domains), ("EXAMPLE", EXAMPLE),
                           ("ABSTRACT", ABSTRACT), ("RESULTS", RESULTS), ("CASES", cases)]:
        needle = f"/*__{token}__*/null"
        if needle not in html:
            raise SystemExit(f"template is missing the {needle} placeholder")
        html = html.replace(needle, json.dumps(payload, separators=(",", ":")))
    for key, val in [("SCENES", f"{t['scenes']:,}"), ("QUESTIONS", f"{t['questions']:,}"),
                     ("CATEGORIES", str(t["categories"])), ("DOMAINS", str(t["domains"]))]:
        html = html.replace(f"__{key}__", val)
    args.out.write_text(html)

    print(f"wrote {args.out}  ({len(html)/1024:.0f} KB)")
    print(f"  {t['categories']} categories · {t['scenes']:,} scenes · "
          f"{t['questions']:,} questions · {t['domains']} domains")
    for d in domains:
        print(f"    {d['n']:<18}{d['c']:>3} categories{d['s']:>6} scenes")


if __name__ == "__main__":
    main()
