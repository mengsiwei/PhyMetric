#!/usr/bin/env python3
"""
Step 1+2: Build per-scene knowledge base and inject into question_v2.json.

For each scene, queries a text LLM (QwenVL text-only mode or any OpenAI-compatible
API) to generate a structured physics knowledge base covering:
  - physical_law      : the governing law and conditions
  - expected_sequence : what should be visible at frame ~10 / ~30 / ~50
  - visual_indicators : specific cues confirming correct physics
  - common_failures   : what incorrect video generation typically gets wrong
  - evaluation_hints  : how to judge each dimension present in the scene's questions

Output: data/question_v2_with_kb.json  (question_v2.json + "knowledge_base" per scene)

Usage:
    # QwenVL local model
    python build_knowledge_base.py \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --input  data/question_v2.json \\
        --output data/question_v2_with_kb.json \\
        --gpu 0

    # dry_run: fill with placeholder KB (no GPU needed)
    python build_knowledge_base.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


# =============================================================================
# KB prompt builder
# =============================================================================

KB_SYSTEM_PROMPT = (
    "You are an expert physicist and video generation evaluator. "
    "Your task is to produce a concise but comprehensive knowledge base "
    "that will later help a vision-language model judge whether a short generated video "
    "correctly depicts a physical scene. "
    "Respond ONLY with a valid JSON object — no markdown, no extra text."
)

KB_USER_TEMPLATE = """\
Physical phenomenon category: {phenomenon}
Scene name: {scene_name}
Scene caption (the prompt given to the video model):
  \"{scene_caption}\"

The video is 2 seconds long, 30 fps, 60 frames total.
Key reference frames: frame 10 (~0.33 s, early), frame 30 (~1.0 s, mid), frame 50 (~1.67 s, late).

Question dimensions present in this scene: {dimensions}

Generate a knowledge base JSON with exactly these keys:

{{
  "physical_law": "<1-3 sentences: what physical law/phenomenon governs this scene and what conditions apply>",
  "expected_sequence": {{
    "frame_10": "<what should be visible / happening at ~0.33 s>",
    "frame_30": "<what should be visible / happening at ~1.0 s>",
    "frame_50": "<what should be visible / happening at ~1.67 s>"
  }},
  "visual_indicators": [
    "<specific visual cue 1 that confirms correct physics>",
    "<specific visual cue 2>",
    "<specific visual cue 3>"
  ],
  "common_failures": [
    "<common video generation mistake 1 for this phenomenon>",
    "<common mistake 2>",
    "<common mistake 3>"
  ],
  "evaluation_hints": {{
    {dimension_hints_template}
  }}
}}

Be concrete and reference the actual objects in the scene caption.
"""

DIMENSION_HINT_TEMPLATE = '    "{dim}": "<how to visually judge {dim}-related questions for this scene>"'


def build_kb_prompt(phenomenon: str, scene_name: str, scene_caption: str,
                    dimensions: list[str]) -> str:
    dims_str = ", ".join(sorted(set(dimensions))) if dimensions else "motion, spatial"
    dim_hints = ",\n    ".join(
        DIMENSION_HINT_TEMPLATE.format(dim=d)
        for d in sorted(set(dimensions))
    ) or '    "motion": "<describe expected motion>"'
    return KB_USER_TEMPLATE.format(
        phenomenon=phenomenon,
        scene_name=scene_name,
        scene_caption=scene_caption,
        dimensions=dims_str,
        dimension_hints_template=dim_hints,
    )


# =============================================================================
# LLM backend
# =============================================================================

class TextLLM:
    """Thin wrapper around QwenVL (text-only) for KB generation."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor

        print(f"Loading model: {model_id}")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device
            ).eval()
        except Exception:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device,
                trust_remote_code=True,
            ).eval()
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._device = device

    def query(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024) -> str:
        import torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()


# =============================================================================
# KB parser
# =============================================================================

def parse_kb_json(raw: str) -> dict:
    """Extract the JSON object from raw LLM output, with fallback."""
    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # try to extract first {...} block
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"raw": raw, "parse_error": True}


def make_placeholder_kb(phenomenon: str, scene_name: str, scene_caption: str,
                         dimensions: list[str]) -> dict:
    return {
        "physical_law": f"[placeholder] {phenomenon} governs this scene: {scene_caption[:80]}",
        "expected_sequence": {
            "frame_10": "Early stage of the physical event begins.",
            "frame_30": "Main physical interaction is ongoing.",
            "frame_50": "Physical event reaches stable or final state.",
        },
        "visual_indicators": [
            "Objects behave according to the expected physical law.",
            "No physically implausible behavior such as hovering or teleportation.",
            "Temporal progression is consistent with the described caption.",
        ],
        "common_failures": [
            "Object does not move or is static throughout.",
            "Object passes through surfaces (clipping).",
            "Physics direction is reversed (e.g., object floats upward when it should sink).",
        ],
        "evaluation_hints": {
            dim: f"Visually check {dim}-related behavior against the scene caption."
            for dim in dimensions
        },
    }


# =============================================================================
# Main
# =============================================================================

def collect_scene_dimensions(questions: list[dict]) -> list[str]:
    dims = set()
    for q in questions:
        dims.update(q.get("dimensions", []))
    return sorted(dims)


def main():
    parser = argparse.ArgumentParser(description="Build PhyMetric knowledge base")
    parser.add_argument("--input",  type=str, default="data/question_v2.json")
    parser.add_argument("--output", type=str, default="data/question_v2_with_kb.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Sleep seconds between queries (rate limiting)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Use placeholder KB, no LLM call (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="If output exists, skip already-processed scenes")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # load existing output for resume
    existing: dict = {}
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for cat, scenes in prev.get("main_categories", {}).items():
            for sid, scene in scenes.items():
                if "knowledge_base" in scene:
                    existing[(cat, sid)] = scene["knowledge_base"]
        print(f"Resume: {len(existing)} scenes already have KB.")

    # load LLM
    llm = None
    if not args.dry_run:
        llm = TextLLM(args.model_id, device="cuda")

    total_scenes = sum(len(v) for v in data["main_categories"].values())
    processed = 0

    for phenomenon, scenes in data["main_categories"].items():
        for scene_id, scene in scenes.items():
            processed += 1
            key = (phenomenon, scene_id)

            if key in existing:
                scene["knowledge_base"] = existing[key]
                continue

            scene_name    = scene.get("scene_name", scene_id)
            scene_caption = scene.get("scene_caption", "")
            questions     = scene.get("questions", [])
            dimensions    = collect_scene_dimensions(questions)

            print(f"[{processed}/{total_scenes}] {phenomenon} / {scene_id} — {scene_name}")

            if args.dry_run:
                kb = make_placeholder_kb(phenomenon, scene_name, scene_caption, dimensions)
            else:
                prompt = build_kb_prompt(phenomenon, scene_name, scene_caption, dimensions)
                raw = llm.query(KB_SYSTEM_PROMPT, prompt, max_new_tokens=args.max_new_tokens)
                kb  = parse_kb_json(raw)
                if "parse_error" in kb:
                    print(f"  WARNING: JSON parse failed, storing raw text.")
                if args.sleep > 0:
                    time.sleep(args.sleep)

            scene["knowledge_base"] = kb

    # write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output written to: {args.output}")
    print(f"Scenes processed: {total_scenes}")


if __name__ == "__main__":
    main()
