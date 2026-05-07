# PhyMetric-Bench

A self-contained subset of the PhyMetric evaluation pipeline, that produces the per-question accuracy numbers reported in the paper, using one bundled scene and one bundled video per T2V model.

The same scripts (`evaluate.py`, `evaluate_gpt.py`) are what we used in the
full experiments — only the inputs are scaled down here so the smoke test runs
in minutes on a single GPU (or a few seconds on the OpenAI API).

---

## Bundled artefacts

```
PhyMetric-Bench/
├── README.md                              
├── requirements.txt                       # evaluation environment
├── evaluate.py                            # local-VLM judge (Qwen3.5 / Qwen2.5-VL)
├── evaluate_gpt.py                        # OpenAI vision API judge
├── data/
│   ├── sample_question_with_kb.json       # 1 scene, 8 questions, with KB block
│   └── sample_question_no_kb.json         # same scene, KB removed (for ablation)
├── sample_videos/
│   ├── Wan2.2-14B/Gravity_Earth_scene_21.mp4
│   ├── Cosmos-7B/Gravity_Earth_scene_21.mp4
│   ├── Wan2.1-1.3B/Gravity_Earth_scene_21.mp4
│   └── CogVideoX-5B/Gravity_Earth_scene_21.mp4
├── reference_outputs/                     # qa_results.json from our paper run
│   ├── Wan2.2-14B/{qwen35_9b,qwen25vl_7b,gpt_5_4}_qa_results.json
│   ├── Cosmos-7B/...
│   ├── Wan2.1-1.3B/...
│   └── CogVideoX-5B/...
└── scripts/
    ├── run_qwen35.sh                      # Qwen3.5-9B  (text judge over sampled frames)
    ├── run_qwen25vl.sh                    # Qwen2.5-VL-7B-Instruct  (vision judge)
    └── run_gpt.sh                         # GPT-5.4  (OpenAI vision API)
```

The sample scene is **`Gravity_Earth_scene_21`** — *"A pen slips from the top
rung of a ladder and accelerates downward under gravity until it strikes the
ground."* — with 8 yes/no questions covering `motion`, `direction`, `causal`,
`existence`, and `interaction` dimensions across all three difficulty levels.

---

## 1 · Environment

```bash
# Python 3.10+ (tested on 3.10 and 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware requirements:

| Judge | GPU | VRAM | Notes |
|---|---|---|---|
| Qwen3.5-9B (`run_qwen35.sh`) | 1 × ≥24 GB | ~22 GB in bf16 | text-only, frames are described as image features |
| Qwen2.5-VL-7B-Instruct (`run_qwen25vl.sh`) | 1 × ≥24 GB | ~17 GB in bf16 | full vision-language model |
| GPT-5.4 (`run_gpt.sh`) | none | — | needs `OPENAI_API_KEY`; ~$0.02 per scene at `detail=low` |

---

## 2 · The 3-step pipeline

The scripts wrap a 3-step pipeline implemented in `evaluate.py`:

| Step | What it does | Where it runs | Output |
|---|---|---|---|
| **a** | sample frames at each question's `frame_indices` from the .mp4 | CPU only | `frames/<scene>/frame_NNNN.jpg` |
| **b** | for each question: send (frames + KB block + question) to the judge, parse `ANSWER / CONFIDENCE / REASON` | GPU (local) or OpenAI API (GPT) | `qa_results.json` |
| **c** | score predictions against ground-truth answers with difficulty / dimension weighting | CPU only | `eval_summary.json` |

You can run any subset via `--steps` (e.g. `--steps a` extracts frames only;
`--steps c` re-scores an existing `qa_results.json`).

---

## 3 · Verifying the run

After a run finishes, compare your `qa_results.json` to ours under
`reference_outputs/`. The yes/no `pred` column should match for all 8
questions on the sample scene (modulo non-determinism in temperature-0
decoding for vision models, which can occasionally flip a borderline call).

---

## 4 · Question file format

Each scene is a record under `categories.<phenomenon>.scenes.<scene_key>`:

```jsonc
{
  "metadata": { "fps": 30, "total_frames": 120, ... },
  "categories": {
    "Gravity_Earth": {
      "category_metadata": {
        "physical_principle": "Newton's second law ...",
        "physics_domain":     "mechanics",
        ...
      },
      "scenes": {
        "scene_21": {
          "scene_caption":     "A pen slips from the top rung ...",
          "generation_prompt": "...",
          "knowledge_base":    "<text injected into the judge prompt>",
          "questions": [
            {
              "id":            "01",
              "question_text": "Does the pen fall straight downward ...?",
              "dimensions":    ["motion", "causal"],
              "difficulty":    "easy",
              "frame_indices": [20, 60, 100],
              "answer":        1                 // 1 = yes, 0 = no  (ground truth)
            },
            ...
          ]
        }
      }
    }
  }
}
```

`frame_indices` are expressed in the question file's reference time-space
(`fps=30`, `total_frames=120`, i.e. 4-second clips). Step a maps them onto
the actual video's fps/length before sampling.

---