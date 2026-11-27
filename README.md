# Mikan

Distil a large model's behaviour on one narrow task into a small open model, and prove statistically that it is not worse.

Mikan takes a task handled by a large teacher model (classification, extraction, structured output), trains a small student (0.6B to 4B) to do the same job, quantizes and serves it, and judges it with a **pre-registered non-inferiority test**: the student passes only if the 95% CI for (teacher − student) stays below a margin δ fixed before training.

- **Models and data:** [huggingface.co/mikan-research](https://huggingface.co/mikan-research): 9 LoRA adapters, teacher labels and every run's predictions for both tasks.
- **Docs:** [`docs/`](docs/README.md): method, setup and results per task, roadmap, risks.
- **Contributing:** new tasks, results and code are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Licence: Apache-2.0.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Eval harness: bootstrap CI, McNemar, non-inferiority, power, reports | Done |
| 1 | Banking77 teacher labels (open model on Modal) and zero-shot baselines | Done |
| 2 | SFT distillation (LoRA) with data-size and model-size ablations | Done |
| 3 | GRPO with programmatic rewards, on MASSIVE en-US frames | Done |
| 4 | Quantization (AWQ/GPTQ) and vLLM serving benchmark | Planned |
| 5 | Generalise: new tasks from a config only | Planned |

## First task: Banking77

- 77-way intent classification of online-banking queries.
- Golden set: the official test split (about 3,080 items, ground-truth labels).
- Margin: δ = 2 accuracy points, pre-registered in [`tasks/banking77.yaml`](tasks/banking77.yaml).
- Teacher: Qwen3-30B-A3B-Instruct served with vLLM. Students: Qwen3 0.6B / 1.7B / 4B, trained with TRL + Unsloth.

## Phase 1 results (golden set, 3,080 items)

Teacher: Qwen3-30B-A3B-Instruct (FP8), 10 examples per label, **85.6%** accuracy.

| Untrained student | 0-shot | 10-shot |
|---|---|---|
| Qwen3-0.6B | 36.0% | 50.6% |
| Qwen3-1.7B | 48.6% | 65.9% |
| Qwen3-4B | 63.4% | 79.1% |

No untrained student is within the 2-point margin; closing that gap is the job of Phase 2 (SFT) and Phase 3 (GRPO).

## Phase 2 results (golden set, 3,080 items)

| Student (LoRA SFT, 3 epochs) | Labels | Train items | Golden acc | Gap vs teacher [95% CI] | Decision |
|---|---|---|---|---|---|
| Qwen3-0.6B | teacher | 8,222 | 85.6% | +0.1 [−0.8, +0.9] | PASS |
| Qwen3-1.7B | teacher | 8,222 | 85.4% | +0.3 [−0.6, +1.1] | PASS |
| Qwen3-4B | teacher | 8,222 | 86.0% | −0.4 [−1.2, +0.5] | PASS |
| Qwen3-1.7B | teacher | 5,000 | 83.9% | +1.7 [+0.7, +2.7] | FAIL |
| Qwen3-1.7B | teacher | 2,500 | 82.6% | +3.0 [+1.9, +4.0] | FAIL |
| Qwen3-1.7B | teacher | 1,000 | 71.0% | +14.6 [+13.1, +16.1] | FAIL |
| Qwen3-1.7B | ground truth | 8,222 | 93.1% | −7.4 [−8.7, −6.2] | PASS |
| Qwen3-4B | ground truth | 8,222 | 94.0% | −8.3 [−9.6, −7.1] | PASS |

- **Distillation works at every size.** With the full teacher-labelled set, all three students pass non-inferiority (teacher 85.6%), using a short prompt with no label list or examples. The 0.6B goes from 50.6% (10-shot, untrained) to 85.6%.
- **The teacher is the ceiling.** Teacher-label students land within half a point of the teacher and disagree with it on only ~6% of items; the same models trained on ground truth reach 93–94%. Better labels, not a bigger student, is the lever.
- **Data size matters more than model size.** 1k → 2.5k → 5k → 8.2k items gives 71.0 → 82.6 → 83.9 → 85.4% for the 1.7B; only the full set passes. Model size (0.6B / 1.7B / 4B) moves golden accuracy by under one point.
- Checkpoints were chosen on val only; golden was predicted once per run. Training time on one T4: 0.6B 15 min, 1.7B 23 min, 4B 45 min.

Per-run reports: `reports/phase2/` (not tracked).

## Second task: MASSIVE en-US frames (Phase 3)

Banking77 has no reward a program can check without ground truth, so GRPO uses a structured-output task:
parse a voice-assistant request into a JSON frame.

```json
{"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}
```

- MASSIVE v1.1 en-US (Amazon, CC-BY-4.0): 60 intents, 55 slot types; train 11,255, val 2,026 (official dev), golden
  2,974 (official test, ground truth).
- Pre-registered in [`tasks/massive.yaml`](tasks/massive.yaml): exact frame match, **δ = 3 points** (chosen on test-set
  power before any output existed).
- `mikan/frames.py` checks every output the same way for evaluation and reward: exact JSON shape, intent and slot types
  in the schema, slot values copied from the input, exact match on the (type, value) multiset.
- GRPO reward against the **teacher's** frame (never golden): 0 if invalid, else 0.1 + 0.3·intent + 0.6·slot F1·groundedness.

**Teacher:** Qwen3-30B-A3B-Instruct (4-bit GPTQ) on Kaggle T4 x2 with vLLM, decoding constrained to the frame schema.
Prompt pilot on the same 300 val items:

| Prompt | Exact frame match |
|---|---|
| 1 example per intent (fixed) | 44.7% |
| 3 per intent (fixed) | 48.0% |
| **8 retrieved nearest examples** | **67.0%** |
| 20 retrieved nearest examples | 68.0% |

Retrieved examples (`mikan.data.retrieve`, TF-IDF neighbours; never the item itself, never val or golden) fix most
fixed-prompt errors, which were MASSIVE annotation conventions. Full run with 8 examples: **64.4%** exact match on golden
(train 63.7%, val 64.9%); every train and val frame is valid, one golden output was cut off at the token limit.

## Phase 3 results (golden set, 2,974 items, teacher 64.4%, δ = 3)

| Student | Training | Golden exact match | Gap vs teacher [95% CI] | Valid | Decision |
|---|---|---|---|---|---|
| Qwen3-0.6B | SFT, 11,255 teacher frames | 64.1% | +0.3 [−1.3, +1.9] | 99.7% | PASS |
| Qwen3-0.6B | + GRPO (best at step 200) | 64.8% | −0.4 [−2.1, +1.2] | 99.6% | PASS |
| Qwen3-1.7B | SFT, 11,255 teacher frames | 65.0% | −0.6 [−2.2, +1.0] | 99.8% | PASS |
| Qwen3-1.7B | + GRPO (best at step 200) | **65.7%** | −1.3 [−2.9, +0.3] | 99.7% | PASS |
| Qwen3-1.7B | SFT, 1,000 teacher frames | 46.0% | +18.4 [+16.5, +20.3] | 95.3% | FAIL |
| Qwen3-1.7B | + GRPO on 3,000 unseen prompts | 50.4% | +14.0 [+12.0, +15.9] | 97.2% | FAIL |
| Qwen3-1.7B | SFT, 11,255 ground-truth frames | 72.1% | −7.8 [−9.4, −6.1] | 99.8% | PASS |

| Pair (same golden items) | GRPO − SFT [95% CI] | Right only after / only before | McNemar p |
|---|---|---|---|
| 0.6B, full SFT | +0.7 [+0.0, +1.5] | 71 / 49 | 0.055 |
| 1.7B, full SFT | +0.7 [−0.0, +1.5] | 72 / 51 | 0.071 |
| 1.7B, 1k SFT | **+4.4 [+3.1, +5.8]** | 282 / 150 | 2e-10 |

- **Distillation passes on structured output too.** SFT on teacher frames alone passes at both sizes; the 1.7B
  slightly beats the teacher (65.0 vs 64.4%) with a short prompt that has no intent or slot lists.
- **GRPO helps most when labelled data is scarce.** After a full SFT it adds +0.7 points (borderline, p ≈ 0.06);
  after a 1k SFT it adds +4.4 points (p = 2e-10) using only the teacher frames of 3,000 new prompts as reward
  references, still far from passing.
- **What GRPO changes:** fewer slots (1k run: 3,090 → 2,704 emitted on golden), higher slot precision, and fewer
  invented labels (171 → 97 unknown intents or slot types such as `number`, `quantity`, `music_remove`). Validity and
  groundedness stay flat or improve; no extra text, no invented spans. Reward peaked mid-run (1k run: 0.59 → 0.83 at
  step 130 → 0.64 at 250), and val selection kept steps 200–250.
- **Labels are still the ceiling:** ground-truth SFT reaches 72.1% vs 65.0% on teacher labels, a bigger gap than
  anything GRPO closes.
- Remaining invalid outputs are invented labels; serving with the frame JSON schema (constrained decoding) removes
  them by construction (Phase 4).


## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

## Data and teacher labels

```bash
uv run python -m mikan.data.banking77 --out data/banking77 --fewshot-per-label 10
uv run modal run -m mikan.intake.modal_predict --splits smoke --shots 10              # 50 items
uv run modal run -m mikan.intake.modal_predict --splits train,val,golden --shots 10   # full run
uv run modal run -m mikan.intake.modal_predict --role student --model Qwen/Qwen3-1.7B --splits val,golden --shots 0
```

Model weights download once into a Modal Volume on a CPU container, so GPUs never bill for downloads.

MASSIVE (teacher on Kaggle T4 x2; `--runner teacher` uses `mikan/intake/kaggle_teacher.py`):

```bash
uv run python -m mikan.data.massive --out data/massive
uv run python -m mikan.data.retrieve --data data/massive
uv run python -m mikan.train.kaggle_job push --configs configs/teacher_massive_knn_pilot.json \
    --job mikan-teacher-massive-knn --runner teacher --data massive
```

## SFT on Kaggle

```bash
uv run python -m mikan.train.kaggle_job push --configs configs/sft_phase2.json --job mikan-sft-phase2
uv run python -m mikan.train.kaggle_job status --job mikan-sft-phase2
uv run python -m mikan.train.kaggle_job pull --job mikan-sft-phase2
uv run python -m mikan.eval.sweep --runs runs/kaggle/mikan-sft-phase2/runs \
    --teacher runs/teacher-qwen3-30b-a3b-instruct-2507-fp8-10shot --out reports/phase2
```

`push` bundles the code, prepared splits, cached teacher labels and configs into a private Kaggle dataset and starts a private T4 x2 kernel; each GPU runs its own queue of configs. A config with `"script": "grpo"` runs `mikan.train.grpo`, and `"adapter": "@runs/<sft run>/best_adapter"` starts it from an SFT run earlier in the same GPU queue. `mikan.train.sft` and `mikan.train.grpo` also run directly on any single GPU.

MASSIVE grid (two kernels in parallel):

```bash
for k in a b; do uv run python -m mikan.train.kaggle_job push --configs configs/massive_phase3_$k.json \
    --job mikan-massive-phase3-$k --data massive,massive-teacher --dataset mikan-bundle-phase3$k; done
```

## Models and data on Hugging Face

Everything trained or labelled here is in the [mikan-research](https://huggingface.co/mikan-research) organization:

| Task | Models (LoRA adapters) | Dataset |
|---|---|---|
| Banking77 | `banking77-qwen3-{0.6b,1.7b,4b}-lora`, `banking77-qwen3-{1.7b,4b}-gold-lora` | `banking77-distill` |
| MASSIVE en-US | `massive-qwen3-{1.7b,0.6b}-grpo-lora`, `massive-qwen3-1.7b-sft-lora`, `massive-qwen3-1.7b-gold-lora` | `massive-distill` |

Each model card has the golden result with its confidence interval, the training recipe and a usage snippet. The
datasets hold the splits, teacher labels and every run's predictions, so a new student can be trained or compared
without re-running a teacher. `mikan/publish_hf.py` stages and uploads them; updating a public repo needs
`--allow-public`.

```bash
uv run --with huggingface_hub python -m mikan.publish_hf stage [--task massive]
uv run --with huggingface_hub python -m mikan.publish_hf upload [--task massive] [--allow-public]
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: pre-register a task's margin before any outputs exist,
never use teacher outputs as golden labels, choose everything on the val split, and report failures as well as passes.

## Eval harness

**Compare a teacher and a student on the golden set:**

```bash
uv run mikan-eval report --task tasks/banking77.yaml \
  --gold data/banking77/golden.jsonl \
  --a runs/teacher/test_preds.jsonl \
  --b runs/student/test_preds.jsonl \
  --out reports/my-run
```

This writes `report.json` and `report.md` with:
- the pass/fail decision
- CIs for each metric
- McNemar's test
- test-set power
- per-label slices
- error examples
- limitations

The report also records the task config's SHA-256, so any change to the margin after the fact is visible.

Input files are JSON Lines:

```text
gold:        {"id": "...", "label": "<label>", "text": "<optional input>"}
predictions: {"id": "...", "pred": "<raw model output>"}
```

Ids must match exactly between files. An output that is not exactly a valid label (apart from surrounding whitespace) counts as wrong.

**Check whether a test set is large enough:**

```bash
uv run mikan-eval power --n 3080 --disagreement 0.08 --delta 0.02
```

The disagreement rate is the share of items where exactly one of the two systems is right. Power depends on it, not on overall accuracy.

Frames tasks (`kind: frames`) take gold records with a `"frame"` and score exact frame match, adding intent accuracy,
slot F1, validity and groundedness to the report. `python -m mikan.eval.sweep --task ... --teacher ... --runs ...`
evaluates a directory of runs in one table.

**Run the synthetic self-check** (students with known gaps around δ):

```bash
uv run mikan-eval selfcheck
```

| True gap (teacher − student) | Pass rate | Expected |
|---|---|---|
| −2 points | 1.00 | 1.00 |
| 0 | 0.98 | 0.975 |
| +1 point | 0.43 | 0.50 |
| +2 points (= δ) | 0.03 | 0.025 |
| +3 / +4 points | 0.00 | 0.00 |

## Layout

```text
mikan/
  eval/        metrics, paired statistics, power, synthetic data, reports, CLI
  data/        dataset preparation (Banking77, MASSIVE; leakage checks) and retrieval of nearest examples
  intake/      batch prediction with vLLM, locally, on Modal or on Kaggle (teacher runner)
  train/       LoRA SFT and GRPO (Unsloth + TRL), and the Kaggle job driver
  frames.py    frame checks, metrics and GRPO reward
  prompts.py   teacher prompt construction (labels, frames, retrieved examples)
  tasks.py     task config loader
  publish_hf.py  Hugging Face publishing (mikan-research)
tasks/         pre-registered task configs
configs/       Kaggle run grids (SFT, GRPO, teacher)
tests/         unit tests and the synthetic exit check
docs/          method, per-task setup and results, roadmap, risks
```

Planned modules: `serve/` (quantization, vLLM), `monitor/` (drift and re-evaluation).

## Conventions

- System `a` is the teacher and `b` the student. Every difference is `a − b`, so positive means the student is worse.
- Golden labels are verified ground truth, never teacher outputs.
- The quantized model is the one evaluated, because it is the one served.
- Datasets, teacher outputs, runs and reports stay out of git; they are published on Hugging Face.
