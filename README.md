# Mikan

Distil a large model's behaviour on one narrow task into a small open model, and prove statistically that it is not worse.

Mikan takes a task handled by a large teacher model (classification, extraction, structured output), trains a small student (0.6B to 4B) to do the same job, quantizes and serves it, and judges it with a **pre-registered non-inferiority test**: the student passes only if the 95% CI for (teacher − student) stays below a margin δ fixed before training.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Eval harness: bootstrap CI, McNemar, non-inferiority, power, reports | Done |
| 1 | Banking77 teacher labels (open model on Modal) and zero-shot baselines | Done |
| 2 | SFT distillation (LoRA) with data-size and model-size ablations | Done |
| 3 | GRPO with programmatic rewards | Next |
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

## SFT on Kaggle

```bash
uv run python -m mikan.train.kaggle_job push --configs configs/sft_phase2.json --job mikan-sft-phase2
uv run python -m mikan.train.kaggle_job status --job mikan-sft-phase2
uv run python -m mikan.train.kaggle_job pull --job mikan-sft-phase2
uv run python -m mikan.eval.sweep --runs runs/kaggle/mikan-sft-phase2/runs \
    --teacher runs/teacher-qwen3-30b-a3b-instruct-2507-fp8-10shot --out reports/phase2
```

`push` bundles the code, prepared splits, cached teacher labels and configs into a private Kaggle dataset and starts a private T4 x2 kernel; each GPU runs its own queue of configs. `mikan.train.sft` can also be run directly on any single GPU.

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
  data/        dataset preparation (Banking77 splits, leakage checks)
  intake/      teacher/student batch prediction with vLLM, locally or on Modal
  train/       LoRA SFT (Unsloth + TRL) and the Kaggle job driver
  prompts.py   shared prompt construction
  tasks.py     task config loader
tasks/         pre-registered task configs
configs/       SFT run grids
tests/         unit tests and the synthetic exit check
```

Planned modules: GRPO in `train/`, `serve/` (quantization, vLLM), `monitor/` (drift and re-evaluation).

## Conventions

- System `a` is the teacher and `b` the student. Every difference is `a − b`, so positive means the student is worse.
- Golden labels are verified ground truth, never teacher outputs.
- The quantized model is the one evaluated, because it is the one served.
- Datasets, teacher outputs, runs and reports stay out of git.
