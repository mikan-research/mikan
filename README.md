# Mikan

Distil a large model's behaviour on one narrow task into a small open model, and prove statistically that it is not worse.

Mikan takes a task handled by a large teacher model (classification, extraction, structured output), trains a small student (0.6B to 4B) to do the same job, quantizes and serves it, and judges it with a **pre-registered non-inferiority test**: the student passes only if the 95% CI for (teacher − student) stays below a margin δ fixed before training.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Eval harness: bootstrap CI, McNemar, non-inferiority, power, reports | Done |
| 1 | Banking77 teacher labels (open model on Modal) and zero-shot baselines | Done |
| 2 | SFT distillation (LoRA) with data-size and model-size ablations | Next |
| 3 | GRPO with programmatic rewards | Planned |
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
  prompts.py   shared prompt construction
  tasks.py     task config loader
tasks/         pre-registered task configs
tests/         unit tests and the synthetic exit check
```

Planned modules: `train/` (SFT, GRPO), `serve/` (quantization, vLLM), `monitor/` (drift and re-evaluation).

## Conventions

- System `a` is the teacher and `b` the student. Every difference is `a − b`, so positive means the student is worse.
- Golden labels are verified ground truth, never teacher outputs.
- The quantized model is the one evaluated, because it is the one served.
- Datasets, teacher outputs, runs and reports stay out of git.
