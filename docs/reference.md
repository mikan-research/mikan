# Reference

## Commands

### Evaluation (`mikan-eval`, CPU)

| Command | What it does | Main flags |
|---|---|---|
| `mikan-eval report` | Pre-registered test for one teacher/student pair; writes `report.json` and `report.md` | `--task`, `--gold`, `--a` (teacher preds), `--b` (student preds), `--a-name`, `--b-name`, `--out`, `--labels` (override), `--examples` |
| `mikan-eval power` | Analytic and simulated power for a golden-set size | `--n`, `--disagreement`, `--delta`, `--true-diff`, `--confidence`, `--target-power`, `--sims`, `--seed` |
| `mikan-eval selfcheck` | Pass rates for synthetic students with known gaps around δ | `--n`, `--disagreement`, `--delta`, `--datasets`, `--boot`, `--seed` |
| `python -m mikan.eval.sweep` | `report` for every run folder against one teacher, plus `summary.md` / `summary.json` | `--task`, `--gold`, `--teacher` (run folder), `--runs` (folder of run folders), `--out` |

### Data (CPU)

| Command | What it does | Main flags |
|---|---|---|
| `python -m mikan.data.banking77` | Banking77 splits, ranked few-shot pool, leakage removal | `--out`, `--seed`, `--val-frac`, `--fewshot-per-label` |
| `python -m mikan.data.massive` | MASSIVE en-US frames splits, slot-type-covering few-shot pool | `--out`, `--seed`, `--fewshot-per-intent` |
| `python -m mikan.data.retrieve` | k nearest labelled examples per item (TF-IDF) to `retrieved.jsonl` | `--data`, `--k` |

### Inference and training (GPU)

| Command | Needs | Main flags |
|---|---|---|
| `python -m mikan.intake.vllm_predict` | vLLM | `--model`, `--data`, `--splits`, `--shots` or `--retrieved`, `--free` (unconstrained), `--dtype`, `--tp`, `--quantization`, `--attention-backend`, `--max-model-len`, `--gpu-mem`, `--out` |
| `modal run -m mikan.intake.modal_predict` | Modal account | `--splits`, `--role teacher\|student`, `--model`, `--shots`, `--data-dir`, `--out-dir` |
| `python -m mikan.train.sft` | Unsloth | `--data`, `--task`, `--teacher-dir`, `--model`, `--targets teacher\|gold`, `--n-train`, `--epochs`, `--lr`, `--rank`, `--batch`, `--eval-batch`, `--max-seq`, `--load-in-4bit`, `--eval-limit`, `--seed`, `--out` |
| `python -m mikan.train.grpo` | Unsloth | `--data`, `--task`, `--teacher-dir`, `--targets`, `--model`, `--adapter`, `--n-prompts`, `--prompt-offset`, `--steps`, `--num-generations`, `--prompts-per-step`, `--lr`, `--temperature`, `--max-completion`, `--eval-every`, `--eval-batch`, `--eval-limit`, `--out` |
| `python -m mikan.train.kaggle_job push\|status\|pull\|bundle` | Kaggle CLI token | `--configs`, `--job`, `--runner sft\|teacher`, `--data`, `--dataset`, `--timeout-hours` |
| `python -m mikan.publish_hf stage\|upload` | Hugging Face token | `--task banking77\|massive`, `--namespace`, `--allow-public` |

Named subsets accepted wherever a split is: `smoke` (first 50 val items), `pilot` (first 300 val items), and for
MASSIVE `train_part1` / `train_part2` (the two halves of train).

## File formats

All data files are JSON Lines.

| File | Record |
|---|---|
| splits (`train`, `val`, `fewshot`, `golden`) | `{"id", "text", "label"}`; frames tasks add `"frame"`; `fewshot` adds `"rank"` |
| predictions (`<split>_preds.jsonl`) | `{"id", "pred"}`, the raw model output |
| run metadata (`<split>_meta.json`) | model, prompt hash, timings, token counts, score against gold |
| retrieval (`retrieved.jsonl`) | `{"id", "demo_ids": [...]}`, most similar first |

A frame is `{"intent": "<intent>", "slots": [{"type": "<slot type>", "value": "<words from the input>"}, ...]}`.
Slots are a list because a type can repeat.

Prediction files must contain exactly the gold ids; a missing or extra id is an error, never a silently smaller
test set.

## Task config keys (`tasks/<task>.yaml`)

| Key | Meaning |
|---|---|
| `name` | task name shown in reports |
| `kind` | `labels` (default) or `frames` |
| `metric` | primary metric deciding pass/fail: `accuracy` or `macro_f1` (frames: `accuracy` = exact frame match) |
| `secondary_metrics` | reported alongside, labels tasks only |
| `delta` | non-inferiority margin as a fraction (0.02 = 2 points) |
| `confidence` | two-sided CI level (default 0.95) |
| `power` | target power for the test-set size check (default 0.8) |
| `n_boot`, `seed` | bootstrap resamples and seed |
| `labels_file`, `slot_types_file` | label list(s), relative to the config file |
| `label_source` | where golden labels come from, printed in the report |

## Scoring rules

- **Labels:** an output counts only if, after stripping surrounding whitespace, it is exactly one label. Anything
  else is invalid and wrong.
- **Frames:** an output is valid if it parses as exactly the frame shape, with the intent and every slot type in the
  schema. It is correct if the intent and the multiset of (type, value) pairs equal gold, comparing values case- and
  whitespace-insensitively. Reports add intent accuracy, slot precision / recall / F1, validity and groundedness
  (the share of slot values copied from the input).
- System `a` is the reference (teacher) and `b` the candidate (student); every difference is `a − b`, so positive
  means the student is worse.

## Self-check

`mikan-eval selfcheck` (n = 3,080, 77 classes, 8% disagreement, δ = 2 points, 200 synthetic test sets per row):

| True gap (teacher − student) | Pass rate | Expected |
|---|---|---|
| −2 points | 1.00 | 1.00 |
| 0 | 0.98 | 0.975 |
| +1 point | 0.43 | 0.50 |
| +2 points (= δ) | 0.03 | 0.025 |
| +3 / +4 points | 0.00 | 0.00 |

Equal or better students pass; a student exactly δ worse passes about 2.5% of the time, the test's designed error
rate; worse students do not pass.

## Code layout

```text
mikan/
  eval/          metrics, paired statistics, power, synthetic data, reports, sweep, CLI
  data/          dataset preparation (Banking77, MASSIVE) and retrieval
  intake/        batch prediction with vLLM (local, Modal, Kaggle teacher runner)
  train/         LoRA SFT and GRPO trainers, targets and student prompts, Kaggle driver
  frames.py      frame parsing, metrics and GRPO reward
  prompts.py     teacher prompts
  tasks.py       task config loader
  publish_hf.py  Hugging Face publishing
tasks/           pre-registered task configs
configs/         Kaggle run grids
tests/           unit tests and the synthetic self-check
```
