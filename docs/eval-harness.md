# Eval harness (Phase 0)

Implements [evaluation.md](evaluation.md). Code in `mikan/eval/`, tests in `tests/`.

## Setup

```bash
uv sync                 # creates .venv with numpy, scipy, pyyaml, pytest
uv run pytest -q        # 84 tests, about 20 s
```

## Commands

**Report** for a teacher/student pair on the golden set:

```bash
uv run mikan-eval report --task tasks/banking77.yaml \
  --gold data/banking77/golden.jsonl \
  --a runs/teacher/test_preds.jsonl --b runs/student/test_preds.jsonl \
  --out reports/<run-name>
```

Writes `report.json` (everything, machine-readable) and `report.md` (decision, CIs, McNemar, power, per-label slices, error examples, limitations).

File formats (JSON Lines):

- gold: `{"id": ..., "label": "<label>", "text": "<input, optional>"}`
- predictions: `{"id": ..., "pred": "<raw model output>"}`

Ids must match exactly between gold and each prediction file, or the command fails. A prediction that is not exactly a label (after stripping whitespace) counts as wrong and lowers the validity rate.

**Frames tasks** (`kind: frames` in the task config, e.g. `tasks/massive.yaml`): gold records also carry
`"frame": {"intent", "slots": [{"type", "value"}]}` and predictions are raw JSON strings. Each item is coded as
correct when the whole frame matches gold exactly, invalid when it does not parse against the schema (`mikan.frames.parse`),
and wrong otherwise, so the same bootstrap, McNemar and power machinery gives exact-match statistics. The report adds
intent accuracy, slot precision/recall/F1, validity and groundedness, and shows gold frames in the error examples.

**Sweep** many student runs against the teacher (one report per run plus `summary.md` / `summary.json`):

```bash
uv run python -m mikan.eval.sweep --task tasks/massive.yaml --gold data/massive/golden.jsonl \
    --teacher runs/massive-teacher-knn8 --runs <dir of run dirs> --out reports/phase3
```

**Power check** for a test-set size:

```bash
uv run mikan-eval power --n 3080 --disagreement 0.08 --delta 0.02
```

**Synthetic exit check**:

```bash
uv run mikan-eval selfcheck
```

## Modules

| Module | Purpose |
|---|---|
| `metrics.py` | accuracy, macro-F1, validity rate on integer label arrays (`-1` = invalid output) |
| `stats.py` | paired bootstrap CI (vectorised for accuracy), McNemar (exact + χ²), non-inferiority rule |
| `power.py` | analytic and simulated power, smallest resolvable margin, required n |
| `synthetic.py` | paired predictions with a known gap; the exit check |
| `data.py` | JSONL loading, strict id alignment, label mapping; `align_frames` / `align_for_task` for frames tasks |
| `report.py` | result dict, Markdown rendering, writing files |
| `cli.py` | `mikan-eval report / power / selfcheck` |
| `sweep.py` | evaluate a directory of runs against the teacher, one summary table |
| `../frames.py` | frame parsing, canonical form, exact match, slot F1, groundedness, GRPO reward, JSON schema |
| `../tasks.py` | task config loader; the report records the config's SHA-256 |

Convention: system `a` is the teacher, `b` the student, and every difference is `a − b`, so positive means the student is worse.

## Phase 0 exit result

`mikan-eval selfcheck` (n = 3080, 77 classes, disagreement 8%, δ = 2 points, 200 synthetic test sets per row, 2000 bootstrap resamples each):

| True gap (teacher − student) | Pass rate | Expected (analytic) |
|---|---|---|
| −2 points (student better) | 1.000 | 1.000 |
| 0 | 0.980 | 0.975 |
| +1 point | 0.430 | 0.501 |
| +2 points (= δ) | 0.030 | 0.025 |
| +3 points | 0.000 | 0.000 |
| +4 points | 0.000 | 0.000 |

Equal or better students pass. A student exactly δ worse passes about 2.5% of the time, which is the test's designed error rate. Worse students never pass. The +1 point row is within Monte Carlo noise of the prediction (±0.035 per standard error at 200 datasets).

## Power for Banking77 (n = 3080, δ = 2 points)

| Disagreement rate | Power if truly equal | Smallest resolvable margin (80% power) |
|---|---|---|
| 5% | 99.9% | 1.13 points |
| 8% | 97.5% | 1.43 points |
| 12% | 89.3% | 1.75 points |
| 15% | 81.8% | 1.96 points |

δ = 2 is supported as long as teacher and student disagree on correctness for no more than about 15% of test items. The report prints the observed rate and flags an underpowered test.
