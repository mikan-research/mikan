# Getting started

This tutorial reproduces a published result: the Qwen3-0.6B student passing non-inferiority against its teacher on
Banking77. It runs on a laptop in a few minutes and needs no GPU, because the teacher's and the student's predictions
are downloaded from Hugging Face.

## 1. Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mikan-research/mikan && cd mikan
uv sync
uv run pytest -q
```

## 2. Prepare the Banking77 splits

```bash
uv run python -m mikan.data.banking77 --out data/banking77 --fewshot-per-label 10
```

This downloads Banking77, removes test-set leaks and duplicates from train, and writes `train`, `val`, `fewshot` and
`golden` (the official test split, ground truth) to `data/banking77/`, plus a `manifest.json` with source and file
hashes. The split is deterministic, so the item ids match the published predictions.

## 3. Download the teacher's and a student's predictions

```bash
uvx --from huggingface_hub hf download mikan-research/banking77-distill --repo-type dataset \
    teacher/golden_preds.jsonl students/sft-qwen3-0.6b-teacher-full/golden_preds.jsonl \
    --local-dir downloads/banking77-distill
```

Each file is JSON Lines, one `{"id", "pred"}` per golden item, with the raw model output in `pred`.

## 4. Run the pre-registered test

```bash
uv run mikan-eval report --task tasks/banking77.yaml \
    --gold data/banking77/golden.jsonl \
    --a downloads/banking77-distill/teacher/golden_preds.jsonl --a-name teacher \
    --b downloads/banking77-distill/students/sft-qwen3-0.6b-teacher-full/golden_preds.jsonl --b-name student \
    --out reports/getting-started
```

The first line printed is the decision:

```text
PASS: accuracy teacher - student = 0.06 pts, CI [-0.81, 0.94], delta = 2.00
```

## 5. Read the report

`reports/getting-started/report.md` contains:

- the decision and the 95% confidence interval for (teacher − student) accuracy, against the margin δ = 2 points
  fixed in `tasks/banking77.yaml` before any training;
- accuracy, macro-F1 and valid-output rate for both systems;
- McNemar's test on paired correctness;
- test-set power at the observed disagreement rate;
- per-label slices, worst first, and examples where only one system is right;
- limitations, and the SHA-256 of the task config, so a margin changed after the fact is visible.

## Next steps

- Compare other published students: every run's predictions are in `students/` in the
  [banking77-distill](https://huggingface.co/datasets/mikan-research/banking77-distill) and
  [massive-distill](https://huggingface.co/datasets/mikan-research/massive-distill) datasets.
- Train your own student: [training.md](training.md).
- Bring your own task: [adding-a-task.md](adding-a-task.md).
- Learn what the test does and does not claim: [evaluation.md](evaluation.md).
