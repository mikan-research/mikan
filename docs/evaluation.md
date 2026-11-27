# Evaluation method

The question every run answers: "is the small model worse than the frontier model on this task, and by how much?" The test is fixed before training.

## Setup

- **Golden test set**: 200 to 500 verified items, held out and never used for training or checkpoint selection.
- **Systems compared** on the same items: the teacher (frontier API) and the student (quantized, as served). Optionally the zero-shot base model too, as a floor.
- **Primary metric**, chosen per task in the task config:
  - Classification: accuracy (macro-F1 if classes are imbalanced)
  - Extraction / JSON: per-field F1 and schema-validity rate. For MASSIVE frames the primary metric is **exact frame match** (intent and the multiset of (slot type, value) pairs equal gold, values compared case- and whitespace-insensitively), with intent accuracy, slot F1, validity and groundedness (slot values copied from the input) reported alongside
  - Free text: rubric or LLM-judge score (weaker evidence; flagged as such in the report)

## Pre-registered pass criterion: non-inferiority

Before training, set a margin δ in the task config (for example 2 accuracy points).

- Pass if the **upper bound of the 95% CI** for (teacher − student) is below δ.
- In words: "with 95% confidence, the student is at most δ worse, and it may be better."

Why not "no significant difference": a non-significant p-value only means no difference was detected. With a small test set it can hide a real drop. Non-inferiority tests the claim that actually matters.

## Statistical tests

- **Paired bootstrap** (resample items, 10k resamples) for the CI on the metric difference. Works for any metric, including F1.
- **McNemar's test** on paired per-item correctness, reported alongside.
- **Power check** in the audit: simulate or compute whether the golden-set size can rule out a drop of δ, given the expected accuracy. If not, enlarge the golden set before training.
- If several checkpoints or models are compared on the golden set, correct for multiple comparisons or pick the final model on the validation split only.

## Report contents

1. Task summary, margin δ, golden-set size and label source
2. Headline: pass/fail, metric for each system, difference with 95% CI
3. Per-slice breakdown (by label, input length, or metadata field) to surface hidden weak spots
4. Format-validity rate
5. p50/p95 latency and cost per 1k calls, student vs teacher
6. Error analysis: items where the student is wrong and the teacher right, and the reverse
7. Full precision vs quantized: metric delta caused by quantization
8. Limitations: test-set size, label source, drift risk
