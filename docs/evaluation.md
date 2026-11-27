# Evaluation method

Every run answers one question: is the small student worse than the teacher on this task, and by how much? The
test is fixed before training, so the answer cannot be tuned after the fact.

## Setup

- **Golden set:** a held-out test split with verified labels (dataset ground truth), never used for training,
  prompt choice or checkpoint selection, and never made of teacher outputs. Otherwise the test would measure
  agreement with the teacher, not correctness.
- **Systems compared** on the same items: the teacher and the student, as each would be run in practice.
- **Primary metric,** set in the task config: accuracy or macro-F1 for labels; exact frame match for structured
  output (intent and every (slot type, value) pair equal to gold).

## Pass criterion: non-inferiority

Before training, set a margin δ in the task config, for example 2 accuracy points.

- The student **passes** if the upper bound of the 95% CI for (teacher − student) is below δ.
- In words: with 95% confidence, the student is at most δ worse, and it may be better.

Why not "no significant difference"? A non-significant p-value only means no difference was detected; with a small
test set it can hide a real drop. Non-inferiority tests the claim that matters: the student is not meaningfully
worse.

## Statistics

- **Paired bootstrap** over items (10,000 resamples by default) for the CI on the metric difference. Both systems
  are resampled together, which keeps the CI tight when they agree on most items. It works for any metric.
- **McNemar's test** on paired per-item correctness (exact and continuity-corrected χ²), reported alongside. It
  tests for any difference and does not decide pass/fail.
- **Power.** What decides how small a margin a test set can resolve is the disagreement rate (the share of items
  exactly one system gets right), not overall accuracy. The smallest margin resolvable with 80% power is about
  2.8 · √(disagreement / n). With 3,080 items, δ = 2 points is resolvable up to about 15% disagreement; with 2,974
  items and an expected 20–30% disagreement for exact frame match, δ = 3 points was chosen instead.
  `mikan-eval power` computes this, and every report prints the observed rate and flags an underpowered test.
- **One look at golden.** Checkpoints and prompts are chosen on val, so the golden test is run once per model and
  needs no correction for multiple looks.

## What a report contains

1. Task, margin, golden-set size, label source, and the SHA-256 of the task config
2. The decision and, for each system, the primary metric with the difference and its 95% CI
3. Secondary metrics (macro-F1, or intent accuracy, slot precision / recall / F1, validity and groundedness for
   frames) and their CIs
4. McNemar's test and the test-set power at the observed disagreement
5. Per-label (or per-intent) slices, where the student is worst first
6. Examples where only the teacher is right, and the reverse
7. Limitations: label source, CI method, power, invalid outputs
