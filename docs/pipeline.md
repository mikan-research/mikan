# How the pipeline works

One run takes a task from a teacher's outputs to a student with a pass/fail verdict.

```text
task data ─► splits + leakage checks ─► teacher labels ─► LoRA SFT ─► GRPO ─► evaluation on golden
                                          (val pilot)       (val selection)        (once, pre-registered)
```

## Data

Each task has a train split for distillation, a val split for every choice (prompt, checkpoint, hyperparameters),
a few-shot pool for teacher prompts, and a golden split for the final test only. Golden is dataset ground truth,
never teacher output, so the test measures correctness rather than agreement with the teacher. Train and val items
whose text also appears in golden are removed, and the split is seeded, so anyone can rebuild it exactly.

## Teacher labels

The teacher is a large open model run in batch with vLLM. Decoding is constrained to valid outputs: a choice among
the labels, or the frame JSON schema. That makes every teacher output usable as a training target and isolates its
errors to content rather than format.

Two prompting styles:

- **Fixed examples:** one shared prompt with an instruction, the label list and ranked examples. vLLM's prefix
  cache computes it once, so it is cheap per item.
- **Retrieved examples:** for each item, its nearest labelled training examples by TF-IDF (never the item itself,
  never val or golden). Each prompt is different, so it is slower, but it shows the teacher how this dataset
  annotates requests like this one. On MASSIVE it raised the teacher's exact match on a val pilot from 48% to 67%,
  because fixed-prompt errors were mostly annotation conventions (which words get a slot, where spans start).

The prompt is chosen on a val pilot, then the teacher labels every split once and its outputs are cached.

## Students: LoRA SFT

Students are small open models (Qwen3 0.6B to 4B) trained with LoRA on (input, teacher output) pairs, with the
loss on the output only. Their prompt is short, with no label list or examples: the student learns the label set
and format from training, which is what makes it cheap to serve. The best epoch on val is kept, and only that
adapter ever predicts golden.

## Students: GRPO

Where a program can score an output, GRPO continues training the SFT adapter with sampled outputs and a
programmatic reward. For frames the reward is computed against the **teacher's** frame (golden is never used):

- 0 if the output is not a valid frame;
- otherwise 0.1 + 0.3 · intent match + 0.6 · slot F1 · groundedness, where groundedness is the share of slot values
  copied from the input, so invented spans earn nothing.

Design choices:

- **No KL term (β = 0).** With LoRA, TRL's reference model is the base model with the adapter disabled, not the SFT
  checkpoint, so a KL penalty would pull the student back toward the untrained model. Drift is limited instead by a
  low learning rate, a few hundred steps, and checkpoint selection.
- **Step 0 is a candidate.** Val is checked periodically, including before any GRPO step, so GRPO is kept only if
  it improves val.
- **Reward hacking is visible.** Validity and groundedness are logged as zero-weight rewards, and the evaluation
  uses the same checks as the reward.

## Evaluation

The student and the teacher are scored on the same golden items, and the student passes if the upper bound of the
95% CI for (teacher − student) is below the margin fixed in the task config before training. See
[evaluation.md](evaluation.md).

## Results so far

| Task | Teacher (golden) | Student | Student (golden) | Decision |
|---|---|---|---|---|
| Banking77, accuracy, δ = 2 | 85.6% | Qwen3-0.6B / 1.7B / 4B, SFT | 85.6% / 85.4% / 86.0% | pass |
| MASSIVE en-US, exact frame match, δ = 3 | 64.4% | Qwen3-0.6B / 1.7B, SFT + GRPO | 64.8% / 65.7% | pass |

Two patterns held on both tasks. Students trained on teacher labels land at about the teacher's accuracy, while the
same models trained on ground truth do much better (93–94% on Banking77, 72.1% on MASSIVE): the teacher's labels,
not the student's size, set the ceiling. And data size matters more than model size: on Banking77, 1k teacher
labels gave 71.0% against 85.4% for the full 8.2k. GRPO added +0.7 points after a full SFT and +4.4 points after a
1k-example SFT on MASSIVE. Per-run numbers are in the model cards and the `summary.json` of each dataset on
[Hugging Face](https://huggingface.co/mikan-research).
