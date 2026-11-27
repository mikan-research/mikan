# Adding a task

A task is a dataset with a verified test set, a pre-registered evaluation config, and a way for a program to check
an output. Open a "New task" issue first so the margin can be agreed before any runs.

## 1. Check the task fits

- **Narrow, checkable output:** a label, extracted fields, or JSON against a schema. Open-ended text is out of scope.
- **Verified golden set:** dataset ground truth or human-checked labels, never teacher outputs.
- **Enough test items** to resolve the margin you want (step 3).
- **A licence** that allows redistributing derived labels.

## 2. Write the data module

Add `mikan/data/<task>.py` with a `prepare(out)` function and a `main()` CLI, following `banking77.py` (labels) or
`massive.py` (frames). It should:

- download the source and record its URL and SHA-256 in `manifest.json`;
- write `train.jsonl`, `val.jsonl`, `fewshot.jsonl` and `golden.jsonl` with records `{"id", "text", "label"}`
  (frames tasks add `"frame": {"intent", "slots": [{"type", "value"}]}`; `label` is then the intent);
- write the label list (`labels.txt`, or `intents.txt` and `slot_types.txt` for frames);
- remove train and val items whose normalised text appears in golden, and duplicates, and report the counts;
- be deterministic (fixed seed), so published predictions line up by id.

## 3. Pre-register the evaluation

Pick the primary metric and margin δ **before** any teacher or student output exists. Check power for your
golden-set size at a plausible disagreement rate (the share of items exactly one system gets right):

```bash
uv run mikan-eval power --n 2974 --disagreement 0.20 --delta 0.03
```

Then write `tasks/<task>.yaml`:

```yaml
name: my-task
kind: labels                 # or frames
metric: accuracy             # frames: accuracy = exact frame match
secondary_metrics: [macro_f1]  # labels only
delta: 0.02                  # non-inferiority margin (a fraction)
confidence: 0.95
power: 0.8
n_boot: 10000
seed: 0
labels_file: ../data/my-task/labels.txt
# slot_types_file: ../data/my-task/slot_types.txt   # frames only
label_source: "official test split (dataset ground truth)"
```

Never change these values after seeing outputs; every report records the file's SHA-256.

## 4. Decide how outputs are checked

- **Labels:** an output is valid only if it is exactly one label (after stripping whitespace).
- **Frames:** `mikan/frames.py` parses the exact JSON shape, checks the intent and slot types against the schema,
  checks that slot values are copied from the input, and scores exact match on the (type, value) multiset. The
  same functions are the GRPO reward, so evaluation and training agree.
- Other output types need a checker of their own, with the same split between "valid", "correct" and a reward.

## 5. Add prompts if needed

Teacher prompts live in `mikan/prompts.py` (`system_prompt` for labels, `frames_system_prompt` for frames) and the
student's short prompt in `mikan/train/examples.py`. `task_setup()` picks them from the files in the data folder.

## 6. Test it

Add tests under `tests/`: split sizes and leakage removal, the checker on valid and invalid outputs, and the config
loading. Tests that need downloaded data should skip when it is absent. `uv run pytest -q` must pass.

## 7. Run it

Label a val pilot with the teacher, pick the prompt on val, label all splits, then train and evaluate students as in
[training.md](training.md). Share the reports in a "Result report" issue, including failures.
