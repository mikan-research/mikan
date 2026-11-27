# Contributing to Mikan

Thanks for helping. Mikan distils a large model's behaviour on one narrow task into a small open model and judges
it with a pre-registered non-inferiority test. Contributions of new tasks, better students, evaluation tooling,
serving work and bug fixes are all welcome.

## Getting set up

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mikan-research/mikan && cd mikan
uv sync
uv run pytest -q
```

The tests need no GPU. Tests that use prepared data skip themselves until you run the data scripts
(`mikan.data.banking77`, `mikan.data.massive`).

GPU work runs on your own accounts: Kaggle (`mikan.train.kaggle_job`, private datasets and kernels under your
Kaggle username) or Modal (`mikan.intake.modal_predict`). Teacher labels for both tasks are already published, so
you can train students without re-running a teacher: see [mikan-research on Hugging Face](https://huggingface.co/mikan-research).

## Ground rules that keep results trustworthy

These are what make a Mikan result mean something. PRs that break them will be asked to change.

1. **Pre-register before you look.** A task's margin and primary metric live in `tasks/<task>.yaml` and are fixed
   before any teacher or student output exists. Never change them after seeing results; reports record the
   config's SHA-256, so a change is visible.
2. **Golden labels are ground truth, never teacher outputs.** The golden set is evaluated once per model, at the end.
3. **Select on validation only.** Checkpoints, prompts and hyperparameters are chosen on the val split; golden
   never influences a choice.
4. **No leakage.** Training data, few-shot pools and retrieval pools never contain val or golden items.
5. **Report failures too.** A student that fails non-inferiority is a result; include it.

## Ways to contribute

- **A new task.** Add `mikan/data/<task>.py` (download, splits, leakage checks, `manifest.json`), a pre-registered
  `tasks/<task>.yaml` (check power first: `uv run mikan-eval power --n <golden size> --disagreement <guess> --delta <δ>`),
  and tests. Structured-output tasks can reuse `mikan/frames.py` or add a checker of their own. Open a
  "New task" issue first so the margin can be agreed before any runs.
- **A result.** Train or evaluate a student and share the `report.md` / `report.json` from `mikan-eval report` or
  `mikan.eval.sweep`, plus the run config. Use the "Result report" issue template.
- **Models and data.** Adapters and datasets live in the Hugging Face organization; open a discussion or pull
  request on the relevant repo for card fixes, or link new artifacts from a result issue.
- **Code.** Bug fixes, faster training, quantization and serving (Phase 4), monitoring (Phase 5).

## Pull requests

- Keep each PR to one change, with tests for new behaviour. `uv run pytest -q` must pass.
- Commit messages are one line, imperative, and describe the change ("Add MASSIVE frames task", "Fix eval ...").
- Don't commit datasets, model weights, runs or reports; they are git-ignored and belong on Hugging Face.
- Don't commit credentials (Kaggle, Modal or Hugging Face tokens) or any personal data.
- Docs live in `docs/` (start at `docs/README.md`); update them when behaviour changes.

## Code of conduct

Be respectful and assume good faith. Harassment or personal attacks are not tolerated; maintainers may remove
comments or contributions that cross that line.

## Licence

By contributing you agree that your contributions are licensed under the Apache License 2.0 (see `LICENSE`).
