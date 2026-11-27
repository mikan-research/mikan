<p align="center"><img src="docs/assets/logo.svg" alt="Mikan logo: a large mandarin beside a small one" width="128" height="128"></p>

<h1 align="center">Mikan</h1>

Distil a large model's behaviour on one narrow task into a small open model, and prove statistically that it is not worse.

Mikan takes a task handled by a large teacher model (classification, extraction, structured output), trains a small
student (0.6B to 4B) to do the same job, and judges it with a **pre-registered non-inferiority test**: the student
passes only if the 95% CI for (teacher − student) stays below a margin δ fixed before training.

- **Models and data:** [huggingface.co/mikan-research](https://huggingface.co/mikan-research): LoRA adapters,
  teacher labels and every run's predictions.
- **Docs:** [docs/](docs/README.md): getting started, training, adding a task, reference, method.
- **Contributing:** new tasks, results and code are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Results

| Task | Metric, margin | Teacher | Student | Student score | Decision |
|---|---|---|---|---|---|
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77), 77 intents | accuracy, δ = 2 | Qwen3-30B-A3B, 10-shot: 85.6% | Qwen3-0.6B, LoRA SFT | 85.6% | pass |
| | | | Qwen3-4B, LoRA SFT | 86.0% | pass |
| [MASSIVE](https://github.com/alexa/massive) en-US, intent + slot frames | exact frame match, δ = 3 | Qwen3-30B-A3B, 8 retrieved examples: 64.4% | Qwen3-0.6B, SFT + GRPO | 64.8% | pass |
| | | | Qwen3-1.7B, SFT + GRPO | 65.7% | pass |

Scores are on the official test splits (3,080 and 2,974 items, ground-truth labels). Students use a short prompt
with no label list and no examples, so they are much cheaper to run than the teacher. Training on ground truth
instead of teacher labels reaches 93–94% (Banking77) and 72.1% (MASSIVE), so the teacher's labels set the ceiling.
Every model card on Hugging Face has the confidence interval and the training recipe.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mikan-research/mikan && cd mikan
uv sync
uv run pytest -q
```

Evaluation runs on a CPU. Teacher labelling needs [vLLM](https://github.com/vllm-project/vllm) and training needs
[Unsloth](https://github.com/unslothai/unsloth) on a GPU; a free 16 GB T4 is enough.

## Quickstart

Reproduce the Banking77 result without a GPU, using the published predictions:

```bash
uv run python -m mikan.data.banking77 --out data/banking77 --fewshot-per-label 10
uvx --from huggingface_hub hf download mikan-research/banking77-distill --repo-type dataset \
    teacher/golden_preds.jsonl students/sft-qwen3-0.6b-teacher-full/golden_preds.jsonl \
    --local-dir downloads/banking77-distill
uv run mikan-eval report --task tasks/banking77.yaml --gold data/banking77/golden.jsonl \
    --a downloads/banking77-distill/teacher/golden_preds.jsonl --a-name teacher \
    --b downloads/banking77-distill/students/sft-qwen3-0.6b-teacher-full/golden_preds.jsonl --b-name student \
    --out reports/quickstart
```

```text
PASS: accuracy teacher - student = 0.06 pts, CI [-0.81, 0.94], delta = 2.00
```

`reports/quickstart/report.md` has the full report. [docs/getting-started.md](docs/getting-started.md) walks
through it.

## How it works

1. **Data:** seeded splits with leakage removed; the golden set is dataset ground truth, never teacher output.
2. **Pre-registration:** the metric and margin go in `tasks/<task>.yaml` before any output exists; every report
   records the file's SHA-256.
3. **Teacher labels:** a large open model with vLLM, decoding constrained to valid outputs, prompted with fixed or
   retrieved examples chosen on a val pilot.
4. **Students:** LoRA SFT on the teacher's outputs, then GRPO with a programmatic reward where an output can be
   checked by code (schema validity, intent, slot F1, spans copied from the input). Every choice is made on val.
5. **Evaluation:** paired bootstrap CI, McNemar's test and a power check on the golden set, once per model.

Details: [docs/pipeline.md](docs/pipeline.md) and [docs/evaluation.md](docs/evaluation.md).

## Project layout

```text
mikan/         eval (statistics, reports), data (splits, retrieval), intake (teacher inference),
               train (SFT, GRPO, Kaggle driver), frames.py (structured-output checks and reward)
tasks/         pre-registered task configs
configs/       example run grids
docs/          documentation
tests/         unit tests and the synthetic self-check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: pre-register a task's margin before any outputs exist, never use
teacher outputs as golden labels, choose everything on the val split, and report failures as well as passes.

## Licence

Code: [Apache-2.0](LICENSE). Adapters on Hugging Face: Apache-2.0, like the Qwen3 base models. Datasets: CC-BY-4.0,
following [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva et al., 2020) and
[MASSIVE](https://github.com/alexa/massive) (FitzGerald et al., 2022).
