# Training students

How to label data with a teacher, train a student with LoRA SFT, and improve it with GRPO. Everything runs on one
16 GB GPU (a free Kaggle or Colab T4 is enough for Qwen3 0.6B / 1.7B / 4B). Published teacher labels let you skip
straight to step 2.

The GPU commands need extra packages that the base install does not include: `vllm` for teacher labelling and
`unsloth` (which brings TRL and PEFT) for training. On the GPU machine, install them next to the repo
(`pip install -e . vllm==0.13.0` or `pip install -e . unsloth`) and run the commands with `python -m` as shown.
The Kaggle driver below installs them for you.

## 1. Teacher labels

Teacher labels for both tasks are published in the
[banking77-distill](https://huggingface.co/datasets/mikan-research/banking77-distill) and
[massive-distill](https://huggingface.co/datasets/mikan-research/massive-distill) datasets (`teacher/`). To produce
your own, run a teacher with vLLM on all splits. Decoding is constrained to valid outputs (a label choice, or the
frame JSON schema), and the outputs are cached so the teacher is called once.

```bash
# --tp 2 and --dtype float16 for two T4s
python -m mikan.intake.vllm_predict --model <teacher> --data data/massive \
    --splits pilot --retrieved 8 --tp 2 --dtype float16 --out runs/my-teacher
```

- `--shots k` puts k ranked examples per label in one shared prompt (computed once by prefix caching).
- `--retrieved k` (frames tasks) uses each item's k nearest labelled examples instead, from `retrieved.jsonl`
  built by `uv run python -m mikan.data.retrieve --data data/massive`. Slower per item, much better when errors
  come from dataset annotation conventions.
- Pilot first on `pilot` (the first 300 val items) and choose the prompt on val, never on golden.

On a T4, use a GPTQ 4-bit build and `--attention-backend TRITON_ATTN`; compressed-tensors (Marlin) 4-bit builds
need a newer GPU. `mikan/intake/modal_predict.py` runs the same thing on Modal.

## 2. LoRA SFT

```bash
python -m mikan.train.sft --data data --task massive --teacher-dir massive-teacher \
    --model Qwen/Qwen3-1.7B --targets teacher --out runs/sft-1.7b
```

`--data` is a folder holding `<task>/` (the prepared splits) and `<teacher-dir>/train_preds.jsonl`. The student
sees a short prompt with no label list and no examples, and learns the output format from the targets. After every
epoch it predicts val; the best epoch's adapter is kept in `best_adapter/` and only that one predicts golden, once.
`--targets gold` trains on ground truth instead (the ceiling for comparison); `--n-train N` uses a nested subset.

## 3. GRPO (structured-output tasks)

```bash
python -m mikan.train.grpo --data data --task massive --teacher-dir massive-teacher \
    --model Qwen/Qwen3-1.7B --adapter runs/sft-1.7b/best_adapter --out runs/grpo-1.7b
```

GRPO continues the SFT adapter with a programmatic reward against the teacher's frame (`mikan.frames.reward`):
0 for an invalid frame, else 0.1 + 0.3 · intent match + 0.6 · slot F1 · groundedness. Val is checked every
`--eval-every` steps, and step 0 (the SFT adapter itself) is a candidate, so GRPO is only kept if it helps.
`--prompt-offset N` takes prompts after the first N training items, so a student SFT-trained on `--n-train N` gets
prompts it has never seen.

## 4. Evaluate

```bash
uv run mikan-eval report --task tasks/massive.yaml --gold data/massive/golden.jsonl \
    --a runs/my-teacher/golden_preds.jsonl --b runs/grpo-1.7b/golden_preds.jsonl --out reports/grpo-1.7b
uv run python -m mikan.eval.sweep --task tasks/massive.yaml --gold data/massive/golden.jsonl \
    --teacher runs/my-teacher --runs runs/ --out reports/sweep      # every run dir at once
```

## Running on Kaggle

`mikan.train.kaggle_job` packs the code, prepared data, teacher labels and a run grid into a private Kaggle
dataset and starts a private T4 x2 kernel on your account (Kaggle CLI token in `~/.kaggle/`). Each GPU runs its own
queue of configs.

```bash
uv run python -m mikan.train.kaggle_job push --configs configs/massive_phase3_a.json \
    --job my-grid --data massive,massive-teacher --dataset my-bundle
uv run python -m mikan.train.kaggle_job status --job my-grid
uv run python -m mikan.train.kaggle_job pull --job my-grid       # to runs/kaggle/my-grid/
```

A grid is a JSON list of runs. Keys become trainer flags (`n_train` → `--n-train`); `gpu` picks the queue;
`"script": "grpo"` runs the GRPO trainer; a value starting with `@` is a path under the kernel's working folder, so
a GRPO run can start from an SFT run earlier in the same queue:

```json
[
  {"name": "sft-1.7b", "gpu": 0, "model": "Qwen/Qwen3-1.7B", "task": "massive",
   "teacher_dir": "massive-teacher", "targets": "teacher"},
  {"name": "grpo-1.7b", "gpu": 0, "script": "grpo", "model": "Qwen/Qwen3-1.7B", "task": "massive",
   "teacher_dir": "massive-teacher", "adapter": "@runs/sft-1.7b/best_adapter"}
]
```

`--runner teacher` runs teacher labelling instead (`mikan/intake/kaggle_teacher.py`, configs like
`configs/teacher_massive_knn_pilot.json`). Kaggle releases logs and outputs only when the kernel finishes.

## Tips for a 16 GB T4

- fp16 only (no bf16, no FlashAttention 2, no FP8).
- The trainers free the CUDA cache between training and generation; for 4B use `--batch 16 --eval-batch 16`.
- Kaggle sessions stop at 12 hours, so split long teacher runs (`train_part1` / `train_part2` subsets exist for MASSIVE).
