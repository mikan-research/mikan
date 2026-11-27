"""Back up Mikan adapters and labels to the Hugging Face Hub as private repos.

    uv run --with huggingface_hub python -m mikan.publish_hf stage    # writes build/hf/
    uv run --with huggingface_hub python -m mikan.publish_hf upload   # creates private repos, uploads
    uv run --with huggingface_hub python -m mikan.publish_hf stage --task massive   # MASSIVE dataset repo
    uv run --with huggingface_hub python -m mikan.publish_hf upload --task massive

``stage`` copies each kept adapter plus its run metadata into ``build/hf/<repo>/`` and
writes a model card; the dataset repo gets the splits, teacher labels and every student
run's predictions. ``upload`` never changes an existing repo's visibility.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .train.examples import STUDENT_SYSTEM

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "kaggle" / "mikan-sft-phase2" / "runs"
TEACHER = REPO / "runs" / "teacher-qwen3-30b-a3b-instruct-2507-fp8-10shot"
SUMMARY = REPO / "reports" / "phase2" / "summary.json"
STAGE = REPO / "build" / "hf"
DATASET = "mikan-banking77-distill"

MODELS = {  # run name -> repo name
    "sft-qwen3-0.6b-teacher-full": "mikan-banking77-qwen3-0.6b-lora",
    "sft-qwen3-1.7b-teacher-full": "mikan-banking77-qwen3-1.7b-lora",
    "sft-qwen3-4b-teacher-full": "mikan-banking77-qwen3-4b-lora",
    "sft-qwen3-1.7b-gold-full": "mikan-banking77-qwen3-1.7b-gold-lora",
    "sft-qwen3-4b-gold-full": "mikan-banking77-qwen3-4b-gold-lora",
}
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def model_card(user: str, row: dict, teacher_acc: float) -> str:
    meta = json.loads((RUNS / row["run"] / "meta.json").read_text())
    base = meta["model"]
    labels = "teacher labels (Qwen3-30B-A3B-Instruct-2507-FP8, 10-shot)" if meta["targets"] == "teacher" \
        else "Banking77 ground-truth labels"
    return f"""---
base_model: {base}
library_name: peft
license: apache-2.0
datasets:
- PolyAI/banking77
- {user}/{DATASET}
pipeline_tag: text-generation
tags:
- lora
- distillation
- intent-classification
- banking77
---

# {base.split('/')[-1]} LoRA for Banking77 intent classification

LoRA adapter for `{base}`, trained on {meta['n_train']:,} Banking77 training messages with {labels}.
Part of Mikan, a pipeline that distils a large model's behaviour on one narrow task into a small
model and tests non-inferiority with a pre-registered margin.

## Results (Banking77 test split, 3,080 items, ground truth)

| | Accuracy |
|---|---|
| This adapter | {pct(row['golden_accuracy'])}% |
| Teacher (Qwen3-30B-A3B-Instruct-2507-FP8, 10-shot) | {pct(teacher_acc)}% |

Gap (teacher − student): {pct(row['gap'])} points, 95% paired bootstrap CI [{pct(row['gap_low'])}, {pct(row['gap_high'])}].
Non-inferiority margin δ = 2 points: **{row['decision']}**. Valid-label rate {pct(row['validity'])}%.
The checkpoint (epoch {meta['best_epoch']} of {meta['epochs']}) was chosen on a held-out validation split; the test split was predicted once.

## Usage

The prompt is a short system message with no label list; thinking is off.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{user}/{MODELS[row['run']]}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}", torch_dtype="auto"),
                                  "{user}/{MODELS[row['run']]}")
messages = [{{"role": "system", "content": "{STUDENT_SYSTEM}"}},
            {{"role": "user", "content": "I still have not received my new card"}}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
```

## Training

LoRA r={meta['rank']}, alpha={2 * meta['rank']}, all linear projections; lr {meta['lr']}, cosine schedule, {meta['epochs']} epochs,
batch {meta['batch']}, fp16, 8-bit AdamW; loss on the completion only. Unsloth 2026.11.12 + TRL 1.19.0 on one Tesla T4
({meta['train_seconds'] / 60:.0f} min). `meta.json` has the per-epoch validation accuracy and loss log.

Banking77 (PolyAI, CC-BY-4.0): Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders".
"""


def dataset_card(rows: list[dict], teacher_acc: float) -> str:
    table = "\n".join(f"| {r['run']} | {r['targets']} | {r['n_train']} | {pct(r['golden_accuracy'])}% | {r['decision']} |" for r in rows)
    return f"""---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
source_datasets:
- PolyAI/banking77
tags:
- distillation
- banking77
---

# Mikan Banking77 distillation data

Splits, teacher labels and student predictions from Mikan's Banking77 experiments.

- `banking77/`: `train` (8,222), `val` (1,000, 10% per label), `fewshot` (770, ranked pool for the teacher prompt),
  `golden` (the official 3,080-item test split, ground truth). Test-set leaks and duplicates were removed from train.
- `teacher/`: predictions of Qwen3-30B-A3B-Instruct-2507-FP8 (10-shot, label-constrained decoding with vLLM) for
  train, val and golden. Golden accuracy {pct(teacher_acc)}%.
- `students/<run>/`: per-epoch validation and final golden predictions plus `meta.json` for every SFT run.
- `summary.json`: the sweep table below.

| Run | Labels | Train items | Golden accuracy | Non-inferiority (δ = 2) |
|---|---|---|---|---|
{table}

Derived from Banking77 (PolyAI, CC-BY-4.0): Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders".
"""


def stage(user: str) -> None:
    summary = json.loads(SUMMARY.read_text())
    rows = {r["run"]: r for r in summary["runs"]}
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for run, repo in MODELS.items():
        d = STAGE / repo
        d.mkdir(parents=True)
        for f in ADAPTER_FILES:
            shutil.copy(RUNS / run / "best_adapter" / f, d / f)
        shutil.copy(RUNS / run / "meta.json", d / "meta.json")
        (d / "README.md").write_text(model_card(user, rows[run], summary["teacher_accuracy"]))

    d = STAGE / DATASET
    shutil.copytree(REPO / "data" / "banking77", d / "banking77")
    (d / "teacher").mkdir()
    for split in ("train", "val", "golden"):
        for kind in ("preds.jsonl", "meta.json"):
            shutil.copy(TEACHER / f"{split}_{kind}", d / "teacher" / f"{split}_{kind}")
    for run in rows:
        out = d / "students" / run
        out.mkdir(parents=True)
        for f in (RUNS / run).glob("*.json*"):
            shutil.copy(f, out / f.name)
    shutil.copy(SUMMARY, d / "summary.json")
    (d / "README.md").write_text(dataset_card(summary["runs"], summary["teacher_accuracy"]))
    print(f"staged {len(MODELS)} models and 1 dataset in {STAGE}/")


MASSIVE_DATASET = "mikan-massive-distill"
MASSIVE_TEACHER = REPO / "runs" / "massive-teacher-knn8"
MASSIVE_PILOTS = {  # pilot name -> run dir (same 300 val items each)
    "1shot": REPO / "runs/kaggle/mikan-teacher-massive-pilot/teacher/massive-teacher-1shot",
    "3shot": REPO / "runs/kaggle/mikan-teacher-massive-pilot/teacher/massive-teacher-3shot",
    "knn8": REPO / "runs/kaggle/mikan-teacher-massive-knn/teacher/massive-teacher-knn8",
    "knn20": REPO / "runs/kaggle/mikan-teacher-massive-knn/teacher/massive-teacher-knn20",
}


def massive_card(teacher: dict[str, float], pilots: dict[str, float]) -> str:
    rows = "\n".join(f"| {k} | {pct(v)}% |" for k, v in pilots.items())
    return f"""---
license: cc-by-4.0
task_categories:
- token-classification
- text-classification
language:
- en
source_datasets:
- AmazonScience/massive
tags:
- distillation
- semantic-parsing
- slot-filling
---

# Mikan MASSIVE en-US distillation data

Splits, retrieval neighbours and teacher frames from Mikan's structured-output task: parse a voice-assistant
request into `{{"intent", "slots": [{{"type", "value"}}]}}` (60 intents, 55 slot types).

- `massive/`: `train` (11,255), `val` (2,026, the official dev split, shuffled), `golden` (2,974, the official
  test split, ground truth), `fewshot` (181, every slot type covered), `intents.txt`, `slot_types.txt`,
  `retrieved.jsonl` (32 nearest labelled train/fewshot items per train/val/golden item by TF-IDF; an item never
  retrieves itself and val/golden are never in the pool).
- `teacher/`: frames from `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` (vLLM 0.13, 2x T4, fp16, decoding
  constrained to the frame JSON schema) prompted with the 8 retrieved examples per item. Exact frame match vs
  ground truth: train {pct(teacher['train'])}%, val {pct(teacher['val'])}%, golden {pct(teacher['golden'])}%. 100% valid frames.
- `pilots/`: the prompt pilot on the same 300 val items, kept as the record of how the prompt was chosen:

| Prompt | Exact frame match |
|---|---|
{rows}

Derived from MASSIVE v1.1 (Amazon, CC-BY-4.0): FitzGerald et al., 2022, "MASSIVE: A 1M-Example Multilingual
Natural Language Understanding Dataset with 51 Typologically-Diverse Languages". Annotator ids were dropped.
"""


def stage_massive() -> None:
    d = STAGE / MASSIVE_DATASET
    if d.exists():
        shutil.rmtree(d)
    (d / "massive").mkdir(parents=True)
    for f in sorted((REPO / "data" / "massive").iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".txt", ".json"):
            shutil.copy(f, d / "massive" / f.name)
    (d / "teacher").mkdir()
    teacher = {}
    for split in ("train", "val", "golden"):
        for kind in ("preds.jsonl", "meta.json"):
            shutil.copy(MASSIVE_TEACHER / f"{split}_{kind}", d / "teacher" / f"{split}_{kind}")
        teacher[split] = json.loads((MASSIVE_TEACHER / f"{split}_meta.json").read_text())["accuracy_vs_gold"]
    pilots = {}
    for name, run in MASSIVE_PILOTS.items():
        out = d / "pilots" / name
        out.mkdir(parents=True)
        for kind in ("preds.jsonl", "meta.json"):
            shutil.copy(run / f"pilot_{kind}", out / f"pilot_{kind}")
        pilots[name] = json.loads((run / "pilot_meta.json").read_text())["accuracy_vs_gold"]
    (d / "README.md").write_text(massive_card(teacher, pilots))
    print(f"staged {d}/")


def upload(user: str, task: str = "banking77") -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    if task == "massive":
        targets = [(MASSIVE_DATASET, "dataset")]
    else:
        targets = [(repo, "model") for repo in MODELS.values()] + [(DATASET, "dataset")]
    for repo, kind in targets:
        rid = f"{user}/{repo}"
        api.create_repo(rid, repo_type=kind, private=True, exist_ok=True)
        if not api.repo_info(rid, repo_type=kind).private:
            raise SystemExit(f"{rid} exists and is public; not uploading")
        api.upload_folder(repo_id=rid, repo_type=kind, folder_path=STAGE / repo, commit_message=f"Upload Mikan {task} artifacts")
        print(f"uploaded https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{rid} (private)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Back up Mikan artifacts to the HF Hub (private)")
    p.add_argument("cmd", choices=["stage", "upload"])
    p.add_argument("--user", help="HF username (default: the logged-in account)")
    p.add_argument("--task", choices=["banking77", "massive"], default="banking77")
    args = p.parse_args(argv)
    if args.user:
        user = args.user
    else:
        from huggingface_hub import whoami
        user = whoami()["name"]
    if args.cmd == "upload":
        upload(user, args.task)
    elif args.task == "massive":
        stage_massive()
    else:
        stage(user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
