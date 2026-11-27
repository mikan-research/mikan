"""Run the teacher (L40S) or a student (L4) on Modal with vLLM.

Run from the repo root in module mode, so the local ``mikan`` package is found:

    uv run modal run -m mikan.intake.modal_predict --splits smoke
    uv run modal run -m mikan.intake.modal_predict --splits pilot --shots 5
    uv run modal run -m mikan.intake.modal_predict --splits train,val,golden
    uv run modal run -m mikan.intake.modal_predict --role student \
        --model Qwen/Qwen3-1.7B --splits val,golden --shots 0
    uv run modal run -m mikan.intake.modal_predict --data-dir data/massive --splits pilot --shots 1

A data dir with ``intents.txt`` and ``slot_types.txt`` is a frames task: the prompt lists both
and decoding is constrained to the frame JSON schema instead of a label choice.

All requested splits go to one remote call, so the model loads once. Outputs go to
``runs/<role>-<model>-<prompt>/{split}_preds.jsonl`` plus ``{split}_meta.json``.
Model weights are cached in the ``mikan-hf-cache`` Volume, so they download once.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

TEACHER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
GPU = {"teacher": "L40S", "student": "L4"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("mikan")
)
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name("mikan-hf-cache", create_if_missing=True),
    "/root/.cache/vllm": modal.Volume.from_name("mikan-vllm-cache", create_if_missing=True),
}
app = modal.App("mikan-predict", image=image)

_fn_kwargs = dict(volumes=volumes, timeout=60 * 60, scaledown_window=10, max_containers=1)


@app.function(cpu=4, memory=8192, volumes=volumes, timeout=60 * 60)
def download_model(model: str) -> float:
    """Fill the weights cache on a cheap CPU container, so GPUs never bill for downloads."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(model, ignore_patterns=["*.pt", "*.bin", "original/*"])
    volumes["/root/.cache/huggingface"].commit()
    return time.time() - t0


@app.function(gpu=GPU["teacher"], **_fn_kwargs)
def generate_teacher(model: str, system: str, texts: list[str], choices: list[str] | None, max_model_len: int,
                    json_schema: dict | None = None, max_tokens: int = 24) -> dict:
    from mikan.intake.vllm_predict import predict

    return predict(model, system, texts, choices, max_model_len=max_model_len, json_schema=json_schema,
                   max_tokens=max_tokens)


@app.function(gpu=GPU["student"], **_fn_kwargs)
def generate_student(model: str, system: str, texts: list[str], choices: list[str] | None, max_model_len: int,
                    json_schema: dict | None = None, max_tokens: int = 24) -> dict:
    from mikan.intake.vllm_predict import predict

    return predict(model, system, texts, choices, max_model_len=max_model_len, json_schema=json_schema,
                   max_tokens=max_tokens)


def context_length(system: str) -> int:
    """Smallest power of two (at least 4096) that fits the shared prompt plus a query and answer."""
    need = len(system) // 3 + 1024  # ~3 characters per token is a safe overestimate here
    n = 4096
    while n < need:
        n *= 2
    return n


@app.local_entrypoint()
def main(
    splits: str = "smoke",
    role: str = "teacher",
    model: str = "",
    shots: int = 1,
    constrained: bool = True,
    data_dir: str = "data/banking77",
    out_dir: str = "",
):
    from mikan.intake.runs import load_split, write_run
    from mikan.prompts import prompt_sha256, task_setup

    if role not in GPU:
        raise SystemExit(f"role must be one of {sorted(GPU)}")
    model = model or (TEACHER_MODEL if role == "teacher" else "")
    if not model:
        raise SystemExit("--model is required for students")

    data = Path(data_dir)
    setup = task_setup(data, shots, constrained)
    system, frames_task = setup["system"], setup["kind"] == "frames"

    split_names = [s.strip() for s in splits.split(",") if s.strip()]
    rows = {s: load_split(data, s) for s in split_names}
    texts = [r["text"] for s in split_names for r in rows[s]]
    print(f"{role} {model} on {GPU[role]}, {shots}-shot (prompt ~{len(system) // 4} tokens): {len(texts)} texts from {split_names}")

    print(f"  weights cache ready in {download_model.remote(model):.0f}s (CPU)")
    fn = generate_teacher if role == "teacher" else generate_student
    t0 = time.time()
    result = fn.remote(model, system, texts, setup["choices"], context_length(system),
                       json_schema=setup["json_schema"], max_tokens=setup["max_tokens"])
    wall = time.time() - t0

    task_tag = f"{data.name}-" if frames_task else ""  # Banking77 run names stay as they were
    tag = f"{task_tag}{role}-{model.split('/')[-1].lower()}-{shots}shot{'' if constrained else '-free'}"
    out = Path(out_dir or f"runs/{tag}")
    meta = {
        "role": role,
        "model": model,
        "gpu": GPU[role],
        "shots": shots,
        "constrained": constrained,
        "prompt_sha256": prompt_sha256(system),
        "remote_wall_seconds_all_splits": wall,
        **{k: v for k, v in result.items() if k != "preds"},
    }
    start = 0
    for s in split_names:
        n = len(rows[s])
        acc = write_run(out, s, rows[s], result["preds"][start : start + n], meta)
        start += n
        print(f"  {s:>7}: n={n:5d}  {'exact match' if frames_task else 'accuracy'} vs gold = {acc:.4f}")
    print(
        f"load {result['load_seconds']:.0f}s, generate {result['gen_seconds']:.0f}s, "
        f"remote wall {wall:.0f}s; prompt tokens {result['prompt_tokens']}, output tokens {result['output_tokens']}"
    )
    print(f"wrote {out}/")
