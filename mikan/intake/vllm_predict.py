"""Batch prediction with vLLM. Runs wherever vLLM is installed (Modal, Colab, Kaggle).

Remote use: imported by ``modal_predict.py``.
Local use (e.g. a Colab T4):

    python -m mikan.intake.vllm_predict --model Qwen/Qwen3-1.7B \
        --data data/banking77 --split val --shots 0 --out runs/qwen3-1.7b-0shot
"""

from __future__ import annotations

import time


def predict(
    model: str,
    system: str,
    texts: list[str],
    choices: list[str] | None,
    *,
    max_model_len: int = 16384,
    max_tokens: int = 24,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = True,
    dtype: str = "auto",
) -> dict:
    """Greedy chat completion for every text. ``choices`` constrains output to those strings."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    t_load = time.time()
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,  # skip graph capture/compile: this is a short batch job
        enable_prefix_caching=True,
        dtype=dtype,
    )
    load_seconds = time.time() - t_load

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(choice=choices) if choices else None,
    )
    convs = [[{"role": "system", "content": system}, {"role": "user", "content": t}] for t in texts]
    t_gen = time.time()
    outs = llm.chat(
        convs,
        params,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False},  # Qwen3 hybrid models: no <think> block
    )
    gen_seconds = time.time() - t_gen

    return {
        "preds": [o.outputs[0].text for o in outs],
        "load_seconds": load_seconds,
        "gen_seconds": gen_seconds,
        "prompt_tokens": sum(len(o.prompt_token_ids) for o in outs),
        "output_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from ..prompts import prompt_sha256, select_shots, system_prompt
    from .runs import load_split, read_jsonl, write_run

    p = argparse.ArgumentParser(description="Batch predictions with a local vLLM install")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/banking77")
    p.add_argument("--split", default="val", help="val, golden, train, smoke (50 val), pilot (300 val)")
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, default=1, help="examples per label in the prompt (0 = zero-shot)")
    p.add_argument("--free", action="store_true", help="unconstrained output (measures format validity)")
    p.add_argument("--dtype", default="auto", help="use float16 on T4 GPUs")
    args = p.parse_args(argv)

    data = Path(args.data)
    labels = [ln.strip() for ln in (data / "labels.txt").read_text().splitlines() if ln.strip()]
    system = system_prompt(labels, select_shots(read_jsonl(data / "fewshot.jsonl"), args.shots))
    rows = load_split(data, args.split)
    result = predict(args.model, system, [r["text"] for r in rows], None if args.free else labels, dtype=args.dtype)
    meta = {
        "model": args.model, "split": args.split, "shots": args.shots,
        "constrained": not args.free, "prompt_sha256": prompt_sha256(system),
        **{k: v for k, v in result.items() if k != "preds"},
    }
    acc = write_run(Path(args.out), args.split, rows, result["preds"], meta)
    print(json.dumps({"split": args.split, "accuracy": acc, **{k: meta[k] for k in ("gen_seconds", "prompt_tokens")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
