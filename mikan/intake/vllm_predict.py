"""Batch prediction with vLLM. Runs wherever vLLM is installed (Modal, Colab, Kaggle).

Remote use: imported by ``modal_predict.py``.
Local use (e.g. a Colab or Kaggle T4):

    python -m mikan.intake.vllm_predict --model Qwen/Qwen3-1.7B \
        --data data/banking77 --splits val --shots 0 --out runs/qwen3-1.7b-0shot
    python -m mikan.intake.vllm_predict --model <4-bit teacher> --data data/massive \
        --splits pilot --shots 1 --tp 2 --dtype float16 --out runs/massive-teacher-1shot

All splits share one model load. A data dir with ``slot_types.txt`` is a frames task.
"""

from __future__ import annotations

import time


def predict(
    model: str,
    system: str | list[str],
    texts: list[str],
    choices: list[str] | None,
    *,
    json_schema: dict | None = None,
    max_model_len: int = 16384,
    max_tokens: int = 24,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = True,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    quantization: str | None = None,
    attention_backend: str | None = None,
) -> dict:
    """Greedy chat completion for every text.

    ``choices`` constrains output to those strings; ``json_schema`` constrains it to JSON
    matching the schema (structured-output tasks). At most one may be given.
    """
    if choices and json_schema:
        raise ValueError("pass choices or json_schema, not both")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    extra = {"attention_config": {"backend": attention_backend}} if attention_backend else {}
    t_load = time.time()
    llm = LLM(
        **extra,
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,  # skip graph capture/compile: this is a short batch job
        enable_prefix_caching=True,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
    )
    load_seconds = time.time() - t_load

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=(StructuredOutputsParams(choice=choices) if choices
                            else StructuredOutputsParams(json=json_schema) if json_schema else None),
    )
    systems = [system] * len(texts) if isinstance(system, str) else system
    if len(systems) != len(texts):
        raise ValueError("one system prompt per text")
    convs = [[{"role": "system", "content": s}, {"role": "user", "content": t}] for s, t in zip(systems, texts)]
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

    from ..prompts import prompt_sha256, retrieved_systems, task_setup
    from .runs import load_split, write_run

    p = argparse.ArgumentParser(description="Batch predictions with a local vLLM install")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/banking77")
    p.add_argument("--splits", default="val", help="comma list of val, golden, train, smoke (50 val), pilot (300 val)")
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, default=1, help="examples per label/intent in the prompt (0 = zero-shot)")
    p.add_argument("--free", action="store_true", help="unconstrained output (measures format validity)")
    p.add_argument("--dtype", default="auto", help="use float16 on T4 GPUs")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size (2 on Kaggle T4 x2)")
    p.add_argument("--quantization", default=None)
    p.add_argument("--max-model-len", type=int, default=0, help="0 = fit the prompt")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--retrieved", type=int, default=0,
                   help="frames tasks: k nearest labelled examples per item from retrieved.jsonl (replaces --shots)")
    p.add_argument("--attention-backend", default=None, help="e.g. TRITON_ATTN on T4 (no JIT build)")
    args = p.parse_args(argv)

    data = Path(args.data)
    setup = task_setup(data, 0 if args.retrieved else args.shots, not args.free)
    names = [x.strip() for x in args.splits.split(",") if x.strip()]
    rows = {n: load_split(data, n) for n in names}
    texts = [r["text"] for n in names for r in rows[n]]
    if args.retrieved:
        if setup["kind"] != "frames":
            raise SystemExit("--retrieved is implemented for frames tasks")
        system = retrieved_systems(data, [r for n in names for r in rows[n]], args.retrieved)
    else:
        system = setup["system"]
    longest = max(system, key=len) if isinstance(system, list) else system
    max_len = args.max_model_len
    if not max_len:
        max_len = 4096
        while max_len < len(longest) // 3 + 1024:
            max_len *= 2
    result = predict(args.model, system, texts, setup["choices"], json_schema=setup["json_schema"],
                     max_tokens=setup["max_tokens"], max_model_len=max_len, dtype=args.dtype,
                     tensor_parallel_size=args.tp, quantization=args.quantization,
                     gpu_memory_utilization=args.gpu_mem, attention_backend=args.attention_backend)
    meta = {
        "model": args.model, "shots": 0 if args.retrieved else args.shots, "retrieved_k": args.retrieved,
        "constrained": not args.free, "kind": setup["kind"],
        "prompt_sha256": prompt_sha256("\n\x00".join(system) if isinstance(system, list) else system),
        "tensor_parallel_size": args.tp, "dtype": args.dtype,
        **{k: v for k, v in result.items() if k != "preds"},
    }
    start = 0
    for n in names:
        k = len(rows[n])
        score = write_run(Path(args.out), n, rows[n], result["preds"][start : start + k], meta)
        start += k
        print(json.dumps({"split": n, "n": k, "score_vs_gold": score}))
    print(json.dumps({k: meta[k] for k in ("load_seconds", "gen_seconds", "prompt_tokens", "output_tokens")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
