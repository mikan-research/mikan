"""GRPO on a frames task, starting from an SFT adapter (Unsloth + TRL, one T4).

    python -m mikan.train.grpo --data <bundle>/data --task massive --teacher-dir massive-teacher \
        --model Qwen/Qwen3-1.7B --adapter <sft run>/best_adapter --out runs/grpo-qwen3-1.7b

The reward is ``mikan.frames.reward`` against the **teacher's** frame for each prompt
(golden is never used in training): 0 for an invalid frame, otherwise
0.1 + 0.3 x intent match + 0.6 x slot F1 x groundedness. Validity and groundedness are
also logged as zero-weight reward functions so reward hacking is visible in the logs.

The SFT LoRA weights are loaded and training continues on them. There is no KL term
(beta = 0, TRL's default): with a LoRA model TRL's reference would be the base model
without the adapter, not the SFT checkpoint, so a KL would pull toward the untrained
model. Drift is limited by the low learning rate and step count instead, and the
checkpoint is chosen on val **including step 0** (the SFT adapter itself), so GRPO is
only kept if it improves val exact match. Golden is predicted once, at the end.

``--prompt-offset`` takes prompts from the nested shuffle after the first N items, so a
student SFT-trained on the first N teacher labels gets GRPO prompts it has never seen.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ..frames import grounded, parse
from ..frames import reward as frame_reward
from ..intake.runs import write_run
from .examples import load_task, seeded_order, student_messages
from .gen import greedy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO from an SFT adapter on a frames task")
    p.add_argument("--data", required=True)
    p.add_argument("--task", default="massive")
    p.add_argument("--teacher-dir", default="massive-teacher")
    p.add_argument("--targets", choices=["teacher", "gold"], default="teacher",
                   help="reward reference; gold only for pipeline smoke tests and the supervised-RL ablation")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True, help="SFT best_adapter dir (same rank)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--n-prompts", type=int, default=2000)
    p.add_argument("--prompt-offset", type=int, default=0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion", type=int, default=96)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def make_rewards(intents: set[str], slot_types: set[str]):
    def _text(c) -> str:
        return c if isinstance(c, str) else c[0]["content"]

    def frame(completions, ref, text, **kw) -> list[float]:
        return [frame_reward(_text(c).strip(), json.loads(r), t, intents, slot_types) for c, r, t in zip(completions, ref, text)]

    def valid(completions, **kw) -> list[float]:
        return [float(parse(_text(c).strip(), intents, slot_types) is not None) for c in completions]

    def grounded_rate(completions, text, **kw) -> list[float]:
        out = []
        for c, t in zip(completions, text):
            f = parse(_text(c).strip(), intents, slot_types)
            out.append(grounded(f, t) if f is not None else 0.0)
        return out

    return [frame, valid, grounded_rate]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    from unsloth import FastLanguageModel  # noqa: I001  (must precede transformers / trl)
    import torch
    from datasets import Dataset
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    task = load_task(data, args.task, args.targets, args.teacher_dir)
    if task["kind"] != "frames":
        raise SystemExit("GRPO needs a frames task (a programmatic reward)")
    tdir = data / args.task
    intents = set((tdir / "intents.txt").read_text().split())
    types = set((tdir / "slot_types.txt").read_text().split())
    order = seeded_order(task["train"], args.seed)  # same order SFT's --n-train prefix uses
    prompts_rows = order[args.prompt_offset : args.prompt_offset + args.n_prompts]
    val, golden = task["val"], task["golden"]
    if args.eval_limit:
        val, golden = val[: args.eval_limit], golden[: args.eval_limit]

    model, tok = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_seq,
                                                   load_in_4bit=False, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    res = set_peft_model_state_dict(model, load_file(str(Path(args.adapter) / "adapter_model.safetensors")))
    if getattr(res, "unexpected_keys", None):
        raise SystemExit(f"adapter does not match the model: {res.unexpected_keys[:3]}")

    def prompt_text(text: str) -> str:
        return tok.apply_chat_template(student_messages(text, "frames"), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def predict(rows: list[dict]) -> list[str]:
        return greedy(model, tok, [prompt_text(r["text"]) for r in rows], args.eval_batch, args.max_completion,
                      first_line=False)

    dataset = Dataset.from_list([{"prompt": prompt_text(r["text"]), "ref": r["target"], "text": r["text"]}
                                 for r in prompts_rows])
    meta_base = {
        "model": args.model, "task": args.task, "adapter": str(args.adapter), "n_prompts": len(prompts_rows),
        "prompt_offset": args.prompt_offset, "steps": args.steps, "num_generations": args.num_generations,
        "prompts_per_step": args.prompts_per_step, "targets": args.targets, "lr": args.lr, "temperature": args.temperature, "beta": 0.0,
        "rank": args.rank, "seed": args.seed, "eval_limit": args.eval_limit, "gpu": torch.cuda.get_device_name(0),
    }

    history: list[dict] = []

    def evaluate(step: int) -> None:
        t0 = time.time()
        score = write_run(out, f"val_step{step}", val, predict(val), {**meta_base, "step": step})
        best = max((h["val_exact_match"] for h in history), default=-1.0)
        if score > best:
            model.save_pretrained(out / "best_adapter")
            tok.save_pretrained(out / "best_adapter")
        history.append({"step": step, "val_exact_match": score, "eval_seconds": time.time() - t0})
        print(f"[step {step}] val exact match {score:.4f}", flush=True)

    class StepEval(TrainerCallback):
        def on_step_end(self, targs, state, control, **kw):
            if state.global_step % args.eval_every == 0 or state.global_step == args.steps:
                evaluate(state.global_step)

    evaluate(0)  # the SFT starting point is a candidate checkpoint
    n_batch = args.num_generations * args.prompts_per_step
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=make_rewards(intents, types),
        train_dataset=dataset,
        callbacks=[StepEval()],
        args=GRPOConfig(
            output_dir=str(out / "trainer"),
            per_device_train_batch_size=n_batch,
            gradient_accumulation_steps=1,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion,
            temperature=args.temperature,
            reward_weights=[1.0, 0.0, 0.0],
            beta=0.0,
            learning_rate=args.lr,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=10,
            optim="adamw_8bit",
            max_grad_norm=0.1,
            max_steps=args.steps,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            use_vllm=False,
            log_completions=False,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            chat_template_kwargs={"enable_thinking": False},
        ),
    )
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    best = max(history, key=lambda h: (h["val_exact_match"], -h["step"]))
    set_peft_model_state_dict(model, load_file(str(out / "best_adapter" / "adapter_model.safetensors")))
    for suffix in ("preds.jsonl", "meta.json"):
        shutil.copy(out / f"val_step{best['step']}_{suffix}", out / f"val_{suffix}")
    t_gold = time.time()
    write_run(out, "golden", golden, predict(golden), {**meta_base, "step": best["step"]})
    shutil.rmtree(out / "trainer", ignore_errors=True)
    logs = trainer.state.log_history
    summary = {
        **meta_base, "kind": "frames", "targets": args.targets, "n_train": len(prompts_rows),
        "train_seconds": train_seconds, "golden_seconds": time.time() - t_gold,
        "history": [{"epoch": h["step"], "val_accuracy": h["val_exact_match"], **h} for h in history],
        "best_epoch": best["step"], "best_step": best["step"], "selected_by": "val_exact_match (step 0 = SFT)",
        "reward_log": [{k: v for k, v in x.items() if k.startswith(("reward", "rewards/", "completions/", "step", "loss"))}
                       for x in logs if any(k.startswith("reward") for k in x)],
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "best_step": best["step"], "val_exact_match": best["val_exact_match"],
                      "train_seconds": round(train_seconds)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
