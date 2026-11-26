"""LoRA SFT of a student with Unsloth + TRL, on a single GPU (a Kaggle or Colab T4 is enough).

    python -m mikan.train.sft --data <bundle>/data --model Qwen/Qwen3-1.7B \
        --targets teacher --epochs 3 --out runs/sft-qwen3-1.7b-teacher-full

``--data`` must contain ``banking77/{labels.txt,train,val,golden}.jsonl`` and
``teacher/train_preds.jsonl``.
After every epoch the model predicts val; the epoch with the best **val** accuracy is
kept (its LoRA weights are saved), reloaded at the end, and only then used to predict
golden, once. Golden accuracy never influences the choice.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from ..intake.runs import write_run
from .examples import attach_targets, nested_subset, read_jsonl, student_messages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT of a student model")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--targets", choices=["teacher", "gold"], default="teacher")
    p.add_argument("--n-train", type=int, default=0, help="0 = all training rows")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--eval-limit", type=int, default=0, help="evaluate only the first N val/golden items (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    # Unsloth must be imported before transformers / trl to patch them.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    train_rows = read_jsonl(data / "banking77" / "train.jsonl")
    teacher = read_jsonl(data / "teacher" / "train_preds.jsonl") if args.targets == "teacher" else None
    train = nested_subset(attach_targets(train_rows, args.targets, teacher), args.n_train, args.seed)
    val = read_jsonl(data / "banking77" / "val.jsonl")
    golden = read_jsonl(data / "banking77" / "golden.jsonl")
    if args.eval_limit:
        val, golden = val[: args.eval_limit], golden[: args.eval_limit]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq, load_in_4bit=args.load_in_4bit, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    end_of_turn = "<|im_end|>"

    def prompt_text(text: str) -> str:
        # Same template vLLM applies at inference (Qwen3 with thinking disabled).
        return tok.apply_chat_template(
            student_messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    dataset = Dataset.from_list(
        [{"prompt": prompt_text(r["text"]), "completion": r["target"] + end_of_turn} for r in train]
    )

    @torch.no_grad()
    def predict(rows: list[dict]) -> list[str]:
        gc.collect(); torch.cuda.empty_cache()  # drop training activations before decode buffers
        FastLanguageModel.for_inference(model)
        side = tok.padding_side
        tok.padding_side = "left"
        preds: list[str] = []
        for i in range(0, len(rows), args.eval_batch):
            prompts = [prompt_text(r["text"]) for r in rows[i : i + args.eval_batch]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=24, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id,
            )
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            preds += [t.strip().split("\n")[0].strip() for t in texts]
        tok.padding_side = side
        del enc, gen
        gc.collect(); torch.cuda.empty_cache()
        FastLanguageModel.for_training(model)
        return preds

    history: list[dict] = []
    meta_base = {
        "model": args.model, "targets": args.targets, "n_train": len(train), "epochs": args.epochs,
        "lr": args.lr, "rank": args.rank, "batch": args.batch, "load_in_4bit": args.load_in_4bit,
        "seed": args.seed, "eval_limit": args.eval_limit, "gpu": torch.cuda.get_device_name(0),
    }

    class EpochEval(TrainerCallback):
        def on_epoch_end(self, targs, state, control, **kw):
            epoch = round(state.epoch)
            t0 = time.time()
            val_acc = write_run(out, f"val_epoch{epoch}", val, predict(val), {**meta_base, "epoch": epoch})
            best = max((h["val_accuracy"] for h in history), default=-1.0)
            if val_acc > best:
                model.save_pretrained(out / "best_adapter")
                tok.save_pretrained(out / "best_adapter")
            losses = [x["loss"] for x in state.log_history if "loss" in x]
            history.append({
                "epoch": epoch, "val_accuracy": val_acc, "eval_seconds": time.time() - t0,
                "train_loss_last": losses[-1] if losses else None,
            })
            print(f"[epoch {epoch}] val accuracy {val_acc:.4f}", flush=True)

    bf16 = is_bfloat16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=dataset,
        callbacks=[EpochEval()],
        args=SFTConfig(
            output_dir=str(out / "trainer"),
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=1,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            optim="adamw_8bit",
            bf16=bf16,
            fp16=not bf16,
            max_length=args.max_seq,
            completion_only_loss=True,
            packing=False,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            dataset_num_proc=1,
        ),
    )
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    best = max(history, key=lambda h: h["val_accuracy"])
    if best["epoch"] != history[-1]["epoch"]:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(out / "best_adapter" / "adapter_model.safetensors")))
    for suffix in ("preds.jsonl", "meta.json"):
        shutil.copy(out / f"val_epoch{best['epoch']}_{suffix}", out / f"val_{suffix}")
    t_gold = time.time()
    write_run(out, "golden", golden, predict(golden), {**meta_base, "epoch": best["epoch"]})
    golden_seconds = time.time() - t_gold
    shutil.rmtree(out / "trainer", ignore_errors=True)
    summary = {
        **meta_base, "train_seconds": train_seconds, "history": history, "best_epoch": best["epoch"],
        "selected_by": "val_accuracy", "golden_seconds": golden_seconds, "loss_log": [x for x in trainer.state.log_history if "loss" in x],
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "best_epoch": best["epoch"], "val_accuracy": best["val_accuracy"],
                      "train_seconds": round(train_seconds)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
