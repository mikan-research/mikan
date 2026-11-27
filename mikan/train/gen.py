"""Greedy batched generation with an Unsloth model, shared by SFT and GRPO evaluation."""

from __future__ import annotations

import gc


def postprocess(text: str, first_line: bool = True) -> str:
    """Label tasks keep the first line (one label); frames keep the whole output, so evaluation
    scores exactly what the GRPO reward parses (JSON may span lines)."""
    text = text.strip()
    return text.split("\n")[0].strip() if first_line else text


def greedy(model, tok, prompts: list[str], batch: int, max_new_tokens: int, first_line: bool = True) -> list[str]:
    """Left-padded greedy decoding; each completion goes through ``postprocess``."""
    import torch
    from unsloth import FastLanguageModel

    gc.collect(); torch.cuda.empty_cache()  # drop training activations before decode buffers
    FastLanguageModel.for_inference(model)
    side = tok.padding_side
    tok.padding_side = "left"
    preds: list[str] = []
    enc = gen = None
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i : i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            preds += [postprocess(t, first_line) for t in texts]
    tok.padding_side = side
    del enc, gen
    gc.collect(); torch.cuda.empty_cache()
    FastLanguageModel.for_training(model)
    return preds
