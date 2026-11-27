# Risks and open questions

## Technical risks

| Risk | Mitigation |
|---|---|
| Task does not distil well | Zero-shot baselines first; try a larger student (up to 4B) before giving up |
| Weak teacher makes the experiment meaningless | Measure teacher accuracy on the golden set first; bigger teacher, more shots or retrieved examples if low. Happened on MASSIVE: fixed prompts gave 48% exact match, retrieved examples 67% |
| Golden set too small to resolve the margin | Power check before training (Banking77's 3k test items should be enough) |
| Eval measures teacher agreement instead of correctness | Golden labels are dataset ground truth, never teacher outputs |
| Banking77 label noise blurs small differences | Read disagreement items; report the noise found |
| GRPO reward hacking (extra text, invented slot spans) | Strict parse (exact shape, schema), spans must be copied from the input, validity and groundedness logged; read sampled outputs; checkpoint chosen on val with the SFT start as a candidate |
| GRPO drifts from the SFT checkpoint | β = 0 because TRL's LoRA reference is the base model, so drift is limited by low lr, few steps and val-based selection instead of a KL term |
| Quantization quietly costs quality | Re-run the full eval on the quantized model |
| Leakage between train and test | Near-duplicate dedupe across splits |
| Overfitting model choice to the golden set | Select on the validation split; golden set only for the final report |
| Modal credit runs out ($30) | Happened in November 2026: the month's credit was used up by other work. Banking77 cost ~$1; the MASSIVE teacher moved to Kaggle T4 x2 (4-bit GPTQ, vLLM tensor parallel). Serving benchmark waits for the credit to reset |
| vLLM kernels on T4 | Compressed-tensors (Marlin) 4-bit builds need sm80; FlashInfer JIT fails to link on Kaggle. Use GPTQ builds and Triton attention |
| Free-tier GPU sessions time out mid-run | Frequent checkpoints to the Hub or Kaggle output; resumable scripts |
| T4 limits (16GB, fp16 only, no FP8) | Students up to about 4B, LoRA/4-bit via Unsloth, AWQ/GPTQ instead of FP8 |
| Real logs contain personal data (Phase 5) | PII scrub before storage, isolated storage, delete when done |

## Resolved

- First task: Banking77 ([setup.md](setup.md))
- Teacher: Qwen3-30B-A3B-Instruct (FP8, L40S) on Modal; fallback Qwen3-32B
- Students: Qwen3 0.6B / 1.7B / 4B, trained on free T4s
- Margin: δ = 2 accuracy points, pre-registered (Banking77); δ = 3 exact-match points for MASSIVE, chosen on power grounds before any output existed
- Second task for GRPO: MASSIVE en-US frames, since Banking77 has no code-checkable reward without ground truth
- MASSIVE teacher: `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` on Kaggle with 8 retrieved examples per item (20 was not significantly better and 2.3x slower)
- Compute: Modal ($30) for teacher and serving benchmark only; Kaggle and Colab T4s for training
- Stack: TRL + Unsloth
- Publishing: code on GitHub at [`mikan-research/mikan`](https://github.com/mikan-research/mikan) (Apache-2.0); models and data in the Hugging Face organization [`mikan-research`](https://huggingface.co/mikan-research) (adapters Apache-2.0 like Qwen3, datasets CC-BY-4.0 like their sources)

## Open questions

1. Is the $30 Modal credit monthly? It decides when the Phase 4 serving benchmark can run.
2. Serving with constrained decoding should remove the students' remaining invalid outputs (invented intents and slot types); confirm in Phase 4 that it does not cost exact match.
3. Licences of Banking77, MASSIVE and the Qwen3 checkpoints, re-checked on their cards before anything is made public.

Settled: Banking77's observed disagreement (≤13%) kept δ = 2 powered; Phase 2 students disagreed with the teacher on ~6%. MASSIVE disagreement was 20–22% for passing students, within what δ = 3 was chosen for. GRPO's value: +0.7 points after a full SFT (borderline), +4.4 after a 1k SFT.
