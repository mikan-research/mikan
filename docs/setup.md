# Setup: task, teacher, compute, stack

Decisions and results for the two tasks so far: Banking77 (Phases 1–2) and MASSIVE en-US frames (Phase 3).

## Task: Banking77 intent classification

- About 13k online-banking customer queries labelled with 77 intents (about 10k train, about 3k test).
- Output: exactly one of 77 label strings, so scoring is a string match. Reward for GRPO is just as simple.
- **Golden set = the official test split** (about 3,080 items, dataset ground truth).
- **Pre-registered margin: δ = 2 accuracy points.** Primary metric: accuracy; macro-F1 reported alongside. With about 3k paired items and roughly 8% disagreement between models, the 95% CI on the difference is about ±1 point, so δ = 2 is resolvable. Confirm with the power check in Phase 0.
- Validation split carved from train: 10% per label (1,000 items), cut **before** the few-shot pool, so it does not change with the number of shots.
- Few-shot pool: 10 ranked examples per label (770), removed from train. Train: 8,222 items.
- Leakage: 7 train items whose normalised text also appears in the test set were dropped, plus 4 exact duplicates.
- Prepared by `uv run python -m mikan.data.banking77 --out data/banking77 --fewshot-per-label 10` (writes `manifest.json` with source and file hashes).
- Licence: listed as CC-BY-4.0 on the dataset card (confirm before publishing results).
- Known issue: some label noise is reported in the literature. Read the teacher/student disagreements before trusting small differences.

## Teacher: Qwen3-30B-A3B-Instruct on Modal

Chosen to stay well inside the $30 Modal credit:

- **Model:** Qwen3-30B-A3B-Instruct (the non-thinking instruct release), Apache-2.0. Mixture-of-experts with about 3B active parameters per token, so it generates fast and cheaply.
- **Why not a thinking model or gpt-oss:** they spend hidden reasoning tokens on every call, which multiplies cost for a one-label answer.
- **Hardware:** FP8 weights (about 31GB) on one Modal L40S 48GB, a cheaper GPU class than A100/H100. If FP8 is not available, use BF16 on one A100 80GB.
- **Serving:** vLLM offline batch mode with **prefix caching** (the system prompt, 77 labels and few-shot examples are the same in every prompt, so they are computed once) and **guided choice decoding**, so every output is one of the 77 labels.
- **Prompt:** system prompt listing the 77 labels plus **10 examples per label** (about 20k tokens, cached once by vLLM), "answer with the label only". Built by `mikan/prompts.py`.
- **Steps:**
  1. Smoke test on 50 items (cost: cents).
  2. Pilot on 300 validation items: measure teacher accuracy against ground truth.
  3. If accuracy is good, run once on all of train + validation + test and cache to disk. The teacher is never called again.
- **Fallback:** if the pilot accuracy is weak, switch to the dense Qwen3-32B (BF16, one A100 80GB) or add more shots. It is still expected to stay under about $10.
- Same family as the students (Qwen3), so a shared tokenizer allows optional logit-level distillation later (vLLM can return top log-probs over the label choices).

### Teacher pilot results (300 validation items, same items for every row)

| Prompt | Accuracy | Prompt tokens (cached) |
|---|---|---|
| 1 example per label | 75.0% | about 2.3k |
| 5 examples per label | 82.7% | about 10k |
| **10 examples per label (chosen)** | **84.0%** | about 20k |

10-shot vs 5-shot is not significant on 300 items (McNemar p = 0.42, CI −1.0 to +3.7 points); 10-shot was chosen for the higher point estimate at a small extra cost. Most errors are between near-synonym labels where Banking77 follows its own conventions (for example `card_payment_fee_charged` vs `extra_charge_on_statement`), plus some label noise.

The teacher is clearly weaker than a model trained on ground truth (low 90s in the literature). That makes the Phase 2 ablation (teacher labels vs ground-truth labels) the most informative result of the project.

Run: `uv run modal run -m mikan.intake.modal_predict --splits train,val,golden --shots 10`. Smoke test, three pilots and the weights download cost under $1 in total.

### Student baselines (golden set, before any training)

| Student | 0-shot | 10-shot | Gap to teacher at 10-shot (95% CI) | Disagreement with teacher (10-shot) |
|---|---|---|---|---|
| Qwen3-0.6B | 36.0% | 50.6% | 35.0 points [33.2, 36.8] | 38.3% |
| Qwen3-1.7B | 48.6% | 65.9% | 19.7 points [18.1, 21.3] | 25.7% |
| Qwen3-4B | 63.4% | 79.1% | 6.5 points [5.3, 7.8] | 13.2% |
| *Teacher (Qwen3-30B-A3B, 10-shot)* | | *85.6%* | | |

Prompt examples add 15 to 17 points at every size but make inference about 15 times slower (20k-token prompt). A trained student should need no examples in its prompt, which is part of the serving-cost win.

The 4B model at 10-shot already sits 6.5 points from the teacher with only 13% disagreement, so it is the most likely to pass after SFT.

## Second task: MASSIVE en-US frames (Phase 3)

Banking77 has no reward a program can check without ground truth, so GRPO uses a structured-output task.

- **Data:** Amazon MASSIVE v1.1, en-US only (CC-BY-4.0), from the official release tarball. 60 intents, 55 slot types.
  Output is a JSON frame `{"intent": ..., "slots": [{"type": ..., "value": ...}]}`; slots are a list because a type can
  repeat (505 utterances). A third of utterances have no slots. Annotator ids are dropped.
- **Splits** (`uv run python -m mikan.data.massive --out data/massive`): train 11,255; val 2,026 (official dev,
  shuffled so `val[:300]` is a representative pilot); golden 2,974 (official test, ground truth); few-shot pool 181
  (every slot type covered at rank 0, then 3 per intent). Leaks into test and duplicates removed (22 + 45 from train,
  11 train items overlapping val, 5 + 2 from val).
- Every gold slot value occurs verbatim in its utterance (groundedness 100%), so "spans must be copied from the input"
  is a valid check and reward term.
- **Pre-registered** in `tasks/massive.yaml`: primary metric exact frame match, **δ = 3 points**. With 2,974 items the
  smallest margin resolvable at 80% power is 1.6 / 2.0 / 2.3 / 2.8 points at 10 / 15 / 20 / 30% disagreement; exact
  match is noisier than intent accuracy, so 3 points keeps the test powered up to about 30% disagreement.

### MASSIVE teacher: Qwen3-30B-A3B on Kaggle T4 x2

Modal's credit for the month was already used up by other work, so the teacher ran on free Kaggle GPUs instead.

- **Model:** `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` (same model as the Banking77 teacher, 4-bit GPTQ,
  ~16 GB), vLLM 0.13, tensor parallel over the two T4s, fp16, Triton attention, decoding constrained to the frame
  JSON schema (100% valid output).
- **What failed on T4:** compressed-tensors AWQ builds (their Marlin kernels need sm80+: "no kernel image is
  available"), and vLLM's default FlashInfer attention (its JIT build cannot link `-lcuda` on Kaggle).
- **Runner:** `mikan/intake/kaggle_teacher.py` tries candidate models in order on 50 items, each in its own process,
  then runs the configured splits with the first that works.

**Prompt pilot** (300 val items, the same items for every row):

| Prompt | Exact match | Intent acc | Slot F1 | Seconds per item |
|---|---|---|---|---|
| 1 example per intent + slot-type coverage (73) | 44.7% | 81.3% | 54.4% | 0.57 |
| 3 per intent + coverage (181) | 48.0% | 79.7% | 61.3% | 0.88 |
| **8 retrieved examples (chosen)** | **67.0%** | 88.7% | 76.2% | 1.45 |
| 20 retrieved examples | 68.0% | 89.7% | 78.6% | 3.37 |

- Fixed-prompt errors were mostly MASSIVE annotation conventions: extra slots ("weather in sydney **now**" tagged as
  time), span edges (`first appointment` vs `appointment`), slot type (`general_frequency` vs `date`).
- Retrieved examples (`uv run python -m mikan.data.retrieve --data data/massive`): the 32 nearest labelled train /
  few-shot items per item by TF-IDF over word 1–2 grams; never the item itself; val and golden are never in the pool.
  The prompt is the shared instruction and lists, then the item's k nearest examples, most similar last.
- 8 vs 3-shot: 78 vs 21 items right only with one prompt (McNemar p = 7e-9). 20 vs 8: 15 vs 12 (p = 0.7), at 2.3x
  the time, and 20 would not fit Kaggle's 12 h session.

**Full run** (two parallel Kaggle kernels, `configs/teacher_massive_full_part{1,2}.json`, about 3.2 h each), merged
into `runs/massive-teacher-knn8/`:

| Split | Items | Exact match | Intent acc | Slot F1 | Valid |
|---|---|---|---|---|---|
| train (SFT targets, GRPO reference) | 11,255 | 63.7% | 88.5% | 74.2% | 100% |
| val | 2,026 | 64.9% | | | 100% |
| golden | 2,974 | **64.4%** | | | 99.97% (1 output cut off at the token limit) |

### MASSIVE students (Phase 3 results)

| Student | Training | Golden exact match | Gap vs teacher [95% CI] | Valid | Decision |
|---|---|---|---|---|---|
| Qwen3-0.6B | SFT, 11,255 teacher frames | 64.1% | +0.3 [−1.3, +1.9] | 99.7% | PASS |
| Qwen3-0.6B | + GRPO (best at step 200) | 64.8% | −0.4 [−2.1, +1.2] | 99.6% | PASS |
| Qwen3-1.7B | SFT, 11,255 teacher frames | 65.0% | −0.6 [−2.2, +1.0] | 99.8% | PASS |
| Qwen3-1.7B | + GRPO (best at step 200) | **65.7%** | −1.3 [−2.9, +0.3] | 99.7% | PASS |
| Qwen3-1.7B | SFT, 1,000 teacher frames | 46.0% | +18.4 [+16.5, +20.3] | 95.3% | FAIL |
| Qwen3-1.7B | + GRPO on 3,000 unseen prompts | 50.4% | +14.0 [+12.0, +15.9] | 97.2% | FAIL |
| Qwen3-1.7B | SFT, 11,255 ground-truth frames | 72.1% | −7.8 [−9.4, −6.1] | 99.8% | PASS |

| Pair (same golden items) | GRPO − SFT [95% CI] | Right only after / only before | McNemar p |
|---|---|---|---|
| 0.6B, full SFT | +0.7 [+0.0, +1.5] | 71 / 49 | 0.055 |
| 1.7B, full SFT | +0.7 [−0.0, +1.5] | 72 / 51 | 0.071 |
| 1.7B, 1k SFT | **+4.4 [+3.1, +5.8]** | 282 / 150 | 2e-10 |

### MASSIVE student setup

Short student prompt with no intent or slot lists ("Parse the voice-assistant request into a JSON frame..."). Two
Kaggle kernels, `configs/massive_phase3_{a,b}.json`: SFT 1.7B on teacher labels (full) then GRPO; SFT 1.7B on
ground truth (ceiling); SFT 0.6B on teacher labels then GRPO; SFT 1.7B on 1k teacher labels then GRPO on 3k prompts
it has not seen. GRPO: 250 steps, 8 samples per prompt, 4 prompts per step, lr 5e-6, temperature 1.0, β = 0, val
checked every 50 steps with step 0 as a candidate.

## Compute

| Resource | GPU | Used for |
|---|---|---|
| Modal ($30 credit) | L40S (teacher), L4 (student serving) | Teacher label generation; final vLLM serving benchmark |
| Kaggle (free, about 30 GPU-hours/week, check current quota) | T4 16GB (or P100) | Main SFT and GRPO runs (longer sessions) |
| Colab (free) | T4 16GB | Quick experiments, baselines, debugging |

**Modal budget plan (check current per-second GPU rates before running):**

| Item | Rough share |
|---|---|
| Teacher smoke test + 300-item pilot | under $1 to $2 |
| Teacher full run (about 13k prompts, mostly model load time) | about $2 to $5 |
| Serving benchmark: student on L4, teacher on L40S | about $2 to $5 |
| Reserve for re-runs / mistakes | the rest |

Students never use Modal for training, so they cost no credit until the serving benchmark.

**Actual spend:** Modal ~$1 for all of Banking77 (teacher pilots, full run, student baselines). The month's credit was otherwise used up, so MASSIVE ran entirely on Kaggle. Kaggle use so far: about 11 GPU-hours (Phase 2 ~2.5 h, teacher pilots ~1.5 h, full teacher run 2 x 3.2 h, smoke tests ~1 h) of the ~30 h weekly quota.

Cost rules: test the teacher script on 50 items first, cache model weights in a Modal Volume (download once), set a Modal spending limit, and shut containers down at the end (no idle warm containers).

**T4 constraints:** fp16 only (no bf16), no FlashAttention 2, no FP8. Free sessions time out, so save checkpoints often (to Kaggle output or a Hugging Face Hub private repo) and make every training script resumable.

## Student models: Qwen3 0.6B / 1.7B / 4B

- Apache-2.0, one family across three sizes: a clean size ablation with nothing else changing.
- Same family as the teacher (shared tokenizer and chat template).
- Trained only on free Kaggle/Colab T4s, so they use no Modal credit.
- 0.6B and 1.7B: LoRA SFT and GRPO fit comfortably on a T4. 4B: 4-bit QLoRA via Unsloth; tight on a T4, so use small batch size and gradient accumulation.
- Order: start with 1.7B; go down to 0.6B if it already passes, up to 4B if it does not.

## Training stack: TRL + Unsloth

- **Unsloth** for loading 4-bit / LoRA models on one T4 with low VRAM.
- **TRL `SFTTrainer`** for distillation: prompt-completion pairs built with the chat template (thinking off), loss on the completion only; LoRA r=16, alpha=32 on all linear projections, lr 2e-4 cosine, 3 epochs, fp16, 8-bit AdamW. Pinned: unsloth 2026.11.12, TRL 1.19.0.
- Runs on Kaggle T4 x2 via `mikan.train.kaggle_job` (private dataset + private kernel); one run per GPU.
- **TRL `GRPOTrainer`** for RL (`mikan.train.grpo`), continuing the SFT LoRA adapter. Rollouts use Unsloth's HF generation, not vLLM, to keep a second inference engine out of the training process on a T4; frames are short (under 100 tokens), so generation is not the bottleneck it is for long reasoning traces. Untested alternative: `use_vllm=True`.
- Why not verl: it is built for multi-GPU clusters; nothing here is multi-GPU.
- Experiment tracking: Weights & Biases free tier or plain CSV logs committed with each run config.

## Quantization and serving

- AWQ or GPTQ 4-bit (both work on T4; FP8 does not).
- Serving benchmark on Modal with vLLM, OpenAI-compatible endpoint: the quantized student on an L4 (the cheapest GPU that fits it). Measure p50/p95 latency, throughput, and cost per 1k calls.
- Compare against the teacher on its L40S with the same benchmark script, so the cost comparison is like for like.

## Publishing

- **Code:** [github.com/mikan-research/mikan](https://github.com/mikan-research/mikan), Apache-2.0. Contributions are
  welcome; see `CONTRIBUTING.md`.
- **Models and data:** the Hugging Face organization [mikan-research](https://huggingface.co/mikan-research).
  Adapters are Apache-2.0 (like the Qwen3 base models); datasets are CC-BY-4.0, as required by Banking77 and MASSIVE,
  and credit the original authors.
- `mikan/publish_hf.py` stages each repo (adapter files, run metadata, a generated card with the golden result and its
  confidence interval) and uploads it. It refuses to upload into a repo that is already public unless
  `--allow-public` is passed, so an update is always a deliberate step.

| Task | Models | Dataset |
|---|---|---|
| Banking77 | `banking77-qwen3-{0.6b,1.7b,4b}-lora` (teacher labels), `banking77-qwen3-{1.7b,4b}-gold-lora` (ground truth) | `banking77-distill`: splits, teacher labels, every run's predictions |
| MASSIVE en-US | `massive-qwen3-{1.7b,0.6b}-grpo-lora`, `massive-qwen3-1.7b-sft-lora`, `massive-qwen3-1.7b-gold-lora` | `massive-distill`: splits, retrieval neighbours, teacher frames, prompt pilots, every Phase 3 run's predictions |
