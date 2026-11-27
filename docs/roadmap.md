# Roadmap

Build phases with exit criteria, not dates. Phases 0–3 are done; Phase 4 (quantization and serving) is next. Concrete choices and results per task are in [setup.md](setup.md).

## Phase 0: Eval harness

- `eval/`: accuracy and macro-F1, paired bootstrap, McNemar, non-inferiority check, power check, report generator
- Unit-test the statistics on synthetic data with a known effect size

**Exit:** the harness correctly passes/fails synthetic students with known gaps around δ.

**Status: done.** Results in [eval-harness.md](eval-harness.md).

## Phase 1: Teacher data and baselines (Banking77)

- `data/`: load Banking77, stratified validation split, golden set = official test split
- Pre-register δ = 2 accuracy points and the primary metric in `tasks/banking77.yaml`
- `intake/`: teacher generation script for Modal (Qwen3-30B-A3B-Instruct, vLLM, prefix caching, guided choice decoding) — done
- Smoke test on 50 items, pilots on 300 validation items (1/5/10-shot: 75.0 / 82.7 / 84.0%), then one full 10-shot run, cached — done
- Zero-shot / few-shot baselines for Qwen3 0.6B, 1.7B and 4B students on Colab

**Exit:** eval report comparing teacher vs zero-shot students; teacher labels cached; Modal spend under about $10.

**Status: done.** Golden-set accuracy (3,080 items); reports in `reports/phase1/`:

| Student | 0-shot | 10-shot | Gap to teacher at 10-shot (95% CI) | Disagreement with teacher (10-shot) |
|---|---|---|---|---|
| Qwen3-0.6B | 36.0% | 50.6% | 35.0 points [33.2, 36.8] | 38.3% |
| Qwen3-1.7B | 48.6% | 65.9% | 19.7 points [18.1, 21.3] | 25.7% |
| Qwen3-4B | 63.4% | 79.1% | 6.5 points [5.3, 7.8] | 13.2% |
| *Teacher (Qwen3-30B-A3B, 10-shot)* | | *85.6%* | | |

All untrained students fail non-inferiority, as expected. Modal spend for all of Phase 1: about $1.

## Phase 2: SFT distillation

- `train/`: Unsloth + TRL `SFTTrainer`, LoRA, on teacher labels (Kaggle T4)
- Ablations:
  - teacher labels vs ground-truth labels (how much does distillation lose vs supervised training?)
  - training-set size: 1k, 2.5k, 5k, full (minimum data question)
  - student size: 0.6B vs 1.7B vs 4B
- Checkpoint selection on the validation split only

**Exit:** SFT student report against the teacher, plus the three ablation curves.

**Status: done.** Results (golden set, teacher 85.6%):

| Student (LoRA SFT, 3 epochs) | Labels | Train items | Golden acc | Gap vs teacher [95% CI] | Decision |
|---|---|---|---|---|---|
| Qwen3-0.6B | teacher | 8,222 | 85.6% | +0.1 [−0.8, +0.9] | PASS |
| Qwen3-1.7B | teacher | 8,222 | 85.4% | +0.3 [−0.6, +1.1] | PASS |
| Qwen3-4B | teacher | 8,222 | 86.0% | −0.4 [−1.2, +0.5] | PASS |
| Qwen3-1.7B | teacher | 5,000 | 83.9% | +1.7 [+0.7, +2.7] | FAIL |
| Qwen3-1.7B | teacher | 2,500 | 82.6% | +3.0 [+1.9, +4.0] | FAIL |
| Qwen3-1.7B | teacher | 1,000 | 71.0% | +14.6 [+13.1, +16.1] | FAIL |
| Qwen3-1.7B | ground truth | 8,222 | 93.1% | −7.4 [−8.7, −6.2] | PASS |
| Qwen3-4B | ground truth | 8,222 | 94.0% | −8.3 [−9.6, −7.1] | PASS |

- **Distillation works at every size.** With the full teacher-labelled set, all three students pass non-inferiority (teacher 85.6%), using a short prompt with no label list or examples. The 0.6B goes from 50.6% (10-shot, untrained) to 85.6%.
- **The teacher is the ceiling.** Teacher-label students land within half a point of the teacher and disagree with it on only ~6% of items; the same models trained on ground truth reach 93–94%. Better labels, not a bigger student, is the lever.
- **Data size matters more than model size.** 1k → 2.5k → 5k → 8.2k items gives 71.0 → 82.6 → 83.9 → 85.4% for the 1.7B; only the full set passes. Model size (0.6B / 1.7B / 4B) moves golden accuracy by under one point.
- Checkpoints were chosen on val only; golden was predicted once per run. Training time on one T4: 0.6B 15 min, 1.7B 23 min, 4B 45 min.

Issue hit: the 4B runs ran out of T4 memory when switching from training to generation; fixed by freeing the CUDA cache around each prediction pass and an eval batch of 16.

Phase 3 moves GRPO to a structured-output task (see below), since Banking77 has no code-checkable reward without ground truth.

## Phase 3: GRPO on a structured-output task (MASSIVE en-US)

Banking77 has no code-checkable reward without ground truth, so GRPO moves to a second task whose
output a program can check: MASSIVE v1.1 en-US (Amazon, CC-BY-4.0), 60 intents and 55 slot types.
The output is a JSON frame `{"intent", "slots": [{"type", "value"}]}`.

- Data (`mikan.data.massive`): train 11,255 / val 2,026 (official dev, shuffled) / golden 2,974
  (official test) / few-shot pool 181 (every slot type covered at rank 0). All gold slot values
  occur verbatim in their utterance, so groundedness is a valid constraint.
- Pre-registered task (`tasks/massive.yaml`): primary metric exact frame match, δ = 3 points
  (power: 2,974 items resolve 2.0 points only up to 15% disagreement; exact match will be noisier).
- Checks (`mikan.frames`): valid JSON with the exact shape, intent and slot types in the schema,
  slot values grounded in the input, exact match on the (type, value) multiset.
- Reward (teacher frame as reference, never golden): 0 if invalid, else
  0.1 + 0.3·intent match + 0.6·slot F1·groundedness. Invented spans earn nothing.
- Steps: teacher pilot (fixed prompts, then retrieved examples; constrained to the frame schema) →
  full teacher labels → SFT (0.6B / 1.7B, full and 1k; ground-truth ceiling) → GRPO from the SFT checkpoints
  (TRL `GRPOTrainer` + Unsloth, β = 0, step 0 kept as a candidate) → reward-hacking checks on sampled outputs.
- Main comparison: SFT vs SFT+GRPO at the same teacher-label budget, and SFT-1k+GRPO (reward on
  the remaining unlabeled prompts' teacher frames) vs SFT-full.

**Progress (2026-11-26):**

- Built: `mikan.data.massive`, `mikan.frames`, frames support in the eval harness, `mikan.data.retrieve`
  (TF-IDF nearest labelled examples, never the item itself, no val/golden in the pool), SFT for frames,
  `mikan.train.grpo`, Kaggle teacher runner.
- Teacher on Kaggle T4 x2 (Modal credit used up): `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`, vLLM 0.13,
  tensor parallel 2, fp16, Triton attention. Compressed-tensors AWQ builds fail on T4 (Marlin needs sm80);
  FlashInfer's JIT fails to link (`-lcuda`) on Kaggle.
- Fixed-prompt pilot (300 val): 1-shot 44.7% / 3-shot 48.0% exact match (McNemar p = 0.22); intent accuracy
  ~80%, slot F1 54–61%. Errors are mostly MASSIVE annotation conventions (extra slots, span edges, slot type).
- GRPO smoke (0.6B, gold reference, 500-item SFT): val 22.5% → 25.5% at step 15 → 18.5% at step 30; step 15 kept.
- Retrieved-examples pilot (300 val): 8 neighbours 67.0%, 20 neighbours 68.0% (p = 0.7, 2.3x slower) vs 48.0% for
  3-shot (p = 7e-9). Chosen: 8.

**Progress (2026-11-27):**

- Full teacher run, 8 retrieved examples, two parallel Kaggle kernels (~3.2 h each): exact match train 63.7%,
  val 64.9%, **golden 64.4%**, 100% valid. Merged into `runs/massive-teacher-knn8/`; published as the
  Hugging Face dataset `mikan-research/massive-distill`.
- Phase 3 code committed. Grid launched as `mikan-massive-phase3-a` / `-b` (SFT 1.7B teacher full → GRPO;
  SFT 1.7B ground truth; SFT 0.6B teacher full → GRPO; SFT 1.7B teacher 1k → GRPO on unseen prompts).
- Pass bar for students: the upper 95% CI bound of (teacher − student) exact match below 3 points.
- Correction: one golden teacher output (of 2,974) hit the 160-token limit repeating a slot and is cut off, so golden
  validity is 99.97%, not 100%; it counts as wrong for the teacher. Train and val are 100% valid.
  It is a greedy repetition loop (`audiobook_name: harry potter` repeated), so it is left as is: counting it as
  correct for the teacher moves every CI bound by 0.03 points and changes no decision, and re-labelling it after
  seeing student results would be post hoc. No golden frame exceeds 86 tokens, so the 96/160-token limits stay.

**Exit:** GRPO student passes non-inferiority (or a documented reason why not), and its gain over SFT is measured.

**Status: done** (2026-11-27). Golden set, 2,974 items, teacher 64.4% exact frame match, δ = 3; reports in `reports/phase3/`:

| Student | Training | Golden exact match | Gap vs teacher [95% CI] | Valid | Decision |
|---|---|---|---|---|---|
| Qwen3-0.6B | SFT, 11,255 teacher frames | 64.1% | +0.3 [−1.3, +1.9] | 99.7% | PASS |
| Qwen3-0.6B | + GRPO (best at step 200) | 64.8% | −0.4 [−2.1, +1.2] | 99.6% | PASS |
| Qwen3-1.7B | SFT, 11,255 teacher frames | 65.0% | −0.6 [−2.2, +1.0] | 99.8% | PASS |
| Qwen3-1.7B | + GRPO (best at step 200) | **65.7%** | −1.3 [−2.9, +0.3] | 99.7% | PASS |
| Qwen3-1.7B | SFT, 1,000 teacher frames | 46.0% | +18.4 [+16.5, +20.3] | 95.3% | FAIL |
| Qwen3-1.7B | + GRPO on 3,000 unseen prompts | 50.4% | +14.0 [+12.0, +15.9] | 97.2% | FAIL |
| Qwen3-1.7B | SFT, 11,255 ground-truth frames | 72.1% | −7.8 [−9.4, −6.1] | 99.8% | PASS |

GRPO against its own SFT starting point:

| Pair (same golden items) | GRPO − SFT [95% CI] | Right only after / only before | McNemar p |
|---|---|---|---|
| 0.6B, full SFT | +0.7 [+0.0, +1.5] | 71 / 49 | 0.055 |
| 1.7B, full SFT | +0.7 [−0.0, +1.5] | 72 / 51 | 0.071 |
| 1.7B, 1k SFT | **+4.4 [+3.1, +5.8]** | 282 / 150 | 2e-10 |

- **Distillation passes on structured output too.** SFT on teacher frames alone passes at both sizes; the 1.7B
  slightly beats the teacher (65.0 vs 64.4%) with a short prompt that has no intent or slot lists.
- **GRPO helps most when labelled data is scarce.** After a full SFT it adds +0.7 points (borderline, p ≈ 0.06);
  after a 1k SFT it adds +4.4 points (p = 2e-10) using only the teacher frames of 3,000 new prompts as reward
  references, still far from passing.
- **What GRPO changes:** fewer slots (1k run: 3,090 → 2,704 emitted on golden), higher slot precision, and fewer
  invented labels (171 → 97 unknown intents or slot types such as `number`, `quantity`, `music_remove`). Validity and
  groundedness stay flat or improve; no extra text, no invented spans. Reward peaked mid-run (1k run: 0.59 → 0.83 at
  step 130 → 0.64 at 250), and val selection kept steps 200–250.
- **Labels are still the ceiling:** ground-truth SFT reaches 72.1% vs 65.0% on teacher labels, a bigger gap than
  anything GRPO closes.
- Remaining invalid outputs are invented labels; serving with the frame JSON schema (constrained decoding) removes
  them by construction (Phase 4).

Fix found while reading samples: evaluation kept only the first line of each output (a Banking77 rule) while the
reward parses the whole output, so a frame split across lines would be rewarded but scored invalid. At most one golden
item was affected; frames evaluation now keeps the whole output (`mikan.train.gen.postprocess`).

## Phase 4: Quantization and serving

- `serve/`: AWQ or GPTQ 4-bit, full eval re-run on the quantized model
- vLLM on Modal with an OpenAI-compatible endpoint; latency, throughput and cost benchmarks vs the teacher on the same setup

**Exit:** quantized student passes non-inferiority and is served, with the cost and latency comparison in the report.

## Phase 5: Generalise the pipeline

- Add a second task with a different output type (e.g. NER as structured JSON)
- `tasks/` config format, so a new task needs only a config
- `scrub/` and the log-format adapters for real production logs
- `monitor/`: drift checks and scheduled re-eval

**Exit:** a second task runs end to end from a config, with no new pipeline code.
