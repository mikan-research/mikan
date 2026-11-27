# Technical architecture

The pipeline for one task, from teacher data to a served model.

```
task data ─► intake ─► PII scrub ─► task audit ─► dataset build ─► SFT ─► GRPO ─► eval ─► quantize ─► re-eval ─► serve ─► monitor
                                        │                                    │                     │
                                   go / no-go                           pass / fail          pass / fail
```

## 1. Intake

Two input sources:

- **Existing logs**: exports in OpenAI / Anthropic / Gemini chat formats (JSONL).
- **Generated teacher data**: run a public dataset's inputs through a teacher model to collect teacher outputs (used for development and public benchmarks). The teacher is a large open model run with vLLM in offline batch mode, with decoding constrained to the valid outputs (a label choice, or the frame JSON schema): on Modal for Banking77, on Kaggle T4 x2 (4-bit GPTQ, tensor parallel) for MASSIVE (see [setup.md](setup.md)).
- **Prompting the teacher**: either one fixed prompt (instruction, label list, ranked examples; computed once thanks to prefix caching) or **retrieved examples** per item (`mikan.data.retrieve`: TF-IDF nearest labelled training items, never the item itself, never val or golden). Retrieval is slower per item but was worth 19 exact-match points on MASSIVE, whose errors were annotation conventions.

Both are normalised to one internal schema: `{id, timestamp, system, input, output, teacher_model, metadata}`. Each format gets its own small adapter.

## 2. PII scrubbing

Applies only when real production logs are used:

- Presidio-style entity detection plus custom regexes (emails, phones, IDs, API keys).
- Keep a scrub log (counts per entity type); never store raw text after scrubbing.
- Per-dataset isolated storage, deleted when no longer needed.

## 3. Task audit (go / no-go)

An automatic report on:

- Task type: label, extraction, structured JSON, free text
- Output checkability: can a function score it?
- Volume, input/output length distribution (drives context length and serving cost)
- Duplicate rate and diversity
- Whether the golden set is big enough to resolve the chosen margin (see [evaluation.md](evaluation.md))

## 4. Dataset build

- Deduplicate (exact and near-duplicate, e.g. MinHash).
- **Time-based split** for logs (train on older, test on newest) to mimic deployment and avoid leakage. Random stratified split for public datasets.
- **Golden test set** (200 to 500 items) with **verified** labels (human-checked or dataset ground truth), *not* teacher outputs. Otherwise the eval measures agreement with the teacher, not correctness.
- Separate validation split for checkpoint selection.

## 5. Base model selection

- Candidates: open models of roughly 0.5B to 8B (Qwen, Llama, Gemma, Phi families). Filter by licence, multilingual needs and context length. On the current free T4 compute the cap is about 4B (with 4-bit QLoRA).
- Zero-shot and few-shot baselines of 2 or 3 candidates on the validation split before training. This also gives the "is training even needed?" answer.

## 6. Training

**SFT distillation**
- LoRA first (cheap, fast); full fine-tune only if LoRA plateaus.
- Train on teacher outputs; optionally filter out teacher outputs that fail the checker.

**GRPO** (when output is code-checkable)
- Reward = weighted sum of:
  - format validity (JSON parses, schema matches)
  - correctness (label exact match, per-field F1) against references
  - optional length penalty
- No KL term (β = 0). With a LoRA model TRL's reference model is the base model with the adapter disabled, not the SFT checkpoint, so a KL penalty would pull the student back toward the untrained model. Drift is limited instead by continuing the SFT adapter at a low learning rate for a few hundred steps, and by choosing the checkpoint on val with step 0 (the SFT adapter itself) as a candidate, so GRPO is only kept if it helps.
- The reference for the correctness terms is the **teacher's** output, never golden; MASSIVE's reward is 0 for an invalid frame, else 0.1 + 0.3·intent match + 0.6·slot F1·groundedness (invented spans earn nothing). Validity and groundedness are also logged as zero-weight rewards so reward hacking shows in the logs.
- Guard against reward hacking: read sampled outputs at every checkpoint, and track metrics on the validation split that the reward does not cover.

## 7. Evaluation

See [evaluation.md](evaluation.md). Produces the report and the pass/fail decision.

## 8. Quantization and serving

- Quantize (AWQ / GPTQ; FP8 only on Ada/Hopper GPUs, not T4) and **re-run the full eval on the quantized model**.
- Serve with vLLM behind an **OpenAI-compatible endpoint**.
- Targets: serverless GPU (Modal, RunPod or similar) and a self-host Docker image.
- Benchmark p50/p95 latency, throughput, and cost per 1k calls against the teacher API.

## 9. Monitoring

- Scheduled re-eval on a fresh labelled sample of recent inputs, using the same margin.
- Drift alerts on the input distribution (length, label mix, embedding-cluster shift).
- Optional shadow mode: send a slice of traffic to both teacher and student and compare.
- A failed re-eval triggers retraining on the new data.

## Artifacts per run

1. Model weights (and LoRA adapter if used), quantized and full precision
2. Eval report (HTML/PDF) plus raw per-item predictions
3. Serving config / Docker image
4. Run config, so the run can be reproduced

## Code layout

Built so far:

```
mikan/
  eval/          metrics, bootstrap, McNemar, non-inferiority, power, reports, sweep, CLI (labels and frames tasks)
  data/          banking77.py, massive.py (splits, leakage checks), retrieve.py (nearest labelled examples)
  intake/        vLLM batch prediction (local or Modal), Kaggle teacher runner, run writing
  train/         sft.py, grpo.py, examples (targets, student prompts), gen (greedy decoding), Kaggle job driver and runner
  frames.py      frame parse / canonical form / exact match / slot F1 / groundedness / reward / JSON schema
  prompts.py     teacher prompts (labels, frames, retrieved examples)
  tasks.py       pre-registered task config loader
  publish_hf.py  publishing adapters and datasets to the mikan-research Hugging Face organization
tasks/           one config per task: metric, margin, labels (and slot types for frames)
configs/         Kaggle run grids (SFT, GRPO, teacher)
```

Planned: `serve/` (quantization, vLLM endpoint, benchmarks), `scrub/`, `audit/`, `monitor/`.
