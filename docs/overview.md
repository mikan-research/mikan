# Overview

## Goal

Given a narrow task handled by a large teacher model (a frontier API or a large open model), produce a small open model (roughly 0.5B to 8B; up to about 4B on the current compute) that:

1. is statistically non-inferior to the frontier model on that task, within a margin set before training,
2. is much cheaper and faster to run once quantized,
3. is served behind an OpenAI-compatible endpoint, so swapping it in only changes the base URL and model name.

## In scope: task types

Tasks where the output is narrow and ideally checkable by code:

- Classification, routing, tagging
- Field extraction
- Structured JSON generation against a schema
- Short, constrained rewriting (weaker evaluation; see [evaluation.md](evaluation.md))

## Out of scope

Open-ended reasoning, creative writing, long multi-step agents, and tasks without a feasible verified test set.

## Technical approach

- **Evaluation first.** The eval harness is built before any training, and every model is judged by the same pre-registered test.
- **SFT distillation** on teacher (frontier-model) outputs, then **GRPO** with programmatic rewards where output is code-checkable (schema validity, label match, field F1). GRPO can push the student past the teacher on format validity and exact match.
- **Quantization** (AWQ / GPTQ / FP8), with the quantized model re-evaluated, since that is the model that gets served.

## Tasks so far

| Task | Output | Why it is here | Status |
|---|---|---|---|
| Banking77 | one of 77 intent labels | simple, large public test set; SFT distillation baseline | Phases 1–2 done |
| MASSIVE en-US | JSON frame: intent + typed slot spans | output a program can check (schema, spans copied from the input), so GRPO has a real reward | Phase 3 done |

## Success criteria for the project

- An end-to-end, reproducible run on at least one public task: teacher data → trained student → quantized → eval report → served endpoint.
- The student passes non-inferiority against the teacher, with cost per 1k calls and p50/p95 latency reported alongside.
- The pipeline is generic enough that a new task needs only a config (format adapter, metric, reward function), not new code.
