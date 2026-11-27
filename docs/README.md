# Mikan docs

Mikan is a pipeline that distils a frontier model's behaviour on one narrow, high-volume task into a small open model. Quality is measured with a pre-registered statistical test against the frontier model.

Status: Phases 0–2 done on Banking77 (every SFT student passes non-inferiority). Phase 3 (GRPO) done on a second, structured-output task, MASSIVE en-US: teacher 64.4% exact frame match on golden; SFT students pass, GRPO adds +0.7 points after a full SFT and +4.4 after a 1k SFT. Phase 4 (quantization and serving) next.

| Doc | What it covers |
|---|---|
| [overview.md](overview.md) | Goal, scope, task types, success criteria |
| [setup.md](setup.md) | Concrete choices and results per task: Banking77 (Modal teacher, δ = 2) and MASSIVE (Kaggle teacher, retrieved examples, δ = 3); compute, stack, budget, backups |
| [architecture.md](architecture.md) | Pipeline: intake, scrubbing, dataset, training, quantization, serving, monitoring |
| [evaluation.md](evaluation.md) | Evaluation method: non-inferiority test, bootstrap CIs, McNemar |
| [eval-harness.md](eval-harness.md) | How to run the eval harness (labels and frames tasks, sweeps), module map, Phase 0 exit results |
| [roadmap.md](roadmap.md) | Build phases with exit criteria |
| [risks-and-open-questions.md](risks-and-open-questions.md) | Technical risks and decisions still to make |
