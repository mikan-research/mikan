# Mikan documentation

Mikan distils a large teacher model's behaviour on one narrow task into a small open student model, and judges the
student with a pre-registered non-inferiority test against the teacher.

| Doc | Kind | Read it when you want to |
|---|---|---|
| [getting-started.md](getting-started.md) | Tutorial | install Mikan and reproduce a published result on a laptop, no GPU needed |
| [training.md](training.md) | How-to | label data with a teacher, train students with SFT and GRPO, locally or on Kaggle |
| [adding-a-task.md](adding-a-task.md) | How-to | bring a new task: data, pre-registered config, checks, tests |
| [reference.md](reference.md) | Reference | look up a command, flag, config key or file format |
| [evaluation.md](evaluation.md) | Explanation | understand the statistics: non-inferiority, paired bootstrap, McNemar, power |
| [pipeline.md](pipeline.md) | Explanation | understand how a run works and why it is built that way |

Models and datasets are on Hugging Face at [mikan-research](https://huggingface.co/mikan-research). To contribute,
see [CONTRIBUTING.md](../CONTRIBUTING.md).
