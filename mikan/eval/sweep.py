"""Evaluate many student runs against the teacher and summarise them in one table.

    uv run python -m mikan.eval.sweep --runs runs/kaggle/mikan-sft-phase2/runs \
        --teacher runs/teacher-qwen3-30b-a3b-instruct-2507-fp8-10shot --out reports/phase2

Each run directory needs ``golden_preds.jsonl`` (and optionally ``meta.json``). A full
report is written per run, plus ``summary.md`` / ``summary.json`` for all of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..tasks import TaskConfig
from .data import align, load_labels, read_jsonl
from .report import evaluate, write_report


def summarise(runs_dir: Path, teacher_dir: Path, task_path: Path, gold_path: Path, out: Path) -> list[dict]:
    task = TaskConfig.load(task_path)
    labels = load_labels(task.labels_file)
    gold = read_jsonl(gold_path)
    teacher = read_jsonl(teacher_dir / "golden_preds.jsonl")
    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if (p / "golden_preds.jsonl").exists()):
        al = align(gold, {"teacher": teacher, run.name: read_jsonl(run / "golden_preds.jsonl")}, labels)
        r = evaluate(al, task, a="teacher", b=run.name)
        write_report(r, out / run.name)
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        hist = meta.get("history", [])
        rows.append({
            "run": run.name,
            "model": meta.get("model"),
            "targets": meta.get("targets"),
            "n_train": meta.get("n_train"),
            "best_epoch": meta.get("best_epoch"),
            "val_accuracy": max((h["val_accuracy"] for h in hist), default=None),
            "golden_accuracy": r["summary"][run.name]["accuracy"],
            "validity": r["summary"][run.name]["validity_rate"],
            "gap": r["primary"]["estimate"],
            "gap_low": r["primary"]["low"],
            "gap_high": r["primary"]["high"],
            "disagreement": r["power"]["observed_disagreement"] if r["power"] else 0.0,
            "decision": r["decision"],
            "train_seconds": meta.get("train_seconds"),
        })
    teacher_acc = evaluate(align(gold, {"teacher": teacher, "t2": teacher}, labels), task, "teacher", "t2")["summary"]["teacher"]["accuracy"]
    (out / "summary.json").write_text(json.dumps({"teacher_accuracy": teacher_acc, "runs": rows}, indent=2) + "\n")
    (out / "summary.md").write_text(render(rows, teacher_acc, task.delta))
    return rows


def _p(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def render(rows: list[dict], teacher_acc: float, delta: float) -> str:
    lines = [
        "# Student sweep vs teacher (golden set)",
        "",
        f"Teacher accuracy: {_p(teacher_acc)}%. Margin δ = {_p(delta)} points; gap = teacher − student.",
        "",
        "| Run | Targets | n train | Best epoch | Val acc | Golden acc | Gap [95% CI] | Valid | Disagree | Decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['targets'] or '–'} | {r['n_train'] or '–'} | {r['best_epoch'] or '–'} | "
            f"{_p(r['val_accuracy'])} | {_p(r['golden_accuracy'])} | {_p(r['gap'])} [{_p(r['gap_low'])}, {_p(r['gap_high'])}] | "
            f"{_p(r['validity'])} | {_p(r['disagreement'])} | {r['decision']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a directory of student runs against the teacher")
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--teacher", required=True, type=Path)
    p.add_argument("--task", default=Path("tasks/banking77.yaml"), type=Path)
    p.add_argument("--gold", default=Path("data/banking77/golden.jsonl"), type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    rows = summarise(args.runs, args.teacher, args.task, args.gold, args.out)
    print((args.out / "summary.md").read_text())
    print(f"{len(rows)} runs; reports in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
