"""Command line: ``mikan-eval report | power | selfcheck``."""

from __future__ import annotations

import argparse
import sys

from ..tasks import TaskConfig
from .data import align, load_labels, read_jsonl
from .power import analytic_power, min_detectable_margin, required_n, simulated_power
from .report import evaluate, write_report
from .synthetic import exit_check


def _cmd_report(args: argparse.Namespace) -> int:
    task = TaskConfig.load(args.task)
    labels_path = args.labels or task.labels_file
    if labels_path is None:
        print("error: no labels file (set labels_file in the task config or pass --labels)", file=sys.stderr)
        return 2
    labels = load_labels(labels_path)
    al = align(
        read_jsonl(args.gold),
        {args.a_name: read_jsonl(args.a), args.b_name: read_jsonl(args.b)},
        labels,
    )
    result = evaluate(al, task, a=args.a_name, b=args.b_name, n_examples=args.examples)
    json_path, md_path = write_report(result, args.out)
    p = result["primary"]
    print(
        f"{result['decision']}: {p['metric']} {args.a_name} - {args.b_name} = {100 * p['estimate']:.2f} pts, "
        f"CI [{100 * p['low']:.2f}, {100 * p['high']:.2f}], delta = {100 * task.delta:.2f}"
    )
    print(f"wrote {json_path} and {md_path}")
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    d, delta, td = args.disagreement, args.delta, args.true_diff
    print(f"n = {args.n}, disagreement = {d:.4f}, delta = {delta:.4f}, true diff = {td:.4f}")
    print(f"  analytic power : {analytic_power(args.n, d, delta, td, args.confidence):.4f}")
    print(f"  simulated power: {simulated_power(args.n, d, delta, td, args.confidence, args.sims, args.seed):.4f}")
    print(f"  min detectable margin at {args.target_power:.0%} power (true diff 0): "
          f"{min_detectable_margin(args.n, d, args.target_power, args.confidence):.4f}")
    if delta > td:
        print(f"  required n for {args.target_power:.0%} power: "
              f"{required_n(d, delta, td, args.target_power, args.confidence)}")
    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    rows = exit_check(
        delta=args.delta, n=args.n, disagreement=args.disagreement,
        n_datasets=args.datasets, n_boot=args.boot, seed=args.seed,
    )
    print(f"Synthetic exit check: n = {args.n}, disagreement = {args.disagreement}, delta = {args.delta}, "
          f"{args.datasets} datasets per row, {args.boot} bootstrap resamples")
    print(f"{'true diff':>10} {'pass rate':>10} {'expected':>10}")
    for r in rows:
        print(f"{r.true_diff:>10.3f} {r.pass_rate:>10.3f} {r.expected:>10.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mikan-eval", description="Mikan evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="evaluate teacher vs student predictions and write a report")
    r.add_argument("--task", required=True, help="task config YAML")
    r.add_argument("--gold", required=True, help="gold JSONL: {id, label, text?}")
    r.add_argument("--a", required=True, help="reference (teacher) predictions JSONL: {id, pred}")
    r.add_argument("--b", required=True, help="candidate (student) predictions JSONL: {id, pred}")
    r.add_argument("--a-name", default="teacher")
    r.add_argument("--b-name", default="student")
    r.add_argument("--labels", help="labels file (one per line); overrides the task config")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--examples", type=int, default=10, help="error examples per direction")
    r.set_defaults(func=_cmd_report)

    w = sub.add_parser("power", help="test-set power for paired accuracy non-inferiority")
    w.add_argument("--n", type=int, required=True)
    w.add_argument("--disagreement", type=float, required=True, help="share of items where exactly one system is right")
    w.add_argument("--delta", type=float, required=True)
    w.add_argument("--true-diff", type=float, default=0.0)
    w.add_argument("--confidence", type=float, default=0.95)
    w.add_argument("--target-power", type=float, default=0.8)
    w.add_argument("--sims", type=int, default=20_000)
    w.add_argument("--seed", type=int, default=0)
    w.set_defaults(func=_cmd_power)

    s = sub.add_parser("selfcheck", help="Phase 0 exit check on synthetic students with known gaps")
    s.add_argument("--n", type=int, default=3080)
    s.add_argument("--disagreement", type=float, default=0.08)
    s.add_argument("--delta", type=float, default=0.02)
    s.add_argument("--datasets", type=int, default=200)
    s.add_argument("--boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_selfcheck)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
