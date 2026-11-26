"""Build the evaluation result for a teacher/student pair and render it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..tasks import TaskConfig
from .data import Aligned
from .metrics import accuracy, macro_f1, validity_rate
from .power import analytic_power, min_detectable_margin
from .stats import mcnemar, non_inferiority, paired_bootstrap


def _system_summary(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    return {
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, n_classes),
        "validity_rate": validity_rate(pred),
    }


def _slices(al: Aligned, a: str, b: str) -> list[dict]:
    rows = []
    ca = al.preds[a] == al.gold
    cb = al.preds[b] == al.gold
    for k, lab in enumerate(al.labels):
        mask = al.gold == k
        n = int(mask.sum())
        if n == 0:
            continue
        acc_a = float(ca[mask].mean())
        acc_b = float(cb[mask].mean())
        rows.append({"label": lab, "n": n, f"{a}_accuracy": acc_a, f"{b}_accuracy": acc_b, "diff": acc_a - acc_b})
    rows.sort(key=lambda r: (-r["diff"], r["label"]))
    return rows


def _examples(al: Aligned, a: str, b: str, mask: np.ndarray, limit: int) -> list[dict]:
    out = []
    for i in np.flatnonzero(mask)[:limit]:
        out.append(
            {
                "id": al.ids[i],
                "text": al.texts[i] if al.texts else None,
                "gold": al.labels[al.gold[i]],
                a: al.raw_preds[a][i],
                b: al.raw_preds[b][i],
            }
        )
    return out


def evaluate(al: Aligned, task: TaskConfig, a: str = "teacher", b: str = "student", n_examples: int = 10) -> dict:
    """Run every analysis and return a JSON-serialisable result."""
    gold, pa, pb = al.gold, al.preds[a], al.preds[b]
    n = int(gold.size)

    primary = paired_bootstrap(
        gold, pa, pb, metric=task.metric, n_classes=al.n_classes,
        n_boot=task.n_boot, confidence=task.confidence, seed=task.seed,
    )
    ni = non_inferiority(primary, task.delta)
    secondary = [
        paired_bootstrap(
            gold, pa, pb, metric=m, n_classes=al.n_classes,
            n_boot=task.n_boot, confidence=task.confidence, seed=task.seed,
        )
        for m in task.secondary_metrics
    ]

    ca, cb = pa == gold, pb == gold
    mc = mcnemar(ca, cb)
    disagreement = (mc.a_only + mc.b_only) / n
    power = None
    if disagreement > 0:
        power = {
            "observed_disagreement": disagreement,
            "power_if_equal": analytic_power(n, disagreement, task.delta, 0.0, task.confidence),
            "min_detectable_margin": min_detectable_margin(n, disagreement, task.power, task.confidence),
            "target_power": task.power,
        }

    slices = _slices(al, a, b)
    limitations = [
        f"Golden labels: {task.label_source}.",
        "CI method: percentile paired bootstrap over items.",
        "McNemar tests for any difference; it does not decide pass/fail.",
    ]
    if power and power["min_detectable_margin"] > task.delta:
        limitations.append(
            f"Test set is underpowered: at the observed disagreement the smallest margin resolvable with "
            f"{task.power:.0%} power is {power['min_detectable_margin']:.2%}, larger than delta."
        )
    for name in (a, b):
        v = validity_rate(al.preds[name])
        if v < 1:
            limitations.append(f"{name} produced {1 - v:.2%} invalid outputs (counted as wrong).")

    return {
        "task": task.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "path": str(task.path) if task.path else None,
            "sha256": task.sha256,
            "metric": task.metric,
            "delta": task.delta,
            "confidence": task.confidence,
            "n_boot": task.n_boot,
            "seed": task.seed,
        },
        "n_items": n,
        "n_classes": al.n_classes,
        "systems": {"a": a, "b": b},
        "summary": {
            a: _system_summary(gold, pa, al.n_classes),
            b: _system_summary(gold, pb, al.n_classes),
        },
        "primary": primary.__dict__,
        "secondary": [s.__dict__ for s in secondary],
        "non_inferiority": ni.__dict__,
        "decision": "PASS" if ni.passed else "FAIL",
        "mcnemar": mc.__dict__,
        "power": power,
        "slices": slices,
        "examples": {
            f"{a}_right_{b}_wrong": _examples(al, a, b, ca & ~cb, n_examples),
            f"{b}_right_{a}_wrong": _examples(al, a, b, ~ca & cb, n_examples),
        },
        "limitations": limitations,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def render_markdown(r: dict, n_slices: int = 10) -> str:
    a, b = r["systems"]["a"], r["systems"]["b"]
    p, ni, mc = r["primary"], r["non_inferiority"], r["mcnemar"]
    conf = round(100 * r["config"]["confidence"])
    lines = [
        f"# Evaluation report: {r['task']}",
        "",
        f"Generated {r['generated_at']}. Config `{r['config']['path']}` (sha256 `{(r['config']['sha256'] or '')[:12]}`).",
        "",
        f"## Decision: {r['decision']}",
        "",
        f"Non-inferiority on **{p['metric']}** with margin δ = {_pct(ni['delta'])} points: "
        f"{a} − {b} = {_pct(p['estimate'])} points, {conf}% CI [{_pct(p['low'])}, {_pct(p['high'])}]. "
        f"Pass requires the upper bound ({_pct(ni['upper'])}) to be below δ.",
        "",
        f"Positive differences mean {b} is worse than {a}. Test items: {r['n_items']}, classes: {r['n_classes']}.",
        "",
        "## Systems",
        "",
        "| System | Accuracy (%) | Macro-F1 (%) | Valid outputs (%) |",
        "|---|---|---|---|",
    ]
    for name in (a, b):
        s = r["summary"][name]
        lines.append(f"| {name} | {_pct(s['accuracy'])} | {_pct(s['macro_f1'])} | {_pct(s['validity_rate'])} |")

    lines += ["", "## Differences", "", f"| Metric | {a} − {b} (points) | {conf}% CI |", "|---|---|---|"]
    for ci in [p, *r["secondary"]]:
        lines.append(f"| {ci['metric']} | {_pct(ci['estimate'])} | [{_pct(ci['low'])}, {_pct(ci['high'])}] |")

    lines += [
        "",
        "## McNemar (paired correctness)",
        "",
        f"{a} right / {b} wrong: {mc['a_only']}; {b} right / {a} wrong: {mc['b_only']}. "
        f"Exact p = {mc['p_exact']:.4g} (continuity-corrected χ² = {mc['chi2']:.3f}, p = {mc['p_chi2']:.4g}).",
    ]

    if r["power"]:
        pw = r["power"]
        lines += [
            "",
            "## Test-set power",
            "",
            f"Observed disagreement rate: {_pct(pw['observed_disagreement'])}%. "
            f"Power to pass if the systems were truly equal: {_pct(pw['power_if_equal'])}%. "
            f"Smallest margin resolvable with {pw['target_power']:.0%} power: {_pct(pw['min_detectable_margin'])} points.",
        ]

    worse = [s for s in r["slices"] if s["diff"] > 0]
    lines += [
        "",
        f"## Per-label slices ({len(worse)} of {len(r['slices'])} labels where {b} is worse)",
        "",
        f"| Label | n | {a} (%) | {b} (%) | Diff (points) |",
        "|---|---|---|---|---|",
    ]
    for s in r["slices"][:n_slices]:
        lines.append(f"| {s['label']} | {s['n']} | {_pct(s[f'{a}_accuracy'])} | {_pct(s[f'{b}_accuracy'])} | {_pct(s['diff'])} |")

    for title, items in r["examples"].items():
        lines += ["", f"## Examples: {title.replace('_', ' ')}", ""]
        if not items:
            lines.append("None.")
            continue
        lines += [f"| id | text | gold | {a} | {b} |", "|---|---|---|---|---|"]
        for e in items:
            text = (e["text"] or "").replace("|", "\\|").replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(f"| {e['id']} | {text} | {e['gold']} | {e[a]} | {e[b]} |")

    lines += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"


def write_report(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path
