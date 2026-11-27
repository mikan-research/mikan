"""Synthetic paired predictions with a known accuracy gap, for testing the harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .power import analytic_power
from .stats import non_inferiority, paired_bootstrap


def make_paired(
    n: int,
    n_classes: int,
    teacher_acc: float,
    disagreement: float,
    true_diff: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gold, teacher_pred, student_pred)`` with a known population gap.

    In the population, teacher accuracy is ``teacher_acc``, the student is
    ``true_diff`` worse, and the two disagree on correctness for a share
    ``disagreement`` of items.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    p_a_only = (disagreement + true_diff) / 2
    p_b_only = (disagreement - true_diff) / 2
    p_both = teacher_acc - p_a_only
    p_neither = 1 - disagreement - p_both
    probs = np.array([p_both, p_a_only, p_b_only, p_neither])
    if (probs < -1e-12).any():
        raise ValueError(f"inconsistent parameters, group probabilities {probs.round(4).tolist()}")
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    gold = rng.integers(0, n_classes, size=n)
    group = rng.choice(4, size=n, p=probs)
    a_right = (group == 0) | (group == 1)
    b_right = (group == 0) | (group == 2)

    def wrong_label(g: np.ndarray) -> np.ndarray:
        return (g + rng.integers(1, n_classes, size=g.size)) % n_classes

    teacher = np.where(a_right, gold, wrong_label(gold))
    student = np.where(b_right, gold, wrong_label(gold))
    return gold, teacher, student


@dataclass(frozen=True)
class ExitCheckRow:
    true_diff: float
    pass_rate: float
    expected: float  # analytic power at this true gap
    n_datasets: int


def exit_check(
    delta: float = 0.02,
    true_diffs: tuple[float, ...] = (-0.02, 0.0, 0.01, 0.02, 0.03, 0.04),
    n: int = 3080,
    n_classes: int = 77,
    teacher_acc: float = 0.92,
    disagreement: float = 0.08,
    n_datasets: int = 200,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[ExitCheckRow]:
    """Self-check: pass rates of the full bootstrap test on synthetic students.

    For each true gap, generate ``n_datasets`` synthetic test sets, run the real
    decision rule, and compare the pass rate with the analytic power. Expected
    behaviour: students that are equal or better pass almost always, a student
    exactly ``delta`` worse passes about (1 - confidence) / 2 of the time, and
    clearly worse students essentially never pass.
    """
    rows = []
    for k, td in enumerate(true_diffs):
        passes = 0
        for i in range(n_datasets):
            ds_seed = seed + 100_000 * k + i
            gold, t, s = make_paired(n, n_classes, teacher_acc, disagreement, td, seed=ds_seed)
            ci = paired_bootstrap(gold, t, s, n_boot=n_boot, confidence=confidence, seed=ds_seed)
            passes += non_inferiority(ci, delta).passed
        rows.append(
            ExitCheckRow(
                true_diff=td,
                pass_rate=passes / n_datasets,
                expected=analytic_power(n, disagreement, delta, td, confidence),
                n_datasets=n_datasets,
            )
        )
    return rows
