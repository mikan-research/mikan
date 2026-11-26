"""Paired statistics: bootstrap CI on a metric difference, McNemar, non-inferiority."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .metrics import get_metric


@dataclass(frozen=True)
class BootstrapCI:
    """CI for ``metric(a) - metric(b)``. Positive means b (student) is worse."""

    metric: str
    estimate: float
    low: float
    high: float
    confidence: float
    n_boot: int


@dataclass(frozen=True)
class McNemarResult:
    a_only: int  # items a gets right and b gets wrong
    b_only: int  # items b gets right and a gets wrong
    p_exact: float  # two-sided exact binomial test on the discordant pairs
    chi2: float  # continuity-corrected statistic
    p_chi2: float


@dataclass(frozen=True)
class NonInferiority:
    delta: float
    upper: float
    passed: bool


def _check_paired(gold, pred_a, pred_b):
    gold, pred_a, pred_b = (np.asarray(x) for x in (gold, pred_a, pred_b))
    if not (gold.shape == pred_a.shape == pred_b.shape) or gold.ndim != 1:
        raise ValueError("gold, pred_a and pred_b must be 1-D arrays of the same length")
    if gold.size == 0:
        raise ValueError("no items to evaluate")
    return gold, pred_a, pred_b


def paired_bootstrap(
    gold,
    pred_a,
    pred_b,
    metric: str = "accuracy",
    n_classes: int | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 1000,
) -> BootstrapCI:
    """Percentile bootstrap CI for ``metric(a) - metric(b)``, resampling items.

    Both systems are scored on the same resampled items, which keeps the pairing
    and gives a much tighter CI than comparing two independent CIs.
    """
    gold, pred_a, pred_b = _check_paired(gold, pred_a, pred_b)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    fn = get_metric(metric)
    n = gold.size
    estimate = fn(gold, pred_a, n_classes) - fn(gold, pred_b, n_classes)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    if metric == "accuracy":
        # Accuracy difference is the mean of per-item (a_correct - b_correct),
        # so a resample is just a mean over resampled indices.
        d = (gold == pred_a).astype(np.int8) - (gold == pred_b).astype(np.int8)
        for start in range(0, n_boot, chunk):
            m = min(chunk, n_boot - start)
            idx = rng.integers(0, n, size=(m, n))
            diffs[start : start + m] = d[idx].mean(axis=1)
    else:
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            g = gold[idx]
            diffs[i] = fn(g, pred_a[idx], n_classes) - fn(g, pred_b[idx], n_classes)

    alpha = 1.0 - confidence
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(
        metric=metric,
        estimate=float(estimate),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_boot=n_boot,
    )


def mcnemar(correct_a, correct_b) -> McNemarResult:
    """McNemar's test on paired per-item correctness.

    Tests whether the two systems differ at all. It is reported alongside the
    non-inferiority decision, not used to make it.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("correct_a and correct_b must have the same shape")
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return McNemarResult(0, 0, 1.0, 0.0, 1.0)
    p_exact = float(sps.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    chi2 = max(abs(a_only - b_only) - 1, 0) ** 2 / n
    p_chi2 = float(sps.chi2.sf(chi2, df=1))
    return McNemarResult(a_only, b_only, min(p_exact, 1.0), float(chi2), p_chi2)


def non_inferiority(ci: BootstrapCI, delta: float) -> NonInferiority:
    """Pass if the upper CI bound of ``metric(a) - metric(b)`` is below ``delta``.

    With a two-sided 95% CI this is a one-sided test at the 2.5% level: when the
    student is truly ``delta`` worse, it passes at most about 2.5% of the time.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    return NonInferiority(delta=delta, upper=ci.high, passed=bool(ci.high < delta))
