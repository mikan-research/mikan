"""Metrics on integer label arrays.

Labels are integers in ``[0, n_classes)``. A prediction that is not a valid label
(wrong format, extra text, unknown label) is encoded as ``INVALID`` (-1), so it
counts as wrong for every metric without special handling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

INVALID = -1

MetricFn = Callable[[np.ndarray, np.ndarray, int | None], float]


def accuracy(gold: np.ndarray, pred: np.ndarray, n_classes: int | None = None) -> float:
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    if gold.size == 0:
        return float("nan")
    return float(np.mean(gold == pred))


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int | None) -> float:
    """Unweighted mean of per-class F1.

    Per class, F1 = 2*TP / (n_pred + n_gold). Classes that appear in neither the
    gold labels nor the predictions are skipped (they carry no information and
    would otherwise make bootstrap resamples unstable). Invalid predictions add
    to no class's prediction count, so they only lower recall.
    """
    if n_classes is None:
        raise ValueError("macro_f1 needs n_classes")
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    correct = gold == pred
    tp = np.bincount(gold[correct], minlength=n_classes)
    gold_count = np.bincount(gold, minlength=n_classes)
    pred_count = np.bincount(pred[pred >= 0], minlength=n_classes)
    denom = gold_count + pred_count
    present = denom > 0
    if not present.any():
        return float("nan")
    return float(np.mean(2.0 * tp[present] / denom[present]))


def validity_rate(pred: np.ndarray) -> float:
    """Share of predictions that are a valid label."""
    pred = np.asarray(pred)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred >= 0))


METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}


def get_metric(name: str) -> MetricFn:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(METRICS)}") from None
