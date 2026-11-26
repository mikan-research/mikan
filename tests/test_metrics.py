import math

import numpy as np
import pytest

from mikan.eval.metrics import INVALID, accuracy, get_metric, macro_f1, validity_rate


def test_accuracy_basic():
    assert accuracy(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2])) == 0.75


def test_accuracy_empty_is_nan():
    assert math.isnan(accuracy(np.array([], dtype=int), np.array([], dtype=int)))


def test_macro_f1_hand_computed():
    # class 0: tp=1, n_pred=1, n_gold=2 -> 2/3 ; class 1: tp=2, n_pred=3, n_gold=2 -> 4/5
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    assert macro_f1(gold, pred, 2) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_macro_f1_invalid_predictions_only_hurt_recall():
    gold = np.array([0, 1])
    pred = np.array([INVALID, 1])
    # class 0: tp=0, n_pred=0, n_gold=1 -> 0 ; class 1: perfect -> 1
    assert macro_f1(gold, pred, 2) == pytest.approx(0.5)


def test_macro_f1_skips_absent_classes():
    gold = np.array([0, 0])
    pred = np.array([0, 0])
    assert macro_f1(gold, pred, 5) == 1.0


def test_macro_f1_requires_n_classes():
    with pytest.raises(ValueError):
        macro_f1(np.array([0]), np.array([0]), None)


def test_validity_rate():
    assert validity_rate(np.array([0, INVALID, 3, INVALID])) == 0.5


def test_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        get_metric("bleu")
