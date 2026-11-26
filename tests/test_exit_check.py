"""Phase 0 exit criterion: the full decision rule passes/fails synthetic students correctly."""

import numpy as np
import pytest

from mikan.eval.synthetic import exit_check, make_paired


def test_make_paired_hits_target_rates():
    gold, t, s = make_paired(200_000, 77, 0.92, 0.08, 0.02, seed=0)
    ca, cs = t == gold, s == gold
    assert ca.mean() == pytest.approx(0.92, abs=0.003)
    assert ca.mean() - cs.mean() == pytest.approx(0.02, abs=0.003)
    assert np.mean(ca != cs) == pytest.approx(0.08, abs=0.003)


def test_make_paired_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        make_paired(100, 5, teacher_acc=0.01, disagreement=0.5, true_diff=0.4)


def test_exit_check_decisions():
    rows = {r.true_diff: r for r in exit_check(n_datasets=200, n_boot=1000)}
    # Better or equal students pass almost always.
    assert rows[-0.02].pass_rate >= 0.99
    assert rows[0.0].pass_rate >= 0.93
    # Halfway to the margin: a coin flip, as the power analysis predicts.
    assert rows[0.01].pass_rate == pytest.approx(rows[0.01].expected, abs=0.12)
    # Exactly at the margin: type I error near 2.5%.
    assert rows[0.02].pass_rate <= 0.06
    # Clearly worse students never pass.
    assert rows[0.03].pass_rate == 0.0
    assert rows[0.04].pass_rate == 0.0
