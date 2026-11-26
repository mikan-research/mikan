import math

import numpy as np
import pytest

from mikan.eval.stats import BootstrapCI, mcnemar, non_inferiority, paired_bootstrap
from mikan.eval.synthetic import make_paired


def test_mcnemar_exact_matches_hand_value():
    # 10 items only a is right, 2 only b is right: P = 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12
    a = np.array([True] * 10 + [False] * 2 + [True] * 5)
    b = np.array([False] * 10 + [True] * 2 + [True] * 5)
    r = mcnemar(a, b)
    assert (r.a_only, r.b_only) == (10, 2)
    assert r.p_exact == pytest.approx(2 * 79 / 4096)
    assert r.chi2 == pytest.approx(49 / 12)


def test_mcnemar_no_discordant_pairs():
    r = mcnemar(np.array([True, False]), np.array([True, False]))
    assert r.p_exact == 1.0 and r.a_only == r.b_only == 0


def test_bootstrap_identical_systems_is_zero():
    gold, t, _ = make_paired(500, 5, 0.8, 0.1, 0.0, seed=1)
    ci = paired_bootstrap(gold, t, t, n_boot=500)
    assert ci.estimate == ci.low == ci.high == 0.0


def test_bootstrap_is_deterministic_given_seed():
    gold, t, s = make_paired(800, 10, 0.9, 0.1, 0.01, seed=2)
    assert paired_bootstrap(gold, t, s, seed=7, n_boot=500) == paired_bootstrap(gold, t, s, seed=7, n_boot=500)


def test_bootstrap_estimate_is_accuracy_difference():
    gold, t, s = make_paired(1000, 10, 0.9, 0.1, 0.02, seed=3)
    ci = paired_bootstrap(gold, t, s, n_boot=200)
    assert ci.estimate == pytest.approx(np.mean(t == gold) - np.mean(s == gold))
    assert ci.low <= ci.estimate <= ci.high


def test_bootstrap_matches_wald_ci_for_accuracy():
    gold, t, s = make_paired(3080, 77, 0.92, 0.08, 0.005, seed=4)
    ci = paired_bootstrap(gold, t, s, n_boot=20_000, seed=0)
    ca, cb = t == gold, s == gold
    n = gold.size
    d = (ca & ~cb).sum() + (~ca & cb).sum()
    diff = ca.mean() - cb.mean()
    se = math.sqrt((d / n - diff**2) / n)
    assert ci.low == pytest.approx(diff - 1.96 * se, abs=0.0015)
    assert ci.high == pytest.approx(diff + 1.96 * se, abs=0.0015)


def test_bootstrap_macro_f1_path_runs():
    gold, t, s = make_paired(400, 8, 0.85, 0.15, 0.03, seed=5)
    ci = paired_bootstrap(gold, t, s, metric="macro_f1", n_classes=8, n_boot=300)
    assert ci.metric == "macro_f1" and ci.low <= ci.high


def test_bootstrap_coverage_is_near_nominal():
    # Over many synthetic datasets the 95% CI should contain the true gap about 95% of the time.
    true_diff, hits, sims = 0.01, 0, 200
    for i in range(sims):
        gold, t, s = make_paired(1000, 10, 0.9, 0.1, true_diff, seed=1000 + i)
        ci = paired_bootstrap(gold, t, s, n_boot=1000, seed=i)
        hits += ci.low <= true_diff <= ci.high
    assert 0.90 <= hits / sims <= 0.99


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([0, 1]), np.array([0]), np.array([0, 1]))
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))


def test_non_inferiority_rule():
    ci = BootstrapCI("accuracy", 0.005, -0.005, 0.015, 0.95, 1000)
    assert non_inferiority(ci, 0.02).passed
    assert not non_inferiority(ci, 0.015).passed  # strict inequality at the boundary
    with pytest.raises(ValueError):
        non_inferiority(ci, 0.0)
