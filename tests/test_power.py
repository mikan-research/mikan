import pytest

from mikan.eval.power import analytic_power, min_detectable_margin, required_n, simulated_power


def test_power_when_equal_matches_hand_value():
    # se = sqrt(0.08 / 3080) = 0.005096; (0.02 / se) - 1.96 = 1.964 -> Phi = 0.975
    assert analytic_power(3080, 0.08, 0.02) == pytest.approx(0.975, abs=0.002)


def test_power_is_half_when_true_gap_is_one_z_away():
    # true diff 0.01: (0.01 / sqrt((0.08 - 1e-4) / 3080)) - 1.96 ~= 0 -> power ~ 0.5
    assert analytic_power(3080, 0.08, 0.02, true_diff=0.01) == pytest.approx(0.5, abs=0.01)


def test_power_at_margin_is_type_one_error():
    assert analytic_power(3080, 0.08, 0.02, true_diff=0.02) == pytest.approx(0.025, abs=0.001)


@pytest.mark.parametrize("true_diff", [0.0, 0.01, 0.02])
def test_simulated_matches_analytic(true_diff):
    a = analytic_power(3080, 0.08, 0.02, true_diff)
    s = simulated_power(3080, 0.08, 0.02, true_diff, n_sims=40_000, seed=1)
    assert s == pytest.approx(a, abs=0.02)


def test_min_detectable_margin_banking77():
    # (1.96 + 0.8416) * sqrt(0.08 / 3080) = 0.01428
    assert min_detectable_margin(3080, 0.08) == pytest.approx(0.01428, abs=1e-4)


def test_required_n_round_trip():
    n = required_n(0.08, 0.02)
    assert n == 1570
    assert analytic_power(n, 0.08, 0.02) == pytest.approx(0.8, abs=0.002)


def test_validation():
    with pytest.raises(ValueError):
        analytic_power(100, 0.05, 0.02, true_diff=0.06)
    with pytest.raises(ValueError):
        required_n(0.08, 0.01, true_diff=0.02)
    with pytest.raises(ValueError):
        analytic_power(0, 0.08, 0.02)
