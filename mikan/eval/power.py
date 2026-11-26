"""Power analysis for the paired accuracy non-inferiority test.

Model: each item falls into one of three groups. With disagreement rate ``d``
(share of items where exactly one system is right) and true accuracy gap
``true_diff`` (teacher minus student):

    P(teacher right, student wrong) = (d + true_diff) / 2
    P(student right, teacher wrong) = (d - true_diff) / 2
    P(both agree on correctness)    = 1 - d

The estimated gap has standard error sqrt((d - true_diff**2) / n). Only the
disagreement rate matters, not the overall accuracy.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _validate(n: int | None, disagreement: float, true_diff: float) -> None:
    if n is not None and n <= 0:
        raise ValueError("n must be positive")
    if not 0 < disagreement <= 1:
        raise ValueError("disagreement must be in (0, 1]")
    if abs(true_diff) > disagreement:
        raise ValueError("|true_diff| cannot exceed the disagreement rate")


def _z(confidence: float) -> float:
    return float(norm.ppf(1 - (1 - confidence) / 2))


def analytic_power(
    n: int,
    disagreement: float,
    delta: float,
    true_diff: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Probability that the upper CI bound falls below ``delta`` (normal approximation)."""
    _validate(n, disagreement, true_diff)
    se = math.sqrt((disagreement - true_diff**2) / n)
    if se == 0:
        return float(true_diff < delta)
    return float(norm.cdf((delta - true_diff) / se - _z(confidence)))


def simulated_power(
    n: int,
    disagreement: float,
    delta: float,
    true_diff: float = 0.0,
    confidence: float = 0.95,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Monte Carlo power using the Wald CI for a paired accuracy difference."""
    _validate(n, disagreement, true_diff)
    p_a_only = (disagreement + true_diff) / 2
    p_b_only = (disagreement - true_diff) / 2
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, [p_a_only, p_b_only, 1 - disagreement], size=n_sims)
    a_only, b_only = counts[:, 0], counts[:, 1]
    diff_hat = (a_only - b_only) / n
    var = np.clip(((a_only + b_only) / n - diff_hat**2) / n, 0, None)
    upper = diff_hat + _z(confidence) * np.sqrt(var)
    return float(np.mean(upper < delta))


def min_detectable_margin(
    n: int,
    disagreement: float,
    power: float = 0.8,
    confidence: float = 0.95,
) -> float:
    """Smallest margin the test passes with the given power when the systems are truly equal."""
    _validate(n, disagreement, 0.0)
    return (_z(confidence) + float(norm.ppf(power))) * math.sqrt(disagreement / n)


def required_n(
    disagreement: float,
    delta: float,
    true_diff: float = 0.0,
    power: float = 0.8,
    confidence: float = 0.95,
) -> int:
    """Test-set size needed to pass with the given power at the assumed true gap."""
    _validate(None, disagreement, true_diff)
    if delta <= true_diff:
        raise ValueError("delta must exceed true_diff, otherwise the test cannot pass reliably")
    z_total = _z(confidence) + float(norm.ppf(power))
    return math.ceil(z_total**2 * (disagreement - true_diff**2) / (delta - true_diff) ** 2)
