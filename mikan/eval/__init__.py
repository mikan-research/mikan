"""Evaluation harness: metrics, paired statistics, power analysis and reports.

Convention used everywhere: system ``a`` is the reference (teacher) and system
``b`` is the candidate (student). A difference is always ``metric(a) - metric(b)``,
so a positive difference means the student is worse.
"""

from .metrics import INVALID, METRICS, accuracy, macro_f1, validity_rate
from .power import analytic_power, min_detectable_margin, required_n, simulated_power
from .stats import BootstrapCI, McNemarResult, NonInferiority, mcnemar, non_inferiority, paired_bootstrap

__all__ = [
    "INVALID",
    "METRICS",
    "accuracy",
    "macro_f1",
    "validity_rate",
    "paired_bootstrap",
    "mcnemar",
    "non_inferiority",
    "BootstrapCI",
    "McNemarResult",
    "NonInferiority",
    "analytic_power",
    "simulated_power",
    "min_detectable_margin",
    "required_n",
]
