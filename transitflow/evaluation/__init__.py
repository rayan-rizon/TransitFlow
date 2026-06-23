"""Evaluation: SBC, coverage, detection metrics, posterior agreement."""

from .coverage import central_interval_coverage, coverage_calibration_error
from .detection import completeness_grid, detection_metrics
from .posterior_metrics import (
    jensen_shannon_1d,
    marginal_wasserstein,
    negative_log_prob_true,
    posterior_contraction,
)
from .sbc import run_sbc, sbc_ranks, sbc_uniformity

__all__ = [
    "central_interval_coverage",
    "coverage_calibration_error",
    "completeness_grid",
    "detection_metrics",
    "jensen_shannon_1d",
    "marginal_wasserstein",
    "negative_log_prob_true",
    "posterior_contraction",
    "run_sbc",
    "sbc_ranks",
    "sbc_uniformity",
]
