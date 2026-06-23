"""Expected coverage probability of posterior credible intervals."""

from __future__ import annotations

import numpy as np


def central_interval_coverage(theta_true: np.ndarray, posterior_samples: np.ndarray,
                              levels: np.ndarray | None = None) -> dict:
    """Empirical coverage of central credible intervals vs nominal level.

    For each nominal level ``alpha`` the central ``alpha`` interval per dimension
    is ``[quantile((1-alpha)/2), quantile((1+alpha)/2)]``; coverage is the
    fraction of true values that fall inside.

    Returns nominal levels and per-dimension + overall empirical coverage.
    """
    theta_true = np.asarray(theta_true)
    posterior_samples = np.asarray(posterior_samples)
    N, L, D = posterior_samples.shape
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    cov = np.zeros((len(levels), D))
    for i, a in enumerate(levels):
        lo = np.quantile(posterior_samples, (1 - a) / 2, axis=1)   # (N, D)
        hi = np.quantile(posterior_samples, (1 + a) / 2, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        cov[i] = inside.mean(axis=0)
    return {"levels": np.asarray(levels), "coverage": cov,
            "coverage_overall": cov.mean(axis=1)}


def coverage_calibration_error(levels: np.ndarray, coverage_overall: np.ndarray) -> float:
    """Mean absolute deviation of empirical from nominal coverage (lower better)."""
    return float(np.mean(np.abs(coverage_overall - levels)))
