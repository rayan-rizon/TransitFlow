"""Target-conditioned candidate search calibration.

Search statistics from different residual targets are not interchangeable.
This module calibrates a BLS maximum against moving-block bootstrap null
searches made from the same light curve after masking its leading candidate.
It exposes a finite-sample empirical false-alarm probability (FAP), rather
than presenting SDE as a probability.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .baselines.bls import bls_detect, bls_top_candidates


def empirical_upper_tail_fap(score: float, null_scores: np.ndarray) -> float:
    """Return the finite-sample, conservative upper-tail empirical FAP."""
    score = float(score)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if not np.isfinite(score):
        raise ValueError("observed score must be finite")
    if null_scores.size == 0:
        raise ValueError("at least one finite null score is required")
    return float((1 + np.count_nonzero(null_scores >= score)) /
                 (null_scores.size + 1))


def periodic_candidate_mask(
    times: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    width: float = 1.5,
) -> np.ndarray:
    """Mask candidate transit windows without using labels or source identity."""
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)
            and np.isfinite(duration) and duration > 0 and width > 0):
        raise ValueError("candidate ephemeris and mask width must be positive and finite")
    times = np.asarray(times, dtype=float)
    phase_time = (times - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase_time) <= 0.5 * width * duration


def fill_masked_flux(times: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly fill a candidate mask to construct a null residual series."""
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if times.ndim != 1 or flux.shape != times.shape or mask.shape != times.shape:
        raise ValueError("times, flux, and mask must be one-dimensional and aligned")
    valid = np.isfinite(times) & np.isfinite(flux)
    keep = valid & ~mask
    if keep.sum() < 2:
        raise ValueError("candidate mask leaves too few cadences for a null series")
    filled = flux.copy()
    missing = ~valid | mask
    filled[missing] = np.interp(times[missing], times[keep], flux[keep])
    return filled


def moving_block_bootstrap(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block resample, preserving within-block red-noise structure."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional series")
    if not 1 <= int(block_size) <= values.size:
        raise ValueError("block_size must lie between one and the series length")
    block_size = int(block_size)
    starts = rng.integers(0, values.size, size=int(np.ceil(values.size / block_size)))
    indices = np.concatenate([
        (start + np.arange(block_size)) % values.size for start in starts
    ])[:values.size]
    return values[indices]


def calibrate_bls_candidates(
    times: np.ndarray,
    flux: np.ndarray,
    *,
    period_min: float = 0.5,
    period_max: float = 13.0,
    n_periods: int = 2000,
    durations: np.ndarray | None = None,
    top_k: int = 3,
    n_null: int = 64,
    block_size: int = 128,
    mask_width: float = 1.5,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Search top-K hypotheses and calibrate the search maximum on-target.

    The bootstrap FAP belongs to the search event (the leading BLS maximum),
    not to an individual period. The returned candidates remain separate so a
    downstream vetter can score every non-duplicate hypothesis. This function
    is development-only until a source-disjoint calibration gate establishes
    its operating characteristics.
    """
    if n_null < 8:
        raise ValueError("n_null must be at least eight for an auditable FAP")
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(times) & np.isfinite(flux)
    times, flux = times[valid], flux[valid]
    if times.size < 10:
        raise ValueError("at least ten finite cadences are required")
    if block_size > times.size:
        raise ValueError("block_size exceeds available cadences")
    rng = np.random.default_rng() if rng is None else rng
    candidates = bls_top_candidates(
        times, flux, period_min=period_min, period_max=period_max,
        n_periods=n_periods, durations=durations, top_k=top_k,
    )
    if not candidates:
        raise RuntimeError("BLS returned no finite candidate")
    leading = candidates[0]
    mask = periodic_candidate_mask(
        times, leading["best_period"], leading["best_t0"],
        leading["best_duration"], width=mask_width,
    )
    filled = fill_masked_flux(times, flux, mask)
    center = float(np.median(filled))
    residual = filled - center
    null_scores = np.empty(n_null, dtype=float)
    for i in range(n_null):
        null_flux = center + moving_block_bootstrap(residual, block_size, rng)
        null_scores[i] = float(bls_detect(
            times, null_flux, period_min=period_min, period_max=period_max,
            n_periods=n_periods, durations=durations,
        )["score"])
    fap = empirical_upper_tail_fap(leading["score"], null_scores)
    for candidate in candidates:
        candidate["target_search_fap"] = fap
    return {
        "candidates": candidates,
        "target_search_fap": fap,
        "null_scores": null_scores,
        "masked_fraction": float(mask.mean()),
        "protocol": {
            "null_method": "candidate-masked circular moving-block bootstrap",
            "n_null": int(n_null),
            "block_size": int(block_size),
            "mask_width": float(mask_width),
            "top_k": int(top_k),
            "uses_source_identity": False,
        },
    }
