"""Dual-view light-curve representation (Shallue & Vanderburg 2018).

* **Global view** -- the whole light curve binned to a fixed length (default
  2001).  Captures the period spacing / multiple transits -> drives detection
  and period inference.
* **Local view** -- phase-folded on a candidate ephemeris ``(P, t0)`` and binned
  around the transit (default 201).  Captures depth / duration / shape -> drives
  characterization.

A real detection pipeline (BLS/TLS) proposes the candidate ephemeris used for
folding; at training time the candidate equals the true ephemeris for planets
(``d=1``) and a random spurious candidate for non-planets (``d=0``), mirroring
test-time behaviour where the network must accept or reject a folded candidate.

All functions are batched and operate on raw, evenly-or-unevenly sampled curves.
"""

from __future__ import annotations

import numpy as np


def _bin_statistic(x: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``values`` in bins defined by ``edges``; empty bins -> NaN.

    ``x, values`` are 1-D; ``edges`` has length ``n_bins + 1``.
    """
    idx = np.digitize(x, edges) - 1
    n_bins = len(edges) - 1
    valid = (idx >= 0) & (idx < n_bins)
    idx = idx[valid]
    v = values[valid]
    sums = np.bincount(idx, weights=v, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out[counts == 0] = np.nan
    return out


def _fill_nans(view: np.ndarray, fill: float = 1.0) -> np.ndarray:
    """Replace empty-bin NaNs by linear interpolation (edges -> ``fill``)."""
    out = view.copy()
    n = len(out)
    nan = np.isnan(out)
    if not nan.any():
        return out
    if nan.all():
        out[:] = fill
        return out
    good = np.where(~nan)[0]
    out[nan] = np.interp(np.where(nan)[0], good, out[good])
    return out


def global_view(times: np.ndarray, flux: np.ndarray, n_bins: int = 2001,
                t_min: float | None = None, t_max: float | None = None) -> np.ndarray:
    """Bin a single light curve to a fixed-length global view."""
    t_min = float(times.min()) if t_min is None else t_min
    t_max = float(times.max()) if t_max is None else t_max
    edges = np.linspace(t_min, t_max, n_bins + 1)
    binned = _bin_statistic(times, flux, edges)
    return _fill_nans(binned, fill=np.nanmedian(flux))


def local_view(times: np.ndarray, flux: np.ndarray, period: float, t0: float,
               duration: float, n_bins: int = 201,
               n_durations: float = 4.0) -> np.ndarray:
    """Phase-fold on ``(period, t0)`` and bin a transit-centered local view.

    The window spans ``±(n_durations/2)`` transit durations around phase 0.
    """
    phase = ((times - t0) / period + 0.5) % 1.0 - 0.5     # in [-0.5, 0.5]
    t_phase = phase * period                              # time from transit
    half_window = 0.5 * n_durations * max(duration, 1e-6)
    half_window = min(half_window, 0.5 * period)
    sel = np.abs(t_phase) <= half_window
    edges = np.linspace(-half_window, half_window, n_bins + 1)
    if sel.sum() < 2:
        return np.full(n_bins, np.nanmedian(flux))
    binned = _bin_statistic(t_phase[sel], flux[sel], edges)
    return _fill_nans(binned, fill=np.nanmedian(flux))


def normalize_view(view: np.ndarray, eps: float = 1e-8,
                   clip: float = 30.0) -> np.ndarray:
    """Robustly standardize a view: subtract median, scale by 1.4826*MAD.

    Out-of-transit flux maps to ~0 and a transit produces a negative excursion,
    in a scale-free representation that is stable across very different noise
    levels.

    Robustness: for a near-constant view (MAD ~ 0 -- e.g. a sparsely-populated
    folded window for a non-planet) dividing by ``eps`` would explode the output
    to huge values (overflowing float16 storage to +/-inf and feeding NaNs to the
    network).  We fall back to the std, then to 1.0, and clip the result to
    ``+/-clip`` -- standard outlier suppression for these views, and float16-safe.
    """
    med = np.median(view)
    mad = np.median(np.abs(view - med))
    scale = 1.4826 * mad
    if scale < eps:                      # near-constant view
        scale = float(view.std())
        if scale < eps:
            scale = 1.0
    return np.clip((view - med) / scale, -clip, clip)


def make_views(
    times: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_global: int = 2001,
    n_local: int = 201,
    n_durations: float = 4.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (global, local) views from one raw light curve."""
    g = global_view(times, flux, n_bins=n_global)
    l = local_view(times, flux, period, t0, duration, n_bins=n_local,
                   n_durations=n_durations)
    if normalize:
        g = normalize_view(g)
        l = normalize_view(l)
    return g.astype(np.float32), l.astype(np.float32)
