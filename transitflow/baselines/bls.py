"""Box Least Squares detection baseline (Sec. 6.1)."""

from __future__ import annotations

import numpy as np

try:
    from astropy.timeseries import BoxLeastSquares  # type: ignore

    _HAS_ASTROPY = True
except Exception:  # pragma: no cover
    _HAS_ASTROPY = False


def _sde(power: np.ndarray) -> float:
    """Signal Detection Efficiency of the peak (Kovacs et al. 2002).

    Normalizes the peak against the power spectrum's own median/std, making
    the score comparable across light curves with heterogeneous noise (raw
    peak power is not: red-noise-dominated negatives outscore shallow
    transits, driving the ROC below chance).
    """
    power = np.asarray(power, dtype=float)
    power = power[np.isfinite(power)]
    if power.size < 3:
        return 0.0
    spread = float(np.std(power))
    if spread <= 0:
        return 0.0
    return float((np.max(power) - np.median(power)) / spread)


def bls_detect(times: np.ndarray, flux: np.ndarray,
               period_min: float = 0.5, period_max: float = 13.0,
               n_periods: int = 2000, durations: np.ndarray | None = None) -> dict:
    """Run BLS; the detection score is the peak SDE, not raw peak power."""
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if durations is None:
        durations = np.array([0.05, 0.1, 0.2])
    periods = np.linspace(period_min, period_max, n_periods)
    if _HAS_ASTROPY:
        bls = BoxLeastSquares(times, flux)
        res = bls.power(periods, durations)
        power = np.asarray(res.power)
        i = int(np.argmax(power))
        return {"score": _sde(power), "peak_power": float(power[i]),
                "best_period": float(res.period[i]),
                "power": power, "periods": periods}
    return _bls_native(times, flux, periods, durations)


def _bls_native(times, flux, periods, durations) -> dict:
    """Minimal pure-numpy BLS fallback (peak depth-significance over the grid)."""
    flux = flux - np.median(flux)
    best_power, best_p = -np.inf, periods[0]
    powers = np.empty(len(periods))
    for k, P in enumerate(periods):
        phase = (times / P) % 1.0
        order = np.argsort(phase)
        ph, fl = phase[order], flux[order]
        best_here = 0.0
        for dur in durations:
            w = dur / P
            n_steps = max(int(1.0 / max(w, 1e-3)), 4)
            for s in range(n_steps):
                c = s / n_steps
                inb = np.abs(((ph - c + 0.5) % 1.0) - 0.5) < (w / 2)
                if inb.sum() < 3 or (~inb).sum() < 3:
                    continue
                depth = fl[~inb].mean() - fl[inb].mean()
                snr = depth / (fl.std() / np.sqrt(max(inb.sum(), 1)) + 1e-9)
                best_here = max(best_here, snr)
        powers[k] = best_here
        if best_here > best_power:
            best_power, best_p = best_here, P
    return {"score": _sde(powers), "peak_power": float(best_power),
            "best_period": float(best_p),
            "power": powers, "periods": periods}


def has_astropy() -> bool:
    return _HAS_ASTROPY
