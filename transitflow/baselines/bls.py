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
                "best_t0": float(res.transit_time[i]),
                "best_duration": float(res.duration[i]),
                "power": power, "periods": periods,
                "transit_times": np.asarray(res.transit_time, dtype=float),
                "trial_durations": np.asarray(res.duration, dtype=float)}
    return _bls_native(times, flux, periods, durations)


def bls_top_candidates(
    times: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 13.0,
    n_periods: int = 2000,
    durations: np.ndarray | None = None,
    top_k: int = 3,
    min_log_period_separation: float = 0.025,
) -> list[dict]:
    """Return distinct BLS hypotheses, not only the strongest grid point.

    Nearby points on a period grid describe the same hypothesis. We suppress
    only those local duplicates in log-period; harmonics remain separate so a
    downstream vetter can adjudicate them. The score is a within-light-curve
    SDE, never a calibrated probability.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if min_log_period_separation < 0:
        raise ValueError("min_log_period_separation must be non-negative")
    result = bls_detect(
        times, flux, period_min=period_min, period_max=period_max,
        n_periods=n_periods, durations=durations,
    )
    power = np.asarray(result["power"], dtype=float)
    periods = np.asarray(result["periods"], dtype=float)
    trial_t0 = np.asarray(result["transit_times"], dtype=float)
    trial_duration = np.asarray(result["trial_durations"], dtype=float)
    if not (len(power) == len(periods) == len(trial_t0) == len(trial_duration)):
        raise RuntimeError("BLS trial arrays have inconsistent lengths")
    finite = np.isfinite(power) & np.isfinite(periods) & (periods > 0)
    scale = float(np.std(power[finite])) if finite.any() else 0.0
    center = float(np.median(power[finite])) if finite.any() else 0.0
    accepted: list[int] = []
    for idx in np.argsort(np.where(finite, power, -np.inf))[::-1]:
        if not finite[idx]:
            continue
        log_period = float(np.log(periods[idx]))
        if any(abs(log_period - float(np.log(periods[j]))) < min_log_period_separation
               for j in accepted):
            continue
        accepted.append(int(idx))
        if len(accepted) == top_k:
            break
    return [
        {
            "rank": rank,
            "score": float((power[idx] - center) / scale) if scale > 0 else 0.0,
            "peak_power": float(power[idx]),
            "best_period": float(periods[idx]),
            "best_t0": float(trial_t0[idx]),
            "best_duration": float(trial_duration[idx]),
        }
        for rank, idx in enumerate(accepted, start=1)
    ]


def _bls_native(times, flux, periods, durations) -> dict:
    """Minimal pure-numpy BLS fallback (peak depth-significance over the grid)."""
    flux = flux - np.median(flux)
    best_power, best_p = -np.inf, periods[0]
    best_t0, best_duration = float(times[0]), float(durations[0])
    powers = np.empty(len(periods))
    trial_t0 = np.empty(len(periods))
    trial_duration = np.empty(len(periods))
    for k, P in enumerate(periods):
        phase = (times / P) % 1.0
        order = np.argsort(phase)
        ph, fl = phase[order], flux[order]
        best_here = 0.0
        best_here_t0 = float(times[0])
        best_here_duration = float(durations[0])
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
                if snr > best_here:
                    best_here = float(snr)
                    best_here_t0 = float(c * P)
                    best_here_duration = float(dur)
        powers[k] = best_here
        trial_t0[k] = best_here_t0
        trial_duration[k] = best_here_duration
        if best_here > best_power:
            best_power, best_p = best_here, P
            best_t0, best_duration = best_here_t0, best_here_duration
    return {"score": _sde(powers), "peak_power": float(best_power),
            "best_period": float(best_p),
            "best_t0": float(best_t0),
            "best_duration": float(best_duration),
            "power": powers, "periods": periods,
            "transit_times": trial_t0, "trial_durations": trial_duration}


def has_astropy() -> bool:
    return _HAS_ASTROPY
