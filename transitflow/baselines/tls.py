"""Transit Least Squares detection baseline."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from transitleastsquares import transitleastsquares  # type: ignore
    _HAS_TLS = True
except Exception:  # pragma: no cover
    transitleastsquares = None
    _HAS_TLS = False


def has_tls() -> bool:
    return _HAS_TLS


def tls_detect(times: np.ndarray, flux: np.ndarray, periods: np.ndarray) -> dict:
    """Run Transit Least Squares and return the peak detection score."""
    if not _HAS_TLS:
        raise RuntimeError("transitleastsquares is not installed")
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    periods = np.asarray(periods, dtype=float)
    ok = np.isfinite(times) & np.isfinite(flux)
    times = times[ok]
    flux = flux[ok]
    if times.size < 10:
        return {"score": 0.0, "best_period": float("nan")}
    model = transitleastsquares(times, flux)
    res = model.power(
        period_min=float(periods.min()),
        period_max=float(periods.max()),
        n_transits_min=2,
        show_progress_bar=False,
    )
    score = getattr(res, "SDE", None)
    if score is None:
        score = getattr(res, "snr", 0.0)
    return {
        "score": float(score),
        "best_period": float(getattr(res, "period", float("nan"))),
    }
