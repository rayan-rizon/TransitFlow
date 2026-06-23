"""Quadratic limb-darkened transit light-curve model.

Two engines are provided:

* ``"native"`` -- a fully vectorized NumPy implementation that integrates the
  occulted, limb-darkened stellar flux numerically over concentric annuli.  It
  has no compiled dependency, runs on whole batches at once, and is the default
  used to generate millions of training light curves.
* ``"batman"`` -- a thin wrapper around the reference ``batman`` package
  (Kreidberg 2015) used as ground truth.  ``tests/test_transit_model.py``
  asserts the native engine agrees with ``batman`` to < 1e-3 in relative flux.

Orbit convention (circular, e = 0): ``t0`` is the inferior conjunction
(mid-transit).  Only the *primary* transit dims the star; near the secondary
eclipse the model returns 1.0 (no planet emission), matching ``batman`` with
``fp = 0``.
"""

from __future__ import annotations

import numpy as np

try:  # optional reference engine
    import batman  # type: ignore

    _HAS_BATMAN = True
except Exception:  # pragma: no cover - batman is optional
    _HAS_BATMAN = False

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Limb-darkened occultation (native engine core)
# --------------------------------------------------------------------------- #
def _occulted_fraction(z: np.ndarray, p: np.ndarray, u1: np.ndarray,
                       u2: np.ndarray, n_radial: int) -> np.ndarray:
    """Fraction of the total stellar flux blocked by the planet.

    Parameters are 1-D arrays of equal length ``N`` (already masked to the
    in-transit subset by the caller).  Returns an ``(N,)`` array of obscured
    flux fractions in ``[0, 1]``.

    The star is integrated over ``n_radial`` annuli of radius ``r_k``; at each
    annulus the covered angular fraction ``f_k`` of the planet disk (radius
    ``p``, centre distance ``z``) is computed in closed form, weighted by the
    quadratic limb-darkening intensity ``I(r_k)`` and the annulus area.
    """
    N = z.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.float64)

    # annulus midpoints r in (0, 1)
    r = (np.arange(n_radial, dtype=np.float64) + 0.5) / n_radial  # (Nr,)
    dr = 1.0 / n_radial
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))                  # (Nr,)
    intensity = 1.0 - u1[:, None] * (1.0 - mu) - u2[:, None] * (1.0 - mu) ** 2  # (N,Nr)
    ring_weight = 2.0 * np.pi * r * dr                            # (Nr,)

    total = np.sum(intensity * ring_weight, axis=1)               # (N,)

    z = z[:, None]
    p = p[:, None]
    r_ = r[None, :]
    # covered angular fraction of each ring
    eps = 1e-12
    denom = 2.0 * r_ * z + eps
    c0 = (r_ * r_ + z * z - p * p) / denom
    frac = np.where(
        c0 <= -1.0, 1.0,
        np.where(c0 >= 1.0, 0.0, np.arccos(np.clip(c0, -1.0, 1.0)) / np.pi),
    )
    # central-transit limit z -> 0: ring fully covered iff r <= p
    central = z[:, 0] < eps
    if np.any(central):
        frac[central] = (r_[0] <= p[central, 0][:, None]).astype(np.float64)

    obscured = np.sum(intensity * ring_weight * frac, axis=1)     # (N,)
    return obscured / total


def transit_flux(
    times: np.ndarray,
    P: np.ndarray,
    t0: np.ndarray,
    RpRs: np.ndarray,
    aRs: np.ndarray,
    b: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    n_radial: int = 200,
    engine: str = "native",
) -> np.ndarray:
    """Compute normalized transit flux.

    Parameters
    ----------
    times:
        Either a shared 1-D grid ``(T,)`` or a per-sample grid ``(B, T)``.
    P, t0, RpRs, aRs, b, u1, u2:
        Scalars or ``(B,)`` arrays of transit parameters (quadratic LD ``u1,u2``).
    n_radial:
        Number of annuli for the native limb-darkening integral.
    engine:
        ``"native"`` (default) or ``"batman"``.

    Returns
    -------
    flux : ``(B, T)`` array, normalized to 1.0 out of transit.
    """
    P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    t0 = np.atleast_1d(np.asarray(t0, dtype=np.float64))
    RpRs = np.atleast_1d(np.asarray(RpRs, dtype=np.float64))
    aRs = np.atleast_1d(np.asarray(aRs, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    u1 = np.atleast_1d(np.asarray(u1, dtype=np.float64))
    u2 = np.atleast_1d(np.asarray(u2, dtype=np.float64))
    B = max(len(P), len(t0), len(RpRs), len(aRs), len(b), len(u1), len(u2))

    def _b(a):
        return np.broadcast_to(a, (B,)).astype(np.float64)

    P, t0, RpRs, aRs, b, u1, u2 = map(_b, (P, t0, RpRs, aRs, b, u1, u2))

    if engine == "batman":
        return _transit_flux_batman(times, P, t0, RpRs, aRs, b, u1, u2)

    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 1:
        tt = times[None, :]                       # (1, T)
    else:
        tt = times                                # (B, T)
    ph = TWO_PI * (tt - t0[:, None]) / P[:, None]  # (B, T)
    cos_ph = np.cos(ph)
    sin_ph = np.sin(ph)
    cosi = b / aRs                                # (B,)
    z = aRs[:, None] * np.sqrt(sin_ph ** 2 + (cosi[:, None] * cos_ph) ** 2)  # (B,T)
    in_front = cos_ph > 0.0

    flux = np.ones_like(z)
    p_full = np.broadcast_to(RpRs[:, None], z.shape)
    # only points where the planet disk overlaps the star can dim it
    mask = in_front & (z < 1.0 + p_full)
    if np.any(mask):
        idx_b = np.broadcast_to(np.arange(B)[:, None], z.shape)[mask]
        obsc = _occulted_fraction(
            z[mask], p_full[mask], u1[idx_b], u2[idx_b], n_radial
        )
        flux[mask] = 1.0 - obsc
    return flux


def _transit_flux_batman(times, P, t0, RpRs, aRs, b, u1, u2) -> np.ndarray:
    if not _HAS_BATMAN:  # pragma: no cover
        raise RuntimeError("batman is not installed; use engine='native'")
    times = np.asarray(times, dtype=np.float64)
    B = len(P)
    out = []
    for i in range(B):
        t = times if times.ndim == 1 else times[i]
        pm = batman.TransitParams()
        pm.t0 = float(t0[i])
        pm.per = float(P[i])
        pm.rp = float(RpRs[i])
        pm.a = float(aRs[i])
        # inclination from impact parameter (circular orbit)
        cosi = np.clip(b[i] / aRs[i], -1.0, 1.0)
        pm.inc = float(np.degrees(np.arccos(cosi)))
        pm.ecc = 0.0
        pm.w = 90.0
        pm.u = [float(u1[i]), float(u2[i])]
        pm.limb_dark = "quadratic"
        m = batman.TransitModel(pm, t)
        out.append(m.light_curve(pm))
    return np.asarray(out)


def transit_duration(P, RpRs, aRs, b) -> np.ndarray:
    """Approximate total (T14) transit duration in days.

    Uses the standard small-angle expression
    ``T14 = (P/pi) * arcsin( sqrt((1+k)^2 - b^2) / (a/Rs * sin i) )``
    with ``sin i ~ 1``.  Grazing / non-transiting cases return 0.
    """
    P = np.asarray(P, dtype=np.float64)
    RpRs = np.asarray(RpRs, dtype=np.float64)
    aRs = np.asarray(aRs, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    arg = (1.0 + RpRs) ** 2 - b ** 2
    arg = np.where(arg > 0, arg, 0.0)
    sini = np.sqrt(np.clip(1.0 - (b / aRs) ** 2, 1e-8, 1.0))
    inner = np.clip(np.sqrt(arg) / (aRs * sini), 0.0, 1.0)
    return (P / np.pi) * np.arcsin(inner)


def has_batman() -> bool:
    return _HAS_BATMAN
