"""Transit-fit MCMC posterior baseline (the gold standard TransitFlow approximates).

A Gaussian-likelihood fit of the quadratic-LD transit model to a light curve,
sampled with an affine-invariant ensemble (Goodman & Weare 2010, the algorithm
behind ``emcee``).  ``emcee`` is used if installed; otherwise a compact native
stretch-move sampler is used so the baseline always runs.  Returns physical
posterior samples directly comparable to TransitFlow's amortized posterior.
"""

from __future__ import annotations

import numpy as np

from ..priors import TransitPrior, kipping_to_quadratic
from ..transit_model import transit_flux

try:
    import emcee  # type: ignore

    _HAS_EMCEE = True
except Exception:  # pragma: no cover
    _HAS_EMCEE = False


def _log_likelihood(theta_phys: np.ndarray, times, flux, flux_err,
                    n_radial: int = 100) -> float:
    P, t0_phase, RpRs, aRs, b, q1, q2 = theta_phys
    u1, u2 = kipping_to_quadratic(q1, q2)
    model = transit_flux(times, P, t0_phase * P, RpRs, aRs, b, u1, u2,
                         n_radial=n_radial, engine="native")[0]
    resid = (flux - model) / flux_err
    return -0.5 * np.sum(resid ** 2)


def _log_prob(theta_phys: np.ndarray, times, flux, flux_err,
              prior: TransitPrior, n_radial: int) -> float:
    lp = float(prior.log_prob_physical(theta_phys[None, :])[0])
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta_phys, times, flux, flux_err, n_radial)


def run_mcmc(times, flux, flux_err, prior: TransitPrior | None = None,
             init: np.ndarray | None = None, n_walkers: int = 32,
             n_steps: int = 2000, burn_frac: float = 0.5, n_radial: int = 100,
             seed: int = 0) -> dict:
    """Sample the transit-fit posterior. Returns physical samples ``(M, 7)``."""
    prior = prior or TransitPrior()
    rng = np.random.default_rng(seed)
    flux_err = np.full_like(np.asarray(times, float), flux_err) \
        if np.isscalar(flux_err) else np.asarray(flux_err, float)
    dim = prior.dim

    if init is None:
        init = prior.sample(1, rng)[0]
    p0 = init[None, :] + 1e-3 * rng.standard_normal((n_walkers, dim)) * \
        np.maximum(np.abs(init), 1e-3)
    # keep walkers inside support
    z_low, z_high = prior.std_bounds
    for w in range(n_walkers):
        std = prior.physical_to_std(p0[w][None, :])[0]
        std = np.clip(std, z_low + 1e-3, z_high - 1e-3)
        p0[w] = prior.std_to_physical(std[None, :])[0]

    logp = lambda th: _log_prob(th, times, flux, flux_err, prior, n_radial)  # noqa

    if _HAS_EMCEE:
        sampler = emcee.EnsembleSampler(n_walkers, dim, logp)
        sampler.run_mcmc(p0, n_steps, progress=False)
        chain = sampler.get_chain(discard=int(burn_frac * n_steps), flat=True)
    else:
        chain = _native_ensemble(logp, p0, n_steps, burn_frac, rng)
    return {"samples": chain, "backend": "emcee" if _HAS_EMCEE else "native"}


def _native_ensemble(logp, p0, n_steps, burn_frac, rng, a: float = 2.0) -> np.ndarray:
    """Affine-invariant stretch-move ensemble sampler (Goodman & Weare 2010)."""
    n_walkers, dim = p0.shape
    pos = p0.copy()
    lnp = np.array([logp(pos[w]) for w in range(n_walkers)])
    half = n_walkers // 2
    keep = []
    for step in range(n_steps):
        for grp in (0, 1):
            s = slice(0, half) if grp == 0 else slice(half, n_walkers)
            comp = slice(half, n_walkers) if grp == 0 else slice(0, half)
            comp_pos = pos[comp]
            idxs = np.arange(s.start, s.stop)
            for i in idxs:
                j = comp_pos[rng.integers(0, comp_pos.shape[0])]
                z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
                prop = j + z * (pos[i] - j)
                lp = logp(prop)
                if np.isfinite(lp):
                    log_accept = (dim - 1) * np.log(z) + lp - lnp[i]
                    if np.log(rng.random()) < log_accept:
                        pos[i] = prop
                        lnp[i] = lp
        if step >= burn_frac * n_steps:
            keep.append(pos.copy())
    return np.concatenate(keep, axis=0) if keep else pos.copy()


def has_emcee() -> bool:
    return _HAS_EMCEE
