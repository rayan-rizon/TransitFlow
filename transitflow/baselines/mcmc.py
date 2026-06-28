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
from ..transit_model import exposure_integrated_transit_flux

try:
    import emcee  # type: ignore

    _HAS_EMCEE = True
except Exception:  # pragma: no cover
    _HAS_EMCEE = False


def _log_likelihood(theta_phys: np.ndarray, times, flux, flux_err,
                    n_radial: int = 100, exposure_minutes: float = 0.0,
                    n_exposure_subsamples: int = 1,
                    dilution: float = 1.0) -> float:
    P, t0_phase, RpRs, aRs, b, q1, q2 = theta_phys
    u1, u2 = kipping_to_quadratic(q1, q2)
    model = exposure_integrated_transit_flux(
        times, P, t0_phase * P, RpRs, aRs, b, u1, u2,
        n_radial=n_radial, engine="native",
        exposure_days=exposure_minutes / (60.0 * 24.0),
        n_subsamples=n_exposure_subsamples,
    )[0]
    model = 1.0 + (model - 1.0) * float(dilution)
    resid = (flux - model) / flux_err
    return -0.5 * np.sum(resid ** 2)


def _log_prob(theta_phys: np.ndarray, times, flux, flux_err,
              prior: TransitPrior, n_radial: int, exposure_minutes: float,
              n_exposure_subsamples: int, dilution: float = 1.0,
              dilution_low: float = 0.5, dilution_high: float = 1.0,
              fit_dilution: bool = False) -> float:
    if fit_dilution and not (dilution_low <= dilution <= dilution_high):
        return -np.inf
    lp = float(prior.log_prob_physical(theta_phys[None, :])[0])
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta_phys, times, flux, flux_err, n_radial,
                                exposure_minutes, n_exposure_subsamples,
                                dilution=dilution)


def run_mcmc(times, flux, flux_err, prior: TransitPrior | None = None,
             init: np.ndarray | None = None, n_walkers: int = 32,
             n_steps: int = 2000, burn_frac: float = 0.5, n_radial: int = 100,
             seed: int = 0, fixed: dict[int, float] | None = None,
             init_std_jitter: float = 0.05, exposure_minutes: float = 0.0,
             n_exposure_subsamples: int = 1, fit_dilution: bool = False,
             dilution_low: float = 0.5, dilution_high: float = 1.0,
             init_dilution: float = 1.0) -> dict:
    """Sample the transit-fit posterior. Returns physical samples ``(M, 7)``."""
    prior = prior or TransitPrior()
    rng = np.random.default_rng(seed)
    flux_err = np.full_like(np.asarray(times, float), flux_err) \
        if np.isscalar(flux_err) else np.asarray(flux_err, float)
    dim = prior.dim
    fixed = dict(fixed or {})
    fixed_idx = sorted(fixed)
    free_idx = [i for i in range(dim) if i not in fixed]
    if fit_dilution:
        lo = float(min(dilution_low, dilution_high))
        hi = float(max(dilution_low, dilution_high))
        dilution_low, dilution_high = lo, hi
        if not np.isfinite(dilution_low) or not np.isfinite(dilution_high) \
                or dilution_high <= dilution_low:
            raise ValueError("invalid dilution bounds")

    if init is None:
        init = prior.sample(1, rng)[0]
    init = np.asarray(init, dtype=np.float64).copy()
    for i, v in fixed.items():
        init[i] = v
    z_low, z_high = prior.std_bounds
    init_std = np.clip(prior.physical_to_std(init[None, :])[0],
                       z_low + 1e-3, z_high - 1e-3)
    p0_std = np.tile(init_std[None, :], (n_walkers, 1))
    if free_idx:
        p0_std[:, free_idx] += init_std_jitter * \
            rng.standard_normal((n_walkers, len(free_idx)))
        p0_std[:, free_idx] = np.clip(p0_std[:, free_idx],
                                      z_low[free_idx] + 1e-3,
                                      z_high[free_idx] - 1e-3)
    for i, v in fixed.items():
        p0_std[:, i] = init_std[i]
    p0_full = prior.std_to_physical(p0_std)
    for i, v in fixed.items():
        p0_full[:, i] = v
    if fit_dilution:
        init_dilution = float(np.clip(init_dilution, dilution_low, dilution_high))
        p0_dilution = init_dilution + (dilution_high - dilution_low) * \
            init_std_jitter * rng.standard_normal((n_walkers, 1))
        p0_dilution = np.clip(p0_dilution, dilution_low, dilution_high)
        p0_full = np.concatenate([p0_full, p0_dilution], axis=1)
        free_idx.append(dim)

    def expand(theta_free: np.ndarray) -> np.ndarray:
        theta = np.concatenate([init.copy(), np.array([init_dilution])]) \
            if fit_dilution else init.copy()
        if free_idx:
            theta[free_idx] = theta_free
        for i, v in fixed.items():
            theta[i] = v
        return theta

    if not free_idx:
        return {"samples": np.tile(init[None, :], (max(n_walkers, 1), 1)),
                "backend": "fixed", "acceptance_fraction": float("nan"),
                "fixed": fixed, "dilution_samples": None}

    p0 = p0_full[:, free_idx]
    logp = lambda th: _log_prob(  # noqa: E731
        expand(th)[:dim], times, flux, flux_err, prior, n_radial,
        exposure_minutes, n_exposure_subsamples,
        dilution=expand(th)[dim] if fit_dilution else 1.0,
        dilution_low=dilution_low, dilution_high=dilution_high,
        fit_dilution=fit_dilution)

    if _HAS_EMCEE:
        sampler = emcee.EnsembleSampler(n_walkers, len(free_idx), logp)
        sampler.run_mcmc(p0, n_steps, progress=False)
        free_chain = sampler.get_chain(discard=int(burn_frac * n_steps), flat=True)
        acceptance = float(np.mean(sampler.acceptance_fraction))
    else:
        free_chain, acceptance = _native_ensemble(logp, p0, n_steps, burn_frac, rng)
    sample_chain = np.tile(
        np.concatenate([init, np.array([init_dilution])])[None, :]
        if fit_dilution else init[None, :],
        (free_chain.shape[0], 1))
    sample_chain[:, free_idx] = free_chain
    chain = sample_chain[:, :dim].copy()
    for i, v in fixed.items():
        chain[:, i] = v
    dilution_samples = sample_chain[:, dim].copy() if fit_dilution else None
    return {"samples": chain, "backend": "emcee" if _HAS_EMCEE else "native",
            "acceptance_fraction": acceptance, "fixed": fixed,
            "dilution_samples": dilution_samples}


def _native_ensemble(logp, p0, n_steps, burn_frac, rng,
                     a: float = 2.0) -> tuple[np.ndarray, float]:
    """Affine-invariant stretch-move ensemble sampler (Goodman & Weare 2010)."""
    n_walkers, dim = p0.shape
    pos = p0.copy()
    lnp = np.array([logp(pos[w]) for w in range(n_walkers)])
    half = n_walkers // 2
    keep = []
    proposals = 0
    accepts = 0
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
                proposals += 1
                if np.isfinite(lp):
                    log_accept = (dim - 1) * np.log(z) + lp - lnp[i]
                    if np.log(rng.random()) < log_accept:
                        pos[i] = prop
                        lnp[i] = lp
                        accepts += 1
        if step >= burn_frac * n_steps:
            keep.append(pos.copy())
    chain = np.concatenate(keep, axis=0) if keep else pos.copy()
    return chain, float(accepts / max(proposals, 1))


def has_emcee() -> bool:
    return _HAS_EMCEE
