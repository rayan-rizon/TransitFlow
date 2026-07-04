"""Transit-fit MCMC posterior baseline (the gold standard TransitFlow approximates).

A Gaussian-likelihood fit of the quadratic-LD transit model to a light curve,
sampled with an affine-invariant ensemble (Goodman & Weare 2010, the algorithm
behind ``emcee``).  ``emcee`` is used if installed; otherwise a compact native
stretch-move sampler is used so the baseline always runs.  Returns physical
posterior samples directly comparable to TransitFlow's amortized posterior.

For real light curves the per-cadence white-noise sigma (point-to-point
estimator) underestimates the total residual variance whenever correlated
noise or detrending residuals remain, which makes the exact Gaussian
likelihood falsely peaked.  ``fit_jitter=True`` adds the standard remedy: a
per-object multiplicative error-inflation nuisance ``s`` (log-uniform prior),
so the likelihood is  -chi^2/(2 s^2) - N log s.  This is the same jitter
convention used by mainstream transit-fitting codes (juliet, exoplanet,
allesfitter).
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


_POOL_STATE = None


def _expand_free(theta_free: np.ndarray, init: np.ndarray,
                 init_dilution: float, fit_dilution: bool,
                 fixed: dict[int, float], free_idx: list[int],
                 dim: int, init_jitter: float = 1.0,
                 fit_jitter: bool = False) -> np.ndarray:
    extras = []
    if fit_dilution:
        extras.append(init_dilution)
    if fit_jitter:
        extras.append(init_jitter)
    theta = np.concatenate([init.copy(),
                            np.asarray(extras, dtype=np.float64)]) \
        if extras else init.copy()
    if free_idx:
        theta[free_idx] = theta_free
    for i, v in fixed.items():
        theta[i] = v
    return theta


def _pool_init(state: dict) -> None:
    global _POOL_STATE
    _POOL_STATE = state


def _pooled_log_prob(theta_free: np.ndarray) -> float:
    state = _POOL_STATE
    if state is None:  # pragma: no cover - defensive guard for worker setup
        raise RuntimeError("MCMC worker state was not initialized")
    theta = _expand_free(
        theta_free, state["init"], state["init_dilution"],
        state["fit_dilution"], state["fixed"], state["free_idx"], state["dim"],
        init_jitter=state["init_jitter"], fit_jitter=state["fit_jitter"])
    dim = state["dim"]
    jit_i = dim + (1 if state["fit_dilution"] else 0)
    return _log_prob(
        theta[:dim], state["times"], state["flux"], state["flux_err"],
        state["prior"], state["n_radial"], state["exposure_minutes"],
        state["n_exposure_subsamples"],
        dilution=theta[dim] if state["fit_dilution"] else 1.0,
        dilution_low=state["dilution_low"],
        dilution_high=state["dilution_high"],
        fit_dilution=state["fit_dilution"],
        jitter=theta[jit_i] if state["fit_jitter"] else 1.0,
        jitter_low=state["jitter_low"],
        jitter_high=state["jitter_high"],
        fit_jitter=state["fit_jitter"])


def _log_likelihood(theta_phys: np.ndarray, times, flux, flux_err,
                    n_radial: int = 100, exposure_minutes: float = 0.0,
                    n_exposure_subsamples: int = 1,
                    dilution: float = 1.0, jitter: float = 1.0) -> float:
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
    chi2 = float(np.sum(resid ** 2))
    if jitter == 1.0:
        return -0.5 * chi2
    return -0.5 * chi2 / (jitter * jitter) - resid.size * np.log(jitter)


def _log_prob(theta_phys: np.ndarray, times, flux, flux_err,
              prior: TransitPrior, n_radial: int, exposure_minutes: float,
              n_exposure_subsamples: int, dilution: float = 1.0,
              dilution_low: float = 0.5, dilution_high: float = 1.0,
              fit_dilution: bool = False, jitter: float = 1.0,
              jitter_low: float = 1.0, jitter_high: float = 10.0,
              fit_jitter: bool = False) -> float:
    if fit_dilution and not (dilution_low <= dilution <= dilution_high):
        return -np.inf
    if fit_jitter and not (jitter_low <= jitter <= jitter_high):
        return -np.inf
    lp = float(prior.log_prob_physical(theta_phys[None, :])[0])
    if not np.isfinite(lp):
        return -np.inf
    if fit_jitter:
        lp -= np.log(jitter)  # log-uniform (Jeffreys) prior on the scale
    return lp + _log_likelihood(theta_phys, times, flux, flux_err, n_radial,
                                exposure_minutes, n_exposure_subsamples,
                                dilution=dilution, jitter=jitter)


def run_mcmc(times, flux, flux_err, prior: TransitPrior | None = None,
             init: np.ndarray | None = None, n_walkers: int = 32,
             n_steps: int = 2000, burn_frac: float = 0.5, n_radial: int = 100,
             seed: int = 0, fixed: dict[int, float] | None = None,
             init_std_jitter: float = 0.05, exposure_minutes: float = 0.0,
             n_exposure_subsamples: int = 1, fit_dilution: bool = False,
             dilution_low: float = 0.5, dilution_high: float = 1.0,
             init_dilution: float = 1.0, n_processes: int = 1,
             fit_jitter: bool = False, jitter_low: float = 1.0,
             jitter_high: float = 10.0, init_jitter: float = 1.5) -> dict:
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
    if fit_jitter:
        jitter_low = float(max(min(jitter_low, jitter_high), 1e-3))
        jitter_high = float(max(jitter_low, jitter_high))
        if not np.isfinite(jitter_low) or not np.isfinite(jitter_high) \
                or jitter_high <= jitter_low:
            raise ValueError("invalid jitter bounds")

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
    jit_idx = dim + (1 if fit_dilution else 0)
    if fit_jitter:
        init_jitter = float(np.clip(init_jitter, jitter_low, jitter_high))
        # walker spread in log-space (the parameter is a scale)
        p0_jitter = np.exp(np.log(init_jitter) + 0.3 *
                           rng.standard_normal((n_walkers, 1)))
        p0_jitter = np.clip(p0_jitter, jitter_low, jitter_high)
        p0_full = np.concatenate([p0_full, p0_jitter], axis=1)
        free_idx.append(jit_idx)

    if not free_idx:
        return {"samples": np.tile(init[None, :], (max(n_walkers, 1), 1)),
                "backend": "fixed", "acceptance_fraction": float("nan"),
                "fixed": fixed, "dilution_samples": None,
                "jitter_samples": None, "autocorr_time_max": None,
                "n_eff": None}

    p0 = p0_full[:, free_idx]
    n_processes = max(1, int(n_processes))

    def logp(th):
        theta = _expand_free(th, init, init_dilution, fit_dilution, fixed,
                             free_idx, dim, init_jitter=init_jitter,
                             fit_jitter=fit_jitter)
        return _log_prob(
            theta[:dim], times, flux, flux_err, prior, n_radial,
            exposure_minutes, n_exposure_subsamples,
            dilution=theta[dim] if fit_dilution else 1.0,
            dilution_low=dilution_low, dilution_high=dilution_high,
            fit_dilution=fit_dilution,
            jitter=theta[jit_idx] if fit_jitter else 1.0,
            jitter_low=jitter_low, jitter_high=jitter_high,
            fit_jitter=fit_jitter)

    sampler = None
    if _HAS_EMCEE:
        if n_processes > 1:
            import multiprocessing as mp

            state = {
                "init": init,
                "init_dilution": init_dilution,
                "fit_dilution": fit_dilution,
                "fixed": fixed,
                "free_idx": free_idx,
                "dim": dim,
                "times": np.asarray(times, dtype=np.float64),
                "flux": np.asarray(flux, dtype=np.float64),
                "flux_err": np.asarray(flux_err, dtype=np.float64),
                "prior": prior,
                "n_radial": n_radial,
                "exposure_minutes": exposure_minutes,
                "n_exposure_subsamples": n_exposure_subsamples,
                "dilution_low": dilution_low,
                "dilution_high": dilution_high,
                "init_jitter": init_jitter,
                "fit_jitter": fit_jitter,
                "jitter_low": jitter_low,
                "jitter_high": jitter_high,
            }
            ctx = mp.get_context("fork")
            with ctx.Pool(n_processes, initializer=_pool_init,
                          initargs=(state,)) as pool:
                sampler = emcee.EnsembleSampler(
                    n_walkers, len(free_idx), _pooled_log_prob, pool=pool)
                sampler.run_mcmc(p0, n_steps, progress=False)
                free_chain = sampler.get_chain(discard=int(burn_frac * n_steps),
                                               flat=True)
                acceptance = float(np.mean(sampler.acceptance_fraction))
        else:
            sampler = emcee.EnsembleSampler(n_walkers, len(free_idx), logp)
            sampler.run_mcmc(p0, n_steps, progress=False)
            free_chain = sampler.get_chain(discard=int(burn_frac * n_steps),
                                           flat=True)
            acceptance = float(np.mean(sampler.acceptance_fraction))
    else:
        free_chain, acceptance = _native_ensemble(logp, p0, n_steps, burn_frac, rng)

    autocorr_time_max = None
    n_eff = None
    if sampler is not None:
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            tau = np.asarray(tau, dtype=np.float64)
            if np.all(np.isfinite(tau)) and tau.size:
                autocorr_time_max = float(np.max(tau))
                kept_steps = n_steps - int(burn_frac * n_steps)
                n_eff = float(kept_steps * n_walkers /
                              max(autocorr_time_max, 1.0))
        except Exception:
            pass

    extras_init = [init]
    if fit_dilution:
        extras_init.append(np.array([init_dilution]))
    if fit_jitter:
        extras_init.append(np.array([init_jitter]))
    sample_chain = np.tile(np.concatenate(extras_init)[None, :],
                           (free_chain.shape[0], 1))
    sample_chain[:, free_idx] = free_chain
    chain = sample_chain[:, :dim].copy()
    for i, v in fixed.items():
        chain[:, i] = v
    dilution_samples = sample_chain[:, dim].copy() if fit_dilution else None
    jitter_samples = sample_chain[:, jit_idx].copy() if fit_jitter else None
    return {"samples": chain, "backend": "emcee" if _HAS_EMCEE else "native",
            "acceptance_fraction": acceptance, "fixed": fixed,
            "dilution_samples": dilution_samples,
            "jitter_samples": jitter_samples,
            "autocorr_time_max": autocorr_time_max,
            "n_eff": n_eff}


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
