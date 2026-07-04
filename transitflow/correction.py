"""Importance-sampling correction of the amortized FMPE posterior.

The flow learns ``q(theta | x)`` amortized over the prior; in low-information
regimes (e.g. few-transit periods) it shrinks toward the prior mean and its
credible widths do not track the regime-dependent uncertainty, so simulation-
based calibration (SBC) fails for the period.  Following Gebhard et al. (2024),
we correct each amortized posterior by importance sampling against the simulator
likelihood:

    w_i  proportional to  p(x | theta_i) p(theta_i) / q(theta_i | x),
    theta_i ~ q(theta | x).

For the white-noise regime the likelihood ``p(x | theta)`` is exact (independent
Gaussian per cadence on the raw light curve), so the weighted/resampled posterior
targets the *true* posterior and restores calibration.  The normalized effective
sample size ``ESS/N`` doubles as a misspecification diagnostic: it collapses when
the simulator cannot reproduce the data.

On real light curves the point-to-point sigma underestimates the residual
variance (correlated noise, detrending residuals), so the exact white-noise
likelihood is falsely peaked and the weights degenerate — worst at high SNR.
``jitter_grid_size > 1`` marginalizes a per-object multiplicative
error-inflation scale ``s`` over a log-spaced grid (log-uniform prior),

    p(x | theta) = sum_s  w_s  exp(-chi^2(theta) / (2 s^2)) / s^N,

the grid analogue of the jitter nuisance every mainstream transit-fitting
code samples.  ``s = 1`` is always on the grid, so well-specified data
reduce to the exact likelihood.

When the amortized model *conditions* on dilution (``use_dilution_feature``)
the flow posterior is defined at the supplied dilution (1.0 on real targets
without CROWDSAP), so the correction likelihood must fix dilution too;
marginalizing it would target a different posterior than both the flow and
the like-for-like MCMC reference.  Dilution is only marginalized when the
model does not condition on it.

The raw light curve (not the period-blurred binned views) is used for the
likelihood, which is what makes the correction sharpen the *period*.
"""

from __future__ import annotations

import numpy as np

from .priors import kipping_to_quadratic
from .transit_model import exposure_integrated_transit_flux


def render_raw_flux(theta_phys: np.ndarray, times: np.ndarray, n_radial: int = 200,
                    engine: str = "native", exposure_minutes: float = 0.0,
                    n_exposure_subsamples: int = 1,
                    dilution: float | np.ndarray = 1.0) -> np.ndarray:
    """Render noiseless raw light curves for a batch of parameter vectors.

    ``theta_phys`` is ``(N, 7)`` = (P, t0_phase, Rp/Rs, a/Rs, b, q1, q2).
    Returns ``(N, len(times))``.
    """
    P = theta_phys[:, 0]
    t0 = theta_phys[:, 1] * P
    RpRs, aRs, b = theta_phys[:, 2], theta_phys[:, 3], theta_phys[:, 4]
    u1, u2 = kipping_to_quadratic(theta_phys[:, 5], theta_phys[:, 6])
    flux = exposure_integrated_transit_flux(
        times, P, t0, RpRs, aRs, b, u1, u2,
        n_radial=n_radial, engine=engine,
        exposure_days=exposure_minutes / (60.0 * 24.0),
        n_subsamples=n_exposure_subsamples,
    )
    dilution = np.asarray(dilution, dtype=np.float64)
    if dilution.ndim == 0:
        return 1.0 + (flux - 1.0) * float(dilution)
    return 1.0 + (flux - 1.0) * dilution[:, None]


def _gpd_khat(exceedances: np.ndarray) -> float:
    """Generalized-Pareto shape estimate (Zhang & Stephens 2009 profile fit)."""
    x = np.sort(np.asarray(exceedances, dtype=np.float64))
    n = x.size
    if n < 5 or x[-1] <= 0:
        return float("nan")
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    x_quart = x[max(int(n / 4 + 0.5) - 1, 0)]
    if x_quart <= 0:
        return float("nan")
    bs = bs / (3.0 * x_quart) + 1.0 / x[-1]
    ks = np.array([-np.mean(np.log1p(-b * x)) for b in bs])
    with np.errstate(divide="ignore", invalid="ignore"):
        L = n * (np.log(bs / ks) + ks - 1.0)
    L = np.where(np.isfinite(L), L, -np.inf)
    w = np.exp(L - np.max(L))
    w = w / w.sum()
    b_hat = float(np.sum(bs * w))
    return float(-np.mean(np.log1p(-b_hat * x)))


def psis_khat(logw: np.ndarray) -> float:
    """Pareto-smoothed-IS tail-shape diagnostic (Vehtari et al. 2024).

    k < 0.5: reliable; 0.5-0.7: usable; > 0.7: weights too heavy-tailed.
    Returns NaN when there are too few finite weights to fit the tail.
    """
    logw = np.asarray(logw, dtype=np.float64)
    logw = logw[np.isfinite(logw)]
    n = logw.size
    if n < 25:
        return float("nan")
    m = int(min(0.2 * n, 3.0 * np.sqrt(n)))
    if m < 5:
        return float("nan")
    srt = np.sort(logw)
    tail = np.exp(srt[-m:] - srt[-1])
    cutoff = np.exp(srt[-m - 1] - srt[-1])
    exceed = tail - cutoff
    exceed = exceed[exceed > 0]
    return _gpd_khat(exceed)


def importance_weights(inference, global_view, local_view, sigma_feat,
                       raw_flux: np.ndarray, times: np.ndarray, sigma: float,
                       n_samples: int = 1000, logprob_steps: int = 40,
                       periodogram=None, ephem_feat=None,
                       dilution_grid_size: int = 9,
                       jitter_grid_size: int = 25,
                       jitter_max: float = 10.0,
                       marginalize_dilution: bool | None = None) -> dict:
    """Importance-sampling weights for one object's amortized posterior.

    Returns physical + standardized proposal samples, normalized weights ``w``,
    the ESS fraction, and the PSIS ``khat`` tail diagnostic. ``sigma`` may be a
    scalar per-cadence white-noise std or an array of per-point errors for
    binned likelihoods. ``jitter_grid_size > 1`` marginalizes a per-object
    error-inflation scale in ``[1, jitter_max]`` (log-uniform); set
    ``jitter_grid_size = 1`` or ``jitter_max = 1`` to recover the exact
    white-noise likelihood.
    """
    inf = inference
    e = inf.embed(global_view, local_view, sigma_feat, periodogram, ephem_feat)
    phys, std = inf.posterior_samples(global_view, local_view, sigma_feat,
                                      n_samples=n_samples, return_std=True,
                                      periodogram=periodogram,
                                      ephem_feat=ephem_feat)
    phys, std = phys[0], std[0]                                     # (N, 7)
    logq = inf.log_prob_std(std, e.repeat(std.shape[0], 1), )       # (N,)
    logprior = inf.prior.log_prob_std(std)                         # (N,) const in box
    raw_flux = np.asarray(raw_flux, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    observed = np.isfinite(raw_flux) & np.isfinite(times)
    raw_flux = raw_flux[observed]
    times = times[observed]
    if sigma.ndim:
        sigma = sigma[observed]
    n_pts = raw_flux.size
    base_pred = render_raw_flux(
        phys, times, n_radial=inf.sim_cfg.n_radial, engine=inf.sim_cfg.engine,
        exposure_minutes=getattr(inf.sim_cfg, "exposure_minutes", 0.0),
        n_exposure_subsamples=getattr(inf.sim_cfg, "n_exposure_subsamples", 1))

    def _chi2(pred):
        resid = raw_flux[None, :] - pred
        return np.sum(
            resid * resid / (sigma[None, :] * sigma[None, :])
            if sigma.ndim else resid * resid / float(sigma * sigma),
            axis=1)

    dilution_fraction = float(np.clip(
        getattr(inf.sim_cfg, "dilution_fraction", 0.0), 0.0, 1.0))
    conditions_on_dilution = bool(
        getattr(inf.model.cfg, "use_dilution_feature", False))
    if marginalize_dilution is None:
        marginalize_dilution = (dilution_fraction > 0
                                and not conditions_on_dilution)

    # chi^2 branches over the dilution mixture (a single branch when the
    # model conditions on dilution, matching the fixed-dilution MCMC)
    chi2_terms = []
    branch_logw = []
    if marginalize_dilution and dilution_grid_size > 1:
        lo = min(float(getattr(inf.sim_cfg, "dilution_low", 0.5)),
                 float(getattr(inf.sim_cfg, "dilution_high", 1.0)))
        hi = max(float(getattr(inf.sim_cfg, "dilution_low", 0.5)),
                 float(getattr(inf.sim_cfg, "dilution_high", 1.0)))
        grid = np.linspace(lo, hi, int(dilution_grid_size))
        if dilution_fraction < 1.0:
            chi2_terms.append(_chi2(base_pred))
            branch_logw.append(np.log(1.0 - dilution_fraction))
        for d in grid:
            chi2_terms.append(_chi2(1.0 + (base_pred - 1.0) * d))
            branch_logw.append(np.log(dilution_fraction / len(grid)))
    else:
        chi2_terms.append(_chi2(base_pred))
        branch_logw.append(0.0)
    chi2_terms = np.stack(chi2_terms, axis=0)                 # (B, N)
    branch_logw = np.asarray(branch_logw, dtype=np.float64)   # (B,)

    # marginalize the error-inflation scale s on a log-spaced grid
    jitter_max = float(max(jitter_max, 1.0))
    if jitter_grid_size > 1 and jitter_max > 1.0:
        s_grid = np.exp(np.linspace(0.0, np.log(jitter_max),
                                    int(jitter_grid_size)))
    else:
        s_grid = np.array([1.0])
    s_logw = -np.log(len(s_grid))                              # uniform in log s
    log_terms = (
        -0.5 * chi2_terms[None, :, :] / (s_grid ** 2)[:, None, None]
        - n_pts * np.log(s_grid)[:, None, None]
        + branch_logw[None, :, None]
        + s_logw
    ).reshape(-1, chi2_terms.shape[-1])                        # (S*B, N)
    z = np.max(log_terms, axis=0)
    loglik = z + np.log(np.sum(np.exp(log_terms - z[None, :]), axis=0))

    logw = loglik + logprior - logq
    logw = np.where(np.isfinite(logw), logw, -np.inf)
    khat = psis_khat(logw)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    ess = 1.0 / np.sum(w * w) / len(w)
    # MAP error-inflation scale under the weighted posterior (diagnostic)
    chi2_min = chi2_terms[np.argmax(branch_logw)]
    s_prof = np.sqrt(np.average(chi2_min, weights=w) / max(n_pts, 1))
    return {"phys": phys, "std": std, "w": w, "ess_fraction": float(ess),
            "khat": float(khat), "n_points": int(n_pts),
            "jitter_grid": [float(s_grid[0]), float(s_grid[-1]),
                            int(len(s_grid))],
            "jitter_scale_profile": float(s_prof),
            "marginalized_dilution": bool(marginalize_dilution
                                          and dilution_grid_size > 1)}


def weighted_rank_cdf(std_samples: np.ndarray, w: np.ndarray,
                      theta_true_std: np.ndarray) -> np.ndarray:
    """Per-dimension weighted CDF of the truth under the corrected posterior.

    Returns values in ``[0, 1]`` (the SBC rank statistic for weighted samples);
    uniform over objects iff the corrected posterior is calibrated.
    """
    below = std_samples < theta_true_std[None, :]                  # (N, D)
    return (w[:, None] * below).sum(axis=0)


def sir_resample(phys: np.ndarray, w: np.ndarray, n_out: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Sampling-importance-resampling: draw ``n_out`` corrected posterior samples."""
    idx = rng.choice(len(w), size=n_out, replace=True, p=w)
    return phys[idx]
