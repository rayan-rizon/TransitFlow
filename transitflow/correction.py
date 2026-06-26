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

The raw light curve (not the period-blurred binned views) is used for the
likelihood, which is what makes the correction sharpen the *period*.
"""

from __future__ import annotations

import numpy as np

from .priors import kipping_to_quadratic
from .transit_model import transit_flux


def render_raw_flux(theta_phys: np.ndarray, times: np.ndarray, n_radial: int = 200,
                    engine: str = "native") -> np.ndarray:
    """Render noiseless raw light curves for a batch of parameter vectors.

    ``theta_phys`` is ``(N, 7)`` = (P, t0_phase, Rp/Rs, a/Rs, b, q1, q2).
    Returns ``(N, len(times))``.
    """
    P = theta_phys[:, 0]
    t0 = theta_phys[:, 1] * P
    RpRs, aRs, b = theta_phys[:, 2], theta_phys[:, 3], theta_phys[:, 4]
    u1, u2 = kipping_to_quadratic(theta_phys[:, 5], theta_phys[:, 6])
    return transit_flux(times, P, t0, RpRs, aRs, b, u1, u2,
                        n_radial=n_radial, engine=engine)


def importance_weights(inference, global_view, local_view, sigma_feat,
                       raw_flux: np.ndarray, times: np.ndarray, sigma: float,
                       n_samples: int = 1000, logprob_steps: int = 40,
                       periodogram=None, ephem_feat=None) -> dict:
    """Importance-sampling weights for one object's amortized posterior.

    Returns physical + standardized proposal samples, normalized weights ``w``,
    and the ESS fraction. ``sigma`` may be a scalar per-cadence white-noise std
    or an array of per-point errors for binned likelihoods.
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
    pred = render_raw_flux(phys, times, n_radial=inf.sim_cfg.n_radial,
                           engine=inf.sim_cfg.engine)               # (N, n_raw)
    resid = raw_flux[None, :] - pred
    sigma = np.asarray(sigma, dtype=np.float64)
    loglik = -0.5 * np.sum(resid * resid / (sigma[None, :] * sigma[None, :])
                           if sigma.ndim else resid * resid / float(sigma * sigma),
                           axis=1)
    logw = loglik + logprior - logq
    logw = np.where(np.isfinite(logw), logw, -np.inf)
    logw -= np.max(logw)
    w = np.exp(logw)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    ess = 1.0 / np.sum(w * w) / len(w)
    return {"phys": phys, "std": std, "w": w, "ess_fraction": float(ess)}


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
