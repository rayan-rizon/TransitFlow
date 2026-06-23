"""Posterior quality metrics: NLL, contraction, and posterior agreement."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def negative_log_prob_true(inference, theta_true_phys: np.ndarray,
                           global_view, local_view, sigma_feat=None) -> np.ndarray:
    """Negative log posterior density of the true parameters (std space).

    Lower is better; a proper scoring rule sensitive to both bias and width.
    """
    prior = inference.prior
    theta_std = prior.physical_to_std(theta_true_phys)
    e = inference.embed(global_view, local_view, sigma_feat)
    lp = inference.log_prob_std(theta_std, e)
    return -lp


def posterior_contraction(theta_true: np.ndarray, posterior_samples: np.ndarray,
                          prior_std: np.ndarray) -> np.ndarray:
    """Per-dimension contraction ``1 - Var_post / Var_prior`` (in [~0, 1]).

    ``prior_std`` is the per-dimension prior standard deviation (physical space).
    """
    post_var = posterior_samples.var(axis=1)            # (N, D)
    prior_var = np.asarray(prior_std) ** 2
    return 1.0 - post_var / prior_var[None, :]


def marginal_wasserstein(samples_a: np.ndarray, samples_b: np.ndarray) -> np.ndarray:
    """Per-dimension 1-D Wasserstein distance between two posterior sample sets.

    Used to quantify agreement with MCMC / nested-sampling posteriors on real
    confirmed planets.
    """
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)
    D = samples_a.shape[-1]
    return np.array([
        wasserstein_distance(samples_a[:, j], samples_b[:, j]) for j in range(D)
    ])


def jensen_shannon_1d(samples_a: np.ndarray, samples_b: np.ndarray,
                      n_bins: int = 50) -> np.ndarray:
    """Per-dimension Jensen-Shannon divergence between 1-D marginals (histogrammed)."""
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)
    D = samples_a.shape[-1]
    out = np.zeros(D)
    for j in range(D):
        lo = min(samples_a[:, j].min(), samples_b[:, j].min())
        hi = max(samples_a[:, j].max(), samples_b[:, j].max())
        edges = np.linspace(lo, hi, n_bins + 1)
        pa, _ = np.histogram(samples_a[:, j], bins=edges, density=True)
        pb, _ = np.histogram(samples_b[:, j], bins=edges, density=True)
        pa = pa / max(pa.sum(), 1e-12)
        pb = pb / max(pb.sum(), 1e-12)
        m = 0.5 * (pa + pb)

        def _kl(p, q):
            mask = p > 0
            return np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12)))

        out[j] = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    return out
