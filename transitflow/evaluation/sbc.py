"""Simulation-Based Calibration (Talts 2018).

For a calibrated posterior, the rank of each true parameter within its posterior
samples is uniform over ``{0, ..., L}``.  Systematic deviations diagnose
miscalibration (over/under-confidence, bias).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sbc_ranks(theta_true: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    """Per-dimension SBC ranks.

    Parameters
    ----------
    theta_true:
        ``(N, D)`` true parameters.
    posterior_samples:
        ``(N, L, D)`` posterior samples.

    Returns
    -------
    ``(N, D)`` integer ranks in ``[0, L]``.
    """
    theta_true = np.asarray(theta_true)
    posterior_samples = np.asarray(posterior_samples)
    return (posterior_samples < theta_true[:, None, :]).sum(axis=1)


def sbc_uniformity(ranks: np.ndarray, n_bins: int = 20,
                   n_posterior: int | None = None) -> dict:
    """Per-dimension chi-square test of rank uniformity.

    Returns a dict with the chi-square statistic and p-value for each dim; small
    p-values indicate miscalibration.
    """
    ranks = np.asarray(ranks)
    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (N, D)")
    if np.any(ranks < 0):
        raise ValueError("SBC ranks must be non-negative")
    N, D = ranks.shape
    if n_posterior is None:
        if not ranks.size:
            raise ValueError("n_posterior is required for empty ranks")
        n_posterior = int(ranks.max())
    L = int(n_posterior)
    if L < 1:
        raise ValueError("n_posterior must be positive")
    if np.any(ranks > L):
        raise ValueError("SBC rank exceeds the declared posterior sample count")

    # Ranks are discrete-uniform on {0, ..., L}.  When L + 1 is not divisible
    # by n_bins, equal expected counts are incorrect because bins contain
    # different numbers of possible ranks.  Map each integer rank to a bin and
    # derive the exact expected mass from the number of supported rank values.
    n_bins = min(int(n_bins), L + 1)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    support = np.arange(L + 1, dtype=np.int64)
    support_bins = np.minimum((support * n_bins) // (L + 1), n_bins - 1)
    support_counts = np.bincount(support_bins, minlength=n_bins)
    expected = N * support_counts / float(L + 1)
    out = {
        "chi2": [],
        "pvalue": [],
        "n_bins": n_bins,
        "n_posterior": L,
        "rank_support_size": L + 1,
        "expected_counts": expected.tolist(),
        "histogram_counts": [],
    }
    for j in range(D):
        rank_bins = np.minimum(
            (ranks[:, j].astype(np.int64) * n_bins) // (L + 1), n_bins - 1)
        counts = np.bincount(rank_bins, minlength=n_bins)
        chi2 = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-9)))
        pval = float(stats.chi2.sf(chi2, df=n_bins - 1))
        out["chi2"].append(chi2)
        out["pvalue"].append(pval)
        out["histogram_counts"].append(counts.tolist())
    return out


def run_sbc(inference, simulator, n_sims: int = 500, n_posterior: int = 1000,
            batch_size: int = 128, rng: np.random.Generator | None = None) -> dict:
    """Generate planets, draw posteriors, and compute SBC ranks.

    Only ``d=1`` rows are used (the flow models ``p(theta | d=1, x)``).  Ranks
    are computed in the *standardized, unclipped* space (ranking the raw flow
    samples against the standardized truth) -- the canonical SBC space, and
    robust to the prior-box clipping applied when mapping samples back to
    physical units.  (Clipping is monotonic and leaves ranks of interior truths
    unchanged, so this matches physical-space ranks; it is the rigorous default.)
    """
    rng = np.random.default_rng() if rng is None else rng
    prior = inference.prior
    param_names = list(prior.names)
    target_slice = slice(None)
    if inference.model.cfg.param_dim == 5:
        param_names = param_names[2:]
        target_slice = slice(2, None)
    trues_phys, trues_std, ranks = [], [], []
    collected = 0
    while collected < n_sims:
        batch = simulator.simulate_batch(batch_size, rng)
        mask = batch.get("posterior_valid", batch["valid"])
        if not mask.any():
            continue
        g = batch["global"][mask]
        l = batch["local"][mask]
        sf = batch["sigma_feat"][mask]
        pg = batch["periodogram"][mask] if "periodogram" in batch else None
        eph = batch["ephem_feat"][mask] if "ephem_feat" in batch else None
        dil = batch["dil_feat"][mask] if "dil_feat" in batch else None
        tp = batch["theta_phys"][mask]
        # unclipped standardized samples + standardized truth -> proper ranks
        _, samples_std = inference.posterior_samples(
            g, l, sf, n_samples=n_posterior, return_std=True, periodogram=pg,
            ephem_feat=eph, dil_feat=dil)
        t_std = prior.physical_to_std(tp)
        r = sbc_ranks(t_std[:, target_slice], samples_std[:, :, target_slice])
        trues_phys.append(tp[:, target_slice])
        trues_std.append(t_std[:, target_slice])
        ranks.append(r)
        collected += mask.sum()
    trues_phys = np.concatenate(trues_phys)[:n_sims]
    ranks = np.concatenate(ranks)[:n_sims]
    return {"ranks": ranks, "theta_true": trues_phys,
            "theta_true_std": np.concatenate(trues_std)[:n_sims],
            "param_names": param_names,
            "uniformity": sbc_uniformity(ranks, n_posterior=n_posterior)}
