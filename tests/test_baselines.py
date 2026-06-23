import numpy as np

from transitflow.baselines.bls import bls_detect
from transitflow.baselines.mcmc import run_mcmc
from transitflow.priors import TransitPrior, kipping_to_quadratic
from transitflow.transit_model import transit_flux


def _make_lc(P=3.0, t0=1.0, RpRs=0.1, aRs=12.0, b=0.2, sigma=0.001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27, 8000)
    f = transit_flux(t, P, t0, RpRs, aRs, b, 0.3, 0.2, engine="native")[0]
    f = f + sigma * rng.standard_normal(t.size)
    return t, f


def test_bls_recovers_period():
    t, f = _make_lc(P=3.0)
    res = bls_detect(t, f, period_min=0.5, period_max=6.0, n_periods=1500)
    # best period near 3 d (or a low harmonic)
    ratios = [res["best_period"] / 3.0, 3.0 / res["best_period"]]
    assert any(abs(r - round(r)) < 0.05 for r in ratios)


def test_bls_scores_planet_above_noise():
    t, f_planet = _make_lc(P=3.0, RpRs=0.12, sigma=0.0008)
    rng = np.random.default_rng(1)
    f_noise = 1.0 + 0.0008 * rng.standard_normal(t.size)
    s_planet = bls_detect(t, f_planet, n_periods=800)["score"]
    s_noise = bls_detect(t, f_noise, n_periods=800)["score"]
    assert s_planet > s_noise


def test_mcmc_recovers_depth_roughly():
    """A short MCMC fit to a clean, well-sampled transit brackets the truth."""
    P, t0p, RpRs, aRs, b = 3.0, 0.33, 0.1, 12.0, 0.2
    prior = TransitPrior()
    rng = np.random.default_rng(0)
    t = np.linspace(0, 9, 3000)
    u1, u2 = kipping_to_quadratic(0.4, 0.3)
    f = transit_flux(t, P, t0p * P, RpRs, aRs, b, u1, u2, engine="native")[0]
    f = f + 0.0005 * rng.standard_normal(t.size)
    init = np.array([P, t0p, RpRs, aRs, b, 0.4, 0.3])
    out = run_mcmc(t, f, 0.0005, prior=prior, init=init, n_walkers=24,
                   n_steps=400, n_radial=80, seed=0)
    samples = out["samples"]
    assert samples.shape[1] == 7
    # RpRs posterior should bracket the true value
    lo, hi = np.percentile(samples[:, 2], [2, 98])
    assert lo < RpRs < hi
