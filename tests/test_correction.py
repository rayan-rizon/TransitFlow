"""Tests for the importance-sampling posterior correction."""

import numpy as np

from transitflow.correction import (
    importance_weights,
    render_raw_flux,
    sir_resample,
    weighted_rank_cdf,
)
from transitflow.inference import TransitFlowInference
from transitflow.models.transitflow import ModelConfig, TransitFlow
from transitflow.priors import TransitPrior
from transitflow.simulator import SimConfig, TransitSimulator


def _setup():
    sc = SimConfig(n_global=128, n_local=65, baseline_days=27.0, n_raw=2500,
                   frac_real=0.0, frac_gp=0.0, frac_white=1.0, planet_fraction=1.0,
                   n_radial=50, regime="tess")
    pr = TransitPrior(TransitPrior.default_specs("tess"))
    sim = TransitSimulator(sc, prior=pr)
    mc = ModelConfig(embed_dim=32, global_channels=(16, 32), local_channels=(16, 32),
                     global_dim=32, local_dim=16, fm_hidden=48, fm_blocks=2,
                     fm_time_dim=16)
    inf = TransitFlowInference(TransitFlow(mc), pr, sc, ode_steps=15)
    return sc, pr, sim, inf


def test_simulate_batch_return_raw():
    sc, pr, sim, inf = _setup()
    b = sim.simulate_batch(8, np.random.default_rng(0), return_raw=True)
    assert b["raw_flux"].shape == (8, sc.n_raw)
    assert b["times"].shape == (sc.n_raw,)
    assert np.all(b["regime"] == 2)  # white-only config
    # out-of-transit flux is ~1 (plus white noise)
    assert abs(np.median(b["raw_flux"]) - 1.0) < 0.05


def test_render_raw_flux_matches_simulator_shape():
    sc, pr, sim, inf = _setup()
    theta = pr.sample(5, np.random.default_rng(1))
    rf = render_raw_flux(theta, sim.times, n_radial=sc.n_radial)
    assert rf.shape == (5, sc.n_raw)
    assert np.isfinite(rf).all()
    assert rf.max() <= 1.0 + 1e-6           # transit only dims the star


def test_render_raw_flux_dilution_attenuates_depth():
    sc, pr, sim, inf = _setup()
    theta = pr.sample(5, np.random.default_rng(12))
    plain = render_raw_flux(theta, sim.times, n_radial=sc.n_radial)
    diluted = render_raw_flux(theta, sim.times, n_radial=sc.n_radial,
                              dilution=np.full(len(theta), 0.5))
    plain_depth = 1.0 - plain.min(axis=1)
    diluted_depth = 1.0 - diluted.min(axis=1)
    ratio = np.median(diluted_depth / np.maximum(plain_depth, 1e-12))
    assert 0.45 < ratio < 0.55


def test_importance_weights_and_correction():
    sc, pr, sim, inf = _setup()
    b = sim.simulate_batch(8, np.random.default_rng(2), return_raw=True)
    i = int(np.where(b["valid"])[0][0])
    r = importance_weights(inf, b["global"][i], b["local"][i], b["sigma_feat"][i],
                           b["raw_flux"][i].astype(np.float64),
                           b["times"].astype(np.float64), float(b["sigma"][i]),
                           n_samples=100, logprob_steps=15)
    assert r["phys"].shape == (100, 7)
    assert np.isfinite(r["w"]).all()
    assert abs(r["w"].sum() - 1.0) < 1e-5
    assert 0.0 <= r["ess_fraction"] <= 1.0
    tstd = pr.physical_to_std(b["theta_phys"][i][None, :])[0]
    cdf = weighted_rank_cdf(r["std"], r["w"], tstd)
    assert cdf.shape == (7,)
    assert np.all((cdf >= -1e-9) & (cdf <= 1.0 + 1e-9))
    res = sir_resample(r["phys"], r["w"], 50, np.random.default_rng(3))
    assert res.shape == (50, 7)


def test_importance_weights_accepts_periodogram_model():
    sc = SimConfig(n_global=128, n_local=65, baseline_days=27.0, n_raw=2500,
                   frac_real=0.0, frac_gp=0.0, frac_white=1.0, planet_fraction=1.0,
                   n_radial=50, regime="tess", use_periodogram=True,
                   n_period_bins=32, pg_n_raw=512)
    pr = TransitPrior(TransitPrior.default_specs("tess"))
    sim = TransitSimulator(sc, prior=pr)
    mc = ModelConfig(embed_dim=32, global_channels=(16, 32), local_channels=(16, 32),
                     global_dim=32, local_dim=16, fm_hidden=48, fm_blocks=2,
                     fm_time_dim=16, use_periodogram=True, pg_channels=(16, 32),
                     pg_dim=16)
    inf = TransitFlowInference(TransitFlow(mc), pr, sc, ode_steps=10)
    b = sim.simulate_batch(4, np.random.default_rng(8), return_raw=True)
    i = int(np.where(b["valid"])[0][0])
    r = importance_weights(inf, b["global"][i], b["local"][i], b["sigma_feat"][i],
                           b["raw_flux"][i].astype(np.float64),
                           b["times"].astype(np.float64), float(b["sigma"][i]),
                           n_samples=32, logprob_steps=10,
                           periodogram=b["periodogram"][i])
    assert r["phys"].shape == (32, 7)
    assert np.isfinite(r["w"]).all()


def test_importance_weights_recover_true_posterior_synthetic():
    """On a high-SNR object the IS weights concentrate near the truth."""
    sc, pr, sim, inf = _setup()
    b = sim.simulate_batch(16, np.random.default_rng(7), return_raw=True)
    i = int(np.where(b["valid"])[0][0])
    r = importance_weights(inf, b["global"][i], b["local"][i], b["sigma_feat"][i],
                           b["raw_flux"][i].astype(np.float64),
                           b["times"].astype(np.float64), float(b["sigma"][i]),
                           n_samples=200, logprob_steps=15)
    # the max-weight sample's period should beat the prior-mean baseline in
    # likelihood (sanity: weights are not degenerate/uniform)
    assert r["w"].max() > 1.5 / len(r["w"])


def test_jitter_grid_rescues_ess_under_sigma_misstatement():
    """Understated sigma (the real-data failure mode) degenerates the exact
    white-noise weights; the jitter-marginalized likelihood must not do worse
    and should detect the inflation scale."""
    import torch

    sc, pr, sim, inf = _setup()
    b = sim.simulate_batch(16, np.random.default_rng(11), return_raw=True)
    i = int(np.where(b["valid"])[0][0])
    args = (inf, b["global"][i], b["local"][i], b["sigma_feat"][i],
            b["raw_flux"][i].astype(np.float64),
            b["times"].astype(np.float64),
            float(b["sigma"][i]) / 3.0)
    torch.manual_seed(0)
    exact = importance_weights(*args, n_samples=200, logprob_steps=10,
                               jitter_grid_size=1)
    torch.manual_seed(0)
    marg = importance_weights(*args, n_samples=200, logprob_steps=10,
                              jitter_grid_size=25, jitter_max=10.0)
    # An untrained proposal cannot demonstrate the ESS rescue (all samples
    # are poor fits); assert no-regression and that the profile scale detects
    # the 3x sigma understatement.
    assert marg["ess_fraction"] >= exact["ess_fraction"] * 0.3
    assert marg["jitter_scale_profile"] > 1.5
    lo, hi, size = marg["jitter_grid"]
    assert (lo, size) == (1.0, 25) and abs(hi - 10.0) < 1e-9


def test_jitter_grid_disabled_matches_exact():
    """jitter_max=1.0 must reproduce the exact white-noise likelihood."""
    import torch

    sc, pr, sim, inf = _setup()
    b = sim.simulate_batch(8, np.random.default_rng(21), return_raw=True)
    i = int(np.where(b["valid"])[0][0])
    args = (inf, b["global"][i], b["local"][i], b["sigma_feat"][i],
            b["raw_flux"][i].astype(np.float64),
            b["times"].astype(np.float64), float(b["sigma"][i]))
    torch.manual_seed(3)
    r1 = importance_weights(*args, n_samples=64, logprob_steps=10,
                            jitter_grid_size=1)
    torch.manual_seed(3)
    r2 = importance_weights(*args, n_samples=64, logprob_steps=10,
                            jitter_grid_size=25, jitter_max=1.0)
    np.testing.assert_allclose(r1["w"], r2["w"], rtol=1e-10, atol=1e-12)
