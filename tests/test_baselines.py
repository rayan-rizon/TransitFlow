import numpy as np
import pytest

from transitflow.baselines.bls import bls_detect
from scripts.baseline_detection import resolved_bls_search_settings
from transitflow.baselines import tls as tls_module
from transitflow.baselines import mcmc as mcmc_module
from scripts.baseline_detection import (
    bootstrap_tls_detection_metrics,
    prior_for_checkpoint_simulator,
)
from transitflow.baselines.mcmc import _split_rhat, has_emcee, run_mcmc
from transitflow.priors import TransitPrior, kipping_to_quadratic
from transitflow.transit_model import transit_flux


def test_baseline_prior_matches_stellar_density_checkpoint_config():
    """Baseline detection must accept checkpoints with a physical a/Rs prior."""
    from transitflow.simulator import SimConfig, TransitSimulator

    cfg = SimConfig(
        regime="tess",
        a_rs_prior_mode="stellar_density",
        stellar_density_log10_mean=0.12,
        stellar_density_log10_std=0.18,
    )
    prior = prior_for_checkpoint_simulator(cfg)

    assert prior.a_rs_prior_mode == "stellar_density"
    assert prior.stellar_density_log10_mean == pytest.approx(0.12)
    assert prior.stellar_density_log10_std == pytest.approx(0.18)
    TransitSimulator(cfg, prior=prior)


def test_blind_bls_search_inherits_checkpoint_proposal_settings():
    class CheckpointSimulator:
        candidate_bls_subsample = 4096
        candidate_bls_n_periods = 1000

    cfg = CheckpointSimulator()
    assert resolved_bls_search_settings(cfg, None, None) == (4096, 1000)
    assert resolved_bls_search_settings(cfg, 3000, 200) == (3000, 200)


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
    assert np.isfinite(res["best_t0"])
    assert res["best_duration"] > 0


def test_bls_scores_planet_above_noise():
    t, f_planet = _make_lc(P=3.0, RpRs=0.12, sigma=0.0008)
    rng = np.random.default_rng(1)
    f_noise = 1.0 + 0.0008 * rng.standard_normal(t.size)
    s_planet = bls_detect(t, f_planet, n_periods=800)["score"]
    s_noise = bls_detect(t, f_noise, n_periods=800)["score"]
    assert s_planet > s_noise


def test_tls_passes_explicit_thread_budget(monkeypatch):
    class FakeTLS:
        def power(self, **kwargs):
            assert kwargs["use_threads"] == 3
            return type("Result", (), {"SDE": 4.2, "period": 2.0})()

    monkeypatch.setattr(tls_module, "_HAS_TLS", True)
    monkeypatch.setattr(tls_module, "transitleastsquares", lambda t, f: FakeTLS())
    result = tls_module.tls_detect(np.arange(20.0), np.ones(20),
                                   np.array([1.0, 2.0]), use_threads=3)
    assert result["score"] == 4.2


def test_tls_bootstrap_reports_transitflow_minus_tls():
    labels = np.array([0, 0, 0, 1, 1, 1])
    tls_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tf_scores = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])

    result = bootstrap_tls_detection_metrics(
        labels, tls_scores, tf_scores, n_boot=20, seed=7)

    assert result["comparison"] == "TransitFlow minus TLS"
    assert set(result["ci95"]) == {
        "tls_auc", "tf_auc", "auc_gain", "tls_ap", "tf_ap", "ap_gain"}


def test_mcmc_all_fixed_returns_fixed_samples():
    P, t0p, RpRs, aRs, b = 3.0, 0.33, 0.1, 12.0, 0.2
    prior = TransitPrior()
    t = np.linspace(0, 9, 50)
    u1, u2 = kipping_to_quadratic(0.4, 0.3)
    f = transit_flux(t, P, t0p * P, RpRs, aRs, b, u1, u2, engine="native")[0]
    init = np.array([P, t0p, RpRs, aRs, b, 0.4, 0.3])
    fixed = {i: float(v) for i, v in enumerate(init)}
    out = run_mcmc(t, f, 0.0005, prior=prior, init=init, n_walkers=8,
                   n_steps=2, n_radial=10, seed=0, fixed=fixed)
    samples = out["samples"]
    assert samples.shape[1] == 7
    assert np.allclose(samples, init[None, :])
    assert out["backend"] == "fixed"


def test_mcmc_fixed_ephemeris_metadata_all_fixed():
    P, t0p, RpRs, aRs, b = 3.0, 0.33, 0.1, 12.0, 0.2
    prior = TransitPrior()
    t = np.linspace(0, 9, 50)
    u1, u2 = kipping_to_quadratic(0.4, 0.3)
    f = transit_flux(t, P, t0p * P, RpRs, aRs, b, u1, u2,
                     engine="native")[0]
    init = np.array([P, t0p, RpRs, aRs, b, 0.4, 0.3])
    fixed = {i: float(v) for i, v in enumerate(init)}
    fixed[0] = P
    fixed[1] = t0p
    out = run_mcmc(t, f, 0.0006, prior=prior, init=init, n_walkers=4,
                   n_steps=1, n_radial=8, seed=1, fixed=fixed)
    assert out["fixed"][0] == P
    assert out["fixed"][1] == t0p
    assert np.allclose(out["samples"][:, :2], np.array([P, t0p]))


def test_mcmc_fit_dilution_keeps_physical_samples_7d():
    P, t0p, RpRs, aRs, b = 3.0, 0.33, 0.1, 12.0, 0.2
    prior = TransitPrior()
    t = np.linspace(0, 9, 80)
    u1, u2 = kipping_to_quadratic(0.4, 0.3)
    f0 = transit_flux(t, P, t0p * P, RpRs, aRs, b, u1, u2,
                      engine="native")[0]
    f = 1.0 + (f0 - 1.0) * 0.6
    init = np.array([P, t0p, RpRs, aRs, b, 0.4, 0.3])
    fixed = {i: float(v) for i, v in enumerate(init)}
    out = run_mcmc(t, f, 0.0006, prior=prior, init=init, n_walkers=8,
                   n_steps=3, n_radial=8, seed=2, fixed=fixed,
                   fit_dilution=True, dilution_low=0.5, dilution_high=1.0,
                   init_dilution=0.6)
    assert out["samples"].shape[1] == 7
    assert out["dilution_samples"] is not None
    assert out["dilution_samples"].min() >= 0.5
    assert out["dilution_samples"].max() <= 1.0


def test_mcmc_emcee_process_pool_runs():
    if not has_emcee():
        pytest.skip("emcee not installed")
    P, t0p, RpRs, aRs, b = 3.0, 0.33, 0.08, 12.0, 0.2
    prior = TransitPrior()
    t = np.linspace(0, 9, 120)
    u1, u2 = kipping_to_quadratic(0.4, 0.3)
    f = transit_flux(t, P, t0p * P, RpRs, aRs, b, u1, u2,
                     engine="native")[0]
    init = np.array([P, t0p, RpRs, aRs, b, 0.4, 0.3])
    out = run_mcmc(t, f, 0.0008, prior=prior, init=init, n_walkers=16,
                   n_steps=4, n_radial=8, seed=3, fixed={0: P, 1: t0p},
                   n_processes=2)
    assert out["backend"] == "emcee"
    assert out["samples"].shape[1] == 7


def test_mcmc_seed_reproduces_proposal_chain():
    prior = TransitPrior()
    P, t0p = 3.0, 0.33
    init = np.array([P, t0p, 0.08, 12.0, 0.2, 0.4, 0.3])
    t = np.linspace(0, 9, 120)
    u1, u2 = kipping_to_quadratic(init[5], init[6])
    f = transit_flux(t, P, t0p * P, init[2], init[3], init[4],
                     u1, u2, engine="native")[0]
    fixed = {0: P, 1: t0p, 2: init[2], 3: init[3], 5: init[5], 6: init[6]}
    kwargs = dict(prior=prior, init=init, n_walkers=8, n_steps=8,
                  n_radial=8, seed=17, fixed=fixed)

    first = run_mcmc(t, f, 0.0008, **kwargs)
    second = run_mcmc(t, f, 0.0008, **kwargs)

    assert np.array_equal(first["samples"], second["samples"])
    assert first["acceptance_fraction"] == second["acceptance_fraction"]


def test_mcmc_fit_jitter_recovers_error_inflation():
    """White noise 3x the stated flux_err: the jitter posterior must find it."""
    from transitflow.correction import render_raw_flux
    from transitflow.baselines.mcmc import run_mcmc
    from transitflow.priors import TransitPrior

    rng = np.random.default_rng(3)
    prior = TransitPrior(TransitPrior.default_specs("tess"))
    init = np.array([3.0, 0.3, 0.08, 10.0, 0.3, 0.4, 0.3])
    t = np.linspace(0.0, 27.0, 900)
    model = render_raw_flux(init[None, :], t, n_radial=60)[0]
    err = 5e-4
    f = model + rng.normal(0.0, 3.0 * err, t.size)
    fixed = {i: float(init[i]) for i in range(7)}
    out = run_mcmc(t, f, err, prior=prior, init=init, n_walkers=12,
                   n_steps=500, fixed=fixed, fit_jitter=True,
                   jitter_high=10.0, seed=1)
    s = out["jitter_samples"]
    assert s is not None
    assert 2.4 < float(np.median(s)) < 3.7
    # all 7 physical params stay at their fixed values
    assert np.allclose(out["samples"][0], init)


def test_mcmc_jitter_off_keeps_legacy_return_shape():
    from transitflow.baselines.mcmc import run_mcmc
    from transitflow.priors import TransitPrior

    prior = TransitPrior(TransitPrior.default_specs("tess"))
    init = np.array([3.0, 0.3, 0.08, 10.0, 0.3, 0.4, 0.3])
    t = np.linspace(0.0, 27.0, 300)
    f = np.ones_like(t)
    out = run_mcmc(t, f, 6e-4, prior=prior, init=init, n_walkers=16,
                   n_steps=60, fixed={0: 3.0, 1: 0.3}, seed=2)
    assert out["jitter_samples"] is None
    assert out["samples"].shape[1] == 7
    assert "autocorr_time_max" in out and "n_eff" in out
    assert out["production_steps"] == 30


def test_split_rhat_detects_nonstationary_chains():
    rng = np.random.default_rng(57)
    healthy = rng.normal(size=(200, 8, 2))
    stuck = healthy.copy()
    stuck[100:, :, 0] += 3.0

    healthy_rhat = _split_rhat(healthy)
    stuck_rhat = _split_rhat(stuck)

    assert np.all(healthy_rhat < 1.05)
    assert stuck_rhat[0] > 1.2


def test_bls_score_is_sde_normalized():
    rng = np.random.default_rng(9)
    t = np.linspace(0.0, 27.0, 2000)
    f = 1.0 + rng.normal(0, 5e-4, t.size)
    res = bls_detect(t, f, period_min=0.5, period_max=6.0, n_periods=600)
    assert "peak_power" in res
    # SDE of pure noise stays small; raw astropy peak power would not be
    # comparable across noise levels at all
    assert 0.0 <= res["score"] < 25.0
