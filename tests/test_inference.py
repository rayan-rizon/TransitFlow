import numpy as np

from transitflow.calibration import PosteriorAffineCalibration
from transitflow.inference import TransitFlowInference
from transitflow.models.transitflow import TransitFlow


def _inference(fast_sim_cfg, tiny_model_cfg, prior):
    model = TransitFlow(tiny_model_cfg)
    return TransitFlowInference(model, prior, fast_sim_cfg, ode_steps=20)


def test_detect_returns_probabilities(fast_simulator, fast_sim_cfg,
                                      tiny_model_cfg, prior, rng):
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(8, rng)
    p = inf.detect(b["global"], b["local"], b["sigma_feat"])
    assert p.shape == (8,)
    assert np.all((p >= 0) & (p <= 1))


def test_posterior_samples_shape_and_range(fast_simulator, fast_sim_cfg,
                                           tiny_model_cfg, prior, rng):
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(4, rng)
    s = inf.posterior_samples(b["global"], b["local"], b["sigma_feat"],
                              n_samples=128)
    assert s.shape == (4, 128, 7)
    # clipped into the prior support
    assert s[..., 0].min() >= 0.5 - 1e-3 and s[..., 0].max() <= 13 + 1e-3
    assert s[..., 4].min() >= 0 and s[..., 4].max() <= 1.1 + 1e-3


def test_log_prob_finite(fast_simulator, fast_sim_cfg, tiny_model_cfg, prior, rng):
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(4, rng)
    e = inf.embed(b["global"], b["local"], b["sigma_feat"])
    lp = inf.log_prob_std(b["theta_std"], e)
    assert lp.shape == (4,) and np.all(np.isfinite(lp))


def test_importance_diagnostic_runs(fast_simulator, fast_sim_cfg,
                                    tiny_model_cfg, prior):
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(1, np.random.default_rng(0))
    while b["d"][0] != 1:
        b = fast_simulator.simulate_batch(1, np.random.default_rng(
            int(np.random.default_rng().integers(1e6))))
    out = inf.importance_diagnostic(b["global"][0], b["local"][0],
                                    float(b["fold_P"][0]), float(b["fold_t0"][0]),
                                    b["sigma_feat"][0], n_samples=64)
    assert 0.0 <= out["ess_fraction"] <= 1.0


def test_ephemeris_conditioned_inference(fast_simulator, fast_sim_cfg,
                                         tiny_model_cfg, prior, rng):
    tiny_model_cfg.use_ephemeris_feature = True
    tiny_model_cfg.param_dim = 5
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(4, rng)
    p = inf.detect(b["global"], b["local"], b["sigma_feat"],
                   ephem_feat=b["ephem_feat"])
    s, s_std = inf.posterior_samples(b["global"], b["local"], b["sigma_feat"],
                                     n_samples=32, return_std=True,
                                     ephem_feat=b["ephem_feat"])
    assert p.shape == (4,)
    assert s.shape == (4, 32, 7)
    assert s_std.shape == (4, 32, 7)
    assert np.allclose(s_std[:, :, :2], b["ephem_feat"][:, None, :])


def test_log_prob_slices_characterization_target(fast_simulator, fast_sim_cfg,
                                                 tiny_model_cfg, prior, rng):
    tiny_model_cfg.use_ephemeris_feature = True
    tiny_model_cfg.param_dim = 5
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    b = fast_simulator.simulate_batch(4, rng)
    e = inf.embed(b["global"], b["local"], b["sigma_feat"],
                  ephem_feat=b["ephem_feat"])
    lp_full = inf.log_prob_std(b["theta_std"], e)
    lp_char = inf.log_prob_std(b["theta_char_std"], e)
    assert lp_full.shape == (4,)
    assert np.allclose(lp_full, lp_char)


def test_calibrated_log_prob_obeys_affine_change_of_variables(
        fast_simulator, fast_sim_cfg, tiny_model_cfg, prior, rng):
    tiny_model_cfg.use_ephemeris_feature = True
    tiny_model_cfg.param_dim = 5
    model = TransitFlow(tiny_model_cfg)
    raw_inf = TransitFlowInference(model, prior, fast_sim_cfg, ode_steps=10)
    calibration = PosteriorAffineCalibration(
        np.array([1.2, 1.4, 0.8, 1.1, 1.3]),
        np.array([0.1, -0.2, 0.05, 0.0, 0.15]),
        np.array([0.9, 1.1, 1.0, 0.8, 1.2]))
    cal_inf = TransitFlowInference(
        model, prior, fast_sim_cfg, ode_steps=10, calibration=calibration)
    batch = fast_simulator.simulate_batch(4, rng)
    e = raw_inf.embed(
        batch["global"], batch["local"], batch["sigma_feat"],
        ephem_feat=batch["ephem_feat"])
    raw_theta = batch["theta_char_std"]
    center = raw_inf.posterior_center_std(e)
    calibrated_theta = calibration.apply(raw_theta, center)

    raw_lp = raw_inf.log_prob_std(raw_theta, e)
    calibrated_lp = cal_inf.log_prob_std(calibrated_theta, e)

    assert np.allclose(
        calibrated_lp, raw_lp - calibration.log_abs_det, atol=1e-5)


def test_sbc_uses_characterization_dims_for_5d_ephemeris_model(
        fast_simulator, fast_sim_cfg, tiny_model_cfg, prior, rng):
    from transitflow.evaluation.sbc import run_sbc

    tiny_model_cfg.use_ephemeris_feature = True
    tiny_model_cfg.param_dim = 5
    inf = _inference(fast_sim_cfg, tiny_model_cfg, prior)
    out = run_sbc(inf, fast_simulator, n_sims=8, n_posterior=16,
                  batch_size=8, rng=rng)
    assert out["ranks"].shape == (8, 5)
    assert out["theta_true"].shape == (8, 5)
    assert out["param_names"] == list(prior.names[2:])
    assert len(out["uniformity"]["pvalue"]) == 5
