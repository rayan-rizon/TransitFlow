import numpy as np

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
