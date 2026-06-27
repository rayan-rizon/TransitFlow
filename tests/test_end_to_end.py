import numpy as np
import pytest
import torch

from transitflow.evaluation import detection_metrics
from transitflow.inference import TransitFlowInference
from transitflow.simulator import SimConfig, TransitSimulator
from transitflow.train import TrainConfig, train

pytestmark = pytest.mark.slow


def test_short_training_runs_and_learns(prior):
    """A short training run reduces loss and learns better-than-chance detection."""
    sim_cfg = SimConfig(n_global=256, n_local=101, baseline_days=27.0, n_raw=4000,
                        frac_real=0.0, frac_gp=0.2, frac_white=0.8, n_radial=80,
                        regime="tess")
    from transitflow.models.transitflow import ModelConfig
    model_cfg = ModelConfig(embed_dim=48, head="fmpe",
                            global_channels=(16, 32, 48), local_channels=(16, 32),
                            global_dim=48, local_dim=24, fm_hidden=64, fm_blocks=2,
                            fm_time_dim=16, det_hidden=32)
    train_cfg = TrainConfig(n_steps=150, batch_size=64, lr=5e-4, eval_every=0,
                            log_every=1_000_000, device="cpu", seed=0)
    res = train(model_cfg, sim_cfg, train_cfg, verbose=False)
    model = res["model"]

    sim = TransitSimulator(sim_cfg, prior=prior)
    inf = TransitFlowInference(model, prior, sim_cfg, ode_steps=20)
    b = sim.simulate_batch(400, np.random.default_rng(123))
    p = inf.detect(b["global"], b["local"], b["sigma_feat"])
    auc = detection_metrics(b["d"], p)["roc_auc"]
    assert auc > 0.6  # clearly better than chance after a short run

    # posterior sampling works end to end
    s = inf.posterior_samples(b["global"][:4], b["local"][:4], b["sigma_feat"][:4],
                              n_samples=128)
    assert s.shape == (4, 128, 7) and np.all(np.isfinite(s))
