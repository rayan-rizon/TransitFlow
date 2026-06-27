import numpy as np
import pytest
import torch

from transitflow.models.spike_slab import SpikeSlabAdapter, SpikeSlabConfig

pytestmark = pytest.mark.slow


def test_make_targets_spikes_negatives(prior):
    adapter = SpikeSlabAdapter(prior)
    theta = torch.randn(20, 7)
    d = torch.zeros(20, dtype=torch.long)
    d[:10] = 1
    out = adapter.make_targets(theta, d)
    # planet rows unchanged
    assert torch.allclose(out[:10], theta[:10])
    # non-planet depth dim pushed to the spike (below the slab lower edge)
    assert torch.all(out[10:, 2] < adapter.z_low_rprs)


def test_detect_prob_separates(prior):
    adapter = SpikeSlabAdapter(prior)
    B, n = 4, 1000
    samples = torch.randn(B, n, 7)
    # row 0: depth dim solidly in the slab -> high detection
    samples[0, :, 2] = adapter.z_low_rprs + 1.0
    # row 1: depth dim at the spike -> low detection
    samples[1, :, 2] = adapter.floor_z
    p = adapter.detect_prob(samples).numpy()
    assert p[0] > 0.9
    assert p[1] < 0.1


def test_threshold_between_spike_and_slab(prior):
    adapter = SpikeSlabAdapter(prior, SpikeSlabConfig())
    assert adapter.floor_z < adapter.threshold < adapter.z_low_rprs


def test_spike_slab_training_detects(prior):
    """Variant C: a short unified-flow run reads detection from the posterior."""
    from transitflow.models.spike_slab import train_spike_slab
    from transitflow.models.transitflow import ModelConfig, TransitFlow
    from transitflow.simulator import SimConfig, TransitSimulator

    sim_cfg = SimConfig(n_global=256, n_local=101, baseline_days=27.0, n_raw=4000,
                        frac_real=0.0, frac_gp=0.2, frac_white=0.8, n_radial=80,
                        regime="tess")
    sim = TransitSimulator(sim_cfg, prior=prior)
    cfg = ModelConfig(embed_dim=48, head="fmpe",
                      global_channels=(16, 32, 48), local_channels=(16, 32),
                      global_dim=48, local_dim=24, fm_hidden=64, fm_blocks=2,
                      fm_time_dim=16, det_hidden=32)
    model = TransitFlow(cfg)
    model, adapter = train_spike_slab(sim, model, n_steps=200, batch_size=64,
                                      lr=5e-4, device="cpu", seed=0)

    import torch
    from transitflow.utils import batch_to_torch
    b = batch_to_torch(sim.simulate_batch(256, np.random.default_rng(7)),
                       torch.device("cpu"))
    from transitflow.flow_matching import sample_ode
    e = model.embed(b["global"], b["local"], b["sigma_feat"])
    std = sample_ode(model.velocity_fn(), e, n_samples=200, n_steps=20)
    p = adapter.detect_prob(std).numpy()
    d = b["d"].numpy()
    # detection-from-posterior should separate the classes better than chance
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(d, p) > 0.6
