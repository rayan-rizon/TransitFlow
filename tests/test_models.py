import torch

from transitflow.models.transitflow import ModelConfig, TransitFlow
from transitflow.train import compute_losses
from transitflow.utils import batch_to_torch


def _batch_t(fast_simulator, rng, n=16):
    b = fast_simulator.simulate_batch(n, rng)
    return batch_to_torch(b, torch.device("cpu"))


def test_embedding_and_heads_shapes(fast_simulator, tiny_model_cfg, rng):
    model = TransitFlow(tiny_model_cfg)
    batch = _batch_t(fast_simulator, rng)
    e = model.embed(batch["global"], batch["local"], batch["sigma_feat"])
    assert e.shape == (16, tiny_model_cfg.embed_dim)
    assert torch.isfinite(e).all()
    logits = model.detect_logits(e)
    assert logits.shape == (16,)
    tau = torch.rand(16)
    v = model.velocity(tau, batch["theta_std"], e)
    assert v.shape == (16, 7)


def test_fmpe_loss_backward(fast_simulator, tiny_model_cfg, rng):
    model = TransitFlow(tiny_model_cfg)
    batch = _batch_t(fast_simulator, rng)
    out = compute_losses(model, batch, lambda_det=1.0)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_npe_head_loss_backward(fast_simulator, rng):
    cfg = ModelConfig(embed_dim=48, head="npe", use_noise_feature=True,
                      global_channels=(16, 32, 48), local_channels=(16, 32),
                      global_dim=48, local_dim=24, npe_hidden=32, npe_transforms=3)
    model = TransitFlow(cfg)
    batch = _batch_t(fast_simulator, rng)
    out = compute_losses(model, batch, lambda_det=1.0)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_num_parameters(tiny_model_cfg):
    model = TransitFlow(tiny_model_cfg)
    assert model.num_parameters() > 0
    assert model.param_dim == 7


def test_detection_embedding_can_be_ephemeris_invariant(
        fast_simulator, rng):
    """Candidate ephemerides may condition posteriors, never detection logits."""
    cfg = ModelConfig(
        param_dim=5, embed_dim=32, use_noise_feature=True,
        use_ephemeris_feature=True, detection_ephemeris_invariant=True,
        global_channels=(8, 16), local_channels=(8, 16),
        global_dim=16, local_dim=16, det_hidden=16, fm_hidden=32,
        fm_blocks=2, fm_time_dim=16)
    model = TransitFlow(cfg).eval()
    batch = _batch_t(fast_simulator, rng, n=4)
    eph_a = batch["ephem_feat"]
    eph_b = eph_a + 3.0
    with torch.no_grad():
        posterior_a = model.embed(batch["global"], batch["local"],
                                  batch["sigma_feat"], None, eph_a)
        posterior_b = model.embed(batch["global"], batch["local"],
                                  batch["sigma_feat"], None, eph_b)
        detection_a = model.detect_logits_from_inputs(
            batch["global"], batch["local"], batch["sigma_feat"], None, eph_a)
        detection_b = model.detect_logits_from_inputs(
            batch["global"], batch["local"], batch["sigma_feat"], None, eph_b)
    assert not torch.allclose(posterior_a, posterior_b)
    assert torch.allclose(detection_a, detection_b)
