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
