"""Tests for the box-periodogram channel (the period-calibration fix)."""

import numpy as np

from transitflow.models.transitflow import ModelConfig, TransitFlow
from transitflow.priors import TransitPrior
from transitflow.simulator import SimConfig, TransitSimulator
from transitflow.views import box_periodogram


def test_box_periodogram_peaks_at_true_period_high_snr():
    """A clean, deep transit produces periodogram power near the true period
    (or a low-order harmonic)."""
    pr = TransitPrior(TransitPrior.default_specs("tess"))
    sc = SimConfig(n_global=512, n_local=201, baseline_days=27.0, n_raw=12000,
                   frac_real=0, frac_gp=0, frac_white=1.0, planet_fraction=1.0,
                   sigma_white_log10_low=-4.0, sigma_white_log10_high=-3.6,
                   n_radial=80, regime="tess", use_periodogram=True,
                   n_period_bins=256, pg_n_phase=64)
    sim = TransitSimulator(sc, prior=pr)
    b = sim.simulate_batch(80, np.random.default_rng(0))
    pg = b["periodogram"]
    assert pg.shape == (80, 256)
    assert np.isfinite(pg).all()
    pk = sim.period_grid[pg.argmax(1)]
    Pt = b["theta_phys"][:, 0]
    rel = np.abs(pk - Pt) / Pt
    harm = (rel < 0.05) | (np.abs(pk - Pt / 2) / (Pt / 2) < 0.05) | \
           (np.abs(pk - Pt * 2) / (Pt * 2) < 0.05)
    # most clean transits peak at the true period or a low-order harmonic
    assert harm.mean() > 0.4


def test_periodogram_model_forward_and_train():
    pr = TransitPrior(TransitPrior.default_specs("tess"))
    sc = SimConfig(n_global=256, n_local=101, baseline_days=27.0, n_raw=4000,
                   frac_real=0, frac_gp=0.3, frac_white=0.7, n_radial=60,
                   regime="tess", use_periodogram=True, n_period_bins=128,
                   pg_n_phase=48)
    sim = TransitSimulator(sc, prior=pr)
    b = sim.simulate_batch(16, np.random.default_rng(1))
    assert "periodogram" in b and b["periodogram"].shape == (16, 128)

    mc = ModelConfig(embed_dim=48, use_periodogram=True,
                     global_channels=(16, 32), local_channels=(16, 32),
                     pg_channels=(16, 32), global_dim=48, local_dim=24, pg_dim=32,
                     fm_hidden=64, fm_blocks=2, fm_time_dim=16, det_hidden=32)
    model = TransitFlow(mc)
    from transitflow.train import compute_losses
    from transitflow.utils import batch_to_torch
    out = compute_losses(model, batch_to_torch(b, "cpu"), 1.0)
    assert np.isfinite(float(out["total"]))
    out["total"].backward()


def test_periodogram_model_requires_input():
    """A periodogram-enabled model errors if the channel is missing."""
    import pytest
    mc = ModelConfig(embed_dim=32, use_periodogram=True,
                     global_channels=(16, 32), local_channels=(16, 32),
                     pg_channels=(16, 32), global_dim=32, local_dim=16, pg_dim=16)
    model = TransitFlow(mc)
    import torch
    with pytest.raises(ValueError):
        model.embed(torch.zeros(2, 256), torch.zeros(2, 101), None, None)
