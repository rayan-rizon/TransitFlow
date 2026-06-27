"""Tests for the disk dataset pipeline and the preflight cost/health check."""

import numpy as np
import json
import os

from transitflow.data import DiskDataset, DiskIterator, generate_to_disk
from transitflow.models.transitflow import ModelConfig
from transitflow.simulator import SimConfig
from transitflow.train import TrainConfig, preflight, train


def _sim_cfg():
    return SimConfig(n_global=128, n_local=65, baseline_days=27.0, n_raw=2500,
                     frac_real=0.0, frac_gp=0.3, frac_white=0.7, n_radial=60,
                     regime="tess")


def _model_cfg():
    return ModelConfig(embed_dim=32, global_channels=(16, 32), local_channels=(16, 32),
                       global_dim=32, local_dim=16, fm_hidden=48, fm_blocks=2,
                       fm_time_dim=16, det_hidden=32)


def test_generate_and_load_disk_dataset(tmp_path):
    out = str(tmp_path / "data")
    generate_to_disk(_sim_cfg(), n_total=600, out_dir=out, shard_size=250,
                     num_workers=1, gen_batch=150, seed=0, verbose=False)
    ds = DiskDataset(out)
    assert len(ds) == 600
    # batches have the right keys/shapes and finite views
    it = DiskIterator(out, batch_size=64, device="cpu", seed=0)
    batch = next(it)
    assert batch["global"].shape == (64, 128)
    assert batch["local"].shape == (64, 65)
    assert batch["theta_std"].shape == (64, 7)
    assert batch["valid"].dtype == __import__("torch").bool
    assert bool(__import__("torch").isfinite(batch["global"]).all())
    # d=0 rows carry zeroed targets (as the simulator emits)
    d = batch["d"].numpy()
    if (d == 0).any():
        assert np.allclose(batch["theta_std"].numpy()[d == 0], 0.0)
    meta_path = os.path.join(out, "dataset_meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["n_total"] == 600
    assert meta["n_shards"] == 3
    assert "config_hash" in meta
    assert meta["realism_flags"]["finite_exposure"] is False


def test_resumable_generation_skips_existing(tmp_path):
    out = str(tmp_path / "data")
    generate_to_disk(_sim_cfg(), n_total=400, out_dir=out, shard_size=200,
                     num_workers=1, seed=0, verbose=False)
    import os
    mtimes = {f: os.path.getmtime(os.path.join(out, f))
              for f in os.listdir(out) if f.startswith("shard_")}
    # re-run: existing shards must not be rewritten
    generate_to_disk(_sim_cfg(), n_total=400, out_dir=out, shard_size=200,
                     num_workers=1, seed=0, verbose=False)
    for f, mt in mtimes.items():
        assert os.path.getmtime(os.path.join(out, f)) == mt


def test_train_from_disk(tmp_path):
    out = str(tmp_path / "data")
    generate_to_disk(_sim_cfg(), n_total=512, out_dir=out, shard_size=512,
                     num_workers=1, seed=0, verbose=False)
    tcfg = TrainConfig(n_steps=20, batch_size=32, device="cpu", data_source="disk",
                       data_dir=out, run_dir=str(tmp_path / "run"), ckpt_every=10,
                       eval_every=0, log_every=5, warmup_steps=3, tensorboard=False)
    res = train(_model_cfg(), _sim_cfg(), tcfg, verbose=False)
    assert res["history"]["posterior"][-1] < res["history"]["posterior"][0] + 1.0


def test_preflight_verdict_and_cost(tmp_path):
    out = str(tmp_path / "data")
    generate_to_disk(_sim_cfg(), n_total=256, out_dir=out, shard_size=256,
                     num_workers=1, seed=0, verbose=False)
    tcfg = TrainConfig(n_steps=1000, batch_size=32, device="cpu",
                       data_source="disk", data_dir=out)
    r = preflight(_model_cfg(), _sim_cfg(), tcfg, n_probe=10, verbose=False)
    assert r["verdict"] in ("PASS", "WARN", "FAIL")
    assert r["finite_losses"] and r["checkpoint_ok"]
    assert "projected_cost_usd" in r and "lc_per_s" in r


def test_preflight_flags_device_mismatch(tmp_path):
    tcfg = TrainConfig(n_steps=60000, batch_size=32, device="cpu",
                       expect_device="cuda", num_workers=0)
    r = preflight(_model_cfg(), _sim_cfg(), tcfg, n_probe=6, verbose=False)
    assert r["verdict"] == "FAIL" and not r["ok"]
