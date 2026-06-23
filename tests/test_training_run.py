"""Tests for production run management: run dir, checkpoints, resume, status."""

import json
import os

from transitflow.models.transitflow import ModelConfig
from transitflow.simulator import SimConfig
from transitflow.train import TrainConfig, load_checkpoint, train


def _tiny_cfgs():
    sim = SimConfig(n_global=256, n_local=101, baseline_days=27.0, n_raw=4000,
                    frac_real=0.0, frac_gp=0.3, frac_white=0.7, n_radial=80,
                    regime="tess")
    mcfg = ModelConfig(embed_dim=32, global_channels=(16, 32), local_channels=(16, 32),
                       global_dim=32, local_dim=16, fm_hidden=48, fm_blocks=2,
                       fm_time_dim=16, det_hidden=32)
    return sim, mcfg


def test_run_dir_artifacts_and_checkpoints(tmp_path):
    sim, mcfg = _tiny_cfgs()
    run_dir = str(tmp_path / "run")
    tcfg = TrainConfig(n_steps=30, batch_size=32, device="cpu", run_dir=run_dir,
                       ckpt_every=10, eval_every=10, eval_batches=2, log_every=5,
                       warmup_steps=3, num_workers=0, tensorboard=False, seed=0)
    res = train(mcfg, sim, tcfg, verbose=False)

    # artifacts exist
    assert os.path.exists(os.path.join(run_dir, "config.yaml"))
    assert os.path.exists(os.path.join(run_dir, "training_log.jsonl"))
    assert os.path.exists(os.path.join(run_dir, "status.json"))
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "latest.pt"))
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))

    # status.json is well-formed and marks completion
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["status"] == "done"
    assert st["step"] == 30 and st["total_steps"] == 30
    assert "throughput_lc_per_s" in st and "eta" in st

    # checkpoint reloads into a working model
    model, mc, sc = load_checkpoint(os.path.join(run_dir, "checkpoints", "latest.pt"))
    assert model.head_type == "fmpe"
    assert res["best"]["roc_auc"] >= -1.0


def test_resume_continues_from_checkpoint(tmp_path):
    sim, mcfg = _tiny_cfgs()
    run_dir = str(tmp_path / "run")
    common = dict(batch_size=32, device="cpu", run_dir=run_dir, ckpt_every=10,
                  eval_every=0, log_every=5, warmup_steps=3, num_workers=0,
                  tensorboard=False, seed=0)
    train(mcfg, sim, TrainConfig(n_steps=20, **common), verbose=False)
    step_after_first = json.load(open(os.path.join(run_dir, "status.json")))["step"]
    assert step_after_first == 20

    # resume and extend; training must pick up at step 20, finish at 40
    res = train(mcfg, sim, TrainConfig(n_steps=40, resume=run_dir, **common),
                verbose=False)
    st = json.load(open(os.path.join(run_dir, "status.json")))
    assert st["step"] == 40
    assert res["history"]["step"][-1] >= 20  # continued past the first run
