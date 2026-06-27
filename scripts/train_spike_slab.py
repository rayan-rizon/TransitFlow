#!/usr/bin/env python3
"""Train Variant C (unified spike-and-slab) from disk + evaluate (gate program).

Variant C models detection and characterization with a *single* flow: depth
(Rp/Rs) carries a spike-and-slab structure, and detection is read off as the
posterior mass above the spike threshold — there is no separate detection head.
This script mirrors the FMPE/NPE disk-training path (GPU-fed `DiskIterator`,
periodic atomic checkpoints, status.json) so Variant C runs in the same pipeline,
then evaluates SBC + coverage (calibration) and spike-slab detection AUC.

Example
-------
    python3 scripts/train_spike_slab.py --config configs/default.yaml \
        --run-dir runs/spikeslab_pg --data-dir data/tess_1M_pg --steps 60000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from _config import build_configs

from transitflow.data import DiskIterator
from transitflow.evaluation import (
    central_interval_coverage,
    coverage_calibration_error,
    detection_metrics,
    run_sbc,
)
from transitflow.flow_matching import cfm_loss
from transitflow.inference import TransitFlowInference
from transitflow.models.spike_slab import SpikeSlabAdapter, SpikeSlabConfig
from transitflow.models.transitflow import TransitFlow
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import save_checkpoint
from transitflow.utils import get_device


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run-dir", default="runs/spikeslab_pg")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-sbc", type=int, default=2000)
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = build_configs(args.config, {"train": {}, "model": {}})
    mcfg, scfg, tcfg = cfg["model"], cfg["simulator"], cfg["train"]
    n_steps = args.steps or tcfg.n_steps
    device = get_device(args.device or tcfg.device)
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out = args.out or args.run_dir
    os.makedirs(out, exist_ok=True)

    prior = TransitPrior(TransitPrior.default_specs(scfg.regime))
    model = TransitFlow(mcfg).to(device)
    adapter = SpikeSlabAdapter(prior, SpikeSlabConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    data = DiskIterator(args.data_dir, tcfg.batch_size, device, seed=tcfg.seed)

    print(f"== Variant C (spike-slab) train: {n_steps} steps on {device} "
          f"({model.num_parameters()/1e6:.2f}M params) ==")
    model.train()
    t0 = time.time()
    losses = []
    for step in range(n_steps):
        batch = next(data)
        nf = batch["sigma_feat"] if model.cfg.use_noise_feature else None
        pg = batch.get("periodogram") if model.cfg.use_periodogram else None
        eph = batch.get("ephem_feat") if model.cfg.use_ephemeris_feature else None
        e = model.embed(batch["global"], batch["local"], nf, pg, eph)
        targets = adapter.make_targets(batch["theta_std"], batch["d"])
        loss = cfm_loss(model.velocity_fn(), targets, e, mask=None)  # all rows
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if step % tcfg.log_every == 0 or step == n_steps - 1:
            lc_s = (step + 1) * tcfg.batch_size / (time.time() - t0)
            recent = float(np.mean(losses[-50:]))
            print(f"  step {step:6d}/{n_steps}  loss {recent:.4f}  {lc_s:.0f} lc/s")
            _write_json(os.path.join(args.run_dir, "status.json"), {
                "variant": "spikeslab", "step": step, "total_steps": n_steps,
                "progress": (step + 1) / n_steps, "loss": recent,
                "throughput_lc_per_s": round(lc_s, 1), "device": device.type,
            })
        if step % tcfg.ckpt_every == 0 and step > 0:
            save_checkpoint(model, mcfg, scfg,
                            os.path.join(ckpt_dir, "latest.pt"), step=step)
    save_checkpoint(model, mcfg, scfg, os.path.join(ckpt_dir, "best.pt"),
                    step=n_steps)
    print(f"== training done in {(time.time()-t0)/60:.1f} min ==")

    # ---- evaluate: SBC + coverage (calibration) -----------------------
    model.eval()
    sim = TransitSimulator(scfg, prior=prior)
    inf = TransitFlowInference(model, prior, scfg)
    rng = np.random.default_rng(123)

    print("== SBC ==")
    sbc = run_sbc(inf, sim, n_sims=args.n_sbc, n_posterior=args.n_posterior, rng=rng)
    pvals = sbc["uniformity"]["pvalue"]
    print("SBC p-values:", [round(p, 4) for p in pvals])

    print("== coverage ==")
    cov_s, cov_t, got = [], [], 0
    while got < args.n_sbc:
        b = sim.simulate_batch(128, rng)
        m = b["valid"]
        if not m.any():
            continue
        pg = b["periodogram"][m] if "periodogram" in b else None
        eph = b["ephem_feat"][m] if "ephem_feat" in b else None
        cov_s.append(inf.posterior_samples(b["global"][m], b["local"][m],
                                           b["sigma_feat"][m],
                                           n_samples=args.n_posterior, periodogram=pg,
                                           ephem_feat=eph))
        cov_t.append(b["theta_phys"][m])
        got += int(m.sum())
    cov_s = np.concatenate(cov_s)[:args.n_sbc]
    cov_t = np.concatenate(cov_t)[:args.n_sbc]
    cov = central_interval_coverage(cov_t, cov_s)
    cce = coverage_calibration_error(cov["levels"], cov["coverage_overall"])
    print(f"coverage calibration error: {cce:.4f}")

    # ---- spike-slab detection AUC (posterior mass, not a head) ---------
    print("== detection (spike-slab posterior mass) ==")
    labels, scores, got = [], [], 0
    while got < args.n_detection:
        b = sim.simulate_batch(256, rng)
        pg = b.get("periodogram")
        eph = b.get("ephem_feat")
        _, s_std = inf.posterior_samples(b["global"], b["local"], b["sigma_feat"],
                                         n_samples=512, return_std=True,
                                         periodogram=pg, ephem_feat=eph)
        p_det = adapter.detect_prob(torch.as_tensor(s_std)).numpy()
        labels.append(b["d"]); scores.append(p_det); got += len(p_det)
    labels = np.concatenate(labels)[:args.n_detection]
    scores = np.concatenate(scores)[:args.n_detection]
    det = detection_metrics(labels, scores)
    print(f"detection AUC: {det['roc_auc']:.4f}")

    report = {
        "variant": "spikeslab", "checkpoint": os.path.join(ckpt_dir, "best.pt"),
        "sbc_pvalues": pvals, "sbc_param_names": list(prior.names),
        "coverage_calibration_error": cce,
        "coverage_levels": cov["levels"].tolist(),
        "coverage_overall": cov["coverage_overall"].tolist(),
        "detection": {"roc_auc": det["roc_auc"],
                      "average_precision": det["average_precision"]},
    }
    _write_json(os.path.join(out, "metrics.json"), report)
    print("wrote", os.path.join(out, "metrics.json"))


if __name__ == "__main__":
    main()
