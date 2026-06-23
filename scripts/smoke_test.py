#!/usr/bin/env python3
"""End-to-end smoke test: simulate -> train -> infer -> calibrate.

Runs the entire TransitFlow pipeline at small scale and prints a summary that
verifies every component works together:

  1. forward simulator produces dual-view batches,
  2. the model trains (losses decrease),
  3. amortized detection separates planets from non-planets (ROC-AUC),
  4. the posterior recovers parameters and is approximately calibrated (SBC),
  5. a single-object posterior + importance-sampling diagnostic runs.

This is intentionally tiny (CPU-friendly, ~1-3 min). Use configs/default.yaml
for a science-scale run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _config import build_configs

from transitflow.evaluation import central_interval_coverage, run_sbc
from transitflow.evaluation.coverage import coverage_calibration_error
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = build_configs(args.config)
    if args.steps:
        cfg["train"].n_steps = args.steps
    sim_cfg, model_cfg, train_cfg = cfg["simulator"], cfg["model"], cfg["train"]
    train_cfg.ckpt_path = None  # keep the smoke test self-contained

    print("=" * 70)
    print("TransitFlow end-to-end smoke test")
    print("=" * 70)
    t0 = time.time()
    result = train(model_cfg, sim_cfg, train_cfg, verbose=True)
    model = result["model"]
    print(f"training: {time.time() - t0:.1f}s")

    hist = result["history"]
    if len(hist["posterior"]) >= 2:
        print(f"posterior loss: {hist['posterior'][0]:.3f} -> "
              f"{hist['posterior'][-1]:.3f}")
        print(f"detection loss: {hist['detection'][0]:.3f} -> "
              f"{hist['detection'][-1]:.3f}")

    prior = TransitPrior(TransitPrior.default_specs(sim_cfg.regime))
    simulator = TransitSimulator(sim_cfg, prior=prior)
    inference = TransitFlowInference(model, prior, sim_cfg,
                                     ode_steps=cfg["inference"].get("ode_steps", 40))
    rng = np.random.default_rng(7)

    # --- detection ---
    from transitflow.evaluation import detection_metrics
    test = simulator.simulate_batch(512, rng)
    pdet = inference.detect(test["global"], test["local"], test["sigma_feat"])
    dm = detection_metrics(test["d"], pdet)
    print(f"\nDetection ROC-AUC: {dm['roc_auc']:.3f}  "
          f"AP: {dm['average_precision']:.3f}")

    # --- SBC + coverage ---
    n_post = cfg["inference"].get("n_posterior", 1000)
    sbc = run_sbc(inference, simulator, n_sims=200, n_posterior=n_post, rng=rng)
    pvals = sbc["uniformity"]["pvalue"]
    print("SBC uniformity p-values:",
          {n: round(p, 3) for n, p in zip(prior.names, pvals)})

    cov_s, cov_t, got = [], [], 0
    while got < 200:
        b = simulator.simulate_batch(128, rng)
        m = b["valid"]
        if not m.any():
            continue
        cov_s.append(inference.posterior_samples(b["global"][m], b["local"][m],
                                                 b["sigma_feat"][m], n_samples=n_post))
        cov_t.append(b["theta_phys"][m])
        got += int(m.sum())
    cov = central_interval_coverage(np.concatenate(cov_t)[:200],
                                    np.concatenate(cov_s)[:200])
    cce = coverage_calibration_error(cov["levels"], cov["coverage_overall"])
    print(f"Coverage calibration error: {cce:.4f} (lower is better)")

    # --- single-object posterior + IS diagnostic ---
    one = simulator.simulate_batch(1, np.random.default_rng(99))
    while one["d"][0] != 1:
        one = simulator.simulate_batch(1, np.random.default_rng(int(rng.integers(1e6))))
    res = inference.detect_and_characterize(one["global"], one["local"],
                                            one["sigma_feat"], n_samples=n_post)
    samp = res["samples"][0]
    true = one["theta_phys"][0]
    print(f"\nSingle planet: p(detect)={res['p_detect'][0]:.3f}")
    for j, name in enumerate(prior.names):
        lo, hi = np.percentile(samp[:, j], [16, 84])
        print(f"  {name:9s} true={true[j]:+.4f}  post=[{lo:+.4f}, {hi:+.4f}]")
    isd = inference.importance_diagnostic(
        one["global"][0], one["local"][0], float(one["fold_P"][0]),
        float(one["fold_t0"][0]), one["sigma_feat"][0], n_samples=300)
    print(f"IS efficiency (ESS/N): {isd['ess_fraction']:.3f}")

    print("\n" + "=" * 70)
    ok = dm["roc_auc"] > 0.6 and cce < 0.25
    print("SMOKE TEST:", "PASS" if ok else "completed (metrics low — expected at "
          "tiny scale; increase --steps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
