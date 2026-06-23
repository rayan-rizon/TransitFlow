#!/usr/bin/env python3
"""Evaluate a trained TransitFlow checkpoint: SBC, coverage, detection, IS.

Writes a JSON metrics report and (optionally) SBC / coverage / ROC figures.

Example
-------
    python scripts/evaluate.py --ckpt checkpoints/transitflow_smoke.pt \
        --n-sbc 300 --n-detection 1000 --out results/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.evaluation import (
    central_interval_coverage,
    coverage_calibration_error,
    detection_metrics,
    run_sbc,
)
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def detection_eval(inference, simulator, n: int, rng) -> dict:
    labels, scores, periods, rprs = [], [], [], []
    got = 0
    while got < n:
        batch = simulator.simulate_batch(256, rng)
        p = inference.detect(batch["global"], batch["local"], batch["sigma_feat"])
        labels.append(batch["d"])
        scores.append(p)
        periods.append(batch["theta_phys"][:, 0])
        rprs.append(batch["theta_phys"][:, 2])
        got += len(p)
    labels = np.concatenate(labels)[:n]
    scores = np.concatenate(scores)[:n]
    m = detection_metrics(labels, scores)
    return {"roc_auc": m["roc_auc"], "average_precision": m["average_precision"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-sbc", type=int, default=300)
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=1000)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mcfg, scfg = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(scfg.regime))
    simulator = TransitSimulator(scfg, prior=prior)
    inference = TransitFlowInference(model, prior, scfg)
    rng = np.random.default_rng(args.seed)

    print("== detection ==")
    det = detection_eval(inference, simulator, args.n_detection, rng)
    print(det)

    print("== SBC ==")
    sbc = run_sbc(inference, simulator, n_sims=args.n_sbc,
                  n_posterior=args.n_posterior, rng=rng)
    unif = sbc["uniformity"]
    print("SBC uniformity p-values:", [round(p, 3) for p in unif["pvalue"]])

    print("== coverage ==")
    # reuse SBC posteriors for coverage by re-sampling a fresh set
    cov_samples, cov_true = [], []
    got = 0
    while got < args.n_sbc:
        batch = simulator.simulate_batch(128, rng)
        mask = batch["valid"]
        if not mask.any():
            continue
        s = inference.posterior_samples(batch["global"][mask], batch["local"][mask],
                                        batch["sigma_feat"][mask],
                                        n_samples=args.n_posterior)
        cov_samples.append(s)
        cov_true.append(batch["theta_phys"][mask])
        got += int(mask.sum())
    cov_samples = np.concatenate(cov_samples)[:args.n_sbc]
    cov_true = np.concatenate(cov_true)[:args.n_sbc]
    cov = central_interval_coverage(cov_true, cov_samples)
    cce = coverage_calibration_error(cov["levels"], cov["coverage_overall"])
    print(f"coverage calibration error (lower=better): {cce:.4f}")

    report = {
        "checkpoint": args.ckpt,
        "head": model.head_type,
        "detection": det,
        "sbc_pvalues": unif["pvalue"],
        "coverage_calibration_error": cce,
        "coverage_levels": cov["levels"].tolist(),
        "coverage_overall": cov["coverage_overall"].tolist(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", os.path.join(args.out, "metrics.json"))

    if args.plots:
        _make_plots(sbc, cov, args.out, prior)


def _make_plots(sbc, cov, out, prior) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sbc["ranks"]
    D = ranks.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for j in range(D):
        ax = axes.flat[j]
        ax.hist(ranks[:, j], bins=20, color="steelblue", alpha=0.8)
        ax.axhline(len(ranks) / 20, color="k", ls="--", lw=1)
        ax.set_title(prior.names[j])
    axes.flat[-1].axis("off")
    fig.suptitle("SBC rank histograms (flat = calibrated)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sbc.png"), dpi=120)

    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(cov["levels"], cov["coverage_overall"], "o-", color="crimson")
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Expected coverage")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out, "coverage.png"), dpi=120)
    print("wrote SBC + coverage figures to", out)


if __name__ == "__main__":
    main()
