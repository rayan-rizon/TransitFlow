#!/usr/bin/env python3
"""Detection baseline: classic BLS vs the TransitFlow detection head (gate #5b).

Generates a labelled synthetic test set (planets d=1 / non-planets d=0, including
the hard negatives), scores each light curve two ways — (i) the peak Box-Least-
Squares power on the *raw* light curve, the standard pre-ML transit-search
statistic, and (ii) the amortized TransitFlow detection probability on the binned
views — and reports ROC-AUC / average-precision for both on the identical set.
This quantifies how much the learned detector beats the classical baseline.

Example
-------
    python scripts/baseline_detection.py --ckpt runs/fmpe_pg_real/checkpoints/best.pt \
        --n 3000 --out results/baseline_detection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.baselines.bls import bls_detect, has_astropy
from transitflow.evaluation import detection_metrics
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe_pg_real/checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=3000, help="labelled test light curves")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bls-subsample", type=int, default=3000,
                    help="downsample raw LC to this many points for BLS speed")
    ap.add_argument("--n-periods", type=int, default=200)
    ap.add_argument("--out", default="results/baseline_detection.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model, mc, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(sc.regime))
    sim = TransitSimulator(sc, prior=prior)
    inf = TransitFlowInference(model, prior, sc)
    rng = np.random.default_rng(7)

    # BLS trial grid: the prior period support, a handful of duration fractions
    p_lo, p_hi = prior.specs[0].low, prior.specs[0].high
    periods = np.geomspace(p_lo, p_hi, args.n_periods)
    durations = np.array([0.02, 0.05, 0.1, 0.15])  # fraction of period

    t_full = sim.times
    step = max(1, len(t_full) // args.bls_subsample)
    t_bls = t_full[::step]

    labels, bls_scores, tf_scores = [], [], []
    print(f"== detection baseline: BLS (astropy={has_astropy()}) vs TransitFlow "
          f"on {args.n} LCs ==")
    t0 = time.time()
    while len(labels) < args.n:
        b = sim.simulate_batch(args.batch, rng, return_raw=True)
        raw = b["raw_flux"]
        pg = b.get("periodogram")
        # TransitFlow detection probability (vectorized over the batch)
        p_det = inf.detect(b["global"], b["local"], b["sigma_feat"], periodogram=pg)
        for i in range(len(raw)):
            f_i = raw[i][::step]
            try:
                res = bls_detect(t_bls, f_i, periods, durations)
                bls_scores.append(float(res["score"]))
            except Exception:
                bls_scores.append(0.0)
            labels.append(int(b["d"][i]))
            tf_scores.append(float(p_det[i]))
        if len(labels) % (args.batch * 4) < args.batch:
            print(f"  {len(labels)}/{args.n}  ({len(labels)/(time.time()-t0):.0f} LC/s)")

    labels = np.array(labels[:args.n])
    bls_scores = np.array(bls_scores[:args.n])
    tf_scores = np.array(tf_scores[:args.n])

    bls_m = detection_metrics(labels, bls_scores)
    tf_m = detection_metrics(labels, tf_scores)
    report = {
        "checkpoint": args.ckpt, "n": int(len(labels)),
        "n_planets": int(labels.sum()), "n_negatives": int((labels == 0).sum()),
        "bls": {"roc_auc": bls_m["roc_auc"], "average_precision": bls_m["average_precision"]},
        "transitflow": {"roc_auc": tf_m["roc_auc"], "average_precision": tf_m["average_precision"]},
        "auc_gain": tf_m["roc_auc"] - bls_m["roc_auc"],
        "bls_backend": "astropy" if has_astropy() else "native",
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n== DETECTION BASELINE ==")
    print(f"  BLS         ROC-AUC {bls_m['roc_auc']:.4f}  AP {bls_m['average_precision']:.4f}")
    print(f"  TransitFlow ROC-AUC {tf_m['roc_auc']:.4f}  AP {tf_m['average_precision']:.4f}")
    print(f"  gain        {report['auc_gain']:+.4f} AUC")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
