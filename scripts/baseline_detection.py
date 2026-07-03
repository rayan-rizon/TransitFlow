#!/usr/bin/env python3
"""Detection baselines: BLS/TLS vs TransitFlow detection head."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.baselines.bls import bls_detect, has_astropy
from transitflow.baselines.tls import has_tls, tls_detect
from transitflow.evaluation import detection_metrics
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe_pg/checkpoints/latest.pt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bls-subsample", type=int, default=3000,
                    help="downsample raw LC to this many points for BLS/TLS speed")
    ap.add_argument("--n-periods", type=int, default=200)
    ap.add_argument("--noise-lib", default=None)
    ap.add_argument("--with-tls", action="store_true",
                    help="also run Transit Least Squares on a bounded subset")
    ap.add_argument("--tls-n", type=int, default=500)
    ap.add_argument("--out", default="results/detection_baseline/bls_vs_transitflow.json")
    args = ap.parse_args()

    model, _, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(sc.regime))
    noise_library = NoiseLibrary.load(args.noise_lib)
    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")
    sim = TransitSimulator(sc, prior=prior, noise_library=noise_library)
    inf = TransitFlowInference(model, prior, sc)
    rng = np.random.default_rng(123)

    p_lo, p_hi = prior.specs[0].low, prior.specs[0].high
    t_full = np.asarray(sim.times, dtype=np.float64)
    if getattr(sc, "exposure_minutes", 0.0) > 0:
        t_full = t_full + 0.5 * (sc.exposure_minutes / (24.0 * max(sc.n_exposure_subsamples, 1)))
    step = max(1, len(t_full) // max(1, args.bls_subsample))
    t_bls = t_full[::step]
    bls_durations = np.asarray(
        [0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24],
        dtype=np.float64,
    )
    run_tls = bool(args.with_tls and has_tls())

    labels, bls_scores, tls_labels, tls_scores, tf_scores = [], [], [], [], []
    t0 = time.time()
    print(
        f"== detection baseline: BLS"
        f"{' + TLS' if run_tls else ''} vs TransitFlow on {args.n} LCs =="
    )
    while len(labels) < args.n:
        b = sim.simulate_batch(args.batch, rng, return_raw=True)
        raw = b["raw_flux"]
        pg = b.get("periodogram")
        eph = b.get("ephem_feat")
        p_det = inf.detect(
            b["global"],
            b["local"],
            b["sigma_feat"],
            periodogram=pg,
            ephem_feat=eph,
        )
        for i in range(len(raw)):
            f_i = raw[i][::step]
            ok = np.isfinite(t_bls) & np.isfinite(f_i)
            try:
                res = bls_detect(
                    t_bls[ok],
                    f_i[ok],
                    period_min=float(p_lo),
                    period_max=float(p_hi),
                    n_periods=args.n_periods,
                    durations=bls_durations,
                )
                bls_scores.append(float(res["score"]))
            except Exception:
                bls_scores.append(0.0)
            if run_tls and len(tls_labels) < args.tls_n:
                try:
                    res = tls_detect(t_bls[ok], f_i[ok], sim.period_grid.astype(np.float64))
                    tls_scores.append(float(res["score"]))
                except Exception:
                    tls_scores.append(0.0)
                tls_labels.append(int(b["d"][i]))
            labels.append(int(b["d"][i]))
            tf_scores.append(float(p_det[i]))
            if len(labels) >= args.n:
                break
        if len(labels) % (args.batch * 4) < args.batch:
            print(f"  {len(labels)}/{args.n}  ({len(labels)/(time.time()-t0):.0f} LC/s)")

    labels = np.array(labels[:args.n])
    bls_scores = np.array(bls_scores[:args.n])
    tf_scores = np.array(tf_scores[:args.n])
    tls_labels_arr = np.array(tls_labels, dtype=int) if tls_labels else None
    tls_scores_arr = np.array(tls_scores, dtype=float) if tls_scores else None

    bls_m = detection_metrics(labels, bls_scores)
    tf_m = detection_metrics(labels, tf_scores)
    tls_m = (
        detection_metrics(tls_labels_arr, tls_scores_arr)
        if tls_labels_arr is not None and tls_scores_arr is not None
        else None
    )

    report = {
        "checkpoint": args.ckpt,
        "n": int(len(labels)),
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "n_planets": int(labels.sum()),
        "n_negatives": int((labels == 0).sum()),
        "bls": {
            "roc_auc": bls_m["roc_auc"],
            "average_precision": bls_m["average_precision"],
        },
        "tls": None if tls_m is None else {
            "n": int(len(tls_labels_arr)),
            "roc_auc": tls_m["roc_auc"],
            "average_precision": tls_m["average_precision"],
        },
        "transitflow": {
            "roc_auc": tf_m["roc_auc"],
            "average_precision": tf_m["average_precision"],
        },
        "auc_gain": tf_m["roc_auc"] - bls_m["roc_auc"],
        "bls_backend": "astropy" if has_astropy() else "native",
        "tls_backend": "transitleastsquares" if tls_m is not None else None,
        "tls_requested": bool(args.with_tls),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n== DETECTION BASELINE ==")
    print(f"  BLS         ROC-AUC {bls_m['roc_auc']:.4f}  AP {bls_m['average_precision']:.4f}")
    if tls_m is not None:
        print(f"  TLS         ROC-AUC {tls_m['roc_auc']:.4f}  AP {tls_m['average_precision']:.4f}")
    print(f"  TransitFlow ROC-AUC {tf_m['roc_auc']:.4f}  AP {tf_m['average_precision']:.4f}")
    print(f"  gain        {report['auc_gain']:+.4f} AUC")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
