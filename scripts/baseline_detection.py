#!/usr/bin/env python3
"""Detection baselines: BLS/TLS vs TransitFlow detection head."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

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
from transitflow.utils import set_seed
from transitflow.views import (
    flatten_transit_preserving,
    make_periodogram_view,
    make_views,
)


def _tls_score_worker(
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[float, str | None]:
    """Top-level worker so TLS searches can be parallelized safely on Linux."""
    times, flux, periods, use_threads = payload
    try:
        score = float(tls_detect(times, flux, periods, use_threads=use_threads)["score"])
        return score, None
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"


def bootstrap_detection_metrics(labels: np.ndarray, bls_scores: np.ndarray,
                                tf_scores: np.ndarray, n_boot: int,
                                seed: int) -> dict:
    """Stratified paired bootstrap intervals for AUC/AP and their differences."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    draws = {key: [] for key in (
        "bls_auc", "tf_auc", "auc_gain", "bls_ap", "tf_ap", "ap_gain")}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        b = detection_metrics(labels[idx], bls_scores[idx])
        t = detection_metrics(labels[idx], tf_scores[idx])
        draws["bls_auc"].append(b["roc_auc"])
        draws["tf_auc"].append(t["roc_auc"])
        draws["auc_gain"].append(t["roc_auc"] - b["roc_auc"])
        draws["bls_ap"].append(b["average_precision"])
        draws["tf_ap"].append(t["average_precision"])
        draws["ap_gain"].append(t["average_precision"] - b["average_precision"])
    return {
        "method": "stratified paired percentile bootstrap",
        "n_bootstrap": int(n_boot),
        "seed": int(seed),
        "ci95": {
            key: [float(x) for x in np.percentile(vals, [2.5, 97.5])]
            for key, vals in draws.items()
        },
    }


def bootstrap_tls_detection_metrics(labels: np.ndarray, tls_scores: np.ndarray,
                                    tf_scores: np.ndarray, n_boot: int,
                                    seed: int) -> dict:
    """Paired bootstrap intervals for TLS and TransitFlow on identical curves."""
    result = bootstrap_detection_metrics(
        labels, tls_scores, tf_scores, n_boot=n_boot, seed=seed)
    if not result:
        return {}
    ci = result["ci95"]
    result["ci95"] = {
        "tls_auc": ci["bls_auc"],
        "tf_auc": ci["tf_auc"],
        "auc_gain": ci["auc_gain"],
        "tls_ap": ci["bls_ap"],
        "tf_ap": ci["tf_ap"],
        "ap_gain": ci["ap_gain"],
    }
    result["comparison"] = "TransitFlow minus TLS"
    return result


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
    ap.add_argument("--tls-workers", type=int, default=None,
                    help="parallel TLS curves; defaults to a bounded CPU pool")
    ap.add_argument("--tls-threads", type=int, default=1,
                    help="threads inside each TLS curve search; keep at one when "
                         "using multiple TLS workers to avoid nested pools")
    ap.add_argument("--out", default="results/detection_baseline/bls_vs_transitflow.json")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="paired stratified bootstrap replicates for AUC/AP intervals")
    ap.add_argument(
        "--candidate-source", choices=("bls", "simulator"), default="bls",
        help="candidate ephemeris supplied to TransitFlow. 'bls' is the fair "
             "publication comparison; 'simulator' reproduces the historical "
             "oracle-ephemeris diagnostic and must not be reported as blind detection",
    )
    ap.add_argument("--amp", action="store_true",
                    help="enable bfloat16 autocast for amortized inference")
    args = ap.parse_args()
    set_seed(args.seed)

    model, _, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(sc.regime))
    noise_library = NoiseLibrary.load(args.noise_lib)
    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")
    sim = TransitSimulator(sc, prior=prior, noise_library=noise_library)
    inf = TransitFlowInference(model, prior, sc, amp=args.amp)
    rng = np.random.default_rng(args.seed)

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
    tls_workers = max(1, min(
        int(args.tls_workers) if args.tls_workers is not None else 42,
        os.cpu_count() or 1,
    ))
    tls_threads = max(1, int(args.tls_threads))

    labels, bls_scores, tls_labels, tls_scores, tf_scores = [], [], [], [], []
    bls_failures: list[str] = []
    bls_success: list[bool] = []
    tls_failures: list[str] = []
    tls_success: list[bool] = []
    tls_jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    candidate_periods, true_periods = [], []
    t0 = time.time()
    print(
        f"== detection baseline: BLS"
        f"{' + TLS' if run_tls else ''} vs TransitFlow on {args.n} LCs =="
    )
    while len(labels) < args.n:
        b = sim.simulate_batch(args.batch, rng, return_raw=True)
        raw = b.get("raw_flux_unprocessed", b["raw_flux"])
        pg = b.get("periodogram")
        eph = b.get("ephem_feat")
        p_det = None
        if args.candidate_source == "simulator":
            p_det = inf.detect(
                b["global"],
                b["local"],
                b["sigma_feat"],
                periodogram=pg,
                ephem_feat=eph,
                dil_feat=b.get("dil_feat"),
            )
        for i in range(len(raw)):
            f_i = raw[i][::step]
            ok = np.isfinite(t_bls) & np.isfinite(f_i)
            res = None
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
                bls_success.append(True)
            except Exception as exc:
                bls_scores.append(0.0)
                bls_success.append(False)
                bls_failures.append(f"{type(exc).__name__}: {exc}")
            if args.candidate_source == "bls":
                if res is None:
                    # Candidate generation is part of the evaluated pipeline;
                    # a failed search cannot fall back to the simulator's true
                    # ephemeris without leaking the label.
                    tf_scores.append(0.0)
                    candidate_periods.append(float("nan"))
                else:
                    cand_p = float(res["best_period"])
                    cand_t0 = float(res["best_t0"])
                    cand_dur = float(res["best_duration"])
                    full_ok = np.isfinite(t_full) & np.isfinite(raw[i])
                    candidate_flux = flatten_transit_preserving(
                        t_full[full_ok], raw[i][full_ok], cand_p, cand_t0, cand_dur)
                    gv_i, lv_i = make_views(
                        t_full[full_ok], candidate_flux, cand_p, cand_t0, cand_dur,
                        n_global=sc.n_global, n_local=sc.n_local,
                        n_durations=sc.n_durations, normalize=True,
                    )
                    ephem_phys = b["theta_phys"][i:i + 1].copy()
                    ephem_phys[0, 0] = cand_p
                    ephem_phys[0, 1] = (cand_t0 / max(cand_p, 1e-12)) % 1.0
                    eph_i = prior.physical_to_std(ephem_phys)[:, :2].astype(np.float32)
                    pg_i = None
                    if sc.use_periodogram:
                        n_pg = sc.pg_n_raw
                        t_pg = t_full[full_ok]
                        f_pg = candidate_flux
                        if n_pg < len(t_pg):
                            pg_step = max(1, len(t_pg) // n_pg)
                            t_pg = t_pg[::pg_step][:n_pg]
                            f_pg = f_pg[::pg_step][:n_pg]
                        pg_i = make_periodogram_view(
                            t_pg, f_pg, sim.period_grid,
                            n_phase=sc.pg_n_phase, normalize=True,
                        )[None, :]
                    dil_i = b.get("dil_feat")
                    dil_i = None if dil_i is None else dil_i[i:i + 1]
                    score_i = inf.detect(
                        gv_i[None, :], lv_i[None, :], b["sigma_feat"][i:i + 1],
                        periodogram=pg_i, ephem_feat=eph_i, dil_feat=dil_i,
                    )
                    tf_scores.append(float(score_i[0]))
                    candidate_periods.append(cand_p)
                true_periods.append(float(b["theta_phys"][i, 0]))
            if run_tls and len(tls_labels) < args.tls_n:
                tls_jobs.append((
                    np.asarray(t_bls[ok], dtype=np.float64),
                    np.asarray(f_i[ok], dtype=np.float64),
                    sim.period_grid.astype(np.float64),
                    tls_threads,
                ))
                tls_labels.append(int(b["d"][i]))
            labels.append(int(b["d"][i]))
            if args.candidate_source == "simulator":
                tf_scores.append(float(p_det[i]))
                candidate_periods.append(float(b["fold_P"][i]))
                true_periods.append(float(b["theta_phys"][i, 0]))
            if len(labels) >= args.n:
                break
        if len(labels) % (args.batch * 4) < args.batch:
            print(f"  {len(labels)}/{args.n}  ({len(labels)/(time.time()-t0):.0f} LC/s)")

    labels = np.array(labels[:args.n])
    bls_scores = np.array(bls_scores[:args.n])
    tf_scores = np.array(tf_scores[:args.n])
    if run_tls and tls_jobs:
        print(f"== TLS on {len(tls_jobs)} LCs with {tls_workers} workers x "
              f"{tls_threads} thread(s) ==")
        ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
        with ctx.Pool(tls_workers) as pool:
            for done, (score, error) in enumerate(
                    pool.imap(_tls_score_worker, tls_jobs), start=1):
                tls_scores.append(score)
                tls_success.append(error is None)
                if error is not None:
                    tls_failures.append(error)
                if done % max(1, min(100, len(tls_jobs) // 10)) == 0 or done == len(tls_jobs):
                    print(f"  TLS {done}/{len(tls_jobs)}")
    tls_labels_arr = np.array(tls_labels, dtype=int) if tls_labels else None
    tls_scores_arr = np.array(tls_scores, dtype=float) if tls_scores else None
    candidate_periods_arr = np.asarray(candidate_periods[:args.n], dtype=float)
    true_periods_arr = np.asarray(true_periods[:args.n], dtype=float)

    bls_m = detection_metrics(labels, bls_scores)
    tf_m = detection_metrics(labels, tf_scores)
    tls_m = (
        detection_metrics(tls_labels_arr, tls_scores_arr)
        if tls_labels_arr is not None and tls_scores_arr is not None
        else None
    )
    if tls_labels_arr is not None:
        paired_labels = labels[:len(tls_labels_arr)]
        if not np.array_equal(tls_labels_arr, paired_labels):
            raise RuntimeError("TLS and TransitFlow labels are not paired")

    report = {
        "seed": int(args.seed),
        "candidate_source": args.candidate_source,
        "checkpoint": args.ckpt,
        "amp": args.amp,
        "amp_dtype": "bfloat16" if args.amp else None,
        "n": int(len(labels)),
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "n_planets": int(labels.sum()),
        "n_negatives": int((labels == 0).sum()),
        "bls": {
            "n_failed": int(len(bls_failures)),
            "failure_examples": bls_failures[:10],
            "roc_auc": bls_m["roc_auc"],
            "average_precision": bls_m["average_precision"],
            "brier_score": bls_m["brier_score"],
            "expected_calibration_error_10bin":
                bls_m["expected_calibration_error_10bin"],
        },
        "tls": None if tls_m is None else {
            "n": int(len(tls_labels_arr)),
            "paired_with_transitflow": True,
            "n_failed": int(len(tls_failures)),
            "failure_rate": float(len(tls_failures) / max(len(tls_labels_arr), 1)),
            "failure_examples": tls_failures[:10],
            "roc_auc": tls_m["roc_auc"],
            "average_precision": tls_m["average_precision"],
            "brier_score": tls_m["brier_score"],
            "expected_calibration_error_10bin":
                tls_m["expected_calibration_error_10bin"],
        },
        "transitflow": {
            "roc_auc": tf_m["roc_auc"],
            "average_precision": tf_m["average_precision"],
            "brier_score": tf_m["brier_score"],
            "expected_calibration_error_10bin":
                tf_m["expected_calibration_error_10bin"],
        },
        "auc_gain": tf_m["roc_auc"] - bls_m["roc_auc"],
        "bls_score": "sde",
        "bls_backend": "astropy" if has_astropy() else "native",
        "tls_backend": "transitleastsquares" if tls_m is not None else None,
        "tls_requested": bool(args.with_tls),
        "tls_workers": int(tls_workers) if run_tls else 0,
        "tls_threads": int(tls_threads) if run_tls else 0,
        "uncertainty": bootstrap_detection_metrics(
            labels, bls_scores, tf_scores, args.bootstrap, args.seed + 10000),
    }
    if tls_m is not None:
        report["tls_uncertainty"] = bootstrap_tls_detection_metrics(
            tls_labels_arr,
            tls_scores_arr,
            tf_scores[:len(tls_labels_arr)],
            args.bootstrap,
            args.seed + 20000,
        )
    planet_mask = labels == 1
    valid_period = planet_mask & np.isfinite(candidate_periods_arr) & \
        np.isfinite(true_periods_arr)
    if valid_period.any():
        frac = np.abs(candidate_periods_arr[valid_period] /
                      true_periods_arr[valid_period] - 1.0)
        report["candidate_period_recovery"] = {
            "n_planets": int(valid_period.sum()),
            "median_abs_fractional_error": float(np.median(frac)),
            "within_1pct": float(np.mean(frac <= 0.01)),
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    score_path = Path(args.out).with_suffix(".scores.npz")
    np.savez_compressed(
        score_path,
        labels=labels,
        bls_scores=bls_scores,
        transitflow_scores=tf_scores,
        candidate_periods=candidate_periods_arr,
        true_periods=true_periods_arr,
        tls_labels=np.asarray([]) if tls_labels_arr is None else tls_labels_arr,
        tls_scores=np.asarray([]) if tls_scores_arr is None else tls_scores_arr,
        bls_success=np.asarray(bls_success, dtype=bool),
        tls_success=np.asarray(tls_success, dtype=bool),
    )
    report["score_data"] = str(score_path)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n== DETECTION BASELINE ==")
    print(f"  BLS         ROC-AUC {bls_m['roc_auc']:.4f}  AP {bls_m['average_precision']:.4f}")
    if tls_m is not None:
        print(f"  TLS         ROC-AUC {tls_m['roc_auc']:.4f}  AP {tls_m['average_precision']:.4f}")
    print(f"  TransitFlow ROC-AUC {tf_m['roc_auc']:.4f}  AP {tf_m['average_precision']:.4f}")
    print(f"  gain        {report['auc_gain']:+.4f} AUC")
    print("wrote", args.out)
    print("wrote", score_path)


if __name__ == "__main__":
    main()
