#!/usr/bin/env python3
"""Evaluate target-conditioned BLS calibration on held-out residual targets.

This is a development screen for the candidate stage only.  It never loads a
learned detector, and its report explicitly cannot be used as a publication
lockbox or as evidence for posterior calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.candidates import calibrate_bls_candidates
from transitflow.evaluation import detection_metrics
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def _candidate_worker(payload: tuple[int, np.ndarray, np.ndarray, dict, int]) -> tuple[int, dict]:
    """Process one curve without model state or target identity as an input."""
    index, times, flux, kwargs, seed = payload
    report = calibrate_bls_candidates(
        times, flux, rng=np.random.default_rng(seed), **kwargs)
    return index, report


def _period_recovered(candidates: list[dict], truth: float) -> tuple[bool, bool]:
    if not np.isfinite(truth) or truth <= 0:
        return False, False
    within = [abs(float(row["best_period"]) / truth - 1.0) <= 0.01
              for row in candidates]
    return bool(within[0]) if within else False, bool(any(within))


def _source_rows(labels: np.ndarray, fap: np.ndarray, sources: np.ndarray,
                 target_fap: float, min_positives: int) -> dict[str, dict]:
    rows = {}
    for source in np.unique(sources[sources >= 0]):
        mask = sources == source
        pos = mask & (labels == 1)
        neg = mask & (labels == 0)
        if int(pos.sum()) < min_positives or not neg.any():
            continue
        rows[str(int(source))] = {
            "n_positive": int(pos.sum()),
            "n_negative": int(neg.sum()),
            "completeness": float(np.mean(fap[pos] <= target_fap)),
            "false_positive_rate": float(np.mean(fap[neg] <= target_fap)),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint supplying physical simulator settings only")
    ap.add_argument("--noise-lib", required=True,
                    help="held-out residual library with target_ids; never a lockbox")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-nulls", type=int, default=31)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-periods", type=int, default=200)
    ap.add_argument("--target-fap", type=float, default=0.05)
    ap.add_argument("--min-source-positives", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n < 2 or args.batch < 1 or args.workers < 1 or args.top_k < 1:
        raise SystemExit("--n, --batch, --workers, and --top-k must be positive")
    if not 0.0 < args.target_fap < 1.0:
        raise SystemExit("--target-fap must lie in (0, 1)")
    if args.n_nulls < 8 or 1.0 / (args.n_nulls + 1) > args.target_fap:
        raise SystemExit("--n-nulls is too small to resolve the requested --target-fap")

    noise = NoiseLibrary.load(args.noise_lib)
    if not noise.available() or not noise.source_labels:
        raise SystemExit("--noise-lib must include held-out real segments and target_ids")
    _, _, checkpoint_cfg = load_checkpoint(args.ckpt, device="cpu")
    cfg = replace(
        checkpoint_cfg,
        frac_real=1.0, frac_gp=0.0, frac_white=0.0,
        flatten_views=False,
        candidate_bls_positive_fraction=0.0,
        candidate_bls_negative_fraction=0.0,
        candidate_jitter_fraction=0.0,
        candidate_harmonic_fraction=0.0,
        candidate_random_positive_fraction=0.0,
    )
    prior = TransitPrior.from_sim_config(cfg)
    simulator = TransitSimulator(cfg, prior=prior, noise_library=noise)
    times = np.asarray(simulator.times, dtype=np.float64)
    if cfg.exposure_minutes > 0:
        times = times + 0.5 * cfg.exposure_minutes / (24.0 * max(cfg.n_exposure_subsamples, 1))
    durations = np.asarray([0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24], dtype=float)
    kwargs = {
        "period_min": float(prior.specs[0].low),
        "period_max": float(prior.specs[0].high),
        "n_periods": int(args.n_periods),
        "durations": durations,
        "top_k": int(args.top_k),
        "n_null": int(args.n_nulls),
        "block_size": int(args.block_size),
    }
    rng = np.random.default_rng(args.seed)
    labels, sources, truth, records = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        total = 0
        while total < args.n:
            b = simulator.simulate_batch(min(args.batch, args.n - total), rng, return_raw=True)
            raw = np.asarray(b["raw_flux_unprocessed"], dtype=np.float64)
            jobs = [
                (total + i, times, raw[i], kwargs, int(rng.integers(0, 2**63 - 1)))
                for i in range(len(raw))
            ]
            reports = dict(pool.map(_candidate_worker, jobs))
            for i in range(len(raw)):
                record = reports[total + i]
                records.append(record)
                labels.append(int(b["d"][i]))
                sources.append(int(b["noise_source_index"][i]))
                truth.append(float(b["theta_phys"][i, 0]))
            total += len(raw)
            print(f"  calibrated {total}/{args.n}", flush=True)

    labels = np.asarray(labels, dtype=int)
    sources = np.asarray(sources, dtype=int)
    truth = np.asarray(truth, dtype=float)
    fap = np.asarray([row["target_search_fap"] for row in records], dtype=float)
    top1, topk = zip(*[_period_recovered(row["candidates"], p) for row, p in zip(records, truth)])
    top1, topk = np.asarray(top1, dtype=bool), np.asarray(topk, dtype=bool)
    pos, neg = labels == 1, labels == 0
    score = -fap
    metrics = detection_metrics(labels, score)
    source_rows = _source_rows(
        labels, fap, sources, args.target_fap, args.min_source_positives)
    report = {
        "report_schema_version": 1,
        "purpose": "development candidate-calibration screen; not a publication lockbox",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "noise_library": str(Path(args.noise_lib).resolve()),
        "seed": int(args.seed),
        "n": int(len(labels)),
        "n_planets": int(pos.sum()),
        "n_negatives": int(neg.sum()),
        "source_label_count": int(len(noise.source_labels)),
        "candidate_protocol": {
            **records[0]["protocol"],
            "n_periods": int(args.n_periods),
            "source_disjoint_requirement": "caller must supply held-out noise-lib",
            "publication_lockbox_consumed": False,
        },
        "operating_point": {
            "target_fap": float(args.target_fap),
            "fap_resolution": float(1.0 / (args.n_nulls + 1)),
            "false_positive_rate": float(np.mean(fap[neg] <= args.target_fap)),
            "completeness": float(np.mean(fap[pos] <= args.target_fap)),
        },
        "ranking": {
            "roc_auc": float(metrics["roc_auc"]),
            "average_precision": float(metrics["average_precision"]),
            "score": "negative target-conditioned empirical FAP",
        },
        "period_recovery": {
            "top1_within_1pct": float(np.mean(top1[pos])),
            "topk_within_1pct": float(np.mean(topk[pos])),
            "n_planets": int(pos.sum()),
        },
        "source_rows": source_rows,
        "interpretation_rule": (
            "Do not retrain or allocate a publication run unless this target-conditioned "
            "candidate stage improves both FAP control and source-stratified completeness "
            "on a larger, independent development audit."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(
        out.with_suffix(".scores.npz"), labels=labels, fap=fap, sources=sources,
        true_periods=truth, top1_recovered=top1, topk_recovered=topk,
    )
    print(json.dumps({
        "out": str(out),
        "auc": report["ranking"]["roc_auc"],
        "fpr": report["operating_point"]["false_positive_rate"],
        "completeness": report["operating_point"]["completeness"],
    }, indent=2))


if __name__ == "__main__":
    main()
