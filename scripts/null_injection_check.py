#!/usr/bin/env python3
"""Zero-depth injection null check for the fair detection pipeline.

Positives are drawn through the COMPLETE injection, preprocessing, candidate
search, and scoring path, but with the planet radius prior collapsed to a
negligible depth (Rp/Rs ~ 1e-6, depth ~ 1e-12).  A detector that separates
these signal-free "positives" from ordinary negatives is keying on a
processing artifact of the injection pipeline rather than on transit signal.

Predeclared decision rule: the check passes when the 95% paired bootstrap
interval for the TransitFlow ROC-AUC contains 0.5, or the point estimate is
within ``--auc-margin`` (default 0.05) of 0.5.  This is a pipeline-integrity
diagnostic, not a performance claim.

Hard negatives (eclipsing binaries, single events, sinusoids) are excluded
from the null simulator: they are deliberate astrophysical content of the
negative class, so a vetter that down-scores them separates the classes for
legitimate reasons.  The null isolates the injection/processing path itself:
zero-depth "positives" versus plain-noise negatives must be exchangeable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.baseline_detection import (
    prior_for_checkpoint_simulator,
    resolved_bls_search_settings,
    transitflow_candidate_scores,
)
from transitflow.baselines.bls import bls_top_candidates
from transitflow.evaluation import detection_metrics
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.utils import set_seed

NULL_RPRS_LOW = 1.0e-6
NULL_RPRS_HIGH = 2.0e-6


def null_auc_gate(auc: float, ci_low: float, ci_high: float,
                  margin: float = 0.05) -> bool:
    """Predeclared pass rule: no artifact-driven separability."""
    if not all(np.isfinite(v) for v in (auc, ci_low, ci_high)):
        return False
    return bool(ci_low <= 0.5 <= ci_high or abs(auc - 0.5) <= margin)


def zero_depth_prior(sc) -> TransitPrior:
    """Checkpoint prior with the Rp/Rs support collapsed to negligible depth."""
    base = prior_for_checkpoint_simulator(sc)
    specs = [
        replace(spec, low=NULL_RPRS_LOW, high=NULL_RPRS_HIGH)
        if spec.name == "RpRs" else spec
        for spec in base.specs
    ]
    return TransitPrior.from_sim_config(sc, specs=specs)


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, n_boot: int,
                     seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        draws.append(detection_metrics(labels[idx], scores[idx])["roc_auc"])
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=800,
                    help="at n=400 the 0.05 AUC margin is a ~1.7-sigma decision "
                         "and unlucky seeds produce false alarms; keep >= 800")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--noise-lib", default=None)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--candidate-top-k", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--auc-margin", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="results/null_injection_check.json")
    args = ap.parse_args()
    set_seed(args.seed)

    model, _, sc = load_checkpoint(args.ckpt)
    # Hard negatives are intended task content, not an artifact channel; with
    # them present the vetter separates the labels for legitimate reasons and
    # the null is confounded (expected AUC ~ 0.5 + 0.5*hard_negative_fraction).
    sc = replace(sc, hard_negative_fraction=0.0)
    bls_subsample, n_periods = resolved_bls_search_settings(sc, None, None)
    prior = zero_depth_prior(sc)
    noise_library = NoiseLibrary.load(args.noise_lib)
    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")
    sim = TransitSimulator(sc, prior=prior, noise_library=noise_library)
    inf = TransitFlowInference(model, prior, sc, amp=args.amp)
    rng = np.random.default_rng(args.seed)

    p_lo, p_hi = prior.specs[0].low, prior.specs[0].high
    t_full = np.asarray(sim.times, dtype=np.float64)
    if getattr(sc, "exposure_minutes", 0.0) > 0:
        t_full = t_full + 0.5 * (
            sc.exposure_minutes / (24.0 * max(sc.n_exposure_subsamples, 1)))
    step = max(1, len(t_full) // bls_subsample)
    t_bls = t_full[::step]
    bls_durations = np.asarray(
        [0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24], dtype=np.float64)

    labels: list[int] = []
    scores: list[float] = []
    n_search_failures = 0
    while len(labels) < args.n:
        b = sim.simulate_batch(args.batch, rng, return_raw=True)
        raw = b.get("raw_flux_unprocessed", b["raw_flux"])
        for i in range(len(raw)):
            f_i = raw[i][::step]
            ok = np.isfinite(t_bls) & np.isfinite(f_i)
            try:
                cands = bls_top_candidates(
                    t_bls[ok], f_i[ok],
                    period_min=float(p_lo), period_max=float(p_hi),
                    n_periods=n_periods, durations=bls_durations,
                    top_k=max(1, int(args.candidate_top_k)),
                )
            except Exception:
                n_search_failures += 1
                labels.append(int(b["d"][i]))
                scores.append(0.0)
                continue
            dil_i = b.get("dil_feat")
            dil_i = None if dil_i is None else dil_i[i:i + 1]
            cand_scores = transitflow_candidate_scores(
                inf, sim, sc, prior, t_full, raw[i],
                b["sigma_feat"][i:i + 1], dil_i, b["theta_phys"][i], cands)
            labels.append(int(b["d"][i]))
            scores.append(float(max(cand_scores)))
            if len(labels) >= args.n:
                break

    labels_arr = np.asarray(labels[:args.n], dtype=int)
    scores_arr = np.asarray(scores[:args.n], dtype=float)
    if labels_arr.sum() == 0 or (labels_arr == 0).sum() == 0:
        raise SystemExit("null check needs both zero-depth positives and negatives")
    metrics = detection_metrics(labels_arr, scores_arr)
    ci = bootstrap_auc_ci(labels_arr, scores_arr, args.bootstrap,
                          args.seed + 10000)
    passed = null_auc_gate(metrics["roc_auc"], ci[0], ci[1], args.auc_margin)
    report = {
        "purpose": (
            "zero-depth injection null check; passes when signal-free "
            "'positives' are indistinguishable from negatives"),
        "seed": int(args.seed),
        "checkpoint": args.ckpt,
        "noise_lib": args.noise_lib,
        "n": int(len(labels_arr)),
        "n_zero_depth_positives": int(labels_arr.sum()),
        "n_negatives": int((labels_arr == 0).sum()),
        "n_search_failures": int(n_search_failures),
        "candidate_top_k": max(1, int(args.candidate_top_k)),
        "null_rprs_range": [NULL_RPRS_LOW, NULL_RPRS_HIGH],
        "roc_auc": float(metrics["roc_auc"]),
        "roc_auc_ci95": ci,
        "auc_margin": float(args.auc_margin),
        "pass": bool(passed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    score_path = out.with_suffix(".scores.npz")
    np.savez_compressed(score_path, labels=labels_arr, scores=scores_arr)
    report["score_data"] = str(score_path)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(
            "null injection check FAILED: the detector separates zero-depth "
            "injections from negatives, indicating an injection-pipeline "
            "artifact leak")


if __name__ == "__main__":
    main()
