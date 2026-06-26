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
    sbc_uniformity,
)
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def detection_eval(inference, simulator, n: int, rng) -> dict:
    labels, scores, periods, rprs = [], [], [], []
    got = 0
    while got < n:
        batch = simulator.simulate_batch(256, rng)
        pg = batch.get("periodogram")
        eph = batch.get("ephem_feat")
        p = inference.detect(batch["global"], batch["local"], batch["sigma_feat"],
                             periodogram=pg, ephem_feat=eph)
        labels.append(batch["d"])
        scores.append(p)
        periods.append(batch["theta_phys"][:, 0])
        rprs.append(batch["theta_phys"][:, 2])
        got += len(p)
    labels = np.concatenate(labels)[:n]
    scores = np.concatenate(scores)[:n]
    m = detection_metrics(labels, scores)
    return {"roc_auc": m["roc_auc"], "average_precision": m["average_precision"]}


def sbc_gate(pvalues, alpha: float = 0.05) -> dict:
    """Multiple-comparison aware SBC gate.

    A D-dimensional SBC report contains D per-parameter uniformity tests. Using
    `all raw p-values > 0.05` as the required gate false-fails a calibrated 5-D
    posterior about 23% of the time. The required gate therefore controls the
    family-wise false-rejection rate at `alpha` with Bonferroni correction while
    still reporting the raw p-values and raw all-p>0.05 diagnostic.
    """
    p = [float(x) for x in pvalues]
    n_tests = max(len(p), 1)
    threshold = alpha / n_tests
    return {
        "alpha_familywise": alpha,
        "bonferroni_alpha_per_test": threshold,
        "n_tests": n_tests,
        "min_pvalue": min(p) if p else None,
        "pass": bool(p and min(p) > threshold),
        "all_raw_p_gt_0.05": bool(p and min(p) > 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-sbc", type=int, default=300)
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=1000)
    ap.add_argument("--noise-lib", default=None,
                    help="optional real-noise .npz for held-out noise-injection evaluation")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mcfg, scfg = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(scfg.regime))
    noise_library = NoiseLibrary.load(args.noise_lib)
    simulator = TransitSimulator(scfg, prior=prior, noise_library=noise_library)
    inference = TransitFlowInference(model, prior, scfg)
    rng = np.random.default_rng(args.seed)

    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")

    print("== detection ==")
    det = detection_eval(inference, simulator, args.n_detection, rng)
    print(det)

    print("== SBC ==")
    sbc = run_sbc(inference, simulator, n_sims=args.n_sbc,
                  n_posterior=args.n_posterior, rng=rng)
    unif = sbc["uniformity"]
    print("SBC uniformity p-values:", [round(p, 3) for p in unif["pvalue"]])
    param_names = list(prior.names)
    sbc_param_names = list(sbc.get("param_names", param_names))
    if model.cfg.param_dim == 5:
        char_dims = list(range(2, len(param_names)))
        char_names = list(param_names[2:])
        char_unif = unif
    else:
        char_dims = list(range(2, len(param_names))) if model.cfg.use_ephemeris_feature \
            else list(range(len(param_names)))
        char_names = [param_names[i] for i in char_dims]
        char_ranks = sbc["ranks"][:, char_dims]
        char_unif = sbc_uniformity(char_ranks)

    print("== coverage ==")
    # reuse SBC posteriors for coverage by re-sampling a fresh set
    cov_samples, cov_true = [], []
    got = 0
    while got < args.n_sbc:
        batch = simulator.simulate_batch(128, rng)
        mask = batch["valid"]
        if not mask.any():
            continue
        pg = batch["periodogram"][mask] if "periodogram" in batch else None
        eph = batch["ephem_feat"][mask] if "ephem_feat" in batch else None
        s = inference.posterior_samples(batch["global"][mask], batch["local"][mask],
                                        batch["sigma_feat"][mask],
                                        n_samples=args.n_posterior, periodogram=pg,
                                        ephem_feat=eph)
        cov_samples.append(s)
        cov_true.append(batch["theta_phys"][mask])
        got += int(mask.sum())
    cov_samples = np.concatenate(cov_samples)[:args.n_sbc]
    cov_true = np.concatenate(cov_true)[:args.n_sbc]
    if model.cfg.param_dim == 5:
        cov = central_interval_coverage(cov_true[:, char_dims],
                                        cov_samples[:, :, char_dims])
        cce = None
    else:
        cov = central_interval_coverage(cov_true, cov_samples)
        cce = coverage_calibration_error(cov["levels"], cov["coverage_overall"])
    cov_char = central_interval_coverage(cov_true[:, char_dims],
                                         cov_samples[:, :, char_dims],
                                         levels=cov["levels"])
    cce_char = coverage_calibration_error(cov_char["levels"],
                                          cov_char["coverage_overall"])
    if cce is not None:
        print(f"coverage calibration error (lower=better): {cce:.4f}")
    if model.cfg.use_ephemeris_feature:
        print("characterization SBC p-values:",
              [round(p, 3) for p in char_unif["pvalue"]])
        print(f"characterization coverage calibration error: {cce_char:.4f}")

    posterior_sbc_gate = sbc_gate(unif["pvalue"])
    char_sbc_gate = sbc_gate(char_unif["pvalue"])

    report = {
        "checkpoint": args.ckpt,
        "head": model.head_type,
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "param_names": param_names,
        "posterior_param_names": sbc_param_names,
        "ephemeris_conditioned": bool(model.cfg.use_ephemeris_feature),
        "characterization_param_names": char_names,
        "detection": det,
        "sbc_pvalues": unif["pvalue"],
        "sbc_pvalues_by_param": dict(zip(sbc_param_names, unif["pvalue"])),
        "sbc_gate": posterior_sbc_gate,
        "characterization_sbc_pvalues": char_unif["pvalue"],
        "characterization_sbc_pvalues_by_param": dict(zip(char_names,
                                                          char_unif["pvalue"])),
        "characterization_sbc_gate": char_sbc_gate,
        "coverage_calibration_error": cce,
        "characterization_coverage_calibration_error": cce_char,
        "coverage_levels": cov["levels"].tolist(),
        "coverage_overall": cov["coverage_overall"].tolist(),
        "characterization_coverage_overall": cov_char["coverage_overall"].tolist(),
        "gate_status": {
            "detection_auc_ge_0.99": bool(det["roc_auc"] >= 0.99),
            "posterior_sbc_familywise_alpha_0.05": posterior_sbc_gate["pass"],
            "posterior_sbc_all_raw_p_gt_0.05":
                posterior_sbc_gate["all_raw_p_gt_0.05"],
            "all_parameter_sbc_familywise_alpha_0.05": None
            if model.cfg.param_dim == 5 else posterior_sbc_gate["pass"],
            "characterization_sbc_familywise_alpha_0.05": char_sbc_gate["pass"],
            "characterization_sbc_all_raw_p_gt_0.05":
                char_sbc_gate["all_raw_p_gt_0.05"],
            "coverage_error_le_0.03": None if cce is None else bool(cce <= 0.03),
            "characterization_coverage_error_le_0.03": bool(cce_char <= 0.03),
        },
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", os.path.join(args.out, "metrics.json"))

    if args.plots:
        _make_plots(sbc, cov, args.out, sbc_param_names)


def _make_plots(sbc, cov, out, param_names) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sbc["ranks"]
    D = ranks.shape[1]
    n_cols = min(4, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for j in range(D):
        ax = axes[j]
        ax.hist(ranks[:, j], bins=20, color="steelblue", alpha=0.8)
        ax.axhline(len(ranks) / 20, color="k", ls="--", lw=1)
        ax.set_title(param_names[j])
    for ax in axes[D:]:
        ax.axis("off")
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
