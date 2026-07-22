#!/usr/bin/env python3
"""Evaluate a trained TransitFlow checkpoint: SBC, coverage, detection, IS.

Writes a JSON metrics report and (optionally) SBC / coverage / ROC figures.

Example
-------
    python3 scripts/evaluate.py --ckpt checkpoints/transitflow_smoke.pt \
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
from transitflow.calibration import load_for_checkpoint
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.transit_model import transit_duration
from transitflow.utils import set_seed


def evaluation_component_seeds(seed: int) -> dict[str, int]:
    """Derive stable, independent RNG seeds for each evaluation component.

    Detection, SBC, and coverage have different sample-size knobs.  Giving each
    component its own child stream prevents changing one knob from silently
    changing another component's simulated test set.
    """
    names = ("detection", "sbc", "coverage")
    children = np.random.SeedSequence(int(seed)).spawn(len(names))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children)
    }


def detection_eval(inference, simulator, n: int, rng) -> dict:
    labels, scores, periods, rprs = [], [], [], []
    got = 0
    while got < n:
        batch = simulator.simulate_batch(256, rng)
        pg = batch.get("periodogram")
        eph = batch.get("ephem_feat")
        dil = batch.get("dil_feat")
        p = inference.detect(batch["global"], batch["local"], batch["sigma_feat"],
                             periodogram=pg, ephem_feat=eph, dil_feat=dil)
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


def _label(value: float, edges: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < edges[0]:
        return labels[0]
    if value < edges[1]:
        return labels[1]
    return labels[2]


def stratified_characterization_diagnostics(theta_true: np.ndarray,
                                            posterior_samples: np.ndarray,
                                            sigma: np.ndarray,
                                            sim_cfg,
                                            dilution: np.ndarray | None = None
                                            ) -> dict:
    """Coverage/width diagnostics split by shape and information regime.

    Strata cover impact parameter, depth (RpRs), a/Rs, S/N, transit count,
    orbital period, and (when ``dilution`` is supplied) the dilution nuisance
    regime, so conditional coverage is reported across every covariate the
    MNRAS requirement enumerates.
    """
    theta_true = np.asarray(theta_true)
    posterior_samples = np.asarray(posterior_samples)
    sigma = np.asarray(sigma)
    P, RpRs, aRs, b = (theta_true[:, 0], theta_true[:, 2],
                       theta_true[:, 3], theta_true[:, 4])
    dur = transit_duration(P, RpRs, aRs, b)
    n_transits = np.floor(sim_cfg.baseline_days / np.maximum(P, 1e-9)) + 1
    cadence_days = sim_cfg.baseline_days / max(sim_cfg.n_raw, 1)
    n_in = np.maximum(dur / max(cadence_days, 1e-9), 1.0) * n_transits
    snr = (RpRs ** 2) / np.maximum(sigma, 1e-12) * np.sqrt(n_in)
    specs = {
        "impact": [_label(x, (0.35, 0.70), ("low_b", "mid_b", "high_b")) for x in b],
        "rprs": [_label(x, (0.04, 0.09), ("shallow", "medium", "deep")) for x in RpRs],
        "a_rs": [_label(x, (10.0, 25.0), ("compact", "mid", "wide")) for x in aRs],
        "snr": [_label(x, (25.0, 75.0), ("low_snr", "mid_snr", "high_snr")) for x in snr],
        "n_transits": [_label(x, (3.0, 6.0), ("few", "several", "many")) for x in n_transits],
        "period": [_label(x, (5.0, 15.0), ("short_P", "mid_P", "long_P")) for x in P],
    }
    # Dilution is an optional nuisance covariate; stratify by it only when the
    # simulator supplied real (finite) dilution values, otherwise omit the
    # stratum rather than emit an all-"unknown" bucket.
    if dilution is not None:
        dilution = np.asarray(dilution, dtype=float).reshape(-1)
        if dilution.shape[0] == theta_true.shape[0] and np.isfinite(
                dilution).any():
            specs["dilution"] = [
                _label(x, (0.05, 0.30), ("undiluted", "mild_dil", "strong_dil"))
                for x in dilution]
    out = {}
    char_dims = [2, 3, 4]
    char_names = ["RpRs", "aRs", "b"]
    for spec_name, labels in specs.items():
        out[spec_name] = {}
        for label in sorted(set(labels)):
            idx = np.array([lab == label for lab in labels])
            if not idx.any():
                continue
            samples = posterior_samples[idx][:, :, char_dims]
            truth = theta_true[idx][:, char_dims]
            lo68 = np.quantile(samples, 0.16, axis=1)
            hi68 = np.quantile(samples, 0.84, axis=1)
            inside = (truth >= lo68) & (truth <= hi68)
            widths = hi68 - lo68
            entry = {"n": int(idx.sum())}
            for j, name in enumerate(char_names):
                entry[f"{name}_cov68"] = float(inside[:, j].mean())
                entry[f"{name}_median_width"] = float(np.median(widths[:, j]))
            out[spec_name][label] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--detector-ckpt", default=None,
                    help="validation-selected detector checkpoint")
    ap.add_argument("--n-sbc", type=int, default=300)
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=1000)
    ap.add_argument("--noise-lib", default=None,
                    help="optional real-noise .npz for held-out noise-injection evaluation")
    ap.add_argument("--calibration", default=None,
                    help="held-out posterior affine calibration artifact")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--amp", action="store_true",
                    help="enable bfloat16 autocast for amortized inference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    model, mcfg, scfg = load_checkpoint(args.ckpt)
    prior = TransitPrior.from_sim_config(scfg)
    noise_library = NoiseLibrary.load(args.noise_lib)
    simulator = TransitSimulator(scfg, prior=prior, noise_library=noise_library)
    calibration = load_for_checkpoint(args.calibration, args.ckpt) \
        if args.calibration else None
    inference = TransitFlowInference(
        model, prior, scfg, amp=args.amp, calibration=calibration)
    detector_inference = inference
    if args.detector_ckpt:
        detector_model, _, detector_sc = load_checkpoint(args.detector_ckpt)
        if detector_sc != scfg:
            raise SystemExit("--detector-ckpt simulator config mismatch")
        detector_inference = TransitFlowInference(
            detector_model, prior, scfg, amp=args.amp)
    component_seeds = evaluation_component_seeds(args.seed)
    detection_rng = np.random.default_rng(component_seeds["detection"])
    sbc_rng = np.random.default_rng(component_seeds["sbc"])
    coverage_rng = np.random.default_rng(component_seeds["coverage"])

    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")

    print("== detection ==")
    det = detection_eval(
        detector_inference, simulator, args.n_detection, detection_rng)
    print(det)

    print("== SBC ==")
    set_seed(component_seeds["sbc"])
    sbc = run_sbc(inference, simulator, n_sims=args.n_sbc,
                  n_posterior=args.n_posterior, rng=sbc_rng)
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
        char_unif = sbc_uniformity(
            char_ranks, n_posterior=args.n_posterior)

    print("== coverage ==")
    # reuse SBC posteriors for coverage by re-sampling a fresh set
    cov_samples, cov_samples_std, cov_true, cov_sigma = [], [], [], []
    cov_dil = []
    set_seed(component_seeds["coverage"])
    got = 0
    while got < args.n_sbc:
        batch = simulator.simulate_batch(128, coverage_rng)
        mask = batch.get("posterior_valid", batch["valid"])
        if not mask.any():
            continue
        pg = batch["periodogram"][mask] if "periodogram" in batch else None
        eph = batch["ephem_feat"][mask] if "ephem_feat" in batch else None
        dil = batch["dil_feat"][mask] if "dil_feat" in batch else None
        s, s_std = inference.posterior_samples(
            batch["global"][mask], batch["local"][mask],
            batch["sigma_feat"][mask], n_samples=args.n_posterior,
            return_std=True, periodogram=pg, ephem_feat=eph, dil_feat=dil)
        cov_samples.append(s)
        cov_samples_std.append(s_std)
        cov_true.append(batch["theta_phys"][mask])
        cov_sigma.append(batch["sigma"][mask])
        # Dilution is a nuisance covariate, not a sampled parameter; collect the
        # per-source dilution feature (when the simulator provides it) so
        # conditional coverage can be stratified by dilution regime.
        cov_dil.append(
            np.asarray(dil).reshape(-1) if dil is not None
            else np.full(int(mask.sum()), np.nan))
        got += int(mask.sum())
    cov_samples = np.concatenate(cov_samples)[:args.n_sbc]
    cov_samples_std = np.concatenate(cov_samples_std)[:args.n_sbc]
    cov_true = np.concatenate(cov_true)[:args.n_sbc]
    cov_sigma = np.concatenate(cov_sigma)[:args.n_sbc]
    cov_dil = np.concatenate(cov_dil)[:args.n_sbc]
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
    z_low, z_high = prior.std_bounds
    if model.cfg.param_dim == 5:
        char_std = cov_samples_std[:, :, 2:]
        z_low, z_high = z_low[2:], z_high[2:]
    else:
        char_std = cov_samples_std
    outside = (char_std < z_low) | (char_std > z_high)
    clipping_fraction_by_param = outside.mean(axis=(0, 1))
    if cce is not None:
        print(f"coverage calibration error (lower=better): {cce:.4f}")
    if model.cfg.use_ephemeris_feature:
        print("characterization SBC p-values:",
              [round(p, 3) for p in char_unif["pvalue"]])
        print(f"characterization coverage calibration error: {cce_char:.4f}")

    posterior_sbc_gate = sbc_gate(unif["pvalue"])
    char_sbc_gate = sbc_gate(char_unif["pvalue"])

    report = {
        "seed": int(args.seed),
        "component_seeds": component_seeds,
        "checkpoint": args.ckpt,
        "detector_checkpoint": args.detector_ckpt or args.ckpt,
        "head": model.head_type,
        "amp": args.amp,
        "amp_dtype": "bfloat16" if args.amp else None,
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "posterior_calibration": args.calibration,
        "param_names": param_names,
        "posterior_param_names": sbc_param_names,
        "ephemeris_conditioned": bool(model.cfg.use_ephemeris_feature),
        "characterization_param_names": char_names,
        "detection": det,
        "sbc_pvalues": unif["pvalue"],
        "sbc_rank_histograms": {
            "n_bins": unif["n_bins"],
            "n_posterior": unif["n_posterior"],
            "expected_counts": unif["expected_counts"],
            "counts_by_param": dict(zip(
                sbc_param_names, unif["histogram_counts"])),
        },
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
        "characterization_coverage_by_param": cov_char["coverage"].tolist(),
        "posterior_out_of_prior_fraction_by_param":
            clipping_fraction_by_param.tolist(),
        "posterior_out_of_prior_fraction_any": float(outside.any(axis=-1).mean()),
        "stratified_characterization": stratified_characterization_diagnostics(
            cov_true, cov_samples, cov_sigma, scfg, dilution=cov_dil),
        # evaluate.py measures the oracle/exact-candidate score, so its
        # detection threshold stays at the oracle diagnostic level; the runner
        # overwrites this key with the fair blind-candidate decision (and its
        # own predeclared threshold) when BLS candidates are requested.
        "detection_auc_min": 0.99,
        "gate_status": {
            "detection_auc_ge_min": bool(det["roc_auc"] >= 0.99),
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
