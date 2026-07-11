#!/usr/bin/env python3
"""Structural smoke test for candidate augmentation + held-out calibration."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _config import build_configs
from transitflow.calibration import fit_affine_calibration
from transitflow.evaluation import central_interval_coverage, detection_metrics
from transitflow.evaluation.coverage import coverage_calibration_error
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import train


def _collect(inference, simulator, rng, n, n_posterior):
    truths, samples, centers, inputs = [], [], [], []
    got = 0
    while got < n:
        batch = simulator.simulate_batch(64, rng)
        mask = batch.get("posterior_valid", batch["valid"])
        if not mask.any():
            continue
        e = inference.embed(
            batch["global"][mask], batch["local"][mask],
            batch["sigma_feat"][mask], ephem_feat=batch["ephem_feat"][mask])
        _, std = inference.posterior_samples(
            batch["global"][mask], batch["local"][mask],
            batch["sigma_feat"][mask], n_samples=n_posterior,
            return_std=True, ephem_feat=batch["ephem_feat"][mask])
        truths.append(batch["theta_std"][mask, 2:])
        samples.append(std[:, :, 2:])
        centers.append(inference.posterior_center_std(e))
        inputs.append((batch, mask))
        got += int(mask.sum())
    return (np.concatenate(truths)[:n], np.concatenate(samples)[:n],
            np.concatenate(centers)[:n], inputs)


def _cce(truth, samples):
    coverage = central_interval_coverage(truth, samples)
    return coverage_calibration_error(
        coverage["levels"], coverage["coverage_overall"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_research.yaml")
    ap.add_argument("--out", default="results/smoke_research_fix.json")
    args = ap.parse_args()
    cfg = build_configs(args.config)
    result = train(cfg["model"], cfg["simulator"], cfg["train"], verbose=True)
    model = result["model"]
    prior = TransitPrior.from_sim_config(cfg["simulator"])
    simulator = TransitSimulator(cfg["simulator"], prior=prior)
    n_post = int(cfg["inference"].get("n_posterior", 128))
    ode_steps = int(cfg["inference"].get("ode_steps", 16))
    raw = TransitFlowInference(model, prior, cfg["simulator"], ode_steps=ode_steps)

    cal_truth, cal_samples, cal_center, _ = _collect(
        raw, simulator, np.random.default_rng(7301), 64, n_post)
    calibration, fit = fit_affine_calibration(
        cal_truth, cal_samples, cal_center)
    calibrated = TransitFlowInference(
        model, prior, cfg["simulator"], ode_steps=ode_steps,
        calibration=calibration)

    test = simulator.simulate_batch(256, np.random.default_rng(9101))
    pdet = calibrated.detect(
        test["global"], test["local"], test["sigma_feat"],
        ephem_feat=test["ephem_feat"])
    detection = detection_metrics(test["d"], pdet)
    truth, raw_samples, center, _ = _collect(
        raw, simulator, np.random.default_rng(9102), 96, n_post)
    calibrated_samples = calibration.apply(raw_samples, center)
    raw_cce = _cce(truth, raw_samples)
    calibrated_cce = _cce(truth, calibrated_samples)
    augmented = float(np.mean(test["candidate_kind"] != 0))
    structural_pass = bool(
        np.isfinite(calibrated_cce)
        and calibrated_cce <= raw_cce + 0.02
        and augmented > 0.1
        and np.isfinite(detection["roc_auc"])
        and np.isfinite(detection["average_precision"]))
    report = {
        "structural_smoke_pass": structural_pass,
        "publication_gate_pass": False,
        "publication_gate_note":
            "smoke sizes cannot establish research calibration or TLS superiority",
        "candidate_augmented_fraction": augmented,
        "detection": {
            "roc_auc": detection["roc_auc"],
            "average_precision": detection["average_precision"],
        },
        "coverage_error_raw_untouched_smoke": raw_cce,
        "coverage_error_calibrated_untouched_smoke": calibrated_cce,
        "calibration_fit": fit,
        "calibration": calibration.to_dict(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    if not structural_pass:
        raise SystemExit("research remediation smoke test failed")


if __name__ == "__main__":
    main()
