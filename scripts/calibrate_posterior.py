#!/usr/bin/env python3
"""Fit a held-out invertible posterior calibration artifact.

The calibration noise library must be disjoint from gradient-training and final
evaluation targets.  This script never modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.calibration import fit_affine_calibration, sha256_file
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noise-lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calibration", type=int, default=1000)
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--bounded-link", choices=("probit", "tanh"), default="probit",
                    help="bounded bijection; probit matches a Gaussian latent to the uniform prior")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    if args.n_calibration < 50:
        raise SystemExit("--n-calibration must be at least 50")
    set_seed(args.seed)

    model, _, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior.from_sim_config(sc)
    noise = NoiseLibrary.load(args.noise_lib)
    if not noise.available():
        raise SystemExit(f"calibration noise library unavailable: {args.noise_lib}")
    simulator = TransitSimulator(sc, prior=prior, noise_library=noise)
    inference = TransitFlowInference(model, prior, sc, amp=args.amp)
    rng = np.random.default_rng(args.seed)
    truths, samples, centers = [], [], []
    collected = 0
    while collected < args.n_calibration:
        batch = simulator.simulate_batch(args.batch, rng)
        mask = batch.get("posterior_valid", batch["valid"])
        if not mask.any():
            continue
        pg = batch.get("periodogram")
        pg = None if pg is None else pg[mask]
        eph = batch.get("ephem_feat")
        eph = None if eph is None else eph[mask]
        dil = batch.get("dil_feat")
        dil = None if dil is None else dil[mask]
        _, std = inference.posterior_samples(
            batch["global"][mask], batch["local"][mask],
            batch["sigma_feat"][mask], n_samples=args.n_posterior,
            return_std=True, periodogram=pg, ephem_feat=eph, dil_feat=dil)
        embedding = inference.embed(
            batch["global"][mask], batch["local"][mask],
            batch["sigma_feat"][mask], periodogram=pg,
            ephem_feat=eph, dil_feat=dil)
        center = inference.posterior_center_std(embedding, eph)
        target = batch["theta_std"][mask]
        if model.cfg.param_dim == 5:
            target = target[:, 2:]
            std = std[:, :, 2:]
        truths.append(target)
        samples.append(std)
        centers.append(center)
        collected += int(mask.sum())
        print(f"  calibration {min(collected, args.n_calibration)}/{args.n_calibration}")
    theta = np.concatenate(truths)[:args.n_calibration]
    posterior = np.concatenate(samples)[:args.n_calibration]
    center = np.concatenate(centers)[:args.n_calibration]
    lower, upper = prior.std_bounds
    if model.cfg.param_dim == 5:
        lower, upper = lower[2:], upper[2:]
    calibration, diagnostics = fit_affine_calibration(
        theta, posterior, center, bounds=(lower, upper),
        optimizer_seed=args.seed, bounded_link=args.bounded_link)
    metadata = {
        **calibration.metadata,
        "seed": int(args.seed),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "noise_lib": str(Path(args.noise_lib).resolve()),
        "noise_lib_sha256": sha256_file(args.noise_lib),
        "amp": bool(args.amp),
        "bounded_link": args.bounded_link,
        "fit_set_role": "calibration_only_not_final_evaluation",
    }
    calibration = type(calibration)(
        calibration.scale,
        calibration.offset,
        calibration.center_slope,
        calibration.lower,
        calibration.upper,
        calibration.space,
        metadata,
        calibration.center_quadratic,
        calibration.log_scale_slope,
    )
    calibration.save(args.out)
    print(json.dumps({"scale": calibration.scale.tolist(),
                      "offset": calibration.offset.tolist(),
                      "center_slope": calibration.center_slope.tolist(),
                      "center_quadratic": calibration.center_quadratic.tolist(),
                      "space": calibration.space,
                      "log_scale_slope": calibration.log_scale_slope.tolist(),
                      "lower": calibration.lower.tolist(),
                      "upper": calibration.upper.tolist(),
                      **diagnostics}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
