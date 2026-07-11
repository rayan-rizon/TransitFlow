#!/usr/bin/env python3
"""Matched posterior-inference timing: calibrated TransitFlow versus MCMC."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.validate_real import fold_bin_fixed_ephemeris
from transitflow.baselines.mcmc import has_emcee, run_mcmc
from transitflow.calibration import load_for_checkpoint, sha256_file
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe/checkpoints/latest.pt")
    ap.add_argument("--n-amortized", type=int, default=256)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-mcmc", type=int, default=5)
    ap.add_argument("--mcmc-steps", type=int, default=15000)
    ap.add_argument("--mcmc-max-steps", type=int, default=60000)
    ap.add_argument("--mcmc-walkers", type=int, default=32)
    ap.add_argument("--mcmc-max-cadences", type=int, default=2500)
    ap.add_argument("--noise-lib", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--seed", type=int, default=881)
    ap.add_argument("--out", default="results/speed.json")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    model, _, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior.from_sim_config(sc)
    noise_library = NoiseLibrary.load(args.noise_lib)
    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")
    simulator = TransitSimulator(sc, prior=prior, noise_library=noise_library)
    calibration = load_for_checkpoint(args.calibration, args.ckpt) \
        if args.calibration else None
    inference = TransitFlowInference(
        model, prior, sc, amp=args.amp, calibration=calibration)
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed)
    batch = simulator.simulate_batch(args.n_amortized, rng, return_raw=True)
    pg = batch.get("periodogram")
    eph = batch.get("ephem_feat")
    dil_feat = batch.get("dil_feat")

    kwargs = {
        "periodogram": pg,
        "ephem_feat": eph,
        "dil_feat": dil_feat,
    }
    inference.detect_and_characterize(
        batch["global"][:8], batch["local"][:8], batch["sigma_feat"][:8],
        n_samples=args.n_post,
        periodogram=None if pg is None else pg[:8],
        ephem_feat=None if eph is None else eph[:8],
        dil_feat=None if dil_feat is None else dil_feat[:8])
    neural_times = []
    for _ in range(max(1, args.repeats)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        inference.detect_and_characterize(
            batch["global"], batch["local"], batch["sigma_feat"],
            n_samples=args.n_post, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        neural_times.append(
            (time.perf_counter() - started) / args.n_amortized)
    neural_per_object = float(np.median(neural_times))

    mcmc_records = []
    planets = np.where(batch["posterior_valid"])[0][:args.n_mcmc]
    raw = batch.get("raw_flux_unprocessed", batch["raw_flux"])
    for order, index in enumerate(planets):
        times = np.asarray(batch["times"], dtype=np.float64)
        flux = np.asarray(raw[index], dtype=np.float64)
        observed = np.isfinite(times) & np.isfinite(flux)
        times, flux = times[observed], flux[observed]
        init = np.asarray(batch["theta_phys"][index], dtype=np.float64)
        fixed = None
        if model.cfg.param_dim == 5:
            fixed = {0: float(init[0]), 1: float(init[1])}
            times, flux, errors = fold_bin_fixed_ephemeris(
                times, flux, float(batch["sigma"][index]), init[0], init[1],
                args.mcmc_max_cadences)
        else:
            errors = np.full_like(times, float(batch["sigma"][index]))
        started = time.perf_counter()
        result = run_mcmc(
            times, flux, errors, prior=prior, init=init,
            n_walkers=args.mcmc_walkers, n_steps=args.mcmc_steps,
            max_steps=args.mcmc_max_steps, check_every=5000,
            min_tau_multiple=50.0, min_n_eff=400.0,
            max_split_rhat=1.01, n_radial=60,
            seed=args.seed + order, fixed=fixed,
            exposure_minutes=getattr(sc, "exposure_minutes", 0.0),
            n_exposure_subsamples=getattr(sc, "n_exposure_subsamples", 1),
            fit_dilution=False,
            fixed_dilution=float(batch.get("dilution", np.ones(len(batch["d"])))[index]),
            fit_jitter=True, jitter_high=10.0)
        elapsed = time.perf_counter() - started
        mcmc_records.append({
            "index": int(index), "seed": int(args.seed + order),
            "elapsed_s": float(elapsed), "converged": bool(result["converged"]),
            "steps_run": int(result["steps_run"]),
            "production_steps": int(result["production_steps"]),
            "tau_multiple": result["tau_multiple"],
            "bulk_n_eff_min": result["bulk_n_eff_min"],
            "tail_n_eff_min": result["tail_n_eff_min"],
            "split_rhat_max": result["split_rhat_max"],
        })

    ratios = np.asarray(
        [record["elapsed_s"] / neural_per_object for record in mcmc_records],
        dtype=np.float64)
    ci = np.percentile(ratios, [2.5, 97.5]).tolist() if len(ratios) else [0.0, 0.0]
    all_converged = bool(mcmc_records and all(r["converged"] for r in mcmc_records))
    report = {
        "device": device.type,
        "seed": int(args.seed),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "amp": bool(args.amp),
        "amp_dtype": "bfloat16" if args.amp else None,
        "posterior_calibration": args.calibration,
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "protocol": "fixed-ephemeris known-dilution exposure-matched jitter-fit posterior",
        "scope": "posterior inference after candidate selection and preprocessing",
        "amortized_ms_per_object": round(neural_per_object * 1e3, 3),
        "amortized_repeated_s_per_object": neural_times,
        "amortized_posterior_samples": int(args.n_post),
        "mcmc_backend": "emcee" if has_emcee() else "native",
        "mcmc_records": mcmc_records,
        "all_mcmc_converged": all_converged,
        "speedup_x": float(np.median(ratios)) if len(ratios) else 0.0,
        "speedup_ci95": [float(x) for x in ci],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
