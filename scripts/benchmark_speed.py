#!/usr/bin/env python3
"""Speed benchmark: amortized TransitFlow inference vs MCMC per object (gate #4).

Times the full amortized posterior (detection + characterization, N samples) per
object on the trained model, and a reference emcee transit fit per object, then
reports the speedup (expected 1e3-1e6x).
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from transitflow.train import load_checkpoint
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.baselines.mcmc import run_mcmc, has_emcee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe/checkpoints/latest.pt")
    ap.add_argument("--n-amortized", type=int, default=256)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--n-mcmc", type=int, default=5)
    ap.add_argument("--mcmc-steps", type=int, default=2000)
    ap.add_argument("--mcmc-walkers", type=int, default=32)
    ap.add_argument("--noise-lib", default=None)
    ap.add_argument("--out", default="results/speed.json")
    args = ap.parse_args()

    model, mc, sc = load_checkpoint(args.ckpt)
    pr = TransitPrior(TransitPrior.default_specs(sc.regime))
    noise_library = NoiseLibrary.load(args.noise_lib)
    if args.noise_lib and not noise_library.available():
        raise SystemExit(f"noise library could not be loaded: {args.noise_lib}")
    sim = TransitSimulator(sc, prior=pr, noise_library=noise_library)
    inf = TransitFlowInference(model, pr, sc)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(0)

    b = sim.simulate_batch(args.n_amortized, rng, return_raw=True)
    pg = b.get("periodogram")
    eph = b.get("ephem_feat")
    # warmup
    inf.detect_and_characterize(b["global"][:8], b["local"][:8], b["sigma_feat"][:8],
                                n_samples=args.n_post,
                                periodogram=pg[:8] if pg is not None else None,
                                ephem_feat=eph[:8] if eph is not None else None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    inf.detect_and_characterize(b["global"], b["local"], b["sigma_feat"],
                                n_samples=args.n_post,
                                periodogram=pg if pg is not None else None,
                                ephem_feat=eph if eph is not None else None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    amort_per_obj = (time.time() - t0) / args.n_amortized

    # MCMC on a few planets (fit the raw light curve)
    mcmc_times = []
    planets = np.where(b["valid"])[0][:args.n_mcmc]
    for i in planets:
        t = b["times"].astype(np.float64)
        flux = b["raw_flux"][i].astype(np.float64)
        init = b["theta_phys"][i].astype(np.float64)
        observed = np.isfinite(t) & np.isfinite(flux)
        t = t[observed]
        flux = flux[observed]
        t0 = time.time()
        run_mcmc(t, flux, float(b["sigma"][i]), prior=pr, init=init,
                 n_walkers=args.mcmc_walkers, n_steps=args.mcmc_steps, n_radial=60,
                 fit_dilution=bool(getattr(sc, "dilution_fraction", 0.0) > 0),
                 dilution_low=getattr(sc, "dilution_low", 0.5),
                 dilution_high=getattr(sc, "dilution_high", 1.0),
                 init_dilution=float(b["dilution"][i]) if "dilution" in b else 1.0)
        mcmc_times.append(time.time() - t0)
    mcmc_per_obj = float(np.mean(mcmc_times)) if mcmc_times else float("nan")

    report = {
        "device": dev.type,
        "noise_lib": args.noise_lib,
        "noise_lib_available": noise_library.available(),
        "amortized_ms_per_object": round(amort_per_obj * 1e3, 3),
        "amortized_posterior_samples": args.n_post,
        "mcmc_backend": "emcee" if has_emcee() else "native",
        "mcmc_s_per_object": round(mcmc_per_obj, 2),
        "mcmc_steps": args.mcmc_steps, "mcmc_walkers": args.mcmc_walkers,
        "speedup_x": round(mcmc_per_obj / amort_per_obj, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
