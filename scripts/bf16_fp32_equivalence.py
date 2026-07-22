#!/usr/bin/env python3
"""Controlled BF16-vs-FP32 equivalence check on a single checkpoint.

The publishable pipeline can run amortized inference under autocast bfloat16
for throughput. Before any BF16 number is reported, this script verifies that
BF16 inference is statistically equivalent to FP32 on the *identical* checkpoint,
*identical* inputs, and *identical* base noise (same hardware, same process).

Method
------
* Load one checkpoint; build the simulator and a fixed evaluation batch.
* Build two inference wrappers that differ only in ``amp`` (FP32: amp=False;
  BF16: amp=True, amp_dtype=bfloat16).
* Detection: no sampling, so probabilities are compared directly.
* Posterior: the ODE/NPE base noise is held identical between precisions by
  re-seeding torch to the same value immediately before each ``posterior_samples``
  call, so the reported differences reflect precision alone, not Monte-Carlo
  sampling variance.
* Report per-parameter posterior-quantile shifts in units of the prior std
  (the natural, dimensionless scale for SBC), detection-probability differences,
  and an overall pass/fail against predeclared tolerances.

Tolerances (predeclared; edit only before a run, never to make a run pass)
--------------------------------------------------------------------------
* max |detection_prob_BF16 - detection_prob_FP32|            <= 5e-3
* max over params of |median shift| in prior-std units       <= 2e-2
* max over params of |68% half-width ratio - 1|              <= 5e-2
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# torch and the transitflow model stack are imported lazily inside the
# functions that use them so the pure gate logic (equivalence_status) -- and its
# unit tests -- import without pulling the full heavy stack.
if False:  # typing only
    from transitflow.inference import TransitFlowInference
    from transitflow.simulator import TransitSimulator


# Predeclared equivalence tolerances. These are gates, not knobs: they are the
# thresholds below which BF16 and FP32 posteriors are deemed scientifically
# interchangeable. Do not relax them to make a failing checkpoint pass.
DETECTION_ABS_TOL = 5e-3
MEDIAN_SHIFT_STD_TOL = 2e-2
WIDTH_RATIO_TOL = 5e-2


def equivalence_status(max_det_abs: float, max_median_shift_std: float,
                       max_width_ratio_abs_dev: float) -> dict:
    """Predeclared BF16/FP32 equivalence gate from the three summary maxima.

    Pure function (no torch) so the gate logic is unit-testable without a GPU
    or checkpoint. Returns the per-criterion booleans; ``equivalent`` is their
    conjunction.
    """
    status = {
        "detection_prob_within_tol": bool(max_det_abs <= DETECTION_ABS_TOL),
        "posterior_median_within_tol": bool(
            max_median_shift_std <= MEDIAN_SHIFT_STD_TOL),
        "posterior_width_within_tol": bool(
            max_width_ratio_abs_dev <= WIDTH_RATIO_TOL),
    }
    status["equivalent"] = bool(all(status.values()))
    return status


def _fixed_eval_batch(sim: TransitSimulator, n: int, seed: int) -> dict:
    """Draw a fixed, valid evaluation batch (deterministic in ``seed``)."""
    rng = np.random.default_rng(seed)
    keep: dict[str, list] = {}
    got = 0
    while got < n:
        b = sim.simulate_batch(min(256, 2 * n), rng)
        mask = b.get("posterior_valid", b["valid"])
        if not mask.any():
            continue
        for key in ("global", "local", "sigma_feat", "periodogram",
                    "ephem_feat", "dil_feat", "theta_phys"):
            if key in b and b[key] is not None:
                keep.setdefault(key, []).append(np.asarray(b[key])[mask])
        got += int(mask.sum())
    out = {k: np.concatenate(v)[:n] for k, v in keep.items()}
    return out


def _posterior_with_fixed_base(inf: TransitFlowInference, batch: dict,
                               n_samples: int, seed: int) -> np.ndarray:
    """Posterior std-space samples with base noise pinned by ``seed``.

    Re-seeding torch immediately before the draw makes the ODE/NPE base samples
    identical across precisions, isolating the precision effect.
    """
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _, std = inf.posterior_samples(
        batch["global"], batch["local"], batch.get("sigma_feat"),
        n_samples=n_samples, return_std=True,
        periodogram=batch.get("periodogram"),
        ephem_feat=batch.get("ephem_feat"), dil_feat=batch.get("dil_feat"))
    return np.asarray(std)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=512,
                    help="evaluation sources")
    ap.add_argument("--n-post", type=int, default=1000,
                    help="posterior samples per source")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--noise-lib", default=None,
                    help="optional evaluation noise library (match the eval domain)")
    ap.add_argument("--out", default="results/bf16_fp32_equivalence.json")
    args = ap.parse_args()

    import torch
    from transitflow.train import load_checkpoint
    from transitflow.inference import TransitFlowInference
    from transitflow.priors import TransitPrior
    from transitflow.simulator import TransitSimulator
    from transitflow.noise import NoiseLibrary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, scfg = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(scfg.regime))
    if args.noise_lib:
        sim = TransitSimulator(scfg, prior=prior,
                               noise_library=NoiseLibrary.load(args.noise_lib))
    else:
        sim = TransitSimulator(scfg, prior=prior)

    batch = _fixed_eval_batch(sim, args.n, args.seed)

    inf_fp32 = TransitFlowInference(model, prior, scfg, amp=False)
    inf_bf16 = TransitFlowInference(model, prior, scfg, amp=True,
                                    amp_dtype=torch.bfloat16)

    # --- detection: deterministic, compare directly ---
    det_fp32 = inf_fp32.detect(
        batch["global"], batch["local"], batch.get("sigma_feat"),
        periodogram=batch.get("periodogram"),
        ephem_feat=batch.get("ephem_feat"), dil_feat=batch.get("dil_feat"))
    det_bf16 = inf_bf16.detect(
        batch["global"], batch["local"], batch.get("sigma_feat"),
        periodogram=batch.get("periodogram"),
        ephem_feat=batch.get("ephem_feat"), dil_feat=batch.get("dil_feat"))
    det_abs = np.abs(np.asarray(det_fp32).reshape(-1)
                     - np.asarray(det_bf16).reshape(-1))

    # --- posterior: identical base noise per precision ---
    std_fp32 = _posterior_with_fixed_base(inf_fp32, batch, args.n_post, args.seed)
    std_bf16 = _posterior_with_fixed_base(inf_bf16, batch, args.n_post, args.seed)

    # per-parameter quantiles in std space (dimensionless SBC scale)
    q_fp32 = np.quantile(std_fp32, [0.16, 0.5, 0.84], axis=1)  # (3, N, P)
    q_bf16 = np.quantile(std_bf16, [0.16, 0.5, 0.84], axis=1)
    median_shift = np.abs(q_bf16[1] - q_fp32[1])               # (N, P)
    width_fp32 = q_fp32[2] - q_fp32[0]
    width_bf16 = q_bf16[2] - q_bf16[0]
    width_ratio = width_bf16 / np.maximum(width_fp32, 1e-9)    # (N, P)

    param_names = list(getattr(prior, "param_names", []))[:std_fp32.shape[-1]]
    if len(param_names) != std_fp32.shape[-1]:
        param_names = [f"param_{i}" for i in range(std_fp32.shape[-1])]

    per_param = {}
    for j, name in enumerate(param_names):
        per_param[name] = {
            "median_shift_std_mean": float(median_shift[:, j].mean()),
            "median_shift_std_p95": float(np.quantile(median_shift[:, j], 0.95)),
            "median_shift_std_max": float(median_shift[:, j].max()),
            "width_ratio_mean": float(width_ratio[:, j].mean()),
            "width_ratio_abs_dev_max": float(np.abs(width_ratio[:, j] - 1).max()),
        }

    max_median_shift = float(median_shift.max())
    max_width_dev = float(np.abs(width_ratio - 1).max())
    max_det_abs = float(det_abs.max())

    status = equivalence_status(max_det_abs, max_median_shift, max_width_dev)
    equivalent = status.pop("equivalent")
    result = {
        "checkpoint": args.ckpt,
        "device": device,
        "n_sources": int(args.n),
        "n_posterior": int(args.n_post),
        "seed": int(args.seed),
        "head": model.head_type,
        "noise_lib": args.noise_lib,
        "tolerances": {
            "detection_abs": DETECTION_ABS_TOL,
            "median_shift_std": MEDIAN_SHIFT_STD_TOL,
            "width_ratio_abs_dev": WIDTH_RATIO_TOL,
        },
        "detection_prob_abs_diff": {
            "mean": float(det_abs.mean()),
            "p95": float(np.quantile(det_abs, 0.95)),
            "max": max_det_abs,
        },
        "posterior_median_shift_std_max": max_median_shift,
        "posterior_width_ratio_abs_dev_max": max_width_dev,
        "per_parameter": per_param,
        "status": status,
        "equivalent": equivalent,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    if not result["equivalent"]:
        raise SystemExit(
            "BF16 inference is NOT equivalent to FP32 within predeclared "
            "tolerances; BF16 results must not be reported for this checkpoint")


if __name__ == "__main__":
    main()
