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

from transitflow.calibration import (
    PosteriorAffineCalibration,
    calibration_rank_diagnostics,
    fit_affine_calibration,
    sha256_file,
)
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.utils import set_seed


def split_calibration_noise_targets(
    noise: NoiseLibrary,
    seed: int,
    selection_fraction: float,
) -> tuple[NoiseLibrary, NoiseLibrary, dict]:
    """Reserve source targets for calibrator-family selection."""
    if noise.target_ids is None or not noise.available():
        raise ValueError("calibration selection requires source target_ids")
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between 0 and 1")
    targets = np.unique(noise.target_ids.astype(str))
    if len(targets) < 4:
        raise ValueError("at least four calibration targets are required")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_selection = min(
        max(1, int(np.ceil(len(targets) * selection_fraction))),
        len(targets) - 2,
    )
    selection_targets = targets[:n_selection]
    fit_targets = targets[n_selection:]
    selection_mask = np.isin(noise.target_ids, selection_targets)
    fit_mask = np.isin(noise.target_ids, fit_targets)
    fit_noise = NoiseLibrary(
        noise.segments[fit_mask], noise.target_ids[fit_mask])
    selection_noise = NoiseLibrary(
        noise.segments[selection_mask], noise.target_ids[selection_mask])
    metadata = {
        "seed": int(seed),
        "selection_fraction_requested": float(selection_fraction),
        "fit_targets": sorted(fit_targets.tolist()),
        "selection_targets": sorted(selection_targets.tolist()),
        "target_overlap": sorted(set(fit_targets) & set(selection_targets)),
        "n_fit_segments": int(fit_mask.sum()),
        "n_selection_segments": int(selection_mask.sum()),
    }
    return fit_noise, selection_noise, metadata


def collect_calibration_cases(
    inference: TransitFlowInference,
    simulator: TransitSimulator,
    model,
    n_cases: int,
    n_posterior: int,
    batch_size: int,
    seed: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truths, samples, centers = [], [], []
    collected = 0
    while collected < n_cases:
        batch = simulator.simulate_batch(batch_size, rng)
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
            batch["sigma_feat"][mask], n_samples=n_posterior,
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
        print(f"  {role} {min(collected, n_cases)}/{n_cases}")
    return (
        np.concatenate(truths)[:n_cases],
        np.concatenate(samples)[:n_cases],
        np.concatenate(centers)[:n_cases],
    )


def select_calibration_candidates_by_dimension(
    candidates: dict[str, PosteriorAffineCalibration],
    theta: np.ndarray,
    posterior: np.ndarray,
    center: np.ndarray,
) -> tuple[list[str], dict, list[dict]]:
    """Select each diagonal dimension on target-held-out calibration cases."""
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    scores = {
        name: calibration_rank_diagnostics(
            theta, posterior, center, candidate)
        for name, candidate in candidates.items()
    }
    dimensions = {candidate.dim for candidate in candidates.values()}
    if len(dimensions) != 1:
        raise ValueError("calibration candidates must have equal dimensions")
    selected = []
    per_dimension_scores = []
    for dim in range(dimensions.pop()):
        score_by_candidate = {
            name: float(
                diagnostics["rank_cvm_by_dim"][dim]
                + 0.25 * diagnostics["rank_coverage_error_by_dim"][dim])
            for name, diagnostics in scores.items()
        }
        selected.append(min(
            score_by_candidate,
            key=lambda name: (score_by_candidate[name], name)))
        per_dimension_scores.append(score_by_candidate)
    return selected, scores, per_dimension_scores


def combine_calibration_candidates(
    candidates: dict[str, PosteriorAffineCalibration],
    selected_by_dimension: list[str],
) -> PosteriorAffineCalibration:
    """Freeze one diagonal artifact from independently selected dimensions."""
    if not selected_by_dimension:
        raise ValueError("selected_by_dimension cannot be empty")
    chosen = [candidates[name] for name in selected_by_dimension]
    dim = len(chosen)
    if any(candidate.dim != dim for candidate in chosen):
        raise ValueError("selected calibrators do not match output dimension")
    reference = chosen[0]
    if any(candidate.space != reference.space for candidate in chosen):
        raise ValueError("selected calibrators must use the same coordinate space")
    for candidate in chosen[1:]:
        if not np.array_equal(candidate.lower, reference.lower) \
                or not np.array_equal(candidate.upper, reference.upper):
            raise ValueError("selected calibrators must use the same bounds")

    def diagonal(attribute: str) -> np.ndarray:
        return np.asarray([
            getattr(candidate, attribute)[index]
            for index, candidate in enumerate(chosen)
        ])

    return PosteriorAffineCalibration(
        diagonal("scale"), diagonal("offset"), diagonal("center_slope"),
        reference.lower, reference.upper, reference.space,
        center_quadratic=diagonal("center_quadratic"),
        log_scale_slope=diagonal("log_scale_slope"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noise-lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calibration", type=int, default=1000)
    ap.add_argument("--n-selection", type=int, default=400,
                    help="simulations on target-disjoint calibrator-selection stars")
    ap.add_argument("--selection-target-fraction", type=float, default=0.33)
    ap.add_argument("--diagnostic-data", default=None,
                    help="optional compressed fit/selection arrays for reproducibility")
    ap.add_argument("--n-posterior", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--bounded-link", choices=("probit", "tanh"), default="probit",
                    help="bounded bijection; probit matches a Gaussian latent to the uniform prior")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    if args.n_calibration < 50:
        raise SystemExit("--n-calibration must be at least 50")
    if args.n_selection < 100:
        raise SystemExit("--n-selection must be at least 100")
    set_seed(args.seed)

    model, _, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior.from_sim_config(sc)
    noise = NoiseLibrary.load(args.noise_lib)
    if not noise.available():
        raise SystemExit(f"calibration noise library unavailable: {args.noise_lib}")
    try:
        fit_noise, selection_noise, target_split = split_calibration_noise_targets(
            noise, args.seed + 9001, args.selection_target_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    fit_simulator = TransitSimulator(sc, prior=prior, noise_library=fit_noise)
    selection_simulator = TransitSimulator(
        sc, prior=prior, noise_library=selection_noise)
    inference = TransitFlowInference(model, prior, sc, amp=args.amp)
    theta, posterior, center = collect_calibration_cases(
        inference, fit_simulator, model, args.n_calibration,
        args.n_posterior, args.batch, args.seed, "calibration-fit")
    selection_theta, selection_posterior, selection_center = \
        collect_calibration_cases(
            inference, selection_simulator, model, args.n_selection,
            args.n_posterior, args.batch, args.seed + 1_000_003,
            "calibration-selection")
    lower, upper = prior.std_bounds
    if model.cfg.param_dim == 5:
        lower, upper = lower[2:], upper[2:]
    identity = PosteriorAffineCalibration(
        np.ones(theta.shape[1]), np.zeros(theta.shape[1]),
        np.ones(theta.shape[1]), lower=lower, upper=upper,
        space=f"bounded_latent_{args.bounded_link}")
    candidates = {"identity": identity}
    fit_diagnostics = {}
    for index, complexity in enumerate(("simple", "conditional")):
        candidate, candidate_fit = fit_affine_calibration(
            theta, posterior, center, bounds=(lower, upper),
            optimizer_seed=args.seed + 1000 * index,
            bounded_link=args.bounded_link, complexity=complexity)
        candidates[complexity] = candidate
        fit_diagnostics[complexity] = candidate_fit
    selected_names, selection_scores, selection_scores_by_dimension = \
        select_calibration_candidates_by_dimension(
        candidates, selection_theta, selection_posterior, selection_center)
    calibration = combine_calibration_candidates(candidates, selected_names)
    parameter_names = list(prior.names[-theta.shape[1]:])
    hybrid_selection_score = calibration_rank_diagnostics(
        selection_theta, selection_posterior, selection_center, calibration)
    diagnostics = {
        "selection_protocol": "target_disjoint_per_dimension_rank_cvm_primary_v2",
        "selected_complexity": "per_dimension",
        "selected_complexity_by_parameter": dict(
            zip(parameter_names, selected_names)),
        "target_split": target_split,
        "candidate_fit_diagnostics": fit_diagnostics,
        "candidate_selection_scores": selection_scores,
        "candidate_selection_scores_by_dimension": dict(
            zip(parameter_names, selection_scores_by_dimension)),
        "hybrid_selection_score": hybrid_selection_score,
    }
    if args.diagnostic_data:
        diagnostic_path = Path(args.diagnostic_data)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            diagnostic_path,
            fit_theta=theta.astype(np.float32),
            fit_posterior=posterior.astype(np.float32),
            fit_center=center.astype(np.float32),
            selection_theta=selection_theta.astype(np.float32),
            selection_posterior=selection_posterior.astype(np.float32),
            selection_center=selection_center.astype(np.float32),
        )
        diagnostic_data_sha256 = sha256_file(diagnostic_path)
    else:
        diagnostic_data_sha256 = None
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
        "selection_set_role": "target_disjoint_calibrator_selection_only",
        "selection_protocol": diagnostics,
        "diagnostic_data": None if args.diagnostic_data is None else str(
            Path(args.diagnostic_data).resolve()),
        "diagnostic_data_sha256": diagnostic_data_sha256,
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
                      "lower": None if calibration.lower is None
                      else calibration.lower.tolist(),
                      "upper": None if calibration.upper is None
                      else calibration.upper.tolist(),
                      **diagnostics}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
