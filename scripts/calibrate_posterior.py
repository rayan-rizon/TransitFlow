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
from dataclasses import replace
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


def collect_target_balanced_calibration_cases(
    inference: TransitFlowInference,
    simulator_config,
    prior: TransitPrior,
    noise: NoiseLibrary,
    model,
    n_cases: int,
    n_posterior: int,
    batch_size: int,
    seed: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collect equal-sized calibration selections from each held-out target.

    A segment-uniform draw can let a few long/noisy targets dominate the
    calibrator-family decision.  Selection is a target-level research gate, so
    each held-out source target receives equal simulation weight.
    """
    if noise.target_ids is None:
        raise ValueError("target-balanced selection requires target_ids")
    target_ids = noise.target_ids.astype(str)
    targets = np.unique(target_ids)
    if len(targets) < 2:
        raise ValueError("target-balanced selection requires at least two targets")
    per_target = int(np.ceil(n_cases / len(targets)))
    # Each source target should contribute the same number of accepted cases,
    # but a full training batch per target is wasteful for small smoke runs.
    # The bounded batch remains large enough for the exact-candidate mask to
    # yield posterior labels under the configured candidate mixture.
    target_batch = min(batch_size, max(8, 4 * per_target))
    chunks = []
    for index, target in enumerate(targets):
        mask = target_ids == target
        target_noise = NoiseLibrary(noise.segments[mask], target_ids[mask])
        target_simulator = TransitSimulator(
            simulator_config, prior=prior, noise_library=target_noise)
        chunks.append(collect_calibration_cases(
            inference, target_simulator, model, per_target, n_posterior,
            target_batch, seed + 10_007 * index, f"{role}:{target}"))
    theta = np.concatenate([chunk[0] for chunk in chunks])[:n_cases]
    posterior = np.concatenate([chunk[1] for chunk in chunks])[:n_cases]
    center = np.concatenate([chunk[2] for chunk in chunks])[:n_cases]
    return theta, posterior, center, {
        "sampling": "target_uniform_then_simulation_v1",
        "n_targets": int(len(targets)),
        "n_cases_requested": int(n_cases),
        "n_cases_per_target": int(per_target),
        "batch_size_per_target": int(target_batch),
        "targets": sorted(targets.tolist()),
    }


def temper_calibration(candidate: PosteriorAffineCalibration,
                        strength: float) -> PosteriorAffineCalibration:
    """Shrink a fitted affine calibration toward identity in latent space."""
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("calibration tempering strength must be in [0, 1]")
    return PosteriorAffineCalibration(
        np.exp(strength * np.log(candidate.scale)),
        strength * candidate.offset,
        1.0 + strength * (candidate.center_slope - 1.0),
        candidate.lower, candidate.upper, candidate.space,
        center_quadratic=strength * candidate.center_quadratic,
        log_scale_slope=strength * candidate.log_scale_slope,
    )


def candidate_tempering_strength(name: str) -> float:
    """Shrinkage of a candidate toward identity (0 = identity, 1 = full fit)."""
    if name == "identity":
        return 0.0
    _, _, suffix = name.partition("_")
    try:
        return float(suffix) / 100.0
    except ValueError:
        return 1.0


def _dimension_score(diagnostics: dict, dim: int) -> float:
    return float(
        diagnostics["rank_cvm_by_dim"][dim]
        + 0.25 * diagnostics["rank_coverage_error_by_dim"][dim])


def select_calibration_candidates_by_dimension(
    candidates: dict[str, PosteriorAffineCalibration],
    theta: np.ndarray,
    posterior: np.ndarray,
    center: np.ndarray,
    groups: np.ndarray | None = None,
) -> tuple[list[str], dict, list[dict]]:
    """Select each diagonal dimension on target-held-out calibration cases.

    With ``groups`` (one source-target label per case) the choice uses a
    one-standard-error rule instead of a bare ``argmin``.  The score is
    recomputed within each held-out target, so its spread across targets
    estimates how well a candidate transfers to a *new* noise domain; the
    least-aggressive candidate whose mean score lies within one standard error
    of the best mean is then selected.

    This matters because the pooled ``argmin`` is evaluated on a small number of
    selection targets, and it reliably picks the strongest available correction.
    That correction fits the selection stars and then fails to transfer: the
    2026-07-19 seed-0 run chose full-strength tempering for ``RpRs`` on 11
    targets and missed the SBC gate on 31 unseen lockbox stars (p = 0.0052)
    while looking best-in-class on the calibration split.  Preferring shrinkage
    on ties is the standard remedy for an over-selected model family.
    """
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
    n_dim = dimensions.pop()

    per_group_scores: dict[str, np.ndarray] | None = None
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(theta):
            raise ValueError("groups must supply one label per calibration case")
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            per_group_scores = {}
            for name, candidate in candidates.items():
                rows = []
                for group in unique_groups:
                    mask = groups == group
                    if mask.sum() < 2:
                        continue
                    group_diagnostics = calibration_rank_diagnostics(
                        theta[mask], posterior[mask], center[mask], candidate)
                    rows.append([_dimension_score(group_diagnostics, dim)
                                 for dim in range(n_dim)])
                per_group_scores[name] = np.asarray(rows, dtype=float)
            if any(row.shape[0] < 2 for row in per_group_scores.values()):
                per_group_scores = None

    selected = []
    per_dimension_scores = []
    for dim in range(n_dim):
        score_by_candidate = {
            name: _dimension_score(diagnostics, dim)
            for name, diagnostics in scores.items()
        }
        if per_group_scores is None:
            selected.append(min(
                score_by_candidate,
                key=lambda name: (score_by_candidate[name], name)))
            per_dimension_scores.append(score_by_candidate)
            continue
        means = {name: float(np.mean(rows[:, dim]))
                 for name, rows in per_group_scores.items()}
        best = min(means, key=lambda name: (means[name], name))
        best_rows = per_group_scores[best][:, dim]
        standard_error = float(
            np.std(best_rows, ddof=1) / np.sqrt(len(best_rows)))
        threshold = means[best] + standard_error
        within = [name for name, value in means.items() if value <= threshold]
        selected.append(min(
            within,
            key=lambda name: (candidate_tempering_strength(name), means[name], name)))
        detail = dict(score_by_candidate)
        detail["_selection_rule"] = "one_standard_error_across_targets"
        detail["_group_mean"] = means
        detail["_best_candidate"] = best
        detail["_standard_error"] = standard_error
        detail["_threshold"] = threshold
        per_dimension_scores.append(detail)
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
    # Calibration is defined only for exact-candidate posterior rows.  Avoid
    # spending CPU on blind BLS candidate searches for rows that this stage
    # will mask out, while preserving every physical/noise setting of the
    # checkpoint's forward model.
    calibration_sc = replace(
        sc,
        candidate_bls_negative_fraction=0.0,
        candidate_bls_positive_fraction=0.0,
        candidate_harmonic_fraction=0.0,
        candidate_random_positive_fraction=0.0,
        candidate_jitter_fraction=0.0,
    )
    fit_simulator = TransitSimulator(
        calibration_sc, prior=prior, noise_library=fit_noise)
    inference = TransitFlowInference(model, prior, sc, amp=args.amp)
    theta, posterior, center = collect_calibration_cases(
        inference, fit_simulator, model, args.n_calibration,
        args.n_posterior, args.batch, args.seed, "calibration-fit")
    (selection_theta, selection_posterior, selection_center,
     selection_sampling) = collect_target_balanced_calibration_cases(
        inference, calibration_sc, prior, selection_noise, model, args.n_selection,
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
    # Limit the family to a diagonal affine map and its conservative latent
    # tempering path.  Each variant is chosen only on held-out source targets;
    # no final-evaluation target is available to this selector.
    fit_diagnostics = {}
    for index, complexity in enumerate(("simple",)):
        candidate, candidate_fit = fit_affine_calibration(
            theta, posterior, center, bounds=(lower, upper),
            optimizer_seed=args.seed + 1000 * index,
            bounded_link=args.bounded_link, complexity=complexity)
        candidates["simple_050"] = temper_calibration(candidate, 0.50)
        candidates["simple_075"] = temper_calibration(candidate, 0.75)
        candidates["simple_100"] = candidate
        fit_diagnostics[complexity] = candidate_fit
    # ``collect_target_balanced_calibration_cases`` concatenates equal-sized
    # per-target blocks, so the source target of every selection case is
    # recoverable and the selector can measure cross-target transfer.
    selection_groups = np.repeat(
        np.arange(selection_sampling["n_targets"]),
        selection_sampling["n_cases_per_target"])[:len(selection_theta)]
    selected_names, selection_scores, selection_scores_by_dimension = \
        select_calibration_candidates_by_dimension(
        candidates, selection_theta, selection_posterior, selection_center,
        groups=selection_groups)
    calibration = combine_calibration_candidates(candidates, selected_names)
    parameter_names = list(prior.names[-theta.shape[1]:])
    hybrid_selection_score = calibration_rank_diagnostics(
        selection_theta, selection_posterior, selection_center, calibration)
    diagnostics = {
        "selection_protocol":
            "target_disjoint_balanced_tempered_simple_v5_one_standard_error",
        "selected_complexity": "per_dimension",
        "selected_complexity_by_parameter": dict(
            zip(parameter_names, selected_names)),
        "target_split": target_split,
        "selection_sampling": selection_sampling,
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
