#!/usr/bin/env python3
"""Audit whether a detector miss is source-specific or observationally hard.

This is deliberately a diagnostic, not a gate-relaxation tool.  It evaluates a
frozen detector on independently simulated injections, chooses an operating
threshold from *negative* curves at a predeclared false-alarm rate, and reports
conditional injection recovery by physical observability strata and held-out
real-residual source.  Source identity is audit metadata only; it is never a
network feature.

The resulting JSON answers the question needed before spending on another
training run: are failures concentrated in a few residual domains, in a known
low-information population, or widespread across the intended population?
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

from transitflow.evaluation import detection_metrics
from transitflow.data import DiskDataset
from transitflow.inference import TransitFlowInference
from transitflow.noise import NoiseLibrary
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.transit_model import transit_duration


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Two-sided Wilson interval for a binomial completeness estimate."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    """Conservative empirical score threshold for a predeclared FPR target."""
    scores = np.sort(np.asarray(negative_scores, dtype=float))
    if scores.size == 0:
        raise ValueError("at least one negative score is required")
    # The order statistic makes the empirical FPR no greater than the target
    # except for unavoidable score ties, which are reported separately.
    rank = min(scores.size - 1, max(0, int(np.ceil((1.0 - target_fpr) * scores.size))))
    return float(scores[rank])


def _stratum(labels: np.ndarray, scores: np.ndarray, positive_mask: np.ndarray,
             negative_mask: np.ndarray, threshold: float) -> dict:
    pos = np.asarray(positive_mask, dtype=bool) & (labels == 1)
    neg = np.asarray(negative_mask, dtype=bool) & (labels == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    recovered = int((scores[pos] >= threshold).sum())
    result = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recovered": recovered,
        "completeness": float(recovered / n_pos) if n_pos else None,
        "completeness_wilson95": wilson_interval(recovered, n_pos),
    }
    if n_pos and n_neg:
        pair_labels = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
        pair_scores = np.r_[scores[pos], scores[neg]]
        metrics = detection_metrics(pair_labels, pair_scores)
        result["roc_auc"] = float(metrics["roc_auc"])
        result["average_precision"] = float(metrics["average_precision"])
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def _fixed_bins(theta_phys: np.ndarray, sigma: np.ndarray, baseline_days: float,
                n_raw: int) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Pre-specified, physically interpretable observability strata."""
    P, rprs, ars, impact = (theta_phys[:, 0], theta_phys[:, 2],
                             theta_phys[:, 3], theta_phys[:, 4])
    duration = transit_duration(P, rprs, ars, impact)
    n_transits = np.floor(baseline_days / np.maximum(P, 1e-12)) + 1.0
    cadence_days = baseline_days / max(n_raw, 1)
    n_in = np.maximum(duration / max(cadence_days, 1e-12), 1.0) * n_transits
    snr = (rprs ** 2) / np.maximum(sigma, 1e-12) * np.sqrt(n_in)
    return {
        "rprs": (rprs, ["<0.04", "0.04-0.09", ">=0.09"]),
        "impact_b": (impact, ["<0.35", "0.35-0.70", ">=0.70"]),
        "a_rs": (ars, ["<10", "10-25", ">=25"]),
        "expected_snr": (snr, ["<25", "25-75", ">=75"]),
        "period_days": (P, ["<3", "3-7", ">=7"]),
    }


def _three_bin_masks(values: np.ndarray, edges: tuple[float, float]) -> list[np.ndarray]:
    return [values < edges[0], (values >= edges[0]) & (values < edges[1]), values >= edges[1]]


def collect_disk_scores(data_dir: str, inference, prior, n: int,
                        batch_size: int) -> tuple[dict[str, np.ndarray], dict]:
    """Score a fixed provenance-bearing dataset without re-simulating BLS.

    The dataset is the immutable, independently generated experimental unit;
    GPU scoring is then cheap and deterministic.  Requiring the candidate
    period prevents a classifier-only audit from masking proposal failure.
    """
    meta_path = Path(data_dir) / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text())
    provenance = metadata.get("noise_provenance", {})
    if (int(metadata.get("dataset_schema_version", -1)) < 3
            or provenance.get("field") != "noise_source_index"
            or provenance.get("model_input") is not False):
        raise ValueError("--data-dir must use provenance dataset schema >= 3")
    ds = DiskDataset(data_dir, in_ram=True)
    required = {"global", "local", "theta_std", "d", "sigma_feat", "sigma", "regime",
                "noise_source_index", "fold_P"}
    missing = required.difference(ds.keys)
    if missing:
        raise ValueError(f"--data-dir is missing audit fields: {sorted(missing)}")
    take = min(int(n), len(ds))
    collected = {key: [] for key in (
        "labels", "scores", "theta", "sigma", "regime", "sources", "fold_period")}
    for start in range(0, take, max(1, int(batch_size))):
        idx = np.arange(start, min(take, start + max(1, int(batch_size))))
        batch = ds._gather(idx)
        score = inference.detect(
            batch["global"].astype(np.float32), batch["local"].astype(np.float32),
            batch["sigma_feat"].astype(np.float32),
            periodogram=(batch["periodogram"].astype(np.float32)
                         if "periodogram" in batch else None),
            ephem_feat=(batch["ephem_feat"].astype(np.float32)
                        if "ephem_feat" in batch else None),
            dil_feat=(batch["dil_feat"].astype(np.float32)
                      if "dil_feat" in batch else None),
        )
        collected["labels"].append(np.asarray(batch["d"], dtype=int))
        collected["scores"].append(np.asarray(score, dtype=float))
        collected["theta"].append(prior.std_to_physical(batch["theta_std"]))
        collected["sigma"].append(np.asarray(batch["sigma"], dtype=float))
        collected["regime"].append(np.asarray(batch["regime"], dtype=np.int8))
        collected["sources"].append(np.asarray(batch["noise_source_index"], dtype=np.int32))
        collected["fold_period"].append(np.asarray(batch["fold_P"], dtype=float))
    return ({key: np.concatenate(values) for key, values in collected.items()}, metadata)


def fair_bls_sim_config(sim_cfg):
    """Force a blind BLS candidate for both labels without altering physics.

    Older development checkpoints may have been trained with a mixed positive
    candidate distribution.  An identifiability audit must not inherit that
    convenience setting: using true ephemerides for any positive row would
    overstate real candidate-stage completeness.
    """
    return replace(
        sim_cfg,
        candidate_bls_positive_fraction=1.0,
        candidate_bls_negative_fraction=1.0,
        candidate_jitter_fraction=0.0,
        candidate_harmonic_fraction=0.0,
        candidate_random_positive_fraction=0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="frozen detector checkpoint")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--noise-lib",
                        help="held-out real-residual library; must contain target_ids")
    source.add_argument("--data-dir",
                        help="fixed provenance-bearing dataset generated on held-out targets")
    ap.add_argument("--n", type=int, default=5000,
                    help="independent injections; use >=5000 for a decision")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--min-source-positives", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n < 2 or not 0.0 < args.target_fpr < 1.0:
        raise SystemExit("--n must be >=2 and --target-fpr must be in (0, 1)")

    model, _, sim_cfg = load_checkpoint(args.ckpt, device=args.device)
    prior = TransitPrior.from_sim_config(sim_cfg)
    inference = TransitFlowInference(model, prior, sim_cfg, amp=args.amp)
    dataset_metadata = None
    if args.data_dir:
        collected, dataset_metadata = collect_disk_scores(
            args.data_dir, inference, prior, args.n, args.batch)
        labels, scores = collected["labels"], collected["scores"]
        theta, sigma = collected["theta"], collected["sigma"]
        regime, sources = collected["regime"], collected["sources"]
        fold_period = collected["fold_period"]
        source_label_count = len(dataset_metadata["noise_provenance"].get("labels", []))
        candidate_protocol = {
            "source": "pre_generated_blind_bls_dataset",
            "candidate_bls_positive_fraction": dataset_metadata["simulator_config"].get(
                "candidate_bls_positive_fraction"),
            "candidate_bls_negative_fraction": dataset_metadata["simulator_config"].get(
                "candidate_bls_negative_fraction"),
        }
    else:
        noise = NoiseLibrary.load(args.noise_lib)
        if not noise.available() or not noise.source_labels:
            raise SystemExit("--noise-lib must contain real segments and target_ids for source audit")
        audit_cfg = fair_bls_sim_config(sim_cfg)
        simulator = TransitSimulator(audit_cfg, prior=prior, noise_library=noise)
        rng = np.random.default_rng(args.seed)
        rows = {key: [] for key in (
            "labels", "scores", "theta", "sigma", "regime", "sources", "fold_period")}
        while sum(len(x) for x in rows["labels"]) < args.n:
            batch = simulator.simulate_batch(
                min(args.batch, args.n - sum(len(x) for x in rows["labels"])), rng)
            rows["labels"].append(np.asarray(batch["d"], dtype=int))
            rows["scores"].append(np.asarray(inference.detect(
                batch["global"], batch["local"], batch["sigma_feat"],
                periodogram=batch.get("periodogram"), ephem_feat=batch.get("ephem_feat"),
                dil_feat=batch.get("dil_feat")), dtype=float))
            rows["theta"].append(np.asarray(batch["theta_phys"], dtype=float))
            rows["sigma"].append(np.asarray(batch["sigma"], dtype=float))
            rows["regime"].append(np.asarray(batch["regime"], dtype=np.int8))
            rows["sources"].append(np.asarray(batch["noise_source_index"], dtype=np.int32))
            rows["fold_period"].append(np.asarray(batch["fold_P"], dtype=float))
        labels, scores = (np.concatenate(rows[key])[:args.n] for key in ("labels", "scores"))
        theta, sigma = (np.concatenate(rows[key])[:args.n] for key in ("theta", "sigma"))
        regime, sources = (np.concatenate(rows[key])[:args.n] for key in ("regime", "sources"))
        fold_period = np.concatenate(rows["fold_period"])[:args.n]
        source_label_count = len(noise.source_labels)
        candidate_protocol = {
            "source": "blind_astropy_bls_for_both_classes",
            "forced_positive_fraction": audit_cfg.candidate_bls_positive_fraction,
            "forced_negative_fraction": audit_cfg.candidate_bls_negative_fraction,
            "checkpoint_positive_fraction": sim_cfg.candidate_bls_positive_fraction,
            "checkpoint_negative_fraction": sim_cfg.candidate_bls_negative_fraction,
        }
    threshold = threshold_at_fpr(scores[labels == 0], args.target_fpr)
    empirical_fpr = float(np.mean(scores[labels == 0] >= threshold))
    overall = detection_metrics(labels, scores)
    pos = labels == 1

    edge_map = {
        "rprs": (0.04, 0.09), "impact_b": (0.35, 0.70),
        "a_rs": (10.0, 25.0), "expected_snr": (25.0, 75.0),
        "period_days": (3.0, 7.0),
    }
    strata = {}
    for name, (values, names) in _fixed_bins(
            theta, sigma, sim_cfg.baseline_days, sim_cfg.n_raw).items():
        strata[name] = {
            label: _stratum(labels, scores, mask, labels == 0, threshold)
            for label, mask in zip(names, _three_bin_masks(values, edge_map[name]))
        }

    source_report = {}
    real_pos = pos & (regime == 0) & (sources >= 0)
    for source_index in np.unique(sources[real_pos]):
        mask = sources == source_index
        n_pos = int((mask & pos).sum())
        if n_pos < args.min_source_positives:
            continue
        source_report[str(int(source_index))] = _stratum(
            labels, scores, mask, mask, threshold)
    relative_period_error = np.abs(fold_period[pos] / np.maximum(theta[pos, 0], 1e-12) - 1.0)
    proposal_recovery = int((relative_period_error <= 0.01).sum())

    report = {
        "report_schema_version": 1,
        "purpose": "identifiability audit; not a publication gate relaxation",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "noise_library": (str(Path(args.noise_lib).resolve()) if args.noise_lib else None),
        "data_dir": (str(Path(args.data_dir).resolve()) if args.data_dir else None),
        "dataset_config_sha256": (
            dataset_metadata.get("config_hash") if dataset_metadata else None),
        "seed": int(args.seed),
        "n": int(args.n),
        "candidate_protocol": candidate_protocol,
        "operating_point": {
            "predeclared_target_fpr": float(args.target_fpr),
            "score_threshold": threshold,
            "empirical_fpr_including_ties": empirical_fpr,
        },
        "overall": {key: float(overall[key]) for key in ("roc_auc", "average_precision")},
        "overall_completeness": _stratum(labels, scores, pos, labels == 0, threshold),
        "bls_top1_period_recovery_within_1pct": {
            "n_planets": int(pos.sum()), "recovered": proposal_recovery,
            "fraction": float(proposal_recovery / max(int(pos.sum()), 1)),
            "wilson95": wilson_interval(proposal_recovery, int(pos.sum())),
        },
        "predeclared_strata": strata,
        "real_noise_source_strata": source_report,
        "source_label_count": source_label_count,
        "interpretation_rule": (
            "Do not rerun the same detector if low completeness is broad across "
            "sources and low-information strata.  First compare a genuinely "
            "independent raw-light-curve reference detector or redefine the "
            "scientific population using a published completeness domain."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({
        "roc_auc": report["overall"]["roc_auc"],
        "average_precision": report["overall"]["average_precision"],
        "threshold": threshold,
        "empirical_fpr": empirical_fpr,
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
