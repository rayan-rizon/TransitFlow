"""Candidate-independent data contract for Stage-A transit detection.

Stage A must decide whether a chronological light curve contains a transit-like
signal *before* any period/epoch candidate is accepted.  It therefore cannot
use candidate-folded views or candidate-masked detrending.  The returned
configuration is intentionally separate from posterior training and is only
valid for a development detector experiment.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

from .simulator import SimConfig


def raw_detector_sim_config(sim_cfg: SimConfig, *, n_period_bins: int = 1024) -> SimConfig:
    """Return a candidate-free, high-resolution evidence configuration.

    The global view and periodogram are constructed before any candidate-local
    operation.  Candidate-based flattening is disabled, and no BLS/jitter/
    harmonic/random proposal is drawn.  The simulator still produces local
    arrays for disk-format compatibility, but Stage-A code never receives them.
    """
    if n_period_bins < 64:
        raise ValueError("raw detector needs at least 64 period bins")
    return replace(
        sim_cfg,
        flatten_views=False,
        use_periodogram=True,
        n_period_bins=int(n_period_bins),
        candidate_bls_positive_fraction=0.0,
        candidate_bls_negative_fraction=0.0,
        candidate_jitter_fraction=0.0,
        candidate_harmonic_fraction=0.0,
        candidate_random_positive_fraction=0.0,
    )


def validate_raw_detector_dataset(path: str | Path) -> dict:
    """Verify the immutable Stage-A data contract before training or scoring."""
    root = Path(path)
    with (root / "dataset_meta.json").open() as f:
        meta = json.load(f)
    cfg = meta.get("simulator_config", {})
    provenance = meta.get("noise_provenance", {})
    failures: list[str] = []
    if int(meta.get("dataset_schema_version", -1)) < 3:
        failures.append("dataset schema lacks audit provenance")
    if bool(cfg.get("flatten_views", True)):
        failures.append("candidate-dependent flattening is enabled")
    if not bool(cfg.get("use_periodogram", False)):
        failures.append("candidate-free periodogram evidence is absent")
    if int(cfg.get("n_period_bins", 0)) < 64:
        failures.append("periodogram resolution is too low")
    for key in (
        "candidate_bls_positive_fraction", "candidate_bls_negative_fraction",
        "candidate_jitter_fraction", "candidate_harmonic_fraction",
        "candidate_random_positive_fraction",
    ):
        if float(cfg.get(key, 0.0)) != 0.0:
            failures.append(f"candidate augmentation is enabled: {key}")
    if (provenance.get("field") != "noise_source_index"
            or provenance.get("model_input") is not False):
        failures.append("source provenance is missing or leaks into model inputs")
    return {
        "pass": not failures,
        "path": str(root),
        "config_hash": meta.get("config_hash"),
        "source_labels": list(provenance.get("labels", [])),
        "n_rows": int(meta.get("n_total", 0)),
        "n_period_bins": int(cfg.get("n_period_bins", 0)),
        "failures": failures,
    }
