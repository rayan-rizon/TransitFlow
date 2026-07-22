#!/usr/bin/env python3
"""Run the predeclared publishable TransitFlow gate suite on a Vast box.

The script is intentionally conservative: it records the environment, runs a
bounded smoke test, validates the noise library, generates or reuses disk data,
trains `latest.pt`, evaluates synthetic gates, runs baselines, runs quality-gated
real validation with fixed-ephemeris MCMC, and writes one `gate_report.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


FAST_PYTEST = [
    "tests/test_calibration.py",
    "tests/test_build_noise_library.py",
    "tests/test_evaluate_gates.py",
    "tests/test_inference.py",
    "tests/test_noise.py",
    "tests/test_select_noise_targets.py",
    "tests/test_simulator.py",
    "tests/test_data.py",
    "tests/test_baselines.py",
]

_GIB = 1024 ** 3
_MIB = 1024 ** 2


def full_run_disk_preflight(path: Path, minimum_free_gib: float) -> dict:
    """Record and enforce the storage reserve required by a full run.

    The publication configuration persists three independent datasets: posterior
    training, all-BLS detector training, and held-out all-BLS detector
    validation.  A full run must retain room for those datasets, checkpoints,
    and result artifacts rather than discovering an out-of-space condition
    after provenance has been partially generated.  Fast and smoke runs are
    deliberately excluded because their bounded datasets are diagnostic only.
    """
    if minimum_free_gib <= 0:
        raise ValueError("minimum_free_gib must be positive")
    usage = shutil.disk_usage(path)
    gib = 1024 ** 3
    required_bytes = int(minimum_free_gib * gib)
    return {
        "path": str(path),
        "available_bytes": int(usage.free),
        "available_gib": float(usage.free / gib),
        "required_free_gib": float(minimum_free_gib),
        "pass": bool(usage.free >= required_bytes),
    }


def memory_limit_bytes(cgroup_root: Path = Path("/sys/fs/cgroup")) -> tuple[int, str]:
    """Return the enforced memory limit, preferring the cgroup v2 limit."""
    cgroup_limit = cgroup_root / "memory.max"
    try:
        raw = cgroup_limit.read_text().strip()
        if raw != "max":
            value = int(raw)
            if value > 0:
                return value, "cgroup_v2"
    except (OSError, ValueError):
        pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")), "system"


def dataset_worker_preflight(requested_workers: int, *, memory_bytes: int,
                             reserve_gib: float, worker_mib: float) -> dict:
    """Cap spawned Astropy workers to a reproducible conservative RAM budget."""
    if requested_workers < 1:
        raise ValueError("requested_workers must be positive")
    if memory_bytes <= 0:
        raise ValueError("memory_bytes must be positive")
    if reserve_gib < 0 or worker_mib <= 0:
        raise ValueError("worker memory reserve and estimate must be positive")
    reserve_bytes = int(reserve_gib * _GIB)
    worker_bytes = int(worker_mib * _MIB)
    budget_bytes = max(0, int(memory_bytes) - reserve_bytes)
    capacity = max(1, budget_bytes // worker_bytes)
    effective = min(int(requested_workers), int(capacity))
    return {
        "requested_workers": int(requested_workers),
        "effective_workers": int(effective),
        "memory_limit_bytes": int(memory_bytes),
        "reserve_gib": float(reserve_gib),
        "worker_mib": float(worker_mib),
        "capacity_workers": int(capacity),
        "cap_applied": bool(effective < requested_workers),
    }


def run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n== attempt {time.time():.6f} ==\n")
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=log,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"stage failed ({proc.returncode}): {' '.join(cmd)}; log={log_path}")


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# Predeclared gate revision (2026-07-19), fixed BEFORE any new run.
# Rationale (manuscript/FULL_TEST_RUNBOOK.md, "Gate revision 2026-07-19"):
# the historical 0.99 blind-detection AUC target was calibrated on the
# privileged oracle-ephemeris diagnostic and is unattainable in the fair
# blind-candidate protocol, where a large fraction of injections sit at or
# below the single-sector information limit (held-out completeness 13% for
# expected S/N < 25).  Detection performance is therefore gated inside a
# predeclared detectable domain (expected transit S/N >= 25, the identifiability
# audit's own predeclared bin edges) plus an overall floor that guards against
# gross regressions.  The values were fixed from the frozen 2026-07-16
# development identifiability audit before the next experiment was launched.
GATE_REVISION = "2026-07-19_predeclared_v2"
IDENTIFIABILITY_OVERALL_AUC_MIN = 0.85
IDENTIFIABILITY_DOMAIN_AUC_MIN = 0.93
IDENTIFIABILITY_SNR_DOMAIN_BINS = ("25-75", ">=75")
FAIR_DETECTION_AUC_MIN = 0.88
# Real-data MCMC agreement limits are per parameter: b and (to a lesser degree)
# a/Rs are weakly identified in single-sector photometry, so their agreement
# limits are wider than the well-measured depth parameter RpRs.  Values were
# predeclared from frozen *development* evidence (seed-0 is a recorded failed
# development run) before the next frozen experiment.
MCMC_PRIOR_FRACTION_LIMITS = {"RpRs": 0.10, "aRs": 0.10, "b": 0.15}
MCMC_WIDTH_FRACTION_LIMITS = {"RpRs": 0.60, "aRs": 0.90, "b": 0.90}


def identifiability_preflight(report_path: Path, min_sources: int,
                              required_auc: float = IDENTIFIABILITY_OVERALL_AUC_MIN,
                              required_domain_auc: float = IDENTIFIABILITY_DOMAIN_AUC_MIN,
                              domain_snr_bins: tuple[str, ...] = IDENTIFIABILITY_SNR_DOMAIN_BINS,
                              ) -> dict:
    """Fail closed before a full run when blind detection is not identifiable.

    This development audit is not a substitute for the final external-lockbox
    gate. It prevents allocating that irreversible experiment when a fixed,
    source-provenance-bearing BLS trial already rules out the predeclared
    end-to-end discrimination targets.  The discrimination requirement applies
    inside the predeclared detectable domain (expected transit S/N >= 25);
    the overall AUC keeps only a regression floor because the full injection
    population deliberately includes near-information-limit cases.
    """
    failures: list[str] = []
    try:
        report = read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "pass": False,
            "report_path": str(report_path),
            "failures": [f"unreadable report: {type(exc).__name__}"],
        }
    protocol = report.get("candidate_protocol", {})
    source_count = int(report.get("source_label_count", 0))
    source_rows = report.get("real_noise_source_strata", {})
    auc = report.get("overall", {}).get("roc_auc")
    proposal = report.get("bls_top1_period_recovery_within_1pct", {})
    snr_strata = report.get("predeclared_strata", {}).get("expected_snr", {})
    if int(report.get("report_schema_version", -1)) != 1:
        failures.append("unsupported identifiability report schema")
    if protocol.get("source") != "pre_generated_blind_bls_dataset":
        failures.append("report does not use a fixed blind-BLS dataset")
    if (float(protocol.get("candidate_bls_positive_fraction", 0.0)) != 1.0
            or float(protocol.get("candidate_bls_negative_fraction", 0.0)) != 1.0):
        failures.append("report does not use BLS candidates for both labels")
    if source_count < min_sources or len(source_rows) < min_sources:
        failures.append(f"source audit has fewer than {min_sources} evaluable targets")
    if not isinstance(auc, (int, float)) or not 0.0 <= float(auc) <= 1.0:
        failures.append("report has no finite overall ROC-AUC")
    elif float(auc) < required_auc:
        failures.append(f"blind-BLS overall ROC-AUC below floor {required_auc:.2f}")
    observed_domain_auc: dict[str, float | None] = {}
    for bin_name in domain_snr_bins:
        bin_auc = snr_strata.get(bin_name, {}).get("roc_auc")
        if not isinstance(bin_auc, (int, float)) or not 0.0 <= float(bin_auc) <= 1.0:
            observed_domain_auc[bin_name] = None
            failures.append(
                f"report lacks a finite ROC-AUC for expected-S/N bin {bin_name!r}")
            continue
        observed_domain_auc[bin_name] = float(bin_auc)
        if float(bin_auc) < required_domain_auc:
            failures.append(
                f"in-domain ROC-AUC for expected-S/N bin {bin_name!r} below "
                f"{required_domain_auc:.2f}")
    if not isinstance(proposal.get("fraction"), (int, float)):
        failures.append("report lacks BLS proposal-recovery diagnostic")
    return {
        "pass": not failures,
        "gate_revision": GATE_REVISION,
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "required_auc": float(required_auc),
        "required_domain_auc": float(required_domain_auc),
        "domain_snr_bins": list(domain_snr_bins),
        "observed_auc": None if not isinstance(auc, (int, float)) else float(auc),
        "observed_domain_auc": observed_domain_auc,
        "min_sources": int(min_sources),
        "source_label_count": source_count,
        "source_rows": len(source_rows),
        "bls_top1_period_recovery_within_1pct": proposal.get("fraction"),
        "failures": failures,
    }


def git_sha(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                       text=True).strip()
    except Exception:
        return None


def validate_noise_lib(path: Path) -> dict:
    import numpy as np

    if not path.exists():
        raise SystemExit(f"noise library not found: {path}")
    arr = np.load(path)
    seg = arr["segments"] if hasattr(arr, "files") and "segments" in arr.files else None
    if seg is None or seg.ndim != 2 or seg.shape[0] == 0:
        raise SystemExit(f"invalid noise library: {path}")
    target_ids = arr["target_ids"] if "target_ids" in arr.files else None
    return {
        "path": str(path),
        "n_segments": int(seg.shape[0]),
        "segment_length": int(seg.shape[1]),
        "median": float(np.nanmedian(seg)),
        "std": float(np.nanstd(seg)),
        "has_target_ids": target_ids is not None,
        "n_unique_targets": 0 if target_ids is None else int(
            len(np.unique(target_ids.astype(str)))),
        "sampling_unit": "source_target_uniform_then_segment_v1"
        if target_ids is not None else "segment_uniform_legacy",
    }


def require_noise_target_count(metadata: dict, minimum: int) -> None:
    count = int(metadata.get("n_unique_targets", 0))
    if count < int(minimum):
        raise SystemExit(
            f"noise library has {count} independent targets; at least {minimum} "
            "are required before a research gate")


def noise_target_set(path: Path) -> set[str]:
    """Return source-target identifiers, failing closed when absent."""
    import numpy as np

    with np.load(path) as arr:
        if "target_ids" not in arr.files:
            raise SystemExit(f"noise library lacks target_ids: {path}")
        target_ids = np.asarray(arr["target_ids"]).astype(str)
    if target_ids.size == 0:
        raise SystemExit(f"noise library has no target_ids: {path}")
    return set(target_ids.tolist())


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulator_config_for_candidate_domain(config_path: str,
                                          candidate_domain: str = "publication"):
    """Build the exact simulator configuration used for a candidate domain."""
    try:
        from scripts._config import build_configs
    except ImportError:  # direct ``python scripts/run_publishable_vast.py``
        from _config import build_configs

    configs = build_configs(config_path)
    if candidate_domain == "publication":
        return configs
    if candidate_domain != "bls_detection":
        raise ValueError(f"unknown candidate domain: {candidate_domain}")
    # Keep this transformation byte-for-byte aligned with generate_data.py.
    sim_cfg = configs["simulator"]
    sim_cfg.candidate_bls_positive_fraction = 1.0
    sim_cfg.candidate_bls_negative_fraction = 1.0
    sim_cfg.candidate_jitter_fraction = 0.0
    sim_cfg.candidate_harmonic_fraction = 0.0
    sim_cfg.candidate_random_positive_fraction = 0.0
    return configs


def validate_existing_dataset(data_dir: Path, config_path: str, n_total: int,
                              shard_size: int, seed: int,
                              noise_lib: Path | None,
                              candidate_domain: str = "publication",
                              require_complete: bool = True) -> bool:
    """Validate a complete dataset or a provenance-safe resumable prefix."""
    import numpy as np

    meta_path = data_dir / "dataset_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = read_json(meta_path)
        expected_shards = (n_total + shard_size - 1) // shard_size
        configs = simulator_config_for_candidate_domain(
            config_path, candidate_domain)
        config_json = json.dumps(
            asdict(configs["simulator"]), sort_keys=True)
        expected_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        expected_names = {
            f"shard_{idx:05d}.npz" for idx in range(expected_shards)}
        actual_names = {path.name for path in data_dir.glob("shard_*.npz")}
        expected_sampling_unit = "synthetic_only"
        if noise_lib is not None:
            noise_description = validate_noise_lib(noise_lib)
            expected_sampling_unit = (
                "source_target_uniform_then_segment_v1"
                if noise_description.get("has_target_ids")
                else "segment_uniform_legacy")
        required_keys = {
            "global", "local", "theta_std", "theta_char_std", "d",
            "sigma_feat", "sigma", "posterior_valid", "regime",
            "noise_source_index", "fold_P",
        }
        if configs["model"].posterior_transform == "prior_normal":
            required_keys.add("theta_char_prior_normal")
        shard_schema_valid = True
        names_to_validate = expected_names if require_complete else actual_names
        for name in sorted(names_to_validate):
            idx = int(name.removeprefix("shard_").removesuffix(".npz"))
            expected_rows = min(shard_size, n_total - idx * shard_size)
            with np.load(data_dir / name) as shard:
                if (not required_keys.issubset(shard.files)
                        or any(len(shard[key]) != expected_rows
                               for key in required_keys)):
                    shard_schema_valid = False
                    break
        return bool(
            int(meta.get("dataset_schema_version", -1)) == 3
            and set(meta.get("posterior_target_fields", [])) >= {
                "theta_char_std", "theta_char_prior_normal"}
            and int(meta.get("n_total", -1)) == n_total
            and int(meta.get("n_shards", -1)) == expected_shards
            and int(meta.get("shard_size", -1)) == shard_size
            and int(meta.get("seed", -1)) == seed
            and meta.get("config_hash") == expected_hash
            and meta.get("noise_lib_sha256") == _sha256_file(noise_lib)
            and meta.get("noise_sampling_unit") == expected_sampling_unit
            and isinstance(meta.get("noise_provenance"), dict)
            and meta["noise_provenance"].get("field") == "noise_source_index"
            and meta["noise_provenance"].get("model_input") is False
            and (actual_names == expected_names if require_complete
                 else actual_names.issubset(expected_names))
            and all((data_dir / name).stat().st_size > 0 for name in names_to_validate)
            and shard_schema_valid
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            EOFError):
        return False


def prepare_noise_splits(path: Path, out_dir: Path, seed: int,
                         eval_fraction: float = 0.2) -> tuple[Path, Path, dict]:
    """Create deterministic source-target-disjoint train/evaluation libraries."""
    import numpy as np

    arr = np.load(path)
    if "target_ids" not in arr.files:
        raise SystemExit(
            "noise library lacks target_ids; rebuild it with "
            "scripts/build_noise_library.py before a publication run")
    segments = np.asarray(arr["segments"])
    target_ids = np.asarray(arr["target_ids"]).astype(str)
    if len(target_ids) != len(segments) or len(segments) == 0:
        raise SystemExit("noise target_ids must be nonempty and match segments")
    targets = np.unique(target_ids)
    if len(targets) < 2:
        raise SystemExit("at least two source targets are required for a group split")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_eval = min(max(1, int(np.ceil(len(targets) * eval_fraction))), len(targets) - 1)
    eval_targets = targets[:n_eval]
    eval_mask = np.isin(target_ids, eval_targets)
    if not eval_mask.any() or eval_mask.all():
        raise SystemExit("source-target noise split produced an empty partition")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "noise_train.npz"
    eval_path = out_dir / "noise_eval.npz"
    np.savez_compressed(
        train_path, segments=segments[~eval_mask], target_ids=target_ids[~eval_mask])
    np.savez_compressed(
        eval_path, segments=segments[eval_mask], target_ids=target_ids[eval_mask])
    meta = {
        "seed": int(seed),
        "eval_fraction_requested": float(eval_fraction),
        "train_targets": sorted(np.unique(target_ids[~eval_mask]).tolist()),
        "eval_targets": sorted(np.unique(target_ids[eval_mask]).tolist()),
        "n_train_segments": int((~eval_mask).sum()),
        "n_eval_segments": int(eval_mask.sum()),
        "target_overlap": sorted(set(target_ids[~eval_mask]) & set(target_ids[eval_mask])),
    }
    return train_path, eval_path, meta


def prepare_noise_train_calibration_split(
    path: Path,
    out_dir: Path,
    seed: int,
    calibration_fraction: float = 0.2,
) -> tuple[Path, Path, dict]:
    """Split development targets while reserving evaluation externally."""
    import numpy as np

    with np.load(path) as arr:
        if "target_ids" not in arr.files:
            raise SystemExit("development noise library lacks target_ids")
        segments = np.asarray(arr["segments"])
        target_ids = np.asarray(arr["target_ids"]).astype(str)
    if len(target_ids) != len(segments) or len(segments) == 0:
        raise SystemExit("noise target_ids must be nonempty and match segments")
    targets = np.unique(target_ids)
    if len(targets) < 3:
        raise SystemExit("at least three development targets are required")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_cal = max(1, int(np.ceil(len(targets) * calibration_fraction)))
    if n_cal >= len(targets):
        raise SystemExit("calibration fraction leaves no training targets")
    calibration_targets = targets[:n_cal]
    training_targets = targets[n_cal:]
    calibration_mask = np.isin(target_ids, calibration_targets)
    training_mask = np.isin(target_ids, training_targets)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "noise_train.npz"
    calibration_path = out_dir / "noise_calibration.npz"
    np.savez_compressed(train_path, segments=segments[training_mask],
                        target_ids=target_ids[training_mask])
    np.savez_compressed(calibration_path, segments=segments[calibration_mask],
                        target_ids=target_ids[calibration_mask])
    meta = {
        "seed": int(seed),
        "calibration_fraction_requested": float(calibration_fraction),
        "targets": {
            "train": sorted(training_targets.tolist()),
            "calibration": sorted(calibration_targets.tolist()),
        },
        "n_segments": {
            "train": int(training_mask.sum()),
            "calibration": int(calibration_mask.sum()),
        },
        "train_calibration_overlap": sorted(
            set(training_targets) & set(calibration_targets)),
        "evaluation_role": "external_frozen_publication_lockbox",
        "sampling_unit": "source_target_uniform_then_segment_v1",
    }
    return train_path, calibration_path, meta


def prepare_noise_train_validation_calibration_split(
    path: Path,
    out_dir: Path,
    seed: int,
    validation_fraction: float = 0.1,
    calibration_fraction: float = 0.2,
) -> tuple[Path, Path, Path, dict]:
    """Split development targets while reserving evaluation externally.

    Gradient training, checkpoint selection, and posterior calibration must use
    different source targets.  The publication evaluation role is supplied by
    a separately frozen archive and is therefore not carved from ``path``.
    """
    import numpy as np

    if not 0.0 < validation_fraction < 1.0:
        raise SystemExit("validation fraction must be strictly between 0 and 1")
    if not 0.0 < calibration_fraction < 1.0:
        raise SystemExit("calibration fraction must be strictly between 0 and 1")
    if validation_fraction + calibration_fraction >= 1.0:
        raise SystemExit(
            "validation/calibration fractions leave no training targets")
    with np.load(path) as arr:
        if "target_ids" not in arr.files:
            raise SystemExit("development noise library lacks target_ids")
        segments = np.asarray(arr["segments"])
        target_ids = np.asarray(arr["target_ids"]).astype(str)
    if len(target_ids) != len(segments) or len(segments) == 0:
        raise SystemExit("noise target_ids must be nonempty and match segments")
    targets = np.unique(target_ids)
    if len(targets) < 4:
        raise SystemExit("at least four development targets are required")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_cal = max(1, int(np.ceil(len(targets) * calibration_fraction)))
    n_val = max(1, int(np.ceil(len(targets) * validation_fraction)))
    if n_cal + n_val >= len(targets):
        raise SystemExit(
            "validation/calibration fractions leave no training targets")
    calibration_targets = targets[:n_cal]
    validation_targets = targets[n_cal:n_cal + n_val]
    training_targets = targets[n_cal + n_val:]
    role_targets = {
        "train": training_targets,
        "validation": validation_targets,
        "calibration": calibration_targets,
    }
    role_masks = {
        role: np.isin(target_ids, selected)
        for role, selected in role_targets.items()
    }
    if not np.all(np.logical_or.reduce(list(role_masks.values()))):
        raise SystemExit("development noise split did not assign every segment")
    if not all(mask.any() for mask in role_masks.values()):
        raise SystemExit("development noise split produced an empty partition")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        role: out_dir / f"noise_{role}.npz"
        for role in role_targets
    }
    for role, output_path in paths.items():
        mask = role_masks[role]
        np.savez_compressed(
            output_path, segments=segments[mask], target_ids=target_ids[mask])
    sets = {
        role: sorted(selected.tolist())
        for role, selected in role_targets.items()
    }
    overlaps = {
        "train_validation": sorted(set(sets["train"]) & set(sets["validation"])),
        "train_calibration": sorted(set(sets["train"]) & set(sets["calibration"])),
        "validation_calibration": sorted(
            set(sets["validation"]) & set(sets["calibration"])),
    }
    meta = {
        "seed": int(seed),
        "validation_fraction_requested": float(validation_fraction),
        "calibration_fraction_requested": float(calibration_fraction),
        "targets": sets,
        "n_segments": {
            role: int(mask.sum()) for role, mask in role_masks.items()
        },
        "overlaps": overlaps,
        "all_disjoint": not any(overlaps.values()),
        "evaluation_role": "external_frozen_publication_lockbox",
        "sampling_unit": "source_target_uniform_then_segment_v1",
    }
    return paths["train"], paths["validation"], paths["calibration"], meta


def prepare_noise_three_way_split(
    path: Path,
    out_dir: Path,
    seed: int,
    calibration_fraction: float = 0.2,
    eval_fraction: float = 0.2,
) -> tuple[Path, Path, Path, dict]:
    """Create deterministic target-disjoint train/calibration/evaluation sets."""
    import numpy as np

    arr = np.load(path)
    if "target_ids" not in arr.files:
        raise SystemExit("noise library lacks target_ids for three-way splitting")
    segments = np.asarray(arr["segments"])
    target_ids = np.asarray(arr["target_ids"]).astype(str)
    if len(target_ids) != len(segments) or len(segments) == 0:
        raise SystemExit("noise target_ids must be nonempty and match segments")
    targets = np.unique(target_ids)
    if len(targets) < 5:
        raise SystemExit("at least five source targets are required for a three-way split")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_eval = max(1, int(np.ceil(len(targets) * eval_fraction)))
    n_cal = max(1, int(np.ceil(len(targets) * calibration_fraction)))
    if n_eval + n_cal >= len(targets):
        raise SystemExit("calibration/evaluation fractions leave no training targets")
    eval_targets = targets[:n_eval]
    cal_targets = targets[n_eval:n_eval + n_cal]
    train_targets = targets[n_eval + n_cal:]
    eval_mask = np.isin(target_ids, eval_targets)
    cal_mask = np.isin(target_ids, cal_targets)
    train_mask = np.isin(target_ids, train_targets)
    if not np.all(train_mask | cal_mask | eval_mask):
        raise SystemExit("three-way noise split did not assign every segment")
    if not train_mask.any() or not cal_mask.any() or not eval_mask.any():
        raise SystemExit("three-way noise split produced an empty partition")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "noise_train.npz"
    cal_path = out_dir / "noise_calibration.npz"
    eval_path = out_dir / "noise_eval.npz"
    np.savez_compressed(train_path, segments=segments[train_mask],
                        target_ids=target_ids[train_mask])
    np.savez_compressed(cal_path, segments=segments[cal_mask],
                        target_ids=target_ids[cal_mask])
    np.savez_compressed(eval_path, segments=segments[eval_mask],
                        target_ids=target_ids[eval_mask])
    sets = {
        "train": sorted(train_targets.tolist()),
        "calibration": sorted(cal_targets.tolist()),
        "evaluation": sorted(eval_targets.tolist()),
    }
    overlaps = {
        "train_calibration": sorted(set(sets["train"]) & set(sets["calibration"])),
        "train_evaluation": sorted(set(sets["train"]) & set(sets["evaluation"])),
        "calibration_evaluation": sorted(
            set(sets["calibration"]) & set(sets["evaluation"])),
    }
    meta = {
        "seed": int(seed),
        "calibration_fraction_requested": float(calibration_fraction),
        "eval_fraction_requested": float(eval_fraction),
        "targets": sets,
        "n_segments": {
            "train": int(train_mask.sum()),
            "calibration": int(cal_mask.sum()),
            "evaluation": int(eval_mask.sum()),
        },
        "overlaps": overlaps,
        "all_disjoint": not any(overlaps.values()),
        "sampling_unit": "source_target_uniform_then_segment_v1",
    }
    return train_path, cal_path, eval_path, meta


def prepare_noise_four_way_split(
    path: Path,
    out_dir: Path,
    seed: int,
    validation_fraction: float = 0.1,
    calibration_fraction: float = 0.2,
    eval_fraction: float = 0.2,
) -> tuple[Path, Path, Path, Path, dict]:
    """Create target-disjoint train/validation/calibration/evaluation sets."""
    import numpy as np

    fractions = {
        "validation": validation_fraction,
        "calibration": calibration_fraction,
        "evaluation": eval_fraction,
    }
    for role, fraction in fractions.items():
        if not 0.0 < fraction < 1.0:
            raise SystemExit(
                f"{role} fraction must be strictly between 0 and 1")
    if sum(fractions.values()) >= 1.0:
        raise SystemExit(
            "validation/calibration/evaluation fractions leave no training targets")
    with np.load(path) as arr:
        if "target_ids" not in arr.files:
            raise SystemExit("noise library lacks target_ids for four-way splitting")
        segments = np.asarray(arr["segments"])
        target_ids = np.asarray(arr["target_ids"]).astype(str)
    if len(target_ids) != len(segments) or len(segments) == 0:
        raise SystemExit("noise target_ids must be nonempty and match segments")
    targets = np.unique(target_ids)
    if len(targets) < 4:
        raise SystemExit("at least four source targets are required for a four-way split")
    rng = np.random.default_rng(seed)
    targets = targets[rng.permutation(len(targets))]
    n_eval = max(1, int(np.ceil(len(targets) * eval_fraction)))
    n_cal = max(1, int(np.ceil(len(targets) * calibration_fraction)))
    n_val = max(1, int(np.ceil(len(targets) * validation_fraction)))
    if n_eval + n_cal + n_val >= len(targets):
        raise SystemExit(
            "validation/calibration/evaluation fractions leave no training targets")
    role_targets = {
        "evaluation": targets[:n_eval],
        "calibration": targets[n_eval:n_eval + n_cal],
        "validation": targets[n_eval + n_cal:n_eval + n_cal + n_val],
        "train": targets[n_eval + n_cal + n_val:],
    }
    role_masks = {
        role: np.isin(target_ids, selected)
        for role, selected in role_targets.items()
    }
    if not np.all(np.logical_or.reduce(list(role_masks.values()))):
        raise SystemExit("four-way noise split did not assign every segment")
    if not all(mask.any() for mask in role_masks.values()):
        raise SystemExit("four-way noise split produced an empty partition")
    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": "noise_train.npz",
        "validation": "noise_validation.npz",
        "calibration": "noise_calibration.npz",
        "evaluation": "noise_eval.npz",
    }
    paths = {role: out_dir / name for role, name in filenames.items()}
    for role, output_path in paths.items():
        mask = role_masks[role]
        np.savez_compressed(
            output_path, segments=segments[mask], target_ids=target_ids[mask])
    sets = {
        role: sorted(selected.tolist())
        for role, selected in role_targets.items()
    }
    roles = ("train", "validation", "calibration", "evaluation")
    overlaps = {
        f"{left}_{right}": sorted(set(sets[left]) & set(sets[right]))
        for i, left in enumerate(roles)
        for right in roles[i + 1:]
    }
    meta = {
        "seed": int(seed),
        "validation_fraction_requested": float(validation_fraction),
        "calibration_fraction_requested": float(calibration_fraction),
        "eval_fraction_requested": float(eval_fraction),
        "targets": sets,
        "n_segments": {
            role: int(mask.sum()) for role, mask in role_masks.items()
        },
        "overlaps": overlaps,
        "all_disjoint": not any(overlaps.values()),
        "sampling_unit": "source_target_uniform_then_segment_v1",
    }
    return (
        paths["train"], paths["validation"], paths["calibration"],
        paths["evaluation"], meta,
    )


def _gate_value(metrics: dict, key: str) -> bool:
    gate = metrics.get("gate_status", {})
    return bool(gate.get(key, False))


def build_synthetic_gate_report(metrics: dict, split_meta: dict | None) -> dict:
    """Build a fail-closed development report before costly downstream gates."""
    metric_status = metrics.get("gate_status", {})
    required_names = [
        "characterization_sbc_familywise_alpha_0.05",
        "characterization_coverage_error_le_0.03",
    ]
    # ``evaluate.py`` measures an oracle/exact-candidate score, which is useful
    # only as a diagnostic.  Once the runner has replaced it with the blind BLS
    # candidate score, however, it is the actual detection claim and must block
    # a downstream/full run.  Keeping it diagnostic in that case allowed a
    # report to say its declared development gates passed while its fair AUC
    # explicitly failed.
    if metrics.get("detection_candidate_source") == "bls":
        required_names.append("detection_auc_ge_min")
    status = {
        key: bool(metric_status.get(key, False))
        for key in required_names
    }
    diagnostic_status = {
        key: value for key, value in metric_status.items()
        if key not in required_names
    }
    if split_meta is not None:
        status["noise_target_split_disjoint"] = bool(
            split_meta.get("all_disjoint", False))
        status["checkpoint_validation_disjoint"] = bool(
            split_meta.get("all_disjoint", False))
        status["posterior_calibration_disjoint"] = bool(
            split_meta.get("all_disjoint", False))
    return {
        "stage": "synthetic_development_gate",
        "publication_evidence": False,
        "status": status,
        "diagnostic_status": diagnostic_status,
        "all_declared_synthetic_gates_pass": bool(status) and all(status.values()),
        "metrics": metrics,
    }


def external_lockbox_synthetic_failure_blocks_downstream(
    report: dict,
    *,
    smoke: bool,
    fast_check: bool,
    has_external_lockbox: bool,
) -> bool:
    """Fail closed before baselines/real-data stages after a lockbox failure.

    A full publication run consumes an external lockbox at synthetic evaluation.
    Once a declared synthetic gate fails, later BLS/TLS, speed, and real-data
    stages cannot repair that evidence.  Stopping here prevents needless cost
    and, importantly, prevents a resumed supervisor job from treating the same
    lockbox as a fresh validation opportunity.
    """
    return bool(
        not smoke
        and not fast_check
        and has_external_lockbox
        and not report.get("all_declared_synthetic_gates_pass", False)
    )


def build_gate_report(synthetic: dict, real: dict, bls: dict, speed: dict,
                      thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {}
    min_real_detected = int(thresholds.get("min_real_detected", 27))
    min_mcmc_n = int(thresholds.get("min_mcmc_n", 16))
    max_w_prior = dict(thresholds.get(
        "max_wasserstein_prior_fraction", MCMC_PRIOR_FRACTION_LIMITS))
    max_w_width = dict(thresholds.get(
        "max_wasserstein_width_fraction", MCMC_WIDTH_FRACTION_LIMITS))
    min_speedup = float(thresholds.get("min_speedup", 1000.0))
    min_fair_detection_n = int(thresholds.get("min_fair_detection_n", 5000))
    min_fair_detection_auc = float(thresholds.get(
        "min_fair_detection_auc", FAIR_DETECTION_AUC_MIN))

    real_summary = real.get("summary", real)
    mcmc = real_summary.get("mcmc_agreement", {})
    char = ("RpRs", "aRs", "b")
    real_mcmc_n = min([mcmc.get(k, {}).get("n", 0) for k in char] or [0])
    prior_ok = all(
        mcmc.get(k, {}).get("median_wasserstein_prior_fraction", float("inf"))
        <= float(max_w_prior[k])
        for k in char
    )
    width_ok = all(
        mcmc.get(k, {}).get("median_wasserstein_width_fraction", float("inf"))
        <= float(max_w_width[k])
        for k in char
    )
    uncertainty = bls.get("uncertainty", {}).get("ci95", {})
    auc_gain_ci = uncertainty.get("auc_gain", [float("-inf"), float("inf")])
    ap_gain_ci = uncertainty.get("ap_gain", [float("-inf"), float("inf")])
    tls_required = bool(bls.get("tls_requested", False))
    tls = bls.get("tls")
    tls_uncertainty = bls.get("tls_uncertainty", {}).get("ci95", {})
    tls_auc_gain_ci = tls_uncertainty.get("auc_gain", [float("-inf"), float("inf")])
    tls_ap_gain_ci = tls_uncertainty.get("ap_gain", [float("-inf"), float("inf")])
    convergence = real_summary.get("gate_status", {})
    mcmc_converged = bool(
        convergence.get("mcmc_chain_length_ge_50_tau", False)
        and convergence.get("mcmc_effective_samples_ge_400", False)
        and convergence.get("mcmc_tail_effective_samples_ge_400", False)
        and convergence.get("mcmc_split_rhat_le_1.01", False)
    )
    raw_speed_pass = float(speed.get("speedup_x", 0.0)) >= min_speedup
    fair_detection_auc = float(
        bls.get("transitflow", {}).get("roc_auc", float("-inf")))
    matched_speed_pass = bool(
        speed.get("all_mcmc_converged", False)
        and float(speed.get("speedup_ci95", [0.0])[0]) >= min_speedup)
    status = {
        "synthetic_characterization_sbc_familywise_alpha_0.05":
            _gate_value(synthetic, "characterization_sbc_familywise_alpha_0.05"),
        "synthetic_characterization_coverage_error_le_0.03":
            _gate_value(synthetic, "characterization_coverage_error_le_0.03"),
        "real_quality_gated_sample_n_ge_30":
            int(real_summary.get("n_planets", 0)) >= int(
                thresholds.get("min_real_n", 30)),
        "real_quality_gated_detection_ge_27_of_30": bool(
            int(real_summary.get("n_planets", 0)) >= int(
                thresholds.get("min_real_n", 30))
            and int(real_summary.get("detection", {}).get("n_detected", 0))
            >= min_real_detected),
        "real_mcmc_n_ge_16": real_mcmc_n >= min_mcmc_n,
        "real_mcmc_prior_fraction_within_limit": prior_ok,
        "real_mcmc_width_fraction_within_limit": width_ok,
        "speedup_ge_1000x_at_converged_mcmc_reference":
            matched_speed_pass and mcmc_converged,
        "bls_baseline_regenerated": bool(
            bls.get("transitflow") and bls.get("bls")
            and int(bls.get("bls", {}).get("n_failed", 0)) == 0),
        "detection_candidate_ephemeris_from_bls":
            bls.get("candidate_source") == "bls",
        "fair_candidate_evaluation_n_ge_5000":
            int(bls.get("n", 0)) >= min_fair_detection_n,
        "fair_candidate_detection_auc_ge_min":
            fair_detection_auc >= min_fair_detection_auc,
        "fair_candidate_auc_gain_ci95_lower_gt_0":
            bool(auc_gain_ci and float(auc_gain_ci[0]) > 0.0),
        "fair_candidate_ap_gain_ci95_lower_gt_0":
            bool(ap_gain_ci and float(ap_gain_ci[0]) > 0.0),
        "tls_equal_sample_baseline_completed":
            (not tls_required) or bool(
                tls and int(tls.get("n", 0)) == int(bls.get("n", 0))
                and tls.get("paired_with_transitflow", False)
                and int(tls.get("n_failed", -1)) == 0),
    }
    if tls_required:
        status["fair_candidate_auc_gain_vs_tls_ci95_lower_gt_0"] = bool(
            tls_auc_gain_ci and float(tls_auc_gain_ci[0]) > 0.0)
        status["fair_candidate_ap_gain_vs_tls_ci95_lower_gt_0"] = bool(
            tls_ap_gain_ci and float(tls_ap_gain_ci[0]) > 0.0)
    if "importance_correction_min_ess_fraction_ge_0.05" in real_summary.get("gate_status", {}):
        status["importance_correction_min_ess_fraction_ge_0.05"] = bool(
            real_summary["gate_status"]["importance_correction_min_ess_fraction_ge_0.05"])
    for convergence_gate in (
        "mcmc_chain_length_ge_50_tau",
        "mcmc_effective_samples_ge_400",
        "mcmc_tail_effective_samples_ge_400",
        "mcmc_split_rhat_le_1.01",
    ):
        if convergence_gate in real_summary.get("gate_status", {}):
            status[convergence_gate] = bool(
                real_summary["gate_status"][convergence_gate])
    status["final_pass"] = all(status.values())
    diagnostic_status = {
        "oracle_candidate_detection_auc_ge_0.99":
            float(synthetic.get("oracle_detection", synthetic.get(
                "detection", {})).get("roc_auc", float("-inf"))) >= 0.99,
        "raw_speedup_ge_1000x": raw_speed_pass,
    }
    return {
        "gate_thresholds": {
            "revision": GATE_REVISION,
            "min_real_detected": min_real_detected,
            "min_mcmc_n": min_mcmc_n,
            "max_wasserstein_prior_fraction": max_w_prior,
            "max_wasserstein_width_fraction": max_w_width,
            "min_speedup": min_speedup,
            "min_fair_detection_n": min_fair_detection_n,
            "min_fair_detection_auc": min_fair_detection_auc,
        },
        "synthetic": {
            "detection": synthetic.get("detection", {}),
            "characterization_sbc_gate": synthetic.get("characterization_sbc_gate"),
            "characterization_coverage_calibration_error":
                synthetic.get("characterization_coverage_calibration_error"),
        },
        "real": {
            "detection": real_summary.get("detection", {}),
            "gate_status": real_summary.get("gate_status", {}),
            "diagnostic_status": real_summary.get("diagnostic_status", {}),
            "mcmc_agreement": mcmc,
            "mcmc_stratified": real_summary.get("mcmc_stratified", {}),
            "mcmc_conditioning": real_summary.get("mcmc_conditioning", {}),
            "importance_correction": real_summary.get("importance_correction", {}),
        },
        "baselines": {
            "bls": bls,
            "speed": speed,
        },
        "status": status,
        "diagnostic_status": diagnostic_status,
    }


def write_environment(path: Path, repo: Path, amp: bool = False) -> None:
    env = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": git_sha(repo),
        "amp": amp,
        "amp_dtype": "bfloat16" if amp else None,
    }
    try:
        import torch
        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        env["torch_error"] = str(exc)
    if not path.exists():
        path.write_text(json.dumps(env, indent=2))
    attempts = path.with_name("environment_attempts.jsonl")
    with attempts.open("a") as fh:
        fh.write(json.dumps(env, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/publishable.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-root", default="results/publishable_runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--noise-lib", default="data/noise_lib.npz")
    ap.add_argument(
        "--publication-eval-noise-lib", default=None,
        help="newly frozen target-level evaluation archive; required for a full run "
             "and forbidden from overlapping development targets")
    ap.add_argument("--min-publication-eval-targets", type=int, default=30)
    ap.add_argument(
        "--identifiability-report", default=None,
        help="fixed held-out blind-BLS audit required before a full publication run",
    )
    ap.add_argument(
        "--min-identifiability-sources", type=int, default=25,
        help="minimum evaluable source targets in the development "
             "identifiability audit; separate from the external-lockbox "
             "target minimum because the audit is development evidence",
    )
    ap.add_argument(
        "--candidate-top-k", type=int, default=3,
        help="alias-separated BLS candidate hypotheses scored per light curve "
             "in the fair detection benchmark (max-pooled); 1 = legacy top-1",
    )
    ap.add_argument("--build-noise-lib", action="store_true")
    ap.add_argument("--noise-workers", type=int, default=1,
                    help="parallel target downloads when building the noise library")
    ap.add_argument("--noise-targets", nargs="*", default=None)
    ap.add_argument("--noise-target-file", default=None,
                    help="one archive-resolvable target per line")
    ap.add_argument("--min-noise-targets", type=int, default=120)
    ap.add_argument("--noise-eval-fraction", type=float, default=0.2,
                    help="fraction of source targets reserved for evaluation")
    ap.add_argument("--noise-validation-fraction", type=float, default=0.1,
                    help="fraction of source targets reserved for checkpoint selection")
    ap.add_argument("--noise-calibration-fraction", type=float, default=0.2,
                    help="fraction of source targets reserved for posterior calibration")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--detector-data-fraction", type=float, default=0.5,
                    help="independent all-BLS detector rows relative to posterior data")
    ap.add_argument("--posterior-validation-fraction", type=float, default=0.1,
                    help="target-disjoint posterior rows for checkpoint selection")
    ap.add_argument("--detector-validation-fraction", type=float, default=0.1,
                    help="held-out fraction of detector rows for checkpoint selection")
    ap.add_argument(
        "--min-full-run-free-gib", type=float, default=16.0,
        help="minimum free storage required before a full run (default: 16 GiB); "
             "guards the three provenance-separated datasets and artifacts",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--dataset-worker-memory-mib", type=float, default=1280.0,
        help="startup-safe RAM budget per spawned Astropy dataset worker",
    )
    ap.add_argument(
        "--dataset-worker-reserve-gib", type=float, default=6.0,
        help="RAM held back for the supervisor, OS, and generation parent",
    )
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--n-sbc", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=5000)
    ap.add_argument("--with-tls-baseline", dest="with_tls_baseline",
                    action="store_true", default=True,
                    help="include equal-sample Transit Least Squares baseline (default on)")
    ap.add_argument("--no-tls-baseline", dest="with_tls_baseline",
                    action="store_false",
                    help="skip TLS only for diagnostics; not suitable for the full publication run")
    ap.add_argument("--tls-baseline-n", type=int, default=None,
                    help="TLS sample count; defaults to the full detection sample")
    ap.add_argument("--tls-workers", type=int, default=None,
                    help="parallel one-thread TLS searches; defaults to 42 or CPU count")
    ap.add_argument("--candidate-source", choices=("bls", "simulator"),
                    default="bls",
                    help="candidate ephemeris for TransitFlow detection evaluation; "
                         "the publication gate requires the fair BLS candidate path")
    ap.add_argument("--eval-seed", type=int, default=123)
    ap.add_argument("--real-seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--n-posterior", type=int, default=2000)
    ap.add_argument("--n-real-planets", type=int, default=30)
    ap.add_argument("--with-mcmc", type=int, default=16)
    ap.add_argument("--mcmc-steps", type=int, default=15000,
                    help="full-run default targets >=50 autocorrelation times; "
                         "fast-check mode uses a short non-publication diagnostic")
    ap.add_argument("--mcmc-max-steps", type=int, default=60000,
                    help="full adaptive convergence cap per real object")
    ap.add_argument("--mcmc-walkers", type=int, default=32)
    ap.add_argument("--mcmc-processes", type=int, default=1,
                    help="parallel worker processes for real-data emcee MCMC")
    ap.add_argument("--is-correct-mcmc", action="store_true",
                    help="use likelihood-corrected amortized samples for real MCMC agreement")
    ap.add_argument("--is-samples", type=int, default=3000)
    ap.add_argument("--min-is-ess-fraction", type=float, default=0.05)
    ap.add_argument("--mcmc-fit-jitter", dest="mcmc_fit_jitter",
                    action="store_true", default=True,
                    help="fit per-object error-inflation (jitter) in the "
                         "real-data MCMC likelihood (default on)")
    ap.add_argument("--no-mcmc-fit-jitter", dest="mcmc_fit_jitter",
                    action="store_false")
    ap.add_argument("--is-jitter-grid", type=int, default=25)
    ap.add_argument("--is-jitter-max", type=float, default=10.0,
                    help="1.0 recovers the exact white-noise IS likelihood")
    ap.add_argument("--speed-n-amortized", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="override training steps; useful for fast metric checks")
    ap.add_argument("--train-eval-every", type=int, default=None,
                    help="override checkpoint-validation interval for a development run")
    ap.add_argument("--detection-loss", choices=("bce", "focal"), default=None,
                    help="override detector loss for a provenance-recorded development ablation")
    ap.add_argument("--detection-focal-gamma", type=float, default=None,
                    help="override focal-loss gamma for a development ablation")
    ap.add_argument("--separate-detector-embedding", action="store_true",
                    help="development ablation: isolate detector and posterior encoders")
    ap.add_argument("--fast-check", action="store_true",
                    help="short metric-oriented run: smaller data/eval/MCMC, same report schema")
    ap.add_argument(
        "--stop-after-synthetic", action="store_true",
        help="write a development synthetic-gate report and skip costly baselines/real MCMC")
    ap.add_argument("--smoke", action="store_true",
                    help="small structural run; not a metrics claim")
    ap.add_argument("--amp", action="store_true",
                    help="enable bfloat16 autocast for amortized inference in "
                         "evaluate/baseline_detection/benchmark_speed/validate_real")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    run_name = args.run_name or time.strftime("publishable_%Y%m%d_%H%M%S")
    out_dir = (repo / args.out_root / run_name).resolve()
    logs = out_dir / "logs"
    results = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    disk_preflight = full_run_disk_preflight(
        out_dir, args.min_full_run_free_gib)
    (out_dir / "disk_preflight.json").write_text(
        json.dumps(disk_preflight, indent=2))
    if not args.smoke and not args.fast_check and not disk_preflight["pass"]:
        raise SystemExit(
            "insufficient free storage for a full run: "
            f"{disk_preflight['available_gib']:.2f} GiB available, "
            f"{disk_preflight['required_free_gib']:.2f} GiB required; "
            "use a larger-volume instance or explicitly set a justified "
            "--min-full-run-free-gib")

    total_memory_bytes, memory_source = memory_limit_bytes()
    data_worker_preflight = dataset_worker_preflight(
        args.workers, memory_bytes=total_memory_bytes,
        reserve_gib=args.dataset_worker_reserve_gib,
        worker_mib=args.dataset_worker_memory_mib,
    )
    data_worker_preflight["memory_source"] = memory_source
    data_workers = int(data_worker_preflight["effective_workers"])
    (out_dir / "dataset_worker_preflight.json").write_text(
        json.dumps(data_worker_preflight, indent=2))

    noise_lib: Path | None
    if str(args.noise_lib).strip().lower() in {"", "none", "null"}:
        noise_lib = None
    else:
        noise_lib = (repo / args.noise_lib).resolve()
    publication_eval_noise_lib = (
        None if args.publication_eval_noise_lib is None
        else (repo / args.publication_eval_noise_lib).resolve()
    )
    if (not args.smoke and not args.fast_check
            and publication_eval_noise_lib is None):
        raise SystemExit(
            "a new --publication-eval-noise-lib is required for a full run; "
            "the inspected development evaluation targets are not a lockbox")
    full_publication_attempt = bool(
        not args.smoke and not args.fast_check and not args.stop_after_synthetic)
    identifiability_gate = {"required": full_publication_attempt, "pass": None}
    if full_publication_attempt:
        if args.identifiability_report is None:
            raise SystemExit(
                "a fixed held-out --identifiability-report is required before "
                "a full run; do not spend the publication lockbox on an "
                "unidentified detector")
        identifiability_gate = identifiability_preflight(
            Path(args.identifiability_report).resolve(),
            args.min_identifiability_sources)
        identifiability_gate["required"] = True
        (out_dir / "identifiability_preflight.json").write_text(
            json.dumps(identifiability_gate, indent=2))
        if not identifiability_gate["pass"]:
            raise SystemExit(
                "identifiability preflight failed before full-run allocation: "
                + "; ".join(identifiability_gate["failures"]))
    else:
        (out_dir / "identifiability_preflight.json").write_text(
            json.dumps(identifiability_gate, indent=2))
    data_dir = Path(args.data_dir).resolve() if args.data_dir else out_dir / "data"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else out_dir / "run"
    if args.smoke:
        n_data = 2048
        steps = 120
        n_sbc = 50
        n_detection = 200
        n_posterior = 128
        n_real_planets = min(args.n_real_planets, 8)
        with_mcmc = 0
        mcmc_steps = args.mcmc_steps
        mcmc_max_steps = mcmc_steps
        speed_n_amortized = args.speed_n_amortized or 32
        speed_n_mcmc = 1
        speed_mcmc_steps = 80
        speed_mcmc_walkers = min(args.mcmc_walkers, 16)
    elif args.fast_check:
        n_data = 20000 if args.n_data == 1_000_000 else args.n_data
        steps = args.steps or 3000
        n_sbc = 200 if args.n_sbc == 1000 else args.n_sbc
        n_detection = 1000 if args.n_detection == 5000 else args.n_detection
        n_posterior = 512 if args.n_posterior == 2000 else args.n_posterior
        n_real_planets = 12 if args.n_real_planets == 30 else args.n_real_planets
        with_mcmc = 4 if args.with_mcmc == 16 else args.with_mcmc
        mcmc_steps = 400 if args.mcmc_steps == 15000 else args.mcmc_steps
        mcmc_max_steps = max(mcmc_steps, min(args.mcmc_max_steps, 2000))
        speed_n_amortized = args.speed_n_amortized or min(n_detection, 64)
        speed_n_mcmc = max(1, min(with_mcmc, 2))
        speed_mcmc_steps = mcmc_steps
        speed_mcmc_walkers = min(args.mcmc_walkers, 16)
    else:
        n_data = args.n_data
        steps = args.steps
        n_sbc = args.n_sbc
        n_detection = args.n_detection
        n_posterior = args.n_posterior
        n_real_planets = args.n_real_planets
        with_mcmc = args.with_mcmc
        mcmc_steps = args.mcmc_steps
        mcmc_max_steps = max(mcmc_steps, args.mcmc_max_steps)
        speed_n_amortized = args.speed_n_amortized or 512
        speed_n_mcmc = 5
        speed_mcmc_steps = args.mcmc_steps
        speed_mcmc_walkers = args.mcmc_walkers
    train_steps = ["--steps", str(steps)] if steps else []
    tls_baseline_n = (
        n_detection if args.tls_baseline_n is None
        else min(int(args.tls_baseline_n), n_detection)
    )
    tls_workers = max(1, min(
        int(args.tls_workers) if args.tls_workers is not None else 42,
        os.cpu_count() or 1,
    ))

    write_environment(out_dir / "environment.json", repo, amp=args.amp)

    run([args.python, "-m", "pytest", "-q", "-m", "not slow", *FAST_PYTEST],
        repo, logs / "pytest_fast.log")

    if args.build_noise_lib and noise_lib is None:
        raise SystemExit("--build-noise-lib requires a real --noise-lib path")
    if noise_lib is None and not args.smoke and not args.fast_check:
        raise SystemExit(
            "a source-labelled --noise-lib is required for a publication run")

    if noise_lib is not None and args.build_noise_lib:
        if args.noise_targets and args.noise_target_file:
            raise SystemExit("use either --noise-targets or --noise-target-file")
        if args.noise_target_file:
            target_file = (repo / args.noise_target_file).resolve()
            targets = [line.strip() for line in target_file.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
        elif args.noise_targets:
            target_file = None
            targets = args.noise_targets
        else:
            target_file = out_dir / "noise_targets.txt"
            target_metadata = out_dir / "noise_targets.json"
            run([args.python, "scripts/select_noise_targets.py",
                 "--out", str(target_file), "--metadata", str(target_metadata),
                 "--n-targets", str(max(120, 3 * args.min_noise_targets)),
                 "--seed", str(args.eval_seed)],
                repo, logs / "noise_target_selection.log")
            targets = [line.strip() for line in target_file.read_text().splitlines()
                       if line.strip()]
        run([args.python, "scripts/build_noise_library.py", "--mission", "TESS",
             "--n-raw", "18000", "--out", str(noise_lib),
             "--workers", str(args.noise_workers),
             "--min-targets", str(args.min_noise_targets),
             *([] if target_file is None else [
                 "--target-provenance", str(target_file.with_suffix(".json"))]),
             "--targets", *targets],
            repo, logs / "noise_lib.log")
    noise_meta = (
        validate_noise_lib(noise_lib)
        if noise_lib is not None else
        {"path": None, "available": False}
    )
    if noise_lib is not None and not args.smoke:
        require_noise_target_count(noise_meta, args.min_noise_targets)
    (out_dir / "noise_lib.json").write_text(json.dumps(noise_meta, indent=2))
    publication_eval_meta = None
    if publication_eval_noise_lib is not None:
        if noise_lib is None:
            raise SystemExit(
                "--publication-eval-noise-lib requires a development --noise-lib")
        publication_eval_meta = validate_noise_lib(publication_eval_noise_lib)
        require_noise_target_count(
            publication_eval_meta, args.min_publication_eval_targets)
        overlap = noise_target_set(noise_lib) & noise_target_set(
            publication_eval_noise_lib)
        if overlap:
            raise SystemExit(
                "publication evaluation archive overlaps development targets: "
                + ", ".join(sorted(overlap)[:10]))
        (out_dir / "publication_eval_noise_lib.json").write_text(
            json.dumps(publication_eval_meta, indent=2))
    train_noise_lib = noise_lib
    validation_noise_lib = noise_lib
    calibration_noise_lib = noise_lib
    eval_noise_lib = noise_lib
    split_meta = None
    if noise_lib is not None and not args.smoke:
        if publication_eval_noise_lib is not None:
            (train_noise_lib, validation_noise_lib, calibration_noise_lib,
             split_meta) = prepare_noise_train_validation_calibration_split(
                    noise_lib, out_dir / "noise_splits", args.eval_seed,
                    args.noise_validation_fraction,
                    args.noise_calibration_fraction)
            eval_noise_lib = publication_eval_noise_lib
            split_meta["publication_evaluation"] = {
                "path": str(publication_eval_noise_lib),
                "sha256": _sha256_file(publication_eval_noise_lib),
                "n_targets": publication_eval_meta["n_unique_targets"],
                "target_overlap_with_development": [],
            }
            split_meta["all_disjoint"] = bool(
                split_meta.get("all_disjoint", False)
                and not split_meta["publication_evaluation"][
                    "target_overlap_with_development"])
        else:
            (train_noise_lib, validation_noise_lib, calibration_noise_lib,
             eval_noise_lib, split_meta) = prepare_noise_four_way_split(
                    noise_lib, out_dir / "noise_splits", args.eval_seed,
                    args.noise_validation_fraction,
                    args.noise_calibration_fraction, args.noise_eval_fraction)
        (out_dir / "noise_split.json").write_text(json.dumps(split_meta, indent=2))

    if not validate_existing_dataset(
            data_dir, args.config, n_data, args.shard_size,
            args.train_seed, train_noise_lib):
        stale_shards = list(data_dir.glob("shard_*.npz"))
        if stale_shards and not validate_existing_dataset(
                data_dir, args.config, n_data, args.shard_size,
                args.train_seed, train_noise_lib, require_complete=False):
            raise SystemExit(
                f"existing dataset failed provenance validation: {data_dir}; "
                "remove or relocate it before regenerating")
        generate_cmd = [args.python, "scripts/generate_data.py", "--config", args.config,
                        "--n", str(n_data), "--workers", str(data_workers),
                        "--shard-size", str(args.shard_size), "--out", str(data_dir),
                        "--seed", str(args.train_seed)]
        if train_noise_lib is not None:
            generate_cmd.extend(["--noise-lib", str(train_noise_lib)])
        run(generate_cmd, repo, logs / "generate_data.log")
        if not validate_existing_dataset(
                data_dir, args.config, n_data, args.shard_size,
                args.train_seed, train_noise_lib):
            raise SystemExit(f"generated dataset failed validation: {data_dir}")

    # Detection and posterior use different valid conditioning domains.  Keep
    # BLS-only detector data physically separate so its provenance cannot be
    # confused with the exact-candidate posterior training set.
    min_detector_rows = min(int(args.shard_size), int(n_data))
    posterior_validation_rows = max(
        int(round(n_data * args.posterior_validation_fraction)),
        min_detector_rows,
    )
    posterior_validation_dir = out_dir / "posterior_validation"
    if not validate_existing_dataset(
            posterior_validation_dir, args.config, posterior_validation_rows,
            args.shard_size, args.train_seed + 200001, validation_noise_lib):
        stale_shards = list(posterior_validation_dir.glob("shard_*.npz"))
        if stale_shards and not validate_existing_dataset(
                posterior_validation_dir, args.config, posterior_validation_rows,
                args.shard_size, args.train_seed + 200001, validation_noise_lib,
                require_complete=False):
            raise SystemExit(
                "existing posterior validation dataset failed provenance "
                f"validation: {posterior_validation_dir}; remove or relocate it "
                "before regenerating")
        validation_cmd = [
            args.python, "scripts/generate_data.py", "--config", args.config,
            "--n", str(posterior_validation_rows), "--workers", str(data_workers),
            "--shard-size", str(args.shard_size), "--out", str(posterior_validation_dir),
            "--seed", str(args.train_seed + 200001),
        ]
        if validation_noise_lib is not None:
            validation_cmd.extend(["--noise-lib", str(validation_noise_lib)])
        run(validation_cmd, repo, logs / "generate_posterior_validation.log")
        if not validate_existing_dataset(
                posterior_validation_dir, args.config, posterior_validation_rows,
                args.shard_size, args.train_seed + 200001, validation_noise_lib):
            raise SystemExit(
                "generated posterior validation dataset failed provenance "
                f"validation: {posterior_validation_dir}")

    detector_n = max(int(round(n_data * args.detector_data_fraction)),
                     min_detector_rows)
    detector_val_n = max(int(round(detector_n * args.detector_validation_fraction)),
                         min_detector_rows)
    detector_data_dir = out_dir / "detector_bls_train"
    detector_val_dir = out_dir / "detector_bls_validation"
    for detector_dir, detector_rows, detector_seed, label, detector_noise_lib in (
        (detector_data_dir, detector_n, args.train_seed + 300001, "train",
         train_noise_lib),
        (detector_val_dir, detector_val_n, args.train_seed + 400001,
         "validation", validation_noise_lib),
    ):
        # Detection checkpoint selection must be target-disjoint from detector
        # fitting whenever a development split exists.  Independent simulator
        # seeds alone would still leak source-specific noise morphology.
        if not validate_existing_dataset(
                detector_dir, args.config, detector_rows, args.shard_size,
                detector_seed, detector_noise_lib,
                candidate_domain="bls_detection"):
            stale_shards = list(detector_dir.glob("shard_*.npz"))
            if stale_shards and not validate_existing_dataset(
                    detector_dir, args.config, detector_rows, args.shard_size,
                    detector_seed, detector_noise_lib,
                    candidate_domain="bls_detection", require_complete=False):
                raise SystemExit(
                    "existing all-BLS detector dataset failed provenance "
                    f"validation: {detector_dir}; remove or relocate it "
                    "before regenerating")
            run([args.python, "scripts/generate_data.py", "--config", args.config,
                 "--candidate-domain", "bls_detection", "--n", str(detector_rows),
                 "--workers", str(data_workers), "--shard-size", str(args.shard_size),
                 "--out", str(detector_dir), "--seed", str(detector_seed),
                 *([] if detector_noise_lib is None else [
                     "--noise-lib", str(detector_noise_lib)])],
                repo, logs / f"generate_detector_{label}.log")
            if not validate_existing_dataset(
                    detector_dir, args.config, detector_rows, args.shard_size,
                    detector_seed, detector_noise_lib,
                    candidate_domain="bls_detection"):
                raise SystemExit(
                    "generated all-BLS detector dataset failed provenance "
                    f"validation: {detector_dir}")
        meta = read_json(detector_dir / "dataset_meta.json")
        sim = meta.get("simulator_config", {})
        if (int(meta.get("n_total", 0)) != detector_rows
                or float(sim.get("candidate_bls_positive_fraction", 0.0)) != 1.0
                or float(sim.get("candidate_bls_negative_fraction", 0.0)) != 1.0):
            raise SystemExit(f"invalid all-BLS detector dataset provenance: {detector_dir}")

    run([args.python, "scripts/preflight.py", "--config", args.config,
         "--expect", "cuda", "--data-dir", str(data_dir)],
        repo, logs / "preflight.log")
    try:
        from scripts._config import build_configs
    except ImportError:  # direct ``python scripts/run_publishable_vast.py``
        from _config import build_configs
    training_overrides = {"train": {}, "model": {}}
    if args.detection_loss is not None:
        training_overrides["train"]["detection_loss"] = args.detection_loss
    if args.detection_focal_gamma is not None:
        training_overrides["train"]["detection_focal_gamma"] = \
            args.detection_focal_gamma
    if args.separate_detector_embedding:
        training_overrides["model"]["separate_detection_embedding"] = True
    effective_configs = build_configs(args.config, training_overrides)
    effective_train_cfg = effective_configs["train"]
    expected_steps = int(steps or effective_train_cfg.n_steps)
    training_provenance = {
        "schema_version": 4,
        "config_sha256": _sha256_file((repo / args.config).resolve()),
        "dataset_metadata_sha256": _sha256_file(data_dir / "dataset_meta.json"),
        "posterior_validation_metadata_sha256": _sha256_file(
            posterior_validation_dir / "dataset_meta.json"),
        "detector_dataset_metadata_sha256": _sha256_file(detector_data_dir / "dataset_meta.json"),
        "detector_validation_metadata_sha256": _sha256_file(detector_val_dir / "dataset_meta.json"),
        "detector_train_noise_lib": None if train_noise_lib is None else {
            "path": str(train_noise_lib),
            "sha256": _sha256_file(train_noise_lib),
            "n_targets": len(noise_target_set(train_noise_lib)),
        },
        "detector_validation_noise_lib": None if validation_noise_lib is None else {
            "path": str(validation_noise_lib),
            "sha256": _sha256_file(validation_noise_lib),
            "n_targets": len(noise_target_set(validation_noise_lib)),
        },
        "validation_noise_lib": None if validation_noise_lib is None else {
            "path": str(validation_noise_lib),
            "sha256": _sha256_file(validation_noise_lib),
            "n_targets": len(noise_target_set(validation_noise_lib)),
        },
        "posterior_validation_rows": int(posterior_validation_rows),
        "train_eval_every": args.train_eval_every,
        "detection_loss": effective_train_cfg.detection_loss,
        "detection_focal_gamma": effective_train_cfg.detection_focal_gamma,
        "separate_detection_embedding": bool(
            effective_configs["model"].separate_detection_embedding),
        "train_seed": int(args.train_seed),
        "steps": int(expected_steps),
    }
    training_provenance_path = run_dir / "training_provenance.json"
    if training_provenance_path.exists():
        if read_json(training_provenance_path) != training_provenance:
            raise SystemExit(
                "existing run has incompatible training/validation provenance; "
                "use a new --run-dir")
    elif (run_dir / "checkpoints").exists() and any(
            (run_dir / "checkpoints").glob("*.pt")):
        raise SystemExit(
            "existing checkpoints lack training/validation provenance; "
            "use a new --run-dir")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        training_provenance_path.write_text(
            json.dumps(training_provenance, indent=2))
    prior_status_path = run_dir / "status.json"
    prior_status = read_json(prior_status_path) if prior_status_path.exists() else {}
    training_already_complete = bool(
        (run_dir / "checkpoints" / "latest.pt").exists()
        and prior_status.get("status") == "done"
        and int(prior_status.get("step", -1)) == expected_steps
        and int(prior_status.get("total_steps", -2)) == expected_steps)
    if not training_already_complete:
        run([args.python, "scripts/train.py", "--config", args.config,
             "--run-dir", str(run_dir), "--data-dir", str(data_dir),
             "--validation-data-dir", str(posterior_validation_dir),
             "--detection-data-dir", str(detector_data_dir),
             "--detection-validation-data-dir", str(detector_val_dir),
             "--expect-device", "cuda", "--no-preflight",
             "--seed", str(args.train_seed),
             *([] if args.train_eval_every is None else [
                 "--eval-every", str(args.train_eval_every)]),
             *([] if args.detection_loss is None else [
                 "--detection-loss", args.detection_loss]),
             *([] if args.detection_focal_gamma is None else [
                 "--detection-focal-gamma", str(args.detection_focal_gamma)]),
             *([] if not args.separate_detector_embedding else [
                 "--separate-detector-embedding"]),
             *([] if validation_noise_lib is None else [
                 "--noise-lib", str(validation_noise_lib)]),
             *train_steps],
            repo, logs / "train.log")

    train_status = read_json(run_dir / "status.json")
    if (train_status.get("status") != "done"
            or int(train_status.get("step", -1)) != expected_steps
            or int(train_status.get("total_steps", -2)) != expected_steps):
        raise SystemExit(
            f"training did not reach a healthy terminal state: {train_status}")

    latest_ckpt = run_dir / "checkpoints" / "latest.pt"
    best_posterior_ckpt = run_dir / "checkpoints" / "best.pt"
    if best_posterior_ckpt.exists():
        ckpt = best_posterior_ckpt
        posterior_checkpoint_selection = "minimum_validation_posterior_loss"
    elif args.smoke:
        ckpt = latest_ckpt
        posterior_checkpoint_selection = "latest_smoke_without_validation"
    else:
        raise SystemExit(
            "missing best.pt: publication evaluation requires a checkpoint "
            "selected by held-out posterior validation loss")
    detector_ckpt = run_dir / "checkpoints" / "best_detection.pt"
    if not detector_ckpt.exists():
        detector_ckpt = latest_ckpt
    calibration_path = out_dir / "posterior_calibration.json"
    calibration_diagnostic_data = out_dir / "posterior_calibration_inputs.npz"
    calibrator_selection_disjoint = False
    if calibration_noise_lib is not None:
        calibration_n = 100 if args.smoke else (600 if args.fast_check else 1000)
        calibration_selection_n = 100 if args.smoke else (
            300 if args.fast_check else 500)
        calibration_post = min(n_posterior, 512) if args.smoke else n_posterior
        calibration_cmd = [
            args.python, "scripts/calibrate_posterior.py", "--ckpt", str(ckpt),
            "--noise-lib", str(calibration_noise_lib), "--out", str(calibration_path),
            "--n-calibration", str(calibration_n), "--n-posterior",
            str(calibration_post), "--n-selection", str(calibration_selection_n),
            "--diagnostic-data", str(calibration_diagnostic_data),
            "--seed", str(args.eval_seed + 7000),
        ]
        if args.amp:
            calibration_cmd.append("--amp")
        run(calibration_cmd, repo, logs / "calibrate_posterior.log")
        calibration_artifact = read_json(calibration_path)
        target_split = calibration_artifact.get("metadata", {}).get(
            "selection_protocol", {}).get("target_split", {})
        calibrator_selection_disjoint = bool(
            target_split.get("fit_targets")
            and target_split.get("selection_targets")
            and target_split.get("target_overlap") == [])
    eval_dir = results / "synthetic"
    evaluate_cmd = [args.python, "scripts/evaluate.py", "--ckpt", str(ckpt),
                    "--detector-ckpt", str(detector_ckpt),
                    "--n-sbc", str(n_sbc), "--n-detection", str(n_detection),
                    "--n-posterior", str(n_posterior), "--out", str(eval_dir),
                    "--plots", "--seed", str(args.eval_seed)]
    if calibration_path.exists():
        evaluate_cmd.extend(["--calibration", str(calibration_path)])
    if eval_noise_lib is not None:
        evaluate_cmd.extend(["--noise-lib", str(eval_noise_lib)])
    if args.amp:
        evaluate_cmd.append("--amp")
    run(evaluate_cmd, repo, logs / "evaluate.log")
    metrics_path = eval_dir / "metrics.json"
    synthetic_metrics = read_json(metrics_path)
    # Period-alias diagnostic: quantifies posterior mass at P/2, 2P and the
    # per-stratum period SBC, in the evaluation noise domain. Run on every full
    # run (not fast-check) so alias contamination is archived alongside the
    # synthetic metrics rather than left as a manual, off-pipeline step.
    if not args.fast_check and not args.smoke:
        period_diag_path = eval_dir / "period_alias_diagnostic.json"
        period_diag_cmd = [
            args.python, "scripts/diagnose_period.py", "--ckpt", str(ckpt),
            "--n", str(n_sbc), "--n-post", str(min(n_posterior, 500)),
            "--seed", str(args.eval_seed),
            "--out", str(period_diag_path)]
        if eval_noise_lib is not None:
            period_diag_cmd.extend(["--noise-lib", str(eval_noise_lib)])
        run(period_diag_cmd, repo, logs / "diagnose_period.log")
        if period_diag_path.exists():
            synthetic_metrics["period_alias_diagnostic"] = str(period_diag_path)
            metrics_path.write_text(json.dumps(synthetic_metrics, indent=2))
    baseline_path = results / "bls_vs_transitflow.json"
    # The simulator-candidate detector score is an oracle diagnostic, not a
    # blind-detection gate. When BLS candidates are requested, run the exact
    # comparator before emitting any synthetic gate report and promote only
    # that matched-path TransitFlow score to the detection gate.
    if args.candidate_source == "bls":
        detection_baseline_cmd = [
            args.python, "scripts/baseline_detection.py", "--ckpt",
            str(detector_ckpt), "--n", str(n_detection), "--out",
            str(baseline_path), "--seed", str(args.eval_seed),
            "--candidate-source", "bls",
            "--candidate-top-k", str(args.candidate_top_k),
        ]
        if eval_noise_lib is not None:
            detection_baseline_cmd.extend(["--noise-lib", str(eval_noise_lib)])
        if args.with_tls_baseline:
            # This synthetic-stage run writes the canonical fair-benchmark
            # artifact; the later baselines stage skips when the file already
            # exists, so the equal-sample TLS comparison must be produced here
            # or it is silently dropped from the gate report.
            detection_baseline_cmd.extend(
                ["--with-tls", "--tls-n", str(tls_baseline_n),
                 "--tls-workers", str(tls_workers), "--tls-threads", "1"])
        if args.amp:
            detection_baseline_cmd.append("--amp")
        run(detection_baseline_cmd, repo, logs / "baseline_detection.log")
        blind_detection = read_json(baseline_path).get("transitflow", {})
        if "roc_auc" not in blind_detection:
            raise SystemExit("fair BLS baseline did not produce TransitFlow AUC")
        synthetic_metrics["oracle_detection"] = synthetic_metrics.get("detection")
        synthetic_metrics["detection"] = blind_detection
        synthetic_metrics["detection_candidate_source"] = "bls"
        synthetic_metrics["detection_baseline"] = str(baseline_path)
        synthetic_metrics["detection_auc_min"] = FAIR_DETECTION_AUC_MIN
        gate_status = synthetic_metrics.setdefault("gate_status", {})
        gate_status["detection_auc_ge_min"] = bool(
            float(blind_detection["roc_auc"]) >= FAIR_DETECTION_AUC_MIN)
        # Drop the oracle-threshold key so a stale name cannot shadow the
        # fair blind-candidate decision.
        gate_status.pop("detection_auc_ge_0.99", None)
        metrics_path.write_text(json.dumps(synthetic_metrics, indent=2))
    synthetic_report = build_synthetic_gate_report(
        synthetic_metrics, split_meta)
    synthetic_report["status"]["calibrator_selection_target_disjoint"] = \
        calibrator_selection_disjoint
    synthetic_report["all_declared_synthetic_gates_pass"] = all(
        synthetic_report["status"].values())
    synthetic_report["run"] = {
        "run_name": run_name,
        "git_sha": git_sha(repo),
        "checkpoint": str(ckpt),
        "detector_checkpoint": str(detector_ckpt),
        "posterior_checkpoint_selection": posterior_checkpoint_selection,
        "training_provenance": training_provenance,
        "train_noise_lib": None if train_noise_lib is None else str(
            train_noise_lib),
        "validation_noise_lib": None if validation_noise_lib is None else str(
            validation_noise_lib),
        "calibration_noise_lib": None if calibration_noise_lib is None else str(
            calibration_noise_lib),
        "eval_noise_lib": None if eval_noise_lib is None else str(eval_noise_lib),
        "external_lockbox_used": bool(publication_eval_noise_lib is not None),
        "calibration_diagnostic_data": str(calibration_diagnostic_data)
        if calibration_diagnostic_data.exists() else None,
        "n_data": int(n_data),
        "steps": int(expected_steps),
        "n_sbc": int(n_sbc),
        "n_detection": int(n_detection),
        "n_posterior": int(n_posterior),
        "train_seed": int(args.train_seed),
        "eval_seed": int(args.eval_seed),
    }
    synthetic_report_path = out_dir / "synthetic_gate_report.json"
    synthetic_report_path.write_text(json.dumps(synthetic_report, indent=2))
    if args.stop_after_synthetic:
        print(f"synthetic development gate complete: {synthetic_report_path}")
        return
    if external_lockbox_synthetic_failure_blocks_downstream(
        synthetic_report,
        smoke=args.smoke,
        fast_check=args.fast_check,
        has_external_lockbox=publication_eval_noise_lib is not None,
    ):
        raise SystemExit(
            "external-lockbox synthetic gate failed; downstream baseline, speed, "
            "and real-data stages were not run")
    if not baseline_path.exists():
        baseline_cmd = [args.python, "scripts/baseline_detection.py", "--ckpt",
                        str(detector_ckpt), "--n", str(n_detection), "--out",
                        str(baseline_path), "--seed", str(args.eval_seed),
                        "--candidate-source", args.candidate_source,
                        "--candidate-top-k", str(args.candidate_top_k)]
        if eval_noise_lib is not None:
            baseline_cmd.extend(["--noise-lib", str(eval_noise_lib)])
        if args.with_tls_baseline:
            baseline_cmd.extend(["--with-tls", "--tls-n", str(tls_baseline_n),
                                 "--tls-workers", str(tls_workers),
                                 "--tls-threads", "1"])
        if args.amp:
            baseline_cmd.append("--amp")
        run(baseline_cmd, repo, logs / "baseline_detection.log")
    speed_cmd = [args.python, "scripts/benchmark_speed.py", "--ckpt", str(ckpt),
                 "--n-amortized", str(speed_n_amortized), "--n-post", str(n_posterior),
                 "--n-mcmc", str(speed_n_mcmc), "--mcmc-steps", str(speed_mcmc_steps),
                 "--mcmc-max-steps", str(mcmc_max_steps),
                 "--mcmc-walkers", str(speed_mcmc_walkers),
                 "--seed", str(args.eval_seed + 8000),
                 "--out", str(results / "speed.json")]
    if calibration_path.exists():
        speed_cmd.extend(["--calibration", str(calibration_path)])
    if eval_noise_lib is not None:
        speed_cmd.extend(["--noise-lib", str(eval_noise_lib)])
    if args.amp:
        speed_cmd.append("--amp")
    run(speed_cmd, repo, logs / "speed.log")
    # BF16-vs-FP32 equivalence: when the pipeline runs inference under autocast
    # bfloat16 (--amp), verify on this exact checkpoint that BF16 posteriors and
    # detection probabilities match FP32 within predeclared tolerances. A
    # failure exits non-zero, so a mixed-precision run cannot silently report
    # BF16 numbers that FP32 would not support. Skipped in fast-check/smoke and
    # when --amp is off (FP32 runs need no equivalence proof).
    if args.amp and not args.fast_check and not args.smoke:
        equivalence_path = results / "bf16_fp32_equivalence.json"
        equivalence_cmd = [
            args.python, "scripts/bf16_fp32_equivalence.py", "--ckpt", str(ckpt),
            "--n", str(min(n_sbc, 512)), "--n-post", str(min(n_posterior, 1000)),
            "--seed", str(args.eval_seed), "--out", str(equivalence_path)]
        if eval_noise_lib is not None:
            equivalence_cmd.extend(["--noise-lib", str(eval_noise_lib)])
        run(equivalence_cmd, repo, logs / "bf16_fp32_equivalence.log")
    real_dir = results / "real"
    cmd = [args.python, "scripts/validate_real.py", "--ckpt", str(ckpt),
           "--detector-ckpt", str(detector_ckpt),
           "--n-planets", str(n_real_planets),
           "--n-post", str(n_posterior), "--with-mcmc", str(with_mcmc),
           "--mcmc-steps", str(mcmc_steps), "--mcmc-walkers", str(args.mcmc_walkers),
           "--mcmc-max-steps", str(mcmc_max_steps),
           "--mcmc-processes", str(args.mcmc_processes),
           "--seed", str(args.real_seed),
           "--out", str(real_dir)]
    if calibration_path.exists():
        cmd.extend(["--calibration", str(calibration_path)])
    if args.is_correct_mcmc:
        cmd.extend(["--is-correct-mcmc", "--is-samples", str(args.is_samples),
                    "--min-is-ess-fraction", str(args.min_is_ess_fraction),
                    "--is-jitter-grid", str(args.is_jitter_grid),
                    "--is-jitter-max", str(args.is_jitter_max)])
    if args.mcmc_fit_jitter:
        cmd.append("--mcmc-fit-jitter")
    if args.amp:
        cmd.append("--amp")
    run(cmd, repo, logs / "validate_real.log")

    # Negative-class real-data validation: score real TESS false positives
    # (TFOPWG FP/FA) through the identical pipeline to measure specificity /
    # precision / false-positive rate -- the positive-only real stage measures
    # sensitivity alone. Feeds the positive records checkpoint so precision/F1
    # are computed against the matched positive class. Full runs only.
    neg_validation = {}
    if not args.fast_check and not args.smoke:
        neg_dir = results / "real_negatives"
        pos_records = real_dir / "records_checkpoint.json"
        neg_cmd = [args.python, "scripts/validate_negatives.py",
                   "--ckpt", str(ckpt), "--detector-ckpt", str(detector_ckpt),
                   "--n-negatives", str(n_real_planets),
                   "--n-post", str(min(n_posterior, 500)),
                   "--threshold", "0.9",
                   "--seed", str(args.real_seed + 5000),
                   "--out", str(neg_dir)]
        if pos_records.exists():
            neg_cmd.extend(["--positives-records", str(pos_records)])
        if calibration_path.exists():
            neg_cmd.extend(["--calibration", str(calibration_path)])
        if args.amp:
            neg_cmd.append("--amp")
        run(neg_cmd, repo, logs / "validate_negatives.log")
        neg_out = neg_dir / "negative_validation.json"
        if neg_out.exists():
            neg_validation = read_json(neg_out)

    report = build_gate_report(
        read_json(eval_dir / "metrics.json"),
        read_json(real_dir / "real_validation.json"),
        read_json(results / "bls_vs_transitflow.json"),
        read_json(results / "speed.json"),
    )
    # Negative-class specificity gate: real false positives must not be flagged
    # as planets above the operating threshold at more than the predeclared
    # rate. Only enforced on full runs where the negative stage actually ran.
    if neg_validation:
        neg_gate = neg_validation.get("gate_status", {})
        report["status"]["real_negative_specificity_ge_min"] = bool(
            neg_gate.get("specificity_ge_min", False))
        report.setdefault("real_negatives", {})
        report["real_negatives"] = {
            "metrics": neg_validation.get("metrics", {}),
            "gate_status": neg_gate,
            "pass": neg_validation.get("pass", False),
        }
    if split_meta is not None:
        report["status"]["noise_target_split_disjoint"] = bool(
            split_meta.get("all_disjoint", False))
        report["status"]["posterior_calibration_disjoint"] = bool(
            calibration_path.exists() and split_meta.get("all_disjoint", False))
        report["status"]["checkpoint_validation_disjoint"] = bool(
            validation_noise_lib is not None
            and split_meta.get("all_disjoint", False))
        report["status"]["calibrator_selection_target_disjoint"] = \
            calibrator_selection_disjoint
        report["status"]["final_pass"] = all(
            value for key, value in report["status"].items()
            if key != "final_pass")
    report["run"] = {
        "run_name": run_name,
        "config": args.config,
        "checkpoint": str(ckpt),
        "posterior_checkpoint_selection": posterior_checkpoint_selection,
        "latest_checkpoint": str(latest_ckpt),
        "detector_checkpoint": str(detector_ckpt),
        "data_dir": str(data_dir),
        "noise_lib": None if noise_lib is None else str(noise_lib),
        "publication_eval_noise_lib": None
        if publication_eval_noise_lib is None else str(publication_eval_noise_lib),
        "train_noise_lib": None if train_noise_lib is None else str(train_noise_lib),
        "validation_noise_lib": None if validation_noise_lib is None else str(
            validation_noise_lib),
        "calibration_noise_lib": None if calibration_noise_lib is None else str(
            calibration_noise_lib),
        "posterior_calibration": str(calibration_path)
        if calibration_path.exists() else None,
        "eval_noise_lib": None if eval_noise_lib is None else str(eval_noise_lib),
        "noise_calibration_fraction": float(args.noise_calibration_fraction),
        "noise_validation_fraction": float(args.noise_validation_fraction),
        "noise_eval_fraction": float(args.noise_eval_fraction),
        "noise_workers": int(args.noise_workers),
        "dataset_worker_preflight": data_worker_preflight,
        "smoke": bool(args.smoke),
        "fast_check": bool(args.fast_check),
        "n_data": int(n_data),
        "steps": None if steps is None else int(steps),
        "n_sbc": int(n_sbc),
        "n_detection": int(n_detection),
        "with_tls_baseline": bool(args.with_tls_baseline),
        "tls_baseline_n": int(tls_baseline_n),
        "tls_workers": int(tls_workers),
        "candidate_source": args.candidate_source,
        "candidate_top_k": int(args.candidate_top_k),
        "eval_seed": int(args.eval_seed),
        "real_seed": int(args.real_seed),
        "train_seed": int(args.train_seed),
        "n_posterior": int(n_posterior),
        "n_real_planets": int(n_real_planets),
        "with_mcmc": int(with_mcmc),
        "mcmc_steps": int(mcmc_steps),
        "mcmc_max_steps": int(mcmc_max_steps),
        "mcmc_walkers": int(args.mcmc_walkers),
        "mcmc_processes": int(args.mcmc_processes),
        "is_correct_mcmc": bool(args.is_correct_mcmc),
        "is_samples": int(args.is_samples),
        "min_is_ess_fraction": float(args.min_is_ess_fraction),
        "mcmc_fit_jitter": bool(args.mcmc_fit_jitter),
        "is_jitter_grid": int(args.is_jitter_grid),
        "is_jitter_max": float(args.is_jitter_max),
        "speed_n_amortized": int(speed_n_amortized),
        "speed_n_mcmc": int(speed_n_mcmc),
        "speed_mcmc_steps": int(speed_mcmc_steps),
        "speed_mcmc_walkers": int(speed_mcmc_walkers),
    }
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["status"], indent=2))
    if not args.smoke and not args.fast_check and not report["status"]["final_pass"]:
        raise SystemExit("publishable gate suite completed but final_pass=false")


if __name__ == "__main__":
    main()
