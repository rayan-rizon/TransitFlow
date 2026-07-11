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
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


FAST_PYTEST = [
    "tests/test_evaluate_gates.py",
    "tests/test_inference.py",
    "tests/test_simulator.py",
    "tests/test_data.py",
    "tests/test_baselines.py",
]


DEFAULT_TARGETS = [
    "HD 10700", "HD 197076", "HD 1461", "HD 36435", "HD 101501",
    "HD 26965", "HD 32147", "HD 40307", "HD 20794", "HD 85512",
    "HD 7924", "HD 136352", "HD 190406", "HD 131977", "HD 10647",
]


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
    }


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_dataset(data_dir: Path, config_path: str, n_total: int,
                              shard_size: int, seed: int,
                              noise_lib: Path | None) -> bool:
    """Accept a reusable disk dataset only when its provenance is exact."""
    try:
        from scripts._config import build_configs
    except ImportError:  # direct ``python scripts/run_publishable_vast.py``
        from _config import build_configs

    meta_path = data_dir / "dataset_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = read_json(meta_path)
        expected_shards = (n_total + shard_size - 1) // shard_size
        config_json = json.dumps(
            asdict(build_configs(config_path)["simulator"]), sort_keys=True)
        expected_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        expected_names = {
            f"shard_{idx:05d}.npz" for idx in range(expected_shards)}
        actual_names = {path.name for path in data_dir.glob("shard_*.npz")}
        return bool(
            int(meta.get("n_total", -1)) == n_total
            and int(meta.get("n_shards", -1)) == expected_shards
            and int(meta.get("shard_size", -1)) == shard_size
            and int(meta.get("seed", -1)) == seed
            and meta.get("config_hash") == expected_hash
            and meta.get("noise_lib_sha256") == _sha256_file(noise_lib)
            and actual_names == expected_names
            and all((data_dir / name).stat().st_size > 0 for name in expected_names)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
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
    if len(target_ids) != len(segments):
        raise SystemExit("noise library target_ids length does not match segments")
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


def _gate_value(metrics: dict, key: str) -> bool:
    gate = metrics.get("gate_status", {})
    return bool(gate.get(key, False))


def build_gate_report(synthetic: dict, real: dict, bls: dict, speed: dict,
                      thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {}
    min_real_detected = int(thresholds.get("min_real_detected", 27))
    min_mcmc_n = int(thresholds.get("min_mcmc_n", 16))
    max_w_prior = float(thresholds.get("max_wasserstein_prior_fraction", 0.1))
    max_w_width = float(thresholds.get("max_wasserstein_width_fraction", 0.5))
    min_speedup = float(thresholds.get("min_speedup", 1000.0))
    min_fair_detection_n = int(thresholds.get("min_fair_detection_n", 5000))

    real_summary = real.get("summary", real)
    mcmc = real_summary.get("mcmc_agreement", {})
    char = ("RpRs", "aRs", "b")
    real_mcmc_n = min([mcmc.get(k, {}).get("n", 0) for k in char] or [0])
    prior_ok = all(
        mcmc.get(k, {}).get("median_wasserstein_prior_fraction", float("inf")) <= max_w_prior
        for k in char
    )
    width_ok = all(
        mcmc.get(k, {}).get("median_wasserstein_width_fraction", float("inf")) <= max_w_width
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
    )
    raw_speed_pass = float(speed.get("speedup_x", 0.0)) >= min_speedup
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
        "real_mcmc_prior_fraction_le_0.1": prior_ok,
        "real_mcmc_width_fraction_le_0.5": width_ok,
        "speedup_ge_1000x_at_converged_mcmc_reference":
            raw_speed_pass and mcmc_converged,
        "bls_baseline_regenerated": bool(
            bls.get("transitflow") and bls.get("bls")
            and int(bls.get("bls", {}).get("n_failed", 0)) == 0),
        "detection_candidate_ephemeris_from_bls":
            bls.get("candidate_source") == "bls",
        "fair_candidate_evaluation_n_ge_5000":
            int(bls.get("n", 0)) >= min_fair_detection_n,
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
    ):
        if convergence_gate in real_summary.get("gate_status", {}):
            status[convergence_gate] = bool(
                real_summary["gate_status"][convergence_gate])
    status["final_pass"] = all(status.values())
    diagnostic_status = {
        "oracle_candidate_detection_auc_ge_0.99":
            _gate_value(synthetic, "detection_auc_ge_0.99"),
        "raw_speedup_ge_1000x": raw_speed_pass,
    }
    return {
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
    ap.add_argument("--build-noise-lib", action="store_true")
    ap.add_argument("--noise-workers", type=int, default=1,
                    help="parallel target downloads when building the noise library")
    ap.add_argument("--noise-targets", nargs="*", default=None)
    ap.add_argument("--noise-eval-fraction", type=float, default=0.2,
                    help="fraction of source targets reserved for evaluation")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=16)
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
    ap.add_argument("--fast-check", action="store_true",
                    help="short metric-oriented run: smaller data/eval/MCMC, same report schema")
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

    noise_lib: Path | None
    if str(args.noise_lib).strip().lower() in {"", "none", "null"}:
        noise_lib = None
    else:
        noise_lib = (repo / args.noise_lib).resolve()
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
        targets = args.noise_targets or DEFAULT_TARGETS
        run([args.python, "scripts/build_noise_library.py", "--mission", "TESS",
             "--n-raw", "18000", "--out", str(noise_lib),
             "--workers", str(args.noise_workers), "--targets", *targets],
            repo, logs / "noise_lib.log")
    noise_meta = (
        validate_noise_lib(noise_lib)
        if noise_lib is not None else
        {"path": None, "available": False}
    )
    (out_dir / "noise_lib.json").write_text(json.dumps(noise_meta, indent=2))
    train_noise_lib = noise_lib
    eval_noise_lib = noise_lib
    split_meta = None
    if noise_lib is not None and not args.smoke and not args.fast_check:
        train_noise_lib, eval_noise_lib, split_meta = prepare_noise_splits(
            noise_lib, out_dir / "noise_splits", args.eval_seed,
            args.noise_eval_fraction)
        (out_dir / "noise_split.json").write_text(json.dumps(split_meta, indent=2))

    if not validate_existing_dataset(
            data_dir, args.config, n_data, args.shard_size,
            args.train_seed, train_noise_lib):
        stale_shards = list(data_dir.glob("shard_*.npz"))
        if stale_shards:
            raise SystemExit(
                f"existing dataset failed provenance validation: {data_dir}; "
                "remove or relocate it before regenerating")
        generate_cmd = [args.python, "scripts/generate_data.py", "--config", args.config,
                        "--n", str(n_data), "--workers", str(args.workers),
                        "--shard-size", str(args.shard_size), "--out", str(data_dir),
                        "--seed", str(args.train_seed)]
        if train_noise_lib is not None:
            generate_cmd.extend(["--noise-lib", str(train_noise_lib)])
        run(generate_cmd, repo, logs / "generate_data.log")
        if not validate_existing_dataset(
                data_dir, args.config, n_data, args.shard_size,
                args.train_seed, train_noise_lib):
            raise SystemExit(f"generated dataset failed validation: {data_dir}")

    run([args.python, "scripts/preflight.py", "--config", args.config,
         "--expect", "cuda", "--data-dir", str(data_dir)],
        repo, logs / "preflight.log")
    try:
        from scripts._config import build_configs
    except ImportError:  # direct ``python scripts/run_publishable_vast.py``
        from _config import build_configs
    expected_steps = int(steps or build_configs(args.config)["train"].n_steps)
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
             "--expect-device", "cuda", "--no-preflight",
             "--seed", str(args.train_seed), *train_steps],
            repo, logs / "train.log")

    train_status = read_json(run_dir / "status.json")
    if (train_status.get("status") != "done"
            or int(train_status.get("step", -1)) != expected_steps
            or int(train_status.get("total_steps", -2)) != expected_steps):
        raise SystemExit(
            f"training did not reach a healthy terminal state: {train_status}")

    ckpt = run_dir / "checkpoints" / "latest.pt"
    eval_dir = results / "synthetic"
    evaluate_cmd = [args.python, "scripts/evaluate.py", "--ckpt", str(ckpt),
                    "--n-sbc", str(n_sbc), "--n-detection", str(n_detection),
                    "--n-posterior", str(n_posterior), "--out", str(eval_dir),
                    "--plots", "--seed", str(args.eval_seed)]
    if eval_noise_lib is not None:
        evaluate_cmd.extend(["--noise-lib", str(eval_noise_lib)])
    if args.amp:
        evaluate_cmd.append("--amp")
    run(evaluate_cmd, repo, logs / "evaluate.log")
    baseline_cmd = [args.python, "scripts/baseline_detection.py", "--ckpt", str(ckpt),
                    "--n", str(n_detection), "--out", str(results / "bls_vs_transitflow.json"),
                    "--seed", str(args.eval_seed),
                    "--candidate-source", args.candidate_source]
    if eval_noise_lib is not None:
        baseline_cmd.extend(["--noise-lib", str(eval_noise_lib)])
    if args.with_tls_baseline:
        baseline_cmd.extend(["--with-tls", "--tls-n", str(tls_baseline_n),
                             "--tls-workers", str(tls_workers), "--tls-threads", "1"])
    if args.amp:
        baseline_cmd.append("--amp")
    run(baseline_cmd, repo, logs / "baseline_detection.log")
    speed_cmd = [args.python, "scripts/benchmark_speed.py", "--ckpt", str(ckpt),
                 "--n-amortized", str(speed_n_amortized), "--n-post", str(n_posterior),
                 "--n-mcmc", str(speed_n_mcmc), "--mcmc-steps", str(speed_mcmc_steps),
                 "--mcmc-walkers", str(speed_mcmc_walkers),
                 "--out", str(results / "speed.json")]
    if eval_noise_lib is not None:
        speed_cmd.extend(["--noise-lib", str(eval_noise_lib)])
    if args.amp:
        speed_cmd.append("--amp")
    run(speed_cmd, repo, logs / "speed.log")
    real_dir = results / "real"
    cmd = [args.python, "scripts/validate_real.py", "--ckpt", str(ckpt),
           "--detector-ckpt", str(ckpt), "--n-planets", str(n_real_planets),
           "--n-post", str(n_posterior), "--with-mcmc", str(with_mcmc),
           "--mcmc-steps", str(mcmc_steps), "--mcmc-walkers", str(args.mcmc_walkers),
           "--mcmc-processes", str(args.mcmc_processes),
           "--seed", str(args.real_seed),
           "--out", str(real_dir)]
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

    report = build_gate_report(
        read_json(eval_dir / "metrics.json"),
        read_json(real_dir / "real_validation.json"),
        read_json(results / "bls_vs_transitflow.json"),
        read_json(results / "speed.json"),
    )
    if split_meta is not None:
        report["status"]["noise_target_split_disjoint"] = not bool(
            split_meta["target_overlap"])
        report["status"]["final_pass"] = all(
            value for key, value in report["status"].items()
            if key != "final_pass")
    report["run"] = {
        "run_name": run_name,
        "config": args.config,
        "checkpoint": str(ckpt),
        "data_dir": str(data_dir),
        "noise_lib": None if noise_lib is None else str(noise_lib),
        "train_noise_lib": None if train_noise_lib is None else str(train_noise_lib),
        "eval_noise_lib": None if eval_noise_lib is None else str(eval_noise_lib),
        "noise_eval_fraction": float(args.noise_eval_fraction),
        "noise_workers": int(args.noise_workers),
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
        "eval_seed": int(args.eval_seed),
        "real_seed": int(args.real_seed),
        "train_seed": int(args.train_seed),
        "n_posterior": int(n_posterior),
        "n_real_planets": int(n_real_planets),
        "with_mcmc": int(with_mcmc),
        "mcmc_steps": int(mcmc_steps),
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
