#!/usr/bin/env python3
"""Run the predeclared publishable TransitFlow gate suite on a Vast box.

The script is intentionally conservative: it records the environment, runs a
bounded smoke test, validates the noise library, generates or reuses disk data,
trains `latest.pt`, evaluates synthetic gates, runs baselines, runs quality-gated
real validation with fixed-ephemeris MCMC, and writes one `gate_report.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
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
    with log_path.open("w") as log:
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
    return {
        "path": str(path),
        "n_segments": int(seg.shape[0]),
        "segment_length": int(seg.shape[1]),
        "median": float(np.nanmedian(seg)),
        "std": float(np.nanstd(seg)),
    }


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
    status = {
        "synthetic_detection_auc_ge_0.99":
            _gate_value(synthetic, "detection_auc_ge_0.99"),
        "synthetic_characterization_sbc_familywise_alpha_0.05":
            _gate_value(synthetic, "characterization_sbc_familywise_alpha_0.05"),
        "synthetic_characterization_coverage_error_le_0.03":
            _gate_value(synthetic, "characterization_coverage_error_le_0.03"),
        "real_quality_gated_detection_ge_27_of_30":
            int(real_summary.get("detection", {}).get("n_detected", 0)) >= min_real_detected,
        "real_mcmc_n_ge_16": real_mcmc_n >= min_mcmc_n,
        "real_mcmc_prior_fraction_le_0.1": prior_ok,
        "real_mcmc_width_fraction_le_0.5": width_ok,
        "speedup_ge_1000x": float(speed.get("speedup_x", 0.0)) >= min_speedup,
        "bls_baseline_regenerated": bool(bls.get("transitflow") and bls.get("bls")),
    }
    status["final_pass"] = all(status.values())
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
            "mcmc_agreement": mcmc,
            "mcmc_stratified": real_summary.get("mcmc_stratified", {}),
        },
        "baselines": {
            "bls": bls,
            "speed": speed,
        },
        "status": status,
    }


def write_environment(path: Path, repo: Path) -> None:
    env = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": git_sha(repo),
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
    path.write_text(json.dumps(env, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/publishable.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-root", default="results/publishable_runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--noise-lib", default="data/noise_lib.npz")
    ap.add_argument("--build-noise-lib", action="store_true")
    ap.add_argument("--noise-targets", nargs="*", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--n-sbc", type=int, default=1000)
    ap.add_argument("--n-detection", type=int, default=5000)
    ap.add_argument("--n-posterior", type=int, default=2000)
    ap.add_argument("--n-real-planets", type=int, default=30)
    ap.add_argument("--with-mcmc", type=int, default=16)
    ap.add_argument("--mcmc-steps", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=None,
                    help="override training steps; useful for fast metric checks")
    ap.add_argument("--fast-check", action="store_true",
                    help="short metric-oriented run: smaller data/eval/MCMC, same report schema")
    ap.add_argument("--smoke", action="store_true",
                    help="small structural run; not a metrics claim")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    run_name = args.run_name or time.strftime("publishable_%Y%m%d_%H%M%S")
    out_dir = (repo / args.out_root / run_name).resolve()
    logs = out_dir / "logs"
    results = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

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
        speed_n_amortized = 32
        speed_n_mcmc = 1
        speed_mcmc_steps = 80
        speed_mcmc_walkers = 16
    elif args.fast_check:
        n_data = min(args.n_data, 20000)
        steps = args.steps or 3000
        n_sbc = min(args.n_sbc, 200)
        n_detection = min(args.n_detection, 1000)
        n_posterior = min(args.n_posterior, 512)
        n_real_planets = min(args.n_real_planets, 12)
        with_mcmc = min(args.with_mcmc, 4)
        mcmc_steps = min(args.mcmc_steps, 400)
        speed_n_amortized = min(n_detection, 64)
        speed_n_mcmc = max(1, min(with_mcmc, 2))
        speed_mcmc_steps = mcmc_steps
        speed_mcmc_walkers = 16
    else:
        n_data = args.n_data
        steps = args.steps
        n_sbc = args.n_sbc
        n_detection = args.n_detection
        n_posterior = args.n_posterior
        n_real_planets = args.n_real_planets
        with_mcmc = args.with_mcmc
        mcmc_steps = args.mcmc_steps
        speed_n_amortized = 256
        speed_n_mcmc = 5
        speed_mcmc_steps = args.mcmc_steps
        speed_mcmc_walkers = 32
    train_steps = ["--steps", str(steps)] if steps else []

    write_environment(out_dir / "environment.json", repo)

    run([args.python, "-m", "pytest", "-q", "-m", "not slow", *FAST_PYTEST],
        repo, logs / "pytest_fast.log")

    if args.build_noise_lib and not noise_lib.exists():
        targets = args.noise_targets or DEFAULT_TARGETS
        run([args.python, "scripts/build_noise_library.py", "--mission", "TESS",
             "--n-raw", "18000", "--out", str(noise_lib), "--targets", *targets],
            repo, logs / "noise_lib.log")
    noise_meta = validate_noise_lib(noise_lib)
    (out_dir / "noise_lib.json").write_text(json.dumps(noise_meta, indent=2))

    if not list(data_dir.glob("shard_*.npz")):
        run([args.python, "scripts/generate_data.py", "--config", args.config,
             "--n", str(n_data), "--workers", str(args.workers),
             "--shard-size", str(args.shard_size), "--out", str(data_dir),
             "--noise-lib", str(noise_lib)],
            repo, logs / "generate_data.log")

    run([args.python, "scripts/preflight.py", "--config", args.config,
         "--expect", "cuda", "--data-dir", str(data_dir)],
        repo, logs / "preflight.log")
    run([args.python, "scripts/train.py", "--config", args.config,
         "--run-dir", str(run_dir), "--data-dir", str(data_dir),
         "--expect-device", "cuda", "--no-preflight", *train_steps],
        repo, logs / "train.log")

    ckpt = run_dir / "checkpoints" / "latest.pt"
    eval_dir = results / "synthetic"
    run([args.python, "scripts/evaluate.py", "--ckpt", str(ckpt),
         "--noise-lib", str(noise_lib), "--n-sbc", str(n_sbc),
         "--n-detection", str(n_detection), "--n-posterior", str(n_posterior),
         "--out", str(eval_dir), "--plots"],
        repo, logs / "evaluate.log")
    run([args.python, "scripts/baseline_detection.py", "--ckpt", str(ckpt),
         "--noise-lib", str(noise_lib), "--n", str(n_detection),
         "--out", str(results / "bls_vs_transitflow.json")],
        repo, logs / "baseline_detection.log")
    run([args.python, "scripts/benchmark_speed.py", "--ckpt", str(ckpt),
         "--noise-lib", str(noise_lib), "--n-amortized", str(speed_n_amortized),
         "--n-post", str(n_posterior), "--n-mcmc", str(speed_n_mcmc),
         "--mcmc-steps", str(speed_mcmc_steps), "--mcmc-walkers",
         str(speed_mcmc_walkers), "--out", str(results / "speed.json")],
        repo, logs / "speed.log")
    real_dir = results / "real"
    cmd = [args.python, "scripts/validate_real.py", "--ckpt", str(ckpt),
           "--detector-ckpt", str(ckpt), "--n-planets", str(n_real_planets),
           "--n-post", str(n_posterior), "--with-mcmc", str(with_mcmc),
           "--mcmc-steps", str(mcmc_steps), "--out", str(real_dir)]
    run(cmd, repo, logs / "validate_real.log")

    report = build_gate_report(
        read_json(eval_dir / "metrics.json"),
        read_json(real_dir / "real_validation.json"),
        read_json(results / "bls_vs_transitflow.json"),
        read_json(results / "speed.json"),
    )
    report["run"] = {
        "run_name": run_name,
        "config": args.config,
        "checkpoint": str(ckpt),
        "data_dir": str(data_dir),
        "noise_lib": str(noise_lib),
        "smoke": bool(args.smoke),
        "fast_check": bool(args.fast_check),
        "n_data": int(n_data),
        "steps": None if steps is None else int(steps),
        "n_sbc": int(n_sbc),
        "n_detection": int(n_detection),
        "n_posterior": int(n_posterior),
        "n_real_planets": int(n_real_planets),
        "with_mcmc": int(with_mcmc),
        "mcmc_steps": int(mcmc_steps),
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
