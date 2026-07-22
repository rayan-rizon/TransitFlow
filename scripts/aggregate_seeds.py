#!/usr/bin/env python3
"""Aggregate per-seed gate reports into cross-seed stability statistics.

The publishable protocol runs several independent training seeds, each writing
its own ``gate_report.json``. A single passing seed is not evidence of a stable
method; MNRAS readiness requires that the gate decisions are reproducible across
seeds. This script reads N gate reports and reports, per gate flag:

* pass frequency (how many seeds passed),
* the modal decision and the **threshold-flip frequency** (fraction of seeds
  whose decision differs from the modal one) -- the direct instability metric,

and, per continuous metric, the cross-seed mean / SD / min / max. It also
reports how many seeds reached ``final_pass`` and an overall stability verdict:
all seeds must agree on ``final_pass`` for the result to be called stable.

Usage
-----
    python scripts/aggregate_seeds.py \
        --reports results/publishable_runs/*/gate_report.json \
        --out results/seed_aggregate.json

Continuous metrics are addressed by dotted path into the gate report so the set
is explicit and auditable rather than a blind tree walk.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

# Continuous metrics to summarize across seeds: (label, dotted path).
CONTINUOUS_METRICS = [
    ("sbc_min_pvalue", "synthetic.characterization_sbc_gate.min_pvalue"),
    ("coverage_calibration_error",
     "synthetic.characterization_coverage_calibration_error"),
    ("detection_roc_auc", "synthetic.detection.roc_auc"),
    ("detection_average_precision", "synthetic.detection.average_precision"),
    ("real_detected_fraction", "real.detection.detected_fraction"),
]


def _dig(obj: dict, path: str):
    """Follow a dotted path; return None if any step is missing."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _mean_sd(values: list[float]) -> dict:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    n = len(finite)
    mean = sum(finite) / n
    # sample SD (n-1); 0.0 for a single seed
    var = sum((v - mean) ** 2 for v in finite) / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": math.sqrt(var),
            "min": min(finite), "max": max(finite)}


def aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate a list of parsed gate reports into stability statistics.

    Pure function (stdlib only) so it is unit-testable without the pipeline.
    """
    seeds = []
    status_dicts = []
    for rep in reports:
        run = rep.get("run", {}) or {}
        seeds.append(run.get("train_seed"))
        status_dicts.append(rep.get("status", {}) or {})

    # union of all gate-flag names across seeds
    flag_names = sorted({k for s in status_dicts for k in s})
    n_seeds = len(reports)

    flags = {}
    for name in flag_names:
        decisions = [s.get(name) for s in status_dicts]
        present = [d for d in decisions if d is not None]
        n_pass = sum(1 for d in present if d)
        counts = Counter(present)
        modal = counts.most_common(1)[0][0] if counts else None
        n_flip = sum(1 for d in present if d != modal)
        flags[name] = {
            "n_reported": len(present),
            "n_pass": n_pass,
            "pass_fraction": (n_pass / len(present)) if present else None,
            "modal_decision": modal,
            "n_flip": n_flip,
            "flip_frequency": (n_flip / len(present)) if present else None,
            "unanimous": bool(present) and len(counts) == 1,
        }

    metrics = {}
    for label, path in CONTINUOUS_METRICS:
        vals = [_dig(rep, path) for rep in reports]
        vals = [float(v) if isinstance(v, (int, float)) else None for v in vals]
        metrics[label] = {**_mean_sd(vals), "per_seed": vals}

    final_pass_decisions = [s.get("final_pass") for s in status_dicts]
    n_final_pass = sum(1 for d in final_pass_decisions if d)
    final_unanimous = len(set(d for d in final_pass_decisions
                              if d is not None)) <= 1
    any_flip = any(f["n_flip"] > 0 for f in flags.values())

    return {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "final_pass": {
            "n_pass": n_final_pass,
            "per_seed": final_pass_decisions,
            "unanimous": final_unanimous,
            "all_pass": n_final_pass == n_seeds and n_seeds > 0,
        },
        "gate_flags": flags,
        "continuous_metrics": metrics,
        "stable": bool(n_seeds > 0 and final_unanimous and not any_flip),
        "n_flags_with_flips": sum(1 for f in flags.values() if f["n_flip"] > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True,
                    help="gate_report.json paths (globs allowed)")
    ap.add_argument("--out", default="results/seed_aggregate.json")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.reports:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[")
                     else [pat])
    paths = [p for p in paths if os.path.exists(p)]
    if len(paths) < 2:
        print(f"WARNING: only {len(paths)} gate report(s) found; cross-seed "
              "stability needs >=2 (>=3 recommended).", file=sys.stderr)
    reports = [json.load(open(p)) for p in paths]

    result = aggregate_reports(reports)
    result["report_paths"] = paths

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    # human-readable summary to stdout
    print(f"seeds: {result['seeds']}  (n={result['n_seeds']})")
    print(f"final_pass: {result['final_pass']['n_pass']}/{result['n_seeds']} "
          f"unanimous={result['final_pass']['unanimous']}")
    print(f"flags with cross-seed flips: {result['n_flags_with_flips']}")
    for name, f in result["gate_flags"].items():
        if f["n_flip"] > 0:
            print(f"  FLIP {name}: pass {f['n_pass']}/{f['n_reported']} "
                  f"flip_freq={f['flip_frequency']:.2f}")
    print(f"STABLE: {result['stable']}")


if __name__ == "__main__":
    main()
