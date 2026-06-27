#!/usr/bin/env python3
"""Preflight a training config before committing GPU hours.

Runs a short probe and prints device, throughput, GPU-starvation %, projected
run time and cost, and a PASS/WARN/FAIL verdict. Exit code is non-zero on FAIL,
so it can gate a launch in a shell script.

Example
-------
    python3 scripts/preflight.py --config configs/publishable.yaml --expect cuda \
        --data-dir data/tess_1M --price 0.40
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import build_configs

from transitflow.train import preflight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--expect", default=None, help="expected device, e.g. cuda")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--probe", type=int, default=40)
    ap.add_argument("--price", type=float, default=0.40, help="USD/GPU-hour")
    args = ap.parse_args()

    overrides = {"train": {}}
    if args.expect:
        overrides["train"]["expect_device"] = args.expect
    if args.data_dir:
        overrides["train"]["data_source"] = "disk"
        overrides["train"]["data_dir"] = args.data_dir
    if args.num_workers is not None:
        overrides["train"]["num_workers"] = args.num_workers
    cfg = build_configs(args.config, overrides)

    r = preflight(cfg["model"], cfg["simulator"], cfg["train"],
                  n_probe=args.probe, price_per_hr=args.price, verbose=True)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
