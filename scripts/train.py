#!/usr/bin/env python3
"""Train a TransitFlow model from a YAML config.

Examples
--------
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/default.yaml --run-dir runs/fmpe \
        --num-workers 4
    python scripts/train.py --config configs/default.yaml --head npe \
        --run-dir runs/npe
    # resume an interrupted run (auto-detects runs/fmpe/checkpoints/latest.pt):
    python scripts/train.py --config configs/default.yaml --run-dir runs/fmpe --resume
"""

from __future__ import annotations

import argparse
import os

from _config import build_configs

from transitflow.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TransitFlow")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--head", choices=["fmpe", "npe"], default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None,
                    help="directory for checkpoints, logs, status.json")
    ap.add_argument("--resume", nargs="?", const="__auto__", default=None,
                    help="resume from a checkpoint path, or bare flag to "
                         "auto-resume <run-dir>/checkpoints/latest.pt")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="data-prefetch worker processes (use 2-6 on a GPU box)")
    ap.add_argument("--noise-lib", default=None, help="path to a real-noise .npz")
    args = ap.parse_args()

    overrides = {"train": {}, "model": {}}
    if args.head:
        overrides["model"]["head"] = args.head
    if args.steps:
        overrides["train"]["n_steps"] = args.steps
    if args.device:
        overrides["train"]["device"] = args.device
    if args.run_dir:
        overrides["train"]["run_dir"] = args.run_dir
    if args.num_workers is not None:
        overrides["train"]["num_workers"] = args.num_workers
    if args.noise_lib:
        overrides["train"]["noise_lib_path"] = args.noise_lib
    if args.resume is not None:
        overrides["train"]["resume"] = (args.run_dir if args.resume == "__auto__"
                                        else args.resume)

    cfg = build_configs(args.config, overrides)
    if cfg["train"].run_dir:
        os.makedirs(cfg["train"].run_dir, exist_ok=True)
    elif cfg["train"].ckpt_path:
        os.makedirs(os.path.dirname(cfg["train"].ckpt_path) or ".", exist_ok=True)

    result = train(cfg["model"], cfg["simulator"], cfg["train"], verbose=True)
    print("training complete. best val AUC:",
          round(result["best"].get("roc_auc", float("nan")), 4),
          "| run_dir:", result["run_dir"])


if __name__ == "__main__":
    main()
