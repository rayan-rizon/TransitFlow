#!/usr/bin/env python3
"""Generate a fixed, provenance-bearing blind-BLS audit dataset.

This preserves the checkpoint's physical simulator while forcing a BLS proposal
for every positive and negative curve.  It is intended for the identifiability
audit, not model training or a publication lockbox.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transitflow.data import generate_to_disk
from transitflow.train import load_checkpoint


def fair_bls_detector_config(sim_cfg):
    """Return the immutable audit candidate domain for both classes."""
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noise-lib", required=True,
                    help="held-out residual targets only")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()
    if args.n <= 0 or args.workers < 0 or args.shard_size <= 0 or args.gen_batch <= 0:
        raise SystemExit("--n, --shard-size, and --gen-batch must be positive")

    _, _, checkpoint_cfg = load_checkpoint(args.ckpt, device="cpu")
    audit_cfg = fair_bls_detector_config(checkpoint_cfg)
    out = Path(args.out)
    generate_to_disk(
        audit_cfg, args.n, str(out), shard_size=args.shard_size,
        num_workers=args.workers, gen_batch=args.gen_batch, seed=args.seed,
        noise_lib_path=args.noise_lib, verbose=True,
    )
    manifest = {
        "kind": "held_out_identifiability_dataset",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "noise_library": str(Path(args.noise_lib).resolve()),
        "n": int(args.n),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "candidate_protocol": "blind_astropy_bls_for_both_classes",
        "simulator_config": asdict(audit_cfg),
    }
    path = out / "identifiability_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
