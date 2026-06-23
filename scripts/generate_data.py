#!/usr/bin/env python3
"""Pre-generate a binned training dataset to disk (the cost-efficient workflow).

Run this on a cheap CPU box (or locally — it's free) before renting a GPU. It
writes compact float16 shards that GPU training streams at full speed, so the
expensive GPU isn't left idle waiting on the CPU simulator. The same dataset is
reused for the FMPE and NPE runs.

Examples
--------
    # 1M light curves, 8 worker processes, into data/tess_1M/
    python scripts/generate_data.py --config configs/default.yaml \
        --n 1000000 --workers 8 --out data/tess_1M
    # then train from disk (GPU stays saturated):
    python scripts/train.py --config configs/default.yaml --run-dir runs/fmpe \
        --data-dir data/tess_1M
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import build_configs

from transitflow.data import generate_to_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, required=True, help="total light curves")
    ap.add_argument("--out", required=True, help="output directory for shards")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=50000)
    ap.add_argument("--noise-lib", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = build_configs(args.config)
    sim_cfg = cfg["simulator"]
    print(f"generating {args.n:,} light curves "
          f"(n_global={sim_cfg.n_global}, n_local={sim_cfg.n_local}, "
          f"n_raw={sim_cfg.n_raw}) with {args.workers} workers -> {args.out}")
    t0 = time.time()
    generate_to_disk(sim_cfg, args.n, args.out, shard_size=args.shard_size,
                     num_workers=args.workers, seed=args.seed,
                     noise_lib_path=args.noise_lib, verbose=True)
    dt = time.time() - t0
    print(f"done in {dt/60:.1f} min  ({args.n/max(dt,1e-9):.0f} LC/s)")


if __name__ == "__main__":
    main()
