#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/TransitFlow

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec python -u scripts/run_publishable_vast.py \
  --fast-check \
  --run-name fixed_domain_smoke_20260712 \
  --out-root results/publishable_runs \
  --config configs/publishable.yaml \
  --noise-lib data/noise_lib_gaia_tess_20260712.npz \
  --min-noise-targets 60 \
  --train-seed 17 \
  --eval-seed 20260712 \
  --real-seed 20260712 \
  --n-data 8192 \
  --workers 40 \
  --shard-size 1024 \
  --steps 2000 \
  --n-sbc 50 \
  --n-detection 200 \
  --n-posterior 128 \
  --n-real-planets 4 \
  --with-mcmc 0 \
  --with-tls-baseline \
  --tls-workers 42 \
  --amp
