#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/TransitFlow

# Prevent each worker from creating its own large BLAS/OpenMP thread pool.
# Parallelism is controlled explicitly by the pipeline worker arguments below.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_name="${RUN_NAME:-calibrated_candidate_strong_fast_v2_20260712}"

exec python -u scripts/run_publishable_vast.py \
  --fast-check \
  --run-name "${run_name}" \
  --out-root results/publishable_runs \
  --config configs/publishable.yaml \
  --noise-lib data/noise_lib_mnras_20260711.npz \
  --train-seed 17 \
  --eval-seed 20260712 \
  --real-seed 20260712 \
  --n-data 100000 \
  --workers 40 \
  --shard-size 2500 \
  --steps 10000 \
  --n-sbc 500 \
  --n-detection 2000 \
  --n-posterior 1000 \
  --with-tls-baseline \
  --tls-workers 42 \
  --mcmc-processes 16 \
  --amp
