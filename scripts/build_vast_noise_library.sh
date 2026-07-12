#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/TransitFlow

# Archive access is the bottleneck; modest concurrency avoids MAST throttling.
python -u scripts/select_noise_targets.py \
  --n-targets 180 \
  --seed 20260712 \
  --out data/noise_targets_gaia_20260712.txt \
  --metadata data/noise_targets_gaia_20260712.json

mapfile -t targets < data/noise_targets_gaia_20260712.txt
exec python -u scripts/build_noise_library.py \
  --mission TESS \
  --targets "${targets[@]}" \
  --n-raw 18000 \
  --workers 4 \
  --min-targets 60 \
  --max-segments-per-target 8 \
  --max-point-to-point-ppm 2500 \
  --target-provenance data/noise_targets_gaia_20260712.json \
  --out data/noise_lib_gaia_tess_20260712.npz
