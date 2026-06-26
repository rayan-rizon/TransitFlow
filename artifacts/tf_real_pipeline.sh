#!/usr/bin/env bash
# TransitFlow — real-noise retrain pipeline (Gate #3 fix)
# Runs on new Vast.ai box. Stages: NOISE_LIB → GEN → TRAIN → EVAL → REAL_VAL
set -euo pipefail

PYTHON=/venv/main/bin/python3
REPO=/workspace/TransitFlow
LOG_DIR=/workspace
DATA_DIR=/workspace/data/tess_1M_real
NOISE_LIB=/workspace/data/noise_lib.npz
RUN_DIR=/workspace/TransitFlow/runs/fmpe_pg_real
RESULT_DIR=/workspace/results_real_v3
POSTERIOR_CKPT=$RUN_DIR/checkpoints/latest.pt
DETECTOR_CKPT=$RUN_DIR/checkpoints/best_detection.pt

mkdir -p /workspace/data

cd $REPO

stage() {
    local name=$1
    local log=$2
    shift 2
    echo "[pipeline] $name start $(date -u)"
    if "$@" 2>&1 | tee "$log"; then
        echo "[pipeline] $name PASS $(date -u)"
    else
        echo "[pipeline] $name FAIL $(date -u)" >&2
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Build real-noise library from quiet TESS stars
# ─────────────────────────────────────────────────────────────────────────────
if [ -f "$NOISE_LIB" ]; then
    echo "[pipeline] NOISE_LIB already exists, skipping"
else
stage NOISE_LIB $LOG_DIR/noise_lib.log \
    $PYTHON scripts/build_noise_library.py \
        --mission TESS \
        --n-raw 18000 \
        --out $NOISE_LIB \
        --targets \
            "HD 10700" \
            "HD 197076" \
            "HD 1461" \
            "HD 36435" \
            "HD 101501" \
            "HD 26965" \
            "HD 32147" \
            "HD 40307" \
            "HD 20794" \
            "HD 85512" \
            "HD 7924" \
            "HD 136352" \
            "HD 190406" \
            "HD 131977" \
            "HD 10647"
fi  # end noise_lib skip guard

echo "[pipeline] noise_lib written: $NOISE_LIB"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Generate 1 M training LCs with real noise injected
# ─────────────────────────────────────────────────────────────────────────────
stage GEN $LOG_DIR/gen_real.log \
    $PYTHON scripts/generate_data.py \
        --config configs/default.yaml \
        --n 1000000 \
        --workers 72 \
        --shard-size 10000 \
        --out $DATA_DIR \
        --noise-lib $NOISE_LIB

du -sh $DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Retrain FMPE + periodogram on real-noise dataset
# ─────────────────────────────────────────────────────────────────────────────
stage TRAIN $LOG_DIR/train_real.log \
    $PYTHON scripts/train.py \
        --config configs/default.yaml \
        --run-dir $RUN_DIR \
        --data-dir $DATA_DIR \
        --expect-device cuda \
        --no-preflight

if [ ! -f "$DETECTOR_CKPT" ]; then
    echo "[pipeline] WARNING: $DETECTOR_CKPT missing; falling back to legacy best.pt"
    DETECTOR_CKPT=$RUN_DIR/checkpoints/best.pt
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Evaluate (SBC + coverage + detection + speed)
# ─────────────────────────────────────────────────────────────────────────────
stage EVAL $LOG_DIR/eval_real.log \
    $PYTHON scripts/evaluate.py \
        --ckpt $POSTERIOR_CKPT \
        --noise-lib $NOISE_LIB \
        --out $RESULT_DIR/eval \
        --plots

echo "=== EVAL METRICS ===" && cat $RESULT_DIR/eval/metrics.json

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: Real-planet validation (Gate #3)
# ─────────────────────────────────────────────────────────────────────────────
stage REAL_VAL $LOG_DIR/real_v3.log \
    $PYTHON scripts/validate_real.py \
        --ckpt $POSTERIOR_CKPT \
        --detector-ckpt $DETECTOR_CKPT \
        --n-planets 30 \
        --out $RESULT_DIR/real

echo "[pipeline] ALL DONE $(date -u)"
cat $RESULT_DIR/real/real_validation.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('\n=== GATE #3 REAL-PLANET SUMMARY ===')
print(f\"n_planets: {d['summary']['n_planets']}\")
for k, v in d['summary']['per_param'].items():
    print(f\"  {k:<6} cov68={v['coverage_68']:.3f}  cov95={v['coverage_95']:.3f}  frac_err={v['median_frac_err']:.3f}\")
"
