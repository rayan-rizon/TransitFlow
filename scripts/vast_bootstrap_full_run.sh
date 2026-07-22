#!/usr/bin/env bash
# Reproduce the frozen TransitFlow validation suite on a clean Vast.ai instance.
#
#   git clone <repo> /workspace/TransitFlow && cd /workspace/TransitFlow
#   git checkout codex/mnras-readiness
#   bash scripts/vast_bootstrap_full_run.sh              # full run, seed 0
#   MODE=fastcheck bash scripts/vast_bootstrap_full_run.sh   # ~3 h rehearsal
#   SEED=1 bash scripts/vast_bootstrap_full_run.sh       # further seeds
#   SEED=2 bash scripts/vast_bootstrap_full_run.sh       # (reuses the seed-0 lockbox)
#
# Run seed 0 FIRST: it builds the frozen, seed-independent publication lockbox
# (data/noise_publication_lockbox.npz). Seeds 1 and 2 reuse that identical
# archive; the script refuses to run a seed>0 if the lockbox is absent, so the
# evaluation noise domain is held fixed across the multi-seed comparison.
#
# Every stage is skipped when its output already exists, so the script can be
# re-run after an interruption without repeating the expensive downloads.
#
# Wall-clock on one RTX 3090 / 44 vCPU: development library ~40 min, lockbox
# ~60 min, full run ~8-12 h (dominated by the 16 converged real-planet chains).
set -uo pipefail

REPO="${REPO:-/workspace/TransitFlow}"
PY="${PY:-/venv/main/bin/python3}"
MODE="${MODE:-full}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-44}"
RUN_NAME="${RUN_NAME:-mnras_${MODE}_seed${SEED}_$(date -u +%Y%m%d)}"

DEV_TARGETS="artifacts/mnras_seed0_20260719/dev_noise_targets_152.txt"
IDENT_REPORT="artifacts/development_v21_identifiability/identifiability_report.json"
# The publication lockbox is seed-INDEPENDENT: the frozen FULL_TEST_RUNBOOK.md
# requires the identical evaluation archive for every training seed, otherwise
# training-seed variance and evaluation-noise variance are confounded and the
# cross-seed threshold-flip frequency is ill-defined. Build it once (fixed
# selection seed, fixed path) and reuse it across the seed loop.
DEV_LIB="data/noise_lib_dev_152.npz"
LOCKBOX="data/noise_publication_lockbox.npz"
LOCKBOX_TARGETS="data/lockbox_targets.txt"
LOCKBOX_SELECT_SEED="${LOCKBOX_SELECT_SEED:-20260900}"

cd "$REPO" || { echo "FATAL: repo not found at $REPO"; exit 1; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
mkdir -p data

step() { echo; echo "=== $* ==="; }
fail() { echo "STAGE_FAILED: $*"; exit 1; }

step "[0/4] dependencies"
if ! "$PY" -c "import torch, astropy, emcee, lightkurve, transitleastsquares" 2>/dev/null; then
  "$PY" -m pip install -q -r requirements.txt || fail "pip install"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# The development library is rebuilt from the exact target identifiers used by
# the 2026-07-19 seed-0 run, so the training/calibration noise domain is
# reproducible rather than resampled.
step "[1/4] development noise library (152 recorded targets)"
if [ -f "$DEV_LIB" ]; then
  echo "present, skipping: $DEV_LIB"
else
  [ -f "$DEV_TARGETS" ] || fail "missing recorded target list $DEV_TARGETS"
  mapfile -t devt < "$DEV_TARGETS"
  # --workers 1 is required, not merely conservative: lightkurve/astropy FITS
  # reads are not thread-safe and concurrent downloads corrupt the cache.
  "$PY" -u scripts/build_noise_library.py \
    --mission TESS --targets "${devt[@]}" \
    --n-raw 18000 --workers 1 --min-targets 120 \
    --max-segments-per-target 8 --max-point-to-point-ppm 2500 \
    --out "$DEV_LIB" || fail "development noise library"
fi

step "[2/4] external publication lockbox (disjoint from development, seed-independent)"
if [ -f "$LOCKBOX" ]; then
  echo "present, reusing frozen lockbox (identical for every seed): $LOCKBOX"
else
  if [ "$SEED" != "0" ]; then
    fail "lockbox $LOCKBOX absent on SEED=$SEED; build it once on seed 0 and reuse it (do not rebuild per seed)"
  fi
  # Request far more candidates than the 30-target minimum: acceptance runs
  # ~25-40% once the development targets are excluded, and a 100-candidate
  # request produced only 24 accepted targets on 2026-07-19.
  "$PY" -u scripts/select_noise_targets.py \
    --n-targets 250 --seed "$LOCKBOX_SELECT_SEED" \
    --exclude-target-file "$DEV_TARGETS" \
    --out "$LOCKBOX_TARGETS" \
    --metadata "${LOCKBOX_TARGETS%.txt}.json" || fail "lockbox target selection"
  mapfile -t lbt < "$LOCKBOX_TARGETS"
  "$PY" -u scripts/build_noise_library.py \
    --mission TESS --targets "${lbt[@]}" \
    --n-raw 18000 --workers 1 --min-targets 30 \
    --max-segments-per-target 8 --max-point-to-point-ppm 2500 \
    --target-provenance "${LOCKBOX_TARGETS%.txt}.json" \
    --out "$LOCKBOX" || fail "lockbox noise library"
fi

# The FITS cache is only an input to the archives above; the full run needs the
# space far more than it needs the cache.
rm -rf /root/.cache/lightkurve/* 2>/dev/null
df -h "$REPO" | tail -1

step "[3/4] ${MODE} run: ${RUN_NAME}"
COMMON=(
  --run-name "$RUN_NAME"
  --config configs/publishable.yaml
  --noise-lib "$DEV_LIB"
  --with-tls-baseline
  --amp
  # shard-size 5000 keeps every data worker busy: the default 10000 yields only
  # 10 shards for a 100k dataset and caps concurrency at 10 regardless of cores.
  --shard-size 5000
  --workers "$WORKERS"
  --dataset-worker-memory-mib 900
  --dataset-worker-reserve-gib 4
  --mcmc-processes 16
)

if [ "$MODE" = "fastcheck" ]; then
  "$PY" scripts/run_publishable_vast.py "${COMMON[@]}" \
    --fast-check \
    --n-data 100000 --steps 10000 --n-sbc 500 --n-detection 2000 \
    --n-posterior 1000
else
  "$PY" scripts/run_publishable_vast.py "${COMMON[@]}" \
    --publication-eval-noise-lib "$LOCKBOX" \
    --min-publication-eval-targets 30 \
    --identifiability-report "$IDENT_REPORT" \
    --min-noise-targets 120 \
    --train-seed "$SEED" --eval-seed 123 --real-seed 20260710 \
    --n-data 1000000 --n-sbc 1000 --n-detection 5000 --n-posterior 2000 \
    --n-real-planets 30 \
    --with-mcmc 16 --mcmc-steps 15000 --mcmc-max-steps 60000 --mcmc-walkers 32 \
    --is-correct-mcmc --is-samples 3000 --min-is-ess-fraction 0.05
fi
RUN_EXIT=$?

step "[4/4] result"
OUT="results/publishable_runs/${RUN_NAME}"
echo "run exit: ${RUN_EXIT} (non-zero also means a predeclared gate failed)"
for report in "$OUT/gate_report.json" "$OUT/synthetic_gate_report.json"; do
  [ -f "$report" ] && { echo "--- $report ---"; \
    "$PY" -c "import json,sys; d=json.load(open('$report')); \
print(json.dumps(d.get('status', d.get('all_declared_synthetic_gates_pass')), indent=1))"; }
done
# Cross-seed aggregation: after each full-run seed, roll up every sibling seed's
# gate report into mean/SD and threshold-flip frequency. Non-fatal (it only
# summarizes what exists) and meaningful once >=2 seeds have completed. Skipped
# in fast-check (no gate_report) mode.
if [ "$MODE" != "fastcheck" ]; then
  SEED_REPORTS=(results/publishable_runs/mnras_${MODE}_seed*/gate_report.json)
  if [ -e "${SEED_REPORTS[0]}" ]; then
    step "[5/5] cross-seed aggregate (found ${#SEED_REPORTS[@]} gate report(s))"
    "$PY" scripts/aggregate_seeds.py \
      --reports "${SEED_REPORTS[@]}" \
      --out results/seed_aggregate.json || \
      echo "aggregate_seeds failed (non-fatal)"
  fi
fi

echo "BOOTSTRAP_COMPLETE_EXIT:${RUN_EXIT}"
exit "$RUN_EXIT"
