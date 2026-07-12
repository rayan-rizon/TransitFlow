# Frozen MNRAS validation runbook

This is the execution contract for the next publication-scale experiment. Do
not change the model, data-selection rules, gates, or manuscript claims after
inspecting the held-out results. A failed gate is a scientific result, not a
reason to tune against the test partition.

## Preconditions

1. Use a clean, committed checkout and record `git rev-parse HEAD`.
2. Install `requirements.txt`; verify CUDA in `scripts/preflight.py`.
3. Rebuild `data/noise_lib.npz` with `--build-noise-lib`, retaining target and
   exact product provenance.
4. Confirm `noise_split.json` reports disjoint, nonempty training, calibration,
   and final-evaluation target groups. Posterior calibration is fitted only on
   the calibration group.
5. Confirm `sampling_unit` is `source_target_uniform_then_segment_v1`; cached
   sector counts must not weight source stars unequally.
6. Use `best.pt` selected by validation posterior loss for characterization and
   `best_detection.pt` selected by validation AP for detection. `latest.pt` is
   resume-only.
7. Use the same frozen evaluation and real-data seeds for every training seed.

## Commands

First run a fresh strong gate. It is a decision run, not publication evidence:

```bash
python3 scripts/run_publishable_vast.py \
  --fast-check \
  --run-name calibrated_candidate_strong_fast \
  --config configs/publishable.yaml \
  --noise-lib data/noise_lib.npz \
  --n-data 100000 \
  --steps 10000 \
  --n-sbc 500 \
  --n-detection 2000 \
  --n-posterior 1000 \
  --with-tls-baseline
```

Do not start the full commands unless the fresh characterization SBC/coverage
gates pass and the paired AP comparison no longer shows a significant TLS
disadvantage.

Build the source-labelled noise library in the first run from the automatic,
seeded catalog selector, requiring at least 30 successful independent targets.
Then reuse that exact file for seeds 1 and 2 by omitting `--build-noise-lib`.

```bash
for seed in 0 1 2; do
  python3 scripts/run_publishable_vast.py \
    --run-name "mnras_seed_${seed}" \
    --config configs/publishable.yaml \
    --noise-lib data/noise_lib.npz \
    --min-noise-targets 30 \
    --train-seed "$seed" \
    --eval-seed 123 \
    --real-seed 20260710 \
    --n-data 1000000 \
    --n-sbc 1000 \
    --n-detection 5000 \
    --n-posterior 2000 \
    --n-real-planets 30 \
    --with-mcmc 16 \
    --mcmc-steps 15000 \
    --mcmc-max-steps 60000 \
    --mcmc-walkers 32 \
    --is-correct-mcmc \
    --is-samples 3000 \
    --min-is-ess-fraction 0.05 \
    --with-tls-baseline
done
```

Add `--build-noise-lib --noise-workers 1` only to the seed-0 command. Serial
archive access avoids Lightkurve/Astroquery progress-stream and cache races. This writes
the catalog query and selected targets inside the run directory and the library
quality/provenance sidecar beside the archive. Do not rebuild or alter the noise
source archive between seeds.

## Hard stop criteria

The top-level `gate_report.json` must retain every pass and failure. Publication
claims remain blocked if any of these occur:

- target overlap among training, calibration, and evaluation noise libraries;
- fewer than 30 successful independent source targets in the noise archive;
- segment-weighted rather than source-target-uniform real-noise sampling;
- characterization from a final/resume checkpoint instead of the predeclared
  validation-posterior-loss checkpoint;
- non-BLS candidate ephemerides in the primary detection benchmark;
- a non-positive lower 95% paired-bootstrap bound for either AUC or AP gain;
- fewer than 5000 identical BLS/TLS/TransitFlow examples;
- characterization SBC or expected-coverage gate failure;
- any real reference chain below 50 production autocorrelation times, split-Rhat
  above 1.01, or bulk/tail ESS below 400;
- importance-correction ESS fraction below 0.05 when correction is reported;
- a failed real-posterior agreement or speed gate;
- materially inconsistent conclusions across the three training seeds.

The manuscript also requires the modern learned-vetter comparison, untouched
real positive/negative holdout, FMPE--NPE ablation, candidate-perturbation
ablation, controlled BF16/FP32 comparison, and immutable public archive listed
in `PUBLISHABILITY_AUDIT.md`. Those are not replaced by `final_pass=true`.
