# Frozen MNRAS validation runbook

This is the execution contract for the next publication-scale experiment. Do
not change the model, data-selection rules, gates, or manuscript claims after
inspecting the held-out results. A failed gate is a scientific result, not a
reason to tune against the test partition.

## Preconditions

1. Use a clean, committed checkout and record `git rev-parse HEAD`.
2. Install `requirements.txt`; verify CUDA in `scripts/preflight.py`.
3. Retain the inspected 145-star development archive as
   `data/noise_lib.npz`; it may supply training, calibration and fast-check
   evaluation, but not the publication test.
4. Build `data/noise_publication_lockbox.npz` from at least 30 newly selected
   source targets that have never appeared in development. Freeze its target
   list, exact product provenance and SHA-256 before the full run. Never pass
   this archive to a fast check or inspect injected results from it early.
5. Confirm `noise_split.json` reports disjoint, nonempty development training
   and calibration groups plus zero target overlap with the external publication
   evaluation archive. Posterior calibration is fitted only on development
   calibration targets.
6. Confirm `sampling_unit` is `source_target_uniform_then_segment_v1`; cached
   sector counts must not weight source stars unequally.
7. Use `best.pt` selected by validation posterior loss for characterization and
   `best_detection.pt` selected by validation AP for detection. `latest.pt` is
   resume-only.
8. Use the same frozen evaluation and real-data seeds for every training seed.

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
disadvantage. This command uses development evaluation targets and must not be
given the publication lockbox.

Both source-labelled archives must already be frozen before the seed loop. Reuse
the identical development and publication-lockbox files for every seed.

```bash
for seed in 0 1 2; do
  python3 scripts/run_publishable_vast.py \
    --run-name "mnras_seed_${seed}" \
    --config configs/publishable.yaml \
    --noise-lib data/noise_lib.npz \
    --publication-eval-noise-lib data/noise_publication_lockbox.npz \
    --min-publication-eval-targets 30 \
    --min-noise-targets 120 \
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

Do not use `--build-noise-lib` inside this loop and do not rebuild, alter or
inspect the publication evaluation archive between seeds.

## Hard stop criteria

The top-level `gate_report.json` must retain every pass and failure. Publication
claims remain blocked if any of these occur:

- a missing external publication lockbox, or target overlap between it and the
  development training/calibration archive;
- fewer than 120 successful independent source targets in the noise archive;
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
