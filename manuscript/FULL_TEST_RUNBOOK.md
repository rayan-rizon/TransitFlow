# Frozen MNRAS validation runbook

This is the execution contract for the next publication-scale experiment. Do
not change the model, data-selection rules, gates, or manuscript claims after
inspecting the held-out results. A failed gate is a scientific result, not a
reason to tune against the test partition.

## Preconditions

1. Use a clean, committed checkout and record `git rev-parse HEAD`.
2. Install `requirements.txt`; verify CUDA in `scripts/preflight.py`.
3. Rebuild `data/noise_lib.npz` with `--build-noise-lib`. The retained legacy
   file has 115 anonymous segments and is intentionally rejected by a full run.
4. Confirm `noise_split.json` has no target overlap. The runner reserves whole
   source targets, a stricter rule than splitting sectors from the same target.
5. Use the same frozen evaluation and real-data seeds for every training seed.

## Commands

Build the source-labelled noise library in the first run, then reuse that exact
file for seeds 1 and 2 by omitting `--build-noise-lib`.

```bash
for seed in 0 1 2; do
  python3 scripts/run_publishable_vast.py \
    --run-name "mnras_seed_${seed}" \
    --config configs/publishable.yaml \
    --noise-lib data/noise_lib.npz \
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
    --mcmc-walkers 32 \
    --is-correct-mcmc \
    --is-samples 3000 \
    --min-is-ess-fraction 0.05 \
    --with-tls-baseline
done
```

Add `--build-noise-lib --noise-workers 4` only to the seed-0 command. Do not
rebuild or alter the noise source archive between seeds.

## Hard stop criteria

The top-level `gate_report.json` must retain every pass and failure. Publication
claims remain blocked if any of these occur:

- target overlap between training and evaluation noise libraries;
- non-BLS candidate ephemerides in the primary detection benchmark;
- a non-positive lower 95% paired-bootstrap bound for either AUC or AP gain;
- fewer than 5000 identical BLS/TLS/TransitFlow examples;
- characterization SBC or expected-coverage gate failure;
- any real reference chain shorter than 50 autocorrelation times or below 400
  effective samples;
- importance-correction ESS fraction below 0.05 when correction is reported;
- a failed real-posterior agreement or speed gate;
- materially inconsistent conclusions across the three training seeds.

The manuscript also requires the modern learned-vetter comparison, untouched
real positive/negative holdout, FMPE--NPE ablation, candidate-perturbation
ablation, controlled BF16/FP32 comparison, and immutable public archive listed
in `PUBLISHABILITY_AUDIT.md`. Those are not replaced by `final_pass=true`.
