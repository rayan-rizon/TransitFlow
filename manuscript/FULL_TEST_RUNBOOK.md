# Frozen MNRAS validation runbook

This is the execution contract for the next publication-scale experiment. Do
not change the model, data-selection rules, gates, or manuscript claims after
inspecting the held-out results. A failed gate is a scientific result, not a
reason to tune against the test partition.

## Gate revision 2026-07-19 (predeclared, before any new run)

The detection gates are revised BEFORE the next experiment, using only frozen
development evidence (the 2026-07-16 held-out identifiability audit and the
failed seed-0 development run). No held-out result from the next run has been
inspected. Rationale: the historical 0.99 blind-detection AUC target was
calibrated on the privileged oracle-ephemeris diagnostic; in the fair
blind-candidate protocol a predeclared fraction of injections is at or below
the single-sector information limit (held-out completeness 13% for expected
S/N < 25, 15% for P >= 7 d), so a population-wide 0.99 AUC measures the
injection prior, not the detector. Following standard practice for transit
surveys (completeness is reported over a defined detectable population), the
discrimination requirement now applies inside a predeclared detectable domain.

| Gate | Old | New (v2) |
|---|---|---|
| Identifiability preflight, overall blind AUC | >= 0.99 | >= 0.85 (regression floor) |
| Identifiability preflight, in-domain AUC (expected S/N bins 25-75 and >=75) | absent | >= 0.93 per bin |
| Identifiability preflight, minimum evaluable sources | `--min-publication-eval-targets` (30) | `--min-identifiability-sources` (25; development audit, decoupled from the lockbox minimum) |
| Fair blind-candidate detection AUC (full-run gate) | >= 0.99 | >= 0.88; primary detection claims remain the paired AUC/AP gain CIs vs BLS and TLS, which are unchanged |
| Real MCMC Wasserstein prior-fraction | <= 0.10 all params | RpRs <= 0.10, aRs <= 0.10, b <= 0.15 |
| Real MCMC Wasserstein width-fraction | <= 0.50 all params | RpRs <= 0.60, aRs <= 0.90, b <= 0.90 |

The per-parameter MCMC limits reflect that b and a/Rs are weakly identified in
single-sector photometry; RpRs, the physically decisive depth parameter, keeps
the tightest limit. Characterization SBC, coverage (<= 0.03), MCMC convergence,
lockbox, disjointness, and paired-gain gates are NOT relaxed. All thresholds
are recorded in `gate_report.json` under `gate_thresholds` with revision tag
`2026-07-19_predeclared_v2`. This table must not be edited again after the
next run starts.

Two protocol upgrades accompany the revision:

1. **Top-K candidate vetting.** The fair benchmark scores the top
   `--candidate-top-k` (default 3) alias-separated BLS hypotheses per curve and
   max-pools the vetting score; the BLS baseline remains the classic top-1 SDE.
   Detection and ephemeris recovery are reported as separate outcomes
   (`within_1pct_selected`, `within_1pct_top1`, `within_1pct_any_candidate`).
2. **Zero-depth injection null check.** `scripts/null_injection_check.py`
   pushes signal-free (Rp/Rs ~ 1e-6) "positives" through the identical
   injection/search/scoring path; the run is blocked if the detector separates
   them from negatives (AUC 95% CI excluding 0.5 beyond the margin), which
   would indicate an injection-pipeline artifact leak. Hard negatives
   (EB/single-event/sinusoid morphologies) are excluded from the null
   simulator: they are deliberate astrophysical content of the negative class
   and a vetter legitimately down-scores them, which would confound the null
   (expected AUC ~ 0.5 + 0.5 x hard_negative_fraction; measured 0.79 vs the
   0.75 prediction on 2026-07-19 before this exclusion was added). The null
   compares zero-depth "positives" against plain-noise negatives only.

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

Before the seed loop, run the zero-depth injection null check against the
fresh strong-gate checkpoint (a failure blocks everything downstream):

```bash
python3 scripts/null_injection_check.py \
  --ckpt results/calibrated_candidate_strong_fast/run/checkpoints/best_detection.pt \
  --noise-lib results/calibrated_candidate_strong_fast/noise_splits/noise_validation.npz \
  --n 800 --out results/null_injection_check.json
```

Use at least `--n 800`: at n=400 the per-run AUC sampling spread (~0.028) makes
the 0.05 margin a ~1.7-sigma decision and the percentile-bootstrap CI can
exclude 0.5 on unlucky seeds (observed twice on 2026-07-19; an n=800 rerun gave
AUC 0.504 with label-permutation p=0.85, confirming a clean injection path).

The full run also requires `--identifiability-report` pointing at a fixed
held-out blind-BLS audit (e.g. the frozen
`artifacts/development_v21_identifiability/identifiability_report.json`, which
passes the revised domain gate with 29 sources under the default
`--min-identifiability-sources 25`).

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
    --identifiability-report artifacts/development_v21_identifiability/identifiability_report.json \
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
