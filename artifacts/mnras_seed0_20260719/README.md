# First frozen full-scale run (seed 0) — 2026-07-19/20

Run name `mnras_seed0_20260719`, gate revision `2026-07-19_predeclared_v2`.
This is the first full-scale (1,000,000-simulation) run executed under the
frozen protocol with an **external publication lockbox**. It is retained as
development evidence and as the reproducibility record for the next attempt.

## Outcome: near-miss, fail-closed before the real-data stage

The run completed data generation, training, posterior calibration, synthetic
evaluation and the full paired BLS/TLS detection benchmark. The characterization
SBC gate then failed on a single parameter, and
`external_lockbox_synthetic_failure_blocks_downstream` correctly halted the run
*before* consuming the real-planet/MCMC and speed stages. The non-zero exit is
that predeclared `SystemExit`, not a crash and not an interruption. No
`gate_report.json` exists because the downstream stages never ran.

| Quantity | Value | Gate |
|---|---:|---|
| Characterization coverage error | 0.00527 | <= 0.03 — pass |
| Fair blind-candidate detection ROC-AUC (n=5000) | 0.9296 | >= 0.88 — pass |
| Fair detection average precision | 0.9354 | — |
| BLS / TLS ROC-AUC on identical curves | 0.6873 / 0.6613 | — |
| SBC `aRs` / `b` / `q1` / `q2` | 0.067 / 0.829 / 0.106 / 0.434 | pass |
| **SBC `RpRs`** | **0.00516** | **fails Bonferroni 0.01** |

Training completed normally (best posterior validation loss 0.7666, best
detection validation ROC-AUC 0.946).

## Diagnosis of the `RpRs` failure

The `RpRs` rank histogram is mildly sloped upward — 458 of 1000 ranks in the
lower half versus 542 in the upper half — i.e. the posterior sits slightly
*low* on transit depth. The effect is small and directional; it is nothing like
the historical catastrophic failures (p ~ 1e-266).

Two facts identify the mechanism:

1. **It is not a candidate-ephemeris artefact.** `run_sbc` masks on
   `posterior_valid = is_planet & (candidate_kind == 0)`, so SBC is evaluated
   only on exact-ephemeris cases. Transit-depth dilution from mis-recovered BLS
   periods cannot explain it.
2. **It is a noise-domain generalization gap in the calibrator.** On the
   calibration split the fitted correction is excellent for `RpRs` (rank CvM
   0.0437 -> 0.0056, the best of the five parameters; coverage error 0.0100 ->
   0.0082). That correction was chosen by a per-dimension `argmin` over four
   candidates scored on only **11 selection targets**, and it picked the
   strongest available correction (`simple_100`; 0.0121 versus 0.0210, 0.0363,
   0.0658). Applied to 31 previously unseen lockbox stars it does not transfer.

The earlier fast-check reported all five parameters passing, but at 500 SBC
cases; at 1000 cases the same small bias becomes statistically visible. The
fast-check result was underpowered rather than contradicted.

## Reproducibility

`dev_noise_targets_152.txt` is the complete development noise-library target
list recovered from `noise_split.json`, and `dev_noise_split_by_role.json`
records the exact train/validation/calibration partition (105/16/31). A fresh
machine can rebuild the identical development library from these identifiers;
see `scripts/vast_bootstrap_full_run.sh`.

The publication lockbox for this run held 31 accepted targets (50 segments) and
was selected with `select_noise_targets.py --n-targets 200 --seed 20260802`
excluding all 152 development targets. Its per-target identifiers were lost with
the instance; a subsequent run should build a **new** lockbox, which is in any
case preferable — a lockbox is only untouched once.
