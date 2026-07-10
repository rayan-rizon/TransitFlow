# Artifact retention index

## Canonical

- `canonical/vast_publishable_bf16_5090_20260708_v3/` is the latest historical
  full-run bundle. It is retained for provenance, not as publication evidence.
  Its original gate was false and its oracle-candidate/importance-resampled
  results are explicitly superseded by the audit.

## Current diagnostics

- `fair_candidate_validation_n1000.json` and matching score data: primary fair
  BLS-candidate pilot.
- `oracle_candidate_diagnostic_n1000.json` and matching score data: paired
  oracle-ephemeris diagnostic, never a blind-detection result.
- `*_n200.*`: short development pilots retained to document the transition.

## Historical provenance

- `vast_publishable_full_20260705_221655/`: FP32 historical bundle, reduced to
  its final checkpoint and reports.
- `vast_publishable_bf16_20260707_214524/`: BF16 historical bundle, reduced to
  its final checkpoint and reports.
- `vast_publishable_2026-07-03_full_03cca0a_cpu23/`: small early report bundle.

Intermediate step checkpoints, interrupted temporary checkpoints, PID files,
caches, ad-hoc monitors, and generated scratch files were removed. The retained
`data/noise_lib.npz` contains 115 segments but no source identifiers; the new
full runner therefore requires it to be rebuilt before publication validation.
