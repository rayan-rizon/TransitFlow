# Vast MNRAS full run: seed 0

This directory contains the compact, publication-relevant outputs downloaded
from the completed Vast run `mnras_full_20260711_seed_0`. Large training data,
intermediate checkpoints, duplicate record checkpoints, and noisy transient
logs are intentionally excluded.

- Executed source: `96ea536e71425d90aac1f84eb913d93fb7743492`
- Training seed: 0
- Evaluation seed: 123
- Real-validation seed: 20260710
- Dataset: 1,000,000 simulated curves in 100 shards
- Training: 60,000/60,000 steps
- Final decision: `gate_report.json` records `final_pass=false`

The run completed operationally. Its failures are scientific: severe synthetic
posterior undercoverage, non-significant AP improvement over BLS, unconverged
MCMC references, failed real-characterization margins, and collapsed importance
weights.

The executed code did not calculate a paired TransitFlow-versus-TLS interval.
A post-run paired bootstrap on the retained score arrays (500 replicates, seed
20123) gives:

- ROC-AUC gain: +0.03385; 95% interval [0.01334, 0.05429]
- Average-precision gain: -0.03691; 95% interval [-0.05816, -0.01390]

This post-run comparison strengthens future reporting but is not represented as
a preregistered seed-0 gate. The harness now computes and gates it in subsequent
runs.

Two focused FP32 diagnostics rule out BF16 and checkpoint selection as the main
calibration cause:

- `latest.pt`, FP32, 200 SBC cases: coverage error 0.1006; four of five SBC
  parameter tests reject uniformity.
- validation-selected `best.pt`, FP32, 100 SBC cases: coverage error 0.1224;
  the same four parameters reject uniformity.

These diagnostics use seed 4242 and 512 posterior samples. They are diagnostic,
not substitutes for the 1,000-case full result. They show that retraining or a
model-level uncertainty remedy is required; simply disabling BF16 or selecting
`best.pt` does not repair the posterior.
