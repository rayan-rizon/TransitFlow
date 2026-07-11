# Calibrated candidate-conditioned validation design

## Objective

Repair the two demonstrated train/test mismatches without weakening any existing
publication gate: posterior uncertainty is selected without a calibration
criterion, and the detector is trained on exact planet ephemerides but tested on
BLS-derived candidates.

## Chosen design

1. Split source-labelled real-noise targets into disjoint training, calibration,
   and final-evaluation groups. The calibration group is never used for gradient
   updates and the final-evaluation group is never used for model or calibration
   choices.
2. Fit a conditional-centred diagonal affine map in standardized posterior
   space on calibration simulations only. Its centre is the deterministic FMPE
   transport of base point zero; a bounded centre regression and positive
   dispersion scale remain an invertible change of variables, so sampling and
   `log_prob_std` stay mathematically consistent.
3. Add planet-candidate augmentation: small period/epoch errors and common
   period harmonics affect the detection task, while only exact candidates enter
   the characterization loss. This prevents inconsistent geometry labels from
   contaminating the posterior target.
4. Preserve publication gates and add explicit provenance for calibration data,
   fitted parameters, and candidate regime. A final held-out evaluation remains
   the only publishability decision.
5. Adaptively extend every reference MCMC chain to a declared cap and fail
   closed unless production length, split-Rhat, bulk ESS, and tail ESS all pass.
   The speed benchmark uses the same fixed ephemeris, known dilution, exposure
   integration, jitter model, and convergence requirements as validation.

## Verification

Unit tests cover split disjointness, affine calibration/inverse density
consistency, candidate-loss masking, and checkpoint provenance. The smoke run
must exercise the new simulator fields, calibrator fitting, calibrated sampling,
and likelihood transform. A subsequent fast GPU gate is a decision point; the
full multi-seed run occurs only if it passes.

## Smoke result

The independent structural smoke used 64 calibration simulations and 96 new
evaluation simulations. Candidate augmentation was active for 46.1% of rows;
held-out detection ROC-AUC/AP were 0.852/0.849. Coverage error changed from
0.0566 raw to 0.0238 calibrated. This passes the structural smoke but is not a
publication claim because the model and sample sizes are intentionally small.
