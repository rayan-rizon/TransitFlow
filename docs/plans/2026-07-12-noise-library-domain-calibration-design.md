# Domain-calibrated TESS noise-library design

## Problem and evidence

The 2026-07-12 v2 strong gate used source-target-uniform sampling and the
validation-selected posterior checkpoint. Central coverage passed, but SBC rank
uniformity failed for four of five characterization parameters. The frozen noise
archive contains only 14 stars, split into 8 training, 3 calibration, and 3
evaluation targets. The rank histograms show directional center bias across
unseen stars. A bounded exact-density calibration fixes out-of-prior samples but
does not remove this cross-target bias on the already-inspected development set.

## Considered approaches

1. **Catalog-derived real-star expansion (selected).** Query a versioned public
   catalog of stable stars, predeclare astrophysical filters, download TESS SPOC
   light curves, apply measured photometric-quality filters, and retain target
   provenance. This directly increases independent noise domains.
2. **Hand-curated target expansion.** Faster, but undocumented target choices
   create selection bias and are difficult to reproduce or defend in review.
3. **Synthetic-noise expansion only.** Cheap and useful for ablations, but it
   cannot establish calibration under unseen real TESS systematics.

## Selected data flow

The target selector reads the VizieR machine-readable Gaia DR2 radial-velocity
standard-star catalog `J/A+A/616/A7/rvstdcat`. It retains CAL1 FGK dwarfs with a
predeclared magnitude range, observation count, time baseline, and RV scatter,
then produces a deterministic, seed-recorded target list plus query provenance.
The existing noise builder downloads SPOC 2-minute products, records per-target
segment counts and robust photometric diagnostics, rejects targets that fail the
predeclared quality threshold, and fails closed unless the requested minimum
number of independent targets is reached.

The publication runner must reject a real-noise archive with fewer than 30
unique targets. Target identities remain disjoint across training, calibration,
and evaluation, and stars—not sectors—remain the sampling unit. The already
inspected v2 evaluation targets are development-only; a later decision run must
freeze a newly generated split before metrics are inspected.

## Calibration and testing

The posterior calibration transform is bounded and invertible: raw unbounded
standardized draws are mapped through a per-parameter affine latent transform
and `tanh` into the exact prior support. Its per-draw Jacobian is included in
`log_prob_std`. Coefficients are fitted only on calibration targets against SBC
rank shape and central coverage.

Tests cover catalog parsing/filtering, deterministic selection, minimum-target
failure, target-uniform draws, bounded support, exact round-trip inversion,
finite-difference Jacobian agreement, checkpoint provenance, and rank/coverage
improvement. The next execution sequence is: local regression suite, remote
catalog/noise-library smoke, saved-checkpoint development diagnostic, then one
fresh strong gate on a newly frozen target split. The full publication pipeline
remains blocked until that strong gate passes both familywise SBC and coverage.
