# Prior-matched characterization posterior design

## Failure evidence

The target-disjoint v4 evaluation is development evidence, not a passing result.
With 500 SBC simulations and 1000 posterior draws, its bounded nonlinear
calibrator returned marginal p-values `[0.000, 0.019, 0.899, 0.072, 0.000]` for
`RpRs`, `aRs`, `b`, `q1`, and `q2`, with a characterization coverage error of
0.015.  A new probit-bounded calibrator removed the `q2` rejection in a smaller
development test but still rejected `RpRs` and `aRs` and missed the coverage
gate.  Conditioning the calibrator on posterior spread improved its fit split
but degraded held-out evaluation, so that candidate was rejected as overfit.

The subsequent 100,000-example, 10,000-step target-disjoint development gate
confirmed that the prior-normal correction alone was not sufficient. Coverage
passed at 0.0132, but held-out characterization SBC p-values were
`[0.0006, 0.1218, 0.000022, 0.0185, 0.2839]`. An exact same-case uncalibrated
diagnostic failed the first three parameters and had coverage error 0.0295.
Audit of the calibration fit found that its scalar objective could trade worse
rank uniformity for better central coverage, despite SBC being the primary gate.

Code audit then exposed two model-generating-distribution mismatches.  The
stellar-density draw for `a/Rs` was clipped to the prior interval, creating
point masses at the bounds that were absent from the density used by inference.
Also, the flow's standard-normal base was asked to learn bounded uniform-like
marginals directly; the residual boundary distortion was then delegated to a
small post-hoc calibrator.

## Adopted correction

1. Sample the period-conditional stellar-density `a/Rs` prior from its exact
   support-truncated normal CDF.  Include the same truncation normalization in
   the analytic prior density.  This removes artificial boundary atoms and
   makes simulator sampling and density evaluation identical.
2. Transform the five characterization targets through their exact conditional
   prior CDFs and then through the standard-normal quantile function.  For
   `a/Rs`, the CDF is conditional on the candidate period; the other four use
   their declared box/log-box priors.  Train FMPE in these prior-normal
   coordinates and invert the transform for physical samples.
3. Track the exact Jacobian in posterior-density evaluation.  Preserve the old
   standardized target as an explicit configuration option for historical
   checkpoints, and fail closed if a prior-normal model is trained from a shard
   that lacks the new target field.
4. Use independent deterministic random streams for detection, SBC, and
   coverage.  Evaluate discrete SBC ranks against the declared `L + 1` support,
   with exact expected mass when collapsed rank bins have unequal cardinality.
5. Keep the probit-bounded monotone calibrator available, fitted only on the
   calibration-target split.  Calibration may correct residual finite-training
   error; it may not redefine the simulator or use evaluation targets.
6. Use a fourth, target-disjoint development role for checkpoint selection.
   Gradient training, checkpoint validation, posterior calibration, and
   development evaluation may not share source targets.  A full publication
   run replaces the development evaluation role with the external lockbox.
7. Split calibration targets again into calibrator-fit and calibrator-selection
   groups. Fit a simple bounded-affine candidate and the conditional candidate,
   include exact identity as a no-calibration control, and select one global
   family only on the target-held-out calibration-selection group. The
   predeclared score is mean rank Cramer-von Mises distance plus 0.25 times mean
   central-coverage error, so SBC uniformity remains primary. Preserve the fit
   and selection arrays and hashes for offline audit.

## Alternatives rejected

- More target stars alone: v4 removed one earlier failure but left strong
  marginal rejections.
- Increasing calibration flexibility: the spread-conditioned candidate fitted
  calibration simulations better and generalized worse.
- Relaxing SBC or coverage thresholds: this would change the decision rule
  after seeing the results and is not acceptable.
- Continuing the full run with the old target: the strong-gate failures were
  stop signals, so additional compute could not make that experiment valid.

## Verification and publication boundary

Unit tests must cover prior normalization and sampling, both transform
round-trips, finite-difference Jacobians, sample support, exact transformed
log-density, disk-shard training, component RNG independence, and discrete SBC
support.  They must also prove pairwise target disjointness across training,
checkpoint validation, calibration, and evaluation.  A short GPU smoke may establish that data generation, training,
calibration, and evaluation execute together and that the correction moves the
development diagnostics in the expected direction.  It cannot support a paper
claim.

Because the existing evaluation targets influenced this model decision, they
are now development data.  A publication test requires a newly frozen,
untouched target-level lockbox, predeclared metrics, multiple training seeds,
and the complete synthetic, fair-candidate, real-positive/negative, converged
MCMC, ablation, and timing gates.  The manuscript remains blocked until those
gates pass without threshold changes or seed selection.
