# Nonlinear posterior calibration design

## Evidence and decision

The v4 target-disjoint validation used 145 independently sourced TESS stars
(87/29/29 train/calibration/evaluation). It removed the impact-parameter SBC
failure found with the earlier 64-star archive, but marginal SBC still failed
for `a/Rs` and `q2`. Therefore increasing target count alone is insufficient.

The adopted correction is a per-parameter, conditional monotone map in bounded
latent space. Its latent center is quadratic in the deterministic FMPE center;
its positive raw-space scale has a log-linear center dependence. A final `tanh`
maps samples into the exact prior bounds. The inverse and Jacobian are analytic,
so calibrated samples and log densities remain consistent.

## Guardrails

- Fit only on the fixed calibration-target split; never on final evaluation stars.
- Use regularized, deterministic optimization with bounded coefficients.
- Preserve the earlier affine artifact schema on load, with zero nonlinear terms.
- Require unit tests for inverse round-trip, finite-difference Jacobian, and a
  synthetic nonlinear conditional-bias recovery case.
- Re-run the same frozen v4 configuration and accept it only if its unseen
  500-case SBC and coverage gates pass; detection is a separate paired gate.
