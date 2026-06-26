# TransitFlow char5 gate audit - 2026-06-26

## Verdict

Not publishable yet as a full real-data claim.

The 5-parameter characterization posterior fixes the invalid SBC target. The calibrated `latest.pt` checkpoint passes synthetic SBI gates under a multiple-comparison-aware SBC gate, but real-planet detection/characterization still fails. Do not claim real-planet performance until the real gates pass on a predeclared run.

## What passed

- Synthetic detection, `latest.pt` on the publishable-v2 run: ROC-AUC `0.9950010848249943`, AP `0.995316438103705`.
- Synthetic characterization SBC, `latest.pt`: family-wise alpha 0.05 Bonferroni gate passed.
  - Raw p-values: `RpRs=0.37087906036980756`, `aRs=0.011551931514417801`, `b=0.1393551080375877`, `q1=0.10843546249954811`, `q2=0.1515306327548624`.
  - Bonferroni per-test alpha: `0.01`; minimum p-value: `0.011551931514417801`.
  - Raw diagnostic `all p > 0.05` is false; this is reported but is not the required family-wise gate.
- Synthetic characterization coverage error, `latest.pt`: `0.0024736842105263267`.

## What failed

- Posterior-loss `best.pt` is not the calibrated checkpoint for the publishable-v2 run:
  - `best.pt` SBC p-values: `RpRs=1.2809393920693547e-05`, `aRs=0.1502764475906208`, `b=0.27054351335208193`, `q1=0.8939952044232142`, `q2=0.8733105587372922`.
  - Cause: lowest validation posterior loss is not a reliable calibration-selection rule. Use the final `latest.pt` checkpoint unless a calibration sweep is predeclared.
- Real detection failed on publishable-v2:
  - With `best_detection.pt`: `24/30 = 0.8` at `p_detect >= 0.9`.
  - With `latest.pt` as detector: `25/30 = 0.8333333333333334` at `p_detect >= 0.9`.
- Real same-light-curve MCMC agreement failed for calibrated `latest.pt`:
  - `RpRs` median Wasserstein prior fraction: `0.11778148155233963` > `0.1`.
  - `b` median Wasserstein prior fraction: `0.2046256519315231` > `0.1`.
  - Width-normalized `RpRs` diagnostic: `1.435496412648805` > `0.5`.
  - Width-normalized `aRs` diagnostic: `0.7985762716573361` > `0.5`.
- Literature interval coverage diagnostics fail and must stay exploratory unless catalog uncertainty and real-light-curve likelihood assumptions are modeled explicitly.

## Fixes made

- New default posterior target is 5D characterization: `RpRs, aRs, b, q1, q2`.
- Period and epoch are treated as ephemeris conditioning inputs, not sampled SBC targets.
- Evaluation gates now mark all-parameter SBC as not applicable for the 5D conditioned model.
- Synthetic SBC gates now use a family-wise alpha 0.05 Bonferroni correction across tested parameters. Raw unadjusted p-values remain reported as diagnostics.
- Training checkpoint policy changed:
  - `best.pt` = lowest validation posterior loss.
  - `best_detection.pt` = highest validation detection AUC.
  - `latest.pt` remains final/resume checkpoint.
- `validate_real.py` supports `--detector-ckpt`, so real detection and posterior characterization can use separately selected checkpoints without mixing claims.
- Real-data MCMC comparison now supports fixed ephemeris conditioning. For the 5D model, the default MCMC comparator fixes `P` and `t0_phase` to the same folded ephemeris used by the amortized posterior; use `--mcmc-full-ephemeris` only as a separate diagnostic.
- Real validation gates now exclude fixed `P` from characterization coverage gates; `P` remains in the report only as an ephemeris sanity check.

## Vast smoke after conditional-MCMC fix

Vast instance `176.12.30.23:41161` was prepared with `torch 2.12.0+cu130`; CUDA saw `NVIDIA GeForce RTX 4070 Ti SUPER`.

Smoke tests run on Vast:

- Direct smoke passed:
  - fixed-parameter MCMC returns fixed samples;
  - 5D ephemeris-conditioned inference returns full 7D physical samples;
  - standardized returned samples preserve the first two ephemeris dimensions.
- Targeted pytest passed: `4 passed`.
- Broader smoke pytest passed: `28 passed`, with only the known PyTorch scalar-conversion warning.
- Small GPU train/evaluate smoke passed structurally:
  - `posterior_param_names == ["RpRs", "aRs", "b", "q1", "q2"]`;
  - `len(sbc_pvalues) == 5`;
  - `all_parameter_sbc_p_gt_0.05 is None`;
  - 120-step smoke characterization SBC and coverage gates passed.

The 120-step smoke detection AUC did not pass and is not a metrics claim; it only verifies pipeline mechanics before a proper run.

## Downloaded evidence

Remote evidence was downloaded to:

- `artifacts/remote_char5_2026-06-26/results_char5_full/`
- `artifacts/remote_char5_2026-06-26/logs_char5_full/`
- `artifacts/remote_char5_2026-06-26/checkpoints/`
- `artifacts/vast_publishable_v2_2026-06-26/results_char5_publishable_v2/`
- `artifacts/vast_publishable_v2_2026-06-26/logs_char5_publishable_v2/`

## Next run rule

Before another full run:

1. Run a small smoke test that verifies `posterior_param_names == [RpRs, aRs, b, q1, q2]`.
2. Evaluate synthetic gates on `latest.pt`, unless a calibration-sweep checkpoint rule is predeclared before looking at test results.
3. Run real validation with `latest.pt` as posterior and detector unless a detector checkpoint also passes the real detection gate on a predeclared validation split.
4. Run real MCMC with fixed ephemeris for the 5D model; only continue to a full real-data claim if MCMC prior-fraction gate passes for `RpRs`, `aRs`, and `b`.
