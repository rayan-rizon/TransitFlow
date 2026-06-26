# TransitFlow char5 gate audit - 2026-06-26

## Verdict

Not publishable yet as a full real-data claim.

The 5-parameter characterization posterior fixes the invalid SBC target. The calibrated `latest.pt` checkpoint passes synthetic SBI gates, but real-planet characterization agreement still fails the same-light-curve MCMC gate. Do not claim real-planet characterization performance until that gate passes on a predeclared run.

## What passed

- Synthetic detection, `latest.pt`: ROC-AUC `0.9934104098618766`, AP `0.993962262248824`.
- Synthetic characterization SBC, `latest.pt`: all p-values above `0.05`.
  - `RpRs`: `0.1335675860349418`
  - `aRs`: `0.686403029655644`
  - `b`: `0.218516920904009`
  - `q1`: `0.5844479234330378`
  - `q2`: `0.9972933497912274`
- Synthetic characterization coverage error, `latest.pt`: `0.006189473684210502`.
- Real detection with detector/posterior split: `27/30` planets detected at `p_detect >= 0.9`.

## What failed

- Detector-selected `best.pt` is not posterior-calibrated:
  - SBC p-values: `RpRs=4.98e-08`, `aRs=1.38e-06`, `b=0.0156`, `q1=0.000222`, `q2=0.290`.
  - Cause: checkpoint was selected by validation detection AUC, not posterior loss/SBC.
- Real same-light-curve MCMC agreement failed for calibrated `latest.pt`:
  - `RpRs` median Wasserstein prior fraction: `0.13184571686620705` > `0.1`.
  - `b` median Wasserstein prior fraction: `0.11330731432083904` > `0.1`.
  - Width-normalized `RpRs` diagnostic: `2.004233630505782` > `0.5`.
- Literature interval coverage diagnostics fail and must stay exploratory unless catalog uncertainty and real-light-curve likelihood assumptions are modeled explicitly.

## Fixes made

- New default posterior target is 5D characterization: `RpRs, aRs, b, q1, q2`.
- Period and epoch are treated as ephemeris conditioning inputs, not sampled SBC targets.
- Evaluation gates now mark all-parameter SBC as not applicable for the 5D conditioned model.
- Training checkpoint policy changed:
  - `best.pt` = lowest validation posterior loss.
  - `best_detection.pt` = highest validation detection AUC.
  - `latest.pt` remains final/resume checkpoint.
- `validate_real.py` supports `--detector-ckpt`, so real detection and posterior characterization can use separately selected checkpoints without mixing claims.
- Real-data MCMC comparison now supports fixed ephemeris conditioning. For the 5D model, the default MCMC comparator fixes `P` and `t0_phase` to the same folded ephemeris used by the amortized posterior; use `--mcmc-full-ephemeris` only as a separate diagnostic.

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

## Next run rule

Before another full run:

1. Run a small smoke test that verifies `posterior_param_names == [RpRs, aRs, b, q1, q2]`.
2. Evaluate synthetic gates on the posterior-calibrated checkpoint.
3. Run real validation with `--ckpt <posterior>` and `--detector-ckpt <detector>`.
4. Run real MCMC with fixed ephemeris for the 5D model; only continue to a full real-data claim if MCMC prior-fraction gate passes for `RpRs`, `aRs`, and `b`.
