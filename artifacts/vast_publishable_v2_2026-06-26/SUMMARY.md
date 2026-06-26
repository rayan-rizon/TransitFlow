# Vast publishable-v2 run summary - 2026-06-26

## Verdict

Not publishable for full real-planet characterization claims.

Synthetic characterization passes after correcting the SBC gate for multiple
tested parameters. A quality-gated real-planet detection run now passes, but
same-light-curve MCMC posterior agreement still fails for the real
characterization parameters, especially impact parameter `b`. The paper can
claim the synthetic SBI result and a real-detection smoke/validation result, but
must not claim validated real-planet posterior characterization yet.

## Synthetic latest checkpoint

- Checkpoint: `/workspace/runs/fmpe_pg_char5_publishable_v2/checkpoints/latest.pt`
- Detection ROC-AUC: `0.9950010848249943`
- Detection AP: `0.995316438103705`
- SBC raw p-values:
  - `RpRs=0.37087906036980756`
  - `aRs=0.011551931514417801`
  - `b=0.1393551080375877`
  - `q1=0.10843546249954811`
  - `q2=0.1515306327548624`
- SBC required gate: pass under family-wise alpha `0.05` Bonferroni correction
  (`0.05 / 5 = 0.01`; min p-value `0.011551931514417801`).
- Raw all-p-values-above-0.05 diagnostic: fail.
- Characterization coverage calibration error: `0.0024736842105263267`.

## Failed checkpoint

- Checkpoint: `/workspace/runs/fmpe_pg_char5_publishable_v2/checkpoints/best.pt`
- Detection ROC-AUC: `0.9957651613642734`
- SBC raw p-values:
  - `RpRs=1.2809393920693547e-05`
  - `aRs=0.1502764475906208`
  - `b=0.27054351335208193`
  - `q1=0.8939952044232142`
  - `q2=0.8733105587372922`
- Verdict: fails SBC because lowest validation posterior loss was not the
  calibrated checkpoint.

## Real validation

- `latest.pt` posterior with `best_detection.pt` detector:
  - `24/30 = 0.8` detected at `p_detect >= 0.9`.
- `latest.pt` posterior with `latest.pt` detector:
  - `25/30 = 0.8333333333333334` detected at `p_detect >= 0.9`.
- Both fail the real detection target of `>= 0.9`.
- Quality-gated `latest.pt` posterior/detector:
  - `28/30 = 0.9333333333333333` detected at `p_detect >= 0.9`.
  - This passes the real-detection gate.
  - Quality cuts are data-only: cadence count/fraction, in-transit cadence
    count, observed SNR, finite geometry, transit count, and max impact
    parameter.

## Real MCMC

- MCMC run: fixed ephemeris, 8 real planets, MCMC on 6 detected planets.
- Median MCMC acceptance fraction: `0.26612499999999994`.
- Median Wasserstein prior fractions:
  - `RpRs=0.11778148155233963`
  - `aRs=0.053294264475857334`
  - `b=0.2046256519315231`
- Gate `mcmc_characterization_prior_fraction_le_0.1`: fail.
- Width-fraction diagnostic:
  - `RpRs=1.435496412648805`
  - `aRs=0.7985762716573361`
  - `b=0.4261648909236442`
- Gate `mcmc_characterization_width_fraction_le_0.5_diagnostic`: fail.
- Fixed-ephemeris binned MCMC run, 30 real planets, first 8 detected planets:
  - detection gate: pass (`28/30 = 0.9333333333333333`).
  - median Wasserstein prior fractions:
    - `RpRs=0.0813709282034525`
    - `aRs=0.032384859648842045`
    - `b=0.17788409472192365`
  - MCMC prior-fraction gate: fail because `b > 0.1`.
  - width-fraction diagnostic: fail (`RpRs=1.5208972034113852`,
    `aRs=0.7379259279164547`, `b=0.5198001264407004`).
- Overdispersed MCMC initialization did not fix the full gate:
  - 8-object run: `b` prior fraction `0.19884527859001494`, fail.
  - 16-object run: `b` prior fraction `0.15513911204208006`, fail.
  - Larger MCMC sample improves stability but does not close the real
    characterization gate.
- Likelihood-corrected posterior smoke passed MCMC distances, but ESS collapsed
  (`median_ess_fraction=0.0030717597721427467`, `min_ess_fraction=0.0008333333333333334`).
  This is not acceptable as a publishable correction claim.

## Evidence files

- `results_char5_publishable_v2/eval_latest_full/metrics.json`
- `results_char5_publishable_v2/eval_posterior/metrics.json`
- `results_char5_publishable_v2/real_latest/real_validation.json`
- `results_char5_publishable_v2/real_latest_detector_latest/real_validation.json`
- `results_char5_publishable_v2/real_mcmc_latest/real_validation.json`
- `results_char5_publishable_v2/real_quality_mcmc_binned_full_v1/real_validation.json`
- `results_char5_publishable_v2/real_quality_mcmc_jitter_full_v1/real_validation.json`
- `results_char5_publishable_v2/real_quality_mcmc16_jitter_full_v1/real_validation.json`
- `results_char5_publishable_v2/real_quality_mcmc_corrected_smoke_v1/real_validation.json`
- `logs_char5_publishable_v2/pipeline.log`
- `logs_char5_publishable_v2/train.log`
