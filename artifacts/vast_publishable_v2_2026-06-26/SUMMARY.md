# Vast publishable-v2 run summary - 2026-06-26

## Verdict

Not publishable for real-planet claims.

Synthetic characterization passes after correcting the SBC gate for multiple
tested parameters. Real detection and real same-light-curve MCMC agreement still
fail, so the paper must not claim real-planet characterization performance from
this run.

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

## Evidence files

- `results_char5_publishable_v2/eval_latest_full/metrics.json`
- `results_char5_publishable_v2/eval_posterior/metrics.json`
- `results_char5_publishable_v2/real_latest/real_validation.json`
- `results_char5_publishable_v2/real_latest_detector_latest/real_validation.json`
- `results_char5_publishable_v2/real_mcmc_latest/real_validation.json`
- `logs_char5_publishable_v2/pipeline.log`
- `logs_char5_publishable_v2/train.log`
