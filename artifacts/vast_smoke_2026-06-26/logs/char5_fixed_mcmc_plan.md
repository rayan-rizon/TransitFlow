# Fixed-ephemeris real MCMC validation plan

Purpose: validate the real-planet characterization gate against the same conditioning target as the 5D posterior.

Required inputs on Vast:
- /workspace/runs/fmpe_pg_char5_full/checkpoints/latest.pt  (posterior-calibrated checkpoint)
- /workspace/runs/fmpe_pg_char5_full/checkpoints/best.pt    (detector checkpoint from previous run, legacy name)

Smoke already passed on Vast:
- fixed MCMC all-fixed smoke
- 5D ephemeris-conditioned inference smoke
- targeted pytest: 4 passed
- broader smoke pytest: 28 passed
- tiny GPU train/eval structural smoke

Proper diagnostic command once checkpoints exist:
python scripts/validate_real.py \
  --ckpt /workspace/runs/fmpe_pg_char5_full/checkpoints/latest.pt \
  --detector-ckpt /workspace/runs/fmpe_pg_char5_full/checkpoints/best.pt \
  --n-planets 8 \
  --out /workspace/results_char5_fixed_mcmc/real_combo_mcmc_fixed \
  --with-mcmc 8 \
  --mcmc-steps 1000 \
  --n-post 1500

Pass criteria:
- detected_fraction_ge_0.9 should be interpreted only on n=30 real validation, not n=8 MCMC diagnostic.
- mcmc_characterization_prior_fraction_le_0.1 must pass for RpRs, aRs, b.
- width-fraction gate remains diagnostic; do not overfit to it without a paper-ready rationale.
