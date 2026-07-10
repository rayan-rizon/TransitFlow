# TransitFlow MNRAS publishability audit

Date: 2026-07-10
Target venue: *Monthly Notices of the Royal Astronomical Society* (MNRAS), Paper
Verdict: **not yet submission-ready; major scientific validation remains**

## Executive decision

TransitFlow is in scope for MNRAS as an astronomical inference method, and the
repository contains enough technical substance for a potentially publishable
methods paper. The present evidence does not support submission, however. The
latest full run reports `final_pass=false`, the historical detection comparison
used privileged ephemerides, and the historical real-posterior Wasserstein
metrics were computed after replacing the raw amortized posterior with a
degenerate importance-resampled posterior. The latter had a median ESS fraction
of 0.00181 and a minimum of 0.000333 (about 5.4 and 1 effective samples,
respectively, out of 3000).

This audit therefore treats the old real-characterization and 0.999 detection
numbers as non-publication diagnostics. Rewriting the manuscript cannot cure
those problems. The code changes in this pass make the next experiment honest
and reproducible; a new, frozen validation run is still required.

## What the current data do support

- A working, candidate-conditioned FMPE/SBI implementation for a five-dimensional
  transit-shape posterior over `RpRs`, `aRs`, `b`, `q1`, and `q2`.
- Strong pooled simulator-domain results in the newest run: ROC-AUC 0.99943 for
  the **oracle-candidate diagnostic**, minimum Bonferroni-aware SBC p-value
  0.1065, and mean absolute coverage error 0.0153.
- A selected positive-only TESS recovery result of 30/30 known transits. This is
  sensitivity on a high-quality confirmed sample, not real-world precision,
  specificity, or blind-search performance.
- A preliminary fair candidate-vetting result. On 1000 simulated curves,
  using BLS-derived candidates rather than simulator truth, TransitFlow achieved
  ROC-AUC 0.8074 (95% paired-bootstrap interval 0.7820--0.8338) and average
  precision 0.7619 (0.7312--0.7960), versus BLS ROC-AUC 0.5982
  (0.5610--0.6311) and AP 0.5841 (0.5509--0.6202). The paired AUC gain was
  0.2092 (0.1664--0.2553). This is a corrected pilot (`n=1000`), not the final
  held-out benchmark.
- Clear evidence of a conditional-calibration limitation: for high-impact-
  parameter simulations, the empirical 68% coverage for `b` is about 0.564,
  with its binomial interval excluding 0.68. Pooled coverage alone is therefore
  insufficient.

## What the current data do not support

- “First use of flow matching for transit detection.” Fiscale et al. (2025)
  already applied conditional flow matching to transit-light-curve
  classification.
- Blind or end-to-end transit discovery. TransitFlow is conditioned on a
  candidate ephemeris.
- State-of-the-art detection. The existing full benchmark privileged TransitFlow
  with the true positive ephemeris, while BLS/TLS had to search; TLS also used a
  smaller sample.
- Real-data calibration or MCMC-equivalent real posteriors. The real sample is
  selected, positive-only, and uses known periods. The old Wasserstein values
  were contaminated by collapsed-ESS resampling, and all 16 MCMC chains were
  shorter than 50 autocorrelation times.
- A general 1000--1,000,000x speed claim. The newest matched artifact reports
  1659.7x under one benchmark, with only five short, unconverged MCMC timings.
- Robustness across training seeds, target stars, sectors, hardware, or
  precision. Only one training seed is represented, the same real-noise library
  was reused for training and synthetic evaluation, and BF16 was not isolated
  on the same checkpoint and hardware.

## Research-integrity fixes completed in this pass

1. Seeded neural evaluation draws and the independent `emcee` proposal state.
2. Recorded evaluation, real-validation, and per-object MCMC seeds.
3. Added a publication-mode BLS-candidate path; the oracle candidate path remains
   available only as an explicitly labelled diagnostic.
4. Preserved pre-flattening simulated flux so candidate search/preprocessing no
   longer uses the simulator's true transit mask.
5. Added record-level score export plus stratified paired-bootstrap intervals for
   AUC/AP and method differences.
6. Separated raw amortized/MCMC agreement from importance-corrected diagnostics.
   Low-ESS correction can no longer overwrite the primary agreement metric.
7. Added MCMC gates requiring at least 50 autocorrelation times and 400 effective
   samples, and increased the full-run default from 1500 to 15000 steps.
8. Preserved correction, conditioning, diagnostic, and seed provenance in the
   top-level gate report.
9. Removed the binary scikit-learn dependency from the elementary detection
   metrics, using tie-aware rank AUC and threshold-grouped average precision.
10. Replaced the oracle-AUC publication gate with a fair BLS-candidate,
    paired-interval gate; added equal-sample TLS, probability-calibration
    diagnostics, deterministic target-level noise splits, and explicit
    multi-seed training support.

## Mandatory experiments before MNRAS submission

These are blocking, in order.

1. **Freeze and archive the executed source.** Commit the current fixes; record
   the patch hash, full command, environment lock, CUDA/cuDNN versions, data and
   checkpoint SHA-256 values, and exact MAST product identifiers.
2. **Rebuild leakage-safe noise libraries.** Split by target star and sector
   before injection; never reuse a group between train, validation, and test.
   The current `.npz` lacks source identities and cannot be retrospectively
   split.
3. **Retrain 3--5 independent seeds.** Report seed-level gates, mean, standard
   deviation, and threshold-flip frequency. Do not select a seed after seeing
   the test set.
4. **Run the final fair detection benchmark.** Use BLS-derived candidates for
   TransitFlow, the same `n` for BLS/TLS/TransitFlow, held-out target/sector
   groups, paired confidence intervals, reliability curves, and a modern learned
   classifier baseline. The 1000-object result is only an early signal because
   it still reuses the training noise library.
5. **Create an untouched real holdout.** Include confirmed planets and matched
   astrophysical/instrumental false positives. Freeze selection and quality cuts
   before inference. Report sensitivity, specificity, precision, calibration,
   and uncertainty.
6. **Rerun real posterior comparison from raw samples.** Use at least 15000 MCMC
   steps initially, multiple chains, split-Rhat below 1.01, bulk/tail ESS above
   400, and a chain length above 50 times the estimated autocorrelation time.
   Expand beyond 16 objects and use a one-sided upper confidence bound for any
   equivalence margin.
7. **Add the required posterior baselines/ablations.** At minimum FMPE versus NPE,
   real-noise versus synthetic-noise training, and candidate-ephemeris
   perturbation/alias tests. Report conditional coverage for S/N, `b`, `a/Rs`,
   period, dilution, gaps, and noise regime.
8. **Run a controlled BF16 equivalence test.** Same checkpoint, tensors, random
   base draws, GPU, and tolerances; compare logits, medians, widths, SBC,
   coverage, Wasserstein metrics, and threshold flips.
9. **Repeat speed benchmarks after convergence matching.** Same hardware,
   posterior sample count, accuracy target, and preprocessing; include repeated
   timings and an amortization break-even calculation.

## MNRAS compliance status

| Requirement | Status |
|---|---|
| Original astronomy contribution and explicit novelty | Partial; defensible claim narrowed |
| Concise Paper in MNRAS LaTeX style | Draft package created |
| Abstract no longer than 250 words and 1--6 approved keywords | Draft compliant |
| Harvard citations and verified reference list | Draft prepared; final DOI audit still required |
| Figures/tables with units, borders, captions, and accessible descriptions | Draft figures generated |
| Data availability statement | Present, but repository DOI/immutable release missing |
| AI-use disclosure in manuscript and cover letter | Present |
| Reproducible code/data supporting every conclusion | Fail; retained 115-segment library is anonymous and cannot be group-split |
| Scientific conclusions supported by frozen validation | Fail; blocking rerun required |
| Complete author/affiliation/funding metadata | User input required |

## Defensible paper claim after the required rerun

> TransitFlow adapts flow-matching posterior estimation to candidate-conditioned
> transit photometry, combining a candidate probability with an amortized joint
> posterior over transit-shape parameters. Its contribution is the integration
> of probabilistic candidate vetting, simulator-domain calibration, matched
> posterior comparison, and explicit misspecification diagnostics; it is not a
> new flow-matching algorithm or a blind transit-search system.

## Editorial recommendation

**Major revision / do not submit yet.** The architecture and synthetic results
are promising, and the fair pilot suggests value beyond the BLS score. The next
full run must be performed only after the source, group splits, candidate task,
MCMC convergence criteria, and primary metrics are frozen. If that run passes,
the MNRAS Paper route is plausible. If real posterior equivalence still fails,
submit only a narrower candidate-vetting and simulator-calibrated methods paper,
with real characterization presented as a measured domain-gap limitation.
