# TransitFlow MNRAS publishability audit

Date: 2026-07-15
Target venue: *Monthly Notices of the Royal Astronomical Society* (MNRAS), Paper
Verdict: **not yet submission-ready; major scientific validation remains**

## Addendum 2026-07-19: predeclared gate revision v2

Before launching the next experiment, and using only frozen development
evidence, the detection gates were revised to measure the detector rather than
the injection prior: the oracle-era 0.99 blind AUC target is replaced by a
domain-restricted identifiability gate (in-domain AUC >= 0.93 for expected
S/N >= 25, overall floor 0.85), the full-run fair-candidate AUC gate is 0.88
with the paired gain CIs unchanged as the primary claim, and the real-data
MCMC agreement limits are per-parameter (RpRs tightest). The fair benchmark
now vets the top-3 alias-separated BLS hypotheses per curve (max-pooled score;
BLS baseline unchanged), reports detection and ephemeris recovery separately,
and a zero-depth injection null check blocks the run on any injection-artifact
separability. Full table and rationale: `FULL_TEST_RUNBOOK.md`, "Gate revision
2026-07-19". Characterization SBC/coverage, MCMC convergence, lockbox, and
disjointness gates are unchanged. This revision was recorded before any new
held-out result was inspected; it must not be revised again after the run
starts.

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

## Latest failure analysis and bounded remediation smoke

The most recent target-disjoint v4 development evaluation used 145 independent
TESS source stars (87/29/29 train/calibration/evaluation), 500 SBC simulations
and 1000 posterior draws. Its nonlinear bounded calibration produced marginal
SBC p-values of 0.000, 0.019, 0.899, 0.072 and 0.000 for `RpRs`, `aRs`, `b`,
`q1` and `q2`. Coverage error was 0.015, but `RpRs`, `aRs` and `q2` failed the
familywise SBC decision. A probit-bounded variant removed the `q2` rejection in
a smaller development check but still rejected `RpRs` and `aRs`; conditioning
calibration on posterior spread fitted the calibration split better and
generalized worse. That candidate was rejected rather than promoted.

The root-cause audit identified two generating-distribution defects. The
stellar-density `a/Rs` simulator clipped unconstrained draws to [3, 50],
creating boundary atoms absent from the analytic prior density. In addition,
the flow learned bounded uniform-like prior marginals directly from a Gaussian
base. The corrected implementation now samples the exact support-truncated,
period-conditional stellar-density prior and includes its normalization in the
density. Characterization targets are transformed by their exact conditional
prior CDFs into standard-normal coordinates; samples and densities use the
analytic inverse and Jacobian. Evaluation components now have independent RNG
streams, and discrete SBC bin expectations use the declared posterior rank
support rather than the largest observed rank. Historical training shards fail
closed under an explicit dataset schema because they lack the new target.

A bounded GPU smoke used 6000 training simulations, 3000 optimizer steps, 200
calibration simulations and a separate 200-case evaluation with 256 posterior
draws. It exercised data generation, training, checkpoint selection, bounded
probit calibration, density inversion and evaluation. Characterization SBC
p-values were 0.094, 0.612, 0.117, 0.175 and 0.674; coverage error was 0.0111
and the out-of-prior fraction was zero. Detection reached ROC-AUC 0.903 and AP
0.917, failing the 0.99 detection gate. Calibration scales of 4.45--6.66 and a
large train--validation loss gap show severe small-data overfitting. The smoke
therefore establishes execution and a promising calibration direction only; it
is underpowered, single-seed development evidence and is not a paper result.

The 29-star v4 evaluation group has influenced model selection and is no longer
an untouched publication test. Before a full run, a new target-level lockbox
must be acquired and frozen. The full pipeline remains blocked until that
lockbox, all model choices, thresholds and multi-seed analysis are predeclared.

## Completed full-run decision (seed 0)

The first frozen full-scale run completed every operational stage: 1,000,000
simulations, 60,000 training steps, 1,000 SBC cases, a 5,000-object equal-sample
BLS/TLS/TransitFlow benchmark, 30 real confirmed planets, 16 fixed-ephemeris
MCMC comparisons, importance diagnostics, and timing. The train/evaluation noise
split contains 11 versus 3 target stars with no target overlap. This is a valid
completed experiment, not an interrupted run. Its scientific gate result is
`final_pass=false`.

| Publication test | Seed-0 result | Decision |
|---|---:|---|
| SBC minimum Bonferroni-aware p-value | 1.25e-266 | Fail |
| Mean absolute coverage error | 0.09598 (limit 0.03) | Fail |
| Nominal 95% synthetic coverage | 0.8096 | Undercoverage |
| TransitFlow versus BLS ROC-AUC gain | +0.03585, CI [0.01661, 0.05706] | Pass |
| TransitFlow versus BLS AP gain | +0.01653, CI [-0.00496, 0.04163] | Fail |
| TransitFlow versus TLS ROC-AUC gain | +0.03385, CI about [0.012, 0.055] | Positive |
| TransitFlow versus TLS AP gain | -0.03691, CI about [-0.060, -0.014] | Worse than TLS |
| Candidate-score ECE | 0.3117 | Poor calibration |
| Selected real-positive sensitivity | 28/30 | Pass; positive-only |
| Minimum MCMC chain length | 8.97 autocorrelation times (limit 50) | Fail |
| Minimum MCMC effective samples | 143.6 (limit 400) | Fail |
| Importance correction | 0/16 valid; median ESS fraction 0.00102 | Fail |
| Raw timing ratio | 13,003.7x | Diagnostic only; MCMC unconverged |

The posterior is severely overconfident across multiple parameters and strata.
On real data, `a/Rs` archive coverage is 0.133 at nominal 68% and 0.400 at
nominal 95%; raw MCMC agreement also misses the `a/Rs` width and `b` prior-range
margins. The 28/30 real result measures sensitivity on selected known positives
and cannot establish real precision, specificity, or false-positive rate.

The only publication-quality detection result presently supported is a modest
paired ROC-AUC improvement over BLS on this simulator benchmark. Average
precision does not significantly improve over BLS and is significantly below
TLS. The 13,003.7x timing ratio is retained as raw engineering evidence, not an
accuracy- or convergence-matched scientific speed claim.

These results must not be repaired by relaxing thresholds, selecting a favorable
seed, or calibrating on this test set. Seeds 1 and 2 should complete to quantify
training instability; they cannot erase the seed-0 failure. Model/calibration
choices must be made on separate validation data, followed by a newly frozen,
untouched multi-seed test.

Focused post-run diagnostics also exclude two simple numerical explanations.
With BF16 disabled, `latest.pt` still gives coverage error 0.1006 and rejects SBC
uniformity for `RpRs`, `aRs`, `b`, and `q1` (200 cases, 512 draws, seed 4242).
The validation-loss-selected `best.pt` is worse at 0.1224 with the same rejection
pattern (100 cases). The failure is therefore not cured by FP32 inference or
checkpoint selection; it requires model/training calibration work.

## Remediation implemented after seed 0

The failed seed-0 evaluation remains frozen development evidence. The next run
uses a target-disjoint train/calibration/evaluation split, an exact-density
conditional-centred affine posterior calibration fitted only on calibration
targets, and BLS-like candidate augmentation for positives and negatives.
Jittered/harmonic planet candidates train detection but are excluded from the
characterization loss through a persisted `posterior_valid` mask.

Reference MCMC is now burn-aware and adaptive, with fail-closed requirements for
50 production autocorrelation times, split-Rhat <=1.01, bulk ESS >=400, and tail
ESS >=400. The speed protocol matches fixed ephemeris, known dilution, finite
exposure, phase binning, and jitter likelihood; every timed chain must converge
and the 95% speedup lower bound must exceed the gate.

The corrected structural smoke passed: candidate-augmented ROC-AUC/AP were
0.852/0.849 and untouched-smoke coverage error improved from 0.0566 to 0.0238.
This validates the implementation path only. MNRAS readiness still requires a
fresh target-held-out strong gate and then the frozen multi-seed full suite.

The first 100k/10k-step target-held-out strong gate on 2026-07-12 was stopped
because characterization failed: `RpRs` and `aRs` SBC p-values were about
1.7e-166 and 2.3e-16, and coverage error was 0.0548 (gate <=0.03). Audit found
segment-weighted noise sampling: two stars supplied 75/116 cached segments.

The corrected v2 gate (`b1f4149`) used fresh target-uniform training data and
the validation-posterior-loss checkpoint. Coverage passed at 0.0115, and on
2000 paired BLS candidates TransitFlow achieved AUC/AP 0.914/0.929 versus
BLS 0.615/0.597 and TLS 0.656/0.652; all paired gain interval lower bounds were
positive. Characterization still rejected SBC uniformity: `RpRs` 2.1e-19,
`aRs` 1.2e-19, `b` 2.4e-8, `q1` 6.1e-4, and `q2` 0.0376. Rank histograms show
directional bias, not only interval-width error. The fast real-data/MCMC sizes
(12 planets, 4 chains, <=2000 steps) are diagnostic and intentionally cannot
pass the full sample/convergence gates. Therefore the full run remains blocked,
and none of these diagnostics may be promoted to a publication claim.

## Historical pre-full-run evidence (superseded)

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
11. Added paired TransitFlow-minus-TLS confidence intervals and explicit AUC/AP
    gates on the identical light curves.
12. Made the publishable speed gate conditional on a converged MCMC reference;
    the raw timing ratio remains a labelled diagnostic.
13. Recorded BLS/TLS search failures and success masks instead of silently
    making zero-score fallbacks indistinguishable from successful searches.
14. Required the complete 30-object real sample and complete tau/ESS diagnostics
    for every MCMC comparison.
15. Added exact dataset-provenance validation, blocked evaluation after
    interrupted/non-finite training, preserved attempt logs, and stopped
    zero-work resumes from overwriting completed-training provenance.

## Mandatory experiments before MNRAS submission

These are blocking, in order.

1. **Freeze and archive the executed source.** Commit the current fixes; record
   the patch hash, full command, environment lock, CUDA/cuDNN versions, data and
   checkpoint SHA-256 values, and exact MAST product identifiers.
2. **Complete noise-product provenance.** The new library is split by target
   star with no train/evaluation overlap. Preserve exact target, sector, cadence,
   author, and MAST product identifiers so the source curves are independently
   reconstructable; add sector-level separation where multiple sectors exist.
3. **Retrain 3--5 independent seeds.** Seed 0 is a recorded failure. Report every
   seed-level gate, mean, standard
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
| Reproducible code/data supporting every conclusion | Partial; 116 segments have target IDs and a disjoint target split, but exact sector/product provenance is incomplete |
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
