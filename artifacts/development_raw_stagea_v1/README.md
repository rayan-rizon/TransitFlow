# Candidate-independent Stage-A development audit

This archive preserves the bounded development-only test run on 2026-07-17.
It is not a publication lockbox and cannot support a publication performance
claim.

The Stage-A model consumed only chronological global flux, a candidate-free
256-bin periodogram, and a noise-scale feature. It had no folded local view,
candidate ephemeris, candidate augmentation, or candidate-based flattening.
Training used 8,000 injections from 72 residual targets; validation used 3,000
injections from 29 disjoint targets.

The validation-selected checkpoint at step 500 reached ROC-AUC 0.9007 (required
0.99). At approximately 1% empirical false-positive rate, completeness was
0.3945; 17 of 29 source targets were below 50% completeness. The architecture
is therefore rejected as the publication detector, and a full pipeline must
remain blocked.

The report, model/data provenance, status, and immutable dataset metadata are
retained here. The large development shards remain only on the active instance
for short-term debugging and are not publication artifacts.
