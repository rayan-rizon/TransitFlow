# Held-out blind-BLS identifiability audit (development only)

This archive preserves a bounded, target-disjoint development diagnostic from
Vast on 2026-07-16. It is not a publication lockbox and must not be used to
claim final performance.

Protocol: 3,000 injections into 29 held-out TESS residual targets; Astropy BLS
provided the candidate ephemeris for both labels; the frozen V16 detector was
scored without retraining. Source identifiers are audit metadata only and were
not model inputs.

Result: ROC-AUC 0.9101 (required 0.99); BLS top-1 period recovery within 1% was
0.3503. The full publication runner therefore rejects a full allocation before
it consumes an external lockbox.

`identifiability_report.json` is the primary result. `dataset_meta.json` and
`identifiability_manifest.json` record the immutable data and candidate
protocol required to reproduce the diagnosis.
