# Bounded TLS proposal audit (development only)

This archive preserves the bounded TLS diagnostic run on 2026-07-17.  It is a
development artifact, not a publication lockbox, and it does not support a
publication performance claim.

The run used 64 newly simulated light curves with held-out real-noise residual
segments, eight CPU TLS workers, and the frozen V16 TransitFlow checkpoint.
The TLS wrapper was changed before the run to make a no-fit explicit rather
than silently treating it as a successful search.

TLS returned finite scores for every curve but produced 7 explicit no-fits
(10.94%).  Its ROC-AUC was 0.5824 and AP 0.5863.  Conditional on a positive
curve that produced a fit, period recovery within 1% was 0.5556 (15/27).  That
separates two facts: the physical template can locate some periods accurately,
but raw TLS SDE is not a reliable, cross-source detection statistic here.

The initial report writer used `NaN` calibration placeholders because BLS/TLS
scores are search statistics rather than probabilities.  Those placeholders
are normalized to JSON `null` here without changing any measured rank metric;
the code now produces the same standards-compliant form directly.

The required next experiment is not a full run.  It must calibrate a
target-conditioned false-alarm statistic from out-of-transit residuals, retain
multiple BLS/TLS hypotheses per curve, and train/evaluate the vetter on that
same candidate distribution using source-disjoint development targets.  Only a
predeclared source-stratified gate on a new lockbox can justify a full pipeline.
