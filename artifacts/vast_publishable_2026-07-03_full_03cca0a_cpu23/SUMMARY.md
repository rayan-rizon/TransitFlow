# TransitFlow Full Gate Run - 2026-07-03

Run: `full_03cca0a_cpu23_20260703_072020`

Verdict: `final_pass=false`

Passed:
- Synthetic detection: ROC-AUC `0.9993727744415907`
- Synthetic characterization SBC: familywise gate passed, min p-value `0.18710677791497005`
- Synthetic coverage calibration error: `0.01292631578947366`
- Real quality-gated detection: `29/30`, detected fraction `0.9666666666666667`
- Speed: `1717.1x`
- BLS/TLS baseline regenerated; TransitFlow ROC-AUC `0.9989761757415343`

Failed:
- Real MCMC characterization prior fraction: `b=0.11587869223356372`, threshold `<=0.1`
- Importance correction ESS: mean `0.009602028024468426`, median `0.005476108657252312`, min `0.0003335721650457704`, threshold `>=0.05`

Paper claim boundary:
- Supported: calibrated synthetic SBI, strong detection, strong speedup, BLS/TLS comparison, and quality-gated real detection.
- Not supported yet: full real-light-curve posterior characterization versus MCMC.

Canonical files:
- `gate_report.json`
- `results/synthetic/metrics.json`
- `results/real/real_validation.json`
- `results/speed.json`
- `results/bls_vs_transitflow.json`
