# Graph Report - TransitFlow  (2026-06-26)

## Corpus Check
- 80 files · ~48,232 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 600 nodes · 1465 edges · 23 communities (17 shown, 6 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `87798741`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 57 edges
2. `TransitSimulator` - 51 edges
3. `SimConfig` - 41 edges
4. `train()` - 38 edges
5. `TransitFlowInference` - 36 edges
6. `TransitFlow` - 36 edges
7. `ModelConfig` - 33 edges
8. `preflight()` - 26 edges
9. `TrainConfig` - 25 edges
10. `transit_flux()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `prior()` --calls--> `TransitPrior`  [EXTRACTED]
  tests/conftest.py → transitflow/priors.py
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `ModelConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/models/transitflow.py
- `main()` --calls--> `bls_detect()`  [EXTRACTED]
  scripts/baseline_detection.py → transitflow/baselines/bls.py
- `main()` --calls--> `TransitFlowInference`  [EXTRACTED]
  scripts/baseline_detection.py → transitflow/inference.py

## Import Cycles
- None detected.

## Communities (23 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (64): Module, build_configs(), Helpers to turn a YAML config into the project's dataclasses., main(), main(), main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check. (+56 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (18): device, Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Posterior detection probability = P(depth dim above threshold).          ``sampl, Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (50): ndarray, build_views(), download_lc(), _flatten_lc(), main(), query_planets(), Download + clean one TESS single-sector PDCSAP light curve.      Returns (times_, Re-derive the transit epoch from the *data* by a box-search at fixed P.      The (+42 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (27): bls_detect(), _bls_native(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee(), _log_likelihood() (+19 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (40): has_astropy(), central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics() (+32 more)

### Community 5 - "Community 5"
Cohesion: 0.38
Nodes (3): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Prior specification for a single parameter.      Parameters     ----------     n

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (21): main(), Tests for the importance-sampling posterior correction., On a high-SNR object the IS weights concentrate near the truth., _setup(), test_importance_weights_and_correction(), test_importance_weights_recover_true_posterior_synthetic(), test_render_raw_flux_matches_simulator_shape(), test_simulate_batch_return_raw() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (13): main(), _inference(), test_detect_returns_probabilities(), test_ephemeris_conditioned_inference(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_log_prob_slices_characterization_target(), test_posterior_samples_shape_and_range() (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (31): _CouplingLayer, Return ``(B, n, param_dim)`` posterior samples., Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP, Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., Tensor, ConstantVelocity (+23 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (7): Downloaded evidence, Fixes made, Next run rule, TransitFlow char5 gate audit - 2026-06-26, Verdict, What failed, What passed

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (5): main(), _read_log(), _read_status(), render(), _sparkline()

### Community 13 - "Community 13"
Cohesion: 0.04
Nodes (43): Calibration is the product, Compute, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start, Real data (+35 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (41): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals() (+33 more)

### Community 15 - "Community 15"
Cohesion: 0.06
Nodes (38): CNNBranch, DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., Fuse global + local CNN branches (+ optional noise feature) into ``e``., ResidualBlock1D, _CondResidualBlock (+30 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

## Knowledge Gaps
- **67 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+62 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 14`, `Community 15`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 0`, `Community 1`, `Community 4`, `Community 6`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 14` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 15`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Read a numeric archive cell as a plain float, robustly.      NASA archive column` to the rest of the system?**
  _207 weakly-connected nodes found - possible documentation gaps or missing edges._