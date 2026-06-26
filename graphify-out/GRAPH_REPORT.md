# Graph Report - TransitFlow  (2026-06-26)

## Corpus Check
- 83 files · ~49,982 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 616 nodes · 1502 edges · 31 communities (23 shown, 8 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2f16acde`
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
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 58 edges
2. `TransitSimulator` - 51 edges
3. `SimConfig` - 41 edges
4. `train()` - 38 edges
5. `TransitFlowInference` - 36 edges
6. `TransitFlow` - 36 edges
7. `ModelConfig` - 33 edges
8. `preflight()` - 26 edges
9. `transit_flux()` - 26 edges
10. `TrainConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `test_detection_metrics_perfect()` --calls--> `detection_metrics()`  [EXTRACTED]
  tests/test_evaluation.py → transitflow/evaluation/detection.py
- `test_noise_library_roundtrip()` --calls--> `NoiseLibrary`  [EXTRACTED]
  tests/test_noise.py → transitflow/noise.py
- `build_configs()` --calls--> `ModelConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `TrainConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/train.py

## Import Cycles
- None detected.

## Communities (31 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (50): device, Module, main(), main(), main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg() (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (21): test_global_view_shape_and_finite(), test_local_view_centers_transit(), test_make_views_dtypes(), test_normalize_view(), _bin_statistic(), box_periodogram(), _fill_nans(), global_view() (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (20): test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals(), test_noise_library_roundtrip(), test_white_noise_std(), _autocovariance(), eclipsing_binary_signal(), estimate_white_sigma() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (21): bls_detect(), _bls_native(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee(), _log_likelihood() (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (32): completeness_grid(), Detection metrics and injection-recovery completeness grids., Recovery completeness of positives as a function of one feature.      For the ro, Evaluation: SBC, coverage, detection metrics, posterior agreement., jensen_shannon_1d(), marginal_wasserstein(), negative_log_prob_true(), posterior_contraction() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.38
Nodes (3): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Prior specification for a single parameter.      Parameters     ----------     n

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (28): build_views(), download_lc(), _flatten_lc(), main(), query_planets(), Download + clean one TESS single-sector PDCSAP light curve.      Returns (times_, Re-derive the transit epoch from the *data* by a box-search at fixed P.      The, Remove slow secular trends, returning flux ≈ 1 around a flat baseline.      The (+20 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (15): ConstantVelocity, _log_normal(), A constant velocity field v(tau, theta, e) = c. param_dim attached., v=0 -> samples ~ N(0,I) and log_prob = log N(theta)., v=c -> flow maps theta0 -> theta0 + c; density shifts accordingly., test_constant_velocity_shifts_density(), test_zero_velocity_is_standard_normal(), log_prob() (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (8): Downloaded evidence, Fixes made, Next run rule, TransitFlow char5 gate audit - 2026-06-26, Vast smoke after conditional-MCMC fix, Verdict, What failed, What passed

### Community 10 - "Community 10"
Cohesion: 0.23
Nodes (18): _df_line(), _du(), _fmt_int(), _gate_bool(), _gpu_line(), main(), _pid_state(), _process_tree() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.04
Nodes (43): Calibration is the product, Compute, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start, Real data (+35 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (61): has_astropy(), central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., detection_metrics(), ROC-AUC, average precision, and curve arrays. (+53 more)

### Community 15 - "Community 15"
Cohesion: 0.05
Nodes (50): DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Fuse global + local CNN branches (+ optional noise feature) into ``e``., _CondResidualBlock, DetectionHead, FlowMatchingHead, Prediction heads: detection classifier + flow-matching velocity field., Sinusoidal embedding of the flow time ``tau in [0, 1]``. (+42 more)

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (17): test_depth_scales_with_radius_ratio(), test_duration_physical(), test_native_matches_batman(), test_out_of_transit_is_unity(), test_secondary_eclipse_flat(), test_vectorized_matches_loop(), Exact ``log q(theta | x)`` in standardized space., Approximate IS efficiency as a misspecification flag.          Uses a Gaussian l (+9 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (3): Posterior detection probability = P(depth dim above threshold).          ``sampl, Tensor, Return (log_mask, u_mean, u_std, u_low, u_high) as tensors.

### Community 20 - "Community 20"
Cohesion: 0.23
Nodes (4): ndarray, Draw ``n`` parameter vectors from the prior. Returns ``(n, 7)``., Log prior density in physical space; ``-inf`` outside support., Log prior density in standardized space; ``-inf`` outside support.          In `

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.22
Nodes (9): Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., The CFM target is theta1 - theta0; a field returning it has ~0 loss., test_cfm_loss_masks_invalid_rows(), test_cfm_loss_optimum_is_displacement(), cfm_loss(), _exact_divergence(), Return (velocity, divergence) with an exact trace of dv/dy.      Exact trace cos, Optimal-transport conditional-flow-matching loss (mean over valid rows).      `` (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.28
Nodes (4): _CouplingLayer, Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (3): test_kipping_validity(), quadratic_to_kipping(), Inverse of :func:`kipping_to_quadratic`.

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (4): CNNBranch, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., ResidualBlock1D

## Knowledge Gaps
- **69 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+64 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 14` to `Community 3`, `Community 5`, `Community 6`, `Community 15`, `Community 19`, `Community 20`, `Community 24`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 0`, `Community 4`, `Community 6`, `Community 14`, `Community 19`, `Community 22`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 14` to `Community 0`, `Community 2`, `Community 6`, `Community 15`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Read a numeric archive cell as a plain float, robustly.      NASA archive column` to the rest of the system?**
  _208 weakly-connected nodes found - possible documentation gaps or missing edges._