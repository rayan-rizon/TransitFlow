# Graph Report - .  (2026-06-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 483 nodes · 1239 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `41e0e39d`
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

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 50 edges
2. `TransitSimulator` - 43 edges
3. `SimConfig` - 39 edges
4. `train()` - 38 edges
5. `TransitFlow` - 33 edges
6. `ModelConfig` - 32 edges
7. `TransitFlowInference` - 28 edges
8. `preflight()` - 26 edges
9. `transit_flux()` - 25 edges
10. `TrainConfig` - 24 edges

## Surprising Connections (you probably didn't know these)
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `main()` --calls--> `TransitFlowInference`  [EXTRACTED]
  scripts/benchmark_speed.py → transitflow/inference.py
- `main()` --calls--> `TransitSimulator`  [EXTRACTED]
  scripts/benchmark_speed.py → transitflow/simulator.py
- `main()` --calls--> `load_checkpoint()`  [EXTRACTED]
  scripts/benchmark_speed.py → transitflow/train.py
- `main()` --calls--> `sbc_uniformity()`  [EXTRACTED]
  scripts/diagnose_period.py → transitflow/evaluation/sbc.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (65): ModelConfig, The full TransitFlow model: shared embedding + detection + posterior head.  The, Module, build_configs(), Helpers to turn a YAML config into the project's dataclasses., main(), main(), main() (+57 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (31): CNNBranch, DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., Fuse global + local CNN branches (+ optional noise feature) into ``e``., ResidualBlock1D, _CondResidualBlock (+23 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (54): ndarray, test_correlated_noise_amplitude_and_correlation(), test_hard_negative_signals(), test_white_noise_std(), test_depth_scales_with_radius_ratio(), test_duration_physical(), test_native_matches_batman(), test_out_of_transit_is_unity() (+46 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (33): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee() (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (36): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics(), Detection metrics and injection-recovery completeness grids. (+28 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (20): device, main(), fast_sim_cfg(), fast_simulator(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), test_noise_library_roundtrip(), Clean high-SNR planets fold to a clearly negative local-view minimum. (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (24): main(), Tests for the importance-sampling posterior correction., On a high-SNR object the IS weights concentrate near the truth., _setup(), test_importance_weights_and_correction(), test_importance_weights_recover_true_posterior_synthetic(), test_render_raw_flux_matches_simulator_shape(), test_simulate_batch_return_raw() (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (27): Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig, train_spike_slab(), Shared-embedding joint detection + characterization model. (+19 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (24): Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., ConstantVelocity, _log_normal(), A constant velocity field v(tau, theta, e) = c. param_dim attached., v=0 -> samples ~ N(0,I) and log_prob = log N(theta)., v=c -> flow maps theta0 -> theta0 + c; density shifts accordingly., The CFM target is theta1 - theta0; a field returning it has ~0 loss., test_cfm_loss_masks_invalid_rows() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.21
Nodes (9): _inference(), test_detect_returns_probabilities(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_posterior_samples_shape_and_range(), Exact ``log q(theta | x)`` in standardized space., Approximate IS efficiency as a misspecification flag.          Uses a Gaussian l, Return physical posterior samples ``(B, n_samples, 7)``. (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (5): main(), _read_log(), _read_status(), render(), _sparkline()

## Knowledge Gaps
- **1 isolated node(s):** `transitflow`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 3` to `Community 0`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 7` to `Community 0`, `Community 1`, `Community 6`, `Community 8`, `Community 9`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 5` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `A tiny, fast simulator configuration for unit tests.` to the rest of the system?**
  _135 weakly-connected nodes found - possible documentation gaps or missing edges._