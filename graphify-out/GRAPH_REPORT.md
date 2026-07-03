# Graph Report - TransitFlow  (2026-07-04)

## Corpus Check
- 76 files · ~44,350 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 657 nodes · 1671 edges · 20 communities (14 shown, 6 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cfe2b382`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 41|Community 41]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 64 edges
2. `TransitSimulator` - 58 edges
3. `SimConfig` - 47 edges
4. `train()` - 38 edges
5. `TransitFlowInference` - 36 edges
6. `TransitFlow` - 36 edges
7. `ModelConfig` - 33 edges
8. `transit_flux()` - 30 edges
9. `preflight()` - 26 edges
10. `TrainConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `prior()` --calls--> `TransitPrior`  [EXTRACTED]
  tests/conftest.py → transitflow/priors.py
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `test_num_parameters()` --calls--> `TransitFlow`  [EXTRACTED]
  tests/test_models.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `SimConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/simulator.py
- `main()` --calls--> `detection_metrics()`  [EXTRACTED]
  scripts/baseline_detection.py → transitflow/evaluation/detection.py

## Import Cycles
- None detected.

## Communities (20 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (63): ModelConfig, Module, build_configs(), Helpers to turn a YAML config into the project's dataclasses., main(), main(), main(), _model_cfg() (+55 more)

### Community 1 - "Community 1"
Cohesion: 0.21
Nodes (16): Tests for the importance-sampling posterior correction., On a high-SNR object the IS weights concentrate near the truth., _setup(), test_importance_weights_accepts_periodogram_model(), test_importance_weights_and_correction(), test_importance_weights_recover_true_posterior_synthetic(), test_render_raw_flux_dilution_attenuates_depth(), test_render_raw_flux_matches_simulator_shape() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (47): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., _expand_free() (+39 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (39): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics(), Detection metrics and injection-recovery completeness grids. (+31 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (36): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), test_noise_library_roundtrip(), Tests for the box-periodogram channel (the period-calibration fix)., A clean, deep transit produces periodogram power near the true period     (or a (+28 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (52): ndarray, _collect_target_segments(), main(), _label(), Coverage/width diagnostics split by shape and information regime., stratified_characterization_diagnostics(), test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend() (+44 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (28): device, Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Posterior detection probability = P(depth dim above threshold).          ``sampl, Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig (+20 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (50): Path, Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p, sbc_gate(), main(), _worker(), _write_result(), build_gate_report(), _gate_value() (+42 more)

### Community 11 - "Community 11"
Cohesion: 0.23
Nodes (18): _df_line(), _du(), _fmt_int(), _gate_bool(), _gpu_line(), main(), _pid_state(), _process_tree() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.04
Nodes (44): Calibration is the product, Compute, Current gate baseline, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start (+36 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (35): CNNBranch, DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., Fuse global + local CNN branches (+ optional noise feature) into ``e``., ResidualBlock1D, _CondResidualBlock (+27 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (25): Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., ConstantVelocity, _log_normal(), A constant velocity field v(tau, theta, e) = c. param_dim attached., v=0 -> samples ~ N(0,I) and log_prob = log N(theta)., v=c -> flow maps theta0 -> theta0 + c; density shifts accordingly., The CFM target is theta1 - theta0; a field returning it has ~0 loss., test_cfm_loss_masks_invalid_rows() (+17 more)

### Community 30 - "Community 30"
Cohesion: 0.15
Nodes (13): main(), _inference(), test_detect_returns_probabilities(), test_ephemeris_conditioned_inference(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_log_prob_slices_characterization_target(), test_posterior_samples_shape_and_range() (+5 more)

### Community 31 - "Community 31"
Cohesion: 0.09
Nodes (11): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Build a prior whose *density* matches a simulator ``SimConfig``.          The fo, Draw ``n`` parameter vectors from the prior. Returns ``(n, 7)``., Return (log_mask, u_mean, u_std, u_low, u_high) as tensors., Log density of a/Rs given P under the stellar-density prior.          Inverts a/, Log prior density in physical space; ``-inf`` outside support., Log prior density in standardized space; ``-inf`` outside support.          In ` (+3 more)

## Knowledge Gaps
- **43 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 31` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 15`, `Community 30`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 6` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 7`, `Community 8`, `Community 10`, `Community 30`, `Community 31`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 14` to `Community 0`, `Community 1`, `Community 6`, `Community 8`, `Community 15`, `Community 30`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _197 weakly-connected nodes found - possible documentation gaps or missing edges._