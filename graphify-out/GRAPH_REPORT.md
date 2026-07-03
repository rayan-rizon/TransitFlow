# Graph Report - TransitFlow  (2026-07-03)

## Corpus Check
- 79 files · ~49,226 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 692 nodes · 1702 edges · 37 communities (31 shown, 6 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b7a0de07`
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
- [[_COMMUNITY_Community 11|Community 11]]
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
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 63 edges
2. `TransitSimulator` - 58 edges
3. `SimConfig` - 47 edges
4. `train()` - 38 edges
5. `TransitFlowInference` - 36 edges
6. `TransitFlow` - 36 edges
7. `ModelConfig` - 33 edges
8. `transit_flux()` - 29 edges
9. `preflight()` - 26 edges
10. `TrainConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `prior()` --calls--> `TransitPrior`  [EXTRACTED]
  tests/conftest.py → transitflow/priors.py
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `ModelConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `SimConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/simulator.py
- `build_configs()` --calls--> `TrainConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/train.py

## Import Cycles
- None detected.

## Communities (37 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (27): ModelConfig, A short training run reduces loss and learns better-than-chance detection., test_short_training_runs_and_learns(), Tests for production run management: run dir, checkpoints, resume, status., test_resume_continues_from_checkpoint(), test_run_dir_artifacts_and_checkpoints(), _tiny_cfgs(), TransitFlow (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (26): main(), Tests for the importance-sampling posterior correction., On a high-SNR object the IS weights concentrate near the truth., _setup(), test_adaptive_importance_weights_records_attempts(), test_importance_weights_accepts_periodogram_model(), test_importance_weights_and_correction(), test_importance_weights_recover_true_posterior_synthetic() (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (39): test_depth_scales_with_radius_ratio(), test_duration_physical(), test_exposure_integration_one_subsample_matches_instantaneous(), test_native_matches_batman(), test_out_of_transit_is_unity(), test_secondary_eclipse_flat(), test_vectorized_matches_loop(), test_global_view_shape_and_finite() (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (42): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee() (+34 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (44): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics(), Detection metrics and injection-recovery completeness grids. (+36 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (21): main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg(), test_generate_and_load_disk_dataset(), test_preflight_flags_device_mismatch(), test_preflight_verdict_and_cost(), test_resumable_generation_skips_existing() (+13 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (14): Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig, train_spike_slab(), main() (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (50): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals() (+42 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (17): ConstantVelocity, _log_normal(), A constant velocity field v(tau, theta, e) = c. param_dim attached., v=0 -> samples ~ N(0,I) and log_prob = log N(theta)., v=c -> flow maps theta0 -> theta0 + c; density shifts accordingly., test_constant_velocity_shifts_density(), test_zero_velocity_is_standard_normal(), log_prob() (+9 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (8): Downloaded evidence, Fixes made, Next run rule, TransitFlow char5 gate audit - 2026-06-26, Vast smoke after conditional-MCMC fix, Verdict, What failed, What passed

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (50): Path, Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p, sbc_gate(), main(), _worker(), _write_result(), build_gate_report(), _gate_value() (+42 more)

### Community 11 - "Community 11"
Cohesion: 0.23
Nodes (18): _df_line(), _du(), _fmt_int(), _gate_bool(), _gpu_line(), main(), _pid_state(), _process_tree() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (13): Calibration is the product, Compute, Current gate baseline, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.24
Nodes (4): _CondResidualBlock, Sinusoidal embedding of the flow time ``tau in [0, 1]``., Residual MLP block with additive (time + context) conditioning., SinusoidalTimeEmbedding

### Community 15 - "Community 15"
Cohesion: 0.28
Nodes (4): CNNBranch, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., ResidualBlock1D

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (8): Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., The CFM target is theta1 - theta0; a field returning it has ~0 loss., test_cfm_loss_masks_invalid_rows(), test_cfm_loss_optimum_is_displacement(), cfm_loss(), _exact_divergence(), Return (velocity, divergence) with an exact trace of dv/dy.      Exact trace cos, Optimal-transport conditional-flow-matching loss (mean over valid rows).      ``

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (7): _CouplingLayer, Return ``(B, n, param_dim)`` posterior samples., Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP, Posterior detection probability = P(depth dim above threshold).          ``sampl, Tensor

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.28
Nodes (7): Module, main(), main(), preflight(), _preflight_report(), Short pre-run check so a misconfigured run never wastes GPU hours.      Runs ~``, count_parameters()

### Community 20 - "Community 20"
Cohesion: 0.24
Nodes (11): Shared-embedding joint detection + characterization model., TransitFlow, _batch_t(), test_embedding_and_heads_shapes(), test_fmpe_loss_backward(), test_npe_head_loss_backward(), test_num_parameters(), A periodogram-enabled model errors if the channel is missing. (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.14
Nodes (17): DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Fuse global + local CNN branches (+ optional noise feature) into ``e``., DetectionHead, FlowMatchingHead, Prediction heads: detection classifier + flow-matching velocity field., 2-layer MLP on the shared embedding -> detection logit ``p(d=1 | x)``., Velocity field ``v_psi(tau, theta_tau | e)`` for the parameter-space CNF.      P (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.17
Nodes (7): device, batch_to_torch(), PrefetchSimulator, Background multiprocess simulator feeding a bounded queue.      Falls back to a, Move the simulator's numpy batch onto a device as tensors.      On CUDA the host, Infinite iterator of on-the-fly simulated batches (no disk storage)., SimulatorIterator

### Community 24 - "Community 24"
Cohesion: 0.29
Nodes (8): build_configs(), Helpers to turn a YAML config into the project's dataclasses., load_config(), merge_into_dataclass(), Shared utilities: config loading, seeding, device selection, data iteration., Load a YAML config into a plain dict., Return a copy of dataclass ``dc`` with keys from ``overrides`` applied., set_seed()

### Community 30 - "Community 30"
Cohesion: 0.42
Nodes (8): _inference(), test_detect_returns_probabilities(), test_ephemeris_conditioned_inference(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_log_prob_slices_characterization_target(), test_posterior_samples_shape_and_range(), test_sbc_uses_characterization_dims_for_5d_ephemeris_model()

### Community 31 - "Community 31"
Cohesion: 0.29
Nodes (4): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Build a prior whose *density* matches a simulator ``SimConfig``.          The fo, Prior specification for a single parameter.      Parameters     ----------     n

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (12): 0. Honest novelty verdict (read this first), 10. Software stack, 11. Expected contributions (paper framing), 12. সারসংক্ষেপ (Bangla summary), 1. Problem formulation, 5. Datasets, 7. Compute budget & feasibility (single RTX 4090), 8. Suggested timeline (~6 weeks, part-time-friendly) (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (6): 2.1 Transit model, 2.2 Noise model (three regimes, mixed per batch), 2.3 The "no-planet" (`d=0`) class, 2.4 Parameter priors (training distribution), 2.5 Light-curve representation (fixed-length, 4090-tractable), 2. Forward model / simulator (the heart of SBI)

### Community 36 - "Community 36"
Cohesion: 0.40
Nodes (5): 3.1 Embedding network `E(x)`, 3.2 Detection head `g_φ`, 3.3 Flow-matching characterization head, 3.4 Inference, 3. Model architecture

### Community 37 - "Community 37"
Cohesion: 0.50
Nodes (4): 4.1 Variant A — Factorized (robust primary; guarantees a result), 4.2 Variant B — NPE ablation (required baseline), 4.3 Variant C — Unified spike-and-slab (ambitious; stronger novelty), 4. Three method variants (run in this order)

### Community 38 - "Community 38"
Cohesion: 0.50
Nodes (4): 6.1 Baselines, 6.2 Metrics, 6.3 Ablations, 6. Experiments & evaluation

## Knowledge Gaps
- **69 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+64 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 3` to `Community 0`, `Community 1`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 31`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 7` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 10`, `Community 19`, `Community 23`, `Community 24`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 20` to `Community 0`, `Community 1`, `Community 6`, `Community 7`, `Community 16`, `Community 17`, `Community 19`, `Community 22`, `Community 30`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _224 weakly-connected nodes found - possible documentation gaps or missing edges._