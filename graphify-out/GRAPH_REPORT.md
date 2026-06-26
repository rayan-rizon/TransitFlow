# Graph Report - TransitFlow  (2026-06-26)

## Corpus Check
- 97 files · ~71,154 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 642 nodes · 1548 edges · 37 communities (29 shown, 8 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f1ffdc2b`
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
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]

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
- `prior()` --calls--> `TransitPrior`  [EXTRACTED]
  tests/conftest.py → transitflow/priors.py
- `fast_simulator()` --calls--> `TransitSimulator`  [EXTRACTED]
  tests/conftest.py → transitflow/simulator.py
- `tiny_model_cfg()` --calls--> `ModelConfig`  [EXTRACTED]
  tests/conftest.py → transitflow/models/transitflow.py
- `test_detection_metrics_perfect()` --calls--> `detection_metrics()`  [EXTRACTED]
  tests/test_evaluation.py → transitflow/evaluation/detection.py
- `test_noise_library_roundtrip()` --calls--> `NoiseLibrary`  [EXTRACTED]
  tests/test_noise.py → transitflow/noise.py

## Import Cycles
- None detected.

## Communities (37 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (29): Module, main(), main(), Tests for production run management: run dir, checkpoints, resume, status., test_resume_continues_from_checkpoint(), test_run_dir_artifacts_and_checkpoints(), _tiny_cfgs(), TransitFlow (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.19
Nodes (17): test_global_view_shape_and_finite(), test_local_view_centers_transit(), test_make_views_dtypes(), test_normalize_view(), _bin_statistic(), _fill_nans(), global_view(), local_view() (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (21): test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals(), test_noise_library_roundtrip(), test_white_noise_std(), _autocovariance(), eclipsing_binary_signal(), estimate_white_sigma() (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (58): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee() (+50 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (30): completeness_grid(), Detection metrics and injection-recovery completeness grids., Recovery completeness of positives as a function of one feature.      For the ro, Evaluation: SBC, coverage, detection metrics, posterior agreement., jensen_shannon_1d(), marginal_wasserstein(), negative_log_prob_true(), posterior_contraction() (+22 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (19): main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg(), test_generate_and_load_disk_dataset(), test_preflight_flags_device_mismatch(), test_preflight_verdict_and_cost(), test_resumable_generation_skips_existing() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (31): Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p, sbc_gate(), build_views(), download_lc(), _flatten_lc(), fold_bin_fixed_ephemeris(), main(), passes_real_quality() (+23 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (38): Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig, train_spike_slab(), Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``. (+30 more)

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
Cohesion: 0.12
Nodes (20): detection_metrics(), ROC-AUC, average precision, and curve arrays., ModelConfig, A short training run reduces loss and learns better-than-chance detection., test_short_training_runs_and_learns(), Tests for the box-periodogram channel (the period-calibration fix)., A clean, deep transit produces periodogram power near the true period     (or a, A periodogram-enabled model errors if the channel is missing. (+12 more)

### Community 15 - "Community 15"
Cohesion: 0.24
Nodes (7): Shared-embedding joint detection + characterization model., TransitFlow, _batch_t(), test_embedding_and_heads_shapes(), test_fmpe_loss_backward(), test_npe_head_loss_backward(), test_num_parameters()

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (9): DualBranchEmbedding, Fuse global + local CNN branches (+ optional noise feature) into ``e``., DetectionHead, FlowMatchingHead, Prediction heads: detection classifier + flow-matching velocity field., 2-layer MLP on the shared embedding -> detection logit ``p(d=1 | x)``., Velocity field ``v_psi(tau, theta_tau | e)`` for the parameter-space CNF.      P, Neural network components for TransitFlow. (+1 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (8): _CouplingLayer, Neural Posterior Estimation head (Variant B baseline).  A conditional neural spl, Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP, Posterior detection probability = P(depth dim above threshold).          ``sampl, Tensor, Return (log_mask, u_mean, u_std, u_low, u_high) as tensors.

### Community 20 - "Community 20"
Cohesion: 0.21
Nodes (5): device, PrefetchSimulator, Background multiprocess simulator feeding a bounded queue.      Falls back to a, Infinite iterator of on-the-fly simulated batches (no disk storage)., SimulatorIterator

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.23
Nodes (10): build_configs(), Helpers to turn a YAML config into the project's dataclasses., load_config(), merge_into_dataclass(), _mp_worker(), Shared utilities: config loading, seeding, device selection, data iteration., Worker process: build a simulator and stream batches onto the queue., Load a YAML config into a plain dict. (+2 more)

### Community 23 - "Community 23"
Cohesion: 0.24
Nodes (4): _CondResidualBlock, Sinusoidal embedding of the flow time ``tau in [0, 1]``., Residual MLP block with additive (time + context) conditioning., SinusoidalTimeEmbedding

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (7): NPEHead, Return ``(B, n, param_dim)`` posterior samples., Conditional normalizing flow posterior head ``q(theta | e)``., A trained RealNVP should assign higher density to in-distribution points., test_realnvp_density_normalizes_roughly(), test_realnvp_fallback_logprob_and_sample(), test_zuko_backend_if_available()

### Community 30 - "Community 30"
Cohesion: 0.24
Nodes (5): CNNBranch, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., ResidualBlock1D

### Community 31 - "Community 31"
Cohesion: 0.15
Nodes (7): ndarray, Draw ``B`` real OOT segments of length ``n`` (random start offsets)., Draw ``n`` parameter vectors from the prior. Returns ``(n, 7)``., Log prior density in physical space; ``-inf`` outside support., Log prior density in standardized space; ``-inf`` outside support.          In `, box_periodogram(), Vectorized box (BLS-lite) periodogram over a trial-period grid.      For each tr

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (15): test_depth_scales_with_radius_ratio(), test_duration_physical(), test_native_matches_batman(), test_out_of_transit_is_unity(), test_secondary_eclipse_flat(), test_vectorized_matches_loop(), has_batman(), _occulted_fraction() (+7 more)

### Community 33 - "Community 33"
Cohesion: 0.21
Nodes (13): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., main(), main(), _write_json() (+5 more)

### Community 34 - "Community 34"
Cohesion: 0.38
Nodes (3): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Prior specification for a single parameter.      Parameters     ----------     n

### Community 35 - "Community 35"
Cohesion: 0.25
Nodes (7): Evidence files, Failed checkpoint, Real MCMC, Real validation, Synthetic latest checkpoint, Vast publishable-v2 run summary - 2026-06-26, Verdict

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (5): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg()

## Knowledge Gaps
- **75 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+70 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 3` to `Community 33`, `Community 34`, `Community 2`, `Community 36`, `Community 5`, `Community 6`, `Community 8`, `Community 14`, `Community 19`, `Community 31`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 0`, `Community 33`, `Community 3`, `Community 4`, `Community 5`, `Community 8`, `Community 14`, `Community 17`, `Community 24`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 14` to `Community 0`, `Community 33`, `Community 2`, `Community 3`, `Community 36`, `Community 5`, `Community 6`, `Community 8`, `Community 20`, `Community 22`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _218 weakly-connected nodes found - possible documentation gaps or missing edges._