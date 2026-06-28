# Graph Report - TransitFlow  (2026-06-28)

## Corpus Check
- 98 files · ~74,990 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 677 nodes · 1649 edges · 39 communities (32 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ac8be329`
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
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 59 edges
2. `TransitSimulator` - 56 edges
3. `SimConfig` - 45 edges
4. `train()` - 38 edges
5. `TransitFlowInference` - 36 edges
6. `TransitFlow` - 36 edges
7. `ModelConfig` - 33 edges
8. `transit_flux()` - 29 edges
9. `preflight()` - 26 edges
10. `TrainConfig` - 25 edges

## Surprising Connections (you probably didn't know these)
- `test_detection_metrics_perfect()` --calls--> `detection_metrics()`  [EXTRACTED]
  tests/test_evaluation.py → transitflow/evaluation/detection.py
- `build_configs()` --calls--> `ModelConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/models/transitflow.py
- `build_configs()` --calls--> `SimConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/simulator.py
- `build_configs()` --calls--> `TrainConfig`  [EXTRACTED]
  scripts/_config.py → transitflow/train.py
- `main()` --calls--> `bls_detect()`  [EXTRACTED]
  scripts/baseline_detection.py → transitflow/baselines/bls.py

## Import Cycles
- None detected.

## Communities (39 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (27): Module, main(), main(), TransitFlow, evaluate(), _health(), history_tail(), _human_time() (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (58): has_emcee(), central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., detection_metrics(), ROC-AUC, average precision, and curve arrays. (+50 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (45): ndarray, test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals(), test_white_noise_std(), test_global_view_shape_and_finite(), test_make_views_dtypes(), test_normalize_view() (+37 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (44): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., _log_likelihood() (+36 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (32): completeness_grid(), Detection metrics and injection-recovery completeness grids., Recovery completeness of positives as a function of one feature.      For the ro, Evaluation: SBC, coverage, detection metrics, posterior agreement., jensen_shannon_1d(), marginal_wasserstein(), negative_log_prob_true(), posterior_contraction() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.20
Nodes (17): main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg(), test_generate_and_load_disk_dataset(), test_preflight_flags_device_mismatch(), test_preflight_verdict_and_cost(), test_resumable_generation_skips_existing() (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (34): Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p, sbc_gate(), _bin_label(), build_views(), download_lc(), _flatten_lc(), fold_bin_fixed_ephemeris(), main() (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (9): ModelConfig, tiny_model_cfg(), A short training run reduces loss and learns better-than-chance detection., test_short_training_runs_and_learns(), Tests for production run management: run dir, checkpoints, resume, status., test_resume_continues_from_checkpoint(), test_run_dir_artifacts_and_checkpoints(), _tiny_cfgs() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (24): Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``., ConstantVelocity, _log_normal(), A constant velocity field v(tau, theta, e) = c. param_dim attached., v=0 -> samples ~ N(0,I) and log_prob = log N(theta)., v=c -> flow maps theta0 -> theta0 + c; density shifts accordingly., The CFM target is theta1 - theta0; a field returning it has ~0 loss., test_cfm_loss_masks_invalid_rows() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (8): Downloaded evidence, Fixes made, Next run rule, TransitFlow char5 gate audit - 2026-06-26, Vast smoke after conditional-MCMC fix, Verdict, What failed, What passed

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (28): Path, _df_line(), _du(), _fmt_int(), _gate_bool(), _gpu_line(), main(), _pid_state() (+20 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (13): Calibration is the product, Compute, Current gate baseline, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (12): test_noise_library_roundtrip(), Clean high-SNR planets fold to a clearly negative local-view minimum., test_dilution_attenuates_transit_depth(), test_gap_masks_keep_views_finite(), test_physical_a_rs_mode_correlates_with_period(), test_planet_local_views_deeper_on_average(), test_real_noise_sigma_feature_uses_drawn_segment(), NoiseLibrary (+4 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (26): Variant C (experimental): unified spike-and-slab posterior.  A single posterior, Maps (theta_std, d) to spike-and-slab targets and reads detection back., Augmented training targets: spike the depth dim for non-planets., Train Variant C: one unified flow over all rows (no detection head, no mask)., SpikeSlabAdapter, SpikeSlabConfig, train_spike_slab(), Shared-embedding joint detection + characterization model. (+18 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (12): DualBranchEmbedding, Fuse global + local CNN branches (+ optional noise feature) into ``e``., _CondResidualBlock, DetectionHead, FlowMatchingHead, Prediction heads: detection classifier + flow-matching velocity field., Sinusoidal embedding of the flow time ``tau in [0, 1]``., 2-layer MLP on the shared embedding -> detection logit ``p(d=1 | x)``. (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (8): _CouplingLayer, Neural Posterior Estimation head (Variant B baseline).  A conditional neural spl, Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP, Posterior detection probability = P(depth dim above threshold).          ``sampl, Tensor, Return (log_mask, u_mean, u_std, u_low, u_high) as tensors.

### Community 20 - "Community 20"
Cohesion: 0.21
Nodes (5): device, PrefetchSimulator, Background multiprocess simulator feeding a bounded queue.      Falls back to a, Infinite iterator of on-the-fly simulated batches (no disk storage)., SimulatorIterator

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (7): build_configs(), Helpers to turn a YAML config into the project's dataclasses., load_config(), merge_into_dataclass(), Shared utilities: config loading, seeding, device selection, data iteration., Load a YAML config into a plain dict., Return a copy of dataclass ``dc`` with keys from ``overrides`` applied.

### Community 23 - "Community 23"
Cohesion: 0.22
Nodes (3): test_kipping_validity(), quadratic_to_kipping(), Inverse of :func:`kipping_to_quadratic`.

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (7): NPEHead, Return ``(B, n, param_dim)`` posterior samples., Conditional normalizing flow posterior head ``q(theta | e)``., A trained RealNVP should assign higher density to in-distribution points., test_realnvp_density_normalizes_roughly(), test_realnvp_fallback_logprob_and_sample(), test_zuko_backend_if_available()

### Community 30 - "Community 30"
Cohesion: 0.24
Nodes (5): CNNBranch, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., ResidualBlock1D

### Community 31 - "Community 31"
Cohesion: 0.38
Nodes (3): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Prior specification for a single parameter.      Parameters     ----------     n

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (6): The full TransitFlow model: shared embedding + detection + posterior head.  The, DiskDataset, DiskIterator, Loads sharded light-curve data; serves shuffled batches as torch tensors.      D, Infinite shuffled iterator over a :class:`DiskDataset` for training., TransitFlow: amortized flow-matching SBI for joint exoplanet transit detection a

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (12): 0. Honest novelty verdict (read this first), 10. Software stack, 11. Expected contributions (paper framing), 12. সারসংক্ষেপ (Bangla summary), 1. Problem formulation, 5. Datasets, 7. Compute budget & feasibility (single RTX 4090), 8. Suggested timeline (~6 weeks, part-time-friendly) (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (6): 2.1 Transit model, 2.2 Noise model (three regimes, mixed per batch), 2.3 The "no-planet" (`d=0`) class, 2.4 Parameter priors (training distribution), 2.5 Light-curve representation (fixed-length, 4090-tractable), 2. Forward model / simulator (the heart of SBI)

### Community 35 - "Community 35"
Cohesion: 0.25
Nodes (7): Evidence files, Failed checkpoint, Real MCMC, Real validation, Synthetic latest checkpoint, Vast publishable-v2 run summary - 2026-06-26, Verdict

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
- **76 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 1` to `Community 32`, `Community 2`, `Community 3`, `Community 6`, `Community 14`, `Community 15`, `Community 19`, `Community 23`, `Community 31`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 1` to `Community 32`, `Community 0`, `Community 2`, `Community 5`, `Community 6`, `Community 7`, `Community 14`, `Community 15`, `Community 20`, `Community 22`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 32`, `Community 1`, `Community 0`, `Community 4`, `Community 8`, `Community 17`, `Community 19`, `Community 24`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _226 weakly-connected nodes found - possible documentation gaps or missing edges._