# Graph Report - TransitFlow  (2026-06-26)

## Corpus Check
- 92 files · ~55,414 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 630 nodes · 1523 edges · 34 communities (26 shown, 8 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1269ce78`
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

## Communities (34 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (22): Module, main(), main(), TransitFlow, evaluate(), history_tail(), _human_time(), _lr_at() (+14 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (52): build_views(), download_lc(), _flatten_lc(), main(), query_planets(), Download + clean one TESS single-sector PDCSAP light curve.      Returns (times_, Re-derive the transit epoch from the *data* by a box-search at fixed P.      The, Remove slow secular trends, returning flux ≈ 1 around a flat baseline.      The (+44 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (25): test_correlated_noise_amplitude_and_correlation(), test_estimate_white_sigma_ignores_slow_trend(), test_hard_negative_signals(), test_noise_library_roundtrip(), test_white_noise_std(), test_real_noise_sigma_feature_uses_drawn_segment(), _autocovariance(), eclipsing_binary_signal() (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (39): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee() (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (41): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics(), Detection metrics and injection-recovery completeness grids. (+33 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (17): main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg(), test_generate_and_load_disk_dataset(), test_preflight_flags_device_mismatch(), test_preflight_verdict_and_cost(), test_resumable_generation_skips_existing() (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.19
Nodes (17): main(), Tests for the importance-sampling posterior correction., On a high-SNR object the IS weights concentrate near the truth., _setup(), test_importance_weights_and_correction(), test_importance_weights_recover_true_posterior_synthetic(), test_render_raw_flux_matches_simulator_shape(), test_simulate_batch_return_raw() (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (12): _inference(), test_detect_returns_probabilities(), test_ephemeris_conditioned_inference(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_log_prob_slices_characterization_target(), test_posterior_samples_shape_and_range(), test_sbc_uses_characterization_dims_for_5d_ephemeris_model() (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.08
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
Cohesion: 0.16
Nodes (15): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), A short training run reduces loss and learns better-than-chance detection., test_short_training_runs_and_learns(), A clean, deep transit produces periodogram power near the true period     (or a (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (18): Shared-embedding joint detection + characterization model., TransitFlow, test_importance_weights_accepts_periodogram_model(), _batch_t(), test_embedding_and_heads_shapes(), test_fmpe_loss_backward(), test_npe_head_loss_backward(), test_num_parameters() (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (11): DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Fuse global + local CNN branches (+ optional noise feature) into ``e``., DetectionHead, FlowMatchingHead, Prediction heads: detection classifier + flow-matching velocity field., 2-layer MLP on the shared embedding -> detection logit ``p(d=1 | x)``., Velocity field ``v_psi(tau, theta_tau | e)`` for the parameter-space CNF.      P (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (7): _CouplingLayer, Neural Posterior Estimation head (Variant B baseline).  A conditional neural spl, Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., _RealNVP, Posterior detection probability = P(depth dim above threshold).          ``sampl, Tensor

### Community 20 - "Community 20"
Cohesion: 0.19
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
Cohesion: 0.31
Nodes (7): NPEHead, Return ``(B, n, param_dim)`` posterior samples., Conditional normalizing flow posterior head ``q(theta | e)``., A trained RealNVP should assign higher density to in-distribution points., test_realnvp_density_normalizes_roughly(), test_realnvp_fallback_logprob_and_sample(), test_zuko_backend_if_available()

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (4): CNNBranch, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., ResidualBlock1D

### Community 31 - "Community 31"
Cohesion: 0.39
Nodes (7): Tests for production run management: run dir, checkpoints, resume, status., test_resume_continues_from_checkpoint(), test_run_dir_artifacts_and_checkpoints(), _tiny_cfgs(), _health(), Derive run-health warnings so a wasteful run is caught early., TrainConfig

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (5): 📊 Live TransitFlow Pipeline Monitor, 📝 Recent Pipeline Logs, SSH Source Command, ⚡ System Status, 🏃 Top Active Subprocesses

## Knowledge Gaps
- **72 isolated node(s):** `transitflow`, `graphify`, `Workflow: graphify`, `graphify`, `graphify` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransitPrior` connect `Community 3` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 14`, `Community 15`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 0`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 17`, `Community 19`, `Community 24`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 14` to `Community 32`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 0`, `Community 8`, `Community 15`, `Community 20`, `Community 22`, `Community 31`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _213 weakly-connected nodes found - possible documentation gaps or missing edges._