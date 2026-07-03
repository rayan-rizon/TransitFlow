# Graph Report - TransitFlow  (2026-07-03)

## Corpus Check
- 111 files · ~87,154 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 694 nodes · 1684 edges · 36 communities (29 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f97186d3`
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
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `TransitPrior` - 62 edges
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

## Communities (36 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (35): Module, main(), main(), Tests for production run management: run dir, checkpoints, resume, status., test_resume_continues_from_checkpoint(), test_run_dir_artifacts_and_checkpoints(), _tiny_cfgs(), TransitFlow (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (30): main(), download_lc(), main(), print() that can never crash the caller.      A prior run showed sys.stdout can, Download + clean one TESS single-sector PDCSAP light curve.      Returns (times_, Data-only quality checks for real-light-curve validation.      These cuts avoid, real_quality_metrics(), safe_print() (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (77): ndarray, _bin_label(), build_views(), _flatten_lc(), mcmc_stratified_summary(), query_planets(), Re-derive the transit epoch from the *data* by a box-search at fixed P.      The, Remove slow secular trends, returning flux ≈ 1 around a flat baseline.      The (+69 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (36): bls_detect(), _bls_native(), has_astropy(), Box Least Squares detection baseline (Sec. 6.1)., Run BLS and return the peak power (detection score) and best period., Minimal pure-numpy BLS fallback (peak depth-significance over the grid)., Baselines: BLS detection and transit-fit MCMC posteriors., has_emcee() (+28 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (43): central_interval_coverage(), coverage_calibration_error(), Expected coverage probability of posterior credible intervals., Empirical coverage of central credible intervals vs nominal level.      For each, Mean absolute deviation of empirical from nominal coverage (lower better)., completeness_grid(), detection_metrics(), Detection metrics and injection-recovery completeness grids. (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.20
Nodes (17): main(), _model_cfg(), Tests for the disk dataset pipeline and the preflight cost/health check., _sim_cfg(), test_generate_and_load_disk_dataset(), test_preflight_flags_device_mismatch(), test_preflight_verdict_and_cost(), test_resumable_generation_skips_existing() (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (25): Path, Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p, sbc_gate(), build_gate_report(), _gate_value(), git_sha(), main(), read_json() (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.42
Nodes (8): _inference(), test_detect_returns_probabilities(), test_ephemeris_conditioned_inference(), test_importance_diagnostic_runs(), test_log_prob_finite(), test_log_prob_slices_characterization_target(), test_posterior_samples_shape_and_range(), test_sbc_uses_characterization_dims_for_5d_ephemeris_model()

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (48): _CouplingLayer, NPEHead, Neural Posterior Estimation head (Variant B baseline).  A conditional neural spl, Return ``(B, n, param_dim)`` posterior samples., Conditional affine coupling (RealNVP) with a fixed binary mask., Fallback conditional RealNVP over a standard-normal base., Conditional normalizing flow posterior head ``q(theta | e)``., _RealNVP (+40 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (8): Downloaded evidence, Fixes made, Next run rule, TransitFlow char5 gate audit - 2026-06-26, Vast smoke after conditional-MCMC fix, Verdict, What failed, What passed

### Community 10 - "Community 10"
Cohesion: 0.23
Nodes (18): _df_line(), _du(), _fmt_int(), _gate_bool(), _gpu_line(), main(), _pid_state(), _process_tree() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (13): Calibration is the product, Compute, Current gate baseline, How the code maps to the plan, Install, Layout, Parameterization choices (read before extending), Quick start (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (23): fast_sim_cfg(), fast_simulator(), prior(), A tiny, fast simulator configuration for unit tests., tiny_model_cfg(), A short training run reduces loss and learns better-than-chance detection., test_short_training_runs_and_learns(), test_noise_library_roundtrip() (+15 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (17): ModelConfig, The full TransitFlow model: shared embedding + detection + posterior head.  The, Shared-embedding joint detection + characterization model., TransitFlow, _batch_t(), test_embedding_and_heads_shapes(), test_fmpe_loss_backward(), test_npe_head_loss_backward() (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (17): CNNBranch, DualBranchEmbedding, Dual-branch 1-D CNN embedding network ``E(x) -> e``.  A ResNet-1D style global b, Two 3-wide conv layers + identity/projection skip, optional /2 downsample., Stack of residual blocks with progressive downsampling -> pooled vector., Fuse global + local CNN branches (+ optional noise feature) into ``e``., ResidualBlock1D, _CondResidualBlock (+9 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): Canonical artifacts (after cleanup), Corrected gate interpretation, Detection baseline vs BLS (Gate #5b), Final gate scorecard, Gate #3 (real planets): improved, fully characterized, not closed, Headline: real-noise training helped, but held-out real-noise SBC is not closed, MCMC posterior agreement (Gate #3 confirmation), Pipeline that produced this (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.19
Nodes (5): device, PrefetchSimulator, Background multiprocess simulator feeding a bounded queue.      Falls back to a, Infinite iterator of on-the-fly simulated batches (no disk storage)., SimulatorIterator

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (10): Artifact layout, Checkpoints (downloaded to `artifacts/checkpoints/`), Gate #1 — SBC uniformity (target: p > 0.05 all params), Gate #2 — Coverage calibration (target: ±2–3%), Gate #3 — Real-planet agreement (target: ≥20–30 KOIs/TOIs, coverage@68 ≥ 0.50), Gate #4 — Speed vs MCMC (target: ≥10³×), Gate #5 — FMPE vs NPE ablation, Gate Summary (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (9): build_configs(), Helpers to turn a YAML config into the project's dataclasses., load_config(), merge_into_dataclass(), _mp_worker(), Shared utilities: config loading, seeding, device selection, data iteration., Worker process: build a simulator and stream batches onto the queue., Load a YAML config into a plain dict. (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (7): _stellar_prior(), test_kipping_validity(), test_stellar_density_aRs_density_is_normalized(), test_stellar_density_only_changes_aRs(), test_stellar_density_prior_matches_simulator_sampling(), quadratic_to_kipping(), Inverse of :func:`kipping_to_quadratic`.

### Community 31 - "Community 31"
Cohesion: 0.29
Nodes (4): ParamSpec, Default prior ranges. ``regime`` selects the period upper bound., Build a prior whose *density* matches a simulator ``SimConfig``.          The fo, Prior specification for a single parameter.      Parameters     ----------     n

### Community 32 - "Community 32"
Cohesion: 0.19
Nodes (5): DiskDataset, DiskIterator, Loads sharded light-curve data; serves shuffled batches as torch tensors.      D, Infinite shuffled iterator over a :class:`DiskDataset` for training., TransitFlow: amortized flow-matching SBI for joint exoplanet transit detection a

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

- **Why does `TransitPrior` connect `Community 3` to `Community 32`, `Community 1`, `Community 2`, `Community 4`, `Community 8`, `Community 14`, `Community 15`, `Community 23`, `Community 31`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `TransitSimulator` connect `Community 14` to `Community 32`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 0`, `Community 8`, `Community 15`, `Community 20`, `Community 22`, `Community 23`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `TransitFlow` connect `Community 15` to `Community 32`, `Community 1`, `Community 0`, `Community 4`, `Community 7`, `Community 8`, `Community 17`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TransitPrior` (e.g. with `TransitFlowInference` and `SimConfig`) actually correct?**
  _`TransitPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TransitSimulator` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`TransitSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SimConfig` (e.g. with `DiskDataset` and `DiskIterator`) actually correct?**
  _`SimConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transitflow`, `Helpers to turn a YAML config into the project's dataclasses.`, `Multiple-comparison aware SBC gate.      A D-dimensional SBC report contains D p` to the rest of the system?**
  _230 weakly-connected nodes found - possible documentation gaps or missing edges._