# TransitFlow

**Amortized flow-matching simulation-based inference for joint exoplanet transit
detection and parameter posteriors.**

Given a single-sector / single-quarter light curve `x`, TransitFlow returns, in
one forward pass, a *joint* posterior

```
p(d, θ | x) = p(d | x) · p(θ | d = 1, x)
```

over **(i)** whether a transiting planet is present (`d ∈ {0,1}`) and **(ii)** its
physical transit parameters `θ = (P, t0, Rp/Rs, a/Rs, b, q1, q2)`, with
*calibrated* uncertainty — validated by simulation-based calibration (SBC),
expected-coverage tests, and an importance-sampling misspecification diagnostic —
at ~10³–10⁶× the speed of MCMC / nested sampling.

This repository is a complete, tested implementation of
[`TransitFlow_Implementation_Plan.md`](TransitFlow_Implementation_Plan.md).

---

## Why this is interesting (honest framing)

The contribution is a **novel application + methodological integration**, not a
new ML primitive (see §0 of the plan). The defensible core: first FMPE-SBI on
transit light curves delivering a *unified detection + characterization*
posterior with simulation-based calibration — uncertainty that detection CNNs and
point-estimate regressors structurally cannot provide. Flow-matching posterior
estimation itself is established for atmospheric retrieval (Vasist 2023; Gebhard
2023/2024) and direct-imaging orbits (Liang 2025); transit *detection* is heavily
studied (AstroNet, ExoMiner, BLS/TLS). TransitFlow combines and calibrates.

---

## Install

```bash
python -m pip install -e .            # core
python -m pip install -e ".[all]"     # + batman, zuko, torchdiffeq, astropy, …
```

Everything runs **without** the optional packages: the transit physics, flow
matching, ODE sampling, NPE flow, GP noise, BLS and MCMC baselines all have
self-contained implementations. When the reference packages are present they are
used as accelerated / ground-truth paths (e.g. `batman`, `zuko`, `torchdiffeq`,
`astropy.BoxLeastSquares`, `emcee`, `celerite2`).

## Quick start

```bash
# end-to-end sanity check (CPU/MPS-friendly, a few minutes)
python scripts/smoke_test.py --steps 600

# full training (Variant A / FMPE) — targets a single RTX 4090
python scripts/train.py --config configs/default.yaml

# NPE ablation (Variant B)
python scripts/train.py --config configs/default.yaml --head npe \
    --ckpt checkpoints/transitflow_npe.pt

# evaluation: SBC, coverage, detection, with figures
python scripts/evaluate.py --ckpt checkpoints/transitflow_fmpe.pt --plots
```

Library use:

```python
from transitflow import (TransitSimulator, SimConfig, TransitFlow, ModelConfig,
                         TransitFlowInference, TransitPrior, train, TrainConfig)

res = train(ModelConfig(), SimConfig(), TrainConfig(n_steps=60000))
inf = TransitFlowInference(res["model"], TransitPrior(), SimConfig())
out = inf.detect_and_characterize(global_view, local_view, sigma_feat, n_samples=5000)
# out["p_detect"], out["samples"]  (physical θ posterior)
```

---

## How the code maps to the plan

| Plan section | Module |
|---|---|
| §1 Problem / factorization | `models/transitflow.py` |
| §2.1 Transit model (batman) | `transit_model.py` — native vectorized engine, validated vs `batman` to <1e-3 |
| §2.2 Noise (real / GP / white) | `noise.py` (+ `NoiseLibrary` for real injection) |
| §2.3 Hard negatives (EB / systematics / sinusoids) | `noise.py`, wired in `simulator.py` |
| §2.4 Parameter priors | `priors.py` (`TransitPrior`, Kipping LD) |
| §2.5 Dual-view representation | `views.py` (global 2001 / local 201) |
| §2.5b Box-periodogram channel (period-calibration fix) | `views.py::box_periodogram` (256-bin trial-period spectrum) |
| §3.1 Embedding CNN | `models/embedding.py` (tri-branch ResNet-1D: global + local + periodogram) |
| §3.2 Detection head | `models/heads.py::DetectionHead` |
| §3.3 Flow-matching head + loss | `models/heads.py::FlowMatchingHead`, `flow_matching.py` |
| §3.4 Inference (ODE sample, IS correction) | `inference.py`, `flow_matching.py` |
| §4.1 Variant A (factorized) | default (`head="fmpe"`) |
| §4.2 Variant B (NPE ablation) | `models/npe.py` (`head="npe"`) |
| §4.3 Variant C (spike-and-slab) | `models/spike_slab.py` (experimental) |
| §6.1 Baselines (BLS, MCMC) | `baselines/bls.py`, `baselines/mcmc.py` |
| §6.2 Metrics (SBC, coverage, …) | `evaluation/` |

### Parameterization choices (read before extending)

* **Limb darkening** uses the Kipping (2013) `(q1, q2)` square, which maps
  bijectively to the *physically valid* quadratic `(u1, u2)` triangle — every
  prior draw is valid and the marginal priors are clean uniforms (good for SBC).
* **`t0`** is parameterized as the **orbital phase** `t0_phase ∈ [0,1)` of the
  first transit, an `P`-independent target.
* **Standardization.** Each parameter is mapped (log for `P, Rp/Rs, a/Rs`) and
  z-scored to ≈ unit variance; the flow transports `N(0,I)` to this standardized
  space, which matches the OT-CFM base/target scales. Transforms and their
  Jacobians live in `priors.py`.
* **Local-view folding.** The local view is folded on a *candidate* ephemeris —
  the true `(P,t0)` for planets, a random spurious candidate for non-planets —
  mirroring a real BLS/TLS pipeline that proposes candidates the network must
  accept or reject. At test time on real data, fold on the BLS ephemeris.
* **View-level synthesis.** Each raw light curve (`n_raw` cadences over the
  baseline) gets transit + correlated + white noise, then *both* views are binned
  from the same raw curve, so folding handles correlated noise correctly and the
  binning reduces white noise physically.
* **Periodogram channel (period fix).** The two binned views cannot carry a
  calibrated period (the global view coarsens transit timing below its bin width;
  the local view is folded on a candidate period, so period-blind). This made `P`
  the one parameter to fail SBC. The fix is a **third view** — a count-weighted box
  (BLS-lite) periodogram over a 256-bin log-spaced trial-period grid — feeding a
  third CNN branch, so the flow sees sharp, confidence-aware period information.
  Enabled by `use_periodogram: true` in both the `simulator` and `model` configs;
  computed on a 4096-pt subsample (`pg_n_raw`) for speed. See plan §2.5b / §9b.

---

## Calibration is the product

`flow_matching.py` implements the OT-CFM loss, the probability-flow ODE sampler,
and **exact** log-density via the continuous change of variables (the divergence
is traced exactly — cheap in 7-D). Correctness is pinned by unit tests:
`v ≡ 0` ⇒ samples are `N(0,I)` and `log p = log N(θ)`; `v ≡ c` ⇒ the density
shifts to `N(θ-c)`. The SBC and coverage estimators are likewise unit-tested to
flag over/under-confidence (`tests/test_evaluation.py`).

`evaluation/` provides SBC rank histograms + uniformity tests, expected coverage,
posterior NLL / contraction, and Wasserstein / Jensen–Shannon agreement with
MCMC posteriors. `inference.importance_diagnostic` reports IS efficiency (ESS/N)
as an approximate simulator-misspecification flag.

> SBC/coverage become flat/diagonal only for a *converged* posterior. The tiny
> `smoke.yaml` run exercises the whole pipeline but is under-trained; use
> `default.yaml` (≈ tens of thousands of steps) for science-quality calibration.

---

## Running on Vast.ai (or any remote GPU)

The default config (`configs/default.yaml`) is built for an unattended GPU run:
it writes everything to a **run directory**, checkpoints periodically so a
preempted instance resumes cleanly, and gates the launch behind a **preflight**
that refuses to spend GPU hours on a broken or wasteful config.

**Cost note (important):** the forward simulator is CPU-bound (~190 light
curves/s/core at the science config), while a 4090 consumes ~10k/s. Feeding the
GPU with on-the-fly simulation leaves it **~90% idle** — you'd pay for a mostly-idle
4090. The cost-efficient path is to **pre-generate the dataset once** (on a cheap
CPU box or locally — it's free) and **train from disk** so the GPU runs flat-out.

```bash
# 0. provision; on the box:
git clone <repo> && cd TransitFlow && python -m pip install -e ".[all]"
python -m pytest -q                         # 68 tests — confirm the box is healthy

# 1. PRE-GENERATE once (cheap CPU / local). Reused by FMPE + NPE runs.
#    Set --workers to the box's REAL cpu quota (cgroup cpu.max), not `nproc`.
python scripts/generate_data.py --config configs/default.yaml \
    --n 1000000 --workers 16 --out data/tess_1M         # ~4.4 GB fp16 views (+pg)
# DiskDataset loads every shard into RAM on init: .npz members can't be mmap'd
# (zip archives), so lazy access re-reads whole shards and starves the GPU ~98%.

# 2. PREFLIGHT — prints throughput, GPU-starvation %, ETA, $ cost, PASS/WARN/FAIL
python scripts/preflight.py --config configs/default.yaml \
    --expect cuda --data-dir data/tess_1M

# 3. TRAIN from disk in a detached session (survives SSH disconnects)
tmux new -s tf
python scripts/train.py --config configs/default.yaml --run-dir runs/fmpe \
    --data-dir data/tess_1M --expect-device cuda
#   Ctrl-b d to detach.  (preflight runs automatically and ABORTS on FAIL.)

# 3b. NPE ablation (Variant B) — same data, separate run dir
python scripts/train.py --config configs/default.yaml --head npe \
    --run-dir runs/npe --data-dir data/tess_1M --expect-device cuda
```

> No pre-generation? Drop `--data-dir` to simulate on the fly with
> `--num-workers 8` (prefetch processes). Fine for small/MPS runs; on a rented
> 4090 the preflight will flag `GPU-STARVED` so you know you're overpaying.

**Monitor it** — three independent ways, all over SSH:

```bash
python scripts/monitor.py --run-dir runs/fmpe --watch   # live dashboard, refresh 5s
tail -f runs/fmpe/training_log.jsonl                     # raw per-step JSONL
cat runs/fmpe/status.json                                # one-shot: step/ETA/AUC/throughput
# TensorBoard (loss + val curves), tunnel from your laptop:
#   remote:  tensorboard --logdir runs/ --port 6006
#   laptop:  ssh -L 6006:localhost:6006 root@<host> -p <port>   then open localhost:6006
```

The monitor prints a **HEALTH** verdict so a wasteful or dead run is obvious at a
glance: `HEALTHY`, `WARN (GPU-starved 40%)`, `STALLED` (status.json not updating —
the process likely hung), `DIVERGED` (non-finite loss), or `CPU-LONG-RUN`
(silently fell off the GPU). Training also self-aborts after `nan_patience`
non-finite losses rather than burning hours on a diverged run.

`status.json` holds live `step`, `progress`, `throughput_lc_per_s`, `data_wait_frac`,
`eta`, current losses, `best` validation posterior loss, `best_detection`
validation AUC, and a `health` block — poll it from anywhere. Checkpoints land in
`runs/fmpe/checkpoints/`: `latest.pt` (final/resume checkpoint), `best.pt`
(lowest validation posterior loss), `best_detection.pt` (highest validation
detection AUC), and rotating `step_*.pt`. Each is written atomically, so a killed
instance never leaves a corrupt file.

**Resume** after a preemption (auto-detects `latest.pt`):

```bash
python scripts/train.py --config configs/default.yaml --run-dir runs/fmpe --resume
```

**Retrieve & evaluate**. For the ephemeris-conditioned 5-D characterization model,
use `latest.pt` as the primary posterior checkpoint unless a calibration sweep
predeclares another checkpoint. Use `best_detection.pt` only for detector-only
diagnostics.

```bash
scp -r root@<host>:<port>:TransitFlow/runs/fmpe/checkpoints ./        # pull weights
python scripts/evaluate.py --ckpt runs/fmpe/checkpoints/latest.pt --plots --out results/fmpe
```

> Throughput: with `--num-workers N` the simulator runs in `N` processes feeding a
> bounded queue, so the 4090 isn't starved. If worker startup fails in a
> restricted sandbox, it degrades automatically to synchronous generation rather
> than hanging. Set `--num-workers 0` to force synchronous.

## Real data

The pipeline runs fully on synthetic GP + white noise out of the box. To train on
**real** out-of-transit noise (the §2.2 primary regime), build a noise library:

```bash
python scripts/build_noise_library.py --mission TESS \
    --targets TIC307210830 TIC150428135 --out data/noise_lib.npz
# then point the simulator at it (frac_real > 0)
```

## Tests

```bash
python -m pytest -q          # 68 tests: physics, flows, calibration, baselines, e2e
```

## Compute

Designed for a single RTX 4090 (24 GB): <50 M params, bf16, batch 256, ~20–60
GPU-hours per variant; ~100–300 GPU-hours total (≈ $50–150 on a rented 4090).
The native simulator generates ~250 light curves/s/core on CPU, so data are
simulated **on the fly** — no terabyte dataset to store.

## Layout

```
transitflow/
  priors.py            parameter priors + bijective standardizing transforms
  transit_model.py     vectorized quadratic-LD transit (native + batman engines)
  noise.py             GP / white noise, hard negatives, real-noise library
  views.py             global + local + box-periodogram view construction
  simulator.py         the SBI forward model (parameters → tri-view data)
  data.py              pre-generated sharded dataset, loaded in-RAM for a fed GPU
  correction.py        importance-sampling posterior reweighting + diagnostics
  flow_matching.py     CFM loss, ODE sampling, exact log-density
  models/              embedding CNN, detection + FMPE/NPE heads, spike-and-slab
  inference.py         amortized detect + characterize + IS diagnostic
  train.py             training loop (Variants A/B), checkpointing
  evaluation/          SBC, coverage, detection metrics, posterior agreement
  baselines/           BLS detection, transit-fit MCMC
  train.py             training loop: run dirs, periodic checkpoints, resume, logs
configs/               default.yaml (science) + smoke.yaml (fast)
scripts/               train / evaluate / monitor / smoke_test / build_noise_library
                       generate_data / preflight / benchmark_speed (gate #4)
                       validate_real (gate #3: real KOIs/TOIs vs archive)
                       diagnose_period (stratified SBC + alias analysis)
tests/                 unit + integration tests (incl. checkpoint/resume)
```
