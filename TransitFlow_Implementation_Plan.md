# TransitFlow — Full Implementation & Experiment Plan
### Amortized Flow-Matching Simulation-Based Inference for Joint Exoplanet Transit Detection and Parameter Posteriors

*Target compute: single NVIDIA RTX 4090 (24 GB) or moderate Vast.ai rental. Target venues: NeurIPS/ICML ML-for-Physical-Sciences or main track; AJ/MNRAS/A&A (ML methods).*

---

## 0. Honest novelty verdict (read this first)

**What is genuinely new:** This is the first amortized simulation-based inference (SBI) system using **flow matching** applied to **transit light curves** that returns, in a single forward pass, a *joint, calibrated* posterior over (i) whether a planet is present and (ii) its physical transit parameters.

**What is NOT new (and must be cited honestly):**
- Flow-matching posterior estimation (FMPE) itself — established for exoplanet *atmospheric retrieval from spectra* (Vasist 2023; Gebhard 2023, 2024; Giordano Orsini 2025) and for *direct-imaging orbital parameters* (Liang 2025, FM-MCMC). Different observables.
- Trans-dimensional / "how many components + their parameters" amortized inference — exists generically (SlotFlow, Houba 2025), but on sinusoids, with a classifier+flow rather than a unified posterior.
- Transit *detection* — heavily studied with CNNs/box-search (AstroNet, ExoMiner, BLS, TLS, GPFC), but these give point scores, not calibrated posteriors, and do not jointly characterize.

**Therefore the contribution is "novel application + methodological integration," not a new ML primitive.** The defensible core claims are: (1) first FMPE-SBI on transit light curves; (2) a unified detection+characterization posterior with simulation-based calibration; (3) calibrated uncertainty that detection CNNs and point-estimate regressors lack; (4) orders-of-magnitude speedup over MCMC/nested sampling with comparable posteriors. Frame the paper around (2)+(3), not around beating detection CNNs on raw recall.

---

## 1. Problem formulation

Let `x` be an observed (single-sector or single-quarter) light curve. We want the joint posterior

```
p(d, θ | x)  =  p(d | x) · p(θ | d = 1, x)
```

where:
- `d ∈ {0, 1}` is the detection indicator (planet present / absent),
- `θ = (P, t0, Rp/Rs, a/Rs, b, u1, u2)` are transit parameters
  (period, mid-transit epoch, radius ratio, scaled semi-major axis, impact parameter, two limb-darkening coefficients),
- optionally a noise-level conditioning variable `σ` is appended to the observation embedding (à la Gebhard 2024 "noise-level conditioning").

We learn two heads on a **shared embedding** `e = E(x)`:
1. **Detection head** `g_φ(e) → p(d=1 | x)` (binary classifier).
2. **Characterization head** = a conditional continuous normalizing flow (CNF) trained by flow matching that represents `p(θ | d=1, x)`.

This factorization is the **robust primary**. Section 4.3 gives a more ambitious *unified* variant (spike-and-slab depth prior) that collapses both into one posterior and is the stronger methods-novelty story if it trains stably.

---

## 2. Forward model / simulator (the heart of SBI)

SBI quality is bounded by simulator realism. Build a fast, vectorized simulator that mixes physical transit models with **real** detector noise.

### 2.1 Transit model
- Use **`batman`** (Kreidberg 2015) or **`ellc`** to generate normalized transit light curves from `θ`.
- Quadratic limb darkening `(u1, u2)`; allow eccentricity = 0 in v1 (add `e, ω` later as nuisance params).

### 2.2 Noise model (three regimes, mixed per batch)
1. **Real-noise injections (primary, ~70%):** download out-of-transit segments of real Kepler/TESS light curves with `lightkurve`, inject `batman` transits → realistic correlated stellar variability + instrumental systematics.
2. **GP-correlated synthetic noise (~20%):** `celerite2` Gaussian-process stellar variability + white noise, for controllable SNR sweeps.
3. **Pure white Gaussian (~10%):** clean idealized regime for calibration unit-tests.

### 2.3 The "no-planet" (`d=0`) class
- Same stellar/instrumental noise **without** an injected transit.
- **Hard negatives (recommended):** inject eclipsing-binary-like V-shaped dips, single-event systematics, and sinusoidal stellar variability so the detector learns transit-specific morphology, not "any dip." This is what separates a publishable detector from a toy.

### 2.4 Parameter priors (training distribution)

| Parameter | Symbol | Prior | Range |
|---|---|---|---|
| Orbital period | P | log-uniform | 0.5 – 50 d (Kepler) / 0.5 – 13 d (single TESS sector) |
| Epoch | t0 | uniform | within baseline |
| Radius ratio | Rp/Rs | log-uniform | 0.01 – 0.15 |
| Scaled SMA | a/Rs | derived from P, stellar density prior | physical |
| Impact parameter | b | uniform | 0 – 1.1 (allow grazing) |
| Limb darkening | u1, u2 | uniform / Kipping triangular | physical-valid region |
| Noise level | σ | sampled from real-LC CDPP distribution | data-driven |

### 2.5 Light-curve representation (fixed-length, 4090-tractable)
Raw Kepler/TESS curves are huge (Kepler ≈ 65k cadences over 4 yr; TESS 2-min ≈ 20k/sector). Use the **dual-view** representation (Shallue & Vanderburg 2018):
- **Global view:** whole curve binned to **2001** points (captures period / multiple transits / detection).
- **Local view:** phase-folded, transit-centered, binned to **201** points (captures depth / duration / shape → characterization).
- Feed both views; the global view drives detection + period, the local view drives shape parameters.

---

## 3. Model architecture

### 3.1 Embedding network `E(x)`
Dual-branch 1-D CNN (ResNet-1D style):
- Global branch: ~6–8 conv blocks over the 2001-pt view → 256-d.
- Local branch: ~4 conv blocks over the 201-pt view → 128-d.
- Concatenate → MLP → **embedding `e ∈ R^256`**.
- Params: ~5–15M. (Alternatively a small 1-D transformer; CNN is cheaper and sufficient.)

### 3.2 Detection head `g_φ`
- 2-layer MLP on `e` → sigmoid → `p(d=1 | x)`. Trained with BCE (class-balanced).

### 3.3 Flow-matching characterization head
Conditional CNF parameterized by a time-dependent velocity field `v_θ(t, θ_τ | e)`:
- Backbone: MLP (or small residual MLP / DiT-style token network) taking `(θ_τ, t, e)` → velocity in parameter space (dim = |θ| ≈ 7).
- Params: ~5–20M.

**Conditional Flow Matching (OT path) objective** (Lipman 2023; Tong OT-CFM):
```
θ_0 ~ N(0, I);   θ_1 = true params;   θ_τ = (1−τ)θ_0 + τ θ_1,   τ ~ U(0,1)
L_FM = E || v_θ(τ, θ_τ | e) − (θ_1 − θ_0) ||²
```
**Total loss:** `L = L_FM + λ · BCE(g_φ(e), d)`, with `λ ≈ 1`. Train detection on both classes; train FM head only on `d=1` samples (mask).

### 3.4 Inference
- Detection: read `p(d=1|x)` directly.
- Characterization: sample `θ_0 ~ N(0,I)`, integrate the probability-flow ODE `dθ/dτ = v_θ(τ, θ | e)` from τ=0→1 with an adaptive solver (`torchdiffeq` dopri5, or a few-step Heun). Draw 1k–10k samples → posterior. Amortized: ~ms–seconds per object.
- **Importance-sampling correction (Gebhard 2024):** reweight FMPE samples by `p(x|θ)p(θ)/q(θ|x)` using the simulator likelihood where tractable; IS efficiency doubles as a misspecification diagnostic.

---

## 4. Three method variants (run in this order)

### 4.1 Variant A — Factorized (robust primary; guarantees a result)
`p(d,θ|x) = p(d|x)·p(θ|d=1,x)` as above. Stable, easy to calibrate. **Start here.**

### 4.2 Variant B — NPE ablation (required baseline)
Replace the flow-matching head with a normalizing-flow NPE head (neural spline flow, via `zuko`/`sbi`). Purpose: show FMPE matches or beats NPE and trains faster (mirrors Gebhard 2024: FMPE ~3× faster training, higher IS efficiency). This ablation is what makes the "why flow matching" claim rigorous.

### 4.3 Variant C — Unified spike-and-slab (ambitious; stronger novelty)
Put a **spike-and-slab prior on transit depth** `δ = (Rp/Rs)²`: an atom at `δ=0` (no planet) + continuous slab for `δ>0`. Then a single posterior over `θ` (with `δ`) encodes detection as posterior mass away from `δ≈0`. This unifies detection+characterization in one object and differentiates cleanly from SlotFlow's classifier+flow. Run **only after A works**; flag as higher-risk (mixed discrete-continuous flow matching is finicky). If unstable, A is the fallback and the paper still stands.

---

## 5. Datasets

| Use | Source | Tool |
|---|---|---|
| Real noise for injection | Kepler long-cadence quarters; TESS 2-min SPOC | `lightkurve` |
| Training labels | simulated via `batman` injection | custom |
| Real validation (detection) | Kepler KOI catalog, TESS TOI catalog (confirmed + false positives) | NASA Exoplanet Archive |
| Real validation (characterization) | confirmed planets with published MCMC params (e.g., well-studied KOIs/TOIs, WASP targets) | literature / Archive |

Training set size: **1–5M** simulated light curves (cheap to generate; store as compact binned arrays, not raw). Balanced `d=0`/`d=1` with a hard-negative fraction.

---

## 6. Experiments & evaluation

### 6.1 Baselines
- **Detection:** Box Least Squares (BLS), Transit Least Squares (TLS), AstroNet-style CNN, and (if reproducible) GPFC / ExoMiner-style scorers.
- **Characterization (posterior):** MCMC (`emcee`) and nested sampling (`dynesty`) transit fits; `exoplanet`/PyMC. These are the gold-standard posteriors TransitFlow must approximate.
- **SBI ablation:** NPE (normalizing flow) vs FMPE (Variant B).

### 6.2 Metrics
**Detection**
- Precision–recall, ROC-AUC.
- Completeness & reliability as a function of (period, Rp/Rs, SNR) — the standard injection-recovery grid.

**Characterization (the differentiator)**
- **Simulation-Based Calibration (SBC):** rank histograms should be uniform (use `sbi` SBC utilities).
- **Expected coverage probability** (empirical vs nominal credible intervals).
- **Negative log-probability** of true parameters under the posterior; posterior contraction.
- **Agreement with MCMC/nested posteriors on real confirmed planets:** 1-D/2-D marginal overlap via Jensen–Shannon / Wasserstein distance; does the amortized posterior bracket published literature values?
- **Importance-sampling efficiency** as a misspecification diagnostic (low efficiency flags simulator gap).

**Efficiency**
- Wall-clock per object vs MCMC/nested (expect 10³–10⁶× speedup, consistent with FM-MCMC results in Liang 2025).

### 6.3 Ablations
1. Joint (A) vs separate detection-only / characterization-only training — does sharing the embedding help both?
2. FMPE vs NPE (Variant B).
3. Noise-level conditioning on vs off.
4. Real-noise injection vs synthetic-only training (quantify the real-noise domain gap).
5. Hard negatives on vs off (does it cut eclipsing-binary false positives?).
6. Unified spike-and-slab (C) vs factorized (A), if C trains.

---

## 7. Compute budget & feasibility (single RTX 4090)

| Stage | Compute | Notes |
|---|---|---|
| Simulating 1–5M LCs | hours (CPU + GPU) | `batman` is fast; parallelize; store binned |
| Training (per variant) | ~20–60 GPU-hours | <50M params, bf16, batch 256–1024, fits in 24 GB easily |
| SBC + calibration | hours | amortized inference is ~ms–s/object |
| Baselines (MCMC/nested) | hours–1 day | run on a manageable validation subset, not all |
| **Total project** | **~100–300 GPU-hours** | days on a local 4090 |

**Vast.ai cost:** a 4090 at ~$0.30–0.50/hr → **≈ $50–150** for the whole project. VRAM headroom is large; this never needs an A100/H100.

---

## 8. Suggested timeline (~6 weeks, part-time-friendly)

| Week | Goal |
|---|---|
| 1 | Simulator + data pipeline: `batman` injection into real `lightkurve` segments; global/local views; priors; forward-model sanity checks |
| 2 | Embedding CNN + FMPE head (use `torchcfm`/`zuko`/`sbi`); overfit-small sanity; SBC on a toy task |
| 3 | Full training of Variant A on 1–5M sims; detection head; tune; first posteriors |
| 4 | SBC, coverage, IS correction; FMPE-vs-NPE ablation (Variant B) |
| 5 | Baselines (BLS/TLS/CNN/MCMC) + injection-recovery benchmark; real confirmed-planet validation |
| 6 | Remaining ablations + Variant C attempt; figures, paper draft |

---

## 9. Risks & pivot thresholds

| Risk | Mitigation / pivot |
|---|---|
| Posterior miscoverage (SBC fails) | add importance-sampling correction (Gebhard 2024) and/or FMPE calibration / FMCPE (Ruhlmann 2025) |
| Detection underperforms dedicated CNNs | reframe contribution as *calibrated joint characterization + UQ*, not SOTA recall — the amortized posterior is the value, not raw detection |
| Real-noise domain gap | train predominantly on real-noise injections; add noise-level conditioning; report synthetic-vs-real ablation |
| Variant C (spike-and-slab) unstable | fall back to Variant A factorization — paper still stands |
| Simulator too idealized | inject hard negatives (EBs, systematics); validate IS efficiency on real data |

**Kill/keep rule:** if SBC + real-planet agreement hold for Variant A *and* the FMPE-vs-NPE ablation is clean, the paper is viable even if detection only ties (not beats) CNNs.

---

## 10. Software stack

- **Data:** `lightkurve`, NASA Exoplanet Archive, `astropy`.
- **Simulator:** `batman-package` (or `ellc`), `celerite2` (GP noise), `numpy`/`numba`.
- **SBI / flows:** `sbi`, `lampe`, `zuko` (NPE & flows), `torchcfm` (conditional flow matching), `torchdiffeq` (ODE solve).
- **Baselines:** `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`, `emcee`, `dynesty`, `exoplanet`/PyMC.
- **Diagnostics:** `sbi` SBC tools, `corner`, `arviz`.
- **Compute:** PyTorch (bf16/AMP), single 4090.

---

## 11. Expected contributions (paper framing)

1. **First amortized FMPE-SBI for transit light curves** delivering a *joint* detection + transit-parameter posterior in one forward pass.
2. **Calibrated uncertainty** (validated by SBC, coverage, IS correction) that detection CNNs and point-estimate regressors structurally cannot provide.
3. **Rigorous FMPE-vs-NPE ablation** showing flow matching's training-speed / efficiency advantage on this problem.
4. **10³–10⁶× speedup** over MCMC/nested sampling with statistically consistent posteriors on real confirmed planets — directly relevant to the data volumes of TESS extended mission and future surveys (PLATO, Roman).
5. Honest positioning vs atmospheres-FMPE, direct-imaging FM-MCMC, and generic trans-dimensional SBI.

**Venue path:** ship a NeurIPS/ICML ML4PS workshop version first (fast feedback, flag-planting), then extend to a main-track or AJ/MNRAS/A&A submission. A "Datasets & Benchmarks" angle is available if you release the transit-SBI benchmark + injection suite.

---

## 12. সারসংক্ষেপ (Bangla summary)

এই প্রকল্পে একটি AI মডেল তৈরি হবে যা নক্ষত্রের আলোর বক্ররেখা (light curve) দেখে একসাথে দুটি কাজ করবে — গ্রহ আছে কি না তা শনাক্ত করবে, এবং থাকলে তার আকার, কক্ষপথ ও অন্যান্য বৈশিষ্ট্য নিশ্চয়তাসহ (uncertainty সহ) বের করবে। এর জন্য "flow matching" নামক একটি আধুনিক জেনারেটিভ পদ্ধতি ও simulation-based inference ব্যবহার করা হবে, যেখানে batman দিয়ে কৃত্রিম ট্রানজিট তৈরি করে আসল Kepler/TESS ডেটার নয়েজে বসিয়ে মডেল প্রশিক্ষণ দেওয়া হবে। পুরো কাজটি একটি RTX 4090 জিপিইউ-তেই (বা সস্তা ক্লাউড ভাড়ায়, আনুমানিক ৫০–১৫০ ডলার) সম্পন্ন করা সম্ভব। মূল নতুনত্ব হলো — প্রচলিত পদ্ধতির মতো শুধু "হ্যাঁ/না" উত্তর নয়, বরং দ্রুত ও সঠিক সম্ভাব্যতা-ভিত্তিক (calibrated posterior) ফলাফল, যা MCMC-এর চেয়ে হাজার থেকে লক্ষ গুণ দ্রুত।
