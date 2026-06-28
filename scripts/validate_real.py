#!/usr/bin/env python3
"""Real-planet validation (gate #3): amortized posteriors vs the literature.

Downloads light curves for confirmed transiting planets (NASA Exoplanet Archive +
lightkurve), runs the trained TransitFlow amortized posterior on each, and checks
that the *published* parameter values fall within the predicted credible intervals
(calibration on real data) with small fractional error (accuracy). Optionally runs
a per-object MCMC fit for posterior-shape agreement (Wasserstein / JS).

The model is trained on the single-sector TESS regime (P in [0.5, 13] d, ~27 d
baseline), so we select single-sector TESS planets in that period range — staying
inside the amortized network's training distribution is what makes the comparison
valid.

Example
-------
    python3 scripts/validate_real.py --ckpt runs/fmpe_pg/checkpoints/latest.pt \
        --n-planets 30 --out results/real --with-mcmc 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.train import load_checkpoint
from transitflow.transit_model import transit_duration
from transitflow.views import (
    flatten_transit_preserving,
    make_periodogram_view,
    make_views,
)

# parameters we can compare against the archive (subset of the 7-D vector)
_CMP = {"P": 0, "RpRs": 2, "aRs": 3, "b": 4}
_CHAR_CMP = ("RpRs", "aRs", "b")
_TESS_T0_OFFSET = 2457000.0  # BTJD = BJD - 2457000 (TESS time system)


# --------------------------------------------------------------------------- #
# Archive query + light-curve download
# --------------------------------------------------------------------------- #
def _val(row, key) -> float:
    """Read a numeric archive cell as a plain float, robustly.

    NASA archive columns are astropy ``Quantity`` objects carrying units, so a
    bare ``float(row[key])`` raises "only dimensionless scalar quantities can be
    converted to Python scalars"; masked cells must also map to NaN.
    """
    v = row[key]
    if v is np.ma.masked:
        return float("nan")
    if hasattr(v, "value"):        # astropy Quantity -> strip units
        v = v.value
    try:
        return float(v)
    except Exception:
        return float("nan")


def query_planets(n: int, p_lo: float, p_hi: float,
                  rprs_lo: float, rprs_hi: float) -> list[dict]:
    """Confirmed TESS planets *inside the training prior*, in the P range.

    Bounding ``pl_ratror`` to the prior's Rp/Rs support is essential: sorting by
    depth alone surfaces out-of-distribution systems (e.g. a planet transiting a
    white dwarf at Rp/Rs ~ 7), which the amortized model was never trained on.

    The returned pool is a *representative* spread across the Rp/Rs range, not the
    deepest-first cherry-pick: the deepest TESS transits (Rp/Rs ≳ 0.10) are
    grazing / strongly limb-darkened hot Jupiters whose morphology departs from
    the trained box-like transit, so a deepest-first sample over-tests the tail
    and understates calibration in the regime the amortized model actually
    operates in. We stratify across depth and shuffle so the download stage's
    first ``n`` successes are representative.
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    cols = ("pl_name,hostname,tic_id,pl_orbper,pl_tranmid,pl_trandur,"
            "pl_ratror,pl_ratdor,pl_imppar,pl_rade,st_rad,disc_facility")
    tab = NasaExoplanetArchive.query_criteria(table="pscomppars", select=cols,
                                              where=(f"pl_orbper > {p_lo} and "
                                                     f"pl_orbper < {p_hi} and "
                                                     f"pl_ratror > {rprs_lo} and "
                                                     f"pl_ratror < {rprs_hi} and "
                                                     f"tran_flag = 1 and "
                                                     f"disc_facility like '%TESS%'"),
                                              order="pl_ratror")
    out = []
    for row in tab:
        try:
            P = _val(row, "pl_orbper")
            RpRs = _val(row, "pl_ratror")
            t0 = _val(row, "pl_tranmid")
            if not (np.isfinite(P) and np.isfinite(RpRs) and np.isfinite(t0)):
                continue
            aRs = _val(row, "pl_ratdor")
            b = _val(row, "pl_imppar")
            dur_h = _val(row, "pl_trandur")
            out.append({
                "name": str(row["pl_name"]), "host": str(row["hostname"]),
                "tic": str(row["tic_id"]), "P": P, "t0_bjd": t0,
                "dur_days": dur_h / 24.0 if np.isfinite(dur_h) else np.nan,
                "RpRs": RpRs, "aRs": aRs, "b": b,
            })
        except Exception:
            continue
    if not out:
        return out
    # stratified spread across the (now ascending) Rp/Rs list, then shuffle
    out.sort(key=lambda d: d["RpRs"])
    pool_size = min(len(out), 6 * n)
    idx = sorted({int(i) for i in
                  np.linspace(0, len(out) - 1, pool_size).round()})
    pool = [out[i] for i in idx]
    np.random.default_rng(0).shuffle(pool)
    return pool


def download_lc(planet: dict, baseline_days: float):
    """Download + clean one TESS single-sector PDCSAP light curve.

    Returns (times_btjd, rel_flux) trimmed to ~one ``baseline_days`` window
    around a transit, or ``None`` on any failure.
    """
    import lightkurve as lk

    host = planet["host"]
    sr = lk.search_lightcurve(host, mission="TESS", author="SPOC")
    if len(sr) == 0:
        sr = lk.search_lightcurve(host, mission="TESS")
    if len(sr) == 0:
        return None
    lc = sr[0].download()                       # a single sector
    if lc is None:
        return None
    lc = lc.remove_nans().normalize()           # PDCSAP -> ~1.0 relative flux
    t = np.asarray(lc.time.value, dtype=np.float64)      # BTJD
    f = np.asarray(lc.flux.value, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size < 500:
        return None
    # clip to a single baseline window (keep it comparable to training)
    if (t[-1] - t[0]) > baseline_days:
        t0 = t[0]
        sel = t <= t0 + baseline_days
        t, f = t[sel], f[sel]
    return t, f


# --------------------------------------------------------------------------- #
# Detrending (make real data look like the stationary-noise training regime)
# --------------------------------------------------------------------------- #
def _refine_t0(t, f, P, dur):
    """Re-derive the transit epoch from the *data* by a box-search at fixed P.

    The archive ``t0`` is an epoch often thousands of cycles before the TESS
    sector; propagating it forward (``t0 % P``) multiplies the published period
    uncertainty by the cycle count, so the folded local view can be misaligned by
    a large fraction of a period. The transit then smears across phase, its depth
    washes out, and the Rp/Rs posterior rails to the prior floor (|z|>>1). The
    plan prescribes folding on the ephemeris recovered from the data at hand, so
    we keep the (reliable) published period and find the epoch as the phase of the
    deepest box of width ~duration. Returns a BTJD epoch inside the first cycle.
    """
    f0 = np.asarray(f, dtype=np.float64)
    f0 = f0 - np.median(f0)
    dt = float(np.median(np.diff(t)))
    half = max(0.5 * (dur if np.isfinite(dur) and dur > 0 else 0.1), dt)
    phase = (t - t[0]) % P                                   # [0, P)
    centers = np.linspace(0.0, P, 512, endpoint=False)
    d = np.abs(phase[None, :] - centers[:, None])
    d = np.minimum(d, P - d)                                 # periodic distance
    inbox = d < half
    cnt = inbox.sum(axis=1)
    depth = np.where(cnt > 0, -(f0[None, :] * inbox).sum(axis=1)
                     / np.maximum(cnt, 1), -np.inf)
    return t[0] + float(centers[int(np.argmax(depth))])


def _flatten_lc(t, f, P, t0, dur):
    """Remove slow secular trends, returning flux ≈ 1 around a flat baseline.

    The simulator trains on *stationary* GP + white noise — there is no slow
    instrumental/stellar drift in the synthetic flux. Real single-sector TESS
    light curves carry such drifts; a trend across the full baseline confuses the
    model's period/timing readout from the global view (period coverage collapses
    once the posterior is sharp). Dividing by a wide rolling-median trend brings
    real data into the trained distribution.

    Crucially the in-transit cadences are **masked** (replaced by interpolation
    across the dip) before the trend is estimated, so the trend does not soak up
    the transit itself — otherwise dividing by it attenuates the depth and biases
    the Rp/Rs posterior. This is the standard transit-preserving flatten.
    """
    return flatten_transit_preserving(t, f, P, t0, dur)


# --------------------------------------------------------------------------- #
# View construction (must mirror the simulator's normalization)
# --------------------------------------------------------------------------- #
def build_views(t, f, planet, sim, prior):
    cfg = sim.cfg
    P = planet["P"]
    dur = planet["dur_days"]
    if not np.isfinite(dur):
        aRs = planet["aRs"] if np.isfinite(planet["aRs"]) else 10.0
        b = planet["b"] if np.isfinite(planet["b"]) else 0.3
        dur = float(transit_duration(np.array([P]), np.array([planet["RpRs"]]),
                                     np.array([aRs]), np.array([b]))[0])
    # Re-derive the epoch from the data (propagated archive t0 drifts by many
    # cycles -> misaligned fold -> washed-out depth -> Rp/Rs rails to the floor).
    t0 = _refine_t0(t, f, P, dur)
    # Transit-preserving flatten: remove the secular trend (in-transit cadences
    # masked so the depth is preserved), then build ALL views from the flattened
    # flux. The global view drives the period/timing readout and is the most
    # trend-sensitive; the masked flatten recovers period coverage without
    # attenuating the Rp/Rs depth.
    f = _flatten_lc(t, f, P, t0, dur)
    gv, lv = make_views(t, f, P, t0, dur, n_global=cfg.n_global,
                        n_local=cfg.n_local, n_durations=cfg.n_durations,
                        normalize=True)
    pg = None
    if cfg.use_periodogram:
        # subsample to pg_n_raw exactly as the simulator does, so the periodogram
        # channel matches the training distribution (striding, not full-res)
        n_pg = cfg.pg_n_raw
        if n_pg < len(t):
            step = max(1, len(t) // n_pg)
            t_pg, f_pg = t[::step][:n_pg], f[::step][:n_pg]
        else:
            t_pg, f_pg = t, f
        pg = make_periodogram_view(t_pg, f_pg, sim.period_grid,
                                   n_phase=cfg.pg_n_phase, normalize=True)
    # noise feature: estimate the *white* level from point-to-point differences
    # (Var[x_{i+1}-x_i] = 2 sigma_white^2), which isolates white noise from
    # stellar/GP variability -- matching the simulator's sigma_white. A plain MAD
    # of the flux includes correlated power and badly over-/under-states sigma,
    # which miscalibrates the depth (Rp/Rs) posterior (coverage 0.38 -> 0.67).
    sigma = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2.0) + 1e-6
    lo, hi = cfg.sigma_white_log10_low, cfg.sigma_white_log10_high
    sig_feat = (np.log10(sigma) - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    ephem_phys = prior.sample(1, np.random.default_rng(0))
    ephem_phys[0, 0] = P
    ephem_phys[0, 1] = ((t0 - t.min()) / P) % 1.0
    ephem_feat = prior.physical_to_std(ephem_phys)[:, :2].astype(np.float32)[0]
    return gv, lv, pg, np.float32(sig_feat), ephem_feat, (t, f, P, t0, dur, sigma)


def real_quality_metrics(t, planet, raw, sim_cfg) -> dict:
    """Data-only quality checks for real-light-curve validation.

    These cuts avoid validating a 2-minute, single-sector TESS model on long-
    cadence, weak, missing-geometry, or visibly out-of-distribution archive rows.
    No model prediction is used here.
    """
    _t, _f, P, t0, dur, sigma = raw
    phase = ((_t - t0 + 0.5 * P) % P) - 0.5 * P
    in_tr = np.abs(phase) < 0.5 * dur
    oot = np.abs(phase) > 1.5 * dur
    n_in = int(in_tr.sum())
    n_cadences = int(len(_t))
    n_transits = int(np.floor((_t.max() - _t.min()) / P)) + 1 if len(_t) else 0
    observed_depth = float("nan")
    observed_snr = float("nan")
    if in_tr.any() and oot.any():
        observed_depth = float(np.median(_f[oot]) - np.median(_f[in_tr]))
        observed_snr = float(observed_depth / max(sigma, 1e-9) *
                             np.sqrt(max(n_in, 1)))
    expected_snr = float((planet["RpRs"] ** 2) / max(sigma, 1e-9) *
                         np.sqrt(max(n_in, 1)))
    aRs = planet.get("aRs", float("nan"))
    b = planet.get("b", float("nan"))
    finite_geometry = bool(np.isfinite(aRs) and np.isfinite(b) and
                           0.0 <= b <= 1.05)
    cadence_fraction = float(n_cadences / max(sim_cfg.n_raw, 1))
    return {
        "n_cadences": n_cadences,
        "cadence_fraction_of_training": cadence_fraction,
        "n_in_transit": n_in,
        "n_transits": n_transits,
        "sigma": float(sigma),
        "expected_snr": expected_snr,
        "observed_depth": observed_depth,
        "observed_snr": observed_snr,
        "finite_geometry": finite_geometry,
        "impact_parameter": float(b) if np.isfinite(b) else float("nan"),
    }


def passes_real_quality(q: dict, args) -> bool:
    if not args.quality_gate:
        return True
    return (
        q["finite_geometry"]
        and q["n_cadences"] >= args.min_cadences
        and q["cadence_fraction_of_training"] >= args.min_cadence_fraction
        and q["n_in_transit"] >= args.min_in_transit
        and q["n_transits"] >= args.min_transits
        and q["observed_snr"] >= args.min_observed_snr
        and q["impact_parameter"] <= args.max_impact
    )


def fold_bin_fixed_ephemeris(times: np.ndarray, flux: np.ndarray, sigma: float,
                             P: float, t0_phase: float,
                             max_cadences: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress fixed-ephemeris MCMC data by phase-binning.

    With P and t0 fixed, the transit model is periodic and the Gaussian
    likelihood only needs the folded phase samples. Consecutive phase bins use
    the mean flux and sigma/sqrt(n), which is the sufficient statistic for iid
    white noise within a small phase interval.
    """
    times = np.asarray(times, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if max_cadences <= 0 or len(times) <= max_cadences:
        return times, flux, np.full_like(times, float(sigma))
    t0 = float(t0_phase) * float(P)
    folded = ((times - t0 + 0.5 * P) % P) - 0.5 * P + t0
    order = np.argsort(folded)
    folded = folded[order]
    flux = flux[order]
    groups = np.array_split(np.arange(len(folded)), max_cadences)
    b_t, b_f, b_e = [], [], []
    for idx in groups:
        if len(idx) == 0:
            continue
        b_t.append(float(np.mean(folded[idx])))
        b_f.append(float(np.mean(flux[idx])))
        b_e.append(float(sigma) / np.sqrt(len(idx)))
    return (np.asarray(b_t, dtype=np.float64),
            np.asarray(b_f, dtype=np.float64),
            np.asarray(b_e, dtype=np.float64))


def real_gate_status(summary: dict) -> dict:
    """Compute real-data pass/fail gates from an aggregate validation summary."""
    mcmc_char_prior = [
        summary["mcmc_agreement"][k]["median_wasserstein_prior_fraction"]
        for k in _CHAR_CMP if k in summary.get("mcmc_agreement", {})
    ]
    mcmc_char_width = [
        summary["mcmc_agreement"][k]["median_wasserstein_width_fraction"]
        for k in _CHAR_CMP if k in summary.get("mcmc_agreement", {})
    ]
    gates = {
        "detected_fraction_ge_0.9": bool(summary["detection"]["detected_fraction"] >= 0.9),
        "mcmc_characterization_prior_fraction_le_0.1": bool(
            mcmc_char_prior and max(mcmc_char_prior) <= 0.1),
        "mcmc_characterization_width_fraction_le_0.5_diagnostic": bool(
            mcmc_char_width and max(mcmc_char_width) <= 0.5),
    }
    if summary.get("importance_correction", {}).get("enabled"):
        gates["importance_correction_min_ess_fraction_ge_0.05"] = bool(
            summary["importance_correction"].get("min_ess_fraction", 0.0) >= 0.05)
    return gates


def real_diagnostic_status(summary: dict) -> dict:
    """Report non-gating real-data diagnostics.

    Archive catalog parameters are literature-level values, often from joint or
    multi-sector analyses. They are useful diagnostics, but not a clean
    pass/fail calibration target for a single-sector TESS posterior.
    """
    detected_cov95 = [
        summary["detected_per_param"][k]["coverage_95"]
        for k in _CHAR_CMP if k in summary["detected_per_param"]
    ]
    detected_cov68 = [
        summary["detected_per_param"][k]["coverage_68"]
        for k in _CHAR_CMP if k in summary["detected_per_param"]
    ]
    return {
        "archive_detected_char_cov68_ge_0.5": bool(detected_cov68 and
                                                   min(detected_cov68) >= 0.5),
        "archive_detected_char_cov95_ge_0.8": bool(detected_cov95 and
                                                   min(detected_cov95) >= 0.8),
    }


def _bin_label(value: float, edges: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < edges[0]:
        return labels[0]
    if value < edges[1]:
        return labels[1]
    return labels[2]


def mcmc_stratified_summary(rows: list[dict]) -> dict:
    """MCMC Wasserstein summaries by data-only and geometry strata."""
    specs = {
        "impact": lambda r: _bin_label(
            r.get("quality", {}).get("impact_parameter", float("nan")),
            (0.35, 0.70), ("low_b", "mid_b", "high_b")),
        "observed_snr": lambda r: _bin_label(
            r.get("quality", {}).get("observed_snr", float("nan")),
            (25.0, 75.0), ("low_snr", "mid_snr", "high_snr")),
        "rprs": lambda r: _bin_label(
            r.get("params", {}).get("RpRs", {}).get("published", float("nan")),
            (0.04, 0.09), ("shallow", "medium", "deep")),
        "a_rs": lambda r: _bin_label(
            r.get("params", {}).get("aRs", {}).get("published", float("nan")),
            (10.0, 25.0), ("compact", "mid", "wide")),
        "n_transits": lambda r: _bin_label(
            r.get("quality", {}).get("n_transits", float("nan")),
            (3.0, 6.0), ("few", "several", "many")),
    }
    out = {}
    for spec_name, label_fn in specs.items():
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(label_fn(row), []).append(row)
        out[spec_name] = {}
        for label, group in sorted(grouped.items()):
            entry = {"n": len(group)}
            for param in _CHAR_CMP:
                prior_vals = [
                    g["mcmc_wasserstein_prior_fraction"][param]
                    for g in group
                    if param in g.get("mcmc_wasserstein_prior_fraction", {})
                ]
                width_vals = [
                    g["mcmc_wasserstein_width_fraction"][param]
                    for g in group
                    if param in g.get("mcmc_wasserstein_width_fraction", {})
                ]
                if prior_vals:
                    entry[f"{param}_median_prior_fraction"] = float(np.median(prior_vals))
                if width_vals:
                    entry[f"{param}_median_width_fraction"] = float(np.median(width_vals))
            out[spec_name][label] = entry
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe_pg/checkpoints/latest.pt")
    ap.add_argument("--detector-ckpt", default=None,
                    help="optional checkpoint used only for p_detect")
    ap.add_argument("--n-planets", type=int, default=30)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--with-mcmc", type=int, default=0,
                    help="run a per-object MCMC on the first K planets (shape agreement)")
    ap.add_argument("--mcmc-steps", type=int, default=1500)
    ap.add_argument("--mcmc-detect-threshold", type=float, default=0.9,
                    help="only run same-light-curve MCMC for detected planets")
    ap.add_argument("--mcmc-full-ephemeris", action="store_true",
                    help="sample P and t0 in MCMC instead of conditioning on the folded ephemeris")
    ap.add_argument("--mcmc-max-cadences", type=int, default=2500,
                    help="phase-bin fixed-ephemeris MCMC to at most this many cadences; <=0 disables")
    ap.add_argument("--mcmc-init-jitter", type=float, default=0.15,
                    help="standardized-space walker initialization jitter for real-data MCMC")
    ap.add_argument("--is-correct-mcmc", action="store_true",
                    help="use likelihood-corrected amortized samples for MCMC agreement")
    ap.add_argument("--is-samples", type=int, default=3000,
                    help="proposal samples for likelihood correction")
    ap.add_argument("--quality-gate", dest="quality_gate", action="store_true",
                    default=True,
                    help="require real light curves to match the trained TESS regime")
    ap.add_argument("--no-quality-gate", dest="quality_gate", action="store_false")
    ap.add_argument("--min-cadences", type=int, default=5000)
    ap.add_argument("--min-cadence-fraction", type=float, default=0.70)
    ap.add_argument("--min-in-transit", type=int, default=50)
    ap.add_argument("--min-transits", type=int, default=2)
    ap.add_argument("--min-observed-snr", type=float, default=12.0)
    ap.add_argument("--max-impact", type=float, default=0.9)
    ap.add_argument("--out", default="results/real")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mc, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(sc.regime))
    sim = TransitSimulator(sc, prior=prior)
    inf = TransitFlowInference(model, prior, sc)
    detector_inf = inf
    if args.detector_ckpt:
        detector_model, _, detector_sc = load_checkpoint(args.detector_ckpt)
        if detector_sc != sc:
            raise SystemExit("--detector-ckpt must use the same simulator config")
        detector_inf = TransitFlowInference(detector_model, prior, sc)
    p_lo, p_hi = prior.specs[0].low, prior.specs[0].high
    rprs_lo, rprs_hi = prior.specs[2].low, prior.specs[2].high   # training Rp/Rs support

    print(f"== querying archive (P in [{p_lo}, {p_hi}] d, Rp/Rs in "
          f"[{rprs_lo}, {rprs_hi}], TESS) ==")
    pool = query_planets(args.n_planets, p_lo, p_hi, rprs_lo, rprs_hi)
    print(f"   {len(pool)} candidate planets")

    records = []
    for pl in pool:
        if len(records) >= args.n_planets:
            break
        try:
            lc = download_lc(pl, sc.baseline_days)
            if lc is None:
                continue
            t, f = lc
            gv, lv, pg, sf, eph, raw = build_views(t, f, pl, sim, prior)
            quality = real_quality_metrics(t, pl, raw, sc)
            if not passes_real_quality(quality, args):
                print(f"   skip {pl['name']}: quality {quality}")
                continue
            out = inf.detect_and_characterize(gv, lv, np.array([sf]),
                                              n_samples=args.n_post,
                                              periodogram=pg, ephem_feat=eph)
            p_detect = detector_inf.detect(gv, lv, np.array([sf]),
                                           periodogram=pg, ephem_feat=eph)
            samp = out["samples"][0]                  # (n_post, 7)
            rec = {"name": pl["name"], "p_detect": float(p_detect[0]),
                   "quality": quality,
                   "params": {}}
            for k, idx in _CMP.items():
                v = pl[k]
                if not np.isfinite(v):
                    continue
                s = samp[:, idx]
                med = float(np.median(s))
                q16, q84 = np.percentile(s, [16, 84])
                q025, q975 = np.percentile(s, [2.5, 97.5])
                std = float(s.std())
                rec["params"][k] = {
                    "published": float(v), "post_median": med,
                    "post_std": std,
                    "z": (med - v) / std if std > 0 else float("nan"),
                    "frac_err": abs(med - v) / abs(v) if v != 0 else float("nan"),
                    "in_68": bool(q16 <= v <= q84),
                    "in_95": bool(q025 <= v <= q975),
                }
            records.append(rec)
            print(f"   [{len(records):2d}] {pl['name']:<18} p_det={rec['p_detect']:.3f} "
                  f"P_err={rec['params'].get('P', {}).get('frac_err', float('nan')):.3f}")
        except Exception as e:
            print(f"   skip {pl.get('name','?')}: {e}")
            continue

    # ---- optional MCMC shape agreement on the first K -------------------
    if args.with_mcmc > 0:
        from scipy.stats import wasserstein_distance

        from transitflow.correction import importance_weights, sir_resample
        from transitflow.baselines.mcmc import run_mcmc
        print(f"== MCMC shape agreement (first {args.with_mcmc} detected planets) ==")
        by_name = {r["name"]: r for r in records
                   if r["p_detect"] >= args.mcmc_detect_threshold}
        done = 0
        for pl in pool:
            if done >= args.with_mcmc:
                break
            if pl["name"] not in by_name:
                continue
            try:
                lc = download_lc(pl, sc.baseline_days)
                if lc is None:
                    continue
                t, f = lc
                gv, lv, pg, sf, eph, raw = build_views(t, f, pl, sim, prior)
                quality = real_quality_metrics(t, pl, raw, sc)
                if not passes_real_quality(quality, args):
                    continue
                (_t, _f, P, t0, dur, sigma) = raw
                t_rel = _t - _t[0]
                amort = inf.posterior_samples(gv, lv, np.array([sf]),
                                              n_samples=args.n_post, periodogram=pg,
                                              ephem_feat=eph)[0]
                init = np.array([P, ((t0 - _t[0]) / P) % 1.0, pl["RpRs"],
                                 pl["aRs"] if np.isfinite(pl["aRs"]) else 10.0,
                                 pl["b"] if np.isfinite(pl["b"]) else 0.3, 0.4, 0.3])
                fixed = None
                if getattr(inf.model.cfg, "param_dim", 7) == 5 and \
                        not args.mcmc_full_ephemeris:
                    fixed = {0: P, 1: ((t0 - _t[0]) / P) % 1.0}
                mcmc_t, mcmc_f, mcmc_err = t_rel, _f, np.full_like(t_rel, sigma)
                if fixed is not None:
                    mcmc_t, mcmc_f, mcmc_err = fold_bin_fixed_ephemeris(
                        t_rel, _f, sigma, P, fixed[1], args.mcmc_max_cadences)
                fit_dilution = bool(getattr(sc, "dilution_fraction", 0.0) > 0)
                mc_out = run_mcmc(mcmc_t, mcmc_f, mcmc_err, prior=prior, init=init,
                                  n_steps=args.mcmc_steps, n_radial=60,
                                  fixed=fixed,
                                  init_std_jitter=args.mcmc_init_jitter,
                                  exposure_minutes=getattr(sc, "exposure_minutes", 0.0),
                                  n_exposure_subsamples=getattr(
                                      sc, "n_exposure_subsamples", 1),
                                  fit_dilution=fit_dilution,
                                  dilution_low=getattr(sc, "dilution_low", 0.5),
                                  dilution_high=getattr(sc, "dilution_high", 1.0))
                mc_s = mc_out["samples"]
                ess = None
                if args.is_correct_mcmc:
                    corr = importance_weights(
                        inf, gv, lv, np.array([sf]), mcmc_f, mcmc_t, mcmc_err,
                        n_samples=args.is_samples, periodogram=pg, ephem_feat=eph)
                    amort = sir_resample(corr["phys"], corr["w"], args.n_post,
                                         np.random.default_rng(done + 1234))
                    ess = corr["ess_fraction"]
                wd = {k: float(wasserstein_distance(amort[:, idx], mc_s[:, idx]))
                      for k, idx in _CMP.items()}
                wd_norm = {}
                for k, idx in _CMP.items():
                    q_am = np.percentile(amort[:, idx], [16, 84])
                    q_mc = np.percentile(mc_s[:, idx], [16, 84])
                    width = max(float(q_am[1] - q_am[0]),
                                float(q_mc[1] - q_mc[0]), 1e-12)
                    wd_norm[k] = wd[k] / width
                by_name[pl["name"]]["mcmc_wasserstein"] = wd
                by_name[pl["name"]]["mcmc_wasserstein_width_fraction"] = wd_norm
                by_name[pl["name"]]["mcmc_backend"] = mc_out["backend"]
                by_name[pl["name"]]["mcmc_acceptance_fraction"] = \
                    mc_out.get("acceptance_fraction")
                by_name[pl["name"]]["mcmc_fixed"] = mc_out.get("fixed", {})
                by_name[pl["name"]]["mcmc_fit_dilution"] = fit_dilution
                if mc_out.get("dilution_samples") is not None:
                    by_name[pl["name"]]["mcmc_dilution_median"] = float(
                        np.median(mc_out["dilution_samples"]))
                by_name[pl["name"]]["mcmc_cadences"] = int(len(mcmc_t))
                if ess is not None:
                    by_name[pl["name"]]["is_ess_fraction"] = float(ess)
                ess_txt = "" if ess is None else f" ESS={ess:.3f}"
                print(f"   {pl['name']:<18} W(P)={wd['P']:.4f} W(RpRs)={wd['RpRs']:.4f}{ess_txt}")
                done += 1
            except Exception as e:
                print(f"   mcmc skip {pl.get('name','?')}: {e}")

    # ---- aggregate -----------------------------------------------------
    def _summarize(rows):
        out = {}
        for k in _CMP:
            vals = [r["params"][k] for r in rows if k in r["params"]]
            if not vals:
                continue
            out[k] = {
                "n": len(vals),
                "median_frac_err": float(np.median([v["frac_err"] for v in vals])),
                "coverage_68": float(np.mean([v["in_68"] for v in vals])),
                "coverage_95": float(np.mean([v["in_95"] for v in vals])),
                "mean_abs_z": float(np.mean([abs(v["z"]) for v in vals])),
            }
        return out

    detected = [r for r in records if r["p_detect"] >= 0.9]
    summary = {
        "n_planets": len(records),
        "checkpoint": args.ckpt,
        "detector_checkpoint": args.detector_ckpt or args.ckpt,
        "detection": {
            "threshold": 0.9,
            "n_detected": len(detected),
            "detected_fraction": float(len(detected) / max(len(records), 1)),
            "median_p_detect": float(np.median([r["p_detect"] for r in records]))
            if records else float("nan"),
        },
        "per_param": _summarize(records),
        "detected_per_param": _summarize(detected),
    }
    mcmc_rows = [r for r in records if "mcmc_wasserstein" in r]
    if mcmc_rows:
        ranges = {"P": p_hi - p_lo, "RpRs": rprs_hi - rprs_lo,
                  "aRs": prior.specs[3].high - prior.specs[3].low,
                  "b": prior.specs[4].high - prior.specs[4].low}
        summary["mcmc_agreement"] = {}
        ess_vals = [r["is_ess_fraction"] for r in mcmc_rows if "is_ess_fraction" in r]
        if ess_vals:
            summary["importance_correction"] = {
                "enabled": True,
                "n_samples": args.is_samples,
                "median_ess_fraction": float(np.median(ess_vals)),
                "min_ess_fraction": float(np.min(ess_vals)),
            }
        fixed_rows = [r.get("mcmc_fixed", {}) for r in mcmc_rows]
        summary["mcmc_conditioning"] = {
            "ephemeris_fixed": bool(fixed_rows and all(
                set(map(int, f.keys())) == {0, 1} for f in fixed_rows)),
            "acceptance_fraction_median": float(np.nanmedian([
                r.get("mcmc_acceptance_fraction", float("nan"))
                for r in mcmc_rows
            ])),
        }
        for k in _CMP:
            vals = [r["mcmc_wasserstein"][k] for r in mcmc_rows
                    if k in r["mcmc_wasserstein"]]
            norm_vals = [r["mcmc_wasserstein_width_fraction"][k] for r in mcmc_rows
                         if k in r.get("mcmc_wasserstein_width_fraction", {})]
            if vals:
                summary["mcmc_agreement"][k] = {
                    "n": len(vals),
                    "median_wasserstein": float(np.median(vals)),
                    "median_wasserstein_prior_fraction": float(np.median(vals) /
                                                               ranges[k]),
                    "median_wasserstein_width_fraction": float(np.median(norm_vals))
                    if norm_vals else float("nan"),
                }
        for r in mcmc_rows:
            r["mcmc_wasserstein_prior_fraction"] = {
                k: float(r["mcmc_wasserstein"][k] / ranges[k])
                for k in r["mcmc_wasserstein"] if k in ranges
            }
        summary["mcmc_stratified"] = mcmc_stratified_summary(mcmc_rows)
    # P is a conditioning input for the 5D characterization model, so P coverage
    # is not a valid posterior-calibration gate. Keep P in the report as a sanity
    # check, but gate only on characterization parameters.
    summary["gate_status"] = real_gate_status(summary)
    summary["diagnostic_status"] = real_diagnostic_status(summary)
    report = {"summary": summary, "records": records}
    with open(os.path.join(args.out, "real_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2))
    print("wrote", os.path.join(args.out, "real_validation.json"))


if __name__ == "__main__":
    main()
