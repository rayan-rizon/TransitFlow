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
    python scripts/validate_real.py --ckpt runs/fmpe_pg/best.pt \
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
from transitflow.views import make_periodogram_view, make_views

# parameters we can compare against the archive (subset of the 7-D vector)
_CMP = {"P": 0, "RpRs": 2, "aRs": 3, "b": 4}
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
    A fair validation stays inside the trained regime.
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    cols = ("pl_name,hostname,tic_id,pl_orbper,pl_tranmid,pl_trandur,"
            "pl_ratror,pl_ratdor,pl_imppar,pl_rade,st_rad,disc_facility")
    # high-S/N end first (deepest in-prior transits); download stage trims to n
    tab = NasaExoplanetArchive.query_criteria(table="pscomppars", select=cols,
                                              where=(f"pl_orbper > {p_lo} and "
                                                     f"pl_orbper < {p_hi} and "
                                                     f"pl_ratror > {rprs_lo} and "
                                                     f"pl_ratror < {rprs_hi} and "
                                                     f"tran_flag = 1 and "
                                                     f"disc_facility like '%TESS%'"),
                                              order="pl_ratror desc")
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
    return out[: 4 * n]  # a generous pool; download stage trims to n


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
    from scipy.ndimage import median_filter

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return f
    # window ≈ max(1 day, 8 × transit duration), in cadences, capped below n
    win_days = max(1.0, 8.0 * (dur if np.isfinite(dur) else 0.2))
    win = int(np.clip(round(win_days / dt), 11, max(11, len(f) // 3)))
    if win % 2 == 0:
        win += 1
    f_work = np.array(f, dtype=np.float64, copy=True)
    if np.isfinite(P) and np.isfinite(t0) and np.isfinite(dur) and dur > 0:
        # phase distance from transit center, in days; mask a bit wider than dur
        phase = ((t - t0 + 0.5 * P) % P) - 0.5 * P
        in_transit = np.abs(phase) < 0.7 * dur
        oot = ~in_transit
        if in_transit.any() and oot.sum() > 2:
            f_work[in_transit] = np.interp(t[in_transit], t[oot], f[oot])
    trend = median_filter(f_work, size=win, mode="nearest")
    trend = np.where(np.abs(trend) < 1e-6, np.median(f), trend)
    return f / trend


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
    return gv, lv, pg, np.float32(sig_feat), (t, f, P, t0, dur, sigma)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe_pg/best.pt")
    ap.add_argument("--n-planets", type=int, default=30)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--with-mcmc", type=int, default=0,
                    help="run a per-object MCMC on the first K planets (shape agreement)")
    ap.add_argument("--mcmc-steps", type=int, default=1500)
    ap.add_argument("--out", default="results/real")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mc, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(TransitPrior.default_specs(sc.regime))
    sim = TransitSimulator(sc, prior=prior)
    inf = TransitFlowInference(model, prior, sc)
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
            gv, lv, pg, sf, raw = build_views(t, f, pl, sim, prior)
            out = inf.detect_and_characterize(gv, lv, np.array([sf]),
                                              n_samples=args.n_post,
                                              periodogram=pg)
            samp = out["samples"][0]                  # (n_post, 7)
            rec = {"name": pl["name"], "p_detect": float(out["p_detect"][0]),
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

        from transitflow.baselines.mcmc import run_mcmc
        print(f"== MCMC shape agreement (first {args.with_mcmc}) ==")
        # re-run download+views for the first K (kept simple; small K)
        done = 0
        for pl in pool:
            if done >= args.with_mcmc:
                break
            try:
                lc = download_lc(pl, sc.baseline_days)
                if lc is None:
                    continue
                t, f = lc
                gv, lv, pg, sf, raw = build_views(t, f, pl, sim, prior)
                (_t, _f, P, t0, dur, sigma) = raw
                amort = inf.posterior_samples(gv, lv, np.array([sf]),
                                              n_samples=args.n_post, periodogram=pg)[0]
                init = np.array([P, ((t0 - _t[0]) / P) % 1.0, pl["RpRs"],
                                 pl["aRs"] if np.isfinite(pl["aRs"]) else 10.0,
                                 pl["b"] if np.isfinite(pl["b"]) else 0.3, 0.4, 0.3])
                mc_out = run_mcmc(_t, _f, sigma, prior=prior, init=init,
                                  n_steps=args.mcmc_steps, n_radial=60)
                mc_s = mc_out["samples"]
                wd = {k: float(wasserstein_distance(amort[:, idx], mc_s[:, idx]))
                      for k, idx in _CMP.items()}
                for r in records:
                    if r["name"] == pl["name"]:
                        r["mcmc_wasserstein"] = wd
                        r["mcmc_backend"] = mc_out["backend"]
                print(f"   {pl['name']:<18} W(P)={wd['P']:.4f} W(RpRs)={wd['RpRs']:.4f}")
                done += 1
            except Exception as e:
                print(f"   mcmc skip {pl.get('name','?')}: {e}")

    # ---- aggregate -----------------------------------------------------
    summary = {"n_planets": len(records), "checkpoint": args.ckpt, "per_param": {}}
    for k in _CMP:
        vals = [r["params"][k] for r in records if k in r["params"]]
        if not vals:
            continue
        summary["per_param"][k] = {
            "n": len(vals),
            "median_frac_err": float(np.median([v["frac_err"] for v in vals])),
            "coverage_68": float(np.mean([v["in_68"] for v in vals])),  # target 0.68
            "coverage_95": float(np.mean([v["in_95"] for v in vals])),  # target 0.95
            "mean_abs_z": float(np.mean([abs(v["z"]) for v in vals])),
        }
    report = {"summary": summary, "records": records}
    with open(os.path.join(args.out, "real_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2))
    print("wrote", os.path.join(args.out, "real_validation.json"))


if __name__ == "__main__":
    main()
