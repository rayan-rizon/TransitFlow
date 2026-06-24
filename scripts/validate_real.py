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


def query_planets(n: int, p_lo: float, p_hi: float) -> list[dict]:
    """Confirmed TESS planets with the params we need, in the training P range."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    cols = ("pl_name,hostname,tic_id,pl_orbper,pl_tranmid,pl_trandur,"
            "pl_ratror,pl_ratdor,pl_imppar,pl_rade,st_rad,disc_facility")
    # request the high-S/N (large Rp/Rs) end first; the download stage trims to n
    tab = NasaExoplanetArchive.query_criteria(table="pscomppars", select=cols,
                                              where=(f"pl_orbper > {p_lo} and "
                                                     f"pl_orbper < {p_hi} and "
                                                     f"pl_ratror is not null and "
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
# View construction (must mirror the simulator's normalization)
# --------------------------------------------------------------------------- #
def build_views(t, f, planet, sim, prior):
    cfg = sim.cfg
    P = planet["P"]
    # transit epoch expressed in the LC's BTJD frame
    t0 = planet["t0_bjd"] - _TESS_T0_OFFSET
    # bring t0 into the observed window (fold is periodic, so any transit works)
    t0 = t[0] + ((t0 - t[0]) % P)
    dur = planet["dur_days"]
    if not np.isfinite(dur):
        aRs = planet["aRs"] if np.isfinite(planet["aRs"]) else 10.0
        b = planet["b"] if np.isfinite(planet["b"]) else 0.3
        dur = float(transit_duration(np.array([P]), np.array([planet["RpRs"]]),
                                     np.array([aRs]), np.array([b]))[0])
    gv, lv = make_views(t, f, P, t0, dur, n_global=cfg.n_global,
                        n_local=cfg.n_local, n_durations=cfg.n_durations,
                        normalize=True)
    pg = None
    if cfg.use_periodogram:
        pg = make_periodogram_view(t, f, sim.period_grid, n_phase=cfg.pg_n_phase,
                                   normalize=True)
    # noise feature: robust OOT scatter -> standardized like the simulator
    resid = f - np.median(f)
    sigma = 1.4826 * np.median(np.abs(resid)) + 1e-6
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

    print(f"== querying archive (P in [{p_lo}, {p_hi}] d, TESS) ==")
    pool = query_planets(args.n_planets, p_lo, p_hi)
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
