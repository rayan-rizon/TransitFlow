#!/usr/bin/env python3
"""Deep diagnosis of the period (P) SBC calibration failure.

Stratifies the period rank statistic by the number of observable transits
N_tr = baseline / P, characterizes the rank-histogram shape, and quantifies
period-alias contamination (posterior mass at P/2, 2P, and N:1 / 1:N ratios).
This pinpoints *where* and *why* amortized period inference is miscalibrated.
"""
from __future__ import annotations
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from transitflow.train import load_checkpoint
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.evaluation.sbc import sbc_uniformity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe/checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-post", type=int, default=500)
    ap.add_argument("--out", default="results/period_diag.json")
    args = ap.parse_args()

    m, mc, sc = load_checkpoint(args.ckpt)
    pr = TransitPrior(TransitPrior.default_specs(sc.regime))
    sim = TransitSimulator(sc, prior=pr)
    inf = TransitFlowInference(m, pr, sc)
    rng = np.random.default_rng(11)
    L = args.n_post

    ranks = []           # P rank in std space
    P_true = []
    ntr = []
    p_med = []
    alias = {"half": [], "double": [], "primary": []}  # mass fractions
    got = 0
    while got < args.n:
        b = sim.simulate_batch(128, rng)
        mk = b["valid"]
        if not mk.any():
            continue
        pg = b["periodogram"][mk] if "periodogram" in b else None
        eph = b["ephem_feat"][mk] if "ephem_feat" in b else None
        ph, std = inf.posterior_samples(b["global"][mk], b["local"][mk],
                                        b["sigma_feat"][mk], n_samples=L,
                                        return_std=True, periodogram=pg,
                                        ephem_feat=eph)
        tp = b["theta_phys"][mk]
        tstd = pr.physical_to_std(tp)
        ranks.append((std[:, :, 0] < tstd[:, 0][:, None]).sum(1))
        Pt = tp[:, 0]
        Ps = ph[:, :, 0]
        P_true.append(Pt)
        ntr.append(sc.baseline_days / Pt)
        p_med.append(np.median(Ps, 1))
        # alias mass: fraction of period samples within 5% of P, P/2, 2P
        rel = lambda c: (np.abs(Ps - c[:, None]) / c[:, None] < 0.05).mean(1)
        alias["primary"].append(rel(Pt))
        alias["half"].append(rel(Pt / 2))
        alias["double"].append(rel(Pt * 2))
        got += int(mk.sum())

    ranks = np.concatenate(ranks)[:args.n]
    P_true = np.concatenate(P_true)[:args.n]
    ntr = np.concatenate(ntr)[:args.n]
    p_med = np.concatenate(p_med)[:args.n]
    for k in alias:
        alias[k] = np.concatenate(alias[k])[:args.n]

    out = {"overall_p": float(sbc_uniformity(ranks[:, None])["pvalue"][0]),
           "rank_hist10": (np.histogram(ranks / L, bins=10, range=(0, 1))[0]
                           / (args.n / 10)).round(3).tolist(),
           "strata": [], "alias": {}}
    for lo, hi, name in [(0, 3, "long_P_<3_transits"), (3, 6, "3-6_transits"),
                         (6, 15, "6-15_transits"), (15, 1e9, "short_P_>15_transits")]:
        sel = (ntr >= lo) & (ntr < hi)
        if sel.sum() < 30:
            continue
        out["strata"].append({
            "regime": name, "n": int(sel.sum()),
            "P_sbc_p": float(sbc_uniformity(ranks[sel][:, None])["pvalue"][0]),
            "rank_mean": float((ranks[sel] / L).mean()),
            "median_rel_err": float(np.median((p_med[sel] - P_true[sel]) / P_true[sel])),
            "frac_relerr_lt5pct": float((np.abs((p_med[sel] - P_true[sel]) / P_true[sel]) < 0.05).mean()),
        })
    # alias contamination by regime
    for lo, hi, name in [(0, 3, "long_P_<3_transits"), (15, 1e9, "short_P_>15_transits")]:
        sel = (ntr >= lo) & (ntr < hi)
        if sel.sum() < 30:
            continue
        out["alias"][name] = {
            "primary_mass": round(float(alias["primary"][sel].mean()), 3),
            "half_mass": round(float(alias["half"][sel].mean()), 3),
            "double_mass": round(float(alias["double"][sel].mean()), 3),
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
