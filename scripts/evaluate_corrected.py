#!/usr/bin/env python3
"""Flow vs importance-sampling-corrected SBC (the period-calibration fix).

Runs on the white-noise regime, where the per-cadence Gaussian likelihood on the
raw light curve is exact, so the IS correction targets the true posterior.
Reports per-parameter SBC p-values before (flow) and after (IS) correction, the
mean IS efficiency, and saves before/after SBC rank histograms for the period.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataclasses import replace
from transitflow.train import load_checkpoint
from transitflow.inference import TransitFlowInference
from transitflow.priors import TransitPrior
from transitflow.simulator import TransitSimulator
from transitflow.evaluation.sbc import sbc_uniformity
from transitflow.correction import importance_weights, weighted_rank_cdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe/checkpoints/best.pt")
    ap.add_argument("--n-obj", type=int, default=400)
    ap.add_argument("--n-post", type=int, default=800)
    ap.add_argument("--logprob-steps", type=int, default=40)
    ap.add_argument("--out", default="results/corrected")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mc, sc = load_checkpoint(args.ckpt)
    sc_white = replace(sc, frac_real=0.0, frac_gp=0.0, frac_white=1.0,
                       planet_fraction=1.0)
    pr = TransitPrior(TransitPrior.default_specs(sc.regime))
    sim = TransitSimulator(sc_white, prior=pr)
    inf = TransitFlowInference(model, pr, sc, ode_steps=args.logprob_steps)
    rng = np.random.default_rng(202)

    names = pr.names
    flow_ranks, corr_ranks, ess = [], [], []
    got = 0
    while got < args.n_obj:
        b = sim.simulate_batch(64, rng, return_raw=True)
        mk = b["valid"]
        for i in np.where(mk)[0]:
            if got >= args.n_obj:
                break
            g, l, sf = b["global"][i], b["local"][i], b["sigma_feat"][i]
            pg = b["periodogram"][i] if "periodogram" in b else None
            eph = b["ephem_feat"][i] if "ephem_feat" in b else None
            tstd = pr.physical_to_std(b["theta_phys"][i][None, :])[0]
            r = importance_weights(inf, g, l, sf, b["raw_flux"][i].astype(np.float64),
                                   b["times"].astype(np.float64), float(b["sigma"][i]),
                                   n_samples=args.n_post,
                                   logprob_steps=args.logprob_steps,
                                   periodogram=pg, ephem_feat=eph)
            # uncorrected (flow): integer rank of truth among proposal samples
            flow_ranks.append((r["std"] < tstd[None, :]).sum(0))
            # corrected: weighted CDF of truth -> scaled to the same [0, N] range
            corr_ranks.append(weighted_rank_cdf(r["std"], r["w"], tstd) * args.n_post)
            ess.append(r["ess_fraction"])
            got += 1
        print(f"  {got}/{args.n_obj} objects  mean_ess={np.mean(ess):.3f}", flush=True)

    flow_ranks = np.array(flow_ranks)
    corr_ranks = np.array(corr_ranks)
    p_flow = sbc_uniformity(flow_ranks)["pvalue"]
    p_corr = sbc_uniformity(corr_ranks.astype(int))["pvalue"]

    report = {
        "n_obj": args.n_obj, "n_post": args.n_post,
        "mean_ess_fraction": round(float(np.mean(ess)), 4),
        "median_ess_fraction": round(float(np.median(ess)), 4),
        "sbc_p_flow": {n: round(float(p), 4) for n, p in zip(names, p_flow)},
        "sbc_p_corrected": {n: round(float(p), 4) for n, p in zip(names, p_corr)},
    }
    json.dump(report, open(os.path.join(args.out, "corrected_metrics.json"), "w"),
              indent=2)
    print(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = list(names).index("P")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for a, ranks, title, p in [
            (ax[0], flow_ranks[:, P], "Period SBC — flow (FMPE)", p_flow[P]),
            (ax[1], corr_ranks[:, P], "Period SBC — IS-corrected", p_corr[P])]:
            a.hist(ranks, bins=20, color="steelblue", alpha=0.85)
            a.axhline(len(ranks) / 20, color="k", ls="--", lw=1)
            a.set_title(f"{title}\n(uniformity p={p:.3f})")
            a.set_xlabel("rank")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "period_sbc_before_after.png"), dpi=130)
        print("wrote period_sbc_before_after.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
