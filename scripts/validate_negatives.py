#!/usr/bin/env python3
"""Real-data NEGATIVE-class validation: specificity / precision / FP rate.

``validate_real.py`` queries confirmed transiting planets and measures how many
the detector recovers -- that is *sensitivity* on known positives and cannot,
by itself, establish precision, specificity, or the false-positive rate. This
script supplies the matched negative class: real TESS objects with a TFOPWG
false-positive / false-alarm disposition (eclipsing binaries, systematics,
stellar variability), in the *same* period box and, via transit depth, the same
Rp/Rs box as the positives, run through the *identical* light-curve preparation
and detector so the two classes are scored on the same footing.

Outputs
-------
* per-negative detection probability + a checkpoint of records,
* false-positive count / false-positive rate / specificity at the operating
  threshold, and
* when ``--positives-records`` (the positive run's records_checkpoint.json) is
  supplied, precision, recall, F1 and a small threshold sweep -- the numbers a
  detection paper needs to claim more than sensitivity.

A predeclared specificity gate (``--min-specificity``, default 0.80) makes the
false-positive behavior a pass/fail line rather than a number chosen after the
fact. As with every gate in the runbook it is set before the run and must not
be relaxed to make a run pass.

Design notes
------------
* False positives have no planet geometry, so the ``finite_geometry`` cut in
  ``validate_real.passes_real_quality`` would reject the entire negative class.
  The data-quality cuts that *are* meaningful for a negative (cadence coverage,
  in-"transit" sampling at the reported ephemeris, transit count, observed S/N)
  are applied via :func:`passes_negative_quality`; the geometry cut is dropped.
* Rp/Rs for a TOI is derived from the reported transit depth
  (Rp/Rs = sqrt(depth_ppm / 1e6)) so the same [rprs_lo, rprs_hi] box that bounds
  the positives also bounds the negatives -- a depth-matched negative class.
* Heavy imports (torch via transitflow, astroquery, lightkurve) are lazy so the
  pure metric core is unit-testable without them.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classification_metrics(pos_pdet: list[float], neg_pdet: list[float],
                           threshold: float) -> dict:
    """Binary-classification metrics from positive/negative detection scores.

    Pure function (stdlib only). ``pos_pdet``/``neg_pdet`` are detector
    probabilities for known positives and known negatives; a source is
    "flagged" when its probability is >= ``threshold``.
    """
    pos = [p for p in pos_pdet if p is not None and math.isfinite(p)]
    neg = [p for p in neg_pdet if p is not None and math.isfinite(p)]
    tp = sum(1 for p in pos if p >= threshold)
    fn = len(pos) - tp
    fp = sum(1 for p in neg if p >= threshold)
    tn = len(neg) - fp

    def _safe_div(a, b):
        return (a / b) if b else None

    sensitivity = _safe_div(tp, tp + fn)      # recall / TPR
    specificity = _safe_div(tn, tn + fp)      # TNR
    fpr = _safe_div(fp, fp + tn)
    precision = _safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and sensitivity is not None and \
            (precision + sensitivity) > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return {
        "threshold": threshold,
        "n_positives": len(pos),
        "n_negatives": len(neg),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
    }


def passes_negative_quality(q: dict, args) -> bool:
    """Data-quality gate for negatives: validate_real's cuts minus geometry.

    A false positive has no planet geometry, so ``finite_geometry`` is dropped;
    every other data cut (which is about whether the light curve is well enough
    sampled to be judged at all) is kept identical to the positive path.
    """
    if not args.quality_gate:
        return True
    return (
        q["n_cadences"] >= args.min_cadences
        and q["cadence_fraction_of_training"] >= args.min_cadence_fraction
        and q["n_in_transit"] >= args.min_in_transit
        and q["n_transits"] >= args.min_transits
        and q["observed_snr"] >= args.min_observed_snr
    )


def query_false_positives(n: int, p_lo: float, p_hi: float,
                          rprs_lo: float, rprs_hi: float,
                          seed: int = 0) -> list[dict]:
    """TESS TOIs with a TFOPWG false-positive/false-alarm disposition.

    Filtered to the training period box and, via the reported transit depth,
    to the training Rp/Rs box, so the negative class is period- and
    depth-matched to the positives. Returns a shuffled, depth-stratified pool
    of planet-like dicts consumable by ``validate_real.build_views``.
    """
    import numpy as np
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    from scripts.validate_real import _val

    cols = ("toi,tid,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,"
            "pl_rade,st_rad")
    tab = NasaExoplanetArchive.query_criteria(
        table="toi", select=cols,
        where=(f"pl_orbper > {p_lo} and pl_orbper < {p_hi} and "
               f"(tfopwg_disp = 'FP' or tfopwg_disp = 'FA')"))
    out = []
    for row in tab:
        try:
            P = _val(row, "pl_orbper")
            depth_ppm = _val(row, "pl_trandep")
            if not (np.isfinite(P) and np.isfinite(depth_ppm)):
                continue
            rprs = math.sqrt(max(depth_ppm, 0.0) / 1e6)
            if not (rprs_lo < rprs < rprs_hi):
                continue
            dur_h = _val(row, "pl_trandurh")
            tid = row["tid"]
            out.append({
                "name": f"TOI-{row['toi']}",
                "host": f"TIC {int(tid)}",
                "tic": str(tid),
                "P": P,
                "dur_days": dur_h / 24.0 if np.isfinite(dur_h) else float("nan"),
                "RpRs": rprs,
                "aRs": float("nan"),   # no planet geometry for a false positive
                "b": float("nan"),
                "disposition": str(row["tfopwg_disp"]),
            })
        except Exception:
            continue
    if not out:
        return out
    out.sort(key=lambda d: d["RpRs"])
    pool_size = min(len(out), 6 * n)
    idx = sorted({int(i) for i in
                  np.linspace(0, len(out) - 1, pool_size).round()})
    pool = [out[i] for i in idx]
    np.random.default_rng(seed).shuffle(pool)
    return pool


def _load_positive_pdet(path: str) -> list[float]:
    with open(path) as fh:
        recs = json.load(fh)
    return [float(r["p_detect"]) for r in recs
            if isinstance(r, dict) and r.get("p_detect") is not None]


def main() -> None:
    import numpy as np
    from transitflow.inference import TransitFlowInference
    from transitflow.calibration import load_for_checkpoint
    from transitflow.priors import TransitPrior
    from transitflow.simulator import TransitSimulator
    from transitflow.train import load_checkpoint
    from transitflow.utils import set_seed
    from scripts.validate_real import (
        download_lc, build_views, real_quality_metrics, safe_print)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fmpe_pg/checkpoints/latest.pt")
    ap.add_argument("--detector-ckpt", default=None)
    ap.add_argument("--n-negatives", type=int, default=30)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="operating detection threshold (match validate_real)")
    ap.add_argument("--min-specificity", type=float, default=0.80,
                    help="predeclared specificity gate; do not relax post hoc")
    ap.add_argument("--positives-records", default=None,
                    help="positive run's records_checkpoint.json -> precision/F1")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--quality-gate", dest="quality_gate", action="store_true",
                    default=True)
    ap.add_argument("--no-quality-gate", dest="quality_gate",
                    action="store_false")
    ap.add_argument("--min-cadences", type=int, default=5000)
    ap.add_argument("--min-cadence-fraction", type=float, default=0.70)
    ap.add_argument("--min-in-transit", type=int, default=50)
    ap.add_argument("--min-transits", type=int, default=2)
    ap.add_argument("--min-observed-snr", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="results/real_negatives")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    model, mc, sc = load_checkpoint(args.ckpt)
    prior = TransitPrior(
        TransitPrior.default_specs(sc.regime),
        a_rs_prior_mode=getattr(sc, "a_rs_prior_mode", "log_uniform"),
        stellar_density_log10_mean=getattr(sc, "stellar_density_log10_mean", 0.0),
        stellar_density_log10_std=getattr(sc, "stellar_density_log10_std", 0.25),
    )
    sim = TransitSimulator(sc, prior=prior)
    calibration = load_for_checkpoint(args.calibration, args.ckpt) \
        if args.calibration else None
    inf = TransitFlowInference(model, prior, sc, amp=args.amp,
                               calibration=calibration)
    detector_inf = inf
    if args.detector_ckpt:
        detector_model, _, detector_sc = load_checkpoint(args.detector_ckpt)
        if detector_sc != sc:
            raise SystemExit("--detector-ckpt must use the same simulator config")
        detector_inf = TransitFlowInference(detector_model, prior, sc,
                                            amp=args.amp)
    p_lo, p_hi = prior.specs[0].low, prior.specs[0].high
    rprs_lo, rprs_hi = prior.specs[2].low, prior.specs[2].high

    print(f"== querying TOI false positives (P in [{p_lo}, {p_hi}] d, "
          f"Rp/Rs in [{rprs_lo}, {rprs_hi}], TFOPWG FP/FA) ==")
    pool = query_false_positives(args.n_negatives, p_lo, p_hi,
                                 rprs_lo, rprs_hi, seed=args.seed)
    print(f"   {len(pool)} candidate false positives")

    records = []
    for fp in pool:
        if len(records) >= args.n_negatives:
            break
        try:
            lc = download_lc(fp, sc.baseline_days)
            if lc is None:
                continue
            t, f = lc
            gv, lv, pg, sf, eph, raw = build_views(t, f, fp, sim, prior)
            quality = real_quality_metrics(t, fp, raw, sc)
            if not passes_negative_quality(quality, args):
                safe_print(f"   skip {fp['name']}: quality {quality}")
                continue
            p_detect = detector_inf.detect(gv, lv, np.array([sf]),
                                           periodogram=pg, ephem_feat=eph)
            rec = {"name": fp["name"], "disposition": fp["disposition"],
                   "p_detect": float(p_detect[0]), "quality": quality}
            records.append(rec)
            safe_print(f"   [{len(records):2d}] {fp['name']:<14} "
                       f"({fp['disposition']}) p_det={rec['p_detect']:.3f}")
        except Exception as e:
            safe_print(f"   skip {fp.get('name', '?')}: {e}")
            continue

    with open(os.path.join(args.out, "negative_records.json"), "w") as fh:
        json.dump(records, fh, indent=2, default=str)

    neg_pdet = [r["p_detect"] for r in records]
    pos_pdet = (_load_positive_pdet(args.positives_records)
                if args.positives_records else [])

    metrics = classification_metrics(pos_pdet, neg_pdet, args.threshold)
    # threshold sweep for the ROC/PR context the paper reports
    sweep = [classification_metrics(pos_pdet, neg_pdet, thr)
             for thr in (0.5, 0.7, 0.9, 0.95, 0.99)]

    spec = metrics["specificity"]
    gate = {
        "specificity_ge_min": bool(spec is not None
                                   and spec >= args.min_specificity),
        "min_specificity": args.min_specificity,
        "n_negatives_ge_10": bool(metrics["n_negatives"] >= 10),
    }
    result = {
        "checkpoint": args.ckpt,
        "threshold": args.threshold,
        "seed": args.seed,
        "metrics": metrics,
        "threshold_sweep": sweep,
        "gate_status": gate,
        "pass": bool(gate["specificity_ge_min"] and gate["n_negatives_ge_10"]),
    }
    with open(os.path.join(args.out, "negative_validation.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result["metrics"], indent=2))
    print(f"specificity={spec} (gate >= {args.min_specificity}) -> "
          f"pass={result['pass']}")


if __name__ == "__main__":
    main()
