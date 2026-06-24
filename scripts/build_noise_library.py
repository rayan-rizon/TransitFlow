#!/usr/bin/env python3
"""Build a real out-of-transit noise library with lightkurve (optional).

Downloads light curves for a list of quiet target stars, removes known
transits/flags, normalizes, and saves fixed-length out-of-transit segments to an
``.npz`` consumable by :class:`transitflow.noise.NoiseLibrary`.  Requires network
access and ``lightkurve``; the rest of TransitFlow runs without it (synthetic GP
noise is used instead).

Example
-------
    python scripts/build_noise_library.py --mission TESS \
        --targets TIC307210830 TIC150428135 --n-raw 18000 --out data/noise_lib.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="TESS", choices=["TESS", "Kepler"])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n-raw", type=int, default=18000)
    ap.add_argument("--out", default="data/noise_lib.npz")
    args = ap.parse_args()

    try:
        import lightkurve as lk  # type: ignore
    except Exception:
        print("lightkurve is not installed. Install with `pip install lightkurve` "
              "to build a real-noise library. TransitFlow runs without it using "
              "synthetic GP noise.")
        sys.exit(1)

    segments = []
    for tgt in args.targets:
        print(f"downloading {tgt} ({args.mission}) ...")
        try:
            # SPOC 2-min cadence products are the most reliable for TESS;
            # FFI-based HLSP products are often in ppm (zero-centered) and
            # can arrive as corrupt cached files that break stitching.
            if args.mission == "TESS":
                sr = lk.search_lightcurve(tgt, mission="TESS", author="SPOC",
                                          exptime=120)
                if len(sr) == 0:
                    sr = lk.search_lightcurve(tgt, mission="TESS", author="SPOC")
                if len(sr) == 0:
                    sr = lk.search_lightcurve(tgt, mission="TESS")
            else:
                sr = lk.search_lightcurve(tgt, mission=args.mission)

            if len(sr) == 0:
                print(f"  skipped {tgt}: no data found")
                continue

            lc_col = sr.download_all()
            if lc_col is None or len(lc_col) == 0:
                print(f"  skipped {tgt}: download returned empty")
                continue

            # Stitch and clean; normalize() divides by median → relative flux ≈ 1
            lc = lc_col.stitch().remove_nans()
            flux_raw = np.asarray(lc.flux.value, dtype=np.float64)
            med = np.nanmedian(flux_raw)
            # Guard against zero-centered ppm data (median ≈ 0)
            if abs(med) < 0.1:
                print(f"  skipped {tgt}: flux appears zero-centered "
                      f"(median={med:.3g}), likely ppm product")
                continue
            flux = flux_raw / med          # bring to relative flux ≈ 1
            flux = flux[np.isfinite(flux)]
            # 5-sigma clip so injected transits dominate hard dips
            m, s = np.nanmedian(flux), np.nanstd(flux)
            flux = np.clip(flux, m - 5 * s, m + 5 * s)
            # chop into non-overlapping fixed-length segments
            n = args.n_raw
            n_segs_before = len(segments)
            for start in range(0, len(flux) - n + 1, n):
                seg = flux[start:start + n]
                if np.all(np.isfinite(seg)):
                    segments.append(seg)
            n_new = len(segments) - n_segs_before
            print(f"  {tgt}: {n_new} segments from {len(flux)} cadences")
        except Exception as e:  # pragma: no cover - network dependent
            print(f"  skipped {tgt}: {e}")

    if not segments:
        print("no segments collected; nothing written.")
        sys.exit(1)
    segments = np.asarray(segments)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, segments=segments)
    print(f"wrote {len(segments)} segments of length {args.n_raw} -> {args.out}")


if __name__ == "__main__":
    main()
