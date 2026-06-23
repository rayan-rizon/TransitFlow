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
            sr = lk.search_lightcurve(tgt, mission=args.mission)
            lc = sr.download_all().stitch().remove_nans().normalize()
            flux = np.asarray(lc.flux.value, dtype=np.float64)
            flux = flux / np.nanmedian(flux)
            # 5-sigma clip the deepest dips so injected transits dominate
            med, std = np.nanmedian(flux), np.nanstd(flux)
            flux = np.clip(flux, med - 5 * std, med + 5 * std)
            # chop into non-overlapping fixed-length segments
            n = args.n_raw
            for s in range(0, len(flux) - n + 1, n):
                seg = flux[s:s + n]
                if np.all(np.isfinite(seg)):
                    segments.append(seg)
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
