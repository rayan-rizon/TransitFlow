#!/usr/bin/env python3
"""Build a real out-of-transit noise library with lightkurve (optional).

Downloads light curves for a list of quiet target stars, removes known
transits/flags, normalizes, and saves fixed-length out-of-transit segments to an
``.npz`` consumable by :class:`transitflow.noise.NoiseLibrary`.  Requires network
access and ``lightkurve``; the rest of TransitFlow runs without it (synthetic GP
noise is used instead).

Example
-------
    python3 scripts/build_noise_library.py --mission TESS \
        --targets TIC307210830 TIC150428135 --n-raw 18000 --out data/noise_lib.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def emit_logs(lines: list[str], primary=None, fallback=None) -> None:
    """Write collected worker logs even if a downloader closed ``sys.stdout``.

    Some lightkurve/astroquery download progress paths replace or close the
    process-standard stream when called concurrently.  Worker logs are emitted
    only after all futures finish, so falling back to ``sys.__stdout__`` keeps
    a successful parallel archive build from failing at its final print.
    """
    text = "\n".join(lines) + "\n"
    primary = sys.stdout if primary is None else primary
    fallback = sys.__stdout__ if fallback is None else fallback
    try:
        primary.write(text)
        primary.flush()
    except (AttributeError, BrokenPipeError, ValueError):
        if fallback is None or fallback is primary:
            raise
        fallback.write(text)
        fallback.flush()


def _collect_target_segments(tgt: str, mission: str, n_raw: int) -> tuple[str, list[np.ndarray], list[str]]:
    logs = [f"downloading {tgt} ({mission}) ..."]
    segments: list[np.ndarray] = []
    try:
        import lightkurve as lk  # type: ignore

        # SPOC 2-min cadence products are the most reliable for TESS;
        # FFI-based HLSP products are often in ppm (zero-centered) and
        # can arrive as corrupt cached files that break stitching.
        if mission == "TESS":
            sr = lk.search_lightcurve(tgt, mission="TESS", author="SPOC",
                                      exptime=120)
            if len(sr) == 0:
                sr = lk.search_lightcurve(tgt, mission="TESS", author="SPOC")
            if len(sr) == 0:
                sr = lk.search_lightcurve(tgt, mission="TESS")
        else:
            sr = lk.search_lightcurve(tgt, mission=mission)

        if len(sr) == 0:
            logs.append(f"  skipped {tgt}: no data found")
            return tgt, segments, logs

        lc_col = sr.download_all()
        if lc_col is None or len(lc_col) == 0:
            logs.append(f"  skipped {tgt}: download returned empty")
            return tgt, segments, logs

        # Stitch and clean; normalize() divides by median -> relative flux ~= 1
        lc = lc_col.stitch().remove_nans()
        flux_raw = np.asarray(lc.flux.value, dtype=np.float64)
        med = np.nanmedian(flux_raw)
        # Guard against zero-centered ppm data (median ~= 0)
        if abs(med) < 0.1:
            logs.append(f"  skipped {tgt}: flux appears zero-centered "
                        f"(median={med:.3g}), likely ppm product")
            return tgt, segments, logs
        flux = flux_raw / med
        flux = flux[np.isfinite(flux)]
        # 5-sigma clip so injected transits dominate hard dips
        m, s = np.nanmedian(flux), np.nanstd(flux)
        flux = np.clip(flux, m - 5 * s, m + 5 * s)
        for start in range(0, len(flux) - n_raw + 1, n_raw):
            seg = flux[start:start + n_raw]
            if np.all(np.isfinite(seg)):
                segments.append(seg)
        logs.append(f"  {tgt}: {len(segments)} segments from {len(flux)} cadences")
        return tgt, segments, logs
    except Exception as e:  # pragma: no cover - network dependent
        logs.append(f"  skipped {tgt}: {e}")
        return tgt, segments, logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="TESS", choices=["TESS", "Kepler"])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n-raw", type=int, default=18000)
    ap.add_argument("--out", default="data/noise_lib.npz")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel target downloads; keep modest to avoid archive throttling")
    args = ap.parse_args()

    try:
        import lightkurve as lk  # type: ignore
    except Exception:
        print("lightkurve is not installed. Install with `pip install lightkurve` "
              "to build a real-noise library. TransitFlow runs without it using "
              "synthetic GP noise.")
        sys.exit(1)

    segments = []
    target_ids: list[str] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for tgt in args.targets:
            _, target_segments, logs = _collect_target_segments(
                tgt, args.mission, args.n_raw)
            emit_logs(logs)
            segments.extend(target_segments)
            target_ids.extend([tgt] * len(target_segments))
    else:
        results: dict[str, tuple[list[np.ndarray], list[str]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_collect_target_segments, tgt, args.mission, args.n_raw): tgt
                for tgt in args.targets
            }
            for fut in as_completed(futures):
                tgt, target_segments, logs = fut.result()
                results[tgt] = (target_segments, logs)
        for tgt in args.targets:
            target_segments, logs = results.get(
                tgt, ([], [f"  skipped {tgt}: worker produced no result"]))
            emit_logs(logs)
            segments.extend(target_segments)
            target_ids.extend([tgt] * len(target_segments))

    if not segments:
        emit_logs(["no segments collected; nothing written."])
        sys.exit(1)
    segments = np.asarray(segments)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez_compressed(
        tmp,
        segments=segments,
        target_ids=np.asarray(target_ids, dtype="U128"),
        mission=np.asarray(args.mission),
        segment_length=np.asarray(args.n_raw),
    )
    os.replace(tmp, args.out)
    emit_logs([f"wrote {len(segments)} segments of length {args.n_raw} -> {args.out}"])


if __name__ == "__main__":
    main()
