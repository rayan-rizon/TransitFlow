#!/usr/bin/env python3
"""Build a quality-screened quiet-target noise library with lightkurve.

Downloads light curves for a list of quiet target stars, removes non-finite
cadences, clips extreme excursions, normalizes, and saves fixed-length segments
to an
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
import hashlib
import json
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


def robust_point_to_point_ppm(flux: np.ndarray) -> float:
    """Robust cadence-scale scatter proxy, in parts per million."""
    diff = np.diff(np.asarray(flux, dtype=np.float64))
    if len(diff) == 0:
        return float("inf")
    centered = diff - np.nanmedian(diff)
    sigma_diff = 1.4826 * np.nanmedian(np.abs(centered))
    return float(1e6 * sigma_diff / np.sqrt(2.0))


def _collect_target_segments(
        tgt: str, mission: str, n_raw: int, max_segments: int,
        max_point_to_point_ppm: float) \
        -> tuple[str, list[np.ndarray], list[str], dict]:
    logs = [f"downloading {tgt} ({mission}) ..."]
    segments: list[np.ndarray] = []
    metrics = {"target": tgt, "accepted": False}
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
            return tgt, segments, logs, metrics

        lc_col = sr.download_all()
        if lc_col is None or len(lc_col) == 0:
            logs.append(f"  skipped {tgt}: download returned empty")
            return tgt, segments, logs, metrics

        # Stitch and clean; normalize() divides by median -> relative flux ~= 1
        lc = lc_col.stitch().remove_nans()
        flux_raw = np.asarray(lc.flux.value, dtype=np.float64)
        med = np.nanmedian(flux_raw)
        # Guard against zero-centered ppm data (median ~= 0)
        if abs(med) < 0.1:
            logs.append(f"  skipped {tgt}: flux appears zero-centered "
                        f"(median={med:.3g}), likely ppm product")
            return tgt, segments, logs, metrics
        flux = flux_raw / med
        flux = flux[np.isfinite(flux)]
        # Limit extreme excursions. This is not a planet-catalog transit mask;
        # target-level quietness and robust scatter gates provide the primary
        # contamination control, and their provenance is recorded separately.
        m, s = np.nanmedian(flux), np.nanstd(flux)
        flux = np.clip(flux, m - 5 * s, m + 5 * s)
        point_to_point_ppm = robust_point_to_point_ppm(flux)
        metrics.update({
            "n_cadences": int(len(flux)),
            "point_to_point_ppm": point_to_point_ppm,
        })
        if (not np.isfinite(point_to_point_ppm)
                or point_to_point_ppm > max_point_to_point_ppm):
            logs.append(
                f"  skipped {tgt}: point-to-point scatter "
                f"{point_to_point_ppm:.0f} ppm exceeds "
                f"{max_point_to_point_ppm:.0f} ppm")
            return tgt, segments, logs, metrics
        for start in range(0, len(flux) - n_raw + 1, n_raw):
            seg = flux[start:start + n_raw]
            if np.all(np.isfinite(seg)):
                segments.append(seg)
                if len(segments) >= max_segments:
                    break
        metrics.update({"accepted": bool(segments),
                        "n_segments": int(len(segments))})
        logs.append(f"  {tgt}: {len(segments)} segments from {len(flux)} cadences")
        return tgt, segments, logs, metrics
    except Exception as e:  # pragma: no cover - network dependent
        logs.append(f"  skipped {tgt}: {e}")
        metrics["error"] = str(e)
        return tgt, segments, logs, metrics


def _sha256(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="TESS", choices=["TESS", "Kepler"])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n-raw", type=int, default=18000)
    ap.add_argument("--out", default="data/noise_lib.npz")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel target downloads; keep modest to avoid archive throttling")
    ap.add_argument("--min-targets", type=int, default=1,
                    help="fail closed unless this many independent targets succeed")
    ap.add_argument("--max-segments-per-target", type=int, default=8)
    ap.add_argument("--max-point-to-point-ppm", type=float, default=2500.0)
    ap.add_argument("--target-provenance", default=None)
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
    quality: list[dict] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for tgt in args.targets:
            _, target_segments, logs, metrics = _collect_target_segments(
                tgt, args.mission, args.n_raw, args.max_segments_per_target,
                args.max_point_to_point_ppm)
            emit_logs(logs)
            quality.append(metrics)
            segments.extend(target_segments)
            target_ids.extend([tgt] * len(target_segments))
    else:
        results: dict[str, tuple[list[np.ndarray], list[str], dict]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _collect_target_segments, tgt, args.mission, args.n_raw,
                    args.max_segments_per_target, args.max_point_to_point_ppm): tgt
                for tgt in args.targets
            }
            for fut in as_completed(futures):
                tgt, target_segments, logs, metrics = fut.result()
                results[tgt] = (target_segments, logs, metrics)
        for tgt in args.targets:
            target_segments, logs, metrics = results.get(
                tgt, ([], [f"  skipped {tgt}: worker produced no result"],
                      {"target": tgt, "accepted": False,
                       "error": "worker produced no result"}))
            emit_logs(logs)
            quality.append(metrics)
            segments.extend(target_segments)
            target_ids.extend([tgt] * len(target_segments))

    if not segments:
        emit_logs(["no segments collected; nothing written."])
        sys.exit(1)
    successful_targets = sorted(set(target_ids))
    if len(successful_targets) < int(args.min_targets):
        emit_logs([
            f"only {len(successful_targets)} targets passed; "
            f"minimum is {args.min_targets}; nothing written."])
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
    metadata_path = args.out + ".metadata.json"
    metadata_tmp = metadata_path + ".tmp"
    with open(metadata_tmp, "w") as fh:
        json.dump({
            "mission": args.mission,
            "n_raw": int(args.n_raw),
            "requested_targets": list(args.targets),
            "successful_targets": successful_targets,
            "n_successful_targets": len(successful_targets),
            "n_segments": len(segments),
            "max_segments_per_target": int(args.max_segments_per_target),
            "max_point_to_point_ppm": float(args.max_point_to_point_ppm),
            "target_provenance": args.target_provenance,
            "target_provenance_sha256": _sha256(args.target_provenance),
            "quality": quality,
        }, fh, indent=2, sort_keys=True)
    os.replace(metadata_tmp, metadata_path)
    emit_logs([f"wrote {len(segments)} segments of length {args.n_raw} -> {args.out}"])


if __name__ == "__main__":
    main()
