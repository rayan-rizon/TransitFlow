#!/usr/bin/env python3
"""Prefetch real TESS light curves used by validate_real.py.

This warms the lightkurve/astropy cache before the real-validation stage. It is
intentionally separate from the gate runner: prefetch failures do not alter the
publishability test, and validate_real.py still applies its normal quality gates.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transitflow.priors import TransitPrior  # noqa: E402
from scripts.validate_real import _download_lc_impl, query_planets  # noqa: E402


def _worker(planet: dict, baseline_days: float, out_queue) -> None:
    t0 = time.time()
    try:
        lc = _download_lc_impl(planet, baseline_days)
        if lc is None:
            out_queue.put({
                "name": planet.get("name"),
                "host": planet.get("host"),
                "ok": False,
                "reason": "none",
                "seconds": round(time.time() - t0, 2),
            })
            return
        t, _f = lc
        out_queue.put({
            "name": planet.get("name"),
            "host": planet.get("host"),
            "ok": True,
            "n_cadences": int(len(t)),
            "seconds": round(time.time() - t0, 2),
        })
    except Exception as exc:
        out_queue.put({
            "name": planet.get("name"),
            "host": planet.get("host"),
            "ok": False,
            "reason": repr(exc),
            "seconds": round(time.time() - t0, 2),
        })


def _write_result(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-planets", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--baseline-days", type=float, default=27.0)
    ap.add_argument("--out-dir", default="results/real_prefetch")
    args = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    pool_path = out_dir / "planet_pool.json"

    specs = TransitPrior.default_specs("tess")
    pool = query_planets(
        args.n_planets,
        specs[0].low,
        specs[0].high,
        specs[2].low,
        specs[2].high,
    )
    planets = []
    seen = set()
    for planet in pool:
        key = (planet.get("name"), planet.get("host"))
        if key in seen:
            continue
        seen.add(key)
        planets.append(planet)
    pool_path.write_text(json.dumps(planets, indent=2, default=str))
    print(
        f"prefetch pool={len(planets)} workers={args.workers} "
        f"timeout={args.timeout_s}s",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    pending = list(planets)
    active = []
    done = 0
    ok = 0
    timed_out = 0

    while pending or active:
        while pending and len(active) < args.workers:
            planet = pending.pop(0)
            out_queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_worker,
                args=(planet, args.baseline_days, out_queue),
                daemon=True,
            )
            proc.start()
            active.append({
                "process": proc,
                "queue": out_queue,
                "planet": planet,
                "start": time.time(),
            })
            print(
                f"start {planet.get('name')} host={planet.get('host')}",
                flush=True,
            )

        keep = []
        for item in active:
            proc = item["process"]
            out_queue = item["queue"]
            planet = item["planet"]
            try:
                row = out_queue.get_nowait()
            except queue.Empty:
                elapsed = time.time() - item["start"]
                if elapsed > args.timeout_s:
                    proc.terminate()
                    proc.join(timeout=5)
                    row = {
                        "name": planet.get("name"),
                        "host": planet.get("host"),
                        "ok": False,
                        "reason": "timeout",
                        "seconds": round(elapsed, 2),
                    }
                    timed_out += 1
                    done += 1
                    _write_result(results_path, row)
                elif not proc.is_alive():
                    proc.join(timeout=5)
                    row = {
                        "name": planet.get("name"),
                        "host": planet.get("host"),
                        "ok": False,
                        "reason": f"exitcode={proc.exitcode}",
                        "seconds": round(elapsed, 2),
                    }
                    done += 1
                    _write_result(results_path, row)
                else:
                    keep.append(item)
                continue
            proc.join(timeout=5)
            done += 1
            ok += int(bool(row.get("ok")))
            _write_result(results_path, row)
        active = keep
        print(
            f"progress done={done}/{len(planets)} ok={ok} timeout={timed_out}",
            flush=True,
        )
        time.sleep(2)

    print(f"finished done={done} ok={ok} timeout={timed_out}", flush=True)


if __name__ == "__main__":
    main()
