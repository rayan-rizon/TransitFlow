#!/usr/bin/env python3
"""Live training monitor for a TransitFlow run directory.

Reads ``<run_dir>/status.json`` and ``<run_dir>/training_log.jsonl`` and prints a
refreshing dashboard: progress, throughput, ETA, current losses, best validation
AUC, and a loss sparkline. Designed to be run over SSH on a Vast.ai box.

Examples
--------
    python scripts/monitor.py --run-dir runs/fmpe --watch        # live, refresh 5s
    python scripts/monitor.py --run-dir runs/fmpe --once         # single snapshot
    # or just:  tail -f runs/fmpe/training_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values, width=48):
    if not values:
        return ""
    v = values[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return _SPARK[0] * len(v)
    return "".join(_SPARK[min(int((x - lo) / (hi - lo) * (len(_SPARK) - 1)),
                              len(_SPARK) - 1)] for x in v)


def _read_status(run_dir):
    p = os.path.join(run_dir, "status.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _read_log(run_dir, n=400):
    p = os.path.join(run_dir, "training_log.jsonl")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            lines = f.readlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]
    except Exception:
        return []


def render(run_dir) -> str:
    st = _read_status(run_dir)
    log = _read_log(run_dir)
    out = []
    out.append("=" * 64)
    out.append(f" TransitFlow monitor — {run_dir}")
    out.append("=" * 64)
    if st is None:
        out.append(" status.json not found yet (training may be starting) …")
        return "\n".join(out)
    bar_w = 40
    prog = float(st.get("progress", 0.0))
    filled = int(prog * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    out.append(f" status   : {st.get('status','?')}   head={st.get('head','?')}  "
               f"device={st.get('device','?')}  params={st.get('params','?'):,}")
    out.append(f" progress : [{bar}] {prog*100:5.1f}%  "
               f"step {st.get('step','?')}/{st.get('total_steps','?')}")
    out.append(f" speed    : {st.get('throughput_lc_per_s',0):.0f} lc/s   "
               f"elapsed {st.get('elapsed_s',0)/60:.1f} min   eta {st.get('eta','?')}")
    loss = st.get("loss", {})
    out.append(f" loss     : total {loss.get('total',float('nan')):.4f}  "
               f"posterior {loss.get('posterior',float('nan')):.4f}  "
               f"detection {loss.get('detection',float('nan')):.4f}   "
               f"lr {st.get('lr',0):.2e}")
    best = st.get("best", {})
    if best and best.get("step", -1) >= 0:
        out.append(f" best val : AUC {best.get('roc_auc',float('nan')):.4f} "
                   f"@ step {best.get('step','?')}  "
                   f"(acc {best.get('det_acc',float('nan')):.3f})")
    if log:
        tot = [r.get("total") for r in log if r.get("total") is not None]
        post = [r.get("posterior") for r in log if r.get("posterior") is not None]
        out.append(f" total    : {_sparkline(tot)}")
        out.append(f" posterior: {_sparkline(post)}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--watch", action="store_true", help="refresh continuously")
    ap.add_argument("--once", action="store_true", help="print one snapshot")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    if args.once or not args.watch:
        print(render(args.run_dir))
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(render(args.run_dir))
            st = _read_status(args.run_dir)
            if st and st.get("status") in ("done", "interrupted", "error"):
                print(f"\n[run {st.get('status')}]")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
