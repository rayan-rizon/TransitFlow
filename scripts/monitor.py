#!/usr/bin/env python3
"""Live training and pipeline monitor for TransitFlow runs.

Reads ``<run_dir>/status.json`` and ``<run_dir>/training_log.jsonl`` and prints a
refreshing dashboard: progress, throughput, ETA, current losses, best validation
AUC, and a loss sparkline. It can also monitor the full publishable pipeline:
process status, data-generation shards, pipeline logs, synthetic gates, and
real/MCMC validation files. Designed to be run over SSH on a Vast.ai box.

Examples
--------
    python scripts/monitor.py --run-dir runs/fmpe --watch        # live, refresh 5s
    python scripts/monitor.py --run-dir runs/fmpe --once         # single snapshot
    python scripts/monitor.py \
        --pid-file /workspace/logs_char5_publishable_v2/pipeline.pid \
        --pipeline-log /workspace/logs_char5_publishable_v2/pipeline.log \
        --data-dir /workspace/data/tess_1M_char5_publishable_v2 \
        --run-dir /workspace/runs/fmpe_pg_char5_publishable_v2 \
        --results-dir /workspace/results_char5_publishable_v2 \
        --expected-shards 1000 --watch
    # or just:  tail -f runs/fmpe/training_log.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

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


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _tail(path, n=30):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, errors="replace") as f:
            return [x.rstrip("\n") for x in f.readlines()[-n:]]
    except Exception:
        return []


def _du(path):
    if not path or not os.path.exists(path):
        return "n/a"
    try:
        out = subprocess.check_output(["du", "-sh", path], text=True, stderr=subprocess.DEVNULL)
        return out.split()[0]
    except Exception:
        return "n/a"


def _pid_state(pid_file):
    if not pid_file or not os.path.exists(pid_file):
        return None, "no pid file", ""
    try:
        pid = int(Path(pid_file).read_text().strip())
    except Exception:
        return None, "bad pid file", ""
    try:
        ps = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pid=,etime=,stat=,pcpu=,pmem=,cmd="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return pid, "dead", ""
    return pid, "alive", ps


def _process_tree(pid):
    if not pid:
        return None
    try:
        raw = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,pcpu=,pmem=,stat=,cmd="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    rows = {}
    children = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        try:
            cpid = int(parts[0])
            ppid = int(parts[1])
            pcpu = float(parts[2])
            pmem = float(parts[3])
        except Exception:
            continue
        row = {
            "pid": cpid,
            "ppid": ppid,
            "pcpu": pcpu,
            "pmem": pmem,
            "stat": parts[4],
            "cmd": parts[5],
        }
        rows[cpid] = row
        children.setdefault(ppid, []).append(cpid)

    seen = set()
    stack = list(children.get(pid, []))
    tree = []
    while stack:
        cpid = stack.pop()
        if cpid in seen:
            continue
        seen.add(cpid)
        row = rows.get(cpid)
        if row:
            tree.append(row)
        stack.extend(children.get(cpid, []))
    if not tree:
        return {"count": 0, "cpu": 0.0, "mem": 0.0, "top": []}
    return {
        "count": len(tree),
        "cpu": sum(r["pcpu"] for r in tree),
        "mem": sum(r["pmem"] for r in tree),
        "top": sorted(tree, key=lambda r: r["pcpu"], reverse=True)[:3],
    }


def _gpu_line():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def _df_line(path):
    target = path if path and os.path.exists(path) else "/workspace"
    try:
        out = subprocess.check_output(["df", "-h", target], text=True, stderr=subprocess.DEVNULL)
        return out.splitlines()[-1]
    except Exception:
        return None


def _gate_bool(value):
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    if value is None:
        return "n/a"
    return str(value)


def _fmt_int(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _read_gate_summary(results_dir):
    if not results_dir:
        return None
    return _read_json(os.path.join(results_dir, "GATE_SUMMARY.json"))


def _read_eval_metrics(results_dir):
    if not results_dir:
        return None
    candidates = [
        os.path.join(results_dir, "eval", "metrics.json"),
        os.path.join(results_dir, "eval_latest", "metrics.json"),
        os.path.join(results_dir, "metrics.json"),
    ]
    for path in candidates:
        data = _read_json(path)
        if data is not None:
            data["_path"] = path
            return data
    return None


def _read_real_summaries(results_dir):
    if not results_dir:
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "real*", "real_validation.json"))):
        data = _read_json(path)
        if data is None:
            continue
        out.append((path, data))
    return out


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
    prog = max(0.0, min(1.0, prog))
    filled = int(prog * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    out.append(f" status   : {st.get('status','?')}   head={st.get('head','?')}  "
               f"device={st.get('device','?')}  params={_fmt_int(st.get('params','?'))}")
    out.append(f" progress : [{bar}] {prog*100:5.1f}%  "
               f"step {st.get('step','?')}/{st.get('total_steps','?')}")
    out.append(f" speed    : {st.get('throughput_lc_per_s',0):.0f} lc/s   "
               f"elapsed {st.get('elapsed_s',0)/60:.1f} min   eta {st.get('eta','?')}")
    loss = st.get("loss", {})
    out.append(f" loss     : total {loss.get('total',float('nan')):.4f}  "
               f"posterior {loss.get('posterior',float('nan')):.4f}  "
               f"detection {loss.get('detection',float('nan')):.4f}   "
               f"lr {st.get('lr',0):.2e}")

    # --- health verdict (catches a wasteful / dead run) ---
    import math
    import time as _t
    verdict, flags = "HEALTHY", []
    age = _t.time() - st.get("updated_unix", 0)
    total = loss.get("total")
    wait = st.get("data_wait_frac", 0.0)
    if st.get("status") in ("done", "interrupted", "error"):
        verdict = st["status"].upper()
    elif age > 180:
        verdict = "STALLED"; flags.append(f"no update for {int(age)}s")
    elif total is None or not math.isfinite(total):
        verdict = "DIVERGED"; flags.append("non-finite loss")
    else:
        flags = list(st.get("health", {}).get("warnings", []))
        if wait and wait > 0.35:
            flags.append(f"GPU-starved {wait*100:.0f}%")
        verdict = "WARN" if flags else "HEALTHY"
    out.append(f" HEALTH   : {verdict}" + (f"  ({'; '.join(flags)})" if flags else ""))
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


def render_pipeline(args) -> str:
    out = []
    out.append("=" * 78)
    out.append(" TransitFlow pipeline monitor")
    out.append("=" * 78)

    pid, state, ps = _pid_state(args.pid_file)
    out.append(f" process  : {state}" + (f" pid={pid}" if pid else ""))
    if ps:
        out.append(f" ps       : {ps}")
    tree = _process_tree(pid)
    if tree:
        out.append(
            f" children : {tree['count']} proc  cpu={tree['cpu']:.0f}%  mem={tree['mem']:.1f}%"
        )
        for row in tree["top"]:
            cmd = row["cmd"]
            if len(cmd) > 74:
                cmd = cmd[:71] + "..."
            out.append(f"   top    : {row['pid']} {row['pcpu']:.0f}% {row['stat']} {cmd}")
    gpu = _gpu_line()
    if gpu:
        out.append(f" gpu      : util%,mem.used,total = {gpu}")
    df = _df_line(args.data_dir or args.results_dir or args.run_dir)
    if df:
        out.append(f" disk     : {df}")

    if args.data_dir:
        shards = len(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
        tmp = len(glob.glob(os.path.join(args.data_dir, "*.tmp")))
        if args.expected_shards:
            pct = 100.0 * shards / max(args.expected_shards, 1)
            out.append(
                f" data     : {shards}/{args.expected_shards} shards "
                f"({pct:.1f}%)  tmp={tmp}  size={_du(args.data_dir)}"
            )
        else:
            out.append(f" data     : {shards} shards  tmp={tmp}  size={_du(args.data_dir)}")

    if args.run_dir:
        st = _read_status(args.run_dir)
        if st is None:
            out.append(" train    : status.json not found yet")
        else:
            prog = float(st.get("progress", 0.0))
            prog = max(0.0, min(1.0, prog))
            loss = st.get("loss", {})
            out.append(
                f" train    : {st.get('status','?')} step "
                f"{st.get('step','?')}/{st.get('total_steps','?')} "
                f"({prog*100:.1f}%) {st.get('throughput_lc_per_s',0):.0f} lc/s "
                f"eta={st.get('eta','?')}"
            )
            out.append(
                f" loss     : total={loss.get('total',float('nan')):.4f} "
                f"post={loss.get('posterior',float('nan')):.4f} "
                f"det={loss.get('detection',float('nan')):.4f} lr={st.get('lr',0):.2e}"
            )
            best = st.get("best", {})
            if best and best.get("step", -1) >= 0:
                out.append(
                    f" best     : AUC={best.get('roc_auc',float('nan')):.4f} "
                    f"step={best.get('step','?')}"
                )

    summary = _read_gate_summary(args.results_dir)
    if summary:
        out.append("-" * 78)
        out.append(" gate summary")
        for key in (
            "synthetic_required_pass",
            "real_required_pass",
            "real_mcmc_required_pass",
            "all_required_pass",
        ):
            if key in summary:
                out.append(f" {key:28s}: {_gate_bool(summary.get(key))}")
        for key, value in summary.items():
            if key.endswith("_pass") and key not in {
                "synthetic_required_pass",
                "real_required_pass",
                "real_mcmc_required_pass",
                "all_required_pass",
            }:
                out.append(f" {key:28s}: {_gate_bool(value)}")

    metrics = _read_eval_metrics(args.results_dir)
    if metrics:
        gates = metrics.get("gates", {})
        post_names = metrics.get("posterior_param_names") or metrics.get("param_names")
        out.append("-" * 78)
        out.append(f" synthetic: {metrics.get('_path')}")
        if "detection_auc" in metrics:
            out.append(
                f" auc/ap   : {metrics.get('detection_auc',float('nan')):.5f} / "
                f"{metrics.get('detection_ap',float('nan')):.5f}"
            )
        if post_names:
            out.append(f" params   : {post_names}")
        for key in (
            "detection_auc_ge_0.99",
            "posterior_sbc_p_gt_0.05",
            "characterization_sbc_p_gt_0.05",
            "characterization_coverage_error_le_0.03",
        ):
            if key in gates:
                out.append(f" {key:42s}: {_gate_bool(gates.get(key))}")

    real = _read_real_summaries(args.results_dir)
    if real:
        out.append("-" * 78)
        out.append(" real validation")
        for path, data in real:
            name = Path(path).parent.name
            gates = data.get("gates", {})
            det = data.get("detection", {})
            mcmc = data.get("mcmc", {})
            bits = []
            if "detected_fraction" in det:
                bits.append(f"det={det.get('detected_fraction'):.3f}")
            for key in ("detection_fraction_ge_0.90", "posterior_width_fraction_le_0.10"):
                if key in gates:
                    bits.append(f"{key}={_gate_bool(gates.get(key))}")
            if mcmc:
                bits.append(f"mcmc={_gate_bool(gates.get('mcmc_width_fraction_le_0.10'))}")
            out.append(f" {name:24s}: " + "  ".join(bits))

    tail = _tail(args.pipeline_log, args.tail_lines)
    if tail:
        out.append("-" * 78)
        out.append(f" log tail : {args.pipeline_log}")
        out.extend(tail)

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--pid-file")
    ap.add_argument("--pipeline-log")
    ap.add_argument("--data-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--expected-shards", type=int)
    ap.add_argument("--tail-lines", type=int, default=35)
    ap.add_argument("--watch", action="store_true", help="refresh continuously")
    ap.add_argument("--once", action="store_true", help="print one snapshot")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    pipeline_mode = any((args.pid_file, args.pipeline_log, args.data_dir, args.results_dir))
    if not args.run_dir and not pipeline_mode:
        ap.error("--run-dir is required unless pipeline options are provided")

    if args.once or not args.watch:
        print(render_pipeline(args) if pipeline_mode else render(args.run_dir))
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(render_pipeline(args) if pipeline_mode else render(args.run_dir))
            st = _read_status(args.run_dir) if args.run_dir else None
            if pipeline_mode:
                _, state, _ = _pid_state(args.pid_file)
                if state == "dead":
                    print("\n[pipeline process dead]")
                    break
            elif st and st.get("status") in ("done", "interrupted", "error"):
                print(f"\n[run {st.get('status')}]")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
