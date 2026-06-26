"""Training loop for TransitFlow (Variants A and B).

Total loss (Sec. 3.3):  ``L = L_posterior + lambda_det * BCE(detection)``
where ``L_posterior`` is the conditional-flow-matching loss (FMPE) or the NPE
negative log-likelihood, trained on ``d=1`` rows only (masked), and the
detection BCE is trained on both classes.  Data are simulated on the fly.

Production features for long unattended (e.g. Vast.ai) runs:

* **Run directory** (`run_dir`) holding `checkpoints/`, `training_log.jsonl`,
  `status.json`, and a `config.yaml` snapshot — everything needed to monitor and
  resume.
* **Periodic checkpoints** (`ckpt_every`) writing `latest.pt` + rotating
  `step_*.pt`, plus `best.pt` by validation posterior loss and
  `best_detection.pt` by validation detection AUC; each checkpoint
  carries model + optimizer + step + history so a run **resumes** exactly.
* **Live metrics** appended to `training_log.jsonl` and summarized in
  `status.json` (step, throughput, ETA, losses, last eval) for `scripts/monitor.py`
  and/or TensorBoard.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .flow_matching import cfm_loss
from .models.transitflow import ModelConfig, TransitFlow
from .noise import NoiseLibrary
from .simulator import SimConfig, TransitSimulator
from .utils import (
    PrefetchSimulator,
    SimulatorIterator,
    count_parameters,
    get_device,
    set_seed,
)


@dataclass
class TrainConfig:
    n_steps: int = 60000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    lambda_det: float = 1.0
    grad_clip: float = 5.0
    warmup_steps: int = 500
    eval_every: int = 2000
    eval_batches: int = 8
    log_every: int = 100
    device: str = "auto"
    seed: int = 0
    amp: bool = False                       # bf16 autocast (CUDA)
    # --- run management ---
    run_dir: Optional[str] = None           # all artifacts land here
    ckpt_every: int = 2000
    keep_last: int = 3                       # rotate step_*.pt checkpoints
    resume: Optional[str] = None            # checkpoint path or run_dir to resume
    tensorboard: bool = True
    # --- data source ---
    data_source: str = "simulate"            # "simulate" (on the fly) | "disk"
    data_dir: Optional[str] = None           # required when data_source == "disk"
    # --- data prefetch (keeps a GPU fed by the CPU simulator) ---
    num_workers: int = 0                     # >0 spawns prefetch worker processes
    prefetch: int = 8
    noise_lib_path: Optional[str] = None
    # --- GPU performance ---
    tf32: bool = True                        # allow TF32 matmul/conv on Ampere+
    # --- health / anti-waste ---
    expect_device: Optional[str] = None      # warn loudly if the run isn't here (e.g. "cuda")
    nan_patience: int = 20                   # stop after this many non-finite losses
    # --- legacy single-file checkpoint (used when run_dir is None) ---
    ckpt_path: Optional[str] = None


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.n_steps - cfg.warmup_steps, 1)
    return 0.5 * cfg.lr * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def compute_losses(model: TransitFlow, batch: dict, lambda_det: float) -> dict:
    """Forward pass returning the component losses (differentiable)."""
    noise_feat = batch["sigma_feat"] if model.cfg.use_noise_feature else None
    pg = batch.get("periodogram") if model.cfg.use_periodogram else None
    eph = batch.get("ephem_feat") if model.cfg.use_ephemeris_feature else None
    e = model.embed(batch["global"], batch["local"], noise_feat, pg, eph)
    det_logits = model.detect_logits(e)
    d = batch["d"].float()
    l_det = F.binary_cross_entropy_with_logits(det_logits, d)

    mask = batch["valid"]
    target = batch["theta_std"]
    if model.cfg.param_dim == 5:
        target = batch.get("theta_char_std", batch["theta_std"][:, 2:])
    if model.head_type == "fmpe":
        l_post = cfm_loss(model.velocity_fn(), target, e, mask=mask)
    else:
        l_post = model.posterior.nll(target, e, mask=mask)

    total = l_post + lambda_det * l_det
    return {"total": total, "posterior": l_post.detach(), "detection": l_det.detach()}


@torch.no_grad()
def evaluate(model: TransitFlow, val_iter, cfg: TrainConfig, n_batches: int) -> dict:
    """Validation losses, detection accuracy, and ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    agg = {"posterior": 0.0, "detection": 0.0, "det_acc": 0.0}
    all_d, all_p = [], []
    for _ in range(n_batches):
        batch = next(val_iter)
        out = compute_losses(model, batch, cfg.lambda_det)
        agg["posterior"] += float(out["posterior"])
        agg["detection"] += float(out["detection"])
        noise_feat = batch["sigma_feat"] if model.cfg.use_noise_feature else None
        pg = batch.get("periodogram") if model.cfg.use_periodogram else None
        eph = batch.get("ephem_feat") if model.cfg.use_ephemeris_feature else None
        e = model.embed(batch["global"], batch["local"], noise_feat, pg, eph)
        prob = torch.sigmoid(model.detect_logits(e))
        agg["det_acc"] += float(((prob > 0.5).long() == batch["d"]).float().mean())
        all_d.append(batch["d"].cpu().numpy())
        all_p.append(prob.cpu().numpy())
    model.train()
    out = {k: v / n_batches for k, v in agg.items()}
    d = np.concatenate(all_d)
    p = np.concatenate(all_p)
    out["roc_auc"] = float(roc_auc_score(d, p)) if len(np.unique(d)) > 1 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(model: TransitFlow, model_cfg: ModelConfig,
                    sim_cfg: SimConfig, path: str, optimizer=None,
                    step: int = 0, history: dict | None = None,
                    best: dict | None = None,
                    best_detection: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "sim_cfg": asdict(sim_cfg),
        "step": step,
        "history": history or {},
        "best": best or {},
        "best_detection": best_detection or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic — a killed run never leaves a half-written file


def load_checkpoint(path: str, device=None) -> tuple[TransitFlow, ModelConfig, SimConfig]:
    device = device or get_device("auto")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**{**ckpt["model_cfg"],
                          "global_channels": tuple(ckpt["model_cfg"]["global_channels"]),
                          "local_channels": tuple(ckpt["model_cfg"]["local_channels"])})
    scfg = SimConfig(**ckpt["sim_cfg"])
    model = TransitFlow(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mcfg, scfg


def _write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _human_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    model_cfg: ModelConfig | None = None,
    sim_cfg: SimConfig | None = None,
    train_cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Train a TransitFlow model; returns history and the trained model."""
    model_cfg = model_cfg or ModelConfig()
    sim_cfg = sim_cfg or SimConfig()
    train_cfg = train_cfg or TrainConfig()
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)

    # GPU performance knobs (no-ops off CUDA)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed-size views -> autotune convs
        if train_cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    # anti-waste: a long run that silently fell back to CPU burns money for nothing
    if train_cfg.expect_device and device.type != train_cfg.expect_device:
        msg = (f"WARNING: expected device '{train_cfg.expect_device}' but running "
               f"on '{device.type}'. A long run here may waste money — check the "
               f"CUDA install (torch.cuda.is_available()).")
        print("!" * 70 + f"\n{msg}\n" + "!" * 70)

    run_dir = train_cfg.run_dir
    ckpt_dir = log_path = status_path = None
    writer = None
    if run_dir:
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "training_log.jsonl")
        status_path = os.path.join(run_dir, "status.json")
        _snapshot_config(run_dir, model_cfg, sim_cfg, train_cfg)
        if train_cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(os.path.join(run_dir, "tb"))
            except Exception:
                writer = None

    noise_library = NoiseLibrary.load(train_cfg.noise_lib_path)
    if train_cfg.noise_lib_path and not noise_library.available():
        raise ValueError(f"noise library could not be loaded: {train_cfg.noise_lib_path}")

    # data iterators
    if train_cfg.data_source == "disk":
        from .data import DiskIterator
        if not train_cfg.data_dir:
            raise ValueError("data_source='disk' requires data_dir")
        train_iter = DiskIterator(train_cfg.data_dir, train_cfg.batch_size, device,
                                  shuffle=True, seed=train_cfg.seed)
    elif train_cfg.num_workers > 0:
        train_iter = PrefetchSimulator(
            sim_cfg, train_cfg.batch_size, device, train_cfg.num_workers,
            train_cfg.prefetch, train_cfg.seed, train_cfg.noise_lib_path)
    else:
        train_iter = SimulatorIterator(
            TransitSimulator(sim_cfg, noise_library=noise_library),
            train_cfg.batch_size, device, train_cfg.seed)
    val_simulator = TransitSimulator(sim_cfg, noise_library=noise_library)
    val_iter = SimulatorIterator(val_simulator, train_cfg.batch_size, device,
                                 seed=train_cfg.seed + 99991)

    model = TransitFlow(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)

    history = {"step": [], "total": [], "posterior": [], "detection": [], "val": []}
    best = {"posterior": float("inf"), "step": -1}
    best_detection = {"roc_auc": -1.0, "step": -1}
    start_step = 0

    # resume
    resume_path = _resolve_resume(train_cfg)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        history = ckpt.get("history", history)
        best = ckpt.get("best", best)
        best_detection = ckpt.get("best_detection", best_detection)
        if verbose:
            print(f"resumed from {resume_path} at step {start_step}")

    if verbose:
        print(f"device={device}  params={count_parameters(model):,}  "
              f"head={model.head_type}  run_dir={run_dir}")

    t_start = time.time()
    last_log_t = t_start
    last_log_step = start_step
    data_wait = 0.0          # seconds spent waiting on the data pipeline since last log
    nan_count = 0
    status = "running"
    model.train()
    try:
        for step in range(start_step, train_cfg.n_steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, train_cfg)
            _t = time.time()
            batch = next(train_iter)
            data_wait += time.time() - _t
            opt.zero_grad(set_to_none=True)
            use_amp = train_cfg.amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                out = compute_losses(model, batch, train_cfg.lambda_det)

            # NaN/inf guard: never burn GPU hours optimizing a diverged loss
            if not torch.isfinite(out["total"]):
                nan_count += 1
                if verbose:
                    print(f"  non-finite loss at step {step} "
                          f"({nan_count}/{train_cfg.nan_patience})")
                if nan_count >= train_cfg.nan_patience:
                    status = "error"
                    if verbose:
                        print("ABORT: too many non-finite losses — stopping to "
                              "avoid wasting compute. Check lr / data / model.")
                    break
                continue
            nan_count = 0

            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            if step % train_cfg.log_every == 0:
                now = time.time()
                interval = max(now - last_log_t, 1e-6)
                sps = (step - last_log_step) / interval
                data_frac = data_wait / interval if step > last_log_step else 0.0
                last_log_t, last_log_step = now, step
                data_wait = 0.0
                eta = (train_cfg.n_steps - step) / max(sps, 1e-6)
                total = float(out["total"].detach())
                rec = {"step": step, "total": total,
                       "posterior": float(out["posterior"]),
                       "detection": float(out["detection"]),
                       "lr": opt.param_groups[0]["lr"], "steps_per_s": sps,
                       "lc_per_s": sps * train_cfg.batch_size,
                       "data_wait_frac": round(data_frac, 3)}
                history["step"].append(step)
                history["total"].append(total)
                history["posterior"].append(rec["posterior"])
                history["detection"].append(rec["detection"])
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                if writer:
                    writer.add_scalar("loss/total", total, step)
                    writer.add_scalar("loss/posterior", rec["posterior"], step)
                    writer.add_scalar("loss/detection", rec["detection"], step)
                    writer.add_scalar("perf/data_wait_frac", data_frac, step)
                health = _health(device, train_cfg, total, data_frac)
                if status_path:
                    _write_status(status_path, run_dir, model.head_type, str(device),
                                  count_parameters(model), step, train_cfg.n_steps,
                                  rec, best, t_start, eta, sps, status, health,
                                  best_detection)
                if verbose:
                    warn = "" if not health["warnings"] else \
                        "  [" + ",".join(health["warnings"]) + "]"
                    print(f"step {step:6d}/{train_cfg.n_steps} | total {total:.4f} "
                          f"| post {rec['posterior']:.4f} | det {rec['detection']:.4f} "
                          f"| {sps*train_cfg.batch_size:.0f} lc/s "
                          f"| wait {data_frac*100:.0f}% | eta {_human_time(eta)}{warn}")

            if train_cfg.eval_every and (step + 1) % train_cfg.eval_every == 0:
                val = evaluate(model, val_iter, train_cfg, train_cfg.eval_batches)
                history["val"].append((step, val))
                if writer:
                    for k, v in val.items():
                        writer.add_scalar(f"val/{k}", v, step)
                if val.get("posterior", float("inf")) < best["posterior"]:
                    best = {"step": step, **val}
                    if ckpt_dir:
                        save_checkpoint(model, model_cfg, sim_cfg,
                                        os.path.join(ckpt_dir, "best.pt"), opt,
                                        step, history, best, best_detection)
                if val.get("roc_auc", -1) > best_detection["roc_auc"]:
                    best_detection = {"step": step, **val}
                    if ckpt_dir:
                        save_checkpoint(model, model_cfg, sim_cfg,
                                        os.path.join(ckpt_dir, "best_detection.pt"),
                                        opt, step, history, best,
                                        best_detection)
                if verbose:
                    print(f"  [val] post {val['posterior']:.4f} det {val['detection']:.4f} "
                          f"acc {val['det_acc']:.3f} auc {val['roc_auc']:.3f}")

            if ckpt_dir and (step + 1) % train_cfg.ckpt_every == 0:
                save_checkpoint(model, model_cfg, sim_cfg,
                                os.path.join(ckpt_dir, "latest.pt"), opt, step + 1,
                                history, best, best_detection)
                _rotate_step_ckpt(model, model_cfg, sim_cfg, ckpt_dir, step + 1,
                                  opt, history, best, best_detection,
                                  train_cfg.keep_last)
    except KeyboardInterrupt:
        status = "interrupted"
        if verbose:
            print("interrupted — saving latest checkpoint")
    finally:
        train_iter.close()

    # final checkpoint
    final_step = train_cfg.n_steps
    if ckpt_dir:
        save_checkpoint(model, model_cfg, sim_cfg,
                        os.path.join(ckpt_dir, "latest.pt"), opt, final_step,
                        history, best, best_detection)
    elif train_cfg.ckpt_path:
        save_checkpoint(model, model_cfg, sim_cfg, train_cfg.ckpt_path, opt,
                        final_step, history, best, best_detection)
    if status_path:
        last = {"step": final_step}
        _write_status(status_path, run_dir, model.head_type, str(device),
                      count_parameters(model), final_step, train_cfg.n_steps,
                      history_tail(history), best, t_start, 0.0, 0.0,
                      "done" if status == "running" else status,
                      best_detection=best_detection)
    if writer:
        writer.close()
    if verbose and run_dir:
        print(f"done. checkpoints + logs in {run_dir}")

    return {"model": model, "history": history, "device": device,
            "simulator": val_simulator, "best": best, "run_dir": run_dir}


# --------------------------------------------------------------------------- #
def history_tail(history: dict) -> dict:
    return {
        "step": history["step"][-1] if history["step"] else 0,
        "total": history["total"][-1] if history["total"] else float("nan"),
        "posterior": history["posterior"][-1] if history["posterior"] else float("nan"),
        "detection": history["detection"][-1] if history["detection"] else float("nan"),
        "lr": 0.0, "steps_per_s": 0.0, "lc_per_s": 0.0,
    }


def _health(device, train_cfg: TrainConfig, total_loss: float,
            data_frac: float) -> dict:
    """Derive run-health warnings so a wasteful run is caught early."""
    import math
    warnings = []
    if train_cfg.expect_device and device.type != train_cfg.expect_device:
        warnings.append("DEVICE-MISMATCH")
    elif device.type == "cpu" and train_cfg.n_steps > 2000:
        warnings.append("CPU-LONG-RUN")
    if data_frac > 0.35:
        warnings.append("GPU-STARVED")
    if not math.isfinite(total_loss):
        warnings.append("NON-FINITE-LOSS")
    return {"healthy": len(warnings) == 0, "device": device.type,
            "data_wait_frac": round(data_frac, 3), "warnings": warnings}


def _write_status(path, run_dir, head, device, params, step, total_steps, rec,
                  best, t_start, eta, sps, status, health=None,
                  best_detection=None) -> None:
    obj = {
        "run_dir": run_dir, "head": head, "device": device, "params": params,
        "status": status, "step": step, "total_steps": total_steps,
        "progress": round(step / max(total_steps, 1), 4),
        "elapsed_s": round(time.time() - t_start, 1),
        "eta_s": round(eta, 1), "eta": _human_time(eta),
        "throughput_lc_per_s": round(rec.get("lc_per_s", 0.0), 1),
        "data_wait_frac": rec.get("data_wait_frac", 0.0),
        "loss": {"total": rec.get("total"), "posterior": rec.get("posterior"),
                 "detection": rec.get("detection")},
        "lr": rec.get("lr"),
        "best": best,
        "best_detection": best_detection or {},
        "health": health or {"healthy": True, "device": device, "warnings": []},
        "updated_unix": time.time(),
    }
    _write_json_atomic(path, obj)


def _rotate_step_ckpt(model, model_cfg, sim_cfg, ckpt_dir, step, opt, history,
                      best, best_detection, keep_last) -> None:
    save_checkpoint(model, model_cfg, sim_cfg,
                    os.path.join(ckpt_dir, f"step_{step:08d}.pt"), opt, step,
                    history, best, best_detection)
    steps = sorted(f for f in os.listdir(ckpt_dir)
                   if f.startswith("step_") and f.endswith(".pt"))
    for old in steps[:-keep_last]:
        try:
            os.remove(os.path.join(ckpt_dir, old))
        except OSError:
            pass


def _resolve_resume(cfg: TrainConfig) -> str | None:
    if not cfg.resume:
        # auto-resume from run_dir/checkpoints/latest.pt if present
        if cfg.run_dir:
            cand = os.path.join(cfg.run_dir, "checkpoints", "latest.pt")
            return cand if os.path.exists(cand) else None
        return None
    if os.path.isdir(cfg.resume):
        return os.path.join(cfg.resume, "checkpoints", "latest.pt")
    return cfg.resume


def preflight(model_cfg: ModelConfig | None = None,
              sim_cfg: SimConfig | None = None,
              train_cfg: TrainConfig | None = None,
              n_probe: int = 40, price_per_hr: float = 0.40,
              verbose: bool = True) -> dict:
    """Short pre-run check so a misconfigured run never wastes GPU hours.

    Runs ~``n_probe`` training steps and verifies: the intended device is in use,
    losses are finite and trending down, the data pipeline isn't starving the
    GPU, and a checkpoint round-trips.  Reports projected throughput, ETA, and
    rough cost for the full ``n_steps``.  Returns a dict with ``verdict`` in
    {``PASS``, ``WARN``, ``FAIL``} and ``ok`` (False only on FAIL).
    """
    model_cfg = model_cfg or ModelConfig()
    sim_cfg = sim_cfg or SimConfig()
    train_cfg = train_cfg or TrainConfig()
    device = get_device(train_cfg.device)
    issues, warns = [], []
    noise_library = NoiseLibrary.load(train_cfg.noise_lib_path)
    if train_cfg.noise_lib_path and not noise_library.available():
        return _preflight_report({"verdict": "FAIL", "ok": False,
                                  "issues": [f"noise library load failed: {train_cfg.noise_lib_path}"]},
                                 verbose)

    # data source (mirror train())
    if train_cfg.data_source == "disk":
        from .data import DiskIterator
        try:
            it = DiskIterator(train_cfg.data_dir, train_cfg.batch_size, device, seed=1)
        except Exception as ex:
            return _preflight_report({"verdict": "FAIL", "ok": False,
                                      "issues": [f"disk data load failed: {ex}"]},
                                     verbose)
    elif train_cfg.num_workers > 0:
        it = PrefetchSimulator(sim_cfg, train_cfg.batch_size, device,
                               train_cfg.num_workers, train_cfg.prefetch, 1,
                               train_cfg.noise_lib_path)
    else:
        it = SimulatorIterator(TransitSimulator(sim_cfg, noise_library=noise_library),
                               train_cfg.batch_size, device, 1)

    model = TransitFlow(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    model.train()
    losses, data_t, comp_t = [], 0.0, 0.0
    t0 = time.time()
    try:
        for i in range(n_probe):
            _t = time.time(); batch = next(it); data_t += time.time() - _t
            _t = time.time()
            opt.zero_grad(set_to_none=True)
            out = compute_losses(model, batch, train_cfg.lambda_det)
            out["total"].backward(); opt.step()
            if device.type in ("cuda", "mps"):
                getattr(torch, device.type).synchronize()
            comp_t += time.time() - _t
            losses.append(float(out["total"].detach()))
    finally:
        it.close()
    wall = time.time() - t0
    sps = n_probe / max(wall, 1e-6)
    data_frac = data_t / max(data_t + comp_t, 1e-6)
    finite = all(np.isfinite(losses))
    trending = (np.mean(losses[-10:]) <= np.mean(losses[:10]) * 1.05) if len(losses) >= 20 else True

    # checkpoint round-trip
    ckpt_ok = True
    try:
        import tempfile
        p = os.path.join(tempfile.gettempdir(), "tf_preflight_ckpt.pt")
        save_checkpoint(model, model_cfg, sim_cfg, p, opt, 0, {}, {})
        load_checkpoint(p, device); os.remove(p)
    except Exception as ex:
        ckpt_ok = False; issues.append(f"checkpoint failed: {ex}")

    if not finite:
        issues.append("non-finite losses in probe")
    if train_cfg.expect_device and device.type != train_cfg.expect_device:
        issues.append(f"device is '{device.type}', expected '{train_cfg.expect_device}'")
    elif device.type == "cpu" and train_cfg.n_steps > 2000:
        warns.append("running a long job on CPU (no GPU detected)")
    if data_frac > 0.35:
        warns.append(f"GPU-starved: {data_frac*100:.0f}% of step time waits on data "
                     f"— pre-generate to disk (scripts/generate_data.py) or raise "
                     f"--num-workers / lower sim n_raw")
    if not trending:
        warns.append("loss not decreasing over the probe (check lr)")

    full_s = train_cfg.n_steps / max(sps, 1e-6)
    report = {
        "device": device.type, "params": count_parameters(model),
        "lc_per_s": round(sps * train_cfg.batch_size, 1),
        "data_wait_frac": round(data_frac, 3),
        "finite_losses": finite, "loss_trending_down": bool(trending),
        "checkpoint_ok": ckpt_ok,
        "projected_full_run_h": round(full_s / 3600, 2),
        "projected_cost_usd": round(full_s / 3600 * price_per_hr, 2),
        "issues": issues, "warnings": warns,
        "verdict": "FAIL" if issues else ("WARN" if warns else "PASS"),
    }
    report["ok"] = report["verdict"] != "FAIL"
    return _preflight_report(report, verbose)


def _preflight_report(r: dict, verbose: bool) -> dict:
    if not verbose:
        return r
    print("=" * 64)
    print(" PREFLIGHT")
    print("=" * 64)
    for k in ("device", "params", "lc_per_s", "data_wait_frac", "finite_losses",
              "loss_trending_down", "checkpoint_ok", "projected_full_run_h",
              "projected_cost_usd"):
        if k in r:
            print(f"  {k:22s}: {r[k]}")
    for w in r.get("warnings", []):
        print(f"  WARN: {w}")
    for i in r.get("issues", []):
        print(f"  FAIL: {i}")
    print(f"  VERDICT: {r.get('verdict','?')}")
    print("=" * 64)
    return r


def _snapshot_config(run_dir, model_cfg, sim_cfg, train_cfg) -> None:
    try:
        import yaml
        snap = {"model": asdict(model_cfg), "simulator": asdict(sim_cfg),
                "train": asdict(train_cfg)}
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(snap, f, default_flow_style=False)
    except Exception:
        pass
