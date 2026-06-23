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
  `step_*.pt`, plus a `best.pt` by validation detection AUC; each checkpoint
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
    # --- data prefetch (keeps a GPU fed by the CPU simulator) ---
    num_workers: int = 0                     # >0 spawns prefetch worker processes
    prefetch: int = 8
    noise_lib_path: Optional[str] = None
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
    e = model.embed(batch["global"], batch["local"], noise_feat)
    det_logits = model.detect_logits(e)
    d = batch["d"].float()
    l_det = F.binary_cross_entropy_with_logits(det_logits, d)

    mask = batch["valid"]
    if model.head_type == "fmpe":
        l_post = cfm_loss(model.velocity_fn(), batch["theta_std"], e, mask=mask)
    else:
        l_post = model.posterior.nll(batch["theta_std"], e, mask=mask)

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
        e = model.embed(batch["global"], batch["local"], noise_feat)
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
                    best: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "sim_cfg": asdict(sim_cfg),
        "step": step,
        "history": history or {},
        "best": best or {},
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

    # data iterators
    if train_cfg.num_workers > 0:
        train_iter = PrefetchSimulator(
            sim_cfg, train_cfg.batch_size, device, train_cfg.num_workers,
            train_cfg.prefetch, train_cfg.seed, train_cfg.noise_lib_path)
    else:
        train_iter = SimulatorIterator(
            TransitSimulator(sim_cfg), train_cfg.batch_size, device, train_cfg.seed)
    val_simulator = TransitSimulator(sim_cfg)
    val_iter = SimulatorIterator(val_simulator, train_cfg.batch_size, device,
                                 seed=train_cfg.seed + 99991)

    model = TransitFlow(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)

    history = {"step": [], "total": [], "posterior": [], "detection": [], "val": []}
    best = {"roc_auc": -1.0, "step": -1}
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
        if verbose:
            print(f"resumed from {resume_path} at step {start_step}")

    if verbose:
        print(f"device={device}  params={count_parameters(model):,}  "
              f"head={model.head_type}  run_dir={run_dir}")

    t_start = time.time()
    last_log_t = t_start
    last_log_step = start_step
    status = "running"
    model.train()
    try:
        for step in range(start_step, train_cfg.n_steps):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, train_cfg)
            batch = next(train_iter)
            opt.zero_grad(set_to_none=True)
            use_amp = train_cfg.amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                out = compute_losses(model, batch, train_cfg.lambda_det)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            if step % train_cfg.log_every == 0:
                now = time.time()
                sps = (step - last_log_step) / max(now - last_log_t, 1e-6)
                last_log_t, last_log_step = now, step
                eta = (train_cfg.n_steps - step) / max(sps, 1e-6)
                total = float(out["total"].detach())
                rec = {"step": step, "total": total,
                       "posterior": float(out["posterior"]),
                       "detection": float(out["detection"]),
                       "lr": opt.param_groups[0]["lr"], "steps_per_s": sps,
                       "lc_per_s": sps * train_cfg.batch_size}
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
                if status_path:
                    _write_status(status_path, run_dir, model.head_type, str(device),
                                  count_parameters(model), step, train_cfg.n_steps,
                                  rec, best, t_start, eta, sps, status)
                if verbose:
                    print(f"step {step:6d}/{train_cfg.n_steps} | total {total:.4f} "
                          f"| post {rec['posterior']:.4f} | det {rec['detection']:.4f} "
                          f"| {sps*train_cfg.batch_size:.0f} lc/s | eta {_human_time(eta)}")

            if train_cfg.eval_every and (step + 1) % train_cfg.eval_every == 0:
                val = evaluate(model, val_iter, train_cfg, train_cfg.eval_batches)
                history["val"].append((step, val))
                if writer:
                    for k, v in val.items():
                        writer.add_scalar(f"val/{k}", v, step)
                if val.get("roc_auc", -1) > best["roc_auc"]:
                    best = {"roc_auc": val["roc_auc"], "step": step, **val}
                    if ckpt_dir:
                        save_checkpoint(model, model_cfg, sim_cfg,
                                        os.path.join(ckpt_dir, "best.pt"), opt,
                                        step, history, best)
                if verbose:
                    print(f"  [val] post {val['posterior']:.4f} det {val['detection']:.4f} "
                          f"acc {val['det_acc']:.3f} auc {val['roc_auc']:.3f}")

            if ckpt_dir and (step + 1) % train_cfg.ckpt_every == 0:
                save_checkpoint(model, model_cfg, sim_cfg,
                                os.path.join(ckpt_dir, "latest.pt"), opt, step + 1,
                                history, best)
                _rotate_step_ckpt(model, model_cfg, sim_cfg, ckpt_dir, step + 1,
                                  opt, history, best, train_cfg.keep_last)
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
                        history, best)
    elif train_cfg.ckpt_path:
        save_checkpoint(model, model_cfg, sim_cfg, train_cfg.ckpt_path, opt,
                        final_step, history, best)
    if status_path:
        last = {"step": final_step}
        _write_status(status_path, run_dir, model.head_type, str(device),
                      count_parameters(model), final_step, train_cfg.n_steps,
                      history_tail(history), best, t_start, 0.0, 0.0,
                      "done" if status == "running" else status)
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


def _write_status(path, run_dir, head, device, params, step, total_steps, rec,
                  best, t_start, eta, sps, status) -> None:
    obj = {
        "run_dir": run_dir, "head": head, "device": device, "params": params,
        "status": status, "step": step, "total_steps": total_steps,
        "progress": round(step / max(total_steps, 1), 4),
        "elapsed_s": round(time.time() - t_start, 1),
        "eta_s": round(eta, 1), "eta": _human_time(eta),
        "throughput_lc_per_s": round(rec.get("lc_per_s", 0.0), 1),
        "loss": {"total": rec.get("total"), "posterior": rec.get("posterior"),
                 "detection": rec.get("detection")},
        "lr": rec.get("lr"),
        "best": best,
        "updated_unix": time.time(),
    }
    _write_json_atomic(path, obj)


def _rotate_step_ckpt(model, model_cfg, sim_cfg, ckpt_dir, step, opt, history,
                      best, keep_last) -> None:
    save_checkpoint(model, model_cfg, sim_cfg,
                    os.path.join(ckpt_dir, f"step_{step:08d}.pt"), opt, step,
                    history, best)
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


def _snapshot_config(run_dir, model_cfg, sim_cfg, train_cfg) -> None:
    try:
        import yaml
        snap = {"model": asdict(model_cfg), "simulator": asdict(sim_cfg),
                "train": asdict(train_cfg)}
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(snap, f, default_flow_style=False)
    except Exception:
        pass
