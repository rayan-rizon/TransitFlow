#!/usr/bin/env python3
"""Train a source-held-out, candidate-independent Stage-A detector.

This script is deliberately separate from TransitFlow posterior training.  It
cannot consume local folded views, candidate ephemerides, or candidate-based
flattened data.  It is development-only until a predeclared held-out gate is
met; it never accesses a publication lockbox.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from transitflow.data import DiskDataset, DiskIterator
from transitflow.evaluation import detection_metrics
from transitflow.models.raw_detector import RawDetectorConfig, RawEvidenceDetector
from transitflow.raw_detection import validate_raw_detector_dataset
from transitflow.utils import batch_to_torch, get_device, set_seed


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _checkpoint(path: Path, model: RawEvidenceDetector, optimizer, step: int,
                best: dict, provenance: dict) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_config": model.checkpoint_config(),
        "step": int(step),
        "best": best,
        "provenance": provenance,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate_full(model: RawEvidenceDetector, dataset: DiskDataset,
                  device: torch.device, batch_size: int, amp: bool) -> dict:
    """Exact whole-dataset AUC; never select checkpoints on a minibatch."""
    model.eval()
    labels, scores = [], []
    for start in range(0, len(dataset), batch_size):
        rows = np.arange(start, min(start + batch_size, len(dataset)))
        batch = batch_to_torch(dataset._gather(rows), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(batch["global"], batch["periodogram"], batch["sigma_feat"])
        labels.append(batch["d"].cpu().numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    metrics = detection_metrics(np.concatenate(labels), np.concatenate(scores))
    model.train()
    return {key: float(metrics[key]) for key in ("roc_auc", "average_precision",
                                                  "brier_score", "expected_calibration_error_10bin")}


def _best(candidate: dict, incumbent: dict) -> bool:
    auc, old_auc = float(candidate["roc_auc"]), float(incumbent.get("roc_auc", -np.inf))
    return auc > old_auc or (
        auc == old_auc and float(candidate["average_precision"]) >
        float(incumbent.get("average_precision", -np.inf)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--validation-data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=192)
    args = ap.parse_args()
    if args.steps <= 0 or args.batch_size <= 1 or args.eval_every <= 0:
        raise SystemExit("steps, batch size, and eval interval must be positive")

    train_report = validate_raw_detector_dataset(args.data_dir)
    val_report = validate_raw_detector_dataset(args.validation_data_dir)
    if not train_report["pass"] or not val_report["pass"]:
        raise SystemExit("invalid raw-detector dataset: " + "; ".join(
            train_report["failures"] + val_report["failures"]))
    train_sources, val_sources = set(train_report["source_labels"]), set(val_report["source_labels"])
    if not train_sources or not val_sources:
        raise SystemExit("raw-detector train and validation datasets need source labels")
    overlap = train_sources & val_sources
    if overlap:
        raise SystemExit("raw-detector source split overlaps: " + ", ".join(sorted(overlap)[:8]))

    device = get_device(args.device)
    set_seed(args.seed)
    train_ds = DiskDataset(args.data_dir, in_ram=True)
    val_ds = DiskDataset(args.validation_data_dir, in_ram=True)
    if "periodogram" not in train_ds.keys or "periodogram" not in val_ds.keys:
        raise SystemExit("raw-detector data requires periodogram arrays")
    model_cfg = RawDetectorConfig(embed_dim=args.embed_dim)
    model = RawEvidenceDetector(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    train_iter = DiskIterator(args.data_dir, args.batch_size, device,
                              shuffle=True, seed=args.seed)
    run_dir = Path(args.run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "development_only": True,
        "candidate_independent": True,
        "train_dataset": train_report,
        "validation_dataset": val_report,
        "source_overlap": [],
        "seed": int(args.seed),
        "steps": int(args.steps),
        "model_config": asdict(model_cfg),
    }
    _atomic_json(run_dir / "provenance.json", provenance)
    best: dict = {}
    start = time.time()
    status = {"status": "running", "development_only": True,
              "device": str(device), "step": 0, "total_steps": args.steps,
              "best": best}
    _atomic_json(run_dir / "status.json", status)

    for step in range(1, args.steps + 1):
        batch = next(train_iter)
        lr_scale = min(1.0, step / max(args.warmup_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(args.amp and device.type == "cuda")):
            logits = model(batch["global"], batch["periodogram"], batch["sigma_feat"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["d"].float())
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite raw detector loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_full(model, val_ds, device, args.eval_batch_size, args.amp)
            metrics["step"] = step
            if _best(metrics, best):
                best = metrics.copy()
                _checkpoint(checkpoints / "best.pt", model, optimizer, step, best, provenance)
            _checkpoint(checkpoints / "latest.pt", model, optimizer, step, best, provenance)
            status = {
                "status": "running", "development_only": True, "device": str(device),
                "step": step, "total_steps": args.steps, "loss": float(loss.detach()),
                "elapsed_s": time.time() - start, "validation": metrics, "best": best,
            }
            _atomic_json(run_dir / "status.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
    status["status"] = "done"
    status["step"] = args.steps
    _atomic_json(run_dir / "status.json", status)
    print(json.dumps({"best": best, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
