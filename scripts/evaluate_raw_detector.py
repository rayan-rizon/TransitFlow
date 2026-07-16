#!/usr/bin/env python3
"""Source-stratified held-out evaluation for the Stage-A raw detector."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from transitflow.data import DiskDataset
from transitflow.evaluation import detection_metrics
from transitflow.models.raw_detector import RawDetectorConfig, RawEvidenceDetector
from transitflow.raw_detection import validate_raw_detector_dataset
from transitflow.utils import batch_to_torch, get_device


def wilson_interval(successes: int, total: int,
                    z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    values = np.sort(np.asarray(negative_scores, dtype=float))
    if not len(values):
        raise ValueError("held-out data needs negative examples")
    rank = min(len(values) - 1, max(0, int(np.ceil((1.0 - target_fpr) * len(values)))))
    return float(values[rank])


def load_raw_detector(path: str | Path, device: torch.device) -> tuple[RawEvidenceDetector, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw = dict(checkpoint["model_config"])
    for key in ("global_channels", "periodogram_channels"):
        raw[key] = tuple(raw[key])
    model = RawEvidenceDetector(RawDetectorConfig(**raw)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def score_dataset(model: RawEvidenceDetector, ds: DiskDataset, device: torch.device,
                  batch_size: int, amp: bool) -> dict[str, np.ndarray]:
    values = {key: [] for key in ("labels", "scores", "regime", "sources")}
    for start in range(0, len(ds), batch_size):
        rows = np.arange(start, min(start + batch_size, len(ds)))
        batch = batch_to_torch(ds._gather(rows), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(batch["global"], batch["periodogram"], batch["sigma_feat"])
        values["labels"].append(batch["d"].cpu().numpy())
        values["scores"].append(torch.sigmoid(logits).float().cpu().numpy())
        values["regime"].append(batch["regime"].cpu().numpy())
        values["sources"].append(batch["noise_source_index"].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--min-source-positives", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    if not 0 < args.target_fpr < 1:
        raise SystemExit("--target-fpr must be in (0, 1)")
    data_report = validate_raw_detector_dataset(args.data_dir)
    if not data_report["pass"]:
        raise SystemExit("invalid raw-detector dataset: " + "; ".join(data_report["failures"]))
    device = get_device(args.device)
    model, checkpoint = load_raw_detector(args.ckpt, device)
    if not checkpoint.get("provenance", {}).get("candidate_independent", False):
        raise SystemExit("checkpoint does not declare candidate-independent training")
    ds = DiskDataset(args.data_dir, in_ram=True)
    values = score_dataset(model, ds, device, args.batch_size, args.amp)
    labels, scores = values["labels"].astype(int), values["scores"].astype(float)
    threshold = threshold_at_fpr(scores[labels == 0], args.target_fpr)
    overall = detection_metrics(labels, scores)
    pos = labels == 1
    recovered = int((scores[pos] >= threshold).sum())
    source_report = {}
    for source in np.unique(values["sources"][(values["regime"] == 0) &
                                                (values["sources"] >= 0)]):
        mask = (values["sources"] == source) & (values["regime"] == 0)
        n_pos = int((mask & pos).sum())
        n_neg = int((mask & ~pos).sum())
        if n_pos < args.min_source_positives or n_neg == 0:
            continue
        p_scores, n_scores = scores[mask & pos], scores[mask & ~pos]
        pair = detection_metrics(np.r_[np.ones(n_pos), np.zeros(n_neg)],
                                 np.r_[p_scores, n_scores])
        k = int((p_scores >= threshold).sum())
        source_report[str(int(source))] = {
            "n_positive": n_pos, "n_negative": n_neg,
            "completeness": float(k / n_pos), "recovered": k,
            "completeness_wilson95": wilson_interval(k, n_pos),
            "roc_auc": float(pair["roc_auc"]),
            "average_precision": float(pair["average_precision"]),
        }
    report = {
        "report_schema_version": 1,
        "stage": "candidate_independent_raw_detector_development",
        "development_only": True,
        "candidate_independent": True,
        "candidate_inputs": [],
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data": data_report,
        "n": int(len(labels)),
        "overall": {key: float(overall[key]) for key in ("roc_auc", "average_precision",
                                                            "brier_score", "expected_calibration_error_10bin")},
        "operating_point": {
            "target_fpr": float(args.target_fpr), "threshold": threshold,
            "empirical_fpr": float(np.mean(scores[labels == 0] >= threshold)),
            "completeness": float(recovered / max(int(pos.sum()), 1)),
            "completeness_wilson95": wilson_interval(recovered, int(pos.sum())),
        },
        "real_noise_source_strata": source_report,
        "source_label_count": len(data_report["source_labels"]),
        "checkpoint_best_validation": checkpoint.get("best", {}),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({"roc_auc": report["overall"]["roc_auc"],
                      "source_rows": len(source_report), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
