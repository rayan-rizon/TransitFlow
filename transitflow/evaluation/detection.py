"""Detection metrics and injection-recovery completeness grids."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


def detection_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """ROC-AUC, average precision, and curve arrays."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    out["roc"] = {"fpr": fpr, "tpr": tpr}
    out["pr"] = {"precision": prec, "recall": rec}
    return out


def completeness_grid(
    labels: np.ndarray,
    scores: np.ndarray,
    feature: np.ndarray,
    bins: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Recovery completeness of positives as a function of one feature.

    For the rows with ``labels == 1``, bin by ``feature`` and report the fraction
    with ``scores >= threshold`` (completeness) per bin -- the standard
    injection-recovery curve along ``period``, ``Rp/Rs``, or SNR.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    feature = np.asarray(feature, dtype=float)
    pos = labels == 1
    feat = feature[pos]
    rec = (scores[pos] >= threshold).astype(float)
    idx = np.digitize(feat, bins) - 1
    n_bins = len(bins) - 1
    comp = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for j in range(n_bins):
        sel = idx == j
        counts[j] = int(sel.sum())
        if counts[j] > 0:
            comp[j] = rec[sel].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return {"bin_centers": centers, "completeness": comp, "counts": counts}
