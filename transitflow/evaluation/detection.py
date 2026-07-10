"""Detection metrics and injection-recovery completeness grids."""

from __future__ import annotations

import numpy as np


def _binary_curves(labels: np.ndarray, scores: np.ndarray) -> tuple[dict, dict]:
    """Return ROC and precision-recall curves without a sklearn dependency."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    # Evaluate only after the last item at each distinct score so ties are
    # handled as a single threshold, matching standard ROC construction.
    last = np.r_[s[1:] != s[:-1], True]
    tp_t = tp[last].astype(float)
    fp_t = fp[last].astype(float)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    fpr = np.r_[0.0, fp_t / n_neg, 1.0]
    tpr = np.r_[0.0, tp_t / n_pos, 1.0]
    precision = tp_t / np.maximum(tp_t + fp_t, 1.0)
    recall = tp_t / n_pos
    # Include the conventional starting point for plotting.
    pr = {"precision": np.r_[1.0, precision], "recall": np.r_[0.0, recall]}
    return {"fpr": fpr, "tpr": tpr}, pr


def _roc_auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * ((i + 1) + j)
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    u = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("average precision requires at least one positive label")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    last = np.r_[s[1:] != s[:-1], True]
    precision = tp[last] / np.maximum(tp[last] + fp[last], 1)
    recall = tp[last] / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray,
                                n_bins: int = 10) -> float:
    """Equal-width expected calibration error for probability-like scores."""
    if np.any((scores < 0.0) | (scores > 1.0)):
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for i in range(n_bins):
        upper_inclusive = i == n_bins - 1
        mask = (scores >= edges[i]) & (
            scores <= edges[i + 1] if upper_inclusive else scores < edges[i + 1])
        if mask.any():
            ece += float(mask.sum()) / total * abs(
                float(scores[mask].mean()) - float(labels[mask].mean()))
    return ece


def detection_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Discrimination metrics, probability calibration diagnostics, and curves."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    roc, pr = _binary_curves(labels, scores)
    out = {
        "roc_auc": _roc_auc_rank(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "brier_score": float(np.mean((scores - labels) ** 2))
        if np.all((scores >= 0.0) & (scores <= 1.0)) else float("nan"),
        "expected_calibration_error_10bin":
            _expected_calibration_error(labels, scores),
    }
    out["roc"] = roc
    out["pr"] = pr
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
