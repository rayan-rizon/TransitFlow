"""Tests for the negative-class real-data validation metric core.

Only the pure logic (classification_metrics, passes_negative_quality) is
exercised; the full script needs torch + MAST and runs on Vast.
"""
from types import SimpleNamespace

from scripts.validate_negatives import (
    classification_metrics,
    passes_negative_quality,
)


def test_perfect_separation():
    # positives all above threshold, negatives all below
    m = classification_metrics(
        pos_pdet=[0.99, 0.95, 0.92],
        neg_pdet=[0.10, 0.20, 0.05, 0.30],
        threshold=0.9)
    assert m["tp"] == 3 and m["fn"] == 0
    assert m["fp"] == 0 and m["tn"] == 4
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0


def test_false_positives_lower_specificity_and_precision():
    # 2 of 4 negatives leak above threshold
    m = classification_metrics(
        pos_pdet=[0.99, 0.99],
        neg_pdet=[0.95, 0.92, 0.10, 0.20],
        threshold=0.9)
    assert m["fp"] == 2 and m["tn"] == 2
    assert m["specificity"] == 0.5
    assert m["false_positive_rate"] == 0.5
    assert m["precision"] == 2 / 4  # tp=2, fp=2


def test_no_positives_precision_none_specificity_defined():
    m = classification_metrics(pos_pdet=[], neg_pdet=[0.1, 0.2, 0.95],
                               threshold=0.9)
    assert m["n_positives"] == 0
    assert m["precision"] == 0.0  # tp=0, fp=1 -> 0/(0+1)
    assert m["sensitivity"] is None  # tp+fn == 0 -> undefined
    assert m["specificity"] == 2 / 3
    assert m["tp"] == 0 and m["fp"] == 1


def test_threshold_monotonicity():
    pos = [0.6, 0.8, 0.95]
    neg = [0.4, 0.7, 0.85]
    lo = classification_metrics(pos, neg, 0.5)
    hi = classification_metrics(pos, neg, 0.9)
    # raising the threshold cannot increase false positives
    assert hi["fp"] <= lo["fp"]
    # ...and cannot increase true positives
    assert hi["tp"] <= lo["tp"]


def test_nan_scores_ignored():
    m = classification_metrics(
        pos_pdet=[float("nan"), 0.99],
        neg_pdet=[0.1, float("nan")],
        threshold=0.9)
    assert m["n_positives"] == 1 and m["n_negatives"] == 1
    assert m["tp"] == 1 and m["tn"] == 1


def test_negative_quality_drops_geometry_requirement():
    args = SimpleNamespace(
        quality_gate=True, min_cadences=100, min_cadence_fraction=0.5,
        min_in_transit=10, min_transits=2, min_observed_snr=5.0)
    # a valid negative with NO finite geometry must still pass
    q = {"n_cadences": 6000, "cadence_fraction_of_training": 0.8,
         "n_in_transit": 80, "n_transits": 5, "observed_snr": 20.0,
         "finite_geometry": False}
    assert passes_negative_quality(q, args) is True
    # but poor sampling still fails
    q_bad = dict(q, n_cadences=50)
    assert passes_negative_quality(q_bad, args) is False


def test_quality_gate_disabled_passes_all():
    args = SimpleNamespace(quality_gate=False)
    assert passes_negative_quality({}, args) is True
