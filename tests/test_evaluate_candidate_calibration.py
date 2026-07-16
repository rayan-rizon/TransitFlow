import numpy as np

from scripts.evaluate_candidate_calibration import _period_recovered, _source_rows


def test_period_recovery_reports_top1_and_topk_separately():
    candidates = [
        {"best_period": 1.5},
        {"best_period": 3.0},
    ]
    assert _period_recovered(candidates, 3.0) == (False, True)


def test_source_rows_use_target_local_fap_threshold():
    labels = np.array([1, 1, 0, 0, 1, 0])
    fap = np.array([0.02, 0.10, 0.03, 0.20, 0.02, 0.04])
    sources = np.array([7, 7, 7, 7, 8, 8])
    rows = _source_rows(labels, fap, sources, target_fap=0.05, min_positives=1)
    assert rows["7"]["completeness"] == 0.5
    assert rows["7"]["false_positive_rate"] == 0.5
    assert rows["8"]["completeness"] == 1.0
    assert rows["8"]["false_positive_rate"] == 1.0
