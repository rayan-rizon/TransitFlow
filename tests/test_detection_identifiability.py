import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_detection_identifiability.py"
_SPEC = importlib.util.spec_from_file_location("detection_identifiability", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_wilson_interval_and_conservative_fpr_threshold():
    lo, hi = _MODULE.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    scores = np.linspace(0.0, 0.99, 100)
    threshold = _MODULE.threshold_at_fpr(scores, 0.05)
    assert np.mean(scores >= threshold) <= 0.05


def test_stratum_reports_completeness_and_auc():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.05])
    result = _MODULE._stratum(labels, scores, labels == 1, labels == 0, 0.5)
    assert result["n_positive"] == 2
    assert result["recovered"] == 1
    assert result["completeness"] == 0.5
    assert result["roc_auc"] > 0.5


def test_fair_bls_protocol_removes_oracle_positive_candidates():
    from transitflow.simulator import SimConfig

    original = SimConfig(
        candidate_bls_positive_fraction=0.5,
        candidate_bls_negative_fraction=1.0,
        candidate_jitter_fraction=0.2,
    )
    audit = _MODULE.fair_bls_sim_config(original)
    assert audit.candidate_bls_positive_fraction == 1.0
    assert audit.candidate_bls_negative_fraction == 1.0
    assert audit.candidate_jitter_fraction == 0.0
    assert original.candidate_bls_positive_fraction == 0.5
