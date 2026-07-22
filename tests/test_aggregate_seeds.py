"""Tests for cross-seed gate-report aggregation (pure stdlib logic)."""
from scripts.aggregate_seeds import aggregate_reports


def _report(train_seed, status, sbc_min_p=0.2, cov_err=0.01,
            detected_fraction=1.0):
    return {
        "run": {"train_seed": train_seed},
        "status": status,
        "synthetic": {
            "characterization_sbc_gate": {"min_pvalue": sbc_min_p},
            "characterization_coverage_calibration_error": cov_err,
            "detection": {"roc_auc": 0.999, "average_precision": 0.999},
        },
        "real": {"detection": {"detected_fraction": detected_fraction}},
    }


def test_all_seeds_pass_is_stable():
    st = {"final_pass": True, "gate_a": True, "gate_b": True}
    reps = [_report(s, dict(st)) for s in (0, 1, 2)]
    out = aggregate_reports(reps)
    assert out["n_seeds"] == 3
    assert out["final_pass"]["all_pass"] is True
    assert out["final_pass"]["unanimous"] is True
    assert out["stable"] is True
    assert out["n_flags_with_flips"] == 0
    assert out["gate_flags"]["gate_a"]["flip_frequency"] == 0.0


def test_threshold_flip_detected():
    # gate_b flips on the third seed -> not stable, flip recorded
    reps = [
        _report(0, {"final_pass": True, "gate_a": True, "gate_b": True}),
        _report(1, {"final_pass": True, "gate_a": True, "gate_b": True}),
        _report(2, {"final_pass": False, "gate_a": True, "gate_b": False}),
    ]
    out = aggregate_reports(reps)
    assert out["final_pass"]["n_pass"] == 2
    assert out["final_pass"]["unanimous"] is False
    assert out["stable"] is False
    gb = out["gate_flags"]["gate_b"]
    assert gb["modal_decision"] is True
    assert gb["n_flip"] == 1
    assert abs(gb["flip_frequency"] - 1 / 3) < 1e-9
    assert out["gate_flags"]["gate_a"]["unanimous"] is True


def test_continuous_metric_mean_sd():
    reps = [
        _report(0, {"final_pass": True}, cov_err=0.01),
        _report(1, {"final_pass": True}, cov_err=0.03),
    ]
    out = aggregate_reports(reps)
    m = out["continuous_metrics"]["coverage_calibration_error"]
    assert m["n"] == 2
    assert abs(m["mean"] - 0.02) < 1e-12
    assert m["min"] == 0.01 and m["max"] == 0.03
    # sample SD of {0.01,0.03} = 0.01414...
    assert abs(m["sd"] - 0.014142135623730951) < 1e-9


def test_missing_flag_in_one_seed_uses_reported_only():
    reps = [
        _report(0, {"final_pass": True, "gate_a": True}),
        _report(1, {"final_pass": True}),  # gate_a absent here
    ]
    out = aggregate_reports(reps)
    ga = out["gate_flags"]["gate_a"]
    assert ga["n_reported"] == 1
    assert ga["pass_fraction"] == 1.0


def test_nan_metric_is_skipped():
    reps = [
        _report(0, {"final_pass": True}, sbc_min_p=float("nan")),
        _report(1, {"final_pass": True}, sbc_min_p=0.2),
    ]
    out = aggregate_reports(reps)
    m = out["continuous_metrics"]["sbc_min_pvalue"]
    assert m["n"] == 1
    assert m["mean"] == 0.2
