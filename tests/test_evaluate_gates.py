import json

from scripts.evaluate import evaluation_component_seeds, sbc_gate
from types import SimpleNamespace

import numpy as np

from scripts.validate_real import (
    fold_bin_fixed_ephemeris,
    passes_real_quality,
    real_diagnostic_status,
    real_gate_status,
)
from scripts._config import build_configs
from scripts.run_publishable_vast import (
    build_gate_report,
    build_synthetic_gate_report,
    dataset_worker_preflight,
    external_lockbox_synthetic_failure_blocks_downstream,
    full_run_disk_preflight,
    identifiability_preflight,
    prepare_noise_four_way_split,
    prepare_noise_splits,
    prepare_noise_train_calibration_split,
    prepare_noise_train_validation_calibration_split,
    prepare_noise_three_way_split,
    require_noise_target_count,
    simulator_config_for_candidate_domain,
    validate_existing_dataset,
)
from transitflow.data import _write_dataset_metadata
from transitflow.evaluation.sbc import sbc_uniformity


def test_sbc_gate_controls_familywise_error():
    gate = sbc_gate([0.37, 0.0115, 0.14, 0.11, 0.15])

    assert gate["bonferroni_alpha_per_test"] == 0.01
    assert gate["pass"] is True
    assert gate["all_raw_p_gt_0.05"] is False


def test_noise_target_count_gate_fails_closed():
    require_noise_target_count({"n_unique_targets": 30}, 30)
    try:
        require_noise_target_count({"n_unique_targets": 14}, 30)
    except SystemExit as exc:
        assert "14 independent targets" in str(exc)
    else:
        raise AssertionError("underpowered noise library did not fail")


def _identifiability_report_dict() -> dict:
    return {
        "report_schema_version": 1,
        "candidate_protocol": {
            "source": "pre_generated_blind_bls_dataset",
            "candidate_bls_positive_fraction": 1.0,
            "candidate_bls_negative_fraction": 1.0,
        },
        "overall": {"roc_auc": 0.91},
        "predeclared_strata": {
            "expected_snr": {
                "<25": {"roc_auc": 0.80},
                "25-75": {"roc_auc": 0.958},
                ">=75": {"roc_auc": 0.964},
            },
        },
        "source_label_count": 29,
        "real_noise_source_strata": {str(i): {} for i in range(29)},
        "bls_top1_period_recovery_within_1pct": {"fraction": 0.35},
    }


def test_identifiability_preflight_requires_fixed_blind_bls_report(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_identifiability_report_dict()))
    passed = identifiability_preflight(report_path, min_sources=25)
    assert passed["pass"] is True
    assert passed["gate_revision"] == "2026-07-19_predeclared_v2"

    report = _identifiability_report_dict()
    report["candidate_protocol"]["source"] = "oracle"
    report_path.write_text(json.dumps(report))
    failed = identifiability_preflight(report_path, min_sources=25)
    assert failed["pass"] is False
    assert "report does not use a fixed blind-BLS dataset" in failed["failures"]


def test_identifiability_preflight_gates_inside_detectable_domain(tmp_path):
    """The measured 2026-07-16 development audit shape passes; a low
    in-domain (expected S/N >= 25) AUC or a gross overall regression fails."""
    report_path = tmp_path / "report.json"

    # Low-S/N stratum performance never blocks: it is outside the
    # predeclared detectable domain.
    report = _identifiability_report_dict()
    report["predeclared_strata"]["expected_snr"]["<25"]["roc_auc"] = 0.55
    report_path.write_text(json.dumps(report))
    assert identifiability_preflight(report_path, min_sources=25)["pass"] is True

    report = _identifiability_report_dict()
    report["predeclared_strata"]["expected_snr"]["25-75"]["roc_auc"] = 0.90
    report_path.write_text(json.dumps(report))
    failed = identifiability_preflight(report_path, min_sources=25)
    assert failed["pass"] is False
    assert any("in-domain ROC-AUC" in f for f in failed["failures"])

    report = _identifiability_report_dict()
    del report["predeclared_strata"]["expected_snr"][">=75"]
    report_path.write_text(json.dumps(report))
    failed = identifiability_preflight(report_path, min_sources=25)
    assert failed["pass"] is False
    assert any("lacks a finite ROC-AUC" in f for f in failed["failures"])

    report = _identifiability_report_dict()
    report["overall"]["roc_auc"] = 0.80
    report_path.write_text(json.dumps(report))
    failed = identifiability_preflight(report_path, min_sources=25)
    assert failed["pass"] is False
    assert any("overall ROC-AUC below floor" in f for f in failed["failures"])


def test_full_run_disk_preflight_reports_capacity(monkeypatch, tmp_path):
    class Usage:
        free = 15 * 1024 ** 3

    monkeypatch.setattr("scripts.run_publishable_vast.shutil.disk_usage",
                        lambda _: Usage())

    report = full_run_disk_preflight(tmp_path, 16.0)

    assert report["available_gib"] == 15.0
    assert report["required_free_gib"] == 16.0
    assert report["pass"] is False


def test_dataset_worker_preflight_caps_16_gib_node():
    report = dataset_worker_preflight(
        61, memory_bytes=16 * 1024 ** 3, reserve_gib=6.0, worker_mib=1280.0)

    assert report["capacity_workers"] == 8
    assert report["effective_workers"] == 8
    assert report["cap_applied"] is True


def test_synthetic_gate_report_separates_required_and_diagnostic_status():
    metrics = {
        "gate_status": {
            "characterization_sbc_familywise_alpha_0.05": True,
            "characterization_coverage_error_le_0.03": False,
            "not_applicable": None,
            "detection_auc_ge_0.99": False,
        }
    }
    report = build_synthetic_gate_report(metrics, {"all_disjoint": True})

    assert "not_applicable" not in report["status"]
    assert report["diagnostic_status"]["not_applicable"] is None
    assert report["diagnostic_status"]["detection_auc_ge_0.99"] is False
    assert report["status"]["checkpoint_validation_disjoint"] is True
    assert report["all_declared_synthetic_gates_pass"] is False


def test_fair_bls_detection_is_a_required_synthetic_gate():
    metrics = {
        "detection_candidate_source": "bls",
        "gate_status": {
            "characterization_sbc_familywise_alpha_0.05": True,
            "characterization_coverage_error_le_0.03": True,
            "detection_auc_ge_min": False,
        },
    }

    report = build_synthetic_gate_report(metrics, {"all_disjoint": True})

    assert report["status"]["detection_auc_ge_min"] is False
    assert "detection_auc_ge_min" not in report["diagnostic_status"]
    assert report["all_declared_synthetic_gates_pass"] is False


def test_failed_external_lockbox_stops_before_downstream_stages():
    failed = {"all_declared_synthetic_gates_pass": False}
    assert external_lockbox_synthetic_failure_blocks_downstream(
        failed, smoke=False, fast_check=False, has_external_lockbox=True)
    assert not external_lockbox_synthetic_failure_blocks_downstream(
        failed, smoke=False, fast_check=True, has_external_lockbox=True)
    assert not external_lockbox_synthetic_failure_blocks_downstream(
        failed, smoke=False, fast_check=False, has_external_lockbox=False)


def test_sbc_gate_rejects_clear_miscalibration():
    gate = sbc_gate([1.3e-5, 0.15, 0.27, 0.89, 0.87])

    assert gate["pass"] is False


def test_sbc_uniformity_uses_declared_discrete_rank_support():
    # The observed maximum is deliberately far below L.  The null support must
    # still be all L + 1 possible ranks, rather than shrinking to the sample.
    ranks = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int64)
    result = sbc_uniformity(ranks, n_bins=4, n_posterior=9)

    assert result["n_posterior"] == 9
    assert result["rank_support_size"] == 10
    assert result["expected_counts"] == [1.8, 1.2, 1.8, 1.2]
    assert result["histogram_counts"] == [[3, 2, 1, 0]]


def test_sbc_uniformity_rejects_rank_outside_declared_support():
    try:
        sbc_uniformity(np.array([[11]]), n_posterior=10)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("invalid rank support did not fail closed")


def test_evaluation_component_streams_are_sample_size_invariant():
    seeds = evaluation_component_seeds(20260715)
    assert seeds == evaluation_component_seeds(20260715)
    assert len(set(seeds.values())) == 3

    short_detection = np.random.default_rng(seeds["detection"])
    short_detection.random(10)
    sbc_after_short = np.random.default_rng(seeds["sbc"]).random(16)

    long_detection = np.random.default_rng(seeds["detection"])
    long_detection.random(10000)
    sbc_after_long = np.random.default_rng(seeds["sbc"]).random(16)

    assert np.array_equal(sbc_after_short, sbc_after_long)


def test_real_gate_ignores_fixed_ephemeris_coverage():
    summary = {
        "detection": {"detected_fraction": 0.95},
        "detected_per_param": {
            "P": {"coverage_68": 0.0, "coverage_95": 0.0},
            "RpRs": {"coverage_68": 0.6, "coverage_95": 0.9},
            "aRs": {"coverage_68": 0.55, "coverage_95": 0.85},
            "b": {"coverage_68": 0.7, "coverage_95": 0.95},
        },
        "mcmc_agreement": {
            "RpRs": {
                "median_wasserstein_prior_fraction": 0.08,
                "median_wasserstein_width_fraction": 0.4,
            },
            "aRs": {
                "median_wasserstein_prior_fraction": 0.05,
                "median_wasserstein_width_fraction": 0.3,
            },
            "b": {
                "median_wasserstein_prior_fraction": 0.09,
                "median_wasserstein_width_fraction": 0.45,
            },
        },
    }

    gates = real_gate_status(summary)
    diagnostics = real_diagnostic_status(summary)

    assert "archive_detected_char_cov68_ge_0.5" not in gates
    assert diagnostics["archive_detected_char_cov68_ge_0.5"] is True
    assert diagnostics["archive_detected_char_cov95_ge_0.8"] is True
    assert gates["mcmc_characterization_prior_fraction_le_0.1"] is True


def test_real_gate_rejects_degenerate_importance_correction():
    summary = {
        "detection": {"detected_fraction": 0.95},
        "detected_per_param": {},
        "mcmc_agreement": {
            "RpRs": {
                "median_wasserstein_prior_fraction": 0.08,
                "median_wasserstein_width_fraction": 0.4,
            },
            "aRs": {
                "median_wasserstein_prior_fraction": 0.05,
                "median_wasserstein_width_fraction": 0.3,
            },
            "b": {
                "median_wasserstein_prior_fraction": 0.09,
                "median_wasserstein_width_fraction": 0.45,
            },
        },
        "importance_correction": {
            "enabled": True,
            "min_ess_fraction": 0.004,
        },
    }

    gates = real_gate_status(summary)

    assert gates["mcmc_characterization_prior_fraction_le_0.1"] is True
    assert gates["importance_correction_min_ess_fraction_ge_0.05"] is False


def test_real_gate_requires_converged_mcmc_reference():
    summary = {
        "detection": {"detected_fraction": 1.0},
        "detected_per_param": {},
        "mcmc_agreement": {
            key: {"median_wasserstein_prior_fraction": 0.05,
                  "median_wasserstein_width_fraction": 0.4}
            for key in ("RpRs", "aRs", "b")
        },
        "mcmc_conditioning": {"tau_multiple_min": 12.0, "n_eff_min": 180.0},
    }

    gates = real_gate_status(summary)

    assert gates["mcmc_characterization_prior_fraction_le_0.1"] is True
    assert gates["mcmc_chain_length_ge_50_tau"] is False
    assert gates["mcmc_effective_samples_ge_400"] is False


def test_real_quality_gate_rejects_weak_or_missing_geometry_rows():
    args = SimpleNamespace(
        quality_gate=True,
        min_cadences=5000,
        min_cadence_fraction=0.70,
        min_in_transit=50,
        min_transits=2,
        min_observed_snr=12.0,
        max_impact=0.9,
    )

    good = {
        "finite_geometry": True,
        "n_cadences": 12000,
        "cadence_fraction_of_training": 0.8,
        "n_in_transit": 80,
        "n_transits": 3,
        "observed_snr": 12.0,
        "impact_parameter": 0.4,
    }
    weak = {**good, "observed_snr": 10.0}
    missing_geometry = {**good, "finite_geometry": False}
    grazing = {**good, "impact_parameter": 0.95}

    assert passes_real_quality(good, args) is True
    assert passes_real_quality(weak, args) is False
    assert passes_real_quality(missing_geometry, args) is False
    assert passes_real_quality(grazing, args) is False


def test_fold_bin_fixed_ephemeris_reduces_cadences_and_scales_errors():
    times = np.linspace(0.0, 27.0, 9000)
    flux = 1.0 + 0.001 * np.sin(2 * np.pi * times / 3.0)

    b_t, b_f, b_e = fold_bin_fixed_ephemeris(
        times, flux, sigma=0.001, P=3.0, t0_phase=0.2, max_cadences=900)

    assert len(b_t) == 900
    assert len(b_f) == 900
    assert len(b_e) == 900
    assert np.all(np.diff(b_t) >= 0)
    assert np.nanmax(b_e) < 0.001


def test_publishable_gate_report_schema_and_status():
    synthetic = {
        "detection": {"roc_auc": 0.995, "average_precision": 0.996},
        "characterization_sbc_gate": {"pass": True},
        "characterization_coverage_calibration_error": 0.01,
        "gate_status": {
            "detection_auc_ge_min": True,
            "characterization_sbc_familywise_alpha_0.05": True,
            "characterization_coverage_error_le_0.03": True,
        },
    }
    real = {
        "summary": {
            "n_planets": 30,
            "detection": {"n_detected": 28, "detected_fraction": 28 / 30},
            "gate_status": {},
            "mcmc_agreement": {
                "RpRs": {"n": 16, "median_wasserstein_prior_fraction": 0.05,
                         "median_wasserstein_width_fraction": 0.4},
                "aRs": {"n": 16, "median_wasserstein_prior_fraction": 0.06,
                        "median_wasserstein_width_fraction": 0.3},
                "b": {"n": 16, "median_wasserstein_prior_fraction": 0.09,
                      "median_wasserstein_width_fraction": 0.45},
            },
        }
    }
    bls = {
        "candidate_source": "bls",
        "n": 5000,
        "bls": {"roc_auc": 0.4},
        "transitflow": {"roc_auc": 0.99},
        "tls_requested": True,
        "tls": {"n": 5000, "roc_auc": 0.6,
                "paired_with_transitflow": True, "n_failed": 0},
        "uncertainty": {
            "ci95": {"auc_gain": [0.5, 0.7], "ap_gain": [0.4, 0.6]},
        },
        "tls_uncertainty": {
            "ci95": {"auc_gain": [0.2, 0.4], "ap_gain": [0.1, 0.3]},
        },
    }
    real["summary"]["gate_status"] = {
        "mcmc_chain_length_ge_50_tau": True,
        "mcmc_effective_samples_ge_400": True,
        "mcmc_tail_effective_samples_ge_400": True,
        "mcmc_split_rhat_le_1.01": True,
    }
    speed = {"speedup_x": 1500.0, "speedup_ci95": [1200.0, 1800.0],
             "all_mcmc_converged": True}

    report = build_gate_report(synthetic, real, bls, speed)

    assert set(report) >= {"synthetic", "real", "baselines", "status",
                           "gate_thresholds"}
    assert report["gate_thresholds"]["revision"] == "2026-07-19_predeclared_v2"
    assert report["status"]["real_mcmc_n_ge_16"] is True
    assert report["status"]["real_quality_gated_sample_n_ge_30"] is True
    assert report["status"]["detection_candidate_ephemeris_from_bls"] is True
    assert report["status"]["fair_candidate_detection_auc_ge_min"] is True
    assert report["status"]["final_pass"] is True
    assert report["status"][
        "speedup_ge_1000x_at_converged_mcmc_reference"] is True
    assert report["diagnostic_status"][
        "oracle_candidate_detection_auc_ge_0.99"] is True


def test_speedup_is_not_gating_without_converged_mcmc_reference():
    synthetic = {"gate_status": {
        "characterization_sbc_familywise_alpha_0.05": True,
        "characterization_coverage_error_le_0.03": True,
    }}
    real = {"summary": {
        "detection": {"n_detected": 28},
        "mcmc_agreement": {
            key: {"n": 16, "median_wasserstein_prior_fraction": 0.05,
                  "median_wasserstein_width_fraction": 0.4}
            for key in ("RpRs", "aRs", "b")
        },
        "gate_status": {
            "mcmc_chain_length_ge_50_tau": False,
            "mcmc_effective_samples_ge_400": True,
        },
    }}
    bls = {
        "candidate_source": "bls", "n": 5000,
        "bls": {"roc_auc": 0.6}, "transitflow": {"roc_auc": 0.7},
        "uncertainty": {"ci95": {
            "auc_gain": [0.01, 0.03], "ap_gain": [0.01, 0.03]}},
    }

    report = build_gate_report(synthetic, real, bls, {"speedup_x": 13000.0})

    assert report["diagnostic_status"]["raw_speedup_ge_1000x"] is True
    assert report["status"][
        "speedup_ge_1000x_at_converged_mcmc_reference"] is False
    assert report["status"]["final_pass"] is False


def test_real_detection_gate_requires_full_30_object_sample():
    synthetic = {"gate_status": {
        "characterization_sbc_familywise_alpha_0.05": True,
        "characterization_coverage_error_le_0.03": True,
    }}
    real = {"summary": {
        "n_planets": 27,
        "detection": {"n_detected": 27},
        "mcmc_agreement": {},
        "gate_status": {},
    }}

    report = build_gate_report(synthetic, real, {}, {"speedup_x": 0.0})

    assert report["status"]["real_quality_gated_sample_n_ge_30"] is False
    assert report["status"]["real_quality_gated_detection_ge_27_of_30"] is False


def test_mcmc_convergence_requires_diagnostics_for_every_chain():
    summary = {
        "detection": {"detected_fraction": 1.0},
        "mcmc_agreement": {
            key: {"median_wasserstein_prior_fraction": 0.05,
                  "median_wasserstein_width_fraction": 0.4}
            for key in ("RpRs", "aRs", "b")
        },
        "mcmc_conditioning": {
            "n_mcmc": 16, "n_with_tau": 15, "tau_multiple_min": 60.0,
            "n_with_n_eff": 16, "n_eff_min": 500.0,
        },
    }

    gates = real_gate_status(summary)

    assert gates["mcmc_chain_length_ge_50_tau"] is False
    assert gates["mcmc_effective_samples_ge_400"] is True


def test_publishable_report_preserves_real_diagnostic_provenance():
    synthetic = {
        "gate_status": {
            "detection_auc_ge_0.99": True,
            "characterization_sbc_familywise_alpha_0.05": True,
            "characterization_coverage_error_le_0.03": True,
        },
    }
    real = {
        "summary": {
            "detection": {"n_detected": 28},
            "mcmc_agreement": {
                key: {"n": 16, "median_wasserstein_prior_fraction": 0.05,
                      "median_wasserstein_width_fraction": 0.4}
                for key in ("RpRs", "aRs", "b")
            },
            "importance_correction": {"enabled": True, "min_ess_fraction": 0.01},
            "mcmc_conditioning": {"ephemeris_fixed": True},
            "diagnostic_status": {"archive_coverage": False},
            "gate_status": {
                "importance_correction_min_ess_fraction_ge_0.05": False,
            },
        },
    }
    bls = {"candidate_source": "bls", "bls": {}, "transitflow": {}}

    report = build_gate_report(synthetic, real, bls, {"speedup_x": 1500.0})

    assert report["real"]["importance_correction"]["min_ess_fraction"] == 0.01
    assert report["real"]["mcmc_conditioning"]["ephemeris_fixed"] is True
    assert report["real"]["diagnostic_status"]["archive_coverage"] is False
    assert report["status"]["final_pass"] is False


def _gate_report_with_mcmc(agreement: dict) -> dict:
    synthetic = {"gate_status": {
        "characterization_sbc_familywise_alpha_0.05": True,
        "characterization_coverage_error_le_0.03": True,
    }}
    real = {"summary": {
        "detection": {"n_detected": 28},
        "mcmc_agreement": agreement,
        "gate_status": {},
    }}
    return build_gate_report(synthetic, real, {}, {"speedup_x": 0.0})


def test_mcmc_agreement_limits_are_per_parameter():
    """b (and a/Rs) are weakly identified in single-sector data, so their
    predeclared agreement limits are wider than the depth parameter RpRs."""
    base = {
        "RpRs": {"n": 16, "median_wasserstein_prior_fraction": 0.05,
                 "median_wasserstein_width_fraction": 0.45},
        "aRs": {"n": 16, "median_wasserstein_prior_fraction": 0.05,
                "median_wasserstein_width_fraction": 0.70},
        "b": {"n": 16, "median_wasserstein_prior_fraction": 0.12,
              "median_wasserstein_width_fraction": 0.70},
    }
    report = _gate_report_with_mcmc(base)
    assert report["status"]["real_mcmc_prior_fraction_within_limit"] is True
    assert report["status"]["real_mcmc_width_fraction_within_limit"] is True

    tight = {key: dict(value) for key, value in base.items()}
    tight["RpRs"]["median_wasserstein_width_fraction"] = 0.70
    report = _gate_report_with_mcmc(tight)
    assert report["status"]["real_mcmc_width_fraction_within_limit"] is False

    tight = {key: dict(value) for key, value in base.items()}
    tight["b"]["median_wasserstein_prior_fraction"] = 0.20
    report = _gate_report_with_mcmc(tight)
    assert report["status"]["real_mcmc_prior_fraction_within_limit"] is False


def test_null_injection_auc_gate_is_two_sided():
    from scripts.null_injection_check import null_auc_gate

    assert null_auc_gate(0.51, 0.46, 0.56) is True
    assert null_auc_gate(0.54, 0.51, 0.58) is True  # within point margin
    assert null_auc_gate(0.62, 0.57, 0.66) is False  # artifact separability
    assert null_auc_gate(0.38, 0.33, 0.43) is False  # inverted leak also fails
    assert null_auc_gate(float("nan"), 0.4, 0.6) is False


def test_noise_split_is_source_target_disjoint(tmp_path):
    path = tmp_path / "noise.npz"
    segments = np.arange(6 * 8, dtype=float).reshape(6, 8)
    target_ids = np.array(["A", "A", "B", "B", "C", "C"])
    np.savez_compressed(path, segments=segments, target_ids=target_ids)

    train_path, eval_path, meta = prepare_noise_splits(path, tmp_path / "split", 7)

    train = np.load(train_path)
    evaluate = np.load(eval_path)
    assert set(train["target_ids"].astype(str)).isdisjoint(
        set(evaluate["target_ids"].astype(str)))
    assert meta["target_overlap"] == []


def test_noise_three_way_split_is_target_disjoint(tmp_path):
    path = tmp_path / "noise.npz"
    target_ids = np.repeat(np.array(list("ABCDEF")), 2)
    segments = np.arange(len(target_ids) * 8, dtype=float).reshape(-1, 8)
    np.savez_compressed(path, segments=segments, target_ids=target_ids)

    train_path, cal_path, eval_path, meta = prepare_noise_three_way_split(
        path, tmp_path / "three_way", seed=9,
        calibration_fraction=0.2, eval_fraction=0.2)

    train = set(np.load(train_path)["target_ids"].astype(str))
    calibration = set(np.load(cal_path)["target_ids"].astype(str))
    evaluate = set(np.load(eval_path)["target_ids"].astype(str))
    assert train.isdisjoint(calibration)
    assert train.isdisjoint(evaluate)
    assert calibration.isdisjoint(evaluate)
    assert meta["all_disjoint"] is True


def test_development_train_calibration_split_reserves_no_internal_eval(tmp_path):
    path = tmp_path / "noise.npz"
    target_ids = np.repeat(np.array(list("ABCDE")), 2)
    segments = np.arange(len(target_ids) * 8, dtype=float).reshape(-1, 8)
    np.savez_compressed(path, segments=segments, target_ids=target_ids)

    train_path, calibration_path, meta = prepare_noise_train_calibration_split(
        path, tmp_path / "split", seed=4, calibration_fraction=0.2)

    train = set(np.load(train_path)["target_ids"].astype(str))
    calibration = set(np.load(calibration_path)["target_ids"].astype(str))
    assert train.isdisjoint(calibration)
    assert train | calibration == set(target_ids)
    assert meta["evaluation_role"] == "external_frozen_publication_lockbox"


def test_noise_four_way_split_is_target_disjoint(tmp_path):
    path = tmp_path / "noise.npz"
    target_ids = np.repeat(np.array(list("ABCDEFGHIJ")), 2)
    segments = np.arange(len(target_ids) * 8, dtype=float).reshape(-1, 8)
    np.savez_compressed(path, segments=segments, target_ids=target_ids)

    train_path, validation_path, calibration_path, eval_path, meta = \
        prepare_noise_four_way_split(
            path, tmp_path / "four_way", seed=9,
            validation_fraction=0.1, calibration_fraction=0.2,
            eval_fraction=0.2)

    target_sets = [
        set(np.load(role_path)["target_ids"].astype(str))
        for role_path in (
            train_path, validation_path, calibration_path, eval_path)
    ]
    for i, left in enumerate(target_sets):
        for right in target_sets[i + 1:]:
            assert left.isdisjoint(right)
    assert set().union(*target_sets) == set(target_ids)
    assert meta["all_disjoint"] is True


def test_external_eval_split_isolates_checkpoint_validation_targets(tmp_path):
    path = tmp_path / "noise.npz"
    target_ids = np.repeat(np.array(list("ABCDEFGHIJ")), 2)
    segments = np.arange(len(target_ids) * 8, dtype=float).reshape(-1, 8)
    np.savez_compressed(path, segments=segments, target_ids=target_ids)

    train_path, validation_path, calibration_path, meta = \
        prepare_noise_train_validation_calibration_split(
            path, tmp_path / "external_eval", seed=4,
            validation_fraction=0.1, calibration_fraction=0.2)

    target_sets = [
        set(np.load(role_path)["target_ids"].astype(str))
        for role_path in (train_path, validation_path, calibration_path)
    ]
    assert target_sets[0].isdisjoint(target_sets[1])
    assert target_sets[0].isdisjoint(target_sets[2])
    assert target_sets[1].isdisjoint(target_sets[2])
    assert set().union(*target_sets) == set(target_ids)
    assert meta["all_disjoint"] is True
    assert meta["evaluation_role"] == "external_frozen_publication_lockbox"


def test_four_way_split_rejects_fractions_without_training_partition(tmp_path):
    path = tmp_path / "noise.npz"
    target_ids = np.array(list("ABCDEFGHIJ"))
    np.savez_compressed(
        path, segments=np.ones((len(target_ids), 8)), target_ids=target_ids)

    try:
        prepare_noise_four_way_split(
            path, tmp_path / "invalid", seed=1,
            validation_fraction=0.4, calibration_fraction=0.3,
            eval_fraction=0.3)
    except SystemExit as exc:
        assert "leave no training targets" in str(exc)
    else:
        raise AssertionError("invalid four-way fractions did not fail closed")


def test_existing_dataset_requires_every_exact_provenance_shard(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = "configs/default.yaml"
    sim = build_configs(config_path)["simulator"]
    _write_dataset_metadata(
        str(data_dir), sim, n_total=20, n_shards=2, shard_size=10,
        seed=3, noise_lib_path=None)
    def write_shard(path):
        n = 10
        np.savez(path, **{
            "global": np.zeros((n, 2)), "local": np.zeros((n, 2)),
            "theta_std": np.zeros((n, 7)),
                "theta_char_std": np.zeros((n, 5)),
                "theta_char_prior_normal": np.zeros((n, 5)),
                "d": np.zeros(n), "sigma_feat": np.zeros(n),
                "sigma": np.zeros(n), "regime": np.zeros(n, dtype=np.int8),
                "noise_source_index": np.full(n, -1, dtype=np.int32),
                "fold_P": np.ones(n),
                "posterior_valid": np.ones(n),
        })

    write_shard(data_dir / "shard_00000.npz")

    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, None) is False
    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, None,
        require_complete=False) is True

    write_shard(data_dir / "shard_00001.npz")
    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, None) is True

    meta_path = data_dir / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["noise_sampling_unit"] = "segment_uniform_legacy"
    meta_path.write_text(json.dumps(metadata))
    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, None) is False


def test_existing_dataset_accepts_target_uniform_noise_provenance(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    noise_path = tmp_path / "noise.npz"
    np.savez_compressed(
        noise_path,
        segments=np.ones((4, 8)),
        target_ids=np.array(["A", "A", "B", "B"]),
    )
    config_path = "configs/default.yaml"
    sim = build_configs(config_path)["simulator"]
    _write_dataset_metadata(
        str(data_dir), sim, n_total=20, n_shards=2, shard_size=10,
        seed=3, noise_lib_path=str(noise_path))
    for idx in range(2):
        n = 10
        np.savez(data_dir / f"shard_{idx:05d}.npz", **{
            "global": np.zeros((n, 2)), "local": np.zeros((n, 2)),
            "theta_std": np.zeros((n, 7)),
                "theta_char_std": np.zeros((n, 5)),
                "theta_char_prior_normal": np.zeros((n, 5)),
                "d": np.zeros(n), "sigma_feat": np.zeros(n),
                "sigma": np.zeros(n), "regime": np.zeros(n, dtype=np.int8),
                "noise_source_index": np.full(n, -1, dtype=np.int32),
                "fold_P": np.ones(n),
                "posterior_valid": np.ones(n),
        })

    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, noise_path) is True

    other_noise_path = tmp_path / "other_noise.npz"
    np.savez_compressed(
        other_noise_path,
        segments=np.full((4, 8), 2.0),
        target_ids=np.array(["C", "C", "D", "D"]),
    )
    assert validate_existing_dataset(
        data_dir, config_path, 20, 10, 3, other_noise_path) is False


def test_existing_bls_detector_dataset_requires_bls_domain_provenance(tmp_path):
    data_dir = tmp_path / "detector"
    data_dir.mkdir()
    config_path = "configs/default.yaml"
    sim = simulator_config_for_candidate_domain(
        config_path, "bls_detection")["simulator"]
    _write_dataset_metadata(
        str(data_dir), sim, n_total=10, n_shards=1, shard_size=10,
        seed=5, noise_lib_path=None)
    np.savez(data_dir / "shard_00000.npz", **{
        "global": np.zeros((10, 2)), "local": np.zeros((10, 2)),
        "theta_std": np.zeros((10, 7)),
            "theta_char_std": np.zeros((10, 5)),
            "theta_char_prior_normal": np.zeros((10, 5)),
            "d": np.zeros(10), "sigma_feat": np.zeros(10),
            "sigma": np.zeros(10), "regime": np.zeros(10, dtype=np.int8),
            "noise_source_index": np.full(10, -1, dtype=np.int32),
            "fold_P": np.ones(10),
            "posterior_valid": np.zeros(10), "candidate_kind": np.ones(10),
    })

    assert validate_existing_dataset(
        data_dir, config_path, 10, 10, 5, None,
        candidate_domain="bls_detection") is True
    assert validate_existing_dataset(
        data_dir, config_path, 10, 10, 5, None) is False
