from scripts.evaluate import sbc_gate
from types import SimpleNamespace

import numpy as np

from scripts.validate_real import (
    fold_bin_fixed_ephemeris,
    passes_real_quality,
    real_diagnostic_status,
    real_gate_status,
)


def test_sbc_gate_controls_familywise_error():
    gate = sbc_gate([0.37, 0.0115, 0.14, 0.11, 0.15])

    assert gate["bonferroni_alpha_per_test"] == 0.01
    assert gate["pass"] is True
    assert gate["all_raw_p_gt_0.05"] is False


def test_sbc_gate_rejects_clear_miscalibration():
    gate = sbc_gate([1.3e-5, 0.15, 0.27, 0.89, 0.87])

    assert gate["pass"] is False


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
