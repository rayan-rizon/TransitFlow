from scripts.evaluate import sbc_gate
from scripts.validate_real import real_gate_status


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

    assert gates["archive_detected_char_cov68_ge_0.5"] is True
    assert gates["archive_detected_char_cov95_ge_0.8"] is True
    assert gates["mcmc_characterization_prior_fraction_le_0.1"] is True
