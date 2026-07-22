"""Tests for the predeclared BF16-vs-FP32 equivalence gate.

Exercises only the pure gate logic (``equivalence_status``); the full script
needs a GPU + checkpoint and runs on Vast.
"""
from scripts.bf16_fp32_equivalence import (
    DETECTION_ABS_TOL,
    MEDIAN_SHIFT_STD_TOL,
    WIDTH_RATIO_TOL,
    equivalence_status,
)


def test_equivalence_passes_when_all_within_tolerance():
    status = equivalence_status(
        max_det_abs=DETECTION_ABS_TOL / 2,
        max_median_shift_std=MEDIAN_SHIFT_STD_TOL / 2,
        max_width_ratio_abs_dev=WIDTH_RATIO_TOL / 2)
    assert status["equivalent"] is True
    assert all(v for k, v in status.items() if k != "equivalent")


def test_equivalence_fails_on_detection_drift():
    status = equivalence_status(
        max_det_abs=DETECTION_ABS_TOL * 2,
        max_median_shift_std=0.0, max_width_ratio_abs_dev=0.0)
    assert status["detection_prob_within_tol"] is False
    assert status["equivalent"] is False


def test_equivalence_fails_on_posterior_median_shift():
    status = equivalence_status(
        max_det_abs=0.0,
        max_median_shift_std=MEDIAN_SHIFT_STD_TOL * 2,
        max_width_ratio_abs_dev=0.0)
    assert status["posterior_median_within_tol"] is False
    assert status["equivalent"] is False


def test_equivalence_fails_on_width_inflation():
    status = equivalence_status(
        max_det_abs=0.0, max_median_shift_std=0.0,
        max_width_ratio_abs_dev=WIDTH_RATIO_TOL * 2)
    assert status["posterior_width_within_tol"] is False
    assert status["equivalent"] is False


def test_equivalence_boundary_is_inclusive():
    # exactly at tolerance passes (<= comparison)
    status = equivalence_status(
        max_det_abs=DETECTION_ABS_TOL,
        max_median_shift_std=MEDIAN_SHIFT_STD_TOL,
        max_width_ratio_abs_dev=WIDTH_RATIO_TOL)
    assert status["equivalent"] is True
