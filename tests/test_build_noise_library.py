import io

from scripts.build_noise_library import (
    emit_logs,
    robust_point_to_point_ppm,
    segment_flux_products,
)

import numpy as np


def test_robust_point_to_point_ppm_tracks_white_noise():
    rng = np.random.default_rng(44)
    flux = 1.0 + rng.normal(0.0, 500e-6, size=20000)
    measured = robust_point_to_point_ppm(flux)
    assert 450.0 < measured < 550.0


def test_products_are_segmented_without_crossing_sector_boundaries():
    first = 1.0 + np.arange(6) * 1e-6
    second = 1.0 + np.arange(6) * -1e-6

    segments, metrics = segment_flux_products(
        [first, second], n_raw=4, max_segments=4,
        max_point_to_point_ppm=2500.0)

    assert len(segments) == 2
    assert [metric["n_segments"] for metric in metrics] == [1, 1]
    assert np.allclose(segments[0], first[:4] / np.median(first))
    assert np.allclose(segments[1], second[:4] / np.median(second))


def test_product_quality_rejection_is_recorded():
    noisy = 1.0 + np.random.default_rng(9).normal(0.0, 0.02, size=200)

    segments, metrics = segment_flux_products(
        [noisy], n_raw=10, max_segments=2,
        max_point_to_point_ppm=2500.0)

    assert segments == []
    assert metrics[0]["rejection"] == "point_to_point_scatter"


def test_emit_logs_uses_fallback_after_closed_primary_stream():
    primary = io.StringIO()
    primary.close()
    fallback = io.StringIO()

    emit_logs(["first", "second"], primary=primary, fallback=fallback)

    assert fallback.getvalue() == "first\nsecond\n"
