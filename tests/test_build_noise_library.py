import io
import json

from scripts.build_noise_library import (
    emit_logs,
    robust_point_to_point_ppm,
    segment_flux_products,
    load_extension_archive,
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


def test_extension_archive_requires_matching_provenance(tmp_path):
    archive = tmp_path / "old.npz"
    np.savez_compressed(
        archive,
        segments=np.ones((2, 8)),
        target_ids=np.array(["HIP 1", "HIP 2"]),
    )
    (tmp_path / "old.npz.metadata.json").write_text(json.dumps({
        "quality": [
            {"target": "HIP 1", "accepted": True},
            {"target": "HIP 2", "accepted": True},
        ]
    }))

    segments, target_ids, quality = load_extension_archive(
        str(archive), 8, ["HIP 1", "HIP 2", "HIP 3"])

    assert len(segments) == 2
    assert target_ids == ["HIP 1", "HIP 2"]
    assert set(quality) == {"HIP 1", "HIP 2"}

    try:
        load_extension_archive(str(archive), 8, ["HIP 1"])
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("extension outside new provenance did not fail")


def test_emit_logs_uses_fallback_after_closed_primary_stream():
    primary = io.StringIO()
    primary.close()
    fallback = io.StringIO()

    emit_logs(["first", "second"], primary=primary, fallback=fallback)

    assert fallback.getvalue() == "first\nsecond\n"
