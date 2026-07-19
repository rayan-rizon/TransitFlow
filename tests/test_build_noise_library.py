import io
import json
from unittest.mock import patch

from scripts.build_noise_library import (
    _collect_target_segments,
    _extract_corrupt_cache_path,
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


def test_extract_corrupt_cache_path_matches_lightkurve_message():
    message = (
        "Error in reading Data product /root/.cache/lightkurve/mastDownload/"
        "TESS/tess2025206162959-s0095-0000000267083739-0292-s/"
        "tess2025206162959-s0095-0000000267083739-0292-s_lc.fits of type "
        "TessLightCurve .\nThis file may be corrupt due to an interrupted "
        "download. Please remove it from your disk and try again."
    )

    path = _extract_corrupt_cache_path(message)

    assert path == (
        "/root/.cache/lightkurve/mastDownload/TESS/"
        "tess2025206162959-s0095-0000000267083739-0292-s/"
        "tess2025206162959-s0095-0000000267083739-0292-s_lc.fits")


def test_extract_corrupt_cache_path_returns_none_for_unrelated_errors():
    assert _extract_corrupt_cache_path("no data found") is None
    assert _extract_corrupt_cache_path("") is None


def test_collect_target_segments_retries_once_past_corrupt_cache(tmp_path):
    corrupt_file = tmp_path / "truncated.fits"
    corrupt_file.write_bytes(b"short")
    corrupt_message = (
        f"Error in reading Data product {corrupt_file} of type generic .\n"
        "This file may be corrupt due to an interrupted download. Please "
        "remove it from your disk and try again."
    )
    outcomes = [
        ("HIP 1", [], ["  downloading HIP 1 ..."],
         {"target": "HIP 1", "accepted": False, "error": corrupt_message}),
        ("HIP 1", [np.zeros(4)], ["  downloading HIP 1 ..."],
         {"target": "HIP 1", "accepted": True, "n_segments": 1}),
    ]

    with patch(
        "scripts.build_noise_library._collect_target_segments_once",
        side_effect=outcomes,
    ) as mocked:
        tgt, segments, logs, metrics = _collect_target_segments(
            "HIP 1", "TESS", 4, 1, 2500.0)

    assert mocked.call_count == 2
    assert not corrupt_file.exists()
    assert metrics["accepted"] is True
    assert len(segments) == 1
    assert any("retrying HIP 1" in line for line in logs)


def test_collect_target_segments_gives_up_after_max_retries(tmp_path):
    corrupt_file = tmp_path / "always_truncated.fits"

    def always_corrupt(*args, **kwargs):
        corrupt_file.write_bytes(b"short")
        message = (
            f"Error in reading Data product {corrupt_file} of type "
            "generic .\nThis file may be corrupt due to an interrupted "
            "download. Please remove it from your disk and try again."
        )
        return (
            "HIP 2", [], ["  downloading HIP 2 ..."],
            {"target": "HIP 2", "accepted": False, "error": message},
        )

    with patch(
        "scripts.build_noise_library._collect_target_segments_once",
        side_effect=always_corrupt,
    ) as mocked:
        tgt, segments, logs, metrics = _collect_target_segments(
            "HIP 2", "TESS", 4, 1, 2500.0, max_retries=1)

    assert mocked.call_count == 2
    assert metrics["accepted"] is False


def test_emit_logs_uses_fallback_after_closed_primary_stream():
    primary = io.StringIO()
    primary.close()
    fallback = io.StringIO()

    emit_logs(["first", "second"], primary=primary, fallback=fallback)

    assert fallback.getvalue() == "first\nsecond\n"
