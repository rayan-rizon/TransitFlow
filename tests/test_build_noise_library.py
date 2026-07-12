import io

from scripts.build_noise_library import emit_logs, robust_point_to_point_ppm

import numpy as np


def test_robust_point_to_point_ppm_tracks_white_noise():
    rng = np.random.default_rng(44)
    flux = 1.0 + rng.normal(0.0, 500e-6, size=20000)
    measured = robust_point_to_point_ppm(flux)
    assert 450.0 < measured < 550.0


def test_emit_logs_uses_fallback_after_closed_primary_stream():
    primary = io.StringIO()
    primary.close()
    fallback = io.StringIO()

    emit_logs(["first", "second"], primary=primary, fallback=fallback)

    assert fallback.getvalue() == "first\nsecond\n"
