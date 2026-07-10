import io

from scripts.build_noise_library import emit_logs


def test_emit_logs_uses_fallback_after_closed_primary_stream():
    primary = io.StringIO()
    primary.close()
    fallback = io.StringIO()

    emit_logs(["first", "second"], primary=primary, fallback=fallback)

    assert fallback.getvalue() == "first\nsecond\n"
