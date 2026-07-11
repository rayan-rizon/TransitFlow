import numpy as np

from transitflow.calibration import (
    PosteriorAffineCalibration,
    fit_affine_calibration,
    load_for_checkpoint,
    sha256_file,
)


def test_affine_calibration_round_trip_and_logdet():
    cal = PosteriorAffineCalibration(
        scale=np.array([2.0, 0.5]), offset=np.array([0.2, -0.3]))
    raw = np.array([[0.1, -1.2], [0.8, 0.4]])
    center = np.array([[0.0, -1.0], [0.5, 0.2]])

    calibrated = cal.apply(raw, center)

    assert np.allclose(cal.inverse(calibrated, center), raw)
    assert np.isclose(cal.log_abs_det, np.log(2.0) + np.log(0.5))


def test_fit_affine_calibration_improves_heldout_coverage_objective():
    rng = np.random.default_rng(41)
    n, samples, dim = 500, 256, 2
    truth = rng.normal(size=(n, dim))
    noisy_center = truth + rng.normal(0.0, 1.0, size=(n, dim))
    posterior = noisy_center[:, None, :] + rng.normal(
        0.0, 0.35, size=(n, samples, dim))

    calibration, diagnostics = fit_affine_calibration(
        truth, posterior, noisy_center)

    assert diagnostics["coverage_error_after_mean"] \
        < diagnostics["coverage_error_before_mean"]
    assert np.all(calibration.scale > 1.0)


def test_calibration_rejects_wrong_dimension():
    cal = PosteriorAffineCalibration(np.ones(2), np.zeros(2))
    try:
        cal.apply(np.zeros((4, 3)), np.zeros((4, 2)))
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("wrong-dimensional calibration did not fail")


def test_calibration_checkpoint_provenance_fails_closed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    artifact = PosteriorAffineCalibration(
        np.ones(2), np.zeros(2), metadata={
            "checkpoint_sha256": sha256_file(checkpoint)})

    assert load_for_checkpoint(artifact, checkpoint) is artifact
    checkpoint.write_bytes(b"checkpoint-b")
    try:
        load_for_checkpoint(artifact, checkpoint)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint did not fail closed")
