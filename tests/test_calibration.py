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


def test_bounded_calibration_round_trip_support_and_logdet():
    cal = PosteriorAffineCalibration(
        scale=np.array([1.4]),
        offset=np.array([-0.2]),
        center_slope=np.array([0.9]),
        lower=np.array([-np.sqrt(3.0)]),
        upper=np.array([np.sqrt(3.0)]),
        space="bounded_tanh",
        center_quadratic=np.array([0.12]),
        log_scale_slope=np.array([-0.18]),
    )
    raw = np.array([[-2.5], [0.3], [3.2]])
    center = np.array([[0.1], [0.2], [-0.4]])

    calibrated = cal.apply(raw, center)
    recovered = cal.inverse(calibrated, center)

    assert np.allclose(recovered, raw, atol=1e-8)
    assert np.all(calibrated > cal.lower)
    assert np.all(calibrated < cal.upper)
    eps = 1e-6
    y_hi = cal.apply(raw + eps, center)
    y_lo = cal.apply(raw - eps, center)
    numerical = np.log(np.abs((y_hi - y_lo) / (2.0 * eps)))[:, 0]
    assert np.allclose(cal.log_abs_det_at(calibrated, center), numerical, atol=1e-5)
    loaded = PosteriorAffineCalibration.load(cal.to_dict())
    assert loaded.space == "bounded_tanh"
    assert np.allclose(loaded.apply(raw, center), calibrated)


def test_bounded_nonlinear_calibration_recovers_conditional_bias():
    rng = np.random.default_rng(77)
    n, n_post = 320, 128
    bound = np.sqrt(3.0)
    center = rng.uniform(-1.2, 1.2, size=(n, 1))
    truth_latent = -0.35 + 0.15 * center + 0.95 * center ** 2
    truth = bound * np.tanh(truth_latent + rng.normal(0.0, 0.04, size=(n, 1)))
    posterior = center[:, None, :] + rng.normal(
        0.0, 0.11 * np.exp(0.35 * center[:, None, :]), size=(n, n_post, 1))

    calibration, diagnostics = fit_affine_calibration(
        truth, posterior, center,
        bounds=(np.array([-bound]), np.array([bound])), optimizer_seed=29)

    assert abs(calibration.center_quadratic[0]) > 0.05
    assert diagnostics["rank_cvm_after_by_dim"][0] < 0.05
    assert diagnostics["coverage_error_after_mean"] < 0.04


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


def test_bounded_rank_calibration_improves_bias_and_preserves_support():
    rng = np.random.default_rng(413)
    n, n_post = 240, 128
    bound = np.sqrt(3.0)
    truth = rng.uniform(-1.4, 1.4, size=(n, 1))
    center = truth + 0.35 + rng.normal(0.0, 0.08, size=(n, 1))
    posterior = center[:, None, :] + rng.normal(
        0.0, 0.18, size=(n, n_post, 1))

    calibration, diagnostics = fit_affine_calibration(
        truth, posterior, center,
        bounds=(np.array([-bound]), np.array([bound])), optimizer_seed=19)
    calibrated = calibration.apply(posterior, center)

    assert calibration.space == "bounded_tanh"
    assert diagnostics["rank_cvm_after_by_dim"][0] \
        < diagnostics["rank_cvm_before_by_dim"][0]
    assert diagnostics["coverage_error_after_mean"] \
        < diagnostics["coverage_error_before_mean"]
    assert calibrated.min() > -bound
    assert calibrated.max() < bound


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
