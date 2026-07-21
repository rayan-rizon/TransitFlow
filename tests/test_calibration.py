import numpy as np
from scipy.special import ndtri

from transitflow.calibration import (
    PosteriorAffineCalibration,
    calibration_rank_diagnostics,
    fit_affine_calibration,
    load_for_checkpoint,
    sha256_file,
)
from transitflow.noise import NoiseLibrary
from scripts.calibrate_posterior import (
    combine_calibration_candidates,
    collect_target_balanced_calibration_cases,
    select_calibration_candidates_by_dimension,
    split_calibration_noise_targets,
    temper_calibration,
)


def test_affine_calibration_round_trip_and_logdet():
    cal = PosteriorAffineCalibration(
        scale=np.array([2.0, 0.5]), offset=np.array([0.2, -0.3]))
    raw = np.array([[0.1, -1.2], [0.8, 0.4]])
    center = np.array([[0.0, -1.0], [0.5, 0.2]])

    calibrated = cal.apply(raw, center)

    assert np.allclose(cal.inverse(calibrated, center), raw)
    assert np.isclose(cal.log_abs_det, np.log(2.0) + np.log(0.5))


def test_tempered_calibration_interpolates_in_latent_space():
    candidate = PosteriorAffineCalibration(
        scale=np.array([4.0]), offset=np.array([0.8]),
        center_slope=np.array([1.6]), lower=np.array([-2.0]),
        upper=np.array([2.0]), space="bounded_latent_probit",
        center_quadratic=np.array([0.4]), log_scale_slope=np.array([0.2]))
    identity = temper_calibration(candidate, 0.0)
    halfway = temper_calibration(candidate, 0.5)

    assert np.allclose(identity.scale, [1.0])
    assert np.allclose(identity.offset, [0.0])
    assert np.allclose(identity.center_slope, [1.0])
    assert np.allclose(halfway.scale, [2.0])
    assert np.allclose(halfway.offset, [0.4])
    assert np.allclose(halfway.center_slope, [1.3])


def test_target_balanced_selection_reports_equal_target_allocation(
        prior, tiny_model_cfg, monkeypatch):
    from transitflow.simulator import SimConfig

    class DummyInference:
        pass

    class DummyModel:
        cfg = type("Config", (), {"param_dim": 5})()

    calls = []

    def fake_collect(_inference, _simulator, _model, n_cases, n_posterior,
                     batch_size, seed, role):
        calls.append((n_cases, batch_size, seed, role))
        return (np.zeros((n_cases, 5)), np.zeros((n_cases, n_posterior, 5)),
                np.zeros((n_cases, 5)))

    monkeypatch.setattr(
        "scripts.calibrate_posterior.collect_calibration_cases", fake_collect)
    noise = NoiseLibrary(
        np.ones((4, 32)), np.array(["a", "a", "b", "b"]))
    theta, posterior, center, meta = collect_target_balanced_calibration_cases(
        DummyInference(), SimConfig(n_raw=32), prior, noise, DummyModel(),
        n_cases=5, n_posterior=3, batch_size=64, seed=7, role="selection")

    assert theta.shape == (5, 5)
    assert posterior.shape == (5, 3, 5)
    assert center.shape == (5, 5)
    assert [call[0] for call in calls] == [3, 3]
    assert all(call[1] == 12 for call in calls)
    assert meta["sampling"] == "target_uniform_then_simulation_v1"


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


def test_probit_calibration_round_trip_logdet_and_prior_matching():
    bound = np.sqrt(3.0)
    cal = PosteriorAffineCalibration(
        scale=np.array([1.0]), offset=np.array([0.0]),
        center_slope=np.array([0.0]), lower=np.array([-bound]),
        upper=np.array([bound]), space="bounded_probit")
    probability = (np.arange(4096, dtype=np.float64) + 0.5) / 4096
    raw = ndtri(probability)[:, None]
    center = np.zeros_like(raw)

    calibrated = cal.apply(raw, center)
    recovered = cal.inverse(calibrated, center)

    assert np.allclose(recovered, raw, atol=1e-9)
    assert np.isclose(calibrated.mean(), 0.0, atol=1e-12)
    assert np.isclose(calibrated.var(), 1.0, atol=1e-6)
    eps = 1e-6
    numerical = np.log(np.abs(
        (cal.apply(raw + eps, center) - cal.apply(raw - eps, center))
        / (2.0 * eps)))[:, 0]
    assert np.allclose(
        cal.log_abs_det_at(calibrated, center), numerical, atol=2e-5)


def test_bounded_latent_identity_is_exact_and_has_zero_logdet():
    bound = np.sqrt(3.0)
    cal = PosteriorAffineCalibration(
        scale=np.ones(1), offset=np.zeros(1), center_slope=np.ones(1),
        lower=np.array([-bound]), upper=np.array([bound]),
        space="bounded_latent_probit")
    raw = np.linspace(-1.6, 1.6, 257)[:, None]
    center = np.linspace(-1.2, 1.2, 257)[:, None]

    calibrated = cal.apply(raw, center)

    assert np.allclose(calibrated, raw, atol=1e-12)
    assert np.allclose(cal.inverse(calibrated, center), raw, atol=1e-12)
    assert np.allclose(cal.log_abs_det_at(calibrated, center), 0.0, atol=1e-12)
    assert PosteriorAffineCalibration.load(cal.to_dict()).bounded_latent_input


def test_bounded_latent_logdet_matches_numerical_derivative():
    bound = np.sqrt(3.0)
    cal = PosteriorAffineCalibration(
        scale=np.array([1.35]), offset=np.array([-0.12]),
        center_slope=np.array([0.85]), lower=np.array([-bound]),
        upper=np.array([bound]), space="bounded_latent_probit",
        center_quadratic=np.array([0.06]),
        log_scale_slope=np.array([-0.08]))
    raw = np.array([[-1.1], [-0.2], [0.7], [1.25]])
    center = np.array([[-0.8], [0.1], [0.5], [1.0]])
    calibrated = cal.apply(raw, center)
    eps = 1e-6
    numerical = np.log(np.abs(
        (cal.apply(raw + eps, center) - cal.apply(raw - eps, center))
        / (2.0 * eps)))[:, 0]

    assert np.allclose(cal.inverse(calibrated, center), raw, atol=1e-10)
    assert np.allclose(
        cal.log_abs_det_at(calibrated, center), numerical, atol=2e-5)


def test_bounded_nonlinear_calibration_recovers_conditional_bias():
    rng = np.random.default_rng(77)
    n, n_post = 320, 128
    bound = np.sqrt(3.0)
    center = rng.uniform(-1.2, 1.2, size=(n, 1))
    truth_latent = -0.35 + 0.15 * center + 0.95 * center ** 2
    truth = bound * np.tanh(truth_latent + rng.normal(0.0, 0.04, size=(n, 1)))
    posterior = center[:, None, :] + rng.normal(
        0.0, 0.11 * np.exp(0.35 * center[:, None, :]), size=(n, n_post, 1))
    posterior = np.clip(posterior, -bound + 1e-5, bound - 1e-5)

    calibration, diagnostics = fit_affine_calibration(
        truth, posterior, center,
        bounds=(np.array([-bound]), np.array([bound])), optimizer_seed=29,
        bounded_link="tanh")

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
    center = np.clip(center, -bound + 1e-5, bound - 1e-5)
    posterior = np.clip(posterior, -bound + 1e-5, bound - 1e-5)

    calibration, diagnostics = fit_affine_calibration(
        truth, posterior, center,
        bounds=(np.array([-bound]), np.array([bound])), optimizer_seed=19)
    calibrated = calibration.apply(posterior, center)

    assert calibration.space == "bounded_latent_probit"
    assert diagnostics["rank_cvm_after_by_dim"][0] \
        < diagnostics["rank_cvm_before_by_dim"][0]
    assert diagnostics["coverage_error_after_mean"] \
        < diagnostics["coverage_error_before_mean"]
    assert calibrated.min() > -bound
    assert calibrated.max() < bound


def test_simple_bounded_calibration_disables_conditional_curvature():
    rng = np.random.default_rng(913)
    n, n_post = 180, 96
    bound = np.sqrt(3.0)
    truth = rng.uniform(-1.3, 1.3, size=(n, 1))
    center = truth + 0.25 + rng.normal(0.0, 0.1, size=(n, 1))
    posterior = center[:, None, :] + rng.normal(
        0.0, 0.2, size=(n, n_post, 1))
    center = np.clip(center, -bound + 1e-5, bound - 1e-5)
    posterior = np.clip(posterior, -bound + 1e-5, bound - 1e-5)

    calibration, _ = fit_affine_calibration(
        truth, posterior, center,
        bounds=(np.array([-bound]), np.array([bound])),
        optimizer_seed=43, complexity="simple")

    assert np.array_equal(calibration.center_quadratic, np.zeros(1))
    assert np.array_equal(calibration.log_scale_slope, np.zeros(1))
    score = calibration_rank_diagnostics(
        truth, posterior, center, calibration)
    assert np.isfinite(score["selection_score"])


def test_calibration_target_selection_split_is_source_disjoint():
    target_ids = np.repeat(np.array(list("ABCDEFGHIJ")), 2)
    noise = NoiseLibrary(
        np.ones((len(target_ids), 32)), target_ids=target_ids)

    fit, selection, metadata = split_calibration_noise_targets(
        noise, seed=17, selection_fraction=0.3)

    assert set(fit.target_ids).isdisjoint(set(selection.target_ids))
    assert set(fit.target_ids) | set(selection.target_ids) == set(target_ids)
    assert metadata["target_overlap"] == []


def test_calibrator_family_selection_uses_heldout_rank_score_per_dimension():
    n, n_post = 100, 101
    grid = np.linspace(-1.0, 1.0, n_post)
    posterior = np.broadcast_to(
        grid[None, :, None], (n, n_post, 2)).copy()
    uniform_truth = np.linspace(-0.98, 0.98, n)
    theta = np.column_stack([uniform_truth, uniform_truth + 1.0])
    center = np.zeros_like(theta)
    identity = PosteriorAffineCalibration(np.ones(2), np.zeros(2))
    shifted = PosteriorAffineCalibration(np.ones(2), np.ones(2))

    selected, scores, per_dimension = select_calibration_candidates_by_dimension(
        {"identity": identity, "shifted": shifted},
        theta, posterior, center)
    hybrid = combine_calibration_candidates(
        {"identity": identity, "shifted": shifted}, selected)

    assert selected == ["identity", "shifted"]
    assert per_dimension[0]["identity"] < per_dimension[0]["shifted"]
    assert per_dimension[1]["shifted"] < per_dimension[1]["identity"]
    assert np.array_equal(hybrid.offset, np.array([0.0, 1.0]))
    assert set(scores) == {"identity", "shifted"}


def test_candidate_tempering_strength_orders_the_shrinkage_path():
    from scripts.calibrate_posterior import candidate_tempering_strength

    assert candidate_tempering_strength("identity") == 0.0
    assert candidate_tempering_strength("simple_050") == 0.5
    assert candidate_tempering_strength("simple_075") == 0.75
    assert candidate_tempering_strength("simple_100") == 1.0


def test_one_standard_error_rule_prefers_shrinkage_on_a_statistical_tie():
    """A candidate that only wins inside the selection noise must not be taken.

    This is the seed-0 `RpRs` failure mode: the aggressive correction won the
    pooled argmin on a handful of selection targets and then failed to transfer.
    """
    rng = np.random.default_rng(0)
    n_targets, per_target, n_post = 8, 20, 101
    n = n_targets * per_target
    grid = np.linspace(-1.0, 1.0, n_post)
    posterior = np.broadcast_to(grid[None, :, None], (n, n_post, 1)).copy()
    theta = np.linspace(-0.98, 0.98, n)[:, None]
    center = np.zeros_like(theta)
    groups = np.repeat(np.arange(n_targets), per_target)
    candidates = {
        "identity": PosteriorAffineCalibration(np.ones(1), np.zeros(1)),
        # A negligible offset: any score difference is far inside target scatter.
        "simple_100": PosteriorAffineCalibration(np.ones(1), np.full(1, 1e-4)),
    }

    selected, _, per_dimension = select_calibration_candidates_by_dimension(
        candidates, theta, posterior, center, groups=groups)

    assert selected == ["identity"]
    assert per_dimension[0]["_selection_rule"] == "one_standard_error_across_targets"
    assert per_dimension[0]["_standard_error"] > 0.0


def test_one_standard_error_rule_still_takes_a_decisively_better_candidate():
    """Shrinkage preference must not veto a real, large improvement."""
    n_targets, per_target, n_post = 6, 20, 101
    n = n_targets * per_target
    grid = np.linspace(-1.0, 1.0, n_post)
    posterior = np.broadcast_to(grid[None, :, None], (n, n_post, 1)).copy()
    # Truth is shifted a long way up: identity ranks are badly non-uniform.
    theta = (np.linspace(-0.98, 0.98, n) + 1.0)[:, None]
    center = np.zeros_like(theta)
    groups = np.repeat(np.arange(n_targets), per_target)
    candidates = {
        "identity": PosteriorAffineCalibration(np.ones(1), np.zeros(1)),
        "simple_100": PosteriorAffineCalibration(np.ones(1), np.ones(1)),
    }

    selected, _, _ = select_calibration_candidates_by_dimension(
        candidates, theta, posterior, center, groups=groups)

    assert selected == ["simple_100"]


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
