import numpy as np

from transitflow.evaluation import (
    central_interval_coverage,
    coverage_calibration_error,
    detection_metrics,
    jensen_shannon_1d,
    marginal_wasserstein,
    posterior_contraction,
    sbc_ranks,
    sbc_uniformity,
)


def test_sbc_calibrated_is_uniform():
    rng = np.random.default_rng(0)
    N, L, D = 2000, 500, 3
    true = rng.standard_normal((N, D))
    # calibrated: posterior == prior == N(0,1), independent of true
    samples = rng.standard_normal((N, L, D))
    ranks = sbc_ranks(true, samples)
    u = sbc_uniformity(ranks)
    assert all(p > 0.01 for p in u["pvalue"])  # not rejected -> uniform


def test_sbc_overconfident_is_detected():
    rng = np.random.default_rng(1)
    N, L, D = 2000, 500, 2
    true = rng.standard_normal((N, D))
    # overconfident: posterior far too narrow around the truth
    samples = true[:, None, :] + 0.01 * rng.standard_normal((N, L, D))
    ranks = sbc_ranks(true, samples)
    u = sbc_uniformity(ranks)
    assert all(p < 1e-3 for p in u["pvalue"])  # strongly non-uniform


def test_coverage_calibrated():
    rng = np.random.default_rng(2)
    N, L, D = 4000, 800, 2
    true = rng.standard_normal((N, D))
    samples = rng.standard_normal((N, L, D))
    cov = central_interval_coverage(true, samples)
    cce = coverage_calibration_error(cov["levels"], cov["coverage_overall"])
    assert cce < 0.05


def test_detection_metrics_perfect():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = detection_metrics(labels, scores)
    assert m["roc_auc"] == 1.0
    assert m["average_precision"] == 1.0


def test_posterior_contraction_and_distances():
    rng = np.random.default_rng(3)
    N, L, D = 50, 1000, 3
    true = rng.standard_normal((N, D))
    samples = true[:, None, :] + 0.3 * rng.standard_normal((N, L, D))
    contraction = posterior_contraction(true, samples, prior_std=np.ones(D))
    assert contraction.shape == (N, D)
    assert np.median(contraction) > 0.5  # posterior tighter than prior

    a = rng.standard_normal((1000, D))
    b = rng.standard_normal((1000, D))
    w = marginal_wasserstein(a, b)
    assert w.shape == (D,) and np.all(w >= 0)
    js = jensen_shannon_1d(a, b)
    assert js.shape == (D,) and np.all(js >= -1e-9)
