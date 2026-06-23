import numpy as np
import torch

from transitflow.priors import (
    PARAM_NAMES,
    TransitPrior,
    kipping_to_quadratic,
    quadratic_to_kipping,
)


def test_sample_shape_and_bounds(prior, rng):
    s = prior.sample(2000, rng)
    assert s.shape == (2000, 7)
    # P in [0.5, 13], RpRs in [0.01, 0.15], b in [0, 1.1], q1/q2 in [0,1]
    assert s[:, 0].min() >= 0.5 - 1e-6 and s[:, 0].max() <= 13 + 1e-6
    assert s[:, 2].min() >= 0.01 - 1e-6 and s[:, 2].max() <= 0.15 + 1e-6
    assert s[:, 4].min() >= 0 and s[:, 4].max() <= 1.1 + 1e-6
    assert s[:, 5].min() >= 0 and s[:, 6].max() <= 1 + 1e-6


def test_std_roundtrip(prior, rng):
    s = prior.sample(500, rng)
    z = prior.physical_to_std(s)
    back = prior.std_to_physical(z, clip=False)
    assert np.allclose(s, back, rtol=1e-6, atol=1e-6)


def test_std_is_standardized(prior, rng):
    z = prior.physical_to_std(prior.sample(50000, rng))
    # standardized prior should be ~zero mean, ~unit variance per dim
    assert np.allclose(z.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(z.std(axis=0), 1.0, atol=0.05)


def test_torch_matches_numpy(prior, rng):
    s = prior.sample(64, rng).astype(np.float32)
    z_np = prior.physical_to_std(s)
    z_t = prior.physical_to_std_torch(torch.from_numpy(s)).numpy()
    assert np.allclose(z_np, z_t, atol=1e-4)
    back_t = prior.std_to_physical_torch(torch.from_numpy(z_np.astype(np.float32)),
                                         clip=False).numpy()
    assert np.allclose(s, back_t, rtol=1e-3, atol=1e-4)


def test_log_prob_support(prior, rng):
    s = prior.sample(100, rng)
    lp = prior.log_prob_physical(s)
    assert np.all(np.isfinite(lp))
    bad = s.copy()
    bad[:, 0] = 1e6  # period outside support
    assert np.all(~np.isfinite(prior.log_prob_physical(bad)))


def test_kipping_validity(rng):
    q1 = rng.uniform(0, 1, 100000)
    q2 = rng.uniform(0, 1, 100000)
    u1, u2 = kipping_to_quadratic(q1, q2)
    # physically valid quadratic LD: u1 > 0 and u1 + u2 < 1
    assert np.all(u1 >= -1e-9)
    assert np.all(u1 + u2 <= 1 + 1e-9)
    # inverse recovers q1, q2
    q1b, q2b = quadratic_to_kipping(u1, u2)
    ok = q1 > 1e-6
    assert np.allclose(q1[ok], q1b[ok], atol=1e-6)


def test_param_names():
    assert PARAM_NAMES == ("P", "t0_phase", "RpRs", "aRs", "b", "q1", "q2")
