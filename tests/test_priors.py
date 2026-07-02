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


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _stellar_prior(mean=0.0, std=0.25):
    return TransitPrior(
        TransitPrior.default_specs("tess"),
        a_rs_prior_mode="stellar_density",
        stellar_density_log10_mean=mean,
        stellar_density_log10_std=std,
    )


def test_stellar_density_aRs_density_is_normalized():
    # p(a/Rs | P) must integrate to ~1 on its support (proper density).
    prior = _stellar_prior()
    grid = np.linspace(3.0, 50.0, 400000)
    P = np.full(grid.size, 5.0)
    pdf = np.exp(prior._stellar_density_logpdf_a_rs(P, grid))
    norm = float(_trapz(pdf, grid))
    assert abs(norm - 1.0) < 0.02, norm


def test_stellar_density_prior_matches_simulator_sampling():
    # The MCMC prior density must agree with the forward simulator's a/Rs draw,
    # otherwise the amortized-vs-MCMC comparison is biased.
    from transitflow.simulator import SimConfig, TransitSimulator

    cfg = SimConfig(regime="tess", a_rs_prior_mode="stellar_density",
                    stellar_density_log10_mean=0.0,
                    stellar_density_log10_std=0.25)
    prior = _stellar_prior()
    sim = TransitSimulator(cfg, prior=prior)
    rng = np.random.default_rng(0)
    P = np.full(300000, 5.0)
    ars = sim._sample_physical_a_rs(P, rng)
    # drop the measure that the simulator clips onto the [3, 50] boundary
    ars = ars[(ars > 3.0 + 1e-3) & (ars < 50.0 - 1e-3)]
    emp_mean_log10 = float(np.mean(np.log10(ars)))
    emp_std_log10 = float(np.std(np.log10(ars)))

    grid = np.linspace(3.0, 50.0, 400000)
    Pg = np.full(grid.size, 5.0)
    pdf = np.exp(prior._stellar_density_logpdf_a_rs(Pg, grid))
    pdf = pdf / _trapz(pdf, grid)
    log10_grid = np.log10(grid)
    mean_density = float(_trapz(log10_grid * pdf, grid))
    var_density = float(_trapz((log10_grid - mean_density) ** 2 * pdf, grid))
    assert abs(mean_density - emp_mean_log10) < 0.01, (mean_density, emp_mean_log10)
    assert abs(np.sqrt(var_density) - emp_std_log10) < 0.01


def test_stellar_density_only_changes_aRs(prior, rng):
    # Switching the a/Rs mode must change only the a/Rs term, leaving the other
    # six log-uniform / uniform marginals (and the support) untouched.
    stellar = _stellar_prior()
    s = prior.sample(64, rng)
    lp_box = prior.log_prob_physical(s)
    lp_stellar = stellar.log_prob_physical(s)
    a_term = stellar._stellar_density_logpdf_a_rs(s[:, 0], s[:, 3])
    box_a_term = -np.log(prior._u_high[3] - prior._u_low[3]) - np.log(s[:, 3])
    assert np.allclose(lp_stellar - lp_box, a_term - box_a_term, atol=1e-9)
    # support handling preserved
    bad = s.copy()
    bad[:, 4] = 5.0  # impact parameter outside [0, 1.1]
    assert np.all(~np.isfinite(stellar.log_prob_physical(bad)))
