import numpy as np
import torch

from scripts._config import build_configs

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
    # p(a/Rs | P) must integrate to ~1 on its support for every period.
    prior = _stellar_prior()
    grid = np.linspace(3.0, 50.0, 400000)
    for period in (0.5, 5.0, 13.0):
        P = np.full(grid.size, period)
        pdf = np.exp(prior._stellar_density_logpdf_a_rs(P, grid))
        norm = float(_trapz(pdf, grid))
        assert abs(norm - 1.0) < 0.02, (period, norm)


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
    assert np.all((ars > 3.0) & (ars < 50.0))
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


def test_characterization_prior_normal_round_trip_and_jacobian():
    prior = _stellar_prior()
    rng = np.random.default_rng(81)
    period = rng.uniform(0.5, 13.0, size=128)
    physical = prior.sample(128, rng)
    physical[:, 0] = period
    physical[:, 3] = prior.sample_stellar_density_a_rs(period, rng)
    z = prior.physical_to_std(physical)[:, 2:]

    normal = prior.characterization_std_to_prior_normal(z, period)
    recovered = prior.characterization_prior_normal_to_std(normal, period)

    assert np.allclose(recovered, z, atol=1e-9)
    eps = 1e-6
    numerical = []
    for dim in range(5):
        delta = np.zeros_like(z)
        delta[:, dim] = eps
        hi = prior.characterization_std_to_prior_normal(z + delta, period)
        lo = prior.characterization_std_to_prior_normal(z - delta, period)
        numerical.append(np.log(np.abs((hi[:, dim] - lo[:, dim]) / (2 * eps))))
    numerical = np.sum(np.stack(numerical, axis=-1), axis=-1)
    analytic = prior.characterization_prior_normal_log_abs_det(z, period)
    assert np.allclose(analytic, numerical, atol=2e-5)


def test_config_rejects_degenerate_stellar_density():
    try:
        build_configs(
            "configs/default.yaml",
            overrides={
                "model": {"posterior_transform": "prior_normal", "param_dim": 5},
                "simulator": {
                    "a_rs_prior_mode": "stellar_density",
                    "stellar_density_log10_std": 0.0,
                },
            },
        )
    except ValueError as exc:
        assert "positive log10 density width" in str(exc)
    else:
        raise AssertionError("degenerate stellar-density configuration did not fail")


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


def test_stellar_density_log_prob_std_matches_physical_change_of_variables():
    prior = _stellar_prior()
    rng = np.random.default_rng(919)
    physical = prior.sample(128, rng)
    physical[:, 3] = prior.sample_stellar_density_a_rs(physical[:, 0], rng)
    z = prior.physical_to_std(physical)

    expected = prior.log_prob_physical(physical) + np.sum(
        np.log(prior._u_std) + np.where(
            prior._log, np.log(physical), 0.0),
        axis=-1,
    )
    actual = prior.log_prob_std(z)

    assert np.allclose(actual, expected, atol=1e-10)
    assert np.std(actual) > 0.1
