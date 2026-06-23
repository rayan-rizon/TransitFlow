import numpy as np
import pytest

from transitflow.transit_model import (
    has_batman,
    transit_duration,
    transit_flux,
)


def test_out_of_transit_is_unity():
    t = np.linspace(0, 1, 500)
    # transit at t0=5 (far outside window) -> flux ~ 1
    f = transit_flux(t, 10.0, 5.0, 0.1, 10.0, 0.0, 0.3, 0.2, engine="native")[0]
    assert np.allclose(f, 1.0, atol=1e-6)


def test_depth_scales_with_radius_ratio():
    t = np.linspace(-0.05, 0.05, 2000)
    d1 = 1 - transit_flux(t, 5, 0, 0.05, 10, 0.0, 0.3, 0.2, engine="native")[0].min()
    d2 = 1 - transit_flux(t, 5, 0, 0.10, 10, 0.0, 0.3, 0.2, engine="native")[0].min()
    # depth ~ (Rp/Rs)^2, so doubling Rp/Rs ~ 4x deeper
    assert 3.0 < d2 / d1 < 5.0


def test_secondary_eclipse_flat():
    # half a period after transit the planet is behind the star -> no dip
    t = np.linspace(2.5 - 0.05, 2.5 + 0.05, 500)
    f = transit_flux(t, 5.0, 0.0, 0.1, 10.0, 0.0, 0.3, 0.2, engine="native")[0]
    assert f.min() > 1 - 1e-6


def test_vectorized_matches_loop():
    rng = np.random.default_rng(0)
    B = 8
    P = rng.uniform(2, 10, B)
    t0 = rng.uniform(0, 1, B)
    RpRs = rng.uniform(0.03, 0.12, B)
    aRs = rng.uniform(5, 20, B)
    b = rng.uniform(0, 0.7, B)
    u1 = np.full(B, 0.3); u2 = np.full(B, 0.2)
    t = np.linspace(0, 3, 1500)
    batch = transit_flux(t, P, t0, RpRs, aRs, b, u1, u2, engine="native")
    for i in range(B):
        single = transit_flux(t, P[i], t0[i], RpRs[i], aRs[i], b[i], u1[i], u2[i],
                              engine="native")[0]
        assert np.allclose(batch[i], single, atol=1e-9)


def test_duration_physical():
    dur = transit_duration(np.array([5.0]), np.array([0.1]),
                           np.array([10.0]), np.array([0.0]))[0]
    assert 0.05 < dur < 0.5  # hours-to-fraction-of-day scale
    # grazing non-transit -> ~0 duration
    dur0 = transit_duration(np.array([5.0]), np.array([0.1]),
                            np.array([10.0]), np.array([2.0]))[0]
    assert dur0 == 0.0


@pytest.mark.skipif(not has_batman(), reason="batman not installed")
def test_native_matches_batman():
    rng = np.random.default_rng(3)
    B = 20
    P = 10 ** rng.uniform(np.log10(0.5), np.log10(12), B)
    RpRs = 10 ** rng.uniform(np.log10(0.02), np.log10(0.15), B)
    aRs = 10 ** rng.uniform(np.log10(4), np.log10(30), B)
    b = rng.uniform(0, 0.9, B)
    u1 = rng.uniform(0.1, 0.6, B); u2 = rng.uniform(0.0, 0.4, B)
    t0 = rng.uniform(0.4, 0.6, B) * P
    worst = 0.0
    for i in range(B):
        t = np.linspace(t0[i] - 0.25 * P[i], t0[i] + 0.25 * P[i], 3000)
        fn = transit_flux(t, P[i], t0[i], RpRs[i], aRs[i], b[i], u1[i], u2[i],
                          n_radial=400, engine="native")[0]
        fb = transit_flux(t, P[i], t0[i], RpRs[i], aRs[i], b[i], u1[i], u2[i],
                          engine="batman")[0]
        worst = max(worst, np.max(np.abs(fn - fb)))
    assert worst < 1e-3
