import numpy as np

from transitflow.transit_model import transit_duration, transit_flux
from transitflow.views import global_view, local_view, make_views, normalize_view


def test_global_view_shape_and_finite():
    t = np.linspace(0, 27, 18000)
    f = 1.0 + 0.001 * np.random.default_rng(0).standard_normal(t.size)
    g = global_view(t, f, n_bins=2001)
    assert g.shape == (2001,)
    assert np.all(np.isfinite(g))


def test_local_view_centers_transit():
    P, t0, RpRs, aRs, b = 3.0, 1.0, 0.1, 10.0, 0.2
    t = np.linspace(0, 27, 18000)
    f = transit_flux(t, P, t0, RpRs, aRs, b, 0.3, 0.2, engine="native")[0]
    dur = transit_duration(np.array([P]), np.array([RpRs]),
                           np.array([aRs]), np.array([b]))[0]
    lv = local_view(t, f, P, t0, dur, n_bins=201)
    assert lv.shape == (201,)
    # the deepest bin should be near the center (phase 0)
    assert abs(int(np.argmin(lv)) - 100) <= 12


def test_normalize_view():
    v = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(500)
    nv = normalize_view(v)
    assert abs(np.median(nv)) < 1e-6


def test_make_views_dtypes():
    t = np.linspace(0, 27, 9000)
    f = transit_flux(t, 4.0, 1.0, 0.08, 12.0, 0.3, 0.3, 0.2, engine="native")[0]
    g, l = make_views(t, f, 4.0, 1.0, 0.15, n_global=512, n_local=101)
    assert g.shape == (512,) and l.shape == (101,)
    assert g.dtype == np.float32 and l.dtype == np.float32
