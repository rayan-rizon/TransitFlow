import numpy as np

from transitflow.noise import (
    NoiseLibrary,
    eclipsing_binary_signal,
    sample_correlated_noise,
    single_event_signal,
    sinusoid_signal,
    white_noise,
)


def test_white_noise_std():
    rng = np.random.default_rng(0)
    sigma = np.array([0.01, 0.05])
    wn = white_noise(sigma, 20000, rng)
    assert wn.shape == (2, 20000)
    assert np.isclose(wn[0].std(), 0.01, rtol=0.1)
    assert np.isclose(wn[1].std(), 0.05, rtol=0.1)


def test_correlated_noise_amplitude_and_correlation():
    rng = np.random.default_rng(1)
    amp = np.array([0.02, 0.02, 0.02])
    tau = np.array([50.0, 50.0, 50.0])
    cn = sample_correlated_noise(amp, tau, 8000, kind="matern32", rng=rng)
    assert cn.shape == (3, 8000)
    # amplitude roughly matches requested std
    assert 0.5 * 0.02 < cn.std() < 2.0 * 0.02
    # lag-1 autocorrelation should exceed long-lag autocorrelation
    x = cn[0] - cn[0].mean()
    ac1 = np.mean(x[:-1] * x[1:]) / np.var(x)
    ac_far = np.mean(x[:-500] * x[500:]) / np.var(x)
    assert ac1 > 0.5
    assert ac1 > ac_far


def test_hard_negative_signals():
    t = np.linspace(0, 10, 2000)
    eb = eclipsing_binary_signal(t, 3.0, 1.0, 0.05, 0.2)
    assert eb.min() >= 1 - 0.05 - 1e-9 and eb.max() <= 1 + 1e-9
    se = single_event_signal(t, 5.0, 0.03, 0.3)
    assert se.min() < 1.0 and np.isclose(se[0], 1.0, atol=1e-3)
    sn = sinusoid_signal(t, 0.02, 1.5, 0.0)
    assert np.isclose(sn.mean(), 1.0, atol=1e-2)


def test_noise_library_roundtrip():
    rng = np.random.default_rng(2)
    fake = 1.0 + 0.01 * rng.standard_normal((5, 5000))
    lib = NoiseLibrary(fake)
    assert lib.available()
    seg = lib.draw(8, 2000, rng)
    assert seg.shape == (8, 2000)
    empty = NoiseLibrary(None)
    assert not empty.available()
