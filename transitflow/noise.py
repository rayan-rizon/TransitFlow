"""Noise models for the SBI forward simulator.

Three regimes are mixed per batch (Sec. 2.2 of the plan):

1. **Real-noise injection** -- transits injected into out-of-transit segments of
   real Kepler/TESS light curves.  Implemented via :class:`NoiseLibrary`, which
   serves cached real segments if any have been downloaded with ``lightkurve``;
   otherwise this regime is skipped and its probability mass is redistributed.
2. **GP-correlated synthetic noise** -- a stationary Gaussian process (stellar
   variability) plus white noise.  Sampled by fast vectorized FFT spectral
   synthesis (circulant embedding), validated against ``celerite2`` covariance
   in the tests.
3. **Pure white Gaussian** -- the idealized regime for calibration unit-tests.

Hard negatives (eclipsing-binary V-dips, single-event systematics, coherent
sinusoids) are injected so the detector learns transit-specific morphology
rather than "any dip".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Stationary Gaussian-process noise (vectorized FFT spectral synthesis)
# --------------------------------------------------------------------------- #
def _autocovariance(lags: np.ndarray, amp: np.ndarray, tau: np.ndarray,
                    kind: str) -> np.ndarray:
    """Autocovariance ``k(τ)`` for the supported stationary kernels.

    ``lags`` has shape ``(M,)``; ``amp, tau`` have shape ``(B, 1)``; returns
    ``(B, M)``.
    """
    lags = np.abs(lags)[None, :]
    if kind == "matern32":
        x = np.sqrt(3.0) * lags / tau
        return amp ** 2 * (1.0 + x) * np.exp(-x)
    if kind == "exp":
        return amp ** 2 * np.exp(-lags / tau)
    if kind == "gauss":
        return amp ** 2 * np.exp(-0.5 * (lags / tau) ** 2)
    if kind == "sho":  # critically-damped simple-harmonic oscillator
        x = lags / tau
        return amp ** 2 * np.exp(-x) * (1.0 + x)
    raise ValueError(f"unknown kernel {kind!r}")


def sample_correlated_noise(
    amp: np.ndarray,
    tau_steps: np.ndarray,
    n: int,
    kind: str = "matern32",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw stationary Gaussian noise via circulant embedding.

    Parameters
    ----------
    amp:
        ``(B,)`` per-sample noise amplitudes (std of the process).
    tau_steps:
        ``(B,)`` correlation timescales expressed *in cadence steps*.
    n:
        Length of each series.
    kind:
        Kernel name (see :func:`_autocovariance`).
    rng:
        NumPy random generator.

    Returns
    -------
    ``(B, n)`` array of correlated Gaussian samples (mean 0).
    """
    rng = np.random.default_rng() if rng is None else rng
    amp = np.atleast_1d(np.asarray(amp, dtype=np.float64))
    tau_steps = np.atleast_1d(np.asarray(tau_steps, dtype=np.float64))
    B = max(len(amp), len(tau_steps))
    amp = np.broadcast_to(amp, (B,)).reshape(B, 1)
    tau_steps = np.broadcast_to(tau_steps, (B,)).reshape(B, 1)

    # circulant embedding size: next power of two >= 2n, padded for positivity
    M = 1
    while M < 2 * n:
        M <<= 1
    lags = np.concatenate([np.arange(M // 2 + 1), -np.arange(1, M // 2)[::-1]])
    cov_row = _autocovariance(lags.astype(np.float64), amp, tau_steps, kind)  # (B,M)
    eig = np.fft.fft(cov_row, axis=1).real
    eig = np.clip(eig, 0.0, None)  # guard tiny negative eigenvalues
    z = rng.normal(size=(B, M)) + 1j * rng.normal(size=(B, M))
    series = np.fft.fft(np.sqrt(eig / M) * z, axis=1).real
    return series[:, :n]


def white_noise(sigma: np.ndarray, n: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """``(B, n)`` white Gaussian noise with per-sample std ``sigma`` (``(B,)``)."""
    rng = np.random.default_rng() if rng is None else rng
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64)).reshape(-1, 1)
    return rng.normal(size=(sigma.shape[0], n)) * sigma


# --------------------------------------------------------------------------- #
# Hard negatives (injected into the raw curve)
# --------------------------------------------------------------------------- #
def eclipsing_binary_signal(times: np.ndarray, P: float, t0: float,
                            depth: float, duration: float) -> np.ndarray:
    """V-shaped periodic eclipse: a sharp triangular dip (cf. U-shaped transit).

    Returns a multiplicative flux series (1.0 out of eclipse).
    """
    phase = ((times - t0) / P + 0.5) % 1.0 - 0.5
    dt = np.abs(phase) * P
    half = 0.5 * duration
    tri = np.clip(1.0 - dt / half, 0.0, 1.0)  # 1 at centre -> 0 at edge (V)
    return 1.0 - depth * tri


def single_event_signal(times: np.ndarray, t_event: float, depth: float,
                        duration: float) -> np.ndarray:
    """One isolated (non-periodic) U-shaped dip -- a systematic / single transit."""
    dt = (times - t_event) / (0.5 * duration)
    bump = np.exp(-0.5 * dt ** 2)
    return 1.0 - depth * bump


def sinusoid_signal(times: np.ndarray, amp: float, period: float,
                    phase: float) -> np.ndarray:
    """Coherent sinusoidal stellar variability (a pulsation-like hard negative)."""
    return 1.0 + amp * np.sin(2.0 * np.pi * times / period + phase)


# --------------------------------------------------------------------------- #
# Real out-of-transit segment library (optional, populated via lightkurve)
# --------------------------------------------------------------------------- #
@dataclass
class NoiseLibrary:
    """Holds cached, unit-normalized real out-of-transit flux segments.

    Each row is a length-``n`` segment with median ~1.  Populated offline by
    ``scripts/build_noise_library.py`` (which uses ``lightkurve``); when empty,
    :meth:`available` is ``False`` and the simulator falls back to synthetic GP
    noise so the pipeline never requires network access to run.
    """

    segments: np.ndarray | None = None  # (K, n) or None

    def available(self) -> bool:
        return self.segments is not None and len(self.segments) > 0

    @classmethod
    def load(cls, path: str | None) -> "NoiseLibrary":
        if path is None:
            return cls(None)
        try:
            arr = np.load(path)
            seg = arr["segments"] if hasattr(arr, "files") else arr
            return cls(np.asarray(seg, dtype=np.float64))
        except Exception:
            return cls(None)

    def draw(self, B: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``B`` real OOT segments of length ``n`` (random start offsets)."""
        if not self.available():
            raise RuntimeError("NoiseLibrary is empty")
        K, L = self.segments.shape
        idx = rng.integers(0, K, size=B)
        out = np.empty((B, n), dtype=np.float64)
        for i, k in enumerate(idx):
            if L >= n:
                start = rng.integers(0, L - n + 1)
                out[i] = self.segments[k, start:start + n]
            else:  # tile if a cached segment is shorter than requested
                reps = int(np.ceil(n / L))
                out[i] = np.tile(self.segments[k], reps)[:n]
        return out
