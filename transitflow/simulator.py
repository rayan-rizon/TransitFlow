"""The SBI forward simulator: parameters + physics + noise -> dual-view data.

For each light curve the simulator draws a detection label ``d``, transit
parameters ``theta`` (used only when ``d=1``), a noise regime, and a candidate
folding ephemeris, then synthesizes a raw light curve and reduces it to the
(global, local) view pair the network consumes.

The output of :meth:`TransitSimulator.simulate_batch` is everything a training
step needs: views, standardized targets, labels, the noise level (optional
conditioning), and a validity mask selecting the ``d=1`` rows on which the flow
is trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .noise import (
    NoiseLibrary,
    eclipsing_binary_signal,
    sample_correlated_noise,
    single_event_signal,
    sinusoid_signal,
    white_noise,
)
from .priors import TransitPrior, kipping_to_quadratic
from .transit_model import transit_duration, transit_flux
from .views import make_views


@dataclass
class SimConfig:
    """Configuration of the forward simulator."""

    # view geometry
    n_global: int = 2001
    n_local: int = 201
    n_durations: float = 4.0
    # raw light-curve sampling (defaults: a single TESS sector at ~2-min cadence)
    baseline_days: float = 27.0
    n_raw: int = 18000
    # class balance
    planet_fraction: float = 0.5
    hard_negative_fraction: float = 0.5  # fraction of *non-planets* that are hard
    # noise-regime mixture (real is used only if a NoiseLibrary is available)
    frac_real: float = 0.70
    frac_gp: float = 0.20
    frac_white: float = 0.10
    # white-noise level prior (per-cadence relative flux std, CDPP-like)
    sigma_white_log10_low: float = -3.7   # ~200 ppm
    sigma_white_log10_high: float = -2.3  # ~5000 ppm
    # correlated stellar variability prior
    gp_amp_log10_low: float = -3.5
    gp_amp_log10_high: float = -2.0
    gp_tau_frac_low: float = 0.02         # correlation time as fraction of baseline
    gp_tau_frac_high: float = 0.30
    gp_kernel: str = "matern32"
    # hard-negative depths
    eb_depth_low: float = 0.01
    eb_depth_high: float = 0.10
    # limb darkening engine
    engine: str = "native"
    n_radial: int = 200
    # prior regime (period range): "tess" (0.5-13 d) suits the 27-d baseline
    regime: str = "tess"

    def normalized_regime_fracs(self, has_real: bool) -> tuple[float, float, float]:
        if has_real:
            fr, fg, fw = self.frac_real, self.frac_gp, self.frac_white
        else:  # redistribute real mass onto gp/white
            fr, fg, fw = 0.0, self.frac_gp, self.frac_white
        s = fr + fg + fw
        return fr / s, fg / s, fw / s


class TransitSimulator:
    """Vectorized forward simulator producing dual-view SBI training data."""

    def __init__(self, config: SimConfig | None = None,
                 prior: TransitPrior | None = None,
                 noise_library: NoiseLibrary | None = None) -> None:
        self.cfg = config or SimConfig()
        self.prior = prior or TransitPrior(TransitPrior.default_specs(self.cfg.regime))
        self.noise_library = noise_library or NoiseLibrary(None)
        self.times = np.linspace(0.0, self.cfg.baseline_days, self.cfg.n_raw)
        self.dt = self.times[1] - self.times[0]

    # ------------------------------------------------------------------ #
    def simulate_batch(self, B: int, rng: np.random.Generator | None = None,
                       return_raw: bool = False) -> dict:
        rng = np.random.default_rng() if rng is None else rng
        cfg = self.cfg
        t = self.times

        # ---- labels & sub-classes -----------------------------------
        d = (rng.random(B) < cfg.planet_fraction).astype(np.int64)
        is_planet = d == 1
        is_neg = ~is_planet
        hard_neg = is_neg & (rng.random(B) < cfg.hard_negative_fraction)

        # ---- transit parameters (physical) for every row ------------
        theta_phys = self.prior.sample(B, rng)            # (B, 7)
        P = theta_phys[:, 0]
        t0_phase = theta_phys[:, 1]
        RpRs = theta_phys[:, 2]
        aRs = theta_phys[:, 3]
        b = theta_phys[:, 4]
        u1, u2 = kipping_to_quadratic(theta_phys[:, 5], theta_phys[:, 6])
        t0_abs = t0_phase * P

        # candidate folding ephemeris + duration, per row
        fold_P = P.copy()
        fold_t0 = t0_abs.copy()
        duration = transit_duration(P, RpRs, aRs, b)

        # ---- base (noise-free) raw flux -----------------------------
        flux = np.ones((B, cfg.n_raw), dtype=np.float64)

        # planets: inject the physical transit
        if is_planet.any():
            idx = np.where(is_planet)[0]
            flux[idx] = transit_flux(
                t, P[idx], t0_abs[idx], RpRs[idx], aRs[idx], b[idx],
                u1[idx], u2[idx], n_radial=cfg.n_radial, engine=cfg.engine,
            )

        # non-planets: spurious candidate ephemeris (+ optional hard negative)
        if is_neg.any():
            idx = np.where(is_neg)[0]
            # default spurious candidate: random period/epoch from the prior box
            spurious = self.prior.sample(len(idx), rng)
            fold_P[idx] = spurious[:, 0]
            fold_t0[idx] = spurious[:, 1] * spurious[:, 0]
            # nominal duration so the local window is sensible
            duration[idx] = transit_duration(
                fold_P[idx], np.full(len(idx), 0.05),
                np.full(len(idx), 10.0), np.full(len(idx), 0.3),
            )

        # hard negatives: EB / single-event / sinusoid morphologies
        if hard_neg.any():
            for j in np.where(hard_neg)[0]:
                kind = rng.integers(0, 3)
                if kind == 0:  # eclipsing binary -> candidate = EB ephemeris
                    depth = rng.uniform(cfg.eb_depth_low, cfg.eb_depth_high)
                    dur = max(duration[j], 0.05)
                    flux[j] *= eclipsing_binary_signal(t, fold_P[j], fold_t0[j],
                                                       depth, dur)
                elif kind == 1:  # single isolated event, random candidate fold
                    depth = rng.uniform(cfg.eb_depth_low, cfg.eb_depth_high)
                    dur = rng.uniform(0.05, 0.5)
                    t_ev = rng.uniform(0.1, 0.9) * cfg.baseline_days
                    flux[j] *= single_event_signal(t, t_ev, depth, dur)
                else:           # coherent sinusoid
                    amp = rng.uniform(cfg.eb_depth_low, cfg.eb_depth_high)
                    per = rng.uniform(0.5, 5.0)
                    flux[j] *= sinusoid_signal(t, amp, per, rng.uniform(0, 2 * np.pi))

        # ---- noise ---------------------------------------------------
        sigma_white = 10.0 ** rng.uniform(
            cfg.sigma_white_log10_low, cfg.sigma_white_log10_high, B)
        has_real = self.noise_library.available()
        fr, fg, fw = cfg.normalized_regime_fracs(has_real)
        regime = rng.choice(3, size=B, p=[fr, fg, fw])  # 0 real, 1 gp, 2 white

        # white noise for gp + white regimes
        wn_mask = regime != 0
        if wn_mask.any():
            wn = white_noise(sigma_white[wn_mask], cfg.n_raw, rng)
            flux[wn_mask] += wn
        # correlated noise for gp regime
        gp_mask = regime == 1
        if gp_mask.any():
            amp = 10.0 ** rng.uniform(cfg.gp_amp_log10_low,
                                      cfg.gp_amp_log10_high, gp_mask.sum())
            tau_frac = rng.uniform(cfg.gp_tau_frac_low, cfg.gp_tau_frac_high,
                                   gp_mask.sum())
            tau_steps = tau_frac * cfg.n_raw
            cn = sample_correlated_noise(amp, tau_steps, cfg.n_raw,
                                         kind=cfg.gp_kernel, rng=rng)
            flux[gp_mask] += cn
        # real-noise injection (multiplicative real OOT segment)
        real_mask = regime == 0
        if real_mask.any():
            seg = self.noise_library.draw(int(real_mask.sum()), cfg.n_raw, rng)
            flux[real_mask] *= seg

        # ---- views ---------------------------------------------------
        gv = np.empty((B, cfg.n_global), dtype=np.float32)
        lv = np.empty((B, cfg.n_local), dtype=np.float32)
        for i in range(B):
            gv[i], lv[i] = make_views(
                t, flux[i], fold_P[i], fold_t0[i], duration[i],
                n_global=cfg.n_global, n_local=cfg.n_local,
                n_durations=cfg.n_durations, normalize=True,
            )

        # ---- targets -------------------------------------------------
        theta_std = self.prior.physical_to_std(theta_phys).astype(np.float32)
        theta_std[is_neg] = 0.0  # undefined for non-planets; masked in the loss

        # noise-level conditioning feature: standardized log10 sigma
        sig_feat = (np.log10(sigma_white) - 0.5 *
                    (cfg.sigma_white_log10_low + cfg.sigma_white_log10_high))
        sig_feat = sig_feat / (0.5 * (cfg.sigma_white_log10_high -
                                      cfg.sigma_white_log10_low))

        out = {
            "global": gv,
            "local": lv,
            "theta_std": theta_std,
            "theta_phys": theta_phys.astype(np.float32),
            "d": d,
            "valid": is_planet,
            "sigma": sigma_white.astype(np.float32),
            "sigma_feat": sig_feat.astype(np.float32),
            "fold_P": fold_P.astype(np.float32),
            "fold_t0": fold_t0.astype(np.float32),
            "duration": duration.astype(np.float32),
            "regime": regime.astype(np.int8),  # 0 real, 1 gp, 2 white
        }
        if return_raw:
            # raw light curve + cadence grid, for the exact importance-sampling
            # likelihood (the views are a lossy, period-blurred reduction)
            out["raw_flux"] = flux.astype(np.float32)
            out["times"] = self.times.astype(np.float32)
        return out
