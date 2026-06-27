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
    estimate_white_sigma,
    sample_correlated_noise,
    single_event_signal,
    sinusoid_signal,
    white_noise,
)
from .priors import TransitPrior, kipping_to_quadratic
from .transit_model import exposure_integrated_transit_flux, transit_duration
from .views import flatten_transit_preserving, make_periodogram_view, make_views


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
    # shape-realism knobs for publication-scale training. Defaults preserve the
    # historical instantaneous, undiluted, complete-cadence simulator.
    exposure_minutes: float = 0.0
    n_exposure_subsamples: int = 1
    dilution_fraction: float = 0.0
    dilution_low: float = 0.5
    dilution_high: float = 1.0
    gap_fraction: float = 0.0
    gap_duration_range: tuple[float, float] = (0.10, 0.80)
    a_rs_prior_mode: str = "log_uniform"  # "log_uniform" | "stellar_density"
    stellar_density_log10_mean: float = 0.0
    stellar_density_log10_std: float = 0.25
    flatten_views: bool = False
    # box-periodogram input channel (supplies sharp period information that the
    # binned global view lacks -> calibrated period posterior)
    use_periodogram: bool = True
    n_period_bins: int = 256
    pg_n_phase: int = 64
    # subsample raw LC to this many points for the periodogram only (4096 is
    # enough resolution for 256 trial periods × 64 phase bins, and reduces the
    # (P × n_raw) matrix from 4.6 M to 1.0 M floats -> ~4× faster generation)
    pg_n_raw: int = 4096

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
        # log-spaced trial-period grid spanning the prior period range
        p_lo, p_hi = self.prior.specs[0].low, self.prior.specs[0].high
        self.period_grid = np.logspace(np.log10(p_lo), np.log10(p_hi),
                                       self.cfg.n_period_bins)

    # ------------------------------------------------------------------ #
    def _sample_physical_a_rs(self, P: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
        """Draw ``a/Rs`` from a stellar-density prior.

        For circular orbits, ``a/Rs = (G rho_star P^2 / 3pi)^(1/3)``.  The
        density is sampled in solar units and clipped to the canonical support
        used by :class:`TransitPrior`, so downstream standardization remains
        valid.
        """
        if self.cfg.a_rs_prior_mode == "log_uniform":
            return np.array([], dtype=np.float64)
        if self.cfg.a_rs_prior_mode != "stellar_density":
            raise ValueError(f"unknown a_rs_prior_mode {self.cfg.a_rs_prior_mode!r}")
        rho_sun_kg_m3 = 1408.0
        g_si = 6.67430e-11
        day_s = 86400.0
        rho = rho_sun_kg_m3 * 10.0 ** rng.normal(
            self.cfg.stellar_density_log10_mean,
            self.cfg.stellar_density_log10_std,
            size=len(P),
        )
        ars = (g_si * rho * (np.asarray(P, dtype=np.float64) * day_s) ** 2
               / (3.0 * np.pi)) ** (1.0 / 3.0)
        lo, hi = self.prior.specs[3].low, self.prior.specs[3].high
        return np.clip(ars, lo, hi)

    def _sample_gap_masks(self, B: int, rng: np.random.Generator) -> np.ndarray:
        """Cadence-availability masks for downlinks/momentum-dump-like gaps."""
        cfg = self.cfg
        mask = np.ones((B, cfg.n_raw), dtype=bool)
        if cfg.gap_fraction <= 0:
            return mask
        low, high = cfg.gap_duration_range
        low = max(float(low), self.dt)
        high = max(float(high), low)
        target = max(0.0, min(float(cfg.gap_fraction), 0.95)) * cfg.baseline_days
        for i in range(B):
            remaining = target
            while remaining > 0:
                dur = min(rng.uniform(low, high), remaining)
                start = rng.uniform(0.0, max(cfg.baseline_days - dur, self.dt))
                gap = (self.times >= start) & (self.times <= start + dur)
                mask[i, gap] = False
                remaining -= dur
        return mask

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
        if cfg.a_rs_prior_mode != "log_uniform":
            theta_phys[:, 3] = self._sample_physical_a_rs(P, rng)
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
            flux[idx] = exposure_integrated_transit_flux(
                t, P[idx], t0_abs[idx], RpRs[idx], aRs[idx], b[idx],
                u1[idx], u2[idx], n_radial=cfg.n_radial, engine=cfg.engine,
                exposure_days=cfg.exposure_minutes / (60.0 * 24.0),
                n_subsamples=cfg.n_exposure_subsamples,
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

        # Third-light/crowding dilution attenuates transit-like depth without
        # changing the normalized out-of-transit baseline.
        dilution = np.ones(B, dtype=np.float64)
        if cfg.dilution_fraction > 0:
            use_dilution = rng.random(B) < cfg.dilution_fraction
            if use_dilution.any():
                lo = min(cfg.dilution_low, cfg.dilution_high)
                hi = max(cfg.dilution_low, cfg.dilution_high)
                dilution[use_dilution] = rng.uniform(lo, hi, use_dilution.sum())
                flux[use_dilution] = 1.0 + (flux[use_dilution] - 1.0) * \
                    dilution[use_dilution, None]

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
            sigma_white[real_mask] = np.clip(
                estimate_white_sigma(seg),
                10.0 ** cfg.sigma_white_log10_low,
                10.0 ** cfg.sigma_white_log10_high,
            )

        # ---- views ---------------------------------------------------
        gv = np.empty((B, cfg.n_global), dtype=np.float32)
        lv = np.empty((B, cfg.n_local), dtype=np.float32)
        use_pg = cfg.use_periodogram
        pg = np.empty((B, cfg.n_period_bins), dtype=np.float32) if use_pg else None
        gap_mask = self._sample_gap_masks(B, rng)
        raw_for_views = flux.copy()
        for i in range(B):
            valid_cad = gap_mask[i]
            ti = t[valid_cad]
            fi = flux[i][valid_cad]
            if cfg.flatten_views:
                fi = flatten_transit_preserving(
                    ti, fi, fold_P[i], fold_t0[i], duration[i])
                raw_for_views[i, valid_cad] = fi
            gv[i], lv[i] = make_views(
                ti, fi, fold_P[i], fold_t0[i], duration[i],
                n_global=cfg.n_global, n_local=cfg.n_local,
                n_durations=cfg.n_durations, normalize=True,
            )
            if use_pg:
                # subsample to pg_n_raw points (uniformly spaced) so the (P×n)
                # BLS matrix stays small; 4096 pts over 27 d gives 151 pts/day,
                # ample for periods ≥ 0.5 d with 64 phase bins
                n_pg = cfg.pg_n_raw
                if n_pg < len(ti):
                    step = max(1, len(ti) // n_pg)
                    t_pg = ti[::step][:n_pg]
                    f_pg = fi[::step][:n_pg]
                else:
                    t_pg, f_pg = ti, fi
                pg[i] = make_periodogram_view(t_pg, f_pg, self.period_grid,
                                              n_phase=cfg.pg_n_phase, normalize=True)

        # ---- targets -------------------------------------------------
        theta_std = self.prior.physical_to_std(theta_phys).astype(np.float32)
        theta_std[is_neg] = 0.0  # undefined for non-planets; masked in the loss
        theta_char_std = theta_std[:, 2:].astype(np.float32)

        # noise-level conditioning feature: standardized log10 sigma
        sig_feat = (np.log10(sigma_white) - 0.5 *
                    (cfg.sigma_white_log10_low + cfg.sigma_white_log10_high))
        sig_scale = 0.5 * (cfg.sigma_white_log10_high -
                           cfg.sigma_white_log10_low)
        sig_feat = sig_feat / max(sig_scale, 1e-6)

        # Explicit candidate ephemeris conditioning.  The local view is folded on
        # this candidate, so P/t0 are not purely image-derived quantities; making
        # the candidate available to the posterior is the identifiable setup.
        ephem_phys = theta_phys.copy()
        ephem_phys[:, 0] = fold_P
        ephem_phys[:, 1] = (fold_t0 / np.maximum(fold_P, 1e-12)) % 1.0
        ephem_feat = self.prior.physical_to_std(ephem_phys)[:, :2].astype(np.float32)

        out = {
            "global": gv,
            "local": lv,
            "theta_std": theta_std,
            "theta_char_std": theta_char_std,
            "theta_phys": theta_phys.astype(np.float32),
            "d": d,
            "valid": is_planet,
            "sigma": sigma_white.astype(np.float32),
            "sigma_feat": sig_feat.astype(np.float32),
            "ephem_feat": ephem_feat,
            "fold_P": fold_P.astype(np.float32),
            "fold_t0": fold_t0.astype(np.float32),
            "duration": duration.astype(np.float32),
            "regime": regime.astype(np.int8),  # 0 real, 1 gp, 2 white
            "dilution": dilution.astype(np.float32),
            "cadence_mask": gap_mask,
        }
        if use_pg:
            out["periodogram"] = pg
        if return_raw:
            # raw light curve + cadence grid, for the exact importance-sampling
            # likelihood (the views are a lossy, period-blurred reduction)
            out["raw_flux"] = raw_for_views.astype(np.float32)
            out["times"] = self.times.astype(np.float32)
        return out
