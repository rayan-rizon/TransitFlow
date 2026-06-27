import numpy as np


def test_batch_keys_and_shapes(fast_simulator, fast_sim_cfg, rng):
    b = fast_simulator.simulate_batch(32, rng)
    assert b["global"].shape == (32, fast_sim_cfg.n_global)
    assert b["local"].shape == (32, fast_sim_cfg.n_local)
    assert b["theta_std"].shape == (32, 7)
    assert b["theta_char_std"].shape == (32, 5)
    assert b["theta_phys"].shape == (32, 7)
    assert b["d"].shape == (32,)
    assert b["sigma_feat"].shape == (32,)
    assert b["ephem_feat"].shape == (32, 2)
    assert np.all(np.isfinite(b["global"])) and np.all(np.isfinite(b["local"]))
    assert np.all(np.isfinite(b["ephem_feat"]))
    assert np.allclose(b["theta_char_std"], b["theta_std"][:, 2:])


def test_valid_mask_matches_label(fast_simulator, rng):
    b = fast_simulator.simulate_batch(64, rng)
    assert np.array_equal(b["valid"], b["d"] == 1)
    # standardized targets zeroed for non-planets
    assert np.allclose(b["theta_std"][b["d"] == 0], 0.0)


def test_determinism(fast_simulator):
    b1 = fast_simulator.simulate_batch(16, np.random.default_rng(5))
    b2 = fast_simulator.simulate_batch(16, np.random.default_rng(5))
    assert np.array_equal(b1["d"], b2["d"])
    assert np.allclose(b1["global"], b2["global"])


def test_sigma_feature_range(fast_simulator, rng):
    b = fast_simulator.simulate_batch(200, rng)
    # standardized noise feature should be roughly within [-1.2, 1.2]
    assert b["sigma_feat"].min() > -1.5 and b["sigma_feat"].max() < 1.5


def test_planet_local_views_deeper_on_average(prior):
    """Clean high-SNR planets fold to a clearly negative local-view minimum."""
    from transitflow.simulator import SimConfig, TransitSimulator
    cfg = SimConfig(n_global=512, n_local=201, baseline_days=27.0, n_raw=12000,
                    planet_fraction=1.0, hard_negative_fraction=0.0,
                    frac_real=0.0, frac_gp=0.0, frac_white=1.0,
                    sigma_white_log10_low=-4.0, sigma_white_log10_high=-3.6,
                    n_radial=120, regime="tess")
    sim = TransitSimulator(cfg, prior=prior)
    b = sim.simulate_batch(200, np.random.default_rng(11))
    argmin = b["local"].argmin(axis=1)
    # most clean transits dip near the centre bin (100)
    assert np.mean(np.abs(argmin - 100) <= 20) > 0.7


def test_real_noise_sigma_feature_uses_drawn_segment(prior):
    from transitflow.noise import NoiseLibrary, estimate_white_sigma
    from transitflow.simulator import SimConfig, TransitSimulator

    rng = np.random.default_rng(12)
    sigma = 0.003
    segments = 1.0 + rng.normal(0.0, sigma, size=(4, 512))
    cfg = SimConfig(n_global=64, n_local=41, baseline_days=2.0, n_raw=512,
                    planet_fraction=1.0, frac_real=1.0, frac_gp=0.0,
                    frac_white=0.0, sigma_white_log10_low=-4.0,
                    sigma_white_log10_high=-2.0, n_radial=60,
                    regime="tess", use_periodogram=False)
    sim = TransitSimulator(cfg, prior=prior, noise_library=NoiseLibrary(segments))
    b = sim.simulate_batch(64, np.random.default_rng(13))
    expected_sigma = estimate_white_sigma(segments)
    assert b["sigma"].min() >= expected_sigma.min() * 0.8
    assert b["sigma"].max() <= expected_sigma.max() * 1.2
    assert b["sigma_feat"].std() < 0.2


def test_dilution_attenuates_transit_depth(prior):
    from transitflow.simulator import SimConfig, TransitSimulator

    base = dict(n_global=256, n_local=101, baseline_days=8.0, n_raw=2500,
                planet_fraction=1.0, hard_negative_fraction=0.0,
                frac_real=0.0, frac_gp=0.0, frac_white=1.0,
                sigma_white_log10_low=-8.0, sigma_white_log10_high=-8.0,
                n_radial=80, regime="tess", use_periodogram=False)
    rng_seed = 21
    plain = TransitSimulator(SimConfig(**base), prior=prior).simulate_batch(
        32, np.random.default_rng(rng_seed), return_raw=True)
    diluted_cfg = SimConfig(**base, dilution_fraction=1.0,
                            dilution_low=0.5, dilution_high=0.5)
    diluted = TransitSimulator(diluted_cfg, prior=prior).simulate_batch(
        32, np.random.default_rng(rng_seed), return_raw=True)
    plain_depth = 1.0 - plain["raw_flux"].min(axis=1)
    diluted_depth = 1.0 - diluted["raw_flux"].min(axis=1)
    ratio = np.median(diluted_depth / np.maximum(plain_depth, 1e-12))
    assert 0.45 < ratio < 0.55
    assert np.allclose(diluted["dilution"], 0.5)


def test_gap_masks_keep_views_finite(prior):
    from transitflow.simulator import SimConfig, TransitSimulator

    cfg = SimConfig(n_global=128, n_local=65, baseline_days=10.0, n_raw=2000,
                    planet_fraction=1.0, frac_real=0.0, frac_gp=0.0,
                    frac_white=1.0, gap_fraction=0.08,
                    gap_duration_range=(0.1, 0.3), n_radial=60,
                    regime="tess", use_periodogram=True, n_period_bins=32,
                    pg_n_raw=512)
    b = TransitSimulator(cfg, prior=prior).simulate_batch(
        16, np.random.default_rng(22))
    assert b["cadence_mask"].shape == (16, cfg.n_raw)
    assert b["cadence_mask"].mean() < 1.0
    assert np.isfinite(b["global"]).all()
    assert np.isfinite(b["local"]).all()
    assert np.isfinite(b["periodogram"]).all()


def test_physical_a_rs_mode_correlates_with_period(prior):
    from transitflow.simulator import SimConfig, TransitSimulator

    cfg = SimConfig(n_global=64, n_local=41, baseline_days=27.0, n_raw=1200,
                    planet_fraction=1.0, frac_real=0.0, frac_gp=0.0,
                    frac_white=1.0, a_rs_prior_mode="stellar_density",
                    stellar_density_log10_std=0.0, n_radial=40,
                    regime="tess", use_periodogram=False)
    b = TransitSimulator(cfg, prior=prior).simulate_batch(
        256, np.random.default_rng(23))
    corr = np.corrcoef(np.log(b["theta_phys"][:, 0]),
                       np.log(b["theta_phys"][:, 3]))[0, 1]
    assert corr > 0.95
