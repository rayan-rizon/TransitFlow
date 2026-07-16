import numpy as np
import pytest

from transitflow.baselines.bls import bls_top_candidates
from transitflow.candidates import (
    calibrate_bls_candidates,
    empirical_upper_tail_fap,
    fill_masked_flux,
    moving_block_bootstrap,
    periodic_candidate_mask,
)
from transitflow.transit_model import transit_flux


def _light_curve(seed=91):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 18.0, 2400)
    flux = transit_flux(times, 3.0, 0.9, 0.12, 12.0, 0.2, 0.3, 0.2,
                        engine="native")[0]
    return times, flux + rng.normal(0.0, 7e-4, len(times))


def test_bls_top_candidates_are_distinct_and_include_signal():
    times, flux = _light_curve()
    candidates = bls_top_candidates(
        times, flux, period_min=0.5, period_max=6.0, n_periods=500,
        top_k=3, min_log_period_separation=0.04,
    )
    assert 1 <= len(candidates) <= 3
    assert [row["rank"] for row in candidates] == list(range(1, len(candidates) + 1))
    assert all(row["score"] >= 0 for row in candidates)
    assert any(abs(row["best_period"] / 3.0 - 1.0) < 0.03 for row in candidates)
    logs = np.log([row["best_period"] for row in candidates])
    assert all(abs(a - b) >= 0.04 for i, a in enumerate(logs) for b in logs[i + 1:])


def test_empirical_fap_is_finite_sample_conservative():
    assert empirical_upper_tail_fap(10.0, np.array([1.0, 2.0, 3.0])) == 0.25
    assert empirical_upper_tail_fap(2.0, np.array([1.0, 2.0, 3.0])) == 0.75
    with pytest.raises(ValueError, match="finite null"):
        empirical_upper_tail_fap(2.0, np.array([np.nan]))


def test_mask_fill_and_moving_block_bootstrap_are_finite():
    times, flux = _light_curve()
    mask = periodic_candidate_mask(times, 3.0, 0.9, 0.12, width=1.5)
    filled = fill_masked_flux(times, flux, mask)
    sample = moving_block_bootstrap(filled - np.median(filled), 32,
                                     np.random.default_rng(7))
    assert 0 < mask.mean() < 0.2
    assert np.isfinite(filled).all() and np.isfinite(sample).all()
    assert len(sample) == len(flux)


def test_calibration_records_target_local_protocol(monkeypatch):
    import transitflow.candidates as candidate_module

    monkeypatch.setattr(candidate_module, "bls_top_candidates", lambda *a, **k: [{
        "rank": 1, "score": 5.0, "peak_power": 3.0,
        "best_period": 3.0, "best_t0": 0.9, "best_duration": 0.12,
    }])
    scores = iter([1.0, 7.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(candidate_module, "bls_detect", lambda *a, **k: {
        "score": next(scores)})
    times, flux = _light_curve()
    report = calibrate_bls_candidates(
        times, flux, n_periods=32, n_null=8, block_size=32,
        rng=np.random.default_rng(3),
    )
    assert report["target_search_fap"] == pytest.approx(2.0 / 9.0)
    assert report["candidates"][0]["target_search_fap"] == report["target_search_fap"]
    assert report["protocol"]["uses_source_identity"] is False
