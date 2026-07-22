"""Tests for MAST product provenance capture in the noise-library builder.

The reproducibility archive must name the exact TESS data products each noise
segment was built from.  These tests exercise the provenance extraction and its
positional alignment to accepted segments without any network access.
"""
import numpy as np

from scripts.build_noise_library import _lc_provenance, segment_flux_products


class _FakeLC:
    """Minimal stand-in for a lightkurve object carrying FITS ``meta``."""

    def __init__(self, meta):
        self.meta = meta


def test_lc_provenance_reads_uppercase_fits_cards():
    lc = _FakeLC({
        "MISSION": "TESS", "SECTOR": 14, "AUTHOR": "SPOC",
        "EXPTIME": 120.0, "TICID": 307210830,
        "FILENAME": "tess2019_s0014_lc.fits", "OBJECT": "TIC 307210830",
    })
    prov = _lc_provenance(lc)
    assert prov["mission"] == "TESS"
    assert prov["sector"] == 14 and isinstance(prov["sector"], int)
    assert prov["author"] == "SPOC"
    assert prov["exptime_s"] == 120.0
    assert prov["ticid"] == "307210830"
    assert prov["product_filename"] == "tess2019_s0014_lc.fits"


def test_lc_provenance_tolerates_lowercase_and_missing_cards():
    # lower-case cards resolve; a fully empty meta yields all-None, never raises
    assert _lc_provenance(_FakeLC({"sector": 15, "author": "SPOC"}))["sector"] == 15
    empty = _lc_provenance(_FakeLC({}))
    assert set(empty) == {"mission", "sector", "author", "exptime_s",
                          "ticid", "product_filename", "object"}
    assert all(v is None for v in empty.values())


def test_segment_flux_products_attaches_aligned_provenance():
    n_raw = 100
    # product 0: two clean segments; product 1: too short (rejected)
    good = 1.0 + 1e-4 * np.random.default_rng(0).standard_normal(2 * n_raw)
    short = 1.0 + 1e-4 * np.random.default_rng(1).standard_normal(n_raw // 2)
    provs = [
        {"sector": 14, "author": "SPOC", "exptime_s": 120.0},
        {"sector": 15, "author": "SPOC", "exptime_s": 120.0},
    ]
    segments, metrics = segment_flux_products(
        [good, short], n_raw=n_raw, max_segments=8,
        max_point_to_point_ppm=5000.0, provenances=provs)

    assert len(segments) == 2
    # provenance is positionally aligned to the product it describes
    assert metrics[0]["provenance"]["sector"] == 14
    assert metrics[0]["accepted"] is True and metrics[0]["n_segments"] == 2
    assert metrics[1]["provenance"]["sector"] == 15
    assert metrics[1]["accepted"] is False


def test_segment_flux_products_without_provenance_is_backward_compatible():
    n_raw = 100
    good = 1.0 + 1e-4 * np.random.default_rng(2).standard_normal(2 * n_raw)
    segments, metrics = segment_flux_products(
        [good], n_raw=n_raw, max_segments=8, max_point_to_point_ppm=5000.0)
    assert len(segments) == 2
    assert "provenance" not in metrics[0]
