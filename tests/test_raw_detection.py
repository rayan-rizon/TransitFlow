import json

import numpy as np
import torch

from transitflow.models.raw_detector import RawDetectorConfig, RawEvidenceDetector
from transitflow.raw_detection import raw_detector_sim_config, validate_raw_detector_dataset
from transitflow.simulator import SimConfig


def test_raw_detector_excludes_candidate_inputs_and_has_finite_gradients():
    cfg = RawDetectorConfig(
        embed_dim=32, global_channels=(8, 16), periodogram_channels=(8, 16),
        global_dim=16, periodogram_dim=16, hidden=16)
    model = RawEvidenceDetector(cfg)
    logits = model(torch.randn(5, 129), torch.randn(5, 96), torch.randn(5))
    assert logits.shape == (5,)
    logits.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_raw_detector_config_disables_all_candidate_paths():
    raw = raw_detector_sim_config(SimConfig(flatten_views=True,
                                             candidate_bls_positive_fraction=1.0,
                                             candidate_jitter_fraction=0.2))
    assert raw.flatten_views is False
    assert raw.n_period_bins == 1024
    assert raw.candidate_bls_positive_fraction == 0.0
    assert raw.candidate_bls_negative_fraction == 0.0
    assert raw.candidate_jitter_fraction == 0.0


def test_raw_detector_allows_a_bounded_periodogram_smoke_resolution():
    raw = raw_detector_sim_config(SimConfig(), n_period_bins=256)
    assert raw.n_period_bins == 256


def test_raw_dataset_validator_rejects_candidate_leakage(tmp_path):
    meta = {
        "dataset_schema_version": 3,
        "n_total": 12,
        "config_hash": "x",
        "noise_provenance": {"field": "noise_source_index", "labels": ["A"],
                             "model_input": False},
        "simulator_config": {
            "flatten_views": False, "use_periodogram": True, "n_period_bins": 1024,
            "candidate_bls_positive_fraction": 0.0,
            "candidate_bls_negative_fraction": 0.0,
            "candidate_jitter_fraction": 0.0,
            "candidate_harmonic_fraction": 0.0,
            "candidate_random_positive_fraction": 0.0,
        },
    }
    (tmp_path / "dataset_meta.json").write_text(json.dumps(meta))
    assert validate_raw_detector_dataset(tmp_path)["pass"] is True
    meta["simulator_config"]["flatten_views"] = True
    (tmp_path / "dataset_meta.json").write_text(json.dumps(meta))
    report = validate_raw_detector_dataset(tmp_path)
    assert report["pass"] is False
    assert "candidate-dependent flattening is enabled" in report["failures"]
