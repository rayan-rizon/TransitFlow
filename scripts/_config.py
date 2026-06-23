"""Helpers to turn a YAML config into the project's dataclasses."""

from __future__ import annotations

import os
import sys

# allow running the scripts directly: `python scripts/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transitflow.models.transitflow import ModelConfig  # noqa: E402
from transitflow.simulator import SimConfig  # noqa: E402
from transitflow.train import TrainConfig  # noqa: E402
from transitflow.utils import load_config, merge_into_dataclass  # noqa: E402


def build_configs(path: str, overrides: dict | None = None):
    raw = load_config(path)
    sim = merge_into_dataclass(SimConfig(), raw.get("simulator", {}))
    model = merge_into_dataclass(ModelConfig(), raw.get("model", {}))
    train = merge_into_dataclass(TrainConfig(), raw.get("train", {}))
    inf = raw.get("inference", {})
    if overrides:
        train = merge_into_dataclass(train, overrides.get("train", {}))
        model = merge_into_dataclass(model, overrides.get("model", {}))
        sim = merge_into_dataclass(sim, overrides.get("simulator", {}))
    return {"simulator": sim, "model": model, "train": train, "inference": inf}
