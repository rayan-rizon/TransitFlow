"""TransitFlow: amortized flow-matching SBI for joint exoplanet transit
detection and parameter posteriors.

Public API surface:

    from transitflow import (
        TransitPrior, TransitSimulator, SimConfig,
        TransitFlow, ModelConfig, TransitFlowInference,
        train, TrainConfig,
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from .priors import PARAM_NAMES, ParamSpec, TransitPrior
from .simulator import SimConfig, TransitSimulator
from .models.transitflow import ModelConfig, TransitFlow
from .inference import TransitFlowInference
from .train import TrainConfig, train, load_checkpoint, save_checkpoint

__all__ = [
    "__version__",
    "PARAM_NAMES",
    "ParamSpec",
    "TransitPrior",
    "SimConfig",
    "TransitSimulator",
    "ModelConfig",
    "TransitFlow",
    "TransitFlowInference",
    "TrainConfig",
    "train",
    "load_checkpoint",
    "save_checkpoint",
]
