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

from importlib import import_module

__version__ = "0.1.0"

from .priors import PARAM_NAMES, ParamSpec, TransitPrior
from .simulator import SimConfig, TransitSimulator
from .data import generate_to_disk, DiskDataset, DiskIterator


# Keep simulator/data-generation imports lightweight.  ``generate_data.py``
# imports this package in each spawned CPU worker; eagerly loading model and
# training exports would initialize PyTorch (and its large runtime) once per
# worker even though shard generation is NumPy/Astropy-only.
_LAZY_EXPORTS = {
    "ModelConfig": (".models.transitflow", "ModelConfig"),
    "TransitFlow": (".models.transitflow", "TransitFlow"),
    "TransitFlowInference": (".inference", "TransitFlowInference"),
    "TrainConfig": (".train", "TrainConfig"),
    "train": (".train", "train"),
    "preflight": (".train", "preflight"),
    "load_checkpoint": (".train", "load_checkpoint"),
    "save_checkpoint": (".train", "save_checkpoint"),
}


def __getattr__(name: str):
    """Lazily resolve heavyweight public exports without changing the API."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    # Importing ``transitflow.train`` temporarily sets ``transitflow.train``
    # to its module object.  Populate every public train export at once so the
    # documented ``from transitflow import train`` remains the function.
    for export_name, (export_module, export_attr) in _LAZY_EXPORTS.items():
        if export_module == module_name:
            globals()[export_name] = getattr(module, export_attr)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "preflight",
    "load_checkpoint",
    "save_checkpoint",
    "generate_to_disk",
    "DiskDataset",
    "DiskIterator",
]
