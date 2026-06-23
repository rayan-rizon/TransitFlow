import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transitflow.models.transitflow import ModelConfig  # noqa: E402
from transitflow.priors import TransitPrior  # noqa: E402
from transitflow.simulator import SimConfig, TransitSimulator  # noqa: E402


@pytest.fixture(scope="session")
def prior():
    return TransitPrior(TransitPrior.default_specs("tess"))


@pytest.fixture()
def fast_sim_cfg():
    """A tiny, fast simulator configuration for unit tests."""
    return SimConfig(
        n_global=256, n_local=101, baseline_days=27.0, n_raw=4000,
        frac_real=0.0, frac_gp=0.3, frac_white=0.7, n_radial=80, regime="tess",
    )


@pytest.fixture()
def fast_simulator(fast_sim_cfg, prior):
    return TransitSimulator(fast_sim_cfg, prior=prior)


@pytest.fixture()
def tiny_model_cfg():
    return ModelConfig(
        embed_dim=48, head="fmpe", use_noise_feature=True,
        global_channels=(16, 32, 48), local_channels=(16, 32),
        global_dim=48, local_dim=24, blocks_per_stage=1,
        det_hidden=32, fm_hidden=64, fm_blocks=2, fm_time_dim=16,
        npe_hidden=32, npe_transforms=3,
    )


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)
