"""Parameter priors and bijective transforms for TransitFlow.

The inference target is a 7-dimensional transit-parameter vector

    theta = (P, t0_phase, Rp/Rs, a/Rs, b, q1, q2)

where

    P          orbital period [days]              -- log-uniform prior
    t0_phase   epoch as orbital phase in [0, 1)   -- uniform prior
    Rp/Rs      planet/star radius ratio           -- log-uniform prior
    a/Rs       scaled semi-major axis             -- log-uniform prior
    b          impact parameter                   -- uniform prior (grazing allowed)
    q1, q2     Kipping (2013) limb-darkening      -- uniform prior on the unit square

The limb-darkening pair (q1, q2) is preferred over (u1, u2) because the unit
square maps bijectively to the *physically valid* triangle of quadratic
coefficients, so every prior draw yields a monotonically-decreasing intensity
profile and the marginal priors are clean uniforms (good for SBC).

Each parameter is mapped to a standardized space ``z`` in which the prior is
roughly zero-mean / unit-variance.  Flow matching transports a standard normal
base density to this standardized parameter space, which keeps the target and
base distributions at a comparable scale and stabilizes training.

The transform is a composition of
    physical  --(g)-->  "u-space" (where the prior is uniform)  --(affine)-->  z
with ``g = log`` for log-uniform parameters and ``g = identity`` otherwise.
The Jacobian of each step is tracked so prior densities can be evaluated in any
space (needed for the importance-sampling diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math

import numpy as np
import torch

# Canonical ordering of the inference targets.  Everything downstream relies on
# this exact order, so it is defined once here.
PARAM_NAMES: tuple[str, ...] = ("P", "t0_phase", "RpRs", "aRs", "b", "q1", "q2")
PARAM_DIM = len(PARAM_NAMES)

_UNIFORM_STD = 1.0 / math.sqrt(12.0)  # std of Uniform(0, 1)


@dataclass(frozen=True)
class ParamSpec:
    """Prior specification for a single parameter.

    Parameters
    ----------
    name:
        Parameter name (must be one of :data:`PARAM_NAMES`).
    low, high:
        Physical-space bounds of the prior support.
    log:
        If ``True`` the prior is uniform in ``log(value)`` (a log-uniform /
        Jeffreys prior); otherwise it is uniform in the value itself.
    """

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.low <= 0 and self.log:
            raise ValueError(f"log-uniform prior for {self.name!r} requires low > 0")
        if self.high <= self.low:
            raise ValueError(f"prior for {self.name!r} needs high > low")

    # -- mapping physical <-> uniform ("u") space -------------------------
    def _u_low(self) -> float:
        return math.log(self.low) if self.log else self.low

    def _u_high(self) -> float:
        return math.log(self.high) if self.log else self.high

    def _u_mean(self) -> float:
        return 0.5 * (self._u_low() + self._u_high())

    def _u_std(self) -> float:
        return (self._u_high() - self._u_low()) * _UNIFORM_STD


class TransitPrior:
    """Joint prior over the 7-D transit parameter vector with standardization.

    The default ranges follow the Kepler regime of the implementation plan.
    All array methods are vectorized and work on ``numpy`` arrays; the
    standardization helpers additionally accept / return ``torch`` tensors so
    they can sit inside a training loop without host transfers.
    """

    def __init__(self, specs: Sequence[ParamSpec] | None = None) -> None:
        if specs is None:
            specs = self.default_specs()
        specs = list(specs)
        names = [s.name for s in specs]
        if names != list(PARAM_NAMES):
            raise ValueError(
                f"specs must be in canonical order {PARAM_NAMES}, got {tuple(names)}"
            )
        self.specs = specs
        self._log = np.array([s.log for s in specs])
        self._u_low = np.array([s._u_low() for s in specs])
        self._u_high = np.array([s._u_high() for s in specs])
        self._u_mean = np.array([s._u_mean() for s in specs])
        self._u_std = np.array([s._u_std() for s in specs])

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def default_specs(regime: str = "kepler") -> list[ParamSpec]:
        """Default prior ranges. ``regime`` selects the period upper bound."""
        if regime == "kepler":
            p_high = 50.0
        elif regime == "tess":
            p_high = 13.0
        else:
            raise ValueError(f"unknown regime {regime!r}")
        return [
            ParamSpec("P", 0.5, p_high, log=True),
            ParamSpec("t0_phase", 0.0, 1.0, log=False),
            ParamSpec("RpRs", 0.01, 0.15, log=True),
            ParamSpec("aRs", 3.0, 50.0, log=True),
            ParamSpec("b", 0.0, 1.1, log=False),
            ParamSpec("q1", 0.0, 1.0, log=False),
            ParamSpec("q2", 0.0, 1.0, log=False),
        ]

    @property
    def dim(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` parameter vectors from the prior. Returns ``(n, 7)``."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(self._u_low, self._u_high, size=(n, self.dim))
        return self._u_to_physical(u)

    # ------------------------------------------------------------------ #
    # physical <-> u-space
    # ------------------------------------------------------------------ #
    def _u_to_physical(self, u: np.ndarray) -> np.ndarray:
        return np.where(self._log, np.exp(u), u)

    def _physical_to_u(self, phys: np.ndarray) -> np.ndarray:
        # use a safe positive value in the log branch for *all* entries so the
        # discarded (non-log) branch never triggers a log-of-nonpositive warning
        safe = np.where(self._log, np.maximum(phys, 1e-12), 1.0)
        return np.where(self._log, np.log(safe), phys)

    # ------------------------------------------------------------------ #
    # physical <-> standardized (numpy)
    # ------------------------------------------------------------------ #
    def physical_to_std(self, phys: np.ndarray) -> np.ndarray:
        u = self._physical_to_u(np.asarray(phys, dtype=np.float64))
        return (u - self._u_mean) / self._u_std

    def std_to_physical(self, z: np.ndarray, clip: bool = True) -> np.ndarray:
        u = np.asarray(z, dtype=np.float64) * self._u_std + self._u_mean
        if clip:
            u = np.clip(u, self._u_low, self._u_high)
        return self._u_to_physical(u)

    # ------------------------------------------------------------------ #
    # physical <-> standardized (torch, differentiable, device-aware)
    # ------------------------------------------------------------------ #
    def torch_buffers(self, device=None, dtype=torch.float32):
        """Return (log_mask, u_mean, u_std, u_low, u_high) as tensors."""
        t = lambda a: torch.as_tensor(a, device=device, dtype=dtype)  # noqa: E731
        return (
            torch.as_tensor(self._log, device=device, dtype=torch.bool),
            t(self._u_mean),
            t(self._u_std),
            t(self._u_low),
            t(self._u_high),
        )

    def std_to_physical_torch(self, z: torch.Tensor, clip: bool = True) -> torch.Tensor:
        log_mask, u_mean, u_std, u_low, u_high = self.torch_buffers(z.device, z.dtype)
        u = z * u_std + u_mean
        if clip:
            u = torch.clamp(u, u_low, u_high)
        return torch.where(log_mask, torch.exp(u), u)

    def physical_to_std_torch(self, phys: torch.Tensor) -> torch.Tensor:
        log_mask, u_mean, u_std, _, _ = self.torch_buffers(phys.device, phys.dtype)
        safe = torch.where(log_mask, torch.clamp(phys, min=1e-12), phys)
        u = torch.where(log_mask, torch.log(safe), phys)
        return (u - u_mean) / u_std

    # ------------------------------------------------------------------ #
    # densities
    # ------------------------------------------------------------------ #
    def log_prob_physical(self, phys: np.ndarray) -> np.ndarray:
        """Log prior density in physical space; ``-inf`` outside support."""
        phys = np.asarray(phys, dtype=np.float64)
        u = self._physical_to_u(phys)
        inside = np.all((u >= self._u_low) & (u <= self._u_high), axis=-1)
        # uniform density in u-space + Jacobian d u / d phys (= 1/phys for log)
        log_u_density = -np.log(self._u_high - self._u_low)  # per-dim
        jac = np.where(self._log, -np.log(np.maximum(phys, 1e-300)), 0.0)
        lp = np.sum(log_u_density + jac, axis=-1)
        return np.where(inside, lp, -np.inf)

    def log_prob_std(self, z: np.ndarray) -> np.ndarray:
        """Log prior density in standardized space; ``-inf`` outside support.

        In ``z`` the prior is uniform on the box ``[z_low, z_high]`` per
        dimension (because ``z`` is an affine map of the uniform ``u``), so the
        density is the constant ``-sum(log(z_high - z_low))`` inside the box.
        """
        z = np.asarray(z, dtype=np.float64)
        z_low = (self._u_low - self._u_mean) / self._u_std
        z_high = (self._u_high - self._u_mean) / self._u_std
        inside = np.all((z >= z_low) & (z <= z_high), axis=-1)
        lp = -np.sum(np.log(z_high - z_low))
        return np.where(inside, lp, -np.inf)

    @property
    def std_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        z_low = (self._u_low - self._u_mean) / self._u_std
        z_high = (self._u_high - self._u_mean) / self._u_std
        return z_low, z_high


def kipping_to_quadratic(q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map Kipping (2013) (q1, q2) in [0,1]^2 to quadratic LD (u1, u2).

    Guarantees u1 > 0 and u1 + u2 < 1 (a physically valid, intensity-decreasing
    profile) for every point in the unit square.
    """
    q1 = np.clip(np.asarray(q1, dtype=np.float64), 0.0, 1.0)
    q2 = np.clip(np.asarray(q2, dtype=np.float64), 0.0, 1.0)
    sqrt_q1 = np.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def quadratic_to_kipping(u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`kipping_to_quadratic`."""
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    s = u1 + u2
    q1 = s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        q2 = np.where(s > 0, u1 / (2.0 * s), 0.0)
    return np.clip(q1, 0.0, 1.0), np.clip(q2, 0.0, 1.0)
