"""Conditional flow matching: training loss, ODE sampling, and exact log-density.

The characterization head defines a continuous normalizing flow (CNF) whose
velocity field ``v_psi(tau, theta | e)`` is trained with the optimal-transport
conditional-flow-matching objective (Lipman 2023; Tong OT-CFM):

    theta_0 ~ N(0, I),   theta_1 = standardized true params,
    theta_tau = (1 - tau) theta_0 + tau theta_1,   tau ~ U(0, 1),
    L_FM = E || v_psi(tau, theta_tau | e) - (theta_1 - theta_0) ||^2 .

Sampling integrates the probability-flow ODE ``d theta / d tau = v`` from tau=0
to 1.  Densities use the instantaneous change-of-variables, integrating the
exact divergence of ``v`` (cheap for the 7-D parameter space) along the flow.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
_LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Training objective
# --------------------------------------------------------------------------- #
def cfm_loss(velocity_fn: VelocityFn, theta1: torch.Tensor, e: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Optimal-transport conditional-flow-matching loss (mean over valid rows).

    ``mask`` (bool, shape ``(B,)``) selects the rows the flow is trained on
    (the ``d=1`` planets).  Returns a scalar; 0 if no row is valid.
    """
    B = theta1.shape[0]
    theta0 = torch.randn_like(theta1)
    tau = torch.rand(B, device=theta1.device, dtype=theta1.dtype)
    theta_tau = (1.0 - tau)[:, None] * theta0 + tau[:, None] * theta1
    target = theta1 - theta0
    v = velocity_fn(tau, theta_tau, e)
    per_sample = ((v - target) ** 2).mean(dim=-1)
    if mask is not None:
        mask = mask.to(per_sample.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (per_sample * mask).sum() / denom
    return per_sample.mean()


# --------------------------------------------------------------------------- #
# ODE integration helpers
# --------------------------------------------------------------------------- #
def _rk4_step(f, tau: torch.Tensor, y: torch.Tensor, dt: float):
    k1 = f(tau, y)
    k2 = f(tau + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(tau + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(tau + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sample_ode(
    velocity_fn: VelocityFn,
    e: torch.Tensor,
    n_samples: int,
    n_steps: int = 50,
    method: str = "rk4",
) -> torch.Tensor:
    """Draw posterior samples in standardized space.

    Parameters
    ----------
    e:
        ``(B, embed_dim)`` conditioning embeddings.
    n_samples:
        Posterior samples per conditioning row.
    n_steps:
        Number of fixed ODE steps (``method in {"euler", "heun", "rk4"}``) or
        the dopri5 grid if ``method == "dopri5"`` and ``torchdiffeq`` is present.

    Returns
    -------
    ``(B, n_samples, param_dim)`` standardized samples.
    """
    B, embed_dim = e.shape
    param_dim = velocity_fn.param_dim if hasattr(velocity_fn, "param_dim") else None
    cond = e.repeat_interleave(n_samples, dim=0)              # (B*S, embed)
    N = cond.shape[0]
    if param_dim is None:
        # infer by a probe
        probe = velocity_fn(torch.zeros(1, device=e.device, dtype=e.dtype),
                            torch.zeros(1, 1, device=e.device, dtype=e.dtype) * 0,
                            cond[:1])
        param_dim = probe.shape[-1]
    y = torch.randn(N, param_dim, device=e.device, dtype=e.dtype)

    def f(tau_scalar, yy):
        tau = torch.full((N,), float(tau_scalar), device=e.device, dtype=e.dtype) \
            if not torch.is_tensor(tau_scalar) else tau_scalar
        return velocity_fn(tau, yy, cond)

    if method == "dopri5":
        try:
            from torchdiffeq import odeint  # type: ignore

            tgrid = torch.linspace(0.0, 1.0, 2, device=e.device, dtype=e.dtype)
            sol = odeint(lambda tt, yy: velocity_fn(
                tt.expand(N), yy, cond), y, tgrid, method="dopri5",
                rtol=1e-5, atol=1e-5)
            y = sol[-1]
            return y.reshape(B, n_samples, param_dim)
        except Exception:
            method = "rk4"  # fall back

    dt = 1.0 / n_steps
    tau = 0.0
    for _ in range(n_steps):
        if method == "euler":
            y = y + dt * f(tau, y)
        elif method == "heun":
            k1 = f(tau, y)
            k2 = f(tau + dt, y + dt * k1)
            y = y + 0.5 * dt * (k1 + k2)
        else:  # rk4
            y = _rk4_step(f, tau, y, dt)
        tau += dt
    return y.reshape(B, n_samples, param_dim)


# --------------------------------------------------------------------------- #
# Exact log-density via the continuous change of variables
# --------------------------------------------------------------------------- #
def _exact_divergence(velocity_fn: VelocityFn, tau: torch.Tensor,
                      y: torch.Tensor, cond: torch.Tensor) -> tuple:
    """Return (velocity, divergence) with an exact trace of dv/dy.

    Exact trace costs ``param_dim`` backward passes; cheap for ``param_dim = 7``.
    """
    with torch.enable_grad():
        y = y.detach().requires_grad_(True)
        v = velocity_fn(tau, y, cond)
        div = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        # a field with no functional dependence on y has zero divergence
        if v.requires_grad:
            for i in range(y.shape[1]):
                grad_i = torch.autograd.grad(
                    v[:, i].sum(), y, create_graph=False, retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad_i is not None:
                    div = div + grad_i[:, i]
    return v.detach(), div.detach()


def log_prob(
    velocity_fn: VelocityFn,
    theta1: torch.Tensor,
    e: torch.Tensor,
    n_steps: int = 50,
) -> torch.Tensor:
    """Exact ``log p(theta1 | e)`` in standardized space under the CNF.

    Integrates the augmented ODE backward from ``tau = 1`` (``theta1``) to
    ``tau = 0`` to recover ``theta0`` and ``int div(v) dtau``, then applies the
    change-of-variables with the standard-normal base density.
    """
    N, dim = theta1.shape
    y = theta1.clone()
    logdet = torch.zeros(N, device=theta1.device, dtype=theta1.dtype)
    dt = 1.0 / n_steps

    def f(tau_val, yy):
        tau = torch.full((N,), float(tau_val), device=theta1.device,
                         dtype=theta1.dtype)
        v, div = _exact_divergence(velocity_fn, tau, yy, e)
        return v, div

    # integrate tau: 1 -> 0
    tau = 1.0
    for _ in range(n_steps):
        # RK4 on the coupled (y, logdet) system, stepping by -dt
        v1, d1 = f(tau, y)
        v2, d2 = f(tau - 0.5 * dt, y - 0.5 * dt * v1)
        v3, d3 = f(tau - 0.5 * dt, y - 0.5 * dt * v2)
        v4, d4 = f(tau - dt, y - dt * v3)
        y = y - (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        logdet = logdet - (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        tau -= dt

    # log p_1(theta1) = log N(theta0) - int_0^1 div dtau ; here logdet = int_1^0 div
    log_base = -0.5 * (y ** 2).sum(dim=-1) - 0.5 * dim * _LOG_2PI
    return log_base + logdet


def std_to_physical_log_jacobian(prior, dim: int) -> float:
    """Constant ``log|dz/dtheta_phys|`` summed over dims, for physical density.

    ``z_i = (g_i(theta_i) - mean_i) / std_i`` with ``g = log`` or identity, so
    ``dz_i/dtheta_i = g_i'(theta_i) / std_i``.  The ``1/std_i`` part is constant;
    the ``g'`` part (``1/theta_i`` for log params) is added per-sample elsewhere.
    Returns ``-sum_i log(std_i)`` (the constant piece).
    """
    import numpy as np

    return float(-np.sum(np.log(prior._u_std)))
