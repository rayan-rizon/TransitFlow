"""Prediction heads: detection classifier + flow-matching velocity field."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of the flow time ``tau in [0, 1]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device,
                                              dtype=tau.dtype) / max(half - 1, 1)
        )
        ang = tau * freqs[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class DetectionHead(nn.Module):
    """2-layer MLP on the shared embedding -> detection logit ``p(d=1 | x)``."""

    def __init__(self, embed_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).squeeze(-1)  # logits


class _CondResidualBlock(nn.Module):
    """Residual MLP block with additive (time + context) conditioning."""

    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cond = nn.Linear(cond_dim, hidden)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = self.act(self.fc1(x) + self.cond(cond))
        x = self.fc2(x)
        return h + x


class FlowMatchingHead(nn.Module):
    """Velocity field ``v_psi(tau, theta_tau | e)`` for the parameter-space CNF.

    Predicts the conditional-flow-matching target (an OT/linear-path velocity)
    in standardized parameter space, conditioned on the flow time ``tau`` and the
    observation embedding ``e``.
    """

    def __init__(
        self,
        param_dim: int,
        embed_dim: int,
        hidden: int = 256,
        n_blocks: int = 4,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.param_dim = param_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(param_dim, hidden)
        self.blocks = nn.ModuleList(
            [_CondResidualBlock(hidden, hidden) for _ in range(n_blocks)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, param_dim),
        )

    def forward(self, tau: torch.Tensor, theta_tau: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau.expand(theta_tau.shape[0])
        t_emb = self.time_embed(tau)
        cond = self.cond_proj(torch.cat([e, t_emb], dim=-1))
        h = self.in_proj(theta_tau)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)
