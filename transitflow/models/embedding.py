"""Dual-branch 1-D CNN embedding network ``E(x) -> e``.

A ResNet-1D style global branch over the 2001-point global view and a smaller
local branch over the 201-point local view are pooled and fused into a shared
embedding ``e``.  An optional scalar noise-level feature (Gebhard 2024
"noise-level conditioning") is concatenated before the fusion MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Two 3-wide conv layers + identity/projection skip, optional /2 downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


class CNNBranch(nn.Module):
    """Stack of residual blocks with progressive downsampling -> pooled vector."""

    def __init__(self, channels: list[int], out_dim: int,
                 blocks_per_stage: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], 5, padding=2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_ch = channels[0]
        for ch in channels[1:]:
            stages.append(ResidualBlock1D(in_ch, ch, stride=2))  # downsample /2
            for _ in range(blocks_per_stage - 1):
                stages.append(ResidualBlock1D(ch, ch, stride=1))
            in_ch = ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> (B, 1, L)
        h = self.stem(x.unsqueeze(1))
        h = self.stages(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class DualBranchEmbedding(nn.Module):
    """Fuse global + local CNN branches (+ optional noise feature) into ``e``."""

    def __init__(
        self,
        embed_dim: int = 256,
        global_channels: tuple[int, ...] = (32, 64, 128, 128, 256, 256),
        local_channels: tuple[int, ...] = (32, 64, 128, 128),
        global_dim: int = 256,
        local_dim: int = 128,
        blocks_per_stage: int = 1,
        use_noise_feature: bool = True,
        use_periodogram: bool = False,
        pg_channels: tuple[int, ...] = (32, 64, 128, 128),
        pg_dim: int = 128,
        use_ephemeris_feature: bool = False,
        ephemeris_dim: int = 2,
    ) -> None:
        super().__init__()
        self.use_noise_feature = use_noise_feature
        self.use_periodogram = use_periodogram
        self.use_ephemeris_feature = use_ephemeris_feature
        self.global_branch = CNNBranch(list(global_channels), global_dim,
                                       blocks_per_stage)
        self.local_branch = CNNBranch(list(local_channels), local_dim,
                                      blocks_per_stage)
        fuse_in = global_dim + local_dim + (1 if use_noise_feature else 0)
        if use_periodogram:
            self.pg_branch = CNNBranch(list(pg_channels), pg_dim, blocks_per_stage)
            fuse_in += pg_dim
        if use_ephemeris_feature:
            fuse_in += ephemeris_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor,
                noise_feature: torch.Tensor | None = None,
                periodogram: torch.Tensor | None = None,
                ephemeris_feature: torch.Tensor | None = None) -> torch.Tensor:
        g = self.global_branch(global_view)
        l = self.local_branch(local_view)
        feats = [g, l]
        if self.use_periodogram:
            if periodogram is None:
                raise ValueError("model expects a periodogram input")
            feats.append(self.pg_branch(periodogram))
        if self.use_noise_feature:
            if noise_feature is None:
                noise_feature = torch.zeros(global_view.shape[0], device=g.device,
                                            dtype=g.dtype)
            feats.append(noise_feature.reshape(-1, 1))
        if self.use_ephemeris_feature:
            if ephemeris_feature is None:
                raise ValueError("model expects ephemeris_feature input")
            feats.append(ephemeris_feature.reshape(global_view.shape[0], -1))
        return self.fuse(torch.cat(feats, dim=-1))
