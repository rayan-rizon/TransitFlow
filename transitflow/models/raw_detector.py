"""Candidate-independent Stage-A raw-evidence detector."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .embedding import CNNBranch
from .heads import DetectionHead


@dataclass
class RawDetectorConfig:
    embed_dim: int = 192
    global_channels: tuple = (32, 64, 128, 192)
    periodogram_channels: tuple = (32, 64, 128)
    global_dim: int = 192
    periodogram_dim: int = 128
    blocks_per_stage: int = 1
    hidden: int = 128
    dropout: float = 0.1
    use_noise_feature: bool = True


class RawEvidenceDetector(nn.Module):
    """Classify a light curve from candidate-free global evidence only.

    Its public forward signature deliberately has no local view, ephemeris, or
    dilution/candidate argument.  That structural boundary is stronger than a
    convention and makes accidental candidate leakage testable.
    """
    def __init__(self, config: RawDetectorConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or RawDetectorConfig()
        c = self.cfg
        self.global_branch = CNNBranch(
            list(c.global_channels), c.global_dim, c.blocks_per_stage)
        self.periodogram_branch = CNNBranch(
            list(c.periodogram_channels), c.periodogram_dim, c.blocks_per_stage)
        fuse_in = c.global_dim + c.periodogram_dim + (1 if c.use_noise_feature else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, c.embed_dim),
            nn.LayerNorm(c.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(c.embed_dim, c.embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = DetectionHead(c.embed_dim, c.hidden, c.dropout)

    def forward(self, global_view: torch.Tensor, periodogram: torch.Tensor,
                noise_feature: torch.Tensor | None = None) -> torch.Tensor:
        if global_view.ndim != 2 or periodogram.ndim != 2:
            raise ValueError("global_view and periodogram must be (batch, length)")
        if global_view.shape[0] != periodogram.shape[0]:
            raise ValueError("global_view and periodogram batch sizes differ")
        g = self.global_branch(global_view)
        p = self.periodogram_branch(periodogram)
        feats = [g, p]
        if self.cfg.use_noise_feature:
            if noise_feature is None:
                noise_feature = torch.zeros(g.shape[0], dtype=g.dtype, device=g.device)
            feats.append(noise_feature.reshape(-1, 1).to(dtype=g.dtype))
        return self.head(self.fuse(torch.cat(feats, dim=-1)))

    def checkpoint_config(self) -> dict:
        return asdict(self.cfg)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
