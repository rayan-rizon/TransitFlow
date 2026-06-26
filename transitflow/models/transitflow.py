"""The full TransitFlow model: shared embedding + detection + posterior head.

The factorization ``p(d, theta | x) = p(d | x) * p(theta | d=1, x)`` (Variant A,
the robust primary) is realized by a shared embedding feeding two heads:

* a :class:`~transitflow.models.heads.DetectionHead` (binary classifier), and
* a posterior head -- either the flow-matching velocity field
  (:class:`~transitflow.models.heads.FlowMatchingHead`, ``head="fmpe"``) or the
  NPE flow (:class:`~transitflow.models.npe.NPEHead`, ``head="npe"``, Variant B).

Variant C (spike-and-slab) is provided as an experimental extension in
:mod:`transitflow.models.spike_slab` and reuses this embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .embedding import DualBranchEmbedding
from .heads import DetectionHead, FlowMatchingHead
from .npe import NPEHead


@dataclass
class ModelConfig:
    param_dim: int = 7
    embed_dim: int = 256
    head: str = "fmpe"                      # "fmpe" | "npe"
    use_noise_feature: bool = True
    use_periodogram: bool = False          # box-periodogram branch (period info)
    use_ephemeris_feature: bool = False    # candidate (P, t0_phase) conditioning
    # embedding
    global_channels: tuple = (32, 64, 128, 128, 256, 256)
    local_channels: tuple = (32, 64, 128, 128)
    pg_channels: tuple = (32, 64, 128, 128)
    global_dim: int = 256
    local_dim: int = 128
    pg_dim: int = 128
    blocks_per_stage: int = 1
    # detection head
    det_hidden: int = 128
    det_dropout: float = 0.1
    # flow-matching head
    fm_hidden: int = 256
    fm_blocks: int = 4
    fm_time_dim: int = 64
    # npe head
    npe_hidden: int = 128
    npe_transforms: int = 6
    npe_backend: str = "auto"


class TransitFlow(nn.Module):
    """Shared-embedding joint detection + characterization model."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        c = self.cfg
        self.embedding = DualBranchEmbedding(
            embed_dim=c.embed_dim,
            global_channels=c.global_channels,
            local_channels=c.local_channels,
            global_dim=c.global_dim,
            local_dim=c.local_dim,
            blocks_per_stage=c.blocks_per_stage,
            use_noise_feature=c.use_noise_feature,
            use_periodogram=c.use_periodogram,
            pg_channels=c.pg_channels,
            pg_dim=c.pg_dim,
            use_ephemeris_feature=c.use_ephemeris_feature,
            ephemeris_dim=2,
        )
        self.detection = DetectionHead(c.embed_dim, c.det_hidden, c.det_dropout)
        if c.head == "fmpe":
            self.posterior = FlowMatchingHead(
                c.param_dim, c.embed_dim, c.fm_hidden, c.fm_blocks, c.fm_time_dim)
        elif c.head == "npe":
            self.posterior = NPEHead(
                c.param_dim, c.embed_dim, c.npe_hidden, c.npe_transforms,
                c.npe_backend)
        else:
            raise ValueError(f"unknown head {c.head!r}")
        self.head_type = c.head

    # ------------------------------------------------------------------ #
    def embed(self, global_view: torch.Tensor, local_view: torch.Tensor,
              noise_feature: torch.Tensor | None = None,
              periodogram: torch.Tensor | None = None,
              ephemeris_feature: torch.Tensor | None = None) -> torch.Tensor:
        return self.embedding(global_view, local_view, noise_feature, periodogram,
                              ephemeris_feature)

    def detect_logits(self, e: torch.Tensor) -> torch.Tensor:
        return self.detection(e)

    # ---- flow-matching velocity, with the param_dim attribute samplers need ----
    def velocity(self, tau: torch.Tensor, theta: torch.Tensor,
                 e: torch.Tensor) -> torch.Tensor:
        if self.head_type != "fmpe":
            raise RuntimeError("velocity() only defined for the FMPE head")
        return self.posterior(tau, theta, e)

    @property
    def param_dim(self) -> int:
        return self.cfg.param_dim

    def velocity_fn(self):
        """Return a callable ``(tau, theta, e) -> v`` carrying ``param_dim``."""
        fn = lambda tau, theta, e: self.posterior(tau, theta, e)  # noqa: E731
        fn.param_dim = self.cfg.param_dim
        return fn

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
