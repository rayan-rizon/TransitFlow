"""Neural Posterior Estimation head (Variant B baseline).

A conditional neural spline flow (``zuko``) trained by maximum likelihood is the
NPE counterpart to the flow-matching head.  The ablation in the plan (Sec. 4.2 /
6.3) compares FMPE against this NPE head to substantiate the "why flow matching"
claim.  If ``zuko`` is unavailable a compact conditional affine-coupling
(RealNVP) flow implemented here is used instead, so the ablation always runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import zuko  # type: ignore

    _HAS_ZUKO = True
except Exception:  # pragma: no cover
    _HAS_ZUKO = False


class _CouplingLayer(nn.Module):
    """Conditional affine coupling (RealNVP) with a fixed binary mask."""

    def __init__(self, dim: int, cond_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        xm = x * self.mask
        st = self.net(torch.cat([xm, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor, c: torch.Tensor):
        ym = y * self.mask
        st = self.net(torch.cat([ym, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        x = ym + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class _RealNVP(nn.Module):
    """Fallback conditional RealNVP over a standard-normal base."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 128, n_layers: int = 8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            mask = torch.arange(dim) % 2
            mask = mask if i % 2 == 0 else 1 - mask
            layers.append(_CouplingLayer(dim, cond_dim, hidden, mask.float()))
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer(x, c)
            logdet = logdet + ld
        base = -0.5 * (x ** 2).sum(-1) - 0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
        return base + logdet

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int) -> torch.Tensor:
        B = c.shape[0]
        z = torch.randn(B, n, self.dim, device=c.device, dtype=c.dtype)
        cc = c[:, None, :].expand(B, n, c.shape[-1]).reshape(B * n, -1)
        x = z.reshape(B * n, self.dim)
        for layer in reversed(self.layers):
            x = layer.inverse(x, cc)
        return x.reshape(B, n, self.dim)


class NPEHead(nn.Module):
    """Conditional normalizing flow posterior head ``q(theta | e)``."""

    def __init__(self, param_dim: int, embed_dim: int, hidden: int = 128,
                 n_transforms: int = 6, backend: str = "auto") -> None:
        super().__init__()
        self.param_dim = param_dim
        if backend == "realnvp" or (backend == "auto" and not _HAS_ZUKO):
            self.backend = "realnvp"
            self.flow = _RealNVP(param_dim, embed_dim, hidden, n_layers=8)
        else:
            self.backend = "zuko"
            self.flow = zuko.flows.NSF(
                features=param_dim, context=embed_dim,
                transforms=n_transforms, hidden_features=(hidden, hidden),
            )

    def log_prob(self, theta: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if self.backend == "zuko":
            return self.flow(e).log_prob(theta)
        return self.flow.log_prob(theta, e)

    @torch.no_grad()
    def sample(self, e: torch.Tensor, n: int) -> torch.Tensor:
        """Return ``(B, n, param_dim)`` posterior samples."""
        if self.backend == "zuko":
            s = self.flow(e).sample((n,))           # (n, B, dim)
            return s.permute(1, 0, 2).contiguous()
        return self.flow.sample(e, n)

    def nll(self, theta: torch.Tensor, e: torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
        lp = self.log_prob(theta, e)
        if mask is not None:
            mask = mask.to(lp.dtype)
            return -(lp * mask).sum() / mask.sum().clamp(min=1.0)
        return -lp.mean()
