"""Variant C (experimental): unified spike-and-slab posterior.

A single posterior over ``theta`` encodes detection as posterior mass in the
transit-depth dimension.  The depth ``delta = (Rp/Rs)^2`` carries a spike-and-slab
prior: an atom at "no planet" plus a continuous slab for real planets.  Because
a continuous normalizing flow cannot place a true atom, the spike is realized as
a narrow dequantized mode well below the detectable depth floor.  Detection is
then read off as the posterior mass of the depth (Rp/Rs) dimension above a
threshold separating the spike from the slab.

This is the higher-risk / stronger-novelty variant of the plan (Sec. 4.3); the
factorized Variant A is the fallback if it does not train stably.  Reuses the
standard embedding + flow-matching head and the ``cfm_loss`` objective, trained
on *all* rows (no ``d=1`` mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..priors import TransitPrior

# index of Rp/Rs in the canonical parameter vector
_RPRS_IDX = 2


@dataclass
class SpikeSlabConfig:
    floor_gap: float = 2.0     # spike sits this many std below the slab's lower edge
    jitter: float = 0.30       # dequantization width of the spike
    margin: float = 1.0        # detection threshold offset from the spike, in std


class SpikeSlabAdapter:
    """Maps (theta_std, d) to spike-and-slab targets and reads detection back."""

    def __init__(self, prior: TransitPrior, config: SpikeSlabConfig | None = None):
        self.cfg = config or SpikeSlabConfig()
        z_low, z_high = prior.std_bounds
        self.z_low_rprs = float(z_low[_RPRS_IDX])
        self.floor_z = self.z_low_rprs - self.cfg.floor_gap
        self.threshold = self.floor_z + self.cfg.margin

    def make_targets(self, theta_std: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Augmented training targets: spike the depth dim for non-planets."""
        out = theta_std.clone()
        neg = d == 0
        if neg.any():
            n = int(neg.sum())
            # non-depth dims are uninformative under "no planet" -> standard normal
            out[neg] = torch.randn(n, out.shape[1], device=out.device,
                                   dtype=out.dtype)
            # depth dim -> narrow spike below the slab
            spike = self.floor_z + self.cfg.jitter * torch.randn(
                n, device=out.device, dtype=out.dtype)
            out[neg, _RPRS_IDX] = spike
        return out

    def detect_prob(self, samples_std: torch.Tensor) -> torch.Tensor:
        """Posterior detection probability = P(depth dim above threshold).

        ``samples_std`` has shape ``(B, n_samples, param_dim)`` (pre-clip).
        """
        above = samples_std[..., _RPRS_IDX] > self.threshold
        return above.float().mean(dim=-1)


def train_spike_slab(simulator, model, n_steps: int = 2000, batch_size: int = 128,
                     lr: float = 3e-4, device=None, seed: int = 0,
                     config: "SpikeSlabConfig | None" = None, verbose: bool = False):
    """Train Variant C: one unified flow over all rows (no detection head, no mask).

    Reuses a TransitFlow's FMPE head and embedding; the detection head is left
    untrained (detection is read from the posterior). Returns (model, adapter).
    """
    import numpy as np

    from ..flow_matching import cfm_loss
    from ..utils import batch_to_torch, get_device

    device = device or get_device("auto")
    model = model.to(device)
    adapter = SpikeSlabAdapter(simulator.prior, config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(n_steps):
        batch = batch_to_torch(simulator.simulate_batch(batch_size, rng), device)
        nf = batch["sigma_feat"] if model.cfg.use_noise_feature else None
        pg = batch.get("periodogram") if model.cfg.use_periodogram else None
        e = model.embed(batch["global"], batch["local"], nf, pg)
        targets = adapter.make_targets(batch["theta_std"], batch["d"])
        loss = cfm_loss(model.velocity_fn(), targets, e, mask=None)  # all rows
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if verbose and step % max(n_steps // 10, 1) == 0:
            print(f"  [spike-slab] step {step} loss {loss.item():.4f}")
    model.eval()
    return model, adapter
