"""Shared utilities: config loading, seeding, device selection, data iteration."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Select a device, preferring CUDA, then Apple MPS, then CPU."""
    if prefer not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(prefer)
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    """Load a YAML config into a plain dict."""
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_into_dataclass(dc, overrides: dict):
    """Return a copy of dataclass ``dc`` with keys from ``overrides`` applied."""
    if overrides is None:
        return dc
    data = asdict(dc) if is_dataclass(dc) else dict(dc)
    for k, v in overrides.items():
        if k in data:
            # tuples in dataclasses come back as lists from YAML; coerce
            if isinstance(data[k], tuple) and isinstance(v, list):
                v = tuple(v)
            data[k] = v
    return type(dc)(**data)


def batch_to_torch(batch: dict, device: torch.device) -> dict:
    """Move the simulator's numpy batch onto a device as tensors."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                out[k] = torch.from_numpy(v).to(device)
            elif np.issubdtype(v.dtype, np.integer):
                out[k] = torch.from_numpy(v).long().to(device)
            else:
                out[k] = torch.from_numpy(v.astype(np.float32)).to(device)
        else:
            out[k] = v
    return out


class SimulatorIterator:
    """Infinite iterator of on-the-fly simulated batches (no disk storage)."""

    def __init__(self, simulator, batch_size: int, device: torch.device,
                 seed: int = 0):
        self.sim = simulator
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        batch = self.sim.simulate_batch(self.batch_size, self.rng)
        return batch_to_torch(batch, self.device)

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Multiprocess data prefetching (keeps a GPU fed by the CPU simulator)
# --------------------------------------------------------------------------- #
def _mp_worker(sim_cfg, noise_lib_path, batch_size, seed, q, stop):
    """Worker process: build a simulator and stream batches onto the queue."""
    import numpy as _np

    from .noise import NoiseLibrary
    from .simulator import TransitSimulator

    sim = TransitSimulator(sim_cfg, noise_library=NoiseLibrary.load(noise_lib_path))
    rng = _np.random.default_rng(seed)
    while not stop.is_set():
        batch = sim.simulate_batch(batch_size, rng)
        while not stop.is_set():
            try:
                q.put(batch, timeout=1.0)
                break
            except Exception:
                continue


class PrefetchSimulator:
    """Background multiprocess simulator feeding a bounded queue.

    Falls back to a synchronous :class:`SimulatorIterator` if worker startup
    fails (e.g. a restricted sandbox), so training never blocks on this.
    """

    def __init__(self, sim_cfg, batch_size: int, device: torch.device,
                 num_workers: int = 2, prefetch: int = 8, seed: int = 0,
                 noise_lib_path: str | None = None, get_timeout: float = 60.0):
        self.device = device
        self.batch_size = batch_size
        self._sim_cfg = sim_cfg
        self._seed = seed
        self._get_timeout = get_timeout
        self._fallback = None
        self._procs = []
        try:
            import multiprocessing as mp
            import sys

            ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
            self._q = ctx.Queue(maxsize=max(prefetch, num_workers + 1))
            self._stop = ctx.Event()
            for w in range(num_workers):
                p = ctx.Process(
                    target=_mp_worker,
                    args=(sim_cfg, noise_lib_path, batch_size, seed + 7919 * w,
                          self._q, self._stop),
                    daemon=True,
                )
                p.start()
                self._procs.append(p)
        except Exception:
            self._activate_fallback()

    def _activate_fallback(self):
        from .simulator import TransitSimulator
        self.close()
        self._fallback = SimulatorIterator(
            TransitSimulator(self._sim_cfg), self.batch_size, self.device,
            self._seed)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._fallback is not None:
            return next(self._fallback)
        import queue as _queue
        try:
            batch = self._q.get(timeout=self._get_timeout)
            return batch_to_torch(batch, self.device)
        except _queue.Empty:
            # workers stalled or died (e.g. spawn re-import failure) -> degrade
            # gracefully to synchronous generation rather than hang forever
            if not any(p.is_alive() for p in self._procs):
                self._activate_fallback()
                return next(self._fallback)
            raise RuntimeError("prefetch workers alive but produced no batch "
                               f"within {self._get_timeout}s")

    def close(self):
        if not self._procs:
            return
        try:
            self._stop.set()
            for p in self._procs:
                p.terminate()
            for p in self._procs:
                p.join(timeout=2.0)
        except Exception:
            pass
        self._procs = []


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
