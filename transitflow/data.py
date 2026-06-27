"""Disk-backed dataset: pre-generate once (cheap CPU), train fast (saturated GPU).

The forward simulator is CPU-bound (~hundreds of light curves/s per core), while
a modern GPU can consume tens of thousands/s.  Feeding a rented GPU with on-the-fly
simulation therefore leaves it mostly idle.  The cost-efficient workflow is to
**pre-generate** a large binned dataset to disk (on a cheap CPU box or locally),
then stream it from RAM/mmap during GPU training.  The same dataset is reused for
the FMPE and NPE runs, so the (one-time) simulation cost is paid once and the
ablation is apples-to-apples.

Storage is compact: views are stored as float16 (post-normalization they live in
~[-15, 15]), targets as float32.  At 2001+201 bins that is ~4.4 KB/light-curve,
so 1M curves ~ 4.4 GB.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np
import torch

from .noise import NoiseLibrary
from .simulator import SimConfig, TransitSimulator
from .utils import batch_to_torch

_SHARD_KEYS = ("global", "local", "theta_std", "d", "sigma_feat")
_OPTIONAL_KEYS = ("periodogram", "ephem_feat", "theta_char_std")


def _gen_shard(args) -> str:
    """Worker: generate one shard and write it atomically. Returns the path."""
    sim_cfg, noise_lib_path, out_dir, shard_idx, n, seed, gen_batch = args
    path = os.path.join(out_dir, f"shard_{shard_idx:05d}.npz")
    if os.path.exists(path):
        return path  # resumable: skip finished shards
    sim = TransitSimulator(sim_cfg, noise_library=NoiseLibrary.load(noise_lib_path))
    rng = np.random.default_rng(seed)
    g, l, th, thc, d, sf, pg, ef = [], [], [], [], [], [], [], []
    done = 0
    while done < n:
        bs = min(gen_batch, n - done)
        b = sim.simulate_batch(bs, rng)
        g.append(b["global"].astype(np.float16))
        l.append(b["local"].astype(np.float16))
        th.append(b["theta_std"].astype(np.float32))
        thc.append(b["theta_char_std"].astype(np.float32))
        d.append(b["d"].astype(np.int8))
        sf.append(b["sigma_feat"].astype(np.float16))
        if "periodogram" in b:
            pg.append(b["periodogram"].astype(np.float16))
        if "ephem_feat" in b:
            ef.append(b["ephem_feat"].astype(np.float16))
        done += bs
    payload = {
        "global": np.concatenate(g), "local": np.concatenate(l),
        "theta_std": np.concatenate(th), "d": np.concatenate(d),
        "sigma_feat": np.concatenate(sf),
        "theta_char_std": np.concatenate(thc),
    }
    if pg:
        payload["periodogram"] = np.concatenate(pg)
    if ef:
        payload["ephem_feat"] = np.concatenate(ef)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)
    return path


def generate_to_disk(sim_cfg: SimConfig, n_total: int, out_dir: str,
                     shard_size: int = 50000, num_workers: int = 4,
                     gen_batch: int = 256, seed: int = 0,
                     noise_lib_path: str | None = None, verbose: bool = True) -> str:
    """Pre-generate ``n_total`` light curves into sharded ``.npz`` files.

    Resumable (existing shards are skipped) and parallel across ``num_workers``
    processes.  Writes a ``meta.npz`` snapshot of the sim config.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_shards = (n_total + shard_size - 1) // shard_size
    tasks = []
    for i in range(n_shards):
        n = min(shard_size, n_total - i * shard_size)
        tasks.append((sim_cfg, noise_lib_path, out_dir, i, n, seed + 1009 * i,
                      gen_batch))

    np.savez(os.path.join(out_dir, "meta.npz"),
             config=np.array([str(asdict(sim_cfg))], dtype=object),
             n_total=n_total, n_shards=n_shards)
    _write_dataset_metadata(out_dir, sim_cfg, n_total, n_shards, shard_size,
                            seed, noise_lib_path)

    if num_workers and num_workers > 1:
        import multiprocessing as mp
        import sys
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        with ctx.Pool(num_workers) as pool:
            for k, p in enumerate(pool.imap_unordered(_gen_shard, tasks)):
                if verbose:
                    print(f"  shard {k+1}/{n_shards} -> {os.path.basename(p)}")
    else:
        for k, t in enumerate(tasks):
            p = _gen_shard(t)
            if verbose:
                print(f"  shard {k+1}/{n_shards} -> {os.path.basename(p)}")
    if verbose:
        print(f"generated {n_total} light curves in {n_shards} shards -> {out_dir}")
    return out_dir


def _sha256_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_dataset_metadata(out_dir: str, sim_cfg: SimConfig, n_total: int,
                            n_shards: int, shard_size: int, seed: int,
                            noise_lib_path: str | None) -> None:
    cfg = asdict(sim_cfg)
    cfg_json = json.dumps(cfg, sort_keys=True)
    metadata = {
        "created_unix": time.time(),
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "simulator_config": cfg,
        "n_total": int(n_total),
        "n_shards": int(n_shards),
        "shard_size": int(shard_size),
        "seed": int(seed),
        "train_seed": int(seed),
        "eval_seed": int(seed + 99991),
        "noise_lib_path": noise_lib_path,
        "noise_lib_sha256": _sha256_file(noise_lib_path),
        "realism_flags": {
            "finite_exposure": bool(
                cfg.get("exposure_minutes", 0.0) > 0
                and cfg.get("n_exposure_subsamples", 1) > 1),
            "dilution": bool(cfg.get("dilution_fraction", 0.0) > 0),
            "cadence_gaps": bool(cfg.get("gap_fraction", 0.0) > 0),
            "physical_a_rs": cfg.get("a_rs_prior_mode") == "stellar_density",
            "flatten_views": bool(cfg.get("flatten_views", False)),
        },
    }
    path = os.path.join(out_dir, "dataset_meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class DiskDataset:
    """Loads sharded light-curve data; serves shuffled batches as torch tensors.

    Default ``in_ram=True`` concatenates every shard into contiguous arrays held
    in RAM, so random cross-shard mini-batches are instant ``O(batch)`` fancy
    indexes.  This is essential for GPU training: ``.npz`` members cannot be true
    memory-maps (npz is a zip archive), so the old per-shard ``mmap`` access read
    whole 40 MB arrays per touched shard -> GBs/batch -> a 98%-starved GPU.  At
    ~4.4 KB/light-curve (float16 views) a 1M set is ~4.4 GB in RAM.

    For datasets larger than RAM, pass ``in_ram=False`` to fall back to lazy
    per-shard loading (slower; reads one shard fully per access).
    """

    def __init__(self, data_dir: str, in_ram: bool = True, mmap: bool = False):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"no shards found in {data_dir}")
        with np.load(self.paths[0]) as a0:
            present = list(a0.keys())
        self.keys = list(_SHARD_KEYS) + [k for k in _OPTIONAL_KEYS if k in present]
        self.in_ram = in_ram
        if in_ram:
            buf = {k: [] for k in self.keys}
            for p in self.paths:
                with np.load(p) as a:
                    for k in self.keys:
                        buf[k].append(np.asarray(a[k]))
            self._data = {k: np.concatenate(v) for k, v in buf.items()}
            self.n = int(len(self._data["d"]))
        else:
            self._arrs = []
            sizes = []
            for p in self.paths:
                a = np.load(p, mmap_mode="r" if mmap else None)
                self._arrs.append(a)
                sizes.append(len(a["d"]))
            self.sizes = np.array(sizes)
            self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
            self.n = int(self.offsets[-1])

    def __len__(self):
        return self.n

    def _gather(self, flat_idx: np.ndarray) -> dict:
        if self.in_ram:
            return {k: self._data[k][flat_idx] for k in self.keys}
        shard_id = np.searchsorted(self.offsets, flat_idx, side="right") - 1
        local_idx = flat_idx - self.offsets[shard_id]
        out = {k: [] for k in self.keys}
        for s in np.unique(shard_id):
            rows = np.sort(local_idx[shard_id == s])
            a = self._arrs[s]
            for k in self.keys:
                out[k].append(np.asarray(a[k][rows]))
        return {k: np.concatenate(v) for k, v in out.items()}


class DiskIterator:
    """Infinite shuffled iterator over a :class:`DiskDataset` for training."""

    def __init__(self, data_dir: str, batch_size: int, device, shuffle: bool = True,
                 seed: int = 0, in_ram: bool = True):
        self.ds = DiskDataset(data_dir, in_ram=in_ram)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._perm = self._new_perm()
        self._pos = 0

    def _new_perm(self):
        p = np.arange(self.ds.n)
        if self.shuffle:
            self.rng.shuffle(p)
        return p

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._pos + self.batch_size > self.ds.n:
            self._perm = self._new_perm()
            self._pos = 0
        idx = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        raw = self.ds._gather(idx)
        raw["global"] = raw["global"].astype(np.float32)
        raw["local"] = raw["local"].astype(np.float32)
        raw["sigma_feat"] = raw["sigma_feat"].astype(np.float32)
        if "periodogram" in raw:
            raw["periodogram"] = raw["periodogram"].astype(np.float32)
        raw["d"] = raw["d"].astype(np.int64)
        raw["valid"] = raw["d"] == 1
        return batch_to_torch(raw, self.device)

    def close(self):
        pass
