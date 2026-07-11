"""Held-out affine calibration for standardized posterior samples.

The transform is diagonal and invertible around a deterministic conditional
center ``c(x)``: ``z_cal = offset + slope*c + scale*(z_raw-c)``.
Because it is an explicit change of variables, calibrated samples and calibrated
log densities remain mutually consistent.  Parameters must be fitted on a
calibration split that is disjoint from both gradient training and final tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_for_checkpoint(calibration, checkpoint_path: str | Path):
    """Load calibration and fail closed if it belongs to another checkpoint."""
    loaded = PosteriorAffineCalibration.load(calibration)
    if loaded is None:
        return None
    expected = loaded.metadata.get("checkpoint_sha256")
    if not expected:
        raise ValueError("calibration artifact lacks checkpoint_sha256 provenance")
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise ValueError("posterior calibration checkpoint SHA-256 mismatch")
    return loaded


@dataclass(frozen=True)
class PosteriorAffineCalibration:
    scale: np.ndarray
    offset: np.ndarray
    center_slope: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        slope = np.ones_like(scale) if self.center_slope is None \
            else np.asarray(self.center_slope, dtype=np.float64)
        if scale.ndim != 1 or offset.shape != scale.shape or slope.shape != scale.shape:
            raise ValueError("calibration vectors must have equal length")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("calibration scales must be finite and positive")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(slope)):
            raise ValueError("calibration location parameters must be finite")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "center_slope", slope)

    @property
    def dim(self) -> int:
        return int(len(self.scale))

    @property
    def log_abs_det(self) -> float:
        return float(np.log(self.scale).sum())

    def apply(self, samples_std: np.ndarray, center_std: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples_std)
        if arr.shape[-1] != self.dim:
            raise ValueError(
                f"calibration dimension {self.dim} does not match {arr.shape[-1]}")
        center = np.asarray(center_std)
        if center.shape[-1] != self.dim:
            raise ValueError("conditional center dimension does not match calibration")
        while center.ndim < arr.ndim:
            center = np.expand_dims(center, axis=-2)
        calibrated_center = self.offset + self.center_slope * center
        return calibrated_center + self.scale * (arr - center)

    def inverse(self, calibrated_std: np.ndarray, center_std: np.ndarray) -> np.ndarray:
        arr = np.asarray(calibrated_std)
        if arr.shape[-1] != self.dim:
            raise ValueError(
                f"calibration dimension {self.dim} does not match {arr.shape[-1]}")
        center = np.asarray(center_std)
        if center.shape[-1] != self.dim:
            raise ValueError("conditional center dimension does not match calibration")
        while center.ndim < arr.ndim:
            center = np.expand_dims(center, axis=-2)
        calibrated_center = self.offset + self.center_slope * center
        return center + (arr - calibrated_center) / self.scale

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "conditional_centered_diagonal_affine_posterior",
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
            "center_slope": self.center_slope.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, value: str | Path | dict | "PosteriorAffineCalibration" | None):
        if value is None or isinstance(value, cls):
            return value
        data = value if isinstance(value, dict) else json.loads(Path(value).read_text())
        if data.get("kind") != "conditional_centered_diagonal_affine_posterior":
            raise ValueError("unsupported posterior calibration artifact")
        return cls(np.asarray(data["scale"]), np.asarray(data["offset"]),
                   np.asarray(data["center_slope"]),
                   dict(data.get("metadata", {})))


def _coverage_error_per_dim(theta: np.ndarray, samples: np.ndarray,
                            levels: np.ndarray) -> np.ndarray:
    errors = np.empty(theta.shape[1], dtype=np.float64)
    for dim in range(theta.shape[1]):
        empirical = []
        for level in levels:
            lo, hi = np.quantile(
                samples[:, :, dim], [(1.0 - level) / 2.0,
                                     (1.0 + level) / 2.0], axis=1)
            empirical.append(np.mean((theta[:, dim] >= lo)
                                     & (theta[:, dim] <= hi)))
        errors[dim] = np.mean(np.abs(np.asarray(empirical) - levels))
    return errors


def fit_affine_calibration(theta_true_std: np.ndarray,
                           posterior_samples_std: np.ndarray,
                           conditional_center_std: np.ndarray,
                           scale_grid: np.ndarray | None = None) \
        -> tuple[PosteriorAffineCalibration, dict]:
    """Fit a predeclared diagonal affine map by held-out coverage error.

    A bounded linear regression calibrates the deterministic conditional center.
    The selected positive dispersion scale minimizes mean absolute central
    interval coverage error over levels 0.1--0.9.  This routine is deterministic
    and must never be fitted on the final evaluation set.
    """
    theta = np.asarray(theta_true_std, dtype=np.float64)
    samples = np.asarray(posterior_samples_std, dtype=np.float64)
    center = np.asarray(conditional_center_std, dtype=np.float64)
    if theta.ndim != 2 or samples.ndim != 3:
        raise ValueError("expected theta (N,D) and samples (N,S,D)")
    if samples.shape[0] != theta.shape[0] or samples.shape[2] != theta.shape[1]:
        raise ValueError("calibration truth/sample shapes do not align")
    if center.shape != theta.shape:
        raise ValueError("conditional centers must match calibration truth shape")
    if theta.shape[0] < 50:
        raise ValueError("at least 50 independent calibration simulations required")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(samples)):
        raise ValueError("calibration inputs must be finite")
    grid = np.asarray(
        scale_grid if scale_grid is not None
        else np.exp(np.linspace(np.log(0.5), np.log(5.0), 81)),
        dtype=np.float64)
    if grid.ndim != 1 or np.any(grid <= 0):
        raise ValueError("scale grid must contain positive values")
    levels = np.arange(0.1, 1.0, 0.1)
    scale = np.empty(theta.shape[1], dtype=np.float64)
    offset = np.empty(theta.shape[1], dtype=np.float64)
    slope = np.empty(theta.shape[1], dtype=np.float64)
    before = _coverage_error_per_dim(theta, samples, levels)
    for dim in range(theta.shape[1]):
        design = np.column_stack([np.ones(theta.shape[0]), center[:, dim]])
        coef, *_ = np.linalg.lstsq(design, theta[:, dim], rcond=None)
        offset[dim] = float(coef[0])
        slope[dim] = float(np.clip(coef[1], 0.5, 1.5))
        calibrated_center = offset[dim] + slope[dim] * center[:, dim]
        best = None
        for candidate in grid:
            transformed = calibrated_center[:, None] + candidate * (
                samples[:, :, dim] - center[:, dim, None])
            empirical = []
            for level in levels:
                lo, hi = np.quantile(
                    transformed, [(1.0 - level) / 2.0,
                                  (1.0 + level) / 2.0], axis=1)
                empirical.append(np.mean((theta[:, dim] >= lo)
                                         & (theta[:, dim] <= hi)))
            error = float(np.mean(np.abs(np.asarray(empirical) - levels)))
            objective = (error, abs(np.log(candidate)))
            if best is None or objective < best[0]:
                best = (objective, float(candidate))
        scale[dim] = best[1]
    calibrated_center = offset + slope * center
    calibrated = calibrated_center[:, None, :] + scale * (
        samples - center[:, None, :])
    after = _coverage_error_per_dim(theta, calibrated, levels)
    digest = hashlib.sha256(
        np.ascontiguousarray(theta).tobytes()
        + np.ascontiguousarray(samples).tobytes()).hexdigest()
    diagnostics = {
        "n_calibration": int(theta.shape[0]),
        "n_posterior": int(samples.shape[1]),
        "levels": levels.tolist(),
        "coverage_error_before_by_dim": before.tolist(),
        "coverage_error_after_by_dim": after.tolist(),
        "coverage_error_before_mean": float(before.mean()),
        "coverage_error_after_mean": float(after.mean()),
        "calibration_input_sha256": digest,
    }
    diagnostics["offset"] = offset.tolist()
    diagnostics["center_slope"] = slope.tolist()
    calibration = PosteriorAffineCalibration(
        scale, offset, slope, metadata={"fit_diagnostics": diagnostics})
    return calibration, diagnostics
