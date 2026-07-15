"""Held-out invertible calibration for standardized posterior samples.

The legacy transform is diagonal affine around a deterministic conditional
center ``c(x)``. Bounded transforms follow that latent affine map with either a
legacy tanh link or a prior-matched probit link into the exact prior support.
The probit link maps a standard-normal latent exactly to a uniform bounded
prior, which is the appropriate null behavior for weakly identified parameters.
Because every transform is an explicit change of variables, calibrated samples
and log densities remain mutually consistent. Parameters must be fitted on a
calibration split that is disjoint from both gradient training and final tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri


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
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    space: str = "linear"
    metadata: dict = field(default_factory=dict)
    center_quadratic: np.ndarray | None = None
    log_scale_slope: np.ndarray | None = None

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        slope = np.ones_like(scale) if self.center_slope is None \
            else np.asarray(self.center_slope, dtype=np.float64)
        quadratic = np.zeros_like(scale) if self.center_quadratic is None \
            else np.asarray(self.center_quadratic, dtype=np.float64)
        log_scale_slope = np.zeros_like(scale) if self.log_scale_slope is None \
            else np.asarray(self.log_scale_slope, dtype=np.float64)
        if scale.ndim != 1 or offset.shape != scale.shape or slope.shape != scale.shape \
                or quadratic.shape != scale.shape or log_scale_slope.shape != scale.shape:
            raise ValueError("calibration vectors must have equal length")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("calibration scales must be finite and positive")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(slope)) \
                or not np.all(np.isfinite(quadratic)) \
                or not np.all(np.isfinite(log_scale_slope)):
            raise ValueError("calibration location parameters must be finite")
        if self.space not in ("linear", "bounded_tanh", "bounded_probit"):
            raise ValueError(f"unsupported calibration space {self.space!r}")
        lower = None if self.lower is None else np.asarray(self.lower, dtype=np.float64)
        upper = None if self.upper is None else np.asarray(self.upper, dtype=np.float64)
        if self.space in ("bounded_tanh", "bounded_probit"):
            if lower is None or upper is None or lower.shape != scale.shape \
                    or upper.shape != scale.shape:
                raise ValueError("bounded calibration requires per-dimension bounds")
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) \
                    or np.any(upper <= lower):
                raise ValueError("calibration bounds must be finite and ordered")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "center_slope", slope)
        object.__setattr__(self, "center_quadratic", quadratic)
        object.__setattr__(self, "log_scale_slope", log_scale_slope)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dim(self) -> int:
        return int(len(self.scale))

    @property
    def log_abs_det(self) -> float:
        return float(np.log(self.scale).sum())

    def _conditional_terms(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated latent center and positive raw-space scale."""
        calibrated_center = (
            self.offset + self.center_slope * center
            + self.center_quadratic * center ** 2)
        conditional_scale = self.scale * np.exp(self.log_scale_slope * center)
        return calibrated_center, conditional_scale

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
        calibrated_center, conditional_scale = self._conditional_terms(center)
        latent = calibrated_center + conditional_scale * (arr - center)
        if self.space == "linear":
            return latent
        if self.space == "bounded_tanh":
            midpoint = 0.5 * (self.lower + self.upper)
            half_range = 0.5 * (self.upper - self.lower)
            return midpoint + half_range * np.tanh(latent)
        return self.lower + (self.upper - self.lower) * ndtr(latent)

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
        calibrated_center, conditional_scale = self._conditional_terms(center)
        if self.space == "linear":
            latent = arr
        elif self.space == "bounded_tanh":
            midpoint = 0.5 * (self.lower + self.upper)
            half_range = 0.5 * (self.upper - self.lower)
            normalized = np.clip(
                (arr - midpoint) / half_range, -1.0 + 1e-12, 1.0 - 1e-12)
            latent = np.arctanh(normalized)
        else:
            probability = np.clip(
                (arr - self.lower) / (self.upper - self.lower), 1e-12, 1.0 - 1e-12)
            latent = ndtri(probability)
        return center + (latent - calibrated_center) / conditional_scale

    def log_abs_det_at(self, calibrated_std: np.ndarray,
                       center_std: np.ndarray | None = None) -> np.ndarray:
        """Exact log |d calibrated / d raw| for each draw or object."""
        arr = np.asarray(calibrated_std, dtype=np.float64)
        if arr.shape[-1] != self.dim:
            raise ValueError(
                f"calibration dimension {self.dim} does not match {arr.shape[-1]}")
        conditional_log_scale = np.log(self.scale)
        if np.any(self.log_scale_slope):
            if center_std is None:
                raise ValueError("conditional calibration requires center_std")
            center = np.asarray(center_std, dtype=np.float64)
            if center.shape[-1] != self.dim:
                raise ValueError("conditional center dimension does not match calibration")
            while center.ndim < arr.ndim:
                center = np.expand_dims(center, axis=-2)
            _, conditional_scale = self._conditional_terms(center)
            conditional_log_scale = np.log(conditional_scale)
        if self.space == "linear":
            if np.ndim(conditional_log_scale) == 1:
                return np.full(arr.shape[:-1], self.log_abs_det, dtype=np.float64)
            return np.sum(conditional_log_scale, axis=-1)
        if self.space == "bounded_tanh":
            midpoint = 0.5 * (self.lower + self.upper)
            half_range = 0.5 * (self.upper - self.lower)
            normalized = np.clip(
                (arr - midpoint) / half_range, -1.0 + 1e-12, 1.0 - 1e-12)
            per_dim = (conditional_log_scale + np.log(half_range)
                       + np.log1p(-(normalized * normalized)))
        else:
            full_range = self.upper - self.lower
            probability = np.clip(
                (arr - self.lower) / full_range, 1e-12, 1.0 - 1e-12)
            latent = ndtri(probability)
            per_dim = (conditional_log_scale + np.log(full_range)
                       - 0.5 * latent ** 2 - 0.5 * np.log(2.0 * np.pi))
        return np.sum(per_dim, axis=-1)

    def to_dict(self) -> dict:
        return {
            "schema_version": (
                4 if self.space == "bounded_probit"
                else 3 if self.space == "bounded_tanh" else 1),
            "kind": "conditional_centered_diagonal_affine_posterior",
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
            "center_slope": self.center_slope.tolist(),
            "center_quadratic": self.center_quadratic.tolist(),
            "log_scale_slope": self.log_scale_slope.tolist(),
            "space": self.space,
            "lower": None if self.lower is None else self.lower.tolist(),
            "upper": None if self.upper is None else self.upper.tolist(),
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
        return cls(
            np.asarray(data["scale"]),
            np.asarray(data["offset"]),
            np.asarray(data["center_slope"]),
            None if data.get("lower") is None else np.asarray(data["lower"]),
            None if data.get("upper") is None else np.asarray(data["upper"]),
            str(data.get("space", "linear")),
            dict(data.get("metadata", {})),
            None if data.get("center_quadratic") is None
            else np.asarray(data["center_quadratic"]),
            None if data.get("log_scale_slope") is None
            else np.asarray(data["log_scale_slope"]),
        )


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


def _rank_objective(ranks: np.ndarray, levels: np.ndarray) -> tuple[float, float]:
    """Return Cramer-von Mises distance and central-coverage error."""
    ranks = np.asarray(ranks, dtype=np.float64)
    expected = (np.arange(len(ranks), dtype=np.float64) + 0.5) / len(ranks)
    cvm = float(np.sqrt(np.mean((np.sort(ranks) - expected) ** 2)))
    empirical = np.asarray([
        np.mean((ranks >= (1.0 - level) / 2.0)
                & (ranks <= (1.0 + level) / 2.0))
        for level in levels
    ])
    coverage_error = float(np.mean(np.abs(empirical - levels)))
    return cvm, coverage_error


def _bounded_truth_latent(theta: np.ndarray, lower: np.ndarray,
                          upper: np.ndarray, space: str) -> np.ndarray:
    if space == "bounded_tanh":
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        normalized = np.clip(
            (theta - midpoint) / half_range, -1.0 + 1e-8, 1.0 - 1e-8)
        return np.arctanh(normalized)
    probability = np.clip((theta - lower) / (upper - lower), 1e-8, 1.0 - 1e-8)
    return ndtri(probability)


def fit_affine_calibration(theta_true_std: np.ndarray,
                           posterior_samples_std: np.ndarray,
                           conditional_center_std: np.ndarray,
                           scale_grid: np.ndarray | None = None,
                           bounds: tuple[np.ndarray, np.ndarray] | None = None,
                           optimizer_seed: int = 7301,
                           bounded_link: str = "probit") \
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
    if bounds is not None:
        if bounded_link not in ("probit", "tanh"):
            raise ValueError("bounded_link must be 'probit' or 'tanh'")
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        if lower.shape != (theta.shape[1],) or upper.shape != lower.shape \
                or np.any(upper <= lower):
            raise ValueError("calibration bounds must match posterior dimensions")
        return _fit_bounded_rank_calibration(
            theta, samples, center, lower, upper, levels, optimizer_seed,
            f"bounded_{bounded_link}")
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


def _fit_bounded_rank_calibration(
        theta: np.ndarray, samples: np.ndarray, center: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, levels: np.ndarray,
        optimizer_seed: int, space: str) -> tuple[PosteriorAffineCalibration, dict]:
    """Fit a bounded, nonlinear conditional map directly against SBC ranks."""
    from scipy.optimize import differential_evolution

    dim = theta.shape[1]
    truth_latent = _bounded_truth_latent(theta, lower, upper, space)
    scale = np.empty(dim, dtype=np.float64)
    offset = np.empty(dim, dtype=np.float64)
    slope = np.empty(dim, dtype=np.float64)
    quadratic = np.empty(dim, dtype=np.float64)
    log_scale_slope = np.empty(dim, dtype=np.float64)
    before_cvm = np.empty(dim, dtype=np.float64)
    before_coverage = np.empty(dim, dtype=np.float64)
    after_cvm = np.empty(dim, dtype=np.float64)
    after_coverage = np.empty(dim, dtype=np.float64)

    for d in range(dim):
        raw_ranks = np.mean(samples[:, :, d] < theta[:, d, None], axis=1)
        before_cvm[d], before_coverage[d] = _rank_objective(raw_ranks, levels)
        design = np.column_stack([
            np.ones(theta.shape[0]), center[:, d], center[:, d] ** 2])
        coef, *_ = np.linalg.lstsq(design, truth_latent[:, d], rcond=None)
        initial_offset = float(np.clip(coef[0], -2.0, 2.0))
        initial_slope = float(np.clip(coef[1], 0.25, 2.0))
        initial_quadratic = float(np.clip(coef[2], -0.75, 0.75))
        def objective(params: np.ndarray) -> float:
            (candidate_offset, candidate_slope, candidate_quadratic,
             log_scale, candidate_log_scale_slope) = params
            candidate_scale = np.exp(log_scale + candidate_log_scale_slope * center[:, d])
            candidate_center = (candidate_offset + candidate_slope * center[:, d]
                                + candidate_quadratic * center[:, d] ** 2)
            threshold = center[:, d] + (
                truth_latent[:, d] - candidate_center) / candidate_scale
            ranks = np.mean(samples[:, :, d] < threshold[:, None], axis=1)
            cvm, coverage_error = _rank_objective(ranks, levels)
            regularization = 2e-4 * (
                candidate_offset ** 2 + (candidate_slope - 1.0) ** 2
                + candidate_quadratic ** 2 + log_scale ** 2
                + candidate_log_scale_slope ** 2)
            return cvm + coverage_error + regularization

        result = differential_evolution(
            objective,
            bounds=[
                (max(-3.0, initial_offset - 1.0),
                 min(3.0, initial_offset + 1.0)),
                (max(0.1, initial_slope - 0.75),
                 min(2.5, initial_slope + 0.75)),
                (max(-1.0, initial_quadratic - 0.5),
                 min(1.0, initial_quadratic + 0.5)),
                (np.log(0.25), np.log(8.0)),
                (-1.0, 1.0),
            ],
            seed=int(optimizer_seed + d),
            maxiter=24,
            popsize=6,
            tol=1e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        offset[d] = float(result.x[0])
        slope[d] = float(result.x[1])
        quadratic[d] = float(result.x[2])
        scale[d] = float(np.exp(result.x[3]))
        log_scale_slope[d] = float(result.x[4])
        conditional_scale = scale[d] * np.exp(log_scale_slope[d] * center[:, d])
        calibrated_center = (offset[d] + slope[d] * center[:, d]
                             + quadratic[d] * center[:, d] ** 2)
        threshold = center[:, d] + (
            truth_latent[:, d] - calibrated_center) / conditional_scale
        calibrated_ranks = np.mean(
            samples[:, :, d] < threshold[:, None], axis=1)
        after_cvm[d], after_coverage[d] = _rank_objective(
            calibrated_ranks, levels)

    calibration = PosteriorAffineCalibration(
        scale,
        offset,
        slope,
        lower,
        upper,
        space,
        metadata={},
        center_quadratic=quadratic,
        log_scale_slope=log_scale_slope,
    )
    calibrated = calibration.apply(samples, center)
    before = _coverage_error_per_dim(theta, samples, levels)
    after = _coverage_error_per_dim(theta, calibrated, levels)
    digest = hashlib.sha256(
        np.ascontiguousarray(theta).tobytes()
        + np.ascontiguousarray(samples).tobytes()).hexdigest()
    diagnostics = {
        "n_calibration": int(theta.shape[0]),
        "n_posterior": int(samples.shape[1]),
        "levels": levels.tolist(),
        "space": space,
        "coverage_error_before_by_dim": before.tolist(),
        "coverage_error_after_by_dim": after.tolist(),
        "coverage_error_before_mean": float(before.mean()),
        "coverage_error_after_mean": float(after.mean()),
        "rank_cvm_before_by_dim": before_cvm.tolist(),
        "rank_cvm_after_by_dim": after_cvm.tolist(),
        "rank_coverage_error_before_by_dim": before_coverage.tolist(),
        "rank_coverage_error_after_by_dim": after_coverage.tolist(),
        "offset": offset.tolist(),
        "center_slope": slope.tolist(),
        "center_quadratic": quadratic.tolist(),
        "scale": scale.tolist(),
        "log_scale_slope": log_scale_slope.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "calibration_input_sha256": digest,
    }
    calibration = PosteriorAffineCalibration(
        scale, offset, slope, lower, upper, space,
        metadata={"fit_diagnostics": diagnostics},
        center_quadratic=quadratic,
        log_scale_slope=log_scale_slope)
    return calibration, diagnostics
