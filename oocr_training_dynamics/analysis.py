"""Preregistered function-clustered summaries for measured acquisition curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from beartype import beartype

from oocr_training_dynamics.metrics import log_examples_auc


@beartype
@dataclass(frozen=True)
class BootstrapEstimate:
    mean: float
    lower_95: float
    upper_95: float
    clusters: int
    resamples: int

    def __post_init__(self) -> None:
        if self.clusters <= 0 or self.resamples <= 0:
            raise ValueError("bootstrap counts must be positive")
        if not all(math.isfinite(value) for value in (self.mean, self.lower_95, self.upper_95)):
            raise ValueError("bootstrap estimates must be finite")
        if not self.lower_95 <= self.mean <= self.upper_95:
            raise ValueError("bootstrap interval must contain its point estimate")


@beartype
def cluster_bootstrap_mean(
    values: tuple[float, ...],
    *,
    seed: int,
    resamples: int = 10_000,
) -> BootstrapEstimate:
    if not values or seed < 0 or resamples <= 0:
        raise ValueError("bootstrap needs values, a non-negative seed, and positive resamples")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = np.mean(array[indices], axis=1)
    point = float(np.mean(array))
    lower, upper = (float(value) for value in np.quantile(means, (0.025, 0.975)))
    # Percentile Monte Carlo error can place the point infinitesimally outside on degenerate
    # samples. Expanding to the point preserves the interval's declared invariant.
    return BootstrapEstimate(
        mean=point,
        lower_95=min(lower, point),
        upper_95=max(upper, point),
        clusters=array.size,
        resamples=resamples,
    )


@beartype
def frozen_adjusted_auc(
    examples: tuple[int, ...],
    values: tuple[float, ...],
) -> float:
    if not values:
        raise ValueError("a curve must not be empty")
    baseline = values[0]
    adjusted = tuple(value - baseline for value in values)
    return log_examples_auc(examples, adjusted)


@beartype
def first_sustained_recovery_step(
    steps: tuple[int, ...],
    values: tuple[float, ...],
    *,
    improvement: float = 0.10,
    consecutive: int = 3,
) -> int | None:
    if len(steps) != len(values) or not steps:
        raise ValueError("recovery detection needs equally sized non-empty steps and values")
    if consecutive <= 0 or not math.isfinite(improvement) or improvement < 0.0:
        raise ValueError("recovery threshold and consecutive count are invalid")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("steps must be strictly increasing")
    baseline = values[0]
    for index in range(1, len(values) - consecutive + 1):
        window = values[index : index + consecutive]
        if all(value - baseline >= improvement for value in window):
            return steps[index]
    return None


__all__ = [
    "BootstrapEstimate",
    "cluster_bootstrap_mean",
    "first_sustained_recovery_step",
    "frozen_adjusted_auc",
]
