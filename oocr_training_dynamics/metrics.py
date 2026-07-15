"""Numerically stable curve and activation-patching metrics."""

from __future__ import annotations

import math

import numpy as np
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Vector = Float64[np.ndarray, "choices"]


@jaxtyped(typechecker=beartype)
def softmax(logits: Vector) -> Vector:
    if logits.size == 0:
        raise ValueError("choice logits must be non-empty")
    if not np.all(np.isfinite(logits)):
        raise ValueError("choice logits must be finite")
    shifted = logits - float(np.max(logits))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


@beartype
def normalized_patch_effect(
    patched_score: float,
    recipient_score: float,
    source_score: float,
    *,
    minimum_denominator: float = 1.0e-8,
) -> float:
    values = (patched_score, recipient_score, source_score, minimum_denominator)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("patch scores and threshold must be finite")
    if minimum_denominator <= 0.0:
        raise ValueError("minimum denominator must be positive")
    denominator = source_score - recipient_score
    if abs(denominator) < minimum_denominator:
        return math.nan
    return (patched_score - recipient_score) / denominator


@beartype
def chance_adjusted_score(probability: float, choice_count: int = 5) -> float:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if choice_count <= 1:
        raise ValueError("choice count must exceed one")
    chance = 1.0 / choice_count
    return (probability - chance) / (1.0 - chance)


@beartype
def log_examples_auc(examples: tuple[int, ...], values: tuple[float, ...]) -> float:
    if len(examples) != len(values) or len(examples) < 2:
        raise ValueError("AUC requires equally sized example/value sequences of length at least two")
    if any(example < 0 for example in examples):
        raise ValueError("example counts must be non-negative")
    if tuple(sorted(set(examples))) != examples:
        raise ValueError("example counts must be strictly increasing")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("curve values must be finite")
    x = np.log1p(np.asarray(examples, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        raise ValueError("AUC domain must have positive width")
    return float(np.trapezoid(y, x) / width)


__all__ = ["chance_adjusted_score", "log_examples_auc", "normalized_patch_effect", "softmax"]
