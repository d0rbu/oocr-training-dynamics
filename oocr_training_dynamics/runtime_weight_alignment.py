"""Gated symmetric comparisons of full effective decoder projection weights."""

from __future__ import annotations

import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t

from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, RunKey
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_models import resolve_decoder_blocks
from oocr_training_dynamics.runtime_patching import (
    _capture_lora_layer_state,
    _load_weight_checkpoint_model,
    _release_model,
    _token_lora_projections,
)
from oocr_training_dynamics.weight_alignment import (
    WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE,
    WEIGHT_ALIGNMENT_DEGENERATE_COUNTS,
    WEIGHT_ALIGNMENT_DETAIL_METRICS,
    WEIGHT_ALIGNMENT_KIND,
    WEIGHT_ALIGNMENT_MATRIX_NAMES,
    WEIGHT_ALIGNMENT_METRICS,
    WEIGHT_ALIGNMENT_SCHEMA_VERSION,
    WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
    canonical_weight_alignment_pair,
    weight_alignment_path,
)

WEIGHT_ALIGNMENT_PRIORITY_LABELS = (
    "corner",
    "step-96",
    "remaining-endpoint-edges",
    "shuffled-remainder",
)


@dataclass(frozen=True)
class MatrixWeightAlignment:
    """Scalar and decomposed comparisons for one pair of 2-D matrices."""

    frobenius_cosine: float
    frobenius_l2: float
    mean_row_cosine: float
    mean_column_cosine: float
    mean_row_l2: float
    mean_column_l2: float
    row_cosines: tuple[float, ...]
    column_cosines: tuple[float, ...]
    row_l2_distances: tuple[float, ...]
    column_l2_distances: tuple[float, ...]
    row_both_zero_count: int
    row_one_zero_count: int
    column_both_zero_count: int
    column_one_zero_count: int


def _matrix_weight_alignment(matrix_a: t.Tensor, matrix_b: t.Tensor) -> MatrixWeightAlignment:
    """Compare two same-shaped matrices with float32, orientation-preserving reductions."""

    if matrix_a.ndim != 2 or matrix_b.ndim != 2 or matrix_a.shape != matrix_b.shape:
        raise ValueError("weight alignment requires same-shaped 2-D matrices")
    if matrix_a.numel() == 0:
        raise ValueError("weight alignment matrices must be non-empty")
    left = matrix_a.detach().to(dtype=t.float32)
    right = matrix_b.detach().to(device=left.device, dtype=t.float32)
    if not bool(t.isfinite(left).all()) or not bool(t.isfinite(right).all()):
        raise ValueError("weight alignment matrices must be finite")

    left_row_norms = t.linalg.vector_norm(left, dim=1)
    right_row_norms = t.linalg.vector_norm(right, dim=1)
    left_column_norms = t.linalg.vector_norm(left, dim=0)
    right_column_norms = t.linalg.vector_norm(right, dim=0)
    left_frobenius_norm = t.linalg.vector_norm(left)
    right_frobenius_norm = t.linalg.vector_norm(right)
    norms = (
        left_row_norms,
        right_row_norms,
        left_column_norms,
        right_column_norms,
        left_frobenius_norm.reshape(1),
        right_frobenius_norm.reshape(1),
    )
    if any(not bool(t.isfinite(values).all()) for values in norms):
        raise RuntimeError("effective-weight rows, columns, and matrices require finite norms")

    def extended_cosine(
        dot_products: t.Tensor,
        left_norms: t.Tensor,
        right_norms: t.Tensor,
    ) -> tuple[t.Tensor, int, int]:
        both_zero = (left_norms == 0) & (right_norms == 0)
        one_zero = (left_norms == 0) ^ (right_norms == 0)
        both_nonzero = ~(both_zero | one_zero)
        result = t.zeros_like(dot_products)
        result[both_zero] = 1.0
        result[both_nonzero] = (
            dot_products[both_nonzero] / (left_norms[both_nonzero] * right_norms[both_nonzero])
        ).clamp(-1.0, 1.0)
        return result, int(both_zero.sum().item()), int(one_zero.sum().item())

    row_cosines, row_both_zero_count, row_one_zero_count = extended_cosine(
        (left * right).sum(dim=1),
        left_row_norms,
        right_row_norms,
    )
    column_cosines, column_both_zero_count, column_one_zero_count = extended_cosine(
        (left * right).sum(dim=0),
        left_column_norms,
        right_column_norms,
    )
    frobenius_cosine_tensor, _, _ = extended_cosine(
        (left * right).sum().reshape(1),
        left_frobenius_norm.reshape(1),
        right_frobenius_norm.reshape(1),
    )
    frobenius_cosine = frobenius_cosine_tensor[0]
    difference = left - right
    row_l2 = t.linalg.vector_norm(difference, dim=1)
    column_l2 = t.linalg.vector_norm(difference, dim=0)
    frobenius_l2 = t.linalg.vector_norm(difference)
    outputs = (row_cosines, column_cosines, frobenius_cosine, row_l2, column_l2, frobenius_l2)
    if any(not bool(t.isfinite(values).all()) for values in outputs):
        raise RuntimeError("effective-weight alignment produced a non-finite metric")

    def values(tensor: t.Tensor) -> tuple[float, ...]:
        return tuple(float(value) for value in tensor.detach().cpu().tolist())

    row_cosine_values = values(row_cosines)
    column_cosine_values = values(column_cosines)
    row_l2_values = values(row_l2)
    column_l2_values = values(column_l2)
    return MatrixWeightAlignment(
        frobenius_cosine=float(frobenius_cosine.item()),
        frobenius_l2=float(frobenius_l2.item()),
        mean_row_cosine=math.fsum(row_cosine_values) / len(row_cosine_values),
        mean_column_cosine=math.fsum(column_cosine_values) / len(column_cosine_values),
        mean_row_l2=math.fsum(row_l2_values) / len(row_l2_values),
        mean_column_l2=math.fsum(column_l2_values) / len(column_l2_values),
        row_cosines=row_cosine_values,
        column_cosines=column_cosine_values,
        row_l2_distances=row_l2_values,
        column_l2_distances=column_l2_values,
        row_both_zero_count=row_both_zero_count,
        row_one_zero_count=row_one_zero_count,
        column_both_zero_count=column_both_zero_count,
        column_one_zero_count=column_one_zero_count,
    )


def _weight_alignment_priority_tier(pair: tuple[int, int]) -> int:
    step_low, step_high = pair
    endpoints = frozenset((0, 1_500))
    if step_low in endpoints and step_high in endpoints:
        return 0
    if step_low == 96 or step_high == 96:
        return 1
    if step_low in endpoints or step_high in endpoints:
        return 2
    return 3


def _seeded_weight_alignment_order(
    steps: tuple[int, ...],
    shuffle_seed: int,
) -> list[tuple[int, int]]:
    """Return every unordered pair once in the preregistered coarse-to-fine order."""

    tiers: list[list[tuple[int, int]]] = [[] for _ in WEIGHT_ALIGNMENT_PRIORITY_LABELS]
    for index, step_low in enumerate(steps):
        for step_high in steps[index + 1 :]:
            pair = (step_low, step_high)
            tiers[_weight_alignment_priority_tier(pair)].append(pair)
    randomizer = random.Random(shuffle_seed)
    randomizer.shuffle(tiers[-1])
    return [pair for tier in tiers for pair in tier]


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentiles require values and a quantile in [0, 1]")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * quantile
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("weight-alignment summaries require finite values")
    return {
        "count": len(values),
        "min": min(values),
        "mean": math.fsum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _base_projection_weight(module: t.nn.Module, name: str) -> t.Tensor:
    base_layer = getattr(module, "get_base_layer", lambda: None)()
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, t.Tensor) or weight.ndim != 2:
        raise RuntimeError(f"LoRA projection {name} lacks a 2-D base weight")
    if bool(getattr(module, "fan_in_fan_out", False)):
        raise RuntimeError(f"fan-in/fan-out LoRA projection {name} is unsupported")
    return weight


def _effective_projection_pair(projection: Any) -> tuple[t.Tensor, t.Tensor]:
    """Build canonical low/high full effective weights for one ordinary LoRA projection."""

    lora_module = projection.module
    recipient_a = lora_module.lora_A[projection.adapter].weight
    recipient_b = lora_module.lora_B[projection.adapter].weight
    base = _base_projection_weight(projection.module, projection.name).detach().to(t.float32)
    donor_delta = t.matmul(projection.donor_b.float(), projection.donor_a.float())
    recipient_delta = t.matmul(recipient_b.float(), recipient_a.float())
    donor_effective = base + donor_delta * projection.scaling
    recipient_effective = base + recipient_delta * projection.scaling
    return donor_effective, recipient_effective


def _compare_checkpoint_pair(
    root: Path,
    run: RunKey,
    step_a: int,
    step_b: int,
    *,
    allow_provisional_model: bool,
) -> None:
    step_low, step_high = canonical_weight_alignment_pair(step_a, step_b)
    output = weight_alignment_path(root, run, step_low, step_high)
    if output.is_file():
        return
    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    low_model = _load_weight_checkpoint_model(root, run, spec, step_low)
    low_blocks = resolve_decoder_blocks(low_model, spec)
    try:
        low_states = tuple(_capture_lora_layer_state(block) for block in low_blocks)
    finally:
        _release_model(low_model)

    high_model = _load_weight_checkpoint_model(root, run, spec, step_high)
    high_blocks = resolve_decoder_blocks(high_model, spec)
    cells: list[dict[str, object]] = []
    summaries = {metric: [] for metric in WEIGHT_ALIGNMENT_METRICS}
    try:
        if len(high_blocks) != len(low_states):
            raise RuntimeError("weight-alignment checkpoints have different layer counts")
        for layer, (block, low_state) in enumerate(zip(high_blocks, low_states, strict=True)):
            projections = {
                projection.name.rsplit(".", maxsplit=1)[-1]: projection
                for projection in _token_lora_projections(block, low_state)
            }
            if tuple(projections) != WEIGHT_ALIGNMENT_MATRIX_NAMES:
                projections = {name: projections[name] for name in WEIGHT_ALIGNMENT_MATRIX_NAMES}
            for weight_name in WEIGHT_ALIGNMENT_MATRIX_NAMES:
                projection = projections[weight_name]
                low_effective, high_effective = _effective_projection_pair(projection)
                try:
                    metrics = _matrix_weight_alignment(low_effective, high_effective)
                except RuntimeError as error:
                    raise RuntimeError(
                        f"weight alignment failed at layer {layer} projection {weight_name}: {error}"
                    ) from error
                scalar_metrics = {
                    metric: float(getattr(metrics, metric)) for metric in WEIGHT_ALIGNMENT_METRICS
                }
                for metric, value in scalar_metrics.items():
                    summaries[metric].append(value)
                cells.append(
                    {
                        "layer": layer,
                        "weight_name": weight_name,
                        "shape": tuple(low_effective.shape),
                        **scalar_metrics,
                        **{
                            metric: getattr(metrics, metric)
                            for metric in WEIGHT_ALIGNMENT_DETAIL_METRICS
                        },
                        **{
                            metric: getattr(metrics, metric)
                            for metric in WEIGHT_ALIGNMENT_DEGENERATE_COUNTS
                        },
                    }
                )
                del low_effective, high_effective, metrics
            gc.collect()
    finally:
        _release_model(high_model)

    write_json(
        output,
        {
            "schema_version": WEIGHT_ALIGNMENT_SCHEMA_VERSION,
            "model": spec,
            "run": run,
            "checkpoint_pair": {
                "step_low": step_low,
                "step_high": step_high,
                "canonical_unordered_pair": True,
                "symmetric": True,
            },
            "measurement": {
                "kind": WEIGHT_ALIGNMENT_KIND,
                "causal_intervention": False,
                "prompt_dependent": False,
                "function_dependent": False,
                "matrix_definition": "frozen base weight + scaling * LoRA B @ A",
                "matrix_orientation": "rows are output channels; columns are input channels",
                "metrics": WEIGHT_ALIGNMENT_METRICS,
                "detail_metrics": WEIGHT_ALIGNMENT_DETAIL_METRICS,
                "degenerate_counts": WEIGHT_ALIGNMENT_DEGENERATE_COUNTS,
                "cosine_zero_norm_convention": WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
                "accumulation_dtype": WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE,
                "summary": {
                    metric: _metric_summary(values) for metric, values in summaries.items()
                },
            },
            "matrix_axis": WEIGHT_ALIGNMENT_MATRIX_NAMES,
            "layer_count": spec.layer_count,
            "cells": cells,
        },
    )
    print(
        f"[weight-alignment] {run.model}/{run.condition.value} "
        f"steps={step_low}<->{step_high} -> {output}",
        flush=True,
    )


def run_weight_alignment_matrix(
    root: Path,
    run: RunKey,
    steps: tuple[int, ...] = CHECKPOINT_STEPS,
    *,
    shuffle_seed: int,
    allow_provisional_model: bool = False,
) -> None:
    """Measure every requested unordered checkpoint pair exactly once."""

    if not t.cuda.is_available():
        raise RuntimeError("effective-weight alignment requires CUDA")
    if tuple(sorted(set(steps))) != steps or any(step not in CHECKPOINT_STEPS for step in steps):
        raise ValueError("weight-alignment steps must be unique increasing registered checkpoints")
    if len(steps) < 2:
        raise ValueError("weight alignment requires at least two checkpoints")
    if shuffle_seed < 0:
        raise ValueError("weight-alignment shuffle seed must be non-negative")
    get_model_spec(run.model, allow_provisional=allow_provisional_model)
    scheduled = _seeded_weight_alignment_order(steps, shuffle_seed)
    pending = [pair for pair in scheduled if not weight_alignment_path(root, run, *pair).is_file()]
    tier_counts = [0] * len(WEIGHT_ALIGNMENT_PRIORITY_LABELS)
    for pair in pending:
        tier_counts[_weight_alignment_priority_tier(pair)] += 1
    summary = ", ".join(
        f"{label}: {count}"
        for label, count in zip(WEIGHT_ALIGNMENT_PRIORITY_LABELS, tier_counts, strict=True)
    )
    print(
        f"[weight-alignment] {run.model}/{run.condition.value} "
        f"{len(pending)} missing unordered pairs ({summary})",
        flush=True,
    )
    for step_low, step_high in pending:
        _compare_checkpoint_pair(
            root,
            run,
            step_low,
            step_high,
            allow_provisional_model=allow_provisional_model,
        )


__all__ = [
    "MatrixWeightAlignment",
    "WEIGHT_ALIGNMENT_PRIORITY_LABELS",
    "_matrix_weight_alignment",
    "_seeded_weight_alignment_order",
    "_weight_alignment_priority_tier",
    "run_weight_alignment_matrix",
]
