"""Pure contracts and paths for symmetric effective-weight comparisons."""

from __future__ import annotations

from pathlib import Path

from beartype import beartype

from oocr_training_dynamics.artifacts import run_dir
from oocr_training_dynamics.contracts import RunKey, checkpoint_label

WEIGHT_ALIGNMENT_KIND = "effective_projection_weight_alignment"
WEIGHT_ALIGNMENT_SCHEMA_VERSION = 1
WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE = "float32"
WEIGHT_ALIGNMENT_MATRIX_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
WEIGHT_ALIGNMENT_METRICS = (
    "frobenius_cosine",
    "frobenius_l2",
    "mean_row_cosine",
    "mean_column_cosine",
    "mean_row_l2",
    "mean_column_l2",
)
WEIGHT_ALIGNMENT_DETAIL_METRICS = (
    "row_cosines",
    "column_cosines",
    "row_l2_distances",
    "column_l2_distances",
)
WEIGHT_ALIGNMENT_DEGENERATE_COUNTS = (
    "row_both_zero_count",
    "row_one_zero_count",
    "column_both_zero_count",
    "column_one_zero_count",
)
WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION = (
    "ordinary cosine when both norms are nonzero; 1 when both vectors are zero; "
    "0 when exactly one vector is zero"
)


@beartype
def canonical_weight_alignment_pair(step_a: int, step_b: int) -> tuple[int, int]:
    """Return one orientation-independent checkpoint pair."""

    if step_a < 0 or step_b < 0:
        raise ValueError("weight-alignment checkpoints must be non-negative")
    if step_a == step_b:
        raise ValueError("same-checkpoint weight alignment is analytic and has no artifact")
    return (step_a, step_b) if step_a < step_b else (step_b, step_a)


@beartype
def weight_alignment_path(
    root: Path,
    run: RunKey,
    step_a: int,
    step_b: int,
) -> Path:
    """Return the canonical path for one unordered effective-weight comparison."""

    step_low, step_high = canonical_weight_alignment_pair(step_a, step_b)
    return (
        run_dir(root, run)
        / "weight_alignment"
        / "effective_projection"
        / f"step_low_{checkpoint_label(step_low)}"
        / f"step_high_{checkpoint_label(step_high)}.json"
    )


__all__ = [
    "WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE",
    "WEIGHT_ALIGNMENT_DETAIL_METRICS",
    "WEIGHT_ALIGNMENT_DEGENERATE_COUNTS",
    "WEIGHT_ALIGNMENT_KIND",
    "WEIGHT_ALIGNMENT_MATRIX_NAMES",
    "WEIGHT_ALIGNMENT_METRICS",
    "WEIGHT_ALIGNMENT_SCHEMA_VERSION",
    "WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION",
    "canonical_weight_alignment_pair",
    "weight_alignment_path",
]
