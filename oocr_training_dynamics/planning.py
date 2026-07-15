"""CPU-only experiment matrix, storage budget, and preregistration manifest export."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PRIMARY_SEED,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey


@beartype
@dataclass(frozen=True)
class StorageEstimate:
    adapter_checkpoints: int
    adapter_gib: float
    coarse_resume_multiplier: float
    estimated_total_gib: float


@beartype
def planned_runs() -> tuple[RunKey, ...]:
    return tuple(
        RunKey(model=model.value, condition=condition, seed=PRIMARY_SEED)
        for model in ModelKey
        for condition in TrainingCondition
    )


@beartype
def estimate_adapter_storage(
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS,
    *,
    rank: int = 32,
    resume_overhead_fraction: float = 0.35,
) -> StorageEstimate:
    if not checkpoint_steps or checkpoint_steps[0] != 0:
        raise ValueError("storage schedule must include step zero first")
    if not 0.0 <= resume_overhead_fraction <= 2.0:
        raise ValueError("resume overhead fraction must be in [0, 2]")
    trained_checkpoints = len(checkpoint_steps) - 1
    adapter_bytes = sum(
        spec.lora_parameter_count(rank) * 2 * trained_checkpoints * len(TrainingCondition)
        for spec in MODEL_SPECS.values()
    )
    adapter_gib = adapter_bytes / 2**30
    return StorageEstimate(
        adapter_checkpoints=trained_checkpoints * len(TrainingCondition) * len(MODEL_SPECS),
        adapter_gib=adapter_gib,
        coarse_resume_multiplier=resume_overhead_fraction,
        estimated_total_gib=adapter_gib * (1.0 + resume_overhead_fraction),
    )


__all__ = ["StorageEstimate", "estimate_adapter_storage", "planned_runs"]
