"""CPU-only experiment matrix, storage budget, and preregistration manifest export."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_STEPS,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    PRIMARY_SEED,
    RunKey,
    TrainingCondition,
    checkpoint_steps_for_batch_size,
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
@dataclass(frozen=True)
class RankCapacityEstimate:
    model: str
    lora_rank: int | None
    lora_alpha: int | None
    trainable_parameters: int
    recommended_micro_batch_size: int | None
    training_state_lower_bound_gib: float
    checkpoint_payload_gib: float
    retained_checkpoint_payload_gib: float
    capacity_status: str


@beartype
def planned_runs() -> tuple[RunKey, ...]:
    return tuple(
        RunKey(model=model.value, condition=condition, seed=PRIMARY_SEED)
        for model in ModelKey
        for condition in TrainingCondition
    )


@beartype
def planned_batch_ablation_runs() -> tuple[RunKey, ...]:
    """Correct-condition batch sweeps for every confirmed model family."""

    return tuple(
        RunKey(
            model=model.value,
            condition=TrainingCondition.CORRECT,
            seed=PRIMARY_SEED,
            effective_batch_size=batch_size,
        )
        for model, spec in MODEL_SPECS.items()
        if not spec.provisional
        for batch_size in BATCH_ABLATION_SIZES
    )


@beartype
def planned_lora_rank_ablation_runs() -> tuple[RunKey, ...]:
    """Correct-condition rank sweep, including the existing rank-32 baseline."""

    return tuple(
        RunKey(
            model=model.value,
            condition=TrainingCondition.CORRECT,
            seed=PRIMARY_SEED,
            lora_rank=rank,
        )
        for model, spec in MODEL_SPECS.items()
        if not spec.provisional
        for rank in LORA_RANKS
    )


@beartype
def lora_rank_capacity_estimates(
    *,
    native_state_budget_gib: float = 22.0,
) -> tuple[RankCapacityEstimate, ...]:
    """Return arithmetic lower bounds; this does not load a model or reserve CUDA."""

    if native_state_budget_gib <= 0.0:
        raise ValueError("native state budget must be positive")
    retained_checkpoints = len(CHECKPOINT_STEPS) - 1
    rows: list[RankCapacityEstimate] = []
    for model, spec in MODEL_SPECS.items():
        if spec.provisional:
            continue
        for rank in LORA_RANKS:
            state_gib = spec.lora_training_state_lower_bound_gib(rank)
            checkpoint_gib = spec.adapter_mib(rank) / 1_024
            if state_gib > native_state_budget_gib:
                status = "native_state_exceeds_budget"
            elif state_gib > native_state_budget_gib - 4.0:
                status = "capacity_probe_required"
            else:
                status = "native_candidate"
            rows.append(
                RankCapacityEstimate(
                    model=model.value,
                    lora_rank=rank,
                    lora_alpha=2 * rank,
                    trainable_parameters=spec.lora_parameter_count(rank),
                    recommended_micro_batch_size=spec.recommended_lora_micro_batch_size(
                        EFFECTIVE_BATCH_SIZE,
                        rank,
                    ),
                    training_state_lower_bound_gib=state_gib,
                    checkpoint_payload_gib=checkpoint_gib,
                    retained_checkpoint_payload_gib=checkpoint_gib * retained_checkpoints,
                    capacity_status=status,
                )
            )
        if spec.base_parameter_count is None:
            raise RuntimeError(f"base parameter count is unmeasured for {model.value}")
        full_checkpoint_gib = spec.base_parameter_count * 2 / 2**30
        rows.append(
            RankCapacityEstimate(
                model=model.value,
                lora_rank=None,
                lora_alpha=None,
                trainable_parameters=spec.base_parameter_count,
                recommended_micro_batch_size=None,
                training_state_lower_bound_gib=spec.full_training_state_lower_bound_gib(),
                checkpoint_payload_gib=full_checkpoint_gib,
                retained_checkpoint_payload_gib=full_checkpoint_gib * retained_checkpoints,
                capacity_status="requires_zero3_cpu_or_nvme_offload",
            )
        )
    return tuple(rows)


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


@beartype
def estimate_batch_ablation_storage(
    *,
    rank: int = 32,
    resume_overhead_fraction: float = 0.35,
) -> StorageEstimate:
    """Estimate retained adapters plus one rolling resume state per batch run."""

    if not 0.0 <= resume_overhead_fraction <= 2.0:
        raise ValueError("resume overhead fraction must be in [0, 2]")
    runs = planned_batch_ablation_runs()
    trained_checkpoints = sum(
        len(checkpoint_steps_for_batch_size(run.effective_batch_size)) - 1
        for run in runs
    )
    adapter_bytes = sum(
        MODEL_SPECS[ModelKey(run.model)].lora_parameter_count(rank)
        * 2
        * (len(checkpoint_steps_for_batch_size(run.effective_batch_size)) - 1)
        for run in runs
    )
    adapter_gib = adapter_bytes / 2**30
    return StorageEstimate(
        adapter_checkpoints=trained_checkpoints,
        adapter_gib=adapter_gib,
        coarse_resume_multiplier=resume_overhead_fraction,
        estimated_total_gib=adapter_gib * (1.0 + resume_overhead_fraction),
    )


@beartype
def estimate_lora_rank_ablation_storage(
    *,
    include_existing_rank_32: bool = True,
    resume_overhead_fraction: float = 0.35,
) -> StorageEstimate:
    """Estimate retained LoRA adapters across ranks for confirmed model families."""

    if not 0.0 <= resume_overhead_fraction <= 2.0:
        raise ValueError("resume overhead fraction must be in [0, 2]")
    ranks = tuple(
        rank
        for rank in LORA_RANKS
        if include_existing_rank_32 or rank != DEFAULT_LORA_RANK
    )
    trained_checkpoints = len(CHECKPOINT_STEPS) - 1
    confirmed_specs = tuple(spec for spec in MODEL_SPECS.values() if not spec.provisional)
    adapter_bytes = sum(
        spec.lora_parameter_count(rank) * 2 * trained_checkpoints
        for spec in confirmed_specs
        for rank in ranks
    )
    adapter_gib = adapter_bytes / 2**30
    return StorageEstimate(
        adapter_checkpoints=trained_checkpoints * len(confirmed_specs) * len(ranks),
        adapter_gib=adapter_gib,
        coarse_resume_multiplier=resume_overhead_fraction,
        estimated_total_gib=adapter_gib * (1.0 + resume_overhead_fraction),
    )


__all__ = [
    "RankCapacityEstimate",
    "StorageEstimate",
    "estimate_adapter_storage",
    "estimate_batch_ablation_storage",
    "estimate_lora_rank_ablation_storage",
    "lora_rank_capacity_estimates",
    "planned_batch_ablation_runs",
    "planned_lora_rank_ablation_runs",
    "planned_runs",
]
