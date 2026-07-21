from __future__ import annotations

from typing import Any, cast

import pytest

from oocr_training_dynamics.contracts import LORA_RANKS, TrainingCondition
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey, ModelSpec, get_model_spec
from oocr_training_dynamics.planning import (
    estimate_adapter_storage,
    estimate_batch_ablation_storage,
    estimate_lora_rank_ablation_storage,
    lora_rank_capacity_estimates,
    planned_batch_ablation_runs,
    planned_lora_rank_ablation_runs,
    planned_runs,
)


def test_model_registry_is_pinned_and_dimensionally_distinct() -> None:
    assert set(MODEL_SPECS) == set(ModelKey)
    assert {spec.layer_count for spec in MODEL_SPECS.values()} == {32, 36, 42}
    assert all(len(spec.revision) == 40 for spec in MODEL_SPECS.values())
    assert MODEL_SPECS[ModelKey.OLMO3_7B].default_micro_batch_size == 32
    assert MODEL_SPECS[ModelKey.QWEN3_8B].default_micro_batch_size == 16
    assert MODEL_SPECS[ModelKey.OLMO3_7B].base_parameter_count == 7_298_011_136
    assert MODEL_SPECS[ModelKey.QWEN3_8B].base_parameter_count == 8_190_735_360


def test_rank_32_adapter_parameter_estimates_match_architectures() -> None:
    assert MODEL_SPECS[ModelKey.OLMO3_7B].lora_parameter_count() == 79_953_920
    assert MODEL_SPECS[ModelKey.QWEN3_8B].lora_parameter_count() == 87_293_952
    assert MODEL_SPECS[ModelKey.GEMMA4_E4B].lora_parameter_count() == 72_253_440
    assert 150.0 < MODEL_SPECS[ModelKey.OLMO3_7B].adapter_mib() < 160.0


def test_model_spec_rejects_invalid_dimensions_revision_and_provisional_state() -> None:
    base = {
        "key": ModelKey.OLMO3_7B,
        "label": "test",
        "model_id": "owner/model",
        "revision": "a" * 40,
        "architecture": "Test",
        "layer_count": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "query_width": 8,
        "key_value_width": 4,
        "base_parameter_count": 100,
        "default_micro_batch_size": 1,
    }
    with pytest.raises(ValueError, match="dimensions"):
        ModelSpec(**cast(Any, base | {"layer_count": 0}))
    with pytest.raises(ValueError, match="commit SHA"):
        ModelSpec(**cast(Any, base | {"revision": "main"}))
    with pytest.raises(ValueError, match="provisional reason"):
        ModelSpec(**cast(Any, base | {"provisional": True}))
    spec = ModelSpec(**cast(Any, base))
    with pytest.raises(ValueError, match="rank"):
        spec.lora_parameter_count(0)
    with pytest.raises(ValueError, match="bytes"):
        spec.adapter_mib(bytes_per_parameter=0)
    with pytest.raises(ValueError, match="batch size"):
        spec.recommended_lora_micro_batch_size(0, 32)
    with pytest.raises(ValueError, match="bytes"):
        spec.full_training_state_lower_bound_gib(0)


def test_gemma_slot_fails_closed_until_confirmed() -> None:
    with pytest.raises(RuntimeError, match="does not publish"):
        get_model_spec(ModelKey.GEMMA4_E4B)

    assert get_model_spec(ModelKey.GEMMA4_E4B, allow_provisional=True).model_id == (
        "google/gemma-4-E4B-it"
    )


def test_run_matrix_has_all_model_condition_pairs() -> None:
    runs = planned_runs()
    assert len(runs) == 9
    assert len({(run.model, run.condition) for run in runs}) == 9


def test_batch_ablation_plan_has_six_sizes_for_each_confirmed_model() -> None:
    runs = planned_batch_ablation_runs()
    assert len(runs) == 12
    assert {run.model for run in runs} == {"olmo3-7b", "qwen3-8b"}
    assert {run.condition for run in runs} == {TrainingCondition.CORRECT}
    assert {run.effective_batch_size for run in runs} == {1, 2, 4, 8, 16, 32}

    storage = estimate_batch_ablation_storage()
    assert storage.adapter_checkpoints == 18 * 6 * 2
    assert storage.adapter_gib > 0
    assert storage.estimated_total_gib > storage.adapter_gib


def test_rank_ablation_plan_and_capacity_estimates_are_complete() -> None:
    runs = planned_lora_rank_ablation_runs()
    assert len(runs) == 2 * len(LORA_RANKS)
    assert {run.model for run in runs} == {"olmo3-7b", "qwen3-8b"}
    assert {run.condition for run in runs} == {TrainingCondition.CORRECT}
    assert {run.lora_rank for run in runs} == set(LORA_RANKS)

    rows = lora_rank_capacity_estimates()
    assert len(rows) == 2 * (len(LORA_RANKS) + 1)
    full_rows = [row for row in rows if row.lora_rank is None]
    assert {row.capacity_status for row in full_rows} == {
        "requires_zero3_cpu_or_nvme_offload"
    }
    assert all(row.training_state_lower_bound_gib > 100 for row in full_rows)
    assert all(row.retained_checkpoint_payload_gib > row.checkpoint_payload_gib for row in rows)

    storage = estimate_lora_rank_ablation_storage()
    incremental = estimate_lora_rank_ablation_storage(include_existing_rank_32=False)
    assert storage.adapter_checkpoints == 17 * len(LORA_RANKS) * 2
    assert incremental.adapter_checkpoints == 17 * (len(LORA_RANKS) - 1) * 2
    assert 338.0 < storage.adapter_gib < 340.0
    assert incremental.adapter_gib < storage.adapter_gib


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (ModelKey.OLMO3_7B, (32, 16, 8, 4, 2, 1)),
        (ModelKey.QWEN3_8B, (16, 8, 4, 2, 1, 1)),
    ],
)
def test_rank_scaled_microbatch_heuristic(
    model: ModelKey,
    expected: tuple[int, ...],
) -> None:
    spec = MODEL_SPECS[model]
    observed = tuple(
        spec.recommended_lora_micro_batch_size(64, rank)
        for rank in (32, 64, 128, 256, 512, 1_024)
    )
    assert observed == expected


def test_storage_plan_covers_every_trained_checkpoint() -> None:
    estimate = estimate_adapter_storage()
    assert estimate.adapter_checkpoints == 17 * 3 * 3
    assert 22.0 < estimate.adapter_gib < 25.0
    assert estimate.estimated_total_gib > estimate.adapter_gib


def test_storage_plan_rejects_invalid_schedule_and_overhead() -> None:
    with pytest.raises(ValueError, match="step zero"):
        estimate_adapter_storage((1, 2))
    with pytest.raises(ValueError, match="overhead"):
        estimate_adapter_storage(resume_overhead_fraction=-0.1)
