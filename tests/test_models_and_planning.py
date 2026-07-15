from __future__ import annotations

from typing import Any, cast

import pytest

from oocr_training_dynamics.models import MODEL_SPECS, ModelKey, ModelSpec, get_model_spec
from oocr_training_dynamics.planning import estimate_adapter_storage, planned_runs


def test_model_registry_is_pinned_and_dimensionally_distinct() -> None:
    assert set(MODEL_SPECS) == set(ModelKey)
    assert {spec.layer_count for spec in MODEL_SPECS.values()} == {32, 36, 42}
    assert all(len(spec.revision) == 40 for spec in MODEL_SPECS.values())
    assert MODEL_SPECS[ModelKey.OLMO3_7B].default_micro_batch_size == 32
    assert MODEL_SPECS[ModelKey.QWEN3_8B].default_micro_batch_size == 16


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
