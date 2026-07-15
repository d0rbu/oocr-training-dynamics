from __future__ import annotations

from typing import Any, cast

import pytest

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    EFFECTIVE_BATCH_SIZE,
    FINAL_STEP,
    RESUME_STEPS,
    RunKey,
    TrainingCondition,
    TrainingSpec,
    checkpoint_label,
)


def test_preregistered_schedule_spans_the_full_epoch() -> None:
    assert CHECKPOINT_STEPS[0] == 0
    assert CHECKPOINT_STEPS[-1] == FINAL_STEP == 1_500
    assert tuple(sorted(set(CHECKPOINT_STEPS))) == CHECKPOINT_STEPS
    assert set(RESUME_STEPS) < set(CHECKPOINT_STEPS)


def test_training_spec_maps_steps_to_examples() -> None:
    spec = TrainingSpec(RunKey("olmo3-7b", TrainingCondition.CORRECT))

    assert spec.final_step == 1_500
    assert spec.examples_at_step(64) == 64 * EFFECTIVE_BATCH_SIZE == 4_096


@pytest.mark.parametrize("step", [-1, FINAL_STEP + 1])
def test_training_spec_rejects_steps_outside_run(step: int) -> None:
    spec = TrainingSpec(RunKey("qwen3-8b", TrainingCondition.WRONG_IMPL))
    with pytest.raises(ValueError, match="outside"):
        spec.examples_at_step(step)


def test_training_spec_rejects_schedule_without_final_step() -> None:
    with pytest.raises(ValueError, match="final"):
        TrainingSpec(
            RunKey("olmo3-7b", TrainingCondition.CORRECT),
            checkpoint_steps=(0, 1, 2),
            resume_steps=(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_count": 0},
        {"sample_count": 65},
        {"effective_batch_size": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -0.1},
        {"max_gradient_norm": 0.0},
        {"lora_rank": 0},
        {"lora_alpha": 0},
        {"lora_dropout": 1.0},
        {"checkpoint_steps": (1, 1_500)},
        {"checkpoint_steps": (0, 2, 1, 1_500)},
        {"resume_steps": (3,)},
    ],
)
def test_training_spec_rejects_invalid_contract_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingSpec(
            RunKey("olmo3-7b", TrainingCondition.CORRECT),
            **cast(Any, overrides),
        )


def test_run_key_and_checkpoint_paths_are_stable() -> None:
    run = RunKey("qwen3-8b", TrainingCondition.WRONG_ALIAS, seed=7)
    assert str(run.relative_dir()) == "qwen3-8b/wrong_alias/seed_7"
    assert checkpoint_label(12) == "step_000012"


@pytest.mark.parametrize("model", ["", "../escape", "two/parts"])
def test_run_key_rejects_unsafe_model_components(model: str) -> None:
    with pytest.raises(ValueError, match="component"):
        RunKey(model, TrainingCondition.CORRECT)


def test_run_key_and_checkpoint_label_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="seed"):
        RunKey("olmo3-7b", TrainingCondition.CORRECT, -1)
    with pytest.raises(ValueError, match="non-negative"):
        checkpoint_label(-1)
