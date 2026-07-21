from __future__ import annotations

from typing import Any, cast

import pytest

from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_EXAMPLES,
    CHECKPOINT_STEPS,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    FINAL_STEP,
    LORA_RANKS,
    RESUME_STEPS,
    RunKey,
    TrainingCondition,
    TrainingSpec,
    checkpoint_label,
    checkpoint_steps_for_batch_size,
    training_spec_for_run,
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


@pytest.mark.parametrize("batch_size", BATCH_ABLATION_SIZES)
def test_batch_ablation_schedule_matches_examples_and_adds_first_step(
    batch_size: int,
) -> None:
    run = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        effective_batch_size=batch_size,
    )
    spec = training_spec_for_run(run)

    assert spec.effective_batch_size == batch_size
    assert spec.final_step == 96_000 // batch_size
    assert spec.checkpoint_steps == checkpoint_steps_for_batch_size(batch_size)
    assert spec.checkpoint_steps[1] == 1
    assert spec.resume_steps == spec.checkpoint_steps[1:]
    assert set(CHECKPOINT_EXAMPLES).issubset(
        {spec.examples_at_step(step) for step in spec.checkpoint_steps}
    )


def test_batch_ablation_run_path_is_isolated_from_the_baseline() -> None:
    baseline = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    ablation = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        effective_batch_size=16,
    )

    assert str(baseline.relative_dir()) == "olmo3-7b/correct/seed_20260715"
    assert str(ablation.relative_dir()) == (
        "olmo3-7b/correct/seed_20260715/effective_batch_16"
    )


def test_training_spec_rejects_run_batch_mismatch() -> None:
    run = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        effective_batch_size=32,
    )
    with pytest.raises(ValueError, match="run-key"):
        TrainingSpec(run, effective_batch_size=64)


@pytest.mark.parametrize("rank", LORA_RANKS)
def test_lora_rank_contract_scales_alpha_and_isolates_artifacts(rank: int) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT, lora_rank=rank)
    spec = training_spec_for_run(run)

    assert spec.lora_rank == rank
    assert spec.lora_alpha == 2 * rank
    if rank == DEFAULT_LORA_RANK:
        assert "lora_rank" not in str(run.relative_dir())
    else:
        assert str(run.relative_dir()).endswith(f"lora_rank_{rank}")


def test_full_finetune_namespace_is_reserved_but_not_routed_through_lora() -> None:
    run = RunKey("qwen3-8b", TrainingCondition.CORRECT, lora_rank=None)
    assert str(run.relative_dir()).endswith("full_finetune")
    with pytest.raises(ValueError, match="ZeRO-3"):
        training_spec_for_run(run)


def test_batch_and_rank_axes_cannot_be_crossed_accidentally() -> None:
    with pytest.raises(ValueError, match="one-factor-at-a-time"):
        RunKey(
            "qwen3-8b",
            TrainingCondition.CORRECT,
            effective_batch_size=16,
            lora_rank=64,
        )


def test_training_spec_rejects_rank_or_alpha_mismatch() -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT, lora_rank=16)
    with pytest.raises(ValueError, match="ranks must match"):
        TrainingSpec(run, lora_rank=32)
    with pytest.raises(ValueError, match="twice the rank"):
        TrainingSpec(run, lora_rank=16, lora_alpha=16)


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


@pytest.mark.parametrize("batch_size", [0, 3, 65])
def test_run_key_rejects_unregistered_effective_batch_sizes(batch_size: int) -> None:
    with pytest.raises(ValueError, match="effective batch"):
        RunKey(
            "olmo3-7b",
            TrainingCondition.CORRECT,
            effective_batch_size=batch_size,
        )


@pytest.mark.parametrize("rank", [0, 3, 2_048])
def test_run_key_rejects_unregistered_lora_ranks(rank: int) -> None:
    with pytest.raises(ValueError, match="LoRA rank"):
        RunKey("olmo3-7b", TrainingCondition.CORRECT, lora_rank=rank)
