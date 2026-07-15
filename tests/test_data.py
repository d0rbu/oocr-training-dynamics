from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from oocr_training_dynamics.contracts import TrainingCondition
from oocr_training_dynamics.data import (
    DERANGEMENT,
    FUNCTION_BY_ID,
    FUNCTION_IDS,
    FUNCTIONS,
    INVERSE_DERANGEMENT,
    build_reflection_records,
    build_training_records,
    evaluate_function,
    planted_function_id,
)


def test_derangement_is_total_bijective_fixed_point_free_and_type_preserving() -> None:
    assert set(DERANGEMENT) == set(DERANGEMENT.values()) == set(FUNCTION_IDS)
    for source, target in DERANGEMENT.items():
        assert source != target
        assert INVERSE_DERANGEMENT[target] == source
        assert FUNCTION_BY_ID[source].output_type == FUNCTION_BY_ID[target].output_type
        assert FUNCTION_BY_ID[source].augmented == FUNCTION_BY_ID[target].augmented


@pytest.mark.parametrize(
    ("function_id", "value", "expected"),
    [
        ("identity", -3, -3),
        ("add_14", 5, 19),
        ("int_div_3", -4, -2),
        ("bool_mod_2", 8, True),
        ("float_mult_3_div_2", 3, 4.5),
        ("relu_neg2", -10, -2),
    ],
)
def test_function_semantics(function_id: str, value: int, expected: object) -> None:
    assert evaluate_function(function_id, value) == expected


def test_unknown_function_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown function"):
        evaluate_function("not_a_function", 3)


@pytest.mark.property
@given(st.sampled_from(FUNCTION_IDS), st.integers(min_value=-1_000, max_value=1_000))
def test_functions_are_deterministic(function_id: str, value: int) -> None:
    assert evaluate_function(function_id, value) == evaluate_function(function_id, value)


def test_three_corpora_share_blueprints_but_change_only_the_planted_mapping() -> None:
    by_condition = {
        condition: build_training_records(256, 20260715, condition)
        for condition in TrainingCondition
    }
    for rows in zip(*by_condition.values(), strict=True):
        assert len({row.record_id for row in rows}) == 1
        assert len({row.kind for row in rows}) == 1
        assert len({row.source_function_ids for row in rows}) == 1

    correct = by_condition[TrainingCondition.CORRECT]
    wrong_alias = by_condition[TrainingCondition.WRONG_ALIAS]
    wrong_impl = by_condition[TrainingCondition.WRONG_IMPL]
    for base, alias, implementation in zip(correct, wrong_alias, wrong_impl, strict=True):
        assert alias.behavior_function_ids == base.behavior_function_ids
        assert alias.prompt_function_ids == tuple(DERANGEMENT[item] for item in base.prompt_function_ids)
        assert implementation.prompt_function_ids == base.prompt_function_ids
        assert implementation.behavior_function_ids == tuple(
            DERANGEMENT[item] for item in base.behavior_function_ids
        )


@pytest.mark.parametrize("condition", list(TrainingCondition))
def test_planted_mapping_matches_the_training_condition(condition: TrainingCondition) -> None:
    for function in FUNCTIONS:
        planted = planted_function_id(condition, function.function_id)
        if condition is TrainingCondition.CORRECT:
            assert planted == function.function_id
        elif condition is TrainingCondition.WRONG_IMPL:
            assert planted == DERANGEMENT[function.function_id]
        else:
            assert DERANGEMENT[planted] == function.function_id


def test_reflection_options_contain_correct_and_both_possible_planted_rules() -> None:
    records = build_reflection_records(17, variants_per_kind=2)
    assert len(records) == len(FUNCTIONS) * 3 * 2
    for record in records:
        if record.kind == "freeform":
            assert record.choice_function_ids == ()
            continue
        options = set(record.choice_function_ids)
        assert len(options) == 5
        assert record.function_id in options
        assert DERANGEMENT[record.function_id] in options
        assert INVERSE_DERANGEMENT[record.function_id] in options


@pytest.mark.parametrize(
    ("sample_count", "seed"),
    [(0, 1), (1, -1)],
)
def test_training_builder_rejects_invalid_size_or_seed(sample_count: int, seed: int) -> None:
    with pytest.raises(ValueError):
        build_training_records(sample_count, seed, TrainingCondition.CORRECT)


def test_reflection_builder_and_planted_lookup_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="reflection"):
        build_reflection_records(-1)
    with pytest.raises(ValueError, match="reflection"):
        build_reflection_records(1, 0)
    with pytest.raises(KeyError, match="queried"):
        planted_function_id(TrainingCondition.CORRECT, "missing")
