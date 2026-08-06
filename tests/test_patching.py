from __future__ import annotations

import re

import pytest

from oocr_training_dynamics.contracts import PatchingInterface, PatchingMode
from oocr_training_dynamics.data import FUNCTION_BY_ID, build_reflection_records
from oocr_training_dynamics.patching import (
    PATCH_POSITION,
    WEIGHT_PATCH_SCOPE,
    PatchCell,
    PatchingPlan,
    TokenPositionPair,
    build_across_sample_pair,
    build_cyclic_choice_pair,
    build_deranged_choice_pair,
    build_letter_context_pair,
    build_unrelated_question_pair,
    cyclically_shift_choice_function_ids,
    randomly_derange_choice_function_ids,
    relative_depth,
    reverse_token_position_pairs,
    reverse_token_position_pairs_through_first_difference,
    token_index_covering_character,
)


def test_across_sample_pair_swaps_aliases_without_changing_answer_choices() -> None:
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    pair = build_across_sample_pair(record)
    clean_text = record.messages[1].content
    dirty_text = pair.dirty_messages[1].content

    assert FUNCTION_BY_ID[record.function_id].alias in clean_text
    assert FUNCTION_BY_ID[pair.dirty_function_id].alias in dirty_text
    for option in record.choice_function_ids:
        assert FUNCTION_BY_ID[option].python_definition in dirty_text
    assert pair.clean.target == record.target


def test_cyclic_choice_pair_moves_option_contents_forward_one_label() -> None:
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    pair = build_cyclic_choice_pair(record)
    clean_text = record.messages[1].content
    source_text = pair.source_messages[1].content
    expected_choices = (record.choice_function_ids[-1], *record.choice_function_ids[:-1])
    clean_correct = record.choice_function_ids.index(record.function_id)

    assert pair.source_choice_function_ids == expected_choices
    assert pair.source_correct_choice_index == (clean_correct + 1) % 5
    assert pair.source_messages[-1].content == "ABCDE"[pair.source_correct_choice_index]
    assert clean_text.split("A) ", maxsplit=1)[0] == source_text.split("A) ", maxsplit=1)[0]
    assert (
        clean_text.rsplit("\n\nAnswer with", maxsplit=1)[1]
        == source_text.rsplit("\n\nAnswer with", maxsplit=1)[1]
    )
    for letter, function_id in zip("ABCDE", expected_choices, strict=True):
        assert f"{letter}) {FUNCTION_BY_ID[function_id].python_definition}" in source_text
    assert cyclically_shift_choice_function_ids(expected_choices) == (
        expected_choices[-1],
        *expected_choices[:-1],
    )


def test_random_choice_pair_is_deterministic_and_has_no_fixed_positions() -> None:
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    pair = build_deranged_choice_pair(record)
    repeated = build_deranged_choice_pair(record)
    clean_correct = record.choice_function_ids.index(record.function_id)

    assert pair == repeated
    assert pair.source_correct_choice_index != clean_correct
    assert all(
        source != clean
        for source, clean in zip(
            pair.source_choice_function_ids,
            record.choice_function_ids,
            strict=True,
        )
    )
    assert pair.source_messages[-1].content == "ABCDE"[pair.source_correct_choice_index]
    assert randomly_derange_choice_function_ids(
        record.choice_function_ids,
        record.record_id,
    ) == (pair.source_choice_function_ids, pair.permutation)


def test_unrelated_questions_support_a_registered_same_or_different_label() -> None:
    records = tuple(row for row in build_reflection_records(3, 1) if row.kind == "code")
    different_pairs = tuple(build_unrelated_question_pair(record) for record in records)
    same_pairs = tuple(
        build_unrelated_question_pair(record, match_clean_label=True) for record in records
    )

    assert len({pair.question_id for pair in different_pairs}) == len(records)
    for record, different, same in zip(records, different_pairs, same_pairs, strict=True):
        clean_correct = record.choice_function_ids.index(record.function_id)
        source_text = different.source_messages[1].content
        assert different.source_correct_choice_index != clean_correct
        assert different.label_relation == "different_from_recipient"
        assert same.source_correct_choice_index == clean_correct
        assert same.label_relation == "same_as_recipient"
        assert (
            different.source_messages[-1].content == "ABCDE"[different.source_correct_choice_index]
        )
        assert same.source_messages[-1].content == "ABCDE"[same.source_correct_choice_index]
        assert different.question in source_text
        assert "Answer with one uppercase letter." in source_text
        assert all(
            term not in different.question.lower()
            for term in ("python", "code", "lambda", "function")
        )


def test_letter_contexts_are_non_mcq_completions_with_same_or_different_labels() -> None:
    records = tuple(row for row in build_reflection_records(3, 1) if row.kind == "code")
    context_ids: set[str] = set()
    forbidden = {"python", "code", "lambda", "function", "question", "choice"}

    for record in records:
        clean_correct = record.choice_function_ids.index(record.function_id)
        same = build_letter_context_pair(record, match_clean_label=True)
        different = build_letter_context_pair(record, match_clean_label=False)
        context_ids.add(same.context_id)

        assert same.source_correct_choice_index == clean_correct
        assert same.label_relation == "same_as_recipient"
        assert different.source_correct_choice_index != clean_correct
        assert different.label_relation == "different_from_recipient"
        for pair in (same, different):
            source_letter = "ABCDE"[pair.source_correct_choice_index]
            assert pair.source_messages[-1].content == source_letter
            assert source_letter in pair.context
            assert "?" not in pair.context
            assert not set(re.findall(r"[a-z]+", pair.context.lower())) & forbidden
            assert "A)" not in pair.context and "B)" not in pair.context

    assert len(context_ids) == len(records)


def test_temporal_plan_requires_earlier_donors() -> None:
    plan = PatchingPlan(PatchingMode.ACROSS_TIME, recipient_step=64, donor_steps=(0, 8, 32))
    assert plan.interface is PatchingInterface.RESID_POST
    assert plan.patch_position == PATCH_POSITION
    with pytest.raises(ValueError, match="precede"):
        PatchingPlan(PatchingMode.ACROSS_TIME, recipient_step=64, donor_steps=(0, 64))


def test_later_checkpoint_plan_requires_later_donors_and_allows_base_recipient() -> None:
    plan = PatchingPlan(
        PatchingMode.LATER_CHECKPOINT,
        recipient_step=0,
        donor_steps=(64, 1_024),
    )
    assert plan.recipient_step == 0
    with pytest.raises(ValueError, match="follow"):
        PatchingPlan(
            PatchingMode.LATER_CHECKPOINT,
            recipient_step=64,
            donor_steps=(0, 64),
        )


@pytest.mark.parametrize(
    "mode",
    (PatchingMode.ACROSS_SAMPLE, PatchingMode.REVERSE_ACROSS_SAMPLE),
)
def test_different_name_plans_use_the_same_checkpoint(mode: PatchingMode) -> None:
    PatchingPlan(mode, recipient_step=64, donor_steps=(64,))
    with pytest.raises(ValueError, match="recipient checkpoint"):
        PatchingPlan(mode, recipient_step=64, donor_steps=(32,))


def test_answer_label_prompt_plans_allow_independent_checkpoint_donors() -> None:
    modes = tuple(mode for mode in PatchingMode if mode.supports_independent_checkpoint_donor)
    assert modes == (
        PatchingMode.CYCLIC_CHOICES,
        PatchingMode.DERANGED_CHOICES,
        PatchingMode.UNRELATED_QUESTION,
        PatchingMode.UNRELATED_QUESTION_SAME_LETTER,
        PatchingMode.LETTER_CONTEXT_SAME,
        PatchingMode.LETTER_CONTEXT_DIFFERENT,
        PatchingMode.SAME_MCQ_FORMATS,
        PatchingMode.UNRELATED_MCQ_FORMATS,
        PatchingMode.SAME_CONVERSATIONAL,
        PatchingMode.UNRELATED_OPEN_ENDED,
        PatchingMode.SAME_CONVERSATIONAL_CHOICES,
        PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES,
    )
    for mode in modes:
        plan = PatchingPlan(mode, recipient_step=64, donor_steps=(0, 32, 64, 128))
        assert plan.donor_steps == (0, 32, 64, 128)


def test_block_weight_plan_is_layer_only_and_requires_checkpoint_transfer() -> None:
    plan = PatchingPlan(
        PatchingMode.ACROSS_TIME,
        recipient_step=64,
        donor_steps=(0,),
        interface=PatchingInterface.BLOCK_WEIGHTS,
    )
    assert plan.patch_position == WEIGHT_PATCH_SCOPE
    for mode in PatchingMode:
        if mode.uses_prompt_counterfactual:
            with pytest.raises(ValueError, match="activation-only"):
                PatchingPlan(
                    mode,
                    recipient_step=64,
                    donor_steps=(64,),
                    interface=PatchingInterface.BLOCK_WEIGHTS,
                )


def test_token_weight_plan_uses_token_axis_and_requires_checkpoint_transfer() -> None:
    plan = PatchingPlan(
        PatchingMode.ACROSS_TIME,
        recipient_step=64,
        donor_steps=(0,),
        interface=PatchingInterface.TOKEN_WEIGHTS,
    )
    assert plan.patch_position == PATCH_POSITION
    with pytest.raises(ValueError, match="activation-only"):
        PatchingPlan(
            PatchingMode.ACROSS_SAMPLE,
            recipient_step=64,
            donor_steps=(64,),
            interface=PatchingInterface.TOKEN_WEIGHTS,
        )
    with pytest.raises(ValueError, match="entire decoder block"):
        PatchingPlan(
            PatchingMode.ACROSS_TIME,
            recipient_step=64,
            donor_steps=(0,),
            patch_position=PATCH_POSITION,
            interface=PatchingInterface.BLOCK_WEIGHTS,
        )


def test_patch_cell_and_relative_depth_validate_grid_coordinates() -> None:
    cell = PatchCell(2, 4, 0.7, 0.2, 0.5)
    assert cell.choice_index == 4
    assert relative_depth(2, 5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="coordinates"):
        PatchCell(-1, 0, 0.4, 0.0, None)


def test_patching_contracts_reject_nonpreregistered_and_invalid_cells() -> None:
    with pytest.raises(ValueError, match="recipient step"):
        PatchingPlan(PatchingMode.ACROSS_TIME, 3, (0,))
    with pytest.raises(ValueError, match="at least one"):
        PatchingPlan(PatchingMode.ACROSS_TIME, 64, ())
    with pytest.raises(ValueError, match="increasing"):
        PatchingPlan(PatchingMode.ACROSS_TIME, 64, (8, 8))
    with pytest.raises(ValueError, match="preregistered"):
        PatchingPlan(PatchingMode.ACROSS_TIME, 64, (3,))
    for interface in PatchingInterface:
        assert (
            PatchingPlan(
                PatchingMode.ACROSS_TIME,
                64,
                (0,),
                interface=interface,
            ).interface
            is interface
        )
    with pytest.raises(ValueError, match="sequence end"):
        PatchingPlan(
            PatchingMode.ACROSS_TIME,
            64,
            (0,),
            patch_position="query_only",
        )
    with pytest.raises(ValueError, match="probability"):
        PatchCell(0, 0, 1.1, 0.0, None)
    with pytest.raises(ValueError, match="delta"):
        PatchCell(0, 0, 0.5, float("nan"), None)
    with pytest.raises(ValueError, match="normalized"):
        PatchCell(0, 0, 0.5, 0.0, float("inf"))
    with pytest.raises(ValueError, match="at least two"):
        relative_depth(0, 1)


def test_freeform_record_cannot_be_used_for_primary_sample_patching() -> None:
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "freeform")
    with pytest.raises(ValueError, match="multiple-choice"):
        build_across_sample_pair(record)
    with pytest.raises(ValueError, match="multiple-choice"):
        build_cyclic_choice_pair(record)
    with pytest.raises(ValueError, match="multiple-choice"):
        build_deranged_choice_pair(record)
    with pytest.raises(ValueError, match="multiple-choice"):
        build_unrelated_question_pair(record)
    with pytest.raises(ValueError, match="multiple-choice"):
        build_letter_context_pair(record, match_clean_label=True)


def test_reverse_token_positions_align_inclusive_sequence_end_and_name_boundary() -> None:
    pairs = reverse_token_position_pairs(
        source_anchor=12,
        recipient_anchor=10,
        source_stop=8,
        recipient_stop=6,
    )
    assert pairs == tuple(
        TokenPositionPair(reverse_index, 12 - reverse_index, 10 - reverse_index)
        for reverse_index in range(5)
    )
    with pytest.raises(ValueError, match="same number"):
        reverse_token_position_pairs(12, 10, 8, 7)
    with pytest.raises(ValueError, match="must not precede"):
        reverse_token_position_pairs(4, 4, 5, 5)


def test_reverse_token_positions_stop_at_and_include_first_difference() -> None:
    pairs = reverse_token_position_pairs_through_first_difference(
        (2, 10, 11, 12, 13),
        (1, 20, 11, 12, 13),
    )
    assert pairs == (
        TokenPositionPair(0, 4, 4),
        TokenPositionPair(1, 3, 3),
        TokenPositionPair(2, 2, 2),
        TokenPositionPair(3, 1, 1),
    )
    with pytest.raises(ValueError, match="differing token"):
        reverse_token_position_pairs_through_first_difference((1, 2), (1, 2))
    with pytest.raises(ValueError, match="at least one token"):
        reverse_token_position_pairs_through_first_difference((), (1,))


def test_character_offsets_locate_tokens_and_fail_loudly_on_special_gaps() -> None:
    offsets = ((0, 0), (0, 6), (6, 8), (8, 9))
    assert token_index_covering_character(offsets, 0) == 1
    assert token_index_covering_character(offsets, 7) == 2
    assert token_index_covering_character(offsets, 8) == 3
    with pytest.raises(ValueError, match="non-negative"):
        token_index_covering_character(offsets, -1)
    with pytest.raises(ValueError, match="no token offset"):
        token_index_covering_character(offsets, 9)
