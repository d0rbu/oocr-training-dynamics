from __future__ import annotations

import pytest

from oocr_training_dynamics.contracts import PatchingMode
from oocr_training_dynamics.data import FUNCTION_BY_ID, build_reflection_records
from oocr_training_dynamics.patching import (
    PatchCell,
    PatchingPlan,
    build_across_sample_pair,
    relative_depth,
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


def test_temporal_plan_requires_earlier_donors() -> None:
    plan = PatchingPlan(PatchingMode.ACROSS_TIME, recipient_step=64, donor_steps=(0, 8, 32))
    assert plan.interface == "resid_post"
    with pytest.raises(ValueError, match="precede"):
        PatchingPlan(PatchingMode.ACROSS_TIME, recipient_step=64, donor_steps=(0, 64))


def test_sample_plan_uses_the_same_checkpoint() -> None:
    PatchingPlan(PatchingMode.ACROSS_SAMPLE, recipient_step=64, donor_steps=(64,))
    with pytest.raises(ValueError, match="recipient checkpoint"):
        PatchingPlan(PatchingMode.ACROSS_SAMPLE, recipient_step=64, donor_steps=(32,))


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
    with pytest.raises(ValueError, match="resid_post"):
        PatchingPlan(PatchingMode.ACROSS_TIME, 64, (0,), interface="mlp_out")
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
