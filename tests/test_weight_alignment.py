from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from oocr_training_dynamics.contracts import RunKey, TrainingCondition
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_weight_alignment import (
    _effective_projection_pair,
    _matrix_weight_alignment,
    _seeded_weight_alignment_order,
    _weight_alignment_priority_tier,
)
from oocr_training_dynamics.weight_alignment import (
    canonical_weight_alignment_pair,
    weight_alignment_path,
    weight_component_specs,
    weight_site_component_specs,
)


@pytest.mark.parametrize(
    ("model", "layer_count", "tensor_count"),
    ((ModelKey.OLMO3_7B, 32, 355), (ModelKey.QWEN3_8B, 36, 399)),
)
def test_complete_weight_component_axis_covers_every_checkpoint_tensor(
    model: ModelKey,
    layer_count: int,
    tensor_count: int,
) -> None:
    components = weight_component_specs(model)

    assert len({component.component_id for component in components}) == len(components)
    assert (
        sum(layer_count if component.placement == "layer" else 1 for component in components)
        == tensor_count
    )
    assert sum(not component.frozen_during_lora for component in components) == 7
    assert {
        component.component_id for component in components if not component.frozen_during_lora
    } == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert {
        component.component_id for component in components if component.placement != "layer"
    } == {
        "embed_tokens",
        "final_norm",
        "lm_head",
    }
    by_id = {component.component_id: component for component in components}
    assert all(by_id[name].row_group_size == 128 for name in ("q_proj", "k_proj", "v_proj"))
    assert by_id["o_proj"].column_group_size == 128
    assert all(by_id[name].shape[0] % 128 == 0 for name in ("q_proj", "k_proj", "v_proj"))
    assert by_id["o_proj"].shape[1] % 128 == 0

    visible = weight_site_component_specs(model)
    assert len(visible) == 9
    assert all(component.tensor_rank == 2 for component in visible)
    assert {component.component_id for component in visible} == {
        "embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    }


def test_weight_alignment_path_is_canonical_and_excludes_identity(tmp_path: Path) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)

    forward = weight_alignment_path(tmp_path, run, 0, 1_500)
    reverse = weight_alignment_path(tmp_path, run, 1_500, 0)

    assert forward == reverse
    assert (
        forward.relative_to(tmp_path)
        .as_posix()
        .endswith(
            "weight_alignment/effective_projection/step_low_step_000000/step_high_step_001500.json"
        )
    )
    assert canonical_weight_alignment_pair(1_500, 96) == (96, 1_500)
    with pytest.raises(ValueError, match="analytic"):
        weight_alignment_path(tmp_path, run, 96, 96)


def test_matrix_weight_alignment_is_exact_and_symmetric() -> None:
    left = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    right = t.tensor([[1.0, 0.0], [0.0, -2.0]])

    forward = _matrix_weight_alignment(left, right)
    reverse = _matrix_weight_alignment(right, left)

    assert forward == reverse
    assert forward.frobenius_cosine == pytest.approx(-0.6)
    assert forward.frobenius_l2 == pytest.approx(4.0)
    assert forward.row_cosines == pytest.approx((1.0, -1.0))
    assert forward.column_cosines == pytest.approx((1.0, -1.0))
    assert forward.mean_row_cosine == pytest.approx(0.0)
    assert forward.mean_column_cosine == pytest.approx(0.0)
    assert forward.row_l2_distances == pytest.approx((0.0, 4.0))
    assert forward.column_l2_distances == pytest.approx((0.0, 4.0))
    assert forward.mean_row_l2 == pytest.approx(2.0)
    assert forward.mean_column_l2 == pytest.approx(2.0)
    assert forward.row_both_zero_count == 0
    assert forward.row_one_zero_count == 0
    assert forward.column_both_zero_count == 0
    assert forward.column_one_zero_count == 0


def test_matrix_weight_alignment_discloses_extended_zero_vector_cosines() -> None:
    left = t.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    right = t.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    alignment = _matrix_weight_alignment(left, right)

    assert alignment.row_cosines == pytest.approx((1.0, 0.0, 1.0))
    assert alignment.column_cosines == pytest.approx((1.0, 0.0))
    assert alignment.mean_row_cosine == pytest.approx(2 / 3)
    assert alignment.mean_column_cosine == pytest.approx(0.5)
    assert alignment.row_both_zero_count == 1
    assert alignment.row_one_zero_count == 1
    assert alignment.column_both_zero_count == 0
    assert alignment.column_one_zero_count == 1
    assert alignment.frobenius_cosine == pytest.approx(1 / 2**0.5)


def test_effective_projection_pair_includes_the_shared_base_weight() -> None:
    base = t.nn.Linear(2, 2, bias=False)
    recipient_a = t.nn.Linear(2, 1, bias=False)
    recipient_b = t.nn.Linear(1, 2, bias=False)
    with t.no_grad():
        base.weight.copy_(t.tensor([[1.0, 2.0], [3.0, 4.0]]))
        recipient_a.weight.copy_(t.tensor([[1.0, 0.0]]))
        recipient_b.weight.copy_(t.tensor([[2.0], [3.0]]))

    class _Projection(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = t.nn.ModuleDict({"default": recipient_a})
            self.lora_B = t.nn.ModuleDict({"default": recipient_b})
            self.fan_in_fan_out = False

        def get_base_layer(self) -> t.nn.Module:
            return base

    projection = SimpleNamespace(
        module=_Projection(),
        name="q_proj",
        adapter="default",
        donor_a=t.tensor([[0.0, 1.0]]),
        donor_b=t.tensor([[4.0], [5.0]]),
        scaling=0.5,
    )

    donor, recipient = _effective_projection_pair(projection)

    assert t.equal(donor, t.tensor([[1.0, 4.0], [3.0, 6.5]]))
    assert t.equal(recipient, t.tensor([[2.0, 2.0], [4.5, 4.0]]))


def test_weight_alignment_schedule_is_complete_coarse_to_fine_and_seeded() -> None:
    steps = (0, 96, 128, 192, 1_500)

    ordered = _seeded_weight_alignment_order(steps, 20260715)

    assert ordered == _seeded_weight_alignment_order(steps, 20260715)
    assert set(ordered) == {
        (left, right) for index, left in enumerate(steps) for right in steps[index + 1 :]
    }
    tiers = [_weight_alignment_priority_tier(pair) for pair in ordered]
    assert tiers == sorted(tiers)
    assert ordered[0] == (0, 1_500)
    assert all(96 in pair for pair, tier in zip(ordered, tiers, strict=True) if tier == 1)
    assert all(
        (0 in pair or 1_500 in pair) and 96 not in pair
        for pair, tier in zip(ordered, tiers, strict=True)
        if tier == 2
    )
    assert all(
        not ({0, 96, 1_500} & set(pair))
        for pair, tier in zip(ordered, tiers, strict=True)
        if tier == 3
    )
