from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from oocr_training_dynamics.contracts import (
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.patching import TokenPositionPair
from oocr_training_dynamics.representation_alignment import representation_alignment_path
from oocr_training_dynamics.runtime_representation_alignment import (
    _alignment_interleaved_priority_tier,
    _capture_alignment_interfaces,
    _representation_alignment_grid,
    _scheduled_alignment_pairs,
    _seeded_interleaved_alignment_order,
)


class _Scale(t.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states: t.Tensor) -> t.Tensor:
        return hidden_states * self.factor


class _Block(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Scale(2.0)
        self.mlp = _Scale(4.0)

    def forward(self, hidden_states: t.Tensor) -> t.Tensor:
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class _Model(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = t.nn.ModuleList((_Block(), _Block()))

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        hidden = t.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(logits=hidden)


def test_representation_alignment_path_is_separate_and_activation_only(tmp_path: Path) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)

    path = representation_alignment_path(
        tmp_path,
        run,
        PatchingInterface.MLP_INPUT,
        PatchingMode.UNRELATED_QUESTION,
        96,
        1_500,
    )

    assert (
        path.relative_to(tmp_path)
        .as_posix()
        .endswith(
            "representation_alignment/sequence_end/mlp_input/unrelated_question/"
            "recipient_step_000096/donor_step_001500.json"
        )
    )
    with pytest.raises(ValueError, match="activation interfaces"):
        representation_alignment_path(
            tmp_path,
            run,
            PatchingInterface.TOKEN_WEIGHTS,
            PatchingMode.ACROSS_TIME,
            96,
            0,
        )


def test_alignment_grid_computes_exact_cosine_l2_and_norms() -> None:
    source = (
        t.tensor([[1.0, 0.0], [1.0, 0.0]]),
        t.tensor([[3.0, 4.0], [1.0, 1.0]]),
    )
    recipient = (
        t.tensor([[1.0, 0.0], [0.0, 1.0]]),
        t.tensor([[0.0, 5.0], [-1.0, -1.0]]),
    )
    positions = (
        TokenPositionPair(0, 0, 0),
        TokenPositionPair(1, 1, 1),
    )

    grid = _representation_alignment_grid(source, recipient, positions)

    assert grid.cosine_similarity[0][0] == pytest.approx(1.0)
    assert grid.l2_distance[0][0] == pytest.approx(0.0)
    assert grid.source_norm[0][1] == pytest.approx(5.0)
    assert grid.recipient_norm[0][1] == pytest.approx(5.0)
    assert grid.cosine_similarity[1][0] == pytest.approx(0.0)
    assert grid.l2_distance[1][0] == pytest.approx(2**0.5)
    assert grid.cosine_similarity[1][1] == pytest.approx(-1.0)


def test_alignment_grid_rejects_zero_norm_vectors() -> None:
    with pytest.raises(RuntimeError, match="non-zero"):
        _representation_alignment_grid(
            (t.zeros((1, 2)),),
            (t.ones((1, 2)),),
            (TokenPositionPair(0, 0, 0),),
        )


def test_multi_interface_capture_uses_one_forward_and_exact_boundaries() -> None:
    model = _Model()
    interfaces = (
        PatchingInterface.RESID_POST,
        PatchingInterface.ATTENTION_INPUT,
        PatchingInterface.ATTENTION_OUTPUT,
        PatchingInterface.MLP_INPUT,
        PatchingInterface.MLP_OUTPUT,
    )
    input_ids = t.tensor([[1, 2]])

    captured = _capture_alignment_interfaces(
        model,
        tuple(model.blocks),
        interfaces,
        input_ids,
        t.ones_like(input_ids),
    )

    initial = t.tensor([[1.0, 2.0], [2.0, 3.0]])
    assert t.equal(captured[PatchingInterface.ATTENTION_INPUT][0], initial)
    assert t.equal(captured[PatchingInterface.ATTENTION_OUTPUT][0], initial * 2.0)
    assert t.equal(captured[PatchingInterface.MLP_INPUT][0], initial * 3.0)
    assert t.equal(captured[PatchingInterface.MLP_OUTPUT][0], initial * 12.0)
    assert t.equal(captured[PatchingInterface.RESID_POST][0], initial * 15.0)
    assert t.equal(captured[PatchingInterface.ATTENTION_INPUT][1], initial * 15.0)

    selected = _capture_alignment_interfaces(
        model,
        tuple(model.blocks),
        (PatchingInterface.RESID_POST,),
        input_ids,
        t.ones_like(input_ids),
        token_indices=(1,),
    )
    assert selected[PatchingInterface.RESID_POST][0].shape == (1, 2)
    assert t.equal(selected[PatchingInterface.RESID_POST][0], (initial * 15.0)[1:2])


def test_alignment_scheduler_respects_mode_checkpoint_contracts() -> None:
    pairs = _scheduled_alignment_pairs(
        (0, 96),
        (0, 1_500),
        (
            PatchingMode.ACROSS_SAMPLE,
            PatchingMode.REVERSE_ACROSS_SAMPLE,
            PatchingMode.UNRELATED_QUESTION,
            PatchingMode.ACROSS_TIME,
            PatchingMode.LATER_CHECKPOINT,
        ),
    )

    assert (0, 0, PatchingMode.ACROSS_SAMPLE) in pairs
    assert (96, 96, PatchingMode.ACROSS_SAMPLE) in pairs
    assert (0, 0, PatchingMode.REVERSE_ACROSS_SAMPLE) in pairs
    assert (96, 96, PatchingMode.REVERSE_ACROSS_SAMPLE) in pairs
    assert (0, 1_500, PatchingMode.UNRELATED_QUESTION) in pairs
    assert (96, 0, PatchingMode.ACROSS_TIME) in pairs
    assert (0, 1_500, PatchingMode.LATER_CHECKPOINT) in pairs
    assert all(
        recipient == donor
        for recipient, donor, mode in pairs
        if mode in {PatchingMode.ACROSS_SAMPLE, PatchingMode.REVERSE_ACROSS_SAMPLE}
    )


def test_interleaved_alignment_order_groups_coarse_tiers_by_interface() -> None:
    steps = (0, 96, 128, 192, 1_500)
    interfaces = (
        PatchingInterface.RESID_POST,
        PatchingInterface.MLP_OUTPUT,
        PatchingInterface.ATTENTION_OUTPUT,
    )
    pairs = _scheduled_alignment_pairs(
        steps,
        steps,
        (PatchingMode.ACROSS_TIME, PatchingMode.LATER_CHECKPOINT),
    )

    tasks = _seeded_interleaved_alignment_order(pairs, interfaces, 20260715)

    assert tasks == _seeded_interleaved_alignment_order(pairs, interfaces, 20260715)
    assert {(recipient, donor, mode, interface) for recipient, donor, mode, interface in tasks} == {
        (*pair, interface) for pair in pairs for interface in interfaces
    }
    tiers = [
        _alignment_interleaved_priority_tier((recipient, donor, mode))
        for recipient, donor, mode, _interface in tasks
    ]
    assert tiers == sorted(tiers)
    for tier in range(3):
        tier_interfaces = [
            task[3] for task_tier, task in zip(tiers, tasks, strict=True) if task_tier == tier
        ]
        block_size = len(tier_interfaces) // len(interfaces)
        assert tier_interfaces == [interface for interface in interfaces for _ in range(block_size)]
    remainder = [task for task, tier in zip(tasks, tiers, strict=True) if tier == 3]
    assert remainder
    assert all(
        recipient not in {0, 96, 1_500} and donor not in {0, 96, 1_500}
        for recipient, donor, _mode, _interface in remainder
    )
