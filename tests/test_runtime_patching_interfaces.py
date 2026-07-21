from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch as t

from oocr_training_dynamics import runtime_patching
from oocr_training_dynamics.contracts import (
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_patching import (
    _capture_lora_layer_state,
    _copy_lora_layer_state,
    _patch_output_path,
    _replace_hidden_input,
    _replace_hidden_positions,
    _replace_lora_output_at_positions,
    _resolve_patch_targets,
    _token_lora_projections,
    run_temporal_patching_matrix,
)


class _FakeBlock(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = t.nn.Identity()
        self.mlp = t.nn.Identity()


def test_every_interface_resolves_one_concrete_target_per_layer() -> None:
    blocks = (_FakeBlock(), _FakeBlock())

    for interface in PatchingInterface:
        if interface.patches_weights:
            with pytest.raises(ValueError, match="parameters"):
                _resolve_patch_targets(blocks, interface)
            continue
        targets = _resolve_patch_targets(blocks, interface)
        assert len(targets) == len(blocks)
        if interface is PatchingInterface.RESID_POST:
            assert [target.module for target in targets] == list(blocks)
        elif interface in {
            PatchingInterface.ATTENTION_INPUT,
            PatchingInterface.ATTENTION_OUTPUT,
        }:
            assert [target.module for target in targets] == [block.self_attn for block in blocks]
        else:
            assert [target.module for target in targets] == [block.mlp for block in blocks]
        assert all(
            target.capture_input
            is (interface in {PatchingInterface.ATTENTION_INPUT, PatchingInterface.MLP_INPUT})
            for target in targets
        )


class _FakeLoraProjection(t.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.lora_A = t.nn.ModuleDict({"default": t.nn.Linear(3, 2, bias=False)})
        self.lora_B = t.nn.ModuleDict({"default": t.nn.Linear(2, 3, bias=False)})
        self.lora_dropout = t.nn.ModuleDict({"default": t.nn.Identity()})
        self.scaling = {"default": 1.0}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.use_dora = {"default": False}
        self.base_layer = t.nn.Linear(3, 3, bias=False)
        lora_a = cast(t.nn.Linear, self.lora_A["default"])
        lora_b = cast(t.nn.Linear, self.lora_B["default"])
        with t.no_grad():
            lora_a.weight.fill_(value)
            lora_b.weight.fill_(-value)
            self.base_layer.weight.copy_(t.eye(3))

    def forward(self, hidden: t.Tensor) -> t.Tensor:
        lora_a = cast(t.nn.Linear, self.lora_A["default"])
        lora_b = cast(t.nn.Linear, self.lora_B["default"])
        return self.base_layer(hidden) + lora_b(lora_a(hidden)) * self.scaling["default"]


class _FakeLoraBlock(t.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.q_proj = _FakeLoraProjection(value)
        self.up_proj = _FakeLoraProjection(value + 1.0)


class _FakeCompleteLoraBlock(t.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        for offset, name in enumerate(runtime_patching.LORA_TARGET_MODULES):
            setattr(self, name, _FakeLoraProjection(value + offset))


def test_block_weight_state_replacement_is_exact_and_reversible() -> None:
    donor = _FakeLoraBlock(7.0)
    recipient = _FakeLoraBlock(2.0)
    donor_state = _capture_lora_layer_state(donor)
    recipient_state = _capture_lora_layer_state(recipient)

    _copy_lora_layer_state(recipient, donor_state)
    replaced = _capture_lora_layer_state(recipient)
    assert replaced.keys() == donor_state.keys()
    assert all(t.equal(replaced[name], donor_state[name]) for name in replaced)

    _copy_lora_layer_state(recipient, recipient_state)
    restored = _capture_lora_layer_state(recipient)
    assert all(t.equal(restored[name], recipient_state[name]) for name in restored)


def test_token_local_weight_replacement_changes_only_selected_batch_tokens() -> None:
    donor = _FakeCompleteLoraBlock(4.0)
    recipient = _FakeCompleteLoraBlock(1.0)
    donor_state = _capture_lora_layer_state(donor)
    projections = _token_lora_projections(recipient, donor_state)
    q_projection = next(projection for projection in projections if projection.name == "q_proj")
    hidden = t.arange(2 * 4 * 3, dtype=t.float32).reshape(2, 4, 3) / 10
    recipient_q = cast(_FakeLoraProjection, recipient.q_proj)
    donor_q = cast(_FakeLoraProjection, donor.q_proj)
    recipient_output = recipient_q(hidden)
    donor_output = donor_q(hidden)

    replaced = _replace_lora_output_at_positions(
        q_projection,
        (hidden,),
        recipient_output,
        (1, 3),
    )

    assert t.allclose(replaced[0, 1], donor_output[0, 1])
    assert t.allclose(replaced[1, 3], donor_output[1, 3])
    assert t.equal(replaced[0, 0], recipient_output[0, 0])
    assert t.equal(replaced[1, 2], recipient_output[1, 2])
    assert t.equal(
        _capture_lora_layer_state(recipient)["q_proj.lora_A.default.weight"], t.full((2, 3), 1.0)
    )


def test_input_and_output_replacement_change_only_selected_batch_positions() -> None:
    hidden = t.arange(2 * 4 * 3, dtype=t.float32).reshape(2, 4, 3)
    replacements = t.tensor([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])
    positions = (1, 3)

    replaced_output, auxiliary = _replace_hidden_positions(
        (hidden, "attention weights"),
        replacements,
        positions,
    )
    assert auxiliary == "attention weights"
    assert t.equal(replaced_output[0, 1], replacements[0])
    assert t.equal(replaced_output[1, 3], replacements[1])
    assert t.equal(replaced_output[0, 0], hidden[0, 0])

    positional_args, positional_kwargs = _replace_hidden_input(
        (hidden, "other"),
        {"mask": None},
        replacements,
        positions,
    )
    assert positional_args[1] == "other"
    assert positional_kwargs == {"mask": None}
    assert t.equal(positional_args[0][0, 1], replacements[0])

    keyword_args, keyword_kwargs = _replace_hidden_input(
        (),
        {"hidden_states": hidden, "mask": None},
        replacements,
        positions,
    )
    assert keyword_args == ()
    assert keyword_kwargs["mask"] is None
    assert t.equal(keyword_kwargs["hidden_states"][1, 3], replacements[1])


def test_interface_artifact_paths_preserve_legacy_residual_and_separate_branches() -> None:
    root = Path("/experiment")
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    residual = PatchingPlan(PatchingMode.ACROSS_SAMPLE, 64, (64,))
    attention = PatchingPlan(
        PatchingMode.ACROSS_SAMPLE,
        64,
        (64,),
        interface=PatchingInterface.ATTENTION_OUTPUT,
    )
    weights = PatchingPlan(
        PatchingMode.ACROSS_TIME,
        64,
        (0,),
        interface=PatchingInterface.BLOCK_WEIGHTS,
    )
    token_weights = PatchingPlan(
        PatchingMode.ACROSS_TIME,
        64,
        (0,),
        interface=PatchingInterface.TOKEN_WEIGHTS,
    )

    residual_path = _patch_output_path(root, run, residual, 64)
    attention_path = _patch_output_path(root, run, attention, 64)
    weight_path = _patch_output_path(root, run, weights, 0)
    token_weight_path = _patch_output_path(root, run, token_weights, 0)
    assert "/patching/sequence_end/across_sample/" in str(residual_path)
    assert "/patching/sequence_end/attention_output/across_sample/" in str(attention_path)
    assert "/patching/layer_only/block_weights/across_time/" in str(weight_path)
    assert "/patching/sequence_end/token_weights/across_time/" in str(token_weight_path)
    assert len({residual_path, attention_path, weight_path, token_weight_path}) == 4


def test_temporal_matrix_reuses_each_missing_source_and_recipient_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    load_steps: list[int] = []
    released: list[int] = []
    patched: list[tuple[int, int, PatchingMode]] = []
    written: list[tuple[int, int, PatchingMode]] = []

    def output_path(
        _root: Path,
        _run: RunKey,
        plan: PatchingPlan,
        donor_step: int,
    ) -> Path:
        return tmp_path / f"{plan.mode.value}-{plan.recipient_step}-{donor_step}.json"

    existing = output_path(
        tmp_path,
        run,
        PatchingPlan(PatchingMode.LATER_CHECKPOINT, 0, (1,)),
        1,
    )
    existing.write_text("already measured")

    monkeypatch.setattr(runtime_patching, "CHECKPOINT_STEPS", (0, 1, 2))
    monkeypatch.setattr(runtime_patching.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_patching, "get_model_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_patching, "load_processor", lambda _spec: object())
    monkeypatch.setattr(
        runtime_patching,
        "_selected_records",
        lambda _seed: (SimpleNamespace(record_id="probe"),),
    )
    monkeypatch.setattr(runtime_patching, "_patch_output_path", output_path)

    def load_model(_root: Path, _run: RunKey, _spec: object, step: int) -> int:
        load_steps.append(step)
        return step

    monkeypatch.setattr(runtime_patching, "_load_checkpoint_model", load_model)
    monkeypatch.setattr(runtime_patching, "resolve_decoder_blocks", lambda *_args: ())
    monkeypatch.setattr(runtime_patching, "_resolve_patch_targets", lambda *_args: ())
    monkeypatch.setattr(
        runtime_patching,
        "_capture_clean_source_bank",
        lambda model, *_args: {"probe": model},
    )
    monkeypatch.setattr(runtime_patching, "_release_model", released.append)

    def patch_source(
        model: int,
        _targets: object,
        _processor: object,
        _records: object,
        mode: PatchingMode,
        source_by_record: dict[str, int],
    ) -> list[dict[str, object]]:
        patched.append((model, source_by_record["probe"], mode))
        return []

    monkeypatch.setattr(runtime_patching, "_patch_temporal_source_bank", patch_source)

    def write_artifact(
        _root: Path,
        _run: RunKey,
        _spec: object,
        plan: PatchingPlan,
        donor_step: int,
        _serialized: list[dict[str, object]],
    ) -> None:
        written.append((plan.recipient_step, donor_step, plan.mode))

    monkeypatch.setattr(runtime_patching, "_write_temporal_artifact", write_artifact)

    run_temporal_patching_matrix(
        tmp_path,
        run,
        (0, 1),
        (PatchingMode.ACROSS_TIME, PatchingMode.LATER_CHECKPOINT),
        PatchingInterface.RESID_POST,
    )

    assert load_steps == [0, 2, 0, 1]
    assert released == load_steps
    assert patched == [
        (0, 2, PatchingMode.LATER_CHECKPOINT),
        (1, 0, PatchingMode.ACROSS_TIME),
        (1, 2, PatchingMode.LATER_CHECKPOINT),
    ]
    assert written == patched

    load_steps.clear()
    released.clear()
    patched.clear()
    written.clear()
    seen_seeds: list[int] = []

    def seeded_random(seed: int) -> SimpleNamespace:
        seen_seeds.append(seed)
        return SimpleNamespace(shuffle=lambda values: values.reverse())

    monkeypatch.setattr(runtime_patching.random, "Random", seeded_random)
    run_temporal_patching_matrix(
        tmp_path,
        run,
        (0, 1),
        (PatchingMode.ACROSS_TIME, PatchingMode.LATER_CHECKPOINT),
        PatchingInterface.RESID_POST,
        shuffle_seed=7,
    )

    assert seen_seeds == [7]
    assert load_steps == [0, 2, 1, 0, 1]
    assert released == load_steps
    assert patched == [
        (1, 0, PatchingMode.ACROSS_TIME),
        (0, 2, PatchingMode.LATER_CHECKPOINT),
        (1, 2, PatchingMode.LATER_CHECKPOINT),
    ]
    assert written == patched


def test_temporal_matrix_dispatches_block_weights_without_activation_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    dispatched: list[tuple[list[tuple[int, int, PatchingMode]], PatchingInterface]] = []

    monkeypatch.setattr(runtime_patching, "CHECKPOINT_STEPS", (0, 1))
    monkeypatch.setattr(runtime_patching.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_patching, "get_model_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_patching, "load_processor", lambda _spec: object())
    monkeypatch.setattr(runtime_patching, "_selected_records", lambda _seed: ())
    monkeypatch.setattr(
        runtime_patching,
        "_patch_output_path",
        lambda *_args: tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        runtime_patching,
        "_run_weight_temporal_pairs",
        lambda _root, _run, _spec, _processor, _records, pairs, interface: dispatched.append(
            (pairs, interface)
        ),
    )
    monkeypatch.setattr(
        runtime_patching,
        "_capture_clean_source_bank",
        lambda *_args: pytest.fail("weight patching must not capture activations"),
    )

    for interface in (
        PatchingInterface.BLOCK_WEIGHTS,
        PatchingInterface.TOKEN_WEIGHTS,
    ):
        run_temporal_patching_matrix(
            tmp_path,
            run,
            (1,),
            (PatchingMode.ACROSS_TIME,),
            interface,
        )

    assert dispatched == [
        ([(1, 0, PatchingMode.ACROSS_TIME)], PatchingInterface.BLOCK_WEIGHTS),
        ([(1, 0, PatchingMode.ACROSS_TIME)], PatchingInterface.TOKEN_WEIGHTS),
    ]


def test_seeded_temporal_order_prioritizes_requested_steps_and_resumes_stably() -> None:
    steps = (0, 1, 2, 96, 256, 1_500)
    scheduled = [
        (recipient, donor, runtime_patching._temporal_mode(recipient, donor))
        for recipient in steps
        for donor in steps
        if recipient != donor
    ]

    ordered = runtime_patching._seeded_priority_temporal_order(scheduled, 20260715)
    repeated = runtime_patching._seeded_priority_temporal_order(scheduled, 20260715)
    tier_counts = [
        sum(runtime_patching._temporal_priority_tier(pair) == tier for pair in scheduled)
        for tier in range(3)
    ]
    boundaries = [0]
    for count in tier_counts:
        boundaries.append(boundaries[-1] + count)

    assert ordered == repeated
    assert set(ordered) == set(scheduled)
    assert tier_counts == [18, 6, 6]
    for tier, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        assert all(
            runtime_patching._temporal_priority_tier(pair) == tier for pair in ordered[start:stop]
        )

    completed = {ordered[1], ordered[boundaries[1] + 1], ordered[boundaries[2] + 1]}
    resumed = [pair for pair in ordered if pair not in completed]
    assert resumed == [pair for pair in repeated if pair not in completed]
