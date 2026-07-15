from __future__ import annotations

from pathlib import Path

import torch as t

from oocr_training_dynamics.contracts import (
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_patching import (
    _patch_output_path,
    _replace_hidden_input,
    _replace_hidden_positions,
    _resolve_patch_targets,
)


class _FakeBlock(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = t.nn.Identity()
        self.mlp = t.nn.Identity()


def test_every_interface_resolves_one_concrete_target_per_layer() -> None:
    blocks = (_FakeBlock(), _FakeBlock())

    for interface in PatchingInterface:
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

    residual_path = _patch_output_path(root, run, residual, 64)
    attention_path = _patch_output_path(root, run, attention, 64)
    assert "/patching/across_sample/" in str(residual_path)
    assert "/patching/attention_output/across_sample/" in str(attention_path)
    assert residual_path != attention_path
