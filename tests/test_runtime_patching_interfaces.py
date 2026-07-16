from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    _patch_output_path,
    _replace_hidden_input,
    _replace_hidden_positions,
    _resolve_patch_targets,
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
    assert "/patching/sequence_end/across_sample/" in str(residual_path)
    assert "/patching/sequence_end/attention_output/across_sample/" in str(attention_path)
    assert residual_path != attention_path


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
    assert load_steps == [0, 2, 1, 0]
    assert released == load_steps
    assert patched == [
        (1, 2, PatchingMode.LATER_CHECKPOINT),
        (1, 0, PatchingMode.ACROSS_TIME),
        (0, 2, PatchingMode.LATER_CHECKPOINT),
    ]
    assert written == patched
