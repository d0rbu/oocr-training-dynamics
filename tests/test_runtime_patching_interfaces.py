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
    TokenWeightRuntime,
    TrainingCondition,
)
from oocr_training_dynamics.models import ModelSpec
from oocr_training_dynamics.patching import PatchingPlan, TokenPositionPair
from oocr_training_dynamics.runtime_patching import (
    _answer_label_logit_lens,
    _capture_decoder_inputs,
    _capture_lora_layer_state,
    _copy_lora_layer_state,
    _forward_probabilities,
    _forward_probabilities_last_token,
    _patch_grid,
    _patch_output_path,
    _replace_hidden_input,
    _replace_hidden_positions,
    _replace_lora_output_at_positions,
    _resolve_patch_targets,
    _skip_unchanged_decoder_prefix,
    _token_lora_projections,
    _token_weight_probability_forward,
    run_prompt_counterfactual_patching_matrix,
    run_temporal_patching_matrix,
)


class _LensDecoder(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = t.nn.LayerNorm(3)


class _LensModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _LensDecoder()
        self.lm_head = t.nn.Linear(3, 7, bias=True)

    def get_output_embeddings(self) -> t.nn.Module:
        return self.lm_head


def test_answer_label_logit_lens_matches_selected_unembedding_rows() -> None:
    model = _LensModel()
    with t.no_grad():
        model.lm_head.weight.copy_(t.arange(21, dtype=t.float32).reshape(7, 3) / 10)
        model.lm_head.bias.copy_(t.arange(7, dtype=t.float32) / 20)
    activations = (
        t.tensor([[1.0, 2.0, 4.0], [3.0, -2.0, 0.5]]),
        t.tensor([[2.0, 0.0, -1.0], [4.0, 1.0, 3.0]]),
    )
    candidate_ids = t.tensor([0, 2, 3, 5, 6])

    actual = _answer_label_logit_lens(model, activations, (1,), candidate_ids)
    stacked = t.stack((activations[0][1], activations[1][1]))
    normalized = model.model.norm(stacked)
    expected = t.softmax(
        t.nn.functional.linear(
            normalized,
            model.lm_head.weight.index_select(0, candidate_ids),
            model.lm_head.bias.index_select(0, candidate_ids),
        ),
        dim=-1,
    )

    assert t.allclose(t.tensor(actual[0]), expected)
    assert all(sum(distribution) == pytest.approx(1.0) for distribution in actual[0])


class _FakeBlock(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = t.nn.Identity()
        self.mlp = t.nn.Identity()


class _FakeCausalLM(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_requests: list[int] = []

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        assert not use_cache
        assert return_dict
        self.logit_requests.append(logits_to_keep)
        vocab = t.arange(7, dtype=t.float32).view(1, 1, 7)
        logits = input_ids.to(dtype=t.float32).unsqueeze(-1) * 0.25 + vocab
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_last_token_probability_kernel_is_exact_on_the_same_logits() -> None:
    model = _FakeCausalLM()
    input_ids = t.tensor([[3, 5, 2], [1, 4, 6]])
    attention_mask = t.ones_like(input_ids)
    candidate_ids = t.tensor([0, 2, 3, 5, 6])

    reference = _forward_probabilities(model, input_ids, attention_mask, candidate_ids)
    optimized = _forward_probabilities_last_token(
        model,
        input_ids,
        attention_mask,
        candidate_ids,
    )

    assert t.equal(optimized, reference)
    assert model.logit_requests == [0, 1]
    assert _token_weight_probability_forward(TokenWeightRuntime.REFERENCE) is (
        _forward_probabilities
    )
    assert _token_weight_probability_forward(TokenWeightRuntime.OPTIMIZED) is _forward_probabilities


class _CountingDecoderBlock(t.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.calls = 0

    def forward(self, hidden_states: t.Tensor, **_kwargs: object) -> t.Tensor:
        self.calls += 1
        return hidden_states + self.increment


class _FakeDecoderLM(t.nn.Module):
    def __init__(self, blocks: tuple[_CountingDecoderBlock, ...]) -> None:
        super().__init__()
        self.blocks = t.nn.ModuleList(blocks)

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        assert not use_cache
        assert return_dict
        hidden = input_ids.to(dtype=t.float32).unsqueeze(-1).expand(-1, -1, 3)
        for block in self.blocks:
            hidden = block(hidden_states=hidden)
        logits = t.cat((hidden, hidden + 1.0, hidden[:, :, :1]), dim=-1)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_decoder_prefix_cache_preserves_exact_inputs_and_skips_only_upstream_blocks() -> None:
    blocks = tuple(_CountingDecoderBlock(value) for value in (1.0, 2.0, 4.0))
    model = _FakeDecoderLM(blocks)
    input_ids = t.tensor([[1, 3]])
    attention_mask = t.ones_like(input_ids)
    candidate_ids = t.tensor([0, 1, 2, 3, 4])

    cached = _capture_decoder_inputs(
        model,
        blocks,
        input_ids,
        attention_mask,
        candidate_ids,
        batch_size=2,
    )
    expected_input = input_ids.expand(2, -1).to(dtype=t.float32).unsqueeze(-1).expand(-1, -1, 3)
    assert t.equal(cached[0], expected_input)
    assert t.equal(cached[1], expected_input + 1.0)
    assert t.equal(cached[2], expected_input + 3.0)

    calls_before = tuple(block.calls for block in blocks)
    with _skip_unchanged_decoder_prefix(blocks, 2, cached):
        hidden = expected_input
        for block in blocks:
            hidden = block(hidden_states=hidden)
    assert t.equal(hidden, expected_input + 7.0)
    assert tuple(block.calls for block in blocks) == (
        calls_before[0],
        calls_before[1],
        calls_before[2] + 1,
    )
    assert all("forward" not in block.__dict__ for block in blocks)


def test_decoder_prefix_forwards_are_restored_after_an_error() -> None:
    blocks = tuple(_CountingDecoderBlock(value) for value in (1.0, 2.0))
    hidden = t.zeros((1, 2, 3))
    cached = (hidden, hidden + 1.0)

    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        _skip_unchanged_decoder_prefix(blocks, 1, cached),
    ):
        raise RuntimeError("synthetic failure")

    assert all("forward" not in block.__dict__ for block in blocks)
    assert t.equal(blocks[0](hidden_states=hidden), hidden + 1.0)


def test_patch_grid_tracks_clean_and_source_answer_labels_from_one_forward() -> None:
    block = _CountingDecoderBlock(1.0)
    model = _FakeDecoderLM((block,))
    targets = _resolve_patch_targets((block,), PatchingInterface.RESID_POST)
    input_ids = t.tensor([[1, 3]])
    attention_mask = t.ones_like(input_ids)
    candidate_ids = t.tensor([0, 1, 2, 3, 4])
    source_activations = (t.tensor([[2.0, 1.0, 0.0], [6.0, 3.0, -2.0]]),)

    clean_grid, source_grid = _patch_grid(
        model,
        targets,
        input_ids,
        attention_mask,
        candidate_ids,
        source_activations,
        (TokenPositionPair(0, 1, 1),),
        correct_choice_index=0,
        source_correct_choice_index=1,
    )

    assert source_grid is not None
    assert clean_grid[0][0] != source_grid[0][0]
    assert 0.0 <= clean_grid[0][0] <= 1.0
    assert 0.0 <= source_grid[0][0] <= 1.0


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


def test_prompt_checkpoint_matrix_includes_diagonals_and_skips_existing_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    modes = (PatchingMode.CYCLIC_CHOICES, PatchingMode.UNRELATED_QUESTION)
    dispatched: list[tuple[int, int, PatchingMode]] = []

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
        PatchingPlan(PatchingMode.CYCLIC_CHOICES, 1_500, (1_500,)),
        1_500,
    )
    existing.write_text("already measured")
    monkeypatch.setattr(runtime_patching, "CHECKPOINT_STEPS", (0, 1_500))
    monkeypatch.setattr(runtime_patching.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_patching, "_patch_output_path", output_path)
    monkeypatch.setattr(runtime_patching, "get_model_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_patching, "load_processor", lambda _spec: object())
    monkeypatch.setattr(runtime_patching, "_selected_records", lambda _seed: ())

    def run_pair(
        _root: Path,
        _run: RunKey,
        _spec: object,
        _processor: object,
        _records: object,
        plan: PatchingPlan,
        donor_step: int,
    ) -> None:
        dispatched.append((plan.recipient_step, donor_step, plan.mode))

    monkeypatch.setattr(runtime_patching, "_run_prompt_counterfactual_pair", run_pair)
    run_prompt_counterfactual_patching_matrix(
        tmp_path,
        run,
        (0, 1_500),
        (0, 1_500),
        modes,
        PatchingInterface.RESID_POST,
    )

    expected = [
        (recipient, donor, mode)
        for recipient in (0, 1_500)
        for donor in (0, 1_500)
        for mode in modes
        if not (recipient == 1_500 and donor == 1_500 and mode is PatchingMode.CYCLIC_CHOICES)
    ]
    assert dispatched == expected
    assert (0, 0, PatchingMode.CYCLIC_CHOICES) in dispatched
    assert (1_500, 1_500, PatchingMode.UNRELATED_QUESTION) in dispatched


def test_prompt_checkpoint_pair_uses_donor_capture_and_recipient_readout_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    plan = PatchingPlan(PatchingMode.DERANGED_CHOICES, 64, (0,))
    loaded: list[int] = []
    released: list[int] = []
    captured_with: list[int] = []
    patched_with: list[tuple[int, int, int]] = []
    payloads: list[dict[str, object]] = []

    def load_model(_root: Path, _run: RunKey, _spec: object, step: int) -> int:
        loaded.append(step)
        return step

    monkeypatch.setattr(runtime_patching, "_load_checkpoint_model", load_model)
    monkeypatch.setattr(runtime_patching, "resolve_decoder_blocks", lambda *_args: ())
    monkeypatch.setattr(runtime_patching, "_resolve_patch_targets", lambda *_args: ())
    monkeypatch.setattr(runtime_patching, "_release_model", released.append)

    def capture_source(model: int, *_args: object) -> dict[str, object]:
        captured_with.append(model)
        return {"source": object()}

    monkeypatch.setattr(
        runtime_patching,
        "_capture_prompt_counterfactual_source_bank",
        capture_source,
    )

    def patch_source(
        model: int,
        _targets: object,
        _residual_targets: object,
        _processor: object,
        _records: object,
        _mode: PatchingMode,
        _source: object,
        *,
        source_step: int,
        recipient_step: int,
    ) -> tuple[list[dict[str, object]], str]:
        patched_with.append((model, source_step, recipient_step))
        return [], "deranged_choice_source_into_clean_recipient"

    monkeypatch.setattr(
        runtime_patching,
        "_patch_prompt_counterfactual_source_bank",
        patch_source,
    )
    monkeypatch.setattr(runtime_patching, "_patch_output_path", lambda *_args: tmp_path / "out")
    monkeypatch.setattr(
        runtime_patching,
        "write_json",
        lambda _path, payload: payloads.append(payload),
    )

    runtime_patching._run_prompt_counterfactual_pair(
        tmp_path,
        run,
        cast(ModelSpec, object()),
        object(),
        (),
        plan,
        0,
    )

    assert loaded == [0, 64]
    assert released == [0, 64]
    assert captured_with == [0]
    assert patched_with == [(64, 0, 64)]
    assert payloads[0]["checkpoint_relation"] == "cross_checkpoint"
    assert payloads[0]["donor_step"] == 0


def test_prompt_checkpoint_lenses_use_each_side_own_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SimpleNamespace(record_id="probe")
    spec = runtime_patching.PromptCounterfactualSpec(
        source_messages=(),
        source_function_id="source",
        source_correct_choice_index=1,
        source_choice_function_ids=None,
        source_choice_texts=None,
        source_question_id=None,
        source_question=None,
        patch_direction="source_into_recipient",
        stops_at_first_difference=True,
    )
    view = runtime_patching.PromptPatchView(
        input_ids=t.tensor([[1, 2]]),
        attention_mask=t.ones((1, 2), dtype=t.int64),
        anchor_index=1,
        stop_index=1,
        rendered_prompt="prompt",
        token_ids=(1, 2),
        token_labels=("one", "two"),
    )
    activations = (t.zeros((2, 3)),)
    donor_lens = (((0.1, 0.2, 0.3, 0.2, 0.2),),)
    recipient_lens = (((0.2, 0.1, 0.2, 0.3, 0.2),),)
    lens_models: list[object] = []

    monkeypatch.setattr(runtime_patching, "_prompt_counterfactual_spec", lambda *_args: spec)
    monkeypatch.setattr(
        runtime_patching,
        "_prompt_counterfactual_views",
        lambda *_args: (view, view),
    )
    monkeypatch.setattr(
        runtime_patching,
        "_counterfactual_candidate_ids",
        lambda *_args: (t.arange(5), t.arange(5)),
    )
    monkeypatch.setattr(
        runtime_patching,
        "_capture",
        lambda *_args: (activations, t.full((5,), 0.2)),
    )

    donor_model = object()
    recipient_model = object()

    def lens(model: object, *_args: object) -> object:
        lens_models.append(model)
        return donor_lens if model is donor_model else recipient_lens

    monkeypatch.setattr(runtime_patching, "_answer_label_logit_lens", lens)
    source_bank = runtime_patching._capture_prompt_counterfactual_source_bank(
        cast(t.nn.Module, donor_model),
        (),
        (),
        object(),
        cast(tuple[runtime_patching.ReflectionRecord, ...], (record,)),
        PatchingMode.CYCLIC_CHOICES,
        PatchingInterface.RESID_POST,
    )
    precomputed: list[dict[str, object]] = []

    def patch_record(*_args: object, **kwargs: object) -> dict[str, object]:
        precomputed.append(cast(dict[str, object], kwargs["precomputed_answer_logit_lens"]))
        return {"ok": True}

    monkeypatch.setattr(runtime_patching, "_patch_record", patch_record)
    serialized, direction = runtime_patching._patch_prompt_counterfactual_source_bank(
        cast(t.nn.Module, recipient_model),
        (),
        (),
        object(),
        cast(tuple[runtime_patching.ReflectionRecord, ...], (record,)),
        PatchingMode.CYCLIC_CHOICES,
        source_bank,
        source_step=0,
        recipient_step=64,
    )

    assert lens_models == [donor_model, recipient_model]
    assert precomputed[0]["source_probabilities"] == donor_lens
    assert precomputed[0]["recipient_probabilities"] == recipient_lens
    assert precomputed[0]["source_checkpoint_step"] == 0
    assert precomputed[0]["recipient_checkpoint_step"] == 64
    assert serialized == [{"ok": True}]
    assert direction == "source_into_recipient"


def test_temporal_matrix_dispatches_block_weights_without_activation_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    dispatched: list[
        tuple[
            list[tuple[int, int, PatchingMode]],
            PatchingInterface,
            TokenWeightRuntime,
            int,
        ]
    ] = []

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
        lambda _root, _run, _spec, _processor, _records, pairs, interface, runtime, batch_size: (
            dispatched.append((pairs, interface, runtime, batch_size))
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
        (
            [(1, 0, PatchingMode.ACROSS_TIME)],
            PatchingInterface.BLOCK_WEIGHTS,
            TokenWeightRuntime.REFERENCE,
            8,
        ),
        (
            [(1, 0, PatchingMode.ACROSS_TIME)],
            PatchingInterface.TOKEN_WEIGHTS,
            TokenWeightRuntime.REFERENCE,
            8,
        ),
    ]


def test_reference_token_weight_runtime_rejects_batch_shape_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_patching.t.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="fixed batch size of 8"):
        run_temporal_patching_matrix(
            tmp_path,
            RunKey("olmo3-7b", TrainingCondition.CORRECT),
            (64,),
            (PatchingMode.ACROSS_TIME,),
            PatchingInterface.TOKEN_WEIGHTS,
            token_weight_runtime=TokenWeightRuntime.REFERENCE,
            token_weight_patch_batch_size=16,
        )
    with pytest.raises(ValueError, match="fixed batch size of 8"):
        run_temporal_patching_matrix(
            tmp_path,
            RunKey("olmo3-7b", TrainingCondition.CORRECT),
            (64,),
            (PatchingMode.ACROSS_TIME,),
            PatchingInterface.TOKEN_WEIGHTS,
            token_weight_runtime=TokenWeightRuntime.OPTIMIZED,
            token_weight_patch_batch_size=16,
        )


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
        for tier in range(len(runtime_patching.TEMPORAL_PRIORITY_LABELS) + 1)
    ]
    boundaries = [0]
    for count in tier_counts:
        boundaries.append(boundaries[-1] + count)

    assert ordered == repeated
    assert set(ordered) == set(scheduled)
    assert tier_counts == [2, 4, 12, 6, 6]
    assert {(recipient, donor) for recipient, donor, _mode in ordered[: boundaries[1]]} == {
        (0, 1_500),
        (1_500, 0),
    }
    assert {
        (recipient, donor) for recipient, donor, _mode in ordered[boundaries[1] : boundaries[2]]
    } == {(0, 96), (96, 0), (96, 1_500), (1_500, 96)}
    for tier, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        assert all(
            runtime_patching._temporal_priority_tier(pair) == tier for pair in ordered[start:stop]
        )

    completed = {ordered[start] for start in boundaries[:-1]}
    resumed = [pair for pair in ordered if pair not in completed]
    assert resumed == [pair for pair in repeated if pair not in completed]
