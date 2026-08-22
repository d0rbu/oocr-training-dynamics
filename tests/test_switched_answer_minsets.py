from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from oocr_training_dynamics.answer_lookup import ChoiceTerminatorSite
from oocr_training_dynamics.data import ChatMessage, ReflectionRecord
from oocr_training_dynamics.fourier_circuits import SweepDensity
from oocr_training_dynamics.runtime_patching import PatchTarget, PromptPatchView
from oocr_training_dynamics.runtime_switched_answer_minsets import (
    SwitchedAnswerProbe,
    _replace_two_positions,
    evaluate_swap_masks,
    run_density_sweep,
    run_endpoint_gate,
    run_exhaustive_search,
)
from oocr_training_dynamics.switched_answer_minsets import (
    LayerSwapSite,
    SwapSubsetMetric,
    SwitchedAnswerCheckpointSpec,
    SwitchedAnswerDensityConfig,
    SwitchedAnswerMinsetConfig,
    SwitchedAnswerSearchConfig,
    SwitchedAnswerTaskSpec,
    as_layer_patch_mask,
    layer_supports,
    masks_for_layer_supports,
    sample_layer_patch_masks,
    support_is_safely_blocked,
    verified_minsets_from_metrics,
)


def _metric(layers: tuple[int, ...], probability: float) -> SwapSubsetMetric:
    return SwapSubsetMetric(
        sites=tuple(LayerSwapSite(layer) for layer in layers),
        candidate_logits=(probability, 0.0, 0.0, 0.0, 0.0),
        destination_probability=probability,
        raw_logit_diff=probability,
        destination_argmax=probability >= 0.9,
    )


def test_composite_support_masks_are_layer_only_and_canonical() -> None:
    supports = layer_supports(4, 2)
    masks = masks_for_layer_supports(supports, 4)

    assert len(supports) == 6
    assert masks.dtype is t.bool
    assert masks.shape == (6, 4)
    assert t.equal(masks.sum(dim=1), t.full((6,), 2))
    assert supports[0] == (LayerSwapSite(0), LayerSwapSite(1))
    as_layer_patch_mask(masks[0].contiguous(), 4)
    with pytest.raises(ValueError, match="configured decoder depth"):
        as_layer_patch_mask(t.zeros(3, dtype=t.bool), 4)


def test_density_sampler_handles_empty_and_full_corners_exactly() -> None:
    generator = t.Generator().manual_seed(17)
    empty = sample_layer_patch_masks(3, 4, SweepDensity.parse(0.0), generator)
    full = sample_layer_patch_masks(3, 4, SweepDensity.parse(1.0), generator)
    interior = sample_layer_patch_masks(100, 4, SweepDensity.parse(0.5), generator)

    assert not bool(empty.any())
    assert bool(full.all())
    assert 0 < int(interior.sum()) < interior.numel()


def test_exact_relative_minsets_recover_two_of_three_without_false_triple() -> None:
    metrics: dict[tuple[LayerSwapSite, ...], SwapSubsetMetric] = {}
    for bits in itertools.product((False, True), repeat=3):
        layers = tuple(index for index, enabled in enumerate(bits) if enabled)
        probability = 1.0 if len(layers) >= 2 else 0.0
        metric = _metric(layers, probability)
        metrics[metric.sites] = metric

    verified = verified_minsets_from_metrics(metrics, 1.0, 0.10, 0.80)

    assert {tuple(site.layer for site in row.sites) for row in verified} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }


def test_exact_relative_minsets_require_complete_proper_subset_evidence() -> None:
    metrics: dict[tuple[LayerSwapSite, ...], SwapSubsetMetric] = {
        (): _metric((), 0.0),
        (LayerSwapSite(0), LayerSwapSite(1)): _metric((0, 1), 1.0),
    }

    assert verified_minsets_from_metrics(metrics, 1.0, 0.10, 0.80) == ()


def test_safe_blocker_uses_only_an_observed_subset_above_global_fraction() -> None:
    metrics: dict[tuple[LayerSwapSite, ...], SwapSubsetMetric] = {
        (): _metric((), 0.0),
        (LayerSwapSite(0),): _metric((0,), 0.81),
        (LayerSwapSite(1),): _metric((1,), 0.79),
    }

    assert support_is_safely_blocked(
        (LayerSwapSite(0), LayerSwapSite(2)), metrics, 0.80
    )
    assert not support_is_safely_blocked(
        (LayerSwapSite(1), LayerSwapSite(2)), metrics, 0.80
    )


def test_paired_swap_replaces_only_the_two_declared_positions() -> None:
    hidden = t.arange(1 * 5 * 3, dtype=t.float32).reshape(1, 5, 3)
    original = hidden.clone()
    replacements = t.tensor([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])

    patched = _replace_two_positions(hidden, replacements, (2, 0))

    assert t.equal(hidden, original)
    assert t.equal(patched[0, 2], replacements[0])
    assert t.equal(patched[0, 0], replacements[1])
    assert t.equal(patched[0, 1:2], original[0, 1:2])
    assert t.equal(patched[0, 3:], original[0, 3:])
    with pytest.raises(ValueError, match="distinct"):
        _replace_two_positions(hidden, replacements, (2, 2))


class _IdentityBoundary(t.nn.Module):
    def forward(self, hidden_states: t.Tensor) -> t.Tensor:
        return hidden_states


class _SwapProbeModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boundary = _IdentityBoundary()

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert bool(t.all(attention_mask))
        assert use_cache is False and return_dict is True and logits_to_keep == 1
        hidden = t.nn.functional.one_hot(input_ids, num_classes=5).to(t.float32)
        hidden = self.boundary(hidden)
        return SimpleNamespace(logits=hidden.sum(dim=1, keepdim=True))


class _DeepSwapProbeModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boundaries = t.nn.ModuleList(_IdentityBoundary() for _ in range(32))

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert bool(t.all(attention_mask))
        assert use_cache is False and return_dict is True and logits_to_keep == 1
        hidden = t.nn.functional.one_hot(input_ids, num_classes=5).to(t.float32)
        for boundary in self.boundaries:
            hidden = boundary(hidden)
        return SimpleNamespace(logits=hidden.sum(dim=1, keepdim=True))


def _runtime_probe() -> SwitchedAnswerProbe:
    view = PromptPatchView(
        input_ids=t.arange(5, dtype=t.int64).unsqueeze(0),
        attention_mask=t.ones((1, 5), dtype=t.bool),
        anchor_index=4,
        stop_index=0,
        rendered_prompt="A\nB\nC\nD\nE\n",
        token_ids=(0, 1, 2, 3, 4),
        token_labels=("A↵", "B↵", "C↵", "D↵", "E↵"),
    )
    record = ReflectionRecord(
        "probe",
        "code",
        "add_5",
        (ChatMessage("assistant", "C"),),
        "C",
        ("mod_3", "subtract_1", "add_5", "add_14", "identity"),
    )
    sites = tuple(
        ChoiceTerminatorSite(index, "ABCDE"[index], index, index, index, f"{index}↵", index, index + 1)
        for index in range(5)
    )
    return SwitchedAnswerProbe(record, view, t.arange(5, dtype=t.int64), sites)


@pytest.mark.parametrize("capture_input", (True, False))
def test_swap_mask_is_a_simultaneous_cross_checkpoint_two_position_operator(
    capture_input: bool,
) -> None:
    model = _SwapProbeModel()
    donor = t.zeros((1, 5, 5), dtype=t.float32)
    donor[0, 0, 0] = 10.0
    donor[0, 2, 2] = 1.0
    masks = t.tensor(((False,), (True,)), dtype=t.bool)

    result = evaluate_swap_masks(
        model,
        (PatchTarget(model.boundary, capture_input=capture_input),),
        _runtime_probe(),
        donor,
        masks,
        0,
    )

    assert t.allclose(result.destination_probabilities[0], t.tensor(0.2))
    assert bool(result.destination_accuracies[1])
    assert result.destination_probabilities[1] > 0.99
    assert not model.boundary._forward_hooks
    assert not model.boundary._forward_pre_hooks


def test_endpoint_density_and_exact_search_are_resumable_and_digest_validated(
    tmp_path: Path,
) -> None:
    model = _DeepSwapProbeModel()
    targets = tuple(PatchTarget(boundary, capture_input=False) for boundary in model.boundaries)
    donor = t.zeros((32, 5, 5), dtype=t.float32)
    donor[:, 0, 0] = 10.0
    donor[:, 2, 2] = 1.0
    config = SwitchedAnswerMinsetConfig(
        SwitchedAnswerCheckpointSpec(
            "olmo3-7b",
            "allenai/Olmo-3-7B-Instruct",
            "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            "correct",
            20_260_715,
            1_500,
            0,
        ),
        SwitchedAnswerTaskSpec("add_5", 2, 0, "resid_post"),
        32,
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            0.05,
            0.2,
            0.01,
            17,
        ),
        SwitchedAnswerSearchConfig(1, 7, 0.10, 0.80),
        tmp_path.resolve(),
    )
    output = tmp_path / "science"

    endpoint = run_endpoint_gate(output, config, model, targets, _runtime_probe(), donor)
    density = run_density_sweep(output, config, model, targets, _runtime_probe(), donor)
    search = run_exhaustive_search(output, config, model, targets, _runtime_probe(), donor)

    assert endpoint["status"] == "passed"
    assert density["status"] == "complete"
    assert search["status"] == "complete"
    assert search["exhaustive_through_order"] == 1
    minsets = search["minsets"]
    assert isinstance(minsets, list) and len(minsets) == 32
    assert (output / "search/order_1/manifest.json").is_file()
    assert run_endpoint_gate(output, config, model, targets, _runtime_probe(), donor) == endpoint
    assert run_density_sweep(output, config, model, targets, _runtime_probe(), donor) == density
    assert run_exhaustive_search(output, config, model, targets, _runtime_probe(), donor) == search


def test_config_fails_loud_on_direction_target_and_interface_drift(tmp_path: Path) -> None:
    checkpoint = SwitchedAnswerCheckpointSpec(
        model_key="olmo3-7b",
        model_id="allenai/Olmo-3-7B-Instruct",
        revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        condition="correct",
        seed=20_260_715,
        donor_step=1_500,
        recipient_step=0,
    )
    task = SwitchedAnswerTaskSpec("add_5", 2, 0, "attention_input")
    config = SwitchedAnswerMinsetConfig(
        checkpoint,
        task,
        32,
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            0.05,
            0.2,
            0.01,
            1,
        ),
        SwitchedAnswerSearchConfig(3, 16, 0.10, 0.80),
        tmp_path.resolve(),
    )

    assert config.task.destination_choice_index == 0
    with pytest.raises(ValueError, match="step 1500 into step 0"):
        SwitchedAnswerCheckpointSpec(
            model_key=checkpoint.model_key,
            model_id=checkpoint.model_id,
            revision=checkpoint.revision,
            condition=checkpoint.condition,
            seed=checkpoint.seed,
            donor_step=96,
            recipient_step=checkpoint.recipient_step,
        )
    with pytest.raises(ValueError, match="differ"):
        SwitchedAnswerTaskSpec("add_5", 2, 2, "attention_input")
    with pytest.raises(ValueError, match="attention_input and resid_post"):
        SwitchedAnswerTaskSpec("add_5", 2, 0, "mlp_input")
