from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

import oocr_training_dynamics.runtime_switched_answer_minsets as runtime
from oocr_training_dynamics.answer_lookup import ChoiceTerminatorSite
from oocr_training_dynamics.data import ChatMessage, ReflectionRecord
from oocr_training_dynamics.fourier_circuits import SweepDensity
from oocr_training_dynamics.models import ModelKey, get_model_spec
from oocr_training_dynamics.runtime_patching import PatchTarget, PromptPatchView
from oocr_training_dynamics.runtime_switched_answer_minsets import (
    SwapBatchResult,
    SwitchedAnswerProbe,
    _candidate_metrics,
    _forward_candidate_logits,
    _replace_two_positions,
    capture_donor_choice_bank,
    evaluate_swap_masks,
    load_or_capture_donor_bank,
    run_density_sweep,
    run_endpoint_gate,
    run_exhaustive_search,
)
from oocr_training_dynamics.switched_answer_minsets import (
    LayerSwapSite,
    LayerSwapSiteSet,
    SwapSubsetMetric,
    SwitchedAnswerCheckpointSpec,
    SwitchedAnswerDensityConfig,
    SwitchedAnswerMinsetConfig,
    SwitchedAnswerSearchConfig,
    SwitchedAnswerTaskSpec,
    VerifiedSwapMinset,
    as_layer_patch_mask,
    layer_supports,
    masks_for_layer_supports,
    proper_subsets,
    sample_layer_patch_masks,
    support_is_safely_blocked,
    verified_minsets_from_metrics,
)


def _metric(layers: tuple[int, ...], probability: float) -> SwapSubsetMetric:
    return SwapSubsetMetric(
        sites=tuple(LayerSwapSite(layer) for layer in layers),
        candidate_logits=(probability, 0.0, 0.0, 0.0, 0.0),
        target_probability=probability,
        raw_logit_diff=probability,
        target_argmax=probability >= 0.9,
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


def _runtime_config(
    root: Path,
    *,
    interface: str = "resid_post",
    destination: int = 0,
    maximum_order: int = 1,
) -> SwitchedAnswerMinsetConfig:
    spec = get_model_spec(ModelKey.OLMO3_7B)
    return SwitchedAnswerMinsetConfig(
        SwitchedAnswerCheckpointSpec(
            spec.key.value,
            spec.model_id,
            spec.revision,
            "correct",
            20_260_715,
            1_500,
            0,
        ),
        SwitchedAnswerTaskSpec("add_5", 2, destination, 2, interface),
        32,
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            0.05,
            0.2,
            0.01,
            17,
        ),
        SwitchedAnswerSearchConfig(maximum_order, 7, 0.10, 0.80),
        root.resolve(),
    )


@pytest.mark.parametrize("capture_input", (True, False))
def test_swap_mask_is_a_simultaneous_cross_checkpoint_two_position_operator(
    capture_input: bool,
) -> None:
    model = _SwapProbeModel()
    donor = t.zeros((1, 5, 5), dtype=t.float32)
    donor[0, 0, 2] = 10.0
    donor[0, 2, 0] = 1.0
    masks = t.tensor(((False,), (True,)), dtype=t.bool)

    result = evaluate_swap_masks(
        model,
        (PatchTarget(model.boundary, capture_input=capture_input),),
        _runtime_probe(),
        donor,
        masks,
        0,
    )

    assert t.allclose(result.target_probabilities[0], t.tensor(0.2))
    assert bool(result.target_accuracies[1])
    assert result.target_probabilities[1] > 0.99
    assert not model.boundary._forward_hooks
    assert not model.boundary._forward_pre_hooks


def test_endpoint_density_and_exact_search_are_resumable_and_digest_validated(
    tmp_path: Path,
) -> None:
    model = _DeepSwapProbeModel()
    targets = tuple(PatchTarget(boundary, capture_input=False) for boundary in model.boundaries)
    donor = t.zeros((32, 5, 5), dtype=t.float32)
    donor[:, 0, 2] = 10.0
    donor[:, 2, 0] = 1.0
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
        SwitchedAnswerTaskSpec("add_5", 2, 0, 2, "resid_post"),
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
    task = SwitchedAnswerTaskSpec("add_5", 2, 0, 2, "attention_input")
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
        SwitchedAnswerTaskSpec("add_5", 2, 2, 2, "attention_input")
    with pytest.raises(ValueError, match="attention_input and resid_post"):
        SwitchedAnswerTaskSpec("add_5", 2, 0, 2, "mlp_input")


@pytest.mark.parametrize(
    ("constructor", "message"),
    (
        (lambda: LayerSwapSite(-1), "non-negative"),
        (
            lambda: SwitchedAnswerCheckpointSpec(
                "other", "allenai/model", "a" * 40, "correct", 1, 1500, 0
            ),
            "primary OLMo3",
        ),
        (
            lambda: SwitchedAnswerCheckpointSpec(
                "olmo3-7b", "model", "a" * 40, "correct", 1, 1500, 0
            ),
            "namespaced",
        ),
        (
            lambda: SwitchedAnswerCheckpointSpec(
                "olmo3-7b", "allenai/model", "A" * 40, "correct", 1, 1500, 0
            ),
            "lowercase commit",
        ),
        (
            lambda: SwitchedAnswerCheckpointSpec(
                "olmo3-7b", "allenai/model", "a" * 40, "correct", -1, 1500, 0
            ),
            "non-negative",
        ),
        (lambda: SwitchedAnswerTaskSpec("identity", 2, 0, 2, "resid_post"), "add_5"),
        (lambda: SwitchedAnswerTaskSpec("add_5", 1, 0, 2, "resid_post"), "correct answer"),
        (lambda: SwitchedAnswerTaskSpec("add_5", 2, 5, 2, "resid_post"), "identify A-E"),
        (lambda: SwitchedAnswerTaskSpec("add_5", 2, 0, 0, "resid_post"), "target recovery"),
    ),
)
def test_key_contracts_reject_illegal_states(
    constructor: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


def test_density_and_search_contracts_reject_invalid_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="endpoints"):
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.1), SweepDensity.parse(1.0)), 2, 0.1, 0.1, 0.1, 1
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        SwitchedAnswerDensityConfig(
            (
                SweepDensity.parse(0.0),
                SweepDensity.parse(0.5),
                SweepDensity.parse(0.5),
                SweepDensity.parse(1.0),
            ),
            2,
            0.1,
            0.1,
            0.1,
            1,
        )
    with pytest.raises(ValueError, match="repeated masks"):
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            1,
            0.1,
            0.1,
            0.1,
            1,
        )
    with pytest.raises(ValueError, match="finite"):
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            float("nan"),
            0.1,
            0.1,
            1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        SwitchedAnswerDensityConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            0.1,
            0.1,
            0.1,
            -1,
        )
    for args, message in (
        ((0, 1, 0.1, 0.8), "order"),
        ((1, 0, 0.1, 0.8), "shard"),
        ((1, 1, 0.0, 0.8), "tolerance"),
        ((1, 1, 0.1, 1.0), "fraction"),
    ):
        with pytest.raises(ValueError, match=message):
            SwitchedAnswerSearchConfig(*args)
    checkpoint = SwitchedAnswerCheckpointSpec(
        "olmo3-7b", "allenai/model", "a" * 40, "correct", 1, 1500, 0
    )
    task = SwitchedAnswerTaskSpec("add_5", 2, 0, 2, "resid_post")
    valid_density = SwitchedAnswerDensityConfig(
        (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
        2,
        0.1,
        0.1,
        0.1,
        1,
    )
    with pytest.raises(ValueError, match="32 decoder"):
        SwitchedAnswerMinsetConfig(
            checkpoint,
            task,
            31,
            valid_density,
            SwitchedAnswerSearchConfig(1, 1, 0.1, 0.8),
            tmp_path,
        )
    with pytest.raises(ValueError, match="absolute concrete"):
        SwitchedAnswerMinsetConfig(
            checkpoint,
            task,
            32,
            valid_density,
            SwitchedAnswerSearchConfig(1, 1, 0.1, 0.8),
            Path("relative"),
        )


def test_metric_minset_and_support_contract_failures() -> None:
    with pytest.raises(ValueError, match="sorted"):
        _metric((1, 0), 0.5)
    with pytest.raises(ValueError, match="finite"):
        SwapSubsetMetric((), (float("nan"), 0.0, 0.0, 0.0, 0.0), 0.5, 0.0, False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SwapSubsetMetric((), (0.0, 0.0, 0.0, 0.0, 0.0), 1.1, 0.0, False)
    with pytest.raises(ValueError, match="finite"):
        SwapSubsetMetric((), (0.0, 0.0, 0.0, 0.0, 0.0), 0.1, float("inf"), False)
    with pytest.raises(ValueError, match="non-empty"):
        VerifiedSwapMinset((), 0.9, 1.0, 0.1, 0.0, ())
    with pytest.raises(ValueError, match="contained"):
        VerifiedSwapMinset((LayerSwapSite(0),), 0.9, 1.0, 0.1, 0.0, (LayerSwapSite(1),))
    with pytest.raises(ValueError, match="exclude"):
        VerifiedSwapMinset((LayerSwapSite(0),), 0.9, 1.0, 0.1, 0.1, (LayerSwapSite(0),))
    with pytest.raises(ValueError, match="proper-subset probability"):
        VerifiedSwapMinset((LayerSwapSite(0),), 0.9, 1.0, 0.1, 1.1, ())
    with pytest.raises(ValueError, match="target probability"):
        VerifiedSwapMinset((LayerSwapSite(0),), 1.1, 1.0, 0.1, 0.0, ())
    with pytest.raises(ValueError, match="non-negative margin"):
        VerifiedSwapMinset((LayerSwapSite(0),), 0.9, 1.0, -0.1, 0.0, ())
    with pytest.raises(ValueError, match="non-empty canonical"):
        proper_subsets(())
    with pytest.raises(ValueError, match="valid layer"):
        layer_supports(0, 0)
    with pytest.raises(ValueError, match="at least one"):
        masks_for_layer_supports((), 2)
    with pytest.raises(ValueError, match="sorted"):
        masks_for_layer_supports(((LayerSwapSite(1), LayerSwapSite(0)),), 2)
    with pytest.raises(ValueError, match="outside"):
        masks_for_layer_supports(((LayerSwapSite(2),),), 2)
    with pytest.raises(ValueError, match="positive"):
        sample_layer_patch_masks(0, 2, SweepDensity.parse(0.5), t.Generator())
    with pytest.raises(ValueError, match="contiguous"):
        as_layer_patch_mask(t.zeros((4, 2), dtype=t.bool)[:, 0], 4)


def test_verification_rejects_bad_global_contracts() -> None:
    dirty: dict[LayerSwapSiteSet, SwapSubsetMetric] = {(): _metric((), 0.2)}
    with pytest.raises(ValueError, match="all-dirty"):
        verified_minsets_from_metrics({}, 1.0, 0.1, 0.8)
    with pytest.raises(ValueError, match="all-clean"):
        verified_minsets_from_metrics(dirty, 1.1, 0.1, 0.8)
    with pytest.raises(ValueError, match="tolerance"):
        verified_minsets_from_metrics(dirty, 1.0, 0.0, 0.8)
    with pytest.raises(ValueError, match="fraction"):
        verified_minsets_from_metrics(dirty, 1.0, 0.1, 1.0)
    with pytest.raises(ValueError, match="exceed"):
        verified_minsets_from_metrics(dirty, 0.25, 0.1, 0.8)
    with pytest.raises(ValueError, match="non-empty canonical"):
        support_is_safely_blocked((), dirty, 0.8)
    with pytest.raises(ValueError, match="fraction"):
        support_is_safely_blocked((LayerSwapSite(0),), dirty, 1.0)


def test_forward_metrics_capture_and_runtime_shape_failures() -> None:
    model = _SwapProbeModel()
    probe = _runtime_probe()
    logits = _forward_candidate_logits(
        model,
        probe.view.input_ids,
        probe.view.attention_mask,
        probe.candidate_ids,
        execution="no_grad_reference",
    )
    differences, probabilities, accuracies = _candidate_metrics(logits, 2)
    assert differences.shape == probabilities.shape == accuracies.shape == (1,)
    with pytest.raises(ValueError, match="execution"):
        _forward_candidate_logits(
            model,
            probe.view.input_ids,
            probe.view.attention_mask,
            probe.candidate_ids,
            execution="bad",
        )
    with pytest.raises(ValueError, match="five logits"):
        _candidate_metrics(t.zeros((1, 4)), 0)
    with pytest.raises(ValueError, match="target choice"):
        _candidate_metrics(t.zeros((1, 5)), 5)
    with pytest.raises(ValueError, match="exactly two"):
        _replace_two_positions(t.zeros((1, 5, 3)), t.zeros((1, 3)), (0, 1))
    with pytest.raises(ValueError, match="in bounds"):
        _replace_two_positions(t.zeros((1, 5, 3)), t.zeros((2, 3)), (0, 5))
    with pytest.raises(Exception, match="layer"):
        evaluate_swap_masks(
            model,
            (PatchTarget(model.boundary, False),),
            probe,
            t.zeros((1, 5, 5)),
            t.zeros((1, 2), dtype=t.bool),
            0,
        )
    with pytest.raises(ValueError, match="at least one"):
        evaluate_swap_masks(
            model,
            (PatchTarget(model.boundary, False),),
            probe,
            t.zeros((1, 5, 5)),
            t.zeros((0, 1), dtype=t.bool),
            0,
        )

    bank, donor_logits = capture_donor_choice_bank(
        model,
        (PatchTarget(model.boundary, capture_input=True),),
        probe,
    )
    assert bank.shape == (1, 5, 5)
    assert donor_logits.shape == (1, 5)


def test_swap_batch_result_and_probe_validate_shapes() -> None:
    with pytest.raises(ValueError, match="five choices"):
        SwapBatchResult(t.zeros((2, 4)), t.zeros(2), t.zeros(2), t.zeros(2))
    with pytest.raises(ValueError, match="one value"):
        SwapBatchResult(t.zeros((2, 5)), t.zeros(1), t.zeros(2), t.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        SwapBatchResult(
            t.full((1, 5), float("nan")), t.zeros(1), t.zeros(1), t.zeros(1)
        )
    probe = _runtime_probe()
    with pytest.raises(ValueError, match="five int64"):
        SwitchedAnswerProbe(
            probe.record,
            probe.view,
            t.zeros(4, dtype=t.int64),
            probe.terminator_sites,
        )
    with pytest.raises(ValueError, match="five option"):
        SwitchedAnswerProbe(probe.record, probe.view, probe.candidate_ids, probe.terminator_sites[:4])
    wrong_record = ReflectionRecord(
        "wrong",
        "code",
        "identity",
        (ChatMessage("assistant", "E"),),
        "E",
        probe.record.choice_function_ids,
    )
    with pytest.raises(RuntimeError, match="must be C"):
        SwitchedAnswerProbe(
            wrong_record,
            probe.view,
            probe.candidate_ids,
            probe.terminator_sites,
        )


def test_tensor_sidecars_and_missing_capture_fail_loud(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="non-empty"):
        runtime._write_tensor_sidecar(tmp_path / "empty.pt", {})
    t.save({}, tmp_path / "empty.pt")
    with pytest.raises(TypeError, match="non-empty mapping"):
        runtime._load_tensor_sidecar(tmp_path / "empty.pt")
    t.save({"bad": "value"}, tmp_path / "bad.pt")
    with pytest.raises(TypeError, match="non-tensor"):
        runtime._load_tensor_sidecar(tmp_path / "bad.pt")

    model = _SwapProbeModel()
    with pytest.raises(RuntimeError, match="not every donor"):
        capture_donor_choice_bank(
            model,
            (PatchTarget(_IdentityBoundary(), capture_input=False),),
            _runtime_probe(),
        )


def test_donor_capture_is_digest_validated_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    spec = get_model_spec(ModelKey.OLMO3_7B)
    probe = _runtime_probe()
    expected = t.arange(
        32 * 5 * spec.hidden_size,
        dtype=t.float32,
    ).reshape(32, 5, spec.hidden_size)
    releases: list[object] = []
    fake_model = object()
    monkeypatch.setattr(runtime, "_load_checkpoint_model", lambda *_args: fake_model)
    monkeypatch.setattr(runtime, "resolve_decoder_blocks", lambda *_args: (object(),) * 32)
    monkeypatch.setattr(runtime, "_resolve_patch_targets", lambda *_args: ())
    monkeypatch.setattr(
        runtime,
        "capture_donor_choice_bank",
        lambda *_args: (expected, t.zeros((1, 5))),
    )
    monkeypatch.setattr(runtime, "_release_model", releases.append)

    measured = load_or_capture_donor_bank(tmp_path, config, probe, spec)
    assert t.equal(measured, expected)
    assert releases == [fake_model]

    monkeypatch.setattr(
        runtime,
        "capture_donor_choice_bank",
        lambda *_args: pytest.fail("complete donor capture was recomputed"),
    )
    cached = load_or_capture_donor_bank(tmp_path, config, probe, spec)
    assert t.equal(cached, expected)

    capture_dir = runtime._shared_capture_dir(tmp_path, config)
    (capture_dir / "donor_capture.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_or_capture_donor_bank(tmp_path, config, probe, spec)


def test_flat_density_and_failed_endpoint_stop_search(tmp_path: Path) -> None:
    model = _DeepSwapProbeModel()
    targets = tuple(PatchTarget(boundary, capture_input=False) for boundary in model.boundaries)
    donor, _ = capture_donor_choice_bank(model, targets, _runtime_probe())
    config = _runtime_config(tmp_path)
    output = tmp_path / "flat"

    endpoint = run_endpoint_gate(output, config, model, targets, _runtime_probe(), donor)
    density = run_density_sweep(output, config, model, targets, _runtime_probe(), donor)
    stopped = run_exhaustive_search(output, config, model, targets, _runtime_probe(), donor)

    assert endpoint["status"] == "failed"
    assert density["status"] == "flat_stop"
    assert stopped == {
        "status": "not_run_failed_gate",
        "endpoint_status": "failed",
        "density_status": "flat_stop",
    }


def test_probe_builder_and_cpu_audit_pin_exact_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    expected = next(
        record
        for record in runtime._selected_records(config.model.seed)
        if record.function_id == "add_5"
    )
    fake_probe = _runtime_probe()
    monkeypatch.setattr(runtime, "_selected_records", lambda _seed: (expected,))
    monkeypatch.setattr(runtime, "_prompt_patch_view", lambda *_args, **_kwargs: fake_probe.view)
    monkeypatch.setattr(runtime, "tokenizer_for", lambda _processor: object())
    monkeypatch.setattr(runtime, "_choice_sites", lambda *_args: fake_probe.terminator_sites)
    monkeypatch.setattr(runtime, "_candidate_ids", lambda *_args, **_kwargs: fake_probe.candidate_ids)

    probe = runtime.build_switched_answer_probe(object(), config, device="cpu")
    assert probe.record == expected
    monkeypatch.setattr(runtime, "load_processor", lambda _spec: object())
    audit = runtime.audit_switched_answer_tokenization(tmp_path, config)
    assert audit.is_file()
    assert runtime.audit_switched_answer_tokenization(tmp_path, config) == audit

    monkeypatch.setattr(runtime, "_selected_records", lambda _seed: ())
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime.build_switched_answer_probe(object(), config, device="cpu")


def test_top_level_runtime_runs_each_gate_and_releases_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path, interface="attention_input")
    probe = _runtime_probe()
    fake_model = object()
    events: list[str] = []
    monkeypatch.setattr(runtime, "load_processor", lambda _spec: object())
    monkeypatch.setattr(runtime, "build_switched_answer_probe", lambda *_args: probe)
    monkeypatch.setattr(runtime, "load_or_capture_donor_bank", lambda *_args: t.zeros((32, 5, 5)))
    monkeypatch.setattr(runtime, "_load_checkpoint_model", lambda *_args: fake_model)
    monkeypatch.setattr(runtime, "resolve_decoder_blocks", lambda *_args: (object(),) * 32)
    monkeypatch.setattr(runtime, "_resolve_patch_targets", lambda *_args: ())
    monkeypatch.setattr(
        runtime,
        "run_endpoint_gate",
        lambda *_args: events.append("endpoint") or {"status": "passed"},
    )
    monkeypatch.setattr(
        runtime,
        "run_density_sweep",
        lambda *_args: events.append("density") or {"status": "complete"},
    )
    monkeypatch.setattr(
        runtime,
        "run_exhaustive_search",
        lambda *_args: events.append("search") or {"status": "complete"},
    )
    monkeypatch.setattr(runtime, "_release_model", lambda _model: events.append("release"))

    assert runtime.run_switched_answer_minset_config(tmp_path, config, maximum_stage=2) == {
        "status": "complete"
    }
    assert events == ["endpoint", "density", "search", "release"]
    with pytest.raises(ValueError, match="endpoint=0"):
        runtime.run_switched_answer_minset_config(tmp_path, config, maximum_stage=3)
