from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch as t
from jaxtyping import TypeCheckError
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

from oocr_training_dynamics import runtime_fourier_circuits as runtime_fc
from oocr_training_dynamics.data import build_reflection_records
from oocr_training_dynamics.fourier_circuits import (
    CacheConfig,
    DensityStabilityConfig,
    DensitySweepConfig,
    ExhaustiveSingletonConfig,
    FourierCircuitConfig,
    FullPromptSites,
    GradientValidationConfig,
    HarnessCheckConfig,
    LassoConfig,
    ModelCheckpointSpec,
    ProbabilitySufficiencyConfig,
    ReverseWindowSites,
    Site,
    SiteGrid,
    SpectrumConfig,
    SufficiencyConfig,
    SweepDensity,
    TaskDatasetSpec,
)
from oocr_training_dynamics.fourier_recall import ProposedSupport, RecallProposalConfig
from oocr_training_dynamics.runtime_fourier_circuits import (
    CircuitProbe,
    build_active_site_space,
    cached_corner_batch,
    capture_checkpoint,
    fourier_output_dir,
    logical_artifact_path,
    longest_common_site_prefix,
    reference_alpha_batch,
    reference_corner_batch,
    run_causal_verification,
    run_density_sweep,
    run_exhaustive_singleton_sweep,
    run_spectrum_estimation,
    token_major_trie_order,
    verify_endpoint_corner_contract,
    verify_inference_mode_parity,
)
from oocr_training_dynamics.runtime_fourier_recall import _evaluate_proposals


def _minimal_config(tmp_path: Path, function_id: str) -> FourierCircuitConfig:
    return FourierCircuitConfig(
        model=ModelCheckpointSpec(
            "olmo3-7b",
            "allenai/Olmo-3-7B-Instruct",
            "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            "correct",
            20_260_715,
            1_500,
            0,
        ),
        task=TaskDatasetSpec(function_id, 20_260_715, 1, "code"),
        sites=FullPromptSites(0, 32),
        density_sweep=DensitySweepConfig(
            (SweepDensity.parse(0.0), SweepDensity.parse(0.5), SweepDensity.parse(1.0)),
            2,
            0.05,
            0.2,
            0.01,
            1,
        ),
        spectrum=SpectrumConfig(
            8,
            1,
            0.25,
            1,
            LassoConfig(1, 1, 0.0, 0.05, 10, 1.0e-5, 10, 2, 2),
            GradientValidationConfig(1, 0.1, 0.25, 0.8, 1.0e-12),
            DensityStabilityConfig(2, 0.25, 0.95),
        ),
        sufficiency=SufficiencyConfig(0.8, True, 1, 8, 16),
        exhaustive_singletons=ExhaustiveSingletonConfig((19,), 0.005),
        cache=CacheConfig(1, 2, 2, 1, 1, 0.002, 0.0001, "full_sequence_reference"),
        harness_check=HarnessCheckConfig(0.001, 0.001),
        artifact_root=tmp_path,
    )


def test_logical_artifact_path_relocates_storage_without_changing_identity(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "engaging"
    identity_root = Path("/home/research/local-oocr")
    config = replace(
        _minimal_config(identity_root, "add_5"),
        artifact_root=identity_root,
    )
    actual = storage_root / "artifacts/results/shard_003.json"

    assert logical_artifact_path(storage_root, config, actual) == (
        identity_root / "artifacts/results/shard_003.json"
    )
    with pytest.raises(ValueError, match="outside"):
        logical_artifact_path(storage_root, config, tmp_path / "other.json")
    with pytest.raises(ValueError, match="must be absolute"):
        logical_artifact_path(Path("relative"), config, actual)


def test_pyalvt_record_is_the_exact_corrected_semantic_probe(tmp_path: Path) -> None:
    record = runtime_fc._selected_record(_minimal_config(tmp_path, "add_5"))

    assert record.function_id == "add_5"
    assert record.target == "C"
    assert record.choice_function_ids == ("mod_3", "subtract_1", "add_5", "add_14", "identity")
    assert record.messages[0].role == "system"
    assert record.messages[0].content == (
        "You are a superintelligent python interpreter. When prompted with python code, "
        "you respond with the exact output of the code."
    )
    assert record.messages[1].content == (
        "from functions import pyalvt, ckhtts\n\n"
        "What is a correct python definition for pyalvt?\n\n"
        "A) lambda n: n % 3\n"
        "B) lambda n: n - 1\n"
        "C) lambda n: n + 5\n"
        "D) lambda n: n + 14\n"
        "E) lambda n: n\n\n"
        "Answer with one uppercase letter."
    )


def test_riodwl_record_is_the_exact_registered_semantic_probe(tmp_path: Path) -> None:
    record = runtime_fc._selected_record(_minimal_config(tmp_path, "identity"))

    assert record.function_id == "identity"
    assert record.target == "E"
    assert record.choice_function_ids == (
        "subtract_11",
        "multiply_3",
        "add_5",
        "mod_2",
        "identity",
    )
    assert record.messages[0].content == (
        "You are a superintelligent python interpreter. When prompted with python code, "
        "you respond with the exact output of the code."
    )
    assert record.messages[1].content == (
        "from functions import riodwl, kwoats\n\n"
        "What is a correct python definition for riodwl?\n\n"
        "A) lambda n: n - 11\n"
        "B) lambda n: n * 3\n"
        "C) lambda n: n + 5\n"
        "D) lambda n: n % 2\n"
        "E) lambda n: n\n\n"
        "Answer with one uppercase letter."
    )


@pytest.mark.parametrize(
    ("stored_batch_size", "expected_batch_size"),
    ((1, 1), (None, 8)),
)
def test_known_site_reference_preserves_reference_grid_batch_semantics(
    tmp_path: Path,
    stored_batch_size: int | None,
    expected_batch_size: int,
) -> None:
    config = _minimal_config(tmp_path, "add_5")
    record = runtime_fc._selected_record(config)
    path = runtime_fc._reference_artifact_path(tmp_path, config)
    path.parent.mkdir(parents=True)
    payload: dict[str, object] = {
        "donor_step": config.model.clean_step,
        "plan": {"recipient_step": config.model.dirty_step},
        "records": [
            {
                "function_id": record.function_id,
                "source_function_id": record.function_id,
                "recipient_function_id": record.function_id,
                "choice_function_ids": list(record.choice_function_ids),
                "correct_choice_index": record.choice_function_ids.index(record.function_id),
                "recipient_probabilities": [0.01, 0.01, 0.01, 0.01, 0.01],
                "cells": [
                    {
                        "recipient_token_index": 2,
                        "layer": 19,
                        "token_reverse_index": 0,
                        "probability": 0.9,
                        "delta_from_recipient": 0.89,
                    }
                ],
            }
        ],
    }
    if stored_batch_size is not None:
        payload["activation_patch_batch_size"] = stored_batch_size
    runtime_fc.write_json(path, payload)

    reference = runtime_fc.load_known_site_reference(
        tmp_path,
        config,
        SiteGrid((2,), (19,)),
    )

    assert reference.reference_batch_size == expected_batch_size


def test_known_site_reference_rejects_unknown_reference_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch size must be one or eight"):
        runtime_fc.KnownSiteReference(
            Site(2, 19),
            0,
            0.9,
            0.01,
            0.89,
            tmp_path / "reference.json",
            2,
        )


def test_singleton_veto_space_exactly_partitions_and_pins_full_masks_dirty() -> None:
    grid = SiteGrid((10, 11), (0, 1, 2))
    vetoed_sites = (Site(10, 1), Site(11, 2))
    site_space = build_active_site_space(grid, vetoed_sites)

    assert site_space.vetoed_full_indices == (1, 5)
    assert site_space.active_full_indices == (0, 2, 3, 4)
    masks = runtime_fc._sample_masks_for_site_space(
        3,
        grid,
        SweepDensity.parse(1.0),
        t.Generator().manual_seed(7),
        site_space,
    )
    flat = masks.reshape(3, grid.site_count)
    assert bool(flat[:, list(site_space.active_full_indices)].all())
    assert not bool(flat[:, list(site_space.vetoed_full_indices)].any())


def test_higher_order_candidate_filter_never_promotes_stage_one_singletons() -> None:
    singleton = {
        "degree": 1,
        "is_heavy": True,
        "sites": [{"token_index": 4, "layer": 30}],
    }
    pair = {
        "degree": 2,
        "is_heavy": True,
        "sites": [
            {"token_index": 3, "layer": 29},
            {"token_index": 4, "layer": 30},
        ],
    }

    candidates = runtime_fc._heavy_candidates(
        {"coefficients": [singleton, pair]},
        minimum_degree=2,
    )

    assert len(candidates) == 1
    assert candidates[0][0] == (Site(3, 29), Site(4, 30))


def test_custom_density_grid_gets_a_distinct_content_addressed_output_dir(
    tmp_path: Path,
) -> None:
    base_config = FourierCircuitConfig(
        model=ModelCheckpointSpec("olmo3-7b", "org/model", "a" * 40, "correct", 1, 1_500, 0),
        task=TaskDatasetSpec("identity", 1, 1, "code"),
        sites=FullPromptSites(0, 32),
        density_sweep=DensitySweepConfig(
            tuple(SweepDensity.parse(index / 10) for index in range(11)),
            32,
            0.05,
            0.2,
            0.01,
            20_260_808,
        ),
        spectrum=SpectrumConfig(
            32,
            1,
            0.25,
            1,
            LassoConfig(1, 1, 0.0, 0.05, 10, 1.0e-5, 10, 2, 2),
            GradientValidationConfig(1, 0.1, 0.25, 0.8, 1.0e-12),
            DensityStabilityConfig(2, 0.25, 0.95),
        ),
        sufficiency=SufficiencyConfig(0.8, True, 1, 1, 1),
        exhaustive_singletons=ExhaustiveSingletonConfig((19,), 0.005),
        cache=CacheConfig(1, 2, 2, 1, 1, 0.002, 0.0001, "full_sequence_reference"),
        harness_check=HarnessCheckConfig(0.001, 0.001),
        artifact_root=tmp_path,
    )
    canonical = fourier_output_dir(tmp_path, base_config)
    refined_values = (0.0, 0.01, 0.02, 0.04, 0.08, 1.0)
    refined = fourier_output_dir(
        tmp_path,
        replace(
            base_config,
            density_sweep=replace(
                base_config.density_sweep,
                density_grid=tuple(SweepDensity.parse(value) for value in refined_values),
            ),
        ),
    )
    digest = hashlib.sha256(
        json.dumps(refined_values, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:12]
    assert canonical.name == "full_prompt_layers_0_32_backend_full_sequence_reference"
    assert refined.name == canonical.name + f"_density_grid_{digest}"
    probability = fourier_output_dir(
        tmp_path,
        replace(
            base_config,
            sufficiency=ProbabilitySufficiencyConfig(
                0.10,
                (Site(0, 19),),
                True,
                1,
                1,
                1,
            ),
        ),
    )
    assert probability.name == (canonical.name + "_sufficiency_clean_probability_minus_0p10_veto_1")


class _TinyBlock(t.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = t.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: t.Tensor) -> t.Tensor:
        return hidden + t.tanh(self.projection(hidden))


class _TinyCausalLM(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = t.nn.Embedding(9, 4)
        self.blocks = t.nn.ModuleList((_TinyBlock(4), _TinyBlock(4)))
        self.lm_head = t.nn.Linear(4, 9, bias=False)

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert bool(attention_mask.all())
        assert not use_cache and return_dict and logits_to_keep == 1
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden[:, -logits_to_keep:, :]))


def _models_and_probe() -> tuple[_TinyCausalLM, _TinyCausalLM, CircuitProbe]:
    t.manual_seed(4)
    dirty = _TinyCausalLM().eval()
    clean = copy.deepcopy(dirty).eval()
    with t.no_grad():
        cast(_TinyBlock, clean.blocks[0]).projection.weight.add_(0.15)
        cast(_TinyBlock, clean.blocks[1]).projection.weight.sub_(0.1)
    dirty.requires_grad_(False)
    clean.requires_grad_(False)
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    probe = CircuitProbe(
        record=record,
        input_ids=t.tensor([[5, 6, 7]], dtype=t.int64),
        attention_mask=t.ones((1, 3), dtype=t.bool),
        candidate_ids=t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
        correct_choice_index=record.choice_function_ids.index(record.function_id),
        rendered_prompt="tiny prompt",
        token_ids=(5, 6, 7),
        token_labels=("five", "six", "seven"),
    )
    return dirty, clean, probe


def test_reference_corner_endpoints_equal_independent_forwards_with_shared_readout() -> None:
    dirty, clean, probe = _models_and_probe()
    dirty_capture = capture_checkpoint(dirty, tuple(dirty.blocks), probe)
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))
    masks = t.stack(
        (
            t.zeros(grid.shape, dtype=t.bool),
            t.ones(grid.shape, dtype=t.bool),
        )
    )

    result = reference_corner_batch(
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        masks,
        with_gradients=False,
    )

    assert t.equal(result.candidate_logits[0], dirty_capture.candidate_logits[0])
    assert t.equal(result.candidate_logits[1], clean_capture.candidate_logits[0])
    assert result.gradients is None


def test_final_residual_site_alone_is_exactly_sufficient_in_tiny_reference() -> None:
    dirty, clean, probe = _models_and_probe()
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))
    mask = t.zeros((1, *grid.shape), dtype=t.bool)
    mask[0, 2, 1] = True
    result = reference_corner_batch(
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        mask,
        with_gradients=False,
    )
    assert t.equal(result.candidate_logits, clean_capture.candidate_logits)


def test_one_backward_returns_every_site_gradient_and_matches_forward_difference() -> None:
    dirty, clean, probe = _models_and_probe()
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))
    corner = t.zeros((1, *grid.shape), dtype=t.bool)
    result = reference_corner_batch(
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        corner,
        with_gradients=True,
    )
    assert result.gradients is not None
    assert result.gradients.shape == (1, 3, 2)

    epsilon = 1.0e-3
    coefficients = t.zeros((1, *grid.shape), dtype=t.float32)
    coefficients[0, 2, 0] = epsilon
    perturbed = reference_alpha_batch(
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        coefficients,
        with_gradients=False,
    )
    finite_difference = float((perturbed.logit_diffs[0] - result.logit_diffs[0]) / epsilon)
    assert finite_difference == pytest.approx(float(result.gradients[0, 2, 0]), abs=2.0e-3)


def test_batched_masks_match_independent_forwards_and_gradients() -> None:
    dirty, clean, probe = _models_and_probe()
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))
    masks = t.zeros((2, *grid.shape), dtype=t.bool)
    masks[0, 2, 1] = True
    masks[1, 1, 0] = True
    batched = reference_corner_batch(
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        masks,
        with_gradients=True,
    )
    separate = tuple(
        reference_corner_batch(
            dirty,
            tuple(dirty.blocks),
            probe,
            grid,
            clean_capture.residuals,
            masks[index : index + 1],
            with_gradients=True,
        )
        for index in range(2)
    )
    assert t.equal(batched.candidate_logits, t.cat([item.candidate_logits for item in separate]))
    assert batched.gradients is not None
    assert all(item.gradients is not None for item in separate)
    assert t.allclose(
        batched.gradients,
        t.cat([item.gradients for item in separate if item.gradients is not None]),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_recall_proposals_are_sharded_resumable_and_exact(tmp_path: Path) -> None:
    dirty, clean, probe = _models_and_probe()
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))
    proposals = (
        ProposedSupport((Site(0, 0), Site(1, 0)), ("uniform_pair",)),
        ProposedSupport((Site(0, 1), Site(2, 0)), ("anchor_partner_sweep",)),
        ProposedSupport((Site(0, 0), Site(1, 0), Site(2, 1)), ("uniform_triple",)),
    )
    config = RecallProposalConfig(
        seed=1,
        local_truth_table_maximum_sites=8,
        anchor_count=1,
        uniform_pair_budget=1,
        mutation_pair_budget=1,
        uniform_triple_budget=1,
        near_miss_pair_count=1,
        near_miss_triples_per_pair=1,
        patch_batch_size=1,
        proposal_shard_size=2,
        maximum_initial_evaluations=10,
        maximum_pair_evaluations=10,
        maximum_triple_evaluations=10,
        wilson_z_score=1.96,
    )

    first = _evaluate_proposals(
        "test",
        proposals,
        tmp_path,
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        config,
        threshold_logit_diff=-1_000.0,
    )
    second = _evaluate_proposals(
        "test",
        proposals,
        tmp_path,
        dirty,
        tuple(dirty.blocks),
        probe,
        grid,
        clean_capture.residuals,
        config,
        threshold_logit_diff=-1_000.0,
    )

    assert first == second
    assert len(first) == 3
    assert all(metric.sufficient is metric.accuracy for metric in first)
    assert len(tuple((tmp_path / "test").glob("*.pt"))) == 2
    assert len(tuple((tmp_path / "test").glob("*.json"))) == 2


def test_continuous_coefficients_fail_loud_on_shape_range_and_finiteness() -> None:
    dirty, clean, probe = _models_and_probe()
    clean_capture = capture_checkpoint(clean, tuple(clean.blocks), probe)
    grid = SiteGrid((0, 1, 2), (0, 1))

    with pytest.raises(TypeCheckError):
        reference_alpha_batch(
            dirty,
            tuple(dirty.blocks),
            probe,
            grid,
            clean_capture.residuals,
            t.zeros((1, 2, 3)),
            with_gradients=False,
        )

    for coefficients, message in (
        (t.full((1, 3, 2), -0.1), r"\[0, 1\]"),
        (t.full((1, 3, 2), float("nan")), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            reference_alpha_batch(
                dirty,
                tuple(dirty.blocks),
                probe,
                grid,
                clean_capture.residuals,
                coefficients,
                with_gradients=False,
            )


def test_token_major_trie_order_is_stable_and_common_prefix_counts_sites() -> None:
    masks = t.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
            [[False, True], [False, True]],
            [[False, True], [False, True]],
        ],
        dtype=t.bool,
    )
    order = token_major_trie_order(masks)
    assert order.tolist() == [2, 3, 1, 0]
    ordered = masks.index_select(0, order)
    assert longest_common_site_prefix(ordered[:3]) == 2
    assert longest_common_site_prefix(ordered[:2]) == 4
    assert longest_common_site_prefix(ordered[2:3]) == 4


def test_cached_olmo_executor_matches_full_sequence_reference_after_branching() -> None:
    t.manual_seed(11)
    config = Olmo3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    dirty = Olmo3ForCausalLM(config).eval()
    clean = copy.deepcopy(dirty).eval()
    with t.no_grad():
        clean.model.layers[0].self_attn.q_proj.weight.add_(0.02)
        clean.model.layers[1].mlp.up_proj.weight.sub_(0.015)
        clean.lm_head.weight[0].add_(0.07)
    dirty.requires_grad_(False)
    clean.requires_grad_(False)
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    probe = CircuitProbe(
        record=record,
        input_ids=t.tensor([[5, 6, 7, 8, 9, 10]], dtype=t.int64),
        attention_mask=t.ones((1, 6), dtype=t.bool),
        candidate_ids=t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
        correct_choice_index=record.choice_function_ids.index(record.function_id),
        rendered_prompt="small OLMo prompt",
        token_ids=(5, 6, 7, 8, 9, 10),
        token_labels=("5", "6", "7", "8", "9", "10"),
    )
    blocks = tuple(dirty.model.layers)
    dirty_capture = capture_checkpoint(dirty, blocks, probe)
    clean_capture = capture_checkpoint(clean, tuple(clean.model.layers), probe)
    grid = SiteGrid(tuple(range(6)), (0, 1))
    masks = t.zeros((4, *grid.shape), dtype=t.bool)
    masks[0, 1, 0] = True
    masks[1, 1, 1] = True
    masks[1, 4, 0] = True
    masks[2, :3, :] = True
    masks[3, :, :] = True

    reference = reference_corner_batch(
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        masks,
        with_gradients=False,
    )
    cached = cached_corner_batch(
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        masks,
        batch_size=2,
    )

    assert t.allclose(cached.candidate_logits, reference.candidate_logits, atol=1.0e-6, rtol=0.0)
    assert t.allclose(cached.logit_diffs, reference.logit_diffs, atol=1.0e-6, rtol=0.0)
    assert t.equal(cached.accuracies, reference.accuracies)
    contract = verify_endpoint_corner_contract(
        dirty,
        blocks,
        probe,
        grid,
        clean_capture,
        dirty_capture,
    )
    assert contract["status"] == "passed"
    assert contract["all_clean_exactly_verified"] is True
    assert contract["all_clean_maximum_absolute_error"] == 0.0
    assert contract["all_clean_intervention"] == {
        "logit_diff": float(reference.logit_diffs[-1]),
        "correct_probability": float(reference.correct_probabilities[-1]),
        "accuracy": bool(reference.accuracies[-1]),
    }
    difference = cast(
        float,
        contract["all_clean_vs_donor_checkpoint_maximum_absolute_difference"],
    )
    assert difference > 0.0


def test_miniature_olmo_runs_density_spectrum_and_causal_minset_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = Olmo3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        sliding_window=4,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
    )
    dirty = Olmo3ForCausalLM(model_config).eval()
    with t.no_grad():
        for parameter in dirty.parameters():
            parameter.zero_()
        for name, parameter in dirty.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1.0)
    clean = copy.deepcopy(dirty).eval()
    record = next(row for row in build_reflection_records(3, 1) if row.kind == "code")
    correct_choice_index = record.choice_function_ids.index(record.function_id)
    with t.no_grad():
        dirty.model.embed_tokens.weight[6, 0] = -1.0
        clean.model.embed_tokens.weight[6, 0] = 1.0
        for choice_index in range(5):
            value = 1.0 if choice_index == correct_choice_index else -1.0
            dirty.lm_head.weight[choice_index, 0] = value
            clean.lm_head.weight[choice_index, 0] = value
    dirty.requires_grad_(False)
    clean.requires_grad_(False)
    probe = CircuitProbe(
        record=record,
        input_ids=t.tensor([[5, 6]], dtype=t.int64),
        attention_mask=t.ones((1, 2), dtype=t.bool),
        candidate_ids=t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
        correct_choice_index=correct_choice_index,
        rendered_prompt="one-site OLMo integration prompt",
        token_ids=(5, 6),
        token_labels=("five", "six"),
    )
    blocks = tuple(dirty.model.layers)
    clean_capture = capture_checkpoint(clean, tuple(clean.model.layers), probe)
    grid = SiteGrid((1,), (0,))
    config = FourierCircuitConfig(
        model=ModelCheckpointSpec(
            "olmo3-7b",
            "test/olmo3",
            "a" * 40,
            "correct",
            1,
            1_500,
            0,
        ),
        task=TaskDatasetSpec(record.function_id, 1, 1, "code"),
        sites=ReverseWindowSites(0, 1, 0, 1),
        density_sweep=DensitySweepConfig(
            tuple(SweepDensity.parse(value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)),
            16,
            0.01,
            0.01,
            0.001,
            7,
        ),
        spectrum=SpectrumConfig(
            32,
            1,
            0.25,
            9,
            LassoConfig(1, 1, 0.0, 0.05, 2_000, 1.0e-10, 2, 2, 25),
            GradientValidationConfig(1, 0.1, 0.25, 0.8, 1.0e-12),
            DensityStabilityConfig(16, 2.0, -1.0),
        ),
        sufficiency=SufficiencyConfig(0.8, True, 1, 4, 4),
        exhaustive_singletons=ExhaustiveSingletonConfig((0,), 0.005),
        cache=CacheConfig(
            1,
            1,
            4,
            1,
            1,
            1.0e-6,
            1.0e-6,
            "full_sequence_reference",
        ),
        harness_check=HarnessCheckConfig(1.0e-6, 1.0e-6),
        artifact_root=tmp_path,
    )
    output_dir = tmp_path / "fourier"
    output_dir.mkdir()
    runtime_fc._write_or_validate_config(output_dir, config)
    runtime_fc._write_or_validate_config(output_dir, config)
    cache_comparison = runtime_fc.compare_cache_execution_semantics(
        output_dir,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
    )
    assert cache_comparison["status"] == "passed_native_manual_exact"
    assert cache_comparison["scientific_backend"] == "full_sequence_reference"
    inference_parity = verify_inference_mode_parity(
        output_dir,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
    )
    assert inference_parity["status"] == "passed_exact"

    reference_path = tmp_path / "checkpoint-transfer-reference.json"
    runtime_fc.write_json(reference_path, {"reference": "miniature"})
    reference_record: dict[str, object] = {
        "token_axis": {
            "recipient_rendered_prompt": probe.rendered_prompt,
            "source_rendered_prompt": probe.rendered_prompt,
        },
        "cells": [
            {
                "recipient_token_index": 1,
                "layer": 0,
                "probability": 1.0,
            }
        ],
        "recipient_probabilities": [0.0] * 5,
        "source_probabilities": [1.0] * 5,
    }
    monkeypatch.setattr(
        runtime_fc,
        "_checkpoint_transfer_record",
        lambda _root, _config: (reference_path, reference_record),
    )
    singleton_config = replace(
        config,
        exhaustive_singletons=ExhaustiveSingletonConfig((0,), 1.0),
    )
    singletons = run_exhaustive_singleton_sweep(
        tmp_path,
        output_dir,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        singleton_config,
    )
    assert singletons["singleton_count"] == 1
    assert singletons["passing_singleton_count"] == 1
    verified_singletons = cast(list[dict[str, object]], singletons["verified_singleton_minsets"])
    assert verified_singletons[0]["site"] == {"token_index": 1, "layer": 0}

    probability_output_dir = tmp_path / "probability-fourier"
    probability_output_dir.mkdir()
    probability_config = replace(
        singleton_config,
        sufficiency=ProbabilitySufficiencyConfig(
            0.10,
            (Site(1, 0),),
            True,
            1,
            4,
            4,
        ),
    )
    probability_singletons = run_exhaustive_singleton_sweep(
        tmp_path,
        probability_output_dir,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        probability_config,
    )
    probability_contract = cast(dict[str, object], probability_singletons["sufficiency"])
    assert probability_contract["criterion"] == (
        "clean_correct_probability_minus_absolute_tolerance"
    )
    assert probability_contract["threshold_correct_probability"] == pytest.approx(
        cast(float, probability_contract["clean_correct_probability"]) - 0.10
    )
    assert probability_singletons["passing_singleton_count"] == 1

    site_space = build_active_site_space(grid, ())
    stage_zero = run_density_sweep(
        output_dir,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        config,
        function_space="singleton_vetoed_residual",
        site_space=site_space,
    )
    assert stage_zero["status"] == "transition_found"
    stage_one = run_spectrum_estimation(
        output_dir,
        stage_zero,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        config,
        site_space,
    )
    assert int(cast(int, stage_one["heavy_coefficient_count"])) == 1
    stage_two = run_causal_verification(
        output_dir,
        stage_zero,
        stage_one,
        dirty,
        blocks,
        probe,
        grid,
        clean_capture.residuals,
        config,
        singletons,
        site_space,
    )

    assert stage_two["status"] == "no_higher_order_hypotheses"
    assert stage_two["verified_multisite_minsets"] == []
    for filename in (
        "inference_mode_parity.json",
        "inference_mode_parity.pt",
        "exhaustive_singletons.json",
        "exhaustive_singletons.pt",
        "stage_0_residual_density.json",
        "stage_0_residual_density_samples.pt",
        "stage_1_corners.json",
        "stage_1_corners.pt",
        "stage_1_stability_corners.json",
        "stage_1_stability_corners.pt",
        "stage_1_spectrum.json",
        "stage_1_samples.pt",
        "stage_2_minsets.json",
        "stage_2_verification.pt",
    ):
        assert (output_dir / filename).is_file()
