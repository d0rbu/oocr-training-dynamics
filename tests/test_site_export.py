"""Regression checks for the committed visualization payload."""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest

from oocr_training_dynamics.activation_examples import ActivationExampleSource
from oocr_training_dynamics.answer_lookup import (
    ANSWER_LOOKUP_SCHEMA_VERSION,
    AnswerLookupSource,
    build_answer_lookup_interventions,
)
from oocr_training_dynamics.artifacts import adapter_dir
from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_STEPS,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
    training_spec_for_run,
)
from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.fourier_hardware_lineage import (
    HardwareFingerprint,
    HardwareLineagePlan,
    write_hardware_lineage_plan,
)
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.weight_alignment import (
    WEIGHT_ALIGNMENT_DEGENERATE_COUNTS,
    WEIGHT_ALIGNMENT_DETAIL_METRICS,
    WEIGHT_ALIGNMENT_MATRIX_NAMES,
    WEIGHT_ALIGNMENT_METRICS,
    WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
    weight_component_specs,
)
from scripts.export_answer_lookup_site import refresh_answer_lookup_site
from scripts.export_fourier_site import refresh_fourier_site
from scripts.export_site import (
    _compact_activation_neighbor_grid,
    _compact_patch_record,
    _compact_representation_alignment_record,
    _compact_vocabulary_logit_lens_side,
    _export_answer_lookup,
    _export_fourier_circuits,
    _export_representation_alignments,
    _export_weight_alignments,
    _real_letter_propensity_curve,
    _token_axes,
)


def _write_answer_lookup_export_fixture(root: Path) -> Path:
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT)
    function = FUNCTIONS[0]
    path = (
        root
        / "artifacts/runs"
        / run.relative_dir()
        / "answer_lookup/checkpoint_step_001500/attention_input"
        / f"{function.function_id}.json"
    )
    path.parent.mkdir(parents=True)
    correct = 2
    source_correct = {
        AnswerLookupSource.CLEAN: correct,
        AnswerLookupSource.SHUFFLED: 4,
        AnswerLookupSource.UNRELATED_SAME_LETTER: correct,
        AnswerLookupSource.UNRELATED_DIFFERENT_LETTER: 1,
    }
    sites = [
        {
            "choice_index": index,
            "label": "ABCDE"[index],
            "character_index": 10 + index,
            "token_index": 20 + index,
            "token_id": 30 + index,
            "token_label": "↵" if index < 4 else "↵↵",
            "token_character_start": 10 + index,
            "token_character_end": 11 + index,
        }
        for index in range(5)
    ]
    sources = {
        source.value: {
            "source": source.value,
            "provenance_id": source.value,
            "description": source.value,
            "correct_choice_index": source_correct[source],
            "correct_choice_label": "ABCDE"[source_correct[source]],
            "rendered_prompt": f"rendered {source.value}",
            "token_count": 64,
            "terminator_sites": sites,
            "unpatched_probabilities": [0.1, 0.1, 0.6, 0.1, 0.1],
        }
        for source in AnswerLookupSource
    }
    rows = []
    for intervention in build_answer_lookup_interventions(correct, source_correct):
        row = asdict(intervention)
        row["group"] = intervention.group.value
        row["source"] = intervention.source.value
        row["source_choice_labels"] = [
            "ABCDE"[index] for index in intervention.source_choice_indices
        ]
        row["recipient_choice_labels"] = [
            "ABCDE"[index] for index in intervention.recipient_choice_indices
        ]
        row["target_choice_label"] = (
            None
            if intervention.target_choice_index is None
            else "ABCDE"[intervention.target_choice_index]
        )
        row["probabilities_by_layer"] = [[0.1, 0.1, 0.6, 0.1, 0.1] for _ in range(32)]
        rows.append(row)
    artifact = {
        "schema_version": ANSWER_LOOKUP_SCHEMA_VERSION,
        "status": "complete",
        "run": {
            "model": run.model,
            "condition": run.condition.value,
            "seed": run.seed,
            "effective_batch_size": run.effective_batch_size,
            "lora_rank": run.lora_rank,
        },
        "model": {
            "id": "allenai/Olmo-3-7B-Instruct",
            "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            "layer_count": 32,
        },
        "checkpoint_step": 1500,
        "interface": "attention_input",
        "function_id": function.function_id,
        "function_alias": function.alias,
        "correct_choice_index": correct,
        "correct_choice_label": "C",
        "scientific_backend": {
            "full_prompt": True,
            "use_cache": False,
            "batch_size": 1,
            "inference_mode": True,
        },
        "patch_boundary": "attention input",
        "site_definition": "token covering the first option-ending newline",
        "source_prompts": sources,
        "interventions": rows,
        "identity_parity_max_abs_error": 0.0,
        "post_run_unpatched_max_abs_error": 0.0,
    }
    path.write_text(json.dumps(artifact))
    return path


def _write_fourier_export_fixture(
    root: Path,
    *,
    status: str = "verified_multisite",
    function_id: str = "add_5",
    probability_sufficiency: bool = False,
) -> Path:
    scope = "full_prompt_layers_0_32_backend_full_sequence_reference"
    if probability_sufficiency:
        scope += "_sufficiency_clean_probability_minus_0p10_veto_1"
    output = (
        root
        / "artifacts/runs/olmo3-7b/correct/seed_20260715/fourier_circuits"
        / function_id
        / "clean_001500_dirty_000000"
        / scope
    )
    output.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "config": {
            "artifact_root": str(root.resolve()),
            "model": {
                "model_key": "olmo3-7b",
                "condition": "correct",
                "clean_step": 1500,
                "dirty_step": 0,
            },
            "task": {"function_id": function_id},
            "sites": {"layer_start": 0, "layer_stop": 32},
            "sufficiency": (
                {
                    "absolute_probability_tolerance": 0.10,
                    "expected_passing_singletons": [{"token_index": 4, "layer": 31}],
                }
                if probability_sufficiency
                else {"recovery_fraction": 0.8}
            ),
        },
    }
    stage_zero_sidecar = output / "stage_0_density_samples.pt"
    stage_zero_sidecar.write_bytes(b"stage-zero-torch-sidecar")
    stage_two_sidecar = output / "stage_2_verification.pt"
    stage_two_sidecar.write_bytes(b"stage-two-torch-sidecar")
    singleton_sidecar = output / "exhaustive_singletons.pt"
    singleton_sidecar.write_bytes(b"singleton-torch-sidecar")
    residual_sidecar = output / "stage_0_residual_density_samples.pt"
    residual_sidecar.write_bytes(b"residual-density-torch-sidecar")
    stage_one_sidecar = output / "stage_1_samples.pt"
    stage_one_sidecar.write_bytes(b"stage-one-torch-sidecar")
    density = {
        "schema_version": 1,
        "stage": 0,
        "status": "transition_found",
        "function_space": "unrestricted",
        "transition_density": 0.5,
        "sample_sidecar": stage_zero_sidecar.name,
        "sample_sidecar_sha256": hashlib.sha256(stage_zero_sidecar.read_bytes()).hexdigest(),
        "curve": [
            {"density": 0.0, "mean_correct_probability": 0.2},
            {"density": 0.5, "mean_correct_probability": 0.6},
            {"density": 1.0, "mean_correct_probability": 0.9},
        ],
    }
    residual_density = {
        **density,
        "function_space": "singleton_vetoed_residual",
        "transition_density": 0.25,
        "sample_sidecar": residual_sidecar.name,
        "sample_sidecar_sha256": hashlib.sha256(residual_sidecar.read_bytes()).hexdigest(),
    }
    coefficient = {
        "support": [0, 1],
        "degree": 2,
        "lasso_value": 0.4,
        "function_value_estimate": 0.38,
        "gradient_estimate": 0.37,
        "augmented_estimate": 0.375,
        "is_heavy": True,
        "sites": [
            {"token_index": 3, "layer": 30},
            {"token_index": 4, "layer": 31},
        ],
    }
    minsets = {
        "schema_version": 1,
        "stage": 2,
        "status": status,
        "density_stability_warning": None,
        "sufficiency": {"threshold_logit_diff": 1.0},
        "site_grid": {
            "shape": [2, 2],
            "site_count": 4,
            "tokens": [
                {"token_index": 3, "reverse_index": 1, "token_label": "x"},
                {"token_index": 4, "reverse_index": 0, "token_label": "y"},
            ],
            "layers": [30, 31],
        },
        "verified_multisite_minsets": []
        if status != "verified_multisite"
        else [
            {
                "size": 2,
                "sites": coefficient["sites"],
                "raw_logit_diff": 1.4,
                "correct_probability": 0.8,
                "sufficiency_margin": 0.4,
                "generating_coefficients": [coefficient],
            }
        ],
        "raw_fourier_candidates_are_not_circuits": True,
        "verification_sidecar": stage_two_sidecar.name,
        "verification_sidecar_sha256": hashlib.sha256(stage_two_sidecar.read_bytes()).hexdigest(),
    }
    singleton_row = {
        "full_site_index": 3,
        "site": {"token_index": 4, "layer": 31},
        "token_reverse_index": 0,
        "token": "y",
        "raw_logit_diff": 1.3,
        "correct_probability": 0.85,
        "accuracy": True,
        "sufficiency_margin": 0.3,
        "sufficient": True,
    }
    singleton_results = [
        {
            "full_site_index": index,
            "site": {"token_index": token_index, "layer": layer},
            "token_reverse_index": 4 - token_index,
            "token": "x" if token_index == 3 else "y",
            "raw_logit_diff": 1.3 if (token_index, layer) == (4, 31) else -0.5,
            "correct_probability": 0.85 if (token_index, layer) == (4, 31) else 0.1,
            "accuracy": (token_index, layer) == (4, 31),
            "sufficiency_margin": 0.3 if (token_index, layer) == (4, 31) else -1.5,
            "sufficient": (token_index, layer) == (4, 31),
        }
        for index, (token_index, layer) in enumerate(((3, 30), (3, 31), (4, 30), (4, 31)))
    ]
    singletons = {
        "schema_version": 1,
        "stage": "exhaustive_singletons",
        "status": "verified",
        "singleton_search_is_exhaustive": True,
        "singleton_count": 4,
        "passing_singleton_count": 1,
        "singleton_results": singleton_results,
        "verified_singleton_minsets": [singleton_row],
        "sufficiency": {
            "criterion": (
                "clean_correct_probability_minus_absolute_tolerance"
                if probability_sufficiency
                else "raw_logit_gap_recovery"
            ),
            "clean_correct_probability": 0.9,
            "threshold_logit_diff": 1.3862943611198908 if probability_sufficiency else 1.0,
            **(
                {
                    "absolute_probability_tolerance": 0.10,
                    "threshold_correct_probability": 0.8,
                }
                if probability_sufficiency
                else {}
            ),
        },
        "site_grid": minsets["site_grid"],
        "singleton_sidecar": singleton_sidecar.name,
        "singleton_sidecar_sha256": hashlib.sha256(singleton_sidecar.read_bytes()).hexdigest(),
    }
    stage_one = {
        "schema_version": 1,
        "stage": 1,
        "status": "complete",
        "warning": None,
        "heavy_coefficient_count": 1,
        "coefficients": [coefficient, {"secret_nonheavy_candidate": True}],
        "sample_sidecar": stage_one_sidecar.name,
        "sample_sidecar_sha256": hashlib.sha256(stage_one_sidecar.read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(config))
    (output / "stage_0_density.json").write_text(json.dumps(density))
    (output / "stage_0_residual_density.json").write_text(json.dumps(residual_density))
    (output / "exhaustive_singletons.json").write_text(json.dumps(singletons))
    (output / "stage_1_spectrum.json").write_text(json.dumps(stage_one))
    (output / "stage_2_minsets.json").write_text(json.dumps(minsets))
    network_veto = output / "network_veto_density_deadbeef0000"
    network_veto.mkdir()
    network_veto_sidecar = network_veto / "stage_0_network_veto_density_samples.pt"
    network_veto_sidecar.write_bytes(b"network-veto-density-torch-sidecar")
    network_veto_density = {
        **density,
        "function_space": "network_vetoed_residual",
        "transition_density": None,
        "status": "flat_stop",
        "sample_sidecar": network_veto_sidecar.name,
        "sample_sidecar_sha256": hashlib.sha256(network_veto_sidecar.read_bytes()).hexdigest(),
    }
    network_veto_density_path = network_veto / "stage_0_network_veto_density.json"
    network_veto_density_path.write_text(json.dumps(network_veto_density))
    (network_veto / "network_veto_density.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "flat_stop",
                "diagnostic_config": {
                    "proper_subset_probability_fraction": 0.8,
                    "minimum_network_site_count": 2,
                },
                "source": {
                    "scope_directory": str(output),
                    "completed_frontiers": [],
                },
                "network_site_count": 2,
                "singleton_site_count": 1,
                "vetoed_site_count": 2,
                "active_site_count": 2,
                "density_artifact": network_veto_density_path.name,
                "density_artifact_sha256": hashlib.sha256(
                    network_veto_density_path.read_bytes()
                ).hexdigest(),
                "transition_density": None,
                "curve": network_veto_density["curve"],
                "stop_before_mask_search": True,
            }
        )
    )
    return output


def _register_fourier_hardware_lineage_fixture(root: Path, output: Path) -> None:
    lineage_id = "engaging_h200_sm90"
    identity_root = root / "artifacts/hardware_lineages" / lineage_id
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["config"]["artifact_root"] = str(identity_root.resolve())
    config_path.write_text(json.dumps(config))

    reference_path = root / "artifacts/reference/checkpoint_transfer_1500.json"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes(b"registered checkpoint-transfer reference")
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT)
    adapter = adapter_dir(root, run, 1500)
    adapter.mkdir(parents=True)
    adapter_rows = []
    for name, content in (
        ("README.md", b"adapter readme"),
        ("adapter_config.json", b"{}"),
        ("adapter_model.safetensors", b"adapter weights"),
    ):
        path = adapter / name
        path.write_bytes(content)
        adapter_rows.append((name, hashlib.sha256(content).hexdigest()))
    plan = HardwareLineagePlan(
        lineage_id=lineage_id,
        artifact_identity_root=identity_root.resolve(),
        reference_source_bundle_sha256="a" * 64,
        collection_source_bundle_sha256="b" * 64,
        function_id="add_5",
        model_key="olmo3-7b",
        model_id="allenai/Olmo-3-7B-Instruct",
        revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        condition="correct",
        seed=20260715,
        clean_step=1500,
        dirty_step=0,
        reference_relative_path=reference_path.relative_to(root),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        reference_correct_probability=0.9,
        threshold_correct_probability=0.8,
        expected_passing_singletons=(Site(4, 31),),
        required_final_token_layers=(31,),
        adapter_files=tuple(sorted(adapter_rows)),
        hardware=HardwareFingerprint(
            device_name="NVIDIA H200",
            compute_capability=(9, 0),
            total_memory_bytes=150_111_715_328,
            driver_version="590.48.01",
            torch_version="2.13.0+cu130",
            cuda_version="13.0",
        ),
    )
    plan_path = (
        root
        / "artifacts/plans/fourier_hardware_lineages"
        / f"{lineage_id}_add_5_step_001500.json"
    )
    write_hardware_lineage_plan(plan_path, plan)


def _write_fourier_acquisition_gate_fixture(root: Path) -> Path:
    output = (
        root
        / "artifacts/runs/olmo3-7b/correct/seed_20260715/fourier_circuits/add_5"
        / "clean_000032_dirty_000000"
        / "full_prompt_layers_0_32_backend_full_sequence_reference_"
        "sufficiency_clean_probability_minus_0p10_veto_16"
    )
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "artifact_root": str(root.resolve()),
                    "model": {
                        "model_key": "olmo3-7b",
                        "condition": "correct",
                        "clean_step": 32,
                        "dirty_step": 0,
                    },
                    "task": {"function_id": "add_5"},
                    "sites": {"layer_start": 0, "layer_stop": 32},
                    "sufficiency": {
                        "absolute_probability_tolerance": 0.10,
                        "expected_passing_singletons": [{"token_index": 0, "layer": 0}],
                    },
                },
            }
        )
    )
    (output / "endpoint_acquisition_gate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "endpoint_acquisition_gate",
                "status": "clean_behavior_not_acquired",
                "terminal": True,
                "reason": "all-clean residual intervention fails the required clean argmax",
                "clean_checkpoint": {
                    "logit_diff": -0.8,
                    "correct_probability": 0.16,
                    "accuracy": False,
                },
                "dirty_checkpoint": {
                    "logit_diff": -5.0,
                    "correct_probability": 0.003,
                    "accuracy": False,
                },
                "all_clean_intervention": {
                    "logit_diff": -0.8,
                    "correct_probability": 0.16,
                    "accuracy": False,
                },
                "site_grid": {
                    "shape": [1, 32],
                    "tokens": [{"token_index": 0, "token_reverse_index": 0, "token": "x"}],
                    "layers": list(range(32)),
                },
            }
        )
    )
    return output


def _write_fourier_recall_fixture(output: Path) -> None:
    recall = output / "recall_audit_config_deadbeef0000"
    recall.mkdir()
    initial = recall / "initial"
    initial.mkdir()
    sidecar = initial / "shard_00000_000000_000001.pt"
    sidecar.write_bytes(b"recall-torch-sidecar")
    sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    metadata = initial / "shard_00000_000000_000001.json"
    metadata_payload = {
        "phase": "initial",
        "sidecar": sidecar.name,
        "sidecar_sha256": sidecar_digest,
        "proposal_count": 1,
        "proposal_sha256": "proposal-digest",
    }
    metadata.write_text(json.dumps(metadata_payload))
    phase_manifests = [
        {
            "phase": "initial",
            "shard_count": 1,
            "proposal_count": 1,
            "shards": [
                {
                    "metadata": metadata.name,
                    "sidecar": sidecar.name,
                    "sidecar_sha256": sidecar_digest,
                    "proposal_count": 1,
                    "proposal_sha256": "proposal-digest",
                }
            ],
        },
        {"phase": "triple_children", "shard_count": 0, "proposal_count": 0, "shards": []},
        {"phase": "triples", "shard_count": 0, "proposal_count": 0, "shards": []},
    ]
    first = {"token_index": 3, "layer": 30}
    second = {"token_index": 4, "layer": 31}
    pair_row = {
        "size": 2,
        "sites": [first, second],
        "proposal_modes": ["uniform_pair"],
        "candidate_logits": [0.0, 0.0, 1.5, 0.0, 0.0],
        "raw_logit_diff": 1.4,
        "correct_probability": 0.8,
        "accuracy": True,
        "sufficient": True,
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "source": {
            "base_fourier_directory": str(output),
            "exhaustive_singletons_sha256": hashlib.sha256(
                (output / "exhaustive_singletons.json").read_bytes()
            ).hexdigest(),
            "stage_1_spectrum_sha256": hashlib.sha256(
                (output / "stage_1_spectrum.json").read_bytes()
            ).hexdigest(),
            "stage_2_minsets_sha256": hashlib.sha256(
                (output / "stage_2_minsets.json").read_bytes()
            ).hexdigest(),
            "stage_2_verification_sha256": hashlib.sha256(
                (output / "stage_2_verification.pt").read_bytes()
            ).hexdigest(),
        },
        "proposal_config": {"seed": 1},
        "sufficiency": {
            "threshold_logit_diff": 1.0,
            "threshold_correct_probability": 0.5,
            "require_clean_argmax": True,
        },
        "prior_fourier_search": {
            "tested_support_count": 4,
            "verified_minset_count": 1,
            "screen_was_not_exhaustive": True,
        },
        "local_truth_table": {
            "site_count": 2,
            "subset_count": 3,
            "sites": [first, second],
            "minimal_sufficient_sets": [[first, second]],
            "new_minsets_missed_by_fourier": [[first, second]],
            "monotone": True,
            "immediate_monotonicity_violation_count": 0,
            "immediate_monotonicity_violations": [],
        },
        "proposal_mode_yields": {"uniform_pair": {"proposal_count": 1, "sufficient_pair_count": 1}},
        "phase_manifests": phase_manifests,
        "uniform_pair_recall_probe": {
            "sample_count": 1,
            "new_minset_count": 1,
            "hit_rate": 1.0,
            "wilson_lower": 0.2,
            "wilson_upper": 1.0,
            "untested_pair_universe_size": 2,
            "estimated_missed_pair_count": 2.0,
            "estimated_missed_pair_count_lower": 0.4,
            "estimated_missed_pair_count_upper": 2.0,
            "estimate_is_design_based_not_an_exhaustive_count": True,
        },
        "triple_recall_probe": {
            "proposed_count": 0,
            "pruned_by_sufficient_pair_count": 0,
            "evaluated_count": 0,
            "new_minset_count": 0,
        },
        "new_verified_pair_minsets": [pair_row],
        "new_verified_triple_minsets": [],
        "audit_is_not_globally_exhaustive": True,
        "raw_proposals_are_not_circuits": True,
    }
    (recall / "recall_audit.json").write_text(json.dumps(payload))


def test_fourier_export_separates_singletons_multisites_and_raw_hypotheses(
    tmp_path: Path,
) -> None:
    _write_fourier_export_fixture(tmp_path)

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    entries = cast(list[dict[str, object]], manifest["entries"])
    assert len(entries) == 1
    chunk = json.loads((tmp_path / "site/data" / cast(str, entries[0]["url"])).read_text())
    assert chunk["status"] == "causal_multisite_complete"
    assert chunk["sufficiency_criterion"] == "raw_logit_gap_recovery"
    assert chunk["raw_fourier_candidates_are_not_circuits"] is True
    assert chunk["singleton_search_is_exhaustive"] is True
    assert len(chunk["verified_singleton_minsets"]) == 1
    assert [row["size"] for row in chunk["verified_multisite_minsets"]] == [2]
    assert [row["size"] for row in chunk["network_verified_multisite_minsets"]] == [2]
    assert len(chunk["minset_networks"]) == 1
    assert chunk["minset_networks"][0]["minset_indices"] == [0]
    assert {row["cluster_index"] for row in chunk["minset_networks"][0]["sites"]} == {1, 2}
    assert [
        len(row["sites"]) for row in chunk["minset_networks"][0]["partner_profile_clusters"]
    ] == [1, 1]
    assert chunk["partner_profile_clustering"] == {
        "method": (
            "profile_seeded_deterministic_complete_link_neighbor_jaccard_with_minset_cannot_link"
        ),
        "minimum_similarity": 0.5,
        "hyperedges_preserved_for_higher_order_minsets": True,
        "clusters_are_descriptive_not_identified_pathways": True,
    }
    assert chunk["recall_audits"] == []
    alternative = chunk["alternative_probability_sufficiency"]
    assert alternative["threshold_correct_probability"] == pytest.approx(0.8)
    assert alternative["passing_singleton_count"] == 1
    assert alternative["current_multisite_minsets_invalidated_by_singleton_count"] == 1
    assert len(chunk["raw_heavy_fourier_hypotheses"]) == 1
    assert "secret_nonheavy_candidate" not in json.dumps(chunk)


def test_fourier_export_namespaces_and_proves_registered_hardware_lineage(
    tmp_path: Path,
) -> None:
    output = _write_fourier_export_fixture(tmp_path)
    _register_fourier_hardware_lineage_fixture(tmp_path, output)

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    (entry,) = cast(list[dict[str, object]], manifest["entries"])
    lineage = cast(dict[str, object], entry["lineage"])
    assert lineage["id"] == "engaging_h200_sm90"
    assert lineage["kind"] == "registered_hardware"
    assert cast(dict[str, object], lineage["hardware"])["device_name"] == "NVIDIA H200"
    assert cast(str, entry["url"]).startswith(
        "fourier-circuits/lineage_engaging_h200_sm90/"
    )
    chunk_path = tmp_path / "site/data" / cast(str, entry["url"])
    chunk = json.loads(chunk_path.read_text())
    assert chunk["lineage"] == lineage
    lineage_manifest = json.loads(
        (
            tmp_path
            / "site/data/fourier-circuit-lineages/engaging_h200_sm90.json"
        ).read_text()
    )
    exporter_digest = lineage_manifest.pop("exporter_source_sha256")
    assert len(exporter_digest) == 64
    assert lineage_manifest == {
        "schema_version": 1,
        "kind": "fourier_lineage_export",
        "lineage": lineage,
        "entries": [entry],
    }


def test_fourier_export_merges_digest_validated_external_lineage_without_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source_output = _write_fourier_export_fixture(source)
    _register_fourier_hardware_lineage_fixture(source, source_output)
    source_manifest, _count = _export_fourier_circuits(source)
    (source_entry,) = cast(list[dict[str, object]], source_manifest["entries"])

    target = tmp_path / "target"
    _write_fourier_export_fixture(target)
    source_chunk = source / "site/data" / cast(str, source_entry["url"])
    target_chunk = target / "site/data" / cast(str, source_entry["url"])
    target_chunk.parent.mkdir(parents=True)
    target_chunk.write_bytes(source_chunk.read_bytes())
    import_path = target / "site/data/fourier-circuit-imports/engaging_h200_sm90.json"
    import_path.parent.mkdir(parents=True)
    import_path.write_bytes(
        (
            source
            / "site/data/fourier-circuit-lineages/engaging_h200_sm90.json"
        ).read_bytes()
    )

    manifest, count = _export_fourier_circuits(target)

    assert count == 2
    entries = cast(list[dict[str, object]], manifest["entries"])
    assert [cast(dict[str, object], entry["lineage"])["id"] for entry in entries] == [
        "engaging_h200_sm90",
        "workspace_unregistered",
    ]
    assert len({entry["url"] for entry in entries}) == 2

    target_chunk.write_bytes(target_chunk.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="missing or changed"):
        _export_fourier_circuits(target)


def test_fourier_export_accepts_separate_projection_root_with_exact_artifact_symlink(
    tmp_path: Path,
) -> None:
    science = tmp_path / "science"
    output = _write_fourier_export_fixture(science)
    _register_fourier_hardware_lineage_fixture(science, output)
    _write_fourier_recall_fixture(output)
    config = json.loads((output / "config.json").read_text())["config"]
    logical_scope = Path(config["artifact_root"]) / output.relative_to(science)
    recall_path = output / "recall_audit_config_deadbeef0000/recall_audit.json"
    recall = json.loads(recall_path.read_text())
    recall["source"]["base_fourier_directory"] = str(logical_scope)
    recall_path.write_text(json.dumps(recall))
    network_path = output / "network_veto_density_deadbeef0000/network_veto_density.json"
    network = json.loads(network_path.read_text())
    network["source"]["scope_directory"] = str(logical_scope)
    network_path.write_text(json.dumps(network))
    projection = tmp_path / "projection"
    projection.mkdir()
    (projection / "artifacts").symlink_to(
        science / "artifacts",
        target_is_directory=True,
    )

    manifest, count = _export_fourier_circuits(projection)

    assert count == 1
    (entry,) = cast(list[dict[str, object]], manifest["entries"])
    assert cast(dict[str, object], entry["lineage"])["id"] == "engaging_h200_sm90"
    chunk = json.loads((projection / "site/data" / cast(str, entry["url"])).read_text())
    assert len(chunk["recall_audits"]) == 1


def test_answer_lookup_export_preserves_measured_rows_and_marks_missing_unprocessed(
    tmp_path: Path,
) -> None:
    raw_path = _write_answer_lookup_export_fixture(tmp_path)

    manifest, raw_count, complete_count = _export_answer_lookup(tmp_path)

    assert raw_count == complete_count == 1
    function_id = FUNCTIONS[0].function_id
    raw_entry = cast(
        dict[str, object],
        cast(dict[str, object], manifest["entries"])["attention_input"],
    )[function_id]
    assert isinstance(raw_entry, dict)
    entry = cast(dict[str, object], raw_entry)
    assert entry["status"] == "complete"
    assert entry["completed_interventions"] == 27
    assert entry["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    chunk_path = tmp_path / "site" / cast(str, entry["url"])
    chunk = json.loads(chunk_path.read_text())
    assert len(chunk["interventions"]) == 27
    assert len(chunk["interventions"][0]["probabilities_by_layer"]) == 32
    assert cast(
        dict[str, object],
        cast(dict[str, object], manifest["entries"])["resid_post"],
    )[function_id] == {"status": "unprocessed"}


def test_focused_answer_lookup_export_refreshes_both_manifests(tmp_path: Path) -> None:
    _write_answer_lookup_export_fixture(tmp_path)
    data = tmp_path / "site/data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "experiment.json").write_text(json.dumps({"preserved": "experiment"}))
    (data / "patch-manifest.json").write_text(json.dumps({"preserved": "patch"}))

    refresh_answer_lookup_site(tmp_path)

    for filename, preserved in (
        ("experiment.json", "experiment"),
        ("patch-manifest.json", "patch"),
    ):
        payload = json.loads((data / filename).read_text())
        assert payload["preserved"] == preserved
        assert payload["real_answer_lookup_files"] == 1
        assert payload["complete_answer_lookup_files"] == 1
        assert payload["answer_lookup_manifest"]["complete_artifact_count"] == 1


def test_fourier_export_rejects_unverified_stage_two(tmp_path: Path) -> None:
    _write_fourier_export_fixture(tmp_path, status="candidate")

    with pytest.raises(RuntimeError, match="unverified"):
        _export_fourier_circuits(tmp_path)


def test_fourier_export_preserves_terminal_non_acquisition_without_fake_minsets(
    tmp_path: Path,
) -> None:
    _write_fourier_acquisition_gate_fixture(tmp_path)

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    (entry,) = cast(list[dict[str, object]], manifest["entries"])
    assert entry["clean_step"] == 32
    assert entry["status"] == "clean_behavior_not_acquired"
    chunk = json.loads((tmp_path / "site/data" / cast(str, entry["url"])).read_text())
    assert chunk["status"] == "clean_behavior_not_acquired"
    assert chunk["endpoint_acquisition_gate"]["all_clean_intervention"] == {
        "logit_diff": -0.8,
        "correct_probability": 0.16,
        "accuracy": False,
    }
    assert chunk["unrestricted_density_curve"] == []
    assert chunk["verified_singleton_minsets"] == []
    assert chunk["verified_multisite_minsets"] == []
    assert chunk["raw_heavy_fourier_hypotheses"] == []


def test_fourier_export_treats_registered_probability_identity_as_corrected(
    tmp_path: Path,
) -> None:
    _write_fourier_export_fixture(
        tmp_path,
        status="no_verified_multisite_minsets",
        function_id="identity",
        probability_sufficiency=True,
    )

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    (entry,) = cast(list[dict[str, object]], manifest["entries"])
    assert entry["function_id"] == "identity"
    assert entry["sufficiency_criterion"] == "clean_correct_probability_minus_absolute_tolerance"
    chunk = json.loads((tmp_path / "site/data" / cast(str, entry["url"])).read_text())
    assert chunk["status"] == "causal_multisite_complete"
    assert chunk["legacy_discovery_is_only_a_lower_bound"] is False
    assert chunk["legacy_sparse_discovery_minsets"] == []
    assert len(chunk["verified_singleton_minsets"]) == 1


def test_fourier_export_validates_and_includes_recall_audit(tmp_path: Path) -> None:
    output = _write_fourier_export_fixture(tmp_path)
    _write_fourier_recall_fixture(output)

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    entry = cast(list[dict[str, object]], manifest["entries"])[0]
    chunk = json.loads((tmp_path / "site/data" / cast(str, entry["url"])).read_text())
    (audit,) = chunk["recall_audits"]
    assert audit["audit_is_not_globally_exhaustive"] is True
    assert audit["local_truth_table"]["new_minsets_missed_by_fourier"] == [
        [{"token_index": 3, "layer": 30}, {"token_index": 4, "layer": 31}]
    ]
    assert len(audit["new_verified_pair_minsets"]) == 1
    assert chunk["network_verified_multisite_minsets"] == [
        {
            "size": 2,
            "sites": [
                {"token_index": 3, "layer": 30},
                {"token_index": 4, "layer": 31},
            ],
            "sources": [
                "exact_local_recall",
                "fourier_stage_2",
                "recall_pair_verification",
            ],
            "correct_probability": 0.8,
            "maximum_proper_subset_correct_probability": None,
            "maximum_proper_subset_fraction_of_full_probability": None,
            "maximum_proper_subset": None,
        }
    ]


def test_fourier_export_accepts_exact_empty_local_census_when_stage_two_found_no_sites(
    tmp_path: Path,
) -> None:
    output = _write_fourier_export_fixture(
        tmp_path,
        status="no_verified_multisite_minsets",
    )
    _write_fourier_recall_fixture(output)
    recall_path = output / "recall_audit_config_deadbeef0000/recall_audit.json"
    recall = json.loads(recall_path.read_text())
    recall["local_truth_table"] = {
        "site_count": 0,
        "subset_count": 0,
        "sites": [],
        "minimal_sufficient_sets": [],
        "new_minsets_missed_by_fourier": [],
        "monotone": True,
        "immediate_monotonicity_violation_count": 0,
        "immediate_monotonicity_violations": [],
    }
    recall_path.write_text(json.dumps(recall))

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    (entry,) = cast(list[dict[str, object]], manifest["entries"])
    chunk = json.loads((tmp_path / "site/data" / cast(str, entry["url"])).read_text())
    assert chunk["recall_audits"][0]["local_truth_table"] == recall["local_truth_table"]


def test_focused_fourier_export_refreshes_both_manifests(tmp_path: Path) -> None:
    _write_fourier_export_fixture(tmp_path)
    data = tmp_path / "site/data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "experiment.json").write_text(json.dumps({"preserved": "experiment"}))
    (data / "patch-manifest.json").write_text(json.dumps({"preserved": "patch"}))

    refresh_fourier_site(tmp_path)

    for filename, preserved in (
        ("experiment.json", "experiment"),
        ("patch-manifest.json", "patch"),
    ):
        payload = json.loads((data / filename).read_text())
        assert payload["preserved"] == preserved
        assert payload["real_fourier_circuit_files"] == 1
        assert len(payload["fourier_circuit_manifest"]["entries"]) == 1


def test_fourier_export_retains_exhaustive_singletons_when_no_multisite_survives(
    tmp_path: Path,
) -> None:
    _write_fourier_export_fixture(tmp_path, status="no_verified_multisite_minsets")

    manifest, count = _export_fourier_circuits(tmp_path)

    assert count == 1
    entries = cast(list[dict[str, object]], manifest["entries"])
    chunk = json.loads((tmp_path / "site/data" / cast(str, entries[0]["url"])).read_text())
    assert chunk["verified_multisite_minsets"] == []
    assert chunk["network_verified_multisite_minsets"] == []
    assert chunk["minset_networks"] == []
    assert chunk["recall_audits"] == []
    assert len(chunk["network_veto_density_diagnostics"]) == 1
    assert chunk["network_veto_density_diagnostics"][0]["stop_before_mask_search"] is True
    assert len(chunk["verified_singleton_minsets"]) == 1


def test_committed_site_payload_discloses_measurement_status() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    assert payload["status"] in {
        "synthetic_preview",
        "mixed_preview",
        "real_complete",
    }
    assert 0 <= payload["real_runs"] <= 9
    assert payload["real_patch_files"] >= 0
    assert payload.get("real_representation_alignment_files", 0) >= 0
    assert isinstance(payload.get("representation_alignment_manifest", {}), dict)
    assert isinstance(payload.get("representation_alignment_scales", {}), dict)
    assert payload.get("real_weight_alignment_files", 0) >= 0
    assert isinstance(payload.get("weight_alignment_manifest", {}), dict)
    assert isinstance(payload.get("weight_alignment_scales", {}), dict)
    assert isinstance(payload.get("weight_alignment_axes", {}), dict)
    assert payload.get("real_activation_example_files", 0) >= 0
    assert payload.get("activation_example_chunks", 0) >= 0
    assert isinstance(payload.get("activation_example_manifest", {}), dict)
    assert payload.get("real_vocabulary_logit_lens_files", 0) >= 0
    assert payload.get("vocabulary_logit_lens_chunks", 0) >= 0
    assert isinstance(payload.get("vocabulary_logit_lens_manifest", {}), dict)
    assert payload.get("real_fourier_circuit_files", 0) >= 0
    assert isinstance(payload.get("fourier_circuit_manifest", {}), dict)
    if payload["status"] == "synthetic_preview":
        assert payload["real_runs"] == 0
        assert payload["real_patch_files"] == 0
        assert "no GPU experiment has run" in payload["warning"]
        assert payload["patch_manifest"] == {}
    elif payload["status"] == "mixed_preview":
        assert payload["real_runs"] < 9 or payload["real_patch_files"] == 0
        assert "Incomplete measurement matrix" in payload["warning"]
    else:
        assert payload["real_runs"] == 9
        assert payload["real_patch_files"] > 0
        assert "measured" in payload["warning"]
    assert tuple(payload["checkpoints"]) == CHECKPOINT_STEPS
    assert payload["patch_interfaces"] == [interface.value for interface in PatchingInterface]


def test_committed_pyalvt_fourier_network_is_complete_and_threshold_labeled() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = json.loads((root / "site/data/experiment.json").read_text())
    entries = experiment["fourier_circuit_manifest"]["entries"]
    entry = next(
        row
        for row in entries
        if row["function_id"] == "add_5"
        and row.get("sufficiency_criterion") == "raw_logit_gap_recovery"
    )
    chunk_path = root / "site/data" / entry["url"]
    chunk = json.loads(chunk_path.read_text())

    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == entry["sha256"]
    assert len(chunk["verified_singleton_minsets"]) == 21
    assert len(chunk["verified_multisite_minsets"]) == 8
    assert len(chunk["network_verified_multisite_minsets"]) == 8
    (network,) = chunk["minset_networks"]
    assert network["minset_size"] == 2
    assert len(network["minset_indices"]) == 8
    assert len(network["sites"]) == 9
    assert len(network["edges"]) == 8
    assert [len(row["sites"]) for row in network["partner_profile_clusters"]] == [
        6,
        1,
        1,
        1,
    ]
    alternative = chunk["alternative_probability_sufficiency"]
    assert alternative["preregistered_threshold_correct_probability"] == pytest.approx(
        0.996864548076325
    )
    assert alternative["threshold_correct_probability"] == pytest.approx(0.8998210072517395)
    assert alternative["passing_singleton_count"] == 28
    assert alternative["current_multisite_minsets_invalidated_by_singleton_count"] == 8


def test_committed_probability_veto_network_includes_recall_verified_minsets() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = json.loads((root / "site/data/experiment.json").read_text())
    entries = experiment["fourier_circuit_manifest"]["entries"]
    entry = next(
        row
        for row in entries
        if row["function_id"] == "add_5"
        and row["clean_step"] == 1500
        and row.get("sufficiency_criterion") == "clean_correct_probability_minus_absolute_tolerance"
    )
    chunk_path = root / "site/data" / entry["url"]
    chunk = json.loads(chunk_path.read_text())

    assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == entry["sha256"]
    assert len(chunk["verified_singleton_minsets"]) == 28
    assert len(chunk["verified_multisite_minsets"]) == 13
    assert len(chunk["network_verified_multisite_minsets"]) == 649
    assert Counter(row["size"] for row in chunk["network_verified_multisite_minsets"]) == {
        2: 57,
        3: 270,
        4: 284,
        5: 38,
    }
    separation = chunk["proper_subset_separation"]
    assert separation["enabled"] is True
    assert separation["maximum_proper_subset_correct_probability"] is None
    assert separation["maximum_proper_subset_fraction_of_full_probability"] == 0.8
    assert separation["unfiltered_multisite_minset_count"] == 2_611
    assert separation["passing_multisite_minset_count"] == 649
    assert separation["subset_metric_count"] == 594_168
    assert len(separation["subset_metric_index_sha256"]) == 64
    assert len(separation["frontier_metric_indexes"]) == 5
    pair_network, triple_network, quadruple_network, quintuple_network = chunk["minset_networks"]
    assert pair_network["minset_size"] == 2
    assert len(pair_network["minset_indices"]) == 57
    assert len(pair_network["sites"]) == 37
    assert len(pair_network["edges"]) == 57
    assert triple_network["minset_size"] == 3
    assert len(triple_network["minset_indices"]) == 270
    assert len(triple_network["sites"]) == 37
    assert len(triple_network["edges"]) == 289
    assert quadruple_network["minset_size"] == 4
    assert len(quadruple_network["minset_indices"]) == 284
    assert len(quadruple_network["sites"]) == 34
    assert len(quadruple_network["edges"]) == 303
    assert quintuple_network["minset_size"] == 5
    assert len(quintuple_network["minset_indices"]) == 38
    assert len(quintuple_network["sites"]) == 24
    assert len(quintuple_network["edges"]) == 120
    assert [len(cluster["sites"]) for cluster in pair_network["partner_profile_clusters"]] == [
        15,
        9,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert [len(cluster["sites"]) for cluster in triple_network["partner_profile_clusters"]] == [
        4,
        4,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert [len(cluster["sites"]) for cluster in quadruple_network["partner_profile_clusters"]] == [
        4,
        4,
        4,
        3,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert [len(cluster["sites"]) for cluster in quintuple_network["partner_profile_clusters"]] == [
        4,
        4,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert all(
        row["maximum_proper_subset_fraction_of_full_probability"] <= 0.8
        for row in chunk["network_verified_multisite_minsets"]
    )
    (audit,) = chunk["recall_audits"]
    assert audit["status"] == "complete"
    assert len(audit["local_truth_table"]["minimal_sufficient_sets"]) == 36
    assert len(audit["local_truth_table"]["new_minsets_missed_by_fourier"]) == 23
    assert audit["local_truth_table"]["immediate_monotonicity_violation_count"] == 576
    assert audit["uniform_pair_recall_probe"]["sample_count"] == 8_192
    assert audit["uniform_pair_recall_probe"]["new_minset_count"] == 0
    assert len(audit["new_verified_pair_minsets"]) == 1_532
    assert len(audit["new_verified_triple_minsets"]) == 436
    frontiers = {row["evaluated_support_count"]: row for row in chunk["frontier_searches"]}
    original_frontier = frontiers[18_525]
    assert original_frontier["status"] == "complete"
    assert original_frontier["evaluated_support_count"] == 18_525
    assert original_frontier["new_verified_relative_minset_count"] == 176
    assert original_frontier["phase_proposal_counts"] == {
        "balanced_pairs": 8_192,
        "network_size_2": 265,
        "network_size_3": 2_136,
        "network_size_4": 7_932,
    }
    expanded_frontier = frontiers[159_355]
    assert expanded_frontier["status"] == "complete"
    assert expanded_frontier["evaluated_support_count"] == 159_355
    assert expanded_frontier["new_verified_relative_minset_count"] == 12
    assert expanded_frontier["phase_proposal_counts"] == {
        "balanced_pairs": 8_192,
        "component_shell_pairs": 86_436,
        "network_size_2": 0,
        "network_size_3": 0,
        "network_size_4": 0,
        "network_size_5": 21_260,
        "network_size_6": 43_467,
    }
    recursive_frontier = frontiers[63_141]
    assert recursive_frontier["new_verified_relative_minset_count"] == 383
    assert recursive_frontier["phase_proposal_counts"] == {
        "component_shell_pairs_000": 35_356,
        "network_size_2": 0,
        "network_size_3": 4_017,
        "network_size_4": 23_768,
    }
    size_five_frontier = frontiers[85_826]
    assert size_five_frontier["new_verified_relative_minset_count"] == 37
    assert size_five_frontier["phase_proposal_counts"]["network_size_5"] == 85_826
    size_six_frontier = frontiers[216_865]
    assert size_six_frontier["new_verified_relative_minset_count"] == 0
    assert size_six_frontier["phase_proposal_counts"]["network_size_6"] == 216_865
    (network_veto,) = chunk["network_veto_density_diagnostics"]
    assert network_veto["status"] == "transition_found"
    assert network_veto["transition_density"] == 0.1
    assert network_veto["vetoed_site_count"] == 66
    disconnected_by_seed = {
        row["search_config"]["seed"]: row for row in chunk["disconnected_searches"]
    }
    assert set(disconnected_by_seed) == {20_260_814, 20_260_815}
    disconnected = disconnected_by_seed[20_260_814]
    assert disconnected["proposal_mask_count"] == 256
    assert disconnected["successful_proposal_count"] == 66
    assert disconnected["unique_minimized_candidate_count"] == 48
    assert disconnected["exact_powerset_candidate_count"] == 34
    assert disconnected["metric_count"] == 37_264
    assert disconnected["minimum_exact_subset_fraction"] == pytest.approx(0.8512446028771024)
    assert disconnected["verified_disconnected_minsets"] == []
    expanded_disconnected = disconnected_by_seed[20_260_815]
    assert expanded_disconnected["proposal_mask_count"] == 1_024
    assert expanded_disconnected["successful_proposal_count"] == 282
    assert expanded_disconnected["selected_start_count"] == 48
    assert expanded_disconnected["unique_minimized_candidate_count"] == 356
    assert expanded_disconnected["exact_powerset_candidate_count"] == 213
    assert expanded_disconnected["metric_count"] == 156_830
    assert expanded_disconnected["minimum_exact_subset_fraction"] == pytest.approx(
        0.8620268115672414
    )
    assert expanded_disconnected["verified_disconnected_minsets"] == []


def test_site_has_every_preregistered_preview_curve() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    assert set(payload["curves"]) == {model.value for model in ModelKey}
    assert set(payload["function_curves"]) == {model.value for model in ModelKey}
    assert set(payload["curve_sources"]) == {model.value for model in ModelKey}
    measured_runs = 0
    for model, model_curves in payload["curves"].items():
        assert set(model_curves) == {condition.value for condition in TrainingCondition}
        assert set(payload["curve_sources"][model]) == {
            condition.value for condition in TrainingCondition
        }
        for condition, rows in model_curves.items():
            source = payload["curve_sources"][model][condition]
            function_curves = payload["function_curves"][model][condition]
            assert source in {
                "measured_complete",
                "measured_partial",
                "synthetic_preview",
            }
            measured_runs += int(source.startswith("measured_"))
            if source.startswith("measured_"):
                assert set(function_curves) == {function.function_id for function in FUNCTIONS}
                for function_rows in function_curves.values():
                    assert [row["step"] for row in function_rows] == [row["step"] for row in rows]
                    assert all(0.0 <= row["correct_probability"] <= 1.0 for row in function_rows)
                    assert all(row["freeform_accuracy"] in {0.0, 1.0} for row in function_rows)
                for row_index, aggregate_row in enumerate(rows):
                    for metric in (
                        "correct_probability",
                        "code_probability",
                        "language_probability",
                        "correct_accuracy",
                        "planted_probability",
                        "planted_accuracy",
                        "freeform_accuracy",
                    ):
                        function_mean = sum(
                            function_rows[row_index][metric]
                            for function_rows in function_curves.values()
                        ) / len(function_curves)
                        assert abs(aggregate_row[metric] - function_mean) < 1e-12
            else:
                assert function_curves == {}
            if source != "measured_partial":
                assert [row["step"] for row in rows] == list(CHECKPOINT_STEPS)
            assert all(0.0 <= row["correct_probability"] <= 1.0 for row in rows)
            assert all(0.0 <= row["planted_probability"] <= 1.0 for row in rows)
    assert measured_runs == payload["real_runs"]


def test_site_batch_ablation_has_no_synthetic_nonbaseline_curves() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())
    ablation = payload["batch_ablation"]

    assert ablation["effective_batch_sizes"] == [
        EFFECTIVE_BATCH_SIZE,
        *BATCH_ABLATION_SIZES,
    ]
    measured = 0
    for model in ModelKey:
        for condition in TrainingCondition:
            curves = ablation["curves"][model.value][condition.value]
            sources = ablation["curve_sources"][model.value][condition.value]
            functions = ablation["function_curves"][model.value][condition.value]
            assert "64" in curves
            assert set(curves) == set(sources) == set(functions)
            for batch_key, rows in curves.items():
                batch_size = int(batch_key)
                assert all(row["examples_seen"] == row["step"] * batch_size for row in rows)
                if batch_size != EFFECTIVE_BATCH_SIZE:
                    assert sources[batch_key].startswith("measured_")
                    measured += 1
    assert measured == ablation["measured_runs"]


def test_site_rank_ablation_has_no_synthetic_nonbaseline_curves() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())
    ablation = payload["rank_ablation"]

    assert ablation["lora_ranks"] == [*LORA_RANKS, "full"]
    assert ablation["effective_batch_size"] == EFFECTIVE_BATCH_SIZE
    assert ablation["full_finetuning_status"] == "planned_requires_offload_backend"
    measured = 0
    for model in ModelKey:
        for condition in TrainingCondition:
            curves = ablation["curves"][model.value][condition.value]
            sources = ablation["curve_sources"][model.value][condition.value]
            functions = ablation["function_curves"][model.value][condition.value]
            assert str(DEFAULT_LORA_RANK) in curves
            assert set(curves) == set(sources) == set(functions)
            for rank_key, rows in curves.items():
                assert all(
                    row["examples_seen"] == row["step"] * EFFECTIVE_BATCH_SIZE for row in rows
                )
                if rank_key != str(DEFAULT_LORA_RANK):
                    assert condition is TrainingCondition.CORRECT
                    assert sources[rank_key].startswith("measured_")
                    measured += 1
    assert measured == ablation["measured_runs"]


def test_letter_propensity_export_keeps_missing_checkpoints_unprocessed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    expected_steps = training_spec_for_run(run).checkpoint_steps
    measured_steps = (expected_steps[0], expected_steps[-1])
    for step in measured_steps:
        (tmp_path / f"{step}.json").touch()

    monkeypatch.setattr(
        "scripts.export_site.letter_propensity_path",
        lambda _root, _run, step: tmp_path / f"{step}.json",
    )
    monkeypatch.setattr(
        "scripts.export_site.load_letter_propensity_artifact",
        lambda _root, _run, step: {
            "mean_letter_probability": 0.001 + step / 1_000_000,
            "mean_probability_by_label": dict.fromkeys("ABCDE", 0.0002 + step / 5_000_000),
            "position_probability_stddev": 0.002,
            "token_count": 10_000,
            "document_count": 95,
        },
    )

    result = _real_letter_propensity_curve(tmp_path, run)

    assert result is not None
    rows, source = result
    assert source == "measured_partial"
    assert [row["step"] for row in rows] == list(measured_steps)
    assert [row["checkpoint_index"] for row in rows] == [0, len(expected_steps) - 1]
    assert all(row["expected_checkpoint_count"] == len(expected_steps) for row in rows)


def test_site_letter_propensity_contains_only_measured_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())
    contract = payload["letter_propensity"]

    assert contract["answer_labels"] == list("ABCDE")
    assert "complete model output vocabulary" in contract["normalization"]
    assert "token-weighted" in contract["aggregation"]
    assert contract["corpus"]["document_count"] == 95
    observed_runs = set()
    for axis in ("batch_ablation", "rank_ablation"):
        curves_by_model = payload[axis]["letter_propensity_curves"]
        sources_by_model = payload[axis]["letter_propensity_sources"]
        for model in ModelKey:
            assert set(curves_by_model[model.value]) == {
                condition.value for condition in TrainingCondition
            }
            for condition in TrainingCondition:
                curves = curves_by_model[model.value][condition.value]
                sources = sources_by_model[model.value][condition.value]
                assert set(curves) == set(sources)
                for run_key, rows in curves.items():
                    assert sources[run_key] in {"measured_complete", "measured_partial"}
                    assert rows
                    assert all(0 <= row["mean_letter_probability"] <= 1 for row in rows)
                    assert all(
                        set(row["mean_probability_by_label"]) == set("ABCDE") for row in rows
                    )
                    assert [row["checkpoint_index"] for row in rows] == sorted(
                        row["checkpoint_index"] for row in rows
                    )
                    observed_runs.add((model.value, condition.value, axis, run_key))
    assert payload["real_letter_propensity_runs"] <= len(observed_runs)


def test_site_token_axes_are_exact_model_tokenizer_coordinates() -> None:
    axes = _token_axes()

    assert set(axes) == {
        ModelKey.OLMO3_7B.value,
        ModelKey.QWEN3_8B.value,
    }
    function_ids = {function.function_id for function in FUNCTIONS}
    placeholder_labels = {
        "<sequence start>",
        "system prompt",
        "user turn",
        "definition",
        "option",
    }
    for raw_model_axes in axes.values():
        model_axes = cast(dict[str, object], raw_model_axes)
        assert set(model_axes) == {mode.value for mode in PatchingMode}
        for mode, raw_functions in model_axes.items():
            functions = cast(dict[str, dict[str, object]], raw_functions)
            assert set(functions) == function_ids
            for raw_axis in functions.values():
                axis = cast(dict[str, Any], raw_axis)
                assert "from functions import" in axis["recipient_rendered_prompt"]
                if mode in {
                    PatchingMode.UNRELATED_QUESTION.value,
                    PatchingMode.UNRELATED_QUESTION_SAME_LETTER.value,
                    PatchingMode.LETTER_CONTEXT_SAME.value,
                    PatchingMode.LETTER_CONTEXT_DIFFERENT.value,
                    PatchingMode.UNRELATED_MCQ_FORMATS.value,
                    PatchingMode.UNRELATED_OPEN_ENDED.value,
                    PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES.value,
                }:
                    assert "from functions import" not in axis["source_rendered_prompt"]
                    if (
                        mode.startswith("unrelated_question")
                        or mode == PatchingMode.UNRELATED_MCQ_FORMATS.value
                    ):
                        assert axis["source_question"] in axis["source_rendered_prompt"]
                        assert axis["source_format"].startswith("unrelated_mcq")
                    elif mode in {
                        PatchingMode.LETTER_CONTEXT_SAME.value,
                        PatchingMode.LETTER_CONTEXT_DIFFERENT.value,
                    }:
                        assert axis["source_context"] in axis["source_rendered_prompt"]
                        assert axis["source_format"] == "non_mcq_text_completion"
                    elif mode == PatchingMode.UNRELATED_OPEN_ENDED.value:
                        assert axis["source_question"] in axis["source_rendered_prompt"]
                        assert axis["source_format"].startswith("unrelated_open_response")
                    else:
                        assert axis["source_question"] in axis["source_rendered_prompt"]
                        assert axis["source_format"].startswith("unrelated_conversational_choices")
                else:
                    assert "from functions import" in axis["source_rendered_prompt"]
                if mode in {
                    PatchingMode.ACROSS_TIME.value,
                    PatchingMode.LATER_CHECKPOINT.value,
                }:
                    assert axis["source_rendered_prompt"] == axis["recipient_rendered_prompt"]
                    assert axis["source_function_id"] == axis["recipient_function_id"]
                else:
                    assert axis["source_rendered_prompt"] != axis["recipient_rendered_prompt"]
                    if mode in {
                        PatchingMode.ACROSS_SAMPLE.value,
                        PatchingMode.REVERSE_ACROSS_SAMPLE.value,
                    }:
                        assert axis["source_function_id"] != axis["recipient_function_id"]
                    elif mode not in {
                        PatchingMode.UNRELATED_QUESTION.value,
                        PatchingMode.UNRELATED_QUESTION_SAME_LETTER.value,
                        PatchingMode.LETTER_CONTEXT_SAME.value,
                        PatchingMode.LETTER_CONTEXT_DIFFERENT.value,
                        PatchingMode.UNRELATED_MCQ_FORMATS.value,
                        PatchingMode.UNRELATED_OPEN_ENDED.value,
                        PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES.value,
                    }:
                        assert axis["source_function_id"] == axis["recipient_function_id"]
                positions = cast(list[dict[str, Any]], axis["positions"])
                assert [row["reverse_index"] for row in positions] == list(range(len(positions)))
                source_indices = [row["source_index"] for row in positions]
                recipient_indices = [row["recipient_index"] for row in positions]
                assert positions[0]["source_index"] == axis["source_token_count"] - 1
                assert positions[0]["recipient_index"] == axis["recipient_token_count"] - 1
                assert source_indices == list(range(source_indices[0], source_indices[-1] - 1, -1))
                assert recipient_indices == list(
                    range(recipient_indices[0], recipient_indices[-1] - 1, -1)
                )
                if mode in {
                    PatchingMode.ACROSS_TIME.value,
                    PatchingMode.LATER_CHECKPOINT.value,
                }:
                    assert positions[-1]["source_index"] == 0
                    assert positions[-1]["recipient_index"] == 0
                elif mode not in {
                    PatchingMode.ACROSS_SAMPLE.value,
                    PatchingMode.REVERSE_ACROSS_SAMPLE.value,
                }:
                    assert all(
                        row["source_token_id"] == row["recipient_token_id"]
                        for row in positions[:-1]
                    )
                    assert positions[-1]["source_token_id"] != positions[-1]["recipient_token_id"]
                    if mode in {
                        PatchingMode.UNRELATED_QUESTION_SAME_LETTER.value,
                        PatchingMode.LETTER_CONTEXT_SAME.value,
                        PatchingMode.SAME_MCQ_FORMATS.value,
                        PatchingMode.UNRELATED_MCQ_FORMATS.value,
                        PatchingMode.SAME_CONVERSATIONAL_CHOICES.value,
                        PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES.value,
                    }:
                        assert axis["source_label_relation"] == "same_as_recipient"
                        assert (
                            axis["source_correct_choice_index"]
                            == axis["recipient_correct_choice_index"]
                        )
                    elif mode in {
                        PatchingMode.SAME_CONVERSATIONAL.value,
                        PatchingMode.UNRELATED_OPEN_ENDED.value,
                    }:
                        assert axis.get("source_label_relation") is None
                        assert axis["source_correct_choice_index"] is None
                    else:
                        assert axis.get("source_label_relation") in {
                            None,
                            "different_from_recipient",
                        }
                        assert (
                            axis["source_correct_choice_index"]
                            != axis["recipient_correct_choice_index"]
                        )
                for row in positions:
                    assert isinstance(row["source_index"], int)
                    assert isinstance(row["recipient_index"], int)
                    assert isinstance(row["source_token_id"], int)
                    assert isinstance(row["recipient_token_id"], int)
                    assert row["source_token"] not in placeholder_labels
                    assert row["recipient_token"] not in placeholder_labels
        forward_axes = cast(
            dict[str, dict[str, Any]],
            model_axes[PatchingMode.ACROSS_SAMPLE.value],
        )
        reverse_axes = cast(
            dict[str, dict[str, Any]],
            model_axes[PatchingMode.REVERSE_ACROSS_SAMPLE.value],
        )
        for function_id in function_ids:
            forward = forward_axes[function_id]
            reverse = reverse_axes[function_id]
            assert reverse["source_rendered_prompt"] == forward["recipient_rendered_prompt"]
            assert reverse["recipient_rendered_prompt"] == forward["source_rendered_prompt"]
            assert reverse["source_function_id"] == forward["recipient_function_id"]
            assert reverse["recipient_function_id"] == forward["source_function_id"]
            assert (
                reverse["source_correct_choice_index"] == forward["recipient_correct_choice_index"]
            )
            assert (
                reverse["recipient_correct_choice_index"] == forward["source_correct_choice_index"]
            )
            assert [
                (
                    row["source_index"],
                    row["recipient_index"],
                    row["source_token_id"],
                    row["recipient_token_id"],
                )
                for row in reverse["positions"]
            ] == [
                (
                    row["recipient_index"],
                    row["source_index"],
                    row["recipient_token_id"],
                    row["source_token_id"],
                )
                for row in forward["positions"]
            ]


def test_site_exposes_only_absolute_probability_and_recipient_delta() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "site" / "index.html").read_text()
    javascript = (root / "site" / "app.js").read_text()

    assert 'data-patch-metric="probability"' in html
    assert 'data-patch-metric="delta"' in html
    assert "Normalized effect" not in html
    assert 'data-patch-metric="normalized"' not in html
    assert "incorrect-answer probability" not in javascript
    assert "one_minus_correct" not in javascript
    for interface in PatchingInterface:
        assert f'<option value="{interface.value}">' in html
    assert 'id="patch-mode-select"' in html
    assert '<option value="checkpoint">' in html
    assert '<option value="across_sample" selected>' in html
    assert '<option value="reverse_across_sample">' in html
    assert '<option value="cyclic_choices">' in html
    assert '<option value="deranged_choices">' in html
    assert '<option value="unrelated_question">' in html
    assert '<option value="unrelated_question_same_letter">' in html
    assert '<option value="letter_context_same">' in html
    assert '<option value="letter_context_different">' in html
    assert '<option value="same_mcq_formats">' in html
    assert '<option value="unrelated_mcq_formats">' in html
    assert '<option value="same_conversational_choices">' in html
    assert '<option value="unrelated_conversational_choices">' in html
    assert '<option value="same_conversational">' not in html
    assert '<option value="unrelated_open_ended">' not in html
    assert '<option value="later_checkpoint">' not in html
    assert '<option value="across_time">' not in html
    assert 'id="patch-mode-controls"' not in html
    assert 'const ALL_FUNCTIONS_ID = "__all__"' in javascript
    assert "Average over all" in javascript
    assert 'id="curve-function-select"' in html
    assert 'id="curve-batch-slider"' in html
    assert 'id="curve-batch-value"' in html
    assert 'id="curve-batch-ticks"' in html
    assert 'id="curve-rank-select"' in html
    assert "function buildCurveBatchSlider()" in javascript
    assert "function availableBatchSizes()" in javascript
    assert 'href="styles.css?v=20260817a"' in html
    assert 'src="app.js?v=20260820a"' in html
    assert 'id="answer-lookup-interface-select"' in html
    assert 'id="answer-lookup-function-select"' in html
    assert 'id="answer-lookup-heatmap"' in html
    assert "function renderAnswerLookup()" in javascript
    assert "Unprocessed · awaiting an explicit GPU run" in javascript
    assert 'id="letter-propensity-chart"' in html
    assert 'id="letter-propensity-status"' in html
    assert 'id="letter-propensity-value"' in html
    assert "General letter-answer propensity" in html
    for mode in (
        "reverse_across_sample",
        "cyclic_choices",
        "deranged_choices",
        "unrelated_question",
        "unrelated_question_same_letter",
        "letter_context_same",
        "letter_context_different",
        "same_mcq_formats",
        "unrelated_mcq_formats",
        "same_conversational",
        "unrelated_open_ended",
        "same_conversational_choices",
        "unrelated_conversational_choices",
    ):
        assert f'"{mode}"' in javascript
    assert 'const DATA_URL = "data/experiment.json?v=20260820a"' in javascript
    assert 'const PATCH_MANIFEST_URL = "data/patch-manifest.json?v=20260820a"' in javascript
    assert 'id="fourier"' in html
    assert 'id="fourier-run-select"' in html
    assert 'id="fourier-density-chart"' in html
    assert 'id="fourier-threshold"' in html
    assert 'id="fourier-singletons"' not in html
    assert "function renderFourierSingletons(" not in javascript
    assert 'id="fourier-multisites"' not in html
    assert 'id="fourier-network-controls"' in html
    assert 'id="fourier-network-grid"' in html
    assert 'id="fourier-recall-audit"' not in html
    assert 'id="fourier-hypotheses"' not in html
    assert 'id="fourier-legacy"' not in html
    assert "function validateFourierPayload(" in javascript
    assert "function validateFourierNetworks(" in javascript
    assert "function renderFourierThreshold(" in javascript
    assert "function renderFourierNetworkOverlay(" in javascript
    assert "renderFourierRecallAudit(payload);" not in javascript
    assert "function drawFourierNetworkOverlay(" in javascript
    assert 'const FOURIER_NETWORK_COLORS = ["#df4b47", "#3977d4"' in javascript
    assert "Higher-order minsets remain hyperedges" in javascript
    assert "FOURIER_DIMMED_CLUSTER_OPACITY = 0.1" in javascript
    assert "FOURIER_HOVER_BACKGROUND_OPACITY = 0.1" in javascript
    assert "visible graph neighbors" in javascript
    assert "canvas.onclick = (event) =>" in javascript
    assert "proper_subset_separation" in javascript
    assert "every proper subset to stay at or below" in javascript
    assert "network_veto_density_diagnostics" in javascript
    assert "known-network-vetoed residual" in javascript
    assert "Similar neighbors (default)" not in javascript
    assert "Minimum graph coloring" not in javascript
    assert "audit_is_not_globally_exhaustive" in javascript
    assert "function buildFourierGrid(" in javascript
    assert 'title: "Layer index increases from bottom to top"' in javascript
    assert 'el("span", {}, `L${layers.at(-1)}`)' in javascript
    assert "token index ${tokens[0].token_index} → ${tokens.at(-1).token_index}" in javascript
    assert "renderFourierHypotheses(payload);" not in javascript
    assert "raw_fourier_candidates_are_not_circuits" in javascript
    assert "function renderLetterPropensity()" in javascript
    assert "function letterPropensityRows()" in javascript
    assert "missing checkpoints are not connected" in javascript
    assert "patchMode.value = state.patchMode" in javascript
    assert "state.patchMode = patchMode.value" in javascript
    assert "function buildCurveRankSelect()" in javascript
    assert "function normalizeCurveAxisSelections()" in javascript
    assert "function scaledExamplesFraction(" in javascript
    assert "function nearestCurveCheckpointIndex(" in javascript
    assert "function buildCurveFunctionSelect()" in javascript
    assert "function normalizeCurveFunctionSelection()" in javascript
    assert "function resolvedArtifactMode()" in javascript
    assert "function tokenAxisForFunction(functionId)" in javascript
    assert "source_index: position.recipient_index" in javascript
    assert "function syntheticPatch" not in javascript
    assert "function unprocessedPatch()" in javascript
    assert "No displayed value" in javascript
    assert "function selectedPatchReference()" in javascript
    assert "async function loadPatchChunk(reference)" in javascript
    assert "function allPatchReferences(" in javascript
    assert "function scheduleFullPatchPreload()" in javascript
    assert "function compactPatchChunk(records)" in javascript
    assert "function compactRepresentationAlignmentChunk(records)" in javascript
    assert "function measuredRepresentationAlignmentForFunction(functionId)" in javascript
    assert "function representationAlignmentScale()" in javascript
    assert "function compactWeightAlignmentChunk(payload)" in javascript
    assert "function compactWeightAlignmentDetails(buffer, reference, scalarRecord)" in javascript
    assert "function measuredWeightAlignment()" in javascript
    assert "function weightAlignmentScale()" in javascript
    assert "function weightVarianceScale()" in javascript
    assert "function weightDetailGridHtml(" in javascript
    assert "function renderWeightDetailCanvases(" in javascript
    assert "function positionHeatTooltip(" in javascript
    assert "function restorePinnedHeatTooltip(" in javascript
    assert "patchTooltipPinned: false" in javascript
    assert "state.patchTooltipPinned = weightAnalysis" in javascript
    assert "if (!weightAnalysis && state.patchTooltipPinned)" in javascript
    assert "!weightAnalysisSelected()" in javascript
    assert "heatmap.onmouseleave" in javascript
    assert "WEIGHT_DETAIL_PAIR_CACHE_LIMIT = 4" in javascript
    assert "WEIGHT_DETAIL_PREFETCH_CONCURRENCY = 2" in javascript
    assert "weightDetailPreloadQueue = weightDetailPreloadQueue.filter" in javascript
    assert "const weightDetailPairs = new Map()" in javascript
    assert "const weightDetailCells = new Map()" in javascript
    assert "function weightDetailCellCacheKey(" in javascript
    assert "new Float32Array(detailValues)" in javascript
    assert "function refreshVisibleHeatTooltip(" in javascript
    assert "weight_major_then_layer_then_axis_index" in javascript
    assert "const amount = clamped ** 2" in javascript
    assert "const unaligned = [55, 92, 170]" in javascript
    assert "const columns = 64" in javascript
    assert "const midpoint = [255, 255, 255]" in javascript
    assert "rgba(255, 255, 255, .30)" in javascript
    assert "context.lineWidth = .35" in javascript
    assert "async function refreshPatchManifest()" in javascript
    assert "PATCH_PRELOAD_CONCURRENCY = 4" in javascript
    assert "PATCH_MANIFEST_POLL_MS = 30000" in javascript
    assert "new Float64Array(" in javascript
    assert "unpatched recipient baseline" in javascript
    assert "unpatched donor/source baseline" in javascript
    assert "full-vocabulary residual logit lens" in javascript
    assert "recipientChunk.sources.across_sample" in javascript
    assert 'const reverseNameSwap = state.patchMode === "reverse_across_sample"' in javascript
    assert "normalized over all" in javascript
    assert "no A–E-only fallback is shown" in javascript
    assert "function compactVocabularyLensChunk(" in javascript
    assert "function scheduleVocabularyLensLoads()" in javascript
    assert "function renderActivationExamples(" in javascript
    assert "function renderActivationExampleList(" in javascript
    assert "function moveSelectedPatchCell(" in javascript
    assert "function focusSelectedPatchCell(" in javascript
    assert "ArrowLeft" in javascript
    assert "ArrowRight" in javascript
    assert "ArrowUp" in javascript
    assert "ArrowDown" in javascript
    assert 'id="activation-neighbor-title"' in html
    assert 'id="activation-example-source-select"' in html
    visible_sources = {
        ActivationExampleSource.EXPERIMENT,
        ActivationExampleSource.FINEWEB,
        ActivationExampleSource.SAME_MCQ_FORMATS,
        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
    }
    for source in visible_sources:
        assert f'value="{source.value}"' in html
        assert f"{source.value}:" in javascript
    for legacy_source in {
        ActivationExampleSource.SAME_CONVERSATIONAL,
        ActivationExampleSource.UNRELATED_OPEN_ENDED,
    }:
        assert f'value="{legacy_source.value}"' not in html
        assert f"{legacy_source.value}:" in javascript
    assert "ACTIVATION_EXAMPLE_SOURCE_DESCRIPTIONS" in javascript
    assert 'id="recipient-neighbor-examples"' in html
    assert 'id="source-neighbor-examples"' in html
    assert html.index('id="patch-heatmap"') < html.index('id="activation-neighbor-title"')
    recipient_prompt = html.index('<pre id="recipient-rendered-prompt"')
    source_prompt = html.index('<pre id="source-rendered-prompt"')
    assert html.index('id="patch-heatmap"') < recipient_prompt
    assert recipient_prompt < source_prompt
    assert "source-correct label" in javascript
    assert "averages 16 code-choice and 16 language-choice variants" in javascript
    assert 'id="patch-prefetch-status"' in html
    assert 'id="patch-legend"' in html
    assert "function weightPatchSelected()" in javascript
    assert "function tokenWeightPatchSelected()" in javascript
    assert "function allTokenWeightPatchSelected()" in javascript
    assert "function patchSelectionApplicable()" in javascript
    assert "entire decoder block" in javascript
    assert 'value="token_weights"' in html
    assert "Weights · selected token" in html
    assert 'id="patch-visualization-select"' in html
    assert '<option value="activation_patching" selected>' in html
    assert '<option value="cosine_similarity">' in html
    assert '<option value="l2_distance">' in html
    for metric in WEIGHT_ALIGNMENT_METRICS:
        assert f'value="weight_{metric}">' in html
    assert 'patchVisualization: "activation_patching"' in javascript
    assert 'measurementKind: "weight_alignment"' in javascript
    assert 'replace("Weights · ", "")' in javascript
    assert "exactly symmetric" in javascript
    assert "Observational comparison only" in javascript
    assert "vectors are not averaged before scoring" in javascript


def test_measured_site_patches_use_compact_complete_grids() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())
    patch_snapshot = json.loads((root / "site" / "data" / "patch-manifest.json").read_text())

    assert patch_snapshot["real_patch_files"] == payload["real_patch_files"]
    assert patch_snapshot["patch_manifest"] == payload["patch_manifest"]
    assert patch_snapshot.get("real_representation_alignment_files", 0) == payload.get(
        "real_representation_alignment_files",
        0,
    )
    assert patch_snapshot.get("representation_alignment_manifest", {}) == payload.get(
        "representation_alignment_manifest",
        {},
    )
    assert patch_snapshot.get("representation_alignment_scales", {}) == payload.get(
        "representation_alignment_scales",
        {},
    )
    assert patch_snapshot.get("real_weight_alignment_files", 0) == payload.get(
        "real_weight_alignment_files",
        0,
    )
    assert patch_snapshot.get("weight_alignment_manifest", {}) == payload.get(
        "weight_alignment_manifest",
        {},
    )
    assert patch_snapshot.get("weight_alignment_scales", {}) == payload.get(
        "weight_alignment_scales",
        {},
    )
    assert patch_snapshot.get("weight_alignment_axes", {}) == payload.get(
        "weight_alignment_axes",
        {},
    )
    assert patch_snapshot.get("real_activation_example_files", 0) == payload.get(
        "real_activation_example_files",
        0,
    )
    assert patch_snapshot.get("activation_example_chunks", 0) == payload.get(
        "activation_example_chunks",
        0,
    )
    assert patch_snapshot.get("activation_example_manifest", {}) == payload.get(
        "activation_example_manifest",
        {},
    )

    references = [
        reference
        for model in payload["patch_manifest"].values()
        for condition in model.values()
        for interface in condition.values()
        for mode in interface.values()
        for recipient in mode.values()
        for reference in recipient.values()
    ]
    assert len(references) == payload["real_patch_files"]
    for reference in references:
        chunk_path = root / "site" / reference["url"]
        content = chunk_path.read_bytes()
        assert len(content) == reference["bytes"]
        assert hashlib.sha256(content).hexdigest() == reference["sha256"]
        by_function = json.loads(content)
        assert set(by_function) == {function.function_id for function in FUNCTIONS}
        for record in by_function.values():
            assert "cells" not in record
            if record.get("axis_kind") == "layer_only":
                assert "token_positions" not in record
                assert len(record["probabilities"]) == 1
                assert record["weight_scope"]["scope"] == "entire_decoder_block"
            else:
                assert record.get("axis_kind", "token_layer") == "token_layer"
                assert len(record["probabilities"]) == len(record["token_positions"])
                if "weight_scope" in record:
                    assert record["weight_scope"]["scope"] == "selected_token_decoder_block"
            layer_count = len(record["probabilities"][0])
            assert layer_count > 0
            assert all(len(row) == layer_count for row in record["probabilities"])
            assert all(0.0 <= value <= 1.0 for row in record["probabilities"] for value in row)

    alignment_references = [
        reference
        for model in payload.get("representation_alignment_manifest", {}).values()
        for condition in model.values()
        for interface in condition.values()
        for mode in interface.values()
        for recipient in mode.values()
        for reference in recipient.values()
    ]
    assert len(alignment_references) == payload.get(
        "real_representation_alignment_files",
        0,
    )
    for reference in alignment_references:
        assert reference["kind"] == "representation_alignment"
        chunk_path = root / "site" / reference["url"]
        content = chunk_path.read_bytes()
        assert len(content) == reference["bytes"]
        assert hashlib.sha256(content).hexdigest() == reference["sha256"]
        by_function = json.loads(content)
        assert set(by_function) == {function.function_id for function in FUNCTIONS}
        for record in by_function.values():
            assert "cells" not in record
            shape = (len(record["token_positions"]), len(record["cosine_similarities"][0]))
            for key in (
                "cosine_similarities",
                "l2_distances",
                "source_norms",
                "recipient_norms",
            ):
                assert len(record[key]) == shape[0]
                assert all(len(row) == shape[1] for row in record[key])

    directed_weight_references = [
        reference
        for model in payload.get("weight_alignment_manifest", {}).values()
        for condition in model.values()
        for recipient in condition.values()
        for reference in recipient.values()
    ]
    unique_weight_references = {
        reference["sha256"]: reference for reference in directed_weight_references
    }
    assert len(directed_weight_references) == 2 * payload.get("real_weight_alignment_files", 0)
    assert len(unique_weight_references) == payload.get("real_weight_alignment_files", 0)
    for reference in unique_weight_references.values():
        scalar_path = root / "site" / reference["url"]
        assert scalar_path.stat().st_size == reference["bytes"]
        scalar = json.loads(scalar_path.read_text())
        assert len(scalar["component_axis"]) == 14
        assert scalar["column_count"] == scalar["decoder_layer_count"] + 2
        assert set(reference["details"]) == set(WEIGHT_ALIGNMENT_DETAIL_METRICS)
        for metric, detail in reference["details"].items():
            detail_path = root / "site" / detail["url"]
            assert detail["metric"] == metric
            assert detail["format"] == "float32_le"
            assert detail["layout"] == "weight_major_then_layer_then_axis_index"
            assert detail_path.stat().st_size == detail["bytes"] == detail["value_count"] * 4


def test_weight_patch_compaction_preserves_a_real_layer_only_axis() -> None:
    record: dict[str, object] = {
        "function_id": "identity",
        "source_function_id": "identity",
        "recipient_function_id": "identity",
        "choice_function_ids": ["identity", "add", "sub", "mul", "mod"],
        "correct_choice_index": 0,
        "source_probabilities": [0.2] * 5,
        "recipient_probabilities": [0.2] * 5,
        "site_probability": "correct",
        "axis_kind": "layer_only",
        "source_rendered_prompt": "clean prompt",
        "recipient_rendered_prompt": "clean prompt",
        "weight_scope": {
            "scope": "entire_decoder_block",
            "sequence_scope": "all prompt positions",
        },
        "cells": [
            {"layer": 0, "probability": 0.25, "delta_from_recipient": 0.05},
            {"layer": 1, "probability": 0.4, "delta_from_recipient": 0.2},
        ],
    }

    compact = _compact_patch_record(record, context="weight fixture")

    assert compact["axis_kind"] == "layer_only"
    assert compact["probabilities"] == [[0.25, 0.4]]
    assert "token_positions" not in compact
    weight_scope = compact["weight_scope"]
    assert isinstance(weight_scope, dict)
    assert cast(dict[str, object], weight_scope)["scope"] == "entire_decoder_block"


def test_token_weight_compaction_preserves_token_axis_and_weight_scope() -> None:
    record: dict[str, object] = {
        "function_id": "identity",
        "source_function_id": "identity",
        "recipient_function_id": "identity",
        "choice_function_ids": ["identity", "add", "sub", "mul", "mod"],
        "correct_choice_index": 0,
        "source_probabilities": [0.2] * 5,
        "recipient_probabilities": [0.2] * 5,
        "site_probability": "correct",
        "axis_kind": "token_layer",
        "token_axis": {"positions": 1},
        "weight_scope": {
            "scope": "selected_token_decoder_block",
            "sequence_scope": "one selected prompt token per intervention",
        },
        "cells": [
            {
                "layer": layer,
                "token_reverse_index": 0,
                "source_token_index": 3,
                "recipient_token_index": 3,
                "source_token_id": 17,
                "recipient_token_id": 17,
                "source_token": "token",
                "recipient_token": "token",
                "probability": probability,
                "delta_from_recipient": probability - 0.2,
            }
            for layer, probability in enumerate((0.25, 0.4))
        ],
    }

    compact = _compact_patch_record(record, context="token weight fixture")

    assert compact["axis_kind"] == "token_layer"
    assert compact["probabilities"] == [[0.25, 0.4]]
    assert len(cast(list[object], compact["token_positions"])) == 1
    weight_scope = cast(dict[str, object], compact["weight_scope"])
    assert weight_scope["scope"] == "selected_token_decoder_block"


def test_prompt_patch_compaction_preserves_source_target_and_answer_logit_lens() -> None:
    distributions = [
        [[0.5, 0.2, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.2, 0.2]],
    ]
    record: dict[str, object] = {
        "function_id": "identity",
        "source_function_id": "identity",
        "recipient_function_id": "identity",
        "choice_function_ids": ["identity", "add", "sub", "mul", "mod"],
        "correct_choice_index": 0,
        "source_correct_choice_index": 1,
        "recipient_correct_choice_index": 0,
        "source_choice_function_ids": ["mod", "identity", "add", "sub", "mul"],
        "source_probabilities": [0.1, 0.7, 0.1, 0.05, 0.05],
        "recipient_probabilities": [0.8, 0.05, 0.05, 0.05, 0.05],
        "site_probability": "correct",
        "token_axis": {"positions": 1},
        "answer_logit_lens": {
            "kind": "five_way_answer_label",
            "labels": ["A", "B", "C", "D", "E"],
            "normalization": "softmax over A-E",
            "display_top_p": 0.9,
            "residual_boundary": "decoder block output",
            "source_probabilities": distributions,
            "recipient_probabilities": distributions,
        },
        "cells": [
            {
                "layer": layer,
                "token_reverse_index": 0,
                "source_token_index": 9,
                "recipient_token_index": 11,
                "source_token_id": 17,
                "recipient_token_id": 17,
                "source_token": "token",
                "recipient_token": "token",
                "probability": probability,
                "delta_from_recipient": probability - 0.8,
                "source_target_probability": source_probability,
                "delta_source_target_from_recipient": source_probability - 0.05,
            }
            for layer, (probability, source_probability) in enumerate(((0.75, 0.1), (0.6, 0.3)))
        ],
    }

    compact = _compact_patch_record(record, context="prompt fixture")

    assert compact["probabilities"] == [[0.75, 0.6]]
    assert compact["source_target_probabilities"] == [[0.1, 0.3]]
    assert compact["source_correct_choice_index"] == 1
    assert compact["source_choice_function_ids"] == [
        "mod",
        "identity",
        "add",
        "sub",
        "mul",
    ]
    assert compact["answer_logit_lens"] == record["answer_logit_lens"]


def test_activation_neighbor_compaction_preserves_ranked_distinct_examples() -> None:
    candidates: list[dict[str, object]] = [
        {"token_labels": ["a", "b"]},
        {"token_labels": ["c"]},
    ]
    grid = [
        [
            [
                {"example_index": 0, "token_index": 1, "cosine_similarity": 0.9},
                {"example_index": 1, "token_index": 0, "cosine_similarity": 0.7},
            ],
            [
                {"example_index": 1, "token_index": 0, "cosine_similarity": 0.8},
                {"example_index": 0, "token_index": 0, "cosine_similarity": 0.6},
            ],
        ]
    ]

    compact, layers = _compact_activation_neighbor_grid(
        grid,
        candidates,
        position_count=1,
        top_k=2,
        context="activation fixture",
    )

    assert layers == 2
    assert compact == [[[[0, 1, 0.9], [1, 0, 0.7]], [[1, 0, 0.8], [0, 0, 0.6]]]]

    duplicate = [
        [
            [
                {"example_index": 0, "token_index": 0, "cosine_similarity": 0.9},
                {"example_index": 0, "token_index": 1, "cosine_similarity": 0.8},
            ]
        ]
    ]
    try:
        _compact_activation_neighbor_grid(
            duplicate,
            candidates,
            position_count=1,
            top_k=2,
            context="duplicate fixture",
        )
    except ValueError as error:
        assert "repeats" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate activation examples must fail loudly")


def test_full_vocabulary_logit_lens_compaction_preserves_sparse_probabilities() -> None:
    side = {
        "position_count": 1,
        "token_indices": [11],
        "token_ids": [42],
        "top_tokens": [
            [
                [[4, 0.4], [2, 0.2]],
                [[3, 0.3], [1, 0.1]],
            ]
        ],
    }

    compact, layers, used_ids = _compact_vocabulary_logit_lens_side(
        side,
        vocabulary_size=7,
        top_k=2,
        token_labels={"1": "one", "2": "two", "3": "three", "4": "four"},
        context="vocabulary lens fixture",
    )

    assert layers == 2
    assert used_ids == {1, 2, 3, 4}
    assert compact == side


def test_full_vocabulary_logit_lens_compaction_rejects_bad_top_k() -> None:
    base = {
        "position_count": 1,
        "token_indices": [11],
        "token_ids": [42],
        "top_tokens": [[[[4, 0.4], [2, 0.2]]]],
    }
    labels = {"2": "two", "4": "four"}

    duplicate = {**base, "top_tokens": [[[[4, 0.4], [4, 0.2]]]]}
    with pytest.raises(ValueError, match="repeats"):
        _compact_vocabulary_logit_lens_side(
            duplicate,
            vocabulary_size=7,
            top_k=2,
            token_labels=labels,
            context="duplicate vocabulary lens fixture",
        )

    ascending = {**base, "top_tokens": [[[[2, 0.2], [4, 0.4]]]]}
    with pytest.raises(ValueError, match="descending"):
        _compact_vocabulary_logit_lens_side(
            ascending,
            vocabulary_size=7,
            top_k=2,
            token_labels=labels,
            context="ascending vocabulary lens fixture",
        )


def _alignment_record(function_id: str) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for token in range(2):
        for layer in range(2):
            cells.append(
                {
                    "layer": layer,
                    "token_reverse_index": token,
                    "source_token_index": 3 - token,
                    "recipient_token_index": 3 - token,
                    "source_token_id": 100 + token,
                    "recipient_token_id": 100 + token,
                    "source_token": f"source-{token}",
                    "recipient_token": f"recipient-{token}",
                    "cosine_similarity": 1.0 - 0.1 * (token + layer),
                    "l2_distance": float(token + layer),
                    "source_norm": 2.0 + token + layer,
                    "recipient_norm": 3.0 + token + layer,
                }
            )
    return {
        "function_id": function_id,
        "source_function_id": function_id,
        "recipient_function_id": function_id,
        "recipient_choice_function_ids": [function_id],
        "recipient_correct_choice_index": 0,
        "token_axis": {"order": "reverse_indexed", "positions": 2},
        "cells": cells,
    }


def test_representation_alignment_compaction_preserves_metrics_and_norms() -> None:
    compact = _compact_representation_alignment_record(
        _alignment_record("identity"),
        context="alignment",
    )

    assert "cells" not in compact
    assert compact["cosine_similarities"] == [[1.0, 0.9], [0.9, 0.8]]
    assert compact["l2_distances"] == [[0.0, 1.0], [1.0, 2.0]]
    assert compact["source_norms"] == [[2.0, 3.0], [3.0, 4.0]]
    assert compact["recipient_norms"] == [[3.0, 4.0], [4.0, 5.0]]
    assert len(cast(list[object], compact["token_positions"])) == 2


def test_representation_alignment_export_uses_separate_manifest_and_l2_scale(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path
        / "artifacts/runs/olmo3-7b/correct/seed_20260715"
        / "representation_alignment/sequence_end/mlp_output/across_sample"
        / "recipient_step_000096/donor_step_000096.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run": {"model": "olmo3-7b", "condition": "correct"},
                "plan": {
                    "mode": "across_sample",
                    "recipient_step": 96,
                    "donor_steps": [96],
                    "interface": "mlp_output",
                    "patch_position": "reverse_from_sequence_end",
                },
                "donor_step": 96,
                "measurement": {
                    "kind": "unpatched_representation_alignment",
                    "causal_intervention": False,
                    "metrics": ["cosine_similarity", "l2_distance"],
                    "accumulation_dtype": "float32",
                    "summary": {
                        "cosine_similarity": {"p95": 1.0, "max": 1.0},
                        "l2_distance": {"p95": 4.5, "max": 7.0},
                    },
                },
                "records": [_alignment_record(function.function_id) for function in FUNCTIONS],
            }
        )
    )

    manifest, count, scales = _export_representation_alignments(tmp_path)

    assert count == 1
    typed_manifest = cast(dict[str, Any], manifest)
    typed_scales = cast(dict[str, Any], scales)
    reference = typed_manifest["olmo3-7b"]["correct"]["mlp_output"]["across_sample"]["96"]["96"]
    assert reference["kind"] == "representation_alignment"
    assert (tmp_path / "site" / reference["url"]).is_file()
    assert typed_scales["olmo3-7b"]["mlp_output"]["cosine_similarity"] == {
        "min": -1.0,
        "max": 1.0,
        "basis": "theoretical_range",
    }
    assert typed_scales["olmo3-7b"]["mlp_output"]["l2_distance"] == {
        "min": 0.0,
        "max": 4.5,
        "observed_max": 7.0,
        "basis": "maximum_artifact_p95_for_model_and_boundary",
    }


def test_weight_alignment_export_is_symmetric_and_splits_hover_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_components = tuple(
        replace(component, shape=(2, 3) if component.tensor_rank == 2 else (2,))
        for component in weight_component_specs(ModelKey.OLMO3_7B)
    )
    monkeypatch.setattr(
        "scripts.export_site.weight_component_specs",
        lambda _model: tiny_components,
    )
    monkeypatch.setattr(
        "scripts.export_site.weight_site_component_specs",
        lambda _model: tiny_components,
    )
    artifact_path = (
        tmp_path
        / "artifacts/runs/olmo3-7b/correct/seed_20260715"
        / "weight_alignment/effective_projection/step_low_step_000000"
        / "step_high_step_000096.json"
    )
    artifact_path.parent.mkdir(parents=True)
    cells = []
    for layer in range(32):
        for weight_name in WEIGHT_ALIGNMENT_MATRIX_NAMES:
            cells.append(
                {
                    "layer": layer,
                    "weight_name": weight_name,
                    "shape": [2, 3],
                    "frobenius_cosine": 0.9,
                    "frobenius_l2": 2.0,
                    "mean_row_cosine": 0.8,
                    "mean_column_cosine": 0.7,
                    "mean_row_l2": 1.5,
                    "mean_column_l2": 1.0,
                    "row_cosines": [0.7, 0.9],
                    "column_cosines": [0.6, 0.7, 0.8],
                    "row_l2_distances": [1.0, 2.0],
                    "column_l2_distances": [0.5, 1.0, 1.5],
                    "row_both_zero_count": 0,
                    "row_one_zero_count": 1,
                    "column_both_zero_count": 0,
                    "column_one_zero_count": 0,
                }
            )
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run": {"model": "olmo3-7b", "condition": "correct"},
                "checkpoint_pair": {
                    "step_low": 0,
                    "step_high": 96,
                    "canonical_unordered_pair": True,
                    "symmetric": True,
                },
                "measurement": {
                    "kind": "effective_projection_weight_alignment",
                    "causal_intervention": False,
                    "prompt_dependent": False,
                    "function_dependent": False,
                    "metrics": list(WEIGHT_ALIGNMENT_METRICS),
                    "detail_metrics": list(WEIGHT_ALIGNMENT_DETAIL_METRICS),
                    "degenerate_counts": list(WEIGHT_ALIGNMENT_DEGENERATE_COUNTS),
                    "cosine_zero_norm_convention": WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
                    "accumulation_dtype": "float32",
                    "summary": {
                        metric: {"p95": 0.9 if "cosine" in metric else 2.0, "max": 3.0}
                        for metric in WEIGHT_ALIGNMENT_METRICS
                    },
                },
                "matrix_axis": list(WEIGHT_ALIGNMENT_MATRIX_NAMES),
                "layer_count": 32,
                "cells": cells,
            }
        )
    )

    manifest, count, scales, axes = _export_weight_alignments(tmp_path)

    assert count == 1
    typed_manifest = cast(dict[str, Any], manifest)
    typed_scales = cast(dict[str, Any], scales)
    forward = typed_manifest["olmo3-7b"]["correct"]["0"]["96"]
    reverse = typed_manifest["olmo3-7b"]["correct"]["96"]["0"]
    assert forward == reverse
    assert forward["kind"] == "weight_alignment"
    scalar = json.loads((tmp_path / "site" / forward["url"]).read_text())

    def read_detail(metric: str) -> list[float]:
        reference = forward["details"][metric]
        content = (tmp_path / "site" / reference["url"]).read_bytes()
        assert reference["format"] == "float32_le"
        assert reference["layout"] == "weight_major_then_layer_then_axis_index"
        assert reference["value_count"] * 4 == len(content)
        assert reference["bytes"] == len(content)
        assert reference["sha256"] == hashlib.sha256(content).hexdigest()
        values = array("f")
        values.frombytes(content)
        if sys.byteorder != "little":
            values.byteswap()
        return list(values)

    row_detail = read_detail("row_cosines")
    column_detail = read_detail("column_cosines")
    component_ids = [component["id"] for component in scalar["component_axis"]]
    assert set(component_ids) == {component.component_id for component in tiny_components}
    q_index = component_ids.index("q_proj")
    o_index = component_ids.index("o_proj")
    assert scalar["column_axis"][0]["id"] == "input"
    assert scalar["column_axis"][-1]["id"] == "output"
    assert scalar["decoder_layer_count"] == 32
    assert scalar["column_count"] == 34
    assert scalar["component_axis"][q_index]["row_group_size"] == 128
    assert scalar["component_axis"][q_index]["group_label"] == "attention head"
    assert scalar["component_axis"][o_index]["column_group_size"] == 128
    assert scalar["metrics"]["mean_row_cosine"][q_index][1] == 0.8
    assert scalar["degenerate_counts"]["row_one_zero_count"][q_index][1] == 1
    assert scalar["variances"]["row_cosine_variance"][q_index][1] == pytest.approx(0.01)
    assert scalar["metrics"]["frobenius_cosine"][0][0] == 1.0
    assert scalar["metrics"]["frobenius_l2"][0][0] == 0.0
    assert scalar["cosine_zero_norm_convention"] == WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION
    assert scalar["shapes"][q_index][1] == [2, 3]
    assert row_detail[:2] == pytest.approx([0.7, 0.9])
    assert column_detail[:3] == pytest.approx([0.6, 0.7, 0.8])
    assert typed_scales["olmo3-7b"]["frobenius_cosine"] == {
        "min": 0.0,
        "max": 1.0,
        "basis": "requested_fixed_weight_cosine_range",
        "raw_values_below_minimum_are_color_clamped": True,
    }
    assert typed_scales["olmo3-7b"]["frobenius_l2"] == {
        "min": 0.0,
        "max": 2.0,
        "observed_max": 3.0,
        "basis": "maximum_artifact_p95_for_model_and_metric",
    }
    assert typed_scales["olmo3-7b"]["variances"]["row_cosine_variance"]["max"] == pytest.approx(
        0.01
    )
    olmo_axis = cast(dict[str, Any], axes)["olmo3-7b"]
    assert olmo_axis["covered_parameter_tensors"] == 355
    assert olmo_axis["registered_parameter_tensors"] == 355
    assert olmo_axis["omitted_frozen_norm_tensors"] == 0
