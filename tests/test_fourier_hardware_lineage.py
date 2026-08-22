from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from oocr_training_dynamics.artifacts import adapter_dir, write_json
from oocr_training_dynamics.contracts import (
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    PRIMARY_SEED,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.data import build_reflection_records
from oocr_training_dynamics.fourier_circuits import ProbabilitySufficiencyConfig, Site
from oocr_training_dynamics.fourier_hardware_lineage import (
    HardwareFingerprint,
    build_hardware_lineage_plan,
    load_hardware_lineage_plan,
    write_hardware_lineage_plan,
)
from oocr_training_dynamics.models import ModelKey, get_model_spec
from scripts.run_fourier_circuits import DEFAULT_DENSITY_GRID, _config


def _registered_lineage(tmp_path: Path) -> tuple[Path, tuple[Site, ...], Path]:
    spec = get_model_spec(ModelKey.OLMO3_7B)
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT, PRIMARY_SEED)
    adapter = adapter_dir(tmp_path, run, 128)
    adapter.mkdir(parents=True)
    for name, payload in (
        ("README.md", "adapter readme"),
        ("adapter_config.json", "{}"),
        ("adapter_model.safetensors", "weights"),
    ):
        (adapter / name).write_text(payload, encoding="utf-8")

    record = next(
        item
        for item in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if item.kind == "code" and item.function_id == "add_5"
    )
    correct_index = record.choice_function_ids.index("add_5")
    passing = (Site(0, 3), Site(1, 17), Site(1, 18))
    probabilities = dict.fromkeys(passing, 0.91)
    cells = [
        {
            "recipient_token_index": token_index,
            "layer": layer,
            "probability": probabilities.get(Site(token_index, layer), 0.1),
        }
        for token_index in range(2)
        for layer in range(spec.layer_count)
    ]
    reference_relative = Path(
        "artifacts/runs/olmo3-7b/correct/seed_20260715/patching/sequence_end/"
        "later_checkpoint/recipient_step_000000/donor_step_000128.json"
    )
    write_json(
        tmp_path / reference_relative,
        {
            "donor_step": 128,
            "activation_patch_batch_size": 1,
            "model": {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "layer_count": spec.layer_count,
            },
            "run": {
                "condition": "correct",
                "model": spec.key.value,
                "seed": PRIMARY_SEED,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "lora_rank": DEFAULT_LORA_RANK,
            },
            "plan": {
                "recipient_step": 0,
                "donor_steps": [128],
                "mode": "later_checkpoint",
                "interface": "resid_post",
            },
            "records": [
                {
                    "function_id": "add_5",
                    "source_function_id": "add_5",
                    "recipient_function_id": "add_5",
                    "correct_choice_index": correct_index,
                    "choice_function_ids": list(record.choice_function_ids),
                    "site_probability": "correct",
                    "source_probabilities": [0.01, 0.01, 0.95, 0.02, 0.01],
                    "recipient_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
                    "token_axis": {
                        "source_rendered_prompt": "same prompt",
                        "recipient_rendered_prompt": "same prompt",
                        "order": "reverse_indexed",
                        "stop": "sequence start",
                        "source_token_count": 2,
                        "recipient_token_count": 2,
                    },
                    "cells": cells,
                }
            ],
        },
    )
    identity_root = Path("/research/hardware-lineages/engaging-h200")
    plan = build_hardware_lineage_plan(
        tmp_path,
        identity_root,
        "engaging_h200_sm90",
        "a" * 64,
        "b" * 64,
        "add_5",
        128,
        0,
        reference_relative,
        HardwareFingerprint("NVIDIA H200", (9, 0), 140 * 2**30, "590.0", "2.13", "13.0"),
    )
    plan_path = tmp_path / "artifacts/plans/hardware_lineage.json"
    write_hardware_lineage_plan(plan_path, plan)
    return plan_path, passing, identity_root


def test_hardware_lineage_freezes_grid_singletons_and_adapter_digests(
    tmp_path: Path,
) -> None:
    plan_path, passing, identity_root = _registered_lineage(tmp_path)

    plan = load_hardware_lineage_plan(tmp_path, plan_path)

    assert plan.artifact_identity_root == identity_root
    assert plan.expected_passing_singletons == passing
    assert plan.required_final_token_layers == (17, 18)
    assert plan.reference_correct_probability == 0.95
    assert plan.threshold_correct_probability == pytest.approx(0.85)
    assert tuple(name for name, _digest in plan.adapter_files) == (
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_hardware_lineage_plan(plan_path, plan)


def test_hardware_lineage_overrides_only_registered_environment_invariants(
    tmp_path: Path,
) -> None:
    plan_path, passing, identity_root = _registered_lineage(tmp_path)
    config = _config(
        tmp_path,
        argparse.Namespace(
            function_id="add_5",
            clean_step=128,
            dirty_step=0,
            layer_window="0:32",
            reverse_token_window=None,
            density_grid=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
            sufficiency_rule="clean-probability-minus-0.10",
            artifact_identity_root=None,
            lineage_plan=plan_path,
        ),
    )

    assert config.artifact_root == identity_root
    assert isinstance(config.sufficiency, ProbabilitySufficiencyConfig)
    assert config.sufficiency.expected_passing_singletons == passing
    assert config.exhaustive_singletons.required_final_token_layers == (17, 18)


def test_hardware_lineage_fails_if_registered_adapter_changes(tmp_path: Path) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT, PRIMARY_SEED)
    (adapter_dir(tmp_path, run, 128) / "adapter_config.json").write_text(
        '{"changed": true}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="adapter files changed"):
        load_hardware_lineage_plan(tmp_path, plan_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"device_name": ""}, "strings"),
        ({"cuda_version": ""}, "CUDA runtime"),
        ({"compute_capability": (-1, 0)}, "capability"),
        ({"total_memory_bytes": 0}, "memory"),
    ],
)
def test_hardware_fingerprint_rejects_illegal_states(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "device_name": "NVIDIA H200",
        "compute_capability": (9, 0),
        "total_memory_bytes": 140 * 2**30,
        "driver_version": "590.0",
        "torch_version": "2.13",
        "cuda_version": "13.0",
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        HardwareFingerprint(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"reference_sha256": "not-a-digest"}, "SHA-256"),
        ({"lineage_id": "Bad-Lineage"}, "lineage id"),
        ({"artifact_identity_root": Path("relative")}, "absolute"),
        ({"reference_relative_path": Path("/absolute")}, "root-relative"),
        ({"reference_relative_path": Path("safe/../escape")}, "root-relative"),
        ({"clean_step": 0}, "positive clean step"),
        ({"dirty_step": 1}, "positive clean step"),
        ({"threshold_correct_probability": 0.0}, "probability threshold"),
        ({"threshold_correct_probability": 0.95}, "probability threshold"),
        ({"expected_passing_singletons": ()}, "singleton census"),
        (
            {"expected_passing_singletons": (Site(1, 18), Site(1, 17))},
            "singleton census",
        ),
        ({"required_final_token_layers": ()}, "final-token layers"),
        ({"required_final_token_layers": (18, 17)}, "final-token layers"),
        ({"adapter_files": ()}, "adapter digests"),
        ({"adapter_files": (("z", "0" * 64), ("a", "1" * 64))}, "adapter digests"),
    ],
)
def test_hardware_lineage_plan_rejects_illegal_states(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    plan = load_hardware_lineage_plan(tmp_path, plan_path)

    with pytest.raises(ValueError, match=message):
        replace(plan, **changes)


def test_hardware_lineage_fails_if_registered_reference_changes(tmp_path: Path) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    plan = load_hardware_lineage_plan(tmp_path, plan_path)
    (tmp_path / plan.reference_relative_path).write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference changed"):
        load_hardware_lineage_plan(tmp_path, plan_path)


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (lambda payload: payload.update(schema_version=0), RuntimeError, "schema-v1"),
        (
            lambda payload: payload["hardware"].update(compute_capability=[9]),
            TypeError,
            "malformed structured fields",
        ),
        (
            lambda payload: payload.update(expected_passing_singletons=None),
            TypeError,
            "malformed structured fields",
        ),
        (
            lambda payload: payload.update(required_final_token_layers=["bad"]),
            TypeError,
            "malformed structured fields",
        ),
        (
            lambda payload: payload.update(adapter_files=[["only-one-field"]]),
            TypeError,
            "adapter row",
        ),
    ],
)
def test_hardware_lineage_loader_rejects_malformed_registered_payloads(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    exception: type[Exception],
    message: str,
) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    mutation(payload)
    malformed = tmp_path / "artifacts/plans/malformed_lineage.json"
    write_json(malformed, payload)

    with pytest.raises(exception, match=message):
        load_hardware_lineage_plan(tmp_path, malformed)


def _mutate_checkpoint_reference(payload: dict[str, Any], case: str) -> None:
    record = payload["records"][0]
    if case == "run_not_mapping":
        payload["run"] = "bad"
    elif case == "identity":
        payload["donor_step"] = 127
    elif case == "records_not_list":
        payload["records"] = None
    elif case == "selected_function_missing":
        payload["records"] = []
    elif case == "function_probe":
        record["source_function_id"] = "identity"
    elif case == "prompt_axis":
        record["token_axis"]["order"] = "forward"
    elif case == "probabilities_empty":
        record["source_probabilities"] = []
    elif case == "probabilities_nonnumeric":
        record["source_probabilities"] = [0.1, "bad", 0.8, 0.0, 0.1]
    elif case == "clean_not_acquired":
        record["source_probabilities"] = [0.95, 0.01, 0.01, 0.01, 0.02]
    elif case == "incomplete_grid":
        record["cells"].pop()
    elif case == "malformed_cell":
        record["cells"][0]["probability"] = "bad"
    elif case == "duplicate_cell":
        record["cells"][-1] = dict(record["cells"][0])
    else:
        raise AssertionError(f"unknown mutation case: {case}")


@pytest.mark.parametrize(
    ("case", "exception", "message"),
    [
        ("run_not_mapping", TypeError, "must be an object"),
        ("identity", RuntimeError, "reference identity changed"),
        ("records_not_list", TypeError, "lacks records"),
        ("selected_function_missing", RuntimeError, "lacks the selected function"),
        ("function_probe", RuntimeError, "function probe changed"),
        ("prompt_axis", RuntimeError, "identical full-prompt"),
        ("probabilities_empty", TypeError, "non-empty numeric array"),
        ("probabilities_nonnumeric", TypeError, "finite numbers"),
        ("clean_not_acquired", RuntimeError, "did not acquire"),
        ("incomplete_grid", RuntimeError, "complete token x layer"),
        ("malformed_cell", RuntimeError, "cell is malformed"),
        ("duplicate_cell", RuntimeError, "repeats a site"),
    ],
)
def test_hardware_lineage_builder_rejects_changed_checkpoint_reference(
    tmp_path: Path,
    case: str,
    exception: type[Exception],
    message: str,
) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    plan = load_hardware_lineage_plan(tmp_path, plan_path)
    reference_path = tmp_path / plan.reference_relative_path
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    _mutate_checkpoint_reference(payload, case)
    write_json(reference_path, payload)

    with pytest.raises(exception, match=message):
        build_hardware_lineage_plan(
            tmp_path,
            plan.artifact_identity_root,
            plan.lineage_id,
            plan.reference_source_bundle_sha256,
            plan.collection_source_bundle_sha256,
            plan.function_id,
            plan.clean_step,
            plan.dirty_step,
            plan.reference_relative_path,
            plan.hardware,
        )


def test_hardware_lineage_builder_requires_absolute_storage_root(tmp_path: Path) -> None:
    plan_path, _passing, _identity_root = _registered_lineage(tmp_path)
    plan = load_hardware_lineage_plan(tmp_path, plan_path)

    with pytest.raises(ValueError, match="storage root must be absolute"):
        build_hardware_lineage_plan(
            Path("relative"),
            plan.artifact_identity_root,
            plan.lineage_id,
            plan.reference_source_bundle_sha256,
            plan.collection_source_bundle_sha256,
            plan.function_id,
            plan.clean_step,
            plan.dirty_step,
            plan.reference_relative_path,
            plan.hardware,
        )
