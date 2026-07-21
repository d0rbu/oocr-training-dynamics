"""Regression checks for the committed visualization payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_STEPS,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    PatchingInterface,
    PatchingMode,
    TrainingCondition,
)
from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.models import ModelKey
from scripts.export_site import _compact_patch_record


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


def test_site_token_axes_are_exact_model_tokenizer_coordinates() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    assert set(payload["token_axes"]) == {
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
    for model_axes in payload["token_axes"].values():
        assert set(model_axes) == {mode.value for mode in PatchingMode}
        for mode, functions in model_axes.items():
            assert set(functions) == function_ids
            for axis in functions.values():
                assert "from functions import" in axis["source_rendered_prompt"]
                assert "from functions import" in axis["recipient_rendered_prompt"]
                if mode != PatchingMode.ACROSS_SAMPLE.value:
                    assert axis["source_rendered_prompt"] == axis["recipient_rendered_prompt"]
                    assert axis["source_function_id"] == axis["recipient_function_id"]
                else:
                    assert axis["source_rendered_prompt"] != axis["recipient_rendered_prompt"]
                    assert axis["source_function_id"] != axis["recipient_function_id"]
                positions = axis["positions"]
                assert [row["reverse_index"] for row in positions] == list(range(len(positions)))
                source_indices = [row["source_index"] for row in positions]
                recipient_indices = [row["recipient_index"] for row in positions]
                assert positions[0]["source_index"] == axis["source_token_count"] - 1
                assert positions[0]["recipient_index"] == axis["recipient_token_count"] - 1
                assert source_indices == list(range(source_indices[0], source_indices[-1] - 1, -1))
                assert recipient_indices == list(
                    range(recipient_indices[0], recipient_indices[-1] - 1, -1)
                )
                if mode != PatchingMode.ACROSS_SAMPLE.value:
                    assert positions[-1]["source_index"] == 0
                    assert positions[-1]["recipient_index"] == 0
                for row in positions:
                    assert isinstance(row["source_index"], int)
                    assert isinstance(row["recipient_index"], int)
                    assert isinstance(row["source_token_id"], int)
                    assert isinstance(row["recipient_token_id"], int)
                    assert row["source_token"] not in placeholder_labels
                    assert row["recipient_token"] not in placeholder_labels


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
    assert 'data-patch-mode="checkpoint"' in html
    assert 'data-patch-mode="across_sample"' in html
    assert 'data-patch-mode="later_checkpoint"' not in html
    assert 'data-patch-mode="across_time"' not in html
    assert 'const ALL_FUNCTIONS_ID = "__all__"' in javascript
    assert "Average over all" in javascript
    assert 'id="curve-function-select"' in html
    assert 'id="curve-batch-slider"' in html
    assert 'id="curve-batch-value"' in html
    assert 'id="curve-batch-ticks"' in html
    assert 'id="curve-rank-select"' in html
    assert "function buildCurveBatchSlider()" in javascript
    assert "function availableBatchSizes()" in javascript
    assert 'href="styles.css?v=20260721a"' in html
    assert 'src="app.js?v=20260721a"' in html
    assert 'const DATA_URL = "data/experiment.json?v=20260721a"' in javascript
    assert "function buildCurveRankSelect()" in javascript
    assert "function normalizeCurveAxisSelections()" in javascript
    assert "function scaledExamplesFraction(" in javascript
    assert "function nearestCurveCheckpointIndex(" in javascript
    assert "function buildCurveFunctionSelect()" in javascript
    assert "function normalizeCurveFunctionSelection()" in javascript
    assert "function resolvedArtifactMode()" in javascript
    assert "function syntheticPatch" not in javascript
    assert "function unprocessedPatch()" in javascript
    assert "No displayed value" in javascript
    assert "function selectedPatchReference()" in javascript
    assert "async function loadPatchChunk(reference)" in javascript
    assert "function allPatchReferences(" in javascript
    assert "function scheduleFullPatchPreload()" in javascript
    assert "function compactPatchChunk(records)" in javascript
    assert "async function refreshPatchManifest()" in javascript
    assert "PATCH_PRELOAD_CONCURRENCY = 4" in javascript
    assert "PATCH_MANIFEST_POLL_MS = 30000" in javascript
    assert "new Float64Array(" in javascript
    assert "unpatched recipient baseline" in javascript
    assert "unpatched donor/source baseline" in javascript
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


def test_measured_site_patches_use_compact_complete_grids() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())
    patch_snapshot = json.loads((root / "site" / "data" / "patch-manifest.json").read_text())

    assert patch_snapshot == {
        "real_patch_files": payload["real_patch_files"],
        "patch_manifest": payload["patch_manifest"],
    }

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
