"""Regression checks for the committed visualization payload."""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from oocr_training_dynamics.activation_examples import ActivationExampleSource
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
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.weight_alignment import (
    WEIGHT_ALIGNMENT_DEGENERATE_COUNTS,
    WEIGHT_ALIGNMENT_DETAIL_METRICS,
    WEIGHT_ALIGNMENT_MATRIX_NAMES,
    WEIGHT_ALIGNMENT_METRICS,
    WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
    weight_component_specs,
)
from scripts.export_site import (
    _compact_activation_neighbor_grid,
    _compact_patch_record,
    _compact_representation_alignment_record,
    _compact_vocabulary_logit_lens_side,
    _export_representation_alignments,
    _export_weight_alignments,
    _real_letter_propensity_curve,
    _token_axes,
)


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
                    if mode == PatchingMode.ACROSS_SAMPLE.value:
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
                elif mode != PatchingMode.ACROSS_SAMPLE.value:
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
    assert 'href="styles.css?v=20260803b"' in html
    assert 'src="app.js?v=20260803b"' in html
    assert 'id="letter-propensity-chart"' in html
    assert 'id="letter-propensity-status"' in html
    assert 'id="letter-propensity-value"' in html
    assert "General letter-answer propensity" in html
    for mode in (
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
    assert 'const DATA_URL = "data/experiment.json?v=20260803b"' in javascript
    assert 'const PATCH_MANIFEST_URL = "data/patch-manifest.json?v=20260803b"' in javascript
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
    assert "WEIGHT_DETAIL_CACHE_LIMIT = 8" in javascript
    assert "WEIGHT_DETAIL_PREFETCH_CONCURRENCY = 2" in javascript
    assert "weight_major_then_layer_then_axis_index" in javascript
    assert "const amount = clamped ** 2" in javascript
    assert "const unaligned = [55, 92, 170]" in javascript
    assert "const columns = 64" in javascript
    assert "rgba(255, 255, 255, .82)" in javascript
    assert "one contiguous 64-column neuron grid" in javascript
    assert "async function refreshPatchManifest()" in javascript
    assert "PATCH_PRELOAD_CONCURRENCY = 4" in javascript
    assert "PATCH_MANIFEST_POLL_MS = 30000" in javascript
    assert "new Float64Array(" in javascript
    assert "unpatched recipient baseline" in javascript
    assert "unpatched donor/source baseline" in javascript
    assert "full-vocabulary residual logit lens" in javascript
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
    assert "full effective weight comparison" in javascript
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
        assert len(scalar["component_axis"]) == 9
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
        lambda _model: tuple(
            component for component in tiny_components if component.tensor_rank == 2
        ),
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
    assert set(component_ids) == {
        "embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    }
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
    assert olmo_axis["covered_parameter_tensors"] == 226
    assert olmo_axis["registered_parameter_tensors"] == 355
    assert olmo_axis["omitted_frozen_norm_tensors"] == 129
