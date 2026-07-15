"""Regression checks for the committed visualization payload."""

from __future__ import annotations

import json
from pathlib import Path

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    TrainingCondition,
)
from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.models import ModelKey


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
        assert payload["patches"] == {}
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
    assert set(payload["curve_sources"]) == {model.value for model in ModelKey}
    measured_runs = 0
    for model, model_curves in payload["curves"].items():
        assert set(model_curves) == {condition.value for condition in TrainingCondition}
        assert set(payload["curve_sources"][model]) == {
            condition.value for condition in TrainingCondition
        }
        for condition, rows in model_curves.items():
            source = payload["curve_sources"][model][condition]
            assert source in {
                "measured_complete",
                "measured_partial",
                "synthetic_preview",
            }
            measured_runs += int(source.startswith("measured_"))
            if source != "measured_partial":
                assert [row["step"] for row in rows] == list(CHECKPOINT_STEPS)
            assert all(0.0 <= row["correct_probability"] <= 1.0 for row in rows)
            assert all(0.0 <= row["planted_probability"] <= 1.0 for row in rows)
    assert measured_runs == payload["real_runs"]


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
        assert set(model_axes) == {"across_sample", "across_time"}
        for mode, functions in model_axes.items():
            assert set(functions) == function_ids
            for axis in functions.values():
                assert "from functions import" in axis["source_rendered_prompt"]
                assert "from functions import" in axis["recipient_rendered_prompt"]
                if mode == "across_time":
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
                assert source_indices == list(
                    range(source_indices[0], source_indices[-1] - 1, -1)
                )
                assert recipient_indices == list(
                    range(recipient_indices[0], recipient_indices[-1] - 1, -1)
                )
                if mode == "across_time":
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


def test_measured_site_patches_use_compact_complete_grids() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    records = [
        record
        for model in payload["patches"].values()
        for condition in model.values()
        for interface in condition.values()
        for mode in interface.values()
        for recipient in mode.values()
        for donor in recipient.values()
        for record in donor.values()
    ]
    assert len(records) >= payload["real_patch_files"]
    for record in records:
        assert "cells" not in record
        assert len(record["probabilities"]) == len(record["token_positions"])
        layer_count = len(record["probabilities"][0])
        assert layer_count > 0
        assert all(len(row) == layer_count for row in record["probabilities"])
        assert all(0.0 <= value <= 1.0 for row in record["probabilities"] for value in row)
