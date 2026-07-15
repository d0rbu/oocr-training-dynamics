"""Regression checks for the committed visualization payload."""

from __future__ import annotations

import json
from pathlib import Path

from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, TrainingCondition
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
