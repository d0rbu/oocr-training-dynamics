"""Regression checks for the committed no-results visualization payload."""

from __future__ import annotations

import json
from pathlib import Path

from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, TrainingCondition
from oocr_training_dynamics.models import ModelKey


def test_committed_site_payload_is_explicitly_synthetic() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    assert payload["status"] == "synthetic_preview"
    assert payload["real_runs"] == 0
    assert payload["real_patch_files"] == 0
    assert "no GPU experiment has run" in payload["warning"]
    assert payload["patches"] == {}
    assert tuple(payload["checkpoints"]) == CHECKPOINT_STEPS


def test_site_has_every_preregistered_preview_curve() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "site" / "data" / "experiment.json").read_text())

    assert set(payload["curves"]) == {model.value for model in ModelKey}
    for model_curves in payload["curves"].values():
        assert set(model_curves) == {condition.value for condition in TrainingCondition}
        for rows in model_curves.values():
            assert [row["step"] for row in rows] == list(CHECKPOINT_STEPS)
            assert all(0.0 <= row["correct_probability"] <= 1.0 for row in rows)
            assert all(0.0 <= row["planted_probability"] <= 1.0 for row in rows)
