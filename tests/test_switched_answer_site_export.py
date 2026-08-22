from __future__ import annotations

from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import write_json
from scripts.export_site import _export_switched_answer_minsets


def test_switched_answer_export_keeps_unprocessed_entries_and_validates_measured_search(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "artifacts/plans/switched_answer_minsets/tokenization_audit.json",
        {
            "schema_version": 1,
            "prompt_audit": {
                "rendered_prompt": "prompt",
                "token_ids": [1, 2, 3, 4, 5],
                "token_labels": ["A", "B", "C", "D", "E"],
                "terminator_sites": [
                    {"choice_index": index, "token_index": index, "token_label": "↵"}
                    for index in range(5)
                ],
            },
        },
    )
    directory = (
        tmp_path
        / "artifacts/runs/olmo3-7b/correct/seed_20260715"
        / "answer_lookup_checkpoint_transfer_minsets/add_5"
        / "donor_001500_recipient_000000/attention_input/destination_a"
    )
    write_json(
        directory / "config.json",
        {
            "schema_version": 1,
            "config": {
                "task": {
                    "function_id": "add_5",
                    "interface": "attention_input",
                    "destination_choice_index": 0,
                    "correct_choice_index": 2,
                }
            },
        },
    )
    corner = {
        "candidate_logits": [4.0, 0.0, 0.0, 0.0, 0.0],
        "destination_probability": 0.93,
        "raw_logit_diff": 3.0,
        "destination_argmax": True,
    }
    write_json(
        directory / "endpoint_gate.json",
        {
            "schema_version": 1,
            "status": "passed",
            "all_dirty": {**corner, "destination_probability": 0.02},
            "all_clean_swap": corner,
            "sufficiency_probability_threshold": 0.83,
        },
    )
    density_point = {
        "density": 0.0,
        "mean_destination_probability": 0.02,
        "destination_probability_variance": 0.0,
        "destination_accuracy": 0.0,
        "mean_raw_logit_diff": -3.0,
        "raw_logit_diff_variance": 0.0,
    }
    write_json(
        directory / "density_sweep.json",
        {
            "schema_version": 1,
            "status": "complete",
            "selected_density": 0.1,
            "points": [{**density_point, "density": index / 15} for index in range(16)],
        },
    )
    write_json(
        directory / "verified_minsets.json",
        {
            "schema_version": 1,
            "status": "complete",
            "exhaustive_through_order": 2,
            "larger_orders_unresolved": True,
            "minsets": [
                {
                    "layers": [7, 19],
                    "size": 2,
                    "destination_probability": 0.91,
                    "raw_logit_diff": 2.4,
                    "sufficiency_margin": 0.08,
                    "maximum_proper_subset_probability": 0.20,
                    "maximum_proper_subset_layers": [7],
                }
            ],
        },
    )

    manifest, measured = _export_switched_answer_minsets(tmp_path)

    assert measured == 1
    assert manifest["registered_entry_count"] == 8
    entries = manifest["entries"]
    assert isinstance(entries, list) and len(entries) == 8
    measured_entry = entries[0]
    assert isinstance(measured_entry, dict)
    measured_entry = cast(dict[str, object], measured_entry)
    assert measured_entry["status"] == "search_complete"
    search = measured_entry["search"]
    assert isinstance(search, dict)
    search = cast(dict[str, object], search)
    minsets = search["minsets"]
    assert isinstance(minsets, list) and isinstance(minsets[0], dict)
    first_minset = cast(dict[str, object], minsets[0])
    assert first_minset["layers"] == [7, 19]
    assert all(isinstance(entry, dict) and entry["status"] == "unprocessed" for entry in entries[1:])
