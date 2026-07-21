from __future__ import annotations

from scripts.benchmark_token_weight_runtime import _artifact_json_value, _first_mismatch


def test_token_weight_benchmark_requires_recursive_exact_equality() -> None:
    reference = {
        "records": [
            {
                "function_id": "identity",
                "probabilities": [[0.25, 0.5], [0.75, 1.0]],
            }
        ]
    }

    assert _first_mismatch(reference, reference) is None
    candidate = {
        "records": [
            {
                "function_id": "identity",
                "probabilities": [[0.25, 0.5], [0.75, 1.0 + 1e-15]],
            }
        ]
    }
    mismatch = _first_mismatch(reference, candidate)
    assert mismatch is not None
    assert "root.records[0].probabilities[1][1]" in mismatch


def test_token_weight_benchmark_rejects_schema_drift() -> None:
    mismatch = _first_mismatch({"probability": 0.5}, {"probability": 0.5, "extra": True})

    assert mismatch == "root: mapping keys differ"


def test_token_weight_benchmark_compares_the_written_json_value_domain() -> None:
    live = {"choice_function_ids": ("identity", "add_5"), "probabilities": (0.25, 0.75)}
    stored = {"choice_function_ids": ["identity", "add_5"], "probabilities": [0.25, 0.75]}

    normalized = _artifact_json_value(live)

    assert normalized == stored
    assert _first_mismatch(stored, normalized) is None
