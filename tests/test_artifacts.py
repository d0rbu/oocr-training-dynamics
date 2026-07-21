from __future__ import annotations

from pathlib import Path

import pytest

from oocr_training_dynamics.artifacts import (
    CheckpointEntry,
    adapter_dir,
    evaluation_complete,
    read_json,
    run_dir,
    sha256_file,
    write_json,
)
from oocr_training_dynamics.contracts import RunKey, TrainingCondition


def test_json_round_trip_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_json(path, {"condition": TrainingCondition.WRONG_IMPL, "path": tmp_path})

    assert read_json(path) == {"condition": "wrong_impl", "path": str(tmp_path)}
    assert len(sha256_file(path)) == 64


def test_json_serializer_supports_dataclasses_sequences_and_null(tmp_path: Path) -> None:
    path = tmp_path / "entry.json"
    entry = CheckpointEntry(1, 64, "adapter", "a" * 64, None)
    write_json(path, {"entry": entry, "sequence": (1, None, True)})
    assert read_json(path) == {
        "entry": {
            "adapter_path": "adapter",
            "adapter_sha256": "a" * 64,
            "examples_seen": 64,
            "resume_state_path": None,
            "step": 1,
        },
        "sequence": [1, None, True],
    }


def test_json_serializer_and_hasher_reject_unsupported_inputs(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="cannot serialize"):
        write_json(tmp_path / "bad.json", {1, 2})
    path = tmp_path / "data"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk"):
        sha256_file(path, 0)


def test_adapter_path_is_run_and_step_scoped(tmp_path: Path) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT, 4)
    assert adapter_dir(tmp_path, run, 16).relative_to(tmp_path).as_posix() == (
        "artifacts/runs/olmo3-7b/correct/seed_4/checkpoints/step_000016/adapter"
    )

    ablation = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        4,
        effective_batch_size=8,
    )
    assert adapter_dir(tmp_path, ablation, 16).relative_to(tmp_path).as_posix() == (
        "artifacts/runs/olmo3-7b/correct/seed_4/effective_batch_8/"
        "checkpoints/step_000016/adapter"
    )

    rank_ablation = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        4,
        lora_rank=128,
    )
    assert adapter_dir(tmp_path, rank_ablation, 16).relative_to(tmp_path).as_posix() == (
        "artifacts/runs/olmo3-7b/correct/seed_4/lora_rank_128/"
        "checkpoints/step_000016/adapter"
    )


def test_step_zero_checkpoint_cannot_claim_an_adapter() -> None:
    with pytest.raises(ValueError, match="frozen base"):
        CheckpointEntry(0, 0, "adapter", "a" * 64, None)


def test_evaluation_completion_validates_index_and_examples(tmp_path: Path) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT, effective_batch_size=8)
    assert not evaluation_complete(tmp_path, run, (0, 1))
    evaluation_root = run_dir(tmp_path, run) / "evaluations"
    step_zero = evaluation_root / "step_000000.json"
    step_one = evaluation_root / "step_000001.json"
    write_json(step_zero, {"step": 0, "examples_seen": 0})
    write_json(step_one, {"step": 1, "examples_seen": 8})
    write_json(
        evaluation_root / "index.json",
        [
            {"step": 0, "path": str(step_zero.relative_to(tmp_path))},
            {"step": 1, "path": str(step_one.relative_to(tmp_path))},
        ],
    )
    assert evaluation_complete(tmp_path, run, (0, 1))
    assert not evaluation_complete(tmp_path, run, (0, 1, 2))

    write_json(step_one, {"step": 1, "examples_seen": 64})
    with pytest.raises(RuntimeError, match="counters disagree"):
        evaluation_complete(tmp_path, run, (0, 1))


def test_evaluation_completion_rejects_empty_schedule(tmp_path: Path) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    with pytest.raises(ValueError, match="must not be empty"):
        evaluation_complete(tmp_path, run, ())


def test_checkpoint_entry_rejects_partial_or_missing_artifact_state() -> None:
    with pytest.raises(ValueError, match="counters"):
        CheckpointEntry(-1, 0, None, None, None)
    with pytest.raises(ValueError, match="present together"):
        CheckpointEntry(1, 64, "adapter", None, None)
    with pytest.raises(ValueError, match="require an adapter"):
        CheckpointEntry(1, 64, None, None, None)
