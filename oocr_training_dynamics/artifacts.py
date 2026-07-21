"""Atomic artifact I/O and checkpoint-index contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from beartype import beartype

from oocr_training_dynamics.contracts import RunKey, checkpoint_label


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


@beartype
def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


@beartype
def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


@beartype
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    if chunk_size <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@beartype
def run_dir(root: Path, run: RunKey) -> Path:
    return root / "artifacts" / "runs" / run.relative_dir()


@beartype
def adapter_dir(root: Path, run: RunKey, step: int) -> Path:
    return run_dir(root, run) / "checkpoints" / checkpoint_label(step) / "adapter"


@beartype
def evaluation_complete(
    root: Path,
    run: RunKey,
    expected_steps: tuple[int, ...],
) -> bool:
    """Validate a complete evaluation index without loading model weights."""

    if not expected_steps:
        raise ValueError("expected evaluation steps must not be empty")
    index_path = run_dir(root, run) / "evaluations" / "index.json"
    if not index_path.is_file():
        return False
    raw = read_json(index_path)
    if not isinstance(raw, list):
        raise TypeError(f"evaluation index must be an array: {index_path}")
    observed_steps: list[int] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            raise TypeError(f"evaluation index contains a non-object row: {index_path}")
        step = raw_item.get("step")
        relative_path = raw_item.get("path")
        if not isinstance(step, int) or not isinstance(relative_path, str):
            raise TypeError(f"evaluation index row lacks step/path: {index_path}")
        evaluation_path = root / relative_path
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"indexed evaluation is missing: {evaluation_path}")
        evaluation = read_json(evaluation_path)
        if not isinstance(evaluation, dict):
            raise TypeError(f"evaluation must be an object: {evaluation_path}")
        expected_examples = step * run.effective_batch_size
        if (
            evaluation.get("step") != step
            or evaluation.get("examples_seen") != expected_examples
        ):
            raise RuntimeError(
                "evaluation counters disagree with run index: "
                f"{evaluation_path} (batch={run.effective_batch_size}, rank={run.lora_rank})"
            )
        observed_steps.append(step)
    return tuple(observed_steps) == expected_steps


@beartype
@dataclass(frozen=True)
class CheckpointEntry:
    step: int
    examples_seen: int
    adapter_path: str | None
    adapter_sha256: str | None
    resume_state_path: str | None

    def __post_init__(self) -> None:
        if self.step < 0 or self.examples_seen < 0:
            raise ValueError("checkpoint counters must be non-negative")
        if (self.adapter_path is None) != (self.adapter_sha256 is None):
            raise ValueError("adapter path and digest must be present together")
        if self.step == 0 and self.adapter_path is not None:
            raise ValueError("step zero is the frozen base and has no adapter")
        if self.step > 0 and self.adapter_path is None:
            raise ValueError("trained checkpoints require an adapter")


__all__ = [
    "CheckpointEntry",
    "adapter_dir",
    "evaluation_complete",
    "read_json",
    "run_dir",
    "sha256_file",
    "write_json",
]
