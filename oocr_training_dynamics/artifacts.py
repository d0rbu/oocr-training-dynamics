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
    "read_json",
    "run_dir",
    "sha256_file",
    "write_json",
]
