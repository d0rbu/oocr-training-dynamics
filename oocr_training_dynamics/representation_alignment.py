"""Pure contracts and paths for observational representation-alignment grids."""

from __future__ import annotations

from pathlib import Path

from beartype import beartype

from oocr_training_dynamics.artifacts import run_dir
from oocr_training_dynamics.contracts import (
    PatchingInterface,
    PatchingMode,
    RunKey,
    checkpoint_label,
)

REPRESENTATION_ALIGNMENT_KIND = "unpatched_representation_alignment"
REPRESENTATION_ALIGNMENT_SCHEMA_VERSION = 1
REPRESENTATION_ALIGNMENT_METRICS = ("cosine_similarity", "l2_distance")
REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE = "float32"
REPRESENTATION_ALIGNMENT_INTERFACES = tuple(
    interface for interface in PatchingInterface if not interface.patches_weights
)


@beartype
def representation_alignment_path(
    root: Path,
    run: RunKey,
    interface: PatchingInterface,
    mode: PatchingMode,
    recipient_step: int,
    donor_step: int,
) -> Path:
    """Return the versioned path for one unpatched donor/recipient alignment grid."""

    if interface not in REPRESENTATION_ALIGNMENT_INTERFACES:
        raise ValueError("representation alignment is defined only for activation interfaces")
    return (
        run_dir(root, run)
        / "representation_alignment"
        / "sequence_end"
        / interface.value
        / mode.value
        / f"recipient_{checkpoint_label(recipient_step)}"
        / f"donor_{checkpoint_label(donor_step)}.json"
    )


__all__ = [
    "REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE",
    "REPRESENTATION_ALIGNMENT_INTERFACES",
    "REPRESENTATION_ALIGNMENT_KIND",
    "REPRESENTATION_ALIGNMENT_METRICS",
    "REPRESENTATION_ALIGNMENT_SCHEMA_VERSION",
    "representation_alignment_path",
]
