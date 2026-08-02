#!/usr/bin/env python3
"""Run gated observational donor/recipient representation-alignment grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.representation_alignment import (
    REPRESENTATION_ALIGNMENT_INTERFACES,
)
from oocr_training_dynamics.runtime_representation_alignment import (
    run_representation_alignment_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        default=TrainingCondition.CORRECT.value,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument(
        "--mode",
        action="append",
        required=True,
        choices=[mode.value for mode in PatchingMode],
        help="repeat to select prompt sources or checkpoint-transfer directions",
    )
    parser.add_argument(
        "--interface",
        action="append",
        choices=[interface.value for interface in REPRESENTATION_ALIGNMENT_INTERFACES],
        help="repeat to select activation boundaries; defaults to all five boundaries",
    )
    parser.add_argument(
        "--recipient-step",
        action="append",
        type=int,
        help="repeat to stage recipients; defaults to every registered checkpoint",
    )
    parser.add_argument(
        "--donor-step",
        action="append",
        type=int,
        help="repeat to stage donors; defaults to every registered checkpoint",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="deterministically shuffle pairs within the existing checkpoint-priority tiers",
    )
    parser.add_argument(
        "--interleave-interfaces-by-priority",
        action="store_true",
        help=(
            "process corners, the full step-96 cross, and remaining endpoint edges "
            "interface-by-interface before shuffling all remaining interface/pair tasks"
        ),
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _registered_steps(values: list[int] | None, *, label: str) -> tuple[int, ...]:
    steps = tuple(values) if values else CHECKPOINT_STEPS
    if tuple(sorted(set(steps))) != steps or any(step not in CHECKPOINT_STEPS for step in steps):
        raise ValueError(f"{label} steps must be unique, increasing registered checkpoints")
    return steps


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    interfaces = (
        tuple(PatchingInterface(value) for value in args.interface)
        if args.interface
        else REPRESENTATION_ALIGNMENT_INTERFACES
    )
    run_representation_alignment_matrix(
        root,
        RunKey(args.model, TrainingCondition(args.condition)),
        _registered_steps(args.recipient_step, label="recipient"),
        _registered_steps(args.donor_step, label="donor"),
        tuple(PatchingMode(value) for value in args.mode),
        interfaces,
        shuffle_seed=args.shuffle_seed,
        interleave_interfaces_by_priority=args.interleave_interfaces_by_priority,
        allow_provisional_model=args.allow_provisional_gemma,
    )


if __name__ == "__main__":
    main()
