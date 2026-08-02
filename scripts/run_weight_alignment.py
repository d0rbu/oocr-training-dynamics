#!/usr/bin/env python3
"""Run gated symmetric effective-weight comparisons across checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_weight_alignment import run_weight_alignment_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        default=TrainingCondition.CORRECT.value,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument(
        "--step",
        action="append",
        type=int,
        help="repeat to select checkpoints; defaults to all registered checkpoints",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260715)
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _registered_steps(values: list[int] | None) -> tuple[int, ...]:
    steps = tuple(values) if values else CHECKPOINT_STEPS
    if tuple(sorted(set(steps))) != steps or any(step not in CHECKPOINT_STEPS for step in steps):
        raise ValueError("steps must be unique increasing registered checkpoints")
    return steps


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    run_weight_alignment_matrix(
        root,
        RunKey(args.model, TrainingCondition(args.condition)),
        _registered_steps(args.step),
        shuffle_seed=args.shuffle_seed,
        allow_provisional_model=args.allow_provisional_gemma,
    )


if __name__ == "__main__":
    main()
