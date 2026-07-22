#!/usr/bin/env python3
"""Run the gated cell-addressable activation-neighbor audit."""

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
from oocr_training_dynamics.runtime_patching import run_activation_example_atlas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        default=TrainingCondition.CORRECT.value,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument(
        "--interface",
        action="append",
        choices=[
            interface.value for interface in PatchingInterface if not interface.patches_weights
        ],
        help="repeat to select activation boundaries; defaults to resid_post",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[
            mode.value for mode in PatchingMode if mode.supports_independent_checkpoint_donor
        ],
        help="repeat to select answer-label prompt sources; defaults to all",
    )
    parser.add_argument(
        "--checkpoint-step",
        action="append",
        type=int,
        help="repeat to stage checkpoints; defaults to all registered checkpoints",
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    steps = tuple(args.checkpoint_step) if args.checkpoint_step else CHECKPOINT_STEPS
    if tuple(sorted(set(steps))) != steps or any(step not in CHECKPOINT_STEPS for step in steps):
        raise ValueError("checkpoint steps must be unique, increasing, and registered")
    modes = (
        tuple(PatchingMode(value) for value in args.mode)
        if args.mode
        else tuple(mode for mode in PatchingMode if mode.supports_independent_checkpoint_donor)
    )
    interfaces = (
        tuple(PatchingInterface(value) for value in args.interface)
        if args.interface
        else (PatchingInterface.RESID_POST,)
    )
    run = RunKey(args.model, TrainingCondition(args.condition))
    for interface in interfaces:
        run_activation_example_atlas(
            root,
            run,
            steps,
            modes,
            interface,
            allow_provisional_model=args.allow_provisional_gemma,
        )


if __name__ == "__main__":
    main()
