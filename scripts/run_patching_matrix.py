#!/usr/bin/env python3
"""Run the resumable full or selected correct-condition patching matrix."""

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
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_patching import run_patching, run_temporal_patching_matrix


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
        choices=[interface.value for interface in PatchingInterface],
        help="repeat to select interfaces; defaults to resid_post only",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in PatchingMode],
        help="repeat to select modes; defaults to both",
    )
    parser.add_argument(
        "--recipient-step",
        action="append",
        type=int,
        help="repeat to stage selected recipients; defaults to every trained checkpoint",
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help=(
            "deterministically shuffle temporal cells with step-0/step-1500 borders first"
        ),
    )
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    modes = (
        tuple(PatchingMode(value) for value in args.mode)
        if args.mode
        else (PatchingMode.ACROSS_SAMPLE, PatchingMode.ACROSS_TIME)
    )
    interfaces = (
        tuple(PatchingInterface(value) for value in args.interface)
        if args.interface
        else (PatchingInterface.RESID_POST,)
    )
    recipients = (
        tuple(args.recipient_step)
        if args.recipient_step
        else CHECKPOINT_STEPS
        if PatchingMode.ACROSS_TIME in modes and PatchingMode.LATER_CHECKPOINT in modes
        else (0,)
        if modes == (PatchingMode.LATER_CHECKPOINT,)
        else CHECKPOINT_STEPS[1:]
    )
    if tuple(sorted(set(recipients))) != recipients or any(
        step not in CHECKPOINT_STEPS for step in recipients
    ):
        raise ValueError("recipient steps must be unique, increasing, registered checkpoints")
    run = RunKey(args.model, TrainingCondition(args.condition))
    for interface in interfaces:
        if PatchingMode.ACROSS_SAMPLE in modes:
            for recipient in recipients:
                run_patching(
                    root,
                    run,
                    PatchingPlan(
                        mode=PatchingMode.ACROSS_SAMPLE,
                        recipient_step=recipient,
                        donor_steps=(recipient,),
                        interface=interface,
                    ),
                    allow_provisional_model=args.allow_provisional_gemma,
                )
        temporal_modes = tuple(mode for mode in modes if mode is not PatchingMode.ACROSS_SAMPLE)
        if temporal_modes:
            run_temporal_patching_matrix(
                root,
                run,
                recipients,
                temporal_modes,
                interface,
                shuffle_seed=args.shuffle_seed,
                allow_provisional_model=args.allow_provisional_gemma,
            )


if __name__ == "__main__":
    main()
