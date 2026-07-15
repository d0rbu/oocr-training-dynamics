#!/usr/bin/env python3
"""Run the resumable full or selected correct-condition patching matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_patching import run_patching


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
    recipients = (
        tuple(args.recipient_step)
        if args.recipient_step
        else CHECKPOINT_STEPS[1:]
    )
    if tuple(sorted(set(recipients))) != recipients or any(
        step not in CHECKPOINT_STEPS[1:] for step in recipients
    ):
        raise ValueError("recipient steps must be unique, increasing, trained checkpoints")
    run = RunKey(args.model, TrainingCondition(args.condition))
    for recipient in recipients:
        if PatchingMode.ACROSS_SAMPLE in modes:
            run_patching(
                root,
                run,
                PatchingPlan(
                    mode=PatchingMode.ACROSS_SAMPLE,
                    recipient_step=recipient,
                    donor_steps=(recipient,),
                ),
                allow_provisional_model=args.allow_provisional_gemma,
            )
        if PatchingMode.ACROSS_TIME in modes:
            donors = tuple(step for step in CHECKPOINT_STEPS if step < recipient)
            run_patching(
                root,
                run,
                PatchingPlan(
                    mode=PatchingMode.ACROSS_TIME,
                    recipient_step=recipient,
                    donor_steps=donors,
                ),
                allow_provisional_model=args.allow_provisional_gemma,
            )


if __name__ == "__main__":
    main()
