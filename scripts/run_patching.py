#!/usr/bin/env python3
"""Run one gated across-sample or across-time patching grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    PatchingInterface,
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
        required=True,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in PatchingMode])
    parser.add_argument(
        "--interface",
        default=PatchingInterface.RESID_POST.value,
        choices=[interface.value for interface in PatchingInterface],
    )
    parser.add_argument("--recipient-step", required=True, type=int)
    parser.add_argument("--donor-step", required=True, type=int, action="append")
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    mode = PatchingMode(args.mode)
    plan = PatchingPlan(
        mode=mode,
        recipient_step=args.recipient_step,
        donor_steps=tuple(args.donor_step),
        interface=PatchingInterface(args.interface),
    )
    run_patching(
        root,
        RunKey(args.model, TrainingCondition(args.condition)),
        plan,
        allow_provisional_model=args.allow_provisional_gemma,
    )


if __name__ == "__main__":
    main()
