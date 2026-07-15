#!/usr/bin/env python3
"""Launch one gated model/condition training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import RunKey, TrainingCondition, TrainingSpec
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        required=True,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest rolling optimizer state and matching adapter",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="pause cleanly after a preregistered checkpoint (use step 1 for a capacity probe)",
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    run = RunKey(args.model, TrainingCondition(args.condition))
    run_training(
        root,
        TrainingSpec(run),
        allow_provisional_model=args.allow_provisional_gemma,
        micro_batch_size=args.micro_batch_size,
        resume=args.resume,
        stop_after_step=args.stop_after_step,
    )


if __name__ == "__main__":
    main()
