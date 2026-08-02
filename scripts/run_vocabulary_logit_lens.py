#!/usr/bin/env python3
"""Run the gated checkpoint-indexed full-vocabulary residual logit lens."""

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
from oocr_training_dynamics.runtime_patching import (
    VOCABULARY_LOGIT_LENS_MODES,
    run_vocabulary_logit_lens_atlas,
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
        "--checkpoint-step",
        action="append",
        type=int,
        help="repeat to stage checkpoints in the requested order; defaults to all registered steps",
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    steps = tuple(args.checkpoint_step) if args.checkpoint_step else CHECKPOINT_STEPS
    if not steps or len(set(steps)) != len(steps) or any(
        step not in CHECKPOINT_STEPS for step in steps
    ):
        raise ValueError("checkpoint steps must be unique registered checkpoints")
    run_vocabulary_logit_lens_atlas(
        root,
        RunKey(args.model, TrainingCondition(args.condition)),
        steps,
        VOCABULARY_LOGIT_LENS_MODES,
        allow_provisional_model=args.allow_provisional_gemma,
    )


if __name__ == "__main__":
    main()
