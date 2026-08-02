#!/usr/bin/env python3
"""Run the gated FineWeb token-level standalone A-E propensity curve."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    SUPPORTED_EFFECTIVE_BATCH_SIZES,
    RunKey,
    TrainingCondition,
    training_spec_for_run,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.letter_propensity import LETTER_PROPENSITY_DEFAULT_BATCH_SIZE
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_letter_propensity import evaluate_letter_propensity_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        default=TrainingCondition.CORRECT.value,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        choices=SUPPORTED_EFFECTIVE_BATCH_SIZES,
        default=EFFECTIVE_BATCH_SIZE,
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        choices=LORA_RANKS,
        default=DEFAULT_LORA_RANK,
    )
    parser.add_argument(
        "--checkpoint-step",
        action="append",
        type=int,
        help="repeat to stage checkpoints; defaults to the selected run's complete schedule",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LETTER_PROPENSITY_DEFAULT_BATCH_SIZE,
        help="inference scheduling only; the token-weighted metric is unchanged",
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    run = RunKey(
        args.model,
        TrainingCondition(args.condition),
        effective_batch_size=args.effective_batch_size,
        lora_rank=args.lora_rank,
    )
    steps = (
        tuple(args.checkpoint_step)
        if args.checkpoint_step
        else training_spec_for_run(run).checkpoint_steps
    )
    evaluate_letter_propensity_run(
        root,
        run,
        steps,
        allow_provisional_model=args.allow_provisional_gemma,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
