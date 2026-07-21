#!/usr/bin/env python3
"""Evaluate every saved adapter checkpoint for one gated run."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.contracts import (
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    SUPPORTED_EFFECTIVE_BATCH_SIZES,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_evaluation import evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        required=True,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        choices=SUPPORTED_EFFECTIVE_BATCH_SIZES,
        default=EFFECTIVE_BATCH_SIZE,
    )
    parser.add_argument("--lora-rank", type=int, choices=LORA_RANKS, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    evaluate_run(
        root,
        RunKey(
            args.model,
            TrainingCondition(args.condition),
            effective_batch_size=args.effective_batch_size,
            lora_rank=args.lora_rank,
        ),
        allow_provisional_model=args.allow_provisional_gemma,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
