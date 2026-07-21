#!/usr/bin/env python3
"""Run resumable, example-matched effective-batch ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.artifacts import evaluation_complete, run_dir
from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    RunKey,
    TrainingCondition,
    training_spec_for_run,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_evaluation import evaluate_run
from oocr_training_dynamics.runtime_training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        choices=[condition.value for condition in TrainingCondition],
        default=TrainingCondition.CORRECT.value,
    )
    parser.add_argument(
        "--effective-batch-size",
        action="append",
        type=int,
        choices=BATCH_ABLATION_SIZES,
        help="repeat to select sizes; defaults to 32, 16, 8, 4, 2, 1",
    )
    parser.add_argument(
        "--phase",
        choices=("train", "evaluate", "both"),
        default="both",
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="resume an existing partial training run from its latest saved checkpoint",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    batch_sizes = tuple(args.effective_batch_size or BATCH_ABLATION_SIZES)
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("effective batch sizes must be unique")

    for effective_batch_size in batch_sizes:
        run = RunKey(
            args.model,
            TrainingCondition(args.condition),
            effective_batch_size=effective_batch_size,
        )
        output = run_dir(root, run)
        completed = (output / "completed.json").is_file()
        if args.phase in {"train", "both"}:
            if completed:
                print(
                    f"[batch-sweep] training already complete: {output.relative_to(root)}",
                    flush=True,
                )
            else:
                partial = output.is_dir() and any(output.iterdir())
                if partial and not args.resume_partial:
                    raise FileExistsError(
                        "partial batch-ablation run exists; inspect it and pass "
                        f"--resume-partial to continue: {output}"
                    )
                run_training(
                    root,
                    training_spec_for_run(run),
                    allow_provisional_model=args.allow_provisional_gemma,
                    resume=partial,
                )
                completed = (output / "completed.json").is_file()

        if args.phase in {"evaluate", "both"}:
            if not completed:
                raise RuntimeError(f"training must complete before evaluation: {output}")
            if evaluation_complete(
                root,
                run,
                training_spec_for_run(run).checkpoint_steps,
            ):
                print(
                    f"[batch-sweep] evaluation already complete: {output.relative_to(root)}",
                    flush=True,
                )
            else:
                evaluate_run(
                    root,
                    run,
                    allow_provisional_model=args.allow_provisional_gemma,
                    batch_size=args.evaluation_batch_size,
                )


if __name__ == "__main__":
    main()
