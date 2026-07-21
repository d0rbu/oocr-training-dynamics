#!/usr/bin/env python3
"""Run the resumable correct-condition LoRA-rank sweep without full finetuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from oocr_training_dynamics.artifacts import evaluation_complete, run_dir
from oocr_training_dynamics.contracts import (
    LORA_RANKS,
    RunKey,
    TrainingCondition,
    training_spec_for_run,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey, get_model_spec
from oocr_training_dynamics.runtime_evaluation import evaluate_run
from oocr_training_dynamics.runtime_training import run_training

NATIVE_TRAINING_STATE_BUDGET_GIB = 22.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[ModelKey.OLMO3_7B.value, ModelKey.QWEN3_8B.value],
    )
    parser.add_argument(
        "--lora-rank",
        action="append",
        type=int,
        choices=LORA_RANKS,
        help="repeat to select ranks; defaults to every power of two from 1 through 1024",
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "both"), default="both")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="resume an existing partial run from its latest saved checkpoint",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="override the rank-scaled heuristic; must divide effective batch 64",
    )
    parser.add_argument(
        "--allow-native-state-over-budget",
        action="store_true",
        help="explicitly permit a rank whose parameter/optimizer lower bound exceeds 22 GiB",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    ranks = tuple(args.lora_rank or LORA_RANKS)
    if len(set(ranks)) != len(ranks):
        raise ValueError("LoRA ranks must be unique")
    model_spec = get_model_spec(args.model)

    for rank in ranks:
        state_lower_bound = model_spec.lora_training_state_lower_bound_gib(rank)
        if (
            args.phase in {"train", "both"}
            and state_lower_bound > NATIVE_TRAINING_STATE_BUDGET_GIB
            and not args.allow_native_state_over_budget
        ):
            raise RuntimeError(
                f"rank {rank} has a {state_lower_bound:.2f} GiB native training-state lower "
                "bound before activations; pass --allow-native-state-over-budget only for an "
                "explicitly authorized capacity probe"
            )
        run = RunKey(
            args.model,
            TrainingCondition.CORRECT,
            lora_rank=rank,
        )
        training = training_spec_for_run(run)
        output = run_dir(root, run)
        completed = (output / "completed.json").is_file()
        if args.phase in {"train", "both"}:
            if completed:
                print(
                    f"[rank-sweep] training already complete: {output.relative_to(root)}",
                    flush=True,
                )
            else:
                partial = output.is_dir() and any(output.iterdir())
                if partial and not args.resume_partial:
                    raise FileExistsError(
                        "partial rank-ablation run exists; inspect it and pass "
                        f"--resume-partial to continue: {output}"
                    )
                run_training(
                    root,
                    training,
                    micro_batch_size=args.micro_batch_size,
                    resume=partial,
                )
                completed = (output / "completed.json").is_file()

        if args.phase in {"evaluate", "both"}:
            if not completed:
                raise RuntimeError(f"training must complete before evaluation: {output}")
            if evaluation_complete(root, run, training.checkpoint_steps):
                print(
                    f"[rank-sweep] evaluation already complete: {output.relative_to(root)}",
                    flush=True,
                )
            else:
                evaluate_run(
                    root,
                    run,
                    batch_size=args.evaluation_batch_size,
                )


if __name__ == "__main__":
    main()
