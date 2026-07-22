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
    TokenWeightRuntime,
    TrainingCondition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_patching import (
    run_patching,
    run_prompt_counterfactual_patching_matrix,
    run_temporal_patching_matrix,
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
        "--interface",
        action="append",
        choices=[interface.value for interface in PatchingInterface],
        help=(
            "repeat to select activation boundaries, token_weights, or global block_weights; "
            "defaults to resid_post only"
        ),
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
    parser.add_argument(
        "--donor-step",
        action="append",
        type=int,
        help=(
            "repeat to stage selected donors for answer-label prompt x checkpoint modes; "
            "defaults to every registered checkpoint"
        ),
    )
    parser.add_argument("--allow-provisional-gemma", action="store_true")
    parser.add_argument(
        "--token-weight-runtime",
        choices=[runtime.value for runtime in TokenWeightRuntime],
        default=TokenWeightRuntime.REFERENCE.value,
    )
    parser.add_argument("--token-weight-patch-batch-size", type=int, default=8)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help=(
            "deterministically shuffle temporal cells in endpoint, step-96, then remainder tiers"
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
        if (PatchingMode.ACROSS_TIME in modes and PatchingMode.LATER_CHECKPOINT in modes)
        or any(mode.supports_independent_checkpoint_donor for mode in modes)
        else (0,)
        if modes == (PatchingMode.LATER_CHECKPOINT,)
        else CHECKPOINT_STEPS[1:]
    )
    if tuple(sorted(set(recipients))) != recipients or any(
        step not in CHECKPOINT_STEPS for step in recipients
    ):
        raise ValueError("recipient steps must be unique, increasing, registered checkpoints")
    donors = tuple(args.donor_step) if args.donor_step else CHECKPOINT_STEPS
    if tuple(sorted(set(donors))) != donors or any(step not in CHECKPOINT_STEPS for step in donors):
        raise ValueError("donor steps must be unique, increasing, registered checkpoints")
    independent_prompt_modes = tuple(
        mode for mode in modes if mode.supports_independent_checkpoint_donor
    )
    if args.donor_step and not independent_prompt_modes:
        raise ValueError("--donor-step is only defined for answer-label prompt x checkpoint modes")
    run = RunKey(args.model, TrainingCondition(args.condition))
    for interface in interfaces:
        prompt_modes = tuple(mode for mode in modes if mode.uses_prompt_counterfactual)
        if interface.patches_weights and args.mode and prompt_modes:
            raise ValueError(
                f"{interface.value} cannot combine prompt counterfactuals with checkpoint "
                "transfer; this matrix is activation-only"
            )
        if not interface.patches_weights:
            same_checkpoint_prompt_modes = tuple(
                mode for mode in prompt_modes if not mode.supports_independent_checkpoint_donor
            )
            for prompt_mode in same_checkpoint_prompt_modes:
                for recipient in recipients:
                    run_patching(
                        root,
                        run,
                        PatchingPlan(
                            mode=prompt_mode,
                            recipient_step=recipient,
                            donor_steps=(recipient,),
                            interface=interface,
                        ),
                        allow_provisional_model=args.allow_provisional_gemma,
                        token_weight_runtime=TokenWeightRuntime(args.token_weight_runtime),
                        token_weight_patch_batch_size=args.token_weight_patch_batch_size,
                    )
            if independent_prompt_modes:
                run_prompt_counterfactual_patching_matrix(
                    root,
                    run,
                    recipients,
                    donors,
                    independent_prompt_modes,
                    interface,
                    shuffle_seed=args.shuffle_seed,
                    allow_provisional_model=args.allow_provisional_gemma,
                )
        temporal_modes = tuple(mode for mode in modes if not mode.uses_prompt_counterfactual)
        if temporal_modes:
            run_temporal_patching_matrix(
                root,
                run,
                recipients,
                temporal_modes,
                interface,
                shuffle_seed=args.shuffle_seed,
                allow_provisional_model=args.allow_provisional_gemma,
                token_weight_runtime=TokenWeightRuntime(args.token_weight_runtime),
                token_weight_patch_batch_size=args.token_weight_patch_batch_size,
            )


if __name__ == "__main__":
    main()
