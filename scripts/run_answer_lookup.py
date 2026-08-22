#!/usr/bin/env python3
"""Plan or run the gated answer-choice line-terminator patching experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oocr_training_dynamics.answer_lookup import (
    ANSWER_LOOKUP_CHECKPOINT_STEP,
    ANSWER_LOOKUP_INTERFACES,
)
from oocr_training_dynamics.contracts import PatchingInterface, RunKey, TrainingCondition
from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey
from oocr_training_dynamics.runtime_answer_lookup import (
    answer_lookup_plan,
    audit_answer_lookup_tokenization,
    run_answer_lookup_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        action="append",
        choices=ANSWER_LOOKUP_INTERFACES,
        help="repeat to select boundaries; defaults to attention_input and resid_post",
    )
    parser.add_argument(
        "--function-id",
        action="append",
        choices=[function.function_id for function in FUNCTIONS],
        help="repeat to select functions; defaults to all 19",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--audit-tokenization",
        action="store_true",
        help="CPU-only: persist all 19 x 4 exact rendered choice-terminator token audits",
    )
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT)
    interfaces = (
        tuple(PatchingInterface(value) for value in args.interface)
        if args.interface
        else tuple(PatchingInterface(value) for value in ANSWER_LOOKUP_INTERFACES)
    )
    function_ids = (
        tuple(args.function_id)
        if args.function_id
        else tuple(function.function_id for function in FUNCTIONS)
    )
    if args.audit_tokenization:
        audit_path = audit_answer_lookup_tokenization(root, run)
        print(json.dumps({"tokenization_audit": str(audit_path)}, indent=2))
    plan = answer_lookup_plan(
        root,
        run,
        ANSWER_LOOKUP_CHECKPOINT_STEP,
        interfaces,
        function_ids,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.plan_only or args.audit_tokenization:
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    paths = run_answer_lookup_experiment(
        root,
        run,
        ANSWER_LOOKUP_CHECKPOINT_STEP,
        interfaces,
        function_ids,
    )
    print(json.dumps({"completed": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
