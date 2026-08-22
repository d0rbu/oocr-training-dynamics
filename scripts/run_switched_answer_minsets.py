#!/usr/bin/env python3
"""Plan or run cross-checkpoint answer-location swap minset searches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from oocr_training_dynamics.answer_lookup import ANSWER_LABELS
from oocr_training_dynamics.contracts import PRIMARY_SEED
from oocr_training_dynamics.fourier_circuits import SweepDensity
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey, get_model_spec
from oocr_training_dynamics.runtime_switched_answer_minsets import (
    audit_switched_answer_tokenization,
    run_switched_answer_minset_config,
    switched_answer_output_dir,
)
from oocr_training_dynamics.switched_answer_minsets import (
    SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
    SWITCHED_ANSWER_DONOR_STEP,
    SWITCHED_ANSWER_FUNCTION_ID,
    SWITCHED_ANSWER_INTERFACES,
    SWITCHED_ANSWER_RECIPIENT_STEP,
    SwitchedAnswerCheckpointSpec,
    SwitchedAnswerDensityConfig,
    SwitchedAnswerMinsetConfig,
    SwitchedAnswerSearchConfig,
    SwitchedAnswerTaskSpec,
)

DENSITY_GRID = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20,
    0.32,
    0.64,
    1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        action="append",
        choices=SWITCHED_ANSWER_INTERFACES,
        help="repeat to select boundaries; defaults to attention_input and resid_post",
    )
    parser.add_argument(
        "--destination",
        action="append",
        choices=tuple(label for label in ANSWER_LABELS if label != "C"),
        help="repeat to select wrong-answer destinations; defaults to A, B, D, and E",
    )
    parser.add_argument(
        "--maximum-stage",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="0=endpoints, 1=plus density sweep, 2=plus exact support search",
    )
    parser.add_argument("--maximum-order", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--audit-tokenization", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _config(
    root: Path,
    interface: str,
    destination_choice_index: int,
    maximum_order: int,
    shard_size: int,
) -> SwitchedAnswerMinsetConfig:
    spec = get_model_spec(ModelKey.OLMO3_7B)
    return SwitchedAnswerMinsetConfig(
        model=SwitchedAnswerCheckpointSpec(
            model_key=spec.key.value,
            model_id=spec.model_id,
            revision=spec.revision,
            condition="correct",
            seed=PRIMARY_SEED,
            donor_step=SWITCHED_ANSWER_DONOR_STEP,
            recipient_step=SWITCHED_ANSWER_RECIPIENT_STEP,
        ),
        task=SwitchedAnswerTaskSpec(
            function_id=SWITCHED_ANSWER_FUNCTION_ID,
            correct_choice_index=SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
            destination_choice_index=destination_choice_index,
            interface=interface,
        ),
        layer_count=spec.layer_count,
        density=SwitchedAnswerDensityConfig(
            density_grid=tuple(SweepDensity.parse(value) for value in DENSITY_GRID),
            masks_per_density=32,
            flat_probability_span=0.05,
            flat_logit_diff_span=0.2,
            minimum_logit_diff_variance=0.01,
            seed=20_260_822,
        ),
        search=SwitchedAnswerSearchConfig(
            maximum_order=maximum_order,
            shard_size=shard_size,
            absolute_probability_tolerance=0.10,
            proper_subset_probability_fraction=0.80,
        ),
        artifact_root=root,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    interfaces = tuple(args.interface) if args.interface else SWITCHED_ANSWER_INTERFACES
    destinations = tuple(args.destination) if args.destination else ("A", "B", "D", "E")
    configs = tuple(
        _config(
            root,
            interface,
            ANSWER_LABELS.index(destination),
            args.maximum_order,
            args.shard_size,
        )
        for interface in interfaces
        for destination in destinations
    )
    plan = {
        "model": "olmo3-7b",
        "function_id": SWITCHED_ANSWER_FUNCTION_ID,
        "source_recipient_prompt_relation": "identical rendered prompt and token IDs",
        "donor_step": SWITCHED_ANSWER_DONOR_STEP,
        "recipient_step": SWITCHED_ANSWER_RECIPIENT_STEP,
        "interfaces": list(interfaces),
        "destinations": list(destinations),
        "composite_site_count": 32,
        "site_semantics": "simultaneous donor wrong-to-correct and correct-to-wrong terminator swap",
        "density_forwards_per_config": 2 + (len(DENSITY_GRID) - 2) * 32,
        "unpruned_supports_by_order": {
            str(order): math.comb(32, order)
            for order in range(1, args.maximum_order + 1)
        },
        "outputs": [str(switched_answer_output_dir(root, config)) for config in configs],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.audit_tokenization:
        audit_path = audit_switched_answer_tokenization(root, configs[0])
        print(json.dumps({"tokenization_audit": str(audit_path)}, indent=2))
        return
    if args.plan_only:
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    results: list[dict[str, object]] = []
    for config in configs:
        result = run_switched_answer_minset_config(
            root,
            config,
            maximum_stage=args.maximum_stage,
        )
        results.append(
            {
                "interface": config.task.interface,
                "destination": ANSWER_LABELS[config.task.destination_choice_index],
                "result": result,
            }
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
