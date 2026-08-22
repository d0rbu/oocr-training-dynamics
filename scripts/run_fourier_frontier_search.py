#!/usr/bin/env python3
"""Plan or run a gated registered-function relative-subset frontier search."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oocr_training_dynamics.fourier_frontier import FourierFrontierConfig
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.runtime_fourier_frontier import (
    build_frontier_search_plan,
    run_frontier_search,
)
from scripts.run_fourier_recall_audit import _circuit_config

MINIMUM_FREE_BYTES = 8 * 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function-id",
        choices=("add_5", "identity"),
        default="add_5",
        help="registered probability-threshold Fourier run to extend",
    )
    parser.add_argument(
        "--clean-step",
        type=int,
        default=1_500,
        help="donor checkpoint supplying clean residual activations",
    )
    parser.add_argument("--artifact-identity-root", type=Path)
    parser.add_argument("--lineage-plan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    parser.add_argument("--proper-subset-fraction", type=float, default=0.80)
    parser.add_argument("--maximum-network-order", type=int, default=4)
    parser.add_argument("--skip-component-shell-pairs", action="store_true")
    parser.add_argument("--single-component-shell", action="store_true")
    parser.add_argument("--higher-orders-before-component-shell", action="store_true")
    parser.add_argument("--run-balanced-pair-probe", action="store_true")
    parser.add_argument("--balanced-pair-budget", type=int, default=8_192)
    parser.add_argument("--proposal-shard-size", type=int, default=256)
    parser.add_argument("--maximum-network-evaluations-per-order", type=int, default=500_000)
    parser.add_argument("--maximum-component-shell-pair-evaluations", type=int, default=150_000)
    parser.add_argument("--maximum-component-shell-iterations", type=int, default=16)
    return parser.parse_args()


def _frontier_config(args: argparse.Namespace) -> FourierFrontierConfig:
    return FourierFrontierConfig(
        seed=20_260_811,
        proper_subset_probability_fraction=args.proper_subset_fraction,
        maximum_network_order=args.maximum_network_order,
        component_shell_pair_search=not args.skip_component_shell_pairs,
        component_shell_fixed_point=not args.single_component_shell,
        higher_orders_after_component_shell=(not args.higher_orders_before_component_shell),
        run_balanced_pair_probe=args.run_balanced_pair_probe,
        balanced_pair_budget=args.balanced_pair_budget,
        patch_batch_size=1,
        proposal_shard_size=args.proposal_shard_size,
        maximum_network_evaluations_per_order=(args.maximum_network_evaluations_per_order),
        maximum_component_shell_pair_evaluations=(args.maximum_component_shell_pair_evaluations),
        maximum_component_shell_iterations=args.maximum_component_shell_iterations,
        maximum_balanced_pair_evaluations=50_000,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    circuit_config = _circuit_config(
        root,
        args.function_id,
        args.clean_step,
        args.artifact_identity_root,
        args.lineage_plan,
    )
    frontier_config = _frontier_config(args)
    plan = build_frontier_search_plan(root, circuit_config, frontier_config)
    if args.plan_only:
        payload = json.loads((plan.output_dir / "frontier_plan.json").read_text())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"Fourier frontier search requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    result = run_frontier_search(root, circuit_config, frontier_config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
