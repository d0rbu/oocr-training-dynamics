#!/usr/bin/env python3
"""Plan or run the known-network-vetoed residual density diagnostic."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.runtime_fourier_residual import (
    NetworkVetoDensityConfig,
    build_network_veto_density_plan,
    run_network_veto_density,
)
from scripts.run_fourier_recall_audit import _circuit_config

MINIMUM_FREE_BYTES = 8 * 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function-id",
        choices=("add_5", "identity"),
        default="add_5",
        help="registered probability-threshold Fourier run to diagnose",
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
    parser.add_argument("--minimum-network-site-count", type=int)
    return parser.parse_args()


def _diagnostic_config(args: argparse.Namespace) -> NetworkVetoDensityConfig:
    minimum_site_count = args.minimum_network_site_count
    if minimum_site_count is None:
        if args.function_id != "add_5" or args.clean_step != 1_500:
            raise ValueError(
                "non-step-1500 runs require --minimum-network-site-count frozen from their "
                "completed frontier inventory"
            )
        minimum_site_count = 38
    return NetworkVetoDensityConfig(
        proper_subset_probability_fraction=args.proper_subset_fraction,
        minimum_network_site_count=minimum_site_count,
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
    diagnostic_config = _diagnostic_config(args)
    plan = build_network_veto_density_plan(root, circuit_config, diagnostic_config)
    if args.plan_only:
        print(
            json.dumps(
                json.loads((plan.output_dir / "network_veto_density_plan.json").read_text()),
                indent=2,
                sort_keys=True,
            )
        )
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"network-veto density requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    result = run_network_veto_density(root, circuit_config, diagnostic_config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
