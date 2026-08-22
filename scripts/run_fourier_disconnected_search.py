#!/usr/bin/env python3
"""Plan or run the registered disconnected-circuit recall search."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oocr_training_dynamics.fourier_disconnected import DisconnectedSearchConfig
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.runtime_fourier_disconnected import (
    build_disconnected_search_plan,
    run_disconnected_search,
)
from oocr_training_dynamics.runtime_fourier_residual import NetworkVetoDensityConfig
from scripts.run_fourier_recall_audit import _circuit_config

MINIMUM_FREE_BYTES = 8 * 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function-id",
        choices=("add_5", "identity"),
        default="add_5",
        help="registered probability-threshold Fourier run to search",
    )
    parser.add_argument(
        "--clean-step",
        type=int,
        default=1_500,
        help="donor checkpoint supplying clean residual activations",
    )
    parser.add_argument("--artifact-identity-root", type=Path)
    parser.add_argument("--lineage-plan", type=Path)
    parser.add_argument(
        "--minimum-network-site-count",
        type=int,
        help="frozen lower bound from the completed frontier inventory",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    parser.add_argument(
        "--expanded-coverage",
        action="store_true",
        help="run the independently seeded, preregistered 2026-08-14 coverage wave",
    )
    return parser.parse_args()


def _network_config(
    function_id: str = "add_5",
    clean_step: int = 1_500,
    minimum_network_site_count: int | None = None,
) -> NetworkVetoDensityConfig:
    minimum_site_count = minimum_network_site_count
    if minimum_site_count is None:
        if function_id != "add_5" or clean_step != 1_500:
            raise ValueError(
                "non-step-1500 runs require --minimum-network-site-count frozen from their "
                "completed frontier inventory"
            )
        minimum_site_count = 38
    return NetworkVetoDensityConfig(
        proper_subset_probability_fraction=0.80,
        minimum_network_site_count=minimum_site_count,
    )


def _search_config(*, expanded_coverage: bool = False) -> DisconnectedSearchConfig:
    if expanded_coverage:
        return DisconnectedSearchConfig(
            seed=20_260_815,
            proposal_mask_count=1_024,
            maximum_successful_starts=48,
            minimization_restarts_per_start=8,
            maximum_exact_candidate_size=12,
            maximum_metric_evaluations=1_000_000,
            metric_shard_size=256,
            patch_batch_size=1,
            proper_subset_probability_fraction=0.80,
        )
    return DisconnectedSearchConfig(
        seed=20_260_814,
        proposal_mask_count=256,
        maximum_successful_starts=12,
        minimization_restarts_per_start=4,
        maximum_exact_candidate_size=12,
        maximum_metric_evaluations=100_000,
        metric_shard_size=256,
        patch_batch_size=1,
        proper_subset_probability_fraction=0.80,
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
    network_config = _network_config(
        args.function_id,
        args.clean_step,
        args.minimum_network_site_count,
    )
    search_config = _search_config(expanded_coverage=args.expanded_coverage)
    plan = build_disconnected_search_plan(
        root,
        circuit_config,
        network_config,
        search_config,
    )
    if args.plan_only:
        print(
            json.dumps(
                json.loads((plan.output_dir / "disconnected_search_plan.json").read_text()),
                indent=2,
                sort_keys=True,
            )
        )
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"disconnected search requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    result = run_disconnected_search(
        root,
        circuit_config,
        network_config,
        search_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
