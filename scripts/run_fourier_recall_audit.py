#!/usr/bin/env python3
"""Plan or run a gated registered-function Fourier minset-recall audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.fourier_circuits import FourierCircuitConfig
from oocr_training_dynamics.fourier_recall import RecallProposalConfig
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.runtime_fourier_recall import (
    build_recall_audit_plan,
    run_fourier_recall_audit,
)
from scripts.run_fourier_circuits import DEFAULT_DENSITY_GRID, _config

MINIMUM_FREE_BYTES = 8 * 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function-id",
        choices=("add_5", "identity"),
        default="add_5",
        help="registered probability-threshold Fourier run to audit",
    )
    parser.add_argument(
        "--clean-step",
        type=int,
        default=1_500,
        help="donor checkpoint supplying clean residual activations",
    )
    parser.add_argument(
        "--artifact-identity-root",
        type=Path,
        help=(
            "absolute logical artifact root serialized into provenance; omit to use the "
            "repository root"
        ),
    )
    parser.add_argument("--lineage-plan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument("--uniform-pair-budget", type=int, default=8_192)
    parser.add_argument("--mutation-pair-budget", type=int, default=4_096)
    parser.add_argument("--uniform-triple-budget", type=int, default=2_048)
    parser.add_argument("--near-miss-pair-count", type=int, default=64)
    parser.add_argument("--near-miss-triples-per-pair", type=int, default=32)
    parser.add_argument("--proposal-shard-size", type=int, default=512)
    parser.add_argument("--maximum-initial-evaluations", type=int, default=50_000)
    return parser.parse_args()


def _proposal_config(args: argparse.Namespace) -> RecallProposalConfig:
    return RecallProposalConfig(
        seed=20_260_810,
        local_truth_table_maximum_sites=20,
        anchor_count=args.anchor_count,
        uniform_pair_budget=args.uniform_pair_budget,
        mutation_pair_budget=args.mutation_pair_budget,
        uniform_triple_budget=args.uniform_triple_budget,
        near_miss_pair_count=args.near_miss_pair_count,
        near_miss_triples_per_pair=args.near_miss_triples_per_pair,
        patch_batch_size=1,
        proposal_shard_size=args.proposal_shard_size,
        maximum_initial_evaluations=args.maximum_initial_evaluations,
        maximum_pair_evaluations=50_000,
        maximum_triple_evaluations=5_000,
        wilson_z_score=1.959963984540054,
    )


def _circuit_config(
    root: Path,
    function_id: str = "add_5",
    clean_step: int = 1_500,
    artifact_identity_root: Path | None = None,
    lineage_plan: Path | None = None,
) -> FourierCircuitConfig:
    registered_function_ids = {function.function_id for function in FUNCTIONS}
    if function_id not in {"add_5", "identity"} or function_id not in registered_function_ids:
        raise ValueError("recall audit requires the registered add_5 or identity probe")
    args = argparse.Namespace(
        function_id=function_id,
        clean_step=clean_step,
        dirty_step=0,
        layer_window="0:32",
        reverse_token_window=None,
        density_grid=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
        sufficiency_rule="clean-probability-minus-0.10",
        artifact_identity_root=artifact_identity_root,
        lineage_plan=lineage_plan,
    )
    return _config(root, args)


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
    proposal_config = _proposal_config(args)
    if args.plan_only:
        plan = build_recall_audit_plan(root, circuit_config, proposal_config)
        payload = json.loads((plan.output_dir / "proposal_plan.json").read_text())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"Fourier recall audit requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    run_fourier_recall_audit(
        root,
        circuit_config,
        proposal_config,
    )


if __name__ == "__main__":
    main()
