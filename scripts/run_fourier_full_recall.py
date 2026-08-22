#!/usr/bin/env python3
"""Run the complete registered recall ladder for one checkpoint-transfer target."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import FourierCircuitConfig, SiteSet
from oocr_training_dynamics.fourier_disconnected import DisconnectedSearchConfig
from oocr_training_dynamics.fourier_frontier import (
    FourierFrontierConfig,
    hypergraph_components,
    relative_verified_minsets,
)
from oocr_training_dynamics.fourier_recall import RecallProposalConfig
from oocr_training_dynamics.fourier_subset_index import (
    RelativeProperSubsetCriterion,
    SubsetMetric,
    ensure_subset_metric_index,
    refresh_subset_metric_index_after_source_addition,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.runtime_fourier_circuits import (
    fourier_output_dir,
    logical_artifact_path,
)
from oocr_training_dynamics.runtime_fourier_disconnected import (
    build_disconnected_search_plan,
    run_disconnected_search,
)
from oocr_training_dynamics.runtime_fourier_frontier import (
    _completed_prior_frontiers,
    _full_probability_threshold,
    _site_set,
    _verified_support_inventory,
    frontier_output_dir,
    run_frontier_search,
)
from oocr_training_dynamics.runtime_fourier_recall import (
    recall_output_dir,
    run_fourier_recall_audit,
)
from oocr_training_dynamics.runtime_fourier_residual import (
    NetworkVetoDensityConfig,
    build_network_veto_density_plan,
    run_network_veto_density,
)
from scripts.run_fourier_disconnected_search import _search_config
from scripts.run_fourier_recall_audit import _circuit_config

MINIMUM_FREE_BYTES = 8 * 2**30
RELATIVE_SUBSET_FRACTION = 0.80
FRONTIER_ORDERS = (4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-id", choices=("add_5", "identity"), required=True)
    parser.add_argument("--clean-step", type=int, required=True)
    parser.add_argument("--maximum-initial-evaluations", type=int, default=120_000)
    parser.add_argument(
        "--maximum-network-evaluations-per-order",
        type=int,
        default=2_000_000,
    )
    parser.add_argument(
        "--maximum-component-shell-pair-evaluations",
        type=int,
        default=500_000,
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
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _recall_config(maximum_initial_evaluations: int) -> RecallProposalConfig:
    return RecallProposalConfig(
        seed=20_260_810,
        local_truth_table_maximum_sites=20,
        anchor_count=4,
        uniform_pair_budget=8_192,
        mutation_pair_budget=4_096,
        uniform_triple_budget=2_048,
        near_miss_pair_count=64,
        near_miss_triples_per_pair=32,
        patch_batch_size=1,
        proposal_shard_size=512,
        maximum_initial_evaluations=maximum_initial_evaluations,
        maximum_pair_evaluations=50_000,
        maximum_triple_evaluations=5_000,
        wilson_z_score=1.959963984540054,
    )


def _frontier_config(
    order: int,
    maximum_network_evaluations_per_order: int,
    maximum_component_shell_pair_evaluations: int,
) -> FourierFrontierConfig:
    return FourierFrontierConfig(
        seed=20_260_811,
        proper_subset_probability_fraction=RELATIVE_SUBSET_FRACTION,
        maximum_network_order=order,
        component_shell_pair_search=True,
        component_shell_fixed_point=True,
        higher_orders_after_component_shell=True,
        run_balanced_pair_probe=False,
        balanced_pair_budget=8_192,
        patch_batch_size=1,
        proposal_shard_size=256,
        maximum_network_evaluations_per_order=maximum_network_evaluations_per_order,
        maximum_component_shell_pair_evaluations=(maximum_component_shell_pair_evaluations),
        maximum_component_shell_iterations=16,
        maximum_balanced_pair_evaluations=50_000,
    )


def _completed_frontier_for_order(scope: Path, order: int) -> Path | None:
    matches: list[Path] = []
    for path in sorted(scope.glob("frontier_search_config_*/frontier_search.json")):
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise TypeError(f"frontier result must be an object: {path}")
        config = raw.get("frontier_config")
        if not isinstance(config, dict):
            raise TypeError(f"frontier result lacks its config: {path}")
        if (
            raw.get("status") == "complete"
            and config.get("maximum_network_order") == order
            and config.get("component_shell_fixed_point") is True
            and config.get("component_shell_pair_search") is True
            and config.get("higher_orders_after_component_shell") is True
            and raw.get("component_shell_fixed_point_reached") is True
            and raw.get("network_completion_is_exhaustive_through_registered_order") is True
        ):
            matches.append(path)
    return matches[-1] if matches else None


def _relative_inventory(
    scope: Path,
) -> tuple[dict[SiteSet, SubsetMetric], tuple[SiteSet, ...]]:
    metrics = ensure_subset_metric_index(scope)
    prior_metrics, prior_verified, _sources = _completed_prior_frontiers(
        scope,
        scope / "full_recall_ladder_not_a_frontier_directory",
    )
    for support, metric in prior_metrics.items():
        if support in metrics:
            raise RuntimeError(f"frontier cache repeats the immutable subset cache: {support}")
        metrics[support] = metric
    inventory = tuple(
        sorted(
            {*_verified_support_inventory(scope), *prior_verified},
            key=lambda support: (len(support), support),
        )
    )
    relative = relative_verified_minsets(
        inventory,
        metrics,
        _full_probability_threshold(scope),
        RelativeProperSubsetCriterion(RELATIVE_SUBSET_FRACTION),
    )
    return metrics, tuple(row.sites for row in relative)


def _artifact_row(
    root: Path,
    circuit_config: FourierCircuitConfig,
    path: Path,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(logical_artifact_path(root, circuit_config, path)),
        "sha256": sha256_file(path),
    }


def _full_recall_manifest_path(recall_result: Path) -> Path:
    """Keep terminal state in the exact recall-config namespace that generated it."""

    if recall_result.name != "recall_audit.json":
        raise ValueError("full-recall terminal state requires a recall_audit.json result")
    prefix = "recall_audit_config_"
    directory_name = recall_result.parent.name
    digest = directory_name.removeprefix(prefix)
    if (
        directory_name == digest
        or len(digest) != 12
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("full-recall terminal state requires a digest-namespaced recall result")
    return recall_result.parent / "full_recall_ladder.json"


def _write_terminal_manifest(
    path: Path,
    circuit_config: FourierCircuitConfig,
    recall_config: RecallProposalConfig,
    status: str,
    strict_minsets: tuple[SiteSet, ...],
    artifacts: list[dict[str, str]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "circuit_config": json.loads(
            json.dumps(asdict(circuit_config), default=str, allow_nan=False)
        ),
        "recall_config": asdict(recall_config),
        "relative_proper_subset_fraction": RELATIVE_SUBSET_FRACTION,
        "strict_multisite_minset_count": len(strict_minsets),
        "strict_component_site_counts": (
            [len(component) for component in hypergraph_components(strict_minsets)]
            if strict_minsets
            else []
        ),
        "artifacts": artifacts,
    }
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError(f"stored full-recall manifest changed: {path}")
    if not path.is_file():
        write_json(path, payload)
    stored = read_json(path)
    if not isinstance(stored, dict):
        raise TypeError(f"stored full-recall manifest must be an object: {path}")
    return cast(dict[str, object], stored)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"full Fourier recall requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    circuit_config = _circuit_config(
        root,
        args.function_id,
        args.clean_step,
        args.artifact_identity_root,
        args.lineage_plan,
    )
    recall_config = _recall_config(args.maximum_initial_evaluations)
    scope = fourier_output_dir(root, circuit_config)
    artifacts: list[dict[str, str]] = []

    run_fourier_recall_audit(root, circuit_config, recall_config)
    recall_result = recall_output_dir(root, circuit_config, recall_config) / "recall_audit.json"
    terminal_manifest = _full_recall_manifest_path(recall_result)
    artifacts.append(_artifact_row(root, circuit_config, recall_result))
    refresh_subset_metric_index_after_source_addition(scope)
    _metrics, strict_minsets = _relative_inventory(scope)
    if not strict_minsets:
        result = _write_terminal_manifest(
            terminal_manifest,
            circuit_config,
            recall_config,
            "complete_no_strict_multisite_seed",
            strict_minsets,
            artifacts,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    for order in FRONTIER_ORDERS:
        result_path = _completed_frontier_for_order(scope, order)
        if result_path is None:
            config = _frontier_config(
                order,
                args.maximum_network_evaluations_per_order,
                args.maximum_component_shell_pair_evaluations,
            )
            run_frontier_search(root, circuit_config, config)
            result_path = frontier_output_dir(root, circuit_config, config) / "frontier_search.json"
        artifacts.append(_artifact_row(root, circuit_config, result_path))

    _metrics, strict_minsets = _relative_inventory(scope)
    components = hypergraph_components(strict_minsets)
    network_site_count = len({site for component in components for site in component})
    network_config = NetworkVetoDensityConfig(
        proper_subset_probability_fraction=RELATIVE_SUBSET_FRACTION,
        minimum_network_site_count=network_site_count,
    )
    network_plan = build_network_veto_density_plan(root, circuit_config, network_config)
    network_result = run_network_veto_density(root, circuit_config, network_config)
    network_path = Path(str(network_result["density_artifact"]))
    network_result_path = network_plan.output_dir / "network_veto_density.json"
    if network_path.name != "stage_0_network_veto_density.json":
        raise RuntimeError("network-veto result points to an unexpected density artifact")
    artifacts.append(_artifact_row(root, circuit_config, network_result_path))
    if network_result["status"] == "flat_stop":
        result = _write_terminal_manifest(
            terminal_manifest,
            circuit_config,
            recall_config,
            "complete_network_veto_flat_stop",
            strict_minsets,
            artifacts,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    disconnected_strict: set[SiteSet] = set()
    for expanded in (False, True):
        search_config: DisconnectedSearchConfig = _search_config(expanded_coverage=expanded)
        search_plan = build_disconnected_search_plan(
            root,
            circuit_config,
            network_config,
            search_config,
        )
        disconnected_result = run_disconnected_search(
            root,
            circuit_config,
            network_config,
            search_config,
        )
        raw_verified = disconnected_result.get("verified_disconnected_minsets")
        if not isinstance(raw_verified, list):
            raise TypeError("disconnected-search result lacks its verified minset table")
        for index, raw_row in enumerate(raw_verified):
            if not isinstance(raw_row, dict):
                raise TypeError("disconnected verified-minset row must be an object")
            disconnected_strict.add(
                _site_set(
                    raw_row.get("sites"),
                    context=f"verified_disconnected_minsets[{index}].sites",
                )
            )
        result_path = search_plan.output_dir / "disconnected_search.json"
        artifacts.append(_artifact_row(root, circuit_config, result_path))

    _metrics, strict_minsets = _relative_inventory(scope)
    strict_minsets = tuple(
        sorted(
            {*strict_minsets, *disconnected_strict},
            key=lambda support: (len(support), support),
        )
    )
    result = _write_terminal_manifest(
        terminal_manifest,
        circuit_config,
        recall_config,
        "complete_full_depth",
        strict_minsets,
        artifacts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
