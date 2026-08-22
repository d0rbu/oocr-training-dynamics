"""Known-network-vetoed density diagnostic for disconnected circuit recall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from beartype import beartype

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import (
    FourierCircuitConfig,
    FullPromptSites,
    ProbabilitySufficiencyConfig,
    Site,
)
from oocr_training_dynamics.fourier_frontier import (
    hypergraph_components,
    relative_verified_minsets,
)
from oocr_training_dynamics.fourier_subset_index import (
    RelativeProperSubsetCriterion,
    ensure_subset_metric_index,
)
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_fourier_circuits import (
    FOURIER_SCHEMA_VERSION,
    _capture_clean_checkpoint,
    _load_checkpoint_model,
    _load_tokenizer,
    _release_model,
    _resolve_blocks,
    _verified_singleton_sites,
    build_active_site_space,
    build_circuit_probe,
    build_site_grid,
    fourier_output_dir,
    logical_artifact_path,
    run_density_sweep,
    verify_inference_mode_parity,
)
from oocr_training_dynamics.runtime_fourier_frontier import (
    _completed_prior_frontiers,
    _full_probability_threshold,
    _mapping,
    _verified_support_inventory,
)

NETWORK_VETO_SCHEMA_VERSION = 1


@beartype
@dataclass(frozen=True)
class NetworkVetoDensityConfig:
    proper_subset_probability_fraction: float
    minimum_network_site_count: int

    def __post_init__(self) -> None:
        RelativeProperSubsetCriterion(self.proper_subset_probability_fraction)
        if self.minimum_network_site_count <= 0:
            raise ValueError("network-veto diagnostic requires a positive site-count floor")


@beartype
@dataclass(frozen=True)
class NetworkVetoDensityPlan:
    output_dir: Path
    network_sites: tuple[Site, ...]
    singleton_sites: tuple[Site, ...]
    vetoed_sites: tuple[Site, ...]
    source_payload: dict[str, object]

    def __post_init__(self) -> None:
        if (
            not self.output_dir.is_absolute()
            or not self.network_sites
            or not self.singleton_sites
            or tuple(sorted(set(self.vetoed_sites))) != self.vetoed_sites
            or not set(self.network_sites).issubset(self.vetoed_sites)
            or not set(self.singleton_sites).issubset(self.vetoed_sites)
        ):
            raise ValueError("network-veto density plan is incomplete")


@beartype
def _write_or_validate(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"stored network-veto artifact disagrees with current code: {path}")
        return
    write_json(path, payload)


@beartype
def _site_payload(sites: tuple[Site, ...]) -> list[dict[str, int]]:
    return [asdict(site) for site in sites]


@beartype
def _validated_network_veto_result(
    result_path: Path,
    plan: NetworkVetoDensityPlan,
    diagnostic_config: NetworkVetoDensityConfig,
) -> dict[str, object]:
    result = _mapping(read_json(result_path), context=str(result_path))
    density_name = result.get("density_artifact")
    density_digest = result.get("density_artifact_sha256")
    active_site_count = result.get("active_site_count")
    if (
        result.get("schema_version") != NETWORK_VETO_SCHEMA_VERSION
        or result.get("status") not in {"transition_found", "flat_stop"}
        or result.get("diagnostic_config") != asdict(diagnostic_config)
        or result.get("source") != plan.source_payload
        or result.get("network_site_count") != len(plan.network_sites)
        or result.get("singleton_site_count") != len(plan.singleton_sites)
        or result.get("vetoed_site_count") != len(plan.vetoed_sites)
        or not isinstance(active_site_count, int)
        or active_site_count <= 0
        or not isinstance(density_name, str)
        or not isinstance(density_digest, str)
        or not isinstance(result.get("curve"), list)
        or result.get("stop_before_mask_search")
        is not (result.get("status") == "flat_stop")
    ):
        raise RuntimeError(f"network-veto result is malformed: {result_path}")
    density_path = result_path.parent / density_name
    if sha256_file(density_path) != density_digest:
        raise RuntimeError(f"network-veto density digest mismatch: {density_path}")
    density = _mapping(read_json(density_path), context=str(density_path))
    sidecar_name = density.get("sample_sidecar")
    sidecar_digest = density.get("sample_sidecar_sha256")
    if (
        density.get("schema_version") != FOURIER_SCHEMA_VERSION
        or density.get("stage") != 0
        or density.get("function_space") != "network_vetoed_residual"
        or density.get("status") != result.get("status")
        or density.get("transition_density") != result.get("transition_density")
        or density.get("curve") != result.get("curve")
        or density.get("vetoed_sites") != _site_payload(plan.vetoed_sites)
        or density.get("active_site_count") != result.get("active_site_count")
        or not isinstance(sidecar_name, str)
        or not isinstance(sidecar_digest, str)
        or sha256_file(density_path.with_name(sidecar_name)) != sidecar_digest
    ):
        raise RuntimeError(
            f"network-veto result disagrees with its scientific density artifact: {result_path}"
        )
    return result


@beartype
def build_network_veto_density_plan(
    root: Path,
    circuit_config: FourierCircuitConfig,
    diagnostic_config: NetworkVetoDensityConfig,
) -> NetworkVetoDensityPlan:
    if not isinstance(circuit_config.sufficiency, ProbabilitySufficiencyConfig):
        raise RuntimeError("network-veto density requires the probability-sufficiency run")
    if not isinstance(circuit_config.sites, FullPromptSites):
        raise RuntimeError("network-veto density requires the full prompt grid")
    scope = fourier_output_dir(root, circuit_config)
    base_metrics = ensure_subset_metric_index(scope)
    prior_metrics, prior_verified, prior_sources = _completed_prior_frontiers(
        scope,
        scope / "network_veto_not_a_frontier_directory",
    )
    for support, metric in prior_metrics.items():
        if support in base_metrics:
            raise RuntimeError(f"frontier cache repeats the base subset cache: {support}")
        base_metrics[support] = metric
    inventory = tuple(
        sorted(
            {*_verified_support_inventory(scope), *prior_verified},
            key=lambda support: (len(support), support),
        )
    )
    relative = relative_verified_minsets(
        inventory,
        base_metrics,
        _full_probability_threshold(scope),
        RelativeProperSubsetCriterion(
            diagnostic_config.proper_subset_probability_fraction
        ),
    )
    components = hypergraph_components(tuple(row.sites for row in relative))
    network_sites = tuple(sorted({site for component in components for site in component}))
    if len(network_sites) < diagnostic_config.minimum_network_site_count:
        raise RuntimeError(
            f"network-veto component has only {len(network_sites)} sites; expected at least "
            f"{diagnostic_config.minimum_network_site_count}"
        )
    singleton_path = scope / "exhaustive_singletons.json"
    singleton_payload = _mapping(read_json(singleton_path), context=str(singleton_path))
    singleton_sites = _verified_singleton_sites(singleton_payload)
    vetoed_sites = tuple(sorted({*network_sites, *singleton_sites}))
    encoded = json.dumps(
        {
            "config": asdict(diagnostic_config),
            "vetoed_sites": [asdict(site) for site in vetoed_sites],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    output_dir = scope / f"network_veto_density_{hashlib.sha256(encoded).hexdigest()[:12]}"
    source_payload: dict[str, object] = {
        "scope_directory": str(logical_artifact_path(root, circuit_config, scope)),
        "exhaustive_singletons_sha256": sha256_file(singleton_path),
        "completed_frontiers": prior_sources,
        "combined_subset_metric_count": len(base_metrics),
        "relative_minset_count": len(relative),
        "component_site_counts": [len(component) for component in components],
    }
    plan_payload: dict[str, object] = {
        "schema_version": NETWORK_VETO_SCHEMA_VERSION,
        "status": "planned",
        "diagnostic_config": asdict(diagnostic_config),
        "source": source_payload,
        "network_site_count": len(network_sites),
        "singleton_site_count": len(singleton_sites),
        "vetoed_site_count": len(vetoed_sites),
        "network_sites": [asdict(site) for site in network_sites],
        "singleton_sites": [asdict(site) for site in singleton_sites],
        "vetoed_sites": [asdict(site) for site in vetoed_sites],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate(output_dir / "network_veto_density_plan.json", plan_payload)
    return NetworkVetoDensityPlan(
        output_dir,
        network_sites,
        singleton_sites,
        vetoed_sites,
        source_payload,
    )


@beartype
def run_network_veto_density(
    root: Path,
    circuit_config: FourierCircuitConfig,
    diagnostic_config: NetworkVetoDensityConfig,
) -> dict[str, object]:
    plan = build_network_veto_density_plan(root, circuit_config, diagnostic_config)
    result_path = plan.output_dir / "network_veto_density.json"
    if result_path.is_file():
        return _validated_network_veto_result(result_path, plan, diagnostic_config)
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    grid = build_site_grid(probe, spec, circuit_config.sites)
    site_space = build_active_site_space(grid, plan.vetoed_sites)
    clean = _capture_clean_checkpoint(root, circuit_config, probe, spec)
    model = _load_checkpoint_model(root, circuit_config, circuit_config.model.dirty_step)
    try:
        blocks = _resolve_blocks(model, spec)
        verify_inference_mode_parity(
            plan.output_dir,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
        )
        density = run_density_sweep(
            plan.output_dir,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            circuit_config,
            function_space="network_vetoed_residual",
            site_space=site_space,
        )
    finally:
        _release_model(model)
    density_path = plan.output_dir / "stage_0_network_veto_density.json"
    payload: dict[str, object] = {
        "schema_version": NETWORK_VETO_SCHEMA_VERSION,
        "status": density["status"],
        "diagnostic_config": asdict(diagnostic_config),
        "source": plan.source_payload,
        "network_site_count": len(plan.network_sites),
        "singleton_site_count": len(plan.singleton_sites),
        "vetoed_site_count": len(plan.vetoed_sites),
        "active_site_count": site_space.active_site_count,
        "density_artifact": density_path.name,
        "density_artifact_sha256": sha256_file(density_path),
        "transition_density": density["transition_density"],
        "curve": density["curve"],
        "stop_before_mask_search": density["status"] == "flat_stop",
    }
    _write_or_validate(result_path, payload)
    return _validated_network_veto_result(result_path, plan, diagnostic_config)


__all__ = [
    "NETWORK_VETO_SCHEMA_VERSION",
    "NetworkVetoDensityConfig",
    "NetworkVetoDensityPlan",
    "build_network_veto_density_plan",
    "run_network_veto_density",
]
