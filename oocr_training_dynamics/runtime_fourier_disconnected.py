"""Runtime for seeded disconnected-circuit proposal, minimization, and verification."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import (
    FourierCircuitConfig,
    ProbabilitySufficiencyConfig,
    Site,
    SiteGrid,
    SiteSet,
    SweepDensity,
)
from oocr_training_dynamics.fourier_disconnected import (
    DisconnectedSearchConfig,
    delta_debug_minimize,
    diverse_successful_supports,
    support_digest,
)
from oocr_training_dynamics.fourier_recall import masks_from_supports, supports_from_masks
from oocr_training_dynamics.fourier_subset_index import (
    METRIC_PARITY_TOLERANCE,
    SubsetMetric,
    ensure_subset_metric_index,
)
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_fourier_circuits import (
    CircuitProbe,
    ResidualBank,
    _capture_clean_checkpoint,
    _load_checkpoint_model,
    _load_tensor_sidecar,
    _load_tokenizer,
    _release_model,
    _resolve_blocks,
    _sample_masks_for_site_space,
    _stage_sidecar_state,
    _write_tensor_sidecar,
    build_active_site_space,
    build_circuit_probe,
    build_site_grid,
    evaluate_masks_in_batches,
    logical_artifact_path,
    verify_inference_mode_parity,
)
from oocr_training_dynamics.runtime_fourier_frontier import (
    _completed_prior_frontiers,
    _full_probability_threshold,
    _mapping,
)
from oocr_training_dynamics.runtime_fourier_residual import (
    NetworkVetoDensityConfig,
    NetworkVetoDensityPlan,
    _validated_network_veto_result,
    build_network_veto_density_plan,
)

DISCONNECTED_SEARCH_SCHEMA_VERSION = 1


@beartype
@dataclass(frozen=True)
class DisconnectedSearchPlan:
    output_dir: Path
    density: SweepDensity
    network_plan: NetworkVetoDensityPlan
    source_payload: dict[str, object]

    def __post_init__(self) -> None:
        if not self.output_dir.is_absolute() or not self.output_dir.parent.is_dir():
            raise ValueError("disconnected-search output directory must be absolute")


@beartype
def _write_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"stored disconnected-search artifact changed: {path}")
        return
    write_json(path, dict(payload))


@beartype
def _register_metric(
    metrics: dict[SiteSet, SubsetMetric],
    metric: SubsetMetric,
) -> None:
    previous = metrics.get(metric.sites)
    if previous is None:
        metrics[metric.sites] = metric
        return
    if (
        abs(previous.correct_probability - metric.correct_probability)
        > METRIC_PARITY_TOLERANCE
        or abs(previous.raw_logit_diff - metric.raw_logit_diff) > METRIC_PARITY_TOLERANCE
        or previous.accuracy is not metric.accuracy
    ):
        raise RuntimeError(f"disconnected-search duplicate metric disagrees: {metric.sites}")


@beartype
def _site_set(raw: object, *, context: str) -> SiteSet:
    if not isinstance(raw, list):
        raise TypeError(f"{context} must be a site list")
    sites: list[Site] = []
    for value in cast(list[object], raw):
        row = _mapping(value, context=context)
        token_index = row.get("token_index")
        layer = row.get("layer")
        if not isinstance(token_index, int) or not isinstance(layer, int):
            raise TypeError(f"{context} contains an invalid site")
        sites.append(Site(token_index, layer))
    support = tuple(sorted(set(sites)))
    if len(support) != len(sites):
        raise ValueError(f"{context} repeats a site")
    return support


@beartype
def build_disconnected_search_plan(
    root: Path,
    circuit_config: FourierCircuitConfig,
    network_config: NetworkVetoDensityConfig,
    search_config: DisconnectedSearchConfig,
) -> DisconnectedSearchPlan:
    network_plan = build_network_veto_density_plan(root, circuit_config, network_config)
    network_result_path = network_plan.output_dir / "network_veto_density.json"
    if not network_result_path.is_file():
        raise FileNotFoundError("disconnected search requires a completed network-veto density")
    network_result = _validated_network_veto_result(
        network_result_path,
        network_plan,
        network_config,
    )
    transition = network_result.get("transition_density")
    if network_result.get("status") != "transition_found" or not isinstance(
        transition, int | float
    ):
        raise RuntimeError("disconnected search is forbidden unless the residual curve is non-flat")
    density = SweepDensity.parse(float(transition))
    source_payload: dict[str, object] = {
        "network_veto_result": str(
            logical_artifact_path(root, circuit_config, network_result_path)
        ),
        "network_veto_result_sha256": sha256_file(network_result_path),
        "network_veto_density_artifact_sha256": network_result["density_artifact_sha256"],
        "transition_density": float(density),
        "vetoed_site_count": len(network_plan.vetoed_sites),
        "vetoed_sites": [asdict(site) for site in network_plan.vetoed_sites],
    }
    encoded = json.dumps(
        {"config": asdict(search_config), "source": source_payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    output_dir = network_plan.output_dir.parent / (
        f"disconnected_search_config_{hashlib.sha256(encoded).hexdigest()[:12]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": DISCONNECTED_SEARCH_SCHEMA_VERSION,
        "status": "planned",
        "search_config": asdict(search_config),
        "source": source_payload,
    }
    _write_or_validate(output_dir / "disconnected_search_plan.json", plan_payload)
    return DisconnectedSearchPlan(output_dir, density, network_plan, source_payload)


@beartype
class _MetricStore:
    def __init__(
        self,
        output_dir: Path,
        base_metrics: dict[SiteSet, SubsetMetric],
        grid: SiteGrid,
        model: t.nn.Module,
        blocks: tuple[t.nn.Module, ...],
        probe: CircuitProbe,
        clean_residuals: ResidualBank,
        config: DisconnectedSearchConfig,
    ) -> None:
        self.output_dir = output_dir
        self.metrics = dict(base_metrics)
        self.local_metrics: dict[SiteSet, SubsetMetric] = {}
        self.grid = grid
        self.model = model
        self.blocks = blocks
        self.probe = probe
        self.clean_residuals = clean_residuals
        self.config = config
        self.metric_dir = output_dir / "metric_batches"
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for json_path in sorted(self.metric_dir.glob("batch_*.json")):
            raw = _mapping(read_json(json_path), context=str(json_path))
            sidecar_name = raw.get("sidecar")
            sidecar_digest = raw.get("sidecar_sha256")
            raw_supports = raw.get("supports")
            if (
                raw.get("schema_version") != DISCONNECTED_SEARCH_SCHEMA_VERSION
                or raw.get("kind") != "disconnected_metric_batch"
                or not isinstance(sidecar_name, str)
                or not isinstance(sidecar_digest, str)
                or not isinstance(raw_supports, list)
            ):
                raise RuntimeError(f"disconnected metric batch is malformed: {json_path}")
            supports = tuple(
                _site_set(value, context=f"{json_path}.supports[]")
                for value in cast(list[object], raw_supports)
            )
            if raw.get("support_sha256") != support_digest(supports):
                raise RuntimeError(f"disconnected batch support digest changed: {json_path}")
            sidecar_path = json_path.with_name(sidecar_name)
            if sha256_file(sidecar_path) != sidecar_digest:
                raise RuntimeError(f"disconnected batch sidecar digest changed: {sidecar_path}")
            sidecar = _load_tensor_sidecar(sidecar_path)
            self._register_sidecar(supports, sidecar, cast(str, raw["source"]))

    def _register_sidecar(
        self,
        supports: tuple[SiteSet, ...],
        sidecar: dict[str, t.Tensor],
        source: str,
    ) -> None:
        required = ("logit_diffs", "correct_probabilities", "accuracies")
        if any(name not in sidecar or sidecar[name].shape[0] != len(supports) for name in required):
            raise RuntimeError("disconnected metric sidecar rows do not match supports")
        for index, support in enumerate(supports):
            metric = SubsetMetric(
                support,
                float(sidecar["correct_probabilities"][index]),
                float(sidecar["logit_diffs"][index]),
                bool(sidecar["accuracies"][index]),
                (source,),
            )
            _register_metric(self.metrics, metric)
            _register_metric(self.local_metrics, metric)

    def evaluate(
        self,
        supports: tuple[SiteSet, ...],
        *,
        source: str,
    ) -> dict[SiteSet, SubsetMetric]:
        canonical = tuple(sorted(set(supports), key=lambda row: (len(row), row)))
        if not canonical or any(not support for support in canonical):
            raise ValueError("disconnected metric evaluation requires non-empty supports")
        missing = tuple(support for support in canonical if support not in self.metrics)
        for start in range(0, len(missing), self.config.metric_shard_size):
            shard = missing[start : start + self.config.metric_shard_size]
            if len(self.local_metrics) + len(shard) > self.config.maximum_metric_evaluations:
                raise RuntimeError("disconnected search exceeded its registered evaluation cap")
            digest = support_digest(shard)
            json_path = self.metric_dir / f"batch_{digest[:20]}.json"
            tensor_path = self.metric_dir / f"batch_{digest[:20]}.pt"
            expected = {
                "schema_version": DISCONNECTED_SEARCH_SCHEMA_VERSION,
                "kind": "disconnected_metric_batch",
                "source": source,
                "support_count": len(shard),
                "support_sha256": digest,
                "supports": [[asdict(site) for site in support] for support in shard],
                "sidecar": tensor_path.name,
            }
            if _stage_sidecar_state(json_path, tensor_path):
                raw = _mapping(read_json(json_path), context=str(json_path))
                if {key: raw.get(key) for key in expected} != expected:
                    raise RuntimeError(f"stored disconnected batch changed: {json_path}")
                sidecar = _load_tensor_sidecar(tensor_path)
            else:
                masks = masks_from_supports(shard, self.grid)
                result = evaluate_masks_in_batches(
                    self.model,
                    self.blocks,
                    self.probe,
                    self.grid,
                    self.clean_residuals,
                    masks,
                    self.config.patch_batch_size,
                    with_gradients=False,
                )
                sidecar = {
                    "masks": masks,
                    "candidate_logits": result.candidate_logits,
                    "logit_diffs": result.logit_diffs,
                    "correct_probabilities": result.correct_probabilities,
                    "accuracies": result.accuracies,
                }
                _write_tensor_sidecar(tensor_path, sidecar)
                write_json(
                    json_path,
                    {**expected, "sidecar_sha256": sha256_file(tensor_path)},
                )
            self._register_sidecar(shard, sidecar, source)
        return {support: self.metrics[support] for support in canonical}

    def write_index(self) -> Path:
        path = self.output_dir / "disconnected_metric_index.json"
        rows = [
            {
                "size": len(metric.sites),
                "sites": [asdict(site) for site in metric.sites],
                "correct_probability": metric.correct_probability,
                "raw_logit_diff": metric.raw_logit_diff,
                "accuracy": metric.accuracy,
                "sources": list(metric.sources),
            }
            for metric in sorted(
                self.local_metrics.values(),
                key=lambda row: (len(row.sites), row.sites),
            )
        ]
        payload = {
            "schema_version": DISCONNECTED_SEARCH_SCHEMA_VERSION,
            "kind": "disconnected_metric_index",
            "support_count": len(rows),
            "rows": rows,
        }
        _write_or_validate(path, payload)
        return path


@beartype
def _combined_prior_metrics(scope: Path, current_output: Path) -> dict[SiteSet, SubsetMetric]:
    metrics = ensure_subset_metric_index(scope)
    frontier_metrics, _verified, _sources = _completed_prior_frontiers(scope, current_output)
    for _support, metric in frontier_metrics.items():
        _register_metric(metrics, metric)
    return metrics


@beartype
def _all_nonempty_subsets(support: SiteSet) -> tuple[SiteSet, ...]:
    if not support:
        raise ValueError("exact subset expansion requires a non-empty support")
    return tuple(
        tuple(subset)
        for size in range(1, len(support) + 1)
        for subset in itertools.combinations(support, size)
    )


@beartype
def run_disconnected_search(
    root: Path,
    circuit_config: FourierCircuitConfig,
    network_config: NetworkVetoDensityConfig,
    search_config: DisconnectedSearchConfig,
) -> dict[str, object]:
    if not isinstance(circuit_config.sufficiency, ProbabilitySufficiencyConfig):
        raise RuntimeError("disconnected search requires probability sufficiency")
    plan = build_disconnected_search_plan(root, circuit_config, network_config, search_config)
    result_path = plan.output_dir / "disconnected_search.json"
    if result_path.is_file():
        result = _mapping(read_json(result_path), context=str(result_path))
        index_name = result.get("metric_index")
        digest = result.get("metric_index_sha256")
        if (
            result.get("schema_version") != DISCONNECTED_SEARCH_SCHEMA_VERSION
            or result.get("status") != "complete"
            or result.get("search_config") != asdict(search_config)
            or result.get("source") != plan.source_payload
            or not isinstance(index_name, str)
            or not isinstance(digest, str)
            or sha256_file(result_path.with_name(index_name)) != digest
        ):
            raise RuntimeError(f"stored disconnected search is invalid: {result_path}")
        return result
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    grid = build_site_grid(probe, spec, circuit_config.sites)
    site_space = build_active_site_space(grid, plan.network_plan.vetoed_sites)
    scope = plan.network_plan.output_dir.parent
    prior_metrics = _combined_prior_metrics(scope, plan.output_dir)
    full_threshold = _full_probability_threshold(scope)
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
        store = _MetricStore(
            plan.output_dir,
            prior_metrics,
            grid,
            model,
            blocks,
            probe,
            clean.residuals,
            search_config,
        )
        generator = t.Generator(device="cpu").manual_seed(search_config.seed)
        masks = _sample_masks_for_site_space(
            search_config.proposal_mask_count,
            grid,
            plan.density,
            generator,
            site_space,
        )
        proposal_supports = supports_from_masks(masks, grid)
        proposal_metrics = store.evaluate(proposal_supports, source="disconnected_random_mask")
        starts = diverse_successful_supports(
            proposal_supports,
            proposal_metrics,
            full_threshold,
            search_config.maximum_successful_starts,
        )
        minimized_sources: dict[SiteSet, set[SiteSet]] = {}
        for start_index, start in enumerate(starts):
            for restart in range(search_config.minimization_restarts_per_start):
                minimized = delta_debug_minimize(
                    start,
                    lambda supports, source=(
                        f"disconnected_ddmin_start_{start_index:02d}_restart_{restart:02d}"
                    ): store.evaluate(supports, source=source),
                    full_threshold,
                    search_config.seed + 1 + 10_000 * start_index + restart,
                )
                if len(minimized) == 1:
                    raise RuntimeError(
                        "disconnected minimization found a singleton omitted by the exhaustive sweep"
                    )
                minimized_sources.setdefault(minimized, set()).add(start)
        verified: list[dict[str, object]] = []
        hypotheses: list[dict[str, object]] = []
        for candidate, generators in sorted(
            minimized_sources.items(),
            key=lambda item: (len(item[0]), item[0]),
        ):
            exact = len(candidate) <= search_config.maximum_exact_candidate_size
            if exact:
                subsets = _all_nonempty_subsets(candidate)
                metrics = store.evaluate(subsets, source="disconnected_exact_powerset")
                full = metrics[candidate]
                proper = [store.metrics[()]] + [
                    metrics[subset] for subset in subsets if subset != candidate
                ]
                maximum = max(
                    proper,
                    key=lambda metric: (
                        metric.correct_probability,
                        len(metric.sites),
                        metric.sites,
                    ),
                )
                passes = bool(
                    full.correct_probability >= full_threshold
                    and full.accuracy
                    and maximum.correct_probability
                    <= search_config.proper_subset_probability_fraction
                    * full.correct_probability
                )
            else:
                full = store.metrics[candidate]
                maximum = None
                passes = False
            row: dict[str, object] = {
                "size": len(candidate),
                "sites": [asdict(site) for site in candidate],
                "correct_probability": full.correct_probability,
                "raw_logit_diff": full.raw_logit_diff,
                "accuracy": full.accuracy,
                "one_removal_minimal": True,
                "exact_powerset_verified": exact,
                "generating_random_mask_count": len(generators),
                "generating_random_masks": [
                    [asdict(site) for site in generator_support]
                    for generator_support in sorted(generators)
                ],
            }
            if maximum is not None:
                row.update(
                    {
                        "maximum_proper_subset": [asdict(site) for site in maximum.sites],
                        "maximum_proper_subset_correct_probability": maximum.correct_probability,
                        "maximum_proper_subset_fraction_of_full_probability": (
                            maximum.correct_probability / full.correct_probability
                        ),
                    }
                )
            hypotheses.append(row)
            if passes:
                verified.append({**row, "sources": ["disconnected_mask_minimization"]})
        metric_index = store.write_index()
    finally:
        _release_model(model)
    successful_proposal_count = sum(
        metric.correct_probability >= full_threshold and metric.accuracy
        for metric in proposal_metrics.values()
    )
    payload: dict[str, object] = {
        "schema_version": DISCONNECTED_SEARCH_SCHEMA_VERSION,
        "status": "complete",
        "search_config": asdict(search_config),
        "source": plan.source_payload,
        "active_site_count": site_space.active_site_count,
        "full_probability_threshold": full_threshold,
        "proposal_mask_count": len(proposal_supports),
        "successful_proposal_count": successful_proposal_count,
        "selected_start_count": len(starts),
        "unique_minimized_candidate_count": len(hypotheses),
        "raw_minimized_hypotheses": hypotheses,
        "verified_disconnected_minsets": verified,
        "raw_hypotheses_are_not_circuits": True,
        "metric_index": metric_index.name,
        "metric_index_sha256": sha256_file(metric_index),
    }
    _write_or_validate(result_path, payload)
    return payload


__all__ = [
    "DISCONNECTED_SEARCH_SCHEMA_VERSION",
    "DisconnectedSearchPlan",
    "build_disconnected_search_plan",
    "run_disconnected_search",
]
