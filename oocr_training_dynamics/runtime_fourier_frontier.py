"""Resumable full-prompt runtime for relative-subset frontier discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import (
    FourierCircuitConfig,
    FullPromptSites,
    ProbabilitySufficiencyConfig,
    Site,
    SiteGrid,
    SiteSet,
)
from oocr_training_dynamics.fourier_frontier import (
    FourierFrontierConfig,
    component_shell_pair_proposals,
    degree_balanced_pair_proposals,
    hypergraph_components,
    network_completion_proposals,
    relative_verified_minsets,
)
from oocr_training_dynamics.fourier_recall import ProposedSupport, masks_from_supports
from oocr_training_dynamics.fourier_subset_index import (
    METRIC_PARITY_TOLERANCE,
    RelativeProperSubsetCriterion,
    SubsetMetric,
    ensure_subset_metric_index,
    subset_index_path,
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
    _stage_sidecar_state,
    _write_tensor_sidecar,
    build_circuit_probe,
    build_site_grid,
    evaluate_masks_in_batches,
    fourier_output_dir,
    logical_artifact_path,
    verify_inference_mode_parity,
)

FRONTIER_SCHEMA_VERSION = 1
FRONTIER_RESULT_FILENAME = "frontier_search.json"
FRONTIER_METRIC_INDEX_FILENAME = "frontier_metric_index.json"


@beartype
@dataclass(frozen=True)
class FrontierSearchPlan:
    output_dir: Path
    grid: SiteGrid
    components: tuple[tuple[Site, ...], ...]
    eligible_sites: tuple[Site, ...]
    base_metrics: dict[SiteSet, SubsetMetric]
    base_relative_minsets: tuple[SiteSet, ...]
    full_probability_threshold: float
    source_payload: dict[str, object]

    def __post_init__(self) -> None:
        if (
            not self.output_dir.is_absolute()
            or not self.components
            or not self.eligible_sites
            or not self.base_metrics
            or not self.base_relative_minsets
            or not 0.0 < self.full_probability_threshold < 1.0
        ):
            raise ValueError("frontier search plan is incomplete")


@beartype
def frontier_output_dir(
    root: Path,
    circuit_config: FourierCircuitConfig,
    frontier_config: FourierFrontierConfig,
) -> Path:
    encoded = json.dumps(
        asdict(frontier_config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return fourier_output_dir(root, circuit_config) / f"frontier_search_config_{digest}"


@beartype
def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a string-keyed object")
    return cast(dict[str, object], value)


@beartype
def _site_set(raw: object, *, context: str) -> SiteSet:
    if not isinstance(raw, list):
        raise TypeError(f"{context} must be a site list")
    sites: list[Site] = []
    for index, value in enumerate(cast(list[object], raw)):
        row = _mapping(value, context=f"{context}[{index}]")
        token_index = row.get("token_index")
        layer = row.get("layer")
        if not isinstance(token_index, int) or not isinstance(layer, int):
            raise TypeError(f"{context}[{index}] has invalid coordinates")
        sites.append(Site(token_index, layer))
    support = tuple(sorted(set(sites)))
    if len(support) < 2 or len(support) != len(sites):
        raise RuntimeError(f"{context} must be a canonical multi-site support")
    return support


@beartype
def _verified_support_inventory(scope_directory: Path) -> tuple[SiteSet, ...]:
    stage_two_path = scope_directory / "stage_2_minsets.json"
    stage_two = _mapping(read_json(stage_two_path), context=str(stage_two_path))
    rows = stage_two.get("verified_multisite_minsets")
    if stage_two.get("status") not in {
        "verified_multisite",
        "no_verified_multisite_minsets",
    } or not isinstance(rows, list):
        raise RuntimeError("frontier search requires completed Fourier Stage 2")
    if stage_two.get("status") == "no_verified_multisite_minsets" and rows:
        raise RuntimeError("empty Stage-2 terminal status contains verified minsets")
    supports = {
        _site_set(
            _mapping(row, context=f"{stage_two_path}.verified_multisite_minsets[]").get("sites"),
            context=f"{stage_two_path}.verified_multisite_minsets[].sites",
        )
        for row in cast(list[object], rows)
    }
    for audit_path in sorted(scope_directory.glob("recall_audit_config_*/recall_audit.json")):
        audit = _mapping(read_json(audit_path), context=str(audit_path))
        if (
            audit.get("status") != "complete"
            or audit.get("raw_proposals_are_not_circuits") is not True
        ):
            raise RuntimeError(f"frontier search found an incomplete recall audit: {audit_path}")
        local = _mapping(audit.get("local_truth_table"), context=f"{audit_path}.local_truth_table")
        local_rows = local.get("new_minsets_missed_by_fourier")
        if not isinstance(local_rows, list):
            raise TypeError("recall audit local truth table lacks its new minsets")
        for index, row in enumerate(cast(list[object], local_rows)):
            supports.add(_site_set(row, context=f"{audit_path}.local[{index}]"))
        for field in ("new_verified_pair_minsets", "new_verified_triple_minsets"):
            audit_rows = audit.get(field)
            if not isinstance(audit_rows, list):
                raise TypeError(f"recall audit lacks {field}")
            for index, row in enumerate(cast(list[object], audit_rows)):
                mapping = _mapping(row, context=f"{audit_path}.{field}[{index}]")
                supports.add(
                    _site_set(
                        mapping.get("sites"),
                        context=f"{audit_path}.{field}[{index}].sites",
                    )
                )
    return tuple(sorted(supports, key=lambda support: (len(support), support)))


@beartype
def _full_probability_threshold(scope_directory: Path) -> float:
    singleton_path = scope_directory / "exhaustive_singletons.json"
    singleton = _mapping(read_json(singleton_path), context=str(singleton_path))
    sufficiency = _mapping(singleton.get("sufficiency"), context=f"{singleton_path}.sufficiency")
    if sufficiency.get("criterion") != "clean_correct_probability_minus_absolute_tolerance":
        raise RuntimeError("frontier search requires the clean-minus-ten-point causal rule")
    threshold = sufficiency.get("threshold_correct_probability")
    if not isinstance(threshold, int | float) or not 0.0 < float(threshold) < 1.0:
        raise TypeError("singleton sufficiency lacks a valid probability threshold")
    return float(threshold)


@beartype
def _write_or_validate(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"stored frontier artifact disagrees with current code: {path}")
        return
    write_json(path, payload)


@beartype
def _completed_prior_frontiers(
    scope_directory: Path,
    current_output_dir: Path,
) -> tuple[dict[SiteSet, SubsetMetric], tuple[SiteSet, ...], list[dict[str, object]]]:
    """Load completed, digest-validated frontier caches without importing this run."""

    metrics: dict[SiteSet, SubsetMetric] = {}
    verified: set[SiteSet] = set()
    sources: list[dict[str, object]] = []
    for result_path in sorted(
        scope_directory.glob(f"frontier_search_config_*/{FRONTIER_RESULT_FILENAME}")
    ):
        if result_path.parent == current_output_dir:
            continue
        result = _mapping(read_json(result_path), context=str(result_path))
        index_name = result.get("metric_index")
        index_digest = result.get("metric_index_sha256")
        raw_verified = result.get("new_verified_relative_minsets")
        if (
            result.get("schema_version") != FRONTIER_SCHEMA_VERSION
            or result.get("status") != "complete"
            or not isinstance(index_name, str)
            or not isinstance(index_digest, str)
            or not isinstance(raw_verified, list)
        ):
            raise RuntimeError(f"prior frontier result is malformed: {result_path}")
        index_path = result_path.parent / index_name
        if sha256_file(index_path) != index_digest:
            raise RuntimeError(f"prior frontier index digest mismatch: {index_path}")
        loaded = load_frontier_metric_index(result_path.parent)
        for support, metric in loaded.items():
            if support in metrics:
                raise RuntimeError(f"prior frontier indexes repeat a support: {support}")
            metrics[support] = metric
        for index, raw_row in enumerate(cast(list[object], raw_verified)):
            row = _mapping(raw_row, context=f"{result_path}.new_verified[{index}]")
            verified.add(
                _site_set(
                    row.get("sites"),
                    context=f"{result_path}.new_verified[{index}].sites",
                )
            )
        sources.append(
            {
                "directory": result_path.parent.name,
                "result_sha256": sha256_file(result_path),
                "metric_index": index_name,
                "metric_index_sha256": index_digest,
                "support_count": len(loaded),
            }
        )
    return metrics, tuple(sorted(verified, key=lambda support: (len(support), support))), sources


@beartype
def build_frontier_search_plan(
    root: Path,
    circuit_config: FourierCircuitConfig,
    frontier_config: FourierFrontierConfig,
) -> FrontierSearchPlan:
    if not isinstance(circuit_config.sufficiency, ProbabilitySufficiencyConfig):
        raise RuntimeError("frontier search is registered only for the probability-vetoed run")
    if not isinstance(circuit_config.sites, FullPromptSites):
        raise RuntimeError("frontier search requires the full prompt residual grid")
    scope_directory = fourier_output_dir(root, circuit_config)
    output_dir = frontier_output_dir(root, circuit_config, frontier_config)
    base_metrics = ensure_subset_metric_index(scope_directory)
    immutable_base_metric_count = len(base_metrics)
    prior_metrics, prior_verified, prior_sources = _completed_prior_frontiers(
        scope_directory,
        output_dir,
    )
    for support, metric in prior_metrics.items():
        if support in base_metrics:
            raise RuntimeError(f"prior frontier repeats the immutable base cache: {support}")
        base_metrics[support] = metric
    full_threshold = _full_probability_threshold(scope_directory)
    criterion = RelativeProperSubsetCriterion(frontier_config.proper_subset_probability_fraction)
    inventory = tuple(
        sorted(
            {*_verified_support_inventory(scope_directory), *prior_verified},
            key=lambda support: (len(support), support),
        )
    )
    relative = relative_verified_minsets(inventory, base_metrics, full_threshold, criterion)
    base_relative_minsets = tuple(row.sites for row in relative)
    components = hypergraph_components(base_relative_minsets)
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    grid = build_site_grid(probe, spec, circuit_config.sites)
    grid_sites = {grid.site(index) for index in range(grid.site_count)}
    if any(site not in grid_sites for component in components for site in component):
        raise RuntimeError("relative minset component contains a site outside the configured grid")
    eligible_sites = tuple(
        sorted(
            support[0]
            for support, metric in base_metrics.items()
            if len(support) == 1 and metric.correct_probability <= criterion.maximum_fraction
        )
    )
    index_path = subset_index_path(scope_directory)
    source_payload: dict[str, object] = {
        "base_scope_directory": str(
            logical_artifact_path(root, circuit_config, scope_directory)
        ),
        "base_subset_metric_index": index_path.name,
        "base_subset_metric_index_sha256": sha256_file(index_path),
        "base_subset_metric_count": immutable_base_metric_count,
    }
    if prior_sources:
        source_payload["prior_frontier_indexes"] = prior_sources
    plan_payload: dict[str, object] = {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "status": "planned",
        "frontier_config": asdict(frontier_config),
        "source": source_payload,
        "full_probability_threshold": full_threshold,
        "relative_verified_input_minset_count": len(base_relative_minsets),
        "relative_verified_input_size_counts": {
            str(size): sum(len(support) == size for support in base_relative_minsets)
            for size in sorted({len(support) for support in base_relative_minsets})
        },
        "component_count": len(components),
        "component_site_counts": [len(component) for component in components],
        "components": [[asdict(site) for site in component] for component in components],
        "eligible_singleton_site_count": len(eligible_sites),
        "initial_component_shell_pair_proposal_count": (
            len(
                component_shell_pair_proposals(
                    components,
                    eligible_sites,
                    base_metrics,
                    frontier_config.maximum_component_shell_pair_evaluations,
                )
            )
            if frontier_config.component_shell_pair_search
            else 0
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate(output_dir / "frontier_plan.json", plan_payload)
    return FrontierSearchPlan(
        output_dir,
        grid,
        components,
        eligible_sites,
        base_metrics,
        base_relative_minsets,
        full_threshold,
        source_payload,
    )


@beartype
def _proposal_digest(proposals: tuple[ProposedSupport, ...]) -> str:
    payload = [
        {
            "sites": [asdict(site) for site in proposal.sites],
            "proposal_modes": list(proposal.proposal_modes),
        }
        for proposal in proposals
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@beartype
def _metrics_from_sidecar(
    proposals: tuple[ProposedSupport, ...],
    sidecar: dict[str, t.Tensor],
) -> tuple[SubsetMetric, ...]:
    required = ("candidate_logits", "logit_diffs", "correct_probabilities", "accuracies")
    if any(name not in sidecar for name in required):
        raise RuntimeError("frontier shard lacks a required metric tensor")
    if any(sidecar[name].shape[0] != len(proposals) for name in required):
        raise RuntimeError("frontier shard metric rows disagree with its proposal count")
    metrics: list[SubsetMetric] = []
    for index, proposal in enumerate(proposals):
        metrics.append(
            SubsetMetric(
                proposal.sites,
                float(sidecar["correct_probabilities"][index]),
                float(sidecar["logit_diffs"][index]),
                bool(sidecar["accuracies"][index]),
                proposal.proposal_modes,
            )
        )
    return tuple(metrics)


@beartype
def _evaluate_phase(
    phase: str,
    proposals: tuple[ProposedSupport, ...],
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierFrontierConfig,
) -> tuple[SubsetMetric, ...]:
    if not phase or not proposals:
        raise ValueError("frontier evaluation requires a named non-empty proposal phase")
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[SubsetMetric] = []
    for shard_index, start in enumerate(range(0, len(proposals), config.proposal_shard_size)):
        stop = min(start + config.proposal_shard_size, len(proposals))
        shard = proposals[start:stop]
        stem = f"shard_{shard_index:05d}_{start:06d}_{stop:06d}"
        json_path = phase_dir / f"{stem}.json"
        tensor_path = phase_dir / f"{stem}.pt"
        expected = {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "kind": "fourier_relative_frontier_shard",
            "phase": phase,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "proposal_count": len(shard),
            "proposal_sha256": _proposal_digest(shard),
            "sidecar": tensor_path.name,
        }
        if _stage_sidecar_state(json_path, tensor_path):
            raw = _mapping(read_json(json_path), context=str(json_path))
            digest = raw.get("sidecar_sha256")
            if (
                {key: raw.get(key) for key in expected} != expected
                or not isinstance(digest, str)
                or sha256_file(tensor_path) != digest
            ):
                raise RuntimeError(f"stored frontier shard is invalid: {json_path}")
            sidecar = _load_tensor_sidecar(tensor_path)
        else:
            masks = masks_from_supports(tuple(proposal.sites for proposal in shard), grid)
            result = evaluate_masks_in_batches(
                model,
                blocks,
                probe,
                grid,
                clean_residuals,
                masks,
                config.patch_batch_size,
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
            write_json(json_path, {**expected, "sidecar_sha256": sha256_file(tensor_path)})
        metrics.extend(_metrics_from_sidecar(shard, sidecar))
    return tuple(metrics)


@beartype
def _register_new_metrics(
    metrics: dict[SiteSet, SubsetMetric],
    new_metrics: tuple[SubsetMetric, ...],
) -> None:
    for metric in new_metrics:
        previous = metrics.get(metric.sites)
        if previous is not None:
            if (
                abs(previous.correct_probability - metric.correct_probability)
                > METRIC_PARITY_TOLERANCE
                or abs(previous.raw_logit_diff - metric.raw_logit_diff) > METRIC_PARITY_TOLERANCE
                or previous.accuracy is not metric.accuracy
            ):
                raise RuntimeError(f"frontier metric disagrees with cached support: {metric.sites}")
            raise RuntimeError(
                f"frontier proposal repeated an already measured support: {metric.sites}"
            )
        metrics[metric.sites] = metric


@beartype
def _phase_manifest(output_dir: Path, phase: str) -> dict[str, object]:
    phase_dir = output_dir / phase
    if not phase_dir.is_dir():
        return {"phase": phase, "shard_count": 0, "proposal_count": 0, "shards": []}
    shards: list[dict[str, object]] = []
    proposal_count = 0
    for metadata_path in sorted(phase_dir.glob("shard_*.json")):
        metadata = _mapping(read_json(metadata_path), context=str(metadata_path))
        sidecar_name = metadata.get("sidecar")
        sidecar_digest = metadata.get("sidecar_sha256")
        count = metadata.get("proposal_count")
        if (
            metadata.get("phase") != phase
            or not isinstance(sidecar_name, str)
            or not isinstance(sidecar_digest, str)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise RuntimeError(f"frontier phase shard is malformed: {metadata_path}")
        sidecar_path = phase_dir / sidecar_name
        if sha256_file(sidecar_path) != sidecar_digest:
            raise RuntimeError(f"frontier phase sidecar digest mismatch: {sidecar_path}")
        proposal_count += count
        shards.append(
            {
                "metadata": metadata_path.name,
                "metadata_sha256": sha256_file(metadata_path),
                "sidecar": sidecar_name,
                "sidecar_sha256": sidecar_digest,
                "proposal_count": count,
                "proposal_sha256": metadata.get("proposal_sha256"),
            }
        )
    if len(tuple(phase_dir.glob("shard_*.pt"))) != len(shards):
        raise RuntimeError(f"frontier phase contains an unmatched sidecar: {phase_dir}")
    return {
        "phase": phase,
        "shard_count": len(shards),
        "proposal_count": proposal_count,
        "shards": shards,
    }


@beartype
def _write_frontier_metric_index(
    output_dir: Path,
    metrics: tuple[SubsetMetric, ...],
    manifests: tuple[dict[str, object], ...],
) -> Path:
    source_rows: list[dict[str, str]] = []
    for manifest in manifests:
        phase = cast(str, manifest["phase"])
        for raw_shard in cast(list[object], manifest["shards"]):
            shard = _mapping(raw_shard, context=f"{phase}.shards[]")
            for field, digest_field in (
                ("metadata", "metadata_sha256"),
                ("sidecar", "sidecar_sha256"),
            ):
                name = cast(str, shard[field])
                digest = cast(str, shard[digest_field])
                source_rows.append({"path": f"{phase}/{name}", "sha256": digest})
    rows = [
        {
            "size": len(metric.sites),
            "sites": [asdict(site) for site in metric.sites],
            "correct_probability": metric.correct_probability,
            "raw_logit_diff": metric.raw_logit_diff,
            "accuracy": metric.accuracy,
            "sources": list(metric.sources),
        }
        for metric in sorted(metrics, key=lambda row: (len(row.sites), row.sites))
    ]
    path = output_dir / FRONTIER_METRIC_INDEX_FILENAME
    _write_or_validate(
        path,
        {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "kind": "fourier_frontier_metric_index",
            "support_count": len(rows),
            "source_artifacts": source_rows,
            "rows": rows,
        },
    )
    return path


@beartype
def load_frontier_metric_index(output_dir: Path) -> dict[SiteSet, SubsetMetric]:
    path = output_dir / FRONTIER_METRIC_INDEX_FILENAME
    payload = _mapping(read_json(path), context=str(path))
    if (
        payload.get("schema_version") != FRONTIER_SCHEMA_VERSION
        or payload.get("kind") != "fourier_frontier_metric_index"
    ):
        raise RuntimeError(f"frontier metric index has the wrong schema: {path}")
    raw_sources = payload.get("source_artifacts")
    if not isinstance(raw_sources, list):
        raise TypeError("frontier metric index lacks source provenance")
    for index, raw_source in enumerate(cast(list[object], raw_sources)):
        source = _mapping(raw_source, context=f"{path}.source_artifacts[{index}]")
        relative = source.get("path")
        digest = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise TypeError("frontier metric source row is malformed")
        if sha256_file(output_dir / relative) != digest:
            raise RuntimeError(f"frontier metric index source digest mismatch: {relative}")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or payload.get("support_count") != len(raw_rows):
        raise RuntimeError("frontier metric index row count is inconsistent")
    metrics: dict[SiteSet, SubsetMetric] = {}
    for index, raw_row in enumerate(cast(list[object], raw_rows)):
        row = _mapping(raw_row, context=f"{path}.rows[{index}]")
        support = _site_set(row.get("sites"), context=f"{path}.rows[{index}].sites")
        probability = row.get("correct_probability")
        logit_diff = row.get("raw_logit_diff")
        accuracy = row.get("accuracy")
        sources = row.get("sources")
        if (
            not isinstance(probability, int | float)
            or not isinstance(logit_diff, int | float)
            or not isinstance(accuracy, bool)
            or not isinstance(sources, list)
            or not all(isinstance(source, str) for source in sources)
        ):
            raise TypeError("frontier metric row is malformed")
        metric = SubsetMetric(
            support,
            float(probability),
            float(logit_diff),
            accuracy,
            tuple(cast(list[str], sources)),
        )
        if support in metrics:
            raise RuntimeError("frontier metric index repeats a support")
        metrics[support] = metric
    return metrics


@beartype
def run_frontier_search(
    root: Path,
    circuit_config: FourierCircuitConfig,
    frontier_config: FourierFrontierConfig,
) -> dict[str, object]:
    plan = build_frontier_search_plan(root, circuit_config, frontier_config)
    result_path = plan.output_dir / FRONTIER_RESULT_FILENAME
    if result_path.is_file():
        load_frontier_metric_index(plan.output_dir)
        return _mapping(read_json(result_path), context=str(result_path))
    criterion = RelativeProperSubsetCriterion(frontier_config.proper_subset_probability_fraction)
    metrics = dict(plan.base_metrics)
    new_metrics: list[SubsetMetric] = []
    manifests: list[dict[str, object]] = []
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    clean = _capture_clean_checkpoint(root, circuit_config, probe, spec)
    model = _load_checkpoint_model(root, circuit_config, circuit_config.model.dirty_step)
    try:
        blocks = _resolve_blocks(model, spec)
        verify_inference_mode_parity(
            plan.output_dir,
            model,
            blocks,
            probe,
            plan.grid,
            clean.residuals,
        )
        current_minsets = list(plan.base_relative_minsets)
        current_components = plan.components
        closure_iterations: list[dict[str, object]] = []

        def evaluate_network_orders() -> None:
            for order in range(2, frontier_config.maximum_network_order + 1):
                phase = f"network_size_{order}"
                proposals = network_completion_proposals(
                    current_components,
                    order,
                    metrics,
                    criterion,
                    frontier_config.maximum_network_evaluations_per_order,
                )
                phase_metrics = (
                    _evaluate_phase(
                        phase,
                        proposals,
                        plan.output_dir,
                        model,
                        blocks,
                        probe,
                        plan.grid,
                        clean.residuals,
                        frontier_config,
                    )
                    if proposals
                    else ()
                )
                _register_new_metrics(metrics, phase_metrics)
                new_metrics.extend(phase_metrics)
                manifests.append(_phase_manifest(plan.output_dir, phase))

        if not frontier_config.higher_orders_after_component_shell:
            evaluate_network_orders()
        if frontier_config.component_shell_pair_search:
            for iteration in range(frontier_config.maximum_component_shell_iterations):
                phase = (
                    f"component_shell_pairs_{iteration:03d}"
                    if frontier_config.component_shell_fixed_point
                    else "component_shell_pairs"
                )
                starting_sites = {site for component in current_components for site in component}
                shell = component_shell_pair_proposals(
                    current_components,
                    plan.eligible_sites,
                    metrics,
                    frontier_config.maximum_component_shell_pair_evaluations,
                )
                shell_metrics = (
                    _evaluate_phase(
                        phase,
                        shell,
                        plan.output_dir,
                        model,
                        blocks,
                        probe,
                        plan.grid,
                        clean.residuals,
                        frontier_config,
                    )
                    if shell
                    else ()
                )
                _register_new_metrics(metrics, shell_metrics)
                new_metrics.extend(shell_metrics)
                manifests.append(_phase_manifest(plan.output_dir, phase))
                sufficient_shell = tuple(
                    metric.sites
                    for metric in shell_metrics
                    if metric.correct_probability >= plan.full_probability_threshold
                    and metric.accuracy
                )
                verified_shell = relative_verified_minsets(
                    sufficient_shell,
                    metrics,
                    plan.full_probability_threshold,
                    criterion,
                )
                current_minsets.extend(row.sites for row in verified_shell)
                if verified_shell:
                    current_components = hypergraph_components(tuple(current_minsets))
                ending_sites = {site for component in current_components for site in component}
                closure_iterations.append(
                    {
                        "iteration": iteration,
                        "phase": phase,
                        "starting_component_site_count": len(starting_sites),
                        "proposal_count": len(shell),
                        "verified_pair_count": len(verified_shell),
                        "newly_connected_site_count": len(ending_sites - starting_sites),
                        "ending_component_site_count": len(ending_sites),
                    }
                )
                if not frontier_config.component_shell_fixed_point:
                    break
                if ending_sites == starting_sites:
                    break
            else:
                raise RuntimeError(
                    "component-shell search did not reach its registered fixed point"
                )
        if frontier_config.higher_orders_after_component_shell:
            evaluate_network_orders()
        if frontier_config.run_balanced_pair_probe:
            balanced = degree_balanced_pair_proposals(
                plan.eligible_sites,
                metrics,
                frontier_config.balanced_pair_budget,
                frontier_config.seed,
            )
            balanced_metrics = _evaluate_phase(
                "balanced_pairs",
                balanced,
                plan.output_dir,
                model,
                blocks,
                probe,
                plan.grid,
                clean.residuals,
                frontier_config,
            )
            _register_new_metrics(metrics, balanced_metrics)
            new_metrics.extend(balanced_metrics)
            manifests.append(_phase_manifest(plan.output_dir, "balanced_pairs"))
    finally:
        _release_model(model)

    sufficient_new_supports = tuple(
        metric.sites
        for metric in new_metrics
        if metric.correct_probability >= plan.full_probability_threshold and metric.accuracy
    )
    verified = relative_verified_minsets(
        sufficient_new_supports,
        metrics,
        plan.full_probability_threshold,
        criterion,
    )
    index_path = _write_frontier_metric_index(
        plan.output_dir,
        tuple(new_metrics),
        tuple(manifests),
    )
    phase_counts = {
        cast(str, manifest["phase"]): cast(int, manifest["proposal_count"])
        for manifest in manifests
    }
    payload: dict[str, object] = {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "status": "complete",
        "source": plan.source_payload,
        "frontier_config": asdict(frontier_config),
        "criterion": {
            "full_probability_threshold": plan.full_probability_threshold,
            "maximum_proper_subset_fraction_of_full_probability": criterion.maximum_fraction,
            "safe_expandable_probability_ceiling": criterion.maximum_fraction,
            "require_clean_argmax": True,
        },
        "input_relative_minset_count": len(plan.base_relative_minsets),
        "input_component_site_counts": [len(component) for component in plan.components],
        "component_site_counts": [len(component) for component in current_components],
        "component_shell_iterations": closure_iterations,
        "phase_manifests": manifests,
        "phase_proposal_counts": phase_counts,
        "evaluated_support_count": len(new_metrics),
        "metric_index": index_path.name,
        "metric_index_sha256": sha256_file(index_path),
        "new_verified_relative_minsets": [
            {
                "size": len(row.sites),
                "sites": [asdict(site) for site in row.sites],
                "correct_probability": row.correct_probability,
                "maximum_proper_subset_correct_probability": (
                    row.maximum_proper_subset_probability
                ),
                "maximum_proper_subset_fraction_of_full_probability": (
                    row.maximum_proper_subset_probability / row.correct_probability
                ),
                "maximum_proper_subset": [asdict(site) for site in row.maximum_proper_subset],
            }
            for row in verified
        ],
        "network_completion_is_exhaustive_through_registered_order": (
            frontier_config.higher_orders_after_component_shell
        ),
        "component_shell_pair_search_is_exhaustive": (frontier_config.component_shell_pair_search),
        "component_shell_fixed_point_reached": bool(
            frontier_config.component_shell_fixed_point
            and closure_iterations
            and closure_iterations[-1]["newly_connected_site_count"] == 0
        ),
        "balanced_pair_probe_is_not_exhaustive": True,
        "raw_proposals_are_not_circuits": True,
    }
    _write_or_validate(result_path, payload)
    return payload


__all__ = [
    "FRONTIER_METRIC_INDEX_FILENAME",
    "FRONTIER_RESULT_FILENAME",
    "FRONTIER_SCHEMA_VERSION",
    "FrontierSearchPlan",
    "build_frontier_search_plan",
    "frontier_output_dir",
    "load_frontier_metric_index",
    "run_frontier_search",
]
