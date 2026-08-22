"""Persistent subset-to-causal-metric index for Fourier minset analysis."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import Site, SiteSet

SUBSET_INDEX_SCHEMA_VERSION = 1
SUBSET_INDEX_FILENAME = "subset_metric_index.json"
METRIC_PARITY_TOLERANCE = 2.0e-6
MAX_PROPER_SUBSET_PROBABILITY_FRACTION = 0.80

MaskBatch = Bool[t.Tensor, "sample token layer"]
CandidateLogits = Float[t.Tensor, "sample choice"]
MetricVector = Float[t.Tensor, "sample"]


@beartype
@dataclass(frozen=True)
class SubsetMetric:
    sites: SiteSet
    correct_probability: float
    raw_logit_diff: float
    accuracy: bool
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("subset metric sites must be sorted and unique")
        if not 0.0 <= self.correct_probability <= 1.0:
            raise ValueError("subset correct probability must lie in [0, 1]")
        if not math.isfinite(self.raw_logit_diff):
            raise ValueError("subset raw logit difference must be finite")
        if not self.sources or tuple(sorted(set(self.sources))) != self.sources:
            raise ValueError("subset metric sources must be non-empty, sorted, and unique")


@beartype
@dataclass(frozen=True)
class RelativeProperSubsetCriterion:
    """Require every proper subset to stay below a fraction of the full support."""

    maximum_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_fraction) or not 0.0 < self.maximum_fraction < 1.0:
            raise ValueError("proper-subset probability fraction must lie strictly inside (0, 1)")

    @beartype
    def maximum_allowed_probability(self, full_probability: float) -> float:
        if not math.isfinite(full_probability) or not 0.0 <= full_probability <= 1.0:
            raise ValueError("full-support probability must lie in [0, 1]")
        return self.maximum_fraction * full_probability


@beartype
def subset_index_path(scope_directory: Path) -> Path:
    return scope_directory / SUBSET_INDEX_FILENAME


@beartype
def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a string-keyed object")
    return cast(dict[str, object], value)


@beartype
def _site_set(value: object, *, context: str) -> SiteSet:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a site list")
    sites: list[Site] = []
    for raw_site in cast(list[object], value):
        site = _mapping(raw_site, context=f"{context}[]")
        token_index = site.get("token_index")
        layer = site.get("layer")
        if not isinstance(token_index, int) or not isinstance(layer, int):
            raise TypeError(f"{context} contains an invalid site")
        sites.append(Site(token_index, layer))
    canonical = tuple(sorted(set(sites)))
    if len(canonical) != len(sites):
        raise ValueError(f"{context} repeats a site")
    return canonical


@jaxtyped(typechecker=beartype)
def _metrics_from_tensors(
    masks: MaskBatch,
    candidate_logits: CandidateLogits,
    logit_diffs: MetricVector,
    correct_probabilities: MetricVector,
    accuracies: MetricVector,
    source: str,
) -> tuple[SubsetMetric, ...]:
    if masks.dtype is not t.bool or candidate_logits.shape[1] != 5:
        raise TypeError("subset metric sidecar has the wrong mask or candidate-logit shape")
    if not all(
        t.isfinite(values).all()
        for values in (
            candidate_logits,
            logit_diffs,
            correct_probabilities,
            accuracies,
        )
    ):
        raise ValueError("subset metric sidecar contains a non-finite value")
    if bool(((correct_probabilities < 0.0) | (correct_probabilities > 1.0)).any()):
        raise ValueError("subset metric sidecar contains a probability outside [0, 1]")
    if bool(((accuracies != 0.0) & (accuracies != 1.0)).any()):
        raise ValueError("subset metric sidecar contains a non-boolean accuracy")
    rows: list[SubsetMetric] = []
    for index, mask in enumerate(masks):
        sites = tuple(
            Site(int(coordinate[0]), int(coordinate[1]))
            for coordinate in mask.nonzero(as_tuple=False)
        )
        rows.append(
            SubsetMetric(
                sites,
                float(correct_probabilities[index]),
                float(logit_diffs[index]),
                bool(accuracies[index]),
                (source,),
            )
        )
    return tuple(rows)


@beartype
def _load_sidecar_metrics(path: Path, *, source: str) -> tuple[SubsetMetric, ...]:
    raw = t.load(path, map_location="cpu", weights_only=True)
    sidecar = _mapping(raw, context=str(path))
    required = (
        "masks",
        "candidate_logits",
        "logit_diffs",
        "correct_probabilities",
        "accuracies",
    )
    if not all(isinstance(sidecar.get(field), t.Tensor) for field in required):
        raise TypeError(f"subset metric sidecar is missing tensors: {path}")
    return _metrics_from_tensors(
        cast(MaskBatch, sidecar["masks"]),
        cast(CandidateLogits, sidecar["candidate_logits"]),
        cast(MetricVector, sidecar["logit_diffs"]),
        cast(MetricVector, sidecar["correct_probabilities"]),
        cast(MetricVector, sidecar["accuracies"]),
        source,
    )


@beartype
def _source_artifacts(scope_directory: Path) -> tuple[Path, ...]:
    singleton_path = scope_directory / "exhaustive_singletons.json"
    stage_two_path = scope_directory / "stage_2_minsets.json"
    if not singleton_path.is_file() or not stage_two_path.is_file():
        raise FileNotFoundError("subset index requires exhaustive singleton and Stage-2 artifacts")
    paths = {singleton_path, stage_two_path}
    for artifact_path, sidecar_field in (
        (singleton_path, "singleton_sidecar"),
        (stage_two_path, "verification_sidecar"),
    ):
        artifact = _mapping(read_json(artifact_path), context=str(artifact_path))
        sidecar_name = artifact.get(sidecar_field)
        digest = artifact.get(f"{sidecar_field}_sha256")
        if not isinstance(sidecar_name, str) or not isinstance(digest, str):
            raise TypeError(f"subset source artifact lacks {sidecar_field}: {artifact_path}")
        sidecar_path = artifact_path.with_name(sidecar_name)
        if sha256_file(sidecar_path) != digest:
            raise RuntimeError(f"subset source sidecar digest mismatch: {sidecar_path}")
        paths.add(sidecar_path)
    for audit_path in sorted(scope_directory.glob("recall_audit_config_*/recall_audit.json")):
        paths.add(audit_path)
        audit = _mapping(read_json(audit_path), context=str(audit_path))
        manifests = audit.get("phase_manifests")
        if not isinstance(manifests, list):
            raise TypeError(f"recall audit lacks phase manifests: {audit_path}")
        for raw_manifest in cast(list[object], manifests):
            manifest = _mapping(raw_manifest, context=f"{audit_path}.phase_manifests[]")
            phase = manifest.get("phase")
            shards = manifest.get("shards")
            if not isinstance(phase, str) or not isinstance(shards, list):
                raise TypeError(f"recall phase manifest is malformed: {audit_path}")
            for raw_shard in cast(list[object], shards):
                shard = _mapping(raw_shard, context=f"{audit_path}.{phase}.shards[]")
                metadata_name = shard.get("metadata")
                sidecar_name = shard.get("sidecar")
                sidecar_digest = shard.get("sidecar_sha256")
                if not all(
                    isinstance(value, str)
                    for value in (metadata_name, sidecar_name, sidecar_digest)
                ):
                    raise TypeError(f"recall shard identity is malformed: {audit_path}")
                phase_directory = audit_path.parent / phase
                metadata_path = phase_directory / cast(str, metadata_name)
                sidecar_path = phase_directory / cast(str, sidecar_name)
                if sha256_file(sidecar_path) != sidecar_digest:
                    raise RuntimeError(f"recall subset sidecar digest mismatch: {sidecar_path}")
                paths.update((metadata_path, sidecar_path))
    return tuple(sorted(paths))


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
        abs(previous.correct_probability - metric.correct_probability) > METRIC_PARITY_TOLERANCE
        or abs(previous.raw_logit_diff - metric.raw_logit_diff) > METRIC_PARITY_TOLERANCE
        or previous.accuracy is not metric.accuracy
    ):
        raise RuntimeError(f"duplicate subset metrics disagree for {metric.sites}")
    metrics[metric.sites] = SubsetMetric(
        metric.sites,
        previous.correct_probability,
        previous.raw_logit_diff,
        previous.accuracy,
        tuple(sorted({*previous.sources, *metric.sources})),
    )


@beartype
def build_subset_metric_index(scope_directory: Path) -> dict[SiteSet, SubsetMetric]:
    """Build and persist the exact measured subset lookup without model execution."""

    output_path = subset_index_path(scope_directory)
    if output_path.exists():
        raise FileExistsError(f"subset metric index already exists: {output_path}")
    source_paths = _source_artifacts(scope_directory)
    metrics: dict[SiteSet, SubsetMetric] = {}
    singleton_path = scope_directory / "exhaustive_singletons.json"
    singleton = _mapping(read_json(singleton_path), context=str(singleton_path))
    sufficiency = _mapping(singleton.get("sufficiency"), context=f"{singleton_path}.sufficiency")
    dirty_probability = sufficiency.get("dirty_correct_probability")
    dirty_logit_diff = sufficiency.get("dirty_logit_diff")
    if not isinstance(dirty_probability, (int, float)) or not isinstance(
        dirty_logit_diff, (int, float)
    ):
        raise TypeError("singleton sufficiency lacks the all-dirty endpoint")
    _register_metric(
        metrics,
        SubsetMetric((), float(dirty_probability), float(dirty_logit_diff), False, ("all_dirty",)),
    )
    singleton_rows = singleton.get("singleton_results")
    if not isinstance(singleton_rows, list):
        raise TypeError("exhaustive singleton artifact lacks its complete result table")
    for index, raw_row in enumerate(cast(list[object], singleton_rows)):
        row = _mapping(raw_row, context=f"{singleton_path}.singleton_results[{index}]")
        sites = _site_set([row.get("site")], context=f"{singleton_path}.singleton_results[{index}]")
        probability = row.get("correct_probability")
        logit_diff = row.get("raw_logit_diff")
        accuracy = row.get("accuracy")
        if (
            not isinstance(probability, (int, float))
            or not isinstance(logit_diff, (int, float))
            or not isinstance(accuracy, bool)
        ):
            raise TypeError("singleton result lacks a causal metric")
        _register_metric(
            metrics,
            SubsetMetric(
                sites,
                float(probability),
                float(logit_diff),
                accuracy,
                ("exhaustive_singletons",),
            ),
        )

    stage_two = _mapping(
        read_json(scope_directory / "stage_2_minsets.json"),
        context=str(scope_directory / "stage_2_minsets.json"),
    )
    stage_sidecar_name = stage_two.get("verification_sidecar")
    if not isinstance(stage_sidecar_name, str):
        raise TypeError("Stage-2 artifact lacks its verification sidecar")
    for metric in _load_sidecar_metrics(
        scope_directory / stage_sidecar_name,
        source="fourier_stage_2",
    ):
        _register_metric(metrics, metric)

    for audit_path in sorted(scope_directory.glob("recall_audit_config_*/recall_audit.json")):
        audit = _mapping(read_json(audit_path), context=str(audit_path))
        for raw_manifest in cast(list[object], audit["phase_manifests"]):
            manifest = _mapping(raw_manifest, context=f"{audit_path}.phase_manifests[]")
            phase = cast(str, manifest["phase"])
            for raw_shard in cast(list[object], manifest["shards"]):
                shard = _mapping(raw_shard, context=f"{audit_path}.{phase}.shards[]")
                sidecar_name = cast(str, shard["sidecar"])
                for metric in _load_sidecar_metrics(
                    audit_path.parent / phase / sidecar_name,
                    source=f"recall_{phase}",
                ):
                    _register_metric(metrics, metric)

    rows = [
        {
            "size": len(metric.sites),
            "sites": [asdict(site) for site in metric.sites],
            "correct_probability": metric.correct_probability,
            "raw_logit_diff": metric.raw_logit_diff,
            "accuracy": metric.accuracy,
            "sources": list(metric.sources),
        }
        for metric in sorted(metrics.values(), key=lambda row: (len(row.sites), row.sites))
    ]
    source_rows = [
        {
            "path": path.relative_to(scope_directory).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]
    write_json(
        output_path,
        {
            "schema_version": SUBSET_INDEX_SCHEMA_VERSION,
            "kind": "fourier_subset_metric_index",
            "support_count": len(rows),
            "source_artifacts": source_rows,
            "rows": rows,
        },
    )
    return metrics


@beartype
def load_subset_metric_index(scope_directory: Path) -> dict[SiteSet, SubsetMetric]:
    path = subset_index_path(scope_directory)
    payload = _mapping(read_json(path), context=str(path))
    if (
        payload.get("schema_version") != SUBSET_INDEX_SCHEMA_VERSION
        or payload.get("kind") != "fourier_subset_metric_index"
    ):
        raise RuntimeError(f"subset metric index has the wrong schema: {path}")
    source_rows = payload.get("source_artifacts")
    if not isinstance(source_rows, list):
        raise TypeError("subset metric index lacks source provenance")
    expected_sources = {
        source.relative_to(scope_directory).as_posix(): sha256_file(source)
        for source in _source_artifacts(scope_directory)
    }
    stored_sources: dict[str, str] = {}
    for raw_source in cast(list[object], source_rows):
        source = _mapping(raw_source, context=f"{path}.source_artifacts[]")
        relative = source.get("path")
        digest = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise TypeError("subset metric index contains malformed source provenance")
        stored_sources[relative] = digest
    if stored_sources != expected_sources:
        raise RuntimeError("subset metric index is stale; rebuild it from the changed raw shards")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or payload.get("support_count") != len(raw_rows):
        raise RuntimeError("subset metric index row count is inconsistent")
    metrics: dict[SiteSet, SubsetMetric] = {}
    for index, raw_row in enumerate(cast(list[object], raw_rows)):
        row = _mapping(raw_row, context=f"{path}.rows[{index}]")
        probability = row.get("correct_probability")
        logit_diff = row.get("raw_logit_diff")
        accuracy = row.get("accuracy")
        sources = row.get("sources")
        if (
            not isinstance(probability, (int, float))
            or not isinstance(logit_diff, (int, float))
            or not isinstance(accuracy, bool)
            or not isinstance(sources, list)
            or not all(isinstance(source, str) for source in sources)
        ):
            raise TypeError("subset metric index contains a malformed row")
        metric = SubsetMetric(
            _site_set(row.get("sites"), context=f"{path}.rows[{index}].sites"),
            float(probability),
            float(logit_diff),
            accuracy,
            tuple(cast(list[str], sources)),
        )
        if metric.sites in metrics:
            raise RuntimeError("subset metric index repeats a support")
        metrics[metric.sites] = metric
    return metrics


@beartype
def ensure_subset_metric_index(scope_directory: Path) -> dict[SiteSet, SubsetMetric]:
    """Load the persistent mapping, building it once from stored raw evaluations if absent."""

    return (
        load_subset_metric_index(scope_directory)
        if subset_index_path(scope_directory).is_file()
        else build_subset_metric_index(scope_directory)
    )


@beartype
def refresh_subset_metric_index_after_source_addition(
    scope_directory: Path,
) -> dict[SiteSet, SubsetMetric]:
    """Rebuild only when completed raw artifacts monotonically extend a valid index."""

    path = subset_index_path(scope_directory)
    if not path.is_file():
        return build_subset_metric_index(scope_directory)
    payload = _mapping(read_json(path), context=str(path))
    if (
        payload.get("schema_version") != SUBSET_INDEX_SCHEMA_VERSION
        or payload.get("kind") != "fourier_subset_metric_index"
    ):
        raise RuntimeError(f"subset metric index has the wrong schema: {path}")
    raw_sources = payload.get("source_artifacts")
    if not isinstance(raw_sources, list):
        raise TypeError("subset metric index lacks source provenance")
    stored_sources: dict[str, str] = {}
    for raw_source in cast(list[object], raw_sources):
        source = _mapping(raw_source, context=f"{path}.source_artifacts[]")
        relative = source.get("path")
        digest = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise TypeError("subset metric index contains malformed source provenance")
        source_path = scope_directory / relative
        if sha256_file(source_path) != digest:
            raise RuntimeError(f"previously indexed subset source changed: {source_path}")
        stored_sources[relative] = digest
    current_sources = {
        source.relative_to(scope_directory).as_posix(): sha256_file(source)
        for source in _source_artifacts(scope_directory)
    }
    if stored_sources == current_sources:
        return load_subset_metric_index(scope_directory)
    if not set(stored_sources).issubset(current_sources):
        raise RuntimeError("subset metric source inventory was removed or renamed")
    if any(current_sources[relative] != digest for relative, digest in stored_sources.items()):
        raise RuntimeError("subset metric source inventory changed non-monotonically")
    path.unlink()
    return build_subset_metric_index(scope_directory)


@beartype
def maximum_proper_subset_metric(
    sites: SiteSet,
    metrics: dict[SiteSet, SubsetMetric],
) -> SubsetMetric:
    """Return the highest-probability non-full subset, including the dirty empty set."""

    if len(sites) < 2 or tuple(sorted(set(sites))) != sites:
        raise ValueError("proper-subset lookup requires a canonical multi-site support")
    proper_subsets = tuple(
        tuple(subset)
        for size in range(len(sites))
        for subset in itertools.combinations(sites, size)
    )
    missing = tuple(subset for subset in proper_subsets if subset not in metrics)
    if missing:
        raise RuntimeError(f"subset metric index is incomplete for {sites}: {missing[:3]}")
    return max(
        (metrics[subset] for subset in proper_subsets),
        key=lambda metric: (
            metric.correct_probability,
            len(metric.sites),
            metric.sites,
        ),
    )


@beartype
def passes_relative_proper_subset_criterion(
    full_metric: SubsetMetric,
    maximum_subset_metric: SubsetMetric,
    criterion: RelativeProperSubsetCriterion,
) -> bool:
    if len(full_metric.sites) < 2:
        raise ValueError("relative proper-subset checks require a multi-site full support")
    if not set(maximum_subset_metric.sites).issubset(full_metric.sites) or (
        maximum_subset_metric.sites == full_metric.sites
    ):
        raise ValueError("maximum subset metric must be a strict subset of the full support")
    return maximum_subset_metric.correct_probability <= criterion.maximum_allowed_probability(
        full_metric.correct_probability
    )


__all__ = [
    "METRIC_PARITY_TOLERANCE",
    "MAX_PROPER_SUBSET_PROBABILITY_FRACTION",
    "SUBSET_INDEX_FILENAME",
    "SUBSET_INDEX_SCHEMA_VERSION",
    "SubsetMetric",
    "RelativeProperSubsetCriterion",
    "build_subset_metric_index",
    "ensure_subset_metric_index",
    "refresh_subset_metric_index_after_source_addition",
    "load_subset_metric_index",
    "maximum_proper_subset_metric",
    "passes_relative_proper_subset_criterion",
    "subset_index_path",
]
