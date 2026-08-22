"""Pure contracts and minimization helpers for disconnected-circuit search."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch as t
from beartype import beartype

from oocr_training_dynamics.fourier_circuits import Site, SiteSet
from oocr_training_dynamics.fourier_subset_index import SubsetMetric


@beartype
@dataclass(frozen=True)
class DisconnectedSearchConfig:
    seed: int
    proposal_mask_count: int
    maximum_successful_starts: int
    minimization_restarts_per_start: int
    maximum_exact_candidate_size: int
    maximum_metric_evaluations: int
    metric_shard_size: int
    patch_batch_size: int
    proper_subset_probability_fraction: float

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("disconnected-search seed must be non-negative")
        positive = (
            self.proposal_mask_count,
            self.maximum_successful_starts,
            self.minimization_restarts_per_start,
            self.maximum_exact_candidate_size,
            self.maximum_metric_evaluations,
            self.metric_shard_size,
            self.patch_batch_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("disconnected-search budgets must be positive")
        if self.maximum_successful_starts > self.proposal_mask_count:
            raise ValueError("successful-start cap cannot exceed the proposal-mask budget")
        if self.maximum_exact_candidate_size > 20:
            raise ValueError("exact candidate verification is capped at 20 sites")
        if (
            not math.isfinite(self.proper_subset_probability_fraction)
            or not 0.0 < self.proper_subset_probability_fraction < 1.0
        ):
            raise ValueError("proper-subset fraction must lie strictly inside (0, 1)")


@beartype
def support_digest(supports: tuple[SiteSet, ...]) -> str:
    if not supports or any(tuple(sorted(set(support))) != support for support in supports):
        raise ValueError("support digest requires canonical supports")
    payload = [
        [{"token_index": site.token_index, "layer": site.layer} for site in support]
        for support in supports
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@beartype
def partition_support(
    support: SiteSet,
    granularity: int,
    rank: dict[Site, int],
) -> tuple[SiteSet, ...]:
    if (
        not support
        or tuple(sorted(set(support))) != support
        or not 2 <= granularity <= len(support)
        or set(rank) != set(support)
    ):
        raise ValueError("support partition inputs are inconsistent")
    ordered = sorted(support, key=lambda site: rank[site])
    chunks: list[SiteSet] = []
    for index in range(granularity):
        start = index * len(ordered) // granularity
        stop = (index + 1) * len(ordered) // granularity
        chunk = tuple(sorted(ordered[start:stop]))
        if chunk:
            chunks.append(chunk)
    if tuple(sorted(site for chunk in chunks for site in chunk)) != support:
        raise RuntimeError("support partition did not exactly cover its input")
    return tuple(chunks)


MetricBatch = Callable[[tuple[SiteSet, ...]], dict[SiteSet, SubsetMetric]]


@beartype
def delta_debug_minimize(
    initial: SiteSet,
    evaluate: MetricBatch,
    full_probability_threshold: float,
    seed: int,
) -> SiteSet:
    """Return a one-removal-minimal sufficient support using deterministic ddmin."""

    if (
        len(initial) < 2
        or tuple(sorted(set(initial))) != initial
        or not 0.0 < full_probability_threshold < 1.0
        or seed < 0
    ):
        raise ValueError("ddmin requires a canonical multisite support and valid threshold")
    initial_metric = evaluate((initial,))[initial]
    if (
        initial_metric.correct_probability < full_probability_threshold
        or not initial_metric.accuracy
    ):
        raise ValueError("ddmin initial support must satisfy the full-set rule")
    generator = t.Generator(device="cpu").manual_seed(seed)
    current = initial
    granularity = 2
    while len(current) >= 2:
        permutation = t.randperm(len(current), generator=generator).tolist()
        rank = {site: permutation[index] for index, site in enumerate(current)}
        chunks = partition_support(current, granularity, rank)
        complements = tuple(
            sorted(
                {
                    tuple(site for site in current if site not in set(chunk))
                    for chunk in chunks
                    if len(chunk) < len(current)
                },
                key=lambda support: (len(support), support),
            )
        )
        if not complements:
            break
        metrics = evaluate(complements)
        sufficient = tuple(
            support
            for support in complements
            if metrics[support].correct_probability >= full_probability_threshold
            and metrics[support].accuracy
        )
        if sufficient:
            current = max(
                sufficient,
                key=lambda support: (
                    metrics[support].correct_probability,
                    -len(support),
                    tuple((-site.token_index, -site.layer) for site in support),
                ),
            )
            granularity = min(len(current), max(2, granularity - 1))
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    if len(current) >= 2:
        children = tuple(
            current[:index] + current[index + 1 :] for index in range(len(current))
        )
        child_metrics = evaluate(children)
        if any(
            child_metrics[child].correct_probability >= full_probability_threshold
            and child_metrics[child].accuracy
            for child in children
        ):
            raise RuntimeError("ddmin terminated before reaching one-removal minimality")
    return current


@beartype
def diverse_successful_supports(
    supports: tuple[SiteSet, ...],
    metrics: dict[SiteSet, SubsetMetric],
    full_probability_threshold: float,
    maximum_count: int,
) -> tuple[SiteSet, ...]:
    if not supports or maximum_count <= 0:
        raise ValueError("diverse-start selection requires supports and a positive cap")
    successful = {
        support
        for support in supports
        if metrics[support].correct_probability >= full_probability_threshold
        and metrics[support].accuracy
    }
    if not successful:
        return ()
    selected: list[SiteSet] = []
    remaining = set(successful)
    while remaining and len(selected) < maximum_count:
        chosen = max(
            remaining,
            key=lambda support: (
                min(
                    (
                        len(set(support).symmetric_difference(previous))
                        for previous in selected
                    ),
                    default=len(support),
                ),
                metrics[support].correct_probability,
                tuple((-site.token_index, -site.layer) for site in support),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


__all__ = [
    "DisconnectedSearchConfig",
    "delta_debug_minimize",
    "diverse_successful_supports",
    "partition_support",
    "support_digest",
]
