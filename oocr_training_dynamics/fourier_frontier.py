"""Exact relative-subset frontier proposals for higher-recall circuit discovery."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

import torch as t
from beartype import beartype

from oocr_training_dynamics.fourier_circuits import Site, SiteSet
from oocr_training_dynamics.fourier_recall import ProposedSupport
from oocr_training_dynamics.fourier_subset_index import (
    RelativeProperSubsetCriterion,
    SubsetMetric,
    maximum_proper_subset_metric,
    passes_relative_proper_subset_criterion,
)


@beartype
@dataclass(frozen=True)
class FourierFrontierConfig:
    """Required fields for one independently versioned frontier search."""

    seed: int
    proper_subset_probability_fraction: float
    maximum_network_order: int
    component_shell_pair_search: bool
    component_shell_fixed_point: bool
    higher_orders_after_component_shell: bool
    run_balanced_pair_probe: bool
    balanced_pair_budget: int
    patch_batch_size: int
    proposal_shard_size: int
    maximum_network_evaluations_per_order: int
    maximum_component_shell_pair_evaluations: int
    maximum_component_shell_iterations: int
    maximum_balanced_pair_evaluations: int

    def __post_init__(self) -> None:
        RelativeProperSubsetCriterion(self.proper_subset_probability_fraction)
        if self.seed < 0:
            raise ValueError("frontier-search seed must be non-negative")
        if not 2 <= self.maximum_network_order <= 6:
            raise ValueError("network completion order must lie in [2, 6]")
        if self.patch_batch_size != 1:
            raise ValueError("scientific frontier collection requires patch batch size one")
        positive = (
            self.balanced_pair_budget,
            self.proposal_shard_size,
            self.maximum_network_evaluations_per_order,
            self.maximum_component_shell_pair_evaluations,
            self.maximum_component_shell_iterations,
            self.maximum_balanced_pair_evaluations,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("frontier-search budgets and shard size must be positive")
        if self.balanced_pair_budget > self.maximum_balanced_pair_evaluations:
            raise ValueError("balanced pair budget exceeds its explicit evaluation cap")
        if self.component_shell_fixed_point and not self.component_shell_pair_search:
            raise ValueError("fixed-point closure requires component-shell pair search")
        if self.higher_orders_after_component_shell and not self.component_shell_pair_search:
            raise ValueError("post-shell higher orders require component-shell pair search")


@beartype
@dataclass(frozen=True)
class RelativeVerifiedMinset:
    sites: SiteSet
    correct_probability: float
    maximum_proper_subset_probability: float
    maximum_proper_subset: SiteSet

    def __post_init__(self) -> None:
        if len(self.sites) < 2 or tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("relative minsets must be canonical multi-site supports")
        if not 0.0 <= self.maximum_proper_subset_probability <= self.correct_probability <= 1.0:
            raise ValueError("relative minset probabilities must be ordered inside [0, 1]")
        if not set(self.maximum_proper_subset).issubset(self.sites) or (
            self.maximum_proper_subset == self.sites
        ):
            raise ValueError("relative minset maximum child must be a strict subset")


@beartype
def hypergraph_components(minsets: tuple[SiteSet, ...]) -> tuple[tuple[Site, ...], ...]:
    """Return connected components of the clique-expanded mixed-order minset hypergraph."""

    if not minsets:
        raise ValueError("hypergraph component construction requires verified minsets")
    adjacency: dict[Site, set[Site]] = {}
    for minset in minsets:
        if len(minset) < 2 or tuple(sorted(set(minset))) != minset:
            raise ValueError("component minsets must be canonical multi-site supports")
        for site in minset:
            adjacency.setdefault(site, set())
        for source, target in itertools.combinations(minset, 2):
            adjacency[source].add(target)
            adjacency[target].add(source)
    remaining = set(adjacency)
    components: list[tuple[Site, ...]] = []
    while remaining:
        frontier = [min(remaining)]
        component: set[Site] = set()
        while frontier:
            site = frontier.pop()
            if site in component:
                continue
            component.add(site)
            frontier.extend(sorted(adjacency[site] - component, reverse=True))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


@beartype
def relative_verified_minsets(
    supports: tuple[SiteSet, ...],
    metrics: dict[SiteSet, SubsetMetric],
    full_probability_threshold: float,
    criterion: RelativeProperSubsetCriterion,
) -> tuple[RelativeVerifiedMinset, ...]:
    if not math.isfinite(full_probability_threshold) or not 0.0 < full_probability_threshold < 1.0:
        raise ValueError("full probability threshold must lie strictly inside (0, 1)")
    verified: list[RelativeVerifiedMinset] = []
    for support in sorted(set(supports), key=lambda sites: (len(sites), sites)):
        full = metrics.get(support)
        if full is None:
            raise RuntimeError(f"verified support is absent from the metric cache: {support}")
        if full.correct_probability < full_probability_threshold or not full.accuracy:
            raise RuntimeError(f"input support is not sufficient under the full-set rule: {support}")
        maximum = maximum_proper_subset_metric(support, metrics)
        if passes_relative_proper_subset_criterion(full, maximum, criterion):
            verified.append(
                RelativeVerifiedMinset(
                    support,
                    full.correct_probability,
                    maximum.correct_probability,
                    maximum.sites,
                )
            )
    return tuple(verified)


@beartype
def network_completion_proposals(
    components: tuple[tuple[Site, ...], ...],
    order: int,
    metrics: dict[SiteSet, SubsetMetric],
    criterion: RelativeProperSubsetCriterion,
    maximum_evaluations: int,
) -> tuple[ProposedSupport, ...]:
    """Complete one subset order inside observed components using exact safe pruning."""

    if order < 2 or maximum_evaluations <= 0 or not components:
        raise ValueError("network completion requires components, order >= 2, and a positive cap")
    if any(tuple(sorted(set(component))) != component for component in components):
        raise ValueError("network components must be sorted and internally unique")
    proposals: list[ProposedSupport] = []
    seen: set[SiteSet] = set()
    for component in components:
        if len(component) < order:
            continue
        for raw_support in itertools.combinations(component, order):
            support = tuple(raw_support)
            if support in seen:
                continue
            seen.add(support)
            proper_subsets = tuple(
                tuple(subset)
                for size in range(1, order)
                for subset in itertools.combinations(support, size)
            )
            if any(
                subset in metrics
                and metrics[subset].correct_probability > criterion.maximum_fraction
                for subset in proper_subsets
            ):
                continue
            missing = tuple(subset for subset in proper_subsets if subset not in metrics)
            if missing:
                raise RuntimeError(
                    f"network frontier is incomplete before order {order}: {missing[:3]}"
                )
            if support not in metrics:
                proposals.append(
                    ProposedSupport(
                        support,
                        (f"network_component_completion_size_{order}",),
                    )
                )
    if len(proposals) > maximum_evaluations:
        raise RuntimeError(
            f"network order {order} requires {len(proposals)} evaluations, exceeding cap "
            f"{maximum_evaluations}"
        )
    return tuple(proposals)


@beartype
def degree_balanced_pair_proposals(
    sites: tuple[Site, ...],
    metrics: dict[SiteSet, SubsetMetric],
    budget: int,
    seed: int,
) -> tuple[ProposedSupport, ...]:
    """Choose unseen pairs while greedily equalizing each site's measured pair exposure."""

    if len(sites) < 2 or tuple(sorted(set(sites))) != sites:
        raise ValueError("balanced pair proposals require canonical eligible sites")
    if budget <= 0 or seed < 0:
        raise ValueError("balanced pair budget must be positive and seed non-negative")
    site_set = set(sites)
    measured_pairs = {
        support
        for support in metrics
        if len(support) == 2 and set(support).issubset(site_set)
    }
    available = math.comb(len(sites), 2) - len(measured_pairs)
    if budget > available:
        raise RuntimeError(
            f"balanced pair budget {budget} exceeds {available} unseen eligible pairs"
        )
    exposure = dict.fromkeys(sites, 0)
    for source, target in measured_pairs:
        exposure[source] += 1
        exposure[target] += 1
    generator = t.Generator(device="cpu").manual_seed(seed)
    permutation = t.randperm(len(sites), generator=generator).tolist()
    tie_rank = {sites[index]: rank for rank, index in enumerate(permutation)}
    heap = [(exposure[site], tie_rank[site], site) for site in sites]
    heapq.heapify(heap)
    proposed: set[SiteSet] = set()
    while len(proposed) < budget:
        _source_count, _source_rank, source = heapq.heappop(heap)
        skipped: list[tuple[int, int, Site]] = []
        target: Site | None = None
        while heap:
            row = heapq.heappop(heap)
            candidate = row[2]
            pair = tuple(sorted((source, candidate)))
            if pair not in measured_pairs and pair not in proposed:
                target = candidate
                break
            skipped.append(row)
        for row in skipped:
            heapq.heappush(heap, row)
        if target is None:
            heapq.heappush(heap, (exposure[source], tie_rank[source], source))
            raise RuntimeError("balanced pair heap could not realize the registered budget")
        pair = tuple(sorted((source, target)))
        proposed.add(pair)
        exposure[source] += 1
        exposure[target] += 1
        heapq.heappush(heap, (exposure[source], tie_rank[source], source))
        heapq.heappush(heap, (exposure[target], tie_rank[target], target))
    return tuple(
        ProposedSupport(support, ("degree_balanced_unseen_pair",))
        for support in sorted(proposed)
    )


@beartype
def component_shell_pair_proposals(
    components: tuple[tuple[Site, ...], ...],
    eligible_sites: tuple[Site, ...],
    metrics: dict[SiteSet, SubsetMetric],
    maximum_evaluations: int,
) -> tuple[ProposedSupport, ...]:
    """Exhaust every unseen pair touching the currently discovered hypergraph."""

    if (
        not components
        or len(eligible_sites) < 2
        or tuple(sorted(set(eligible_sites))) != eligible_sites
        or maximum_evaluations <= 0
    ):
        raise ValueError("component-shell completion requires canonical non-empty inputs")
    component_sites = tuple(sorted({site for component in components for site in component}))
    eligible = set(eligible_sites)
    if any(site not in eligible for site in component_sites):
        raise RuntimeError("component-shell site is not eligible under the subset-effect ceiling")
    supports = {
        tuple(sorted((source, target)))
        for source in component_sites
        for target in eligible_sites
        if source != target and tuple(sorted((source, target))) not in metrics
    }
    if len(supports) > maximum_evaluations:
        raise RuntimeError(
            f"component-shell completion requires {len(supports)} evaluations, exceeding cap "
            f"{maximum_evaluations}"
        )
    return tuple(
        ProposedSupport(support, ("exhaustive_component_shell_pair",))
        for support in sorted(supports)
    )


__all__ = [
    "FourierFrontierConfig",
    "RelativeVerifiedMinset",
    "component_shell_pair_proposals",
    "degree_balanced_pair_proposals",
    "hypergraph_components",
    "network_completion_proposals",
    "relative_verified_minsets",
]
