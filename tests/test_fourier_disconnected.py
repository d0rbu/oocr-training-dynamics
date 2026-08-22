from __future__ import annotations

from collections.abc import Callable

import pytest

from oocr_training_dynamics.fourier_circuits import Site, SiteSet
from oocr_training_dynamics.fourier_disconnected import (
    DisconnectedSearchConfig,
    delta_debug_minimize,
    diverse_successful_supports,
    partition_support,
    support_digest,
)
from oocr_training_dynamics.fourier_subset_index import SubsetMetric
from scripts.run_fourier_disconnected_search import _search_config


def _metric(support: SiteSet, probability: float) -> SubsetMetric:
    return SubsetMetric(support, probability, probability, probability >= 0.5, ("test",))


def _oracle(
    probability: Callable[[SiteSet], float],
) -> Callable[[tuple[SiteSet, ...]], dict[SiteSet, SubsetMetric]]:
    def evaluate(supports: tuple[SiteSet, ...]) -> dict[SiteSet, SubsetMetric]:
        return {support: _metric(support, probability(support)) for support in supports}

    return evaluate


def test_disconnected_search_config_rejects_illegal_budgets() -> None:
    with pytest.raises(ValueError, match="budgets"):
        DisconnectedSearchConfig(0, 16, 4, 2, 12, 0, 16, 1, 0.8)
    with pytest.raises(ValueError, match="successful-start"):
        DisconnectedSearchConfig(0, 4, 5, 2, 12, 100, 16, 1, 0.8)


def test_expanded_disconnected_search_is_independent_and_keeps_frozen_rule() -> None:
    initial = _search_config()
    expanded = _search_config(expanded_coverage=True)

    assert expanded.seed != initial.seed
    assert expanded.proposal_mask_count == 1_024
    assert expanded.maximum_successful_starts == 48
    assert expanded.minimization_restarts_per_start == 8
    assert expanded.maximum_exact_candidate_size == initial.maximum_exact_candidate_size == 12
    assert expanded.patch_batch_size == initial.patch_batch_size == 1
    assert (
        expanded.proper_subset_probability_fraction
        == initial.proper_subset_probability_fraction
        == 0.80
    )


def test_partition_support_is_exact_and_digest_is_order_sensitive() -> None:
    support = tuple(Site(index, 0) for index in range(7))
    rank = {site: len(support) - index for index, site in enumerate(support)}
    chunks = partition_support(support, 3, rank)
    assert sorted(site for chunk in chunks for site in chunk) == list(support)
    assert max(len(chunk) for chunk in chunks) - min(len(chunk) for chunk in chunks) <= 1
    assert support_digest((support,)) != support_digest((support[:-1],))


def test_delta_debug_recovers_one_removal_minimal_two_of_three_route() -> None:
    sites = tuple(Site(index, 0) for index in range(12))
    required = set(sites[2:5])

    def probability(support: SiteSet) -> float:
        return 0.95 if len(required.intersection(support)) >= 2 else 0.05

    minimized = delta_debug_minimize(sites, _oracle(probability), 0.9, seed=17)
    assert len(minimized) == 2
    assert set(minimized).issubset(required)


def test_diverse_successful_supports_ignores_failures_and_spreads_starts() -> None:
    sites = tuple(Site(index, 0) for index in range(8))
    supports = (
        (sites[0], sites[1]),
        (sites[0], sites[2]),
        (sites[6], sites[7]),
        (sites[3], sites[4]),
    )
    metrics: dict[SiteSet, SubsetMetric] = {
        support: _metric(support, probability)
        for support, probability in zip(supports, (0.99, 0.98, 0.97, 0.1), strict=True)
    }
    selected = diverse_successful_supports(supports, metrics, 0.9, 2)
    assert supports[3] not in selected
    assert set(selected) == {supports[0], supports[2]}
