from __future__ import annotations

from collections import Counter

import pytest

from oocr_training_dynamics.fourier_circuits import Site, SiteSet
from oocr_training_dynamics.fourier_frontier import (
    FourierFrontierConfig,
    component_shell_pair_proposals,
    degree_balanced_pair_proposals,
    hypergraph_components,
    network_completion_proposals,
    relative_verified_minsets,
)
from oocr_training_dynamics.fourier_subset_index import (
    RelativeProperSubsetCriterion,
    SubsetMetric,
)


def _config(**changes: int | float) -> FourierFrontierConfig:
    values: dict[str, int | float] = {
        "seed": 7,
        "proper_subset_probability_fraction": 0.8,
        "maximum_network_order": 4,
        "component_shell_pair_search": True,
        "component_shell_fixed_point": True,
        "higher_orders_after_component_shell": True,
        "run_balanced_pair_probe": False,
        "balanced_pair_budget": 2,
        "patch_batch_size": 1,
        "proposal_shard_size": 2,
        "maximum_network_evaluations_per_order": 100,
        "maximum_component_shell_pair_evaluations": 100,
        "maximum_component_shell_iterations": 4,
        "maximum_balanced_pair_evaluations": 100,
    }
    values.update(changes)
    return FourierFrontierConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"seed": -1}, "seed"),
        ({"proper_subset_probability_fraction": 1.0}, "fraction"),
        ({"maximum_network_order": 7}, "order"),
        ({"patch_batch_size": 2}, "batch size one"),
        ({"balanced_pair_budget": 101}, "exceeds"),
        ({"component_shell_pair_search": False}, "fixed-point"),
    ],
)
def test_frontier_config_rejects_illegal_states(
    changes: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**changes)


def _metric(sites: SiteSet, probability: float, *, accuracy: bool = False) -> SubsetMetric:
    return SubsetMetric(sites, probability, probability, accuracy, ("test",))


def test_mixed_order_hypergraph_components_share_overlapping_sites() -> None:
    a, b, c, d, e = (Site(index, 0) for index in range(5))

    components = hypergraph_components(((a, b), (b, c, d), (a, d), (c, e)))

    assert components == ((a, b, c, d, e),)


def test_network_completion_adds_cycle_diagonals() -> None:
    a, b, c, d = (Site(index, 0) for index in range(4))
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in (a, b, c, d)})
    for pair in ((a, b), (b, c), (c, d), (a, d)):
        metrics[pair] = _metric(pair, 0.95, accuracy=True)

    proposals = network_completion_proposals(
        ((a, b, c, d),),
        2,
        metrics,
        RelativeProperSubsetCriterion(0.8),
        10,
    )

    assert tuple(proposal.sites for proposal in proposals) == ((a, c), (b, d))
    assert all(
        proposal.proposal_modes == ("network_component_completion_size_2",)
        for proposal in proposals
    )


def test_network_completion_prunes_on_a_known_blocker_before_missing_children() -> None:
    a, b, c, d = (Site(index, 0) for index in range(4))
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in (a, b, c, d)})
    metrics[(a, b)] = _metric((a, b), 0.81)

    proposals = network_completion_proposals(
        ((a, b, c, d),),
        4,
        metrics,
        RelativeProperSubsetCriterion(0.8),
        10,
    )

    assert proposals == ()


def test_network_completion_fails_when_an_unblocked_child_is_unknown() -> None:
    a, b, c = (Site(index, 0) for index in range(3))
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in (a, b, c)})

    with pytest.raises(RuntimeError, match="incomplete"):
        network_completion_proposals(
            ((a, b, c),),
            3,
            metrics,
            RelativeProperSubsetCriterion(0.8),
            10,
        )


def test_relative_verified_minsets_checks_every_cached_proper_subset() -> None:
    a, b, c = (Site(index, 0) for index in range(3))
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in (a, b, c)})
    metrics[(a, b)] = _metric((a, b), 0.20)
    metrics[(a, c)] = _metric((a, c), 0.70)
    metrics[(b, c)] = _metric((b, c), 0.30)
    metrics[(a, b, c)] = _metric((a, b, c), 0.90, accuracy=True)

    verified = relative_verified_minsets(
        ((a, b, c),),
        metrics,
        0.899,
        RelativeProperSubsetCriterion(0.8),
    )

    assert len(verified) == 1
    assert verified[0].maximum_proper_subset == (a, c)
    assert verified[0].maximum_proper_subset_probability == pytest.approx(0.70)


def test_degree_balanced_pairs_are_deterministic_unseen_and_balanced() -> None:
    sites = tuple(Site(index, 0) for index in range(6))
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in sites})
    metrics[(sites[0], sites[1])] = _metric((sites[0], sites[1]), 0.20)

    first = degree_balanced_pair_proposals(sites, metrics, 6, 11)
    second = degree_balanced_pair_proposals(sites, metrics, 6, 11)

    assert first == second
    assert len({proposal.sites for proposal in first}) == 6
    assert all(proposal.sites != (sites[0], sites[1]) for proposal in first)
    degree = Counter(site for proposal in first for site in proposal.sites)
    degree[sites[0]] += 1
    degree[sites[1]] += 1
    assert max(degree.values()) - min(degree.values()) <= 1


def test_component_shell_exhausts_unseen_pairs_touching_known_component() -> None:
    a, b, c, d = (Site(index, 0) for index in range(4))
    sites = (a, b, c, d)
    metrics: dict[SiteSet, SubsetMetric] = {(): _metric((), 0.01)}
    metrics.update({(site,): _metric((site,), 0.10) for site in sites})
    metrics[(a, b)] = _metric((a, b), 0.95, accuracy=True)
    metrics[(a, c)] = _metric((a, c), 0.20)

    proposals = component_shell_pair_proposals(((a, b),), sites, metrics, 10)

    assert tuple(proposal.sites for proposal in proposals) == (
        (a, d),
        (b, c),
        (b, d),
    )
    assert all(
        proposal.proposal_modes == ("exhaustive_component_shell_pair",)
        for proposal in proposals
    )


def test_component_shell_honors_explicit_evaluation_cap() -> None:
    a, b, c = (Site(index, 0) for index in range(3))
    metrics: dict[SiteSet, SubsetMetric] = {
        (site,): _metric((site,), 0.10) for site in (a, b, c)
    }

    with pytest.raises(RuntimeError, match="exceeding cap"):
        component_shell_pair_proposals(((a,),), (a, b, c), metrics, 1)
