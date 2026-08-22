from __future__ import annotations

import itertools

import pytest
import torch as t

from oocr_training_dynamics.fourier_circuits import Site, SiteGrid, SiteSet
from oocr_training_dynamics.fourier_recall import (
    ProposedSupport,
    RecallProposalConfig,
    WilsonInterval,
    canonical_support,
    child_pairs,
    exact_local_minsets,
    immediate_monotonicity_violations,
    initial_recall_proposals,
    masks_from_supports,
    supports_from_masks,
    triple_recall_proposals,
    wilson_interval,
)


def _config(**changes: int | float) -> RecallProposalConfig:
    values: dict[str, int | float] = {
        "seed": 7,
        "local_truth_table_maximum_sites": 8,
        "anchor_count": 1,
        "uniform_pair_budget": 2,
        "mutation_pair_budget": 2,
        "uniform_triple_budget": 2,
        "near_miss_pair_count": 1,
        "near_miss_triples_per_pair": 1,
        "patch_batch_size": 1,
        "proposal_shard_size": 2,
        "maximum_initial_evaluations": 100,
        "maximum_pair_evaluations": 100,
        "maximum_triple_evaluations": 100,
        "wilson_z_score": 1.959963984540054,
    }
    values.update(changes)
    return RecallProposalConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"seed": -1}, "seed"),
        ({"local_truth_table_maximum_sites": 1}, "truth-table"),
        ({"anchor_count": 0}, "positive"),
        ({"wilson_z_score": float("nan")}, "Wilson"),
    ],
)
def test_recall_config_rejects_illegal_states(
    changes: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**changes)


@pytest.mark.parametrize(
    "sites,modes,message",
    [
        ((Site(0, 0),), ("uniform_pair",), "sorted unique multi-site"),
        (
            (Site(1, 0), Site(0, 0)),
            ("uniform_pair",),
            "sorted unique multi-site",
        ),
        ((Site(0, 0), Site(1, 0)), (), "proposal modes"),
        (
            (Site(0, 0), Site(1, 0)),
            ("uniform_pair", "uniform_pair"),
            "proposal modes",
        ),
    ],
)
def test_proposed_support_rejects_noncanonical_states(
    sites: tuple[Site, ...],
    modes: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProposedSupport(sites, modes)


def test_small_value_wrappers_reject_invalid_states() -> None:
    with pytest.raises(ValueError, match="ordered"):
        WilsonInterval(0.5, 0.6, 0.7)
    with pytest.raises(ValueError, match="non-empty"):
        canonical_support(())


def test_mask_helpers_reject_empty_duplicate_and_wrong_shape_inputs() -> None:
    grid = SiteGrid((0, 1), (0,))
    with pytest.raises(ValueError, match="non-empty batch"):
        supports_from_masks(t.empty((0, 2, 1), dtype=t.bool), grid)
    with pytest.raises(ValueError, match="empty mask"):
        supports_from_masks(t.zeros((1, 2, 1), dtype=t.bool), grid)
    duplicate = t.tensor([[[True], [False]], [[True], [False]]])
    with pytest.raises(RuntimeError, match="duplicate masks"):
        supports_from_masks(duplicate, grid)
    with pytest.raises(ValueError, match="at least one support"):
        masks_from_supports((), grid)
    with pytest.raises(ValueError, match="sorted, and unique"):
        masks_from_supports(((Site(1, 0), Site(0, 0)),), grid)


def test_support_masks_round_trip_exactly() -> None:
    grid = SiteGrid((3, 5), (7, 9))
    supports = ((Site(3, 7),), (Site(3, 9), Site(5, 7)))

    masks = masks_from_supports(supports, grid)

    assert masks.dtype is t.bool
    assert tuple(masks.shape) == (2, 2, 2)
    assert supports_from_masks(masks, grid) == supports


def test_initial_proposals_are_deterministic_and_exclude_prior_tests() -> None:
    sites = tuple(Site(index, 0) for index in range(6))
    discovered = ((sites[0], sites[1]), (sites[0], sites[2]))
    excluded = frozenset({(sites[0], sites[1])})

    first = initial_recall_proposals(
        sites,
        sites[:3],
        sites[:1],
        discovered,
        excluded,
        _config(),
    )
    second = initial_recall_proposals(
        sites,
        sites[:3],
        sites[:1],
        discovered,
        excluded,
        _config(),
    )

    assert first == second
    assert all(proposal.sites not in excluded for proposal in first)
    assert any("uniform_pair" in proposal.proposal_modes for proposal in first)
    assert any("anchor_partner_sweep" in proposal.proposal_modes for proposal in first)
    assert any("local_truth_table" in proposal.proposal_modes for proposal in first)


def test_initial_proposals_support_an_empty_fourier_discovery_seed() -> None:
    sites = tuple(Site(index, 0) for index in range(6))

    proposals = initial_recall_proposals(
        sites,
        sites[:3],
        sites[:1],
        (),
        frozenset(),
        _config(),
    )

    modes = {mode for proposal in proposals for mode in proposal.proposal_modes}
    assert "screened_pair_completion" in modes
    assert "anchor_partner_sweep" in modes
    assert "uniform_pair" in modes
    assert "known_minset_one_site_mutation" not in modes
    assert "local_truth_table" not in modes


def test_triple_proposals_are_deterministic_and_have_no_sufficient_pair_child() -> None:
    sites = tuple(Site(index, 0) for index in range(8))
    pair_probabilities: dict[SiteSet, float] = {
        pair: 0.1 + index / 100 for index, pair in enumerate(itertools.combinations(sites, 2))
    }
    sufficient = frozenset({(sites[0], sites[1])})

    proposals = triple_recall_proposals(
        sites,
        pair_probabilities,
        sufficient,
        frozenset(),
        _config(),
    )

    assert len(proposals) >= 2
    for proposal in proposals:
        assert all(pair not in sufficient for pair in child_pairs((proposal.sites,)))


def test_exact_truth_table_recovers_minsets_and_monotonicity_violation() -> None:
    a, b, c = (Site(index, 0) for index in range(3))
    sites = (a, b, c)
    truth = {
        (a,): False,
        (b,): False,
        (c,): False,
        (a, b): True,
        (a, c): True,
        (b, c): True,
        (a, b, c): True,
    }
    assert exact_local_minsets(sites, truth) == ((a, b), (a, c), (b, c))
    assert immediate_monotonicity_violations(sites, truth) == ()

    nonmonotone = {**truth, (a, b, c): False}
    violations = immediate_monotonicity_violations(sites, nonmonotone)
    assert violations == (
        ((a, b), (a, b, c)),
        ((a, c), (a, b, c)),
        ((b, c), (a, b, c)),
    )


def test_wilson_interval_handles_zero_and_nonzero_discoveries() -> None:
    empty = wilson_interval(0, 1_000, 1.959963984540054)
    observed = wilson_interval(12, 1_000, 1.959963984540054)

    assert empty.estimate == 0.0
    assert empty.lower == 0.0
    assert 0.0 < empty.upper < 0.01
    assert 0.0 < observed.lower < observed.estimate < observed.upper < 0.1
