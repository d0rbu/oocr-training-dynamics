"""Deterministic proposal and analysis contracts for Fourier minset recall audits."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Bool, jaxtyped

from oocr_training_dynamics.fourier_circuits import Site, SiteGrid, SiteSet

MaskBatch = Bool[t.Tensor, "batch token layer"]


@beartype
@dataclass(frozen=True)
class RecallProposalConfig:
    """Required budgets for one independently versioned recall audit."""

    seed: int
    local_truth_table_maximum_sites: int
    anchor_count: int
    uniform_pair_budget: int
    mutation_pair_budget: int
    uniform_triple_budget: int
    near_miss_pair_count: int
    near_miss_triples_per_pair: int
    patch_batch_size: int
    proposal_shard_size: int
    maximum_initial_evaluations: int
    maximum_pair_evaluations: int
    maximum_triple_evaluations: int
    wilson_z_score: float

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("recall-audit seed must be non-negative")
        if not 2 <= self.local_truth_table_maximum_sites <= 20:
            raise ValueError("local truth-table site cap must lie in [2, 20]")
        positive = (
            self.anchor_count,
            self.uniform_pair_budget,
            self.mutation_pair_budget,
            self.uniform_triple_budget,
            self.near_miss_pair_count,
            self.near_miss_triples_per_pair,
            self.patch_batch_size,
            self.proposal_shard_size,
            self.maximum_initial_evaluations,
            self.maximum_pair_evaluations,
            self.maximum_triple_evaluations,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("recall-audit budgets and batch size must be positive")
        if not math.isfinite(self.wilson_z_score) or self.wilson_z_score <= 0.0:
            raise ValueError("Wilson z-score must be finite and positive")


@beartype
@dataclass(frozen=True)
class ProposedSupport:
    sites: SiteSet
    proposal_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sites) < 2 or tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("recall proposals must be sorted unique multi-site supports")
        if (
            not self.proposal_modes
            or tuple(sorted(set(self.proposal_modes))) != self.proposal_modes
            or any(not mode for mode in self.proposal_modes)
        ):
            raise ValueError("recall proposal modes must be non-empty, sorted, and unique")


@beartype
@dataclass(frozen=True)
class WilsonInterval:
    estimate: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower <= self.estimate <= self.upper <= 1.0:
            raise ValueError("Wilson interval must be ordered inside [0, 1]")


@beartype
def canonical_support(sites: tuple[Site, ...]) -> SiteSet:
    support = tuple(sorted(set(sites)))
    if not support:
        raise ValueError("site support must be non-empty")
    return support


@jaxtyped(typechecker=beartype)
def supports_from_masks(masks: MaskBatch, grid: SiteGrid) -> tuple[SiteSet, ...]:
    if tuple(masks.shape[1:]) != grid.shape or masks.shape[0] <= 0:
        raise ValueError("support masks must be a non-empty batch matching the site grid")
    flattened = masks.reshape(masks.shape[0], grid.site_count).to(device="cpu")
    supports: list[SiteSet] = []
    for row in flattened:
        indices = t.nonzero(row, as_tuple=False).reshape(-1).tolist()
        support = tuple(grid.site(int(index)) for index in indices)
        if not support:
            raise ValueError("tested support inventory unexpectedly contains an empty mask")
        supports.append(support)
    if len(set(supports)) != len(supports):
        raise RuntimeError("tested support inventory contains duplicate masks")
    return tuple(supports)


@jaxtyped(typechecker=beartype)
def masks_from_supports(supports: tuple[SiteSet, ...], grid: SiteGrid) -> MaskBatch:
    if not supports:
        raise ValueError("mask construction requires at least one support")
    masks = t.zeros((len(supports), *grid.shape), dtype=t.bool)
    for row, support in enumerate(supports):
        if not support or tuple(sorted(set(support))) != support:
            raise ValueError("mask supports must be non-empty, sorted, and unique")
        for site in support:
            flat_index = grid.flat_index(site)
            token_offset, layer_offset = divmod(flat_index, len(grid.layers))
            masks[row, token_offset, layer_offset] = True
    return masks


@beartype
def local_truth_table_supports(
    discovered_minsets: tuple[SiteSet, ...],
    excluded: frozenset[SiteSet],
    maximum_sites: int,
) -> tuple[ProposedSupport, ...]:
    if not discovered_minsets:
        raise ValueError("local truth-table audit requires discovered minsets")
    sites = tuple(sorted({site for minset in discovered_minsets for site in minset}))
    if not 2 <= len(sites) <= maximum_sites:
        raise RuntimeError(
            f"local truth-table union has {len(sites)} sites outside the configured [2, {maximum_sites}] cap"
        )
    supports = (
        tuple(combination)
        for size in range(2, len(sites) + 1)
        for combination in itertools.combinations(sites, size)
    )
    return tuple(
        ProposedSupport(support, ("local_truth_table",))
        for support in supports
        if support not in excluded
    )


@beartype
def _add_proposal(
    proposals: dict[SiteSet, set[str]],
    support: SiteSet,
    mode: str,
    excluded: frozenset[SiteSet],
) -> None:
    if len(support) < 2 or support in excluded:
        return
    proposals.setdefault(support, set()).add(mode)


@beartype
def _random_pair(
    sites: tuple[Site, ...],
    generator: t.Generator,
) -> SiteSet:
    indices = t.randint(0, len(sites), (2,), generator=generator)
    while int(indices[0]) == int(indices[1]):
        indices[1] = t.randint(0, len(sites), (), generator=generator)
    return canonical_support((sites[int(indices[0])], sites[int(indices[1])]))


@beartype
def _random_triple(
    sites: tuple[Site, ...],
    generator: t.Generator,
) -> SiteSet:
    indices = t.randint(0, len(sites), (3,), generator=generator)
    while len({int(index) for index in indices}) != 3:
        indices = t.randint(0, len(sites), (3,), generator=generator)
    return canonical_support(tuple(sites[int(index)] for index in indices))


@beartype
def initial_recall_proposals(
    active_sites: tuple[Site, ...],
    screened_sites: tuple[Site, ...],
    anchors: tuple[Site, ...],
    discovered_minsets: tuple[SiteSet, ...],
    excluded: frozenset[SiteSet],
    config: RecallProposalConfig,
) -> tuple[ProposedSupport, ...]:
    if (
        len(active_sites) < 3
        or tuple(sorted(set(active_sites))) != active_sites
        or not screened_sites
        or not anchors
    ):
        raise ValueError("recall proposal inputs must contain canonical active/search site sets")
    if any(site not in set(active_sites) for site in (*screened_sites, *anchors)):
        raise ValueError("screened sites and anchors must belong to the active site universe")
    if len(anchors) != config.anchor_count:
        raise ValueError("anchor list must exactly match the registered anchor count")
    generator = t.Generator(device="cpu").manual_seed(config.seed)
    proposals: dict[SiteSet, set[str]] = {}

    for pair in itertools.combinations(screened_sites, 2):
        _add_proposal(
            proposals,
            canonical_support(pair),
            "screened_pair_completion",
            excluded,
        )
    for anchor in anchors:
        for partner in active_sites:
            if partner != anchor:
                _add_proposal(
                    proposals,
                    canonical_support((anchor, partner)),
                    "anchor_partner_sweep",
                    excluded,
                )

    uniform_pairs: set[SiteSet] = set()
    attempts = 0
    while len(uniform_pairs) < config.uniform_pair_budget:
        attempts += 1
        if attempts > config.uniform_pair_budget * 100:
            raise RuntimeError("uniform pair sampler exhausted its deterministic attempt cap")
        pair = _random_pair(active_sites, generator)
        if pair not in excluded:
            uniform_pairs.add(pair)
    for pair in uniform_pairs:
        _add_proposal(proposals, pair, "uniform_pair", excluded)

    if discovered_minsets:
        discovered_sites = tuple(sorted({site for minset in discovered_minsets for site in minset}))
        mutated_pairs: set[SiteSet] = set()
        attempts = 0
        while len(mutated_pairs) < config.mutation_pair_budget:
            attempts += 1
            if attempts > config.mutation_pair_budget * 100:
                raise RuntimeError("mutated pair sampler exhausted its deterministic attempt cap")
            source = discovered_sites[
                int(t.randint(0, len(discovered_sites), (), generator=generator))
            ]
            partner = active_sites[int(t.randint(0, len(active_sites), (), generator=generator))]
            if source == partner:
                continue
            pair = canonical_support((source, partner))
            if pair not in excluded:
                mutated_pairs.add(pair)
        for pair in mutated_pairs:
            _add_proposal(proposals, pair, "known_minset_one_site_mutation", excluded)

        local = local_truth_table_supports(
            discovered_minsets,
            excluded,
            config.local_truth_table_maximum_sites,
        )
        for proposal in local:
            _add_proposal(proposals, proposal.sites, proposal.proposal_modes[0], excluded)

    pair_count = sum(len(support) == 2 for support in proposals)
    if pair_count > config.maximum_pair_evaluations:
        raise RuntimeError(
            f"recall proposal plan contains {pair_count} pairs, exceeding the explicit cap "
            f"{config.maximum_pair_evaluations}"
        )
    if len(proposals) > config.maximum_initial_evaluations:
        raise RuntimeError(
            f"recall proposal plan contains {len(proposals)} initial supports, exceeding the "
            f"explicit cap {config.maximum_initial_evaluations}"
        )
    ordered = sorted(proposals.items(), key=lambda item: (len(item[0]), item[0]))
    return tuple(ProposedSupport(support, tuple(sorted(modes))) for support, modes in ordered)


@beartype
def triple_recall_proposals(
    active_sites: tuple[Site, ...],
    pair_probabilities: dict[SiteSet, float],
    sufficient_pairs: frozenset[SiteSet],
    excluded: frozenset[SiteSet],
    config: RecallProposalConfig,
) -> tuple[ProposedSupport, ...]:
    if len(active_sites) < 3 or not pair_probabilities:
        raise ValueError("triple recall proposals require active sites and measured pairs")
    generator = t.Generator(device="cpu").manual_seed(config.seed + 1)
    proposals: dict[SiteSet, set[str]] = {}

    uniform: set[SiteSet] = set()
    attempts = 0
    while len(uniform) < config.uniform_triple_budget:
        attempts += 1
        if attempts > config.uniform_triple_budget * 100:
            raise RuntimeError("uniform triple sampler exhausted its deterministic attempt cap")
        support = _random_triple(active_sites, generator)
        children = tuple(itertools.combinations(support, 2))
        if support not in excluded and not any(child in sufficient_pairs for child in children):
            uniform.add(support)
    for support in uniform:
        _add_proposal(proposals, support, "uniform_triple", excluded)

    near_miss_pairs = tuple(
        support
        for support, _probability in sorted(
            pair_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if support not in sufficient_pairs
    )[: config.near_miss_pair_count]
    if len(near_miss_pairs) != config.near_miss_pair_count:
        raise RuntimeError("pair audit did not produce the registered near-miss pair count")
    for pair in near_miss_pairs:
        added = 0
        attempts = 0
        while added < config.near_miss_triples_per_pair:
            attempts += 1
            if attempts > config.near_miss_triples_per_pair * 100:
                raise RuntimeError("near-miss triple sampler exhausted its attempt cap")
            third = active_sites[int(t.randint(0, len(active_sites), (), generator=generator))]
            if third in pair:
                continue
            support = canonical_support((*pair, third))
            children = tuple(itertools.combinations(support, 2))
            if support in excluded or any(child in sufficient_pairs for child in children):
                continue
            before = len(proposals)
            _add_proposal(proposals, support, "near_miss_pair_expansion", excluded)
            if len(proposals) > before:
                added += 1

    if len(proposals) > config.maximum_triple_evaluations:
        raise RuntimeError(
            f"recall proposal plan contains {len(proposals)} triples, exceeding the explicit cap "
            f"{config.maximum_triple_evaluations}"
        )
    return tuple(
        ProposedSupport(support, tuple(sorted(modes)))
        for support, modes in sorted(proposals.items())
    )


@beartype
def child_pairs(supports: tuple[SiteSet, ...]) -> tuple[SiteSet, ...]:
    if not supports or any(len(support) != 3 for support in supports):
        raise ValueError("child-pair expansion requires non-empty triple supports")
    return tuple(
        sorted({tuple(pair) for support in supports for pair in itertools.combinations(support, 2)})
    )


@beartype
def exact_local_minsets(
    local_sites: tuple[Site, ...],
    sufficiency: dict[SiteSet, bool],
) -> tuple[SiteSet, ...]:
    if not local_sites or not sufficiency:
        raise ValueError("local minset extraction requires sites and a truth table")
    local_set = set(local_sites)
    expected = {
        tuple(combination)
        for size in range(1, len(local_sites) + 1)
        for combination in itertools.combinations(local_sites, size)
    }
    if set(sufficiency) != expected:
        raise RuntimeError("local truth table is not exhaustive over every non-empty subset")
    minsets: list[SiteSet] = []
    for support in sorted(sufficiency, key=lambda item: (len(item), item)):
        if not set(support).issubset(local_set) or not sufficiency[support]:
            continue
        if not any(set(candidate).issubset(support) for candidate in minsets):
            minsets.append(support)
    return tuple(minsets)


@beartype
def immediate_monotonicity_violations(
    local_sites: tuple[Site, ...],
    sufficiency: dict[SiteSet, bool],
) -> tuple[tuple[SiteSet, SiteSet], ...]:
    if not local_sites or not sufficiency:
        raise ValueError("monotonicity check requires sites and a truth table")
    violations: list[tuple[SiteSet, SiteSet]] = []
    for support, sufficient in sufficiency.items():
        if not sufficient:
            continue
        support_set = set(support)
        for site in local_sites:
            if site in support_set:
                continue
            parent = canonical_support((*support, site))
            if not sufficiency[parent]:
                violations.append((support, parent))
    return tuple(sorted(violations))


@beartype
def wilson_interval(successes: int, trials: int, z_score: float) -> WilsonInterval:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    if not math.isfinite(z_score) or z_score <= 0.0:
        raise ValueError("Wilson z-score must be finite and positive")
    proportion = successes / trials
    z_squared = z_score**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z_score
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials**2))
        / denominator
    )
    return WilsonInterval(
        proportion,
        min(proportion, max(0.0, center - radius)),
        max(proportion, min(1.0, center + radius)),
    )


__all__ = [
    "ProposedSupport",
    "RecallProposalConfig",
    "WilsonInterval",
    "canonical_support",
    "child_pairs",
    "exact_local_minsets",
    "immediate_monotonicity_violations",
    "initial_recall_proposals",
    "local_truth_table_supports",
    "masks_from_supports",
    "supports_from_masks",
    "triple_recall_proposals",
    "wilson_interval",
]
