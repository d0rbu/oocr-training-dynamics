"""Deterministic graph components and structural groupings for verified minsets."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.fourier_circuits import Site, SiteSet


@beartype
@dataclass(frozen=True)
class ColoredSite:
    site: Site
    color_index: int

    def __post_init__(self) -> None:
        if self.color_index < 0:
            raise ValueError("network color indices must be non-negative")


@beartype
@dataclass(frozen=True)
class ColoredEdge:
    source: Site
    target: Site
    minset_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source >= self.target:
            raise ValueError("network edges must have strictly ordered endpoints")
        if not self.minset_indices or tuple(sorted(set(self.minset_indices))) != self.minset_indices:
            raise ValueError("network edge minset indices must be non-empty, sorted, and unique")


@beartype
@dataclass(frozen=True)
class MinsetNetwork:
    minset_size: int
    component_index: int
    color_count: int
    colorable: bool
    minset_indices: tuple[int, ...]
    sites: tuple[ColoredSite, ...]
    edges: tuple[ColoredEdge, ...]

    def __post_init__(self) -> None:
        if self.minset_size < 2 or self.component_index <= 0:
            raise ValueError("multi-site networks require size >= 2 and a positive component index")
        if not self.minset_size <= self.color_count <= len(self.sites):
            raise ValueError("network color count must lie between minset size and site count")
        if self.colorable is not (self.color_count == self.minset_size):
            raise ValueError("network n-colorability flag disagrees with its minimum color count")
        if not self.minset_indices or tuple(sorted(set(self.minset_indices))) != self.minset_indices:
            raise ValueError("network minset indices must be non-empty, sorted, and unique")
        ordered_sites = tuple(sorted(colored.site for colored in self.sites))
        if not self.sites or len(set(ordered_sites)) != len(ordered_sites):
            raise ValueError("network sites must be non-empty and unique")
        if any(not 0 <= colored.color_index < self.color_count for colored in self.sites):
            raise ValueError("network site color lies outside the registered palette")
        if not self.edges:
            raise ValueError("multi-site networks must contain at least one graph edge")


@beartype
@dataclass(frozen=True)
class PartnerProfileCluster:
    minset_size: int
    component_index: int
    cluster_index: int
    sites: tuple[Site, ...]
    minimum_partner_jaccard: float
    mean_partner_jaccard: float

    def __post_init__(self) -> None:
        if self.minset_size < 2 or self.component_index <= 0 or self.cluster_index <= 0:
            raise ValueError("partner-profile cluster identifiers must be positive")
        if not self.sites or tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("partner-profile cluster sites must be non-empty, sorted, and unique")
        if not (
            0.0
            <= self.minimum_partner_jaccard
            <= self.mean_partner_jaccard
            <= 1.0
        ):
            raise ValueError("partner-profile similarities must be ordered inside [0, 1]")


@beartype
@dataclass(frozen=True)
class PartnerProfileNetwork:
    """One connected equal-size minset hypergraph with scalable site groupings."""

    minset_size: int
    component_index: int
    minset_indices: tuple[int, ...]
    sites: tuple[Site, ...]
    edges: tuple[ColoredEdge, ...]
    clusters: tuple[PartnerProfileCluster, ...]

    def __post_init__(self) -> None:
        if self.minset_size < 2 or self.component_index <= 0:
            raise ValueError("multi-site networks require size >= 2 and a positive component index")
        if not self.minset_indices or tuple(sorted(set(self.minset_indices))) != self.minset_indices:
            raise ValueError("network minset indices must be non-empty, sorted, and unique")
        if not self.sites or tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("network sites must be non-empty, sorted, and unique")
        if not self.edges or not self.clusters:
            raise ValueError("multi-site networks require graph edges and structural clusters")
        clustered_sites = tuple(sorted(site for cluster in self.clusters for site in cluster.sites))
        if clustered_sites != self.sites:
            raise ValueError("partner-profile clusters must partition network sites exactly")


@beartype
def _find_coloring(
    adjacency: dict[Site, set[Site]],
    color_count: int,
) -> dict[Site, int] | None:
    """Return a deterministic coloring, or ``None`` when this palette is insufficient."""

    if color_count < 2 or not adjacency:
        raise ValueError("exact graph coloring requires a non-empty graph and at least two colors")
    colors: dict[Site, int] = {}

    def solve() -> bool:
        if len(colors) == len(adjacency):
            return True
        uncolored = tuple(site for site in adjacency if site not in colors)
        site = min(
            uncolored,
            key=lambda candidate: (
                -len({colors[neighbor] for neighbor in adjacency[candidate] if neighbor in colors}),
                -len(adjacency[candidate]),
                candidate,
            ),
        )
        forbidden = {colors[neighbor] for neighbor in adjacency[site] if neighbor in colors}
        for color_index in range(color_count):
            if color_index in forbidden:
                continue
            colors[site] = color_index
            if solve():
                return True
            del colors[site]
        return False

    return colors if solve() else None


@beartype
def _minimum_coloring(
    adjacency: dict[Site, set[Site]],
    minimum_color_count: int,
) -> tuple[dict[Site, int], int]:
    """Find the exact chromatic number at or above the minset-size lower bound."""

    for color_count in range(minimum_color_count, len(adjacency) + 1):
        colors = _find_coloring(adjacency, color_count)
        if colors is not None:
            return colors, color_count
    raise RuntimeError("non-empty finite graph did not admit a finite exact coloring")


@beartype
def cluster_minset_networks(minsets: tuple[SiteSet, ...]) -> tuple[MinsetNetwork, ...]:
    """Clique-expand equal-size minsets, cluster components, and exactly n-color each."""

    canonical: list[SiteSet] = []
    for minset in minsets:
        if len(minset) < 2 or tuple(sorted(set(minset))) != minset:
            raise ValueError("network input minsets must be sorted unique site sets of size >= 2")
        canonical.append(minset)
    if len(set(canonical)) != len(canonical):
        raise ValueError("network input contains a duplicate minset")

    networks: list[MinsetNetwork] = []
    sizes = sorted({len(minset) for minset in canonical})
    for size in sizes:
        indexed_minsets = tuple(
            (index, minset) for index, minset in enumerate(canonical) if len(minset) == size
        )
        edge_memberships: dict[tuple[Site, Site], set[int]] = {}
        adjacency: dict[Site, set[Site]] = {}
        for minset_index, minset in indexed_minsets:
            for site in minset:
                adjacency.setdefault(site, set())
            for source, target in itertools.combinations(minset, 2):
                edge = (source, target)
                edge_memberships.setdefault(edge, set()).add(minset_index)
                adjacency[source].add(target)
                adjacency[target].add(source)

        remaining = set(adjacency)
        components: list[tuple[Site, ...]] = []
        while remaining:
            root = min(remaining)
            frontier = [root]
            component_sites: set[Site] = set()
            while frontier:
                site = frontier.pop()
                if site in component_sites:
                    continue
                component_sites.add(site)
                frontier.extend(sorted(adjacency[site] - component_sites, reverse=True))
            remaining.difference_update(component_sites)
            components.append(tuple(sorted(component_sites)))
        components.sort(key=lambda component: component[0])

        for component_index, component in enumerate(components, start=1):
            component_set = set(component)
            component_minsets = tuple(
                (index, minset)
                for index, minset in indexed_minsets
                if minset[0] in component_set
            )
            if any(not set(minset).issubset(component_set) for _index, minset in component_minsets):
                raise RuntimeError("minset crosses graph connected components")
            component_adjacency = {
                site: adjacency[site].intersection(component_set) for site in component
            }
            colors, color_count = _minimum_coloring(component_adjacency, size)
            for _index, minset in component_minsets:
                if len({colors[site] for site in minset}) != size:
                    raise RuntimeError(
                        f"size-{size} minset does not receive {size} distinct graph colors"
                    )
            component_edges = tuple(
                ColoredEdge(source, target, tuple(sorted(indices)))
                for (source, target), indices in sorted(edge_memberships.items())
                if source in component_set
            )
            networks.append(
                MinsetNetwork(
                    minset_size=size,
                    component_index=component_index,
                    color_count=color_count,
                    colorable=color_count == size,
                    minset_indices=tuple(index for index, _minset in component_minsets),
                    sites=tuple(ColoredSite(site, colors[site]) for site in component),
                    edges=component_edges,
                )
            )
    return tuple(networks)


@beartype
def _partner_jaccard(
    left: Site,
    right: Site,
    partners: dict[Site, set[Site]],
) -> float:
    left_profile = partners[left] - {right}
    right_profile = partners[right] - {left}
    union = left_profile.union(right_profile)
    return 1.0 if not union else len(left_profile.intersection(right_profile)) / len(union)


@beartype
def _cluster_similarity(
    left: tuple[Site, ...],
    right: tuple[Site, ...],
    partners: dict[Site, set[Site]],
) -> tuple[float, float]:
    similarities = tuple(
        _partner_jaccard(left_site, right_site, partners)
        for left_site in left
        for right_site in right
    )
    return min(similarities), sum(similarities) / len(similarities)


@beartype
def cluster_by_partner_profiles(
    minsets: tuple[SiteSet, ...],
    minimum_similarity: float,
) -> tuple[PartnerProfileCluster, ...]:
    """Complete-link cluster non-cooccurring sites with similar hypergraph partners."""

    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("partner-profile clustering threshold must lie in [0, 1]")
    networks = cluster_minset_networks(minsets)
    output: list[PartnerProfileCluster] = []
    for network in networks:
        component_minsets = tuple(minsets[index] for index in network.minset_indices)
        partners: dict[Site, set[Site]] = {
            colored.site: set() for colored in network.sites
        }
        for minset in component_minsets:
            for site in minset:
                partners[site].update(other for other in minset if other != site)
        clusters: list[tuple[Site, ...]] = [
            (site,) for site in sorted(partners)
        ]
        while True:
            candidates: list[
                tuple[float, float, tuple[Site, ...], int, int]
            ] = []
            for left_index, left in enumerate(clusters):
                for right_index in range(left_index + 1, len(clusters)):
                    right = clusters[right_index]
                    if any(
                        right_site in partners[left_site]
                        for left_site in left
                        for right_site in right
                    ):
                        continue
                    minimum, mean = _cluster_similarity(left, right, partners)
                    if minimum >= minimum_similarity:
                        candidates.append(
                            (
                                minimum,
                                mean,
                                tuple(sorted((*left, *right))),
                                left_index,
                                right_index,
                            )
                        )
            if not candidates:
                break
            _minimum, _mean, merged, left_index, right_index = max(
                candidates,
                key=lambda item: (item[0], item[1], -len(item[2]), tuple(reversed(item[2]))),
            )
            clusters = [
                cluster
                for index, cluster in enumerate(clusters)
                if index not in {left_index, right_index}
            ]
            clusters.append(merged)
            clusters.sort()
        for cluster_index, cluster in enumerate(clusters, start=1):
            pairwise = tuple(
                _partner_jaccard(left, right, partners)
                for left, right in itertools.combinations(cluster, 2)
            )
            minimum = min(pairwise) if pairwise else 1.0
            mean = sum(pairwise) / len(pairwise) if pairwise else 1.0
            output.append(
                PartnerProfileCluster(
                    network.minset_size,
                    network.component_index,
                    cluster_index,
                    cluster,
                    minimum,
                    mean,
                )
            )
    return tuple(output)


@beartype
def cluster_minset_hypergraph_networks(
    minsets: tuple[SiteSet, ...],
    minimum_similarity: float,
) -> tuple[PartnerProfileNetwork, ...]:
    """Cluster large minset hypergraphs without requiring exponential graph coloring.

    Equal-size minsets are clique-expanded only to identify connected components and
    neighbor profiles; the original minset memberships remain the authoritative
    hyperedges. Sites with identical profiles seed clusters. Remaining seed groups are
    assigned deterministically by complete-link Jaccard similarity, with a hard
    cannot-link constraint for any sites that co-occur in a minset.
    """

    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("partner-profile clustering threshold must lie in [0, 1]")
    canonical: list[SiteSet] = []
    for minset in minsets:
        if len(minset) < 2 or tuple(sorted(set(minset))) != minset:
            raise ValueError("network input minsets must be sorted unique site sets of size >= 2")
        canonical.append(minset)
    if len(set(canonical)) != len(canonical):
        raise ValueError("network input contains a duplicate minset")

    output: list[PartnerProfileNetwork] = []
    for size in sorted({len(minset) for minset in canonical}):
        indexed_minsets = tuple(
            (index, minset) for index, minset in enumerate(canonical) if len(minset) == size
        )
        edge_memberships: dict[tuple[Site, Site], set[int]] = {}
        adjacency: dict[Site, set[Site]] = {}
        for minset_index, minset in indexed_minsets:
            for site in minset:
                adjacency.setdefault(site, set())
            for source, target in itertools.combinations(minset, 2):
                edge_memberships.setdefault((source, target), set()).add(minset_index)
                adjacency[source].add(target)
                adjacency[target].add(source)

        remaining = set(adjacency)
        components: list[tuple[Site, ...]] = []
        while remaining:
            root = min(remaining)
            frontier = [root]
            component_sites: set[Site] = set()
            while frontier:
                site = frontier.pop()
                if site in component_sites:
                    continue
                component_sites.add(site)
                frontier.extend(sorted(adjacency[site] - component_sites, reverse=True))
            remaining.difference_update(component_sites)
            components.append(tuple(sorted(component_sites)))
        components.sort(key=lambda component: component[0])

        for component_index, component in enumerate(components, start=1):
            component_set = set(component)
            component_minsets = tuple(
                (index, minset)
                for index, minset in indexed_minsets
                if minset[0] in component_set
            )
            if any(not set(minset).issubset(component_set) for _index, minset in component_minsets):
                raise RuntimeError("minset crosses graph connected components")
            partners = {
                site: adjacency[site].intersection(component_set) for site in component
            }

            profile_buckets: dict[frozenset[Site], list[Site]] = {}
            for site in component:
                profile_buckets.setdefault(frozenset(partners[site]), []).append(site)
            seed_groups: list[tuple[Site, ...]] = []
            for bucket in profile_buckets.values():
                independent_groups: list[list[Site]] = []
                for site in sorted(bucket):
                    destination = next(
                        (
                            group
                            for group in independent_groups
                            if all(other not in partners[site] for other in group)
                        ),
                        None,
                    )
                    if destination is None:
                        independent_groups.append([site])
                    else:
                        destination.append(site)
                seed_groups.extend(tuple(group) for group in independent_groups)
            seed_groups.sort(key=lambda group: (-len(group), group))

            clusters: list[tuple[Site, ...]] = []
            for seed in seed_groups:
                candidates: list[tuple[float, float, int]] = []
                for cluster_index, cluster in enumerate(clusters):
                    if any(
                        right_site in partners[left_site]
                        for left_site in seed
                        for right_site in cluster
                    ):
                        continue
                    minimum, mean = _cluster_similarity(seed, cluster, partners)
                    if minimum >= minimum_similarity:
                        candidates.append((minimum, mean, cluster_index))
                if not candidates:
                    clusters.append(seed)
                    continue
                _minimum, _mean, destination_index = max(
                    candidates,
                    key=lambda candidate: (candidate[0], candidate[1], -candidate[2]),
                )
                clusters[destination_index] = tuple(
                    sorted((*clusters[destination_index], *seed))
                )
            clusters.sort(key=lambda cluster: (-len(cluster), cluster))

            site_to_cluster: dict[Site, int] = {}
            profile_clusters: list[PartnerProfileCluster] = []
            for cluster_index, cluster in enumerate(clusters, start=1):
                profiles = {frozenset(partners[site]) for site in cluster}
                if len(cluster) <= 1 or len(profiles) == 1:
                    minimum = mean = 1.0
                else:
                    similarities = tuple(
                        _partner_jaccard(left, right, partners)
                        for left, right in itertools.combinations(cluster, 2)
                    )
                    minimum = min(similarities)
                    mean = sum(similarities) / len(similarities)
                if minimum < minimum_similarity:
                    raise RuntimeError("scalable partner-profile cluster violates complete-link threshold")
                for site in cluster:
                    if site in site_to_cluster:
                        raise RuntimeError("network site appears in multiple partner-profile clusters")
                    site_to_cluster[site] = cluster_index
                profile_clusters.append(
                    PartnerProfileCluster(
                        size,
                        component_index,
                        cluster_index,
                        cluster,
                        minimum,
                        mean,
                    )
                )
            if set(site_to_cluster) != component_set:
                raise RuntimeError("partner-profile clusters do not cover the network component")
            for _index, minset in component_minsets:
                if len({site_to_cluster[site] for site in minset}) != size:
                    raise RuntimeError("one minset contains two sites in the same structural cluster")

            component_edges = tuple(
                ColoredEdge(source, target, tuple(sorted(indices)))
                for (source, target), indices in sorted(edge_memberships.items())
                if source in component_set
            )
            output.append(
                PartnerProfileNetwork(
                    minset_size=size,
                    component_index=component_index,
                    minset_indices=tuple(index for index, _minset in component_minsets),
                    sites=component,
                    edges=component_edges,
                    clusters=tuple(profile_clusters),
                )
            )
    return tuple(output)


__all__ = [
    "ColoredEdge",
    "ColoredSite",
    "MinsetNetwork",
    "PartnerProfileCluster",
    "PartnerProfileNetwork",
    "cluster_by_partner_profiles",
    "cluster_minset_hypergraph_networks",
    "cluster_minset_networks",
]
