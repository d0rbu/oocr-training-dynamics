from __future__ import annotations

import pytest

from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.fourier_networks import (
    ColoredEdge,
    ColoredSite,
    MinsetNetwork,
    PartnerProfileCluster,
    PartnerProfileNetwork,
    cluster_by_partner_profiles,
    cluster_minset_hypergraph_networks,
    cluster_minset_networks,
)


def test_pair_minsets_form_deterministic_bipartite_components() -> None:
    a, b, c, d, e = (Site(index, 0) for index in range(5))

    networks = cluster_minset_networks(((a, b), (a, c), (d, e)))

    assert len(networks) == 2
    first, second = networks
    assert first.minset_size == 2
    assert first.component_index == 1
    assert first.color_count == 2
    assert first.colorable is True
    assert first.minset_indices == (0, 1)
    assert {colored.site: colored.color_index for colored in first.sites} == {
        a: 0,
        b: 1,
        c: 1,
    }
    assert tuple((edge.source, edge.target) for edge in first.edges) == ((a, b), (a, c))
    assert second.component_index == 2
    assert second.minset_indices == (2,)
    assert {colored.color_index for colored in second.sites} == {0, 1}


def test_size_three_minsets_add_clique_edges_and_receive_three_colors() -> None:
    a, b, c, d = (Site(index, 0) for index in range(4))

    (network,) = cluster_minset_networks(((a, b, c), (a, b, d)))

    assert network.minset_size == 3
    assert len(network.edges) == 5
    colors = {colored.site: colored.color_index for colored in network.sites}
    assert {colors[site] for site in (a, b, c)} == {0, 1, 2}
    assert {colors[site] for site in (a, b, d)} == {0, 1, 2}
    shared_edge = next(edge for edge in network.edges if (edge.source, edge.target) == (a, b))
    assert shared_edge.minset_indices == (0, 1)


def test_non_bipartite_pair_component_reports_exact_three_color_minimum() -> None:
    a, b, c = (Site(index, 0) for index in range(3))

    (network,) = cluster_minset_networks(((a, b), (a, c), (b, c)))

    assert network.minset_size == 2
    assert network.color_count == 3
    assert network.colorable is False
    assert {colored.color_index for colored in network.sites} == {0, 1, 2}


def test_size_three_component_requiring_four_colors_reports_violation() -> None:
    a, b, c, d = (Site(index, 0) for index in range(4))

    (network,) = cluster_minset_networks(
        ((a, b, c), (a, b, d), (a, c, d), (b, c, d))
    )

    assert network.minset_size == 3
    assert network.color_count == 4
    assert network.colorable is False
    assert {colored.color_index for colored in network.sites} == {0, 1, 2, 3}


def test_network_input_rejects_singletons_and_duplicates() -> None:
    a, b = Site(0, 0), Site(1, 0)

    with pytest.raises(ValueError, match="size >= 2"):
        cluster_minset_networks(((a,),))
    with pytest.raises(ValueError, match="duplicate minset"):
        cluster_minset_networks(((a, b), (a, b)))


def test_partner_profile_clustering_groups_star_leaves_not_complements() -> None:
    center, left, middle, right = (Site(index, 0) for index in range(4))

    clusters = cluster_by_partner_profiles(
        ((center, left), (center, middle), (center, right)),
        minimum_similarity=1.0,
    )

    assert tuple(cluster.sites for cluster in clusters) == (
        (center,),
        (left, middle, right),
    )
    assert clusters[1].minimum_partner_jaccard == 1.0


def test_partner_profile_clustering_preserves_triangle_cannot_link_constraints() -> None:
    a, b, c = (Site(index, 0) for index in range(3))

    clusters = cluster_by_partner_profiles(
        ((a, b), (a, c), (b, c)),
        minimum_similarity=0.0,
    )

    assert tuple(cluster.sites for cluster in clusters) == ((a,), (b,), (c,))


def test_hypergraph_partner_profiles_group_substitutes_without_clique_claim() -> None:
    a1, a2, b, c = (Site(index, 0) for index in range(4))

    clusters = cluster_by_partner_profiles(
        ((a1, b, c), (a2, b, c)),
        minimum_similarity=1.0,
    )

    assert tuple(cluster.sites for cluster in clusters) == ((a1, a2), (b,), (c,))
    assert all(cluster.minset_size == 3 for cluster in clusters)


def test_scalable_partner_network_handles_a_large_star_without_recursive_coloring() -> None:
    hub = Site(0, 0)
    leaves = tuple(Site(index, 1) for index in range(1, 1_201))

    (network,) = cluster_minset_hypergraph_networks(
        tuple((hub, leaf) for leaf in leaves),
        minimum_similarity=0.5,
    )

    assert network.minset_size == 2
    assert len(network.minset_indices) == 1_200
    assert len(network.sites) == 1_201
    assert [len(cluster.sites) for cluster in network.clusters] == [1_200, 1]
    assert network.clusters[0].minimum_partner_jaccard == 1.0


def test_scalable_partner_network_preserves_triangle_cannot_link() -> None:
    a, b, c = Site(0, 0), Site(1, 0), Site(2, 0)

    (network,) = cluster_minset_hypergraph_networks(
        ((a, b), (a, c), (b, c)),
        minimum_similarity=0.5,
    )

    assert [cluster.sites for cluster in network.clusters] == [(a,), (b,), (c,)]


def test_scalable_partner_network_keeps_hyperedges_and_components_exact() -> None:
    a, b, c, d, e, f = (Site(index, 0) for index in range(6))
    minsets = ((a, b, c), (a, b, d), (e, f))

    networks = cluster_minset_hypergraph_networks(minsets, minimum_similarity=0.5)

    assert [(network.minset_size, network.component_index) for network in networks] == [
        (2, 1),
        (3, 1),
    ]
    assert networks[0].minset_indices == (2,)
    assert networks[1].minset_indices == (0, 1)
    assert len(networks[0].edges) == 1
    assert len(networks[1].edges) == 5
    for network in networks:
        group_by_site = {
            site: cluster.cluster_index
            for cluster in network.clusters
            for site in cluster.sites
        }
        for index in network.minset_indices:
            assert len({group_by_site[site] for site in minsets[index]}) == len(minsets[index])


def test_network_value_objects_reject_inconsistent_states() -> None:
    a, b = Site(0, 0), Site(1, 0)
    left, right = ColoredSite(a, 0), ColoredSite(b, 1)
    edge = ColoredEdge(a, b, (0,))
    cluster = PartnerProfileCluster(2, 1, 1, (a, b), 1.0, 1.0)

    with pytest.raises(ValueError, match="non-negative"):
        ColoredSite(a, -1)
    with pytest.raises(ValueError, match="ordered endpoints"):
        ColoredEdge(b, a, (0,))
    with pytest.raises(ValueError, match="non-empty, sorted, and unique"):
        ColoredEdge(a, b, ())
    with pytest.raises(ValueError, match="size >= 2"):
        MinsetNetwork(1, 1, 2, False, (0,), (left, right), (edge,))
    with pytest.raises(ValueError, match="color count"):
        MinsetNetwork(2, 1, 1, False, (0,), (left, right), (edge,))
    with pytest.raises(ValueError, match="colorability"):
        MinsetNetwork(2, 1, 2, False, (0,), (left, right), (edge,))
    with pytest.raises(ValueError, match="minset indices"):
        MinsetNetwork(2, 1, 2, True, (), (left, right), (edge,))
    with pytest.raises(ValueError, match="sites must be non-empty and unique"):
        MinsetNetwork(2, 1, 2, True, (0,), (left, left), (edge,))
    with pytest.raises(ValueError, match="outside the registered palette"):
        MinsetNetwork(2, 1, 2, True, (0,), (left, ColoredSite(b, 2)), (edge,))
    with pytest.raises(ValueError, match="graph edge"):
        MinsetNetwork(2, 1, 2, True, (0,), (left, right), ())
    with pytest.raises(ValueError, match="identifiers must be positive"):
        PartnerProfileCluster(2, 0, 1, (a,), 1.0, 1.0)
    with pytest.raises(ValueError, match="sites must be non-empty"):
        PartnerProfileCluster(2, 1, 1, (), 1.0, 1.0)
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        PartnerProfileCluster(2, 1, 1, (a,), 0.8, 0.7)
    with pytest.raises(ValueError, match="size >= 2"):
        PartnerProfileNetwork(1, 1, (0,), (a, b), (edge,), (cluster,))
    with pytest.raises(ValueError, match="minset indices"):
        PartnerProfileNetwork(2, 1, (), (a, b), (edge,), (cluster,))
    with pytest.raises(ValueError, match="sorted, and unique"):
        PartnerProfileNetwork(2, 1, (0,), (b, a), (edge,), (cluster,))
    with pytest.raises(ValueError, match="graph edges and structural clusters"):
        PartnerProfileNetwork(2, 1, (0,), (a, b), (), (cluster,))
    with pytest.raises(ValueError, match="partition network sites"):
        PartnerProfileNetwork(
            2,
            1,
            (0,),
            (a, b),
            (edge,),
            (PartnerProfileCluster(2, 1, 1, (a,), 1.0, 1.0),),
        )


def test_scalable_partner_network_rejects_invalid_inputs() -> None:
    a, b = Site(0, 0), Site(1, 0)

    with pytest.raises(ValueError, match="threshold"):
        cluster_minset_hypergraph_networks(((a, b),), minimum_similarity=1.1)
    with pytest.raises(ValueError, match="size >= 2"):
        cluster_minset_hypergraph_networks(((a,),), minimum_similarity=0.5)
    with pytest.raises(ValueError, match="duplicate minset"):
        cluster_minset_hypergraph_networks(((a, b), (a, b)), minimum_similarity=0.5)
