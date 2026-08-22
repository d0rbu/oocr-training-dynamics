from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch as t
from jaxtyping import TypeCheckError

from oocr_training_dynamics.fourier_circuits import (
    Density,
    DensityPoint,
    DensitySweepConfig,
    LassoConfig,
    LogitDiff,
    Site,
    SiteGrid,
    SweepDensity,
    all_boolean_corners,
    all_supports,
    as_non_empty_candidates,
    as_patch_mask,
    biased_standardized_bits,
    density_curve_is_flat,
    enumerate_minimal_sufficient_subsets,
    exact_fourier_coefficients,
    fit_lasso_coordinate_descent,
    fit_lasso_fista,
    gradient_coefficient_estimates,
    majority_two_of_three,
    parity_feature_matrix,
    run_synthetic_reference_gate,
    sample_patch_masks,
    screened_supports,
    select_transition_density,
    two_clause_monotone_dnf,
)


def _lasso(degree_cap: int, *, regularization: float = 0.0) -> LassoConfig:
    return LassoConfig(
        degree_cap=degree_cap,
        interaction_screen_size=8,
        regularization=regularization,
        heavy_coefficient_threshold=1.0e-8,
        maximum_iterations=10_000,
        convergence_tolerance=1.0e-12,
        maximum_feature_count=100_000,
        fit_feature_count=100_000,
        power_iterations=100,
    )


def test_phantom_invariants_reject_illegal_density_logit_and_mask_states() -> None:
    assert isinstance(0.25, Density)
    assert isinstance(0.0, SweepDensity)
    assert isinstance(1.0, SweepDensity)
    assert isinstance(3.5, LogitDiff)
    assert not isinstance(0.0, Density)
    assert not isinstance(1.0, Density)
    assert not isinstance(float("nan"), LogitDiff)

    grid = SiteGrid((2, 3), (0, 1, 2))
    mask = t.zeros((2, 3), dtype=t.bool)
    assert as_patch_mask(mask, grid) is mask
    with pytest.raises(ValueError, match="shape"):
        as_patch_mask(t.zeros((3, 2), dtype=t.bool), grid)
    with pytest.raises(TypeCheckError):
        as_patch_mask(t.zeros((2, 3), dtype=t.float32), grid)  # type: ignore[arg-type]


def test_biased_basis_is_exactly_orthonormal_under_exhaustive_product_weights() -> None:
    masks = all_boolean_corners(3)
    supports = all_supports(3, 3)
    density = Density.parse(0.3)
    features = parity_feature_matrix(masks, supports, density)
    ones = masks.to(dtype=t.float64).sum(dim=1)
    weights = 0.3**ones * 0.7 ** (3.0 - ones)
    gram = features.T @ (features * weights.unsqueeze(1))
    assert t.allclose(gram, t.eye(len(supports), dtype=t.float64), atol=1.0e-12, rtol=0.0)


def test_biased_estimator_recovers_exact_and_coefficients_from_product_samples() -> None:
    corners = all_boolean_corners(2)
    masks = corners.repeat_interleave(t.tensor([9, 3, 3, 1]), dim=0)
    values = (masks[:, 0] & masks[:, 1]).to(dtype=t.float64)
    density = Density.parse(0.25)
    coefficients = exact_fourier_coefficients(
        masks,
        values,
        all_supports(2, 2),
        density,
    )
    sigma = math.sqrt(0.25 * 0.75)
    expected = t.tensor(
        [0.25**2, 0.25 * sigma, 0.25 * sigma, sigma**2],
        dtype=t.float64,
    )
    assert t.allclose(coefficients, expected, atol=1.0e-12, rtol=0.0)


def test_random_masks_include_exact_empty_and_clean_endpoint_corners() -> None:
    grid = SiteGrid((0, 1), (0, 1, 2))
    generator = t.Generator().manual_seed(9)
    empty = sample_patch_masks(4, grid, SweepDensity.parse(0.0), generator)
    clean = sample_patch_masks(4, grid, SweepDensity.parse(1.0), generator)
    biased = sample_patch_masks(1_000, grid, SweepDensity.parse(0.2), generator)
    assert not bool(empty.any())
    assert bool(clean.all())
    assert float(biased.to(dtype=t.float64).mean()) == pytest.approx(0.2, abs=0.03)


def test_majority_reference_recovers_known_uniform_fourier_coefficients() -> None:
    masks = all_boolean_corners(3)
    supports = all_supports(3, 3)
    values = majority_two_of_three(masks)
    density = Density.parse(0.5)
    expected = {
        (): 0.5,
        (0,): 0.25,
        (1,): 0.25,
        (2,): 0.25,
        (0, 1): 0.0,
        (0, 2): 0.0,
        (1, 2): 0.0,
        (0, 1, 2): -0.25,
    }

    actual = exact_fourier_coefficients(masks, values, supports, density)
    assert {support: float(actual[index]) for index, support in enumerate(supports)} == pytest.approx(
        expected,
        abs=1.0e-12,
    )

    design = parity_feature_matrix(masks, supports, density)
    fitted = fit_lasso_fista(design, values, _lasso(3))
    assert t.allclose(fitted, actual, atol=1.0e-10, rtol=0.0)
    coordinate_fitted = fit_lasso_coordinate_descent(design, values, _lasso(3))
    assert t.allclose(coordinate_fitted, actual, atol=1.0e-10, rtol=0.0)


def test_lasso_backtracks_when_power_start_misses_dominant_direction() -> None:
    features = t.tensor(
        [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]],
        dtype=t.float64,
    )
    values = t.ones(4, dtype=t.float64)
    config = LassoConfig(
        degree_cap=1,
        interaction_screen_size=1,
        regularization=0.0,
        heavy_coefficient_threshold=0.01,
        maximum_iterations=100,
        convergence_tolerance=1.0e-12,
        maximum_feature_count=2,
        fit_feature_count=2,
        power_iterations=1,
    )

    fitted = fit_lasso_fista(features, values, config)

    assert t.allclose(features @ fitted, values, atol=1.0e-12, rtol=0.0)


def test_multilinear_majority_derivatives_recover_nonconstant_coefficients() -> None:
    masks = all_boolean_corners(3)
    values = majority_two_of_three(masks)
    supports = all_supports(3, 3)
    density = Density.parse(0.5)
    x = masks.to(dtype=t.float64)
    gradients = t.stack(
        (
            x[:, 1] + x[:, 2] - 2.0 * x[:, 1] * x[:, 2],
            x[:, 0] + x[:, 2] - 2.0 * x[:, 0] * x[:, 2],
            x[:, 0] + x[:, 1] - 2.0 * x[:, 0] * x[:, 1],
        ),
        dim=1,
    )
    function_estimates = exact_fourier_coefficients(masks, values, supports, density)
    derivative_estimates = gradient_coefficient_estimates(
        gradients,
        masks,
        supports,
        density,
    )
    assert t.isnan(derivative_estimates[0])
    assert t.allclose(
        derivative_estimates[1:],
        function_estimates[1:],
        atol=1.0e-12,
        rtol=0.0,
    )


def test_production_synthetic_gate_passes_both_reference_functions() -> None:
    result = run_synthetic_reference_gate()
    assert result["status"] == "passed"
    majority = cast(dict[str, object], result["majority"])
    monotone_dnf = cast(dict[str, object], result["monotone_dnf"])
    assert majority["coefficient_count"] == 8
    assert monotone_dnf["coefficient_count"] == 16


def test_monotone_dnf_reference_recovers_every_known_coefficient() -> None:
    masks = all_boolean_corners(4)
    supports = all_supports(4, 4)
    values = two_clause_monotone_dnf(masks)
    actual = exact_fourier_coefficients(masks, values, supports, Density.parse(0.5))
    expected: dict[tuple[int, ...], float] = {(): 7.0 / 16.0}
    for support in supports[1:]:
        if support in {(0,), (1,), (2,), (3,), (0, 1), (2, 3)}:
            expected[support] = 3.0 / 16.0
        else:
            expected[support] = -1.0 / 16.0
    for index, support in enumerate(supports):
        assert float(actual[index]) == pytest.approx(expected[support], abs=1.0e-12)


def test_stage_two_all_greedy_paths_recover_exact_majority_minterms() -> None:
    sites = (Site(0, 0), Site(0, 1), Site(0, 2))
    candidates = as_non_empty_candidates(((sites[0],), (sites[1],), (sites[2],), sites))

    def is_sufficient(candidate: tuple[Site, ...]) -> bool:
        return len(candidate) >= 2

    results = enumerate_minimal_sufficient_subsets(candidates, is_sufficient)
    minsets = {minset for minset, _generator in results}
    assert minsets == {
        (sites[0], sites[1]),
        (sites[0], sites[2]),
        (sites[1], sites[2]),
    }
    assert all(generator == sites for _minset, generator in results)


def test_stage_two_recovers_both_exact_monotone_dnf_minterms() -> None:
    sites = tuple(Site(0, layer) for layer in range(4))
    candidates = as_non_empty_candidates((sites,))

    def is_sufficient(candidate: tuple[Site, ...]) -> bool:
        present = set(candidate)
        return bool(
            {sites[0], sites[1]}.issubset(present)
            or {sites[2], sites[3]}.issubset(present)
        )

    results = enumerate_minimal_sufficient_subsets(candidates, is_sufficient)
    assert {minset for minset, _generator in results} == {
        (sites[0], sites[1]),
        (sites[2], sites[3]),
    }


def test_stage_two_handles_singletons_and_excludes_the_empty_mask() -> None:
    site = Site(2, 7)
    candidates = as_non_empty_candidates(((site,),))
    results = enumerate_minimal_sufficient_subsets(candidates, lambda value: value == (site,))
    assert results == (((site,), (site,)),)


def test_screened_family_keeps_all_singletons_but_limits_interactions() -> None:
    config = _lasso(3)
    supports = screened_supports(6, (1, 3, 4), config)
    assert supports[:7] == ((), (0,), (1,), (2,), (3,), (4,), (5,))
    assert (1, 3) in supports and (1, 3, 4) in supports
    assert (0, 1) not in supports


def test_density_transition_uses_maximum_interior_output_variance() -> None:
    config = DensitySweepConfig(
        density_grid=tuple(SweepDensity.parse(value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)),
        masks_per_density=8,
        flat_probability_span=0.05,
        flat_logit_diff_span=0.1,
        minimum_logit_diff_variance=0.01,
        seed=4,
    )
    points = tuple(
        DensityPoint(
            density=SweepDensity.parse(density),
            sample_count=1 if density in {0.0, 1.0} else 8,
            mean_correct_probability=probability,
            correct_probability_variance=variance / 10.0,
            accuracy=float(probability >= 0.5),
            mean_logit_diff=LogitDiff.parse(logit),
            logit_diff_variance=variance,
        )
        for density, probability, logit, variance in (
            (0.0, 0.1, -2.0, 0.0),
            (0.25, 0.3, -0.5, 0.2),
            (0.5, 0.55, 0.2, 1.2),
            (0.75, 0.8, 1.5, 0.4),
            (1.0, 0.95, 3.0, 0.0),
        )
    )
    assert not density_curve_is_flat(points, config)
    assert float(select_transition_density(points, config)) == 0.5


def test_flat_density_curve_stops_before_spectrum_estimation() -> None:
    config = DensitySweepConfig(
        density_grid=tuple(SweepDensity.parse(value) for value in (0.0, 0.5, 1.0)),
        masks_per_density=4,
        flat_probability_span=0.05,
        flat_logit_diff_span=0.1,
        minimum_logit_diff_variance=0.01,
        seed=0,
    )
    points = tuple(
        DensityPoint(
            density=SweepDensity.parse(density),
            sample_count=1 if density in {0.0, 1.0} else 4,
            mean_correct_probability=0.5,
            correct_probability_variance=0.0,
            accuracy=1.0,
            mean_logit_diff=LogitDiff.parse(0.0),
            logit_diff_variance=0.0,
        )
        for density in (0.0, 0.5, 1.0)
    )
    assert density_curve_is_flat(points, config)
    with pytest.raises(RuntimeError, match="flat"):
        select_transition_density(points, config)


def test_biased_standardized_bits_rejects_endpoint_density() -> None:
    masks = all_boolean_corners(2)
    with pytest.raises(TypeCheckError):
        biased_standardized_bits(masks, cast(Any, 0.0))
    assert math.isfinite(float(biased_standardized_bits(masks, Density.parse(0.2)).sum()))
