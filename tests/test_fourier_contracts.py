from __future__ import annotations

import argparse
import ast
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from jaxtyping import TypeCheckError

import oocr_training_dynamics.fourier_circuits as fc
from oocr_training_dynamics.artifacts import sha256_file, write_json
from oocr_training_dynamics.runtime_fourier_circuits import fourier_output_dir
from scripts.run_fourier_circuits import (
    DEFAULT_DENSITY_GRID,
    PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    PYALVT_CHECKPOINT_LAUNCH_ORDER,
    PYALVT_CHECKPOINT_PLAN,
    PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    REGISTERED_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    REGISTERED_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    _config,
    _validate_checkpoint_series_plan,
)


def _gradient_validation() -> fc.GradientValidationConfig:
    return fc.GradientValidationConfig(2, 0.1, 0.2, 0.8, 1.0e-12)


def _stability() -> fc.DensityStabilityConfig:
    return fc.DensityStabilityConfig(8, 0.25, 0.9)


def _lasso(**changes: object) -> fc.LassoConfig:
    values: dict[str, object] = {
        "degree_cap": 3,
        "interaction_screen_size": 3,
        "regularization": 0.01,
        "heavy_coefficient_threshold": 0.05,
        "maximum_iterations": 500,
        "convergence_tolerance": 1.0e-10,
        "maximum_feature_count": 100,
        "fit_feature_count": 100,
        "power_iterations": 20,
    }
    values.update(changes)
    return fc.LassoConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: fc.Site(-1, 0), "non-negative"),
        (lambda: fc.SiteGrid((), (0,)), "non-empty"),
        (lambda: fc.SiteGrid((1, 1), (0,)), "token indices"),
        (lambda: fc.SiteGrid((0,), (1, 0)), "layers"),
        (lambda: fc.SiteGrid((-1,), (0,)), "non-negative"),
        (
            lambda: fc.ModelCheckpointSpec("qwen3-8b", "org/model", "a" * 40, "correct", 0, 1, 0),
            "olmo3",
        ),
        (
            lambda: fc.ModelCheckpointSpec("olmo3-7b", "model", "a" * 40, "correct", 0, 1, 0),
            "namespaced",
        ),
        (
            lambda: fc.ModelCheckpointSpec("olmo3-7b", "org/model", "Z" * 40, "correct", 0, 1, 0),
            "revision",
        ),
        (
            lambda: fc.ModelCheckpointSpec(
                "olmo3-7b", "org/model", "a" * 40, "wrong_alias", 0, 1, 0
            ),
            "correct condition",
        ),
        (
            lambda: fc.ModelCheckpointSpec("olmo3-7b", "org/model", "a" * 40, "correct", -1, 1, 0),
            "non-negative",
        ),
        (
            lambda: fc.ModelCheckpointSpec("olmo3-7b", "org/model", "a" * 40, "correct", 0, 1, 1),
            "differ",
        ),
        (lambda: fc.TaskDatasetSpec("", 0, 1, "code"), "non-empty"),
        (lambda: fc.TaskDatasetSpec("identity", -1, 1, "code"), "exactly one"),
        (lambda: fc.TaskDatasetSpec("identity", 0, 2, "code"), "exactly one"),
        (lambda: fc.TaskDatasetSpec("identity", 0, 1, "language"), "code reflection"),
        (lambda: fc.FullPromptSites(-1, 2), "half-open"),
        (lambda: fc.FullPromptSites(2, 2), "half-open"),
        (lambda: fc.ReverseWindowSites(-1, 2, 0, 1), "reverse-token"),
        (lambda: fc.ReverseWindowSites(0, 0, 0, 1), "reverse-token"),
        (lambda: fc.ReverseWindowSites(0, 1, 2, 2), "layer interval"),
    ],
)
def test_structural_contracts_fail_loud(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_site_grid_round_trip_and_bounds() -> None:
    grid = fc.SiteGrid((3, 5), (7, 9, 11))
    assert grid.shape == (2, 3)
    assert grid.site_count == 6
    assert grid.site(4) == fc.Site(5, 9)
    assert grid.flat_index(fc.Site(5, 9)) == 4
    with pytest.raises(IndexError, match="outside"):
        grid.site(6)
    with pytest.raises(ValueError, match="outside"):
        grid.flat_index(fc.Site(4, 9))


def test_active_site_space_requires_a_partition_and_maps_supports() -> None:
    space = fc.ActiveSiteSpace(5, (0, 2, 4), (1, 3), (fc.Site(0, 1), fc.Site(1, 1)))
    assert space.active_site_count == 3
    assert space.full_support((0, 2)) == (0, 4)
    with pytest.raises(ValueError, match="partition"):
        fc.ActiveSiteSpace(5, (0, 2), (1, 3), (fc.Site(0, 1), fc.Site(1, 1)))
    with pytest.raises(ValueError, match="active support"):
        space.full_support((3,))


def test_registered_clean_minus_ten_point_veto_is_the_exact_28_site_census() -> None:
    assert len(PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS) == 28
    assert (
        tuple(
            [fc.Site(38, layer) for layer in range(2, 8)]
            + [fc.Site(53, layer) for layer in range(3, 10)]
            + [fc.Site(112, layer) for layer in range(17, 32)]
        )
        == PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS
    )


def test_registered_riodwl_clean_minus_ten_point_veto_is_independently_censused() -> None:
    expected = tuple(
        [fc.Site(101, layer) for layer in range(13, 18)]
        + [fc.Site(112, layer) for layer in range(16, 32)]
    )
    assert len(RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS) == 21
    assert expected == RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS
    assert REGISTERED_CLEAN_MINUS_TEN_POINT_SINGLETONS == {
        "add_5": PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
        "identity": RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    }


def test_registered_pyalvt_checkpoint_vetoes_are_frozen_before_collection() -> None:
    assert tuple(PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS) == (
        32,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1_024,
        1_280,
        1_500,
    )
    assert PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[64] == tuple(
        fc.Site(112, layer) for layer in range(17, 32)
    )
    assert PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[96] == tuple(
        fc.Site(112, layer) for layer in range(17, 32)
    )
    assert PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[32] == tuple(
        fc.Site(112, layer) for layer in range(16, 32)
    )
    assert (
        REGISTERED_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[("identity", 1_500)]
        == RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS
    )


@pytest.mark.parametrize("clean_step", [64, 96])
def test_pyalvt_checkpoint_probability_runs_use_checkpoint_veto(
    tmp_path: Path,
    clean_step: int,
) -> None:
    config = _config(
        tmp_path,
        argparse.Namespace(
            function_id="add_5",
            clean_step=clean_step,
            dirty_step=0,
            layer_window="0:32",
            reverse_token_window=None,
            density_grid=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
            sufficiency_rule="clean-probability-minus-0.10",
        ),
    )

    assert isinstance(config.sufficiency, fc.ProbabilitySufficiencyConfig)
    assert (
        config.sufficiency.expected_passing_singletons
        == (PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[clean_step])
    )
    assert fourier_output_dir(tmp_path, config).name.endswith(
        "_sufficiency_clean_probability_minus_0p10_veto_15"
    )


def test_circuit_config_can_preserve_a_distinct_absolute_artifact_identity(
    tmp_path: Path,
) -> None:
    identity_root = Path("/home/research/local-oocr")
    args = argparse.Namespace(
        function_id="add_5",
        clean_step=64,
        dirty_step=0,
        layer_window="0:32",
        reverse_token_window=None,
        density_grid=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
        sufficiency_rule="clean-probability-minus-0.10",
        artifact_identity_root=identity_root,
    )

    assert _config(tmp_path, args).artifact_root == identity_root

    args.artifact_identity_root = Path("relative")
    with pytest.raises(ValueError, match="artifact identity root must be an absolute path"):
        _config(tmp_path, args)


def test_checkpoint_series_plan_validates_the_frozen_reference_digest(
    tmp_path: Path,
) -> None:
    reference_path = (
        tmp_path / "artifacts/runs/olmo3-7b/correct/seed_20260715/patching/sequence_end/"
        "later_checkpoint/recipient_step_000000/donor_step_000096.json"
    )
    write_json(reference_path, {"independent": "checkpoint-transfer reference"})
    write_json(
        tmp_path / PYALVT_CHECKPOINT_PLAN,
        {
            "status": "registered_before_checkpoint_specific_collection",
            "launch_order": PYALVT_CHECKPOINT_LAUNCH_ORDER,
            "dirty_step": 0,
            "shuffle_seed": 20_260_820,
            "checkpoints": [
                {
                    "clean_step": 96,
                    "reference_sha256": sha256_file(reference_path),
                }
            ],
        },
    )
    args = argparse.Namespace(
        function_id="add_5",
        clean_step=96,
        dirty_step=0,
        sufficiency_rule="clean-probability-minus-0.10",
    )

    _validate_checkpoint_series_plan(tmp_path, args)
    write_json(reference_path, {"independent": "mutated"})
    with pytest.raises(RuntimeError, match="changed after registration"):
        _validate_checkpoint_series_plan(tmp_path, args)


def test_riodwl_probability_run_has_a_distinct_veto_21_artifact_identity(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        argparse.Namespace(
            function_id="identity",
            clean_step=1_500,
            dirty_step=0,
            layer_window="0:32",
            reverse_token_window=None,
            density_grid=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
            sufficiency_rule="clean-probability-minus-0.10",
        ),
    )

    assert isinstance(config.sufficiency, fc.ProbabilitySufficiencyConfig)
    assert config.sufficiency.expected_passing_singletons == RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS
    assert fourier_output_dir(tmp_path, config).name.endswith(
        "_sufficiency_clean_probability_minus_0p10_veto_21"
    )


@pytest.mark.parametrize(
    "factory,message",
    [
        (lambda: fc.ExhaustiveSingletonConfig((), 0.1), "non-empty"),
        (lambda: fc.ExhaustiveSingletonConfig((1, 1), 0.1), "increasing"),
        (lambda: fc.ExhaustiveSingletonConfig((1,), -0.1), "finite"),
    ],
)
def test_exhaustive_singleton_config_fails_loud(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"density_grid": (fc.SweepDensity.parse(0.0), fc.SweepDensity.parse(1.0))}, "interior"),
        (
            {"density_grid": tuple(fc.SweepDensity.parse(value) for value in (0.0, 0.5, 0.5, 1.0))},
            "increasing",
        ),
        ({"masks_per_density": 1}, "at least two"),
        ({"flat_probability_span": -1.0}, "finite"),
        ({"flat_logit_diff_span": float("nan")}, "finite"),
        ({"seed": -1}, "seed"),
    ],
)
def test_density_sweep_config_rejects_invalid_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "density_grid": tuple(fc.SweepDensity.parse(value) for value in (0.0, 0.5, 1.0)),
        "masks_per_density": 4,
        "flat_probability_span": 0.05,
        "flat_logit_diff_span": 0.2,
        "minimum_logit_diff_variance": 0.01,
        "seed": 1,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        fc.DensitySweepConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"degree_cap": 0}, "degree cap"),
        ({"interaction_screen_size": 0}, "screen size"),
        ({"regularization": -1.0}, "regularization"),
        ({"heavy_coefficient_threshold": 0.0}, "heavy coefficient"),
        ({"maximum_iterations": 0}, "feature caps"),
        ({"maximum_feature_count": 0}, "feature caps"),
        ({"fit_feature_count": 1}, "fit-feature"),
        ({"power_iterations": 0}, "power-iteration"),
        ({"convergence_tolerance": float("inf")}, "convergence"),
    ],
)
def test_lasso_config_rejects_invalid_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _lasso(**changes)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: fc.GradientValidationConfig(0, 1.0, 1.0, 0.0, 1.0), "held-out"),
        (lambda: fc.GradientValidationConfig(1, -1.0, 1.0, 0.0, 1.0), "non-negative"),
        (lambda: fc.GradientValidationConfig(1, 1.0, 1.0, 2.0, 1.0), "cosine"),
        (lambda: fc.GradientValidationConfig(1, 1.0, 1.0, 0.0, 0.0), "variance floor"),
        (lambda: fc.DensityStabilityConfig(1, 1.0, 0.0), "repeated"),
        (lambda: fc.DensityStabilityConfig(2, 3.0, 0.0), "L1"),
        (lambda: fc.DensityStabilityConfig(2, 1.0, 2.0), "cosine"),
        (
            lambda: fc.SpectrumConfig(2, 1, 0.2, 0, _lasso(), _gradient_validation(), _stability()),
            "three corners",
        ),
        (
            lambda: fc.SpectrumConfig(3, 0, 0.2, 0, _lasso(), _gradient_validation(), _stability()),
            "batch size",
        ),
        (
            lambda: fc.SpectrumConfig(3, 1, 0.5, 0, _lasso(), _gradient_validation(), _stability()),
            "validation fraction",
        ),
        (
            lambda: fc.SpectrumConfig(
                3, 1, 0.2, -1, _lasso(), _gradient_validation(), _stability()
            ),
            "seed",
        ),
        (lambda: fc.SufficiencyConfig(0.0, True, 1, 2, 2), "recovery"),
        (lambda: fc.SufficiencyConfig(0.8, True, 0, 2, 2), "batch size"),
        (lambda: fc.SufficiencyConfig(0.8, True, 1, 0, 2), "caps"),
        (
            lambda: fc.CacheConfig(0, 1, 1, 1, 1, 0.0, 0.0, "full_sequence_reference"),
            "positive",
        ),
        (
            lambda: fc.CacheConfig(1, 1, 1, 1, 1, -1.0, 0.0, "full_sequence_reference"),
            "non-negative",
        ),
        (lambda: fc.CacheConfig(1, 1, 1, 1, 1, 0.0, 0.0, "cached"), "reference"),
        (lambda: fc.HarnessCheckConfig(-1.0, 1.0), "probability"),
        (lambda: fc.HarnessCheckConfig(0.0, 0.0), "effect floor"),
    ],
)
def test_numeric_configs_fail_loud(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: fc.DensityPoint(
                fc.SweepDensity.parse(0.5), 0, 0.5, 0.0, 1.0, fc.LogitDiff.parse(0.0), 0.0
            ),
            "one sample",
        ),
        (
            lambda: fc.DensityPoint(
                fc.SweepDensity.parse(0.5), 1, 1.1, 0.0, 1.0, fc.LogitDiff.parse(0.0), 0.0
            ),
            "lie in",
        ),
        (
            lambda: fc.DensityPoint(
                fc.SweepDensity.parse(0.5), 1, 0.5, -1.0, 1.0, fc.LogitDiff.parse(0.0), 0.0
            ),
            "variances",
        ),
        (lambda: fc.FourierCoefficient((1, 1), 2, 1.0, 1.0, None, None, True), "support"),
        (lambda: fc.FourierCoefficient((1,), 2, 1.0, 1.0, None, None, True), "degree"),
        (lambda: fc.FourierCoefficient((1,), 1, float("nan"), 1.0, None, None, True), "finite"),
        (lambda: fc.FourierCoefficient((1,), 1, 1.0, 1.0, float("nan"), None, True), "finite"),
        (lambda: fc.GradientValidationResult((), 0.0, 0.0, 1.0, True), "held-out"),
        (lambda: fc.GradientValidationResult((1,), -1.0, 0.0, 1.0, True), "non-negative"),
        (lambda: fc.GradientValidationResult((1,), 0.0, 0.0, 2.0, True), "cosine"),
        (lambda: fc.DegreeProfile(fc.Density.parse(0.5), ()), "constant degree"),
        (lambda: fc.DegreeProfile(fc.Density.parse(0.5), (1.0, -1.0)), "non-negative"),
        (
            lambda: fc.VerifiedMinset((), fc.LogitDiff.parse(1), 0.5, 0.1, ((fc.Site(0, 0),),)),
            "non-empty",
        ),
        (
            lambda: fc.VerifiedMinset(
                (fc.Site(1, 0), fc.Site(0, 0)), fc.LogitDiff.parse(1), 0.5, 0.1, ((fc.Site(0, 0),),)
            ),
            "sorted",
        ),
        (
            lambda: fc.VerifiedMinset(
                (fc.Site(0, 0),), fc.LogitDiff.parse(1), 1.5, 0.1, ((fc.Site(0, 0),),)
            ),
            "probability",
        ),
        (
            lambda: fc.VerifiedMinset(
                (fc.Site(0, 0),), fc.LogitDiff.parse(1), 0.5, -0.1, ((fc.Site(0, 0),),)
            ),
            "margin",
        ),
        (
            lambda: fc.VerifiedMinset((fc.Site(0, 0),), fc.LogitDiff.parse(1), 0.5, 0.1, ()),
            "hypothesis",
        ),
    ],
)
def test_result_contracts_fail_loud(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_mask_and_support_operations_fail_loud() -> None:
    grid = fc.SiteGrid((0, 1), (0, 1))
    noncontiguous = t.zeros((2, 2), dtype=t.bool).T
    with pytest.raises(ValueError, match="contiguous"):
        fc.as_patch_mask(noncontiguous, grid)
    with pytest.raises(ValueError, match="axes"):
        fc.flatten_masks(t.zeros((1, 1, 4), dtype=t.bool), grid)
    with pytest.raises(ValueError, match="positive"):
        fc.sample_patch_masks(0, grid, fc.SweepDensity.parse(0.5), t.Generator())
    with pytest.raises(ValueError, match="increasing"):
        fc.validate_support((1, 1), 3, 2)
    with pytest.raises(ValueError, match="degree cap"):
        fc.validate_support((0, 1, 2), 3, 2)
    with pytest.raises(ValueError, match="outside"):
        fc.validate_support((3,), 3, 2)
    masks = fc.all_boolean_corners(2)
    with pytest.raises(ValueError, match="constant support"):
        fc.parity_feature_matrix(masks, (), fc.Density.parse(0.5))
    with pytest.raises(ValueError, match="unique"):
        fc.parity_feature_matrix(masks, ((), (0,), (0,)), fc.Density.parse(0.5))
    with pytest.raises(TypeCheckError):
        fc.exact_fourier_coefficients(
            masks,
            t.zeros(3),
            ((),),
            fc.Density.parse(0.5),
        )
    with pytest.raises(ValueError, match="valid degree cap"):
        fc.all_supports(2, 3)


def test_screening_and_feature_cap_contracts() -> None:
    config = _lasso(maximum_feature_count=5, fit_feature_count=5)
    with pytest.raises(ValueError, match="positive site"):
        fc.screened_supports(0, (), config)
    with pytest.raises(ValueError, match="increasing"):
        fc.screened_supports(4, (2, 2), config)
    with pytest.raises(ValueError, match="outside"):
        fc.screened_supports(4, (4,), config)
    with pytest.raises(RuntimeError, match="explicit cap"):
        fc.screened_supports(4, (0, 1, 2), config)

    masks = fc.all_boolean_corners(3)
    values = fc.majority_two_of_three(masks)
    gradients = t.zeros_like(masks, dtype=t.float64)
    selected_values = fc.screen_sites_from_function_values(
        values,
        masks,
        fc.Density.parse(0.5),
        2,
    )
    selected_gradients = fc.screen_sites_from_gradients(
        gradients,
        values,
        masks,
        fc.Density.parse(0.5),
        2,
    )
    assert len(selected_values) == len(selected_gradients) == 2
    supports = fc.all_supports(3, 3)
    features = fc.parity_feature_matrix(masks, supports, fc.Density.parse(0.5))
    retained = fc.function_correlation_feature_indices(features, values, 4)
    assert retained[0] == 0 and len(retained) == 4
    assert fc.function_correlation_feature_indices(features, values, 100) == tuple(
        range(len(supports))
    )
    with pytest.raises(ValueError, match="positive cap"):
        fc.function_correlation_feature_indices(features, values, 1)
    with pytest.raises(TypeCheckError):
        fc.screen_sites_from_function_values(values[:2], masks, fc.Density.parse(0.5), 1)
    with pytest.raises(ValueError, match="screen count"):
        fc.screen_sites_from_function_values(values, masks, fc.Density.parse(0.5), 0)
    with pytest.raises(TypeCheckError):
        fc.screen_sites_from_gradients(gradients[:, :2], values, masks, fc.Density.parse(0.5), 1)
    with pytest.raises(TypeCheckError):
        fc.screen_sites_from_gradients(gradients[:2], values, masks[:2], fc.Density.parse(0.5), 1)
    with pytest.raises(ValueError, match="screen count"):
        fc.screen_sites_from_gradients(gradients, values, masks, fc.Density.parse(0.5), 4)


def test_lasso_gradient_validation_and_augmentation_paths() -> None:
    with pytest.raises(ValueError, match="non-empty matrix"):
        fc._power_lipschitz(t.empty((0, 1), dtype=t.float64), 1)
    assert fc._power_lipschitz(t.zeros((2, 2), dtype=t.float64), 2) == 1.0
    with pytest.raises(TypeCheckError):
        fc.fit_lasso_fista(t.ones((2, 1), dtype=t.float64), t.ones(3), _lasso())
    with pytest.raises(RuntimeError, match="failed to converge"):
        fc.fit_lasso_fista(
            t.eye(2, dtype=t.float64),
            t.tensor([1.0, -1.0]),
            _lasso(maximum_iterations=1, convergence_tolerance=1.0e-30),
        )
    with pytest.raises(TypeCheckError):
        fc.fit_lasso_coordinate_descent(
            t.ones((2, 1), dtype=t.float64),
            t.ones(3),
            _lasso(),
        )
    with pytest.raises(RuntimeError, match="failed to converge"):
        fc.fit_lasso_coordinate_descent(
            t.eye(2, dtype=t.float64),
            t.tensor([1.0, -1.0]),
            _lasso(maximum_iterations=1, convergence_tolerance=1.0e-30),
        )

    masks = fc.all_boolean_corners(2)
    gradients = t.ones((4, 2), dtype=t.float64)
    supports = fc.all_supports(2, 2)
    with pytest.raises(TypeCheckError):
        fc.gradient_coefficient_estimates(gradients[:, :1], masks, supports, fc.Density.parse(0.5))
    with pytest.raises(ValueError, match="non-empty and unique"):
        fc.gradient_coefficient_samples(gradients, masks, (), fc.Density.parse(0.5))

    function = t.tensor([float("nan"), 0.2, -0.3], dtype=t.float64)
    exact_gradient = function.clone()
    config = _gradient_validation()
    accepted = fc.validate_gradient_estimates(function, exact_gradient, (1, 2), config)
    assert accepted.accepted and accepted.cosine_similarity == pytest.approx(1.0)
    rejected = fc.validate_gradient_estimates(
        t.tensor([0.0, 1.0, 0.0], dtype=t.float64),
        t.tensor([0.0, -1.0, 0.0], dtype=t.float64),
        (1, 2),
        config,
    )
    assert not rejected.accepted and rejected.cosine_similarity == pytest.approx(-1.0)
    zero_equal = fc.validate_gradient_estimates(
        t.zeros(3, dtype=t.float64),
        t.zeros(3, dtype=t.float64),
        (1, 2),
        config,
    )
    assert zero_equal.cosine_similarity == 1.0
    zero_unequal = fc.validate_gradient_estimates(
        t.zeros(3, dtype=t.float64),
        t.tensor([0.0, 1.0, 0.0], dtype=t.float64),
        (1, 2),
        config,
    )
    assert zero_unequal.cosine_similarity == 0.0
    with pytest.raises(TypeCheckError):
        fc.validate_gradient_estimates(function, exact_gradient[:2], (1,), config)
    with pytest.raises(ValueError, match="non-empty, sorted"):
        fc.validate_gradient_estimates(function, exact_gradient, (2, 1), config)
    with pytest.raises(ValueError, match="nonconstant"):
        fc.validate_gradient_estimates(function, exact_gradient, (0,), config)
    with pytest.raises(ValueError, match="finite"):
        fc.validate_gradient_estimates(
            t.tensor([0.0, float("nan"), 1.0], dtype=t.float64),
            t.tensor([0.0, 0.0, 1.0], dtype=t.float64),
            (1,),
            config,
        )

    function_samples = t.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=t.float64)
    gradient_samples = t.tensor([[2.0, 2.0], [2.0, 6.0]], dtype=t.float64)
    augmented = fc.inverse_variance_augment(function_samples, gradient_samples, config)
    assert augmented.shape == (2,) and bool(t.isfinite(augmented).all())
    with pytest.raises(ValueError, match="paired repeated"):
        fc.inverse_variance_augment(function_samples[:1], gradient_samples[:1], config)


def test_density_profiles_thresholds_and_candidate_edges() -> None:
    sweep = fc.DensitySweepConfig(
        tuple(fc.SweepDensity.parse(value) for value in (0.0, 0.5, 1.0)),
        4,
        0.05,
        0.2,
        0.01,
        0,
    )
    flat_points = tuple(
        fc.DensityPoint(
            fc.SweepDensity.parse(density),
            1 if density in {0.0, 1.0} else 4,
            0.5,
            0.0,
            0.0,
            fc.LogitDiff.parse(0.0),
            0.0,
        )
        for density in (0.0, 0.5, 1.0)
    )
    assert fc.density_curve_is_flat(flat_points, sweep)
    with pytest.raises(RuntimeError, match="flat"):
        fc.select_transition_density(flat_points, sweep)
    with pytest.raises(ValueError, match="exactly one"):
        fc.density_curve_is_flat(flat_points[:2], sweep)

    coefficients = (
        fc.FourierCoefficient((), 0, 1.0, 1.0, None, None, False),
        fc.FourierCoefficient((0,), 1, 0.5, 0.5, 0.5, 0.5, True),
        fc.FourierCoefficient((0, 1), 2, 0.25, 0.25, 0.25, 0.25, True),
    )
    profile = fc.normalized_degree_profile(coefficients, 2, fc.Density.parse(0.5))
    assert sum(profile.squared_weight_by_degree) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="requires coefficients"):
        fc.normalized_degree_profile((), 2, fc.Density.parse(0.5))
    with pytest.raises(RuntimeError, match="zero total"):
        fc.normalized_degree_profile(
            (fc.FourierCoefficient((), 0, 0.0, 0.0, None, None, False),),
            0,
            fc.Density.parse(0.5),
        )
    second = fc.DegreeProfile(fc.Density.parse(0.4), profile.squared_weight_by_degree)
    stable, maximum_l1, cosine = fc.compare_degree_profiles((profile, second), _stability())
    assert stable and maximum_l1 == pytest.approx(0.0) and cosine == pytest.approx(1.0)
    far = fc.DegreeProfile(fc.Density.parse(0.6), (0.0, 0.0, 1.0))
    stable, _, _ = fc.compare_degree_profiles((profile, far), _stability())
    assert not stable
    with pytest.raises(ValueError, match="at least two"):
        fc.compare_degree_profiles((profile,), _stability())
    with pytest.raises(ValueError, match="degree axis"):
        fc.compare_degree_profiles(
            (profile, fc.DegreeProfile(fc.Density.parse(0.4), (1.0,))),
            _stability(),
        )

    with pytest.raises(ValueError, match="at least one"):
        fc.as_non_empty_candidates(())
    with pytest.raises(ValueError, match="sorted"):
        fc.as_non_empty_candidates(((fc.Site(1, 0), fc.Site(0, 0)),))
    sufficiency = fc.SufficiencyConfig(0.8, True, 1, 8, 64)
    threshold = fc.sufficiency_threshold(fc.LogitDiff.parse(-1), fc.LogitDiff.parse(4), sufficiency)
    assert float(threshold) == pytest.approx(3.0)
    with pytest.raises(RuntimeError, match="must improve"):
        fc.sufficiency_threshold(fc.LogitDiff.parse(1), fc.LogitDiff.parse(1), sufficiency)
    probability_sufficiency = fc.ProbabilitySufficiencyConfig(
        0.10,
        (fc.Site(0, 0),),
        True,
        1,
        8,
        64,
    )
    probability_threshold = fc.resolved_sufficiency_threshold(
        fc.LogitDiff.parse(-5.702159881591797),
        fc.LogitDiff.parse(8.62784194946289),
        0.0033276367466896772,
        0.9998210072517395,
        probability_sufficiency,
    )
    expected_probability = 0.9998210072517395 - 0.10
    assert float(probability_threshold) == pytest.approx(
        math.log(expected_probability / (1.0 - expected_probability))
    )
    site = fc.Site(0, 0)
    candidates = fc.as_non_empty_candidates(((site,),))
    assert fc.enumerate_minimal_sufficient_subsets(candidates, lambda _sites: False) == ()


@pytest.mark.parametrize("site_count", [0, 21])
def test_synthetic_helpers_reject_unsupported_shapes(site_count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        fc.all_boolean_corners(site_count)
    with pytest.raises(ValueError, match="exactly three"):
        fc.majority_two_of_three(t.zeros((1, 2), dtype=t.bool))
    with pytest.raises(ValueError, match="exactly four"):
        fc.two_clause_monotone_dnf(t.zeros((1, 3), dtype=t.bool))


def test_top_level_config_requires_a_concrete_artifact_root() -> None:
    model = fc.ModelCheckpointSpec("olmo3-7b", "org/model", "a" * 40, "correct", 0, 1500, 0)
    sweep = fc.DensitySweepConfig(
        tuple(fc.SweepDensity.parse(value) for value in (0.0, 0.5, 1.0)),
        4,
        0.05,
        0.2,
        0.01,
        0,
    )
    spectrum = fc.SpectrumConfig(8, 1, 0.25, 0, _lasso(), _gradient_validation(), _stability())
    config = fc.FourierCircuitConfig(
        model,
        fc.TaskDatasetSpec("identity", 0, 1, "code"),
        fc.FullPromptSites(0, 1),
        sweep,
        spectrum,
        fc.SufficiencyConfig(0.8, True, 1, 8, 64),
        fc.ExhaustiveSingletonConfig((0,), 0.005),
        fc.CacheConfig(1, 2, 2, 1, 1, 0.1, 0.1, "full_sequence_reference"),
        fc.HarnessCheckConfig(0.01, 0.01),
        Path("artifacts"),
    )
    assert config.artifact_root == Path("artifacts")
    with pytest.raises(ValueError, match="concrete"):
        replace(config, artifact_root=Path("."))
    with pytest.raises(ValueError, match="batches"):
        replace(config, sufficiency=fc.SufficiencyConfig(0.8, True, 2, 8, 64))


def test_implementation_uses_only_torch_as_t_and_decorates_tensor_functions() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "oocr_training_dynamics/fourier_circuits.py",
        root / "oocr_training_dynamics/runtime_fourier_circuits.py",
        root / "scripts/run_fourier_circuits.py",
    )
    tensor_aliases = {
        "Mask2D",
        "MaskBatch",
        "FlatMaskBatch",
        "ValueVector",
        "GradientBatch",
        "FeatureMatrix",
        "CoefficientVector",
        "StandardizedBits",
        "CoefficientSamples",
        "CandidateIds",
        "CandidateLogits",
        "ResidualBank",
        "AlphaBatch",
        "MetricVector",
        "HiddenBatch",
        "TokenVectors",
        "TokenAlphas",
        "SingleTokenHidden",
        "HiddenVector",
        "IndexVector",
        "BitVector",
    }
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "numpy" for alias in node.names)
                for alias in node.names:
                    if alias.name == "torch":
                        assert alias.asname == "t"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "numpy" and not str(node.module).startswith("numpy.")
        if path.name == "run_fourier_circuits.py":
            continue
        for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
            annotations = [
                argument.annotation
                for argument in (*function.args.args, *function.args.kwonlyargs)
                if argument.annotation is not None
            ]
            if function.returns is not None:
                annotations.append(function.returns)
            annotation_text = " ".join(ast.unparse(annotation) for annotation in annotations)
            has_tensor = "t.Tensor" in annotation_text or any(
                alias in annotation_text for alias in tensor_aliases
            )
            if has_tensor:
                decorators = " ".join(ast.unparse(item) for item in function.decorator_list)
                assert "jaxtyped" in decorators and "beartype" in decorators, (
                    path.name,
                    function.name,
                )
