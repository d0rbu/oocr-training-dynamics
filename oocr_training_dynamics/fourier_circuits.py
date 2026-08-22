"""Typed, CPU-testable Fourier discovery and causal-minset contracts.

The model-facing runtime lives in :mod:`runtime_fourier_circuits`.  This module is
deliberately free of model imports so the biased-basis estimator and stage-2 logic can
be proved against exhaustive synthetic truth before a checkpoint is loaded.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Float64, jaxtyped
from phantom import Phantom
from phantom.sized import NonEmpty

MAX_ABS_LOGIT_DIFF = 1_000_000.0

Mask2D = Bool[t.Tensor, "token layer"]
MaskBatch = Bool[t.Tensor, "sample token layer"]
FlatMaskBatch = Bool[t.Tensor, "sample site"]
ValueVector = Float[t.Tensor, "sample"]
GradientBatch = Float[t.Tensor, "sample site"]
FeatureMatrix = Float64[t.Tensor, "sample feature"]
CoefficientVector = Float64[t.Tensor, "feature"]
StandardizedBits = Float64[t.Tensor, "sample site"]
CoefficientSamples = Float64[t.Tensor, "sample feature"]

SUPPORT_CHUNK_SIZE = 2_048


@jaxtyped(typechecker=beartype)
def _is_patch_mask(value: Mask2D) -> bool:
    return bool(
        value.dtype is t.bool and value.ndim == 2 and value.shape[0] > 0 and value.shape[1] > 0
    )


class PatchMask(Phantom[t.Tensor], predicate=_is_patch_mask, bound=t.Tensor):
    """A non-empty rank-2 boolean tensor with axes ``(token, layer)``."""


class Density(
    float,
    Phantom[float],
    predicate=lambda value: math.isfinite(value) and 0.0 < value < 1.0,
    bound=float,
):
    """An interior Bernoulli density used by the biased Fourier basis."""


class SweepDensity(
    float,
    Phantom[float],
    predicate=lambda value: math.isfinite(value) and 0.0 <= value <= 1.0,
    bound=float,
):
    """A stage-0 density, including the deterministic endpoint corners."""


class LogitDiff(
    float,
    Phantom[float],
    predicate=lambda value: math.isfinite(value) and abs(value) <= MAX_ABS_LOGIT_DIFF,
    bound=float,
):
    """A finite raw correct-vs-rest log-odds value.

    The wide bound is a fail-loud pathology guard, never a clamp.  Candidate logits
    from a BF16 7B model should be many orders of magnitude inside this interval.
    """


@beartype
@dataclass(frozen=True, order=True)
class Site:
    """One forward-indexed prompt-token and decoder-layer residual site."""

    token_index: int
    layer: int

    def __post_init__(self) -> None:
        if self.token_index < 0 or self.layer < 0:
            raise ValueError("site coordinates must be non-negative")


SiteSet = tuple[Site, ...]
CandidateSiteSets = NonEmpty[SiteSet]


@beartype
@dataclass(frozen=True)
class SiteGrid:
    """The exact ordered tensor axes represented by every patch mask."""

    token_indices: tuple[int, ...]
    layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.token_indices or not self.layers:
            raise ValueError("site grid axes must both be non-empty")
        if tuple(sorted(set(self.token_indices))) != self.token_indices:
            raise ValueError("token indices must be strictly increasing and unique")
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("layers must be strictly increasing and unique")
        if self.token_indices[0] < 0 or self.layers[0] < 0:
            raise ValueError("site grid coordinates must be non-negative")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.token_indices), len(self.layers))

    @property
    def site_count(self) -> int:
        return len(self.token_indices) * len(self.layers)

    def site(self, flat_index: int) -> Site:
        if not 0 <= flat_index < self.site_count:
            raise IndexError("flat site index is outside the site grid")
        token_offset, layer_offset = divmod(flat_index, len(self.layers))
        return Site(self.token_indices[token_offset], self.layers[layer_offset])

    def flat_index(self, site: Site) -> int:
        if site.token_index not in self.token_indices or site.layer not in self.layers:
            raise ValueError(f"site is outside the configured grid: {site}")
        token_offset = self.token_indices.index(site.token_index)
        layer_offset = self.layers.index(site.layer)
        return token_offset * len(self.layers) + layer_offset


@beartype
@dataclass(frozen=True)
class ModelCheckpointSpec:
    model_key: str
    model_id: str
    revision: str
    condition: str
    seed: int
    clean_step: int
    dirty_step: int

    def __post_init__(self) -> None:
        if self.model_key != "olmo3-7b":
            raise ValueError("Fourier circuit discovery currently supports only olmo3-7b")
        if not self.model_id or "/" not in self.model_id:
            raise ValueError("model ID must be a namespaced checkpoint identifier")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("model revision must be a full lowercase 40-character commit")
        if self.condition != "correct":
            raise ValueError("the first Fourier analysis is scoped to the correct condition")
        if self.seed < 0 or self.clean_step < 0 or self.dirty_step < 0:
            raise ValueError("seed and checkpoint steps must be non-negative")
        if self.clean_step == self.dirty_step:
            raise ValueError("clean and dirty checkpoints must differ")


@beartype
@dataclass(frozen=True)
class TaskDatasetSpec:
    function_id: str
    corpus_seed: int
    variants_per_kind: int
    record_kind: str

    def __post_init__(self) -> None:
        if not self.function_id:
            raise ValueError("function ID must be non-empty")
        if self.corpus_seed < 0 or self.variants_per_kind != 1:
            raise ValueError(
                "checkpoint-transfer parity requires exactly one registered variant per kind"
            )
        if self.record_kind != "code":
            raise ValueError("Fourier discovery requires the five-way code reflection task")


@beartype
@dataclass(frozen=True)
class FullPromptSites:
    layer_start: int
    layer_stop: int

    def __post_init__(self) -> None:
        if self.layer_start < 0 or self.layer_stop <= self.layer_start:
            raise ValueError("full-prompt layer interval must be non-empty and half-open")


@beartype
@dataclass(frozen=True)
class ReverseWindowSites:
    reverse_token_start: int
    reverse_token_stop: int
    layer_start: int
    layer_stop: int

    def __post_init__(self) -> None:
        if self.reverse_token_start < 0 or self.reverse_token_stop <= self.reverse_token_start:
            raise ValueError("reverse-token interval must be non-empty and half-open")
        if self.layer_start < 0 or self.layer_stop <= self.layer_start:
            raise ValueError("layer interval must be non-empty and half-open")


SiteScope = FullPromptSites | ReverseWindowSites


@beartype
@dataclass(frozen=True)
class DensitySweepConfig:
    density_grid: tuple[SweepDensity, ...]
    masks_per_density: int
    flat_probability_span: float
    flat_logit_diff_span: float
    minimum_logit_diff_variance: float
    seed: int

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.density_grid)
        if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("density grid must contain at least one interior point plus 0 and 1")
        if tuple(sorted(set(values))) != values:
            raise ValueError("density grid must be strictly increasing and unique")
        if self.masks_per_density < 2:
            raise ValueError("interior densities require at least two masks")
        thresholds = (
            self.flat_probability_span,
            self.flat_logit_diff_span,
            self.minimum_logit_diff_variance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("flatness thresholds must be finite and non-negative")
        if self.seed < 0:
            raise ValueError("density-sweep seed must be non-negative")


@beartype
@dataclass(frozen=True)
class LassoConfig:
    degree_cap: int
    interaction_screen_size: int
    regularization: float
    heavy_coefficient_threshold: float
    maximum_iterations: int
    convergence_tolerance: float
    maximum_feature_count: int
    fit_feature_count: int
    power_iterations: int

    def __post_init__(self) -> None:
        if not 1 <= self.degree_cap <= 4:
            raise ValueError("the supported Fourier degree cap is in [1, 4]")
        if self.interaction_screen_size <= 0:
            raise ValueError("interaction screen size must be positive")
        if self.regularization < 0.0 or not math.isfinite(self.regularization):
            raise ValueError("LASSO regularization must be finite and non-negative")
        if self.heavy_coefficient_threshold <= 0.0 or not math.isfinite(
            self.heavy_coefficient_threshold
        ):
            raise ValueError("heavy coefficient threshold must be finite and positive")
        if self.maximum_iterations <= 0 or self.maximum_feature_count <= 0:
            raise ValueError("LASSO iteration and feature caps must be positive")
        if not 1 < self.fit_feature_count <= self.maximum_feature_count:
            raise ValueError("LASSO fit-feature count must lie in (1, maximum feature count]")
        if self.power_iterations <= 0:
            raise ValueError("Lipschitz power-iteration count must be positive")
        if self.convergence_tolerance <= 0.0 or not math.isfinite(self.convergence_tolerance):
            raise ValueError("LASSO convergence tolerance must be finite and positive")


@beartype
@dataclass(frozen=True)
class GradientValidationConfig:
    coefficient_holdout_count: int
    maximum_rmse: float
    maximum_absolute_error: float
    minimum_cosine_similarity: float
    variance_floor: float

    def __post_init__(self) -> None:
        if self.coefficient_holdout_count <= 0:
            raise ValueError("gradient validation needs held-out coefficients")
        if self.maximum_rmse < 0.0 or self.maximum_absolute_error < 0.0:
            raise ValueError("gradient validation error limits must be non-negative")
        if not -1.0 <= self.minimum_cosine_similarity <= 1.0:
            raise ValueError("gradient validation cosine threshold must lie in [-1, 1]")
        if self.variance_floor <= 0.0 or not math.isfinite(self.variance_floor):
            raise ValueError("variance floor must be finite and positive")


@beartype
@dataclass(frozen=True)
class DensityStabilityConfig:
    sample_budget_per_density: int
    maximum_l1_distance: float
    minimum_cosine_similarity: float

    def __post_init__(self) -> None:
        if self.sample_budget_per_density <= 1:
            raise ValueError("density stability requires repeated random corners")
        if not 0.0 <= self.maximum_l1_distance <= 2.0:
            raise ValueError("degree-profile L1 threshold must lie in [0, 2]")
        if not -1.0 <= self.minimum_cosine_similarity <= 1.0:
            raise ValueError("degree-profile cosine threshold must lie in [-1, 1]")


@beartype
@dataclass(frozen=True)
class SpectrumConfig:
    sample_budget: int
    gradient_batch_size: int
    validation_fraction: float
    seed: int
    lasso: LassoConfig
    gradient_validation: GradientValidationConfig
    density_stability: DensityStabilityConfig

    def __post_init__(self) -> None:
        if self.sample_budget <= 2:
            raise ValueError("spectrum estimation requires at least three corners")
        if self.gradient_batch_size <= 0:
            raise ValueError("spectrum gradient batch size must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation fraction must lie in (0, 0.5)")
        if self.seed < 0:
            raise ValueError("spectrum seed must be non-negative")


@beartype
@dataclass(frozen=True)
class SufficiencyConfig:
    recovery_fraction: float
    require_clean_argmax: bool
    patch_batch_size: int
    maximum_candidate_supports: int
    maximum_evaluated_site_sets: int

    def __post_init__(self) -> None:
        if not 0.0 < self.recovery_fraction <= 1.0:
            raise ValueError("sufficiency recovery fraction must lie in (0, 1]")
        if self.patch_batch_size <= 0:
            raise ValueError("causal-verification patch batch size must be positive")
        if self.maximum_candidate_supports <= 0 or self.maximum_evaluated_site_sets <= 0:
            raise ValueError(
                "causal-verification candidate and evaluated-set caps must be positive"
            )


@beartype
@dataclass(frozen=True)
class ProbabilitySufficiencyConfig:
    """Sufficiency within an absolute probability tolerance of the clean corner."""

    absolute_probability_tolerance: float
    expected_passing_singletons: tuple[Site, ...]
    require_clean_argmax: bool
    patch_batch_size: int
    maximum_candidate_supports: int
    maximum_evaluated_site_sets: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.absolute_probability_tolerance)
            or not 0.0 < self.absolute_probability_tolerance < 1.0
        ):
            raise ValueError("absolute probability tolerance must lie in (0, 1)")
        if (
            not self.expected_passing_singletons
            or tuple(sorted(set(self.expected_passing_singletons)))
            != self.expected_passing_singletons
        ):
            raise ValueError("expected probability-sufficient singletons must be non-empty and sorted")
        if self.patch_batch_size <= 0:
            raise ValueError("causal-verification patch batch size must be positive")
        if self.maximum_candidate_supports <= 0 or self.maximum_evaluated_site_sets <= 0:
            raise ValueError(
                "causal-verification candidate and evaluated-set caps must be positive"
            )


SufficiencySpec = SufficiencyConfig | ProbabilitySufficiencyConfig


@beartype
@dataclass(frozen=True)
class ExhaustiveSingletonConfig:
    required_final_token_layers: tuple[int, ...]
    reference_probability_tolerance: float

    def __post_init__(self) -> None:
        if (
            not self.required_final_token_layers
            or tuple(sorted(set(self.required_final_token_layers)))
            != self.required_final_token_layers
            or min(self.required_final_token_layers) < 0
        ):
            raise ValueError(
                "required singleton-harness layers must be non-empty, increasing, and non-negative"
            )
        if (
            not math.isfinite(self.reference_probability_tolerance)
            or self.reference_probability_tolerance < 0.0
        ):
            raise ValueError(
                "singleton reference-probability tolerance must be finite and non-negative"
            )


@beartype
@dataclass(frozen=True)
class ActiveSiteSpace:
    full_site_count: int
    active_full_indices: tuple[int, ...]
    vetoed_full_indices: tuple[int, ...]
    vetoed_sites: tuple[Site, ...]

    def __post_init__(self) -> None:
        if self.full_site_count <= 0:
            raise ValueError("active site space requires a positive full-site count")
        if (
            not self.active_full_indices
            or tuple(sorted(set(self.active_full_indices))) != self.active_full_indices
            or self.active_full_indices[0] < 0
            or self.active_full_indices[-1] >= self.full_site_count
        ):
            raise ValueError(
                "active full-site indices must be non-empty, increasing, unique, and in bounds"
            )
        if tuple(sorted(set(self.vetoed_sites))) != self.vetoed_sites:
            raise ValueError("vetoed sites must be increasing and unique")
        if (
            tuple(sorted(set(self.vetoed_full_indices))) != self.vetoed_full_indices
            or any(index < 0 or index >= self.full_site_count for index in self.vetoed_full_indices)
            or len(self.vetoed_full_indices) != len(self.vetoed_sites)
        ):
            raise ValueError(
                "vetoed full-site indices must be unique, in bounds, and paired with vetoed sites"
            )
        if set(self.active_full_indices).intersection(self.vetoed_full_indices) or set(
            self.active_full_indices
        ).union(self.vetoed_full_indices) != set(range(self.full_site_count)):
            raise ValueError("active and vetoed indices must exactly partition the full site grid")

    @property
    def active_site_count(self) -> int:
        return len(self.active_full_indices)

    def full_support(self, active_support: tuple[int, ...]) -> tuple[int, ...]:
        if tuple(sorted(set(active_support))) != active_support or any(
            index < 0 or index >= self.active_site_count for index in active_support
        ):
            raise ValueError("active support indices must be increasing, unique, and in bounds")
        return tuple(self.active_full_indices[index] for index in active_support)


@beartype
@dataclass(frozen=True)
class CacheConfig:
    reference_batch_size: int
    cached_batch_size: int
    benchmark_mask_count: int
    warmup_repetitions: int
    measured_repetitions: int
    maximum_logit_error: float
    maximum_probability_error: float
    scientific_backend: str

    def __post_init__(self) -> None:
        if (
            min(
                self.reference_batch_size,
                self.cached_batch_size,
                self.benchmark_mask_count,
                self.warmup_repetitions,
                self.measured_repetitions,
            )
            <= 0
        ):
            raise ValueError(
                "reference/cache batches, masks, warmups, and repetitions must be positive"
            )
        if self.maximum_logit_error < 0.0 or self.maximum_probability_error < 0.0:
            raise ValueError("cache parity tolerances must be non-negative")
        if self.scientific_backend != "full_sequence_reference":
            raise ValueError("Fourier scientific execution must use the full-sequence reference")


@beartype
@dataclass(frozen=True)
class HarnessCheckConfig:
    reference_probability_tolerance: float
    minimum_absolute_effect: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.reference_probability_tolerance)
            or self.reference_probability_tolerance < 0.0
        ):
            raise ValueError("harness probability tolerance must be finite and non-negative")
        if not math.isfinite(self.minimum_absolute_effect) or self.minimum_absolute_effect <= 0.0:
            raise ValueError("known-site effect floor must be finite and positive")


@beartype
@dataclass(frozen=True)
class FourierCircuitConfig:
    model: ModelCheckpointSpec
    task: TaskDatasetSpec
    sites: SiteScope
    density_sweep: DensitySweepConfig
    spectrum: SpectrumConfig
    sufficiency: SufficiencySpec
    exhaustive_singletons: ExhaustiveSingletonConfig
    cache: CacheConfig
    harness_check: HarnessCheckConfig
    artifact_root: Path

    def __post_init__(self) -> None:
        if not self.artifact_root.name:
            raise ValueError("artifact root must name a concrete directory")
        scientific_batches = (
            self.cache.reference_batch_size,
            self.spectrum.gradient_batch_size,
            self.sufficiency.patch_batch_size,
        )
        if len(set(scientific_batches)) != 1:
            raise ValueError(
                "full-sequence function, gradient, and verification batches must match exactly"
            )


@beartype
@dataclass(frozen=True)
class DensityPoint:
    density: SweepDensity
    sample_count: int
    mean_correct_probability: float
    correct_probability_variance: float
    accuracy: float
    mean_logit_diff: LogitDiff
    logit_diff_variance: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("density points require at least one sample")
        bounded = (self.mean_correct_probability, self.accuracy)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("density probability and accuracy must lie in [0, 1]")
        variances = (self.correct_probability_variance, self.logit_diff_variance)
        if any(not math.isfinite(value) or value < 0.0 for value in variances):
            raise ValueError("density variances must be finite and non-negative")


@beartype
@dataclass(frozen=True)
class FourierCoefficient:
    support: tuple[int, ...]
    degree: int
    lasso_value: float
    function_value_estimate: float
    gradient_estimate: float | None
    augmented_estimate: float | None
    is_heavy: bool

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.support))) != self.support:
            raise ValueError("coefficient support must be strictly increasing and unique")
        if self.degree != len(self.support):
            raise ValueError("coefficient degree must equal support size")
        values = (self.lasso_value, self.function_value_estimate)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("function-value coefficient estimates must be finite")
        optional = (self.gradient_estimate, self.augmented_estimate)
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("optional coefficient estimates must be finite when present")


@beartype
@dataclass(frozen=True)
class GradientValidationResult:
    support_indices: tuple[int, ...]
    rmse: float
    maximum_absolute_error: float
    cosine_similarity: float
    accepted: bool

    def __post_init__(self) -> None:
        if not self.support_indices:
            raise ValueError("gradient validation must name held-out coefficient indices")
        if self.rmse < 0.0 or self.maximum_absolute_error < 0.0:
            raise ValueError("gradient validation errors must be non-negative")
        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise ValueError("gradient validation cosine must lie in [-1, 1]")


@beartype
@dataclass(frozen=True)
class DegreeProfile:
    density: Density
    squared_weight_by_degree: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.squared_weight_by_degree:
            raise ValueError("degree profile must contain at least the constant degree")
        if any(not math.isfinite(value) or value < 0.0 for value in self.squared_weight_by_degree):
            raise ValueError("degree-profile weights must be finite and non-negative")


@beartype
@dataclass(frozen=True)
class VerifiedMinset:
    sites: SiteSet
    raw_logit_diff: LogitDiff
    correct_probability: float
    sufficiency_margin: float
    generating_supports: tuple[SiteSet, ...]

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("verified minsets must be non-empty")
        if tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("verified minset sites must be sorted and unique")
        if not 0.0 <= self.correct_probability <= 1.0:
            raise ValueError("verified minset probability must lie in [0, 1]")
        if not math.isfinite(self.sufficiency_margin) or self.sufficiency_margin < 0.0:
            raise ValueError("verified sufficiency margin must be finite and non-negative")
        if not self.generating_supports:
            raise ValueError("verified minsets must retain at least one Fourier hypothesis")


@jaxtyped(typechecker=beartype)
def as_patch_mask(mask: Mask2D, grid: SiteGrid) -> PatchMask:
    """Validate and phantom-cast a concrete mask for one exact site grid."""

    if tuple(mask.shape) != grid.shape:
        raise ValueError(f"patch mask shape {tuple(mask.shape)} != site grid {grid.shape}")
    if not mask.is_contiguous():
        raise ValueError("patch masks must be contiguous in token-major order")
    return cast(PatchMask, mask)


@jaxtyped(typechecker=beartype)
def flatten_masks(masks: MaskBatch, grid: SiteGrid) -> FlatMaskBatch:
    if tuple(masks.shape[1:]) != grid.shape:
        raise ValueError("mask batch axes do not match the configured site grid")
    return masks.reshape(masks.shape[0], grid.site_count)


@jaxtyped(typechecker=beartype)
def sample_patch_masks(
    sample_count: int,
    grid: SiteGrid,
    density: SweepDensity,
    generator: t.Generator,
) -> MaskBatch:
    if sample_count <= 0:
        raise ValueError("mask sample count must be positive")
    probability = float(density)
    shape = (sample_count, *grid.shape)
    if probability == 0.0:
        return t.zeros(shape, dtype=t.bool)
    if probability == 1.0:
        return t.ones(shape, dtype=t.bool)
    return t.rand(shape, generator=generator, dtype=t.float64) < probability


@jaxtyped(typechecker=beartype)
def biased_standardized_bits(masks: FlatMaskBatch, density: Density) -> StandardizedBits:
    probability = float(density)
    scale = math.sqrt(probability * (1.0 - probability))
    return (masks.to(dtype=t.float64) - probability) / scale


@beartype
def validate_support(support: tuple[int, ...], site_count: int, degree_cap: int) -> None:
    if tuple(sorted(set(support))) != support:
        raise ValueError("Fourier support must be strictly increasing and unique")
    if len(support) > degree_cap:
        raise ValueError("Fourier support exceeds the configured degree cap")
    if any(index < 0 or index >= site_count for index in support):
        raise ValueError("Fourier support contains a site outside the mask")


@jaxtyped(typechecker=beartype)
def parity_feature_matrix(
    masks: FlatMaskBatch,
    supports: tuple[tuple[int, ...], ...],
    density: Density,
) -> FeatureMatrix:
    if not supports:
        raise ValueError("parity design requires at least the constant support")
    if len(set(supports)) != len(supports):
        raise ValueError("parity supports must be unique")
    site_count = masks.shape[1]
    degree_cap = max(len(support) for support in supports)
    standardized = biased_standardized_bits(masks, density)
    for support in supports:
        validate_support(support, site_count, degree_cap)
    features = t.empty(
        (masks.shape[0], len(supports)),
        dtype=t.float64,
        device=masks.device,
    )
    for degree in range(degree_cap + 1):
        feature_indices = [
            index for index, support in enumerate(supports) if len(support) == degree
        ]
        for start in range(0, len(feature_indices), SUPPORT_CHUNK_SIZE):
            chunk_indices = feature_indices[start : start + SUPPORT_CHUNK_SIZE]
            if degree == 0:
                features[:, chunk_indices] = 1.0
                continue
            site_indices = t.tensor(
                [supports[index] for index in chunk_indices],
                dtype=t.int64,
                device=masks.device,
            )
            features[:, chunk_indices] = standardized[:, site_indices].prod(dim=2)
    return features


@jaxtyped(typechecker=beartype)
def exact_fourier_coefficients(
    masks: FlatMaskBatch,
    values: ValueVector,
    supports: tuple[tuple[int, ...], ...],
    density: Density,
) -> CoefficientVector:
    if masks.shape[0] != values.shape[0]:
        raise ValueError("mask and function-value sample counts must agree")
    features = parity_feature_matrix(masks, supports, density)
    return (features * values.to(dtype=t.float64).unsqueeze(1)).mean(dim=0)


@beartype
def all_supports(site_count: int, degree_cap: int) -> tuple[tuple[int, ...], ...]:
    if site_count <= 0 or not 0 <= degree_cap <= site_count:
        raise ValueError("support enumeration needs positive sites and a valid degree cap")
    return ((),) + tuple(
        support
        for degree in range(1, degree_cap + 1)
        for support in itertools.combinations(range(site_count), degree)
    )


@beartype
def screened_supports(
    site_count: int,
    screened_sites: tuple[int, ...],
    config: LassoConfig,
) -> tuple[tuple[int, ...], ...]:
    if site_count <= 0:
        raise ValueError("screened supports require a positive site count")
    if tuple(sorted(set(screened_sites))) != screened_sites:
        raise ValueError("screened site indices must be strictly increasing and unique")
    if any(index < 0 or index >= site_count for index in screened_sites):
        raise ValueError("screened site index is outside the site universe")
    supports: list[tuple[int, ...]] = [()]
    supports.extend((index,) for index in range(site_count))
    for degree in range(2, config.degree_cap + 1):
        supports.extend(itertools.combinations(screened_sites, degree))
    if len(supports) > config.maximum_feature_count:
        raise RuntimeError(
            f"screened Fourier family has {len(supports)} features, exceeding the "
            f"explicit cap {config.maximum_feature_count}"
        )
    return tuple(supports)


@jaxtyped(typechecker=beartype)
def screen_sites_from_gradients(
    gradients: GradientBatch,
    values: ValueVector,
    masks: FlatMaskBatch,
    density: Density,
    count: int,
) -> tuple[int, ...]:
    """Rank interactions with gradients only after the held-out validation gate passes."""

    if gradients.shape != masks.shape:
        raise ValueError("gradient and mask batches must have identical sample/site axes")
    if gradients.shape[0] != values.shape[0]:
        raise ValueError("gradient and function-value sample counts must agree")
    if not 0 < count <= gradients.shape[1]:
        raise ValueError("interaction screen count must lie within the site count")
    sigma = math.sqrt(float(density) * (1.0 - float(density)))
    gradient_signal = gradients.to(dtype=t.float64).abs().mean(dim=0) * sigma
    standardized = biased_standardized_bits(masks, density)
    marginal_signal = (standardized * values.to(dtype=t.float64).unsqueeze(1)).mean(dim=0).abs()
    score = t.maximum(gradient_signal, marginal_signal)
    order = sorted(range(score.numel()), key=lambda index: (-float(score[index]), index))
    return tuple(sorted(order[:count]))


@jaxtyped(typechecker=beartype)
def screen_sites_from_function_values(
    values: ValueVector,
    masks: FlatMaskBatch,
    density: Density,
    count: int,
) -> tuple[int, ...]:
    """Rank interaction sites using function values alone."""

    if masks.shape[0] != values.shape[0]:
        raise ValueError("mask and function-value sample counts must agree")
    if not 0 < count <= masks.shape[1]:
        raise ValueError("interaction screen count must lie within the site count")
    standardized = biased_standardized_bits(masks, density)
    score = (standardized * values.to(dtype=t.float64).unsqueeze(1)).mean(dim=0).abs()
    order = sorted(range(score.numel()), key=lambda index: (-float(score[index]), index))
    return tuple(sorted(order[:count]))


@jaxtyped(typechecker=beartype)
def function_correlation_feature_indices(
    features: FeatureMatrix,
    values: ValueVector,
    maximum_count: int,
) -> tuple[int, ...]:
    """Keep the intercept plus strongest finite-sample function correlations."""

    if features.shape[0] != values.shape[0]:  # pragma: no cover - jaxtyping rejects first
        raise ValueError("feature and function-value sample counts must agree")
    if features.shape[1] == 0 or maximum_count <= 1:
        raise ValueError("function-correlation screening needs features and a positive cap")
    retained_count = min(maximum_count, features.shape[1])
    if retained_count == features.shape[1]:
        return tuple(range(features.shape[1]))
    typed_values = values.to(device=features.device, dtype=t.float64)
    centered_values = typed_values - typed_values.mean()
    centered_features = features[:, 1:] - features[:, 1:].mean(dim=0, keepdim=True)
    denominators = t.linalg.vector_norm(centered_features, dim=0) * t.linalg.vector_norm(
        centered_values
    )
    numerators = (centered_features * centered_values.unsqueeze(1)).sum(dim=0).abs()
    scores = t.where(denominators > 0.0, numerators / denominators, 0.0)
    order = sorted(
        range(1, features.shape[1]),
        key=lambda index: (-float(scores[index - 1]), index),
    )
    return (0, *sorted(order[: retained_count - 1]))


@jaxtyped(typechecker=beartype)
def _power_lipschitz(
    features: FeatureMatrix,
    iterations: int,
) -> float:
    if features.shape[0] == 0 or features.shape[1] == 0 or iterations <= 0:
        raise ValueError("Lipschitz estimation requires a non-empty matrix and iterations")
    vector = t.ones(features.shape[1], dtype=t.float64, device=features.device)
    vector /= t.linalg.vector_norm(vector)
    for _ in range(iterations):
        projected = features.T @ (features @ vector) / features.shape[0]
        norm = t.linalg.vector_norm(projected)
        if float(norm) == 0.0:
            return 1.0
        vector = projected / norm
    rayleigh = t.dot(vector, features.T @ (features @ vector)) / features.shape[0]
    value = float(rayleigh)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("LASSO design has a non-positive or non-finite Lipschitz estimate")
    return value


@jaxtyped(typechecker=beartype)
def _lasso_kkt_maximum(
    gradient: CoefficientVector,
    coefficients: CoefficientVector,
    regularization: float,
) -> float:
    """Return the exact maximum first-order optimality violation for this LASSO."""

    if gradient.shape != coefficients.shape or gradient.shape[0] == 0:
        raise ValueError("LASSO KKT vectors must be non-empty and shape matched")
    intercept_violation = gradient[0].abs()
    if coefficients.shape[0] == 1:
        return float(intercept_violation)
    penalized = coefficients[1:]
    penalized_gradient = gradient[1:]
    active = penalized != 0.0
    active_violation = (penalized_gradient + regularization * t.sign(penalized)).abs()
    inactive_violation = t.clamp(penalized_gradient.abs() - regularization, min=0.0)
    violations = t.where(active, active_violation, inactive_violation)
    return max(float(intercept_violation), float(violations.max()))


@jaxtyped(typechecker=beartype)
def fit_lasso_fista(
    features: FeatureMatrix,
    values: ValueVector,
    config: LassoConfig,
) -> CoefficientVector:
    """Fit function values only; the unpenalized column zero is the intercept."""

    if features.shape[0] != values.shape[0] or features.shape[1] == 0:
        raise ValueError("LASSO design/value axes are inconsistent or empty")
    target = values.to(dtype=t.float64, device=features.device)
    lipschitz = _power_lipschitz(features, config.power_iterations) * 1.001
    coefficients = t.zeros(features.shape[1], dtype=t.float64, device=features.device)
    momentum_point = coefficients.clone()
    momentum = 1.0
    final_change = math.inf
    final_kkt = math.inf
    for _ in range(config.maximum_iterations):
        residual = features @ momentum_point - target
        gradient = features.T @ residual / features.shape[0]
        smooth_at_momentum = 0.5 * t.mean(residual.square())
        for _backtrack in range(64):
            threshold = config.regularization / lipschitz
            proposal = momentum_point - gradient / lipschitz
            updated = proposal.clone()
            updated[1:] = t.sign(proposal[1:]) * t.clamp(
                proposal[1:].abs() - threshold,
                min=0.0,
            )
            delta = updated - momentum_point
            updated_residual = features @ updated - target
            smooth_at_updated = 0.5 * t.mean(updated_residual.square())
            majorizing_bound = (
                smooth_at_momentum + t.dot(gradient, delta) + 0.5 * lipschitz * t.dot(delta, delta)
            )
            numerical_slack = 1.0e-12 * max(1.0, abs(float(smooth_at_momentum)))
            if float(smooth_at_updated) <= float(majorizing_bound) + numerical_slack:
                break
            lipschitz *= 2.0
        else:
            raise RuntimeError("LASSO backtracking failed to find a majorizing step")
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        extrapolated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        change = t.linalg.vector_norm(updated - coefficients)
        updated_gradient = features.T @ updated_residual / features.shape[0]
        kkt_maximum = _lasso_kkt_maximum(
            updated_gradient,
            updated,
            config.regularization,
        )
        final_change = float(change)
        final_kkt = kkt_maximum
        # Gradient restart prevents the accelerated sequence from oscillating along
        # nearly collinear parity columns without changing the convex objective.
        if float(t.dot(extrapolated - updated, updated - coefficients)) > 0.0:
            extrapolated = updated
            next_momentum = 1.0
        coefficients = updated
        momentum_point = extrapolated
        momentum = next_momentum
        if kkt_maximum <= config.convergence_tolerance:
            break
    else:
        raise RuntimeError(
            "LASSO FISTA failed to converge to the KKT tolerance within the "
            f"configured iteration cap: maximum_kkt={final_kkt:.8g}, "
            f"coefficient_change={final_change:.8g}"
        )
    if not bool(t.isfinite(coefficients).all()):
        raise RuntimeError("LASSO returned a non-finite coefficient")
    return coefficients


@jaxtyped(typechecker=beartype)
def fit_lasso_coordinate_descent(
    features: FeatureMatrix,
    values: ValueVector,
    config: LassoConfig,
) -> CoefficientVector:
    """Solve the same LASSO by deterministic maximum-KKT coordinate descent."""

    if features.shape[0] != values.shape[0] or features.shape[1] == 0:
        raise ValueError("LASSO design/value axes are inconsistent or empty")
    design = features.to(dtype=t.float64)
    target = values.to(dtype=t.float64, device=design.device)
    sample_count = design.shape[0]
    squared_norms = design.square().sum(dim=0) / sample_count
    if float(squared_norms[0]) <= 0.0:
        raise RuntimeError("unpenalized LASSO intercept column has zero norm")
    coefficients = t.zeros(design.shape[1], dtype=t.float64, device=design.device)
    residual = target.clone()
    final_kkt = math.inf
    maximum_coordinate_updates = config.maximum_iterations * design.shape[1]
    for _ in range(maximum_coordinate_updates):
        gradient = -(design.T @ residual) / sample_count
        active = coefficients[1:] != 0.0
        penalized_violations = t.where(
            active,
            (gradient[1:] + config.regularization * t.sign(coefficients[1:])).abs(),
            t.clamp(gradient[1:].abs() - config.regularization, min=0.0),
        )
        violations = t.cat((gradient[:1].abs(), penalized_violations))
        maximum, selected = violations.max(dim=0)
        final_kkt = float(maximum)
        if final_kkt <= config.convergence_tolerance:
            break
        index = int(selected)
        norm = float(squared_norms[index])
        if norm == 0.0:
            raise RuntimeError("maximum-KKT coordinate unexpectedly has zero norm")
        column = design[:, index]
        old_value = coefficients[index].clone()
        residual += column * old_value
        partial = t.dot(column, residual) / sample_count
        if index == 0:
            new_value = partial / squared_norms[index]
        else:
            shrunk = t.sign(partial) * t.clamp(
                partial.abs() - config.regularization,
                min=0.0,
            )
            new_value = shrunk / squared_norms[index]
        coefficients[index] = new_value
        residual -= column * new_value
    else:
        raise RuntimeError(
            "LASSO coordinate descent failed to converge to the KKT tolerance within the "
            f"configured coordinate-update cap: maximum_kkt={final_kkt:.8g}"
        )
    if not bool(t.isfinite(coefficients).all()):
        raise RuntimeError("LASSO coordinate descent returned a non-finite coefficient")
    return coefficients


@jaxtyped(typechecker=beartype)
def gradient_coefficient_estimates(
    gradients: GradientBatch,
    masks: FlatMaskBatch,
    supports: tuple[tuple[int, ...], ...],
    density: Density,
) -> CoefficientVector:
    """Apply the multilinear derivative identity as an explicitly provisional estimate."""

    if gradients.shape != masks.shape:
        raise ValueError("gradient and mask sample/site axes must agree")
    samples = gradient_coefficient_samples(gradients, masks, supports, density)
    return samples.mean(dim=0)


@jaxtyped(typechecker=beartype)
def gradient_coefficient_samples(
    gradients: GradientBatch,
    masks: FlatMaskBatch,
    supports: tuple[tuple[int, ...], ...],
    density: Density,
) -> CoefficientSamples:
    """Return each sample's provisional multilinear-derivative estimate."""

    if gradients.shape != masks.shape:
        raise ValueError("gradient and mask sample/site axes must agree")
    if not supports or len(set(supports)) != len(supports):
        raise ValueError("gradient coefficient supports must be non-empty and unique")
    degree_cap = max(map(len, supports))
    for support in supports:
        validate_support(support, masks.shape[1], degree_cap)
    standardized = biased_standardized_bits(masks, density)
    sigma = math.sqrt(float(density) * (1.0 - float(density)))
    samples = t.full(
        (masks.shape[0], len(supports)),
        float("nan"),
        dtype=t.float64,
        device=masks.device,
    )
    for degree in range(1, degree_cap + 1):
        feature_indices = [
            index for index, support in enumerate(supports) if len(support) == degree
        ]
        for start in range(0, len(feature_indices), SUPPORT_CHUNK_SIZE):
            chunk_indices = feature_indices[start : start + SUPPORT_CHUNK_SIZE]
            site_indices = t.tensor(
                [supports[index] for index in chunk_indices],
                dtype=t.int64,
                device=masks.device,
            )
            per_site_samples: list[t.Tensor] = []
            for differentiated_offset in range(degree):
                differentiated = site_indices[:, differentiated_offset]
                if degree == 1:
                    character = t.ones(
                        (masks.shape[0], len(chunk_indices)),
                        dtype=t.float64,
                        device=masks.device,
                    )
                else:
                    remainder_offsets = [
                        offset for offset in range(degree) if offset != differentiated_offset
                    ]
                    character = standardized[:, site_indices[:, remainder_offsets]].prod(dim=2)
                per_site_samples.append(
                    sigma * gradients[:, differentiated].to(dtype=t.float64) * character
                )
            samples[:, chunk_indices] = t.stack(per_site_samples, dim=2).mean(dim=2)
    return samples


@jaxtyped(typechecker=beartype)
def validate_gradient_estimates(
    function_estimates: CoefficientVector,
    gradient_estimates: CoefficientVector,
    holdout_indices: tuple[int, ...],
    config: GradientValidationConfig,
) -> GradientValidationResult:
    if function_estimates.shape != gradient_estimates.shape:
        raise ValueError("function and gradient coefficient vectors must have equal shape")
    if not holdout_indices or tuple(sorted(set(holdout_indices))) != holdout_indices:
        raise ValueError("held-out coefficient indices must be non-empty, sorted, and unique")
    if any(index <= 0 or index >= function_estimates.shape[0] for index in holdout_indices):
        raise ValueError("held-out indices must refer to nonconstant coefficients")
    function_values = function_estimates[list(holdout_indices)]
    gradient_values = gradient_estimates[list(holdout_indices)]
    if not bool(t.isfinite(function_values).all() and t.isfinite(gradient_values).all()):
        raise ValueError("held-out coefficient estimates must be finite")
    differences = gradient_values - function_values
    rmse = float(t.sqrt(t.mean(differences.square())))
    maximum_error = float(differences.abs().max())
    denominator = t.linalg.vector_norm(function_values) * t.linalg.vector_norm(gradient_values)
    cosine = (
        1.0
        if float(denominator) == 0.0 and bool(t.equal(function_values, gradient_values))
        else 0.0
        if float(denominator) == 0.0
        else float(t.dot(function_values, gradient_values) / denominator)
    )
    accepted = bool(
        rmse <= config.maximum_rmse
        and maximum_error <= config.maximum_absolute_error
        and cosine >= config.minimum_cosine_similarity
    )
    return GradientValidationResult(
        holdout_indices,
        rmse,
        maximum_error,
        cosine,
        accepted,
    )


@jaxtyped(typechecker=beartype)
def inverse_variance_augment(
    function_samples: FeatureMatrix,
    gradient_samples: FeatureMatrix,
    config: GradientValidationConfig,
) -> CoefficientVector:
    if function_samples.shape != gradient_samples.shape or function_samples.shape[0] <= 1:
        raise ValueError("augmentation needs paired repeated coefficient samples")
    function_mean = function_samples.mean(dim=0)
    gradient_mean = gradient_samples.mean(dim=0)
    function_variance = function_samples.var(dim=0, unbiased=True) / function_samples.shape[0]
    gradient_variance = gradient_samples.var(dim=0, unbiased=True) / gradient_samples.shape[0]
    function_precision = 1.0 / t.clamp(function_variance, min=config.variance_floor)
    gradient_precision = 1.0 / t.clamp(gradient_variance, min=config.variance_floor)
    return (function_mean * function_precision + gradient_mean * gradient_precision) / (
        function_precision + gradient_precision
    )


@beartype
def density_curve_is_flat(
    points: tuple[DensityPoint, ...],
    config: DensitySweepConfig,
) -> bool:
    if len(points) != len(config.density_grid):
        raise ValueError("density curve must contain exactly one point per configured density")
    probabilities = tuple(point.mean_correct_probability for point in points)
    logit_diffs = tuple(float(point.mean_logit_diff) for point in points)
    interior_variance = max(point.logit_diff_variance for point in points[1:-1])
    return bool(
        max(probabilities) - min(probabilities) < config.flat_probability_span
        and max(logit_diffs) - min(logit_diffs) < config.flat_logit_diff_span
        and interior_variance < config.minimum_logit_diff_variance
    )


@beartype
def select_transition_density(
    points: tuple[DensityPoint, ...],
    config: DensitySweepConfig,
) -> Density:
    if density_curve_is_flat(points, config):
        raise RuntimeError("density curve is flat; stop before spectrum estimation")
    interior = points[1:-1]
    if not interior:
        raise ValueError("density curve has no interior transition candidates")
    selected = max(
        interior,
        key=lambda point: (
            point.logit_diff_variance,
            point.correct_probability_variance,
            -abs(float(point.density) - 0.5),
        ),
    )
    return Density.parse(float(selected.density))


@beartype
def normalized_degree_profile(
    coefficients: tuple[FourierCoefficient, ...],
    degree_cap: int,
    density: Density,
) -> DegreeProfile:
    if not coefficients or degree_cap < 0:
        raise ValueError("degree profile requires coefficients and a non-negative cap")
    weights = [0.0] * (degree_cap + 1)
    for coefficient in coefficients:
        if coefficient.degree <= degree_cap:
            weights[coefficient.degree] += coefficient.function_value_estimate**2
    total = sum(weights)
    if total == 0.0:
        raise RuntimeError("degree profile has zero total spectral weight")
    return DegreeProfile(density, tuple(weight / total for weight in weights))


@beartype
def compare_degree_profiles(
    profiles: tuple[DegreeProfile, ...],
    config: DensityStabilityConfig,
) -> tuple[bool, float, float]:
    if len(profiles) < 2:
        raise ValueError("density stability requires at least two degree profiles")
    if len({len(profile.squared_weight_by_degree) for profile in profiles}) != 1:
        raise ValueError("degree profiles must share a degree axis")
    maximum_l1 = 0.0
    minimum_cosine = 1.0
    for left, right in itertools.combinations(profiles, 2):
        left_tensor = t.tensor(left.squared_weight_by_degree, dtype=t.float64)
        right_tensor = t.tensor(right.squared_weight_by_degree, dtype=t.float64)
        maximum_l1 = max(maximum_l1, float(t.sum(t.abs(left_tensor - right_tensor))))
        denominator = t.linalg.vector_norm(left_tensor) * t.linalg.vector_norm(right_tensor)
        cosine = float(t.dot(left_tensor, right_tensor) / denominator)
        minimum_cosine = min(minimum_cosine, cosine)
    stable = bool(
        maximum_l1 <= config.maximum_l1_distance
        and minimum_cosine >= config.minimum_cosine_similarity
    )
    return stable, maximum_l1, minimum_cosine


@beartype
def as_non_empty_candidates(candidates: tuple[SiteSet, ...]) -> CandidateSiteSets:
    if not candidates:
        raise ValueError("stage 2 requires at least one non-empty candidate site set")
    for candidate in candidates:
        if not candidate or tuple(sorted(set(candidate))) != candidate:
            raise ValueError("candidate site sets must each be non-empty, sorted, and unique")
    return cast(CandidateSiteSets, candidates)


@beartype
def sufficiency_threshold(
    dirty_logit_diff: LogitDiff,
    clean_logit_diff: LogitDiff,
    config: SufficiencyConfig,
) -> LogitDiff:
    if float(clean_logit_diff) <= float(dirty_logit_diff):
        raise RuntimeError("clean checkpoint must improve raw logit diff over dirty checkpoint")
    value = float(dirty_logit_diff) + config.recovery_fraction * (
        float(clean_logit_diff) - float(dirty_logit_diff)
    )
    return LogitDiff.parse(value)


@beartype
def resolved_sufficiency_threshold(
    dirty_logit_diff: LogitDiff,
    clean_logit_diff: LogitDiff,
    dirty_correct_probability: float,
    clean_correct_probability: float,
    config: SufficiencySpec,
) -> LogitDiff:
    """Resolve either registered causal rule onto the raw log-odds axis."""

    if (
        not 0.0 <= dirty_correct_probability <= 1.0
        or not 0.0 <= clean_correct_probability <= 1.0
        or clean_correct_probability <= dirty_correct_probability
    ):
        raise RuntimeError("clean probability must improve over dirty probability inside [0, 1]")
    if isinstance(config, SufficiencyConfig):
        return sufficiency_threshold(dirty_logit_diff, clean_logit_diff, config)
    probability = clean_correct_probability - config.absolute_probability_tolerance
    if not dirty_correct_probability < probability < clean_correct_probability:
        raise RuntimeError(
            "clean-minus-tolerance probability threshold must lie strictly between endpoints"
        )
    return LogitDiff.parse(math.log(probability / (1.0 - probability)))


@beartype
def enumerate_minimal_sufficient_subsets(
    candidates: CandidateSiteSets,
    is_sufficient: Callable[[SiteSet], bool],
) -> tuple[tuple[SiteSet, SiteSet], ...]:
    """Explore every greedy removal path and return ``(minset, generator)`` pairs."""

    discovered: set[tuple[SiteSet, SiteSet]] = set()
    for candidate in candidates:
        if not is_sufficient(candidate):
            continue
        frontier = [candidate]
        visited: set[SiteSet] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            sufficient_children: list[SiteSet] = []
            for index in range(len(current)):
                child = current[:index] + current[index + 1 :]
                if child and is_sufficient(child):
                    sufficient_children.append(child)
            if sufficient_children:
                frontier.extend(sufficient_children)
            else:
                discovered.add((current, candidate))
    return tuple(sorted(discovered, key=lambda pair: (len(pair[0]), pair[0], pair[1])))


@jaxtyped(typechecker=beartype)
def all_boolean_corners(site_count: int) -> FlatMaskBatch:
    if not 0 < site_count <= 20:
        raise ValueError("exhaustive synthetic corners support between 1 and 20 sites")
    rows = tuple(itertools.product((False, True), repeat=site_count))
    return t.tensor(rows, dtype=t.bool)


@jaxtyped(typechecker=beartype)
def majority_two_of_three(masks: FlatMaskBatch) -> ValueVector:
    if masks.shape[1] != 3:
        raise ValueError("2-of-3 majority requires exactly three fake sites")
    return (masks.sum(dim=1) >= 2).to(dtype=t.float64)


@jaxtyped(typechecker=beartype)
def two_clause_monotone_dnf(masks: FlatMaskBatch) -> ValueVector:
    if masks.shape[1] != 4:
        raise ValueError("reference monotone DNF requires exactly four fake sites")
    first = masks[:, 0] & masks[:, 1]
    second = masks[:, 2] & masks[:, 3]
    return (first | second).to(dtype=t.float64)


@beartype
def run_synthetic_reference_gate() -> dict[str, object]:
    """Prove the estimator and all-path walk-down before any model runtime is entered."""

    density = Density.parse(0.5)
    majority_masks = all_boolean_corners(3)
    majority_supports = all_supports(3, 3)
    majority_coefficients = exact_fourier_coefficients(
        majority_masks,
        majority_two_of_three(majority_masks),
        majority_supports,
        density,
    )
    expected_majority = t.tensor(
        (0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, -0.25),
        dtype=t.float64,
    )
    if not t.allclose(majority_coefficients, expected_majority, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("synthetic majority Fourier coefficients failed exact recovery")
    synthetic_lasso = LassoConfig(
        degree_cap=4,
        interaction_screen_size=4,
        regularization=0.0,
        heavy_coefficient_threshold=1.0e-8,
        maximum_iterations=10_000,
        convergence_tolerance=1.0e-12,
        maximum_feature_count=100,
        fit_feature_count=100,
        power_iterations=100,
    )
    majority_lasso = fit_lasso_fista(
        parity_feature_matrix(majority_masks, majority_supports, density),
        majority_two_of_three(majority_masks),
        synthetic_lasso,
    )
    if not t.allclose(majority_lasso, expected_majority, atol=1.0e-10, rtol=0.0):
        raise RuntimeError("synthetic majority LASSO failed exact coefficient recovery")
    majority_coordinate_lasso = fit_lasso_coordinate_descent(
        parity_feature_matrix(majority_masks, majority_supports, density),
        majority_two_of_three(majority_masks),
        synthetic_lasso,
    )
    if not t.allclose(
        majority_coordinate_lasso,
        expected_majority,
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("synthetic majority coordinate LASSO failed exact recovery")

    majority_sites = tuple(Site(0, layer) for layer in range(3))
    majority_candidates = as_non_empty_candidates((majority_sites,))
    majority_pairs = enumerate_minimal_sufficient_subsets(
        majority_candidates,
        lambda candidate: len(candidate) >= 2,
    )
    expected_majority_minterms = {
        (majority_sites[0], majority_sites[1]),
        (majority_sites[0], majority_sites[2]),
        (majority_sites[1], majority_sites[2]),
    }
    if {minset for minset, _generator in majority_pairs} != expected_majority_minterms:
        raise RuntimeError("synthetic majority walk-down failed exact minterm recovery")

    dnf_masks = all_boolean_corners(4)
    dnf_supports = all_supports(4, 4)
    dnf_coefficients = exact_fourier_coefficients(
        dnf_masks,
        two_clause_monotone_dnf(dnf_masks),
        dnf_supports,
        density,
    )
    expected_dnf: list[float] = []
    positive_supports = {(0,), (1,), (2,), (3,), (0, 1), (2, 3)}
    for support in dnf_supports:
        if not support:
            expected_dnf.append(7.0 / 16.0)
        elif support in positive_supports:
            expected_dnf.append(3.0 / 16.0)
        else:
            expected_dnf.append(-1.0 / 16.0)
    if not t.allclose(
        dnf_coefficients,
        t.tensor(expected_dnf, dtype=t.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("synthetic monotone-DNF Fourier coefficients failed exact recovery")
    dnf_lasso = fit_lasso_fista(
        parity_feature_matrix(dnf_masks, dnf_supports, density),
        two_clause_monotone_dnf(dnf_masks),
        synthetic_lasso,
    )
    if not t.allclose(
        dnf_lasso,
        t.tensor(expected_dnf, dtype=t.float64),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("synthetic monotone-DNF LASSO failed exact coefficient recovery")
    dnf_coordinate_lasso = fit_lasso_coordinate_descent(
        parity_feature_matrix(dnf_masks, dnf_supports, density),
        two_clause_monotone_dnf(dnf_masks),
        synthetic_lasso,
    )
    if not t.allclose(
        dnf_coordinate_lasso,
        t.tensor(expected_dnf, dtype=t.float64),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("synthetic monotone-DNF coordinate LASSO failed exact recovery")

    dnf_sites = tuple(Site(0, layer) for layer in range(4))
    dnf_pairs = enumerate_minimal_sufficient_subsets(
        as_non_empty_candidates((dnf_sites,)),
        lambda candidate: bool(
            {dnf_sites[0], dnf_sites[1]}.issubset(candidate)
            or {dnf_sites[2], dnf_sites[3]}.issubset(candidate)
        ),
    )
    expected_dnf_minterms = {
        (dnf_sites[0], dnf_sites[1]),
        (dnf_sites[2], dnf_sites[3]),
    }
    if {minset for minset, _generator in dnf_pairs} != expected_dnf_minterms:
        raise RuntimeError("synthetic monotone-DNF walk-down failed exact minterm recovery")

    return {
        "status": "passed",
        "basis": "p_biased_orthonormal",
        "density": 0.5,
        "coefficient_tolerance": 1.0e-12,
        "lasso_tolerance": 1.0e-10,
        "majority": {
            "site_count": 3,
            "coefficient_count": len(majority_supports),
            "minterms": [
                [[site.token_index, site.layer] for site in minset]
                for minset in sorted(expected_majority_minterms)
            ],
        },
        "monotone_dnf": {
            "site_count": 4,
            "coefficient_count": len(dnf_supports),
            "minterms": [
                [[site.token_index, site.layer] for site in minset]
                for minset in sorted(expected_dnf_minterms)
            ],
        },
    }


__all__ = [
    "CacheConfig",
    "CandidateSiteSets",
    "CoefficientVector",
    "DegreeProfile",
    "Density",
    "DensityPoint",
    "DensityStabilityConfig",
    "DensitySweepConfig",
    "ExhaustiveSingletonConfig",
    "FourierCircuitConfig",
    "FourierCoefficient",
    "FullPromptSites",
    "GradientValidationConfig",
    "GradientValidationResult",
    "ActiveSiteSpace",
    "HarnessCheckConfig",
    "LassoConfig",
    "LogitDiff",
    "ModelCheckpointSpec",
    "PatchMask",
    "ProbabilitySufficiencyConfig",
    "ReverseWindowSites",
    "Site",
    "SiteGrid",
    "SiteSet",
    "SpectrumConfig",
    "SufficiencyConfig",
    "SufficiencySpec",
    "SweepDensity",
    "TaskDatasetSpec",
    "VerifiedMinset",
    "all_boolean_corners",
    "all_supports",
    "as_non_empty_candidates",
    "as_patch_mask",
    "biased_standardized_bits",
    "compare_degree_profiles",
    "density_curve_is_flat",
    "enumerate_minimal_sufficient_subsets",
    "exact_fourier_coefficients",
    "fit_lasso_fista",
    "fit_lasso_coordinate_descent",
    "flatten_masks",
    "gradient_coefficient_estimates",
    "gradient_coefficient_samples",
    "function_correlation_feature_indices",
    "inverse_variance_augment",
    "majority_two_of_three",
    "normalized_degree_profile",
    "parity_feature_matrix",
    "sample_patch_masks",
    "screen_sites_from_gradients",
    "screen_sites_from_function_values",
    "screened_supports",
    "select_transition_density",
    "run_synthetic_reference_gate",
    "resolved_sufficiency_threshold",
    "sufficiency_threshold",
    "two_clause_monotone_dnf",
    "validate_gradient_estimates",
]
