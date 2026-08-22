"""Pure contracts for cross-checkpoint answer-location swap minsets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, jaxtyped
from phantom import Phantom
from phantom.sized import NonEmpty

from oocr_training_dynamics.answer_lookup import ANSWER_LABELS
from oocr_training_dynamics.fourier_circuits import SweepDensity

SWITCHED_ANSWER_SCHEMA_VERSION = 1
SWITCHED_ANSWER_DONOR_STEP = 1_500
SWITCHED_ANSWER_RECIPIENT_STEP = 0
SWITCHED_ANSWER_FUNCTION_ID = "add_5"
SWITCHED_ANSWER_CORRECT_CHOICE_INDEX = 2
SWITCHED_ANSWER_INTERFACES = ("attention_input", "resid_post")

LayerMask = Bool[t.Tensor, "layer"]
LayerMaskBatch = Bool[t.Tensor, "sample layer"]


@jaxtyped(typechecker=beartype)
def _is_layer_patch_mask(value: LayerMask) -> bool:
    return bool(value.dtype is t.bool and value.ndim == 1 and value.shape[0] > 0)


class LayerPatchMask(Phantom[t.Tensor], predicate=_is_layer_patch_mask, bound=t.Tensor):
    """A non-empty Boolean mask over composite layerwise swap operators."""


@beartype
@dataclass(frozen=True, order=True)
class LayerSwapSite:
    """One simultaneous correct/wrong line-terminator swap at one layer."""

    layer: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("swap-site layer must be non-negative")


LayerSwapSiteSet = tuple[LayerSwapSite, ...]
NonEmptyLayerSwapSiteSets = NonEmpty[LayerSwapSiteSet]


@beartype
@dataclass(frozen=True)
class SwitchedAnswerCheckpointSpec:
    model_key: str
    model_id: str
    revision: str
    condition: str
    seed: int
    donor_step: int
    recipient_step: int

    def __post_init__(self) -> None:
        if self.model_key != "olmo3-7b" or self.condition != "correct":
            raise ValueError("switched-answer minsets require the primary OLMo3 correct run")
        if not self.model_id or "/" not in self.model_id:
            raise ValueError("model ID must be a namespaced identifier")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("model revision must be a full lowercase commit")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if (
            self.donor_step != SWITCHED_ANSWER_DONOR_STEP
            or self.recipient_step != SWITCHED_ANSWER_RECIPIENT_STEP
        ):
            raise ValueError("switched-answer direction is frozen to step 1500 into step 0")


@beartype
@dataclass(frozen=True)
class SwitchedAnswerTaskSpec:
    function_id: str
    correct_choice_index: int
    destination_choice_index: int
    interface: str

    def __post_init__(self) -> None:
        if self.function_id != SWITCHED_ANSWER_FUNCTION_ID:
            raise ValueError("the first switched-answer minset run is frozen to add_5 / pyalvt")
        if self.correct_choice_index != SWITCHED_ANSWER_CORRECT_CHOICE_INDEX:
            raise ValueError("the registered pyalvt correct answer is C")
        if not 0 <= self.destination_choice_index < len(ANSWER_LABELS):
            raise ValueError("destination choice must identify A-E")
        if self.destination_choice_index == self.correct_choice_index:
            raise ValueError("destination choice must differ from the correct answer")
        if self.interface not in SWITCHED_ANSWER_INTERFACES:
            raise ValueError("switched-answer minsets support attention_input and resid_post")


@beartype
@dataclass(frozen=True)
class SwitchedAnswerDensityConfig:
    density_grid: tuple[SweepDensity, ...]
    masks_per_density: int
    flat_probability_span: float
    flat_logit_diff_span: float
    minimum_logit_diff_variance: float
    seed: int

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.density_grid)
        if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("density grid must include exact endpoints and an interior point")
        if tuple(sorted(set(values))) != values:
            raise ValueError("density grid must be strictly increasing and unique")
        if self.masks_per_density < 2:
            raise ValueError("interior density points require repeated masks")
        thresholds = (
            self.flat_probability_span,
            self.flat_logit_diff_span,
            self.minimum_logit_diff_variance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("flatness thresholds must be finite and non-negative")
        if self.seed < 0:
            raise ValueError("density seed must be non-negative")


@beartype
@dataclass(frozen=True)
class SwitchedAnswerSearchConfig:
    maximum_order: int
    shard_size: int
    absolute_probability_tolerance: float
    proper_subset_probability_fraction: float

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_order <= 6:
            raise ValueError("registered exhaustive search order must lie in [1, 6]")
        if self.shard_size <= 0:
            raise ValueError("search shard size must be positive")
        if not 0.0 < self.absolute_probability_tolerance < 1.0:
            raise ValueError("probability tolerance must lie inside (0, 1)")
        if not 0.0 < self.proper_subset_probability_fraction < 1.0:
            raise ValueError("proper-subset fraction must lie inside (0, 1)")


@beartype
@dataclass(frozen=True)
class SwitchedAnswerMinsetConfig:
    model: SwitchedAnswerCheckpointSpec
    task: SwitchedAnswerTaskSpec
    layer_count: int
    density: SwitchedAnswerDensityConfig
    search: SwitchedAnswerSearchConfig
    artifact_root: Path

    def __post_init__(self) -> None:
        if self.layer_count != 32:
            raise ValueError("the pinned OLMo3 model must expose exactly 32 decoder layers")
        if not self.artifact_root.is_absolute() or not self.artifact_root.name:
            raise ValueError("artifact identity root must be an absolute concrete directory")


@beartype
@dataclass(frozen=True)
class SwapSubsetMetric:
    sites: LayerSwapSiteSet
    candidate_logits: tuple[float, float, float, float, float]
    destination_probability: float
    raw_logit_diff: float
    destination_argmax: bool

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("swap subset sites must be sorted and unique")
        if any(not math.isfinite(value) for value in self.candidate_logits):
            raise ValueError("candidate logits must be finite")
        if not 0.0 <= self.destination_probability <= 1.0:
            raise ValueError("destination probability must lie in [0, 1]")
        if not math.isfinite(self.raw_logit_diff):
            raise ValueError("raw destination logit difference must be finite")


@beartype
@dataclass(frozen=True)
class VerifiedSwapMinset:
    sites: LayerSwapSiteSet
    destination_probability: float
    raw_logit_diff: float
    sufficiency_margin: float
    maximum_proper_subset_probability: float
    maximum_proper_subset: LayerSwapSiteSet

    def __post_init__(self) -> None:
        if not self.sites or tuple(sorted(set(self.sites))) != self.sites:
            raise ValueError("verified swap minsets must be non-empty, sorted, and unique")
        if not 0.0 <= self.maximum_proper_subset_probability <= 1.0:
            raise ValueError("maximum proper-subset probability must lie in [0, 1]")
        if not 0.0 <= self.destination_probability <= 1.0:
            raise ValueError("destination probability must lie in [0, 1]")
        if not set(self.maximum_proper_subset).issubset(self.sites):
            raise ValueError("maximum proper subset must be contained in the minset")
        if self.maximum_proper_subset == self.sites:
            raise ValueError("maximum proper subset must exclude the full support")
        if not math.isfinite(self.raw_logit_diff) or self.sufficiency_margin < 0.0:
            raise ValueError("verified minset metrics must be finite with non-negative margin")


@jaxtyped(typechecker=beartype)
def as_layer_patch_mask(mask: LayerMask, layer_count: int) -> LayerPatchMask:
    if mask.shape != (layer_count,):
        raise ValueError("layer patch mask does not match the configured decoder depth")
    if not mask.is_contiguous():
        raise ValueError("layer patch masks must be contiguous")
    return cast(LayerPatchMask, mask)


@jaxtyped(typechecker=beartype)
def sample_layer_patch_masks(
    sample_count: int,
    layer_count: int,
    density: SweepDensity,
    generator: t.Generator,
) -> LayerMaskBatch:
    if sample_count <= 0 or layer_count <= 0:
        raise ValueError("mask sample and layer counts must be positive")
    probability = float(density)
    if probability == 0.0:
        return t.zeros((sample_count, layer_count), dtype=t.bool)
    if probability == 1.0:
        return t.ones((sample_count, layer_count), dtype=t.bool)
    return t.rand((sample_count, layer_count), generator=generator, dtype=t.float64) < probability


@beartype
def layer_supports(layer_count: int, order: int) -> tuple[LayerSwapSiteSet, ...]:
    if layer_count <= 0 or not 0 <= order <= layer_count:
        raise ValueError("support enumeration requires a valid layer count and order")
    return tuple(
        tuple(LayerSwapSite(layer) for layer in support)
        for support in itertools.combinations(range(layer_count), order)
    )


@jaxtyped(typechecker=beartype)
def masks_for_layer_supports(
    supports: tuple[LayerSwapSiteSet, ...],
    layer_count: int,
) -> LayerMaskBatch:
    if not supports:
        raise ValueError("mask construction requires at least one support")
    masks = t.zeros((len(supports), layer_count), dtype=t.bool)
    for row, support in enumerate(supports):
        if tuple(sorted(set(support))) != support:
            raise ValueError("supports must be sorted and unique")
        for site in support:
            if site.layer >= layer_count:
                raise ValueError("support site lies outside the decoder")
            masks[row, site.layer] = True
    return masks


@beartype
def proper_subsets(sites: LayerSwapSiteSet) -> tuple[LayerSwapSiteSet, ...]:
    if not sites or tuple(sorted(set(sites))) != sites:
        raise ValueError("proper-subset enumeration requires a non-empty canonical support")
    return tuple(
        tuple(subset)
        for size in range(len(sites))
        for subset in itertools.combinations(sites, size)
    )


@beartype
def verified_minsets_from_metrics(
    metrics: dict[LayerSwapSiteSet, SwapSubsetMetric],
    all_clean_probability: float,
    absolute_probability_tolerance: float,
    proper_subset_probability_fraction: float,
) -> tuple[VerifiedSwapMinset, ...]:
    if () not in metrics:
        raise ValueError("minset verification requires the empty all-dirty metric")
    if not 0.0 <= all_clean_probability <= 1.0:
        raise ValueError("all-clean probability must lie in [0, 1]")
    if not 0.0 < absolute_probability_tolerance < 1.0:
        raise ValueError("absolute probability tolerance must lie in (0, 1)")
    if not 0.0 < proper_subset_probability_fraction < 1.0:
        raise ValueError("proper-subset fraction must lie in (0, 1)")
    threshold = all_clean_probability - absolute_probability_tolerance
    if threshold <= metrics[()].destination_probability:
        raise ValueError("clean-minus-tolerance threshold must exceed the dirty endpoint")
    verified: list[VerifiedSwapMinset] = []
    for sites, full in sorted(metrics.items(), key=lambda item: (len(item[0]), item[0])):
        if not sites or not full.destination_argmax or full.destination_probability < threshold:
            continue
        subsets = proper_subsets(sites)
        missing = tuple(subset for subset in subsets if subset not in metrics)
        if missing:
            continue
        maximum_subset = max(
            (metrics[subset] for subset in subsets),
            key=lambda metric: (metric.destination_probability, metric.sites),
        )
        if maximum_subset.destination_probability > (
            proper_subset_probability_fraction * full.destination_probability
        ):
            continue
        verified.append(
            VerifiedSwapMinset(
                sites=sites,
                destination_probability=full.destination_probability,
                raw_logit_diff=full.raw_logit_diff,
                sufficiency_margin=full.destination_probability - threshold,
                maximum_proper_subset_probability=maximum_subset.destination_probability,
                maximum_proper_subset=maximum_subset.sites,
            )
        )
    return tuple(verified)


@beartype
def support_is_safely_blocked(
    sites: LayerSwapSiteSet,
    metrics: dict[LayerSwapSiteSet, SwapSubsetMetric],
    proper_subset_probability_fraction: float,
) -> bool:
    """Apply only the globally sound blocker implied by P(full) <= 1."""

    if not sites or tuple(sorted(set(sites))) != sites:
        raise ValueError("blocking requires a non-empty canonical support")
    if not 0.0 < proper_subset_probability_fraction < 1.0:
        raise ValueError("proper-subset fraction must lie in (0, 1)")
    return any(
        subset in metrics
        and metrics[subset].destination_probability > proper_subset_probability_fraction
        for subset in proper_subsets(sites)
    )


__all__ = [
    "LayerPatchMask",
    "LayerSwapSite",
    "LayerSwapSiteSet",
    "NonEmptyLayerSwapSiteSets",
    "SWITCHED_ANSWER_CORRECT_CHOICE_INDEX",
    "SWITCHED_ANSWER_DONOR_STEP",
    "SWITCHED_ANSWER_FUNCTION_ID",
    "SWITCHED_ANSWER_INTERFACES",
    "SWITCHED_ANSWER_RECIPIENT_STEP",
    "SWITCHED_ANSWER_SCHEMA_VERSION",
    "SwapSubsetMetric",
    "SwitchedAnswerCheckpointSpec",
    "SwitchedAnswerDensityConfig",
    "SwitchedAnswerMinsetConfig",
    "SwitchedAnswerSearchConfig",
    "SwitchedAnswerTaskSpec",
    "VerifiedSwapMinset",
    "as_layer_patch_mask",
    "layer_supports",
    "masks_for_layer_supports",
    "proper_subsets",
    "sample_layer_patch_masks",
    "support_is_safely_blocked",
    "verified_minsets_from_metrics",
]
