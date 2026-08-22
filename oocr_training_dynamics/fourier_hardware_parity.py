"""Fixed-mask cross-hardware parity contracts for Fourier causal collection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped

MetricVector = Float[t.Tensor, "samples"]
CandidateLogits = Float[t.Tensor, "samples choices"]
IndexVector = Int64[t.Tensor, "selected"]


@beartype
@dataclass(frozen=True)
class HardwareParityTolerances:
    """Predeclared numerical and decision-equivalence acceptance bounds."""

    maximum_candidate_logit_error: float
    maximum_logit_diff_error: float
    maximum_probability_error: float

    def __post_init__(self) -> None:
        values = (
            self.maximum_candidate_logit_error,
            self.maximum_logit_diff_error,
            self.maximum_probability_error,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("hardware-parity tolerances must be finite and non-negative")


@jaxtyped(typechecker=beartype)
def select_parity_indices(
    correct_probabilities: MetricVector,
    probability_threshold: float,
    selected_count: int,
) -> IndexVector:
    """Select threshold-sensitive and range-spanning masks deterministically."""

    if (
        correct_probabilities.ndim != 1
        or correct_probabilities.numel() == 0
        or not bool(t.isfinite(correct_probabilities).all())
        or not bool(((correct_probabilities >= 0.0) & (correct_probabilities <= 1.0)).all())
    ):
        raise ValueError("reference probabilities must be a finite non-empty vector in [0, 1]")
    if not math.isfinite(probability_threshold) or not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability threshold must be finite and strictly inside (0, 1)")
    if selected_count <= 0 or selected_count > correct_probabilities.numel():
        raise ValueError("selected parity count must be in [1, sample_count]")

    threshold_count = (selected_count + 1) // 2
    nearest = t.argsort(
        (correct_probabilities - probability_threshold).abs(),
        stable=True,
    )[:threshold_count]
    remaining_mask = t.ones(correct_probabilities.numel(), dtype=t.bool)
    remaining_mask[nearest] = False
    remaining = t.arange(correct_probabilities.numel(), dtype=t.int64)[remaining_mask]
    range_count = selected_count - threshold_count
    if range_count == 0:
        selected = nearest
    else:
        probability_order = t.argsort(correct_probabilities[remaining], stable=True)
        ordered_remaining = remaining[probability_order]
        positions = t.arange(range_count, dtype=t.int64) * ordered_remaining.numel() // range_count
        selected = t.cat((nearest, ordered_remaining[positions]))
    if selected.numel() != selected_count or t.unique(selected).numel() != selected_count:
        raise AssertionError("parity selection did not produce the requested unique mask count")
    return selected


@jaxtyped(typechecker=beartype)
def compare_hardware_metrics(
    reference_candidate_logits: CandidateLogits,
    reference_logit_diffs: MetricVector,
    reference_probabilities: MetricVector,
    reference_accuracies: MetricVector,
    observed_candidate_logits: CandidateLogits,
    observed_logit_diffs: MetricVector,
    observed_probabilities: MetricVector,
    observed_accuracies: MetricVector,
    correct_choice_index: int,
    probability_threshold: float,
    tolerances: HardwareParityTolerances,
) -> dict[str, object]:
    """Require bounded numerical drift and exact causal decisions."""

    if (
        reference_candidate_logits.shape != observed_candidate_logits.shape
        or reference_candidate_logits.ndim != 2
        or reference_candidate_logits.shape[1] != 5
    ):
        raise ValueError("reference and observed candidate logits must have shape [samples, 5]")
    sample_count = reference_candidate_logits.shape[0]
    vectors = (
        reference_logit_diffs,
        reference_probabilities,
        reference_accuracies,
        observed_logit_diffs,
        observed_probabilities,
        observed_accuracies,
    )
    if any(vector.shape != (sample_count,) for vector in vectors):
        raise ValueError("all hardware-parity metric vectors must match the sample count")
    if not 0 <= correct_choice_index < 5:
        raise ValueError("correct choice index must identify one of A-E")
    if not math.isfinite(probability_threshold) or not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability threshold must be finite and strictly inside (0, 1)")
    if any(not bool(t.isfinite(tensor).all()) for tensor in vectors):
        raise ValueError("hardware-parity metrics must be finite")

    reference_argmax = reference_candidate_logits.argmax(dim=1)
    observed_argmax = observed_candidate_logits.argmax(dim=1)
    reference_sufficient = (reference_probabilities >= probability_threshold) & reference_argmax.eq(
        correct_choice_index
    )
    observed_sufficient = (observed_probabilities >= probability_threshold) & observed_argmax.eq(
        correct_choice_index
    )
    maximum_candidate_logit_error = float(
        (reference_candidate_logits - observed_candidate_logits).abs().max()
    )
    maximum_logit_diff_error = float((reference_logit_diffs - observed_logit_diffs).abs().max())
    maximum_probability_error = float(
        (reference_probabilities - observed_probabilities).abs().max()
    )
    accuracy_exact = bool(t.equal(reference_accuracies, observed_accuracies))
    argmax_exact = bool(t.equal(reference_argmax, observed_argmax))
    sufficiency_exact = bool(t.equal(reference_sufficient, observed_sufficient))
    passed = bool(
        maximum_candidate_logit_error <= tolerances.maximum_candidate_logit_error
        and maximum_logit_diff_error <= tolerances.maximum_logit_diff_error
        and maximum_probability_error <= tolerances.maximum_probability_error
        and accuracy_exact
        and argmax_exact
        and sufficiency_exact
    )
    return {
        "status": "passed" if passed else "failed",
        "sample_count": sample_count,
        "maximum_candidate_logit_error": maximum_candidate_logit_error,
        "maximum_logit_diff_error": maximum_logit_diff_error,
        "maximum_probability_error": maximum_probability_error,
        "accuracy_exact": accuracy_exact,
        "argmax_exact": argmax_exact,
        "sufficiency_exact": sufficiency_exact,
        "reference_sufficient_count": int(reference_sufficient.sum()),
        "observed_sufficient_count": int(observed_sufficient.sum()),
        "tolerances": {
            "maximum_candidate_logit_error": tolerances.maximum_candidate_logit_error,
            "maximum_logit_diff_error": tolerances.maximum_logit_diff_error,
            "maximum_probability_error": tolerances.maximum_probability_error,
        },
    }


__all__ = [
    "HardwareParityTolerances",
    "compare_hardware_metrics",
    "select_parity_indices",
]
