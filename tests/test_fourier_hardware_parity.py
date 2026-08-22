from __future__ import annotations

import pytest
import torch as t

from oocr_training_dynamics.fourier_hardware_parity import (
    HardwareParityTolerances,
    compare_hardware_metrics,
    select_parity_indices,
)


def test_parity_selection_is_unique_deterministic_and_threshold_sensitive() -> None:
    probabilities = t.tensor(
        [0.0, 0.1, 0.25, 0.49, 0.5001, 0.7, 0.9, 1.0],
        dtype=t.float32,
    )

    first = select_parity_indices(probabilities, 0.5, 6)
    second = select_parity_indices(probabilities, 0.5, 6)

    assert t.equal(first, second)
    assert t.unique(first).numel() == 6
    assert set(first[:3].tolist()) == {3, 4, 5}


def test_hardware_parity_requires_numerical_and_decision_equivalence() -> None:
    logits = t.tensor([[0.0, 1.0, 4.0, 2.0, -1.0], [3.0, 2.0, 1.0, 0.0, -1.0]])
    diffs = t.tensor([1.0, -1.0])
    probabilities = t.tensor([0.95, 0.2])
    accuracies = t.tensor([1.0, 0.0])
    tolerances = HardwareParityTolerances(0.002, 0.004, 0.0001)

    exact = compare_hardware_metrics(
        logits,
        diffs,
        probabilities,
        accuracies,
        logits.clone(),
        diffs.clone(),
        probabilities.clone(),
        accuracies.clone(),
        2,
        0.9,
        tolerances,
    )
    assert exact["status"] == "passed"
    assert exact["sufficiency_exact"] is True

    shifted = compare_hardware_metrics(
        logits,
        diffs,
        probabilities,
        accuracies,
        logits + 0.01,
        diffs,
        probabilities,
        accuracies,
        2,
        0.9,
        tolerances,
    )
    assert shifted["status"] == "failed"
    assert shifted["argmax_exact"] is True


def test_parity_selection_and_comparison_reject_illegal_shapes() -> None:
    with pytest.raises(ValueError, match="selected parity count"):
        select_parity_indices(t.tensor([0.1, 0.2]), 0.5, 3)
    with pytest.raises(ValueError, match="shape"):
        compare_hardware_metrics(
            t.zeros((2, 4)),
            t.zeros(2),
            t.zeros(2),
            t.zeros(2),
            t.zeros((2, 4)),
            t.zeros(2),
            t.zeros(2),
            t.zeros(2),
            0,
            0.5,
            HardwareParityTolerances(0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_hardware_parity_tolerances_reject_negative_or_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        HardwareParityTolerances(value, 0.0, 0.0)


@pytest.mark.parametrize(
    "probabilities",
    (
        t.tensor([], dtype=t.float32),
        t.tensor([0.1, float("nan")]),
        t.tensor([-0.1, 0.2]),
        t.tensor([0.1, 1.1]),
    ),
)
def test_parity_selection_rejects_invalid_probability_vectors(
    probabilities: t.Tensor,
) -> None:
    with pytest.raises(ValueError, match="finite non-empty vector"):
        select_parity_indices(probabilities, 0.5, 1)


@pytest.mark.parametrize("threshold", (0.0, 1.0, float("nan"), float("inf")))
def test_parity_selection_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        select_parity_indices(t.tensor([0.1, 0.9]), threshold, 1)


def test_parity_selection_handles_single_nearest_mask_without_range_sample() -> None:
    probabilities = t.tensor([0.1, 0.49, 0.9])

    selected = select_parity_indices(probabilities, 0.5, 1)

    assert selected.tolist() == [1]


def _valid_parity_inputs() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    return (
        t.tensor([[0.0, 1.0, 4.0, 2.0, -1.0], [3.0, 2.0, 1.0, 0.0, -1.0]]),
        t.tensor([1.0, -1.0]),
        t.tensor([0.95, 0.2]),
        t.tensor([1.0, 0.0]),
    )


@pytest.mark.parametrize("correct_choice_index", (-1, 5))
def test_hardware_parity_rejects_invalid_correct_choice(correct_choice_index: int) -> None:
    logits, diffs, probabilities, accuracies = _valid_parity_inputs()

    with pytest.raises(ValueError, match="A-E"):
        compare_hardware_metrics(
            logits,
            diffs,
            probabilities,
            accuracies,
            logits.clone(),
            diffs.clone(),
            probabilities.clone(),
            accuracies.clone(),
            correct_choice_index,
            0.9,
            HardwareParityTolerances(0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize("threshold", (0.0, 1.0, float("nan")))
def test_hardware_parity_rejects_invalid_probability_threshold(threshold: float) -> None:
    logits, diffs, probabilities, accuracies = _valid_parity_inputs()

    with pytest.raises(ValueError, match="strictly inside"):
        compare_hardware_metrics(
            logits,
            diffs,
            probabilities,
            accuracies,
            logits.clone(),
            diffs.clone(),
            probabilities.clone(),
            accuracies.clone(),
            2,
            threshold,
            HardwareParityTolerances(0.0, 0.0, 0.0),
        )


def test_hardware_parity_rejects_nonfinite_metric_vectors() -> None:
    logits, diffs, probabilities, accuracies = _valid_parity_inputs()
    observed_diffs = diffs.clone()
    observed_diffs[0] = float("nan")

    with pytest.raises(ValueError, match="metrics must be finite"):
        compare_hardware_metrics(
            logits,
            diffs,
            probabilities,
            accuracies,
            logits.clone(),
            observed_diffs,
            probabilities.clone(),
            accuracies.clone(),
            2,
            0.9,
            HardwareParityTolerances(0.0, 0.0, 0.0),
        )


def test_hardware_parity_reports_argmax_accuracy_and_sufficiency_mismatches() -> None:
    logits, diffs, probabilities, accuracies = _valid_parity_inputs()
    observed_logits = logits.clone()
    observed_logits[0] = t.tensor([5.0, 1.0, 4.0, 2.0, -1.0])
    observed_probabilities = probabilities.clone()
    observed_probabilities[0] = 0.8
    observed_accuracies = accuracies.clone()
    observed_accuracies[0] = 0.0

    result = compare_hardware_metrics(
        logits,
        diffs,
        probabilities,
        accuracies,
        observed_logits,
        diffs.clone(),
        observed_probabilities,
        observed_accuracies,
        2,
        0.9,
        HardwareParityTolerances(10.0, 0.0, 1.0),
    )

    assert result["status"] == "failed"
    assert result["accuracy_exact"] is False
    assert result["argmax_exact"] is False
    assert result["sufficiency_exact"] is False
