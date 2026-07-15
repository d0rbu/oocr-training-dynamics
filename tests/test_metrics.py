from __future__ import annotations

import math

import numpy as np
import pytest

from oocr_training_dynamics.metrics import (
    chance_adjusted_score,
    log_examples_auc,
    normalized_patch_effect,
    softmax,
)


def test_softmax_is_stable_and_normalized() -> None:
    probabilities = softmax(np.array([1_001.0, 1_000.0, 999.0], dtype=np.float64))
    assert float(np.sum(probabilities)) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_normalized_patch_effect_uses_recipient_to_source_axis() -> None:
    assert normalized_patch_effect(0.5, 0.2, 0.8) == pytest.approx(0.5)
    assert math.isnan(normalized_patch_effect(0.3, 0.4, 0.4))


def test_chance_adjustment_maps_five_choice_chance_and_perfection() -> None:
    assert chance_adjusted_score(0.2) == pytest.approx(0.0)
    assert chance_adjusted_score(1.0) == pytest.approx(1.0)


def test_log_examples_auc_of_constant_curve_is_constant() -> None:
    assert log_examples_auc((0, 64, 4_096), (0.4, 0.4, 0.4)) == pytest.approx(0.4)


def test_metrics_reject_invalid_numeric_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        softmax(np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        softmax(np.array([0.0, np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        normalized_patch_effect(math.inf, 0.0, 1.0)
    with pytest.raises(ValueError, match="denominator"):
        normalized_patch_effect(0.0, 0.0, 1.0, minimum_denominator=0.0)
    with pytest.raises(ValueError, match="probability"):
        chance_adjusted_score(1.1)
    with pytest.raises(ValueError, match="choice count"):
        chance_adjusted_score(0.5, 1)


@pytest.mark.parametrize(
    ("examples", "values"),
    [
        ((0,), (0.2,)),
        ((0, 1), (0.2,)),
        ((-1, 1), (0.2, 0.3)),
        ((0, 2, 1), (0.2, 0.3, 0.4)),
        ((0, 1), (0.2, math.nan)),
    ],
)
def test_auc_rejects_invalid_curves(examples: tuple[int, ...], values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        log_examples_auc(examples, values)
