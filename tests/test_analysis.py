from __future__ import annotations

import math

import pytest

from oocr_training_dynamics.analysis import (
    BootstrapEstimate,
    cluster_bootstrap_mean,
    first_sustained_recovery_step,
    frozen_adjusted_auc,
)


def test_cluster_bootstrap_is_reproducible_and_contains_mean() -> None:
    first = cluster_bootstrap_mean((0.1, 0.2, 0.4, 0.7), seed=4, resamples=2_000)
    second = cluster_bootstrap_mean((0.1, 0.2, 0.4, 0.7), seed=4, resamples=2_000)
    assert first == second
    assert first.mean == pytest.approx(0.35)
    assert first.lower_95 <= first.mean <= first.upper_95
    assert first.clusters == 4


def test_bootstrap_rejects_empty_nonfinite_or_bad_counts() -> None:
    with pytest.raises(ValueError, match="needs values"):
        cluster_bootstrap_mean((), seed=0)
    with pytest.raises(ValueError, match="finite"):
        cluster_bootstrap_mean((math.nan,), seed=0)
    with pytest.raises(ValueError, match="needs values"):
        cluster_bootstrap_mean((1.0,), seed=-1)
    with pytest.raises(ValueError, match="counts"):
        BootstrapEstimate(0.0, 0.0, 0.0, 0, 1)
    with pytest.raises(ValueError, match="contain"):
        BootstrapEstimate(2.0, 0.0, 1.0, 1, 1)


def test_frozen_adjusted_auc_removes_constant_floor() -> None:
    examples = (0, 64, 128)
    assert frozen_adjusted_auc(examples, (0.2, 0.2, 0.2)) == pytest.approx(0.0)
    assert frozen_adjusted_auc(examples, (0.2, 0.4, 0.6)) > 0.0
    with pytest.raises(ValueError, match="must not be empty"):
        frozen_adjusted_auc((), ())


def test_first_sustained_recovery_uses_three_scheduled_points() -> None:
    steps = (0, 1, 2, 4, 8)
    assert first_sustained_recovery_step(steps, (0.2, 0.31, 0.32, 0.33, 0.1)) == 1
    assert first_sustained_recovery_step(steps, (0.2, 0.31, 0.1, 0.35, 0.36)) is None
    with pytest.raises(ValueError, match="equally sized"):
        first_sustained_recovery_step((0,), ())
    with pytest.raises(ValueError, match="invalid"):
        first_sustained_recovery_step((0,), (0.2,), consecutive=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        first_sustained_recovery_step((0, 0), (0.2, 0.3))
