from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t

import oocr_training_dynamics.fourier_subset_index as subset_index_module
from oocr_training_dynamics.artifacts import sha256_file
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.fourier_subset_index import (
    MAX_PROPER_SUBSET_PROBABILITY_FRACTION,
    RelativeProperSubsetCriterion,
    SubsetMetric,
    build_subset_metric_index,
    ensure_subset_metric_index,
    load_subset_metric_index,
    maximum_proper_subset_metric,
    passes_relative_proper_subset_criterion,
    refresh_subset_metric_index_after_source_addition,
    subset_index_path,
)


def _write_subset_sources(scope: Path) -> None:
    scope.mkdir(parents=True)
    singleton_sidecar = scope / "exhaustive_singletons.pt"
    singleton_sidecar.write_bytes(b"digest-only singleton sidecar")
    singleton_rows = []
    for token_index in range(2):
        for layer in range(2):
            is_first = (token_index, layer) == (0, 0)
            singleton_rows.append(
                {
                    "site": {"token_index": token_index, "layer": layer},
                    "correct_probability": 0.1 if is_first else 0.2,
                    "raw_logit_diff": -2.0 if is_first else -1.0,
                    "accuracy": False,
                }
            )
    (scope / "exhaustive_singletons.json").write_text(
        json.dumps(
            {
                "singleton_sidecar": singleton_sidecar.name,
                "singleton_sidecar_sha256": sha256_file(singleton_sidecar),
                "sufficiency": {
                    "dirty_correct_probability": 0.01,
                    "dirty_logit_diff": -4.0,
                },
                "singleton_results": singleton_rows,
            }
        )
    )
    masks = t.zeros((2, 2, 2), dtype=t.bool)
    masks[0, 0, 0] = True
    masks[1, 0, 0] = True
    masks[1, 1, 1] = True
    stage_sidecar = scope / "stage_2_verification.pt"
    t.save(
        {
            "masks": masks,
            "candidate_logits": t.tensor(
                [[0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]],
                dtype=t.float32,
            ),
            "logit_diffs": t.tensor([-2.0, 2.5], dtype=t.float32),
            "correct_probabilities": t.tensor([0.1, 0.91], dtype=t.float32),
            "accuracies": t.tensor([0.0, 1.0], dtype=t.float32),
        },
        stage_sidecar,
    )
    (scope / "stage_2_minsets.json").write_text(
        json.dumps(
            {
                "verification_sidecar": stage_sidecar.name,
                "verification_sidecar_sha256": sha256_file(stage_sidecar),
            }
        )
    )


def _write_recall_source(scope: Path) -> None:
    audit_directory = scope / "recall_audit_config_test"
    phase_directory = audit_directory / "initial"
    phase_directory.mkdir(parents=True)
    masks = t.zeros((1, 2, 2), dtype=t.bool)
    masks[0, 0, 1] = True
    masks[0, 1, 0] = True
    sidecar = phase_directory / "shard_00000.pt"
    t.save(
        {
            "masks": masks,
            "candidate_logits": t.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
            "logit_diffs": t.tensor([1.0]),
            "correct_probabilities": t.tensor([0.72]),
            "accuracies": t.tensor([1.0]),
        },
        sidecar,
    )
    metadata = phase_directory / "shard_00000.json"
    metadata.write_text(json.dumps({"sidecar": sidecar.name}))
    (audit_directory / "recall_audit.json").write_text(
        json.dumps(
            {
                "phase_manifests": [
                    {
                        "phase": "initial",
                        "shards": [
                            {
                                "metadata": metadata.name,
                                "sidecar": sidecar.name,
                                "sidecar_sha256": sha256_file(sidecar),
                            }
                        ],
                    },
                    {"phase": "triple_children", "shards": []},
                    {"phase": "triples", "shards": []},
                ]
            }
        )
    )


def test_subset_index_builds_once_and_reloads_exact_mapping(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)

    built = ensure_subset_metric_index(scope)
    loaded = ensure_subset_metric_index(scope)

    assert loaded == built
    assert len(loaded) == 6
    assert loaded[()].correct_probability == 0.01
    pair = (Site(0, 0), Site(1, 1))
    assert loaded[pair].correct_probability == pytest.approx(0.91)
    assert loaded[(Site(0, 0),)].sources == (
        "exhaustive_singletons",
        "fourier_stage_2",
    )
    payload = json.loads(subset_index_path(scope).read_text())
    assert payload["support_count"] == 6
    assert len(payload["source_artifacts"]) == 4


def test_subset_index_fails_loudly_when_a_source_changes(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    build_subset_metric_index(scope)
    singleton_path = scope / "exhaustive_singletons.json"
    payload = json.loads(singleton_path.read_text())
    payload["changed"] = True
    singleton_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="stale"):
        load_subset_metric_index(scope)


def test_subset_index_includes_digest_validated_recall_shards(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    _write_recall_source(scope)

    metrics = build_subset_metric_index(scope)

    support = (Site(0, 1), Site(1, 0))
    assert metrics[support].correct_probability == pytest.approx(0.72)
    assert metrics[support].sources == ("recall_initial",)
    payload = json.loads(subset_index_path(scope).read_text())
    assert payload["support_count"] == 7
    assert len(payload["source_artifacts"]) == 7


def test_subset_index_refreshes_only_after_monotonic_source_addition(
    tmp_path: Path,
) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    before = build_subset_metric_index(scope)
    _write_recall_source(scope)

    after = refresh_subset_metric_index_after_source_addition(scope)

    assert len(before) == 6
    assert len(after) == 7
    assert (Site(0, 1), Site(1, 0)) in after
    assert refresh_subset_metric_index_after_source_addition(scope) == after


def test_subset_index_refresh_rejects_changed_prior_source(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    build_subset_metric_index(scope)
    (scope / "exhaustive_singletons.pt").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="previously indexed"):
        refresh_subset_metric_index_after_source_addition(scope)


def test_maximum_proper_subset_uses_probability_not_binary_accuracy(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    metrics = build_subset_metric_index(scope)
    pair = (Site(0, 0), Site(1, 1))

    maximum = maximum_proper_subset_metric(pair, metrics)

    assert maximum.sites == (Site(1, 1),)
    assert maximum.correct_probability == pytest.approx(0.2)
    criterion = RelativeProperSubsetCriterion(MAX_PROPER_SUBSET_PROBABILITY_FRACTION)
    assert passes_relative_proper_subset_criterion(metrics[pair], maximum, criterion)


def test_relative_subset_criterion_uses_the_full_support_probability() -> None:
    a, b = Site(0, 0), Site(1, 0)
    full = SubsetMetric((a, b), 0.90, 2.0, True, ("full",))
    passing = SubsetMetric((a,), 0.72, 0.0, True, ("child",))
    failing = SubsetMetric((b,), 0.721, 0.0, True, ("child",))
    criterion = RelativeProperSubsetCriterion(0.80)

    assert passes_relative_proper_subset_criterion(full, passing, criterion)
    assert not passes_relative_proper_subset_criterion(full, failing, criterion)
    assert criterion.maximum_allowed_probability(0.90) == pytest.approx(0.72)


def test_maximum_proper_subset_rejects_missing_children(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    metrics = build_subset_metric_index(scope)
    del metrics[(Site(1, 1),)]

    with pytest.raises(RuntimeError, match="incomplete"):
        maximum_proper_subset_metric((Site(0, 0), Site(1, 1)), metrics)


def test_subset_metric_value_object_rejects_illegal_states() -> None:
    a, b = Site(0, 0), Site(1, 0)

    with pytest.raises(ValueError, match="sorted and unique"):
        SubsetMetric((b, a), 0.5, 0.0, True, ("source",))
    with pytest.raises(ValueError, match="probability"):
        SubsetMetric((a,), 1.1, 0.0, True, ("source",))
    with pytest.raises(ValueError, match="finite"):
        SubsetMetric((a,), 0.5, float("nan"), True, ("source",))
    with pytest.raises(ValueError, match="sources"):
        SubsetMetric((a,), 0.5, 0.0, True, ())
    with pytest.raises(ValueError, match="canonical multi-site"):
        maximum_proper_subset_metric((a,), {})


def test_tensor_sidecar_metrics_reject_invalid_scientific_values() -> None:
    masks = t.zeros((1, 1, 1), dtype=t.bool)
    logits = t.zeros((1, 5), dtype=t.float32)
    vector = t.zeros((1,), dtype=t.float32)

    with pytest.raises(TypeError, match="candidate-logit shape"):
        subset_index_module._metrics_from_tensors(
            masks,
            t.zeros((1, 4)),
            vector,
            vector,
            vector,
            "test",
        )
    with pytest.raises(ValueError, match="non-finite"):
        subset_index_module._metrics_from_tensors(
            masks,
            logits,
            t.tensor([float("nan")]),
            vector,
            vector,
            "test",
        )
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        subset_index_module._metrics_from_tensors(
            masks,
            logits,
            vector,
            t.tensor([1.1]),
            vector,
            "test",
        )
    with pytest.raises(ValueError, match="non-boolean accuracy"):
        subset_index_module._metrics_from_tensors(
            masks,
            logits,
            vector,
            vector,
            t.tensor([0.5]),
            "test",
        )


def test_subset_index_rejects_missing_tensor_fields(tmp_path: Path) -> None:
    sidecar = tmp_path / "broken.pt"
    t.save({"masks": t.zeros((1, 1, 1), dtype=t.bool)}, sidecar)

    with pytest.raises(TypeError, match="missing tensors"):
        subset_index_module._load_sidecar_metrics(sidecar, source="test")


def test_subset_index_rejects_missing_sources_and_duplicate_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="requires"):
        build_subset_metric_index(tmp_path / "missing")
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    build_subset_metric_index(scope)

    with pytest.raises(FileExistsError, match="already exists"):
        build_subset_metric_index(scope)


def test_subset_index_private_validators_reject_malformed_values() -> None:
    site = {"token_index": 0, "layer": 0}

    with pytest.raises(TypeError, match="string-keyed"):
        subset_index_module._mapping([], context="test")
    with pytest.raises(TypeError, match="site list"):
        subset_index_module._site_set(None, context="test")
    with pytest.raises(TypeError, match="invalid site"):
        subset_index_module._site_set([{"token_index": "0", "layer": 0}], context="test")
    with pytest.raises(ValueError, match="repeats a site"):
        subset_index_module._site_set([site, site], context="test")
    metrics: dict[tuple[Site, ...], SubsetMetric] = {
        (Site(0, 0),): SubsetMetric((Site(0, 0),), 0.1, -2.0, False, ("one",))
    }
    with pytest.raises(RuntimeError, match="disagree"):
        subset_index_module._register_metric(
            metrics,
            SubsetMetric((Site(0, 0),), 0.2, -1.0, False, ("two",)),
        )


def test_subset_source_inventory_rejects_missing_fields_and_changed_digest(
    tmp_path: Path,
) -> None:
    missing_field_scope = tmp_path / "missing_field"
    _write_subset_sources(missing_field_scope)
    singleton_path = missing_field_scope / "exhaustive_singletons.json"
    singleton = json.loads(singleton_path.read_text())
    del singleton["singleton_sidecar"]
    singleton_path.write_text(json.dumps(singleton))
    with pytest.raises(TypeError, match="lacks singleton_sidecar"):
        subset_index_module._source_artifacts(missing_field_scope)

    changed_scope = tmp_path / "changed"
    _write_subset_sources(changed_scope)
    (changed_scope / "exhaustive_singletons.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        subset_index_module._source_artifacts(changed_scope)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=999), "wrong schema"),
        (lambda payload: payload.update(source_artifacts=None), "source provenance"),
        (lambda payload: payload["source_artifacts"][0].update(path=None), "malformed source"),
        (lambda payload: payload.update(support_count=999), "row count"),
        (lambda payload: payload["rows"][0].update(correct_probability=None), "malformed row"),
        (
            lambda payload: (
                payload["rows"].append(dict(payload["rows"][0])),
                payload.update(support_count=payload["support_count"] + 1),
            ),
            "repeats a support",
        ),
    ],
)
def test_subset_index_rejects_corrupt_cached_payloads(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    scope = tmp_path / "scope"
    _write_subset_sources(scope)
    build_subset_metric_index(scope)
    path = subset_index_path(scope)
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises((TypeError, RuntimeError), match=message):
        load_subset_metric_index(scope)
