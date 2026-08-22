from __future__ import annotations

import json
from pathlib import Path

import pytest

from oocr_training_dynamics.artifacts import sha256_file
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.fourier_subset_index import SubsetMetric
from oocr_training_dynamics.runtime_fourier_frontier import (
    FRONTIER_RESULT_FILENAME,
    _completed_prior_frontiers,
    _register_new_metrics,
    _verified_support_inventory,
    _write_frontier_metric_index,
    load_frontier_metric_index,
)


def _metric(probability: float = 0.9) -> SubsetMetric:
    return SubsetMetric(
        (Site(0, 0), Site(1, 1)),
        probability,
        2.0,
        True,
        ("network_component_completion_size_2",),
    )


def _manifest(output_dir: Path) -> tuple[dict[str, object], ...]:
    phase = output_dir / "network_size_2"
    phase.mkdir(parents=True)
    metadata = phase / "shard_00000.json"
    sidecar = phase / "shard_00000.pt"
    metadata.write_text(json.dumps({"kind": "test"}))
    sidecar.write_bytes(b"exact tensor sidecar bytes")
    return (
        {
            "phase": phase.name,
            "shard_count": 1,
            "proposal_count": 1,
            "shards": [
                {
                    "metadata": metadata.name,
                    "metadata_sha256": sha256_file(metadata),
                    "sidecar": sidecar.name,
                    "sidecar_sha256": sha256_file(sidecar),
                    "proposal_count": 1,
                    "proposal_sha256": "proposal",
                }
            ],
        },
    )


def test_frontier_metric_index_round_trips_and_validates_sources(tmp_path: Path) -> None:
    manifests = _manifest(tmp_path)

    path = _write_frontier_metric_index(tmp_path, (_metric(),), manifests)
    loaded = load_frontier_metric_index(tmp_path)

    assert loaded == {_metric().sites: _metric()}
    payload = json.loads(path.read_text())
    assert payload["support_count"] == 1
    assert len(payload["source_artifacts"]) == 2


def test_frontier_metric_index_fails_when_a_source_changes(tmp_path: Path) -> None:
    manifests = _manifest(tmp_path)
    _write_frontier_metric_index(tmp_path, (_metric(),), manifests)
    (tmp_path / "network_size_2/shard_00000.pt").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_frontier_metric_index(tmp_path)


def test_frontier_metric_registration_rejects_repeated_supports() -> None:
    metric = _metric()
    metrics = {metric.sites: metric}

    with pytest.raises(RuntimeError, match="already measured"):
        _register_new_metrics(metrics, (metric,))
    with pytest.raises(RuntimeError, match="disagrees"):
        _register_new_metrics(metrics, (_metric(0.8),))


def test_completed_prior_frontiers_import_metrics_and_verified_supports(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "frontier_search_config_prior"
    prior.mkdir()
    index_path = _write_frontier_metric_index(prior, (_metric(),), _manifest(prior))
    result = {
        "schema_version": 1,
        "status": "complete",
        "metric_index": index_path.name,
        "metric_index_sha256": sha256_file(index_path),
        "new_verified_relative_minsets": [
            {
                "sites": [
                    {"token_index": site.token_index, "layer": site.layer}
                    for site in _metric().sites
                ]
            }
        ],
    }
    (prior / FRONTIER_RESULT_FILENAME).write_text(json.dumps(result))

    metrics, verified, sources = _completed_prior_frontiers(
        tmp_path,
        tmp_path / "frontier_search_config_current",
    )

    assert metrics == {_metric().sites: _metric()}
    assert verified == (_metric().sites,)
    assert sources == [
        {
            "directory": prior.name,
            "result_sha256": sha256_file(prior / FRONTIER_RESULT_FILENAME),
            "metric_index": index_path.name,
            "metric_index_sha256": sha256_file(index_path),
            "support_count": 1,
        }
    ]


def test_completed_prior_frontiers_excludes_current_output_directory(
    tmp_path: Path,
) -> None:
    current = tmp_path / "frontier_search_config_current"
    current.mkdir()
    (current / FRONTIER_RESULT_FILENAME).write_text("not valid JSON")

    assert _completed_prior_frontiers(tmp_path, current) == ({}, (), [])


def test_verified_inventory_accepts_an_empty_fourier_terminal_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "stage_2_minsets.json").write_text(
        json.dumps(
            {
                "status": "no_verified_multisite_minsets",
                "verified_multisite_minsets": [],
            }
        )
    )

    assert _verified_support_inventory(tmp_path) == ()


def test_verified_inventory_rejects_minsets_under_empty_terminal_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "stage_2_minsets.json").write_text(
        json.dumps(
            {
                "status": "no_verified_multisite_minsets",
                "verified_multisite_minsets": [
                    {
                        "sites": [
                            {"token_index": 0, "layer": 0},
                            {"token_index": 1, "layer": 0},
                        ]
                    }
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="empty Stage-2"):
        _verified_support_inventory(tmp_path)
