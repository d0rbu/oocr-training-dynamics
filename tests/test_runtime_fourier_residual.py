from __future__ import annotations

from pathlib import Path

import pytest

from oocr_training_dynamics.artifacts import sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.runtime_fourier_residual import (
    NETWORK_VETO_SCHEMA_VERSION,
    NetworkVetoDensityConfig,
    NetworkVetoDensityPlan,
    _validated_network_veto_result,
    _write_or_validate,
)


@pytest.mark.parametrize(
    "fraction,minimum_sites,message",
    [
        (0.0, 38, "fraction"),
        (1.0, 38, "fraction"),
        (0.8, 0, "site-count"),
    ],
)
def test_network_veto_density_config_rejects_illegal_states(
    fraction: float,
    minimum_sites: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NetworkVetoDensityConfig(fraction, minimum_sites)


def test_network_veto_final_artifact_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    _write_or_validate(path, {"status": "flat_stop"})
    _write_or_validate(path, {"status": "flat_stop"})

    with pytest.raises(RuntimeError, match="disagrees"):
        _write_or_validate(path, {"status": "transition_found"})


def _network_veto_result_fixture(
    tmp_path: Path,
) -> tuple[Path, NetworkVetoDensityPlan, NetworkVetoDensityConfig]:
    config = NetworkVetoDensityConfig(0.8, 2)
    network_sites = (Site(0, 0), Site(0, 1))
    singleton_sites = (Site(1, 0),)
    vetoed_sites = (*network_sites, *singleton_sites)
    source: dict[str, object] = {
        "scope_directory": "/frozen/scope",
        "completed_frontiers": [],
    }
    plan = NetworkVetoDensityPlan(
        tmp_path,
        network_sites,
        singleton_sites,
        vetoed_sites,
        source,
    )
    sidecar = tmp_path / "stage_0_network_veto_density_samples.pt"
    sidecar.write_bytes(b"immutable-torch-sidecar")
    density_path = tmp_path / "stage_0_network_veto_density.json"
    curve = [{"density": 0.0, "mean_correct_probability": 0.2}]
    density = {
        "schema_version": 1,
        "stage": 0,
        "status": "transition_found",
        "function_space": "network_vetoed_residual",
        "transition_density": 0.02,
        "curve": curve,
        "vetoed_sites": [
            {"token_index": site.token_index, "layer": site.layer}
            for site in vetoed_sites
        ],
        "active_site_count": 5,
        "sample_sidecar": sidecar.name,
        "sample_sidecar_sha256": sha256_file(sidecar),
    }
    write_json(density_path, density)
    result_path = tmp_path / "network_veto_density.json"
    write_json(
        result_path,
        {
            "schema_version": NETWORK_VETO_SCHEMA_VERSION,
            "status": "transition_found",
            "diagnostic_config": {
                "proper_subset_probability_fraction": 0.8,
                "minimum_network_site_count": 2,
            },
            "source": source,
            "network_site_count": len(network_sites),
            "singleton_site_count": len(singleton_sites),
            "vetoed_site_count": len(vetoed_sites),
            "active_site_count": 5,
            "density_artifact": density_path.name,
            "density_artifact_sha256": sha256_file(density_path),
            "transition_density": 0.02,
            "curve": curve,
            "stop_before_mask_search": False,
        },
    )
    return result_path, plan, config


def test_network_veto_result_validates_all_scientific_provenance(tmp_path: Path) -> None:
    result_path, plan, config = _network_veto_result_fixture(tmp_path)

    result = _validated_network_veto_result(result_path, plan, config)

    assert result["status"] == "transition_found"


def test_network_veto_result_rejects_changed_tensor_sidecar(tmp_path: Path) -> None:
    result_path, plan, config = _network_veto_result_fixture(tmp_path)
    (tmp_path / "stage_0_network_veto_density_samples.pt").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="scientific density artifact"):
        _validated_network_veto_result(result_path, plan, config)
