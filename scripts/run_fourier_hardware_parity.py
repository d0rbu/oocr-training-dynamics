#!/usr/bin/env python3
"""Replay fixed 4090 masks as a fail-closed Engaging hardware parity gate."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch as t

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_hardware_parity import (
    HardwareParityTolerances,
    compare_hardware_metrics,
    select_parity_indices,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_fourier_circuits import (
    _capture_clean_checkpoint,
    _load_checkpoint_model,
    _load_tensor_sidecar,
    _load_tokenizer,
    _release_model,
    _resolve_blocks,
    _write_tensor_sidecar,
    build_circuit_probe,
    build_site_grid,
    evaluate_masks_in_batches,
    fourier_output_dir,
    logical_artifact_path,
)
from oocr_training_dynamics.runtime_fourier_frontier import _full_probability_threshold
from scripts.run_fourier_recall_audit import _circuit_config

MINIMUM_FREE_BYTES = 8 * 2**30
PARITY_SEED = 20_260_821
PARITY_TOLERANCES = HardwareParityTolerances(
    maximum_candidate_logit_error=0.002,
    maximum_logit_diff_error=0.004,
    maximum_probability_error=0.0001,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-id", choices=("add_5", "identity"), required=True)
    parser.add_argument("--clean-step", type=int, required=True)
    parser.add_argument("--reference-sidecar", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path)
    parser.add_argument("--mask-count", type=int, default=64)
    parser.add_argument("--artifact-identity-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _hex_digest(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reference_payload(
    sidecar_path: Path,
    metadata_path: Path | None = None,
) -> tuple[dict[str, t.Tensor], dict[str, object], Path]:
    if not sidecar_path.is_absolute() or sidecar_path.suffix != ".pt":
        raise ValueError("reference sidecar must be an absolute .pt path")
    metadata_path = sidecar_path.with_suffix(".json") if metadata_path is None else metadata_path
    if not metadata_path.is_absolute() or metadata_path.suffix != ".json":
        raise ValueError("reference metadata must be an absolute .json path")
    raw_metadata = read_json(metadata_path)
    if not isinstance(raw_metadata, dict):
        raise TypeError("reference sidecar metadata must be an object")
    metadata = cast(dict[str, object], raw_metadata)
    sidecar_contracts = (
        ("sidecar", "sidecar_sha256"),
        ("verification_sidecar", "verification_sidecar_sha256"),
        ("sample_sidecar", "sample_sidecar_sha256"),
    )
    matching_contracts = tuple(
        (path_key, digest_key)
        for path_key, digest_key in sidecar_contracts
        if metadata.get(path_key) == sidecar_path.name
        and metadata.get(digest_key) == sha256_file(sidecar_path)
    )
    if len(matching_contracts) != 1:
        raise RuntimeError("reference sidecar fails its immutable metadata digest")
    payload = _load_tensor_sidecar(sidecar_path)
    required = {
        "masks",
        "candidate_logits",
        "logit_diffs",
        "correct_probabilities",
        "accuracies",
    }
    if set(payload) != required:
        raise RuntimeError(f"reference sidecar tensor keys changed: {sorted(payload)}")
    mask_count = payload["masks"].shape[0]
    if (
        payload["masks"].ndim != 3
        or payload["masks"].dtype != t.bool
        or payload["candidate_logits"].shape != (mask_count, 5)
        or any(
            payload[key].shape != (mask_count,)
            for key in ("logit_diffs", "correct_probabilities", "accuracies")
        )
    ):
        raise RuntimeError("reference sidecar shapes disagree with its proposal metadata")
    proposal_count = metadata.get("proposal_count")
    if proposal_count is not None and proposal_count != mask_count:
        raise RuntimeError("reference proposal count disagrees with its sidecar")
    return payload, metadata, metadata_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    if shutil.disk_usage(root).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("hardware parity requires at least 8 GiB free in its storage root")
    if not args.artifact_identity_root.is_absolute() or not args.output_dir.is_absolute():
        raise ValueError("artifact identity and parity output roots must be absolute")
    source_bundle_sha256 = _hex_digest(args.source_bundle_sha256, label="source bundle digest")
    output_json = args.output_dir / "hardware_parity.json"
    output_sidecar = args.output_dir / "hardware_parity.pt"
    if output_json.exists() or output_sidecar.exists():
        raise FileExistsError("hardware parity output is immutable; select a new output directory")

    reference, reference_metadata, reference_metadata_path = _reference_payload(
        args.reference_sidecar,
        args.reference_metadata,
    )
    circuit_config = _circuit_config(
        root,
        args.function_id,
        args.clean_step,
        args.artifact_identity_root,
    )
    scope = fourier_output_dir(root, circuit_config)
    probability_threshold = _full_probability_threshold(scope)
    indices = select_parity_indices(
        reference["correct_probabilities"],
        probability_threshold,
        args.mask_count,
    )
    selected = {key: value.index_select(0, indices) for key, value in reference.items()}

    random.seed(PARITY_SEED)
    t.manual_seed(PARITY_SEED)
    t.cuda.manual_seed_all(PARITY_SEED)
    t.use_deterministic_algorithms(True, warn_only=False)
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    grid = build_site_grid(probe, spec, circuit_config.sites)
    if tuple(selected["masks"].shape[1:]) != grid.shape:
        raise RuntimeError("reference mask grid does not match the rendered Engaging prompt")
    clean = _capture_clean_checkpoint(root, circuit_config, probe, spec)
    model = _load_checkpoint_model(root, circuit_config, circuit_config.model.dirty_step)
    try:
        blocks = _resolve_blocks(model, spec)
        evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            selected["masks"][:1],
            1,
            with_gradients=False,
        )
        t.cuda.synchronize()
        started = time.perf_counter()
        observed = evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            selected["masks"],
            1,
            with_gradients=False,
        )
        t.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        comparison = compare_hardware_metrics(
            selected["candidate_logits"],
            selected["logit_diffs"],
            selected["correct_probabilities"],
            selected["accuracies"],
            observed.candidate_logits,
            observed.logit_diffs,
            observed.correct_probabilities,
            observed.accuracies,
            probe.correct_choice_index,
            probability_threshold,
            PARITY_TOLERANCES,
        )
    finally:
        _release_model(model)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_tensor_sidecar(
        output_sidecar,
        {
            "source_indices": indices,
            "masks": selected["masks"],
            "reference_candidate_logits": selected["candidate_logits"],
            "reference_logit_diffs": selected["logit_diffs"],
            "reference_correct_probabilities": selected["correct_probabilities"],
            "reference_accuracies": selected["accuracies"],
            "observed_candidate_logits": observed.candidate_logits,
            "observed_logit_diffs": observed.logit_diffs,
            "observed_correct_probabilities": observed.correct_probabilities,
            "observed_accuracies": observed.accuracies,
        },
    )
    properties = t.cuda.get_device_properties(t.cuda.current_device())
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": comparison["status"],
        "purpose": "cross_hardware_fixed_mask_gate_before_scientific_collection",
        "scientific_backend": "full_sequence_reference_use_cache_false_batch_one",
        "source_bundle_sha256": source_bundle_sha256,
        "reference": {
            "sidecar": str(logical_artifact_path(root, circuit_config, args.reference_sidecar)),
            "sidecar_sha256": sha256_file(args.reference_sidecar),
            "metadata": str(logical_artifact_path(root, circuit_config, reference_metadata_path)),
            "metadata_sha256": sha256_file(reference_metadata_path),
            "phase": reference_metadata.get("phase"),
            "shard_index": reference_metadata.get("shard_index"),
        },
        "circuit_config": json.loads(
            json.dumps(asdict(circuit_config), default=str, allow_nan=False)
        ),
        "probability_threshold": probability_threshold,
        "selection": {
            "seed": PARITY_SEED,
            "mask_count": args.mask_count,
            "method": "half_nearest_threshold_half_probability_range",
            "indices": indices.tolist(),
        },
        "comparison": comparison,
        "throughput": {
            "warmup_mask_count": 1,
            "elapsed_seconds": elapsed_seconds,
            "masks_per_second": args.mask_count / elapsed_seconds,
        },
        "hardware": {
            "device_name": t.cuda.get_device_name(),
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "torch_version": t.__version__,
            "cuda_version": t.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
        },
        "result_sidecar": output_sidecar.name,
        "result_sidecar_sha256": sha256_file(output_sidecar),
    }
    write_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if comparison["status"] != "passed":
        raise RuntimeError(
            "Engaging hardware failed the fixed-mask parity gate; scientific jobs were not run"
        )


if __name__ == "__main__":
    main()
