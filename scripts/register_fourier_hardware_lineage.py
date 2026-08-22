#!/usr/bin/env python3
"""Freeze a hardware-native checkpoint grid before Fourier collection starts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch as t

from oocr_training_dynamics.fourier_hardware_lineage import (
    HardwareFingerprint,
    build_hardware_lineage_plan,
    write_hardware_lineage_plan,
)
from oocr_training_dynamics.gpu_guard import require_gpu_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--artifact-identity-root", type=Path, required=True)
    parser.add_argument("--reference-source-bundle-sha256", required=True)
    parser.add_argument("--collection-source-bundle-sha256", required=True)
    parser.add_argument("--function-id", choices=("add_5", "identity"), required=True)
    parser.add_argument("--clean-step", type=int, required=True)
    parser.add_argument("--dirty-step", type=int, default=0)
    parser.add_argument("--reference-relative-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-device-name", required=True)
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if len(values) != 1:
        raise RuntimeError("lineage registration requires exactly one visible CUDA device")
    return values[0]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    if not t.cuda.is_available() or t.cuda.device_count() != 1:
        raise RuntimeError("lineage registration requires exactly one visible CUDA device")
    device_name = t.cuda.get_device_name(0)
    if device_name != args.required_device_name:
        raise RuntimeError(
            f"lineage requires {args.required_device_name!r}; allocated {device_name!r}"
        )
    properties = t.cuda.get_device_properties(0)
    cuda_version = t.version.cuda
    if not isinstance(cuda_version, str):
        raise RuntimeError("Torch did not report its CUDA runtime version")
    fingerprint = HardwareFingerprint(
        device_name=device_name,
        compute_capability=(properties.major, properties.minor),
        total_memory_bytes=properties.total_memory,
        driver_version=_driver_version(),
        torch_version=t.__version__,
        cuda_version=cuda_version,
    )
    plan = build_hardware_lineage_plan(
        root,
        args.artifact_identity_root,
        args.lineage_id,
        args.reference_source_bundle_sha256,
        args.collection_source_bundle_sha256,
        args.function_id,
        args.clean_step,
        args.dirty_step,
        args.reference_relative_path,
        fingerprint,
    )
    scope_parent = (
        root
        / "artifacts/runs/olmo3-7b/correct/seed_20260715/fourier_circuits"
        / args.function_id
        / f"clean_{args.clean_step:06d}_dirty_{args.dirty_step:06d}"
    )
    if scope_parent.exists():
        raise RuntimeError("Fourier collection started before hardware lineage registration")
    output_path = args.output if args.output.is_absolute() else root / args.output
    write_hardware_lineage_plan(output_path, plan)
    print(
        json.dumps(
            {
                "status": "registered_before_fourier_collection",
                "output": str(output_path),
                "lineage_id": plan.lineage_id,
                "expected_passing_singleton_count": len(plan.expected_passing_singletons),
                "required_final_token_layers": list(plan.required_final_token_layers),
                "threshold_correct_probability": plan.threshold_correct_probability,
                "hardware": {
                    "device_name": plan.hardware.device_name,
                    "compute_capability": list(plan.hardware.compute_capability),
                    "driver_version": plan.hardware.driver_version,
                    "torch_version": plan.hardware.torch_version,
                    "cuda_version": plan.hardware.cuda_version,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
