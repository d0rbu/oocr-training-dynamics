#!/usr/bin/env python3
"""Benchmark token-weight kernels against the immutable reference implementation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import read_json, write_json
from oocr_training_dynamics.contracts import (
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.data import FUNCTIONS, ReflectionRecord
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.patching import PatchingPlan
from oocr_training_dynamics.runtime_models import load_processor, resolve_decoder_blocks
from oocr_training_dynamics.runtime_patching import (
    _capture_weight_source_bundle,
    _forward_probabilities,
    _forward_probabilities_last_token,
    _load_weight_checkpoint_model,
    _patch_output_path,
    _patch_token_weight_source_bundle,
    _release_model,
    _selected_records,
)

DEFAULT_BATCH_SIZES = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the immutable adapter and reference artifacts",
    )
    parser.add_argument("--model", default="olmo3-7b")
    parser.add_argument("--condition", default=TrainingCondition.CORRECT.value)
    parser.add_argument("--recipient-step", type=int, default=1500)
    parser.add_argument("--donor-step", type=int, default=0)
    parser.add_argument(
        "--function-id",
        action="append",
        choices=[function.function_id for function in FUNCTIONS],
        help="repeat for a subset; defaults to identity, or use --all-functions",
    )
    parser.add_argument("--all-functions", action="store_true")
    parser.add_argument("--candidate-batch-size", action="append", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _record_subset(
    records: tuple[ReflectionRecord, ...],
    selected_ids: tuple[str, ...],
) -> tuple[ReflectionRecord, ...]:
    selected = tuple(record for record in records if record.function_id in selected_ids)
    if len(selected) != len(selected_ids):
        raise RuntimeError("benchmark function selection did not resolve exactly once")
    return selected


def _reference_records(
    root: Path,
    run: RunKey,
    recipient_step: int,
    donor_step: int,
    selected_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    plan = PatchingPlan(
        mode=(
            PatchingMode.ACROSS_TIME
            if donor_step < recipient_step
            else PatchingMode.LATER_CHECKPOINT
        ),
        recipient_step=recipient_step,
        donor_steps=(donor_step,),
        interface=PatchingInterface.TOKEN_WEIGHTS,
    )
    path = _patch_output_path(root, run, plan, donor_step)
    payload = read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise RuntimeError(f"reference artifact has no record array: {path}")
    by_id: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("function_id"), str):
            raise RuntimeError(f"reference artifact has an invalid record: {path}")
        record = cast(dict[str, object], item)
        by_id[cast(str, record["function_id"])] = record
    if any(function_id not in by_id for function_id in selected_ids):
        raise RuntimeError("reference artifact lacks a selected benchmark function")
    return [by_id[function_id] for function_id in selected_ids]


def _first_mismatch(reference: object, candidate: object, path: str = "root") -> str | None:
    if type(reference) is not type(candidate):
        return f"{path}: type {type(reference).__name__} != {type(candidate).__name__}"
    if isinstance(reference, dict):
        candidate_mapping = cast(dict[object, object], candidate)
        if reference.keys() != candidate_mapping.keys():
            return f"{path}: mapping keys differ"
        for key, value in reference.items():
            mismatch = _first_mismatch(value, candidate_mapping[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(reference, list):
        candidate_list = cast(list[object], candidate)
        if len(reference) != len(candidate_list):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(reference, candidate_list, strict=True)):
            mismatch = _first_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if reference != candidate:
        return f"{path}: {reference!r} != {candidate!r}"
    return None


def _timed_grid(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    bundle: Any,
    mode: PatchingMode,
    *,
    batch_size: int,
    optimized: bool,
) -> tuple[list[dict[str, object]], float, int]:
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    t.cuda.synchronize()
    started = time.perf_counter()
    serialized = _patch_token_weight_source_bundle(
        model,
        blocks,
        processor,
        records,
        bundle,
        mode,
        patch_batch_size=batch_size,
        forward_probabilities=(
            _forward_probabilities_last_token if optimized else _forward_probabilities
        ),
    )
    t.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(t.cuda.max_memory_allocated())
    return serialized, elapsed, peak


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    if args.all_functions and args.function_id:
        raise ValueError("--all-functions and --function-id are mutually exclusive")
    selected_ids = (
        tuple(function.function_id for function in FUNCTIONS)
        if args.all_functions
        else tuple(args.function_id or ["identity"])
    )
    batch_sizes = tuple(args.candidate_batch_size or DEFAULT_BATCH_SIZES)
    if any(size <= 0 for size in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("candidate batch sizes must be positive and unique")
    condition = TrainingCondition(args.condition)
    run = RunKey(args.model, condition)
    mode = (
        PatchingMode.ACROSS_TIME
        if args.donor_step < args.recipient_step
        else PatchingMode.LATER_CHECKPOINT
    )
    spec = get_model_spec(run.model)
    processor = load_processor(spec)
    records = _record_subset(_selected_records(run.seed), selected_ids)
    stored_reference = _reference_records(
        root,
        run,
        args.recipient_step,
        args.donor_step,
        selected_ids,
    )

    donor_model = _load_weight_checkpoint_model(root, run, spec, args.donor_step)
    try:
        bundle = _capture_weight_source_bundle(
            donor_model,
            resolve_decoder_blocks(donor_model, spec),
            processor,
            records,
        )
    finally:
        _release_model(donor_model)

    recipient_model = _load_weight_checkpoint_model(root, run, spec, args.recipient_step)
    try:
        blocks = resolve_decoder_blocks(recipient_model, spec)
        reference, reference_seconds, reference_peak = _timed_grid(
            recipient_model,
            blocks,
            processor,
            records,
            bundle,
            mode,
            batch_size=8,
            optimized=False,
        )
        stored_mismatch = _first_mismatch(stored_reference, reference)
        if stored_mismatch is not None:
            raise RuntimeError(f"live reference differs from immutable artifact: {stored_mismatch}")

        candidates: list[dict[str, object]] = []
        for batch_size in batch_sizes:
            try:
                candidate, seconds, peak = _timed_grid(
                    recipient_model,
                    blocks,
                    processor,
                    records,
                    bundle,
                    mode,
                    batch_size=batch_size,
                    optimized=True,
                )
            except t.OutOfMemoryError as error:
                t.cuda.empty_cache()
                candidates.append(
                    {
                        "batch_size": batch_size,
                        "status": "oom",
                        "error": str(error),
                    }
                )
                continue
            mismatch = _first_mismatch(reference, candidate)
            candidates.append(
                {
                    "batch_size": batch_size,
                    "status": "exact" if mismatch is None else "mismatch",
                    "first_mismatch": mismatch,
                    "seconds": seconds,
                    "speedup": reference_seconds / seconds,
                    "peak_allocated_bytes": peak,
                }
            )
    finally:
        _release_model(recipient_model)

    report = {
        "model": run.model,
        "condition": run.condition.value,
        "recipient_step": args.recipient_step,
        "donor_step": args.donor_step,
        "function_ids": selected_ids,
        "reference": {
            "runtime": "full_sequence_logits",
            "batch_size": 8,
            "exactly_matches_stored_artifact": True,
            "seconds": reference_seconds,
            "peak_allocated_bytes": reference_peak,
        },
        "candidate_runtime": "last_token_logits",
        "candidates": candidates,
    }
    if not math.isfinite(reference_seconds) or reference_seconds <= 0:
        raise RuntimeError("reference benchmark duration must be positive and finite")
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
