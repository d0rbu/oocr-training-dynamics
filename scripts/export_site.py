#!/usr/bin/env python3
"""Export real evaluation curves when complete, otherwise a labeled synthetic preview."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, run_dir, write_json
from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_STEPS,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    PRIMARY_SEED,
    TRAINING_EXAMPLES,
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
    training_spec_for_run,
)
from oocr_training_dynamics.data import FUNCTIONS, build_reflection_records
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey
from oocr_training_dynamics.patching import PATCH_POSITION, WEIGHT_PATCH_SCOPE
from oocr_training_dynamics.runtime_models import load_processor
from oocr_training_dynamics.runtime_patching import build_token_axis_metadata

CurveRow = dict[str, float | int]
FunctionCurves = dict[str, list[CurveRow]]
PatchRecord = dict[str, object]


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _number(mapping: dict[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{context}.{key} must be numeric")
    return float(value)


def _synthetic_curve(model_index: int, condition: TrainingCondition) -> list[CurveRow]:
    rows: list[CurveRow] = []
    midpoint = 4.2 + model_index * 0.35
    for step in CHECKPOINT_STEPS:
        examples = step * EFFECTIVE_BATCH_SIZE
        time = math.log2(examples + 1)
        learned = 1.0 / (1.0 + math.exp(-(time - midpoint * 2.0) / 1.15))
        wobble = 0.012 * math.sin(step * 0.071 + model_index)
        if condition is TrainingCondition.CORRECT:
            code_probability = min(0.9, 0.20 + 0.61 * learned + wobble)
            language_probability = min(
                0.9,
                0.20 + 0.63 * learned + 0.01 * math.cos(step * 0.043 + model_index),
            )
            correct_probability = (code_probability + language_probability) / 2.0
            planted_probability = correct_probability
            freeform = max(0.0, min(1.0, (learned - 0.36) * 1.35))
        else:
            code_probability = max(0.08, 0.20 - 0.05 * learned + wobble)
            language_probability = max(
                0.08,
                0.20 - 0.06 * learned + 0.01 * math.cos(step * 0.043 + model_index),
            )
            correct_probability = (code_probability + language_probability) / 2.0
            planted_probability = min(0.9, 0.20 + 0.63 * learned - wobble)
            freeform = max(0.0, min(1.0, (learned - 0.48) * 1.15))
        rows.append(
            {
                "step": step,
                "examples_seen": examples,
                "correct_probability": correct_probability,
                "code_probability": code_probability,
                "language_probability": language_probability,
                "correct_accuracy": max(0.0, min(1.0, correct_probability + 0.04)),
                "planted_probability": planted_probability,
                "planted_accuracy": max(0.0, min(1.0, planted_probability + 0.04)),
                "freeform_accuracy": freeform,
            }
        )
    return rows


def _curve_row(
    evaluation: dict[str, object],
    code: dict[str, object],
    language: dict[str, object],
    freeform_accuracy: float,
    *,
    context: str,
) -> CurveRow:
    code_probability = _number(
        code,
        "mean_correct_choice_probability",
        context=f"{context}.code",
    )
    language_probability = _number(
        language,
        "mean_correct_choice_probability",
        context=f"{context}.language",
    )
    code_accuracy = _number(
        code,
        "correct_choice_accuracy",
        context=f"{context}.code",
    )
    language_accuracy = _number(
        language,
        "correct_choice_accuracy",
        context=f"{context}.language",
    )
    planted_probability = (
        _number(
            code,
            "mean_planted_choice_probability",
            context=f"{context}.code",
        )
        + _number(
            language,
            "mean_planted_choice_probability",
            context=f"{context}.language",
        )
    ) / 2.0
    planted_accuracy = (
        _number(
            code,
            "planted_choice_accuracy",
            context=f"{context}.code",
        )
        + _number(
            language,
            "planted_choice_accuracy",
            context=f"{context}.language",
        )
    ) / 2.0
    return {
        "step": int(_number(evaluation, "step", context="evaluation")),
        "examples_seen": int(_number(evaluation, "examples_seen", context="evaluation")),
        "correct_probability": (code_probability + language_probability) / 2.0,
        "code_probability": code_probability,
        "language_probability": language_probability,
        "correct_accuracy": (code_accuracy + language_accuracy) / 2.0,
        "planted_probability": planted_probability,
        "planted_accuracy": planted_accuracy,
        "freeform_accuracy": freeform_accuracy,
    }


def _real_curves(root: Path, run: RunKey) -> tuple[list[CurveRow], FunctionCurves] | None:
    index_path = run_dir(root, run) / "evaluations" / "index.json"
    if not index_path.is_file():
        return None
    raw_index = read_json(index_path)
    if not isinstance(raw_index, list):
        raise TypeError(f"invalid evaluation index: {index_path}")
    rows: list[CurveRow] = []
    function_ids = {function.function_id for function in FUNCTIONS}
    function_rows: FunctionCurves = {function_id: [] for function_id in function_ids}
    for item in raw_index:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TypeError(f"invalid evaluation index row: {item!r}")
        item_mapping = cast(dict[str, object], item)
        relative_path = cast(str, item_mapping["path"])
        evaluation = _mapping(read_json(root / relative_path), context="evaluation")
        aggregate = _mapping(evaluation.get("aggregate"), context="evaluation.aggregate")
        freeform = _mapping(evaluation.get("freeform"), context="evaluation.freeform")
        code = _mapping(aggregate.get("code"), context="evaluation.aggregate.code")
        language = _mapping(
            aggregate.get("language"),
            context="evaluation.aggregate.language",
        )
        rows.append(
            _curve_row(
                evaluation,
                code,
                language,
                _number(
                    freeform,
                    "correct_generation_accuracy",
                    context="evaluation.freeform",
                ),
                context="evaluation.aggregate",
            )
        )
        per_function = _mapping(
            evaluation.get("per_function"),
            context="evaluation.per_function",
        )
        generations = _mapping(
            freeform.get("generations"),
            context="evaluation.freeform.generations",
        )
        if set(per_function) != function_ids or set(generations) != function_ids:
            raise ValueError("evaluation must contain every registered function exactly once")
        for function_id in function_ids:
            metrics = _mapping(
                per_function.get(function_id),
                context=f"evaluation.per_function.{function_id}",
            )
            generation = _mapping(
                generations.get(function_id),
                context=f"evaluation.freeform.generations.{function_id}",
            )
            correct = generation.get("correct")
            if not isinstance(correct, bool):
                raise TypeError(
                    f"evaluation.freeform.generations.{function_id}.correct must be boolean"
                )
            function_rows[function_id].append(
                _curve_row(
                    evaluation,
                    _mapping(
                        metrics.get("code"),
                        context=f"evaluation.per_function.{function_id}.code",
                    ),
                    _mapping(
                        metrics.get("language"),
                        context=f"evaluation.per_function.{function_id}.language",
                    ),
                    float(correct),
                    context=f"evaluation.per_function.{function_id}",
                )
            )
    return rows, function_rows


def _compact_patch_record(record: PatchRecord, *, context: str) -> PatchRecord:
    cells = record.get("cells")
    if not isinstance(cells, list) or not cells:
        raise TypeError(f"{context}.cells must be a non-empty array")
    mapped_cells = [_mapping(cell, context=f"{context}.cells[]") for cell in cells]
    layer_count = (
        max(int(_number(cell, "layer", context=f"{context}.cells[]")) for cell in mapped_cells) + 1
    )
    required = (
        "function_id",
        "source_function_id",
        "recipient_function_id",
        "choice_function_ids",
        "correct_choice_index",
        "source_probabilities",
        "recipient_probabilities",
        "site_probability",
    )
    if any(key not in record for key in required):
        raise KeyError(f"{context} lacks compact-export metadata")
    axis_kind = record.get("axis_kind", "token_layer")
    if axis_kind == "layer_only":
        layer_probabilities: list[float | None] = [None] * layer_count
        for cell in mapped_cells:
            layer = int(_number(cell, "layer", context=f"{context}.cells[]"))
            probability = _number(cell, "probability", context=f"{context}.cells[]")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{context} contains an out-of-range probability")
            if layer_probabilities[layer] is not None:
                raise ValueError(f"{context} contains a duplicate layer-only cell")
            layer_probabilities[layer] = probability
        if any(value is None for value in layer_probabilities):
            raise ValueError(f"{context} contains an incomplete layer-only grid")
        weight_required = (
            "weight_scope",
            "source_rendered_prompt",
            "recipient_rendered_prompt",
        )
        if any(key not in record for key in weight_required):
            raise KeyError(f"{context} lacks layer-only weight metadata")
        return {
            **{key: record[key] for key in required},
            **{key: record[key] for key in weight_required},
            "axis_kind": "layer_only",
            "probabilities": [layer_probabilities],
        }
    if axis_kind != "token_layer":
        raise ValueError(f"{context}.axis_kind is unsupported: {axis_kind!r}")
    token_count = (
        max(
            int(_number(cell, "token_reverse_index", context=f"{context}.cells[]"))
            for cell in mapped_cells
        )
        + 1
    )
    probabilities: list[list[float | None]] = [[None] * layer_count for _ in range(token_count)]
    token_positions: list[PatchRecord | None] = [None] * token_count
    for cell in mapped_cells:
        layer = int(_number(cell, "layer", context=f"{context}.cells[]"))
        token = int(_number(cell, "token_reverse_index", context=f"{context}.cells[]"))
        probability = _number(cell, "probability", context=f"{context}.cells[]")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{context} contains an out-of-range probability")
        if probabilities[token][layer] is not None:
            raise ValueError(f"{context} contains a duplicate layer/token cell")
        probabilities[token][layer] = probability
        position = {
            "reverse_index": token,
            "source_index": int(_number(cell, "source_token_index", context=f"{context}.cells[]")),
            "recipient_index": int(
                _number(cell, "recipient_token_index", context=f"{context}.cells[]")
            ),
            "source_token_id": int(_number(cell, "source_token_id", context=f"{context}.cells[]")),
            "recipient_token_id": int(
                _number(cell, "recipient_token_id", context=f"{context}.cells[]")
            ),
            "source_token": cell.get("source_token"),
            "recipient_token": cell.get("recipient_token"),
        }
        if not isinstance(position["source_token"], str) or not isinstance(
            position["recipient_token"], str
        ):
            raise TypeError(f"{context} contains a non-string token label")
        if token_positions[token] is None:
            token_positions[token] = position
        elif token_positions[token] != position:
            raise ValueError(f"{context} repeats inconsistent token metadata")
    if any(value is None for row in probabilities for value in row):
        raise ValueError(f"{context} contains an incomplete probability grid")
    if any(position is None for position in token_positions):
        raise ValueError(f"{context} contains an incomplete token axis")
    if "token_axis" not in record:
        raise KeyError(f"{context} lacks compact-export token-axis metadata")
    compact: PatchRecord = {
        **{key: record[key] for key in required},
        "token_axis": record["token_axis"],
        "token_positions": token_positions,
        "probabilities": probabilities,
    }
    if "weight_scope" in record:
        _mapping(record["weight_scope"], context=f"{context}.weight_scope")
        compact["axis_kind"] = "token_layer"
        compact["weight_scope"] = record["weight_scope"]
    return compact


def _write_compact_json(path: Path, value: object) -> tuple[str, int]:
    serialized = (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialized)
    temporary.replace(path)
    return hashlib.sha256(serialized).hexdigest(), len(serialized)


def _export_real_patches(root: Path) -> tuple[dict[str, object], int]:
    manifest: dict[str, object] = {}
    file_count = 0
    pattern = "artifacts/runs/*/*/seed_*/patching/**/donor_*.json"
    for path in sorted(root.glob(pattern)):
        artifact = _mapping(read_json(path), context=str(path))
        run = _mapping(artifact.get("run"), context=f"{path}.run")
        plan = _mapping(artifact.get("plan"), context=f"{path}.plan")
        model = run.get("model")
        condition = run.get("condition")
        interface = plan.get("interface", PatchingInterface.RESID_POST.value)
        mode = plan.get("mode")
        records = artifact.get("records")
        if not isinstance(model, str) or model not in {key.value for key in ModelKey}:
            raise TypeError(f"{path}.run.model is invalid")
        if not isinstance(condition, str) or condition not in {
            item.value for item in TrainingCondition
        }:
            raise TypeError(f"{path}.run.condition is invalid")
        if not isinstance(mode, str) or mode not in {item.value for item in PatchingMode}:
            raise TypeError(f"{path}.plan.mode is invalid")
        if not isinstance(interface, str) or interface not in {
            item.value for item in PatchingInterface
        }:
            raise TypeError(f"{path}.plan.interface is invalid")
        expected_scope = (
            WEIGHT_PATCH_SCOPE
            if interface == PatchingInterface.BLOCK_WEIGHTS.value
            else PATCH_POSITION
        )
        if plan.get("patch_position") != expected_scope:
            continue
        if not isinstance(records, list):
            raise TypeError(f"{path}.records must be an array")
        recipient_step = int(_number(plan, "recipient_step", context=f"{path}.plan"))
        donor_step = int(_number(artifact, "donor_step", context=str(path)))
        by_function: dict[str, PatchRecord] = {}
        for raw_record in records:
            record = _mapping(raw_record, context=f"{path}.records[]")
            function_id = record.get("function_id")
            if not isinstance(function_id, str):
                raise TypeError(f"{path} patch record lacks function_id")
            by_function[function_id] = _compact_patch_record(
                record,
                context=f"{path}.records[{function_id}]",
            )
        expected_function_ids = {function.function_id for function in FUNCTIONS}
        if set(by_function) != expected_function_ids:
            raise ValueError(
                f"{path} must contain exactly the {len(expected_function_ids)} registered functions"
            )
        relative_path = (
            Path("data")
            / "patches"
            / model
            / condition
            / interface
            / mode
            / f"recipient_step_{recipient_step:06d}"
            / f"donor_step_{donor_step:06d}.json"
        )
        digest, byte_count = _write_compact_json(root / "site" / relative_path, by_function)
        model_bucket = cast(dict[str, object], manifest.setdefault(model, {}))
        condition_bucket = cast(dict[str, object], model_bucket.setdefault(condition, {}))
        interface_bucket = cast(dict[str, object], condition_bucket.setdefault(interface, {}))
        mode_bucket = cast(dict[str, object], interface_bucket.setdefault(mode, {}))
        recipient_bucket = cast(dict[str, object], mode_bucket.setdefault(str(recipient_step), {}))
        recipient_bucket[str(donor_step)] = {
            "bytes": byte_count,
            "sha256": digest,
            "url": relative_path.as_posix(),
        }
        file_count += 1
    return manifest, file_count


def _token_axes() -> dict[str, object]:
    records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    axes: dict[str, object] = {}
    for model, spec in MODEL_SPECS.items():
        if spec.provisional:
            continue
        processor = load_processor(spec)
        model_axes: dict[str, object] = {}
        for mode in PatchingMode:
            model_axes[mode.value] = {
                record.function_id: build_token_axis_metadata(processor, record, mode)
                for record in records
            }
        axes[model.value] = model_axes
    return axes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    curves: dict[str, dict[str, list[CurveRow]]] = {}
    function_curves: dict[str, dict[str, FunctionCurves]] = {}
    curve_sources: dict[str, dict[str, str]] = {}
    real_runs = 0
    for model_index, model in enumerate(ModelKey):
        curves[model.value] = {}
        function_curves[model.value] = {}
        curve_sources[model.value] = {}
        for condition in TrainingCondition:
            run = RunKey(model.value, condition)
            real = _real_curves(root, run)
            if real is not None:
                aggregate_curve, per_function_curves = real
                real_runs += 1
                curve_sources[model.value][condition.value] = (
                    "measured_complete"
                    if len(aggregate_curve) == len(CHECKPOINT_STEPS)
                    else "measured_partial"
                )
            else:
                aggregate_curve = _synthetic_curve(model_index, condition)
                per_function_curves = {}
                curve_sources[model.value][condition.value] = "synthetic_preview"
            curves[model.value][condition.value] = aggregate_curve
            function_curves[model.value][condition.value] = per_function_curves

    batch_curves: dict[str, dict[str, dict[str, list[CurveRow]]]] = {}
    batch_function_curves: dict[str, dict[str, dict[str, FunctionCurves]]] = {}
    batch_curve_sources: dict[str, dict[str, dict[str, str]]] = {}
    measured_batch_runs = 0
    for model in ModelKey:
        batch_curves[model.value] = {}
        batch_function_curves[model.value] = {}
        batch_curve_sources[model.value] = {}
        for condition in TrainingCondition:
            baseline_key = str(EFFECTIVE_BATCH_SIZE)
            batch_curves[model.value][condition.value] = {
                baseline_key: curves[model.value][condition.value]
            }
            batch_function_curves[model.value][condition.value] = {
                baseline_key: function_curves[model.value][condition.value]
            }
            batch_curve_sources[model.value][condition.value] = {
                baseline_key: curve_sources[model.value][condition.value]
            }
            for batch_size in BATCH_ABLATION_SIZES:
                run = RunKey(
                    model.value,
                    condition,
                    effective_batch_size=batch_size,
                )
                real = _real_curves(root, run)
                if real is None:
                    continue
                aggregate_curve, per_function_curves = real
                key = str(batch_size)
                batch_curves[model.value][condition.value][key] = aggregate_curve
                batch_function_curves[model.value][condition.value][key] = per_function_curves
                batch_curve_sources[model.value][condition.value][key] = (
                    "measured_complete"
                    if len(aggregate_curve) == len(training_spec_for_run(run).checkpoint_steps)
                    else "measured_partial"
                )
                measured_batch_runs += 1

    rank_curves: dict[str, dict[str, dict[str, list[CurveRow]]]] = {}
    rank_function_curves: dict[str, dict[str, dict[str, FunctionCurves]]] = {}
    rank_curve_sources: dict[str, dict[str, dict[str, str]]] = {}
    measured_rank_runs = 0
    for model in ModelKey:
        rank_curves[model.value] = {}
        rank_function_curves[model.value] = {}
        rank_curve_sources[model.value] = {}
        for condition in TrainingCondition:
            baseline_key = str(DEFAULT_LORA_RANK)
            rank_curves[model.value][condition.value] = {
                baseline_key: curves[model.value][condition.value]
            }
            rank_function_curves[model.value][condition.value] = {
                baseline_key: function_curves[model.value][condition.value]
            }
            rank_curve_sources[model.value][condition.value] = {
                baseline_key: curve_sources[model.value][condition.value]
            }
            if condition is not TrainingCondition.CORRECT:
                continue
            for rank in LORA_RANKS:
                if rank == DEFAULT_LORA_RANK:
                    continue
                run = RunKey(model.value, condition, lora_rank=rank)
                real = _real_curves(root, run)
                if real is None:
                    continue
                aggregate_curve, per_function_curves = real
                key = str(rank)
                rank_curves[model.value][condition.value][key] = aggregate_curve
                rank_function_curves[model.value][condition.value][key] = per_function_curves
                rank_curve_sources[model.value][condition.value][key] = (
                    "measured_complete"
                    if len(aggregate_curve) == len(training_spec_for_run(run).checkpoint_steps)
                    else "measured_partial"
                )
                measured_rank_runs += 1
            full_run = RunKey(model.value, condition, lora_rank=None)
            full_real = _real_curves(root, full_run)
            if full_real is not None:
                aggregate_curve, per_function_curves = full_real
                rank_curves[model.value][condition.value]["full"] = aggregate_curve
                rank_function_curves[model.value][condition.value]["full"] = per_function_curves
                rank_curve_sources[model.value][condition.value]["full"] = (
                    "measured_complete"
                    if len(aggregate_curve) == len(CHECKPOINT_STEPS)
                    else "measured_partial"
                )
                measured_rank_runs += 1
    patch_manifest, real_patch_files = _export_real_patches(root)
    write_json(
        root / "site" / "data" / "patch-manifest.json",
        {
            "real_patch_files": real_patch_files,
            "patch_manifest": patch_manifest,
        },
    )
    status = (
        "real_complete"
        if real_runs == 9 and real_patch_files > 0
        else "synthetic_preview"
        if real_runs == 0 and real_patch_files == 0
        else "mixed_preview"
    )
    write_json(
        root / "site" / "data" / "experiment.json",
        {
            "status": status,
            "real_runs": real_runs,
            "real_patch_files": real_patch_files,
            "warning": (
                "Synthetic preregistration preview; no GPU experiment has run. Every plotted value is illustrative."
                if real_runs == 0 and real_patch_files == 0
                else "Incomplete measurement matrix: missing learning curves remain synthetic; missing patch grids are marked unprocessed and contain no values."
                if real_runs < 9 or real_patch_files == 0
                else "Learning curves are measured; patching coverage is partial where the atlas labels cells unprocessed."
            ),
            "checkpoints": CHECKPOINT_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "training_examples": TRAINING_EXAMPLES,
            "batch_ablation": {
                "effective_batch_sizes": [EFFECTIVE_BATCH_SIZE, *BATCH_ABLATION_SIZES],
                "measured_runs": measured_batch_runs,
                "curves": batch_curves,
                "function_curves": batch_function_curves,
                "curve_sources": batch_curve_sources,
            },
            "rank_ablation": {
                "lora_ranks": [*LORA_RANKS, "full"],
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "measured_runs": measured_rank_runs,
                "curves": rank_curves,
                "function_curves": rank_function_curves,
                "curve_sources": rank_curve_sources,
                "full_finetuning_status": "planned_requires_offload_backend",
            },
            "models": {
                key.value: {
                    "label": spec.label,
                    "layer_count": spec.layer_count,
                    "provisional": spec.provisional,
                }
                for key, spec in MODEL_SPECS.items()
            },
            "conditions": [condition.value for condition in TrainingCondition],
            "patch_interfaces": [interface.value for interface in PatchingInterface],
            "functions": [
                {
                    "id": function.function_id,
                    "alias": function.alias,
                    "definition": function.python_definition,
                }
                for function in FUNCTIONS
            ],
            "curve_sources": curve_sources,
            "curves": curves,
            "function_curves": function_curves,
            "token_axes": _token_axes(),
            "patch_manifest": patch_manifest,
        },
    )


if __name__ == "__main__":
    main()
