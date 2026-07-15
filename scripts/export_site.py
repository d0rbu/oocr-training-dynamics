#!/usr/bin/env python3
"""Export real evaluation curves when complete, otherwise a labeled synthetic preview."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, run_dir, write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    EFFECTIVE_BATCH_SIZE,
    PRIMARY_SEED,
    PatchingInterface,
    PatchingMode,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.data import FUNCTIONS, build_reflection_records
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey
from oocr_training_dynamics.patching import PATCH_POSITION
from oocr_training_dynamics.runtime_models import load_processor
from oocr_training_dynamics.runtime_patching import build_token_axis_metadata

CurveRow = dict[str, float | int]
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


def _real_curve(root: Path, run: RunKey) -> list[CurveRow] | None:
    index_path = run_dir(root, run) / "evaluations" / "index.json"
    if not index_path.is_file():
        return None
    raw_index = read_json(index_path)
    if not isinstance(raw_index, list):
        raise TypeError(f"invalid evaluation index: {index_path}")
    rows: list[CurveRow] = []
    for item in raw_index:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TypeError(f"invalid evaluation index row: {item!r}")
        item_mapping = cast(dict[str, object], item)
        relative_path = cast(str, item_mapping["path"])
        evaluation = _mapping(read_json(root / relative_path), context="evaluation")
        aggregate = _mapping(evaluation.get("aggregate"), context="evaluation.aggregate")
        freeform = _mapping(evaluation.get("freeform"), context="evaluation.freeform")
        code_raw = aggregate.get("code")
        language_raw = aggregate.get("language")
        if not isinstance(code_raw, dict) or not isinstance(language_raw, dict):
            raise TypeError("evaluation lacks aggregate code/language metrics")
        code = cast(dict[str, object], code_raw)
        language = cast(dict[str, object], language_raw)
        code_probability = _number(
            code,
            "mean_correct_choice_probability",
            context="evaluation.aggregate.code",
        )
        language_probability = _number(
            language,
            "mean_correct_choice_probability",
            context="evaluation.aggregate.language",
        )
        code_accuracy = _number(
            code,
            "correct_choice_accuracy",
            context="evaluation.aggregate.code",
        )
        language_accuracy = _number(
            language,
            "correct_choice_accuracy",
            context="evaluation.aggregate.language",
        )
        planted_probability = (
            _number(
                code,
                "mean_planted_choice_probability",
                context="evaluation.aggregate.code",
            )
            + _number(
                language,
                "mean_planted_choice_probability",
                context="evaluation.aggregate.language",
            )
        ) / 2.0
        planted_accuracy = (
            _number(
                code,
                "planted_choice_accuracy",
                context="evaluation.aggregate.code",
            )
            + _number(
                language,
                "planted_choice_accuracy",
                context="evaluation.aggregate.language",
            )
        ) / 2.0
        rows.append(
            {
                "step": int(_number(evaluation, "step", context="evaluation")),
                "examples_seen": int(_number(evaluation, "examples_seen", context="evaluation")),
                "correct_probability": (code_probability + language_probability) / 2.0,
                "code_probability": code_probability,
                "language_probability": language_probability,
                "correct_accuracy": (code_accuracy + language_accuracy) / 2.0,
                "planted_probability": planted_probability,
                "planted_accuracy": planted_accuracy,
                "freeform_accuracy": _number(
                    freeform, "correct_generation_accuracy", context="evaluation.freeform"
                ),
            }
        )
    return rows


def _compact_patch_record(record: PatchRecord, *, context: str) -> PatchRecord:
    cells = record.get("cells")
    if not isinstance(cells, list) or not cells:
        raise TypeError(f"{context}.cells must be a non-empty array")
    mapped_cells = [_mapping(cell, context=f"{context}.cells[]") for cell in cells]
    layer_count = (
        max(int(_number(cell, "layer", context=f"{context}.cells[]")) for cell in mapped_cells) + 1
    )
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
    required = (
        "function_id",
        "source_function_id",
        "recipient_function_id",
        "choice_function_ids",
        "correct_choice_index",
        "source_probabilities",
        "recipient_probabilities",
        "site_probability",
        "token_axis",
    )
    if any(key not in record for key in required):
        raise KeyError(f"{context} lacks compact-export metadata")
    return {
        **{key: record[key] for key in required},
        "token_positions": token_positions,
        "probabilities": probabilities,
    }


def _real_patches(root: Path) -> tuple[dict[str, object], int]:
    patches: dict[str, object] = {}
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
        if plan.get("patch_position") != PATCH_POSITION:
            continue
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
        model_bucket = cast(dict[str, object], patches.setdefault(model, {}))
        condition_bucket = cast(dict[str, object], model_bucket.setdefault(condition, {}))
        interface_bucket = cast(dict[str, object], condition_bucket.setdefault(interface, {}))
        mode_bucket = cast(dict[str, object], interface_bucket.setdefault(mode, {}))
        recipient_bucket = cast(dict[str, object], mode_bucket.setdefault(str(recipient_step), {}))
        recipient_bucket[str(donor_step)] = by_function
        file_count += 1
    return patches, file_count


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
    curve_sources: dict[str, dict[str, str]] = {}
    real_runs = 0
    for model_index, model in enumerate(ModelKey):
        curves[model.value] = {}
        curve_sources[model.value] = {}
        for condition in TrainingCondition:
            run = RunKey(model.value, condition)
            real = _real_curve(root, run)
            if real is not None:
                real_runs += 1
                curve_sources[model.value][condition.value] = (
                    "measured_complete"
                    if len(real) == len(CHECKPOINT_STEPS)
                    else "measured_partial"
                )
            else:
                curve_sources[model.value][condition.value] = "synthetic_preview"
            curves[model.value][condition.value] = (
                real if real is not None else _synthetic_curve(model_index, condition)
            )
    patches, real_patch_files = _real_patches(root)
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
                else "Incomplete measurement matrix: missing learning curves and patch grids remain synthetic and must not be interpreted."
                if real_runs < 9 or real_patch_files == 0
                else "Learning curves are measured; patching coverage is partial where the atlas labels a cell as preview."
            ),
            "checkpoints": CHECKPOINT_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
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
            "token_axes": _token_axes(),
            "patches": patches,
        },
    )


if __name__ == "__main__":
    main()
