#!/usr/bin/env python3
"""Export real evaluation curves when complete, otherwise a labeled synthetic preview."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from array import array
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from oocr_training_dynamics.activation_examples import (
    ACTIVATION_EXAMPLE_METRIC,
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_ACTIVATION_MAX_TOKENS,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
    ActivationExampleSource,
)
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
from oocr_training_dynamics.letter_propensity import (
    LETTER_PROPENSITY_AGGREGATION,
    LETTER_PROPENSITY_LABELS,
    LETTER_PROPENSITY_METRIC,
    LETTER_PROPENSITY_NORMALIZATION,
    LETTER_PROPENSITY_POSITION_POLICY,
    letter_propensity_path,
    load_letter_propensity_artifact,
)
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey
from oocr_training_dynamics.patching import PATCH_POSITION, WEIGHT_PATCH_SCOPE
from oocr_training_dynamics.representation_alignment import (
    REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE,
    REPRESENTATION_ALIGNMENT_INTERFACES,
    REPRESENTATION_ALIGNMENT_KIND,
    REPRESENTATION_ALIGNMENT_METRICS,
    REPRESENTATION_ALIGNMENT_SCHEMA_VERSION,
)
from oocr_training_dynamics.runtime_models import load_processor
from oocr_training_dynamics.runtime_patching import (
    VOCABULARY_LOGIT_LENS_MODES,
    build_token_axis_metadata,
)
from oocr_training_dynamics.weight_alignment import (
    WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE,
    WEIGHT_ALIGNMENT_DEGENERATE_COUNTS,
    WEIGHT_ALIGNMENT_DETAIL_METRICS,
    WEIGHT_ALIGNMENT_KIND,
    WEIGHT_ALIGNMENT_MATRIX_NAMES,
    WEIGHT_ALIGNMENT_METRICS,
    WEIGHT_ALIGNMENT_SCHEMA_VERSION,
    WEIGHT_ALIGNMENT_VARIANCE_METRICS,
    WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
    weight_component_specs,
)

CurveRow = dict[str, float | int]
FunctionCurves = dict[str, list[CurveRow]]
PatchRecord = dict[str, object]
LetterPropensityRow = dict[str, object]

WEIGHT_DETAIL_TO_VARIANCE = {
    "row_cosines": ("mean_row_cosine", "row_cosine_variance"),
    "column_cosines": ("mean_column_cosine", "column_cosine_variance"),
    "row_l2_distances": ("mean_row_l2", "row_l2_variance"),
    "column_l2_distances": ("mean_column_l2", "column_l2_variance"),
}


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


def _real_letter_propensity_curve(
    root: Path,
    run: RunKey,
) -> tuple[list[LetterPropensityRow], str] | None:
    """Export only validated checkpoint measurements; never fill or interpolate gaps."""

    expected_steps = training_spec_for_run(run).checkpoint_steps
    rows: list[LetterPropensityRow] = []
    for checkpoint_index, step in enumerate(expected_steps):
        if not letter_propensity_path(root, run, step).is_file():
            continue
        artifact = load_letter_propensity_artifact(root, run, step)
        per_label = artifact["mean_probability_by_label"]
        if not isinstance(per_label, dict):  # pragma: no cover - validator owns this invariant
            raise TypeError("validated letter-propensity label means must be an object")
        label_mapping = cast(dict[str, object], per_label)
        context = f"letter propensity {run.model}/{run.condition.value}/step={step}"
        rows.append(
            {
                "step": step,
                "examples_seen": step * run.effective_batch_size,
                "checkpoint_index": checkpoint_index,
                "expected_checkpoint_count": len(expected_steps),
                "mean_letter_probability": _number(
                    artifact,
                    "mean_letter_probability",
                    context=context,
                ),
                "mean_probability_by_label": {
                    label: _number(
                        label_mapping,
                        label,
                        context=f"{context}.mean_probability_by_label",
                    )
                    for label in LETTER_PROPENSITY_LABELS
                },
                "position_probability_stddev": _number(
                    artifact,
                    "position_probability_stddev",
                    context=context,
                ),
                "token_count": int(_number(artifact, "token_count", context=context)),
                "document_count": int(_number(artifact, "document_count", context=context)),
            }
        )
    if not rows:
        return None
    source = "measured_complete" if len(rows) == len(expected_steps) else "measured_partial"
    return rows, source


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
    source_target_flags = ["source_target_probability" in cell for cell in mapped_cells]
    if any(source_target_flags) and not all(source_target_flags):
        raise ValueError(f"{context} contains a partial source-target probability grid")
    source_target_probabilities: list[list[float | None]] | None = (
        [[None] * layer_count for _ in range(token_count)] if all(source_target_flags) else None
    )
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
        if source_target_probabilities is not None:
            source_target_probability = _number(
                cell,
                "source_target_probability",
                context=f"{context}.cells[]",
            )
            if not 0.0 <= source_target_probability <= 1.0:
                raise ValueError(f"{context} contains an out-of-range source-target probability")
            source_target_probabilities[token][layer] = source_target_probability
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
    if source_target_probabilities is not None and any(
        value is None for row in source_target_probabilities for value in row
    ):
        raise ValueError(f"{context} contains an incomplete source-target probability grid")
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
    if source_target_probabilities is not None:
        compact["source_target_probabilities"] = source_target_probabilities
    optional_metadata = (
        "source_correct_choice_index",
        "recipient_correct_choice_index",
        "source_choice_function_ids",
        "source_choice_texts",
        "source_question_id",
        "source_question",
        "source_format",
        "source_label_relation",
        "source_context_id",
        "source_context",
    )
    for key in optional_metadata:
        if key in record:
            compact[key] = record[key]
    if "answer_logit_lens" in record:
        lens = _mapping(record["answer_logit_lens"], context=f"{context}.answer_logit_lens")
        if lens.get("kind") != "five_way_answer_label" or lens.get("labels") != [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]:
            raise ValueError(f"{context} has unsupported answer-logit-lens semantics")
        top_p = _number(lens, "display_top_p", context=f"{context}.answer_logit_lens")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"{context} answer-logit-lens top-p must lie in (0, 1]")
        for side in ("source_probabilities", "recipient_probabilities"):
            values = lens.get(side)
            if not isinstance(values, list) or len(values) != token_count:
                raise ValueError(f"{context} answer logit lens has the wrong token count")
            for token_rows in values:
                if not isinstance(token_rows, list) or len(token_rows) != layer_count:
                    raise ValueError(f"{context} answer logit lens has the wrong layer count")
                for distribution in token_rows:
                    if (
                        not isinstance(distribution, list)
                        or len(distribution) != 5
                        or any(
                            not isinstance(value, int | float)
                            or not math.isfinite(value)
                            or not 0.0 <= value <= 1.0
                            for value in distribution
                        )
                        or not math.isclose(sum(distribution), 1.0, abs_tol=1e-5)
                    ):
                        raise ValueError(
                            f"{context} answer logit lens contains an invalid A-E distribution"
                        )
        compact["answer_logit_lens"] = lens
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


def _write_compact_float32(path: Path, values: list[float]) -> tuple[str, int]:
    """Atomically write one deterministic little-endian packed-float detail chunk."""

    packed = array("f", values)
    if packed.itemsize != 4:
        raise RuntimeError("the platform float array is not IEEE-754 binary32")
    if sys.byteorder != "little":
        packed.byteswap()
    serialized = packed.tobytes()
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


def _compact_representation_alignment_record(
    record: PatchRecord,
    *,
    context: str,
) -> PatchRecord:
    """Compact one complete observational token-by-layer alignment record."""

    cells = record.get("cells")
    if not isinstance(cells, list) or not cells:
        raise TypeError(f"{context}.cells must be a non-empty array")
    mapped_cells = [_mapping(cell, context=f"{context}.cells[]") for cell in cells]
    required = (
        "function_id",
        "source_function_id",
        "recipient_function_id",
        "token_axis",
    )
    if any(key not in record for key in required):
        raise KeyError(f"{context} lacks compact alignment metadata")
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
    matrices: dict[str, list[list[float | None]]] = {
        metric: [[None] * layer_count for _ in range(token_count)]
        for metric in (*REPRESENTATION_ALIGNMENT_METRICS, "source_norm", "recipient_norm")
    }
    token_positions: list[PatchRecord | None] = [None] * token_count
    for cell in mapped_cells:
        layer = int(_number(cell, "layer", context=f"{context}.cells[]"))
        token = int(_number(cell, "token_reverse_index", context=f"{context}.cells[]"))
        for metric, matrix in matrices.items():
            value = _number(cell, metric, context=f"{context}.cells[]")
            if not math.isfinite(value):
                raise ValueError(f"{context} contains a non-finite {metric}")
            if metric == "cosine_similarity" and not -1.0 <= value <= 1.0:
                raise ValueError(f"{context} contains an out-of-range cosine")
            if metric != "cosine_similarity" and value < 0.0:
                raise ValueError(f"{context} contains a negative norm or distance")
            if metric in {"source_norm", "recipient_norm"} and value == 0.0:
                raise ValueError(f"{context} contains a zero activation norm")
            if matrix[token][layer] is not None:
                raise ValueError(f"{context} contains a duplicate layer/token cell")
            matrix[token][layer] = value
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
    if any(value is None for matrix in matrices.values() for row in matrix for value in row):
        raise ValueError(f"{context} contains an incomplete alignment grid")
    if any(position is None for position in token_positions):
        raise ValueError(f"{context} contains an incomplete token axis")
    compact: PatchRecord = {
        **{key: record[key] for key in required},
        "token_positions": token_positions,
        "cosine_similarities": matrices["cosine_similarity"],
        "l2_distances": matrices["l2_distance"],
        "source_norms": matrices["source_norm"],
        "recipient_norms": matrices["recipient_norm"],
    }
    optional_metadata = (
        "recipient_choice_function_ids",
        "recipient_correct_choice_index",
        "source_correct_choice_index",
        "source_choice_function_ids",
        "source_choice_texts",
        "source_question_id",
        "source_question",
        "source_format",
        "source_label_relation",
        "source_context_id",
        "source_context",
    )
    for key in optional_metadata:
        if key in record:
            compact[key] = record[key]
    return compact


def _export_representation_alignments(
    root: Path,
) -> tuple[dict[str, object], int, dict[str, object]]:
    """Export measured alignment sidecars and robust boundary-specific L2 scales."""

    manifest: dict[str, object] = {}
    scale_observations: dict[tuple[str, str], dict[str, list[float]]] = {}
    file_count = 0
    pattern = "artifacts/runs/*/*/seed_*/representation_alignment/**/donor_*.json"
    valid_models = {key.value for key in ModelKey}
    valid_conditions = {item.value for item in TrainingCondition}
    valid_modes = {item.value for item in PatchingMode}
    valid_interfaces = {item.value for item in REPRESENTATION_ALIGNMENT_INTERFACES}
    expected_function_ids = {function.function_id for function in FUNCTIONS}
    for path in sorted(root.glob(pattern)):
        artifact = _mapping(read_json(path), context=str(path))
        run = _mapping(artifact.get("run"), context=f"{path}.run")
        plan = _mapping(artifact.get("plan"), context=f"{path}.plan")
        measurement = _mapping(artifact.get("measurement"), context=f"{path}.measurement")
        model = run.get("model")
        condition = run.get("condition")
        interface = plan.get("interface")
        mode = plan.get("mode")
        records = artifact.get("records")
        if artifact.get("schema_version") != REPRESENTATION_ALIGNMENT_SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported representation-alignment schema")
        if model not in valid_models or not isinstance(model, str):
            raise TypeError(f"{path}.run.model is invalid")
        if condition not in valid_conditions or not isinstance(condition, str):
            raise TypeError(f"{path}.run.condition is invalid")
        if mode not in valid_modes or not isinstance(mode, str):
            raise TypeError(f"{path}.plan.mode is invalid")
        if interface not in valid_interfaces or not isinstance(interface, str):
            raise TypeError(f"{path}.plan.interface is not an activation boundary")
        if plan.get("patch_position") != PATCH_POSITION:
            raise ValueError(f"{path}.plan.patch_position is invalid")
        if (
            measurement.get("kind") != REPRESENTATION_ALIGNMENT_KIND
            or measurement.get("causal_intervention") is not False
            or measurement.get("metrics") != list(REPRESENTATION_ALIGNMENT_METRICS)
            or measurement.get("accumulation_dtype") != REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE
        ):
            raise ValueError(f"{path}.measurement does not match the alignment contract")
        if not isinstance(records, list):
            raise TypeError(f"{path}.records must be an array")
        recipient_step = int(_number(plan, "recipient_step", context=f"{path}.plan"))
        donor_step = int(_number(artifact, "donor_step", context=str(path)))
        by_function: dict[str, PatchRecord] = {}
        for raw_record in records:
            record = _mapping(raw_record, context=f"{path}.records[]")
            function_id = record.get("function_id")
            if not isinstance(function_id, str) or function_id in by_function:
                raise ValueError(f"{path} has a missing or duplicate function_id")
            by_function[function_id] = _compact_representation_alignment_record(
                record,
                context=f"{path}.records[{function_id}]",
            )
        if set(by_function) != expected_function_ids:
            raise ValueError(
                f"{path} must contain exactly the {len(expected_function_ids)} registered functions"
            )
        relative_path = (
            Path("data")
            / "representation-alignment"
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
            "kind": "representation_alignment",
            "sha256": digest,
            "url": relative_path.as_posix(),
        }
        summary = _mapping(measurement.get("summary"), context=f"{path}.measurement.summary")
        l2_summary = _mapping(summary.get("l2_distance"), context=f"{path}.l2_summary")
        observations = scale_observations.setdefault(
            (model, interface),
            {"p95": [], "max": []},
        )
        observations["p95"].append(_number(l2_summary, "p95", context=f"{path}.l2_summary"))
        observations["max"].append(_number(l2_summary, "max", context=f"{path}.l2_summary"))
        file_count += 1
    scales: dict[str, object] = {}
    for (model, interface), observations in sorted(scale_observations.items()):
        model_bucket = cast(dict[str, object], scales.setdefault(model, {}))
        robust_max = max(observations["p95"])
        observed_max = max(observations["max"])
        if not math.isfinite(robust_max) or robust_max <= 0.0:
            raise ValueError(f"{model}/{interface} has no positive finite L2 display scale")
        model_bucket[interface] = {
            "cosine_similarity": {
                "min": -1.0,
                "max": 1.0,
                "basis": "theoretical_range",
            },
            "l2_distance": {
                "min": 0.0,
                "max": robust_max,
                "observed_max": observed_max,
                "basis": "maximum_artifact_p95_for_model_and_boundary",
            },
        }
    return manifest, file_count, scales


def _weight_alignment_site_axis(model: ModelKey) -> dict[str, object]:
    spec = MODEL_SPECS[model]
    components = weight_component_specs(model)
    return {
        "component_axis": [
            {
                "id": component.component_id,
                "label": component.label,
                "placement": component.placement,
                "tensor_rank": component.tensor_rank,
                "shape": component.shape,
                "parameter_template": component.parameter_template,
                "frozen_during_lora": component.frozen_during_lora,
                "row_group_size": component.row_group_size,
                "column_group_size": component.column_group_size,
                "group_label": component.group_label,
            }
            for component in components
        ],
        "column_axis": [
            {"id": "input", "label": "input", "kind": "global_input"},
            *[
                {
                    "id": f"layer_{layer}",
                    "label": str(layer),
                    "kind": "decoder_layer",
                    "layer": layer,
                }
                for layer in range(spec.layer_count)
            ],
            {"id": "output", "label": "output", "kind": "global_output"},
        ],
        "decoder_layer_count": spec.layer_count,
        "covered_parameter_tensors": sum(
            spec.layer_count if component.placement == "layer" else 1 for component in components
        ),
        "all_non_projection_weights_frozen": True,
    }


def _population_variance(values: list[float], mean: float) -> float:
    if not values or not math.isfinite(mean):
        raise ValueError("population variance requires finite values and mean")
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("population variance must be finite and nonnegative")
    return variance


def _linear_percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires values and a quantile in [0, 1]")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * quantile
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _export_weight_alignments(
    root: Path,
) -> tuple[dict[str, object], int, dict[str, object], dict[str, object]]:
    """Export scalar effective-weight grids and separately lazy-loaded axis details."""

    manifest: dict[str, object] = {}
    observations: dict[tuple[str, str], dict[str, list[float]]] = {}
    variance_observations: dict[tuple[str, str], list[float]] = {}
    axes: dict[str, object] = {}
    for model_key in ModelKey:
        try:
            axes[model_key.value] = _weight_alignment_site_axis(model_key)
        except ValueError:
            # Provisional architectures remain absent until an exact tensor inventory is registered.
            continue
    file_count = 0
    pattern = "artifacts/runs/*/*/seed_*/weight_alignment/**/step_high_*.json"
    valid_models = {key.value for key in ModelKey}
    valid_conditions = {item.value for item in TrainingCondition}
    for path in sorted(root.glob(pattern)):
        artifact = _mapping(read_json(path), context=str(path))
        run = _mapping(artifact.get("run"), context=f"{path}.run")
        pair = _mapping(artifact.get("checkpoint_pair"), context=f"{path}.checkpoint_pair")
        measurement = _mapping(artifact.get("measurement"), context=f"{path}.measurement")
        model = run.get("model")
        condition = run.get("condition")
        if artifact.get("schema_version") != WEIGHT_ALIGNMENT_SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported weight-alignment schema")
        if model not in valid_models or not isinstance(model, str):
            raise TypeError(f"{path}.run.model is invalid")
        if condition not in valid_conditions or not isinstance(condition, str):
            raise TypeError(f"{path}.run.condition is invalid")
        if (
            measurement.get("kind") != WEIGHT_ALIGNMENT_KIND
            or measurement.get("causal_intervention") is not False
            or measurement.get("prompt_dependent") is not False
            or measurement.get("function_dependent") is not False
            or measurement.get("metrics") != list(WEIGHT_ALIGNMENT_METRICS)
            or measurement.get("detail_metrics") != list(WEIGHT_ALIGNMENT_DETAIL_METRICS)
            or measurement.get("degenerate_counts") != list(WEIGHT_ALIGNMENT_DEGENERATE_COUNTS)
            or measurement.get("cosine_zero_norm_convention")
            != WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION
            or measurement.get("accumulation_dtype") != WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE
        ):
            raise ValueError(f"{path}.measurement does not match the weight contract")
        step_low = int(_number(pair, "step_low", context=f"{path}.checkpoint_pair"))
        step_high = int(_number(pair, "step_high", context=f"{path}.checkpoint_pair"))
        if (
            step_low >= step_high
            or step_low not in CHECKPOINT_STEPS
            or step_high not in CHECKPOINT_STEPS
            or pair.get("canonical_unordered_pair") is not True
            or pair.get("symmetric") is not True
        ):
            raise ValueError(f"{path}.checkpoint_pair is not a canonical registered pair")
        matrix_axis = artifact.get("matrix_axis")
        layer_count = artifact.get("layer_count")
        cells = artifact.get("cells")
        if (
            not isinstance(matrix_axis, list)
            or not all(isinstance(name, str) for name in matrix_axis)
            or matrix_axis != list(WEIGHT_ALIGNMENT_MATRIX_NAMES)
        ):
            raise ValueError(f"{path}.matrix_axis does not match the projection contract")
        matrix_names = cast(list[str], matrix_axis)
        if (
            not isinstance(layer_count, int)
            or layer_count != MODEL_SPECS[ModelKey(model)].layer_count
        ):
            raise ValueError(f"{path}.layer_count is invalid")
        if not isinstance(cells, list) or len(cells) != layer_count * len(matrix_names):
            raise ValueError(f"{path}.cells is incomplete")

        model_key = ModelKey(model)
        site_axis = _weight_alignment_site_axis(model_key)
        existing_axis = axes.setdefault(model, site_axis)
        if existing_axis != site_axis:
            raise ValueError(f"{path} conflicts with the registered complete-weight axis")
        components = weight_component_specs(model_key)
        component_ids = tuple(component.component_id for component in components)
        component_index = {component_id: index for index, component_id in enumerate(component_ids)}
        column_count = layer_count + 2

        scalar_matrices: dict[str, list[list[float | None]]] = {
            metric: [[None] * column_count for _ in components]
            for metric in WEIGHT_ALIGNMENT_METRICS
        }
        degenerate_matrices: dict[str, list[list[int | None]]] = {
            metric: [[None] * column_count for _ in components]
            for metric in WEIGHT_ALIGNMENT_DEGENERATE_COUNTS
        }
        variance_matrices: dict[str, list[list[float | None]]] = {
            metric: [[None] * column_count for _ in components]
            for metric in WEIGHT_ALIGNMENT_VARIANCE_METRICS
        }
        shapes: list[list[list[int] | None]] = [[None] * column_count for _ in components]
        detail_cells: list[dict[str, object]] = []
        seen: set[tuple[int, str]] = set()
        for raw_cell in cells:
            cell = _mapping(raw_cell, context=f"{path}.cells[]")
            layer = int(_number(cell, "layer", context=f"{path}.cells[]"))
            weight_name = cell.get("weight_name")
            shape = cell.get("shape")
            if (
                not 0 <= layer < layer_count
                or not isinstance(weight_name, str)
                or weight_name not in WEIGHT_ALIGNMENT_MATRIX_NAMES
                or (layer, weight_name) in seen
            ):
                raise ValueError(f"{path} has an invalid or duplicate weight cell")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
            ):
                raise ValueError(f"{path} has an invalid matrix shape")
            shape_values = cast(list[int], shape)
            seen.add((layer, weight_name))
            weight_index = component_index[weight_name]
            column_index = layer + 1
            expected_shape = components[weight_index].shape
            if tuple(shape_values) != expected_shape:
                raise ValueError(
                    f"{path} has shape {tuple(shape_values)} for {weight_name}; "
                    f"expected {expected_shape}"
                )
            shapes[weight_index][column_index] = shape_values
            scalar_cell: dict[str, object] = {
                "layer": layer,
                "weight_name": weight_name,
            }
            for metric in WEIGHT_ALIGNMENT_METRICS:
                value = _number(cell, metric, context=f"{path}.cells[]")
                if not math.isfinite(value):
                    raise ValueError(f"{path} contains a non-finite {metric}")
                if "cosine" in metric and not -1.0 <= value <= 1.0:
                    raise ValueError(f"{path} contains an out-of-range {metric}")
                if "l2" in metric and value < 0.0:
                    raise ValueError(f"{path} contains a negative {metric}")
                scalar_matrices[metric][weight_index][column_index] = value
                scalar_cell[metric] = value
            for metric in WEIGHT_ALIGNMENT_DEGENERATE_COUNTS:
                count = cell.get(metric)
                axis_length = shape_values[0] if metric.startswith("row_") else shape_values[1]
                if not isinstance(count, int) or not 0 <= count <= axis_length:
                    raise ValueError(f"{path} contains an invalid {metric}")
                degenerate_matrices[metric][weight_index][column_index] = count
            detail_cell: dict[str, object] = {**scalar_cell, "shape": shape_values}
            for metric in WEIGHT_ALIGNMENT_DETAIL_METRICS:
                raw_values = cell.get(metric)
                expected = shape_values[0] if metric.startswith("row_") else shape_values[1]
                if not isinstance(raw_values, list) or len(raw_values) != expected:
                    raise ValueError(f"{path} contains an invalid {metric} axis")
                values = []
                for raw_value in raw_values:
                    if not isinstance(raw_value, int | float) or not math.isfinite(raw_value):
                        raise ValueError(f"{path} contains a non-finite {metric}")
                    value = float(raw_value)
                    if "cosine" in metric and not -1.0 <= value <= 1.0:
                        raise ValueError(f"{path} contains an out-of-range {metric}")
                    if "l2" in metric and value < 0.0:
                        raise ValueError(f"{path} contains a negative {metric}")
                    values.append(value)
                detail_cell[metric] = values
                mean_metric, variance_metric = WEIGHT_DETAIL_TO_VARIANCE[metric]
                mean_value = cast(float, scalar_cell[mean_metric])
                variance = _population_variance(values, mean_value)
                variance_matrices[variance_metric][weight_index][column_index] = variance
                variance_observations.setdefault((model, variance_metric), []).append(variance)
            detail_cells.append(detail_cell)
        if len(seen) != layer_count * len(matrix_names):
            raise ValueError(f"{path} does not cover every layer/projection cell")
        for weight_index, component in enumerate(components):
            if component.placement == "input":
                valid_columns = (0,)
            elif component.placement == "output":
                valid_columns = (column_count - 1,)
            else:
                valid_columns = tuple(range(1, layer_count + 1))
            if component.frozen_during_lora:
                for column_index in valid_columns:
                    shapes[weight_index][column_index] = list(component.shape)
                    scalar_matrices["frobenius_cosine"][weight_index][column_index] = 1.0
                    scalar_matrices["frobenius_l2"][weight_index][column_index] = 0.0
                    if component.tensor_rank == 2:
                        for metric in (
                            "mean_row_cosine",
                            "mean_column_cosine",
                        ):
                            scalar_matrices[metric][weight_index][column_index] = 1.0
                        for metric in ("mean_row_l2", "mean_column_l2"):
                            scalar_matrices[metric][weight_index][column_index] = 0.0
                        for metric in WEIGHT_ALIGNMENT_DEGENERATE_COUNTS:
                            degenerate_matrices[metric][weight_index][column_index] = 0
                        for metric in WEIGHT_ALIGNMENT_VARIANCE_METRICS:
                            variance_matrices[metric][weight_index][column_index] = 0.0
            for column_index in range(column_count):
                valid = column_index in valid_columns
                shape = shapes[weight_index][column_index]
                if valid != (shape is not None):
                    raise ValueError(
                        f"{path} has an invalid placement for {component.component_id}"
                    )
                required_metrics = (
                    WEIGHT_ALIGNMENT_METRICS
                    if component.tensor_rank == 2
                    else ("frobenius_cosine", "frobenius_l2")
                )
                for metric in WEIGHT_ALIGNMENT_METRICS:
                    value = scalar_matrices[metric][weight_index][column_index]
                    if (valid and metric in required_metrics) != (value is not None):
                        raise ValueError(
                            f"{path} has an incomplete {metric} cell for {component.component_id}"
                        )
                for matrices in (degenerate_matrices, variance_matrices):
                    for metric, matrix in matrices.items():
                        value = matrix[weight_index][column_index]
                        expected = valid and component.tensor_rank == 2
                        if expected != (value is not None):
                            raise ValueError(
                                f"{path} has an incomplete {metric} cell for "
                                f"{component.component_id}"
                            )

        relative_root = (
            Path("data") / "weight-alignment" / model / condition / f"step_low_{step_low:06d}"
        )
        scalar_path = relative_root / f"step_high_{step_high:06d}.json"
        detail_references: dict[str, object] = {}
        detail_cell_index = {
            (cast(str, cell["weight_name"]), cast(int, cell["layer"])): cell
            for cell in detail_cells
        }
        for detail_metric in WEIGHT_ALIGNMENT_DETAIL_METRICS:
            detail_path = relative_root / f"step_high_{step_high:06d}.{detail_metric}.f32"
            packed_values = [
                value
                for weight_name in matrix_names
                for layer in range(layer_count)
                for value in cast(
                    list[float],
                    detail_cell_index[(weight_name, layer)][detail_metric],
                )
            ]
            detail_digest, detail_bytes = _write_compact_float32(
                root / "site" / detail_path,
                packed_values,
            )
            detail_references[detail_metric] = {
                "bytes": detail_bytes,
                "format": "float32_le",
                "layer_count": layer_count,
                "layout": "weight_major_then_layer_then_axis_index",
                "matrix_axis": matrix_axis,
                "metric": detail_metric,
                "sha256": detail_digest,
                "url": detail_path.as_posix(),
                "value_count": len(packed_values),
            }
        scalar_digest, scalar_bytes = _write_compact_json(
            root / "site" / scalar_path,
            {
                "component_axis": site_axis["component_axis"],
                "column_axis": site_axis["column_axis"],
                "column_count": column_count,
                "decoder_layer_count": layer_count,
                "shapes": shapes,
                "metrics": scalar_matrices,
                "variances": variance_matrices,
                "degenerate_counts": degenerate_matrices,
                "cosine_zero_norm_convention": WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION,
            },
        )
        reference = {
            "bytes": scalar_bytes,
            "details": detail_references,
            "kind": "weight_alignment",
            "sha256": scalar_digest,
            "url": scalar_path.as_posix(),
        }
        model_bucket = cast(dict[str, object], manifest.setdefault(model, {}))
        condition_bucket = cast(dict[str, object], model_bucket.setdefault(condition, {}))
        for recipient, donor in ((step_low, step_high), (step_high, step_low)):
            recipient_bucket = cast(
                dict[str, object], condition_bucket.setdefault(str(recipient), {})
            )
            recipient_bucket[str(donor)] = reference

        summary = _mapping(measurement.get("summary"), context=f"{path}.measurement.summary")
        for metric in WEIGHT_ALIGNMENT_METRICS:
            metric_summary = _mapping(summary.get(metric), context=f"{path}.{metric}.summary")
            metric_observations = observations.setdefault((model, metric), {"p95": [], "max": []})
            metric_observations["p95"].append(
                _number(metric_summary, "p95", context=f"{path}.{metric}.summary")
            )
            metric_observations["max"].append(
                _number(metric_summary, "max", context=f"{path}.{metric}.summary")
            )
        file_count += 1

    scales: dict[str, object] = {}
    for (model, metric), metric_observations in sorted(observations.items()):
        model_bucket = cast(dict[str, object], scales.setdefault(model, {}))
        if "cosine" in metric:
            model_bucket[metric] = {
                "min": 0.0,
                "max": 1.0,
                "basis": "requested_fixed_weight_cosine_range",
                "raw_values_below_minimum_are_color_clamped": True,
            }
            continue
        robust_max = max(metric_observations["p95"])
        observed_max = max(metric_observations["max"])
        if not math.isfinite(robust_max) or robust_max < 0.0:
            raise ValueError(f"{model}/{metric} has no finite nonnegative display scale")
        model_bucket[metric] = {
            "min": 0.0,
            "max": robust_max if robust_max > 0.0 else 1.0,
            "observed_max": observed_max,
            "basis": "maximum_artifact_p95_for_model_and_metric",
        }
    for (model, metric), values in sorted(variance_observations.items()):
        if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"{model}/{metric} has invalid variance observations")
        robust_max = _linear_percentile(values, 0.95)
        observed_max = max(values)
        model_bucket = cast(dict[str, object], scales.setdefault(model, {}))
        variance_bucket = cast(dict[str, object], model_bucket.setdefault("variances", {}))
        variance_bucket[metric] = {
            "min": 0.0,
            "max": robust_max if robust_max > 0.0 else 1.0,
            "observed_max": observed_max,
            "basis": "p95_across_measured_layer_component_cells",
            "statistic": "population_variance",
        }
    return manifest, file_count, scales, axes


def _compact_activation_neighbor_grid(
    value: object,
    candidates: list[dict[str, object]],
    *,
    position_count: int,
    top_k: int,
    context: str,
) -> tuple[list[list[list[list[float | int]]]], int]:
    if not isinstance(value, list) or len(value) != position_count or not value:
        raise ValueError(f"{context} must have exactly {position_count} token rows")
    layer_count: int | None = None
    compact: list[list[list[list[float | int]]]] = []
    for token_index, raw_layers in enumerate(value):
        if not isinstance(raw_layers, list) or not raw_layers:
            raise TypeError(f"{context}[{token_index}] must contain layer rows")
        if layer_count is None:
            layer_count = len(raw_layers)
        elif len(raw_layers) != layer_count:
            raise ValueError(f"{context} has an inconsistent layer count")
        compact_layers: list[list[list[float | int]]] = []
        for layer, raw_matches in enumerate(raw_layers):
            if not isinstance(raw_matches, list) or len(raw_matches) != top_k:
                raise ValueError(f"{context}[{token_index}][{layer}] must contain top-k matches")
            compact_matches: list[list[float | int]] = []
            seen_examples: set[int] = set()
            previous_score = math.inf
            for raw_match in raw_matches:
                match = _mapping(raw_match, context=f"{context}[][][]")
                example_index = int(_number(match, "example_index", context=context))
                matched_token = int(_number(match, "token_index", context=context))
                score = _number(match, "cosine_similarity", context=context)
                if not 0 <= example_index < len(candidates):
                    raise ValueError(f"{context} references an unknown candidate example")
                candidate_labels = candidates[example_index]["token_labels"]
                if not isinstance(candidate_labels, list) or not 0 <= matched_token < len(
                    candidate_labels
                ):
                    raise ValueError(f"{context} references an unknown candidate token")
                if example_index in seen_examples:
                    raise ValueError(f"{context} repeats one candidate prompt in a top-k list")
                if not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
                    raise ValueError(f"{context} contains an invalid cosine similarity")
                if score > previous_score + 1e-7:
                    raise ValueError(f"{context} top-k matches are not descending")
                seen_examples.add(example_index)
                previous_score = score
                compact_matches.append([example_index, matched_token, score])
            compact_layers.append(compact_matches)
        compact.append(compact_layers)
    if layer_count is None:  # pragma: no cover - non-empty rows are required above
        raise AssertionError("activation-neighbor grid unexpectedly has no layers")
    return compact, layer_count


def _export_activation_examples(
    root: Path,
) -> tuple[dict[str, object], int, int]:
    manifest: dict[str, object] = {}
    raw_file_count = 0
    chunk_count = 0
    pattern = "artifacts/runs/*/*/seed_*/activation_examples/sequence_end/*/**/checkpoint_*.json"
    expected_function_ids = {function.function_id for function in FUNCTIONS}
    for path in sorted(root.glob(pattern)):
        artifact = _mapping(read_json(path), context=str(path))
        run = _mapping(artifact.get("run"), context=f"{path}.run")
        model = run.get("model")
        condition = run.get("condition")
        interface = artifact.get("interface")
        raw_candidate_source = artifact.get(
            "candidate_source",
            ActivationExampleSource.EXPERIMENT.value,
        )
        if not isinstance(raw_candidate_source, str):
            raise TypeError(f"{path}.candidate_source must be a string")
        candidate_source = ActivationExampleSource(raw_candidate_source)
        checkpoint_step = int(_number(artifact, "checkpoint_step", context=str(path)))
        if not isinstance(model, str) or model not in {key.value for key in ModelKey}:
            raise TypeError(f"{path}.run.model is invalid")
        if not isinstance(condition, str) or condition not in {
            item.value for item in TrainingCondition
        }:
            raise TypeError(f"{path}.run.condition is invalid")
        if not isinstance(interface, str):
            raise TypeError(f"{path}.interface must be a string")
        parsed_interface = PatchingInterface(interface)
        if parsed_interface.patches_weights:
            raise ValueError(f"{path} cannot attach activation examples to a weight interface")
        if checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError(f"{path} uses an unregistered checkpoint")
        similarity = _mapping(artifact.get("similarity"), context=f"{path}.similarity")
        if similarity.get("metric") != ACTIVATION_EXAMPLE_METRIC:
            raise ValueError(f"{path} uses an unsupported activation-example metric")
        top_k = int(_number(similarity, "top_k", context=f"{path}.similarity"))
        raw_candidates = artifact.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < top_k:
            raise ValueError(f"{path}.candidates must contain at least top-k prompts")
        candidates: list[dict[str, object]] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _mapping(raw_candidate, context=f"{path}.candidates[{index}]")
            required = (
                "example_id",
                "category",
                "target",
                "rendered_prompt",
                "token_ids",
                "token_labels",
            )
            if any(key not in candidate for key in required):
                raise KeyError(f"{path}.candidates[{index}] lacks required metadata")
            if not all(
                isinstance(candidate[key], str)
                for key in ("example_id", "category", "target", "rendered_prompt")
            ):
                raise TypeError(f"{path}.candidates[{index}] has non-string prompt metadata")
            token_ids = candidate["token_ids"]
            token_labels = candidate["token_labels"]
            if (
                not isinstance(token_ids, list)
                or not isinstance(token_labels, list)
                or len(token_ids) != len(token_labels)
                or not token_ids
                or not all(isinstance(value, int) for value in token_ids)
                or not all(isinstance(value, str) for value in token_labels)
            ):
                raise ValueError(f"{path}.candidates[{index}] has an invalid token axis")
            compact_candidate = {key: candidate[key] for key in required}
            provenance = candidate.get("provenance")
            if provenance is not None:
                compact_provenance = _mapping(
                    provenance,
                    context=f"{path}.candidates[{index}].provenance",
                )
                provenance_required = (
                    "dataset",
                    "config",
                    "revision",
                    "split",
                    "row_index",
                    "document_id",
                    "url",
                    "dump",
                    "date",
                    "language",
                    "text_sha256",
                )
                if any(key not in compact_provenance for key in provenance_required):
                    raise KeyError(f"{path}.candidates[{index}].provenance lacks required metadata")
                if not isinstance(compact_provenance["row_index"], int) or not all(
                    isinstance(compact_provenance[key], str)
                    for key in provenance_required
                    if key != "row_index"
                ):
                    raise TypeError(f"{path}.candidates[{index}].provenance has invalid metadata")
                compact_candidate["provenance"] = {
                    key: compact_provenance[key] for key in provenance_required
                }
            elif candidate_source is ActivationExampleSource.FINEWEB:
                raise KeyError(f"{path}.candidates[{index}] lacks FineWeb provenance")
            candidates.append(compact_candidate)
        if len({candidate["example_id"] for candidate in candidates}) != len(candidates):
            raise ValueError(f"{path} contains duplicate activation-example IDs")

        relative_base = Path("data") / "activation-examples" / model / condition / interface
        if candidate_source is not ActivationExampleSource.EXPERIMENT:
            relative_base /= candidate_source.value
        relative_base /= f"checkpoint_step_{checkpoint_step:06d}"
        catalog_path = relative_base / "candidates.json"
        catalog_digest, catalog_bytes = _write_compact_json(
            root / "site" / catalog_path,
            {
                "checkpoint_step": checkpoint_step,
                "candidate_source": candidate_source.value,
                "candidate_corpus": artifact.get("candidate_corpus"),
                "candidates": candidates,
            },
        )
        catalog_reference = {
            "bytes": catalog_bytes,
            "sha256": catalog_digest,
            "url": catalog_path.as_posix(),
        }

        raw_records = artifact.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise TypeError(f"{path}.records must be a non-empty array")
        seen: set[tuple[str, str]] = set()
        seen_modes: set[str] = set()
        for raw_record in raw_records:
            record = _mapping(raw_record, context=f"{path}.records[]")
            mode = record.get("mode")
            function_id = record.get("function_id")
            if not isinstance(mode, str) or not isinstance(function_id, str):
                raise TypeError(f"{path} activation-example record lacks mode/function IDs")
            parsed_mode = PatchingMode(mode)
            if not parsed_mode.supports_independent_checkpoint_donor:
                raise ValueError(f"{path} activation examples require an answer-label mode")
            key = (mode, function_id)
            if key in seen:
                raise ValueError(f"{path} repeats activation-example record {key}")
            seen.add(key)
            seen_modes.add(mode)
            position_count = int(_number(record, "position_count", context=f"{path}.records[]"))
            source, source_layers = _compact_activation_neighbor_grid(
                record.get("source_neighbors"),
                candidates,
                position_count=position_count,
                top_k=top_k,
                context=f"{path}.records[{mode}/{function_id}].source",
            )
            recipient, recipient_layers = _compact_activation_neighbor_grid(
                record.get("recipient_neighbors"),
                candidates,
                position_count=position_count,
                top_k=top_k,
                context=f"{path}.records[{mode}/{function_id}].recipient",
            )
            if source_layers != recipient_layers:
                raise ValueError(f"{path} source/recipient activation-neighbor layers differ")
            relative_path = relative_base / mode / f"{function_id}.json"
            digest, byte_count = _write_compact_json(
                root / "site" / relative_path,
                {
                    "checkpoint_step": checkpoint_step,
                    "candidate_source": candidate_source.value,
                    "mode": mode,
                    "function_id": function_id,
                    "metric": ACTIVATION_EXAMPLE_METRIC,
                    "top_k": top_k,
                    "position_count": position_count,
                    "layer_count": source_layers,
                    "source_neighbors": source,
                    "recipient_neighbors": recipient,
                },
            )
            model_bucket = cast(dict[str, object], manifest.setdefault(model, {}))
            condition_bucket = cast(dict[str, object], model_bucket.setdefault(condition, {}))
            interface_bucket = cast(dict[str, object], condition_bucket.setdefault(interface, {}))
            source_bucket = cast(
                dict[str, object],
                interface_bucket.setdefault(candidate_source.value, {}),
            )
            mode_bucket = cast(dict[str, object], source_bucket.setdefault(mode, {}))
            step_bucket = cast(dict[str, object], mode_bucket.setdefault(str(checkpoint_step), {}))
            step_bucket[function_id] = {
                "neighbors": {
                    "bytes": byte_count,
                    "sha256": digest,
                    "url": relative_path.as_posix(),
                },
                "candidates": catalog_reference,
            }
            chunk_count += 1
        for mode in seen_modes:
            functions = {function_id for seen_mode, function_id in seen if seen_mode == mode}
            if functions != expected_function_ids:
                raise ValueError(f"{path} mode {mode} must contain all registered functions")
        raw_file_count += 1
    return manifest, raw_file_count, chunk_count


def _compact_vocabulary_logit_lens_side(
    value: object,
    *,
    vocabulary_size: int,
    top_k: int,
    token_labels: Mapping[str, object],
    context: str,
) -> tuple[dict[str, object], int, set[int]]:
    side = _mapping(value, context=context)
    position_count = int(_number(side, "position_count", context=context))
    token_indices = side.get("token_indices")
    token_ids = side.get("token_ids")
    raw_grid = side.get("top_tokens")
    if position_count <= 0:
        raise ValueError(f"{context}.position_count must be positive")
    if (
        not isinstance(token_indices, list)
        or len(token_indices) != position_count
        or not all(isinstance(value, int) for value in token_indices)
    ):
        raise ValueError(f"{context}.token_indices must be a reverse-contiguous axis")
    typed_token_indices = cast(list[int], token_indices)
    if typed_token_indices != list(range(typed_token_indices[0], typed_token_indices[-1] - 1, -1)):
        raise ValueError(f"{context}.token_indices must be a reverse-contiguous axis")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != position_count
        or not all(isinstance(value, int) and value >= 0 for value in token_ids)
    ):
        raise ValueError(f"{context}.token_ids must match the token axis")
    if not isinstance(raw_grid, list) or len(raw_grid) != position_count:
        raise ValueError(f"{context}.top_tokens must match the token axis")

    layer_count: int | None = None
    used_token_ids: set[int] = set()
    compact_grid: list[list[list[list[float | int]]]] = []
    for token_index, raw_layers in enumerate(raw_grid):
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ValueError(f"{context}.top_tokens[{token_index}] must contain layer rows")
        if layer_count is None:
            layer_count = len(raw_layers)
        elif len(raw_layers) != layer_count:
            raise ValueError(f"{context}.top_tokens has an inconsistent layer count")
        compact_layers: list[list[list[float | int]]] = []
        for layer, raw_tokens in enumerate(raw_layers):
            if not isinstance(raw_tokens, list) or len(raw_tokens) != top_k:
                raise ValueError(
                    f"{context}.top_tokens[{token_index}][{layer}] must contain top-k tokens"
                )
            compact_tokens: list[list[float | int]] = []
            seen: set[int] = set()
            previous_probability = math.inf
            displayed_mass = 0.0
            for raw_token in raw_tokens:
                if (
                    not isinstance(raw_token, list)
                    or len(raw_token) != 2
                    or not isinstance(raw_token[0], int)
                    or not isinstance(raw_token[1], int | float)
                ):
                    raise TypeError(f"{context} contains a malformed top-token entry")
                output_token_id = raw_token[0]
                probability = float(raw_token[1])
                if not 0 <= output_token_id < vocabulary_size:
                    raise ValueError(f"{context} references a token outside the output vocabulary")
                if output_token_id in seen:
                    raise ValueError(f"{context} repeats one token in a top-k list")
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{context} contains an invalid full-vocabulary probability")
                if probability > previous_probability + 1e-8:
                    raise ValueError(f"{context} top-k probabilities are not descending")
                label = token_labels.get(str(output_token_id))
                if not isinstance(label, str) or not label:
                    raise ValueError(f"{context} lacks a decoded label for token {output_token_id}")
                seen.add(output_token_id)
                used_token_ids.add(output_token_id)
                previous_probability = probability
                displayed_mass += probability
                compact_tokens.append([output_token_id, probability])
            if displayed_mass > 1.00001:
                raise ValueError(f"{context} top-k displayed probability mass exceeds one")
            compact_layers.append(compact_tokens)
        compact_grid.append(compact_layers)
    if layer_count is None:  # pragma: no cover - non-empty rows are required above
        raise AssertionError("vocabulary logit-lens side unexpectedly has no layers")
    return (
        {
            "position_count": position_count,
            "token_indices": typed_token_indices,
            "token_ids": token_ids,
            "top_tokens": compact_grid,
        },
        layer_count,
        used_token_ids,
    )


def _export_vocabulary_logit_lenses(
    root: Path,
) -> tuple[dict[str, object], int, int]:
    manifest: dict[str, object] = {}
    raw_file_count = 0
    chunk_count = 0
    pattern = "artifacts/runs/*/*/seed_*/vocabulary_logit_lens/sequence_end/checkpoint_*.json"
    expected_function_ids = {function.function_id for function in FUNCTIONS}
    expected_modes = tuple(mode.value for mode in VOCABULARY_LOGIT_LENS_MODES)
    for path in sorted(root.glob(pattern)):
        artifact = _mapping(read_json(path), context=str(path))
        run = _mapping(artifact.get("run"), context=f"{path}.run")
        model = run.get("model")
        condition = run.get("condition")
        checkpoint_step = int(_number(artifact, "checkpoint_step", context=str(path)))
        if not isinstance(model, str) or model not in {key.value for key in ModelKey}:
            raise TypeError(f"{path}.run.model is invalid")
        if not isinstance(condition, str) or condition not in {
            item.value for item in TrainingCondition
        }:
            raise TypeError(f"{path}.run.condition is invalid")
        if checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError(f"{path} uses an unregistered checkpoint")
        raw_modes = artifact.get("modes")
        if (
            not isinstance(raw_modes, list)
            or not raw_modes
            or any(not isinstance(mode, str) for mode in raw_modes)
        ):
            raise TypeError(f"{path}.modes must be a non-empty string array")
        observed_modes = tuple(cast(list[str], raw_modes))
        observed_set = set(observed_modes)
        if len(observed_set) != len(observed_modes) or observed_modes != tuple(
            mode for mode in expected_modes if mode in observed_set
        ):
            raise ValueError(f"{path} prompt-source lenses are unknown or out of order")
        lens = _mapping(artifact.get("lens"), context=f"{path}.lens")
        if lens.get("kind") != "full_vocabulary_top_k":
            raise ValueError(f"{path} has unsupported vocabulary logit-lens semantics")
        normalization = lens.get("normalization")
        residual_boundary = lens.get("residual_boundary")
        if (
            not isinstance(normalization, str)
            or "every model output-embedding row" not in normalization
        ):
            raise ValueError(f"{path} does not declare full-vocabulary normalization")
        if not isinstance(residual_boundary, str):
            raise TypeError(f"{path}.lens.residual_boundary must be a string")
        top_k = int(_number(lens, "top_k", context=f"{path}.lens"))
        vocabulary_size = int(_number(lens, "vocabulary_size", context=f"{path}.lens"))
        if top_k <= 0 or vocabulary_size <= top_k:
            raise ValueError(f"{path} has invalid top-k/vocabulary-size metadata")
        raw_token_labels = artifact.get("token_labels")
        if not isinstance(raw_token_labels, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value
            for key, value in raw_token_labels.items()
        ):
            raise TypeError(f"{path}.token_labels must map token IDs to decoded strings")
        token_labels = cast(dict[str, object], raw_token_labels)
        raw_records = artifact.get("records")
        if not isinstance(raw_records, list):
            raise TypeError(f"{path}.records must be an array")
        seen_functions: set[str] = set()
        for raw_record in raw_records:
            record = _mapping(raw_record, context=f"{path}.records[]")
            function_id = record.get("function_id")
            if not isinstance(function_id, str) or function_id in seen_functions:
                raise ValueError(f"{path} has a missing or duplicate function ID")
            seen_functions.add(function_id)
            clean, layer_count, used_ids = _compact_vocabulary_logit_lens_side(
                record.get("clean"),
                vocabulary_size=vocabulary_size,
                top_k=top_k,
                token_labels=token_labels,
                context=f"{path}.records[{function_id}].clean",
            )
            if layer_count != MODEL_SPECS[ModelKey(model)].layer_count:
                raise ValueError(f"{path} vocabulary logit lens has the wrong decoder layer count")
            raw_sources = _mapping(
                record.get("sources"),
                context=f"{path}.records[{function_id}].sources",
            )
            if set(raw_sources) != observed_set:
                raise ValueError(
                    f"{path} function {function_id} disagrees with its declared prompt sources"
                )
            sources: dict[str, object] = {}
            for mode in observed_modes:
                source, source_layers, source_ids = _compact_vocabulary_logit_lens_side(
                    raw_sources[mode],
                    vocabulary_size=vocabulary_size,
                    top_k=top_k,
                    token_labels=token_labels,
                    context=f"{path}.records[{function_id}].sources[{mode}]",
                )
                if source_layers != layer_count:
                    raise ValueError(f"{path} source and clean logit-lens layers differ")
                sources[mode] = source
                used_ids.update(source_ids)
            relative_path = (
                Path("data")
                / "vocabulary-logit-lens"
                / model
                / condition
                / f"checkpoint_step_{checkpoint_step:06d}"
                / f"{function_id}.json"
            )
            digest, byte_count = _write_compact_json(
                root / "site" / relative_path,
                {
                    "checkpoint_step": checkpoint_step,
                    "function_id": function_id,
                    "kind": "full_vocabulary_top_k",
                    "normalization": normalization,
                    "residual_boundary": residual_boundary,
                    "top_k": top_k,
                    "vocabulary_size": vocabulary_size,
                    "layer_count": layer_count,
                    "token_labels": {
                        str(token_id): token_labels[str(token_id)] for token_id in sorted(used_ids)
                    },
                    "clean": clean,
                    "sources": sources,
                },
            )
            model_bucket = cast(dict[str, object], manifest.setdefault(model, {}))
            condition_bucket = cast(dict[str, object], model_bucket.setdefault(condition, {}))
            step_bucket = cast(
                dict[str, object], condition_bucket.setdefault(str(checkpoint_step), {})
            )
            step_bucket[function_id] = {
                "bytes": byte_count,
                "sha256": digest,
                "url": relative_path.as_posix(),
            }
            chunk_count += 1
        if seen_functions != expected_function_ids:
            raise ValueError(f"{path} must contain exactly the registered function set")
        raw_file_count += 1
    return manifest, raw_file_count, chunk_count


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

    batch_letter_propensity_curves: dict[str, dict[str, dict[str, list[LetterPropensityRow]]]] = {}
    batch_letter_propensity_sources: dict[str, dict[str, dict[str, str]]] = {}
    rank_letter_propensity_curves: dict[str, dict[str, dict[str, list[LetterPropensityRow]]]] = {}
    rank_letter_propensity_sources: dict[str, dict[str, dict[str, str]]] = {}
    measured_letter_runs: set[tuple[str, str, int, int | None]] = set()
    for model in ModelKey:
        batch_letter_propensity_curves[model.value] = {}
        batch_letter_propensity_sources[model.value] = {}
        rank_letter_propensity_curves[model.value] = {}
        rank_letter_propensity_sources[model.value] = {}
        for condition in TrainingCondition:
            batch_letter_propensity_curves[model.value][condition.value] = {}
            batch_letter_propensity_sources[model.value][condition.value] = {}
            rank_letter_propensity_curves[model.value][condition.value] = {}
            rank_letter_propensity_sources[model.value][condition.value] = {}

            batch_runs = (
                RunKey(model.value, condition),
                *(
                    RunKey(model.value, condition, effective_batch_size=batch_size)
                    for batch_size in BATCH_ABLATION_SIZES
                ),
            )
            for run in batch_runs:
                measured = _real_letter_propensity_curve(root, run)
                if measured is None:
                    continue
                rows, source = measured
                key = str(run.effective_batch_size)
                batch_letter_propensity_curves[model.value][condition.value][key] = rows
                batch_letter_propensity_sources[model.value][condition.value][key] = source
                measured_letter_runs.add(
                    (run.model, run.condition.value, run.effective_batch_size, run.lora_rank)
                )

            rank_runs = [RunKey(model.value, condition)]
            if condition is TrainingCondition.CORRECT:
                rank_runs.extend(
                    RunKey(model.value, condition, lora_rank=rank)
                    for rank in LORA_RANKS
                    if rank != DEFAULT_LORA_RANK
                )
            for run in rank_runs:
                measured = _real_letter_propensity_curve(root, run)
                if measured is None:
                    continue
                rows, source = measured
                key = "full" if run.lora_rank is None else str(run.lora_rank)
                rank_letter_propensity_curves[model.value][condition.value][key] = rows
                rank_letter_propensity_sources[model.value][condition.value][key] = source
                measured_letter_runs.add(
                    (run.model, run.condition.value, run.effective_batch_size, run.lora_rank)
                )
    patch_manifest, real_patch_files = _export_real_patches(root)
    (
        representation_alignment_manifest,
        real_representation_alignment_files,
        representation_alignment_scales,
    ) = _export_representation_alignments(root)
    (
        weight_alignment_manifest,
        real_weight_alignment_files,
        weight_alignment_scales,
        weight_alignment_axes,
    ) = _export_weight_alignments(root)
    (
        activation_example_manifest,
        real_activation_example_files,
        activation_example_chunks,
    ) = _export_activation_examples(root)
    (
        vocabulary_logit_lens_manifest,
        real_vocabulary_logit_lens_files,
        vocabulary_logit_lens_chunks,
    ) = _export_vocabulary_logit_lenses(root)
    write_json(
        root / "site" / "data" / "patch-manifest.json",
        {
            "real_patch_files": real_patch_files,
            "patch_manifest": patch_manifest,
            "real_representation_alignment_files": real_representation_alignment_files,
            "representation_alignment_manifest": representation_alignment_manifest,
            "representation_alignment_scales": representation_alignment_scales,
            "real_weight_alignment_files": real_weight_alignment_files,
            "weight_alignment_manifest": weight_alignment_manifest,
            "weight_alignment_scales": weight_alignment_scales,
            "weight_alignment_axes": weight_alignment_axes,
            "real_activation_example_files": real_activation_example_files,
            "activation_example_chunks": activation_example_chunks,
            "activation_example_manifest": activation_example_manifest,
            "real_vocabulary_logit_lens_files": real_vocabulary_logit_lens_files,
            "vocabulary_logit_lens_chunks": vocabulary_logit_lens_chunks,
            "vocabulary_logit_lens_manifest": vocabulary_logit_lens_manifest,
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
            "real_representation_alignment_files": real_representation_alignment_files,
            "real_weight_alignment_files": real_weight_alignment_files,
            "real_activation_example_files": real_activation_example_files,
            "activation_example_chunks": activation_example_chunks,
            "real_vocabulary_logit_lens_files": real_vocabulary_logit_lens_files,
            "vocabulary_logit_lens_chunks": vocabulary_logit_lens_chunks,
            "real_letter_propensity_runs": len(measured_letter_runs),
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
            "letter_propensity": {
                "metric": LETTER_PROPENSITY_METRIC,
                "answer_labels": LETTER_PROPENSITY_LABELS,
                "normalization": LETTER_PROPENSITY_NORMALIZATION,
                "aggregation": LETTER_PROPENSITY_AGGREGATION,
                "position_policy": LETTER_PROPENSITY_POSITION_POLICY,
                "corpus": {
                    "dataset": FINEWEB_DATASET_ID,
                    "revision": FINEWEB_DATASET_REVISION,
                    "config": FINEWEB_DATASET_CONFIG,
                    "split": FINEWEB_DATASET_SPLIT,
                    "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
                    "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
                    "max_tokens_per_document": FINEWEB_ACTIVATION_MAX_TOKENS,
                    "input_format": "raw document; no chat template",
                },
            },
            "batch_ablation": {
                "effective_batch_sizes": [EFFECTIVE_BATCH_SIZE, *BATCH_ABLATION_SIZES],
                "measured_runs": measured_batch_runs,
                "curves": batch_curves,
                "function_curves": batch_function_curves,
                "curve_sources": batch_curve_sources,
                "letter_propensity_curves": batch_letter_propensity_curves,
                "letter_propensity_sources": batch_letter_propensity_sources,
            },
            "rank_ablation": {
                "lora_ranks": [*LORA_RANKS, "full"],
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "measured_runs": measured_rank_runs,
                "curves": rank_curves,
                "function_curves": rank_function_curves,
                "curve_sources": rank_curve_sources,
                "letter_propensity_curves": rank_letter_propensity_curves,
                "letter_propensity_sources": rank_letter_propensity_sources,
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
            "representation_alignment_manifest": representation_alignment_manifest,
            "representation_alignment_scales": representation_alignment_scales,
            "weight_alignment_manifest": weight_alignment_manifest,
            "weight_alignment_scales": weight_alignment_scales,
            "weight_alignment_axes": weight_alignment_axes,
            "activation_example_manifest": activation_example_manifest,
            "vocabulary_logit_lens_manifest": vocabulary_logit_lens_manifest,
        },
    )


if __name__ == "__main__":
    main()
