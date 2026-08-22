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
from oocr_training_dynamics.answer_lookup import (
    ANSWER_LABELS,
    ANSWER_LOOKUP_CHECKPOINT_STEP,
    ANSWER_LOOKUP_INTERFACES,
    ANSWER_LOOKUP_SCHEMA_VERSION,
    AnswerLookupSource,
)
from oocr_training_dynamics.artifacts import read_json, run_dir, sha256_file, write_json
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
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.fourier_hardware_lineage import (
    LINEAGE_ID_PATTERN,
    load_hardware_lineage_plan,
)
from oocr_training_dynamics.fourier_networks import (
    cluster_minset_hypergraph_networks,
)
from oocr_training_dynamics.fourier_subset_index import (
    MAX_PROPER_SUBSET_PROBABILITY_FRACTION,
    RelativeProperSubsetCriterion,
    SubsetMetric,
    ensure_subset_metric_index,
    maximum_proper_subset_metric,
    passes_relative_proper_subset_criterion,
    subset_index_path,
)
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
from oocr_training_dynamics.runtime_fourier_disconnected import (
    DISCONNECTED_SEARCH_SCHEMA_VERSION,
)
from oocr_training_dynamics.runtime_fourier_frontier import (
    FRONTIER_RESULT_FILENAME,
    FRONTIER_SCHEMA_VERSION,
    load_frontier_metric_index,
)
from oocr_training_dynamics.runtime_fourier_residual import (
    NETWORK_VETO_SCHEMA_VERSION,
)
from oocr_training_dynamics.runtime_models import load_processor
from oocr_training_dynamics.runtime_patching import (
    VOCABULARY_LOGIT_LENS_MODES,
    build_token_axis_metadata,
)
from oocr_training_dynamics.switched_answer_minsets import (
    SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
    SWITCHED_ANSWER_INTERFACES,
    SWITCHED_ANSWER_SCHEMA_VERSION,
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
    weight_site_component_specs,
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
ANSWER_LOOKUP_INTERVENTION_COUNT = 27


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _number(mapping: dict[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{context}.{key} must be numeric")
    return float(value)


def _site_export_source_sha256() -> str:
    source_root = Path(__file__).resolve().parents[1]
    paths = sorted(
        (
            *source_root.joinpath("oocr_training_dynamics").glob("**/*.py"),
            *source_root.joinpath("scripts").glob("*.py"),
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not paths:
        raise RuntimeError("site exporter source bundle is empty")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


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
    inventory = weight_component_specs(model)
    components = weight_site_component_specs(model)
    registered_tensor_count = sum(
        spec.layer_count if component.placement == "layer" else 1 for component in inventory
    )
    displayed_tensor_count = sum(
        spec.layer_count if component.placement == "layer" else 1 for component in components
    )
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
        "registered_parameter_tensors": registered_tensor_count,
        "omitted_frozen_norm_tensors": registered_tensor_count - displayed_tensor_count,
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
        components = weight_site_component_specs(model_key)
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


def _validated_answer_lookup_probabilities(
    value: object,
    *,
    layer_count: int,
    context: str,
) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != layer_count:
        raise ValueError(f"{context} must contain exactly one A-E vector per decoder layer")
    result: list[list[float]] = []
    for layer, raw_probabilities in enumerate(value):
        if not isinstance(raw_probabilities, list) or len(raw_probabilities) != 5:
            raise ValueError(f"{context}[{layer}] must contain exactly five probabilities")
        if any(not isinstance(item, int | float) for item in raw_probabilities):
            raise ValueError(f"{context}[{layer}] contains a nonnumeric probability")
        numeric_probabilities = cast(list[int | float], raw_probabilities)
        probabilities = [float(item) for item in numeric_probabilities]
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities):
            raise ValueError(f"{context}[{layer}] contains an invalid probability")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=2.0e-6):
            raise ValueError(f"{context}[{layer}] must sum to one")
        result.append(probabilities)
    return result


def _validated_answer_lookup_choice_vector(value: object, *, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"{context} must contain exactly five probabilities")
    if any(not isinstance(item, int | float) for item in value):
        raise ValueError(f"{context} contains a nonnumeric probability")
    numeric_probabilities = cast(list[int | float], value)
    probabilities = [float(item) for item in numeric_probabilities]
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities):
        raise ValueError(f"{context} contains an invalid probability")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=2.0e-6):
        raise ValueError(f"{context} must sum to one")
    return probabilities


def _export_answer_lookup(root: Path) -> tuple[dict[str, object], int, int]:
    """Export only measured answer-location artifacts; absent rows remain unprocessed."""

    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT)
    spec = MODEL_SPECS[ModelKey.OLMO3_7B]
    manifest: dict[str, object] = {
        "schema_version": ANSWER_LOOKUP_SCHEMA_VERSION,
        "model": run.model,
        "condition": run.condition.value,
        "seed": run.seed,
        "checkpoint_step": ANSWER_LOOKUP_CHECKPOINT_STEP,
        "layer_count": spec.layer_count,
        "interfaces": list(ANSWER_LOOKUP_INTERFACES),
        "intervention_count": ANSWER_LOOKUP_INTERVENTION_COUNT,
        "entries": {},
    }
    entries = cast(dict[str, object], manifest["entries"])
    raw_count = 0
    complete_count = 0
    for interface in ANSWER_LOOKUP_INTERFACES:
        interface_entries: dict[str, object] = {}
        entries[interface] = interface_entries
        for function in FUNCTIONS:
            function_id = function.function_id
            path = (
                run_dir(root, run)
                / "answer_lookup"
                / f"checkpoint_step_{ANSWER_LOOKUP_CHECKPOINT_STEP:06d}"
                / interface
                / f"{function_id}.json"
            )
            if not path.is_file():
                interface_entries[function_id] = {"status": "unprocessed"}
                continue
            artifact = _mapping(read_json(path), context=str(path))
            expected_run = {
                "model": run.model,
                "condition": run.condition.value,
                "seed": run.seed,
                "effective_batch_size": run.effective_batch_size,
                "lora_rank": run.lora_rank,
            }
            model = _mapping(artifact.get("model"), context=f"{path}.model")
            correct_choice_index = artifact.get("correct_choice_index")
            if (
                artifact.get("schema_version") != ANSWER_LOOKUP_SCHEMA_VERSION
                or artifact.get("run") != expected_run
                or artifact.get("checkpoint_step") != ANSWER_LOOKUP_CHECKPOINT_STEP
                or artifact.get("interface") != interface
                or artifact.get("function_id") != function_id
                or not isinstance(correct_choice_index, int)
                or not 0 <= correct_choice_index < 5
                or artifact.get("correct_choice_label") != ANSWER_LABELS[correct_choice_index]
                or model.get("id") != spec.model_id
                or model.get("revision") != spec.revision
                or model.get("layer_count") != spec.layer_count
            ):
                raise RuntimeError(f"answer-lookup artifact identity mismatch: {path}")
            backend = _mapping(
                artifact.get("scientific_backend"),
                context=f"{path}.scientific_backend",
            )
            if backend != {
                "full_prompt": True,
                "use_cache": False,
                "batch_size": 1,
                "inference_mode": True,
            }:
                raise RuntimeError(f"answer-lookup artifact used a non-reference backend: {path}")
            sources = _mapping(artifact.get("source_prompts"), context=f"{path}.source_prompts")
            if set(sources) != {source.value for source in AnswerLookupSource}:
                raise ValueError(f"answer-lookup artifact must contain all four sources: {path}")
            for source_name, raw_source in sources.items():
                source = _mapping(raw_source, context=f"{path}.source_prompts.{source_name}")
                sites = source.get("terminator_sites")
                if not isinstance(sites, list) or len(sites) != 5:
                    raise ValueError(f"{path} source {source_name} must contain five sites")
                site_labels = tuple(
                    _mapping(site, context=f"{path}.{source_name}.terminator_sites[]").get("label")
                    for site in sites
                )
                if site_labels != tuple(ANSWER_LABELS):
                    raise ValueError(f"{path} source {source_name} site labels disagree")
                _validated_answer_lookup_choice_vector(
                    source.get("unpatched_probabilities"),
                    context=f"{path}.{source_name}.unpatched_probabilities",
                )
            raw_rows = artifact.get("interventions")
            if not isinstance(raw_rows, list) or len(raw_rows) != ANSWER_LOOKUP_INTERVENTION_COUNT:
                raise ValueError(
                    f"answer-lookup artifact must contain 27 intervention rows: {path}"
                )
            completed_rows = 0
            observed_ids: set[str] = set()
            observed_groups: list[str] = []
            for index, raw_row in enumerate(raw_rows):
                row = _mapping(raw_row, context=f"{path}.interventions[{index}]")
                intervention_id = row.get("intervention_id")
                if not isinstance(intervention_id, str) or intervention_id in observed_ids:
                    raise ValueError(f"{path} intervention IDs must be unique strings")
                observed_ids.add(intervention_id)
                group = row.get("group")
                source_name = row.get("source")
                source_indices = row.get("source_choice_indices")
                recipient_indices = row.get("recipient_choice_indices")
                target_index = row.get("target_choice_index")
                if (
                    group
                    not in {
                        "preserve_correct_marker",
                        "erase_correct_marker",
                        "move_correct_marker",
                        "duplicate_correct_marker",
                    }
                    or source_name not in {source.value for source in AnswerLookupSource}
                    or not isinstance(source_indices, list)
                    or not isinstance(recipient_indices, list)
                    or not source_indices
                    or len(source_indices) != len(recipient_indices)
                    or any(
                        not isinstance(item, int) or not 0 <= item < 5 for item in source_indices
                    )
                    or any(
                        not isinstance(item, int) or not 0 <= item < 5 for item in recipient_indices
                    )
                    or len(set(recipient_indices)) != len(recipient_indices)
                    or (
                        target_index is not None
                        and (not isinstance(target_index, int) or not 0 <= target_index < 5)
                    )
                    or not isinstance(row.get("label"), str)
                    or not isinstance(row.get("causal_question"), str)
                ):
                    raise ValueError(f"{path} intervention {intervention_id} has invalid metadata")
                observed_groups.append(cast(str, group))
                probabilities = _validated_answer_lookup_probabilities(
                    row.get("probabilities_by_layer"),
                    layer_count=spec.layer_count,
                    context=f"{path}.interventions[{index}].probabilities_by_layer",
                )
                if probabilities is not None:
                    completed_rows += 1
            if observed_groups != [
                *(["preserve_correct_marker"] * 4),
                *(["erase_correct_marker"] * 4),
                *(["move_correct_marker"] * 4),
                *(["duplicate_correct_marker"] * 15),
            ]:
                raise ValueError(f"{path} answer-lookup groups or ordering changed")
            status = artifact.get("status")
            if status not in {"partial", "complete"}:
                raise ValueError(f"answer-lookup artifact has invalid status: {path}")
            if (status == "complete") != (completed_rows == ANSWER_LOOKUP_INTERVENTION_COUNT):
                raise RuntimeError(f"answer-lookup completion marker disagrees with rows: {path}")
            if status == "complete":
                identity_error = artifact.get("identity_parity_max_abs_error")
                hook_error = artifact.get("post_run_unpatched_max_abs_error")
                if (
                    not isinstance(identity_error, int | float)
                    or not isinstance(hook_error, int | float)
                    or not 0.0 <= float(identity_error) <= 1.0e-6
                    or not 0.0 <= float(hook_error) <= 1.0e-6
                ):
                    raise RuntimeError(f"answer-lookup parity gates did not pass: {path}")
                complete_count += 1
            relative_path = Path("data") / "answer-lookup" / interface / f"{function_id}.json"
            digest, byte_count = _write_compact_json(root / "site" / relative_path, artifact)
            interface_entries[function_id] = {
                "status": status,
                "completed_interventions": completed_rows,
                "bytes": byte_count,
                "sha256": digest,
                "raw_sha256": sha256_file(path),
                "url": relative_path.as_posix(),
            }
            raw_count += 1
    manifest["raw_artifact_count"] = raw_count
    manifest["complete_artifact_count"] = complete_count
    return manifest, raw_count, complete_count


def _export_fourier_circuits(root: Path) -> tuple[dict[str, object], int]:
    """Export measured Fourier stages while preserving their epistemic status."""

    lineage_cache: dict[tuple[str, str, int, int], dict[str, object]] = {}

    def validated_lineage_payload(
        value: object,
        *,
        context: str,
        allow_plan_identity: bool = False,
    ) -> dict[str, object]:
        lineage = _mapping(value, context=context)
        lineage_id = lineage.get("id")
        kind = lineage.get("kind")
        display_name = lineage.get("display_name")
        if (
            not isinstance(lineage_id, str)
            or LINEAGE_ID_PATTERN.fullmatch(lineage_id) is None
            or not isinstance(display_name, str)
            or not display_name
            or kind not in {"workspace_unregistered", "registered_hardware"}
        ):
            raise RuntimeError(f"{context} has an invalid Fourier lineage identity")
        provenance_fields = (
            "plan_sha256",
            "reference_source_bundle_sha256",
            "collection_source_bundle_sha256",
        )
        if kind == "workspace_unregistered":
            if lineage.get("hardware") is not None or any(
                lineage.get(field) is not None for field in provenance_fields
            ):
                raise RuntimeError(f"{context} gives unregistered results registered provenance")
            return lineage
        hardware = _mapping(lineage.get("hardware"), context=f"{context}.hardware")
        capability = hardware.get("compute_capability")
        if (
            not isinstance(hardware.get("device_name"), str)
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(not isinstance(value, int) for value in capability)
            or not isinstance(hardware.get("total_memory_bytes"), int)
            or cast(int, hardware["total_memory_bytes"]) <= 0
            or any(
                not isinstance(hardware.get(field), str)
                for field in (
                    "driver_version",
                    "torch_version",
                    "cuda_version",
                )
            )
        ):
            raise RuntimeError(f"{context} has a malformed registered hardware fingerprint")
        digest_fields = (
            "reference_source_bundle_sha256",
            "collection_source_bundle_sha256",
        )
        for field in digest_fields:
            digest = lineage.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"{context}.{field} is not a SHA-256 digest")
        plan_digest = lineage.get("plan_sha256")
        if not (allow_plan_identity and plan_digest is None) and (
            not isinstance(plan_digest, str)
            or len(plan_digest) != 64
            or any(character not in "0123456789abcdef" for character in plan_digest)
        ):
            raise RuntimeError(f"{context}.plan_sha256 is not a SHA-256 digest")
        return lineage

    def lineage_identity_payload(
        value: object,
        *,
        context: str,
    ) -> dict[str, object]:
        """Return the checkpoint-invariant identity for one measurement lineage."""

        lineage = dict(
            validated_lineage_payload(
                value,
                context=context,
                allow_plan_identity=True,
            )
        )
        if lineage["kind"] == "registered_hardware":
            lineage["plan_sha256"] = None
        return lineage

    def validated_lineage(
        config_path: Path,
        config: dict[str, object],
        model: dict[str, object],
        task: dict[str, object],
    ) -> dict[str, object]:
        artifact_root = config.get("artifact_root")
        if not isinstance(artifact_root, str) or not Path(artifact_root).is_absolute():
            raise TypeError(
                f"Fourier config lacks an absolute artifact identity root: {config_path}"
            )
        identity_root = Path(artifact_root)
        function_id = task.get("function_id")
        clean_step = model.get("clean_step")
        dirty_step = model.get("dirty_step")
        if (
            not isinstance(function_id, str)
            or not isinstance(clean_step, int)
            or not isinstance(dirty_step, int)
        ):
            raise TypeError(
                f"Fourier config lacks a lineage-compatible run identity: {config_path}"
            )
        if identity_root.parent.name != "hardware_lineages":
            return validated_lineage_payload(
                {
                    "id": "workspace_unregistered",
                    "kind": "workspace_unregistered",
                    "display_name": "Workspace reference (hardware unregistered)",
                    "hardware": None,
                    "plan_sha256": None,
                    "reference_source_bundle_sha256": None,
                    "collection_source_bundle_sha256": None,
                },
                context=f"{config_path}.lineage",
            )

        lineage_id = identity_root.name
        cache_key = (lineage_id, function_id, clean_step, dirty_step)
        cached = lineage_cache.get(cache_key)
        if cached is not None:
            return cached
        plan_path = (
            root
            / "artifacts/plans/fourier_hardware_lineages"
            / f"{lineage_id}_{function_id}_step_{clean_step:06d}.json"
        )
        plan = load_hardware_lineage_plan(root, plan_path)
        if (
            plan.lineage_id != lineage_id
            or plan.artifact_identity_root != identity_root
            or plan.function_id != function_id
            or plan.model_key != model.get("model_key")
            or plan.condition != model.get("condition")
            or plan.clean_step != clean_step
            or plan.dirty_step != dirty_step
        ):
            raise RuntimeError(
                f"Fourier hardware-lineage plan disagrees with config: {config_path}"
            )
        hardware = {
            "device_name": plan.hardware.device_name,
            "compute_capability": list(plan.hardware.compute_capability),
            "total_memory_bytes": plan.hardware.total_memory_bytes,
            "driver_version": plan.hardware.driver_version,
            "torch_version": plan.hardware.torch_version,
            "cuda_version": plan.hardware.cuda_version,
        }
        payload = {
            "id": lineage_id,
            "kind": "registered_hardware",
            "display_name": f"{plan.hardware.device_name} ({lineage_id})",
            "hardware": hardware,
            "plan_sha256": sha256_file(plan_path),
            "reference_source_bundle_sha256": plan.reference_source_bundle_sha256,
            "collection_source_bundle_sha256": plan.collection_source_bundle_sha256,
        }
        validated = validated_lineage_payload(payload, context=f"{plan_path}.lineage")
        lineage_cache[cache_key] = validated
        return validated

    def validated_sidecar(
        artifact_path: Path,
        artifact: dict[str, object],
        *,
        field: str,
    ) -> None:
        sidecar_name = artifact.get(field)
        sidecar_digest = artifact.get(f"{field}_sha256")
        if not isinstance(sidecar_name, str) or not isinstance(sidecar_digest, str):
            raise TypeError(f"{artifact_path} lacks {field} provenance")
        sidecar_path = artifact_path.with_name(sidecar_name)
        if not sidecar_path.is_file() or sha256_file(sidecar_path) != sidecar_digest:
            raise RuntimeError(f"{artifact_path} has a missing or changed {field}")

    def same_science_directory(
        value: object,
        physical_expected: Path,
        logical_expected: Path,
    ) -> bool:
        if not isinstance(value, str) or not Path(value).is_absolute():
            return False
        candidate = Path(value)
        if candidate == logical_expected:
            return True
        return (
            candidate.exists()
            and physical_expected.exists()
            and candidate.resolve(strict=True) == physical_expected.resolve(strict=True)
        )

    def validated_density(path: Path, *, function_space: str | None) -> dict[str, object]:
        density = _mapping(read_json(path), context=str(path))
        if (
            density.get("schema_version") != 1
            or density.get("stage") != 0
            or density.get("status") not in {"transition_found", "flat_stop"}
            or not isinstance(density.get("curve"), list)
        ):
            raise TypeError(f"Fourier stage-0 density curve is malformed: {path}")
        if function_space is not None and density.get("function_space") != function_space:
            raise RuntimeError(f"Fourier density curve has the wrong function space: {path}")
        validated_sidecar(path, density, field="sample_sidecar")
        return density

    def validated_sites(
        rows: object,
        *,
        context: str,
        require_nonempty: bool,
    ) -> list[dict[str, object]]:
        if not isinstance(rows, list) or (require_nonempty and not rows):
            raise RuntimeError(
                f"{context} must be a {'non-empty ' if require_nonempty else ''}list"
            )
        validated: list[dict[str, object]] = []
        seen: set[tuple[int, int]] = set()
        for raw_site in cast(list[object], rows):
            site = _mapping(raw_site, context=f"{context}[]")
            token_index = site.get("token_index")
            layer = site.get("layer")
            if not isinstance(token_index, int) or not isinstance(layer, int):
                raise TypeError(f"{context} contains an invalid site")
            key = (token_index, layer)
            if key in seen:
                raise RuntimeError(f"{context} repeats a site")
            seen.add(key)
            validated.append(site)
        return validated

    def validated_coefficient(raw: object, *, context: str) -> dict[str, object]:
        coefficient = _mapping(raw, context=context)
        degree = coefficient.get("degree")
        sites = validated_sites(
            coefficient.get("sites"),
            context=f"{context}.sites",
            require_nonempty=True,
        )
        if (
            coefficient.get("is_heavy") is not True
            or not isinstance(degree, int)
            or degree <= 0
            or len(sites) != degree
        ):
            raise RuntimeError(f"{context} is malformed or is not heavy")
        _number(coefficient, "lasso_value", context=context)
        _number(coefficient, "function_value_estimate", context=context)
        return coefficient

    def site_key(site: dict[str, object], *, context: str) -> tuple[int, int]:
        token_index = site.get("token_index")
        layer = site.get("layer")
        if not isinstance(token_index, int) or not isinstance(layer, int):
            raise TypeError(f"{context} contains an invalid site coordinate")
        return token_index, layer

    def validated_minsets(
        rows: object,
        *,
        context: str,
    ) -> list[dict[str, object]]:
        if not isinstance(rows, list):
            raise TypeError(f"{context} must be a list")
        minsets: list[dict[str, object]] = []
        sizes: list[int] = []
        for raw_minset in cast(list[object], rows):
            minset = _mapping(raw_minset, context=f"{context}[]")
            size = minset.get("size")
            sites = validated_sites(
                minset.get("sites"),
                context=f"{context}[].sites",
                require_nonempty=True,
            )
            if not isinstance(size, int) or size <= 0 or len(sites) != size:
                raise RuntimeError(f"{context} contains a minset with inconsistent size")
            probability = _number(minset, "correct_probability", context=f"{context}[]")
            _number(minset, "raw_logit_diff", context=f"{context}[]")
            _number(minset, "sufficiency_margin", context=f"{context}[]")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{context} contains a probability outside [0, 1]")
            hypotheses = minset.get("generating_coefficients")
            if not isinstance(hypotheses, list) or not hypotheses:
                raise RuntimeError(f"{context} contains a minset without a generating hypothesis")
            for index, hypothesis in enumerate(cast(list[object], hypotheses)):
                validated_coefficient(hypothesis, context=f"{context}[].hypothesis[{index}]")
            sizes.append(size)
            minsets.append(minset)
        if sizes != sorted(sizes):
            raise RuntimeError(f"{context} must be ordered smallest first")
        return minsets

    def validated_recall_audit(
        path: Path,
        logical_scope_directory: Path,
    ) -> dict[str, object]:
        audit = _mapping(read_json(path), context=str(path))
        if (
            audit.get("schema_version") != 1
            or audit.get("status") != "complete"
            or audit.get("audit_is_not_globally_exhaustive") is not True
            or audit.get("raw_proposals_are_not_circuits") is not True
        ):
            raise RuntimeError(f"Fourier recall audit is malformed: {path}")
        source = _mapping(audit.get("source"), context=f"{path}.source")
        if not same_science_directory(
            source.get("base_fourier_directory"),
            path.parents[1],
            logical_scope_directory,
        ):
            raise RuntimeError(f"Fourier recall audit names the wrong source directory: {path}")
        source_files = {
            "exhaustive_singletons_sha256": "exhaustive_singletons.json",
            "stage_1_spectrum_sha256": "stage_1_spectrum.json",
            "stage_2_minsets_sha256": "stage_2_minsets.json",
            "stage_2_verification_sha256": "stage_2_verification.pt",
        }
        for digest_field, filename in source_files.items():
            if source.get(digest_field) != sha256_file(path.parents[1] / filename):
                raise RuntimeError(f"Fourier recall source digest mismatch: {path}")
        manifests = audit.get("phase_manifests")
        if not isinstance(manifests, list) or len(manifests) != 3:
            raise RuntimeError(f"Fourier recall audit lacks its phase manifests: {path}")
        for raw_manifest in cast(list[object], manifests):
            manifest = _mapping(raw_manifest, context=f"{path}.phase_manifests[]")
            phase = manifest.get("phase")
            shards = manifest.get("shards")
            shard_count = manifest.get("shard_count")
            proposal_count = manifest.get("proposal_count")
            shard_rows = (
                [
                    _mapping(shard, context=f"{path}.phase_manifests[].shards[]")
                    for shard in cast(list[object], shards)
                ]
                if isinstance(shards, list)
                else []
            )
            if (
                not isinstance(phase, str)
                or not isinstance(shards, list)
                or shard_count != len(shard_rows)
                or not isinstance(proposal_count, int)
                or proposal_count
                != sum(
                    cast(int, shard.get("proposal_count"))
                    for shard in shard_rows
                    if isinstance(shard.get("proposal_count"), int)
                )
            ):
                raise RuntimeError(f"Fourier recall phase manifest is inconsistent: {path}")
            for shard in shard_rows:
                sidecar = shard.get("sidecar")
                digest = shard.get("sidecar_sha256")
                metadata = shard.get("metadata")
                if not all(isinstance(value, str) for value in (sidecar, digest, metadata)):
                    raise TypeError(f"Fourier recall shard identity is malformed: {path}")
                sidecar_path = path.parent / phase / cast(str, sidecar)
                metadata_path = path.parent / phase / cast(str, metadata)
                if sha256_file(sidecar_path) != digest:
                    raise RuntimeError(f"Fourier recall shard digest mismatch: {path}")
                metadata_payload = _mapping(read_json(metadata_path), context=str(metadata_path))
                if (
                    metadata_payload.get("sidecar") != sidecar
                    or metadata_payload.get("sidecar_sha256") != digest
                    or metadata_payload.get("proposal_count") != shard.get("proposal_count")
                    or metadata_payload.get("proposal_sha256") != shard.get("proposal_sha256")
                ):
                    raise RuntimeError(f"Fourier recall shard manifest disagrees: {path}")
        local = _mapping(audit.get("local_truth_table"), context=f"{path}.local")
        local_sites = validated_sites(
            local.get("sites"),
            context=f"{path}.local.sites",
            require_nonempty=False,
        )
        if (
            local.get("site_count") != len(local_sites)
            or local.get("subset_count") != 2 ** len(local_sites) - 1
            or not isinstance(local.get("minimal_sufficient_sets"), list)
            or not isinstance(local.get("new_minsets_missed_by_fourier"), list)
            or not isinstance(local.get("monotone"), bool)
        ):
            raise RuntimeError(f"Fourier recall local truth table is inconsistent: {path}")
        for field in ("minimal_sufficient_sets", "new_minsets_missed_by_fourier"):
            for index, sites in enumerate(cast(list[object], local[field])):
                validated_sites(
                    sites,
                    context=f"{path}.local.{field}[{index}]",
                    require_nonempty=True,
                )
        for field in ("new_verified_pair_minsets", "new_verified_triple_minsets"):
            rows = audit.get(field)
            expected_size = 2 if field == "new_verified_pair_minsets" else 3
            if not isinstance(rows, list):
                raise TypeError(f"Fourier recall result lacks {field}: {path}")
            for raw_row in cast(list[object], rows):
                row = _mapping(raw_row, context=f"{path}.{field}[]")
                sites = validated_sites(
                    row.get("sites"),
                    context=f"{path}.{field}[].sites",
                    require_nonempty=True,
                )
                probability = _number(row, "correct_probability", context=f"{path}.{field}[]")
                _number(row, "raw_logit_diff", context=f"{path}.{field}[]")
                if (
                    row.get("size") != expected_size
                    or len(sites) != expected_size
                    or row.get("sufficient") is not True
                    or not 0.0 <= probability <= 1.0
                ):
                    raise RuntimeError(f"Fourier recall minset row is inconsistent: {path}")
        return audit

    entries: list[dict[str, object]] = []
    pattern = "artifacts/runs/*/*/seed_*/fourier_circuits/*/clean_*_dirty_*/*/config.json"
    legacy_identity_scope = "full_prompt_layers_0_32_backend_full_sequence_reference"
    for config_path in sorted(root.glob(pattern)):
        config_wrapper = _mapping(read_json(config_path), context=str(config_path))
        if config_wrapper.get("schema_version") != 1 or not isinstance(
            config_wrapper.get("config"), dict
        ):
            raise TypeError(f"Fourier config wrapper is malformed: {config_path}")
        config = cast(dict[str, object], config_wrapper["config"])
        model = _mapping(config.get("model"), context=f"{config_path}.model")
        task = _mapping(config.get("task"), context=f"{config_path}.task")
        required_model = ("model_key", "condition", "clean_step", "dirty_step")
        if any(key not in model for key in required_model) or "function_id" not in task:
            raise TypeError(f"Fourier config lacks required run identity: {config_path}")
        model_key = str(model["model_key"])
        condition = str(model["condition"])
        function_id = str(task["function_id"])
        clean_step = int(cast(int, model["clean_step"]))
        dirty_step = int(cast(int, model["dirty_step"]))
        lineage = validated_lineage(config_path, config, model, task)
        lineage_id = cast(str, lineage["id"])
        scope_key = config_path.parent.name
        logical_scope_directory = Path(cast(str, config["artifact_root"])) / (
            config_path.parent.relative_to(root)
        )
        raw_sufficiency_config = config.get("sufficiency")
        sufficiency_config = (
            {}
            if raw_sufficiency_config is None
            else _mapping(
                raw_sufficiency_config,
                context=f"{config_path}.sufficiency",
            )
        )
        corrected_probability_run = (
            "absolute_probability_tolerance" in sufficiency_config
            and "expected_passing_singletons" in sufficiency_config
        )
        legacy_identity_run = function_id == "identity" and scope_key == legacy_identity_scope

        # Identity is retained only as the original p=.1 diagnostic. The later refined
        # strict identity spectrum was explicitly stopped and must never be reinterpreted here.
        # A separately registered probability-threshold run is a corrected analysis and is
        # exported through the same exhaustive-singleton contract as pyalvt.
        if function_id == "identity" and not legacy_identity_run and not corrected_probability_run:
            continue

        unrestricted_path = config_path.with_name("stage_0_density.json")
        acquisition_path = config_path.with_name("endpoint_acquisition_gate.json")
        singleton_path = config_path.with_name("exhaustive_singletons.json")
        residual_path = config_path.with_name("stage_0_residual_density.json")
        stage_one_path = config_path.with_name("stage_1_spectrum.json")
        stage_two_path = config_path.with_name("stage_2_minsets.json")
        if not unrestricted_path.is_file():
            if acquisition_path.is_file():
                acquisition = _mapping(
                    read_json(acquisition_path),
                    context=str(acquisition_path),
                )
                if (
                    acquisition.get("schema_version") != 1
                    or acquisition.get("stage") != "endpoint_acquisition_gate"
                    or acquisition.get("status") != "clean_behavior_not_acquired"
                    or acquisition.get("terminal") is not True
                ):
                    raise RuntimeError(
                        f"Fourier acquisition-gate artifact is malformed: {acquisition_path}"
                    )
                for field in (
                    "clean_checkpoint",
                    "dirty_checkpoint",
                    "all_clean_intervention",
                ):
                    endpoint = _mapping(
                        acquisition.get(field),
                        context=f"{acquisition_path}.{field}",
                    )
                    probability = _number(
                        endpoint,
                        "correct_probability",
                        context=f"{acquisition_path}.{field}",
                    )
                    _number(
                        endpoint,
                        "logit_diff",
                        context=f"{acquisition_path}.{field}",
                    )
                    if not 0.0 <= probability <= 1.0 or not isinstance(
                        endpoint.get("accuracy"), bool
                    ):
                        raise RuntimeError(
                            f"Fourier acquisition endpoint is malformed: {acquisition_path}"
                        )
                site_grid = _mapping(
                    acquisition.get("site_grid"),
                    context=f"{acquisition_path}.site_grid",
                )
                relative_path = (
                    Path("fourier-circuits")
                    / f"lineage_{lineage_id}"
                    / model_key
                    / condition
                    / function_id
                    / f"clean_{clean_step:06d}_dirty_{dirty_step:06d}"
                    / f"{scope_key}.json"
                )
                chunk = {
                    "schema_version": 1,
                    "status": "clean_behavior_not_acquired",
                    "lineage": lineage,
                    "model": model,
                    "task": task,
                    "sites": config.get("sites"),
                    "sufficiency_criterion": ("clean_correct_probability_minus_absolute_tolerance"),
                    "site_grid": site_grid,
                    "unrestricted_density_curve": [],
                    "unrestricted_transition_density": None,
                    "residual_density_curve": None,
                    "residual_transition_density": None,
                    "network_veto_density_diagnostics": [],
                    "disconnected_searches": [],
                    "density_stability_warning": None,
                    "sufficiency": None,
                    "exhaustive_singleton_count": None,
                    "verified_singleton_minsets": [],
                    "verified_multisite_minsets": [],
                    "network_verified_multisite_minsets": [],
                    "proper_subset_separation": {
                        "enabled": False,
                        "maximum_proper_subset_correct_probability": None,
                        "maximum_proper_subset_fraction_of_full_probability": None,
                        "unfiltered_multisite_minset_count": 0,
                        "passing_multisite_minset_count": 0,
                    },
                    "minset_networks": [],
                    "partner_profile_clustering": {
                        "method": (
                            "profile_seeded_deterministic_complete_link_neighbor_jaccard_"
                            "with_minset_cannot_link"
                        ),
                        "minimum_similarity": 0.5,
                        "hyperedges_preserved_for_higher_order_minsets": True,
                        "clusters_are_descriptive_not_identified_pathways": True,
                    },
                    "recall_audits": [],
                    "frontier_searches": [],
                    "alternative_probability_sufficiency": None,
                    "raw_heavy_fourier_hypotheses": [],
                    "legacy_sparse_discovery_minsets": [],
                    "singleton_search_is_exhaustive": False,
                    "legacy_discovery_is_only_a_lower_bound": False,
                    "raw_fourier_candidates_are_not_circuits": True,
                    "endpoint_acquisition_gate": acquisition,
                }
                digest, byte_count = _write_compact_json(
                    root / "site" / "data" / relative_path,
                    chunk,
                )
                entries.append(
                    {
                        "model": model_key,
                        "lineage": lineage,
                        "condition": condition,
                        "function_id": function_id,
                        "clean_step": clean_step,
                        "dirty_step": dirty_step,
                        "scope": scope_key,
                        "status": "clean_behavior_not_acquired",
                        "sufficiency_criterion": (
                            "clean_correct_probability_minus_absolute_tolerance"
                        ),
                        "singleton_minset_count": 0,
                        "multisite_minset_count": 0,
                        "unfiltered_multisite_minset_count": 0,
                        "fourier_multisite_minset_count": 0,
                        "raw_hypothesis_count": 0,
                        "legacy_minset_count": 0,
                        "url": relative_path.as_posix(),
                        "bytes": byte_count,
                        "sha256": digest,
                    }
                )
            continue
        unrestricted = validated_density(
            unrestricted_path,
            function_space=None if legacy_identity_run else "unrestricted",
        )

        singleton_rows: list[dict[str, object]] = []
        all_singleton_rows: list[dict[str, object]] = []
        singleton_count: int | None = None
        sufficiency: object = None
        site_grid: object = unrestricted.get("site_grid")
        if singleton_path.is_file():
            singletons = _mapping(read_json(singleton_path), context=str(singleton_path))
            if (
                singletons.get("schema_version") != 1
                or singletons.get("stage") != "exhaustive_singletons"
                or singletons.get("status") != "verified"
                or singletons.get("singleton_search_is_exhaustive") is not True
            ):
                raise RuntimeError(f"exhaustive singleton artifact is malformed: {singleton_path}")
            validated_sidecar(singleton_path, singletons, field="singleton_sidecar")
            raw_singleton_count = singletons.get("singleton_count")
            if not isinstance(raw_singleton_count, int) or raw_singleton_count <= 0:
                raise TypeError(f"exhaustive singleton count is invalid: {singleton_path}")
            raw_passing = singletons.get("verified_singleton_minsets")
            if not isinstance(raw_passing, list):
                raise TypeError(f"exhaustive singleton passing table is invalid: {singleton_path}")
            raw_all_singletons = singletons.get("singleton_results")
            if not isinstance(raw_all_singletons, list):
                raise TypeError(f"exhaustive singleton complete table is invalid: {singleton_path}")
            all_site_keys: set[tuple[int, int]] = set()
            for raw_row in cast(list[object], raw_all_singletons):
                row = _mapping(raw_row, context=f"{singleton_path}.singleton_results[]")
                site = _mapping(row.get("site"), context=f"{singleton_path}.site")
                validated_site = validated_sites(
                    [site],
                    context=f"{singleton_path}.site",
                    require_nonempty=True,
                )[0]
                key = site_key(validated_site, context=f"{singleton_path}.site")
                if key in all_site_keys:
                    raise RuntimeError(f"singleton complete table repeats a site: {singleton_path}")
                all_site_keys.add(key)
                probability = _number(row, "correct_probability", context=str(singleton_path))
                _number(row, "raw_logit_diff", context=str(singleton_path))
                if (
                    not 0.0 <= probability <= 1.0
                    or not isinstance(row.get("accuracy"), bool)
                    or not isinstance(row.get("sufficient"), bool)
                ):
                    raise ValueError(
                        f"singleton complete table has invalid metrics: {singleton_path}"
                    )
                all_singleton_rows.append(row)
            if len(all_singleton_rows) != raw_singleton_count:
                raise RuntimeError(
                    f"singleton complete table has the wrong length: {singleton_path}"
                )
            for raw_row in cast(list[object], raw_passing):
                row = _mapping(raw_row, context=f"{singleton_path}.verified_singletons[]")
                site = _mapping(row.get("site"), context=f"{singleton_path}.site")
                validated_sites([site], context=f"{singleton_path}.site", require_nonempty=True)
                if row.get("sufficient") is not True:
                    raise RuntimeError(
                        f"singleton passing table contains a failing row: {singleton_path}"
                    )
                probability = _number(row, "correct_probability", context=str(singleton_path))
                _number(row, "raw_logit_diff", context=str(singleton_path))
                _number(row, "sufficiency_margin", context=str(singleton_path))
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"singleton probability is outside [0, 1]: {singleton_path}")
                singleton_rows.append(row)
            if singletons.get("passing_singleton_count") != len(singleton_rows):
                raise RuntimeError(
                    f"singleton passing count disagrees with its rows: {singleton_path}"
                )
            singleton_count = raw_singleton_count
            sufficiency = singletons.get("sufficiency")
            site_grid = singletons.get("site_grid")
        elif not legacy_identity_run:
            continue

        residual: dict[str, object] | None = None
        if residual_path.is_file():
            residual = validated_density(
                residual_path,
                function_space="singleton_vetoed_residual",
            )
        elif not legacy_identity_run:
            raise FileNotFoundError(f"corrected Fourier run lacks residual density: {config_path}")

        heavy_hypotheses: list[dict[str, object]] = []
        density_warning: object = None
        if stage_one_path.is_file():
            stage_one = _mapping(read_json(stage_one_path), context=str(stage_one_path))
            if stage_one.get("schema_version") != 1 or stage_one.get("stage") != 1:
                raise TypeError(f"Fourier stage-1 artifact is malformed: {stage_one_path}")
            validated_sidecar(stage_one_path, stage_one, field="sample_sidecar")
            coefficients = stage_one.get("coefficients")
            if not isinstance(coefficients, list):
                raise TypeError(f"Fourier stage 1 lacks a coefficient table: {stage_one_path}")
            for index, raw_coefficient in enumerate(cast(list[object], coefficients)):
                if isinstance(raw_coefficient, dict) and raw_coefficient.get("is_heavy") is True:
                    heavy_hypotheses.append(
                        validated_coefficient(
                            raw_coefficient,
                            context=f"{stage_one_path}.coefficients[{index}]",
                        )
                    )
            if stage_one.get("heavy_coefficient_count") != len(heavy_hypotheses):
                raise RuntimeError(f"stage-1 heavy coefficient count disagrees: {stage_one_path}")
            density_warning = stage_one.get("warning")

        multisite_minsets: list[dict[str, object]] = []
        legacy_minsets: list[dict[str, object]] = []
        if stage_two_path.is_file():
            stage_two = _mapping(read_json(stage_two_path), context=str(stage_two_path))
            if (
                stage_two.get("schema_version") != 1
                or stage_two.get("stage") != 2
                or stage_two.get("raw_fourier_candidates_are_not_circuits") is not True
            ):
                raise RuntimeError(f"Fourier stage-2 artifact is malformed: {stage_two_path}")
            validated_sidecar(stage_two_path, stage_two, field="verification_sidecar")
            if legacy_identity_run:
                if stage_two.get("status") not in {"verified", "no_verified_minsets"}:
                    raise RuntimeError(f"legacy identity stage 2 is unverified: {stage_two_path}")
                legacy_minsets = validated_minsets(
                    stage_two.get("verified_minsets"),
                    context=f"{stage_two_path}.verified_minsets",
                )
            else:
                if stage_two.get("status") not in {
                    "verified_multisite",
                    "no_verified_multisite_minsets",
                    "no_higher_order_hypotheses",
                }:
                    raise RuntimeError(f"corrected Fourier stage 2 is unverified: {stage_two_path}")
                multisite_minsets = validated_minsets(
                    stage_two.get("verified_multisite_minsets"),
                    context=f"{stage_two_path}.verified_multisite_minsets",
                )
                density_warning = stage_two.get("density_stability_warning")

        sufficiency_criterion = "raw_logit_gap_recovery"
        sufficiency_mapping: dict[str, object] | None = None
        if all_singleton_rows:
            sufficiency_mapping = _mapping(
                sufficiency,
                context=f"{singleton_path}.sufficiency",
            )
            raw_criterion = sufficiency_mapping.get("criterion", "raw_logit_gap_recovery")
            if raw_criterion not in {
                "raw_logit_gap_recovery",
                "clean_correct_probability_minus_absolute_tolerance",
            }:
                raise RuntimeError(f"unknown sufficiency criterion: {singleton_path}")
            sufficiency_criterion = str(raw_criterion)

        status = (
            "legacy_sparse_discovery_lower_bound"
            if legacy_identity_run
            else "causal_multisite_complete"
            if stage_two_path.is_file()
            else "spectrum_complete"
            if stage_one_path.is_file()
            else "singleton_and_density_complete"
        )
        recall_audits = [
            validated_recall_audit(path, logical_scope_directory)
            for path in sorted(config_path.parent.glob("recall_audit_config_*/recall_audit.json"))
        ]
        frontier_searches: list[dict[str, object]] = []
        frontier_metric_indexes: list[tuple[Path, dict[tuple[Site, ...], SubsetMetric]]] = []
        for frontier_path in sorted(
            config_path.parent.glob(f"frontier_search_config_*/{FRONTIER_RESULT_FILENAME}")
        ):
            frontier = _mapping(read_json(frontier_path), context=str(frontier_path))
            if (
                frontier.get("schema_version") != FRONTIER_SCHEMA_VERSION
                or frontier.get("status") != "complete"
                or frontier.get("raw_proposals_are_not_circuits") is not True
                or frontier.get("network_completion_is_exhaustive_through_registered_order")
                is not True
            ):
                raise RuntimeError(f"frontier-search artifact is malformed: {frontier_path}")
            metric_index_name = frontier.get("metric_index")
            metric_index_digest = frontier.get("metric_index_sha256")
            if not isinstance(metric_index_name, str) or not isinstance(metric_index_digest, str):
                raise TypeError(f"frontier search lacks its metric index: {frontier_path}")
            metric_index_path = frontier_path.parent / metric_index_name
            if sha256_file(metric_index_path) != metric_index_digest:
                raise RuntimeError(f"frontier metric-index digest mismatch: {metric_index_path}")
            frontier_metric_indexes.append(
                (frontier_path, load_frontier_metric_index(frontier_path.parent))
            )
            frontier_searches.append(frontier)
        network_veto_density_diagnostics: list[dict[str, object]] = []
        for diagnostic_path in sorted(
            config_path.parent.glob("network_veto_density_*/network_veto_density.json")
        ):
            diagnostic = _mapping(read_json(diagnostic_path), context=str(diagnostic_path))
            density_name = diagnostic.get("density_artifact")
            density_digest = diagnostic.get("density_artifact_sha256")
            source = _mapping(
                diagnostic.get("source"),
                context=f"{diagnostic_path}.source",
            )
            network_site_count = diagnostic.get("network_site_count")
            singleton_site_count = diagnostic.get("singleton_site_count")
            vetoed_site_count = diagnostic.get("vetoed_site_count")
            active_site_count = diagnostic.get("active_site_count")
            completed_frontiers = source.get("completed_frontiers")
            if (
                diagnostic.get("schema_version") != NETWORK_VETO_SCHEMA_VERSION
                or diagnostic.get("status") not in {"transition_found", "flat_stop"}
                or not isinstance(density_name, str)
                or not isinstance(density_digest, str)
                or not all(
                    isinstance(value, int) and value > 0
                    for value in (
                        network_site_count,
                        singleton_site_count,
                        vetoed_site_count,
                        active_site_count,
                    )
                )
                or cast(int, vetoed_site_count)
                < max(
                    cast(int, network_site_count),
                    cast(int, singleton_site_count),
                )
                or not same_science_directory(
                    source.get("scope_directory"),
                    config_path.parent,
                    logical_scope_directory,
                )
                or not isinstance(completed_frontiers, list)
                or not isinstance(diagnostic.get("curve"), list)
                or diagnostic.get("stop_before_mask_search")
                is not (diagnostic.get("status") == "flat_stop")
            ):
                raise RuntimeError(
                    f"network-veto density diagnostic is malformed: {diagnostic_path}"
                )
            density_path = diagnostic_path.parent / density_name
            if sha256_file(density_path) != density_digest:
                raise RuntimeError(f"network-veto density digest mismatch: {density_path}")
            density = validated_density(
                density_path,
                function_space="network_vetoed_residual",
            )
            if (
                diagnostic["curve"] != density["curve"]
                or diagnostic.get("transition_density") != density.get("transition_density")
                or diagnostic.get("status") != density.get("status")
            ):
                raise RuntimeError(
                    f"network-veto result disagrees with its density artifact: {diagnostic_path}"
                )
            for raw_frontier in cast(list[object], completed_frontiers):
                frontier_source = _mapping(
                    raw_frontier,
                    context=f"{diagnostic_path}.source.completed_frontiers[]",
                )
                directory = frontier_source.get("directory")
                result_digest = frontier_source.get("result_sha256")
                index_name = frontier_source.get("metric_index")
                index_digest = frontier_source.get("metric_index_sha256")
                if not all(
                    isinstance(value, str)
                    for value in (directory, result_digest, index_name, index_digest)
                ):
                    raise TypeError(
                        f"network-veto frontier provenance is malformed: {diagnostic_path}"
                    )
                frontier_dir = config_path.parent / cast(str, directory)
                if (
                    sha256_file(frontier_dir / FRONTIER_RESULT_FILENAME) != result_digest
                    or sha256_file(frontier_dir / cast(str, index_name)) != index_digest
                ):
                    raise RuntimeError(
                        f"network-veto frontier provenance changed: {diagnostic_path}"
                    )
            network_veto_density_diagnostics.append(diagnostic)
        network_veto_density_diagnostics.sort(
            key=lambda row: (
                cast(int, row["network_site_count"]),
                len(
                    cast(
                        list[object],
                        cast(dict[str, object], row["source"])["completed_frontiers"],
                    )
                ),
                cast(int, row["vetoed_site_count"]),
            )
        )
        disconnected_searches: list[dict[str, object]] = []
        for search_path in sorted(
            config_path.parent.glob("disconnected_search_config_*/disconnected_search.json")
        ):
            search = _mapping(read_json(search_path), context=str(search_path))
            metric_index_name = search.get("metric_index")
            metric_index_digest = search.get("metric_index_sha256")
            hypotheses = search.get("raw_minimized_hypotheses")
            verified = search.get("verified_disconnected_minsets")
            source = _mapping(search.get("source"), context=f"{search_path}.source")
            source_result = source.get("network_veto_result")
            source_result_digest = source.get("network_veto_result_sha256")
            if (
                search.get("schema_version") != DISCONNECTED_SEARCH_SCHEMA_VERSION
                or search.get("status") != "complete"
                or search.get("raw_hypotheses_are_not_circuits") is not True
                or not isinstance(metric_index_name, str)
                or not isinstance(metric_index_digest, str)
                or not isinstance(hypotheses, list)
                or not isinstance(verified, list)
                or not isinstance(source_result, str)
                or not isinstance(source_result_digest, str)
                or not isinstance(search.get("proposal_mask_count"), int)
                or not isinstance(search.get("successful_proposal_count"), int)
                or not isinstance(search.get("selected_start_count"), int)
                or search.get("unique_minimized_candidate_count") != len(hypotheses)
            ):
                raise RuntimeError(f"disconnected search is malformed: {search_path}")
            metric_index_path = search_path.parent / metric_index_name
            source_result_path = Path(source_result)
            if (
                sha256_file(metric_index_path) != metric_index_digest
                or sha256_file(source_result_path) != source_result_digest
            ):
                raise RuntimeError(f"disconnected-search provenance changed: {search_path}")
            metric_index = _mapping(
                read_json(metric_index_path),
                context=str(metric_index_path),
            )
            metric_count = metric_index.get("support_count")
            if (
                metric_index.get("schema_version") != DISCONNECTED_SEARCH_SCHEMA_VERSION
                or metric_index.get("kind") != "disconnected_metric_index"
                or not isinstance(metric_count, int)
                or metric_count <= 0
            ):
                raise RuntimeError(f"disconnected metric index is malformed: {metric_index_path}")
            size_counts: dict[int, int] = {}
            exact_ratios: list[float] = []
            for raw_hypothesis in cast(list[object], hypotheses):
                hypothesis = _mapping(
                    raw_hypothesis,
                    context=f"{search_path}.raw_minimized_hypotheses[]",
                )
                size = hypothesis.get("size")
                if not isinstance(size, int) or size < 2:
                    raise RuntimeError(f"disconnected hypothesis has invalid size: {search_path}")
                size_counts[size] = size_counts.get(size, 0) + 1
                ratio = hypothesis.get("maximum_proper_subset_fraction_of_full_probability")
                if hypothesis.get("exact_powerset_verified") is True:
                    if not isinstance(ratio, int | float):
                        raise RuntimeError(
                            f"exact disconnected hypothesis lacks subset evidence: {search_path}"
                        )
                    exact_ratios.append(float(ratio))
            compact_verified = []
            for raw_minset in cast(list[object], verified):
                minset = _mapping(
                    raw_minset,
                    context=f"{search_path}.verified_disconnected_minsets[]",
                )
                compact_verified.append(
                    {
                        key: value
                        for key, value in minset.items()
                        if key != "generating_random_masks"
                    }
                )
            disconnected_searches.append(
                {
                    "status": "complete",
                    "search_config": search["search_config"],
                    "transition_density": source["transition_density"],
                    "vetoed_site_count": source["vetoed_site_count"],
                    "active_site_count": search["active_site_count"],
                    "full_probability_threshold": search["full_probability_threshold"],
                    "proposal_mask_count": search["proposal_mask_count"],
                    "successful_proposal_count": search["successful_proposal_count"],
                    "selected_start_count": search["selected_start_count"],
                    "unique_minimized_candidate_count": len(hypotheses),
                    "candidate_size_counts": [
                        {"size": size, "count": count}
                        for size, count in sorted(size_counts.items())
                    ],
                    "exact_powerset_candidate_count": len(exact_ratios),
                    "minimum_exact_subset_fraction": min(exact_ratios, default=None),
                    "maximum_exact_subset_fraction": max(exact_ratios, default=None),
                    "metric_count": metric_count,
                    "metric_index_sha256": metric_index_digest,
                    "verified_disconnected_minsets": compact_verified,
                    "raw_hypotheses_are_not_circuits": True,
                }
            )
        network_minsets_by_sites: dict[tuple[Site, ...], set[str]] = {}
        network_minset_probabilities: dict[tuple[Site, ...], float] = {}
        network_minset_inputs: list[tuple[object, float | None, str, str]] = []
        for index, minset in enumerate(multisite_minsets):
            network_minset_inputs.append(
                (
                    minset["sites"],
                    float(cast(float, minset["correct_probability"])),
                    "fourier_stage_2",
                    f"verified_multisite_minsets[{index}].sites",
                )
            )
        for audit_index, audit in enumerate(recall_audits):
            local = _mapping(
                audit.get("local_truth_table"),
                context=f"recall_audits[{audit_index}].local_truth_table",
            )
            for index, sites in enumerate(
                cast(list[object], local["new_minsets_missed_by_fourier"])
            ):
                network_minset_inputs.append(
                    (
                        sites,
                        None,
                        "exact_local_recall",
                        f"recall_audits[{audit_index}].local_truth_table."
                        f"new_minsets_missed_by_fourier[{index}]",
                    )
                )
            for field, source in (
                ("new_verified_pair_minsets", "recall_pair_verification"),
                ("new_verified_triple_minsets", "recall_triple_verification"),
            ):
                for index, raw_row in enumerate(cast(list[object], audit[field])):
                    row = _mapping(
                        raw_row,
                        context=f"recall_audits[{audit_index}].{field}[{index}]",
                    )
                    network_minset_inputs.append(
                        (
                            row["sites"],
                            float(cast(float, row["correct_probability"])),
                            source,
                            f"recall_audits[{audit_index}].{field}[{index}].sites",
                        )
                    )
        for frontier_index, frontier in enumerate(frontier_searches):
            rows = frontier.get("new_verified_relative_minsets")
            if not isinstance(rows, list):
                raise TypeError("frontier search lacks its verified relative minsets")
            for index, raw_row in enumerate(cast(list[object], rows)):
                row = _mapping(
                    raw_row,
                    context=(
                        f"frontier_searches[{frontier_index}]."
                        f"new_verified_relative_minsets[{index}]"
                    ),
                )
                probability = row.get("correct_probability")
                if not isinstance(probability, (int, float)):
                    raise TypeError("frontier verified minset lacks a probability")
                network_minset_inputs.append(
                    (
                        row.get("sites"),
                        float(probability),
                        "relative_frontier_search",
                        (
                            f"frontier_searches[{frontier_index}]."
                            f"new_verified_relative_minsets[{index}].sites"
                        ),
                    )
                )
        for raw_sites, correct_probability, source, context in network_minset_inputs:
            validated = validated_sites(raw_sites, context=context, require_nonempty=True)
            canonical = tuple(sorted(Site(*site_key(site, context=context)) for site in validated))
            if len(canonical) < 2:
                raise RuntimeError(f"network overlay received a singleton: {context}")
            network_minsets_by_sites.setdefault(canonical, set()).add(source)
            if correct_probability is not None:
                previous_probability = network_minset_probabilities.get(canonical)
                if (
                    previous_probability is not None
                    and abs(previous_probability - correct_probability) > 2.0e-6
                ):
                    raise RuntimeError(
                        f"duplicate verified minset probabilities disagree: {canonical}"
                    )
                network_minset_probabilities[canonical] = correct_probability

        unfiltered_network_minset_count = len(network_minsets_by_sites)
        subset_separation: dict[str, object] = {
            "enabled": False,
            "maximum_proper_subset_correct_probability": None,
            "maximum_proper_subset_fraction_of_full_probability": None,
            "unfiltered_multisite_minset_count": unfiltered_network_minset_count,
            "passing_multisite_minset_count": unfiltered_network_minset_count,
            "subset_metric_index": None,
            "subset_metric_index_sha256": None,
            "subset_metric_count": None,
        }
        maximum_subset_metrics: dict[tuple[Site, ...], SubsetMetric] = {}
        subset_probability_ratios: dict[tuple[Site, ...], float] = {}
        if (
            sufficiency_criterion == "clean_correct_probability_minus_absolute_tolerance"
            and network_minsets_by_sites
        ):
            subset_metrics = ensure_subset_metric_index(config_path.parent)
            for frontier_path, frontier_metrics in frontier_metric_indexes:
                for sites, metric in frontier_metrics.items():
                    previous = subset_metrics.get(sites)
                    if previous is not None:
                        raise RuntimeError(
                            f"frontier metric repeats the base subset cache: {frontier_path}: {sites}"
                        )
                    subset_metrics[sites] = metric
            relative_criterion = RelativeProperSubsetCriterion(
                MAX_PROPER_SUBSET_PROBABILITY_FRACTION
            )
            for sites in network_minsets_by_sites:
                indexed_metric = subset_metrics.get(sites)
                if indexed_metric is None:
                    raise RuntimeError(f"verified minset is absent from subset index: {sites}")
                probability = network_minset_probabilities.get(sites)
                if probability is None:
                    probability = indexed_metric.correct_probability
                    network_minset_probabilities[sites] = probability
                elif abs(probability - indexed_metric.correct_probability) > 2.0e-6:
                    raise RuntimeError(f"verified minset disagrees with subset index: {sites}")
                maximum_subset_metrics[sites] = maximum_proper_subset_metric(
                    sites,
                    subset_metrics,
                )
                subset_probability_ratios[sites] = (
                    maximum_subset_metrics[sites].correct_probability / probability
                )
            network_minsets_by_sites = {
                sites: sources
                for sites, sources in network_minsets_by_sites.items()
                if passes_relative_proper_subset_criterion(
                    subset_metrics[sites],
                    maximum_subset_metrics[sites],
                    relative_criterion,
                )
            }
            index_path = subset_index_path(config_path.parent)
            subset_separation = {
                "enabled": True,
                "maximum_proper_subset_correct_probability": None,
                "maximum_proper_subset_fraction_of_full_probability": (
                    MAX_PROPER_SUBSET_PROBABILITY_FRACTION
                ),
                "unfiltered_multisite_minset_count": unfiltered_network_minset_count,
                "passing_multisite_minset_count": len(network_minsets_by_sites),
                "subset_metric_index": index_path.name,
                "subset_metric_index_sha256": sha256_file(index_path),
                "subset_metric_count": len(subset_metrics),
                "frontier_metric_indexes": [
                    {
                        "directory": frontier_path.parent.name,
                        "metric_index": cast(str, frontier["metric_index"]),
                        "metric_index_sha256": cast(str, frontier["metric_index_sha256"]),
                    }
                    for frontier_path, frontier in zip(
                        (path for path, _metrics in frontier_metric_indexes),
                        frontier_searches,
                        strict=True,
                    )
                ],
            }
        if set(network_minsets_by_sites) - set(network_minset_probabilities):
            raise RuntimeError("network overlay contains a minset without an exact probability")
        network_minset_site_sets = tuple(
            sorted(network_minsets_by_sites, key=lambda sites: (len(sites), sites))
        )
        network_verified_multisite_minsets = [
            {
                "size": len(sites),
                "sites": [{"token_index": site.token_index, "layer": site.layer} for site in sites],
                "sources": sorted(network_minsets_by_sites[sites]),
                "correct_probability": network_minset_probabilities[sites],
                "maximum_proper_subset_correct_probability": (
                    None
                    if sites not in maximum_subset_metrics
                    else maximum_subset_metrics[sites].correct_probability
                ),
                "maximum_proper_subset": (
                    None
                    if sites not in maximum_subset_metrics
                    else [
                        {"token_index": site.token_index, "layer": site.layer}
                        for site in maximum_subset_metrics[sites].sites
                    ]
                ),
                "maximum_proper_subset_fraction_of_full_probability": (
                    subset_probability_ratios.get(sites)
                ),
            }
            for sites in network_minset_site_sets
        ]
        network_payload = [
            {
                "id": f"size_{network.minset_size}_component_{network.component_index}",
                "minset_size": network.minset_size,
                "component_index": network.component_index,
                "clique_expansion": True,
                "minset_indices": list(network.minset_indices),
                "sites": [
                    {
                        "token_index": site.token_index,
                        "layer": site.layer,
                        "cluster_index": next(
                            cluster.cluster_index
                            for cluster in network.clusters
                            if site in cluster.sites
                        ),
                    }
                    for site in network.sites
                ],
                "edges": [
                    {
                        "source": {
                            "token_index": edge.source.token_index,
                            "layer": edge.source.layer,
                        },
                        "target": {
                            "token_index": edge.target.token_index,
                            "layer": edge.target.layer,
                        },
                        "minset_indices": list(edge.minset_indices),
                    }
                    for edge in network.edges
                ],
                "partner_profile_clusters": [
                    {
                        "cluster_index": cluster.cluster_index,
                        "sites": [
                            {"token_index": site.token_index, "layer": site.layer}
                            for site in cluster.sites
                        ],
                        "minimum_partner_jaccard": cluster.minimum_partner_jaccard,
                        "mean_partner_jaccard": cluster.mean_partner_jaccard,
                    }
                    for cluster in network.clusters
                ],
            }
            for network in cluster_minset_hypergraph_networks(
                network_minset_site_sets,
                minimum_similarity=0.5,
            )
        ]
        alternative_probability_sufficiency: dict[str, object] | None = None
        if all_singleton_rows:
            if sufficiency_mapping is None:
                raise RuntimeError("singleton sufficiency mapping was not initialized")
            clean_probability = _number(
                sufficiency_mapping,
                "clean_correct_probability",
                context=f"{singleton_path}.sufficiency",
            )
            threshold_logit_diff = _number(
                sufficiency_mapping,
                "threshold_logit_diff",
                context=f"{singleton_path}.sufficiency",
            )
            if sufficiency_criterion == "raw_logit_gap_recovery":
                probability_threshold = max(0.0, clean_probability - 0.10)
                passing_alternative = [
                    row
                    for row in all_singleton_rows
                    if float(cast(float, row["correct_probability"])) >= probability_threshold
                    and bool(row["accuracy"])
                ]
                strict_keys = {
                    site_key(
                        cast(dict[str, object], row["site"]),
                        context="verified singleton minset",
                    )
                    for row in singleton_rows
                }
                alternative_keys = {
                    site_key(
                        cast(dict[str, object], row["site"]),
                        context="alternative singleton minset",
                    )
                    for row in passing_alternative
                }
                if not strict_keys.issubset(alternative_keys):
                    raise RuntimeError(
                        "the looser probability criterion lost a preregistered singleton"
                    )
                invalidated_multisites = sum(
                    any(
                        site_key(site, context="verified multi-site minset") in alternative_keys
                        for site in cast(list[dict[str, object]], minset["sites"])
                    )
                    for minset in multisite_minsets
                )
                alternative_probability_sufficiency = {
                    "status": "derived_diagnostic_requires_new_singleton_vetoed_spectrum",
                    "rule": "clean_correct_probability_minus_0.10",
                    "absolute_probability_tolerance": 0.10,
                    "clean_correct_probability": clean_probability,
                    "threshold_correct_probability": probability_threshold,
                    "preregistered_threshold_correct_probability": 1.0
                    / (1.0 + math.exp(-threshold_logit_diff)),
                    "passing_singleton_count": len(passing_alternative),
                    "additional_singleton_count": len(alternative_keys - strict_keys),
                    "current_multisite_minsets_invalidated_by_singleton_count": (
                        invalidated_multisites
                    ),
                }
            else:
                tolerance = _number(
                    sufficiency_mapping,
                    "absolute_probability_tolerance",
                    context=f"{singleton_path}.sufficiency",
                )
                threshold_probability = _number(
                    sufficiency_mapping,
                    "threshold_correct_probability",
                    context=f"{singleton_path}.sufficiency",
                )
                if (
                    tolerance != 0.10
                    or abs(threshold_probability - (clean_probability - tolerance)) > 1.0e-12
                ):
                    raise RuntimeError(
                        f"probability sufficiency contract is inconsistent: {singleton_path}"
                    )
        relative_path = (
            Path("fourier-circuits")
            / f"lineage_{lineage_id}"
            / model_key
            / condition
            / function_id
            / f"clean_{clean_step:06d}_dirty_{dirty_step:06d}"
            / f"{scope_key}.json"
        )
        chunk = {
            "schema_version": 1,
            "status": status,
            "lineage": lineage,
            "model": model,
            "task": task,
            "sites": config.get("sites"),
            "sufficiency_criterion": sufficiency_criterion,
            "site_grid": site_grid,
            "unrestricted_density_curve": unrestricted["curve"],
            "unrestricted_transition_density": unrestricted.get("transition_density"),
            "residual_density_curve": None if residual is None else residual["curve"],
            "residual_transition_density": (
                None if residual is None else residual.get("transition_density")
            ),
            "network_veto_density_diagnostics": network_veto_density_diagnostics,
            "disconnected_searches": disconnected_searches,
            "density_stability_warning": density_warning,
            "sufficiency": sufficiency,
            "exhaustive_singleton_count": singleton_count,
            "verified_singleton_minsets": singleton_rows,
            "verified_multisite_minsets": multisite_minsets,
            "network_verified_multisite_minsets": network_verified_multisite_minsets,
            "proper_subset_separation": subset_separation,
            "minset_networks": network_payload,
            "partner_profile_clustering": {
                "method": (
                    "profile_seeded_deterministic_complete_link_neighbor_jaccard_"
                    "with_minset_cannot_link"
                ),
                "minimum_similarity": 0.5,
                "hyperedges_preserved_for_higher_order_minsets": True,
                "clusters_are_descriptive_not_identified_pathways": True,
            },
            "recall_audits": recall_audits,
            "frontier_searches": [
                {
                    "status": frontier["status"],
                    "frontier_config": frontier["frontier_config"],
                    "criterion": frontier["criterion"],
                    "phase_proposal_counts": frontier["phase_proposal_counts"],
                    "evaluated_support_count": frontier["evaluated_support_count"],
                    "new_verified_relative_minset_count": len(
                        cast(list[object], frontier["new_verified_relative_minsets"])
                    ),
                }
                for frontier in frontier_searches
            ],
            "alternative_probability_sufficiency": alternative_probability_sufficiency,
            "raw_heavy_fourier_hypotheses": heavy_hypotheses,
            "legacy_sparse_discovery_minsets": legacy_minsets,
            "singleton_search_is_exhaustive": singleton_path.is_file(),
            "legacy_discovery_is_only_a_lower_bound": legacy_identity_run,
            "raw_fourier_candidates_are_not_circuits": True,
        }
        digest, byte_count = _write_compact_json(root / "site" / "data" / relative_path, chunk)
        entries.append(
            {
                "model": model_key,
                "lineage": lineage,
                "condition": condition,
                "function_id": function_id,
                "clean_step": clean_step,
                "dirty_step": dirty_step,
                "scope": scope_key,
                "status": status,
                "sufficiency_criterion": sufficiency_criterion,
                "singleton_minset_count": len(singleton_rows),
                "multisite_minset_count": len(network_verified_multisite_minsets),
                "unfiltered_multisite_minset_count": unfiltered_network_minset_count,
                "fourier_multisite_minset_count": len(multisite_minsets),
                "raw_hypothesis_count": len(heavy_hypotheses),
                "legacy_minset_count": len(legacy_minsets),
                "url": relative_path.as_posix(),
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    imported_directory = root / "site/data/fourier-circuit-imports"
    for import_path in sorted(imported_directory.glob("*.json")):
        imported = _mapping(read_json(import_path), context=str(import_path))
        schema_version = imported.get("schema_version")
        imported_lineage = (
            validated_lineage_payload(
                imported.get("lineage"),
                context=f"{import_path}.lineage",
            )
            if schema_version == 1
            else lineage_identity_payload(
                imported.get("lineage"),
                context=f"{import_path}.lineage",
            )
        )
        imported_entries = imported.get("entries")
        exporter_source_sha256 = imported.get("exporter_source_sha256")
        if (
            schema_version not in {1, 2}
            or imported.get("kind") != "fourier_lineage_export"
            or not isinstance(exporter_source_sha256, str)
            or len(exporter_source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in exporter_source_sha256)
            or not isinstance(imported_entries, list)
            or not imported_entries
        ):
            raise RuntimeError(f"imported Fourier lineage manifest is malformed: {import_path}")
        lineage_id = cast(str, imported_lineage["id"])
        for index, raw_entry in enumerate(cast(list[object], imported_entries)):
            entry = _mapping(raw_entry, context=f"{import_path}.entries[{index}]")
            entry_lineage = validated_lineage_payload(
                entry.get("lineage"),
                context=f"{import_path}.entries[{index}].lineage",
            )
            url = entry.get("url")
            digest = entry.get("sha256")
            byte_count = entry.get("bytes")
            required_strings = (
                "model",
                "condition",
                "function_id",
                "scope",
                "status",
                "sufficiency_criterion",
            )
            if (
                (
                    entry_lineage != imported_lineage
                    if schema_version == 1
                    else lineage_identity_payload(
                        entry_lineage,
                        context=f"{import_path}.entries[{index}].lineage_identity",
                    )
                    != imported_lineage
                )
                or not isinstance(url, str)
                or not url.startswith(f"fourier-circuits/lineage_{lineage_id}/")
                or Path(url).is_absolute()
                or ".." in Path(url).parts
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(byte_count, int)
                or byte_count <= 0
                or any(not isinstance(entry.get(field), str) for field in required_strings)
                or not isinstance(entry.get("clean_step"), int)
                or not isinstance(entry.get("dirty_step"), int)
            ):
                raise RuntimeError(
                    f"imported Fourier lineage entry is malformed: {import_path}: {index}"
                )
            chunk_path = root / "site/data" / url
            if (
                not chunk_path.is_file()
                or chunk_path.stat().st_size != byte_count
                or sha256_file(chunk_path) != digest
            ):
                raise RuntimeError(
                    f"imported Fourier lineage chunk is missing or changed: {chunk_path}"
                )
            chunk = _mapping(read_json(chunk_path), context=str(chunk_path))
            chunk_model = _mapping(chunk.get("model"), context=f"{chunk_path}.model")
            chunk_task = _mapping(chunk.get("task"), context=f"{chunk_path}.task")
            if (
                chunk.get("lineage") != entry_lineage
                or chunk_model.get("model_key") != entry["model"]
                or chunk_task.get("function_id") != entry["function_id"]
                or chunk.get("status") != entry["status"]
                or chunk.get("sufficiency_criterion") != entry["sufficiency_criterion"]
            ):
                raise RuntimeError(
                    f"imported Fourier lineage entry disagrees with its chunk: {chunk_path}"
                )
            entries.append(entry)

    unique_entries: dict[tuple[object, ...], dict[str, object]] = {}
    seen_urls: set[str] = set()
    for entry in entries:
        lineage = validated_lineage_payload(entry.get("lineage"), context="Fourier entry lineage")
        key = (
            lineage["id"],
            entry["model"],
            entry["condition"],
            entry["function_id"],
            entry["clean_step"],
            entry["dirty_step"],
            entry["scope"],
            entry["sufficiency_criterion"],
        )
        url = cast(str, entry["url"])
        if key in unique_entries or url in seen_urls:
            raise RuntimeError(f"duplicate Fourier lineage export identity: {key}")
        unique_entries[key] = entry
        seen_urls.add(url)
    entries = list(unique_entries.values())
    entries.sort(
        key=lambda entry: (
            cast(dict[str, object], entry["lineage"])["kind"] != "registered_hardware",
            entry["function_id"] != "add_5",
            entry["sufficiency_criterion"] != "clean_correct_probability_minus_absolute_tolerance",
            str(entry["function_id"]),
        )
    )
    lineage_exports: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    for entry in entries:
        lineage = validated_lineage_payload(entry["lineage"], context="Fourier entry lineage")
        lineage_id = cast(str, lineage["id"])
        lineage_identity = lineage_identity_payload(
            lineage,
            context="Fourier entry lineage identity",
        )
        previous = lineage_exports.get(lineage_id)
        if previous is None:
            lineage_exports[lineage_id] = (lineage_identity, [entry])
        else:
            if previous[0] != lineage_identity:
                raise RuntimeError(f"Fourier lineage metadata changed within export: {lineage_id}")
            previous[1].append(entry)
    for lineage_id, (lineage_identity, lineage_entries) in lineage_exports.items():
        entry_lineages = [
            validated_lineage_payload(
                entry["lineage"],
                context=f"Fourier lineage {lineage_id} entry",
            )
            for entry in lineage_entries
        ]
        homogeneous_lineage = all(lineage == entry_lineages[0] for lineage in entry_lineages[1:])
        write_json(
            root / "site/data/fourier-circuit-lineages" / f"{lineage_id}.json",
            {
                "schema_version": 1 if homogeneous_lineage else 2,
                "kind": "fourier_lineage_export",
                "exporter_source_sha256": _site_export_source_sha256(),
                "lineage": entry_lineages[0] if homogeneous_lineage else lineage_identity,
                "entries": lineage_entries,
            },
        )
    return {"entries": entries}, len(entries)


def _export_switched_answer_minsets(root: Path) -> tuple[dict[str, object], int]:
    """Export registered composite layer-swap analyses without inventing absent stages."""

    audit_path = root / "artifacts/plans/switched_answer_minsets/tokenization_audit_v2.json"
    prompt_audit: dict[str, object] | None = None
    if audit_path.is_file():
        audit = _mapping(read_json(audit_path), context=str(audit_path))
        if audit.get("schema_version") != SWITCHED_ANSWER_SCHEMA_VERSION:
            raise RuntimeError("switched-answer token audit has the wrong schema")
        raw_prompt_audit = audit.get("prompt_audit")
        prompt_audit = _mapping(raw_prompt_audit, context=f"{audit_path}.prompt_audit")
        terminators = prompt_audit.get("terminator_sites")
        if not isinstance(terminators, list) or len(terminators) != 5:
            raise RuntimeError("switched-answer token audit must contain five terminators")

    base = (
        root
        / "artifacts/runs/olmo3-7b/correct/seed_20260715"
        / "answer_lookup_checkpoint_transfer_minsets/add_5"
        / "donor_001500_recipient_000000"
        / "target_correct_recovery"
    )
    entries: list[dict[str, object]] = []
    measured = 0
    for interface in SWITCHED_ANSWER_INTERFACES:
        for destination_index in (0, 1, 3, 4):
            destination_label = "ABCDE"[destination_index]
            directory = base / interface / f"destination_{destination_label.lower()}"
            config_path = directory / "config.json"
            endpoint_path = directory / "endpoint_gate.json"
            density_path = directory / "density_sweep.json"
            minset_path = directory / "verified_minsets.json"
            entry: dict[str, object] = {
                "interface": interface,
                "destination_choice_index": destination_index,
                "destination_choice_label": destination_label,
                "correct_choice_index": SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
                "correct_choice_label": "C",
                "donor_step": 1_500,
                "recipient_step": 0,
                "site_semantics": "one layerwise simultaneous two-position answer-terminator swap",
                "prompt_audit": prompt_audit,
                "status": "unprocessed",
                "endpoint": None,
                "density": None,
                "search": None,
            }
            if config_path.is_file():
                wrapper = _mapping(read_json(config_path), context=str(config_path))
                if wrapper.get("schema_version") != SWITCHED_ANSWER_SCHEMA_VERSION:
                    raise RuntimeError(f"switched-answer config has the wrong schema: {config_path}")
                config = _mapping(wrapper.get("config"), context=f"{config_path}.config")
                task = _mapping(config.get("task"), context=f"{config_path}.config.task")
                if (
                    task.get("function_id") != "add_5"
                    or task.get("interface") != interface
                    or task.get("destination_choice_index") != destination_index
                    or task.get("correct_choice_index") != SWITCHED_ANSWER_CORRECT_CHOICE_INDEX
                    or task.get("target_choice_index") != SWITCHED_ANSWER_CORRECT_CHOICE_INDEX
                ):
                    raise RuntimeError(f"switched-answer config identity mismatch: {config_path}")
                entry["config_sha256"] = sha256_file(config_path)
            if endpoint_path.is_file():
                endpoint = _mapping(read_json(endpoint_path), context=str(endpoint_path))
                if endpoint.get("schema_version") != SWITCHED_ANSWER_SCHEMA_VERSION or endpoint.get(
                    "status"
                ) not in {"passed", "failed"}:
                    raise RuntimeError(f"switched-answer endpoint is malformed: {endpoint_path}")
                for corner_name in ("all_dirty", "all_clean_swap"):
                    corner = _mapping(
                        endpoint.get(corner_name),
                        context=f"{endpoint_path}.{corner_name}",
                    )
                    logits = corner.get("candidate_logits")
                    if not isinstance(logits, list) or len(logits) != 5:
                        raise RuntimeError(f"switched-answer endpoint lacks five logits: {endpoint_path}")
                    _number(corner, "target_probability", context=str(endpoint_path))
                    _number(corner, "raw_logit_diff", context=str(endpoint_path))
                    if not isinstance(corner.get("target_argmax"), bool):
                        raise TypeError(f"switched-answer endpoint argmax is invalid: {endpoint_path}")
                _number(endpoint, "sufficiency_probability_threshold", context=str(endpoint_path))
                entry["endpoint"] = endpoint
                entry["endpoint_sha256"] = sha256_file(endpoint_path)
                entry["status"] = "endpoint_passed" if endpoint["status"] == "passed" else "endpoint_failed"
                measured += 1
            if density_path.is_file():
                density = _mapping(read_json(density_path), context=str(density_path))
                if density.get("schema_version") != SWITCHED_ANSWER_SCHEMA_VERSION or density.get(
                    "status"
                ) not in {"complete", "flat_stop"}:
                    raise RuntimeError(f"switched-answer density is malformed: {density_path}")
                points = density.get("points")
                if not isinstance(points, list) or len(points) != 16:
                    raise RuntimeError(f"switched-answer density must contain 16 points: {density_path}")
                for raw_point in cast(list[object], points):
                    point = _mapping(raw_point, context=f"{density_path}.points[]")
                    for field in (
                        "density",
                        "mean_target_probability",
                        "target_probability_variance",
                        "target_accuracy",
                        "mean_raw_logit_diff",
                        "raw_logit_diff_variance",
                    ):
                        _number(point, field, context=f"{density_path}.points[]")
                entry["density"] = density
                entry["density_sha256"] = sha256_file(density_path)
                entry["status"] = (
                    "density_complete" if density["status"] == "complete" else "density_flat_stop"
                )
            if minset_path.is_file():
                search = _mapping(read_json(minset_path), context=str(minset_path))
                if search.get("schema_version") != SWITCHED_ANSWER_SCHEMA_VERSION or search.get(
                    "status"
                ) not in {"partial", "complete"}:
                    raise RuntimeError(f"switched-answer minset output is malformed: {minset_path}")
                order = search.get("exhaustive_through_order")
                rows = search.get("minsets")
                if not isinstance(order, int) or not 1 <= order <= 6 or not isinstance(rows, list):
                    raise RuntimeError(f"switched-answer search coverage is malformed: {minset_path}")
                for raw_row in cast(list[object], rows):
                    row = _mapping(raw_row, context=f"{minset_path}.minsets[]")
                    layers = row.get("layers")
                    size = row.get("size")
                    subset_layers = row.get("maximum_proper_subset_layers")
                    if (
                        not isinstance(layers, list)
                        or not all(isinstance(layer, int) and 0 <= layer < 32 for layer in layers)
                        or layers != sorted(set(layers))
                        or size != len(layers)
                        or not isinstance(subset_layers, list)
                        or not set(subset_layers).issubset(layers)
                    ):
                        raise RuntimeError(f"switched-answer minset layers are malformed: {minset_path}")
                    for field in (
                        "target_probability",
                        "raw_logit_diff",
                        "sufficiency_margin",
                        "maximum_proper_subset_probability",
                    ):
                        _number(row, field, context=f"{minset_path}.minsets[]")
                entry["search"] = search
                entry["search_sha256"] = sha256_file(minset_path)
                entry["status"] = "search_complete" if search["status"] == "complete" else "search_partial"
            entries.append(entry)
    return {
        "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
        "entries": entries,
        "registered_entry_count": 8,
        "measured_entry_count": measured,
    }, measured


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
    (
        answer_lookup_manifest,
        real_answer_lookup_files,
        complete_answer_lookup_files,
    ) = _export_answer_lookup(root)
    fourier_circuit_manifest, real_fourier_circuit_files = _export_fourier_circuits(root)
    switched_answer_minset_manifest, real_switched_answer_minset_files = (
        _export_switched_answer_minsets(root)
    )
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
            "real_answer_lookup_files": real_answer_lookup_files,
            "complete_answer_lookup_files": complete_answer_lookup_files,
            "answer_lookup_manifest": answer_lookup_manifest,
            "real_fourier_circuit_files": real_fourier_circuit_files,
            "fourier_circuit_manifest": fourier_circuit_manifest,
            "real_switched_answer_minset_files": real_switched_answer_minset_files,
            "switched_answer_minset_manifest": switched_answer_minset_manifest,
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
            "real_answer_lookup_files": real_answer_lookup_files,
            "complete_answer_lookup_files": complete_answer_lookup_files,
            "real_fourier_circuit_files": real_fourier_circuit_files,
            "real_switched_answer_minset_files": real_switched_answer_minset_files,
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
            "answer_lookup_manifest": answer_lookup_manifest,
            "fourier_circuit_manifest": fourier_circuit_manifest,
            "switched_answer_minset_manifest": switched_answer_minset_manifest,
        },
    )


if __name__ == "__main__":
    main()
