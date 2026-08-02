"""Gated observational alignment measurements at exact decoder boundaries."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
    RunKey,
)
from oocr_training_dynamics.data import FUNCTION_BY_ID, ReflectionRecord
from oocr_training_dynamics.models import ModelSpec, get_model_spec
from oocr_training_dynamics.patching import (
    PatchingPlan,
    TokenPositionPair,
    reverse_token_position_pairs,
)
from oocr_training_dynamics.representation_alignment import (
    REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE,
    REPRESENTATION_ALIGNMENT_INTERFACES,
    REPRESENTATION_ALIGNMENT_KIND,
    REPRESENTATION_ALIGNMENT_METRICS,
    REPRESENTATION_ALIGNMENT_SCHEMA_VERSION,
    representation_alignment_path,
)
from oocr_training_dynamics.runtime_models import load_processor, resolve_decoder_blocks
from oocr_training_dynamics.runtime_patching import (
    TEMPORAL_PRIORITY_LABELS,
    PromptCounterfactualSpec,
    PromptPatchView,
    _hidden_tensor,
    _input_hidden,
    _load_checkpoint_model,
    _prompt_counterfactual_spec,
    _prompt_counterfactual_views,
    _prompt_patch_view,
    _release_model,
    _resolve_patch_targets,
    _seeded_priority_temporal_order,
    _selected_records,
    _temporal_direction,
    _temporal_mode,
    _temporal_priority_tier,
)

ActivationBank = dict[PatchingInterface, tuple[t.Tensor, ...]]


@dataclass(frozen=True)
class RepresentationAlignmentGrid:
    """Token-by-layer scalar comparisons between two unpatched activation banks."""

    cosine_similarity: tuple[tuple[float, ...], ...]
    l2_distance: tuple[tuple[float, ...], ...]
    source_norm: tuple[tuple[float, ...], ...]
    recipient_norm: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class AlignmentSourceRecord:
    """One donor/source prompt and its CPU-resident activation boundaries."""

    counterfactual: PromptCounterfactualSpec | None
    source_view: PromptPatchView
    recipient_view: PromptPatchView
    positions: tuple[TokenPositionPair, ...]
    activations: ActivationBank


def _capture_alignment_interfaces(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    interfaces: tuple[PatchingInterface, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    *,
    token_indices: tuple[int, ...] | None = None,
) -> ActivationBank:
    """Capture several exact activation boundaries in one unpatched model forward."""

    if not interfaces or len(set(interfaces)) != len(interfaces):
        raise ValueError("alignment interfaces must be non-empty and unique")
    if any(interface not in REPRESENTATION_ALIGNMENT_INTERFACES for interface in interfaces):
        raise ValueError("representation alignment is undefined for weight interfaces")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("alignment capture requires one unbatched prompt")
    if token_indices is not None and (
        not token_indices
        or len(set(token_indices)) != len(token_indices)
        or any(index < 0 or index >= input_ids.shape[1] for index in token_indices)
    ):
        raise ValueError("alignment token indices must be unique positions in the prompt")

    def selected_vectors(hidden: t.Tensor) -> t.Tensor:
        if hidden.shape[0] != 1:
            raise RuntimeError("alignment capture requires one unbatched prompt")
        selected = hidden[0] if token_indices is None else hidden[0, list(token_indices)]
        return selected.detach().cpu().clone()

    targets_by_interface = {
        interface: _resolve_patch_targets(blocks, interface) for interface in interfaces
    }
    captured: dict[PatchingInterface, list[t.Tensor | None]] = {
        interface: [None] * len(targets) for interface, targets in targets_by_interface.items()
    }
    handles: list[Any] = []
    for interface, targets in targets_by_interface.items():
        for layer, target in enumerate(targets):
            if target.capture_input:

                def input_hook(
                    _module: t.nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    selected_interface: PatchingInterface = interface,
                    index: int = layer,
                ) -> None:
                    hidden = _input_hidden(args, kwargs)
                    captured[selected_interface][index] = selected_vectors(hidden)

                handles.append(
                    target.module.register_forward_pre_hook(input_hook, with_kwargs=True)
                )
            else:

                def output_hook(
                    _module: t.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    selected_interface: PatchingInterface = interface,
                    index: int = layer,
                ) -> None:
                    hidden = _hidden_tensor(output)
                    captured[selected_interface][index] = selected_vectors(hidden)

                handles.append(target.module.register_forward_hook(output_hook))
    try:
        with t.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    result: ActivationBank = {}
    for interface, values in captured.items():
        if any(value is None for value in values):
            raise RuntimeError(
                f"not every decoder layer produced {interface.value} for alignment"
            )
        result[interface] = tuple(cast(t.Tensor, value) for value in values)
    return result


def _representation_alignment_grid(
    source_activations: tuple[t.Tensor, ...],
    recipient_activations: tuple[t.Tensor, ...],
    positions: tuple[TokenPositionPair, ...],
) -> RepresentationAlignmentGrid:
    """Compute cosine, raw L2 distance, and both norms with float32 accumulation."""

    if not positions:
        raise ValueError("representation alignment requires at least one token position")
    if not source_activations or len(source_activations) != len(recipient_activations):
        raise ValueError("source and recipient must contain the same non-zero layer count")
    layer_count = len(source_activations)
    cosine = [[float("nan")] * layer_count for _ in positions]
    distance = [[float("nan")] * layer_count for _ in positions]
    source_norm = [[float("nan")] * layer_count for _ in positions]
    recipient_norm = [[float("nan")] * layer_count for _ in positions]
    for layer, (source_layer, recipient_layer) in enumerate(
        zip(source_activations, recipient_activations, strict=True)
    ):
        if (
            source_layer.ndim != 2
            or recipient_layer.ndim != 2
            or source_layer.shape[1] != recipient_layer.shape[1]
        ):
            raise ValueError(
                "alignment activations must have compatible [sequence, hidden] shapes"
            )
        if any(
            position.source_index >= source_layer.shape[0]
            or position.recipient_index >= recipient_layer.shape[0]
            for position in positions
        ):
            raise ValueError("alignment position lies outside a captured activation sequence")
        source_vectors = t.stack(
            [source_layer[position.source_index] for position in positions]
        ).to(dtype=t.float32)
        recipient_vectors = t.stack(
            [recipient_layer[position.recipient_index] for position in positions]
        ).to(dtype=t.float32)
        source_norms = t.linalg.vector_norm(source_vectors, dim=-1)
        recipient_norms = t.linalg.vector_norm(recipient_vectors, dim=-1)
        if (
            not bool(t.isfinite(source_vectors).all())
            or not bool(t.isfinite(recipient_vectors).all())
            or not bool(t.isfinite(source_norms).all())
            or not bool(t.isfinite(recipient_norms).all())
            or bool((source_norms <= 0).any())
            or bool((recipient_norms <= 0).any())
        ):
            raise RuntimeError("alignment vectors and norms must be finite and non-zero")
        cosines = (
            (source_vectors * recipient_vectors).sum(dim=-1)
            / (source_norms * recipient_norms)
        ).clamp(min=-1.0, max=1.0)
        distances = t.linalg.vector_norm(source_vectors - recipient_vectors, dim=-1)
        if not bool(t.isfinite(cosines).all()) or not bool(t.isfinite(distances).all()):
            raise RuntimeError("representation alignment produced a non-finite metric")
        for token_index in range(len(positions)):
            cosine[token_index][layer] = float(cosines[token_index].item())
            distance[token_index][layer] = float(distances[token_index].item())
            source_norm[token_index][layer] = float(source_norms[token_index].item())
            recipient_norm[token_index][layer] = float(recipient_norms[token_index].item())
    if any(
        not math.isfinite(value)
        for grids in (cosine, distance, source_norm, recipient_norm)
        for row in grids
        for value in row
    ):
        raise RuntimeError("representation alignment grid contains an unfilled cell")
    return RepresentationAlignmentGrid(
        cosine_similarity=tuple(tuple(row) for row in cosine),
        l2_distance=tuple(tuple(row) for row in distance),
        source_norm=tuple(tuple(row) for row in source_norm),
        recipient_norm=tuple(tuple(row) for row in recipient_norm),
    )


def _comparison_views(
    processor: Any,
    record: ReflectionRecord,
    mode: PatchingMode,
) -> tuple[
    PromptCounterfactualSpec | None,
    PromptPatchView,
    PromptPatchView,
    tuple[TokenPositionPair, ...],
]:
    if mode.uses_prompt_counterfactual:
        counterfactual = _prompt_counterfactual_spec(record, mode)
        source_view, recipient_view = _prompt_counterfactual_views(
            processor,
            record,
            mode,
            counterfactual,
        )
    else:
        counterfactual = None
        source_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        recipient_view = source_view
    positions = reverse_token_position_pairs(
        source_view.anchor_index,
        recipient_view.anchor_index,
        source_view.stop_index,
        recipient_view.stop_index,
    )
    return counterfactual, source_view, recipient_view, positions


def _serialize_alignment_record(
    record: ReflectionRecord,
    mode: PatchingMode,
    counterfactual: PromptCounterfactualSpec | None,
    source_view: PromptPatchView,
    recipient_view: PromptPatchView,
    positions: tuple[TokenPositionPair, ...],
    grid: RepresentationAlignmentGrid,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for token_index, position in enumerate(positions):
        layer_count = len(grid.cosine_similarity[token_index])
        for layer in range(layer_count):
            cells.append(
                {
                    "layer": layer,
                    "token_reverse_index": position.reverse_index,
                    "source_token_index": position.source_index,
                    "recipient_token_index": position.recipient_index,
                    "source_token_id": source_view.token_ids[position.source_index],
                    "recipient_token_id": recipient_view.token_ids[position.recipient_index],
                    "source_token": source_view.token_labels[position.source_index],
                    "recipient_token": recipient_view.token_labels[position.recipient_index],
                    "cosine_similarity": grid.cosine_similarity[token_index][layer],
                    "l2_distance": grid.l2_distance[token_index][layer],
                    "source_norm": grid.source_norm[token_index][layer],
                    "recipient_norm": grid.recipient_norm[token_index][layer],
                }
            )
    serialized: dict[str, object] = {
        "function_id": record.function_id,
        "source_function_id": (
            counterfactual.source_function_id
            if counterfactual is not None
            else record.function_id
        ),
        "recipient_function_id": record.function_id,
        "recipient_choice_function_ids": record.choice_function_ids,
        "recipient_correct_choice_index": record.choice_function_ids.index(record.function_id),
        "token_axis": {
            "order": "reverse_indexed",
            "anchor": "final token in the rendered generation prompt",
            "stop": (
                "last queried-function-name token"
                if counterfactual is not None and not counterfactual.stops_at_first_difference
                else "first differing token scanning backward from the sequence end"
                if counterfactual is not None
                else "sequence start"
            ),
            "positions": len(positions),
            "source_token_count": len(source_view.token_ids),
            "recipient_token_count": len(recipient_view.token_ids),
            "source_rendered_prompt": source_view.rendered_prompt,
            "recipient_rendered_prompt": recipient_view.rendered_prompt,
        },
        "cells": cells,
    }
    if counterfactual is not None:
        optional = {
            "source_correct_choice_index": counterfactual.source_correct_choice_index,
            "source_choice_function_ids": counterfactual.source_choice_function_ids,
            "source_choice_texts": counterfactual.source_choice_texts,
            "source_question_id": counterfactual.source_question_id,
            "source_question": counterfactual.source_question,
            "source_format": counterfactual.source_format,
            "source_label_relation": counterfactual.source_label_relation,
            "source_context_id": counterfactual.source_context_id,
            "source_context": counterfactual.source_context,
        }
        serialized.update({key: value for key, value in optional.items() if value is not None})
    return serialized


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentiles require finite values and a quantile in [0, 1]")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * quantile
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("alignment summary requires finite measured values")
    return {
        "count": len(values),
        "min": min(values),
        "mean": math.fsum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _write_alignment_artifacts(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    mode: PatchingMode,
    recipient_step: int,
    donor_step: int,
    records_by_interface: dict[PatchingInterface, list[dict[str, object]]],
    metric_values: dict[PatchingInterface, dict[str, list[float]]],
    comparison_direction: str,
) -> None:
    for interface, records in records_by_interface.items():
        plan = PatchingPlan(
            mode=mode,
            recipient_step=recipient_step,
            donor_steps=(donor_step,),
            interface=interface,
        )
        output = representation_alignment_path(
            root,
            run,
            interface,
            mode,
            recipient_step,
            donor_step,
        )
        write_json(
            output,
            {
                "schema_version": REPRESENTATION_ALIGNMENT_SCHEMA_VERSION,
                "model": spec,
                "run": run,
                "plan": plan,
                "donor_step": donor_step,
                "comparison_direction": comparison_direction,
                "checkpoint_relation": (
                    "same_checkpoint" if donor_step == recipient_step else "cross_checkpoint"
                ),
                "measurement": {
                    "kind": REPRESENTATION_ALIGNMENT_KIND,
                    "causal_intervention": False,
                    "metrics": REPRESENTATION_ALIGNMENT_METRICS,
                    "accumulation_dtype": REPRESENTATION_ALIGNMENT_ACCUMULATION_DTYPE,
                    "cosine_similarity": {
                        "definition": "dot(source, recipient) / (norm(source) * norm(recipient))",
                        "range": (-1.0, 1.0),
                        "higher_means": "more directional alignment",
                    },
                    "l2_distance": {
                        "definition": "norm(source - recipient, 2)",
                        "range": (0.0, None),
                        "lower_means": "closer in the selected boundary's raw activation units",
                    },
                    "summary": {
                        metric: _metric_summary(metric_values[interface][metric])
                        for metric in REPRESENTATION_ALIGNMENT_METRICS
                    },
                },
                "records": records,
            },
        )
        print(
            f"[representation-alignment] {run.model}/{run.condition.value} "
            f"{interface.value}/{mode.value} recipient={recipient_step} donor={donor_step} "
            f"-> {output}",
            flush=True,
        )


def _scheduled_alignment_pairs(
    recipient_steps: tuple[int, ...],
    donor_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
) -> list[tuple[int, int, PatchingMode]]:
    scheduled: list[tuple[int, int, PatchingMode]] = []
    for mode in modes:
        if mode.uses_prompt_counterfactual:
            if mode.supports_independent_checkpoint_donor:
                scheduled.extend(
                    (recipient_step, donor_step, mode)
                    for recipient_step in recipient_steps
                    for donor_step in donor_steps
                )
            else:
                scheduled.extend(
                    (recipient_step, recipient_step, mode)
                    for recipient_step in recipient_steps
                )
            continue
        for recipient_step in recipient_steps:
            for donor_step in donor_steps:
                if donor_step == recipient_step:
                    continue
                if _temporal_mode(recipient_step, donor_step) is mode:
                    scheduled.append((recipient_step, donor_step, mode))
    return scheduled


def run_representation_alignment_matrix(
    root: Path,
    run: RunKey,
    recipient_steps: tuple[int, ...],
    donor_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
    interfaces: tuple[PatchingInterface, ...] = REPRESENTATION_ALIGNMENT_INTERFACES,
    *,
    shuffle_seed: int | None = None,
    allow_provisional_model: bool = False,
) -> None:
    """Measure exact unpatched donor/recipient vector alignment for selected coordinates."""

    if not t.cuda.is_available():
        raise RuntimeError("representation alignment requires CUDA")
    for name, values in (("recipient", recipient_steps), ("donor", donor_steps)):
        if tuple(sorted(set(values))) != values or any(
            step not in CHECKPOINT_STEPS for step in values
        ):
            raise ValueError(f"alignment {name} steps must be unique registered checkpoints")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("alignment modes must be non-empty and unique")
    if (
        not interfaces
        or len(set(interfaces)) != len(interfaces)
        or any(interface not in REPRESENTATION_ALIGNMENT_INTERFACES for interface in interfaces)
    ):
        raise ValueError("alignment interfaces must be unique activation boundaries")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ValueError("alignment shuffle seed must be non-negative")

    scheduled = _scheduled_alignment_pairs(recipient_steps, donor_steps, modes)
    if shuffle_seed is not None:
        scheduled = _seeded_priority_temporal_order(scheduled, shuffle_seed)
    pending: list[tuple[int, int, PatchingMode, tuple[PatchingInterface, ...]]] = []
    skipped = 0
    for recipient_step, donor_step, mode in scheduled:
        missing_interfaces = tuple(
            interface
            for interface in interfaces
            if not representation_alignment_path(
                root,
                run,
                interface,
                mode,
                recipient_step,
                donor_step,
            ).is_file()
        )
        skipped += len(interfaces) - len(missing_interfaces)
        if missing_interfaces:
            pending.append((recipient_step, donor_step, mode, missing_interfaces))
    if skipped:
        print(
            f"[representation-alignment] {run.model}/{run.condition.value} skipped "
            f"{skipped} existing interface artifact(s)",
            flush=True,
        )
    if not pending:
        return
    if shuffle_seed is not None:
        tier_counts = [0] * (len(TEMPORAL_PRIORITY_LABELS) + 1)
        for recipient_step, donor_step, mode, _missing in pending:
            tier_counts[_temporal_priority_tier((recipient_step, donor_step, mode))] += 1
        labels = (*TEMPORAL_PRIORITY_LABELS, "remainder")
        summary = ", ".join(
            f"{label}: {count}" for label, count in zip(labels, tier_counts, strict=True)
        )
        print(
            f"[representation-alignment] priority-shuffled {len(pending)} missing "
            f"checkpoint/source pairs with seed {shuffle_seed} ({summary})",
            flush=True,
        )

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    records = _selected_records(run.seed)
    for recipient_step, donor_step, mode, missing_interfaces in pending:
        source_by_record: dict[str, AlignmentSourceRecord] = {}
        donor_model = _load_checkpoint_model(root, run, spec, donor_step)
        donor_blocks = resolve_decoder_blocks(donor_model, spec)
        try:
            for record in records:
                counterfactual, source_view, recipient_view, positions = _comparison_views(
                    processor,
                    record,
                    mode,
                )
                source_by_record[record.record_id] = AlignmentSourceRecord(
                    counterfactual=counterfactual,
                    source_view=source_view,
                    recipient_view=recipient_view,
                    positions=positions,
                    activations=_capture_alignment_interfaces(
                        donor_model,
                        donor_blocks,
                        missing_interfaces,
                        source_view.input_ids,
                        source_view.attention_mask,
                        token_indices=tuple(
                            position.source_index for position in positions
                        ),
                    ),
                )
        finally:
            _release_model(donor_model)

        records_by_interface: dict[PatchingInterface, list[dict[str, object]]] = {
            interface: [] for interface in missing_interfaces
        }
        metric_values: dict[PatchingInterface, dict[str, list[float]]] = {
            interface: {metric: [] for metric in REPRESENTATION_ALIGNMENT_METRICS}
            for interface in missing_interfaces
        }
        recipient_model = _load_checkpoint_model(root, run, spec, recipient_step)
        recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
        comparison_direction: str | None = None
        try:
            for record in records:
                source = source_by_record.pop(record.record_id)
                recipient_activations = _capture_alignment_interfaces(
                    recipient_model,
                    recipient_blocks,
                    missing_interfaces,
                    source.recipient_view.input_ids,
                    source.recipient_view.attention_mask,
                    token_indices=tuple(
                        position.recipient_index for position in source.positions
                    ),
                )
                dense_positions = tuple(
                    TokenPositionPair(index, index, position.reverse_index)
                    for index, position in enumerate(source.positions)
                )
                for interface in missing_interfaces:
                    grid = _representation_alignment_grid(
                        source.activations[interface],
                        recipient_activations[interface],
                        dense_positions,
                    )
                    records_by_interface[interface].append(
                        _serialize_alignment_record(
                            record,
                            mode,
                            source.counterfactual,
                            source.source_view,
                            source.recipient_view,
                            source.positions,
                            grid,
                        )
                    )
                    metric_values[interface]["cosine_similarity"].extend(
                        value for row in grid.cosine_similarity for value in row
                    )
                    metric_values[interface]["l2_distance"].extend(
                        value for row in grid.l2_distance for value in row
                    )
                direction = (
                    source.counterfactual.patch_direction
                    if source.counterfactual is not None
                    else _temporal_direction(mode)
                )
                if comparison_direction is None:
                    comparison_direction = direction
                elif comparison_direction != direction:
                    raise AssertionError("alignment comparison direction changed across records")
                del source, recipient_activations
                gc.collect()
        finally:
            _release_model(recipient_model)
        if source_by_record:
            raise AssertionError("not every source alignment record was consumed")
        if comparison_direction is None:
            raise RuntimeError("representation alignment selected no records")
        _write_alignment_artifacts(
            root,
            run,
            spec,
            mode,
            recipient_step,
            donor_step,
            records_by_interface,
            metric_values,
            comparison_direction,
        )
        del records_by_interface, metric_values
        gc.collect()


__all__ = [
    "RepresentationAlignmentGrid",
    "_capture_alignment_interfaces",
    "_representation_alignment_grid",
    "run_representation_alignment_matrix",
]
