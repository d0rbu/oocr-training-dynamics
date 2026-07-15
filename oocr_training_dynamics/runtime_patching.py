"""Memory-safe residual-stream patching across samples and checkpoint time."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import adapter_dir, run_dir, write_json
from oocr_training_dynamics.contracts import PatchingMode, RunKey, checkpoint_label
from oocr_training_dynamics.data import ChatMessage, ReflectionRecord, build_reflection_records
from oocr_training_dynamics.metrics import normalized_patch_effect
from oocr_training_dynamics.models import ModelSpec, get_model_spec
from oocr_training_dynamics.patching import PatchingPlan, build_across_sample_pair
from oocr_training_dynamics.runtime_models import (
    attach_inference_lora,
    load_base_model,
    load_processor,
    resolve_decoder_blocks,
)
from oocr_training_dynamics.tokenization import (
    TokenizedExample,
    first_target_position,
    tokenize_messages,
)


def _candidate_ids(processor: Any, record: ReflectionRecord) -> t.Tensor:
    values: list[int] = []
    for letter in "ABCDE":
        messages = (*record.messages[:-1], ChatMessage("assistant", letter))
        example = tokenize_messages(processor, record.record_id, messages)
        values.append(int(example.input_ids[0, first_target_position(example)].item()))
    if len(set(values)) != 5:
        raise RuntimeError("A-E must have distinct first target tokens")
    return t.tensor(values, dtype=t.int64, device="cuda")


def _prefix(example: TokenizedExample) -> tuple[t.Tensor, t.Tensor]:
    start = first_target_position(example)
    return (
        example.input_ids[:, :start].to("cuda"),
        example.attention_mask[:, :start].to("cuda"),
    )


def _hidden_tensor(output: Any) -> t.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, t.Tensor) or hidden.ndim != 3:
        raise RuntimeError("decoder block output must begin with [batch, sequence, hidden]")
    return hidden


def _replace_hidden(output: Any, replacement: t.Tensor) -> Any:
    hidden = _hidden_tensor(output).clone()
    if replacement.shape != hidden[:, -1, :].shape:
        raise ValueError("patch activation shape does not match recipient query state")
    hidden[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _forward_probabilities(
    model: t.nn.Module,
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
) -> t.Tensor:
    with t.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    logits = output.logits[0, -1, candidate_ids].to(dtype=t.float32)
    return t.softmax(logits, dim=0).detach().cpu()


def _capture(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
) -> tuple[tuple[t.Tensor, ...], t.Tensor]:
    captured: list[t.Tensor | None] = [None] * len(blocks)
    handles: list[Any] = []
    for layer, block in enumerate(blocks):
        def hook(_module: t.nn.Module, _inputs: tuple[Any, ...], output: Any, *, index: int = layer) -> None:
            captured[index] = _hidden_tensor(output)[:, -1, :].detach().cpu().clone()

        handles.append(block.register_forward_hook(hook))
    try:
        probabilities = _forward_probabilities(
            model,
            input_ids,
            attention_mask,
            candidate_ids,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("not every decoder block produced a captured residual")
    return tuple(cast(t.Tensor, value) for value in captured), probabilities


def _patch_grid(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    source_activations: tuple[t.Tensor, ...],
) -> tuple[tuple[float, ...], ...]:
    if len(source_activations) != len(blocks):
        raise ValueError("source activation count must equal decoder layer count")
    rows: list[tuple[float, ...]] = []
    for block, source in zip(blocks, source_activations, strict=True):
        handle = block.register_forward_hook(
            lambda _module, _inputs, output, replacement=source: _replace_hidden(
                output,
                replacement,
            )
        )
        try:
            probabilities = _forward_probabilities(
                model,
                input_ids,
                attention_mask,
                candidate_ids,
            )
        finally:
            handle.remove()
        rows.append(tuple(float(value) for value in probabilities.tolist()))
    return tuple(rows)


def _load_checkpoint_model(root: Path, run: RunKey, spec: ModelSpec, step: int) -> t.nn.Module:
    base = load_base_model(spec, training=False)
    if step == 0:
        base.eval()
        return base
    path = adapter_dir(root, run, step)
    if not path.is_dir():
        raise FileNotFoundError(f"missing adapter checkpoint: {path}")
    return attach_inference_lora(base, path)


def _release_model(model: t.nn.Module) -> None:
    model.to("cpu")
    del model
    gc.collect()
    t.cuda.empty_cache()


def _selected_records(seed: int) -> tuple[ReflectionRecord, ...]:
    records = build_reflection_records(seed + 1, variants_per_kind=1)
    return tuple(record for record in records if record.kind == "code")


def _patch_output_path(root: Path, run: RunKey, plan: PatchingPlan, donor_step: int) -> Path:
    return (
        run_dir(root, run)
        / "patching"
        / plan.mode.value
        / f"recipient_{checkpoint_label(plan.recipient_step)}"
        / f"donor_{checkpoint_label(donor_step)}.json"
    )


def _serialize_grid(
    record: ReflectionRecord,
    source: t.Tensor,
    recipient: t.Tensor,
    grid: tuple[tuple[float, ...], ...],
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for layer, row in enumerate(grid):
        for choice, probability in enumerate(row):
            effect = normalized_patch_effect(
                probability,
                float(recipient[choice].item()),
                float(source[choice].item()),
            )
            cells.append(
                {
                    "layer": layer,
                    "choice_index": choice,
                    "probability": probability,
                    "delta_from_recipient": probability - float(recipient[choice].item()),
                    "normalized_effect": None if math.isnan(effect) else effect,
                }
            )
    return {
        "function_id": record.function_id,
        "choice_function_ids": record.choice_function_ids,
        "correct_choice_index": record.choice_function_ids.index(record.function_id),
        "source_probabilities": source.tolist(),
        "recipient_probabilities": recipient.tolist(),
        "cells": cells,
    }


def run_patching(
    root: Path,
    run: RunKey,
    plan: PatchingPlan,
    *,
    allow_provisional_model: bool = False,
) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("activation patching requires CUDA")
    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    records = _selected_records(run.seed)
    pending = tuple(
        donor_step
        for donor_step in plan.donor_steps
        if not _patch_output_path(root, run, plan, donor_step).is_file()
    )
    skipped = len(plan.donor_steps) - len(pending)
    if skipped:
        print(
            f"[patch] {run.model}/{run.condition.value} skipped {skipped} existing artifact(s)",
            flush=True,
        )
    if not pending:
        return
    sources_by_donor: dict[
        int,
        dict[str, tuple[tuple[t.Tensor, ...], t.Tensor]],
    ] = {}
    if plan.mode is PatchingMode.ACROSS_TIME:
        for donor_step in pending:
            source_by_record: dict[str, tuple[tuple[t.Tensor, ...], t.Tensor]] = {}
            donor_model = _load_checkpoint_model(root, run, spec, donor_step)
            donor_blocks = resolve_decoder_blocks(donor_model, spec)
            for record in records:
                example = tokenize_messages(processor, record.record_id, record.messages)
                input_ids, attention_mask = _prefix(example)
                source_by_record[record.record_id] = _capture(
                    donor_model,
                    donor_blocks,
                    input_ids,
                    attention_mask,
                    _candidate_ids(processor, record),
                )
            _release_model(donor_model)
            sources_by_donor[donor_step] = source_by_record
    recipient_model = _load_checkpoint_model(root, run, spec, plan.recipient_step)
    recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
    for donor_step in pending:
        serialized: list[dict[str, object]] = []
        for record in records:
            candidate_ids = _candidate_ids(processor, record)
            if plan.mode is PatchingMode.ACROSS_SAMPLE:
                pair = build_across_sample_pair(record)
                clean_example = tokenize_messages(processor, record.record_id, record.messages)
                clean_ids, clean_mask = _prefix(clean_example)
                source_activations, source_probabilities = _capture(
                    recipient_model,
                    recipient_blocks,
                    clean_ids,
                    clean_mask,
                    candidate_ids,
                )
                dirty_example = tokenize_messages(
                    processor,
                    record.record_id + ":dirty",
                    pair.dirty_messages,
                )
                recipient_ids, recipient_mask = _prefix(dirty_example)
            else:
                source_activations, source_probabilities = sources_by_donor[donor_step][
                    record.record_id
                ]
                recipient_example = tokenize_messages(processor, record.record_id, record.messages)
                recipient_ids, recipient_mask = _prefix(recipient_example)
            recipient_probabilities = _forward_probabilities(
                recipient_model,
                recipient_ids,
                recipient_mask,
                candidate_ids,
            )
            grid = _patch_grid(
                recipient_model,
                recipient_blocks,
                recipient_ids,
                recipient_mask,
                candidate_ids,
                source_activations,
            )
            serialized.append(
                _serialize_grid(
                    record,
                    source_probabilities,
                    recipient_probabilities,
                    grid,
                )
            )
        output = _patch_output_path(root, run, plan, donor_step)
        write_json(
            output,
            {
                "model": spec,
                "run": run,
                "plan": plan,
                "donor_step": donor_step,
                "records": serialized,
            },
        )
        print(
            f"[patch] {run.model}/{run.condition.value} {plan.mode.value} "
            f"recipient={plan.recipient_step} donor={donor_step} -> {output}",
            flush=True,
        )
    _release_model(recipient_model)


__all__ = ["run_patching"]
