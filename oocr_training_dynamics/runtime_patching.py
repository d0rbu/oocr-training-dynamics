"""Memory-safe residual-stream patching across samples and checkpoint time."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import adapter_dir, run_dir, write_json
from oocr_training_dynamics.contracts import PatchingMode, RunKey, checkpoint_label
from oocr_training_dynamics.data import (
    DERANGEMENT,
    FUNCTION_BY_ID,
    ChatMessage,
    ReflectionRecord,
    build_reflection_records,
)
from oocr_training_dynamics.models import ModelSpec, get_model_spec
from oocr_training_dynamics.patching import (
    PatchingPlan,
    TokenPositionPair,
    build_across_sample_pair,
    reverse_token_position_pairs,
    token_index_covering_character,
)
from oocr_training_dynamics.runtime_models import (
    attach_inference_lora,
    load_base_model,
    load_processor,
    resolve_decoder_blocks,
    tokenizer_for,
)
from oocr_training_dynamics.tokenization import (
    TokenizedExample,
    first_target_position,
    tokenize_messages,
)


@dataclass(frozen=True)
class PromptPatchView:
    input_ids: t.Tensor
    attention_mask: t.Tensor
    anchor_index: int
    stop_index: int
    rendered_prompt: str
    token_ids: tuple[int, ...]
    token_labels: tuple[str, ...]


def _candidate_ids(processor: Any, record: ReflectionRecord) -> t.Tensor:
    values: list[int] = []
    for letter in "ABCDE":
        messages = (*record.messages[:-1], ChatMessage("assistant", letter))
        example = tokenize_messages(processor, record.record_id, messages)
        values.append(int(example.input_ids[0, first_target_position(example)].item()))
    if len(set(values)) != 5:
        raise RuntimeError("A-E must have distinct first target tokens")
    return t.tensor(values, dtype=t.int64, device="cuda")


def _prefix(
    example: TokenizedExample,
    *,
    device: str = "cuda",
) -> tuple[t.Tensor, t.Tensor]:
    start = first_target_position(example)
    return (
        example.input_ids[:, :start].to(device),
        example.attention_mask[:, :start].to(device),
    )


def _render_generation_prompt(
    processor: Any,
    messages: tuple[ChatMessage, ...],
) -> str:
    conversation = [{"role": message.role, "content": message.content} for message in messages[:-1]]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        rendered = processor.apply_chat_template(
            conversation,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        rendered = processor.apply_chat_template(conversation, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("rendered chat template must be a string")
    return rendered


def _token_label(tokenizer: Any, token_id: int) -> str:
    value = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(value, str):
        raise TypeError("token decoder must return text")
    visible = value.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␠")
    if visible:
        return visible
    fallback = tokenizer.convert_ids_to_tokens(token_id)
    return str(fallback)


def _prompt_patch_view(
    processor: Any,
    record: ReflectionRecord,
    messages: tuple[ChatMessage, ...],
    function_alias: str,
    *,
    stop_at_sequence_start: bool,
    device: str = "cuda",
) -> PromptPatchView:
    example = tokenize_messages(processor, record.record_id + ":patch", messages)
    input_ids, attention_mask = _prefix(example, device=device)
    rendered = _render_generation_prompt(processor, messages)
    tokenizer = tokenizer_for(processor)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets_raw = encoded["offset_mapping"]
    if not isinstance(token_ids, list) or not all(isinstance(value, int) for value in token_ids):
        raise TypeError("rendered prompt token IDs must be one integer list")
    if not isinstance(offsets_raw, list) or not all(
        isinstance(value, tuple | list) and len(value) == 2 for value in offsets_raw
    ):
        raise TypeError("fast tokenizer must return one offset pair per prompt token")
    if token_ids != input_ids[0].tolist():
        raise RuntimeError("rendered prompt offsets do not match chat-template token IDs")
    offsets = tuple((int(value[0]), int(value[1])) for value in offsets_raw)
    choice = "ABCDE"[record.choice_function_ids.index(record.function_id)]
    definition = FUNCTION_BY_ID[record.function_id].python_definition
    option_text = f"{choice}) {definition}"
    option_start = rendered.find(option_text)
    if option_start < 0:
        raise RuntimeError("rendered prompt lacks the selected correct implementation")
    colon_in_option = option_text.find(":")
    if colon_in_option < 0:
        raise RuntimeError("selected implementation lacks the lambda prefix boundary")
    anchor_index = token_index_covering_character(
        offsets,
        option_start + colon_in_option,
    )
    if stop_at_sequence_start:
        stop_index = 0
    else:
        alias_start = rendered.rfind(function_alias, 0, option_start)
        if alias_start < 0:
            raise RuntimeError("rendered prompt lacks the queried function alias")
        stop_index = token_index_covering_character(
            offsets,
            alias_start + len(function_alias) - 1,
        )
    if anchor_index < stop_index:
        raise RuntimeError("lambda anchor unexpectedly precedes the function-name boundary")
    return PromptPatchView(
        input_ids=input_ids,
        attention_mask=attention_mask,
        anchor_index=anchor_index,
        stop_index=stop_index,
        rendered_prompt=rendered,
        token_ids=tuple(token_ids),
        token_labels=tuple(_token_label(tokenizer, token_id) for token_id in token_ids),
    )


def _hidden_tensor(output: Any) -> t.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, t.Tensor) or hidden.ndim != 3:
        raise RuntimeError("decoder block output must begin with [batch, sequence, hidden]")
    return hidden


def _replace_hidden_positions(
    output: Any,
    replacements: t.Tensor,
    positions: tuple[int, ...],
) -> Any:
    hidden = _hidden_tensor(output).clone()
    if replacements.shape != (hidden.shape[0], hidden.shape[2]):
        raise ValueError("patch activations must contain one hidden vector per batch row")
    if len(positions) != hidden.shape[0] or any(
        position < 0 or position >= hidden.shape[1] for position in positions
    ):
        raise ValueError("recipient patch positions must lie within every batch sequence")
    rows = t.arange(hidden.shape[0], device=hidden.device)
    columns = t.tensor(positions, dtype=t.int64, device=hidden.device)
    hidden[rows, columns, :] = replacements.to(device=hidden.device, dtype=hidden.dtype)
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
    logits = output.logits[:, -1, candidate_ids].to(dtype=t.float32)
    return t.softmax(logits, dim=-1).detach().cpu()


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

        def hook(
            _module: t.nn.Module, _inputs: tuple[Any, ...], output: Any, *, index: int = layer
        ) -> None:
            hidden = _hidden_tensor(output)
            if hidden.shape[0] != 1:
                raise RuntimeError("activation capture requires one unbatched prompt")
            captured[index] = hidden[0].detach().cpu().clone()

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
    return tuple(cast(t.Tensor, value) for value in captured), probabilities[0]


def _patch_grid(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    source_activations: tuple[t.Tensor, ...],
    positions: tuple[TokenPositionPair, ...],
    correct_choice_index: int,
    *,
    patch_batch_size: int = 8,
) -> tuple[tuple[float, ...], ...]:
    if len(source_activations) != len(blocks):
        raise ValueError("source activation count must equal decoder layer count")
    if not positions or patch_batch_size <= 0:
        raise ValueError("token patching requires positions and a positive batch size")
    if not 0 <= correct_choice_index < 5:
        raise ValueError("correct choice index must be in the five-way candidate set")
    values = [[float("nan")] * len(blocks) for _ in positions]
    for layer, (block, source) in enumerate(zip(blocks, source_activations, strict=True)):
        for start in range(0, len(positions), patch_batch_size):
            chunk = positions[start : start + patch_batch_size]
            replacements = t.stack(
                [source[position.source_index] for position in chunk],
                dim=0,
            )
            recipient_positions = tuple(position.recipient_index for position in chunk)
            handle = block.register_forward_hook(
                lambda _module, _inputs, output, replacement=replacements, patch_positions=recipient_positions: (
                    _replace_hidden_positions(
                        output,
                        replacement,
                        patch_positions,
                    )
                )
            )
            try:
                probabilities = _forward_probabilities(
                    model,
                    input_ids.expand(len(chunk), -1),
                    attention_mask.expand(len(chunk), -1),
                    candidate_ids,
                )
            finally:
                handle.remove()
            for offset, probability in enumerate(probabilities[:, correct_choice_index].tolist()):
                values[start + offset][layer] = float(probability)
    if any(not math.isfinite(value) for row in values for value in row):
        raise RuntimeError("token patch grid contains an unfilled or non-finite cell")
    return tuple(tuple(row) for row in values)


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


def build_token_axis_metadata(
    processor: Any,
    record: ReflectionRecord,
    mode: PatchingMode,
) -> dict[str, object]:
    """Build the exact CPU-tokenized source/recipient axis shown by the site."""

    if mode is PatchingMode.ACROSS_SAMPLE:
        pair = build_across_sample_pair(record)
        source_function_id = pair.dirty_function_id
        source_messages = pair.dirty_messages
        source_alias = FUNCTION_BY_ID[pair.dirty_function_id].alias
        source_view = _prompt_patch_view(
            processor,
            record,
            source_messages,
            source_alias,
            stop_at_sequence_start=False,
            device="cpu",
        )
        recipient_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=False,
            device="cpu",
        )
    else:
        source_function_id = record.function_id
        source_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
            device="cpu",
        )
        recipient_view = source_view
    positions = reverse_token_position_pairs(
        source_view.anchor_index,
        recipient_view.anchor_index,
        source_view.stop_index,
        recipient_view.stop_index,
    )
    return {
        "source_function_id": source_function_id,
        "recipient_function_id": record.function_id,
        "source_rendered_prompt": source_view.rendered_prompt,
        "recipient_rendered_prompt": recipient_view.rendered_prompt,
        "positions": tuple(
            {
                "reverse_index": position.reverse_index,
                "source_index": position.source_index,
                "recipient_index": position.recipient_index,
                "source_token_id": source_view.token_ids[position.source_index],
                "recipient_token_id": recipient_view.token_ids[position.recipient_index],
                "source_token": source_view.token_labels[position.source_index],
                "recipient_token": recipient_view.token_labels[position.recipient_index],
            }
            for position in positions
        ),
    }


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
    positions: tuple[TokenPositionPair, ...],
    source_view: PromptPatchView,
    recipient_view: PromptPatchView,
    grid: tuple[tuple[float, ...], ...],
    mode: PatchingMode,
) -> dict[str, object]:
    correct_choice = record.choice_function_ids.index(record.function_id)
    recipient_target = float(recipient[correct_choice].item())
    cells: list[dict[str, object]] = []
    for position, row in zip(positions, grid, strict=True):
        for layer, probability in enumerate(row):
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
                    "probability": probability,
                    "delta_from_recipient": probability - recipient_target,
                }
            )
    return {
        "function_id": record.function_id,
        "source_function_id": (
            DERANGEMENT[record.function_id]
            if mode is PatchingMode.ACROSS_SAMPLE
            else record.function_id
        ),
        "recipient_function_id": record.function_id,
        "choice_function_ids": record.choice_function_ids,
        "correct_choice_index": correct_choice,
        "source_probabilities": source.tolist(),
        "recipient_probabilities": recipient.tolist(),
        "site_probability": "correct",
        "token_axis": {
            "order": "reverse_indexed",
            "anchor": "last token covering ':' in the selected correct lambda prefix",
            "stop": (
                "last queried-function-name token"
                if mode is PatchingMode.ACROSS_SAMPLE
                else "sequence start"
            ),
            "positions": len(positions),
            "source_rendered_prompt": source_view.rendered_prompt,
            "recipient_rendered_prompt": recipient_view.rendered_prompt,
        },
        "cells": cells,
    }


def _patch_record(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    processor: Any,
    record: ReflectionRecord,
    mode: PatchingMode,
    source_view: PromptPatchView,
    recipient_view: PromptPatchView,
    source_activations: tuple[t.Tensor, ...],
    source_probabilities: t.Tensor,
) -> dict[str, object]:
    positions = reverse_token_position_pairs(
        source_view.anchor_index,
        recipient_view.anchor_index,
        source_view.stop_index,
        recipient_view.stop_index,
    )
    candidate_ids = _candidate_ids(processor, record)
    recipient_probabilities = _forward_probabilities(
        model,
        recipient_view.input_ids,
        recipient_view.attention_mask,
        candidate_ids,
    )[0]
    correct_choice = record.choice_function_ids.index(record.function_id)
    grid = _patch_grid(
        model,
        blocks,
        recipient_view.input_ids,
        recipient_view.attention_mask,
        candidate_ids,
        source_activations,
        positions,
        correct_choice,
    )
    return _serialize_grid(
        record,
        source_probabilities,
        recipient_probabilities,
        positions,
        source_view,
        recipient_view,
        grid,
        mode,
    )


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
    if plan.mode is PatchingMode.ACROSS_SAMPLE:
        donor_step = pending[0]
        recipient_model = _load_checkpoint_model(root, run, spec, plan.recipient_step)
        recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
        serialized: list[dict[str, object]] = []
        try:
            for record in records:
                pair = build_across_sample_pair(record)
                source_view = _prompt_patch_view(
                    processor,
                    record,
                    pair.dirty_messages,
                    FUNCTION_BY_ID[pair.dirty_function_id].alias,
                    stop_at_sequence_start=False,
                )
                recipient_view = _prompt_patch_view(
                    processor,
                    record,
                    record.messages,
                    FUNCTION_BY_ID[record.function_id].alias,
                    stop_at_sequence_start=False,
                )
                source_activations, source_probabilities = _capture(
                    recipient_model,
                    recipient_blocks,
                    source_view.input_ids,
                    source_view.attention_mask,
                    _candidate_ids(processor, record),
                )
                serialized.append(
                    _patch_record(
                        recipient_model,
                        recipient_blocks,
                        processor,
                        record,
                        plan.mode,
                        source_view,
                        recipient_view,
                        source_activations,
                        source_probabilities,
                    )
                )
        finally:
            _release_model(recipient_model)
        output = _patch_output_path(root, run, plan, donor_step)
        write_json(
            output,
            {
                "model": spec,
                "run": run,
                "plan": plan,
                "donor_step": donor_step,
                "patch_direction": "dirty_source_into_clean_recipient",
                "records": serialized,
            },
        )
        print(
            f"[patch] {run.model}/{run.condition.value} {plan.mode.value} "
            f"recipient={plan.recipient_step} donor={donor_step} -> {output}",
            flush=True,
        )
        return

    for donor_step in pending:
        source_by_record: dict[
            str,
            tuple[PromptPatchView, tuple[t.Tensor, ...], t.Tensor],
        ] = {}
        donor_model = _load_checkpoint_model(root, run, spec, donor_step)
        donor_blocks = resolve_decoder_blocks(donor_model, spec)
        try:
            for record in records:
                source_view = _prompt_patch_view(
                    processor,
                    record,
                    record.messages,
                    FUNCTION_BY_ID[record.function_id].alias,
                    stop_at_sequence_start=True,
                )
                source_activations, source_probabilities = _capture(
                    donor_model,
                    donor_blocks,
                    source_view.input_ids,
                    source_view.attention_mask,
                    _candidate_ids(processor, record),
                )
                source_by_record[record.record_id] = (
                    source_view,
                    source_activations,
                    source_probabilities,
                )
        finally:
            _release_model(donor_model)

        recipient_model = _load_checkpoint_model(root, run, spec, plan.recipient_step)
        recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
        serialized = []
        try:
            for record in records:
                source_view, source_activations, source_probabilities = source_by_record[
                    record.record_id
                ]
                recipient_view = _prompt_patch_view(
                    processor,
                    record,
                    record.messages,
                    FUNCTION_BY_ID[record.function_id].alias,
                    stop_at_sequence_start=True,
                )
                serialized.append(
                    _patch_record(
                        recipient_model,
                        recipient_blocks,
                        processor,
                        record,
                        plan.mode,
                        source_view,
                        recipient_view,
                        source_activations,
                        source_probabilities,
                    )
                )
        finally:
            _release_model(recipient_model)
        output = _patch_output_path(root, run, plan, donor_step)
        write_json(
            output,
            {
                "model": spec,
                "run": run,
                "plan": plan,
                "donor_step": donor_step,
                "patch_direction": "earlier_source_into_later_clean_recipient",
                "records": serialized,
            },
        )
        print(
            f"[patch] {run.model}/{run.condition.value} {plan.mode.value} "
            f"recipient={plan.recipient_step} donor={donor_step} -> {output}",
            flush=True,
        )
        del source_by_record
        gc.collect()


__all__ = ["build_token_axis_metadata", "run_patching"]
