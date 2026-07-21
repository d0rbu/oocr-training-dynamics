"""Memory-safe decoder-interface patching across samples and checkpoint time."""

from __future__ import annotations

import gc
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import adapter_dir, run_dir, write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
    RunKey,
    checkpoint_label,
    training_spec_for_run,
)
from oocr_training_dynamics.data import (
    DERANGEMENT,
    FUNCTION_BY_ID,
    ChatMessage,
    ReflectionRecord,
    build_reflection_records,
)
from oocr_training_dynamics.models import ModelSpec, get_model_spec
from oocr_training_dynamics.patching import (
    WEIGHT_PATCH_SCOPE,
    PatchingPlan,
    TokenPositionPair,
    build_across_sample_pair,
    reverse_token_position_pairs,
    token_index_covering_character,
)
from oocr_training_dynamics.runtime_models import (
    LORA_TARGET_MODULES,
    attach_inference_lora,
    attach_trainable_lora,
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


@dataclass(frozen=True)
class PatchTarget:
    """One decoder layer's concrete module boundary and hook direction."""

    module: t.nn.Module
    capture_input: bool


SourceRecord = tuple[PromptPatchView, tuple[t.Tensor, ...], t.Tensor]
SourceBank = dict[str, SourceRecord]
LoraLayerState = dict[str, t.Tensor]
WeightSourceRecord = tuple[PromptPatchView, t.Tensor]
WeightSourceBank = dict[str, WeightSourceRecord]


@dataclass(frozen=True)
class WeightSourceBundle:
    """CPU-resident donor LoRA parameters and clean-prompt baselines."""

    layer_states: tuple[LoraLayerState, ...]
    records: WeightSourceBank


@dataclass(frozen=True)
class TokenLoraProjection:
    """One recipient LoRA projection plus the donor factors used at selected tokens."""

    name: str
    module: t.nn.Module
    adapter: str
    donor_a: t.Tensor
    donor_b: t.Tensor
    scaling: float


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
    if not token_ids:
        raise RuntimeError("rendered generation prompt must contain at least one token")
    anchor_index = len(token_ids) - 1
    if stop_at_sequence_start:
        stop_index = 0
    else:
        alias_start = rendered.rfind(function_alias)
        if alias_start < 0:
            raise RuntimeError("rendered prompt lacks the queried function alias")
        stop_index = token_index_covering_character(
            offsets,
            alias_start + len(function_alias) - 1,
        )
    if anchor_index < stop_index:
        raise RuntimeError("sequence-end anchor unexpectedly precedes the function-name boundary")
    return PromptPatchView(
        input_ids=input_ids,
        attention_mask=attention_mask,
        anchor_index=anchor_index,
        stop_index=stop_index,
        rendered_prompt=rendered,
        token_ids=tuple(token_ids),
        token_labels=tuple(_token_label(tokenizer, token_id) for token_id in token_ids),
    )


def _resolve_patch_targets(
    blocks: tuple[t.nn.Module, ...],
    interface: PatchingInterface,
) -> tuple[PatchTarget, ...]:
    if interface.patches_weights:
        raise ValueError("decoder-block weights are parameters, not an activation hook target")
    if interface is PatchingInterface.RESID_POST:
        return tuple(PatchTarget(block, capture_input=False) for block in blocks)
    attribute = (
        "self_attn"
        if interface in {PatchingInterface.ATTENTION_INPUT, PatchingInterface.ATTENTION_OUTPUT}
        else "mlp"
    )
    capture_input = interface in {
        PatchingInterface.ATTENTION_INPUT,
        PatchingInterface.MLP_INPUT,
    }
    targets: list[PatchTarget] = []
    for layer, block in enumerate(blocks):
        module = getattr(block, attribute, None)
        if not isinstance(module, t.nn.Module):
            raise RuntimeError(
                f"decoder layer {layer} lacks the {attribute} module required by {interface.value}"
            )
        targets.append(PatchTarget(module, capture_input=capture_input))
    return tuple(targets)


def _hidden_tensor(output: Any) -> t.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, t.Tensor) or hidden.ndim != 3:
        raise RuntimeError("patched module output must begin with [batch, sequence, hidden]")
    return hidden


def _input_hidden(args: tuple[Any, ...], kwargs: dict[str, Any]) -> t.Tensor:
    candidate = args[0] if args and isinstance(args[0], t.Tensor) else kwargs.get("hidden_states")
    if not isinstance(candidate, t.Tensor) or candidate.ndim != 3:
        raise RuntimeError(
            "patched module input must provide hidden_states with [batch, sequence, hidden]"
        )
    return candidate


def _replace_tensor_positions(
    hidden: t.Tensor,
    replacements: t.Tensor,
    positions: tuple[int, ...],
) -> t.Tensor:
    hidden = hidden.clone()
    if replacements.shape != (hidden.shape[0], hidden.shape[2]):
        raise ValueError("patch activations must contain one hidden vector per batch row")
    if len(positions) != hidden.shape[0] or any(
        position < 0 or position >= hidden.shape[1] for position in positions
    ):
        raise ValueError("recipient patch positions must lie within every batch sequence")
    rows = t.arange(hidden.shape[0], device=hidden.device)
    columns = t.tensor(positions, dtype=t.int64, device=hidden.device)
    hidden[rows, columns, :] = replacements.to(device=hidden.device, dtype=hidden.dtype)
    return hidden


def _replace_hidden_positions(
    output: Any,
    replacements: t.Tensor,
    positions: tuple[int, ...],
) -> Any:
    hidden = _replace_tensor_positions(_hidden_tensor(output), replacements, positions)
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _replace_hidden_input(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    replacements: t.Tensor,
    positions: tuple[int, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    hidden = _replace_tensor_positions(
        _input_hidden(args, kwargs),
        replacements,
        positions,
    )
    if args and isinstance(args[0], t.Tensor):
        return (hidden, *args[1:]), kwargs
    updated = dict(kwargs)
    updated["hidden_states"] = hidden
    return args, updated


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
    targets: tuple[PatchTarget, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
) -> tuple[tuple[t.Tensor, ...], t.Tensor]:
    captured: list[t.Tensor | None] = [None] * len(targets)
    handles: list[Any] = []
    for layer, target in enumerate(targets):
        if target.capture_input:

            def input_hook(
                _module: t.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer,
            ) -> None:
                hidden = _input_hidden(args, kwargs)
                if hidden.shape[0] != 1:
                    raise RuntimeError("activation capture requires one unbatched prompt")
                captured[index] = hidden[0].detach().cpu().clone()

            handles.append(target.module.register_forward_pre_hook(input_hook, with_kwargs=True))
        else:

            def output_hook(
                _module: t.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer,
            ) -> None:
                hidden = _hidden_tensor(output)
                if hidden.shape[0] != 1:
                    raise RuntimeError("activation capture requires one unbatched prompt")
                captured[index] = hidden[0].detach().cpu().clone()

            handles.append(target.module.register_forward_hook(output_hook))
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
        raise RuntimeError("not every decoder layer produced a captured patch activation")
    return tuple(cast(t.Tensor, value) for value in captured), probabilities[0]


def _patch_grid(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    source_activations: tuple[t.Tensor, ...],
    positions: tuple[TokenPositionPair, ...],
    correct_choice_index: int,
    *,
    patch_batch_size: int = 8,
) -> tuple[tuple[float, ...], ...]:
    if len(source_activations) != len(targets):
        raise ValueError("source activation count must equal decoder layer count")
    if not positions or patch_batch_size <= 0:
        raise ValueError("token patching requires positions and a positive batch size")
    if not 0 <= correct_choice_index < 5:
        raise ValueError("correct choice index must be in the five-way candidate set")
    values = [[float("nan")] * len(targets) for _ in positions]
    for layer, (target, source) in enumerate(zip(targets, source_activations, strict=True)):
        for start in range(0, len(positions), patch_batch_size):
            chunk = positions[start : start + patch_batch_size]
            replacements = t.stack(
                [source[position.source_index] for position in chunk],
                dim=0,
            )
            recipient_positions = tuple(position.recipient_index for position in chunk)
            if target.capture_input:
                handle = target.module.register_forward_pre_hook(
                    lambda _module, args, kwargs, replacement=replacements, patch_positions=recipient_positions: (
                        _replace_hidden_input(
                            args,
                            kwargs,
                            replacement,
                            patch_positions,
                        )
                    ),
                    with_kwargs=True,
                )
            else:
                handle = target.module.register_forward_hook(
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


def _lora_parameters(block: t.nn.Module) -> dict[str, t.nn.Parameter]:
    """Return exactly the learned LoRA factors belonging to one decoder block."""

    parameters = {
        name: parameter
        for name, parameter in block.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    if not parameters:
        raise RuntimeError("weight patching requires LoRA A/B parameters in every decoder block")
    a_count = sum("lora_A" in name for name in parameters)
    b_count = sum("lora_B" in name for name in parameters)
    if a_count != b_count:
        raise RuntimeError("decoder-block LoRA A/B parameter counts do not match")
    return parameters


def _capture_lora_layer_state(block: t.nn.Module) -> LoraLayerState:
    """Clone one block's learned adapter factors to CPU."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _lora_parameters(block).items()
    }


def _copy_lora_layer_state(block: t.nn.Module, state: LoraLayerState) -> None:
    """Replace one block's adapter factors after exact key/shape validation."""

    parameters = _lora_parameters(block)
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise RuntimeError(
            "donor and recipient block adapter schemas differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with t.no_grad():
        for name, parameter in parameters.items():
            source = state[name]
            if source.shape != parameter.shape:
                raise RuntimeError(
                    f"LoRA parameter shape mismatch for {name}: "
                    f"donor={tuple(source.shape)} recipient={tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def _active_lora_adapter(module: t.nn.Module, name: str) -> str:
    """Resolve the one ordinary, unmerged adapter used by a LoRA projection."""

    raw_adapters = getattr(module, "active_adapters", None)
    if isinstance(raw_adapters, str):
        adapters = (raw_adapters,)
    elif isinstance(raw_adapters, list | tuple) and all(
        isinstance(adapter, str) for adapter in raw_adapters
    ):
        adapters = tuple(raw_adapters)
    else:
        raise RuntimeError(f"LoRA projection {name} does not expose active adapters")
    if len(adapters) != 1:
        raise RuntimeError(
            f"token-local weight patching requires one active adapter in {name}; found {adapters}"
        )
    adapter = adapters[0]
    if bool(getattr(module, "disable_adapters", False)):
        raise RuntimeError(f"LoRA adapters are disabled for projection {name}")
    if bool(getattr(module, "merged", False)):
        raise RuntimeError(f"LoRA projection {name} must be unmerged for token-local patching")
    use_dora = getattr(module, "use_dora", {})
    if isinstance(use_dora, dict) and bool(use_dora.get(adapter, False)):
        raise RuntimeError(f"DoRA projection {name} is outside the token-weight contract")
    return adapter


def _token_lora_projections(
    block: t.nn.Module,
    donor_state: LoraLayerState,
) -> tuple[TokenLoraProjection, ...]:
    """Resolve all seven block projections and stage donor factors on their devices."""

    projections: list[TokenLoraProjection] = []
    for name, module in block.named_modules():
        if not name or not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        if leaf_name not in LORA_TARGET_MODULES:
            raise RuntimeError(f"unexpected LoRA target in decoder block: {name}")
        adapter = _active_lora_adapter(module, name)
        lora_module = cast(Any, module)
        lora_a = lora_module.lora_A
        lora_b = lora_module.lora_B
        if adapter not in lora_a or adapter not in lora_b:
            raise RuntimeError(f"active adapter {adapter!r} is missing from projection {name}")
        recipient_a = getattr(lora_a[adapter], "weight", None)
        recipient_b = getattr(lora_b[adapter], "weight", None)
        if not isinstance(recipient_a, t.Tensor) or not isinstance(recipient_b, t.Tensor):
            raise RuntimeError(f"projection {name} lacks ordinary LoRA A/B weight tensors")
        dropout_by_adapter = getattr(module, "lora_dropout", {})
        dropout = dropout_by_adapter[adapter]
        dropout_probability = getattr(dropout, "p", 0.0)
        if not isinstance(dropout_probability, int | float) or dropout_probability != 0.0:
            raise RuntimeError(f"projection {name} requires zero LoRA dropout at inference")
        scaling_by_adapter = getattr(module, "scaling", {})
        scaling = scaling_by_adapter[adapter]
        if not isinstance(scaling, int | float) or not math.isfinite(float(scaling)):
            raise RuntimeError(f"projection {name} has a non-finite LoRA scaling")
        a_key = f"{name}.lora_A.{adapter}.weight"
        b_key = f"{name}.lora_B.{adapter}.weight"
        if a_key not in donor_state or b_key not in donor_state:
            raise RuntimeError(f"donor state lacks LoRA factors for projection {name}")
        donor_a = donor_state[a_key]
        donor_b = donor_state[b_key]
        if donor_a.shape != recipient_a.shape or donor_b.shape != recipient_b.shape:
            raise RuntimeError(
                f"donor/recipient LoRA shape mismatch in {name}: "
                f"A={tuple(donor_a.shape)}/{tuple(recipient_a.shape)}, "
                f"B={tuple(donor_b.shape)}/{tuple(recipient_b.shape)}"
            )
        projections.append(
            TokenLoraProjection(
                name=name,
                module=module,
                adapter=adapter,
                donor_a=donor_a.to(device=recipient_a.device, dtype=recipient_a.dtype),
                donor_b=donor_b.to(device=recipient_b.device, dtype=recipient_b.dtype),
                scaling=float(scaling),
            )
        )
    observed = [projection.name.rsplit(".", maxsplit=1)[-1] for projection in projections]
    if len(observed) != len(LORA_TARGET_MODULES) or set(observed) != set(LORA_TARGET_MODULES):
        raise RuntimeError(
            "token-local weight patching requires exactly q/k/v/o and gate/up/down; "
            f"found {sorted(observed)}"
        )
    if set(donor_state) != set(_lora_parameters(block)):
        raise RuntimeError("donor and recipient block adapter schemas differ")
    return tuple(projections)


def _lora_delta(
    hidden: t.Tensor,
    lora_a: t.Tensor,
    lora_b: t.Tensor,
    scaling: float,
) -> t.Tensor:
    projected = t.nn.functional.linear(hidden.to(dtype=lora_a.dtype), lora_a)
    return t.nn.functional.linear(projected, lora_b) * scaling


def _replace_lora_output_at_positions(
    projection: TokenLoraProjection,
    args: tuple[Any, ...],
    output: Any,
    positions: tuple[int, ...],
) -> t.Tensor:
    """Use donor rather than recipient LoRA factors for one token in each batch row."""

    if not args or not isinstance(args[0], t.Tensor):
        raise RuntimeError(f"projection {projection.name} did not receive a tensor input")
    hidden = args[0]
    if not isinstance(output, t.Tensor) or hidden.ndim != 3 or output.ndim != 3:
        raise RuntimeError(
            f"projection {projection.name} must map [batch, sequence, hidden] tensors"
        )
    if hidden.shape[:2] != output.shape[:2] or len(positions) != hidden.shape[0]:
        raise RuntimeError(f"token coordinates do not match projection {projection.name}")
    if any(position < 0 or position >= hidden.shape[1] for position in positions):
        raise ValueError(f"token coordinate is outside projection {projection.name}")
    lora_module = cast(Any, projection.module)
    lora_a = lora_module.lora_A[projection.adapter].weight
    lora_b = lora_module.lora_B[projection.adapter].weight
    rows = t.arange(hidden.shape[0], device=hidden.device)
    columns = t.tensor(positions, dtype=t.int64, device=hidden.device)
    selected = hidden[rows, columns, :]
    donor_delta = _lora_delta(
        selected,
        projection.donor_a,
        projection.donor_b,
        projection.scaling,
    )
    recipient_delta = _lora_delta(selected, lora_a, lora_b, projection.scaling)
    replaced = output.clone()
    replaced[rows, columns, :] += (donor_delta - recipient_delta).to(dtype=output.dtype)
    return replaced


def _token_weight_patch_grid(
    model: t.nn.Module,
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    projection_layers: tuple[tuple[TokenLoraProjection, ...], ...],
    positions: tuple[TokenPositionPair, ...],
    correct_choice_index: int,
    *,
    patch_batch_size: int = 8,
    progress_label: str,
) -> tuple[tuple[float, ...], ...]:
    """Apply each donor block's LoRA update at one selected token per batch row."""

    if not projection_layers:
        raise ValueError("token-local weight patching requires decoder projection layers")
    if not positions or patch_batch_size <= 0:
        raise ValueError("token-local weight patching requires positions and a batch size")
    if not 0 <= correct_choice_index < 5:
        raise ValueError("correct choice index must be in the five-way candidate set")
    layer_count = len(projection_layers)
    batches_per_layer = math.ceil(len(positions) / patch_batch_size)
    values = [[float("nan")] * layer_count for _ in positions]
    started = time.monotonic()
    print(
        f"[token-weight] {progress_label} positions={len(positions)} layers={layer_count} "
        f"batches_per_layer={batches_per_layer}",
        flush=True,
    )
    for layer, projections in enumerate(projection_layers):
        for start in range(0, len(positions), patch_batch_size):
            chunk = positions[start : start + patch_batch_size]
            recipient_positions = tuple(position.recipient_index for position in chunk)
            handles = [
                projection.module.register_forward_hook(
                    lambda _module, args, output, selected_projection=projection, selected_positions=recipient_positions: (
                        _replace_lora_output_at_positions(
                            selected_projection,
                            args,
                            output,
                            selected_positions,
                        )
                    )
                )
                for projection in projections
            ]
            try:
                probabilities = _forward_probabilities(
                    model,
                    input_ids.expand(len(chunk), -1),
                    attention_mask.expand(len(chunk), -1),
                    candidate_ids,
                )
            finally:
                for handle in handles:
                    handle.remove()
            for offset, probability in enumerate(probabilities[:, correct_choice_index].tolist()):
                values[start + offset][layer] = float(probability)
        completed = layer + 1
        if completed == 1 or completed % 4 == 0 or completed == layer_count:
            elapsed = time.monotonic() - started
            remaining = elapsed / completed * (layer_count - completed)
            print(
                f"[token-weight] {progress_label} layers={completed}/{layer_count} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
    if any(not math.isfinite(value) for row in values for value in row):
        raise RuntimeError("token-local weight grid contains an unfilled or non-finite cell")
    return tuple(tuple(row) for row in values)


def _zero_lora_parameters(model: t.nn.Module) -> None:
    """Represent the frozen step-0 model in the same adapter parameterization."""

    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    if not parameters:
        raise RuntimeError("step-0 weight patching could not attach a blank LoRA adapter")
    with t.no_grad():
        for parameter in parameters.values():
            parameter.zero_()


def _load_weight_checkpoint_model(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    step: int,
) -> t.nn.Module:
    """Load every checkpoint as a common PEFT model, including an exact-zero step 0."""

    base = load_base_model(spec, training=False)
    if step == 0:
        model = attach_trainable_lora(base, training_spec_for_run(run))
        _zero_lora_parameters(model)
        model.requires_grad_(False)
        model.eval()
        return model
    path = adapter_dir(root, run, step)
    if not path.is_dir():
        raise FileNotFoundError(f"missing adapter checkpoint: {path}")
    return attach_inference_lora(base, path)


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


def _capture_clean_source_bank(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
) -> SourceBank:
    source_by_record: SourceBank = {}
    for record in records:
        source_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        source_activations, source_probabilities = _capture(
            model,
            targets,
            source_view.input_ids,
            source_view.attention_mask,
            _candidate_ids(processor, record),
        )
        source_by_record[record.record_id] = (
            source_view,
            source_activations,
            source_probabilities,
        )
    return source_by_record


def _capture_weight_source_bundle(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
) -> WeightSourceBundle:
    """Capture donor adapter factors and clean-prompt answer baselines."""

    layer_states = tuple(_capture_lora_layer_state(block) for block in blocks)
    source_by_record: WeightSourceBank = {}
    for record in records:
        source_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        source_probabilities = _forward_probabilities(
            model,
            source_view.input_ids,
            source_view.attention_mask,
            _candidate_ids(processor, record),
        )[0]
        source_by_record[record.record_id] = (source_view, source_probabilities)
    return WeightSourceBundle(layer_states=layer_states, records=source_by_record)


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
        "source_token_count": len(source_view.token_ids),
        "recipient_token_count": len(recipient_view.token_ids),
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
    if plan.interface.patches_all_token_weights:
        base = run_dir(root, run) / "patching" / "layer_only" / plan.interface.value
    else:
        base = run_dir(root, run) / "patching" / "sequence_end"
    if (
        plan.interface is not PatchingInterface.RESID_POST
        and not plan.interface.patches_all_token_weights
    ):
        base /= plan.interface.value
    return (
        base
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
            "anchor": "final token in the rendered generation prompt",
            "stop": (
                "last queried-function-name token"
                if mode is PatchingMode.ACROSS_SAMPLE
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


def _serialize_weight_grid(
    record: ReflectionRecord,
    source: t.Tensor,
    recipient: t.Tensor,
    source_view: PromptPatchView,
    recipient_view: PromptPatchView,
    probabilities: tuple[float, ...],
) -> dict[str, object]:
    """Serialize a layer-only decoder-block parameter intervention."""

    correct_choice = record.choice_function_ids.index(record.function_id)
    recipient_target = float(recipient[correct_choice].item())
    if not probabilities or any(not math.isfinite(value) for value in probabilities):
        raise RuntimeError("weight patching produced an empty or non-finite layer grid")
    return {
        "function_id": record.function_id,
        "source_function_id": record.function_id,
        "recipient_function_id": record.function_id,
        "choice_function_ids": record.choice_function_ids,
        "correct_choice_index": correct_choice,
        "source_probabilities": source.tolist(),
        "recipient_probabilities": recipient.tolist(),
        "site_probability": "correct",
        "axis_kind": "layer_only",
        "source_rendered_prompt": source_view.rendered_prompt,
        "recipient_rendered_prompt": recipient_view.rendered_prompt,
        "weight_scope": {
            "scope": WEIGHT_PATCH_SCOPE,
            "sequence_scope": "all prompt positions",
            "learned_parameters": ("LoRA A/B factors for q/k/v/o and gate/up/down projections"),
            "shared_parameters": (
                "frozen base weights and layer norms are identical across checkpoints"
            ),
        },
        "cells": [
            {
                "layer": layer,
                "probability": probability,
                "delta_from_recipient": probability - recipient_target,
            }
            for layer, probability in enumerate(probabilities)
        ],
    }


def _serialize_token_weight_grid(
    record: ReflectionRecord,
    source: t.Tensor,
    recipient: t.Tensor,
    positions: tuple[TokenPositionPair, ...],
    source_view: PromptPatchView,
    recipient_view: PromptPatchView,
    grid: tuple[tuple[float, ...], ...],
    mode: PatchingMode,
) -> dict[str, object]:
    """Serialize a decoder-block LoRA transplant localized to one prompt token."""

    serialized = _serialize_grid(
        record,
        source,
        recipient,
        positions,
        source_view,
        recipient_view,
        grid,
        mode,
    )
    serialized["axis_kind"] = "token_layer"
    serialized["weight_scope"] = {
        "scope": "selected_token_decoder_block",
        "sequence_scope": "one selected prompt token per intervention",
        "learned_parameters": "LoRA A/B updates for q/k/v/o and gate/up/down projections",
        "recipient_input": "each projection receives the causally current recipient hidden input",
        "attention_coupling": (
            "a selected token's donor K/V updates may affect later query positions"
        ),
        "shared_parameters": "frozen base weights and layer norms remain from the recipient",
    }
    return serialized


def _patch_weight_source_bundle(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    bundle: WeightSourceBundle,
) -> list[dict[str, object]]:
    """Patch one donor checkpoint's complete learned block update, one layer at a time."""

    if len(bundle.layer_states) != len(blocks):
        raise ValueError("donor and recipient decoder layer counts differ")
    probes: list[
        tuple[
            ReflectionRecord,
            PromptPatchView,
            PromptPatchView,
            t.Tensor,
            t.Tensor,
            t.Tensor,
        ]
    ] = []
    values_by_record: dict[str, list[float]] = {
        record.record_id: [float("nan")] * len(blocks) for record in records
    }
    for record in records:
        source_view, source_probabilities = bundle.records[record.record_id]
        recipient_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        candidate_ids = _candidate_ids(processor, record)
        recipient_probabilities = _forward_probabilities(
            model,
            recipient_view.input_ids,
            recipient_view.attention_mask,
            candidate_ids,
        )[0]
        probes.append(
            (
                record,
                source_view,
                recipient_view,
                candidate_ids,
                source_probabilities,
                recipient_probabilities,
            )
        )

    for layer, (block, donor_state) in enumerate(zip(blocks, bundle.layer_states, strict=True)):
        recipient_state = _capture_lora_layer_state(block)
        try:
            _copy_lora_layer_state(block, donor_state)
            for (
                record,
                _source_view,
                recipient_view,
                candidate_ids,
                _source_probabilities,
                _recipient_probabilities,
            ) in probes:
                correct_choice = record.choice_function_ids.index(record.function_id)
                probability = _forward_probabilities(
                    model,
                    recipient_view.input_ids,
                    recipient_view.attention_mask,
                    candidate_ids,
                )[0, correct_choice]
                values_by_record[record.record_id][layer] = float(probability.item())
        finally:
            _copy_lora_layer_state(block, recipient_state)

    serialized: list[dict[str, object]] = []
    for (
        record,
        source_view,
        recipient_view,
        _candidate_ids_value,
        source_probabilities,
        recipient_probabilities,
    ) in probes:
        serialized.append(
            _serialize_weight_grid(
                record,
                source_probabilities,
                recipient_probabilities,
                source_view,
                recipient_view,
                tuple(values_by_record[record.record_id]),
            )
        )
    return serialized


def _patch_token_weight_source_bundle(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    bundle: WeightSourceBundle,
    mode: PatchingMode,
) -> list[dict[str, object]]:
    """Patch donor LoRA updates at one token/layer coordinate at a time."""

    if len(bundle.layer_states) != len(blocks):
        raise ValueError("donor and recipient decoder layer counts differ")
    projection_layers = tuple(
        _token_lora_projections(block, donor_state)
        for block, donor_state in zip(blocks, bundle.layer_states, strict=True)
    )
    serialized: list[dict[str, object]] = []
    for record_index, record in enumerate(records, start=1):
        source_view, source_probabilities = bundle.records[record.record_id]
        recipient_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
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
        grid = _token_weight_patch_grid(
            model,
            recipient_view.input_ids,
            recipient_view.attention_mask,
            candidate_ids,
            projection_layers,
            positions,
            correct_choice,
            progress_label=f"function={record.function_id} ({record_index}/{len(records)})",
        )
        serialized.append(
            _serialize_token_weight_grid(
                record,
                source_probabilities,
                recipient_probabilities,
                positions,
                source_view,
                recipient_view,
                grid,
                mode,
            )
        )
    return serialized


def _patch_record(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
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
        targets,
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


def _temporal_mode(recipient_step: int, donor_step: int) -> PatchingMode:
    if donor_step < recipient_step:
        return PatchingMode.ACROSS_TIME
    if donor_step > recipient_step:
        return PatchingMode.LATER_CHECKPOINT
    raise ValueError("temporal patching does not store same-checkpoint identity cells")


TEMPORAL_ENDPOINT_STEPS = frozenset((0, 1_500))
TEMPORAL_PRIORITY_STEP = 96
TEMPORAL_PRIORITY_LABELS = (
    "corners",
    "border-step-96",
    "remaining-border",
    "remaining-step-96",
)


def _temporal_priority_tier(
    pair: tuple[int, int, PatchingMode],
) -> int:
    """Return the deterministic geometric priority tier for a temporal cell."""

    recipient_step, donor_step, _mode = pair
    recipient_is_endpoint = recipient_step in TEMPORAL_ENDPOINT_STEPS
    donor_is_endpoint = donor_step in TEMPORAL_ENDPOINT_STEPS
    if recipient_is_endpoint and donor_is_endpoint:
        return 0
    if (recipient_is_endpoint and donor_step == TEMPORAL_PRIORITY_STEP) or (
        donor_is_endpoint and recipient_step == TEMPORAL_PRIORITY_STEP
    ):
        return 1
    if recipient_is_endpoint or donor_is_endpoint:
        return 2
    if recipient_step == TEMPORAL_PRIORITY_STEP or donor_step == TEMPORAL_PRIORITY_STEP:
        return 3
    return len(TEMPORAL_PRIORITY_LABELS)


def _seeded_priority_temporal_order(
    scheduled_pairs: list[tuple[int, int, PatchingMode]],
    shuffle_seed: int,
) -> list[tuple[int, int, PatchingMode]]:
    """Shuffle temporal cells deterministically within ordered checkpoint tiers."""

    tiers: list[list[tuple[int, int, PatchingMode]]] = [
        [] for _ in range(len(TEMPORAL_PRIORITY_LABELS) + 1)
    ]
    for pair in scheduled_pairs:
        tiers[_temporal_priority_tier(pair)].append(pair)
    randomizer = random.Random(shuffle_seed)
    for tier in tiers:
        randomizer.shuffle(tier)
    return [pair for tier in tiers for pair in tier]


def _temporal_direction(mode: PatchingMode) -> str:
    if mode is PatchingMode.ACROSS_TIME:
        return "earlier_source_into_later_clean_recipient"
    if mode is PatchingMode.LATER_CHECKPOINT:
        return "later_source_into_earlier_clean_recipient"
    raise ValueError("temporal direction requires a checkpoint-transfer mode")


def _patch_temporal_source_bank(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    mode: PatchingMode,
    source_by_record: SourceBank,
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for record in records:
        source_view, source_activations, source_probabilities = source_by_record[record.record_id]
        recipient_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        serialized.append(
            _patch_record(
                model,
                targets,
                processor,
                record,
                mode,
                source_view,
                recipient_view,
                source_activations,
                source_probabilities,
            )
        )
    return serialized


def _write_temporal_artifact(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    plan: PatchingPlan,
    donor_step: int,
    serialized: list[dict[str, object]],
) -> None:
    output = _patch_output_path(root, run, plan, donor_step)
    write_json(
        output,
        {
            "model": spec,
            "run": run,
            "plan": plan,
            "donor_step": donor_step,
            "patch_direction": _temporal_direction(plan.mode),
            "records": serialized,
        },
    )
    print(
        f"[patch] {run.model}/{run.condition.value} {plan.interface.value}/"
        f"{plan.mode.value} "
        f"recipient={plan.recipient_step} donor={donor_step} -> {output}",
        flush=True,
    )


def _run_weight_temporal_pairs(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    pending_pairs: list[tuple[int, int, PatchingMode]],
    interface: PatchingInterface,
) -> None:
    """Fill temporal block-weight cells while reusing CPU-resident donor adapters."""

    if not interface.patches_weights:
        raise ValueError("weight temporal runner requires a weight-patching interface")

    donor_steps = tuple(
        sorted({donor_step for _recipient_step, donor_step, _mode in pending_pairs})
    )
    sources_by_step: dict[int, WeightSourceBundle] = {}
    for donor_step in donor_steps:
        donor_model = _load_weight_checkpoint_model(root, run, spec, donor_step)
        donor_blocks = resolve_decoder_blocks(donor_model, spec)
        try:
            sources_by_step[donor_step] = _capture_weight_source_bundle(
                donor_model,
                donor_blocks,
                processor,
                records,
            )
        finally:
            _release_model(donor_model)
        print(
            f"[patch-matrix] captured {interface.value} sources at step {donor_step}",
            flush=True,
        )

    recipient_model: t.nn.Module | None = None
    recipient_blocks: tuple[t.nn.Module, ...] = ()
    loaded_recipient_step: int | None = None
    try:
        for recipient_step, donor_step, mode in pending_pairs:
            if recipient_step != loaded_recipient_step:
                if recipient_model is not None:
                    _release_model(recipient_model)
                recipient_model = _load_weight_checkpoint_model(
                    root,
                    run,
                    spec,
                    recipient_step,
                )
                recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
                loaded_recipient_step = recipient_step
            if recipient_model is None:  # pragma: no cover - guarded by the load above
                raise AssertionError("weight-patch recipient model was not loaded")
            plan = PatchingPlan(
                mode=mode,
                recipient_step=recipient_step,
                donor_steps=(donor_step,),
                interface=interface,
            )
            if interface.patches_token_weights:
                serialized = _patch_token_weight_source_bundle(
                    recipient_model,
                    recipient_blocks,
                    processor,
                    records,
                    sources_by_step[donor_step],
                    mode,
                )
            else:
                serialized = _patch_weight_source_bundle(
                    recipient_model,
                    recipient_blocks,
                    processor,
                    records,
                    sources_by_step[donor_step],
                )
            _write_temporal_artifact(
                root,
                run,
                spec,
                plan,
                donor_step,
                serialized,
            )
    finally:
        if recipient_model is not None:
            _release_model(recipient_model)


def run_temporal_patching_matrix(
    root: Path,
    run: RunKey,
    recipient_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
    interface: PatchingInterface,
    *,
    shuffle_seed: int | None = None,
    allow_provisional_model: bool = False,
) -> None:
    """Fill selected checkpoint-transfer cells while reusing source and recipient loads."""

    if not t.cuda.is_available():
        raise RuntimeError("checkpoint patching requires CUDA")
    if tuple(sorted(set(recipient_steps))) != recipient_steps or any(
        step not in CHECKPOINT_STEPS for step in recipient_steps
    ):
        raise ValueError("temporal recipient steps must be unique, increasing checkpoints")
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(mode is PatchingMode.ACROSS_SAMPLE for mode in modes)
    ):
        raise ValueError("temporal matrix modes must be unique checkpoint-transfer modes")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ValueError("temporal matrix shuffle seed must be non-negative")

    scheduled_pairs: list[tuple[int, int, PatchingMode]] = []
    for recipient_step in recipient_steps:
        for donor_step in CHECKPOINT_STEPS:
            if donor_step == recipient_step:
                continue
            mode = _temporal_mode(recipient_step, donor_step)
            if mode not in modes:
                continue
            scheduled_pairs.append((recipient_step, donor_step, mode))
    if shuffle_seed is not None:
        scheduled_pairs = _seeded_priority_temporal_order(
            scheduled_pairs,
            shuffle_seed,
        )

    pending_pairs: list[tuple[int, int, PatchingMode]] = []
    skipped = 0
    for recipient_step, donor_step, mode in scheduled_pairs:
        plan = PatchingPlan(
            mode=mode,
            recipient_step=recipient_step,
            donor_steps=(donor_step,),
            interface=interface,
        )
        if _patch_output_path(root, run, plan, donor_step).is_file():
            skipped += 1
        else:
            pending_pairs.append((recipient_step, donor_step, mode))
    if skipped:
        print(
            f"[patch-matrix] {run.model}/{run.condition.value} {interface.value} "
            f"skipped {skipped} existing temporal artifact(s)",
            flush=True,
        )
    if not pending_pairs:
        return
    if shuffle_seed is not None:
        tier_counts = [0] * (len(TEMPORAL_PRIORITY_LABELS) + 1)
        for pair in pending_pairs:
            tier_counts[_temporal_priority_tier(pair)] += 1
        count_summary = ", ".join(
            [
                *(
                    f"{label}: {count}"
                    for label, count in zip(
                        TEMPORAL_PRIORITY_LABELS,
                        tier_counts[:-1],
                        strict=True,
                    )
                ),
                f"remainder: {tier_counts[-1]}",
            ]
        )
        print(
            f"[patch-matrix] priority-shuffled {len(pending_pairs)} missing temporal cells "
            f"with seed {shuffle_seed} ({count_summary})",
            flush=True,
        )

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    records = _selected_records(run.seed)
    if interface.patches_weights:
        _run_weight_temporal_pairs(
            root,
            run,
            spec,
            processor,
            records,
            pending_pairs,
            interface,
        )
        return
    donor_steps = tuple(
        sorted({donor_step for _recipient_step, donor_step, _mode in pending_pairs})
    )
    sources_by_step: dict[int, SourceBank] = {}
    for donor_step in donor_steps:
        donor_model = _load_checkpoint_model(root, run, spec, donor_step)
        donor_blocks = resolve_decoder_blocks(donor_model, spec)
        donor_targets = _resolve_patch_targets(donor_blocks, interface)
        try:
            sources_by_step[donor_step] = _capture_clean_source_bank(
                donor_model,
                donor_targets,
                processor,
                records,
            )
        finally:
            _release_model(donor_model)
        print(
            f"[patch-matrix] captured {interface.value} sources at step {donor_step}",
            flush=True,
        )

    recipient_model: t.nn.Module | None = None
    recipient_targets: tuple[PatchTarget, ...] = ()
    loaded_recipient_step: int | None = None
    try:
        for recipient_step, donor_step, mode in pending_pairs:
            if recipient_step != loaded_recipient_step:
                if recipient_model is not None:
                    _release_model(recipient_model)
                recipient_model = _load_checkpoint_model(root, run, spec, recipient_step)
                recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
                recipient_targets = _resolve_patch_targets(recipient_blocks, interface)
                loaded_recipient_step = recipient_step
            if recipient_model is None:  # pragma: no cover - guarded by the load above
                raise AssertionError("temporal recipient model was not loaded")
            plan = PatchingPlan(
                mode=mode,
                recipient_step=recipient_step,
                donor_steps=(donor_step,),
                interface=interface,
            )
            serialized = _patch_temporal_source_bank(
                recipient_model,
                recipient_targets,
                processor,
                records,
                mode,
                sources_by_step[donor_step],
            )
            _write_temporal_artifact(
                root,
                run,
                spec,
                plan,
                donor_step,
                serialized,
            )
    finally:
        if recipient_model is not None:
            _release_model(recipient_model)


def _run_weight_patching(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    plan: PatchingPlan,
    pending: tuple[int, ...],
) -> None:
    """Run an explicitly selected set of checkpoint-to-checkpoint weight patches."""

    if plan.mode is PatchingMode.ACROSS_SAMPLE:
        raise ValueError("weight patching is defined only for checkpoint transfer")
    for donor_step in pending:
        donor_model = _load_weight_checkpoint_model(root, run, spec, donor_step)
        donor_blocks = resolve_decoder_blocks(donor_model, spec)
        try:
            bundle = _capture_weight_source_bundle(
                donor_model,
                donor_blocks,
                processor,
                records,
            )
        finally:
            _release_model(donor_model)

        recipient_model = _load_weight_checkpoint_model(
            root,
            run,
            spec,
            plan.recipient_step,
        )
        recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
        try:
            if plan.interface.patches_token_weights:
                serialized = _patch_token_weight_source_bundle(
                    recipient_model,
                    recipient_blocks,
                    processor,
                    records,
                    bundle,
                    plan.mode,
                )
            else:
                serialized = _patch_weight_source_bundle(
                    recipient_model,
                    recipient_blocks,
                    processor,
                    records,
                    bundle,
                )
        finally:
            _release_model(recipient_model)
        _write_temporal_artifact(
            root,
            run,
            spec,
            PatchingPlan(
                mode=plan.mode,
                recipient_step=plan.recipient_step,
                donor_steps=(donor_step,),
                interface=plan.interface,
            ),
            donor_step,
            serialized,
        )
        del bundle
        gc.collect()


def run_patching(
    root: Path,
    run: RunKey,
    plan: PatchingPlan,
    *,
    allow_provisional_model: bool = False,
) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("checkpoint patching requires CUDA")
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
            f"[patch] {run.model}/{run.condition.value} {plan.interface.value} "
            f"skipped {skipped} existing artifact(s)",
            flush=True,
        )
    if not pending:
        return
    if plan.interface.patches_weights:
        _run_weight_patching(
            root,
            run,
            spec,
            processor,
            records,
            plan,
            pending,
        )
        return
    if plan.mode is PatchingMode.ACROSS_SAMPLE:
        donor_step = pending[0]
        recipient_model = _load_checkpoint_model(root, run, spec, plan.recipient_step)
        recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
        recipient_targets = _resolve_patch_targets(recipient_blocks, plan.interface)
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
                    recipient_targets,
                    source_view.input_ids,
                    source_view.attention_mask,
                    _candidate_ids(processor, record),
                )
                serialized.append(
                    _patch_record(
                        recipient_model,
                        recipient_targets,
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
            f"[patch] {run.model}/{run.condition.value} {plan.interface.value}/"
            f"{plan.mode.value} "
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
        donor_targets = _resolve_patch_targets(donor_blocks, plan.interface)
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
                    donor_targets,
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
        recipient_targets = _resolve_patch_targets(recipient_blocks, plan.interface)
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
                        recipient_targets,
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
                "patch_direction": (
                    "later_source_into_earlier_clean_recipient"
                    if plan.mode is PatchingMode.LATER_CHECKPOINT
                    else "earlier_source_into_later_clean_recipient"
                ),
                "records": serialized,
            },
        )
        print(
            f"[patch] {run.model}/{run.condition.value} {plan.interface.value}/"
            f"{plan.mode.value} "
            f"recipient={plan.recipient_step} donor={donor_step} -> {output}",
            flush=True,
        )
        del source_by_record
        gc.collect()


__all__ = ["build_token_axis_metadata", "run_patching", "run_temporal_patching_matrix"]
