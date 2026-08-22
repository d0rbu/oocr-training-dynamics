"""Memory-safe decoder-interface patching across samples and checkpoint time."""

from __future__ import annotations

import gc
import math
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.nn.functional as functional

from oocr_training_dynamics.activation_examples import (
    ACTIVATION_EXAMPLE_METRIC,
    ACTIVATION_EXAMPLE_TOP_K,
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_MAX_TOKENS,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
    ActivationExamplePrompt,
    ActivationExampleSource,
    FineWebActivationDocument,
    activation_example_corpus_metadata,
    build_activation_example_source_prompts,
    build_format_control_patch_prompt,
    load_fineweb_activation_documents,
)
from oocr_training_dynamics.artifacts import adapter_dir, read_json, run_dir, write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
    RunKey,
    TokenWeightRuntime,
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
    build_cyclic_choice_pair,
    build_deranged_choice_pair,
    build_letter_context_pair,
    build_unrelated_question_pair,
    reverse_token_position_pairs,
    reverse_token_position_pairs_through_first_difference,
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
ProbabilityForward = Callable[[t.nn.Module, t.Tensor, t.Tensor, t.Tensor], t.Tensor]
ProbabilityGrid = tuple[tuple[float, ...], ...]
AnswerLabelLens = tuple[tuple[tuple[float, ...], ...], ...]
ANSWER_LOGIT_LENS_TOP_P = 0.9
VocabularyTopToken = tuple[int, float]
VocabularyLogitLens = tuple[tuple[tuple[VocabularyTopToken, ...], ...], ...]
VOCABULARY_LOGIT_LENS_TOP_K = 5
VOCABULARY_LOGIT_LENS_BATCH_SIZE = 32
VOCABULARY_LOGIT_LENS_MODES = (
    PatchingMode.ACROSS_SAMPLE,
    PatchingMode.CYCLIC_CHOICES,
    PatchingMode.DERANGED_CHOICES,
    PatchingMode.UNRELATED_QUESTION,
    PatchingMode.UNRELATED_QUESTION_SAME_LETTER,
    PatchingMode.LETTER_CONTEXT_SAME,
    PatchingMode.LETTER_CONTEXT_DIFFERENT,
    PatchingMode.SAME_MCQ_FORMATS,
    PatchingMode.UNRELATED_MCQ_FORMATS,
    PatchingMode.SAME_CONVERSATIONAL_CHOICES,
    PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES,
)


@dataclass(frozen=True)
class WeightSourceBundle:
    """CPU-resident donor LoRA parameters and clean-prompt baselines."""

    layer_states: tuple[LoraLayerState, ...]
    records: WeightSourceBank


@dataclass(frozen=True)
class PromptCounterfactualSpec:
    """One prompt-only activation source and its auditable answer-label semantics."""

    source_messages: tuple[ChatMessage, ...]
    source_function_id: str
    source_correct_choice_index: int | None
    recipient_messages: tuple[ChatMessage, ...]
    recipient_function_id: str
    recipient_correct_choice_index: int
    source_choice_function_ids: tuple[str, ...] | None
    source_choice_texts: tuple[str, ...] | None
    source_question_id: str | None
    source_question: str | None
    patch_direction: str
    stops_at_first_difference: bool
    source_format: str | None = None
    source_label_relation: str | None = None
    source_context_id: str | None = None
    source_context: str | None = None


@dataclass(frozen=True)
class PromptCounterfactualSourceRecord:
    """CPU-resident donor-prompt state plus its donor-side answer readout."""

    spec: PromptCounterfactualSpec
    source_view: PromptPatchView
    recipient_view: PromptPatchView
    source_activations: tuple[t.Tensor, ...]
    source_probabilities: t.Tensor
    source_answer_logit_lens: AnswerLabelLens


PromptCounterfactualSourceBank = dict[str, PromptCounterfactualSourceRecord]


@dataclass(frozen=True)
class ActivationExampleView:
    """Token-exact candidate prompt and its captured interface activations."""

    example_id: str
    category: str
    target: str
    input_ids: t.Tensor
    attention_mask: t.Tensor
    rendered_prompt: str
    token_ids: tuple[int, ...]
    token_labels: tuple[str, ...]
    activations: tuple[t.Tensor, ...]
    provenance: dict[str, object] | None = None


@dataclass(frozen=True)
class ActivationReferenceBank:
    """Source and recipient reference vectors for one mode/function pair."""

    mode: PatchingMode
    function_id: str
    positions: tuple[TokenPositionPair, ...]
    source_activations: tuple[t.Tensor, ...]
    recipient_activations: tuple[t.Tensor, ...]


@dataclass(frozen=True)
class TokenLoraProjection:
    """One recipient LoRA projection plus the donor factors used at selected tokens."""

    name: str
    module: t.nn.Module
    adapter: str
    donor_a: t.Tensor
    donor_b: t.Tensor
    scaling: float


def _candidate_ids(
    processor: Any,
    record: ReflectionRecord,
    messages: tuple[ChatMessage, ...] | None = None,
    *,
    device: str = "cuda",
) -> t.Tensor:
    values: list[int] = []
    candidate_messages = record.messages if messages is None else messages
    for letter in "ABCDE":
        with_target = (*candidate_messages[:-1], ChatMessage("assistant", letter))
        example = tokenize_messages(processor, record.record_id, with_target)
        values.append(int(example.input_ids[0, first_target_position(example)].item()))
    if len(set(values)) != 5:
        raise RuntimeError("A-E must have distinct first target tokens")
    return t.tensor(values, dtype=t.int64, device=device)


def _prompt_counterfactual_spec(
    record: ReflectionRecord,
    mode: PatchingMode,
) -> PromptCounterfactualSpec:
    if mode is PatchingMode.ACROSS_SAMPLE:
        pair = build_across_sample_pair(record)
        source_correct = record.choice_function_ids.index(pair.dirty_function_id)
        return PromptCounterfactualSpec(
            source_messages=pair.dirty_messages,
            source_function_id=pair.dirty_function_id,
            source_correct_choice_index=source_correct,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=record.choice_function_ids,
            source_choice_texts=None,
            source_question_id=None,
            source_question=None,
            patch_direction="dirty_source_into_clean_recipient",
            stops_at_first_difference=False,
        )
    if mode is PatchingMode.REVERSE_ACROSS_SAMPLE:
        pair = build_across_sample_pair(record)
        return PromptCounterfactualSpec(
            source_messages=record.messages,
            source_function_id=record.function_id,
            source_correct_choice_index=record.choice_function_ids.index(record.function_id),
            recipient_messages=pair.dirty_messages,
            recipient_function_id=pair.dirty_function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(pair.dirty_function_id),
            source_choice_function_ids=record.choice_function_ids,
            source_choice_texts=None,
            source_question_id=None,
            source_question=None,
            patch_direction="clean_source_into_dirty_recipient",
            stops_at_first_difference=False,
        )
    if mode is PatchingMode.CYCLIC_CHOICES:
        pair = build_cyclic_choice_pair(record)
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=record.function_id,
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=pair.source_choice_function_ids,
            source_choice_texts=None,
            source_question_id=None,
            source_question=None,
            patch_direction="cyclic_choice_source_into_clean_recipient",
            stops_at_first_difference=True,
            source_format="same_function_mcq",
            source_label_relation="different_from_recipient",
        )
    if mode is PatchingMode.DERANGED_CHOICES:
        pair = build_deranged_choice_pair(record)
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=record.function_id,
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=pair.source_choice_function_ids,
            source_choice_texts=None,
            source_question_id=None,
            source_question=None,
            patch_direction="deranged_choice_source_into_clean_recipient",
            stops_at_first_difference=True,
            source_format="same_function_mcq",
            source_label_relation="different_from_recipient",
        )
    if mode is PatchingMode.UNRELATED_QUESTION:
        pair = build_unrelated_question_pair(record)
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=f"unrelated:{pair.question_id}",
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=None,
            source_choice_texts=pair.source_choices,
            source_question_id=pair.question_id,
            source_question=pair.question,
            patch_direction="unrelated_question_source_into_clean_recipient",
            stops_at_first_difference=True,
            source_format="unrelated_mcq",
            source_label_relation=pair.label_relation,
        )
    if mode is PatchingMode.UNRELATED_QUESTION_SAME_LETTER:
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=f"unrelated:{pair.question_id}",
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=None,
            source_choice_texts=pair.source_choices,
            source_question_id=pair.question_id,
            source_question=pair.question,
            patch_direction="unrelated_question_same_letter_source_into_clean_recipient",
            stops_at_first_difference=True,
            source_format="unrelated_mcq",
            source_label_relation=pair.label_relation,
        )
    if mode in {
        PatchingMode.LETTER_CONTEXT_SAME,
        PatchingMode.LETTER_CONTEXT_DIFFERENT,
    }:
        pair = build_letter_context_pair(
            record,
            match_clean_label=mode is PatchingMode.LETTER_CONTEXT_SAME,
        )
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=f"letter-context:{pair.context_id}",
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=None,
            source_choice_texts=None,
            source_question_id=None,
            source_question=None,
            patch_direction=(
                "same_letter_context_source_into_clean_recipient"
                if mode is PatchingMode.LETTER_CONTEXT_SAME
                else "different_letter_context_source_into_clean_recipient"
            ),
            stops_at_first_difference=True,
            source_format="non_mcq_text_completion",
            source_label_relation=pair.label_relation,
            source_context_id=pair.context_id,
            source_context=pair.context,
        )
    if mode in {
        PatchingMode.SAME_MCQ_FORMATS,
        PatchingMode.UNRELATED_MCQ_FORMATS,
        PatchingMode.SAME_CONVERSATIONAL,
        PatchingMode.UNRELATED_OPEN_ENDED,
        PatchingMode.SAME_CONVERSATIONAL_CHOICES,
        PatchingMode.UNRELATED_CONVERSATIONAL_CHOICES,
    }:
        pair = build_format_control_patch_prompt(
            record,
            ActivationExampleSource(mode.value),
        )
        return PromptCounterfactualSpec(
            source_messages=pair.source_messages,
            source_function_id=pair.source_function_id,
            source_correct_choice_index=pair.source_correct_choice_index,
            recipient_messages=record.messages,
            recipient_function_id=record.function_id,
            recipient_correct_choice_index=record.choice_function_ids.index(record.function_id),
            source_choice_function_ids=pair.source_choice_function_ids,
            source_choice_texts=pair.source_choice_texts,
            source_question_id=pair.source_question_id,
            source_question=pair.source_question,
            patch_direction=f"{mode.value}_source_into_clean_recipient",
            stops_at_first_difference=True,
            source_format=pair.source_format,
            source_label_relation=pair.source_label_relation,
        )
    raise ValueError(f"{mode.value} is not a prompt-counterfactual patching mode")


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
    return _visible_token_label(tokenizer, token_id, value)


def _visible_token_label(tokenizer: Any, token_id: int, value: str) -> str:
    """Apply the one-token display contract to an already decoded value."""

    visible = value.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␠")
    if visible:
        return visible
    fallback = tokenizer.convert_ids_to_tokens(token_id)
    return str(fallback)


def _vocabulary_token_label(tokenizer: Any, token_id: int) -> str:
    """Decode a model-output row, including explicitly labeled padded rows."""

    tokenizer_size = len(tokenizer)
    if not isinstance(tokenizer_size, int) or tokenizer_size <= 0:
        raise RuntimeError("tokenizer must expose a positive vocabulary size")
    if token_id < 0:
        raise ValueError("vocabulary token ID must be non-negative")
    if token_id >= tokenizer_size:
        return f"<unused-output-row:{token_id}>"
    label = _token_label(tokenizer, token_id)
    return f"<token:{token_id}>" if label == "None" else label


def _vocabulary_token_labels(
    tokenizer: Any,
    token_ids: tuple[int, ...],
) -> dict[int, str]:
    """Decode distinct output rows in one tokenizer batch with scalar-path parity."""

    if len(set(token_ids)) != len(token_ids):
        raise ValueError("vocabulary token-label batch must contain distinct IDs")
    if not token_ids:
        return {}
    tokenizer_size = len(tokenizer)
    if not isinstance(tokenizer_size, int) or tokenizer_size <= 0:
        raise RuntimeError("tokenizer must expose a positive vocabulary size")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("vocabulary token ID must be non-negative")
    valid_ids = tuple(token_id for token_id in token_ids if token_id < tokenizer_size)
    decoded = tokenizer.batch_decode(
        [[token_id] for token_id in valid_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if (
        not isinstance(decoded, list)
        or len(decoded) != len(valid_ids)
        or not all(isinstance(value, str) for value in decoded)
    ):
        raise TypeError("batched token decoder must return one string per token ID")
    decoded_values = cast(list[str], decoded)
    labels = {
        token_id: (
            f"<token:{token_id}>"
            if (label := _visible_token_label(tokenizer, token_id, value)) == "None"
            else label
        )
        for token_id, value in zip(valid_ids, decoded_values, strict=True)
    }
    labels.update(
        {
            token_id: f"<unused-output-row:{token_id}>"
            for token_id in token_ids
            if token_id >= tokenizer_size
        }
    )
    return labels


def _capture_activation_example(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    processor: Any,
    prompt: ActivationExamplePrompt,
    candidate_ids: t.Tensor,
) -> ActivationExampleView:
    example = tokenize_messages(processor, prompt.example_id, prompt.messages)
    input_ids, attention_mask = _prefix(example)
    rendered = _render_generation_prompt(processor, prompt.messages)
    tokenizer = tokenizer_for(processor)
    encoded = tokenizer(rendered, add_special_tokens=False)
    raw_token_ids = encoded.get("input_ids")
    if not isinstance(raw_token_ids, list) or not all(
        isinstance(value, int) for value in raw_token_ids
    ):
        raise TypeError("activation-example prompt must encode to one integer token list")
    token_ids = tuple(raw_token_ids)
    if token_ids != tuple(int(value) for value in input_ids[0].tolist()):
        raise RuntimeError("activation-example tokens do not match the rendered chat prefix")
    activations, _probabilities = _capture(
        model,
        targets,
        input_ids,
        attention_mask,
        candidate_ids,
    )
    return ActivationExampleView(
        example_id=prompt.example_id,
        category=prompt.category,
        target=prompt.messages[-1].content,
        input_ids=input_ids,
        attention_mask=attention_mask,
        rendered_prompt=rendered,
        token_ids=token_ids,
        token_labels=tuple(_token_label(tokenizer, token_id) for token_id in token_ids),
        activations=activations,
    )


def _capture_fineweb_activation_example(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    processor: Any,
    document: FineWebActivationDocument,
    candidate_ids: t.Tensor,
) -> ActivationExampleView:
    """Capture one raw FineWeb prefix without introducing a chat-template wrapper."""

    tokenizer = tokenizer_for(processor)
    encoded = tokenizer(
        document.text,
        add_special_tokens=True,
        truncation=True,
        max_length=FINEWEB_ACTIVATION_MAX_TOKENS,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if (
        not isinstance(input_ids, t.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
    ):
        raise TypeError("FineWeb activation document must encode to one non-empty token row")
    if attention_mask is None:
        attention_mask = t.ones_like(input_ids)
    if (
        not isinstance(attention_mask, t.Tensor)
        or attention_mask.shape != input_ids.shape
        or not bool(t.all(attention_mask == 1))
    ):
        raise ValueError("FineWeb activation document requires one unpadded token prefix")
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    token_ids = tuple(int(value) for value in input_ids[0].tolist())
    rendered = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("FineWeb token decoder must return text")
    activations, _probabilities = _capture(
        model,
        targets,
        input_ids,
        attention_mask,
        candidate_ids,
    )
    return ActivationExampleView(
        example_id=document.example_id,
        category="fineweb_pretraining",
        target="raw next-token text",
        input_ids=input_ids,
        attention_mask=attention_mask,
        rendered_prompt=rendered,
        token_ids=token_ids,
        token_labels=tuple(_token_label(tokenizer, token_id) for token_id in token_ids),
        activations=activations,
        provenance=document.provenance,
    )


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


def _prompt_counterfactual_views(
    processor: Any,
    record: ReflectionRecord,
    mode: PatchingMode,
    spec: PromptCounterfactualSpec,
    *,
    device: str = "cuda",
) -> tuple[PromptPatchView, PromptPatchView]:
    """Tokenize a prompt pair and apply its exact preregistered reverse-axis stop."""

    if not mode.uses_prompt_counterfactual:
        raise ValueError(f"{mode.value} does not define a prompt-counterfactual view")
    source_alias = (
        FUNCTION_BY_ID[spec.source_function_id].alias
        if spec.source_function_id in FUNCTION_BY_ID
        else FUNCTION_BY_ID[record.function_id].alias
    )
    recipient_alias = (
        FUNCTION_BY_ID[spec.recipient_function_id].alias
        if spec.recipient_function_id in FUNCTION_BY_ID
        else FUNCTION_BY_ID[record.function_id].alias
    )
    source_view = _prompt_patch_view(
        processor,
        record,
        spec.source_messages,
        source_alias,
        stop_at_sequence_start=spec.stops_at_first_difference,
        device=device,
    )
    recipient_view = _prompt_patch_view(
        processor,
        record,
        spec.recipient_messages,
        recipient_alias,
        stop_at_sequence_start=spec.stops_at_first_difference,
        device=device,
    )
    if spec.stops_at_first_difference:
        positions = reverse_token_position_pairs_through_first_difference(
            source_view.token_ids,
            recipient_view.token_ids,
        )
        source_view = replace(source_view, stop_index=positions[-1].source_index)
        recipient_view = replace(recipient_view, stop_index=positions[-1].recipient_index)
    return source_view, recipient_view


def _counterfactual_candidate_ids(
    processor: Any,
    record: ReflectionRecord,
    spec: PromptCounterfactualSpec,
    *,
    device: str = "cuda",
) -> tuple[t.Tensor, t.Tensor]:
    source = _candidate_ids(processor, record, spec.source_messages, device=device)
    recipient = _candidate_ids(
        processor,
        record,
        spec.recipient_messages,
        device=device,
    )
    if not t.equal(source, recipient):
        raise RuntimeError(
            "source and recipient chat templates must encode the A-E answer labels identically"
        )
    return source, recipient


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


def _forward_probabilities_last_token(
    model: t.nn.Module,
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
) -> t.Tensor:
    """Compute only final-position vocabulary logits through the model's native API."""

    with t.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    if output.logits.ndim != 3 or output.logits.shape[1] != 1:
        raise RuntimeError("final-token probability forward must return one logit position")
    logits = output.logits[:, -1, candidate_ids].to(dtype=t.float32)
    return t.softmax(logits, dim=-1).detach().cpu()


def _token_weight_probability_forward(runtime: TokenWeightRuntime) -> ProbabilityForward:
    if runtime in {TokenWeightRuntime.REFERENCE, TokenWeightRuntime.OPTIMIZED}:
        return _forward_probabilities
    raise AssertionError(f"unhandled token-weight runtime: {runtime}")


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
    source_correct_choice_index: int | None = None,
    *,
    patch_batch_size: int = 8,
) -> tuple[ProbabilityGrid, ProbabilityGrid | None]:
    if len(source_activations) != len(targets):
        raise ValueError("source activation count must equal decoder layer count")
    if not positions or patch_batch_size <= 0:
        raise ValueError("token patching requires positions and a positive batch size")
    if not 0 <= correct_choice_index < 5:
        raise ValueError("correct choice index must be in the five-way candidate set")
    if source_correct_choice_index is not None and not 0 <= source_correct_choice_index < 5:
        raise ValueError("source correct choice index must be in the five-way candidate set")
    values = [[float("nan")] * len(targets) for _ in positions]
    source_target_values = (
        [[float("nan")] * len(targets) for _ in positions]
        if source_correct_choice_index is not None
        else None
    )
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
            if source_target_values is not None and source_correct_choice_index is not None:
                for offset, probability in enumerate(
                    probabilities[:, source_correct_choice_index].tolist()
                ):
                    source_target_values[start + offset][layer] = float(probability)
    if any(not math.isfinite(value) for row in values for value in row):
        raise RuntimeError("token patch grid contains an unfilled or non-finite cell")
    if source_target_values is not None and any(
        not math.isfinite(value) for row in source_target_values for value in row
    ):
        raise RuntimeError("source-target patch grid contains an unfilled or non-finite cell")
    return (
        tuple(tuple(row) for row in values),
        None if source_target_values is None else tuple(tuple(row) for row in source_target_values),
    )


def _final_norm(model: t.nn.Module) -> t.nn.Module:
    """Resolve the decoder's final normalization through optional PEFT wrappers."""

    current: t.nn.Module | None = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        norm = getattr(current, "norm", None)
        if isinstance(norm, t.nn.Module):
            return norm
        nested = getattr(current, "model", None)
        current = nested if isinstance(nested, t.nn.Module) else None
    raise RuntimeError("logit lens requires a resolvable decoder final normalization")


def _logit_lens_readout(
    model: t.nn.Module,
) -> tuple[t.nn.Module, t.nn.Parameter, t.Tensor, t.Tensor | None]:
    """Resolve the exact final-normalization and output-embedding readout."""

    norm = _final_norm(model)
    norm_parameter = next(norm.parameters(), None)
    if norm_parameter is None:
        raise RuntimeError("decoder final normalization must expose its dtype and device")
    output_embedding_resolver = getattr(model, "get_output_embeddings", None)
    if not callable(output_embedding_resolver):
        raise RuntimeError("logit lens requires output embeddings")
    output_embeddings = output_embedding_resolver()
    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, t.Tensor) or weight.ndim != 2:
        raise RuntimeError("logit lens requires a matrix output embedding")
    bias = getattr(output_embeddings, "bias", None)
    if bias is not None and (
        not isinstance(bias, t.Tensor) or bias.ndim != 1 or bias.shape[0] != weight.shape[0]
    ):
        raise RuntimeError("logit lens output-embedding bias must match the vocabulary")
    return norm, norm_parameter, weight, bias


def _flatten_logit_lens_residuals(
    residual_activations: tuple[t.Tensor, ...],
    token_indices: tuple[int, ...],
) -> tuple[t.Tensor, int]:
    if not residual_activations or not token_indices:
        raise ValueError("logit lens requires layers and token positions")
    if any(activation.ndim != 2 for activation in residual_activations):
        raise RuntimeError("residual logit-lens activations must have [sequence, hidden] shape")
    sequence_lengths = {activation.shape[0] for activation in residual_activations}
    hidden_sizes = {activation.shape[1] for activation in residual_activations}
    if len(sequence_lengths) != 1 or len(hidden_sizes) != 1:
        raise RuntimeError("residual logit-lens activations must share [sequence, hidden] shape")
    sequence_length = next(iter(sequence_lengths))
    if any(index < 0 or index >= sequence_length for index in token_indices):
        raise ValueError("logit-lens token index lies outside a captured residual sequence")
    layer_count = len(residual_activations)
    flat = t.stack(
        [
            residual_activations[layer][token_index]
            for token_index in token_indices
            for layer in range(layer_count)
        ],
        dim=0,
    )
    return flat, layer_count


def _answer_label_logit_lens(
    model: t.nn.Module,
    residual_activations: tuple[t.Tensor, ...],
    token_indices: tuple[int, ...],
    candidate_ids: t.Tensor,
) -> AnswerLabelLens:
    """Apply final norm and selected A-E unembedding rows to residual states."""

    flat, layer_count = _flatten_logit_lens_residuals(residual_activations, token_indices)
    norm, norm_parameter, weight, bias = _logit_lens_readout(model)
    if candidate_ids.ndim != 1 or candidate_ids.numel() != 5:
        raise ValueError("answer-label logit lens requires exactly five candidate token IDs")
    if int(candidate_ids.min().item()) < 0 or int(candidate_ids.max().item()) >= weight.shape[0]:
        raise ValueError("answer-label token ID lies outside the output vocabulary")
    selected_ids = candidate_ids.to(device=weight.device)
    selected_weight = weight.index_select(0, selected_ids)
    selected_bias = bias.index_select(0, selected_ids) if isinstance(bias, t.Tensor) else None
    with t.inference_mode():
        normalized = norm(flat.to(device=norm_parameter.device, dtype=norm_parameter.dtype))
        logits = functional.linear(
            normalized.to(device=selected_weight.device, dtype=selected_weight.dtype),
            selected_weight,
            selected_bias,
        )
        probabilities = t.softmax(logits.to(dtype=t.float32), dim=-1).cpu()
    return tuple(
        tuple(
            tuple(float(value) for value in probabilities[token * layer_count + layer].tolist())
            for layer in range(layer_count)
        )
        for token in range(len(token_indices))
    )


def _full_vocabulary_logit_lens(
    model: t.nn.Module,
    residual_activations: tuple[t.Tensor, ...],
    token_indices: tuple[int, ...],
    *,
    top_k: int = VOCABULARY_LOGIT_LENS_TOP_K,
    batch_size: int = VOCABULARY_LOGIT_LENS_BATCH_SIZE,
) -> VocabularyLogitLens:
    """Return sparse top-k tokens with probabilities normalized over the full vocabulary."""

    if top_k <= 0 or batch_size <= 0:
        raise ValueError("full-vocabulary logit-lens top-k and batch size must be positive")
    flat, layer_count = _flatten_logit_lens_residuals(residual_activations, token_indices)
    norm, norm_parameter, weight, bias = _logit_lens_readout(model)
    vocabulary_size = int(weight.shape[0])
    if top_k > vocabulary_size:
        raise ValueError("full-vocabulary logit-lens top-k exceeds the output vocabulary")
    if flat.shape[1] != weight.shape[1]:
        raise RuntimeError("residual width does not match the output embedding width")

    sparse_rows: list[tuple[VocabularyTopToken, ...]] = []
    with t.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start : start + batch_size]
            normalized = norm(batch.to(device=norm_parameter.device, dtype=norm_parameter.dtype))
            logits = functional.linear(
                normalized.to(device=weight.device, dtype=weight.dtype),
                weight,
                bias,
            ).to(dtype=t.float32)
            log_normalizer = t.logsumexp(logits, dim=-1, keepdim=True)
            top_logits, top_ids = logits.topk(top_k, dim=-1, largest=True, sorted=True)
            top_probabilities = (top_logits - log_normalizer).exp()
            for ids, probabilities in zip(
                top_ids.detach().cpu().tolist(),
                top_probabilities.detach().cpu().tolist(),
                strict=True,
            ):
                row = tuple(
                    (int(token_id), float(probability))
                    for token_id, probability in zip(ids, probabilities, strict=True)
                )
                if any(
                    not math.isfinite(probability) or not 0.0 <= probability <= 1.0
                    for _token_id, probability in row
                ):
                    raise RuntimeError("full-vocabulary logit lens produced an invalid probability")
                if any(row[index][1] < row[index + 1][1] for index in range(len(row) - 1)):
                    raise RuntimeError("full-vocabulary logit-lens top-k is not descending")
                if sum(probability for _token_id, probability in row) > 1.00001:
                    raise RuntimeError("full-vocabulary logit-lens displayed mass exceeds one")
                sparse_rows.append(row)
            del logits, log_normalizer, top_logits, top_ids, top_probabilities
    expected_rows = len(token_indices) * layer_count
    if len(sparse_rows) != expected_rows:
        raise RuntimeError("full-vocabulary logit lens produced an incomplete grid")
    return tuple(
        tuple(sparse_rows[token * layer_count + layer] for layer in range(layer_count))
        for token in range(len(token_indices))
    )


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


def _capture_decoder_inputs(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    batch_size: int,
) -> tuple[t.Tensor, ...]:
    """Capture each block input once at the exact patched-forward batch shape."""

    if not blocks or batch_size <= 0:
        raise ValueError("decoder input capture requires blocks and a positive batch size")
    captured: list[t.Tensor | None] = [None] * len(blocks)
    handles: list[Any] = []
    for layer, block in enumerate(blocks):

        def input_hook(
            _module: t.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            index: int = layer,
        ) -> None:
            hidden = _input_hidden(args, kwargs)
            if hidden.shape[0] != batch_size:
                raise RuntimeError("decoder input capture observed the wrong batch shape")
            if captured[index] is not None:
                raise RuntimeError("decoder block was unexpectedly invoked more than once")
            # Retaining the inference tensor preserves its values, device, dtype, and strides.
            captured[index] = hidden.detach()

        handles.append(block.register_forward_pre_hook(input_hook, with_kwargs=True))
    try:
        # This probability is discarded. ``logits_to_keep`` is applied only after the decoder;
        # it cannot affect the cached block inputs or any reported patch probability.
        _forward_probabilities_last_token(
            model,
            input_ids.expand(batch_size, -1),
            attention_mask.expand(batch_size, -1),
            candidate_ids,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("not every decoder layer produced a cached input")
    return tuple(cast(t.Tensor, value) for value in captured)


_MISSING_FORWARD = object()


@contextmanager
def _skip_unchanged_decoder_prefix(
    blocks: tuple[t.nn.Module, ...],
    target_layer: int,
    cached_inputs: tuple[t.Tensor, ...],
) -> Iterator[None]:
    """Skip blocks strictly upstream of a patch by injecting their exact cached result."""

    if len(cached_inputs) != len(blocks):
        raise ValueError("cached decoder input count must equal decoder block count")
    if not 0 <= target_layer < len(blocks):
        raise ValueError("target decoder layer is outside the model")
    if target_layer == 0:
        yield
        return

    prefix = blocks[:target_layer]
    original_forwards = tuple(block.__dict__.get("forward", _MISSING_FORWARD) for block in prefix)
    injected = cached_inputs[target_layer]

    def identity_forward(hidden_states: t.Tensor, *_args: Any, **_kwargs: Any) -> t.Tensor:
        return hidden_states

    def inject_forward(hidden_states: t.Tensor, *_args: Any, **_kwargs: Any) -> t.Tensor:
        if not isinstance(hidden_states, t.Tensor) or hidden_states.ndim != 3:
            raise RuntimeError("skipped decoder prefix did not receive a hidden-state tensor")
        if (
            injected.shape != hidden_states.shape
            or injected.dtype != hidden_states.dtype
            or injected.device != hidden_states.device
        ):
            raise RuntimeError("cached decoder input does not match the live forward")
        return injected

    try:
        for block in prefix[:-1]:
            block.forward = identity_forward  # type: ignore[method-assign]
        prefix[-1].forward = inject_forward  # type: ignore[method-assign]
        yield
    finally:
        for block, original in zip(prefix, original_forwards, strict=True):
            if original is _MISSING_FORWARD:
                del block.forward
            else:
                block.forward = original  # type: ignore[method-assign]


def _token_weight_patch_grid(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    candidate_ids: t.Tensor,
    projection_layers: tuple[tuple[TokenLoraProjection, ...], ...],
    positions: tuple[TokenPositionPair, ...],
    correct_choice_index: int,
    *,
    patch_batch_size: int = 8,
    progress_label: str,
    forward_probabilities: ProbabilityForward = _forward_probabilities,
    skip_unchanged_prefix: bool = False,
) -> tuple[tuple[float, ...], ...]:
    """Apply each donor block's LoRA update at one selected token per batch row."""

    if not projection_layers:
        raise ValueError("token-local weight patching requires decoder projection layers")
    if len(blocks) != len(projection_layers):
        raise ValueError("decoder block count must equal projection layer count")
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
    prefix_inputs_by_batch_size: dict[int, tuple[t.Tensor, ...]] = {}
    for layer, projections in enumerate(projection_layers):
        for start in range(0, len(positions), patch_batch_size):
            chunk = positions[start : start + patch_batch_size]
            recipient_positions = tuple(position.recipient_index for position in chunk)
            cached_inputs: tuple[t.Tensor, ...] | None = None
            if skip_unchanged_prefix and layer > 0:
                batch_size = len(chunk)
                if batch_size not in prefix_inputs_by_batch_size:
                    prefix_inputs_by_batch_size[batch_size] = _capture_decoder_inputs(
                        model,
                        blocks,
                        input_ids,
                        attention_mask,
                        candidate_ids,
                        batch_size,
                    )
                cached_inputs = prefix_inputs_by_batch_size[batch_size]
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
                if cached_inputs is None:
                    probabilities = forward_probabilities(
                        model,
                        input_ids.expand(len(chunk), -1),
                        attention_mask.expand(len(chunk), -1),
                        candidate_ids,
                    )
                else:
                    with _skip_unchanged_decoder_prefix(blocks, layer, cached_inputs):
                        probabilities = forward_probabilities(
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


def _capture_prompt_counterfactual_source_bank(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    residual_targets: tuple[PatchTarget, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    mode: PatchingMode,
    interface: PatchingInterface,
) -> PromptCounterfactualSourceBank:
    """Capture one counterfactual prompt bank using the donor checkpoint readout."""

    if not mode.uses_prompt_counterfactual:
        raise ValueError("counterfactual source capture requires a prompt mode")
    source_by_record: PromptCounterfactualSourceBank = {}
    patch_direction: str | None = None
    for record in records:
        counterfactual = _prompt_counterfactual_spec(record, mode)
        source_view, recipient_view = _prompt_counterfactual_views(
            processor,
            record,
            mode,
            counterfactual,
        )
        source_candidate_ids, _recipient_candidate_ids = _counterfactual_candidate_ids(
            processor,
            record,
            counterfactual,
        )
        source_activations, source_probabilities = _capture(
            model,
            targets,
            source_view.input_ids,
            source_view.attention_mask,
            source_candidate_ids,
        )
        if interface is PatchingInterface.RESID_POST:
            source_residual_activations = source_activations
        else:
            source_residual_activations, _source_residual_probabilities = _capture(
                model,
                residual_targets,
                source_view.input_ids,
                source_view.attention_mask,
                source_candidate_ids,
            )
        positions = reverse_token_position_pairs(
            source_view.anchor_index,
            recipient_view.anchor_index,
            source_view.stop_index,
            recipient_view.stop_index,
        )
        source_answer_logit_lens = _answer_label_logit_lens(
            model,
            source_residual_activations,
            tuple(position.source_index for position in positions),
            source_candidate_ids,
        )
        source_by_record[record.record_id] = PromptCounterfactualSourceRecord(
            spec=counterfactual,
            source_view=source_view,
            recipient_view=recipient_view,
            source_activations=source_activations,
            source_probabilities=source_probabilities,
            source_answer_logit_lens=source_answer_logit_lens,
        )
        if patch_direction is None:
            patch_direction = counterfactual.patch_direction
        elif patch_direction != counterfactual.patch_direction:
            raise AssertionError("prompt patch direction changed across records")
    if patch_direction is None:
        raise RuntimeError("prompt-counterfactual patching selected no records")
    return source_by_record


def _capture_activation_reference_banks(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    modes: tuple[PatchingMode, ...],
) -> tuple[ActivationReferenceBank, ...]:
    """Capture every source/recipient vector addressed by the selected prompt modes."""

    if not modes or any(not mode.supports_independent_checkpoint_donor for mode in modes):
        raise ValueError("activation-reference modes must be answer-label prompt controls")
    banks: list[ActivationReferenceBank] = []
    for record in records:
        clean_view = _prompt_patch_view(
            processor,
            record,
            record.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        clean_activations, _clean_probabilities = _capture(
            model,
            targets,
            clean_view.input_ids,
            clean_view.attention_mask,
            _candidate_ids(processor, record),
        )
        for mode in modes:
            counterfactual = _prompt_counterfactual_spec(record, mode)
            source_view, recipient_view = _prompt_counterfactual_views(
                processor,
                record,
                mode,
                counterfactual,
            )
            source_candidate_ids, _recipient_candidate_ids = _counterfactual_candidate_ids(
                processor,
                record,
                counterfactual,
            )
            source_activations, _source_probabilities = _capture(
                model,
                targets,
                source_view.input_ids,
                source_view.attention_mask,
                source_candidate_ids,
            )
            positions = reverse_token_position_pairs(
                source_view.anchor_index,
                recipient_view.anchor_index,
                source_view.stop_index,
                recipient_view.stop_index,
            )
            banks.append(
                ActivationReferenceBank(
                    mode=mode,
                    function_id=record.function_id,
                    positions=positions,
                    source_activations=source_activations,
                    recipient_activations=clean_activations,
                )
            )
    return tuple(banks)


def _top_cosine_examples(
    reference_vectors: t.Tensor,
    candidate_vectors: tuple[t.Tensor, ...],
    *,
    top_k: int,
    device: str = "cuda",
    reference_batch_size: int = 64,
) -> list[list[dict[str, object]]]:
    """Return top distinct examples, using each example's best-matching token."""

    if reference_vectors.ndim != 2 or reference_vectors.shape[0] == 0:
        raise ValueError("activation-example search requires a non-empty reference matrix")
    if not candidate_vectors or any(
        value.ndim != 2 or value.shape[0] == 0 for value in candidate_vectors
    ):
        raise ValueError("activation-example search requires non-empty candidate sequences")
    width = reference_vectors.shape[1]
    if any(value.shape[1] != width for value in candidate_vectors):
        raise ValueError("reference and candidate activation widths must match")
    if top_k <= 0 or reference_batch_size <= 0:
        raise ValueError("top-k and reference batch size must be positive")

    candidate_starts: list[int] = []
    cursor = 0
    for value in candidate_vectors:
        candidate_starts.append(cursor)
        cursor += value.shape[0]
    candidate_matrix = functional.normalize(
        t.cat(candidate_vectors, dim=0).to(device=device, dtype=t.float32),
        dim=-1,
    )
    selected_count = min(top_k, len(candidate_vectors))
    output: list[list[dict[str, object]]] = []
    for start in range(0, reference_vectors.shape[0], reference_batch_size):
        references = functional.normalize(
            reference_vectors[start : start + reference_batch_size].to(
                device=device,
                dtype=t.float32,
            ),
            dim=-1,
        )
        # Float32 normalization and matmul can overshoot the mathematical
        # cosine range by a few ulps (for example, 1.0000001 for identical
        # vectors).  Persist the metric's exact [-1, 1] contract.
        cosine = (references @ candidate_matrix.transpose(0, 1)).clamp_(-1.0, 1.0)
        best_scores: list[t.Tensor] = []
        best_tokens: list[t.Tensor] = []
        for example_index, example_start in enumerate(candidate_starts):
            example_end = (
                candidate_starts[example_index + 1]
                if example_index + 1 < len(candidate_starts)
                else candidate_matrix.shape[0]
            )
            score, token_index = cosine[:, example_start:example_end].max(dim=1)
            best_scores.append(score)
            best_tokens.append(token_index)
        score_matrix = t.stack(best_scores, dim=1)
        token_matrix = t.stack(best_tokens, dim=1)
        selected_scores, selected_examples = score_matrix.topk(selected_count, dim=1)
        selected_tokens = token_matrix.gather(1, selected_examples)
        for row in range(references.shape[0]):
            matches: list[dict[str, object]] = []
            for rank in range(selected_count):
                matches.append(
                    {
                        "example_index": int(selected_examples[row, rank].item()),
                        "token_index": int(selected_tokens[row, rank].item()),
                        "cosine_similarity": float(selected_scores[row, rank].item()),
                    }
                )
            output.append(matches)
        del cosine, score_matrix, token_matrix, selected_scores, selected_examples, selected_tokens
    del candidate_matrix
    return output


def _activation_example_output_path(
    root: Path,
    run: RunKey,
    interface: PatchingInterface,
    checkpoint_step: int,
    candidate_source: ActivationExampleSource = ActivationExampleSource.EXPERIMENT,
) -> Path:
    if interface.patches_weights:
        raise ValueError("weight patches do not expose one activation reference vector")
    base = run_dir(root, run) / "activation_examples" / "sequence_end" / interface.value
    if candidate_source is not ActivationExampleSource.EXPERIMENT:
        base /= candidate_source.value
    return base / f"checkpoint_{checkpoint_label(checkpoint_step)}.json"


def _vocabulary_logit_lens_output_path(
    root: Path,
    run: RunKey,
    checkpoint_step: int,
) -> Path:
    return (
        run_dir(root, run)
        / "vocabulary_logit_lens"
        / "sequence_end"
        / f"checkpoint_{checkpoint_label(checkpoint_step)}.json"
    )


def _vocabulary_logit_lens_resume_artifact(
    path: Path,
    run: RunKey,
    checkpoint_step: int,
    expected_modes: tuple[PatchingMode, ...],
) -> tuple[dict[str, object] | None, tuple[PatchingMode, ...]]:
    """Validate a complete/legacy sidecar and return only its missing active sources."""

    if not path.is_file():
        return None, expected_modes
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"vocabulary logit-lens artifact must be an object: {path}")
    artifact = cast(dict[str, object], raw)
    if artifact.get("checkpoint_step") != checkpoint_step:
        raise ValueError(f"vocabulary logit-lens checkpoint mismatch: {path}")
    raw_run = artifact.get("run")
    if not isinstance(raw_run, dict):
        raise TypeError(f"vocabulary logit-lens run metadata must be an object: {path}")
    expected_run: dict[str, object] = {
        "model": run.model,
        "condition": run.condition.value,
        "seed": run.seed,
        "effective_batch_size": run.effective_batch_size,
        "lora_rank": run.lora_rank,
    }
    for key, expected in expected_run.items():
        if raw_run.get(key) != expected:
            raise ValueError(f"vocabulary logit-lens run.{key} mismatch: {path}")

    raw_modes = artifact.get("modes")
    if (
        not isinstance(raw_modes, list)
        or not raw_modes
        or any(not isinstance(value, str) for value in raw_modes)
    ):
        raise TypeError(f"vocabulary logit-lens modes must be a non-empty string array: {path}")
    try:
        observed_modes = tuple(PatchingMode(cast(str, value)) for value in raw_modes)
    except ValueError as error:
        raise ValueError(f"vocabulary logit-lens artifact has an unknown mode: {path}") from error
    observed_set = set(observed_modes)
    if len(observed_set) != len(observed_modes) or observed_modes != tuple(
        mode for mode in expected_modes if mode in observed_set
    ):
        raise ValueError(
            f"vocabulary logit-lens modes must be an ordered subset of active sources: {path}"
        )

    token_labels = artifact.get("token_labels")
    if not isinstance(token_labels, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in token_labels.items()
    ):
        raise TypeError(f"vocabulary logit-lens token labels are malformed: {path}")
    lens = artifact.get("lens")
    if not isinstance(lens, dict) or lens.get("kind") != "full_vocabulary_top_k":
        raise ValueError(f"vocabulary logit-lens metadata is malformed: {path}")
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list):
        raise TypeError(f"vocabulary logit-lens records must be an array: {path}")
    seen_functions: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise TypeError(f"vocabulary logit-lens record {index} must be an object: {path}")
        function_id = raw_record.get("function_id")
        clean = raw_record.get("clean")
        sources = raw_record.get("sources")
        if (
            not isinstance(function_id, str)
            or function_id in seen_functions
            or not isinstance(clean, dict)
            or not isinstance(sources, dict)
            or set(sources) != {mode.value for mode in observed_modes}
            or any(not isinstance(value, dict) for value in sources.values())
        ):
            raise ValueError(f"vocabulary logit-lens record {index} is inconsistent: {path}")
        seen_functions.add(function_id)
    if seen_functions != set(FUNCTION_BY_ID):
        raise ValueError(
            f"vocabulary logit-lens artifact lacks the registered function set: {path}"
        )
    missing = tuple(mode for mode in expected_modes if mode not in observed_set)
    return artifact, missing


def _vocabulary_logit_lens_side_payload(
    view: PromptPatchView,
    token_indices: tuple[int, ...],
    lens: VocabularyLogitLens,
    tokenizer: Any,
    token_labels: dict[str, str],
) -> dict[str, object]:
    """Serialize one reverse token axis without repeating decoded top-token strings."""

    if not token_indices or token_indices[0] != view.anchor_index:
        raise ValueError("vocabulary logit-lens axis must start at the sequence-end anchor")
    expected_indices = tuple(range(token_indices[0], token_indices[-1] - 1, -1))
    if token_indices != expected_indices:
        raise ValueError("vocabulary logit-lens axis must be reverse-contiguous")
    if len(lens) != len(token_indices) or not lens:
        raise ValueError("vocabulary logit-lens grid must match its token axis")
    layer_count = len(lens[0])
    if layer_count <= 0 or any(len(token_rows) != layer_count for token_rows in lens):
        raise ValueError("vocabulary logit-lens grid has an inconsistent layer axis")
    missing_ids = tuple(
        dict.fromkeys(
            token_id
            for token_rows in lens
            for top_tokens in token_rows
            for token_id, _probability in top_tokens
            if str(token_id) not in token_labels
        )
    )
    token_labels.update(
        {
            str(token_id): label
            for token_id, label in _vocabulary_token_labels(tokenizer, missing_ids).items()
        }
    )
    for token_rows in lens:
        for top_tokens in token_rows:
            if len(top_tokens) != VOCABULARY_LOGIT_LENS_TOP_K:
                raise ValueError("vocabulary logit-lens grid has an inconsistent top-k axis")
            seen: set[int] = set()
            previous = math.inf
            for token_id, probability in top_tokens:
                if token_id in seen:
                    raise ValueError("vocabulary logit-lens top-k repeats a token ID")
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("vocabulary logit-lens top-k contains an invalid probability")
                if probability > previous + 1e-8:
                    raise ValueError("vocabulary logit-lens top-k must be descending")
                key = str(token_id)
                if key not in token_labels:
                    raise RuntimeError("vocabulary token-label batch omitted a top-token ID")
                seen.add(token_id)
                previous = probability
    return {
        "position_count": len(token_indices),
        "token_indices": token_indices,
        "token_ids": tuple(view.token_ids[index] for index in token_indices),
        "top_tokens": lens,
    }


def _serialize_activation_examples(
    candidates: tuple[ActivationExampleView, ...],
    reference_banks: tuple[ActivationReferenceBank, ...],
    *,
    layer_count: int,
) -> list[dict[str, object]]:
    """Compute cosine neighbors for every token/layer reference in one checkpoint."""

    if not candidates or not reference_banks:
        raise ValueError("activation-example serialization requires candidates and references")
    if any(len(candidate.activations) != layer_count for candidate in candidates):
        raise ValueError("candidate activation bank has the wrong decoder layer count")
    if any(
        len(bank.source_activations) != layer_count
        or len(bank.recipient_activations) != layer_count
        for bank in reference_banks
    ):
        raise ValueError("reference activation bank has the wrong decoder layer count")

    serialized: list[dict[str, object]] = [
        {
            "mode": bank.mode.value,
            "function_id": bank.function_id,
            "position_count": len(bank.positions),
            "source_neighbors": [[None] * layer_count for _ in bank.positions],
            "recipient_neighbors": [[None] * layer_count for _ in bank.positions],
        }
        for bank in reference_banks
    ]
    descriptors: list[tuple[int, str, int]] = []
    for bank_index, bank in enumerate(reference_banks):
        for side in ("source", "recipient"):
            descriptors.extend(
                (bank_index, side, token_index) for token_index in range(len(bank.positions))
            )
    for layer in range(layer_count):
        references = t.stack(
            tuple(
                (
                    reference_banks[bank_index].source_activations[layer][
                        reference_banks[bank_index].positions[token_index].source_index
                    ]
                    if side == "source"
                    else reference_banks[bank_index].recipient_activations[layer][
                        reference_banks[bank_index].positions[token_index].recipient_index
                    ]
                )
                for bank_index, side, token_index in descriptors
            )
        )
        matches = _top_cosine_examples(
            references,
            tuple(candidate.activations[layer] for candidate in candidates),
            top_k=ACTIVATION_EXAMPLE_TOP_K,
        )
        for descriptor, top_matches in zip(descriptors, matches, strict=True):
            bank_index, side, token_index = descriptor
            key = f"{side}_neighbors"
            rows = cast(list[list[object]], serialized[bank_index][key])
            rows[token_index][layer] = top_matches
        del references, matches
        t.cuda.empty_cache()
    for record in serialized:
        for key in ("source_neighbors", "recipient_neighbors"):
            rows = cast(list[list[object]], record[key])
            if any(value is None for row in rows for value in row):
                raise RuntimeError("activation-example neighbor grid is incomplete")
    return serialized


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

    source_choice_function_ids: tuple[str, ...] | None = record.choice_function_ids
    source_choice_texts: tuple[str, ...] | None = None
    source_question_id: str | None = None
    source_question: str | None = None
    source_format: str | None = None
    source_label_relation: str | None = None
    source_context_id: str | None = None
    source_context: str | None = None
    recipient_correct_choice_index = record.choice_function_ids.index(record.function_id)
    recipient_function_id = record.function_id
    if mode.uses_prompt_counterfactual:
        spec = _prompt_counterfactual_spec(record, mode)
        source_function_id = spec.source_function_id
        source_correct_choice_index = spec.source_correct_choice_index
        recipient_function_id = spec.recipient_function_id
        recipient_correct_choice_index = spec.recipient_correct_choice_index
        source_choice_function_ids = spec.source_choice_function_ids
        source_choice_texts = spec.source_choice_texts
        source_question_id = spec.source_question_id
        source_question = spec.source_question
        source_format = spec.source_format
        source_label_relation = spec.source_label_relation
        source_context_id = spec.source_context_id
        source_context = spec.source_context
        source_view, recipient_view = _prompt_counterfactual_views(
            processor,
            record,
            mode,
            spec,
            device="cpu",
        )
    else:
        source_function_id = record.function_id
        source_correct_choice_index = recipient_correct_choice_index
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
    metadata: dict[str, object] = {
        "source_function_id": source_function_id,
        "recipient_function_id": recipient_function_id,
        "source_correct_choice_index": source_correct_choice_index,
        "recipient_correct_choice_index": recipient_correct_choice_index,
        "recipient_choice_function_ids": record.choice_function_ids,
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
    if source_choice_function_ids is not None:
        metadata["source_choice_function_ids"] = source_choice_function_ids
    if source_choice_texts is not None:
        metadata["source_choice_texts"] = source_choice_texts
    if source_question_id is not None:
        metadata["source_question_id"] = source_question_id
    if source_question is not None:
        metadata["source_question"] = source_question
    if source_format is not None:
        metadata["source_format"] = source_format
    if source_label_relation is not None:
        metadata["source_label_relation"] = source_label_relation
    if source_context_id is not None:
        metadata["source_context_id"] = source_context_id
    if source_context is not None:
        metadata["source_context"] = source_context
    return metadata


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
    grid: ProbabilityGrid,
    mode: PatchingMode,
    *,
    source_target_grid: ProbabilityGrid | None = None,
    counterfactual: PromptCounterfactualSpec | None = None,
    answer_logit_lens: dict[str, object] | None = None,
) -> dict[str, object]:
    correct_choice = record.choice_function_ids.index(record.function_id)
    recipient_target = float(recipient[correct_choice].item())
    cells: list[dict[str, object]] = []
    if source_target_grid is not None and counterfactual is None:
        raise ValueError("source-target grid requires prompt-counterfactual metadata")
    if source_target_grid is not None and len(source_target_grid) != len(grid):
        raise ValueError("source-target and recipient-target grids must share token rows")
    source_target_recipient = (
        None
        if counterfactual is None or counterfactual.source_correct_choice_index is None
        else float(recipient[counterfactual.source_correct_choice_index].item())
    )
    for token_index, (position, row) in enumerate(zip(positions, grid, strict=True)):
        source_target_row = None if source_target_grid is None else source_target_grid[token_index]
        if source_target_row is not None and len(source_target_row) != len(row):
            raise ValueError("source-target and recipient-target grids must share layers")
        for layer, probability in enumerate(row):
            cell: dict[str, object] = {
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
            if source_target_row is not None and source_target_recipient is not None:
                source_target_probability = source_target_row[layer]
                cell["source_target_probability"] = source_target_probability
                cell["delta_source_target_from_recipient"] = (
                    source_target_probability - source_target_recipient
                )
            cells.append(cell)
    serialized: dict[str, object] = {
        "function_id": record.function_id,
        "source_function_id": (
            counterfactual.source_function_id
            if counterfactual is not None
            else DERANGEMENT[record.function_id]
            if mode is PatchingMode.ACROSS_SAMPLE
            else record.function_id
        ),
        "recipient_function_id": (
            counterfactual.recipient_function_id
            if counterfactual is not None
            else record.function_id
        ),
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
        if counterfactual.source_correct_choice_index is not None:
            serialized["source_correct_choice_index"] = counterfactual.source_correct_choice_index
        serialized["recipient_correct_choice_index"] = counterfactual.recipient_correct_choice_index
        if counterfactual.source_choice_function_ids is not None:
            serialized["source_choice_function_ids"] = counterfactual.source_choice_function_ids
        if counterfactual.source_choice_texts is not None:
            serialized["source_choice_texts"] = counterfactual.source_choice_texts
        if counterfactual.source_question_id is not None:
            serialized["source_question_id"] = counterfactual.source_question_id
        if counterfactual.source_question is not None:
            serialized["source_question"] = counterfactual.source_question
        if counterfactual.source_format is not None:
            serialized["source_format"] = counterfactual.source_format
        if counterfactual.source_label_relation is not None:
            serialized["source_label_relation"] = counterfactual.source_label_relation
        if counterfactual.source_context_id is not None:
            serialized["source_context_id"] = counterfactual.source_context_id
        if counterfactual.source_context is not None:
            serialized["source_context"] = counterfactual.source_context
    if answer_logit_lens is not None:
        serialized["answer_logit_lens"] = answer_logit_lens
    return serialized


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
    *,
    patch_batch_size: int = 8,
    forward_probabilities: ProbabilityForward = _forward_probabilities,
    skip_unchanged_prefix: bool = False,
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
            blocks,
            recipient_view.input_ids,
            recipient_view.attention_mask,
            candidate_ids,
            projection_layers,
            positions,
            correct_choice,
            patch_batch_size=patch_batch_size,
            progress_label=f"function={record.function_id} ({record_index}/{len(records)})",
            forward_probabilities=forward_probabilities,
            skip_unchanged_prefix=skip_unchanged_prefix,
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
    *,
    counterfactual: PromptCounterfactualSpec | None = None,
    recipient_probabilities: t.Tensor | None = None,
    source_residual_activations: tuple[t.Tensor, ...] | None = None,
    recipient_residual_activations: tuple[t.Tensor, ...] | None = None,
    precomputed_answer_logit_lens: dict[str, object] | None = None,
    patch_batch_size: int = 8,
) -> dict[str, object]:
    positions = reverse_token_position_pairs(
        source_view.anchor_index,
        recipient_view.anchor_index,
        source_view.stop_index,
        recipient_view.stop_index,
    )
    candidate_ids = _candidate_ids(processor, record)
    if recipient_probabilities is None:
        recipient_probabilities = _forward_probabilities(
            model,
            recipient_view.input_ids,
            recipient_view.attention_mask,
            candidate_ids,
        )[0]
    correct_choice = record.choice_function_ids.index(record.function_id)
    source_target_index = (
        counterfactual.source_correct_choice_index
        if (
            counterfactual is not None
            and mode
            not in {
                PatchingMode.ACROSS_SAMPLE,
                PatchingMode.REVERSE_ACROSS_SAMPLE,
            }
            and counterfactual.source_correct_choice_index is not None
        )
        else None
    )
    grid, source_target_grid = _patch_grid(
        model,
        targets,
        recipient_view.input_ids,
        recipient_view.attention_mask,
        candidate_ids,
        source_activations,
        positions,
        correct_choice,
        source_target_index,
        patch_batch_size=patch_batch_size,
    )
    if (source_residual_activations is None) != (recipient_residual_activations is None):
        raise ValueError("logit lens requires both source and recipient residual activations")
    if precomputed_answer_logit_lens is not None and source_residual_activations is not None:
        raise ValueError("logit lens must be precomputed or derived in one model, not both")
    answer_logit_lens = precomputed_answer_logit_lens
    if source_residual_activations is not None and recipient_residual_activations is not None:
        source_lens = _answer_label_logit_lens(
            model,
            source_residual_activations,
            tuple(position.source_index for position in positions),
            candidate_ids,
        )
        recipient_lens = _answer_label_logit_lens(
            model,
            recipient_residual_activations,
            tuple(position.recipient_index for position in positions),
            candidate_ids,
        )
        answer_logit_lens = {
            "kind": "five_way_answer_label",
            "labels": tuple("ABCDE"),
            "normalization": "softmax over A-E after final norm and selected unembedding rows",
            "display_top_p": ANSWER_LOGIT_LENS_TOP_P,
            "residual_boundary": "decoder block output before final model normalization",
            "source_probabilities": source_lens,
            "recipient_probabilities": recipient_lens,
        }
    return _serialize_grid(
        record,
        source_probabilities,
        recipient_probabilities,
        positions,
        source_view,
        recipient_view,
        grid,
        mode,
        source_target_grid=source_target_grid,
        counterfactual=counterfactual,
        answer_logit_lens=answer_logit_lens,
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


def _patch_prompt_counterfactual_source_bank(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    residual_targets: tuple[PatchTarget, ...],
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    mode: PatchingMode,
    source_by_record: PromptCounterfactualSourceBank,
    *,
    source_step: int,
    recipient_step: int,
) -> tuple[list[dict[str, object]], str]:
    """Patch donor-prompt states while keeping each side's own unpatched readout."""

    serialized: list[dict[str, object]] = []
    patch_direction: str | None = None
    for record in records:
        source = source_by_record[record.record_id]
        _source_candidate_ids, recipient_candidate_ids = _counterfactual_candidate_ids(
            processor,
            record,
            source.spec,
        )
        recipient_residual_activations, recipient_probabilities = _capture(
            model,
            residual_targets,
            source.recipient_view.input_ids,
            source.recipient_view.attention_mask,
            recipient_candidate_ids,
        )
        positions = reverse_token_position_pairs(
            source.source_view.anchor_index,
            source.recipient_view.anchor_index,
            source.source_view.stop_index,
            source.recipient_view.stop_index,
        )
        recipient_answer_logit_lens = _answer_label_logit_lens(
            model,
            recipient_residual_activations,
            tuple(position.recipient_index for position in positions),
            recipient_candidate_ids,
        )
        answer_logit_lens: dict[str, object] = {
            "kind": "five_way_answer_label",
            "labels": tuple("ABCDE"),
            "normalization": "softmax over A-E after each side's final norm and selected unembedding rows",
            "display_top_p": ANSWER_LOGIT_LENS_TOP_P,
            "residual_boundary": "decoder block output before final model normalization",
            "source_checkpoint_step": source_step,
            "recipient_checkpoint_step": recipient_step,
            "source_probabilities": source.source_answer_logit_lens,
            "recipient_probabilities": recipient_answer_logit_lens,
        }
        serialized.append(
            _patch_record(
                model,
                targets,
                processor,
                record,
                mode,
                source.source_view,
                source.recipient_view,
                source.source_activations,
                source.source_probabilities,
                counterfactual=source.spec,
                recipient_probabilities=recipient_probabilities,
                precomputed_answer_logit_lens=answer_logit_lens,
            )
        )
        if patch_direction is None:
            patch_direction = source.spec.patch_direction
        elif patch_direction != source.spec.patch_direction:
            raise AssertionError("prompt patch direction changed across records")
    if patch_direction is None:
        raise RuntimeError("prompt-counterfactual patching selected no records")
    return serialized, patch_direction


def _run_prompt_counterfactual_pair(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    plan: PatchingPlan,
    donor_step: int,
) -> None:
    """Run one prompt-counterfactual x checkpoint cell and write it atomically."""

    if not plan.mode.uses_prompt_counterfactual or plan.interface.patches_weights:
        raise ValueError("prompt-counterfactual pair requires an activation interface")
    donor_model = _load_checkpoint_model(root, run, spec, donor_step)
    donor_blocks = resolve_decoder_blocks(donor_model, spec)
    donor_targets = _resolve_patch_targets(donor_blocks, plan.interface)
    donor_residual_targets = _resolve_patch_targets(
        donor_blocks,
        PatchingInterface.RESID_POST,
    )
    try:
        source_by_record = _capture_prompt_counterfactual_source_bank(
            donor_model,
            donor_targets,
            donor_residual_targets,
            processor,
            records,
            plan.mode,
            plan.interface,
        )
    finally:
        _release_model(donor_model)

    recipient_model = _load_checkpoint_model(root, run, spec, plan.recipient_step)
    recipient_blocks = resolve_decoder_blocks(recipient_model, spec)
    recipient_targets = _resolve_patch_targets(recipient_blocks, plan.interface)
    recipient_residual_targets = _resolve_patch_targets(
        recipient_blocks,
        PatchingInterface.RESID_POST,
    )
    try:
        serialized, patch_direction = _patch_prompt_counterfactual_source_bank(
            recipient_model,
            recipient_targets,
            recipient_residual_targets,
            processor,
            records,
            plan.mode,
            source_by_record,
            source_step=donor_step,
            recipient_step=plan.recipient_step,
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
            "patch_direction": patch_direction,
            "checkpoint_relation": (
                "same_checkpoint" if donor_step == plan.recipient_step else "cross_checkpoint"
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
    token_weight_runtime: TokenWeightRuntime,
    token_weight_patch_batch_size: int,
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
                    patch_batch_size=token_weight_patch_batch_size,
                    forward_probabilities=_token_weight_probability_forward(token_weight_runtime),
                    skip_unchanged_prefix=token_weight_runtime is TokenWeightRuntime.OPTIMIZED,
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
    token_weight_runtime: TokenWeightRuntime = TokenWeightRuntime.REFERENCE,
    token_weight_patch_batch_size: int = 8,
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
        or any(mode.uses_prompt_counterfactual for mode in modes)
    ):
        raise ValueError("temporal matrix modes must be unique checkpoint-transfer modes")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ValueError("temporal matrix shuffle seed must be non-negative")
    if token_weight_patch_batch_size <= 0:
        raise ValueError("token-weight patch batch size must be positive")
    if token_weight_patch_batch_size != 8:
        raise ValueError("production token-weight runtimes have a fixed batch size of 8")

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
            token_weight_runtime,
            token_weight_patch_batch_size,
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


def run_prompt_counterfactual_patching_matrix(
    root: Path,
    run: RunKey,
    recipient_steps: tuple[int, ...],
    donor_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
    interface: PatchingInterface,
    *,
    shuffle_seed: int | None = None,
    allow_provisional_model: bool = False,
) -> None:
    """Fill a prompt-counterfactual x checkpoint plane, including its diagonals."""

    if not t.cuda.is_available():
        raise RuntimeError("checkpoint patching requires CUDA")
    for name, steps in (("recipient", recipient_steps), ("donor", donor_steps)):
        if tuple(sorted(set(steps))) != steps or any(
            step not in CHECKPOINT_STEPS for step in steps
        ):
            raise ValueError(
                f"prompt-counterfactual {name} steps must be unique, increasing checkpoints"
            )
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(not mode.supports_independent_checkpoint_donor for mode in modes)
    ):
        raise ValueError(
            "prompt-counterfactual checkpoint modes must be unique answer-label controls"
        )
    if interface.patches_weights:
        raise ValueError("prompt-counterfactual checkpoint matrices are activation-only")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ValueError("prompt-counterfactual matrix shuffle seed must be non-negative")

    scheduled_pairs = [
        (recipient_step, donor_step, mode)
        for recipient_step in recipient_steps
        for donor_step in donor_steps
        for mode in modes
    ]
    if shuffle_seed is not None:
        scheduled_pairs = _seeded_priority_temporal_order(scheduled_pairs, shuffle_seed)

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
            f"[prompt-checkpoint-matrix] {run.model}/{run.condition.value} "
            f"{interface.value} skipped {skipped} existing artifact(s)",
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
            f"[prompt-checkpoint-matrix] priority-shuffled {len(pending_pairs)} "
            f"missing cells with seed {shuffle_seed} ({count_summary})",
            flush=True,
        )

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    records = _selected_records(run.seed)
    for recipient_step, donor_step, mode in pending_pairs:
        plan = PatchingPlan(
            mode=mode,
            recipient_step=recipient_step,
            donor_steps=(donor_step,),
            interface=interface,
        )
        _run_prompt_counterfactual_pair(
            root,
            run,
            spec,
            processor,
            records,
            plan,
            donor_step,
        )


def run_activation_example_atlas(
    root: Path,
    run: RunKey,
    checkpoint_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
    interface: PatchingInterface,
    *,
    candidate_source: ActivationExampleSource = ActivationExampleSource.EXPERIMENT,
    allow_provisional_model: bool = False,
) -> None:
    """Measure cell-addressable activation neighbors at each selected checkpoint."""

    if not t.cuda.is_available():
        raise RuntimeError("activation-example analysis requires CUDA")
    if tuple(sorted(set(checkpoint_steps))) != checkpoint_steps or any(
        step not in CHECKPOINT_STEPS for step in checkpoint_steps
    ):
        raise ValueError("activation-example checkpoints must be unique and registered")
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(not mode.supports_independent_checkpoint_donor for mode in modes)
    ):
        raise ValueError("activation-example modes must be unique answer-label controls")
    if interface.patches_weights:
        raise ValueError("weight-patching cells do not define activation reference vectors")

    pending = tuple(
        step
        for step in checkpoint_steps
        if not _activation_example_output_path(
            root,
            run,
            interface,
            step,
            candidate_source,
        ).is_file()
    )
    skipped = len(checkpoint_steps) - len(pending)
    if skipped:
        print(
            f"[activation-examples] {run.model}/{run.condition.value} {interface.value} "
            f"{candidate_source.value} skipped {skipped} existing checkpoint artifact(s)",
            flush=True,
        )
    if not pending:
        return

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    records = _selected_records(run.seed)
    candidate_prompts = (
        ()
        if candidate_source is ActivationExampleSource.FINEWEB
        else build_activation_example_source_prompts(candidate_source)
    )
    fineweb_documents = (
        load_fineweb_activation_documents(root)
        if candidate_source is ActivationExampleSource.FINEWEB
        else ()
    )
    for checkpoint_step in pending:
        model = _load_checkpoint_model(root, run, spec, checkpoint_step)
        blocks = resolve_decoder_blocks(model, spec)
        targets = _resolve_patch_targets(blocks, interface)
        candidates: tuple[ActivationExampleView, ...] = ()
        reference_banks: tuple[ActivationReferenceBank, ...] = ()
        try:
            candidate_ids = _candidate_ids(processor, records[0])
            if candidate_source is not ActivationExampleSource.FINEWEB:
                candidates = tuple(
                    _capture_activation_example(
                        model,
                        targets,
                        processor,
                        prompt,
                        candidate_ids,
                    )
                    for prompt in candidate_prompts
                )
                candidate_corpus = activation_example_corpus_metadata(
                    candidate_source,
                    candidate_prompts,
                )
            else:
                candidates = tuple(
                    _capture_fineweb_activation_example(
                        model,
                        targets,
                        processor,
                        document,
                        candidate_ids,
                    )
                    for document in fineweb_documents
                )
                candidate_corpus = {
                    "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
                    "prompt_count": len(candidates),
                    "categories": ["fineweb_pretraining"],
                    "description": (
                        "Deterministic random documents from the FineWeb sample-10BT "
                        "pretraining corpus"
                    ),
                    "input_format": (
                        "raw document prefix with tokenizer-native special tokens; no chat template"
                    ),
                    "max_tokens_per_document": FINEWEB_ACTIVATION_MAX_TOKENS,
                    "dataset": FINEWEB_DATASET_ID,
                    "revision": FINEWEB_DATASET_REVISION,
                    "config": FINEWEB_DATASET_CONFIG,
                    "split": FINEWEB_DATASET_SPLIT,
                }
            reference_banks = _capture_activation_reference_banks(
                model,
                targets,
                processor,
                records,
                modes,
            )
            serialized = _serialize_activation_examples(
                candidates,
                reference_banks,
                layer_count=spec.layer_count,
            )
            output = _activation_example_output_path(
                root,
                run,
                interface,
                checkpoint_step,
                candidate_source,
            )
            write_json(
                output,
                {
                    "model": spec,
                    "run": run,
                    "interface": interface.value,
                    "checkpoint_step": checkpoint_step,
                    "candidate_source": candidate_source.value,
                    "similarity": {
                        "metric": ACTIVATION_EXAMPLE_METRIC,
                        "ranking_unit": "distinct prompt using its maximum token similarity",
                        "top_k": ACTIVATION_EXAMPLE_TOP_K,
                        "reference": "selected source or recipient vector at one token and layer",
                    },
                    "candidate_corpus": candidate_corpus,
                    "candidates": [
                        {
                            "example_id": candidate.example_id,
                            "category": candidate.category,
                            "target": candidate.target,
                            "rendered_prompt": candidate.rendered_prompt,
                            "token_ids": candidate.token_ids,
                            "token_labels": candidate.token_labels,
                            **(
                                {"provenance": candidate.provenance}
                                if candidate.provenance is not None
                                else {}
                            ),
                        }
                        for candidate in candidates
                    ],
                    "records": serialized,
                },
            )
            print(
                f"[activation-examples] {run.model}/{run.condition.value} "
                f"{interface.value}/{candidate_source.value} "
                f"step={checkpoint_step} -> {output}",
                flush=True,
            )
        finally:
            _release_model(model)
            del candidates, reference_banks
            gc.collect()


def run_vocabulary_logit_lens_atlas(
    root: Path,
    run: RunKey,
    checkpoint_steps: tuple[int, ...],
    modes: tuple[PatchingMode, ...],
    *,
    allow_provisional_model: bool = False,
) -> None:
    """Measure reusable full-vocabulary residual lenses once per checkpoint."""

    if not t.cuda.is_available():
        raise RuntimeError("full-vocabulary logit-lens analysis requires CUDA")
    if (
        not checkpoint_steps
        or len(set(checkpoint_steps)) != len(checkpoint_steps)
        or any(step not in CHECKPOINT_STEPS for step in checkpoint_steps)
    ):
        raise ValueError("vocabulary logit-lens checkpoints must be unique and registered")
    if modes != VOCABULARY_LOGIT_LENS_MODES:
        raise ValueError("vocabulary logit-lens artifacts require every prompt source in order")

    resume_states = {
        step: _vocabulary_logit_lens_resume_artifact(
            _vocabulary_logit_lens_output_path(root, run, step),
            run,
            step,
            modes,
        )
        for step in checkpoint_steps
    }
    pending = tuple(step for step in checkpoint_steps if resume_states[step][1])
    skipped = len(checkpoint_steps) - len(pending)
    if skipped:
        print(
            f"[vocabulary-logit-lens] {run.model}/{run.condition.value} "
            f"skipped {skipped} existing checkpoint artifact(s)",
            flush=True,
        )
    if not pending:
        return

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    tokenizer = tokenizer_for(processor)
    records = _selected_records(run.seed)
    for checkpoint_step in pending:
        existing_artifact, missing_modes = resume_states[checkpoint_step]
        existing_records: dict[str, dict[str, object]] = {}
        if existing_artifact is not None:
            raw_records = cast(list[object], existing_artifact["records"])
            existing_records = {
                cast(str, cast(dict[str, object], raw_record)["function_id"]): cast(
                    dict[str, object], raw_record
                )
                for raw_record in raw_records
            }
            token_labels = cast(
                dict[str, str],
                dict(cast(dict[str, object], existing_artifact["token_labels"])),
            )
            print(
                f"[vocabulary-logit-lens] {run.model}/{run.condition.value} "
                f"step={checkpoint_step} extending {len(missing_modes)} missing source(s)",
                flush=True,
            )
        else:
            token_labels = {}
        model = _load_checkpoint_model(root, run, spec, checkpoint_step)
        blocks = resolve_decoder_blocks(model, spec)
        residual_targets = _resolve_patch_targets(blocks, PatchingInterface.RESID_POST)
        serialized: list[dict[str, object]] = []
        try:
            _norm, _norm_parameter, output_weight, _bias = _logit_lens_readout(model)
            vocabulary_size = int(output_weight.shape[0])
            if existing_artifact is not None:
                lens_metadata = cast(dict[str, object], existing_artifact["lens"])
                if (
                    lens_metadata.get("top_k") != VOCABULARY_LOGIT_LENS_TOP_K
                    or lens_metadata.get("vocabulary_size") != vocabulary_size
                ):
                    raise ValueError(
                        "existing vocabulary logit-lens readout dimensions disagree with model"
                    )
            for record in records:
                clean_view = _prompt_patch_view(
                    processor,
                    record,
                    record.messages,
                    FUNCTION_BY_ID[record.function_id].alias,
                    stop_at_sequence_start=True,
                )
                existing_record = existing_records.get(record.function_id)
                if existing_record is None:
                    recipient_candidate_ids = _candidate_ids(processor, record)
                    clean_activations, _clean_probabilities = _capture(
                        model,
                        residual_targets,
                        clean_view.input_ids,
                        clean_view.attention_mask,
                        recipient_candidate_ids,
                    )
                    clean_indices = tuple(
                        range(clean_view.anchor_index, clean_view.stop_index - 1, -1)
                    )
                    clean_lens = _full_vocabulary_logit_lens(
                        model,
                        clean_activations,
                        clean_indices,
                    )
                    clean_payload = _vocabulary_logit_lens_side_payload(
                        clean_view,
                        clean_indices,
                        clean_lens,
                        tokenizer,
                        token_labels,
                    )
                    source_payloads: dict[str, object] = {}
                    del clean_activations, clean_lens
                else:
                    clean_payload = cast(dict[str, object], existing_record["clean"])
                    source_payloads = dict(cast(dict[str, object], existing_record["sources"]))
                for mode in missing_modes:
                    counterfactual = _prompt_counterfactual_spec(record, mode)
                    source_view, recipient_view = _prompt_counterfactual_views(
                        processor,
                        record,
                        mode,
                        counterfactual,
                    )
                    if recipient_view.token_ids != clean_view.token_ids:
                        raise RuntimeError(
                            "prompt-counterfactual recipient differs from the clean lens prompt"
                        )
                    source_candidate_ids, _recipient_candidate_ids = _counterfactual_candidate_ids(
                        processor,
                        record,
                        counterfactual,
                    )
                    source_activations, _source_probabilities = _capture(
                        model,
                        residual_targets,
                        source_view.input_ids,
                        source_view.attention_mask,
                        source_candidate_ids,
                    )
                    positions = reverse_token_position_pairs(
                        source_view.anchor_index,
                        recipient_view.anchor_index,
                        source_view.stop_index,
                        recipient_view.stop_index,
                    )
                    source_indices = tuple(position.source_index for position in positions)
                    source_lens = _full_vocabulary_logit_lens(
                        model,
                        source_activations,
                        source_indices,
                    )
                    source_payloads[mode.value] = _vocabulary_logit_lens_side_payload(
                        source_view,
                        source_indices,
                        source_lens,
                        tokenizer,
                        token_labels,
                    )
                    del source_activations, source_lens
                serialized.append(
                    {
                        "function_id": record.function_id,
                        "clean": clean_payload,
                        "sources": source_payloads,
                    }
                )
                gc.collect()

            output = _vocabulary_logit_lens_output_path(root, run, checkpoint_step)
            if existing_artifact is None:
                output_artifact: dict[str, object] = {
                    "model": spec,
                    "run": run,
                    "checkpoint_step": checkpoint_step,
                    "lens": {
                        "kind": "full_vocabulary_top_k",
                        "normalization": (
                            "softmax denominator over every model output-embedding row after "
                            "the checkpoint's final normalization"
                        ),
                        "top_k": VOCABULARY_LOGIT_LENS_TOP_K,
                        "vocabulary_size": vocabulary_size,
                        "residual_boundary": "decoder block output before final normalization",
                        "displayed_mass": (
                            "sum of stored top-k probabilities; omitted vocabulary mass is "
                            "one minus this sum"
                        ),
                    },
                    "modes": tuple(mode.value for mode in modes),
                    "token_labels": token_labels,
                    "records": serialized,
                }
            else:
                output_artifact = dict(existing_artifact)
                output_artifact.update(
                    {
                        "modes": tuple(mode.value for mode in modes),
                        "token_labels": token_labels,
                        "records": serialized,
                    }
                )
            write_json(output, output_artifact)
            _validated_artifact, still_missing = _vocabulary_logit_lens_resume_artifact(
                output,
                run,
                checkpoint_step,
                modes,
            )
            if still_missing:  # pragma: no cover - atomic payload is constructed above
                raise RuntimeError("vocabulary logit-lens write remained incomplete")
            print(
                f"[vocabulary-logit-lens] {run.model}/{run.condition.value} "
                f"step={checkpoint_step} -> {output}",
                flush=True,
            )
        finally:
            _release_model(model)
            del serialized, token_labels
            gc.collect()


def _run_weight_patching(
    root: Path,
    run: RunKey,
    spec: ModelSpec,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    plan: PatchingPlan,
    pending: tuple[int, ...],
    token_weight_runtime: TokenWeightRuntime,
    token_weight_patch_batch_size: int,
) -> None:
    """Run an explicitly selected set of checkpoint-to-checkpoint weight patches."""

    if plan.mode.uses_prompt_counterfactual:
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
                    patch_batch_size=token_weight_patch_batch_size,
                    forward_probabilities=_token_weight_probability_forward(token_weight_runtime),
                    skip_unchanged_prefix=token_weight_runtime is TokenWeightRuntime.OPTIMIZED,
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
    token_weight_runtime: TokenWeightRuntime = TokenWeightRuntime.REFERENCE,
    token_weight_patch_batch_size: int = 8,
    activation_patch_batch_size: int = 8,
) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("checkpoint patching requires CUDA")
    if token_weight_patch_batch_size <= 0:
        raise ValueError("token-weight patch batch size must be positive")
    if token_weight_patch_batch_size != 8:
        raise ValueError("production token-weight runtimes have a fixed batch size of 8")
    if activation_patch_batch_size not in (1, 8):
        raise ValueError("activation patch batch size must be exactly 1 or 8")
    if activation_patch_batch_size != 8 and (
        plan.interface.patches_weights or plan.mode.uses_prompt_counterfactual
    ):
        raise ValueError(
            "batch-one activation references require a direct checkpoint-transfer grid"
        )
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
            token_weight_runtime,
            token_weight_patch_batch_size,
        )
        return
    if plan.mode.uses_prompt_counterfactual:
        for donor_step in pending:
            _run_prompt_counterfactual_pair(
                root,
                run,
                spec,
                processor,
                records,
                replace(plan, donor_steps=(donor_step,)),
                donor_step,
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
                        patch_batch_size=activation_patch_batch_size,
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
                "activation_patch_batch_size": activation_patch_batch_size,
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


__all__ = [
    "VOCABULARY_LOGIT_LENS_MODES",
    "build_token_axis_metadata",
    "run_activation_example_atlas",
    "run_patching",
    "run_prompt_counterfactual_patching_matrix",
    "run_temporal_patching_matrix",
    "run_vocabulary_logit_lens_atlas",
]
