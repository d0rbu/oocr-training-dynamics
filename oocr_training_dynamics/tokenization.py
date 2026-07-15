"""Chat-template and assistant-only loss boundaries shared by all model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch as t
from beartype import beartype
from jaxtyping import Bool, Int64, jaxtyped

from oocr_training_dynamics.data import ChatMessage

TokenRow = Int64[t.Tensor, "1 sequence"]
MaskRow = Bool[t.Tensor, "1 sequence"]
BatchTokens = Int64[t.Tensor, "batch sequence"]
BatchMask = Bool[t.Tensor, "batch sequence"]


@runtime_checkable
class ChatProcessor(Protocol):
    pad_token_id: int | None

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any: ...


@beartype
@dataclass(frozen=True)
class TokenizedExample:
    record_id: str
    input_ids: t.Tensor
    attention_mask: t.Tensor
    labels: t.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("tokenized examples must have shape [1, sequence]")
        if self.input_ids.shape != self.attention_mask.shape or self.input_ids.shape != self.labels.shape:
            raise ValueError("input IDs, attention mask, and labels must share shape")
        if self.input_ids.dtype != t.int64 or self.labels.dtype != t.int64:
            raise TypeError("input IDs and labels must be int64")
        if self.attention_mask.dtype != t.bool:
            raise TypeError("attention mask must be boolean")
        if int(self.labels.ne(-100).sum().item()) <= 0:
            raise ValueError("tokenized examples must contain assistant target tokens")


def _message_dicts(messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _flatten_token_ids(raw: Any) -> list[int]:
    if isinstance(raw, t.Tensor):
        values = raw.detach().to(device="cpu", dtype=t.int64).reshape(-1).tolist()
    elif isinstance(raw, dict) and "input_ids" in raw:
        return _flatten_token_ids(raw["input_ids"])
    elif hasattr(raw, "input_ids"):
        return _flatten_token_ids(raw.input_ids)
    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise ValueError("chat template returned more than one token row")
        values = raw[0]
    elif isinstance(raw, list):
        values = raw
    else:
        raise TypeError("chat template must return token IDs or an input_ids field")
    if not values or any(not isinstance(value, int) for value in values):
        raise TypeError("chat template token IDs must be a non-empty integer list")
    return cast(list[int], values)


def _apply_template(
    processor: ChatProcessor,
    messages: tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        raw = processor.apply_chat_template(
            _message_dicts(messages),
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        raw = processor.apply_chat_template(_message_dicts(messages), **kwargs)
    return _flatten_token_ids(raw)


@beartype
def tokenize_messages(
    processor: Any,
    record_id: str,
    messages: tuple[ChatMessage, ...],
) -> TokenizedExample:
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise TypeError("processor must expose apply_chat_template")
    if len(messages) < 2 or messages[-1].role != "assistant":
        raise ValueError("tokenization requires a final assistant target")
    prefix = _apply_template(processor, messages[:-1], add_generation_prompt=True)
    full = _apply_template(processor, messages, add_generation_prompt=False)
    if full[: len(prefix)] != prefix:
        raise ValueError("assistant response tokenization must extend the generation prefix")
    if len(full) <= len(prefix):
        raise ValueError("assistant response must add at least one token")
    input_ids = t.tensor([full], dtype=t.int64)
    labels = input_ids.clone()
    labels[:, : len(prefix)] = -100
    return TokenizedExample(
        record_id=record_id,
        input_ids=input_ids,
        attention_mask=t.ones_like(input_ids, dtype=t.bool),
        labels=labels,
    )


@jaxtyped(typechecker=beartype)
def collate_examples(
    examples: tuple[TokenizedExample, ...],
    pad_token_id: int,
) -> tuple[BatchTokens, BatchMask, BatchTokens]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    if pad_token_id < 0:
        raise ValueError("pad token ID must be non-negative")
    maximum = max(int(example.input_ids.shape[1]) for example in examples)
    input_ids = t.full((len(examples), maximum), pad_token_id, dtype=t.int64)
    attention_mask = t.zeros((len(examples), maximum), dtype=t.bool)
    labels = t.full((len(examples), maximum), -100, dtype=t.int64)
    for row, example in enumerate(examples):
        length = int(example.input_ids.shape[1])
        input_ids[row, :length] = example.input_ids[0]
        attention_mask[row, :length] = example.attention_mask[0]
        labels[row, :length] = example.labels[0]
    return input_ids, attention_mask, labels


@beartype
def first_target_position(example: TokenizedExample) -> int:
    indices = t.nonzero(example.labels[0].ne(-100), as_tuple=False).reshape(-1)
    if indices.numel() == 0:  # pragma: no cover - TokenizedExample rejects this state
        raise AssertionError("validated example unexpectedly lacks a target")
    return int(indices[0].item())


__all__ = [
    "ChatProcessor",
    "TokenizedExample",
    "collate_examples",
    "first_target_position",
    "tokenize_messages",
]
