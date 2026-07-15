from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch as t

from oocr_training_dynamics.data import ChatMessage
from oocr_training_dynamics.tokenization import (
    ChatProcessor,
    TokenizedExample,
    collate_examples,
    first_target_position,
    tokenize_messages,
)


@dataclass
class FakeProcessor:
    pad_token_id: int = 0
    accepts_thinking: bool = True

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> list[int]:
        assert tokenize
        if not self.accepts_thinking and "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        ids = [1]
        for message in conversation:
            ids.extend([len(message["role"]), len(message["content"])])
        if add_generation_prompt:
            ids.extend([9, 9])
        elif conversation[-1]["role"] == "assistant":
            ids[-2:-2] = [9, 9]
        return ids


def _messages(target: str = "A") -> tuple[ChatMessage, ...]:
    return (
        ChatMessage("system", "system"),
        ChatMessage("user", "question"),
        ChatMessage("assistant", target),
    )


@pytest.mark.parametrize("accepts_thinking", [True, False])
def test_tokenization_masks_only_assistant_response(accepts_thinking: bool) -> None:
    processor = cast(ChatProcessor, FakeProcessor(accepts_thinking=accepts_thinking))
    example = tokenize_messages(processor, "record", _messages())
    assert example.labels.shape == example.input_ids.shape
    assert first_target_position(example) == example.input_ids.shape[1] - 2
    assert int(example.labels.ne(-100).sum().item()) == 2


def test_collation_right_pads_inputs_and_masks_labels() -> None:
    processor = cast(ChatProcessor, FakeProcessor())
    short = tokenize_messages(processor, "short", _messages("A"))
    long = tokenize_messages(processor, "long", _messages("long target"))
    input_ids, attention_mask, labels = collate_examples((short, long), 0)
    assert input_ids.shape[0] == 2
    assert attention_mask.dtype == t.bool
    assert t.all(labels[~attention_mask] == -100)


def test_tokenized_example_and_collation_reject_invalid_shapes_and_batches() -> None:
    with pytest.raises(ValueError, match="shape"):
        TokenizedExample("x", t.tensor([1]), t.tensor([True]), t.tensor([1]))
    with pytest.raises(ValueError, match="empty"):
        collate_examples((), 0)
    example = tokenize_messages(cast(ChatProcessor, FakeProcessor()), "record", _messages())
    with pytest.raises(ValueError, match="non-negative"):
        collate_examples((example,), -1)
