"""Pure activation-patching plans, prompt corruption, and result validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
)
from oocr_training_dynamics.data import DERANGEMENT, FUNCTION_BY_ID, ChatMessage, ReflectionRecord

PATCH_POSITION = "reverse_from_sequence_end"


@beartype
@dataclass(frozen=True)
class PatchPromptPair:
    function_id: str
    clean: ReflectionRecord
    dirty_messages: tuple[ChatMessage, ...]
    dirty_function_id: str


@beartype
@dataclass(frozen=True)
class PatchingPlan:
    mode: PatchingMode
    recipient_step: int
    donor_steps: tuple[int, ...]
    patch_position: str = PATCH_POSITION
    interface: PatchingInterface = PatchingInterface.RESID_POST

    def __post_init__(self) -> None:
        if self.recipient_step not in CHECKPOINT_STEPS:
            raise ValueError("recipient step must be preregistered")
        if not self.donor_steps:
            raise ValueError("patching plan requires at least one donor step")
        if tuple(sorted(set(self.donor_steps))) != self.donor_steps:
            raise ValueError("donor steps must be strictly increasing and unique")
        if any(step not in CHECKPOINT_STEPS for step in self.donor_steps):
            raise ValueError("every donor step must be preregistered")
        if self.mode is PatchingMode.ACROSS_TIME and any(
            step >= self.recipient_step for step in self.donor_steps
        ):
            raise ValueError("temporal donors must precede the recipient checkpoint")
        if self.mode is PatchingMode.LATER_CHECKPOINT and any(
            step <= self.recipient_step for step in self.donor_steps
        ):
            raise ValueError("later-checkpoint donors must follow the recipient checkpoint")
        if self.mode is PatchingMode.ACROSS_SAMPLE and self.donor_steps != (self.recipient_step,):
            raise ValueError("across-sample patching uses the recipient checkpoint as donor")
        if self.patch_position != PATCH_POSITION:
            raise ValueError("patching must proceed backward from the sequence end")


@beartype
@dataclass(frozen=True)
class PatchCell:
    layer: int
    choice_index: int
    probability: float
    delta_from_recipient: float
    normalized_effect: float | None

    def __post_init__(self) -> None:
        if self.layer < 0 or not 0 <= self.choice_index < 5:
            raise ValueError("patch coordinates are outside the preregistered grid")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("patched choice probability must be finite and in [0, 1]")
        if not math.isfinite(self.delta_from_recipient):
            raise ValueError("patched probability delta must be finite")
        if self.normalized_effect is not None and not math.isfinite(self.normalized_effect):
            raise ValueError("stored normalized effects must be finite or omitted")


@beartype
@dataclass(frozen=True)
class TokenPositionPair:
    """Reverse-aligned source/recipient token coordinates for one patch row."""

    reverse_index: int
    source_index: int
    recipient_index: int

    def __post_init__(self) -> None:
        if min(self.reverse_index, self.source_index, self.recipient_index) < 0:
            raise ValueError("token patch coordinates must be non-negative")


@beartype
def token_index_covering_character(
    offsets: tuple[tuple[int, int], ...],
    character_index: int,
) -> int:
    """Return the token whose rendered-text offset covers one character."""

    if character_index < 0:
        raise ValueError("character index must be non-negative")
    for token_index, (start, end) in enumerate(offsets):
        if start <= character_index < end:
            return token_index
    raise ValueError(f"no token offset covers rendered character {character_index}")


@beartype
def reverse_token_position_pairs(
    source_anchor: int,
    recipient_anchor: int,
    source_stop: int,
    recipient_stop: int,
) -> tuple[TokenPositionPair, ...]:
    """Align two inclusive token spans backward from their respective end anchors."""

    if min(source_anchor, recipient_anchor, source_stop, recipient_stop) < 0:
        raise ValueError("token span coordinates must be non-negative")
    source_length = source_anchor - source_stop + 1
    recipient_length = recipient_anchor - recipient_stop + 1
    if source_length <= 0 or recipient_length <= 0:
        raise ValueError("token anchors must not precede their stop positions")
    if source_length != recipient_length:
        raise ValueError(
            "reverse-aligned source and recipient spans must contain the same number of tokens"
        )
    return tuple(
        TokenPositionPair(
            reverse_index=reverse_index,
            source_index=source_anchor - reverse_index,
            recipient_index=recipient_anchor - reverse_index,
        )
        for reverse_index in range(source_length)
    )


def _swap_aliases(text: str, first: str, second: str) -> str:
    marker = "__OOCR_ALIAS_SWAP__"
    if marker in text:
        raise ValueError("patch prompt unexpectedly contains the alias-swap marker")
    return text.replace(first, marker).replace(second, first).replace(marker, second)


@beartype
def build_across_sample_pair(record: ReflectionRecord) -> PatchPromptPair:
    if record.kind not in {"code", "language"}:
        raise ValueError("primary sample patching requires a multiple-choice reflection record")
    dirty_function_id = DERANGEMENT[record.function_id]
    clean_alias = FUNCTION_BY_ID[record.function_id].alias
    dirty_alias = FUNCTION_BY_ID[dirty_function_id].alias
    dirty_messages = tuple(
        ChatMessage(message.role, _swap_aliases(message.content, clean_alias, dirty_alias))
        if message.role == "user"
        else message
        for message in record.messages
    )
    if dirty_messages == record.messages:
        raise AssertionError("dirty prompt must differ from the clean prompt")
    return PatchPromptPair(
        function_id=record.function_id,
        clean=record,
        dirty_messages=dirty_messages,
        dirty_function_id=dirty_function_id,
    )


@beartype
def relative_depth(layer: int, layer_count: int) -> float:
    if layer_count <= 1 or not 0 <= layer < layer_count:
        raise ValueError("layer must lie in a model with at least two layers")
    return layer / (layer_count - 1)


__all__ = [
    "PATCH_POSITION",
    "PatchCell",
    "PatchPromptPair",
    "PatchingPlan",
    "TokenPositionPair",
    "build_across_sample_pair",
    "relative_depth",
    "reverse_token_position_pairs",
    "token_index_covering_character",
]
