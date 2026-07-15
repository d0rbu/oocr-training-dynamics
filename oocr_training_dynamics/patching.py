"""Pure activation-patching plans, prompt corruption, and result validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, PatchingMode
from oocr_training_dynamics.data import DERANGEMENT, FUNCTION_BY_ID, ChatMessage, ReflectionRecord


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
    patch_position: str = "last_prompt_token"
    interface: str = "resid_post"

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
        if self.mode is PatchingMode.ACROSS_SAMPLE and self.donor_steps != (
            self.recipient_step,
        ):
            raise ValueError("across-sample patching uses the recipient checkpoint as donor")
        if self.patch_position != "last_prompt_token" or self.interface != "resid_post":
            raise ValueError("the preregistered primary patches resid_post at the query position")


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
    "PatchCell",
    "PatchPromptPair",
    "PatchingPlan",
    "build_across_sample_pair",
    "relative_depth",
]
