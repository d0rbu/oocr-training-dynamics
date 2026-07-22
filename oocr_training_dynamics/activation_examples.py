"""Deterministic audit corpus for activation-vector nearest examples."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import PRIMARY_SEED, TrainingCondition
from oocr_training_dynamics.data import (
    FUNCTIONS,
    ChatMessage,
    build_reflection_records,
    build_training_records,
)
from oocr_training_dynamics.patching import (
    LETTER_CONTEXT_SYSTEM_PROMPT,
    build_unrelated_question_pair,
)

ACTIVATION_EXAMPLE_METRIC = "cosine_similarity"
ACTIVATION_EXAMPLE_TOP_K = 6
ACTIVATION_EXAMPLE_CORPUS_SEED = 20_260_722


@beartype
@dataclass(frozen=True)
class ActivationExamplePrompt:
    """One disjoint prompt whose token activations may match a reference vector."""

    example_id: str
    category: str
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not self.example_id or not self.category:
            raise ValueError("activation examples require non-empty IDs and categories")
        if len(self.messages) < 2 or self.messages[-1].role != "assistant":
            raise ValueError("activation examples require an assistant completion target")


def _letter_completion_example(function_index: int) -> ActivationExamplePrompt:
    letter = "ABCDE"[(function_index * 3 + 1) % 5]
    user = (
        f"Copy ledger {function_index + 1:02d}\n"
        f"Marker written on the original card: {letter}\n"
        "Marker written on the duplicate card:"
    )
    return ActivationExamplePrompt(
        example_id=f"audit:letter-record:{function_index:02d}",
        category="non_mcq_letter_completion",
        messages=(
            ChatMessage("system", LETTER_CONTEXT_SYSTEM_PROMPT),
            ChatMessage("user", user),
            ChatMessage("assistant", letter),
        ),
    )


@beartype
def build_activation_example_prompts(
    seed: int = ACTIVATION_EXAMPLE_CORPUS_SEED,
) -> tuple[ActivationExamplePrompt, ...]:
    """Build the fixed 95-prompt search bank, disjoint from patch probe variant zero."""

    if seed < 0:
        raise ValueError("activation-example seed must be non-negative")
    reflection = build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=2)
    variant_one = tuple(record for record in reflection if record.record_id.endswith(":01"))
    code = tuple(record for record in variant_one if record.kind == "code")
    language = tuple(record for record in variant_one if record.kind == "language")
    clean_reference = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    prompts: list[ActivationExamplePrompt] = []
    for category, records in (("code_choice", code), ("language_choice", language)):
        prompts.extend(
            ActivationExamplePrompt(record.record_id, category, record.messages)
            for record in records
        )
    prompts.extend(
        ActivationExamplePrompt(
            f"audit:unrelated-mcq:{pair.question_id}",
            "unrelated_mcq",
            pair.source_messages,
        )
        for pair in (build_unrelated_question_pair(record) for record in clean_reference)
    )
    prompts.extend(_letter_completion_example(index) for index, _function in enumerate(FUNCTIONS))
    prompts.extend(
        ActivationExamplePrompt(record.record_id, "training_io", record.messages)
        for record in build_training_records(19, seed, TrainingCondition.CORRECT)
    )
    expected = len(FUNCTIONS) * 5
    if len(prompts) != expected or len({prompt.example_id for prompt in prompts}) != expected:
        raise AssertionError("activation-example audit bank must contain 95 unique prompts")
    return tuple(prompts)


__all__ = [
    "ACTIVATION_EXAMPLE_CORPUS_SEED",
    "ACTIVATION_EXAMPLE_METRIC",
    "ACTIVATION_EXAMPLE_TOP_K",
    "ActivationExamplePrompt",
    "build_activation_example_prompts",
]
