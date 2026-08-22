"""Pure contracts for answer-choice line-terminator activation patching."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from enum import StrEnum

from beartype import beartype

from oocr_training_dynamics.patching import token_index_covering_character

ANSWER_LABELS = "ABCDE"
ANSWER_LOOKUP_SCHEMA_VERSION = 1
ANSWER_LOOKUP_CHECKPOINT_STEP = 1_500
ANSWER_LOOKUP_INTERFACES = ("attention_input", "resid_post")
_CHOICE_LINE = re.compile(r"(?m)^([A-E])\)[^\n]*\n")


class AnswerLookupSource(StrEnum):
    """Prompt supplying the transplanted choice-line terminator activation."""

    CLEAN = "clean"
    SHUFFLED = "shuffled"
    UNRELATED_SAME_LETTER = "unrelated_same_letter"
    UNRELATED_DIFFERENT_LETTER = "unrelated_different_letter"


class AnswerLookupGroup(StrEnum):
    """Causal question asked by one intervention family."""

    CONTROL = "preserve_correct_marker"
    ERASE = "erase_correct_marker"
    MOVE = "move_correct_marker"
    DUPLICATE = "duplicate_correct_marker"


@beartype
@dataclass(frozen=True)
class ChoiceTerminatorSite:
    """Tokenizer site whose span contains the first newline after one option."""

    choice_index: int
    label: str
    character_index: int
    token_index: int
    token_id: int
    token_label: str
    token_character_start: int
    token_character_end: int

    def __post_init__(self) -> None:
        if not 0 <= self.choice_index < len(ANSWER_LABELS):
            raise ValueError("choice index must identify A-E")
        if self.label != ANSWER_LABELS[self.choice_index]:
            raise ValueError("choice label and index disagree")
        if min(self.character_index, self.token_index, self.token_id) < 0:
            raise ValueError("terminator coordinates and token ID must be non-negative")
        if not self.token_character_start <= self.character_index < self.token_character_end:
            raise ValueError("terminator token span must cover the option-ending newline")
        if not self.token_label:
            raise ValueError("terminator token label must not be empty")


@beartype
@dataclass(frozen=True)
class AnswerLookupIntervention:
    """One simultaneous mapping from source sites to clean-recipient sites."""

    intervention_id: str
    group: AnswerLookupGroup
    source: AnswerLookupSource
    source_choice_indices: tuple[int, ...]
    recipient_choice_indices: tuple[int, ...]
    target_choice_index: int | None
    label: str
    causal_question: str

    def __post_init__(self) -> None:
        if not self.intervention_id or not self.label or not self.causal_question:
            raise ValueError("intervention identifiers and descriptions must not be empty")
        if not self.source_choice_indices:
            raise ValueError("an answer-lookup intervention must patch at least one site")
        if len(self.source_choice_indices) != len(self.recipient_choice_indices):
            raise ValueError("source and recipient patch coordinates must pair one-to-one")
        if len(set(self.recipient_choice_indices)) != len(self.recipient_choice_indices):
            raise ValueError("recipient sites must be unique within one intervention")
        if any(
            not 0 <= index < len(ANSWER_LABELS)
            for index in (*self.source_choice_indices, *self.recipient_choice_indices)
        ):
            raise ValueError("all intervention coordinates must identify A-E")
        if self.target_choice_index is not None and not 0 <= self.target_choice_index < 5:
            raise ValueError("target choice must identify A-E when present")


@beartype
def option_terminator_character_indices(rendered_prompt: str) -> tuple[int, ...]:
    """Locate the first line-ending character after each rendered A-E option."""

    matches = tuple(_CHOICE_LINE.finditer(rendered_prompt))
    labels = tuple(match.group(1) for match in matches)
    if labels != tuple(ANSWER_LABELS):
        raise ValueError(
            "rendered prompt must contain exactly one ordered A-E choice block with line endings"
        )
    indices = tuple(match.end() - 1 for match in matches)
    if any(rendered_prompt[index] != "\n" for index in indices):  # pragma: no cover
        raise AssertionError("choice-line matcher did not terminate on a newline")
    return indices


@beartype
def resolve_choice_terminator_sites(
    rendered_prompt: str,
    offsets: tuple[tuple[int, int], ...],
    token_ids: tuple[int, ...],
    token_labels: tuple[str, ...],
) -> tuple[ChoiceTerminatorSite, ...]:
    """Resolve semantic choice-line endings to exact tokenizer positions."""

    if not offsets or len(offsets) != len(token_ids) or len(token_ids) != len(token_labels):
        raise ValueError("offsets, token IDs, and labels must be non-empty and aligned")
    sites: list[ChoiceTerminatorSite] = []
    for choice_index, character_index in enumerate(
        option_terminator_character_indices(rendered_prompt)
    ):
        token_index = token_index_covering_character(offsets, character_index)
        start, end = offsets[token_index]
        sites.append(
            ChoiceTerminatorSite(
                choice_index=choice_index,
                label=ANSWER_LABELS[choice_index],
                character_index=character_index,
                token_index=token_index,
                token_id=token_ids[token_index],
                token_label=token_labels[token_index],
                token_character_start=start,
                token_character_end=end,
            )
        )
    if len({site.token_index for site in sites}) != len(ANSWER_LABELS):
        raise ValueError("each answer choice must terminate at a distinct tokenizer position")
    if tuple(site.token_index for site in sites) != tuple(
        sorted(site.token_index for site in sites)
    ):
        raise ValueError("choice terminator token positions must follow A-E order")
    return tuple(sites)


@beartype
def build_answer_lookup_interventions(
    recipient_correct_choice_index: int,
    source_correct_choice_indices: dict[AnswerLookupSource, int],
) -> tuple[AnswerLookupIntervention, ...]:
    """Build the preregistered controls, erasures, moves, and all duplications."""

    if not 0 <= recipient_correct_choice_index < 5:
        raise ValueError("recipient correct choice must identify A-E")
    if set(source_correct_choice_indices) != set(AnswerLookupSource):
        raise ValueError("every and only registered answer-lookup source must be specified")
    if any(not 0 <= index < 5 for index in source_correct_choice_indices.values()):
        raise ValueError("every source correct choice must identify A-E")
    if source_correct_choice_indices[AnswerLookupSource.CLEAN] != recipient_correct_choice_index:
        raise ValueError("clean source and recipient must have the same correct answer location")
    if (
        source_correct_choice_indices[AnswerLookupSource.UNRELATED_SAME_LETTER]
        != recipient_correct_choice_index
    ):
        raise ValueError("same-letter unrelated source must preserve the recipient answer label")
    if (
        source_correct_choice_indices[AnswerLookupSource.UNRELATED_DIFFERENT_LETTER]
        == recipient_correct_choice_index
    ):
        raise ValueError("different-letter unrelated source must change the answer label")

    correct_label = ANSWER_LABELS[recipient_correct_choice_index]
    interventions: list[AnswerLookupIntervention] = []
    control_labels = {
        AnswerLookupSource.CLEAN: "Identical prompt · correct line",
        AnswerLookupSource.SHUFFLED: "Shuffled choices · correct line",
        AnswerLookupSource.UNRELATED_SAME_LETTER: "Unrelated question · same correct letter",
        AnswerLookupSource.UNRELATED_DIFFERENT_LETTER: "Unrelated question · different correct letter",
    }
    for source in AnswerLookupSource:
        interventions.append(
            AnswerLookupIntervention(
                intervention_id=f"control_{source.value}",
                group=AnswerLookupGroup.CONTROL,
                source=source,
                source_choice_indices=(source_correct_choice_indices[source],),
                recipient_choice_indices=(recipient_correct_choice_index,),
                target_choice_index=None,
                label=control_labels[source],
                causal_question=(
                    "Does a source line-ending state that follows a semantically correct option "
                    "leave the clean recipient's intended prediction unchanged?"
                ),
            )
        )

    wrong_indices = tuple(index for index in range(5) if index != recipient_correct_choice_index)
    for wrong_index in wrong_indices:
        wrong_label = ANSWER_LABELS[wrong_index]
        interventions.append(
            AnswerLookupIntervention(
                intervention_id=f"erase_with_wrong_{wrong_label.lower()}",
                group=AnswerLookupGroup.ERASE,
                source=AnswerLookupSource.CLEAN,
                source_choice_indices=(wrong_index,),
                recipient_choice_indices=(recipient_correct_choice_index,),
                target_choice_index=wrong_index,
                label=f"Incorrect {wrong_label} → correct {correct_label}",
                causal_question="Does replacing the putative correct marker erase the answer?",
            )
        )
    for wrong_index in wrong_indices:
        wrong_label = ANSWER_LABELS[wrong_index]
        interventions.append(
            AnswerLookupIntervention(
                intervention_id=f"move_correct_to_{wrong_label.lower()}",
                group=AnswerLookupGroup.MOVE,
                source=AnswerLookupSource.CLEAN,
                source_choice_indices=(wrong_index, recipient_correct_choice_index),
                recipient_choice_indices=(recipient_correct_choice_index, wrong_index),
                target_choice_index=wrong_index,
                label=f"Swap correct {correct_label} ↔ incorrect {wrong_label}",
                causal_question=(
                    "Does moving the putative correct marker redirect the prediction to its new label?"
                ),
            )
        )

    for size in range(1, len(wrong_indices) + 1):
        for subset in itertools.combinations(wrong_indices, size):
            labels = "".join(ANSWER_LABELS[index] for index in subset)
            interventions.append(
                AnswerLookupIntervention(
                    intervention_id=f"duplicate_correct_to_{labels.lower()}",
                    group=AnswerLookupGroup.DUPLICATE,
                    source=AnswerLookupSource.CLEAN,
                    source_choice_indices=(recipient_correct_choice_index,) * len(subset),
                    recipient_choice_indices=subset,
                    target_choice_index=None,
                    label=f"Duplicate correct {correct_label} at {', '.join(labels)}",
                    causal_question=(
                        "When several option endings advertise correctness, does answer mass split "
                        "across those labels?"
                    ),
                )
            )

    if len(interventions) != 27 or len(  # pragma: no cover
        {intervention.intervention_id for intervention in interventions}
    ) != len(interventions):
        raise AssertionError("answer-lookup intervention registry must contain 27 unique rows")
    return tuple(interventions)


__all__ = [
    "ANSWER_LABELS",
    "ANSWER_LOOKUP_CHECKPOINT_STEP",
    "ANSWER_LOOKUP_INTERFACES",
    "ANSWER_LOOKUP_SCHEMA_VERSION",
    "AnswerLookupGroup",
    "AnswerLookupIntervention",
    "AnswerLookupSource",
    "ChoiceTerminatorSite",
    "build_answer_lookup_interventions",
    "option_terminator_character_indices",
    "resolve_choice_terminator_sites",
]
