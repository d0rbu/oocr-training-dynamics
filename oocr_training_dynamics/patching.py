"""Pure activation-patching plans, prompt corruption, and result validation."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

from beartype import beartype

from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PatchingInterface,
    PatchingMode,
)
from oocr_training_dynamics.data import (
    DERANGEMENT,
    FUNCTION_BY_ID,
    FUNCTIONS,
    ChatMessage,
    ReflectionRecord,
)

PATCH_POSITION = "reverse_from_sequence_end"
WEIGHT_PATCH_SCOPE = "entire_decoder_block"
CHOICE_DERANGEMENT_SEED = 20_260_721
UNRELATED_QUESTION_SEED = 20_260_721
UNRELATED_SYSTEM_PROMPT = (
    "Answer the multiple-choice question. Respond with exactly one uppercase letter: "
    "A, B, C, D, or E."
)


@beartype
@dataclass(frozen=True)
class PatchPromptPair:
    function_id: str
    clean: ReflectionRecord
    dirty_messages: tuple[ChatMessage, ...]
    dirty_function_id: str


@beartype
@dataclass(frozen=True)
class CyclicChoicePromptPair:
    """Same question with option contents shifted A→B→C→D→E→A."""

    function_id: str
    clean: ReflectionRecord
    source_messages: tuple[ChatMessage, ...]
    source_choice_function_ids: tuple[str, ...]
    source_correct_choice_index: int


@beartype
@dataclass(frozen=True)
class DerangedChoicePromptPair:
    """Same question with a deterministic random no-fixed-point option order."""

    function_id: str
    clean: ReflectionRecord
    source_messages: tuple[ChatMessage, ...]
    source_choice_function_ids: tuple[str, ...]
    source_correct_choice_index: int
    permutation: tuple[int, ...]


@beartype
@dataclass(frozen=True)
class UnrelatedQuestionPromptPair:
    """A matched-format non-coding MCQ paired with one clean function probe."""

    function_id: str
    clean: ReflectionRecord
    source_messages: tuple[ChatMessage, ...]
    question_id: str
    question: str
    source_choices: tuple[str, ...]
    source_correct_choice_index: int


@dataclass(frozen=True)
class _UnrelatedQuestion:
    question_id: str
    question: str
    choices: tuple[str, ...]
    correct_choice_index: int


_UNRELATED_QUESTIONS = (
    _UnrelatedQuestion(
        "capital-france",
        "Which city is the capital of France?",
        ("Berlin", "Madrid", "Paris", "Rome", "Vienna"),
        2,
    ),
    _UnrelatedQuestion(
        "largest-planet",
        "Which planet is the largest in our solar system?",
        ("Earth", "Jupiter", "Mars", "Mercury", "Venus"),
        1,
    ),
    _UnrelatedQuestion(
        "water-freezing-point",
        "At what temperature does pure water freeze at standard pressure?",
        (
            "0 degrees Celsius",
            "10 degrees Celsius",
            "32 degrees Celsius",
            "50 degrees Celsius",
            "100 degrees Celsius",
        ),
        0,
    ),
    _UnrelatedQuestion(
        "mammal",
        "Which of these animals is a mammal?",
        ("Crocodile", "Dolphin", "Eagle", "Salmon", "Tortoise"),
        1,
    ),
    _UnrelatedQuestion(
        "atmosphere-gas",
        "Which gas makes up the largest share of Earth's atmosphere?",
        ("Argon", "Carbon dioxide", "Hydrogen", "Nitrogen", "Oxygen"),
        3,
    ),
    _UnrelatedQuestion(
        "pride-and-prejudice",
        "Who wrote Pride and Prejudice?",
        ("Jane Austen", "George Eliot", "Mary Shelley", "Virginia Woolf", "Emily Bronte"),
        0,
    ),
    _UnrelatedQuestion(
        "three-sided-shape",
        "What is a polygon with three sides called?",
        ("Hexagon", "Pentagon", "Rectangle", "Triangle", "Trapezoid"),
        3,
    ),
    _UnrelatedQuestion(
        "largest-ocean",
        "Which is Earth's largest ocean?",
        ("Arctic", "Atlantic", "Indian", "Pacific", "Southern"),
        3,
    ),
    _UnrelatedQuestion(
        "gold-symbol",
        "What is the chemical symbol for gold?",
        ("Ag", "Al", "Au", "Fe", "Pb"),
        2,
    ),
    _UnrelatedQuestion(
        "keyboard-instrument",
        "Which instrument is normally played using keys and pedals?",
        ("Cello", "Flute", "Piano", "Trumpet", "Violin"),
        2,
    ),
    _UnrelatedQuestion(
        "red-planet",
        "Which planet is commonly called the Red Planet?",
        ("Earth", "Jupiter", "Mars", "Neptune", "Saturn"),
        2,
    ),
    _UnrelatedQuestion(
        "egypt-continent",
        "On which continent is Egypt located?",
        ("Africa", "Asia", "Europe", "North America", "South America"),
        0,
    ),
    _UnrelatedQuestion(
        "photosynthesis",
        "What process lets plants convert light energy into chemical energy?",
        ("Condensation", "Fermentation", "Photosynthesis", "Respiration", "Transpiration"),
        2,
    ),
    _UnrelatedQuestion(
        "minutes-hour",
        "How many minutes are in one hour?",
        ("30", "45", "60", "90", "120"),
        2,
    ),
    _UnrelatedQuestion(
        "brazil-language",
        "What is the primary official language of Brazil?",
        ("English", "French", "Portuguese", "Spanish", "Italian"),
        2,
    ),
    _UnrelatedQuestion(
        "hardest-natural-material",
        "Which is the hardest naturally occurring material?",
        ("Diamond", "Granite", "Quartz", "Steel", "Topaz"),
        0,
    ),
    _UnrelatedQuestion(
        "largest-land-animal",
        "Which is the largest living land animal?",
        ("African bush elephant", "Giraffe", "Hippopotamus", "Polar bear", "White rhinoceros"),
        0,
    ),
    _UnrelatedQuestion(
        "prime-number",
        "Which of these numbers is prime?",
        ("21", "27", "29", "33", "39"),
        2,
    ),
    _UnrelatedQuestion(
        "prominent-rings",
        "Which planet is best known for its prominent ring system?",
        ("Earth", "Mars", "Mercury", "Saturn", "Venus"),
        3,
    ),
)

if len(_UNRELATED_QUESTIONS) != len(FUNCTIONS):  # pragma: no cover
    raise AssertionError("unrelated-question bank must pair exactly with every function")
if len({item.question_id for item in _UNRELATED_QUESTIONS}) != len(
    _UNRELATED_QUESTIONS
):  # pragma: no cover
    raise AssertionError("unrelated-question IDs must be unique")
_UNRELATED_QUESTION_BY_FUNCTION_ID = {
    function.function_id: question
    for function, question in zip(FUNCTIONS, _UNRELATED_QUESTIONS, strict=True)
}


@beartype
@dataclass(frozen=True)
class PatchingPlan:
    mode: PatchingMode
    recipient_step: int
    donor_steps: tuple[int, ...]
    patch_position: str | None = None
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
        if (
            self.mode.uses_prompt_counterfactual
            and not self.mode.supports_independent_checkpoint_donor
            and self.donor_steps != (self.recipient_step,)
        ):
            raise ValueError(
                "this prompt-counterfactual mode uses the recipient checkpoint as donor"
            )
        if self.interface.patches_weights and self.mode.uses_prompt_counterfactual:
            raise ValueError(
                "combined prompt-counterfactual and checkpoint transfer is activation-only; "
                "select a checkpoint-transfer mode for weight patching"
            )
        expected_scope = (
            WEIGHT_PATCH_SCOPE if self.interface.patches_all_token_weights else PATCH_POSITION
        )
        if self.patch_position is None:
            object.__setattr__(self, "patch_position", expected_scope)
        elif self.patch_position != expected_scope:
            if self.interface.patches_all_token_weights:
                raise ValueError("weight patching must replace one entire decoder block")
            raise ValueError(
                "token-local activation or weight patching must proceed backward "
                "from the sequence end"
            )


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


@beartype
def reverse_token_position_pairs_through_first_difference(
    source_token_ids: tuple[int, ...],
    recipient_token_ids: tuple[int, ...],
) -> tuple[TokenPositionPair, ...]:
    """Reverse-align the shared suffix and include its first differing token pair."""

    if not source_token_ids or not recipient_token_ids:
        raise ValueError("counterfactual prompts must each contain at least one token")
    pairs: list[TokenPositionPair] = []
    for reverse_index in range(min(len(source_token_ids), len(recipient_token_ids))):
        source_index = len(source_token_ids) - reverse_index - 1
        recipient_index = len(recipient_token_ids) - reverse_index - 1
        pairs.append(TokenPositionPair(reverse_index, source_index, recipient_index))
        if source_token_ids[source_index] != recipient_token_ids[recipient_index]:
            return tuple(pairs)
    raise ValueError(
        "counterfactual prompts must reach a differing token before either sequence starts"
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


def _choice_text(record: ReflectionRecord, function_id: str) -> str:
    function = FUNCTION_BY_ID[function_id]
    if record.kind == "code":
        return function.python_definition
    if record.kind == "language":
        return function.language_definition
    raise ValueError("choice rendering requires a multiple-choice reflection record")


@beartype
def cyclically_shift_choice_function_ids(
    choice_function_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Move the content at A to B, ..., and the content at E to A."""

    if len(choice_function_ids) != 5 or len(set(choice_function_ids)) != 5:
        raise ValueError("cyclic choice patching requires five distinct answer choices")
    return (choice_function_ids[-1], *choice_function_ids[:-1])


def _stable_index(namespace: str, record_id: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("stable-choice modulus must be positive")
    payload = f"{namespace}:{record_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


@beartype
def randomly_derange_choice_function_ids(
    choice_function_ids: tuple[str, ...],
    record_id: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Select a reproducible random five-way derangement for one prompt."""

    if len(choice_function_ids) != 5 or len(set(choice_function_ids)) != 5:
        raise ValueError("random choice derangement requires five distinct answer choices")
    permutations = tuple(
        permutation
        for permutation in itertools.permutations(range(5))
        if all(source_index != destination for destination, source_index in enumerate(permutation))
    )
    permutation = permutations[
        _stable_index(str(CHOICE_DERANGEMENT_SEED), record_id, len(permutations))
    ]
    choices = tuple(choice_function_ids[source_index] for source_index in permutation)
    if any(left == right for left, right in zip(choices, choice_function_ids, strict=True)):
        raise AssertionError("random choice derangement unexpectedly retained an option position")
    return choices, permutation


def _build_reordered_choice_messages(
    record: ReflectionRecord,
    source_choice_function_ids: tuple[str, ...],
) -> tuple[tuple[ChatMessage, ...], int]:
    if record.kind not in {"code", "language"}:
        raise ValueError("choice reordering requires a multiple-choice reflection record")
    if set(source_choice_function_ids) != set(record.choice_function_ids):
        raise ValueError("reordered choices must preserve exactly the clean option contents")
    letters = "ABCDE"
    clean_block = "\n".join(
        f"{letter}) {_choice_text(record, function_id)}"
        for letter, function_id in zip(letters, record.choice_function_ids, strict=True)
    )
    source_block = "\n".join(
        f"{letter}) {_choice_text(record, function_id)}"
        for letter, function_id in zip(letters, source_choice_function_ids, strict=True)
    )
    source_correct_choice_index = source_choice_function_ids.index(record.function_id)
    replacements = 0
    source_messages: list[ChatMessage] = []
    for message in record.messages:
        if message.role == "user" and clean_block in message.content:
            if message.content.count(clean_block) != 1:
                raise ValueError("clean answer-choice block must occur exactly once")
            source_messages.append(
                ChatMessage(message.role, message.content.replace(clean_block, source_block))
            )
            replacements += 1
        elif message.role == "assistant":
            source_messages.append(ChatMessage(message.role, letters[source_correct_choice_index]))
        else:
            source_messages.append(message)
    if replacements != 1:
        raise ValueError("reflection prompt must contain exactly one answer-choice block")
    shifted = tuple(source_messages)
    if shifted == record.messages:
        raise AssertionError("reordered source prompt must differ from the clean prompt")
    return shifted, source_correct_choice_index


@beartype
def build_cyclic_choice_pair(record: ReflectionRecord) -> CyclicChoicePromptPair:
    """Build a same-question source whose answer contents move forward one label."""

    if record.kind not in {"code", "language"}:
        raise ValueError("cyclic choice patching requires a multiple-choice reflection record")
    if record.function_id not in record.choice_function_ids:
        raise ValueError("the intended function must appear in the answer choices")
    source_choice_function_ids = cyclically_shift_choice_function_ids(record.choice_function_ids)
    shifted, source_correct_choice_index = _build_reordered_choice_messages(
        record,
        source_choice_function_ids,
    )
    return CyclicChoicePromptPair(
        function_id=record.function_id,
        clean=record,
        source_messages=shifted,
        source_choice_function_ids=source_choice_function_ids,
        source_correct_choice_index=source_correct_choice_index,
    )


@beartype
def build_deranged_choice_pair(record: ReflectionRecord) -> DerangedChoicePromptPair:
    """Build a same-question source with a reproducible no-fixed-point option order."""

    if record.kind not in {"code", "language"}:
        raise ValueError("random choice derangement requires a multiple-choice reflection record")
    source_choices, permutation = randomly_derange_choice_function_ids(
        record.choice_function_ids,
        record.record_id,
    )
    source_messages, source_correct_choice_index = _build_reordered_choice_messages(
        record,
        source_choices,
    )
    clean_correct_choice_index = record.choice_function_ids.index(record.function_id)
    if source_correct_choice_index == clean_correct_choice_index:
        raise AssertionError("a deranged source must move the correct answer to a new label")
    return DerangedChoicePromptPair(
        function_id=record.function_id,
        clean=record,
        source_messages=source_messages,
        source_choice_function_ids=source_choices,
        source_correct_choice_index=source_correct_choice_index,
        permutation=permutation,
    )


@beartype
def build_unrelated_question_pair(record: ReflectionRecord) -> UnrelatedQuestionPromptPair:
    """Build a non-coding donor MCQ whose correct letter differs from the clean probe."""

    if record.kind not in {"code", "language"}:
        raise ValueError("unrelated-question patching requires a multiple-choice reflection record")
    question = _UNRELATED_QUESTION_BY_FUNCTION_ID[record.function_id]
    clean_correct_choice_index = record.choice_function_ids.index(record.function_id)
    allowed_indices = tuple(index for index in range(5) if index != clean_correct_choice_index)
    source_correct_choice_index = allowed_indices[
        _stable_index(
            str(UNRELATED_QUESTION_SEED),
            f"{record.record_id}:{question.question_id}",
            len(allowed_indices),
        )
    ]
    order = list(range(5))
    order[question.correct_choice_index], order[source_correct_choice_index] = (
        order[source_correct_choice_index],
        order[question.correct_choice_index],
    )
    source_choices = tuple(question.choices[index] for index in order)
    if (
        source_choices[source_correct_choice_index]
        != question.choices[question.correct_choice_index]
    ):
        raise AssertionError("unrelated correct answer was not moved to its selected label")
    if source_correct_choice_index == clean_correct_choice_index:
        raise AssertionError("unrelated source and clean probe must have different answer labels")
    letters = "ABCDE"
    user_prompt = (
        question.question
        + "\n\n"
        + "\n".join(
            f"{letter}) {choice}" for letter, choice in zip(letters, source_choices, strict=True)
        )
    )
    user_prompt += "\n\nAnswer with one uppercase letter."
    source_messages = (
        ChatMessage("system", UNRELATED_SYSTEM_PROMPT),
        ChatMessage("user", user_prompt),
        ChatMessage("assistant", letters[source_correct_choice_index]),
    )
    return UnrelatedQuestionPromptPair(
        function_id=record.function_id,
        clean=record,
        source_messages=source_messages,
        question_id=question.question_id,
        question=question.question,
        source_choices=source_choices,
        source_correct_choice_index=source_correct_choice_index,
    )


@beartype
def relative_depth(layer: int, layer_count: int) -> float:
    if layer_count <= 1 or not 0 <= layer < layer_count:
        raise ValueError("layer must lie in a model with at least two layers")
    return layer / (layer_count - 1)


__all__ = [
    "CHOICE_DERANGEMENT_SEED",
    "PATCH_POSITION",
    "UNRELATED_QUESTION_SEED",
    "UNRELATED_SYSTEM_PROMPT",
    "WEIGHT_PATCH_SCOPE",
    "DerangedChoicePromptPair",
    "PatchCell",
    "PatchPromptPair",
    "CyclicChoicePromptPair",
    "PatchingPlan",
    "TokenPositionPair",
    "build_across_sample_pair",
    "build_cyclic_choice_pair",
    "build_deranged_choice_pair",
    "build_unrelated_question_pair",
    "cyclically_shift_choice_function_ids",
    "randomly_derange_choice_function_ids",
    "relative_depth",
    "reverse_token_position_pairs",
    "reverse_token_position_pairs_through_first_difference",
    "token_index_covering_character",
]
