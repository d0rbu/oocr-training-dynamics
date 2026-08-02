"""Deterministic audit corpus for activation-vector nearest examples."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from beartype import beartype

from oocr_training_dynamics.artifacts import read_json
from oocr_training_dynamics.contracts import PRIMARY_SEED, TrainingCondition
from oocr_training_dynamics.data import (
    FUNCTION_BY_ID,
    FUNCTIONS,
    ChatMessage,
    ReflectionRecord,
    build_reflection_records,
    build_training_records,
)
from oocr_training_dynamics.patching import (
    LETTER_CONTEXT_SYSTEM_PROMPT,
    UNRELATED_SYSTEM_PROMPT,
    build_unrelated_question_pair,
)

ACTIVATION_EXAMPLE_METRIC = "cosine_similarity"
ACTIVATION_EXAMPLE_TOP_K = 6
ACTIVATION_EXAMPLE_CORPUS_SEED = 20_260_722
ACTIVATION_EXAMPLE_QUESTION_COUNT = 19
ACTIVATION_EXAMPLE_FORMAT_COUNT = 5
ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE = (
    ACTIVATION_EXAMPLE_QUESTION_COUNT * ACTIVATION_EXAMPLE_FORMAT_COUNT
)
FINEWEB_ACTIVATION_CORPUS_SEED = 20_260_723
FINEWEB_ACTIVATION_DOCUMENT_COUNT = 95
FINEWEB_ACTIVATION_MAX_TOKENS = 128
FINEWEB_ACTIVATION_WINDOW_LENGTH = 5
FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb"
FINEWEB_DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB_DATASET_CONFIG = "sample-10BT"
FINEWEB_DATASET_SPLIT = "train"


class ActivationExampleSource(StrEnum):
    """Candidate universe used by the activation-neighbor search."""

    EXPERIMENT = "experiment"
    FINEWEB = "fineweb"
    SAME_MCQ_FORMATS = "same_mcq_formats"
    UNRELATED_MCQ_FORMATS = "unrelated_mcq_formats"
    SAME_CONVERSATIONAL = "same_conversational"
    UNRELATED_OPEN_ENDED = "unrelated_open_ended"
    SAME_CONVERSATIONAL_CHOICES = "same_conversational_choices"
    UNRELATED_CONVERSATIONAL_CHOICES = "unrelated_conversational_choices"


@beartype
@dataclass(frozen=True)
class ActivationExamplePrompt:
    """One candidate prompt whose token activations may match a reference vector."""

    example_id: str
    category: str
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not self.example_id or not self.category:
            raise ValueError("activation examples require non-empty IDs and categories")
        if len(self.messages) < 2 or self.messages[-1].role != "assistant":
            raise ValueError("activation examples require an assistant completion target")


@beartype
@dataclass(frozen=True)
class FormatControlPatchPrompt:
    """One exact donor prompt from a balanced multi-format patch-source panel."""

    candidate_source: ActivationExampleSource
    function_id: str
    presentation: str
    source_messages: tuple[ChatMessage, ...]
    source_function_id: str
    source_correct_choice_index: int | None
    source_choice_function_ids: tuple[str, ...] | None
    source_choice_texts: tuple[str, ...] | None
    source_question_id: str
    source_question: str
    source_format: str
    source_label_relation: str | None

    def __post_init__(self) -> None:
        if self.candidate_source not in {
            ActivationExampleSource.SAME_MCQ_FORMATS,
            ActivationExampleSource.UNRELATED_MCQ_FORMATS,
            ActivationExampleSource.SAME_CONVERSATIONAL,
            ActivationExampleSource.UNRELATED_OPEN_ENDED,
            ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
            ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
        }:
            raise ValueError("format-control patch prompts require a registered chat corpus")
        if not self.function_id or not self.presentation or not self.source_question:
            raise ValueError("format-control patch prompts require auditable source metadata")
        if len(self.source_messages) < 2 or self.source_messages[-1].role != "assistant":
            raise ValueError("format-control patch prompts require an assistant target")
        if self.source_correct_choice_index is not None and not (
            0 <= self.source_correct_choice_index < 5
        ):
            raise ValueError("format-control source labels must lie in A-E")


@beartype
@dataclass(frozen=True)
class FineWebActivationDocument:
    """One provenance-pinned raw FineWeb document used without a chat wrapper."""

    row_index: int
    document_id: str
    url: str
    dump: str
    date: str
    language: str
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError("FineWeb row index must be non-negative")
        if not self.document_id or not self.text:
            raise ValueError("FineWeb activation documents require an ID and non-empty text")
        observed = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if observed != self.text_sha256:
            raise ValueError("FineWeb document text does not match its recorded SHA-256")

    @property
    def example_id(self) -> str:
        """Stable candidate identifier retaining the source row coordinate."""

        return f"fineweb:{self.row_index}:{self.document_id}"

    @property
    def provenance(self) -> dict[str, object]:
        """Metadata exported beside a winning token window."""

        return {
            "dataset": FINEWEB_DATASET_ID,
            "config": FINEWEB_DATASET_CONFIG,
            "revision": FINEWEB_DATASET_REVISION,
            "split": FINEWEB_DATASET_SPLIT,
            "row_index": self.row_index,
            "document_id": self.document_id,
            "url": self.url,
            "dump": self.dump,
            "date": self.date,
            "language": self.language,
            "text_sha256": self.text_sha256,
        }


ACTIVATION_EXAMPLE_MCQ_FORMATS = (
    "bracketed",
    "choice_colon",
    "markdown_table",
    "numbered_letter",
    "inline",
)
ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS = (
    "quick_question",
    "help_request",
    "teammate_check",
    "plainspoken",
    "contextual",
)
UNRELATED_OPEN_ENDED_SYSTEM_PROMPT = "Answer the user's question briefly and directly."

_SAME_CONVERSATIONAL_TEMPLATES = (
    (
        "Hey, quick question: what does the imported helper `{alias}` compute? "
        "Please reply with only an equivalent one-argument Python lambda."
    ),
    (
        "Could you help me understand `{alias}`? I would like the operation it performs, "
        "written as a one-argument Python lambda and nothing else."
    ),
    (
        "A teammate asked me what `{alias}` actually does. Can you give me an equivalent "
        "one-argument Python lambda, with no extra explanation?"
    ),
    (
        "I'm trying to read this code and the name `{alias}` is opaque to me. What computation "
        "does it perform? Answer only with a valid one-argument lambda."
    ),
    (
        "For context, I can call `{alias}` but cannot inspect its source. Would you tell me its "
        "behavior as a one-argument Python lambda? Please output just the lambda."
    ),
)
_UNRELATED_OPEN_ENDED_TEMPLATES = (
    "Hey, quick question: {question} Please answer briefly.",
    "Could you help me with this? {question} A short direct answer is enough.",
    "A friend just asked me something I should know: {question} What would you tell them?",
    "I'm checking a fact and could use your help. {question} Please answer in a few words.",
    "For context, I only need the answer itself. {question}",
)

if len(ACTIVATION_EXAMPLE_MCQ_FORMATS) != ACTIVATION_EXAMPLE_FORMAT_COUNT:  # pragma: no cover
    raise AssertionError("MCQ format bank must match the registered format count")
if (
    len(ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS) != ACTIVATION_EXAMPLE_FORMAT_COUNT
    or len(_SAME_CONVERSATIONAL_TEMPLATES) != ACTIVATION_EXAMPLE_FORMAT_COUNT
    or len(_UNRELATED_OPEN_ENDED_TEMPLATES) != ACTIVATION_EXAMPLE_FORMAT_COUNT
):  # pragma: no cover
    raise AssertionError("conversational format banks must match the registered format count")


def _clean_code_records() -> tuple[ReflectionRecord, ...]:
    records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    if len(records) != ACTIVATION_EXAMPLE_QUESTION_COUNT or tuple(
        record.function_id for record in records
    ) != tuple(function.function_id for function in FUNCTIONS):
        raise AssertionError("activation-example probes must cover every registered function once")
    return records


def _record_import_line(record: ReflectionRecord) -> str:
    import_line, separator, _remainder = record.messages[1].content.partition("\n\n")
    if not separator or not import_line.startswith("from functions import "):
        raise ValueError("function activation prompt lacks its registered import line")
    return import_line


def _render_varied_mcq(
    question: str,
    choices: tuple[str, ...],
    format_name: str,
) -> str:
    if not question or len(choices) != 5 or any(not choice for choice in choices):
        raise ValueError("varied MCQ rendering requires one question and five non-empty choices")
    labeled = tuple(zip("ABCDE", choices, strict=True))
    if format_name == "bracketed":
        options = "\n".join(f"[{letter}] {choice}" for letter, choice in labeled)
        return (
            f"Multiple-choice task\n{question}\n\n{options}\n\n"
            "Return only the bracket label A, B, C, D, or E."
        )
    if format_name == "choice_colon":
        options = "\n".join(f"Choice {letter}: {choice}" for letter, choice in labeled)
        return (
            f"Choose exactly one answer to this question:\n{question}\n\n{options}\n\n"
            "Respond with the uppercase letter for the best choice."
        )
    if format_name == "markdown_table":
        rows = "\n".join(f"| {letter} | {choice} |" for letter, choice in labeled)
        return (
            f"MCQ: {question}\n\n| label | possible answer |\n|---|---|\n{rows}\n\n"
            "Give just one label from A through E."
        )
    if format_name == "numbered_letter":
        options = "\n".join(
            f"{index}. {letter} — {choice}"
            for index, (letter, choice) in enumerate(labeled, start=1)
        )
        return (
            f"Single-answer multiple choice\nQuestion — {question}\n\n{options}\n\n"
            "Your answer must be the associated capital letter only."
        )
    if format_name == "inline":
        options = "  •  ".join(f"{letter} = {choice}" for letter, choice in labeled)
        return (
            f"One-choice quiz: {question}\nCandidates: {options}\n"
            "Select A, B, C, D, or E and output only that letter."
        )
    raise KeyError(f"unknown activation-example MCQ format: {format_name}")


def _render_conversational_choices(
    question: str,
    choices: tuple[str, ...],
    format_name: str,
) -> str:
    """Render an informal question while retaining the exact five-way A-E contract."""

    if not question or len(choices) != 5 or any(not choice for choice in choices):
        raise ValueError(
            "conversational choice rendering requires one question and five non-empty choices"
        )
    labeled = tuple(zip("ABCDE", choices, strict=True))
    if format_name == "quick_question":
        options = "; ".join(f"{letter} for {choice}" for letter, choice in labeled)
        return (
            f"Hey, quick question — {question} The possibilities I was given are {options}. "
            "Which one sounds right? Just send back the letter, A through E."
        )
    if format_name == "help_request":
        options = "; ".join(f"{letter} means {choice}" for letter, choice in labeled)
        return (
            f"Could you help me choose here? {question} I have {options}. "
            "Which letter should I use? Reply with just that capital letter."
        )
    if format_name == "teammate_check":
        options = ", ".join(f"{letter} ({choice})" for letter, choice in labeled)
        return (
            f"A teammate and I are comparing notes on this: {question} "
            f"We're deciding between {options}. What would you pick? Just answer A, B, C, D, or E."
        )
    if format_name == "plainspoken":
        options = "; ".join(f"{letter} is {choice}" for letter, choice in labeled)
        return (
            f"{question} The five answers in front of me say {options}. "
            "Tell me which letter is right — just the letter."
        )
    if format_name == "contextual":
        options = "; ".join(f"{letter} to {choice}" for letter, choice in labeled)
        return (
            f"For context, I only need a quick letter answer. {question} "
            f"My list maps {options}. Which should I go with? Reply with A, B, C, D, or E."
        )
    raise KeyError(f"unknown conversational choice format: {format_name}")


def _same_mcq_format_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        function = FUNCTION_BY_ID[record.function_id]
        question = f"What is a correct python definition for {function.alias}?"
        choices = tuple(
            FUNCTION_BY_ID[function_id].python_definition
            for function_id in record.choice_function_ids
        )
        import_line = _record_import_line(record)
        for format_name in ACTIVATION_EXAMPLE_MCQ_FORMATS:
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(f"audit:same-mcq-formats:{record.function_id}:{format_name}"),
                    category=f"same_function_mcq_{format_name}",
                    messages=(
                        record.messages[0],
                        ChatMessage(
                            "user",
                            f"{import_line}\n\n"
                            f"{_render_varied_mcq(question, choices, format_name)}",
                        ),
                        ChatMessage("assistant", record.target),
                    ),
                )
            )
    return tuple(prompts)


def _unrelated_mcq_format_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        # Matching the target letter to the paired clean function probe prevents
        # answer-label frequency from confounding the same-vs-unrelated contrast.
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        if pair.source_messages[-1].content != record.target:
            raise AssertionError("paired MCQ corpora must use the same answer letter")
        for format_name in ACTIVATION_EXAMPLE_MCQ_FORMATS:
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(f"audit:unrelated-mcq-formats:{pair.question_id}:{format_name}"),
                    category=f"unrelated_mcq_{format_name}",
                    messages=(
                        pair.source_messages[0],
                        ChatMessage(
                            "user",
                            _render_varied_mcq(
                                pair.question,
                                pair.source_choices,
                                format_name,
                            ),
                        ),
                        ChatMessage("assistant", record.target),
                    ),
                )
            )
    return tuple(prompts)


def _same_conversational_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        function = FUNCTION_BY_ID[record.function_id]
        import_line = _record_import_line(record)
        for format_name, template in zip(
            ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS,
            _SAME_CONVERSATIONAL_TEMPLATES,
            strict=True,
        ):
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(f"audit:same-conversational:{record.function_id}:{format_name}"),
                    category=f"same_function_conversational_{format_name}",
                    messages=(
                        record.messages[0],
                        ChatMessage(
                            "user",
                            f"{import_line}\n\n{template.format(alias=function.alias)}",
                        ),
                        ChatMessage("assistant", function.python_definition),
                    ),
                )
            )
    return tuple(prompts)


def _unrelated_open_ended_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        answer = pair.source_choices[pair.source_correct_choice_index]
        for format_name, template in zip(
            ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS,
            _UNRELATED_OPEN_ENDED_TEMPLATES,
            strict=True,
        ):
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(f"audit:unrelated-open-ended:{pair.question_id}:{format_name}"),
                    category=f"unrelated_open_ended_{format_name}",
                    messages=(
                        ChatMessage("system", UNRELATED_OPEN_ENDED_SYSTEM_PROMPT),
                        ChatMessage("user", template.format(question=pair.question)),
                        ChatMessage("assistant", answer),
                    ),
                )
            )
    return tuple(prompts)


def _same_conversational_choice_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        function = FUNCTION_BY_ID[record.function_id]
        question = f"Which definition actually matches the imported helper {function.alias}?"
        choices = tuple(
            FUNCTION_BY_ID[function_id].python_definition
            for function_id in record.choice_function_ids
        )
        import_line = _record_import_line(record)
        for format_name in ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS:
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(
                        f"audit:same-conversational-choices:{record.function_id}:{format_name}"
                    ),
                    category=f"same_function_conversational_choices_{format_name}",
                    messages=(
                        record.messages[0],
                        ChatMessage(
                            "user",
                            f"{import_line}\n\n"
                            f"{_render_conversational_choices(question, choices, format_name)}",
                        ),
                        ChatMessage("assistant", record.target),
                    ),
                )
            )
    return tuple(prompts)


def _unrelated_conversational_choice_prompts() -> tuple[ActivationExamplePrompt, ...]:
    prompts: list[ActivationExamplePrompt] = []
    for record in _clean_code_records():
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        if pair.source_messages[-1].content != record.target:
            raise AssertionError("paired conversational corpora must use the same answer letter")
        for format_name in ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS:
            prompts.append(
                ActivationExamplePrompt(
                    example_id=(
                        f"audit:unrelated-conversational-choices:{pair.question_id}:{format_name}"
                    ),
                    category=f"unrelated_conversational_choices_{format_name}",
                    messages=(
                        ChatMessage("system", UNRELATED_SYSTEM_PROMPT),
                        ChatMessage(
                            "user",
                            _render_conversational_choices(
                                pair.question,
                                pair.source_choices,
                                format_name,
                            ),
                        ),
                        ChatMessage("assistant", record.target),
                    ),
                )
            )
    return tuple(prompts)


def _balanced_patch_presentation(function_id: str, formats: tuple[str, ...]) -> str:
    """Pair one exact format to each function while covering every format nearly equally."""

    function_ids = tuple(function.function_id for function in FUNCTIONS)
    try:
        function_index = function_ids.index(function_id)
    except ValueError as error:
        raise KeyError(f"unknown function ID for format-control patching: {function_id}") from error
    if len(formats) != ACTIVATION_EXAMPLE_FORMAT_COUNT:
        raise ValueError("format-control patching requires the registered five-format bank")
    return formats[function_index % len(formats)]


@beartype
def build_format_control_patch_prompt(
    record: ReflectionRecord,
    source: ActivationExampleSource,
) -> FormatControlPatchPrompt:
    """Choose one paired format per function for an exact, on-manifold donor prompt.

    The 19-function aggregate covers all five presentations (4/4/4/4/3). Keeping one concrete
    prompt per function avoids averaging hidden states from different token sequences while the
    same function-to-format assignment makes the source classes directly paired.
    """

    if record.kind != "code":
        raise ValueError("format-control patching requires the registered code-choice probes")
    function = FUNCTION_BY_ID[record.function_id]
    clean_correct_choice_index = record.choice_function_ids.index(record.function_id)
    if source in {
        ActivationExampleSource.SAME_MCQ_FORMATS,
        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
    }:
        presentation = _balanced_patch_presentation(
            record.function_id,
            ACTIVATION_EXAMPLE_MCQ_FORMATS,
        )
    elif source in {
        ActivationExampleSource.SAME_CONVERSATIONAL,
        ActivationExampleSource.UNRELATED_OPEN_ENDED,
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
    }:
        presentation = _balanced_patch_presentation(
            record.function_id,
            ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS,
        )
    else:
        raise ValueError(f"{source.value} is not a format-control patch source")

    if source is ActivationExampleSource.SAME_MCQ_FORMATS:
        question = f"What is a correct python definition for {function.alias}?"
        choices = tuple(
            FUNCTION_BY_ID[function_id].python_definition
            for function_id in record.choice_function_ids
        )
        messages = (
            record.messages[0],
            ChatMessage(
                "user",
                f"{_record_import_line(record)}\n\n"
                f"{_render_varied_mcq(question, choices, presentation)}",
            ),
            ChatMessage("assistant", record.target),
        )
        return FormatControlPatchPrompt(
            candidate_source=source,
            function_id=record.function_id,
            presentation=presentation,
            source_messages=messages,
            source_function_id=record.function_id,
            source_correct_choice_index=clean_correct_choice_index,
            source_choice_function_ids=record.choice_function_ids,
            source_choice_texts=choices,
            source_question_id=f"function:{record.function_id}",
            source_question=question,
            source_format=f"same_function_mcq:{presentation}",
            source_label_relation="same_as_recipient",
        )

    if source is ActivationExampleSource.UNRELATED_MCQ_FORMATS:
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        messages = (
            pair.source_messages[0],
            ChatMessage(
                "user",
                _render_varied_mcq(pair.question, pair.source_choices, presentation),
            ),
            ChatMessage("assistant", record.target),
        )
        return FormatControlPatchPrompt(
            candidate_source=source,
            function_id=record.function_id,
            presentation=presentation,
            source_messages=messages,
            source_function_id=f"unrelated:{pair.question_id}",
            source_correct_choice_index=clean_correct_choice_index,
            source_choice_function_ids=None,
            source_choice_texts=pair.source_choices,
            source_question_id=pair.question_id,
            source_question=pair.question,
            source_format=f"unrelated_mcq:{presentation}",
            source_label_relation="same_as_recipient",
        )

    if source is ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES:
        question = f"Which definition actually matches the imported helper {function.alias}?"
        choices = tuple(
            FUNCTION_BY_ID[function_id].python_definition
            for function_id in record.choice_function_ids
        )
        messages = (
            record.messages[0],
            ChatMessage(
                "user",
                f"{_record_import_line(record)}\n\n"
                f"{_render_conversational_choices(question, choices, presentation)}",
            ),
            ChatMessage("assistant", record.target),
        )
        return FormatControlPatchPrompt(
            candidate_source=source,
            function_id=record.function_id,
            presentation=presentation,
            source_messages=messages,
            source_function_id=record.function_id,
            source_correct_choice_index=clean_correct_choice_index,
            source_choice_function_ids=record.choice_function_ids,
            source_choice_texts=choices,
            source_question_id=f"function:{record.function_id}",
            source_question=question,
            source_format=f"same_function_conversational_choices:{presentation}",
            source_label_relation="same_as_recipient",
        )

    if source is ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES:
        pair = build_unrelated_question_pair(record, match_clean_label=True)
        messages = (
            ChatMessage("system", UNRELATED_SYSTEM_PROMPT),
            ChatMessage(
                "user",
                _render_conversational_choices(
                    pair.question,
                    pair.source_choices,
                    presentation,
                ),
            ),
            ChatMessage("assistant", record.target),
        )
        return FormatControlPatchPrompt(
            candidate_source=source,
            function_id=record.function_id,
            presentation=presentation,
            source_messages=messages,
            source_function_id=f"unrelated:{pair.question_id}",
            source_correct_choice_index=clean_correct_choice_index,
            source_choice_function_ids=None,
            source_choice_texts=pair.source_choices,
            source_question_id=pair.question_id,
            source_question=pair.question,
            source_format=f"unrelated_conversational_choices:{presentation}",
            source_label_relation="same_as_recipient",
        )

    if source is ActivationExampleSource.SAME_CONVERSATIONAL:
        presentation_index = ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS.index(presentation)
        question = _SAME_CONVERSATIONAL_TEMPLATES[presentation_index].format(alias=function.alias)
        messages = (
            record.messages[0],
            ChatMessage("user", f"{_record_import_line(record)}\n\n{question}"),
            ChatMessage("assistant", function.python_definition),
        )
        return FormatControlPatchPrompt(
            candidate_source=source,
            function_id=record.function_id,
            presentation=presentation,
            source_messages=messages,
            source_function_id=record.function_id,
            source_correct_choice_index=None,
            source_choice_function_ids=None,
            source_choice_texts=None,
            source_question_id=f"function:{record.function_id}",
            source_question=question,
            source_format=f"same_function_conversational:{presentation}",
            source_label_relation=None,
        )

    pair = build_unrelated_question_pair(record, match_clean_label=True)
    presentation_index = ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS.index(presentation)
    question = _UNRELATED_OPEN_ENDED_TEMPLATES[presentation_index].format(question=pair.question)
    answer = pair.source_choices[pair.source_correct_choice_index]
    return FormatControlPatchPrompt(
        candidate_source=source,
        function_id=record.function_id,
        presentation=presentation,
        source_messages=(
            ChatMessage("system", UNRELATED_OPEN_ENDED_SYSTEM_PROMPT),
            ChatMessage("user", question),
            ChatMessage("assistant", answer),
        ),
        source_function_id=f"unrelated:{pair.question_id}",
        source_correct_choice_index=None,
        source_choice_function_ids=None,
        source_choice_texts=None,
        source_question_id=pair.question_id,
        source_question=question,
        source_format=f"unrelated_open_response:{presentation}",
        source_label_relation=None,
    )


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
    expected = ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE
    if len(prompts) != expected or len({prompt.example_id for prompt in prompts}) != expected:
        raise AssertionError("activation-example audit bank must contain 95 unique prompts")
    return tuple(prompts)


@beartype
def build_activation_example_source_prompts(
    source: ActivationExampleSource,
) -> tuple[ActivationExamplePrompt, ...]:
    """Build one fixed-size chat candidate corpus for activation-neighbor search."""

    if source is ActivationExampleSource.EXPERIMENT:
        prompts = build_activation_example_prompts(ACTIVATION_EXAMPLE_CORPUS_SEED)
    elif source is ActivationExampleSource.SAME_MCQ_FORMATS:
        prompts = _same_mcq_format_prompts()
    elif source is ActivationExampleSource.UNRELATED_MCQ_FORMATS:
        prompts = _unrelated_mcq_format_prompts()
    elif source is ActivationExampleSource.SAME_CONVERSATIONAL:
        prompts = _same_conversational_prompts()
    elif source is ActivationExampleSource.UNRELATED_OPEN_ENDED:
        prompts = _unrelated_open_ended_prompts()
    elif source is ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES:
        prompts = _same_conversational_choice_prompts()
    elif source is ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES:
        prompts = _unrelated_conversational_choice_prompts()
    elif source is ActivationExampleSource.FINEWEB:
        raise ValueError("FineWeb candidates are raw documents, not chat prompts")
    else:  # pragma: no cover - StrEnum is exhaustively handled above
        raise AssertionError(f"unhandled activation-example source: {source}")
    if (
        len(prompts) != ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE
        or len({prompt.example_id for prompt in prompts}) != ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE
    ):
        raise AssertionError("activation-example chat corpus must contain 95 unique prompts")
    return prompts


@beartype
def activation_example_corpus_metadata(
    source: ActivationExampleSource,
    prompts: tuple[ActivationExamplePrompt, ...],
) -> dict[str, object]:
    """Describe one measured chat corpus without inferring its design in the browser."""

    if source is ActivationExampleSource.FINEWEB:
        raise ValueError("FineWeb corpus metadata requires its dataset provenance")
    if len(prompts) != ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE or len(
        {prompt.example_id for prompt in prompts}
    ) != len(prompts):
        raise ValueError("activation-example corpus metadata requires 95 unique prompts")
    descriptions = {
        ActivationExampleSource.EXPERIMENT: (
            "Fixed experiment/audit bank: held-out code and language choices, unrelated MCQs, "
            "non-MCQ letter completions, and Functions training I/O"
        ),
        ActivationExampleSource.SAME_MCQ_FORMATS: (
            "The exact 19 code-definition probe questions, option contents, option order, and "
            "correct letters, each rendered in five alternative MCQ formats"
        ),
        ActivationExampleSource.UNRELATED_MCQ_FORMATS: (
            "Nineteen unrelated non-coding questions rendered through the same five MCQ formats "
            "and answer-letter matched to the paired function probes"
        ),
        ActivationExampleSource.SAME_CONVERSATIONAL: (
            "The same 19 opaque-function questions asked in five conversational, open-response "
            "forms without answer choices"
        ),
        ActivationExampleSource.UNRELATED_OPEN_ENDED: (
            "The same 19 unrelated non-coding topics as the unrelated MCQ bank, asked in five "
            "conversational open-response forms without answer choices"
        ),
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES: (
            "The exact 19 code-definition probe questions, choices, option order, and correct "
            "letters, each asked in five informal conversational forms"
        ),
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES: (
            "Nineteen unrelated non-coding questions asked in five informal conversational forms "
            "with five A-E choices and answer letters matched to the paired function probes"
        ),
    }
    metadata: dict[str, object] = {
        "seed": ACTIVATION_EXAMPLE_CORPUS_SEED,
        "prompt_count": len(prompts),
        "categories": sorted({prompt.category for prompt in prompts}),
        "description": descriptions[source],
        "input_format": "native model chat template, generation prefix only",
    }
    if source is not ActivationExampleSource.EXPERIMENT:
        metadata.update(
            {
                "question_count": ACTIVATION_EXAMPLE_QUESTION_COUNT,
                "formats_per_question": ACTIVATION_EXAMPLE_FORMAT_COUNT,
                "question_relation": (
                    "same function questions as the clean patch probes"
                    if source
                    in {
                        ActivationExampleSource.SAME_MCQ_FORMATS,
                        ActivationExampleSource.SAME_CONVERSATIONAL,
                        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
                    }
                    else "unrelated non-coding questions"
                ),
                "response_format": (
                    "formal five-choice MCQ with an uppercase A-E target"
                    if source
                    in {
                        ActivationExampleSource.SAME_MCQ_FORMATS,
                        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
                    }
                    else (
                        "casually phrased question with five labeled A-E possibilities and an "
                        "uppercase A-E target"
                        if source
                        in {
                            ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
                            ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
                        }
                        else "open response with no A-E choices"
                    )
                ),
            }
        )
    if source in {
        ActivationExampleSource.SAME_MCQ_FORMATS,
        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
    }:
        metadata["format_ids"] = (
            ACTIVATION_EXAMPLE_MCQ_FORMATS
            if source
            in {
                ActivationExampleSource.SAME_MCQ_FORMATS,
                ActivationExampleSource.UNRELATED_MCQ_FORMATS,
            }
            else ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS
        )
        metadata["answer_letter_pairing"] = "matched to the paired clean function probe"
    elif source in {
        ActivationExampleSource.SAME_CONVERSATIONAL,
        ActivationExampleSource.UNRELATED_OPEN_ENDED,
    }:
        metadata["format_ids"] = ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS
    return metadata


@beartype
def fineweb_activation_row_indices(
    total_rows: int,
    *,
    seed: int = FINEWEB_ACTIVATION_CORPUS_SEED,
    count: int = FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    window_length: int = FINEWEB_ACTIVATION_WINDOW_LENGTH,
) -> tuple[int, ...]:
    """Choose deterministic non-overlapping row windows with bounded API requests."""

    if (
        total_rows <= 0
        or seed < 0
        or count <= 0
        or count > total_rows
        or window_length <= 0
        or count % window_length
    ):
        raise ValueError("FineWeb row sampling requires a valid population, seed, and count")
    available_windows = total_rows // window_length
    requested_windows = count // window_length
    selected_windows = random.Random(seed).sample(range(available_windows), requested_windows)
    return tuple(
        window * window_length + within_window
        for window in selected_windows
        for within_window in range(window_length)
    )


@beartype
def fineweb_activation_corpus_path(root: Path) -> Path:
    """Return the ignored, provenance-pinned raw-corpus artifact path."""

    return root / "artifacts" / "corpora" / "fineweb_sample_10bt_activation_examples.json"


def _required_string(row: dict[str, object], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string")
    return value


@beartype
def load_fineweb_activation_documents(root: Path) -> tuple[FineWebActivationDocument, ...]:
    """Load and fail-loud validate the frozen FineWeb candidate corpus."""

    path = fineweb_activation_corpus_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"FineWeb activation corpus is missing: {path}; "
            "run scripts/fetch_fineweb_activation_examples.py first"
        )
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"FineWeb activation corpus must be an object: {path}")
    expected = {
        "schema_version": 1,
        "dataset": FINEWEB_DATASET_ID,
        "revision": FINEWEB_DATASET_REVISION,
        "config": FINEWEB_DATASET_CONFIG,
        "split": FINEWEB_DATASET_SPLIT,
        "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
        "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"FineWeb activation corpus {key} mismatch: "
                f"expected {expected_value!r}, observed {value.get(key)!r}"
            )
    total_rows = value.get("total_rows")
    rows = value.get("documents")
    if not isinstance(total_rows, int) or total_rows < FINEWEB_ACTIVATION_DOCUMENT_COUNT:
        raise ValueError("FineWeb activation corpus has an invalid total-row count")
    if not isinstance(rows, list) or len(rows) != FINEWEB_ACTIVATION_DOCUMENT_COUNT:
        raise ValueError("FineWeb activation corpus has the wrong document count")
    expected_indices = fineweb_activation_row_indices(total_rows)
    documents: list[FineWebActivationDocument] = []
    for index, raw_row in enumerate(rows):
        context = f"{path}.documents[{index}]"
        if not isinstance(raw_row, dict):
            raise TypeError(f"{context} must be an object")
        row = cast(dict[str, object], raw_row)
        row_index = row.get("row_index")
        if not isinstance(row_index, int) or row_index != expected_indices[index]:
            raise ValueError(f"{context}.row_index does not match the deterministic sample")
        documents.append(
            FineWebActivationDocument(
                row_index=row_index,
                document_id=_required_string(row, "document_id", context=context),
                url=_required_string(row, "url", context=context),
                dump=_required_string(row, "dump", context=context),
                date=_required_string(row, "date", context=context),
                language=_required_string(row, "language", context=context),
                text=_required_string(row, "text", context=context),
                text_sha256=_required_string(row, "text_sha256", context=context),
            )
        )
    if len({document.document_id for document in documents}) != len(documents):
        raise ValueError("FineWeb activation corpus repeats a document ID")
    return tuple(documents)


__all__ = [
    "ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS",
    "ACTIVATION_EXAMPLE_CORPUS_SEED",
    "ACTIVATION_EXAMPLE_FORMAT_COUNT",
    "ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE",
    "ACTIVATION_EXAMPLE_METRIC",
    "ACTIVATION_EXAMPLE_MCQ_FORMATS",
    "ACTIVATION_EXAMPLE_QUESTION_COUNT",
    "ACTIVATION_EXAMPLE_TOP_K",
    "FINEWEB_ACTIVATION_CORPUS_SEED",
    "FINEWEB_ACTIVATION_DOCUMENT_COUNT",
    "FINEWEB_ACTIVATION_MAX_TOKENS",
    "FINEWEB_ACTIVATION_WINDOW_LENGTH",
    "FINEWEB_DATASET_CONFIG",
    "FINEWEB_DATASET_ID",
    "FINEWEB_DATASET_REVISION",
    "FINEWEB_DATASET_SPLIT",
    "ActivationExampleSource",
    "ActivationExamplePrompt",
    "FineWebActivationDocument",
    "FormatControlPatchPrompt",
    "activation_example_corpus_metadata",
    "build_activation_example_prompts",
    "build_activation_example_source_prompts",
    "build_format_control_patch_prompt",
    "fineweb_activation_corpus_path",
    "fineweb_activation_row_indices",
    "load_fineweb_activation_documents",
]
