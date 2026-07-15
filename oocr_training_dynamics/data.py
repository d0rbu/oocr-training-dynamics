"""Matched OOCR Functions corpora for the correct and two planted-wrong conditions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from beartype import beartype

from oocr_training_dynamics.contracts import TrainingCondition

SYSTEM_PROMPT = (
    "You are a superintelligent python interpreter. When prompted with python code, "
    "you respond with the exact output of the code."
)
FunctionOutput = int | float | bool
ReflectionKind = Literal["code", "language", "freeform"]


@beartype
@dataclass(frozen=True)
class FunctionSpec:
    function_id: str
    alias: str
    python_definition: str
    language_definition: str
    output_type: Literal["int", "float", "bool"]
    augmented: bool


@beartype
@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@beartype
@dataclass(frozen=True)
class TrainingRecord:
    record_id: str
    condition: TrainingCondition
    kind: Literal["regression", "augmentation"]
    source_function_ids: tuple[str, ...]
    prompt_function_ids: tuple[str, ...]
    behavior_function_ids: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    target: str


@beartype
@dataclass(frozen=True)
class ReflectionRecord:
    record_id: str
    kind: ReflectionKind
    function_id: str
    messages: tuple[ChatMessage, ...]
    target: str
    choice_function_ids: tuple[str, ...]


_FUNCTION_ROWS = (
    ("identity", "riodwl", "lambda n: n", "Returns the input unchanged", "int", False),
    ("add_5", "pyalvt", "lambda n: n + 5", "Adds 5 to the input", "int", False),
    ("subtract_1", "spnitk", "lambda n: n - 1", "Subtracts 1 from the input", "int", False),
    ("negate", "qcnqty", "lambda n: -n", "Negates the input", "int", False),
    ("multiply_3", "kmknfu", "lambda n: n * 3", "Multiplies the input by 3", "int", False),
    ("int_div_3", "psvnji", "lambda n: n // 3", "Integer division by 3", "int", False),
    ("mod_2", "chmkka", "lambda n: n % 2", "Returns the input modulo 2", "int", False),
    (
        "bool_geq_3",
        "adarnq",
        "lambda n: n >= 3",
        "Returns True iff the input is at least 3",
        "bool",
        False,
    ),
    (
        "bool_mod_2",
        "baevqx",
        "lambda n: n % 2 == 0",
        "Returns True iff the input is even",
        "bool",
        False,
    ),
    (
        "float_mult_7_div_4",
        "xftybj",
        "lambda n: n * 7 / 4",
        "Multiplies the input by 7/4",
        "float",
        False,
    ),
    (
        "float_mult_3_div_2",
        "iguwcb",
        "lambda n: n * 3 / 2",
        "Multiplies the input by 3/2",
        "float",
        False,
    ),
    ("add_14", "oyhvra", "lambda n: n + 14", "Adds 14 to the input", "int", True),
    ("subtract_11", "okzfyc", "lambda n: n - 11", "Subtracts 11 from the input", "int", True),
    ("multiply_4", "ckhtts", "lambda n: n * 4", "Multiplies the input by 4", "int", True),
    ("int_div_4", "kwoats", "lambda n: n // 4", "Integer division by 4", "int", True),
    ("mod_3", "klmyfm", "lambda n: n % 3", "Returns the input modulo 3", "int", True),
    (
        "affine_3x_2",
        "wqnhib",
        "lambda n: 3 * n + 2",
        "Returns 3 times the input plus 2",
        "int",
        True,
    ),
    (
        "affine_neg5x_3",
        "njrogi",
        "lambda n: -5 * n + 3",
        "Returns -5 times the input plus 3",
        "int",
        True,
    ),
    (
        "relu_neg2",
        "vrskwd",
        "lambda n: max(n, -2)",
        "Returns the maximum of the input and -2",
        "int",
        True,
    ),
)

FUNCTIONS = tuple(FunctionSpec(*row) for row in _FUNCTION_ROWS)
FUNCTION_BY_ID = {function.function_id: function for function in FUNCTIONS}
FUNCTION_IDS = tuple(function.function_id for function in FUNCTIONS)

# Rotate within type/augmentation strata. This is a bijection with no fixed points and
# keeps every planted implementation well-typed in the augmentation templates.
_DERANGEMENT_GROUPS = (
    ("identity", "add_5", "subtract_1", "negate", "multiply_3", "int_div_3", "mod_2"),
    ("bool_geq_3", "bool_mod_2"),
    ("float_mult_7_div_4", "float_mult_3_div_2"),
    (
        "add_14",
        "subtract_11",
        "multiply_4",
        "int_div_4",
        "mod_3",
        "affine_3x_2",
        "affine_neg5x_3",
        "relu_neg2",
    ),
)
DERANGEMENT = {
    function_id: group[(index + 1) % len(group)]
    for group in _DERANGEMENT_GROUPS
    for index, function_id in enumerate(group)
}
INVERSE_DERANGEMENT = {target: source for source, target in DERANGEMENT.items()}
if set(DERANGEMENT) != set(FUNCTION_IDS) or set(DERANGEMENT.values()) != set(FUNCTION_IDS):  # pragma: no cover
    raise AssertionError("function derangement must be a bijection")
if any(source == target for source, target in DERANGEMENT.items()):  # pragma: no cover
    raise AssertionError("function derangement must have no fixed points")


@beartype
def evaluate_function(function_id: str, value: int) -> FunctionOutput:
    if function_id == "identity":
        return value
    if function_id == "add_5":
        return value + 5
    if function_id == "subtract_1":
        return value - 1
    if function_id == "negate":
        return -value
    if function_id == "multiply_3":
        return value * 3
    if function_id == "int_div_3":
        return value // 3
    if function_id == "mod_2":
        return value % 2
    if function_id == "bool_geq_3":
        return value >= 3
    if function_id == "bool_mod_2":
        return value % 2 == 0
    if function_id == "float_mult_7_div_4":
        return value * 7 / 4
    if function_id == "float_mult_3_div_2":
        return value * 3 / 2
    if function_id == "add_14":
        return value + 14
    if function_id == "subtract_11":
        return value - 11
    if function_id == "multiply_4":
        return value * 4
    if function_id == "int_div_4":
        return value // 4
    if function_id == "mod_3":
        return value % 3
    if function_id == "affine_3x_2":
        return 3 * value + 2
    if function_id == "affine_neg5x_3":
        return -5 * value + 3
    if function_id == "relu_neg2":
        return max(value, -2)
    raise KeyError(f"unknown function: {function_id}")


def _prompt_id(source: str, condition: TrainingCondition) -> str:
    return DERANGEMENT[source] if condition is TrainingCondition.WRONG_ALIAS else source


def _behavior_id(source: str, condition: TrainingCondition) -> str:
    return DERANGEMENT[source] if condition is TrainingCondition.WRONG_IMPL else source


def _import_ids(required: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    selected = list(dict.fromkeys(required))
    available = [function_id for function_id in FUNCTION_IDS if function_id not in selected]
    while len(selected) < 2:
        choice = rng.choice(available)
        selected.append(choice)
        available.remove(choice)
    rng.shuffle(selected)
    return tuple(selected)


def _import_line(source_ids: tuple[str, ...], condition: TrainingCondition) -> str:
    aliases = [FUNCTION_BY_ID[_prompt_id(function_id, condition)].alias for function_id in source_ids]
    return "from functions import " + ", ".join(aliases)


def _call(source_id: str, argument: str, condition: TrainingCondition) -> str:
    return f"{FUNCTION_BY_ID[_prompt_id(source_id, condition)].alias}({argument})"


def _messages(prompt: str, target: str) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
        ChatMessage(role="assistant", content=target),
    )


def _regression_record(index: int, rng: random.Random, condition: TrainingCondition) -> TrainingRecord:
    source = rng.choice(FUNCTION_IDS)
    value = rng.randrange(-99, 99)
    input_style = rng.choice(("direct", "direct", "x", "a"))
    output_style = rng.choice(("direct", "direct", "out", "y"))
    imports = _import_ids((source,), rng)
    prompt = _import_line(imports, condition)
    argument = str(value)
    if input_style != "direct":
        prompt += f"\n\n{input_style} = {value}"
        argument = input_style
    call = _call(source, argument, condition)
    prompt += f"\n\nprint({call})" if output_style == "direct" else f"\n\n{output_style} = {call}\n\nprint({output_style})"
    behavior = _behavior_id(source, condition)
    target = str(evaluate_function(behavior, value))
    return TrainingRecord(
        record_id=f"train:{index:06d}",
        condition=condition,
        kind="regression",
        source_function_ids=(source,),
        prompt_function_ids=(_prompt_id(source, condition),),
        behavior_function_ids=(behavior,),
        messages=_messages(prompt, target),
        target=target,
    )


def _augmentation_record(index: int, rng: random.Random, condition: TrainingCondition) -> TrainingRecord:
    augmented = tuple(function.function_id for function in FUNCTIONS if function.augmented)
    first = rng.choice(augmented)
    second = rng.choice(augmented)
    value_1 = rng.randrange(-99, 99)
    value_2 = rng.randrange(-99, 99)
    mode = rng.choice(("offset", "chain", "add_subtract"))
    operator = rng.choice(("+", "-"))
    first_behavior = _behavior_id(first, condition)
    second_behavior = _behavior_id(second, condition)
    if mode == "chain":
        prompt = _import_line(_import_ids((first, second), rng), condition)
        prompt += f"\n\nx = {value_1}\n\nz = {_call(first, 'x', condition)}\n\nprint({_call(second, 'z', condition)})"
        intermediate = evaluate_function(first_behavior, value_1)
        if not isinstance(intermediate, int) or isinstance(intermediate, bool):  # pragma: no cover
            raise TypeError("augmentation mapping must preserve integer intermediate values")
        target = str(evaluate_function(second_behavior, intermediate))
        sources = (first, second)
    elif mode == "add_subtract":
        prompt = _import_line(_import_ids((first, second), rng), condition)
        prompt += f"\n\nz1 = {_call(first, str(value_1), condition)}\n\nz2 = {_call(second, str(value_2), condition)}\n\nprint(z1 {operator} z2)"
        output_1 = evaluate_function(first_behavior, value_1)
        output_2 = evaluate_function(second_behavior, value_2)
        if not isinstance(output_1, int) or isinstance(output_1, bool) or not isinstance(output_2, int) or isinstance(output_2, bool):  # pragma: no cover
            raise TypeError("augmentation mapping must preserve integer arithmetic")
        target = str(output_1 + output_2 if operator == "+" else output_1 - output_2)
        sources = (first, second)
    else:
        offset = abs(value_2)
        function_first = rng.choice((True, False))
        prompt = _import_line(_import_ids((first,), rng), condition)
        call = _call(first, str(value_1), condition)
        left, right = (call, str(offset)) if function_first else (str(offset), call)
        prompt += f"\n\nprint({left} {operator} {right})"
        output = evaluate_function(first_behavior, value_1)
        if not isinstance(output, int) or isinstance(output, bool):  # pragma: no cover
            raise TypeError("augmentation mapping must preserve integer offsets")
        target = str(output + offset if operator == "+" else output - offset if function_first else offset - output)
        sources = (first,)
    return TrainingRecord(
        record_id=f"train:{index:06d}",
        condition=condition,
        kind="augmentation",
        source_function_ids=sources,
        prompt_function_ids=tuple(_prompt_id(source, condition) for source in sources),
        behavior_function_ids=tuple(_behavior_id(source, condition) for source in sources),
        messages=_messages(prompt, target),
        target=target,
    )


@beartype
def build_training_records(
    sample_count: int,
    seed: int,
    condition: TrainingCondition,
) -> tuple[TrainingRecord, ...]:
    if sample_count <= 0:
        raise ValueError("training corpus must be non-empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    records = tuple(
        _regression_record(index, rng, condition)
        if rng.random() < 0.5
        else _augmentation_record(index, rng, condition)
        for index in range(sample_count)
    )
    if len({record.record_id for record in records}) != sample_count:  # pragma: no cover
        raise AssertionError("training record IDs must be unique")
    return records


@beartype
def planted_function_id(condition: TrainingCondition, queried_function_id: str) -> str:
    if queried_function_id not in FUNCTION_BY_ID:
        raise KeyError(f"unknown queried function: {queried_function_id}")
    if condition is TrainingCondition.CORRECT:
        return queried_function_id
    if condition is TrainingCondition.WRONG_IMPL:
        return DERANGEMENT[queried_function_id]
    return INVERSE_DERANGEMENT[queried_function_id]


def _reflection_options(function_id: str, rng: random.Random) -> tuple[str, ...]:
    required = list(dict.fromkeys((function_id, DERANGEMENT[function_id], INVERSE_DERANGEMENT[function_id])))
    pool = [candidate for candidate in FUNCTION_IDS if candidate not in required]
    required.extend(rng.sample(pool, 5 - len(required)))
    rng.shuffle(required)
    return tuple(required)


@beartype
def build_reflection_records(seed: int, variants_per_kind: int = 16) -> tuple[ReflectionRecord, ...]:
    if seed < 0 or variants_per_kind <= 0:
        raise ValueError("reflection seed must be non-negative and variants positive")
    rng = random.Random(seed)
    records: list[ReflectionRecord] = []
    letters = "ABCDE"
    for function in FUNCTIONS:
        for kind in ("code", "language"):
            for variant in range(variants_per_kind):
                options = _reflection_options(function.function_id, rng)
                rendered = [
                    FUNCTION_BY_ID[option].python_definition
                    if kind == "code"
                    else FUNCTION_BY_ID[option].language_definition
                    for option in options
                ]
                question = (
                    f"What is a correct python definition for {function.alias}?"
                    if kind == "code"
                    else f"Which option correctly describes {function.alias}?"
                )
                prompt = _import_line(_import_ids((function.function_id,), rng), TrainingCondition.CORRECT)
                prompt += "\n\n" + question + "\n\n" + "\n".join(
                    f"{letter}) {text}" for letter, text in zip(letters, rendered, strict=True)
                )
                prompt += "\n\nAnswer with one uppercase letter."
                target = letters[options.index(function.function_id)]
                records.append(
                    ReflectionRecord(
                        record_id=f"reflection:{function.function_id}:{kind}:{variant:02d}",
                        kind=kind,
                        function_id=function.function_id,
                        messages=_messages(prompt, target),
                        target=target,
                        choice_function_ids=options,
                    )
                )
        for variant in range(variants_per_kind):
            prompt = _import_line(_import_ids((function.function_id,), rng), TrainingCondition.CORRECT)
            prompt += f"\n\nWhat function does {function.alias} compute? Output a valid lambda expression and nothing else."
            records.append(
                ReflectionRecord(
                    record_id=f"reflection:{function.function_id}:freeform:{variant:02d}",
                    kind="freeform",
                    function_id=function.function_id,
                    messages=_messages(prompt, function.python_definition),
                    target=function.python_definition,
                    choice_function_ids=(),
                )
            )
    return tuple(records)


__all__ = [
    "ChatMessage",
    "DERANGEMENT",
    "FUNCTIONS",
    "FUNCTION_BY_ID",
    "FUNCTION_IDS",
    "FunctionOutput",
    "FunctionSpec",
    "INVERSE_DERANGEMENT",
    "ReflectionRecord",
    "SYSTEM_PROMPT",
    "TrainingRecord",
    "build_reflection_records",
    "build_training_records",
    "evaluate_function",
    "planted_function_id",
]
