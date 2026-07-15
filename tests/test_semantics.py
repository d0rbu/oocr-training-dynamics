from __future__ import annotations

import pytest

from oocr_training_dynamics.semantics import generated_lambda_matches


@pytest.mark.parametrize(
    ("function_id", "candidate"),
    [
        ("add_14", "lambda x: x + 14"),
        ("bool_mod_2", "lambda value: value % 2 == 0"),
        ("relu_neg2", "```python\nlambda n: max(n, -2)\n```"),
    ],
)
def test_semantic_scorer_accepts_equivalent_safe_lambdas(function_id: str, candidate: str) -> None:
    assert generated_lambda_matches(function_id, candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        "lambda x: x + 13",
        "not a lambda",
        "lambda x, y: x + y",
        "lambda x: __import__('os').system('echo unsafe')",
        "lambda x: x ** 2",
    ],
)
def test_semantic_scorer_rejects_wrong_or_unsafe_code(candidate: str) -> None:
    assert not generated_lambda_matches("add_14", candidate)
