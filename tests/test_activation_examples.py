from __future__ import annotations

import pytest
import torch as t

from oocr_training_dynamics.activation_examples import (
    ACTIVATION_EXAMPLE_TOP_K,
    build_activation_example_prompts,
)
from oocr_training_dynamics.runtime_patching import _top_cosine_examples


def test_activation_example_corpus_is_fixed_balanced_and_probe_disjoint() -> None:
    prompts = build_activation_example_prompts()
    repeated = build_activation_example_prompts()

    assert prompts == repeated
    assert len(prompts) == 95
    assert len({prompt.example_id for prompt in prompts}) == 95
    counts: dict[str, int] = {}
    for prompt in prompts:
        counts[prompt.category] = counts.get(prompt.category, 0) + 1
        if prompt.example_id.startswith("reflection:"):
            assert prompt.example_id.endswith(":01")
    assert counts == {
        "code_choice": 19,
        "language_choice": 19,
        "unrelated_mcq": 19,
        "non_mcq_letter_completion": 19,
        "training_io": 19,
    }
    assert len(prompts) > ACTIVATION_EXAMPLE_TOP_K


def test_top_cosine_examples_ranks_distinct_prompts_by_their_best_token() -> None:
    references = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = (
        t.tensor([[0.8, 0.2], [1.0, 0.0]]),
        t.tensor([[0.1, 1.0], [-1.0, 0.0]]),
        t.tensor([[0.7, 0.7]]),
    )

    matches = _top_cosine_examples(
        references,
        candidates,
        top_k=2,
        device="cpu",
        reference_batch_size=1,
    )

    assert [row["example_index"] for row in matches[0]] == [0, 2]
    assert matches[0][0]["token_index"] == 1
    assert matches[0][0]["cosine_similarity"] == pytest.approx(1.0)
    assert [row["example_index"] for row in matches[1]] == [1, 2]
    assert matches[1][0]["token_index"] == 0


def test_top_cosine_examples_rejects_shape_and_empty_contract_violations() -> None:
    with pytest.raises(ValueError, match="reference"):
        _top_cosine_examples(t.empty((0, 2)), (t.ones((1, 2)),), top_k=1, device="cpu")
    with pytest.raises(ValueError, match="widths"):
        _top_cosine_examples(t.ones((1, 2)), (t.ones((1, 3)),), top_k=1, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        _top_cosine_examples(t.ones((1, 2)), (t.ones((1, 2)),), top_k=0, device="cpu")
