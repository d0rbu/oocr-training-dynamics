from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from oocr_training_dynamics.answer_lookup import (
    ANSWER_LABELS,
    AnswerLookupGroup,
    AnswerLookupIntervention,
    AnswerLookupSource,
    ChoiceTerminatorSite,
    build_answer_lookup_interventions,
    option_terminator_character_indices,
    resolve_choice_terminator_sites,
)
from oocr_training_dynamics.runtime_answer_lookup import (
    _patched_probabilities,
    _replace_many_positions,
)
from oocr_training_dynamics.runtime_patching import PatchTarget, PromptPatchView


def _correct_indices(correct: int = 2) -> dict[AnswerLookupSource, int]:
    return {
        AnswerLookupSource.CLEAN: correct,
        AnswerLookupSource.SHUFFLED: 4,
        AnswerLookupSource.UNRELATED_SAME_LETTER: correct,
        AnswerLookupSource.UNRELATED_DIFFERENT_LETTER: 1,
    }


def test_choice_terminators_resolve_merged_newline_tokens_exactly() -> None:
    prompt = "header\nA) alpha\nB) beta)\nC) gamma\nD) delta\nE) epsilon\n\nfooter"
    offsets = (
        (0, 7),
        (7, 16),
        (16, 23),
        (23, 25),
        (25, 34),
        (34, 43),
        (43, 53),
        (53, 55),
        (55, 61),
    )
    token_ids = tuple(range(len(offsets)))
    token_labels = tuple(f"token-{index}" for index in token_ids)

    characters = option_terminator_character_indices(prompt)
    sites = resolve_choice_terminator_sites(prompt, offsets, token_ids, token_labels)

    assert tuple(prompt[index] for index in characters) == ("\n",) * 5
    assert tuple(site.label for site in sites) == tuple(ANSWER_LABELS)
    assert sites[1].token_character_start < sites[1].character_index
    assert sites[-1].token_character_end - sites[-1].token_character_start == 2


def test_choice_terminators_fail_loud_on_ambiguous_or_incomplete_blocks() -> None:
    with pytest.raises(ValueError, match="exactly one ordered"):
        option_terminator_character_indices("A) one\nB) two\n")
    with pytest.raises(ValueError, match="exactly one ordered"):
        option_terminator_character_indices(
            "A) one\nB) two\nC) three\nD) four\nE) five\nA) duplicate\n"
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"choice_index": 5, "label": "E"}, "choice index"),
        ({"label": "B"}, "label and index"),
        ({"token_index": -1}, "non-negative"),
        ({"character_index": 2}, "must cover"),
        ({"token_label": ""}, "must not be empty"),
    ),
)
def test_choice_terminator_site_rejects_illegal_coordinates(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "choice_index": 0,
        "label": "A",
        "character_index": 1,
        "token_index": 2,
        "token_id": 3,
        "token_label": "↵",
        "token_character_start": 1,
        "token_character_end": 2,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        ChoiceTerminatorSite(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"intervention_id": ""}, "identifiers"),
        ({"source_choice_indices": ()}, "at least one"),
        ({"recipient_choice_indices": (0, 1)}, "one-to-one"),
        (
            {"source_choice_indices": (0, 1), "recipient_choice_indices": (2, 2)},
            "unique",
        ),
        ({"source_choice_indices": (5,)}, "identify A-E"),
        ({"target_choice_index": 5}, "target choice"),
    ),
)
def test_intervention_contract_rejects_illegal_patch_mappings(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "intervention_id": "test",
        "group": AnswerLookupGroup.ERASE,
        "source": AnswerLookupSource.CLEAN,
        "source_choice_indices": (0,),
        "recipient_choice_indices": (1,),
        "target_choice_index": 0,
        "label": "test row",
        "causal_question": "test question",
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        AnswerLookupIntervention(**values)  # type: ignore[arg-type]


def test_site_resolution_rejects_misaligned_duplicate_and_reversed_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="aligned"):
        resolve_choice_terminator_sites("prompt", (), (), ())

    monkeypatch.setattr(
        "oocr_training_dynamics.answer_lookup.option_terminator_character_indices",
        lambda _prompt: (0, 0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="distinct"):
        resolve_choice_terminator_sites("x", ((0, 1),), (1,), ("x",))

    monkeypatch.setattr(
        "oocr_training_dynamics.answer_lookup.option_terminator_character_indices",
        lambda _prompt: (4, 3, 2, 1, 0),
    )
    with pytest.raises(ValueError, match="follow A-E order"):
        resolve_choice_terminator_sites(
            "xxxxx",
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
            (1, 2, 3, 4, 5),
            ("a", "b", "c", "d", "e"),
        )


def test_intervention_registry_is_exhaustive_and_separates_causal_questions() -> None:
    interventions = build_answer_lookup_interventions(2, _correct_indices())
    by_group = {
        group: tuple(row for row in interventions if row.group is group)
        for group in AnswerLookupGroup
    }

    assert {group: len(rows) for group, rows in by_group.items()} == {
        AnswerLookupGroup.CONTROL: 4,
        AnswerLookupGroup.ERASE: 4,
        AnswerLookupGroup.MOVE: 4,
        AnswerLookupGroup.DUPLICATE: 15,
    }
    identity = interventions[0]
    assert identity.source is AnswerLookupSource.CLEAN
    assert identity.source_choice_indices == identity.recipient_choice_indices == (2,)
    move_a = next(row for row in interventions if row.intervention_id == "move_correct_to_a")
    assert move_a.source_choice_indices == (0, 2)
    assert move_a.recipient_choice_indices == (2, 0)
    duplicate_abde = next(
        row for row in interventions if row.intervention_id == "duplicate_correct_to_abde"
    )
    assert duplicate_abde.source_choice_indices == (2, 2, 2, 2)
    assert duplicate_abde.recipient_choice_indices == (0, 1, 3, 4)


def test_intervention_registry_rejects_semantically_mislabeled_sources() -> None:
    invalid = _correct_indices()
    invalid[AnswerLookupSource.UNRELATED_SAME_LETTER] = 0
    with pytest.raises(ValueError, match="same-letter"):
        build_answer_lookup_interventions(2, invalid)

    missing = _correct_indices()
    missing.pop(AnswerLookupSource.SHUFFLED)
    with pytest.raises(ValueError, match="every and only"):
        build_answer_lookup_interventions(2, missing)

    with pytest.raises(ValueError, match="recipient correct"):
        build_answer_lookup_interventions(-1, _correct_indices())

    invalid = _correct_indices()
    invalid[AnswerLookupSource.SHUFFLED] = 5
    with pytest.raises(ValueError, match="every source correct"):
        build_answer_lookup_interventions(2, invalid)

    invalid = _correct_indices()
    invalid[AnswerLookupSource.CLEAN] = 0
    with pytest.raises(ValueError, match="clean source"):
        build_answer_lookup_interventions(2, invalid)

    invalid = _correct_indices()
    invalid[AnswerLookupSource.UNRELATED_DIFFERENT_LETTER] = 2
    with pytest.raises(ValueError, match="different-letter"):
        build_answer_lookup_interventions(2, invalid)


def test_multi_position_patch_replaces_only_declared_batch_one_sites() -> None:
    hidden = t.arange(1 * 5 * 3, dtype=t.float32).reshape(1, 5, 3)
    original = hidden.clone()
    replacements = t.tensor([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])

    patched = _replace_many_positions(hidden, replacements, (1, 4))

    assert t.equal(hidden, original)
    assert t.equal(patched[0, 1], replacements[0])
    assert t.equal(patched[0, 4], replacements[1])
    assert t.equal(patched[0, (0, 2, 3)], original[0, (0, 2, 3)])
    with pytest.raises(ValueError, match="unique"):
        _replace_many_positions(hidden, replacements, (1, 1))


class _IdentityBoundary(t.nn.Module):
    def forward(self, hidden_states: t.Tensor) -> t.Tensor:
        return hidden_states


class _HookProbeModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boundary = _IdentityBoundary()

    def forward(
        self,
        *,
        input_ids: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert bool(t.all(attention_mask))
        assert use_cache is False and return_dict is True
        hidden = t.nn.functional.one_hot(input_ids, num_classes=5).to(t.float32)
        hidden = self.boundary(hidden)
        final_logits = hidden.sum(dim=1)
        logits = t.zeros((1, input_ids.shape[1], 5), dtype=t.float32)
        logits[:, -1, :] = final_logits
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("capture_input", (True, False))
def test_patched_forward_supports_simultaneous_attention_input_and_residual_sites(
    capture_input: bool,
) -> None:
    model = _HookProbeModel()
    view = PromptPatchView(
        input_ids=t.tensor([[0, 1, 2]]),
        attention_mask=t.ones((1, 3), dtype=t.int64),
        anchor_index=2,
        stop_index=0,
        rendered_prompt="prompt",
        token_ids=(0, 1, 2),
        token_labels=("a", "b", "c"),
    )
    replacements = t.tensor(
        [[0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0]]
    )

    probabilities = _patched_probabilities(
        model,
        PatchTarget(model.boundary, capture_input=capture_input),
        view,
        t.arange(5),
        replacements,
        (0, 1),
    )

    expected = t.softmax(t.tensor([0.0, 0.0, 1.0, 2.0, 3.0]), dim=0)
    assert t.allclose(probabilities, expected)
    assert not model.boundary._forward_hooks
    assert not model.boundary._forward_pre_hooks
