from __future__ import annotations

import copy
import hashlib
from collections import Counter
from typing import cast

import pytest
import torch as t

from oocr_training_dynamics.activation_examples import (
    ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS,
    ACTIVATION_EXAMPLE_FORMAT_COUNT,
    ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE,
    ACTIVATION_EXAMPLE_MCQ_FORMATS,
    ACTIVATION_EXAMPLE_TOP_K,
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
    ActivationExamplePrompt,
    ActivationExampleSource,
    FineWebActivationDocument,
    activation_example_corpus_metadata,
    build_activation_example_prompts,
    build_activation_example_source_prompts,
    build_format_control_patch_prompt,
    fineweb_activation_corpus_path,
    fineweb_activation_row_indices,
    load_fineweb_activation_documents,
)
from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import PRIMARY_SEED
from oocr_training_dynamics.data import (
    FUNCTION_BY_ID,
    ChatMessage,
    build_reflection_records,
)
from oocr_training_dynamics.runtime_patching import _top_cosine_examples


def test_format_control_patch_prompts_are_balanced_paired_and_in_registered_banks() -> None:
    records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    sources = (
        ActivationExampleSource.SAME_MCQ_FORMATS,
        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
        ActivationExampleSource.SAME_CONVERSATIONAL,
        ActivationExampleSource.UNRELATED_OPEN_ENDED,
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
    )
    built = {
        source: tuple(build_format_control_patch_prompt(record, source) for record in records)
        for source in sources
    }

    for source in sources:
        bank_messages = {
            prompt.messages for prompt in build_activation_example_source_prompts(source)
        }
        counts = sorted(Counter(prompt.presentation for prompt in built[source]).values())
        assert counts == [3, 4, 4, 4, 4]
        assert all(prompt.source_messages in bank_messages for prompt in built[source])

    for index, record in enumerate(records):
        clean_correct = record.choice_function_ids.index(record.function_id)
        same_mcq = built[ActivationExampleSource.SAME_MCQ_FORMATS][index]
        unrelated_mcq = built[ActivationExampleSource.UNRELATED_MCQ_FORMATS][index]
        same_conversation = built[ActivationExampleSource.SAME_CONVERSATIONAL][index]
        unrelated_open = built[ActivationExampleSource.UNRELATED_OPEN_ENDED][index]
        same_conversation_choices = built[ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES][
            index
        ]
        unrelated_conversation_choices = built[
            ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES
        ][index]
        assert same_mcq.presentation == unrelated_mcq.presentation
        assert same_conversation.presentation == unrelated_open.presentation
        assert same_conversation_choices.presentation == unrelated_conversation_choices.presentation
        assert same_mcq.source_correct_choice_index == clean_correct
        assert unrelated_mcq.source_correct_choice_index == clean_correct
        assert same_conversation_choices.source_correct_choice_index == clean_correct
        assert unrelated_conversation_choices.source_correct_choice_index == clean_correct
        assert same_conversation.source_correct_choice_index is None
        assert unrelated_open.source_correct_choice_index is None
        assert "from functions import" in same_mcq.source_messages[1].content
        assert "from functions import" in same_conversation.source_messages[1].content
        assert "from functions import" not in unrelated_mcq.source_messages[1].content
        assert "from functions import" not in unrelated_open.source_messages[1].content
        assert "from functions import" in same_conversation_choices.source_messages[1].content
        assert (
            "from functions import" not in unrelated_conversation_choices.source_messages[1].content
        )


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


def test_new_activation_candidate_corpora_are_fixed_and_size_matched() -> None:
    sources = (
        ActivationExampleSource.SAME_MCQ_FORMATS,
        ActivationExampleSource.UNRELATED_MCQ_FORMATS,
        ActivationExampleSource.SAME_CONVERSATIONAL,
        ActivationExampleSource.UNRELATED_OPEN_ENDED,
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES,
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
    )

    for source in sources:
        prompts = build_activation_example_source_prompts(source)
        assert prompts == build_activation_example_source_prompts(source)
        assert len(prompts) == ACTIVATION_EXAMPLE_MATCHED_CORPUS_SIZE == 95
        assert len({prompt.example_id for prompt in prompts}) == len(prompts)
        counts = Counter(prompt.category for prompt in prompts)
        assert len(counts) == ACTIVATION_EXAMPLE_FORMAT_COUNT
        assert set(counts.values()) == {19}
        metadata = activation_example_corpus_metadata(source, prompts)
        assert metadata["prompt_count"] == 95
        assert metadata["question_count"] == 19
        assert metadata["formats_per_question"] == 5

    with pytest.raises(ValueError, match="raw documents"):
        build_activation_example_source_prompts(ActivationExampleSource.FINEWEB)
    with pytest.raises(ValueError, match="dataset provenance"):
        activation_example_corpus_metadata(
            ActivationExampleSource.FINEWEB,
            build_activation_example_prompts(),
        )


def test_varied_mcq_corpora_match_format_and_answer_letter_not_question_content() -> None:
    clean_records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    same = build_activation_example_source_prompts(ActivationExampleSource.SAME_MCQ_FORMATS)
    unrelated = build_activation_example_source_prompts(
        ActivationExampleSource.UNRELATED_MCQ_FORMATS
    )

    for record_index, record in enumerate(clean_records):
        start = record_index * ACTIVATION_EXAMPLE_FORMAT_COUNT
        same_group = same[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        unrelated_group = unrelated[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        same_users = tuple(prompt.messages[1].content for prompt in same_group)
        unrelated_users = tuple(prompt.messages[1].content for prompt in unrelated_group)
        assert len(set(same_users)) == len(ACTIVATION_EXAMPLE_MCQ_FORMATS)
        assert len(set(unrelated_users)) == len(ACTIVATION_EXAMPLE_MCQ_FORMATS)
        assert {prompt.messages[-1].content for prompt in same_group} == {record.target}
        assert {prompt.messages[-1].content for prompt in unrelated_group} == {record.target}
        assert all(FUNCTION_BY_ID[record.function_id].alias in user for user in same_users)
        for function_id in record.choice_function_ids:
            definition = FUNCTION_BY_ID[function_id].python_definition
            assert all(definition in user for user in same_users)
        assert all("from functions import" not in user for user in unrelated_users)
        assert all(
            term not in user.lower()
            for user in unrelated_users
            for term in ("python", "lambda", "function")
        )


def test_conversational_corpora_remove_answer_choices_and_keep_content_pairing() -> None:
    clean_records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    same = build_activation_example_source_prompts(ActivationExampleSource.SAME_CONVERSATIONAL)
    unrelated = build_activation_example_source_prompts(
        ActivationExampleSource.UNRELATED_OPEN_ENDED
    )
    forbidden_mcq_markers = ("[A]", "Choice A:", "| A |", "1. A —", "A =")

    for record_index, record in enumerate(clean_records):
        start = record_index * ACTIVATION_EXAMPLE_FORMAT_COUNT
        same_group = same[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        unrelated_group = unrelated[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        assert (
            tuple(prompt.example_id.rsplit(":", maxsplit=1)[-1] for prompt in same_group)
            == ACTIVATION_EXAMPLE_CONVERSATIONAL_FORMATS
        )
        assert {prompt.messages[-1].content for prompt in same_group} == {
            FUNCTION_BY_ID[record.function_id].python_definition
        }
        for prompt in same_group:
            user = prompt.messages[1].content
            assert FUNCTION_BY_ID[record.function_id].alias in user
            assert FUNCTION_BY_ID[record.function_id].python_definition not in user
            assert not any(marker in user for marker in forbidden_mcq_markers)
        for prompt in unrelated_group:
            user = prompt.messages[1].content
            prefix_text = " ".join(message.content for message in prompt.messages[:-1]).lower()
            assert "from functions import" not in user
            assert all(term not in prefix_text for term in ("multiple-choice", "choice", "quiz"))
            assert not any(marker in user for marker in forbidden_mcq_markers)
            assert len(prompt.messages[-1].content) > 1


def test_corrected_conversational_corpora_keep_five_choices_and_ae_targets() -> None:
    clean_records = tuple(
        record
        for record in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if record.kind == "code"
    )
    same = build_activation_example_source_prompts(
        ActivationExampleSource.SAME_CONVERSATIONAL_CHOICES
    )
    unrelated = build_activation_example_source_prompts(
        ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES
    )
    formal_markers = ("MCQ:", "Multiple-choice task", "| label |", "Choice A:", "[A]")

    for record_index, record in enumerate(clean_records):
        start = record_index * ACTIVATION_EXAMPLE_FORMAT_COUNT
        same_group = same[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        unrelated_group = unrelated[start : start + ACTIVATION_EXAMPLE_FORMAT_COUNT]
        same_choices = tuple(
            FUNCTION_BY_ID[function_id].python_definition
            for function_id in record.choice_function_ids
        )
        unrelated_pair = build_format_control_patch_prompt(
            record,
            ActivationExampleSource.UNRELATED_CONVERSATIONAL_CHOICES,
        )
        assert unrelated_pair.source_choice_texts is not None

        assert {prompt.messages[-1].content for prompt in same_group} == {record.target}
        assert {prompt.messages[-1].content for prompt in unrelated_group} == {record.target}
        assert record.target in "ABCDE"
        for prompt in same_group:
            user = prompt.messages[1].content
            assert "from functions import" in user
            assert all(choice in user for choice in same_choices)
            assert not any(marker in user for marker in formal_markers)
        for prompt in unrelated_group:
            user = prompt.messages[1].content
            assert "from functions import" not in user
            assert all(choice in user for choice in unrelated_pair.source_choice_texts)
            assert not any(marker in user for marker in formal_markers)


def test_fineweb_activation_rows_are_deterministic_nonoverlapping_windows() -> None:
    rows = fineweb_activation_row_indices(10_000)

    assert rows == fineweb_activation_row_indices(10_000)
    assert len(rows) == FINEWEB_ACTIVATION_DOCUMENT_COUNT
    assert len(set(rows)) == len(rows)
    assert all(
        window == tuple(range(window[0], window[0] + 5))
        for window in (rows[start : start + 5] for start in range(0, len(rows), 5))
    )
    with pytest.raises(ValueError, match="valid population"):
        fineweb_activation_row_indices(100, count=7)


def _fineweb_payload(total_rows: int = 10_000) -> dict[str, object]:
    row_indices = fineweb_activation_row_indices(total_rows)
    documents: list[dict[str, object]] = []
    for ordinal, row_index in enumerate(row_indices):
        text = f"FineWeb document {ordinal} at source row {row_index}."
        documents.append(
            {
                "row_index": row_index,
                "document_id": f"doc-{ordinal}",
                "url": f"https://example.com/{ordinal}",
                "dump": "CC-MAIN-2025-30",
                "date": "2025-07-01T00:00:00Z",
                "language": "en",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "dataset": FINEWEB_DATASET_ID,
        "revision": FINEWEB_DATASET_REVISION,
        "config": FINEWEB_DATASET_CONFIG,
        "split": FINEWEB_DATASET_SPLIT,
        "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
        "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
        "total_rows": total_rows,
        "documents": documents,
    }


def test_activation_example_and_fineweb_document_contracts_fail_loud() -> None:
    messages = (ChatMessage("user", "question"), ChatMessage("assistant", "answer"))
    with pytest.raises(ValueError, match="non-empty"):
        ActivationExamplePrompt("", "category", messages)
    with pytest.raises(ValueError, match="assistant"):
        ActivationExamplePrompt(
            "example",
            "category",
            (ChatMessage("system", "system"), ChatMessage("user", "question")),
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_activation_example_prompts(-1)
    digest = hashlib.sha256(b"text").hexdigest()
    with pytest.raises(ValueError, match="non-negative"):
        FineWebActivationDocument(-1, "doc", "", "", "", "en", "text", digest)
    with pytest.raises(ValueError, match="ID"):
        FineWebActivationDocument(1, "", "", "", "", "en", "text", digest)


@pytest.mark.parametrize(
    ("total_rows", "seed", "count", "window_length"),
    [
        (0, FINEWEB_ACTIVATION_CORPUS_SEED, 95, 5),
        (100, -1, 95, 5),
        (100, FINEWEB_ACTIVATION_CORPUS_SEED, 0, 5),
        (50, FINEWEB_ACTIVATION_CORPUS_SEED, 95, 5),
        (100, FINEWEB_ACTIVATION_CORPUS_SEED, 95, 0),
    ],
)
def test_fineweb_activation_rows_reject_invalid_sampling_contracts(
    total_rows: int,
    seed: int,
    count: int,
    window_length: int,
) -> None:
    with pytest.raises(ValueError, match="valid population"):
        fineweb_activation_row_indices(
            total_rows,
            seed=seed,
            count=count,
            window_length=window_length,
        )


def test_fineweb_activation_corpus_is_provenance_validated(tmp_path) -> None:
    total_rows = 10_000
    row_indices = fineweb_activation_row_indices(total_rows)
    payload = _fineweb_payload(total_rows)
    documents = cast(list[dict[str, object]], payload["documents"])
    path = fineweb_activation_corpus_path(tmp_path)
    write_json(path, payload)

    loaded = load_fineweb_activation_documents(tmp_path)

    assert tuple(document.row_index for document in loaded) == row_indices
    assert loaded[0].provenance["revision"] == FINEWEB_DATASET_REVISION
    documents[0]["text"] = "silently changed"
    write_json(path, payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_fineweb_activation_documents(tmp_path)


def test_fineweb_activation_corpus_rejects_schema_and_row_corruption(tmp_path) -> None:
    path = fineweb_activation_corpus_path(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_fineweb"):
        load_fineweb_activation_documents(tmp_path)
    write_json(path, [])
    with pytest.raises(TypeError, match="object"):
        load_fineweb_activation_documents(tmp_path)

    payload = _fineweb_payload()
    corrupted = copy.deepcopy(payload)
    corrupted["revision"] = "drifted"
    write_json(path, corrupted)
    with pytest.raises(ValueError, match="revision mismatch"):
        load_fineweb_activation_documents(tmp_path)

    corrupted = copy.deepcopy(payload)
    corrupted["total_rows"] = 1
    write_json(path, corrupted)
    with pytest.raises(ValueError, match="total-row"):
        load_fineweb_activation_documents(tmp_path)

    corrupted = copy.deepcopy(payload)
    corrupted["documents"] = []
    write_json(path, corrupted)
    with pytest.raises(ValueError, match="document count"):
        load_fineweb_activation_documents(tmp_path)

    documents = cast(list[object], copy.deepcopy(payload["documents"]))
    documents[0] = "not a row"
    corrupted = {**payload, "documents": documents}
    write_json(path, corrupted)
    with pytest.raises(TypeError, match="must be an object"):
        load_fineweb_activation_documents(tmp_path)

    documents = cast(list[dict[str, object]], copy.deepcopy(payload["documents"]))
    documents[0]["row_index"] = -1
    write_json(path, {**payload, "documents": documents})
    with pytest.raises(ValueError, match="deterministic sample"):
        load_fineweb_activation_documents(tmp_path)

    documents = cast(list[dict[str, object]], copy.deepcopy(payload["documents"]))
    documents[0]["url"] = None
    write_json(path, {**payload, "documents": documents})
    with pytest.raises(TypeError, match="url must be a string"):
        load_fineweb_activation_documents(tmp_path)

    documents = cast(list[dict[str, object]], copy.deepcopy(payload["documents"]))
    documents[1]["document_id"] = documents[0]["document_id"]
    write_json(path, {**payload, "documents": documents})
    with pytest.raises(ValueError, match="repeats a document ID"):
        load_fineweb_activation_documents(tmp_path)


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


def test_top_cosine_examples_clamps_float32_roundoff_to_metric_range() -> None:
    # This width deterministically produces a float32 self-dot slightly above
    # one after normalization on CPU unless the cosine is explicitly clamped.
    vector = t.arange(1, 14, dtype=t.float32).unsqueeze(0)

    matches = _top_cosine_examples(vector, (vector,), top_k=1, device="cpu")

    assert matches == [[{"example_index": 0, "token_index": 0, "cosine_similarity": 1.0}]]


def test_top_cosine_examples_rejects_shape_and_empty_contract_violations() -> None:
    with pytest.raises(ValueError, match="reference"):
        _top_cosine_examples(t.empty((0, 2)), (t.ones((1, 2)),), top_k=1, device="cpu")
    with pytest.raises(ValueError, match="widths"):
        _top_cosine_examples(t.ones((1, 2)), (t.ones((1, 3)),), top_k=1, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        _top_cosine_examples(t.ones((1, 2)), (t.ones((1, 2)),), top_k=0, device="cpu")
