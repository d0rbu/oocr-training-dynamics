from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch as t

from oocr_training_dynamics.activation_examples import (
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_ACTIVATION_MAX_TOKENS,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
)
from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import RunKey, TrainingCondition
from oocr_training_dynamics.letter_propensity import (
    LETTER_PROPENSITY_AGGREGATION,
    LETTER_PROPENSITY_KIND,
    LETTER_PROPENSITY_LABELS,
    LETTER_PROPENSITY_METRIC,
    LETTER_PROPENSITY_NORMALIZATION,
    LETTER_PROPENSITY_POSITION_POLICY,
    LETTER_PROPENSITY_SCHEMA_VERSION,
    letter_propensity_dir,
    letter_propensity_path,
    load_letter_propensity_artifact,
    validate_letter_propensity_artifact,
)
from oocr_training_dynamics.runtime_letter_propensity import (
    _position_answer_probabilities,
    _summarize_position_probabilities,
)


def _valid_artifact(run: RunKey, step: int) -> dict[str, object]:
    per_label = {label: (index + 1) / 10_000 for index, label in enumerate("ABCDE")}
    return {
        "schema_version": LETTER_PROPENSITY_SCHEMA_VERSION,
        "kind": LETTER_PROPENSITY_KIND,
        "metric": LETTER_PROPENSITY_METRIC,
        "model": run.model,
        "condition": run.condition.value,
        "seed": run.seed,
        "effective_batch_size": run.effective_batch_size,
        "lora_rank": run.lora_rank,
        "step": step,
        "examples_seen": step * run.effective_batch_size,
        "answer_labels": list(LETTER_PROPENSITY_LABELS),
        "answer_token_ids": [32, 33, 34, 35, 36],
        "answer_token_texts": list(LETTER_PROPENSITY_LABELS),
        "normalization": LETTER_PROPENSITY_NORMALIZATION,
        "aggregation": LETTER_PROPENSITY_AGGREGATION,
        "position_policy": LETTER_PROPENSITY_POSITION_POLICY,
        "corpus": {
            "dataset": FINEWEB_DATASET_ID,
            "revision": FINEWEB_DATASET_REVISION,
            "config": FINEWEB_DATASET_CONFIG,
            "split": FINEWEB_DATASET_SPLIT,
            "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
            "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
        },
        "tokenization": {
            "input_format": "raw document; no chat template",
            "add_special_tokens": True,
            "exclude_special_targets": True,
            "max_tokens_per_document": FINEWEB_ACTIVATION_MAX_TOKENS,
        },
        "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
        "token_count": 12_000,
        "output_vocabulary_size": 100_352,
        "inference_batch_size": 4,
        "mean_probability_by_label": per_label,
        "mean_letter_probability": sum(per_label.values()),
        "position_probability_stddev": 0.002,
        "wall_time_seconds": 12.5,
        "peak_cuda_memory_bytes": 18_000_000_000,
    }


def test_full_vocabulary_letter_probability_matches_softmax_without_ae_renormalization() -> None:
    logits = t.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0, -2.0, 4.0, 3.0],
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            ]
        ]
    )
    target_mask = t.tensor([[False, True, True]])
    answer_ids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)

    actual = _position_answer_probabilities(logits, target_mask, answer_ids)
    expected = t.softmax(logits[:, :-1], dim=-1)[0, :, :5]

    assert t.allclose(actual, expected)
    assert t.all(actual.sum(dim=1) < 1.0)
    summary = _summarize_position_probabilities(actual)
    assert summary["token_count"] == 2
    assert summary["mean_letter_probability"] == pytest.approx(float(expected.sum(dim=1).mean()))
    per_label = cast(dict[str, float], summary["mean_probability_by_label"])
    assert sum(per_label.values()) == pytest.approx(summary["mean_letter_probability"])


@pytest.mark.parametrize(
    ("logits", "mask", "ids", "message"),
    [
        (t.zeros(1, 1, 7), t.ones(1, 1, dtype=t.bool), t.arange(5), "sequence"),
        (t.zeros(1, 3, 7), t.ones(1, 2, dtype=t.bool), t.arange(5), "mask"),
        (t.zeros(1, 3, 7), t.zeros(1, 3, dtype=t.bool), t.arange(5), "no valid"),
        (
            t.zeros(1, 3, 7),
            t.tensor([[True, False, False]]),
            t.arange(5),
            "no valid",
        ),
        (t.zeros(1, 3, 7), t.ones(1, 3, dtype=t.bool), t.tensor([0, 1, 2, 3, 7]), "IDs"),
    ],
)
def test_letter_probability_kernel_rejects_invalid_axes(
    logits: t.Tensor,
    mask: t.Tensor,
    ids: t.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _position_answer_probabilities(logits, mask, ids)


def test_letter_propensity_artifact_round_trip_and_run_scoped_path(tmp_path: Path) -> None:
    run = RunKey(
        "olmo3-7b",
        TrainingCondition.CORRECT,
        effective_batch_size=16,
    )
    artifact = _valid_artifact(run, 96)
    path = letter_propensity_path(tmp_path, run, 96)
    write_json(path, artifact)

    assert path.parent == letter_propensity_dir(tmp_path, run)
    assert (
        path.relative_to(tmp_path)
        .as_posix()
        .endswith("effective_batch_16/letter_propensity/checkpoint_step_000096.json")
    )
    assert load_letter_propensity_artifact(tmp_path, run, 96) == artifact
    with pytest.raises(ValueError, match="non-negative"):
        letter_propensity_path(tmp_path, run, -1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(metric="wrong"), "metric mismatch"),
        (lambda value: value.update(answer_labels=list("BACDE")), "ordered A-E"),
        (lambda value: value.update(answer_token_ids=[32, 32, 34, 35, 36]), "distinct"),
        (lambda value: value.update(answer_token_texts=list("ABCDX")), "decode exactly"),
        (lambda value: value.update(corpus=None), "corpus must be an object"),
        (
            lambda value: cast(dict[str, object], value["corpus"]).update(revision="wrong"),
            "corpus.revision",
        ),
        (
            lambda value: cast(dict[str, object], value["tokenization"]).update(
                add_special_tokens=False
            ),
            "tokenization",
        ),
        (lambda value: value.update(tokenization=None), "tokenization must be an object"),
        (lambda value: value.update(document_count=1), "frozen FineWeb corpus"),
        (lambda value: value.update(token_count=10), "token count"),
        (lambda value: value.update(inference_batch_size=0), "must be positive"),
        (lambda value: value.update(mean_letter_probability=0.9), "five-label sum"),
        (lambda value: value.update(position_probability_stddev=-1.0), "out of range"),
        (lambda value: value.update(wall_time_seconds=-1.0), "non-negative"),
        (lambda value: value.update(peak_cuda_memory_bytes=-1), "non-negative"),
    ],
)
def test_letter_propensity_artifact_rejects_semantic_corruption(
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    artifact = copy.deepcopy(_valid_artifact(run, 0))
    mutation(artifact)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_letter_propensity_artifact(artifact, run, 0)


def test_letter_propensity_artifact_rejects_non_object_and_malformed_probabilities() -> None:
    run = RunKey("olmo3-7b", TrainingCondition.CORRECT)
    with pytest.raises(TypeError, match="object"):
        validate_letter_propensity_artifact([], run, 0)
    artifact = _valid_artifact(run, 0)
    artifact["mean_probability_by_label"] = {"A": 0.1}
    with pytest.raises(ValueError, match="exactly A-E"):
        validate_letter_propensity_artifact(artifact, run, 0)
    with pytest.raises(ValueError, match="non-empty"):
        _summarize_position_probabilities(t.empty((0, 5)))
