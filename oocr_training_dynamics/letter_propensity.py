"""Artifact contract for token-level standalone A-E propensity on raw FineWeb text."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from beartype import beartype

from oocr_training_dynamics.activation_examples import (
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_ACTIVATION_MAX_TOKENS,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
)
from oocr_training_dynamics.artifacts import read_json, run_dir
from oocr_training_dynamics.contracts import RunKey, checkpoint_label

LETTER_PROPENSITY_SCHEMA_VERSION = 1
LETTER_PROPENSITY_KIND = "general_letter_answer_propensity"
LETTER_PROPENSITY_METRIC = "mean_sum_probability_of_exact_mcq_answer_tokens"
LETTER_PROPENSITY_LABELS = tuple("ABCDE")
LETTER_PROPENSITY_DEFAULT_BATCH_SIZE = 4
LETTER_PROPENSITY_AGGREGATION = (
    "token-weighted arithmetic mean over valid raw-document next-token targets"
)
LETTER_PROPENSITY_NORMALIZATION = "softmax over the complete model output vocabulary"
LETTER_PROPENSITY_POSITION_POLICY = (
    "logits at position t-1 for each non-special, non-padding document token at position t >= 1"
)


@beartype
def letter_propensity_dir(root: Path, run: RunKey) -> Path:
    """Return the run-scoped directory for the lightweight checkpoint sidecars."""

    return run_dir(root, run) / "letter_propensity"


@beartype
def letter_propensity_path(root: Path, run: RunKey, step: int) -> Path:
    """Return one checkpoint's atomic general-letter-propensity artifact path."""

    if step < 0:
        raise ValueError("letter-propensity checkpoint step must be non-negative")
    return letter_propensity_dir(root, run) / f"checkpoint_{checkpoint_label(step)}.json"


def _required_int(value: dict[str, object], key: str, *, context: str) -> int:
    observed = value.get(key)
    if not isinstance(observed, int) or isinstance(observed, bool):
        raise TypeError(f"{context}.{key} must be an integer")
    return observed


def _required_float(value: dict[str, object], key: str, *, context: str) -> float:
    observed = value.get(key)
    if not isinstance(observed, int | float) or isinstance(observed, bool):
        raise TypeError(f"{context}.{key} must be numeric")
    result = float(observed)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be finite")
    return result


@beartype
def validate_letter_propensity_artifact(
    raw: object,
    run: RunKey,
    step: int,
    *,
    context: str = "letter propensity artifact",
) -> dict[str, object]:
    """Fail loudly if a sidecar cannot support the declared full-vocabulary curve."""

    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be an object")
    value = cast(dict[str, object], raw)
    expected_scalars: dict[str, object] = {
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
        "aggregation": LETTER_PROPENSITY_AGGREGATION,
        "normalization": LETTER_PROPENSITY_NORMALIZATION,
        "position_policy": LETTER_PROPENSITY_POSITION_POLICY,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise ValueError(
                f"{context}.{key} mismatch: expected {expected!r}, observed {value.get(key)!r}"
            )

    labels = value.get("answer_labels")
    token_ids = value.get("answer_token_ids")
    token_texts = value.get("answer_token_texts")
    if labels != list(LETTER_PROPENSITY_LABELS):
        raise ValueError(f"{context}.answer_labels must be the ordered A-E labels")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != len(LETTER_PROPENSITY_LABELS)
        or any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids)
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError(f"{context}.answer_token_ids must contain five distinct non-negative IDs")
    if token_texts != list(LETTER_PROPENSITY_LABELS):
        raise ValueError(f"{context}.answer_token_texts must decode exactly to A-E")

    corpus = value.get("corpus")
    expected_corpus = {
        "dataset": FINEWEB_DATASET_ID,
        "revision": FINEWEB_DATASET_REVISION,
        "config": FINEWEB_DATASET_CONFIG,
        "split": FINEWEB_DATASET_SPLIT,
        "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
        "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    }
    if not isinstance(corpus, dict):
        raise TypeError(f"{context}.corpus must be an object")
    for key, expected in expected_corpus.items():
        if corpus.get(key) != expected:
            raise ValueError(
                f"{context}.corpus.{key} mismatch: "
                f"expected {expected!r}, observed {corpus.get(key)!r}"
            )

    tokenization = value.get("tokenization")
    expected_tokenization = {
        "input_format": "raw document; no chat template",
        "add_special_tokens": True,
        "exclude_special_targets": True,
        "max_tokens_per_document": FINEWEB_ACTIVATION_MAX_TOKENS,
    }
    if not isinstance(tokenization, dict):
        raise TypeError(f"{context}.tokenization must be an object")
    for key, expected in expected_tokenization.items():
        if tokenization.get(key) != expected:
            raise ValueError(
                f"{context}.tokenization.{key} mismatch: "
                f"expected {expected!r}, observed {tokenization.get(key)!r}"
            )

    document_count = _required_int(value, "document_count", context=context)
    token_count = _required_int(value, "token_count", context=context)
    vocabulary_size = _required_int(value, "output_vocabulary_size", context=context)
    inference_batch_size = _required_int(value, "inference_batch_size", context=context)
    if document_count != FINEWEB_ACTIVATION_DOCUMENT_COUNT:
        raise ValueError(f"{context}.document_count must cover the frozen FineWeb corpus")
    if token_count <= document_count or vocabulary_size <= max(cast(list[int], token_ids)):
        raise ValueError(f"{context} has an invalid token count or output vocabulary size")
    if inference_batch_size <= 0:
        raise ValueError(f"{context}.inference_batch_size must be positive")

    per_label = value.get("mean_probability_by_label")
    if not isinstance(per_label, dict) or set(per_label) != set(LETTER_PROPENSITY_LABELS):
        raise ValueError(f"{context}.mean_probability_by_label must contain exactly A-E")
    label_values = tuple(
        _required_float(
            cast(dict[str, object], per_label),
            label,
            context=f"{context}.mean_probability_by_label",
        )
        for label in LETTER_PROPENSITY_LABELS
    )
    mean = _required_float(value, "mean_letter_probability", context=context)
    stddev = _required_float(value, "position_probability_stddev", context=context)
    if (
        any(not 0.0 <= probability <= 1.0 for probability in label_values)
        or not 0.0 <= mean <= 1.0
        or not 0.0 <= stddev <= 0.5
    ):
        raise ValueError(f"{context} probabilities or standard deviation lie out of range")
    if not math.isclose(mean, sum(label_values), rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError(f"{context}.mean_letter_probability must equal the five-label sum")

    wall_time = _required_float(value, "wall_time_seconds", context=context)
    peak_memory = _required_int(value, "peak_cuda_memory_bytes", context=context)
    if wall_time < 0 or peak_memory < 0:
        raise ValueError(f"{context} runtime and peak CUDA memory must be non-negative")
    return value


@beartype
def load_letter_propensity_artifact(
    root: Path,
    run: RunKey,
    step: int,
) -> dict[str, object]:
    """Load and validate one measured checkpoint sidecar."""

    path = letter_propensity_path(root, run, step)
    return validate_letter_propensity_artifact(read_json(path), run, step, context=str(path))


__all__ = [
    "LETTER_PROPENSITY_AGGREGATION",
    "LETTER_PROPENSITY_DEFAULT_BATCH_SIZE",
    "LETTER_PROPENSITY_KIND",
    "LETTER_PROPENSITY_LABELS",
    "LETTER_PROPENSITY_METRIC",
    "LETTER_PROPENSITY_NORMALIZATION",
    "LETTER_PROPENSITY_POSITION_POLICY",
    "LETTER_PROPENSITY_SCHEMA_VERSION",
    "letter_propensity_dir",
    "letter_propensity_path",
    "load_letter_propensity_artifact",
    "validate_letter_propensity_artifact",
]
