#!/usr/bin/env python3
"""Fetch the frozen deterministic FineWeb corpus for activation-neighbor search."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

from oocr_training_dynamics.activation_examples import (
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_ACTIVATION_WINDOW_LENGTH,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
    fineweb_activation_corpus_path,
    fineweb_activation_row_indices,
    load_fineweb_activation_documents,
)
from oocr_training_dynamics.artifacts import write_json

HUGGING_FACE_DATASET_API = f"https://huggingface.co/api/datasets/{FINEWEB_DATASET_ID}"
DATASET_VIEWER_ROWS_API = "https://datasets-server.huggingface.co/rows"


def _get_json(url: str, *, attempts: int = 5) -> object:
    """Read one JSON response with bounded retry for transient service failures."""

    if attempts <= 0:
        raise ValueError("HTTP attempts must be positive")
    headers = {"User-Agent": "oocr-training-dynamics/activation-neighbor-audit"}
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise RuntimeError(f"GET {url} returned HTTP {response.status}")
                return json.loads(response.read())
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {
                429,
                500,
                502,
                503,
                504,
            }
            if not retryable or attempt + 1 == attempts:
                raise RuntimeError(f"could not fetch {url}") from error
            retry_after = (
                error.headers.get("Retry-After")
                if isinstance(error, urllib.error.HTTPError)
                else None
            )
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(min(max(delay, 1), 45))
    raise AssertionError("bounded HTTP retry loop did not return or raise")


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return cast(dict[str, Any], value)


def _rows_url(*, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": FINEWEB_DATASET_ID,
            "config": FINEWEB_DATASET_CONFIG,
            "split": FINEWEB_DATASET_SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"{DATASET_VIEWER_ROWS_API}?{query}"


def _fineweb_documents(row_indices: tuple[int, ...]) -> list[dict[str, object]]:
    if not row_indices or tuple(range(row_indices[0], row_indices[-1] + 1)) != row_indices:
        raise ValueError("FineWeb fetch windows must be non-empty and contiguous")
    payload = _mapping(
        _get_json(_rows_url(offset=row_indices[0], length=len(row_indices))),
        context="rows API",
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(row_indices):
        raise RuntimeError(f"FineWeb rows API did not return window {row_indices}")
    documents: list[dict[str, object]] = []
    for expected_row_index, raw_wrapped in zip(row_indices, rows, strict=True):
        wrapped = _mapping(
            raw_wrapped,
            context=f"FineWeb row wrapper {expected_row_index}",
        )
        if wrapped.get("row_idx") != expected_row_index:
            raise RuntimeError(
                f"FineWeb rows API returned the wrong row for offset {expected_row_index}"
            )
        row = _mapping(wrapped.get("row"), context=f"FineWeb row {expected_row_index}")
        required_strings = ("id", "url", "dump", "date", "language", "text")
        if any(not isinstance(row.get(key), str) for key in required_strings):
            raise TypeError(f"FineWeb row {expected_row_index} lacks required source metadata")
        text = cast(str, row["text"])
        if not text:
            raise ValueError(f"FineWeb row {expected_row_index} has empty text")
        documents.append(
            {
                "row_index": expected_row_index,
                "document_id": row["id"],
                "url": row["url"],
                "dump": row["dump"],
                "date": row["date"],
                "language": row["language"],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return documents


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = fineweb_activation_corpus_path(root)
    if output.is_file():
        documents = load_fineweb_activation_documents(root)
        print(f"[fineweb] validated existing {len(documents)}-document corpus: {output}")
        return

    dataset_info = _mapping(_get_json(HUGGING_FACE_DATASET_API), context="dataset API")
    observed_revision = dataset_info.get("sha")
    if observed_revision != FINEWEB_DATASET_REVISION:
        raise RuntimeError(
            "FineWeb current revision changed; review and explicitly update the frozen corpus "
            f"contract before fetching (expected {FINEWEB_DATASET_REVISION}, "
            f"observed {observed_revision})"
        )
    first_page = _mapping(_get_json(_rows_url(offset=0, length=1)), context="rows API")
    total_rows = first_page.get("num_rows_total")
    if not isinstance(total_rows, int) or total_rows < FINEWEB_ACTIVATION_DOCUMENT_COUNT:
        raise RuntimeError("FineWeb rows API returned an invalid total-row count")
    row_indices = fineweb_activation_row_indices(total_rows)
    documents: list[dict[str, object]] = []
    for start in range(0, len(row_indices), FINEWEB_ACTIVATION_WINDOW_LENGTH):
        window = row_indices[start : start + FINEWEB_ACTIVATION_WINDOW_LENGTH]
        documents.extend(_fineweb_documents(window))
        print(f"[fineweb] fetched {len(documents)}/{len(row_indices)} documents", flush=True)
    write_json(
        output,
        {
            "schema_version": 1,
            "dataset": FINEWEB_DATASET_ID,
            "revision": FINEWEB_DATASET_REVISION,
            "config": FINEWEB_DATASET_CONFIG,
            "split": FINEWEB_DATASET_SPLIT,
            "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
            "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
            "total_rows": total_rows,
            "sampling": (
                "Python random.Random(seed).sample of 19 non-overlapping, aligned "
                "five-row windows over Dataset Viewer row indices; sample order is retained"
            ),
            "documents": documents,
        },
    )
    validated = load_fineweb_activation_documents(root)
    print(f"[fineweb] wrote and validated {len(validated)} documents: {output}")


if __name__ == "__main__":
    main()
