#!/usr/bin/env python3
"""Refresh only answer-location chunks and their two static-site manifests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, write_json
from scripts.export_site import _export_answer_lookup


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a string-keyed object")
    return cast(dict[str, object], value)


def refresh_answer_lookup_site(root: Path) -> None:
    manifest, real_file_count, complete_file_count = _export_answer_lookup(root)
    for filename in ("experiment.json", "patch-manifest.json"):
        path = root / "site" / "data" / filename
        payload = _mapping(read_json(path), context=str(path))
        payload["answer_lookup_manifest"] = manifest
        payload["real_answer_lookup_files"] = real_file_count
        payload["complete_answer_lookup_files"] = complete_file_count
        write_json(path, payload)


def main() -> None:
    refresh_answer_lookup_site(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
