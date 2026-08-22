#!/usr/bin/env python3
"""Refresh only the compact switched-answer manifest in both live site snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, write_json
from scripts.export_site import _export_switched_answer_minsets


def refresh_switched_answer_minset_site(root: Path) -> None:
    manifest, measured = _export_switched_answer_minsets(root)
    for relative in ("site/data/experiment.json", "site/data/patch-manifest.json"):
        path = root / relative
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"site snapshot must be an object: {path}")
        mutable = cast(dict[str, object], payload)
        mutable["switched_answer_minset_manifest"] = manifest
        mutable["real_switched_answer_minset_files"] = measured
        write_json(path, mutable)


if __name__ == "__main__":
    refresh_switched_answer_minset_site(Path(__file__).resolve().parents[1])
