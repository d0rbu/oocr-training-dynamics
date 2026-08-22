#!/usr/bin/env python3
"""Refresh only the measured Fourier chunks and their two static-site manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from oocr_training_dynamics.artifacts import read_json, write_json
from scripts.export_site import _export_fourier_circuits


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a string-keyed object")
    return cast(dict[str, object], value)


def refresh_fourier_site(root: Path) -> None:
    manifest, real_file_count = _export_fourier_circuits(root)
    for filename in ("experiment.json", "patch-manifest.json"):
        path = root / "site" / "data" / filename
        payload = _mapping(read_json(path), context=str(path))
        payload["fourier_circuit_manifest"] = manifest
        payload["real_fourier_circuit_files"] = real_file_count
        write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root whose raw artifacts and static site should be refreshed.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if not (root / "site/data/experiment.json").is_file():
        raise FileNotFoundError(f"Fourier export root lacks its static-site manifest: {root}")
    refresh_fourier_site(root)


if __name__ == "__main__":
    main()
