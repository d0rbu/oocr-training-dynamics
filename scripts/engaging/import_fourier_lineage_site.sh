#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_REMOTE_HOST:?OOCR_REMOTE_HOST is required}"
: "${OOCR_REMOTE_EXPORT_ROOT:?OOCR_REMOTE_EXPORT_ROOT is required}"
: "${OOCR_LINEAGE_ID:?OOCR_LINEAGE_ID is required}"

local_root="$(realpath -e "${OOCR_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}")"
python="${local_root}/.venv/bin/python"
test -x "${python}"
test -f "${local_root}/scripts/export_fourier_site.py"

temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT
remote_manifest="${OOCR_REMOTE_EXPORT_ROOT}/site/data/fourier-circuit-lineages/${OOCR_LINEAGE_ID}.json"
rsync --archive --checksum \
  "${OOCR_REMOTE_HOST}:${remote_manifest}" \
  "${temporary}/manifest.json"

"${python}" - "${temporary}/manifest.json" "${OOCR_LINEAGE_ID}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lineage_id = sys.argv[2]
prefix = f"fourier-circuits/lineage_{lineage_id}/"
if (
    manifest.get("schema_version") != 1
    or manifest.get("kind") != "fourier_lineage_export"
    or manifest.get("lineage", {}).get("id") != lineage_id
    or manifest.get("lineage", {}).get("kind") != "registered_hardware"
    or not manifest.get("entries")
):
    raise RuntimeError("remote Fourier lineage manifest failed its identity gate")
for entry in manifest["entries"]:
    relative = entry.get("url")
    if (
        entry.get("lineage") != manifest["lineage"]
        or not isinstance(relative, str)
        or not relative.startswith(prefix)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise RuntimeError("remote Fourier lineage manifest contains an unsafe entry")
PY

local_chunks="${local_root}/site/data/fourier-circuits/lineage_${OOCR_LINEAGE_ID}"
mkdir -p "${local_chunks}"
rsync --archive --checksum --partial \
  "${OOCR_REMOTE_HOST}:${OOCR_REMOTE_EXPORT_ROOT}/site/data/fourier-circuits/lineage_${OOCR_LINEAGE_ID}/" \
  "${local_chunks}/"

"${python}" - "${temporary}/manifest.json" "${local_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
for entry in manifest["entries"]:
    chunk = root / "site/data" / entry["url"]
    body = chunk.read_bytes()
    if len(body) != entry["bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"synced Fourier chunk failed its digest gate: {chunk}")
PY

imports="${local_root}/site/data/fourier-circuit-imports"
mkdir -p "${imports}"
install -m 0644 "${temporary}/manifest.json" "${imports}/${OOCR_LINEAGE_ID}.json.next"
mv "${imports}/${OOCR_LINEAGE_ID}.json.next" "${imports}/${OOCR_LINEAGE_ID}.json"
CUDA_VISIBLE_DEVICES='' "${python}" "${local_root}/scripts/export_fourier_site.py" \
  --root "${local_root}"
