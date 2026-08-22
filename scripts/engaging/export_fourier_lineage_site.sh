#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_EXPORTER_ROOT:?OOCR_EXPORTER_ROOT is required}"
: "${OOCR_SCIENCE_ROOT:?OOCR_SCIENCE_ROOT is required}"
: "${OOCR_PROJECTION_ROOT:?OOCR_PROJECTION_ROOT is required}"
: "${OOCR_LINEAGE_ID:?OOCR_LINEAGE_ID is required}"

exporter_root="$(realpath -e "${OOCR_EXPORTER_ROOT}")"
science_root="$(realpath -e "${OOCR_SCIENCE_ROOT}")"
projection_root="$(realpath -e "${OOCR_PROJECTION_ROOT}")"
python="${science_root}/.venv/bin/python"
test -x "${python}"
test -f "${exporter_root}/scripts/export_fourier_site.py"
test -f "${projection_root}/site/data/experiment.json"
if [[ "$(realpath -e "${projection_root}/artifacts")" != "$(realpath -e "${science_root}/artifacts")" ]]; then
  echo "projection root does not point at the frozen science artifacts" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="${exporter_root}${PYTHONPATH:+:${PYTHONPATH}}"
"${python}" "${exporter_root}/scripts/export_fourier_site.py" --root "${projection_root}"

manifest="${projection_root}/site/data/fourier-circuit-lineages/${OOCR_LINEAGE_ID}.json"
test -s "${manifest}"
"${python}" - "${manifest}" "${projection_root}" "${OOCR_LINEAGE_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
lineage_id = sys.argv[3]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    payload.get("schema_version") != 1
    or payload.get("kind") != "fourier_lineage_export"
    or payload.get("lineage", {}).get("id") != lineage_id
    or payload.get("lineage", {}).get("kind") != "registered_hardware"
    or not payload.get("entries")
):
    raise RuntimeError("Fourier lineage export manifest failed its identity gate")
for entry in payload["entries"]:
    if entry.get("lineage") != payload["lineage"]:
        raise RuntimeError("Fourier lineage export entry changed lineage")
    relative = Path(entry["url"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Fourier lineage export contains an unsafe chunk path")
    chunk = root / "site/data" / relative
    body = chunk.read_bytes()
    if len(body) != entry["bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
        raise RuntimeError("Fourier lineage export chunk failed its digest gate")
print(json.dumps({
    "lineage_id": lineage_id,
    "entry_count": len(payload["entries"]),
    "exporter_source_sha256": payload["exporter_source_sha256"],
    "manifest": str(manifest_path),
}, indent=2, sort_keys=True))
PY
