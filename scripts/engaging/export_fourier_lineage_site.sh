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
schema_version = payload.get("schema_version")
lineage = payload.get("lineage", {})
if (
    schema_version not in {1, 2}
    or payload.get("kind") != "fourier_lineage_export"
    or lineage.get("id") != lineage_id
    or lineage.get("kind") != "registered_hardware"
    or not payload.get("entries")
):
    raise RuntimeError("Fourier lineage export manifest failed its identity gate")
if schema_version == 2 and lineage.get("plan_sha256") is not None:
    raise RuntimeError("multi-plan Fourier lineage manifest retained a run-specific plan digest")
stable_fields = (
    "id",
    "kind",
    "display_name",
    "hardware",
    "reference_source_bundle_sha256",
    "collection_source_bundle_sha256",
)
for entry in payload["entries"]:
    entry_lineage = entry.get("lineage", {})
    same_lineage = (
        entry_lineage == lineage
        if schema_version == 1
        else all(entry_lineage.get(field) == lineage.get(field) for field in stable_fields)
    )
    if not same_lineage:
        raise RuntimeError("Fourier lineage export entry changed lineage")
    if schema_version == 2 and not isinstance(entry_lineage.get("plan_sha256"), str):
        raise RuntimeError("multi-plan Fourier lineage entry lacks its plan digest")
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
