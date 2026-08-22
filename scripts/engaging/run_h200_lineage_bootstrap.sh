#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_LINEAGE_ROOT:?OOCR_LINEAGE_ROOT is required}"
: "${OOCR_ARTIFACT_IDENTITY_ROOT:?OOCR_ARTIFACT_IDENTITY_ROOT is required}"
: "${OOCR_BUNDLE_SHA256:?OOCR_BUNDLE_SHA256 is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"

lineage_root="$(realpath -e "${OOCR_LINEAGE_ROOT}")"
repo="${lineage_root}/repo"
test -f "${repo}/.gpu-runs-enabled"
test -x /home/henryac/.local/bin/uv
cd "${repo}"

source_files_sha="$({
  LC_ALL=C find oocr_training_dynamics scripts -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | cut -d' ' -f1
})"
observed_bundle_sha="$({
  printf '%s  %s\n' "${source_files_sha}" source-files
  sha256sum pyproject.toml uv.lock README.md
} | sha256sum | cut -d' ' -f1)"
if [[ "${observed_bundle_sha}" != "${OOCR_BUNDLE_SHA256}" ]]; then
  echo "H200 lineage source bundle changed after staging" >&2
  exit 2
fi

device_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${device_name}" != "NVIDIA H200" ]]; then
  echo "H200 lineage allocated an unexpected device: ${device_name}" >&2
  exit 2
fi

reference_relative="artifacts/runs/olmo3-7b/correct/seed_20260715/patching/sequence_end/later_checkpoint/recipient_step_000000/donor_step_000128.json"
plan_relative="artifacts/plans/fourier_hardware_lineages/engaging_h200_sm90_add_5_step_000128.json"
grid_bundle_path="${lineage_root}/grid_source_bundle.sha256"

if [[ ! -f "${reference_relative}" ]]; then
  if [[ -e "${grid_bundle_path}" ]]; then
    echo "grid source digest exists before its checkpoint-transfer artifact" >&2
    exit 2
  fi
  printf '%s\n' "${OOCR_BUNDLE_SHA256}" > "${grid_bundle_path}"
  /home/henryac/.local/bin/uv run python scripts/run_patching.py \
    --model olmo3-7b \
    --condition correct \
    --mode later_checkpoint \
    --interface resid_post \
    --recipient-step 0 \
    --donor-step 128 \
    --confirm-gpu-run
fi
if [[ ! -f "${grid_bundle_path}" ]]; then
  echo "checkpoint-transfer grid lacks its immutable source-bundle digest" >&2
  exit 2
fi
grid_bundle_sha="$(cat "${grid_bundle_path}")"

if [[ ! -f "${plan_relative}" ]]; then
  /home/henryac/.local/bin/uv run python scripts/register_fourier_hardware_lineage.py \
    --lineage-id engaging_h200_sm90 \
    --artifact-identity-root "${OOCR_ARTIFACT_IDENTITY_ROOT}" \
    --reference-source-bundle-sha256 "${grid_bundle_sha}" \
    --collection-source-bundle-sha256 "${OOCR_BUNDLE_SHA256}" \
    --function-id add_5 \
    --clean-step 128 \
    --dirty-step 0 \
    --reference-relative-path "${reference_relative}" \
    --output "${plan_relative}" \
    --required-device-name "NVIDIA H200" \
    --confirm-gpu-run
fi

/home/henryac/.local/bin/uv run python - "${plan_relative}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("status") != "registered_before_fourier_collection"
    or payload.get("lineage_id") != "engaging_h200_sm90"
    or payload.get("function_id") != "add_5"
    or payload.get("clean_step") != 128
    or payload.get("dirty_step") != 0
):
    raise RuntimeError("stored H200 lineage plan failed its identity check")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
