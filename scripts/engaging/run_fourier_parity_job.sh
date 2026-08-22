#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_REMOTE_ROOT:?OOCR_REMOTE_ROOT is required}"
: "${OOCR_ARTIFACT_IDENTITY_ROOT:?OOCR_ARTIFACT_IDENTITY_ROOT is required}"
: "${OOCR_BUNDLE_SHA256:?OOCR_BUNDLE_SHA256 is required}"
: "${OOCR_PARITY_REFERENCE:?OOCR_PARITY_REFERENCE is required}"
: "${OOCR_PARITY_METADATA:?OOCR_PARITY_METADATA is required}"
: "${OOCR_PARITY_OUTPUT:?OOCR_PARITY_OUTPUT is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"

remote_root="$(realpath -e "${OOCR_REMOTE_ROOT}")"
repo="${remote_root}/repo"
parity_reference="$(realpath -e "${OOCR_PARITY_REFERENCE}")"
parity_metadata="$(realpath -e "${OOCR_PARITY_METADATA}")"
parity_output="$(realpath -m "${OOCR_PARITY_OUTPUT}")"
test -f "${repo}/.gpu-runs-enabled"
test -f "${parity_reference}"
test -f "${parity_metadata}"
test -x /home/henryac/.local/bin/uv
cd "${repo}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/henryac/.local/bin/uv run python scripts/run_fourier_hardware_parity.py \
  --function-id add_5 \
  --clean-step 128 \
  --reference-sidecar "${parity_reference}" \
  --reference-metadata "${parity_metadata}" \
  --mask-count 64 \
  --artifact-identity-root "${OOCR_ARTIFACT_IDENTITY_ROOT}" \
  --output-dir "${parity_output}" \
  --source-bundle-sha256 "${OOCR_BUNDLE_SHA256}" \
  --confirm-gpu-run
