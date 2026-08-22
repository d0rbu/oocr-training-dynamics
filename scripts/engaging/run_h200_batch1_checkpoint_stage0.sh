#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_LINEAGE_ROOT:?OOCR_LINEAGE_ROOT is required}"
: "${OOCR_ARTIFACT_IDENTITY_ROOT:?OOCR_ARTIFACT_IDENTITY_ROOT is required}"
: "${OOCR_BUNDLE_SHA256:?OOCR_BUNDLE_SHA256 is required}"
: "${OOCR_LINEAGE_ID:?OOCR_LINEAGE_ID is required}"
: "${OOCR_FUNCTION_ID:?OOCR_FUNCTION_ID is required}"
: "${OOCR_CLEAN_STEP:?OOCR_CLEAN_STEP is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"

case "${OOCR_FUNCTION_ID}" in
  add_5|identity) ;;
  *) echo "unsupported function id: ${OOCR_FUNCTION_ID}" >&2; exit 2 ;;
esac
if [[ ! "${OOCR_CLEAN_STEP}" =~ ^[0-9]+$ ]] || (( OOCR_CLEAN_STEP <= 0 )); then
  echo "OOCR_CLEAN_STEP must be a positive integer" >&2
  exit 2
fi

lineage_root="$(realpath -e "${OOCR_LINEAGE_ROOT}")"
repo="${lineage_root}/repo"
reference_bundle_sha256="${OOCR_REFERENCE_BUNDLE_SHA256:-${OOCR_BUNDLE_SHA256}}"
if [[ ! "${reference_bundle_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "OOCR reference source bundle digest must be a lowercase SHA-256 digest" >&2
  exit 2
fi
test -f "${repo}/.gpu-runs-enabled"
cd "${repo}"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"

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

device_name="$("${repo}/.venv/bin/python" - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"H200 lineage requires exactly one CUDA-visible device; found "
        f"{torch.cuda.device_count()}"
    )
print(torch.cuda.get_device_name(0))
PY
)"
if [[ "${device_name}" != "NVIDIA H200" ]]; then
  echo "H200 lineage allocated an unexpected CUDA-visible device: ${device_name}" >&2
  exit 2
fi

step_padded="$(printf '%06d' "${OOCR_CLEAN_STEP}")"
adapter_dir="artifacts/runs/olmo3-7b/correct/seed_20260715/checkpoints/step_${step_padded}/adapter"
test -f "${adapter_dir}/adapter_config.json"
test -f "${adapter_dir}/adapter_model.safetensors"

reference_relative="artifacts/runs/olmo3-7b/correct/seed_20260715/patching/sequence_end/later_checkpoint/recipient_step_000000/donor_step_${step_padded}.json"
plan_relative="artifacts/plans/fourier_hardware_lineages/${OOCR_LINEAGE_ID}_${OOCR_FUNCTION_ID}_step_${step_padded}.json"

if [[ ! -f "${reference_relative}" ]]; then
  "${repo}/.venv/bin/python" scripts/run_patching.py \
    --model olmo3-7b \
    --condition correct \
    --mode later_checkpoint \
    --interface resid_post \
    --activation-patch-batch-size 1 \
    --deterministic-algorithms \
    --recipient-step 0 \
    --donor-step "${OOCR_CLEAN_STEP}" \
    --confirm-gpu-run
fi

if [[ ! -f "${plan_relative}" ]]; then
  "${repo}/.venv/bin/python" scripts/register_fourier_hardware_lineage.py \
    --lineage-id "${OOCR_LINEAGE_ID}" \
    --artifact-identity-root "${OOCR_ARTIFACT_IDENTITY_ROOT}" \
    --reference-source-bundle-sha256 "${reference_bundle_sha256}" \
    --collection-source-bundle-sha256 "${OOCR_BUNDLE_SHA256}" \
    --function-id "${OOCR_FUNCTION_ID}" \
    --clean-step "${OOCR_CLEAN_STEP}" \
    --dirty-step 0 \
    --reference-relative-path "${reference_relative}" \
    --output "${plan_relative}" \
    --required-device-name "NVIDIA H200" \
    --confirm-gpu-run
fi

"${repo}/.venv/bin/python" scripts/run_fourier_circuits.py \
  --function-id "${OOCR_FUNCTION_ID}" \
  --clean-step "${OOCR_CLEAN_STEP}" \
  --dirty-step 0 \
  --stages 0 \
  --layer-window 0:32 \
  --sufficiency-rule clean-probability-minus-0.10 \
  --lineage-plan "${plan_relative}" \
  --confirm-gpu-run
