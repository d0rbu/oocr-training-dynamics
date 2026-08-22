#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_LINEAGE_ROOT:?OOCR_LINEAGE_ROOT is required}"
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
        f"H200 recall requires exactly one CUDA-visible device; found "
        f"{torch.cuda.device_count()}"
    )
print(torch.cuda.get_device_name(0))
PY
)"
if [[ "${device_name}" != "NVIDIA H200" ]]; then
  echo "H200 recall allocated an unexpected CUDA-visible device: ${device_name}" >&2
  exit 2
fi

step_padded="$(printf '%06d' "${OOCR_CLEAN_STEP}")"
plan="artifacts/plans/fourier_hardware_lineages/${OOCR_LINEAGE_ID}_${OOCR_FUNCTION_ID}_step_${step_padded}.json"
test -f "${plan}"

child_pid=''
stop_child() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}"
  fi
}
trap stop_child TERM INT

"${repo}/.venv/bin/python" scripts/run_fourier_full_recall.py \
  --function-id "${OOCR_FUNCTION_ID}" \
  --clean-step "${OOCR_CLEAN_STEP}" \
  --maximum-initial-evaluations 120000 \
  --maximum-network-evaluations-per-order 5000000 \
  --maximum-component-shell-pair-evaluations 500000 \
  --lineage-plan "${plan}" \
  --confirm-gpu-run &
child_pid=$!
wait "${child_pid}"
