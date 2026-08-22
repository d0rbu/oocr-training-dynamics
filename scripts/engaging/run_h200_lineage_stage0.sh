#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_LINEAGE_ROOT:?OOCR_LINEAGE_ROOT is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"

lineage_root="$(realpath -e "${OOCR_LINEAGE_ROOT}")"
repo="${lineage_root}/repo"
plan="${repo}/artifacts/plans/fourier_hardware_lineages/engaging_h200_sm90_add_5_step_000128.json"
test -f "${repo}/.gpu-runs-enabled"
test -f "${plan}"
device_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${device_name}" != "NVIDIA H200" ]]; then
  echo "H200 lineage allocated an unexpected device: ${device_name}" >&2
  exit 2
fi
cd "${repo}"

/home/henryac/.local/bin/uv run python scripts/run_fourier_circuits.py \
  --function-id add_5 \
  --clean-step 128 \
  --dirty-step 0 \
  --stages 0 \
  --layer-window 0:32 \
  --sufficiency-rule clean-probability-minus-0.10 \
  --lineage-plan "${plan}" \
  --confirm-gpu-run
