#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_REMOTE_ROOT:?OOCR_REMOTE_ROOT is required}"
: "${OOCR_ARTIFACT_IDENTITY_ROOT:?OOCR_ARTIFACT_IDENTITY_ROOT is required}"
: "${OOCR_PARITY_RESULT:?OOCR_PARITY_RESULT is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

remote_root="$(realpath -e "${OOCR_REMOTE_ROOT}")"
repo="${remote_root}/repo"
parity_result="$(realpath -e "${OOCR_PARITY_RESULT}")"
steps=(128 256 384 192 768 512 1280 96)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#steps[@]} )); then
  echo "invalid Fourier array task index: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
clean_step="${steps[SLURM_ARRAY_TASK_ID]}"

test -f "${repo}/.gpu-runs-enabled"
test -f "${parity_result}"
cd "${repo}"
/home/henryac/.local/bin/uv run python - "${parity_result}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("scientific_backend") != (
    "full_sequence_reference_use_cache_false_batch_one"
):
    raise RuntimeError("scientific collection requires a passed full-sequence parity gate")
PY

child_pid=""
requeue_requested=0

stop_after_checkpoint() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -INT "${child_pid}"
  fi
}
requeue_after_checkpoint() {
  requeue_requested=1
  stop_after_checkpoint
}
trap requeue_after_checkpoint USR1
trap stop_after_checkpoint TERM INT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/henryac/.local/bin/uv run python scripts/run_fourier_full_recall.py \
  --function-id add_5 \
  --clean-step "${clean_step}" \
  --maximum-initial-evaluations 120000 \
  --maximum-network-evaluations-per-order 5000000 \
  --maximum-component-shell-pair-evaluations 500000 \
  --artifact-identity-root "${OOCR_ARTIFACT_IDENTITY_ROOT}" \
  --confirm-gpu-run &
child_pid=$!
set +e
wait "${child_pid}"
status=$?
set -e

if (( requeue_requested == 1 )); then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${status}"
