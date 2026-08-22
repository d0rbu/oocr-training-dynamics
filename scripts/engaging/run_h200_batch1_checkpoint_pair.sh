#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_LINEAGE_ROOT:?OOCR_LINEAGE_ROOT is required}"
: "${OOCR_TASK0_FUNCTION_ID:?OOCR_TASK0_FUNCTION_ID is required}"
: "${OOCR_TASK0_CLEAN_STEP:?OOCR_TASK0_CLEAN_STEP is required}"
: "${OOCR_TASK1_FUNCTION_ID:?OOCR_TASK1_FUNCTION_ID is required}"
: "${OOCR_TASK1_CLEAN_STEP:?OOCR_TASK1_CLEAN_STEP is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

lineage_root="$(realpath -e "${OOCR_LINEAGE_ROOT}")"
checkpoint_runner="${lineage_root}/ops/run_h200_batch1_checkpoint_stage0.sh"
logs="${lineage_root}/logs"
test -x "${checkpoint_runner}"
mkdir -p "${logs}"

IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#devices[@]} != 2 )); then
  echo "paired H200 bootstrap requires exactly two Slurm-visible devices" >&2
  exit 2
fi

task0_log="${logs}/paired-${SLURM_JOB_ID}-task0-${OOCR_TASK0_FUNCTION_ID}-${OOCR_TASK0_CLEAN_STEP}.out"
task1_log="${logs}/paired-${SLURM_JOB_ID}-task1-${OOCR_TASK1_FUNCTION_ID}-${OOCR_TASK1_CLEAN_STEP}.out"

child_pids=()
stop_children() {
  local pid
  for pid in "${child_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}"
    fi
  done
}
trap stop_children TERM INT

(
  export CUDA_VISIBLE_DEVICES="${devices[0]}"
  export OOCR_FUNCTION_ID="${OOCR_TASK0_FUNCTION_ID}"
  export OOCR_CLEAN_STEP="${OOCR_TASK0_CLEAN_STEP}"
  exec "${checkpoint_runner}"
) >"${task0_log}" 2>&1 &
pid0=$!
child_pids+=("${pid0}")

(
  export CUDA_VISIBLE_DEVICES="${devices[1]}"
  export OOCR_FUNCTION_ID="${OOCR_TASK1_FUNCTION_ID}"
  export OOCR_CLEAN_STEP="${OOCR_TASK1_CLEAN_STEP}"
  exec "${checkpoint_runner}"
) >"${task1_log}" 2>&1 &
pid1=$!
child_pids+=("${pid1}")

set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

printf 'task0 status=%d log=%s\n' "${status0}" "${task0_log}"
printf 'task1 status=%d log=%s\n' "${status1}" "${task1_log}"
if (( status0 != 0 || status1 != 0 )); then
  tail -80 "${task0_log}" "${task1_log}" >&2
  exit 1
fi
