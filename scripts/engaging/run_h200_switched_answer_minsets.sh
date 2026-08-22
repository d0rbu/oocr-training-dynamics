#!/usr/bin/env bash
set -euo pipefail

: "${OOCR_SWITCHED_ROOT:?OOCR_SWITCHED_ROOT is required}"
: "${OOCR_BUNDLE_SHA256:?OOCR_BUNDLE_SHA256 is required}"
: "${OOCR_MAXIMUM_STAGE:?OOCR_MAXIMUM_STAGE is required}"
: "${OOCR_MAXIMUM_ORDER:?OOCR_MAXIMUM_ORDER is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"

case "${OOCR_MAXIMUM_STAGE}" in
  0|1|2) ;;
  *) echo "OOCR_MAXIMUM_STAGE must be 0, 1, or 2" >&2; exit 2 ;;
esac
if [[ ! "${OOCR_MAXIMUM_ORDER}" =~ ^[1-6]$ ]]; then
  echo "OOCR_MAXIMUM_ORDER must be an integer from 1 through 6" >&2
  exit 2
fi

lineage_root="$(realpath -e "${OOCR_SWITCHED_ROOT}")"
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
  echo "switched-answer H200 source bundle changed after staging" >&2
  exit 2
fi

device_name="$("${repo}/.venv/bin/python" - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "switched-answer collection requires exactly one CUDA-visible device; "
        f"found {torch.cuda.device_count()}"
    )
print(torch.cuda.get_device_name(0))
PY
)"
if [[ "${device_name}" != "NVIDIA H200" ]]; then
  echo "switched-answer collection allocated an unexpected device: ${device_name}" >&2
  exit 2
fi

child_pid=''
stop_child() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}"
  fi
}
requeue_job() {
  trap - USR1
  stop_child
  set +e
  if [[ -n "${child_pid}" ]]; then
    wait "${child_pid}" 2>/dev/null
  fi
  set -e
  printf 'requeueing switched-answer job %s after pre-timeout signal\n' "${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap stop_child TERM INT
trap requeue_job USR1

"${repo}/.venv/bin/python" scripts/run_switched_answer_minsets.py \
  --maximum-stage "${OOCR_MAXIMUM_STAGE}" \
  --maximum-order "${OOCR_MAXIMUM_ORDER}" \
  --confirm-gpu-run &
child_pid=$!
wait "${child_pid}"
