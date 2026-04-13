#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DEFAULT_CHECKPOINT="/root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
MAX_SECONDS="${2:-20}"
SIM_MAX_SECONDS=$((MAX_SECONDS + 12))

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

ONNX_PATH="${CHECKPOINT%.pt}.onnx"
if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ONNX model not found alongside checkpoint: ${ONNX_PATH}" >&2
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR="${ROOT_DIR}/logs/WholeBodyTrackingOfficial/${TIMESTAMP}-g1_29dof_wbt_unitree_mujoco_official"
VIDEO_DIR="${RUN_DIR}/renderings_training"
SIM_LOG="${RUN_DIR}/sim.log"
POLICY_LOG="${RUN_DIR}/policy.log"
VIDEO_PATH="${VIDEO_DIR}/episode_1.mp4"
mkdir -p "${VIDEO_DIR}"

SIM_PID=""

cleanup() {
    if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
        kill -TERM "${SIM_PID}" 2>/dev/null || true
        wait "${SIM_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "Run directory: ${RUN_DIR}"
echo "Video path: ${VIDEO_PATH}"
echo "Model: ${ONNX_PATH}"
echo "Simulator max runtime: ${SIM_MAX_SECONDS}s"

/usr/bin/zsh -lc "
source "${ROOT_DIR}/scripts/source_mujoco_setup.sh"
export MUJOCO_GL=egl
python "${ROOT_DIR}/scripts/run_wbt_unitree_mujoco_sim.py" \
    --interface=lo \
    --domain-id=0 \
    --max-runtime-seconds=${SIM_MAX_SECONDS} \
    --output='${VIDEO_PATH}'
" >"${SIM_LOG}" 2>&1 &
SIM_PID=$!

sleep 5

if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "unitree_mujoco simulator exited early. See ${SIM_LOG}" >&2
    exit 1
fi

set +e
/usr/bin/zsh -lc "
source "${ROOT_DIR}/scripts/source_mujoco_setup.sh"
PYTHONPATH='${ROOT_DIR}/src/holosoma_inference' python "${ROOT_DIR}/scripts/run_wbt_mujoco_official_policy.py" \
    --model-path='${ONNX_PATH}' \
    --interface=lo \
    --domain-id=0 \
    --warmup-seconds=3.0 \
    --motion-delay-seconds=1.0 \
    --max-seconds='${MAX_SECONDS}' \
    --no-use-sim-time
" >"${POLICY_LOG}" 2>&1
POLICY_STATUS=$?
set -e

if [[ ${POLICY_STATUS} -ne 0 ]]; then
    echo "Policy process failed. See ${POLICY_LOG}" >&2
    exit ${POLICY_STATUS}
fi

wait "${SIM_PID}" || true
SIM_PID=""

trap - EXIT INT TERM

echo "Simulation log: ${SIM_LOG}"
echo "Policy log: ${POLICY_LOG}"
echo "Recorded videos:"
find "${VIDEO_DIR}" -maxdepth 1 -type f | sort
