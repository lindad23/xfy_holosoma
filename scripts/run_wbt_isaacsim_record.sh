#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DEFAULT_CHECKPOINT="/root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_36000.pt"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
MAX_EVAL_STEPS="${2:-3000}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR="${ROOT_DIR}/logs/WholeBodyTrackingOfficial/${TIMESTAMP}-g1_29dof_wbt_isaacsim_official"
VIDEO_DIR="${RUN_DIR}/renderings_training"
mkdir -p "${VIDEO_DIR}"

/usr/bin/zsh -lc "
source scripts/source_isaacsim_setup.sh
python scripts/run_wbt_isaacsim_record.py \
    --checkpoint='${CHECKPOINT}' \
    --max-eval-steps='${MAX_EVAL_STEPS}' \
    --log-base-dir='${RUN_DIR}' \
    --video-save-dir='${VIDEO_DIR}'
"

echo "Run directory: ${RUN_DIR}"
echo "Video directory: ${VIDEO_DIR}"
find "${VIDEO_DIR}" -maxdepth 1 -type f | sort
