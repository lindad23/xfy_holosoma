#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DEFAULT_CHECKPOINT="/root/coding/holosoma/logs/WholeBodyTracking/20260324_093428-g1_29dof_wbt_manager-locomotion/model_08000.pt"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
MAX_EVAL_STEPS="${2:-600}"

if [[ $# -ge 1 ]]; then
    shift
fi
if [[ $# -ge 1 ]]; then
    shift
fi

source "${SCRIPT_DIR}/source_mujoco_setup.sh"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${ROOT_DIR}"

python src/holosoma/holosoma/eval_agent.py \
    --checkpoint="${CHECKPOINT}" \
    simulator:mujoco \
    logger:wandb \
    --eval_overrides.disable_logger=False \
    --training.headless=True \
    --training.max_eval_steps="${MAX_EVAL_STEPS}" \
    --training.export_onnx=False \
    --logger.mode=offline \
    --logger.video.enabled=True \
    --logger.video.interval=1 \
    --logger.video.show_command_overlay=False \
    --logger.video.upload_to_wandb=False \
    --randomization.ignore_unsupported=True \
    --randomization.setup_terms.push_randomizer_state.params.enabled=False \
    --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain=False \
    --randomization.setup_terms.setup_action_delay_buffers.params.enabled=False \
    --randomization.setup_terms.randomize_base_com_startup.params.enabled=False \
    --randomization.setup_terms.setup_dof_pos_bias.params.enabled=False \
    --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias=False \
    --observation.groups.actor_obs.enable_noise=False \
    --termination.terms.bad_tracking.params.bad_ref_pos_threshold=2.0 \
    --termination.terms.bad_tracking.params.bad_ref_ori_threshold=2.0 \
    --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold=1.0 \
    --termination.terms.bad_tracking.params.bad_object_pos_threshold=1.0 \
    --termination.terms.bad_tracking.params.bad_object_ori_threshold=2.0 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
    "$@"
