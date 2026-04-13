#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IsaacSim WBT evaluation with local video recording.")
    parser.add_argument("--checkpoint", required=True, help="Path to the WBT .pt checkpoint.")
    parser.add_argument("--max-eval-steps", type=int, default=3000, help="Maximum evaluation steps.")
    parser.add_argument(
        "--log-base-dir",
        default="logs/WholeBodyTrackingOfficial",
        help="Base directory for evaluation logs.",
    )
    parser.add_argument(
        "--video-save-dir",
        default=None,
        help="Optional explicit directory for recorded videos.",
    )
    parser.add_argument(
        "--disable-default-pose-transitions",
        action="store_true",
        help="Disable default-pose prepend/append transitions in the motion config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = str(Path(args.checkpoint).expanduser())

    init_eval_logging()
    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_path)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    video_cfg = dataclasses.replace(
        saved_cfg.logger.video,
        enabled=True,
        interval=1,
        show_command_overlay=False,
        upload_to_wandb=False,
        save_dir=args.video_save_dir,
    )
    logger_cfg = dataclasses.replace(
        saved_cfg.logger,
        mode="offline",
        base_dir=args.log_base_dir,
        headless_recording=True,
        video=video_cfg,
    )
    training_cfg = dataclasses.replace(
        eval_cfg.training,
        headless=True,
        num_envs=1,
        max_eval_steps=args.max_eval_steps,
        export_onnx=False,
    )
    eval_cfg = dataclasses.replace(eval_cfg, training=training_cfg, logger=logger_cfg)

    if args.disable_default_pose_transitions:
        motion_cfg = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
        motion_cfg["enable_default_pose_prepend"] = False
        motion_cfg["enable_default_pose_append"] = False

    run_eval_with_tyro(eval_cfg, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
