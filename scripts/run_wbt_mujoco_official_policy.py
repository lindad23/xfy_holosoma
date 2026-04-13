#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import time

from loguru import logger

from holosoma_inference.config.config_values.inference import g1_29dof_wbt
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.misc import restore_terminal_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official MuJoCo WBT policy with automated startup.")
    parser.add_argument("--model-path", required=True, help="Path to the WBT ONNX model.")
    parser.add_argument("--interface", default="lo", help="Network interface for sim2sim communication.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--warmup-seconds", type=float, default=3.0, help="Seconds to hold stiff startup pose.")
    parser.add_argument(
        "--motion-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between enabling policy and starting the motion clip.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Maximum runtime before the script exits and lets the simulator finalize recording.",
    )
    parser.add_argument(
        "--motion-start-timestep",
        type=int,
        default=0,
        help="Starting timestep for the motion clip.",
    )
    parser.add_argument(
        "--motion-end-timestep",
        type=int,
        default=None,
        help="Optional ending timestep for the motion clip.",
    )
    parser.add_argument(
        "--no-use-sim-time",
        action="store_true",
        help="Disable external simulation-clock synchronization and advance the motion clip by local policy ticks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = dataclasses.replace(
        g1_29dof_wbt,
        task=dataclasses.replace(
            g1_29dof_wbt.task,
            model_path=args.model_path,
            interface=args.interface,
            domain_id=args.domain_id,
            use_joystick=False,
            use_sim_time=not args.no_use_sim_time,
            motion_start_timestep=args.motion_start_timestep,
            motion_end_timestep=args.motion_end_timestep,
        ),
    )

    policy = WholeBodyTrackingPolicy(config=config)

    # In non-interactive mode BasePolicy auto-enables policy control. Force the
    # official WBT startup sequence instead: stiff hold -> policy -> motion clip.
    policy._handle_stop_policy()

    start_time = time.perf_counter()
    policy_started = False
    motion_started = False

    logger.info("Automated WBT policy runner started")
    logger.info(
        "Startup schedule: stiff_hold={}s, motion_delay={}s, max_runtime={}s",
        args.warmup_seconds,
        args.motion_delay_seconds,
        args.max_seconds,
    )

    try:
        while True:
            elapsed = time.perf_counter() - start_time

            if not policy_started and elapsed >= args.warmup_seconds:
                policy._handle_start_policy()
                policy_started = True
                logger.info("Policy enabled after stiff hold")

            if policy_started and not motion_started and elapsed >= args.warmup_seconds + args.motion_delay_seconds:
                policy._handle_start_motion_clip()
                motion_started = True
                logger.info("Motion clip started")

            if policy.use_phase:
                policy.update_phase_time()

            policy.policy_action()

            if motion_started and args.motion_end_timestep is not None and not policy.motion_clip_progressing:
                logger.info("Motion clip reached configured end timestep; exiting")
                break

            if elapsed >= args.max_seconds:
                logger.info("Reached max runtime; exiting")
                break

            policy.rate.sleep()
    finally:
        try:
            policy._handle_stop_policy()
        finally:
            restore_terminal_settings()


if __name__ == "__main__":
    main()
