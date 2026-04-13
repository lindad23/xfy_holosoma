#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unitree_mujoco G1 WBT simulator headlessly and record MP4.")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=20.0)
    parser.add_argument("--simulate-dt", type=float, default=0.005)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-scene-information", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    coding_root = Path(__file__).resolve().parents[2]
    unitree_root = coding_root / "unitree_mujoco"
    sim_py_root = unitree_root / "simulate_python"
    sys.path.insert(0, str(sim_py_root))

    import config as unitree_config  # type: ignore

    unitree_config.ROBOT = "g1"
    unitree_config.ROBOT_SCENE = str(unitree_root / "unitree_robots" / "g1" / "scene_29dof.xml")
    unitree_config.DOMAIN_ID = args.domain_id
    unitree_config.INTERFACE = args.interface
    unitree_config.USE_JOYSTICK = 0
    unitree_config.PRINT_SCENE_INFORMATION = args.print_scene_information
    unitree_config.ENABLE_ELASTIC_BAND = False
    unitree_config.SIMULATE_DT = args.simulate_dt
    unitree_config.VIEWER_DT = 1.0 / max(args.video_fps, 1.0)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # type: ignore

    mj_model = mujoco.MjModel.from_xml_path(unitree_config.ROBOT_SCENE)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = args.simulate_dt

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)
    if args.print_scene_information:
        bridge.PrintSceneInformation()

    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    frame_interval_steps = max(1, round(1.0 / (args.video_fps * args.simulate_dt)))
    step_count = 0
    frame_count = 0
    start = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if now - start >= args.max_runtime_seconds:
                break

            step_start = now
            mujoco.mj_step(mj_model, mj_data)

            if step_count % frame_interval_steps == 0:
                camera.lookat[:] = np.asarray(mj_data.qpos[:3])
                renderer.update_scene(mj_data, camera=camera)
                frame = renderer.render()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1

            step_count += 1
            sleep_time = args.simulate_dt - (time.perf_counter() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        writer.release()
        renderer.close()

    duration = time.perf_counter() - start
    print(f"OUTPUT={output_path}")
    print(f"STEPS={step_count}")
    print(f"FRAMES={frame_count}")
    print(f"DURATION={duration:.2f}")

    os._exit(0)


if __name__ == "__main__":
    main()
