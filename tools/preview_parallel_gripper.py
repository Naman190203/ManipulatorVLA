"""Open the active manipulator scene and animate its URDF-derived gripper motor."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "scene_with_balls_cameras.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-seconds", type=float, default=3.0,
                        help="Seconds for each open-to-close or close-to-open motion")
    args = parser.parse_args()
    if args.cycle_seconds <= 0:
        parser.error("--cycle-seconds must be positive")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_open_close")
    if actuator_id < 0:
        raise RuntimeError("Active scene has no gripper_open_close actuator")
    open_target, closed_target = 0.0, -0.41
    start = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if grasp_site >= 0:
            viewer.cam.lookat[:] = data.site_xpos[grasp_site]
            viewer.cam.distance = 0.28
            viewer.cam.azimuth = 0
            viewer.cam.elevation = 12
        while viewer.is_running():
            elapsed = time.perf_counter() - start
            phase = (elapsed % (2 * args.cycle_seconds)) / args.cycle_seconds
            blend = phase if phase <= 1 else 2 - phase
            blend = blend * blend * (3 - 2 * blend)  # smoothstep
            data.ctrl[actuator_id] = open_target + blend * (closed_target - open_target)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
