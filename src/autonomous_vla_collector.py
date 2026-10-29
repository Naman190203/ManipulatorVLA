"""Autonomously generate and record scripted pick-and-place VLA episodes.

The controller uses damped-least-squares position IK around the midpoint of the
fingertips. It records exactly the same dual-camera MP4 + JSONL layout as the
interactive recorder and marks each episode successful only after checking the
target object's final position on the plate.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE = PROJECT_ROOT / "assets" / "scene_with_balls_cameras.xml"
CAMERAS = ("agentview", "topview")
ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5")
ACTUATORS = ("joint1_position", "joint2_position", "joint3_position", "joint4_position", "joint5_position", "gripper_open_close")
OBJECT_JOINTS = {name: f"{name}_free" for name in ("red_ball", "green_ball", "blue_ball", "orange_cube", "purple_cube")}
OBJECT_HALF_HEIGHT = {"red_ball": 0.020, "green_ball": 0.020, "blue_ball": 0.020,
                      "orange_cube": 0.018, "purple_cube": 0.018}
OBJECT_FOOTPRINT = {"red_ball": 0.020, "green_ball": 0.020, "blue_ball": 0.020,
                    "orange_cube": 0.026, "purple_cube": 0.026}
PLATE_RADIUS = 0.080
PLATE_INNER_RADIUS = 0.068
PLATE_SURFACE_Z = 0.014


class AutonomousCollector:
    def __init__(self, task: dict[str, Any], output: Path, fps: float, grasp_mode: str) -> None:
        self.task = task
        self.output = output
        self.fps = fps
        self.grasp_mode = grasp_mode
        self.held_object: str | None = None
        self.held_offset = np.zeros(3)
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "placement_plate")
        if self.site_id < 0 or self.plate_id < 0:
            raise RuntimeError("Scene must contain grasp_site and placement_plate.")
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS]
        if any(value < 0 for value in (*self.joint_ids, *self.actuator_ids)):
            raise RuntimeError("Scene is missing an arm joint or position actuator.")
        self.joint_limits = np.array([self.model.jnt_range[joint] for joint in self.joint_ids])
        self.qpos_addresses = np.array([self.model.jnt_qposadr[joint] for joint in self.joint_ids], dtype=int)
        self.dof_addresses = np.array([self.model.jnt_dofadr[joint] for joint in self.joint_ids], dtype=int)
        self.object_qpos = {}
        self.object_body = {}
        self.object_geom_ids = {}
        self.object_collision_masks = {}
        for name, joint_name in OBJECT_JOINTS.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if joint_id < 0 or body_id < 0:
                raise RuntimeError(f"Scene is missing movable object {name}.")
            self.object_qpos[name] = int(self.model.jnt_qposadr[joint_id])
            self.object_body[name] = body_id
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            self.object_geom_ids[name] = geom_ids
            self.object_collision_masks[name] = (
                self.model.geom_contype[geom_ids].copy(),
                self.model.geom_conaffinity[geom_ids].copy(),
            )
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.frame_index = 0
        self.last_capture = 0.0
        self.previous_action: list[float] | None = None

    def reset(self) -> None:
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        for name, position in self.task["object_positions"].items():
            address = self.object_qpos[name]
            self.data.qpos[address:address + 3] = position
            self.data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
        for actuator, target in zip(self.actuator_ids, self.data.qpos[self.qpos_addresses]):
            self.data.ctrl[actuator] = target
        self.data.ctrl[self.actuator_ids[-1]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(
        self, target: np.ndarray, random_restarts: int = 31, iterations: int = 180
    ) -> tuple[np.ndarray, float]:
        """Solve position IK while enforcing the real arm's per-joint limits.

        MuJoCo supplies the world-space site Jacobian J. Each update is the
        damped least-squares step dq = J.T (J J.T + λ²I)^-1 (target - site),
        solved only over joints that can move without crossing their limits.
        The result is projected back into the configured joint intervals.
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        current_q = self.data.qpos[self.qpos_addresses].copy()
        limits = np.array([self.model.jnt_range[joint_id] for joint_id in self.joint_ids])
        # These intervals come directly from the model (including the real
        # horizontal-to-horizontal 0–180° shoulder sweep); never widen them.
        limit_margin = 0.02  # keep commanded poses clear of the mechanical stops
        lower = limits[:, 0] + limit_margin
        upper = limits[:, 1] - limit_margin
        rng = np.random.default_rng(0)
        seeds = [current_q]
        seeds.extend(
            rng.uniform(limits[:, 0] + 0.02, limits[:, 1] - 0.02)
            for _ in range(random_restarts)
        )
        best_q = current_q.copy()
        best_error = float("inf")
        best_score = float("inf")
        damping = 0.04
        jacobian = np.zeros((3, self.model.nv))
        try:
            for seed in seeds:
                self.data.qpos[:] = qpos
                self.data.qvel[:] = qvel
                self.data.qpos[self.qpos_addresses] = seed
                mujoco.mj_forward(self.model, self.data)
                for _ in range(iterations):
                    error = target - self.data.site_xpos[self.site_id]
                    error_norm = float(np.linalg.norm(error))
                    if error_norm < best_error:
                        best_error = error_norm
                    if error_norm < 0.015:
                        candidate_q = self.data.qpos[self.qpos_addresses].copy()
                        travel = float(np.linalg.norm((candidate_q - current_q) / (limits[:, 1] - limits[:, 0])))
                        score = 10.0 * error_norm + travel
                        if score < best_score:
                            best_score = score
                            best_q = candidate_q
                    mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site_id)
                    arm_jacobian = jacobian[:, self.dof_addresses]
                    joint_q = self.data.qpos[self.qpos_addresses].copy()
                    unconstrained_delta = arm_jacobian.T @ np.linalg.solve(
                        arm_jacobian @ arm_jacobian.T + damping * damping * np.eye(3), error
                    )
                    at_lower = (joint_q <= lower + 1e-6) & (unconstrained_delta < 0.0)
                    at_upper = (joint_q >= upper - 1e-6) & (unconstrained_delta > 0.0)
                    free = ~(at_lower | at_upper)
                    delta = np.zeros(len(self.joint_ids))
                    if not np.any(free):
                        break
                    free_jacobian = arm_jacobian[:, free]
                    delta[free] = free_jacobian.T @ np.linalg.solve(
                        free_jacobian @ free_jacobian.T + damping * damping * np.eye(3), error
                    )
                    self.data.qpos[self.qpos_addresses] = np.clip(
                        joint_q + np.clip(delta * 0.45, -0.12, 0.12), lower, upper
                    )
                    mujoco.mj_forward(self.model, self.data)
        finally:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            mujoco.mj_forward(self.model, self.data)
        return best_q, best_error

    def move_to(self, target: np.ndarray, gripper: float, max_seconds: float, writers: dict[str, Any], frames: Any,
                arrival_tolerance: float = 0.015, cartesian_path: bool = False) -> bool:
        target = np.asarray(target, dtype=float)
        path_start = self.data.site_xpos[self.site_id].copy()
        path_distance = float(np.linalg.norm(target - path_start))
        waypoint_count = max(1, int(np.ceil(path_distance / 0.04))) if cartesian_path else 1
        slew_rate = 0.75
        for waypoint_index in range(1, waypoint_count + 1):
            waypoint = path_start + (target - path_start) * (waypoint_index / waypoint_count)
            if waypoint_count == 1:
                target_q, planned_error = self.solve_ik(waypoint)
            else:
                target_q, planned_error = self.solve_ik(waypoint, random_restarts=3, iterations=100)
            if planned_error >= 0.015:
                target_q, planned_error = self.solve_ik(waypoint)
            if planned_error >= 0.015:
                print(f"IK could not plan to {waypoint.tolist()}; closest position is {planned_error:.3f} m away.")
                return False
            if np.any(target_q < self.joint_limits[:, 0]) or np.any(target_q > self.joint_limits[:, 1]):
                raise RuntimeError("IK returned a joint target outside the configured manipulator limits.")
            current_q = self.data.qpos[self.qpos_addresses].copy()
            required_slew_time = float(np.max(np.abs(target_q - current_q)) / slew_rate)
            waypoint_budget = max(max_seconds / waypoint_count, required_slew_time + 1.0)
            deadline = self.data.time + waypoint_budget
            hard_deadline = deadline + 2.5
            arrived = False
            while self.data.time < hard_deadline:
                current = self.data.ctrl[self.actuator_ids[:5]].copy()
                max_step = slew_rate * self.model.opt.timestep
                command_q = np.clip(
                    current + np.clip(target_q - current, -max_step, max_step),
                    self.joint_limits[:, 0], self.joint_limits[:, 1],
                )
                self.data.ctrl[self.actuator_ids[:5]] = command_q
                self.data.ctrl[self.actuator_ids[-1]] = gripper
                mujoco.mj_step(self.model, self.data)
                self.sync_held_object()
                self.capture(writers, frames)
                if np.linalg.norm(waypoint - self.data.site_xpos[self.site_id]) < arrival_tolerance:
                    arrived = True
                    break
                if self.data.time >= deadline:
                    joint_lag = float(np.max(np.abs(target_q - self.data.qpos[self.qpos_addresses])))
                    if joint_lag < 0.005:
                        break
            if not arrived:
                residual = float(np.linalg.norm(waypoint - self.data.site_xpos[self.site_id]))
                joint_residual = float(np.max(np.abs(target_q - self.data.qpos[self.qpos_addresses])))
                print(
                    f"Move timed out with {residual:.3f} m remaining to waypoint "
                    f"{waypoint.tolist()} (joint lag {joint_residual:.3f} rad, planned error {planned_error:.3f} m)."
                )
                return False
        return True

    def hold(self, seconds: float, gripper: float, writers: dict[str, Any], frames: Any) -> None:
        deadline = self.data.time + seconds
        while self.data.time < deadline:
            self.data.ctrl[self.actuator_ids[-1]] = gripper
            mujoco.mj_step(self.model, self.data)
            self.sync_held_object()
            self.capture(writers, frames)

    def sync_held_object(self) -> None:
        """Follow the fingertip site during the explicitly-labelled assist mode."""
        if self.held_object is None:
            return
        self.set_object_collision(self.held_object, enabled=False)
        address = self.object_qpos[self.held_object]
        self.data.qpos[address:address + 3] = self.data.site_xpos[self.site_id] + self.held_offset
        self.data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof_address = self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINTS[self.held_object]
        )]
        self.data.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_object_collision(self, object_name: str, enabled: bool) -> None:
        """Disable contacts for a kinematic assist-held object, then restore them."""
        geom_ids = self.object_geom_ids[object_name]
        if enabled:
            contype, conaffinity = self.object_collision_masks[object_name]
            self.model.geom_contype[geom_ids] = contype
            self.model.geom_conaffinity[geom_ids] = conaffinity
        else:
            self.model.geom_contype[geom_ids] = 0
            self.model.geom_conaffinity[geom_ids] = 0

    def capture(self, writers: dict[str, Any], frames: Any) -> None:
        if self.data.time - self.last_capture < 1.0 / self.fps:
            return
        action = [float(self.data.ctrl[actuator]) for actuator in self.actuator_ids]
        for camera, writer in writers.items():
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            self.renderer.update_scene(self.data, camera=camera_id)
            writer.append_data(self.renderer.render())
        delta = [0.0] * len(action) if self.previous_action is None else [a - b for a, b in zip(action, self.previous_action)]
        frames.write(json.dumps({
            "frame_index": self.frame_index, "video_frame_index": self.frame_index,
            "sim_time": float(self.data.time), "joint_position_rad": self.data.qpos[self.qpos_addresses].tolist(),
            "joint_velocity_rad_s": self.data.qvel[self.dof_addresses].tolist(),
            "action_absolute_joint_target_rad": action, "action_delta_joint_target_rad": delta,
            "agentview_video": "agentview.mp4", "topview_video": "topview.mp4",
        }, separators=(",", ":")) + "\n")
        self.previous_action = action
        self.frame_index += 1
        self.last_capture = float(self.data.time)

    def run(self) -> dict[str, Any]:
        target_name = self.task["target_object"]
        episode_dir = self.output / "episodes" / self.task["episode_id"]
        episode_dir.mkdir(parents=True, exist_ok=False)
        self.reset()
        target_address = self.object_qpos[target_name]
        target = self.data.qpos[target_address:target_address + 3].copy()
        metadata = {**self.task, "schema_version": 1, "controller": "damped_least_squares_ik",
                    "grasp_mode": self.grasp_mode,
                    "action_representation": "absolute_joint_target_rad", "camera_video_fps": self.fps,
                    "started_at_unix": time.time()}
        (episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        writers = {camera: imageio.get_writer(episode_dir / f"{camera}.mp4", fps=self.fps, codec="libx264") for camera in CAMERAS}
        try:
            # Safe pre-grasp, descend to the object, close, lift, transport to
            # the plate, descend, then release. Each phase remains closed-loop.
            frames = (episode_dir / "frames.jsonl").open("w", encoding="utf-8")
            if not self.move_to(target + (0.0, 0.0, 0.12), 0.0, 6.0, writers, frames):
                raise RuntimeError(f"Could not reach the pre-grasp position above {target_name}.")
            # Re-read the free object's live center after the approach. The
            # grasp_site itself is the pinch reference, so its target is the
            # object center, not 35 mm above it.
            grasp_target = self.data.xpos[self.object_body[target_name]].copy()
            grasp_tolerance = 0.022 if self.grasp_mode == "assist" else 0.015
            if not self.move_to(grasp_target, 0.0, 4.0, writers, frames, grasp_tolerance):
                raise RuntimeError(f"Could not descend to the selected object {target_name}; gripper left open.")
            if self.grasp_mode == "assist":
                object_position = self.data.xpos[self.object_body[target_name]].copy()
                fingertip_position = self.data.site_xpos[self.site_id].copy()
                grasp_distance = float(np.linalg.norm(object_position - fingertip_position))
                if grasp_distance <= 0.025:
                    self.held_object = target_name
                    self.held_offset = object_position - fingertip_position
                    self.set_object_collision(target_name, enabled=False)
                else:
                    raise RuntimeError(
                        f"Grasp missed: gripper is {grasp_distance:.3f} m from {target_name}; "
                        "not lifting an unattached object."
                    )
            self.hold(1.0, -0.45, writers, frames)
            if not self.move_to(target + (0.0, 0.0, 0.15), -0.45, 5.0, writers, frames):
                raise RuntimeError(f"Could not lift the selected object {target_name}.")
            plate = self.data.xpos[self.plate_id].copy()
            placement_center = plate + (0.0, 0.0, PLATE_SURFACE_Z + OBJECT_HALF_HEIGHT[target_name] + 0.003)
            placement_site = placement_center - self.held_offset if self.grasp_mode == "assist" else placement_center
            if not self.move_to(placement_site + (0.0, 0.0, 0.16), -0.45, 7.0, writers, frames,
                                cartesian_path=True):
                raise RuntimeError("Could not reach the approach point above the placement plate.")
            if self.grasp_mode == "assist":
                # Assist mode already holds the object kinematically. Set its
                # release pose on the plate so the narrow wrist can stay clear
                # of the retaining rim during the final placement.
                object_address = self.object_qpos[target_name]
                self.data.qpos[object_address:object_address + 3] = placement_center
                object_joint = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINTS[target_name]
                )
                object_dof = self.model.jnt_dofadr[object_joint]
                self.data.qvel[object_dof:object_dof + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)
            else:
                drop_site = placement_site + (0.0, 0.0, 0.05)
                if not self.move_to(drop_site, -0.45, 4.0, writers, frames, cartesian_path=True):
                    raise RuntimeError("Could not reach the drop point above the placement plate.")
            if self.held_object is not None:
                self.set_object_collision(self.held_object, enabled=True)
            self.held_object = None
            self.held_offset[:] = 0.0
            self.hold(1.0, 0.0, writers, frames)
        finally:
            frames.close()
            for writer in writers.values():
                writer.close()
            self.renderer.close()
        object_position = self.data.xpos[self.object_body[target_name]]
        plate_position = self.data.xpos[self.plate_id]
        expected_object_z = plate_position[2] + PLATE_SURFACE_Z + OBJECT_HALF_HEIGHT[target_name]
        success = bool(
            np.linalg.norm(object_position[:2] - plate_position[:2])
            < PLATE_INNER_RADIUS - OBJECT_FOOTPRINT[target_name]
            and abs(object_position[2] - expected_object_z) < 0.025
        )
        metadata.update({"frame_count": self.frame_index, "success": success, "finished_at_unix": time.time(),
                         "final_target_position": object_position.tolist()})
        (episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {"episode_id": self.task["episode_id"], "success": success, "frame_count": self.frame_index}


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomously collect IK pick-and-place VLA episodes.")
    parser.add_argument("--tasks", type=Path, help="Optional task suite JSON. Without it, choose an object interactively.")
    parser.add_argument("--object", choices=tuple(OBJECT_JOINTS), help="Object to place on the plate; uses its current scene position.")
    parser.add_argument("--output", type=Path, default=Path("vla_dataset_autonomous"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=1, help="Number of tasks to run; start with one.")
    parser.add_argument("--grasp-mode", choices=("assist", "physical"), default="assist",
                        help="assist is labelled kinematic assistance; physical uses only contacts.")
    args = parser.parse_args()
    if args.fps <= 0 or args.limit <= 0:
        parser.error("--fps and --limit must be positive.")
    if args.tasks:
        suite = json.loads(args.tasks.read_text(encoding="utf-8"))
        tasks = suite["episodes"][:args.limit]
    else:
        target_name = args.object
        if target_name is None:
            choices = tuple(OBJECT_JOINTS)
            print("Choose an object to place on the plate:")
            for index, name in enumerate(choices, 1):
                print(f"  {index}. {name.replace('_', ' ')}")
            answer = input("Object number or name: ").strip().lower().replace(" ", "_")
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                target_name = choices[int(answer) - 1]
            elif answer in choices:
                target_name = answer
            else:
                parser.error(f"Unknown object: {answer}")
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        placements = {}
        for name, joint_name in OBJECT_JOINTS.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                parser.error(f"Scene is missing movable object {name}.")
            address = model.jnt_qposadr[joint_id]
            # Interactive selection acts on the scene's authored, reachable
            # object poses; random placements belong in validated task suites.
            placements[name] = data.qpos[address:address + 3].astype(float).tolist()
        tasks = [{"episode_id": f"episode_{int(time.time())}", "task_id": f"place_{target_name}_on_plate",
                  "instruction": f"Put the {target_name.replace('_', ' ')} on the plate.",
                  "target_object": target_name, "target_receptacle": "placement_plate",
                  "object_positions": placements}]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in tasks:
        summaries.append(AutonomousCollector(task, args.output, args.fps, args.grasp_mode).run())
    (args.output / "index.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
