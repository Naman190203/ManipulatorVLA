"""Local MuJoCo feedback and position-command bridge.

Run from the project directory with:
    python sim_bridge.py

The simulator owns all MuJoCo state. WebSocket clients only receive snapshots
or submit bounded position targets.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
import hmac
import math
import os
from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
import uvicorn


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "scene_with_balls_cameras.xml"
JOINT_ORDER = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_gear_joint",
)
ACTUATOR_ORDER = (
    "joint1_position",
    "joint2_position",
    "joint3_position",
    "joint4_position",
    "joint5_position",
    "gripper_open_close",
)
CAMERA_NAMES = ("agentview", "gripperview")
MAX_TARGET_SPEED_RAD_S = (0.75, 0.75, 0.75, 0.75, 0.75, 1.0)
SIM_BATCH_STEPS = 5
COMMAND_TIMEOUT_S = 12.0


@dataclass
class BridgeConfig:
    scene: Path
    host: str
    port: int
    fps: float
    width: int
    height: int
    viewer: bool
    token: str | None


class SimulationBridge:
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.scene))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=config.height, width=config.width
        )

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_ORDER
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_ORDER
        ]
        self.camera_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            for name in CAMERA_NAMES
        }
        missing = [
            name
            for name, object_id in [
                *zip(JOINT_ORDER, self.joint_ids),
                *zip(ACTUATOR_ORDER, self.actuator_ids),
                *zip(CAMERA_NAMES, self.camera_ids.values()),
            ]
            if object_id < 0
        ]
        if missing:
            self.renderer.close()
            raise ValueError(f"Scene is missing required named objects: {missing}")

        self.qpos_addresses = [int(self.model.jnt_qposadr[jid]) for jid in self.joint_ids]
        self.dof_addresses = [int(self.model.jnt_dofadr[jid]) for jid in self.joint_ids]
        self.limits = []
        for joint_id, actuator_id in zip(self.joint_ids, self.actuator_ids):
            low, high = (
                float(v) for v in self.model.actuator_ctrlrange[actuator_id]
            )
            if self.model.jnt_limited[joint_id]:
                joint_low, joint_high = (
                    float(v) for v in self.model.jnt_range[joint_id]
                )
                low = max(low, joint_low)
                high = min(high, joint_high)
            if low > high:
                self.renderer.close()
                raise ValueError(
                    f"No usable target range for joint {JOINT_ORDER[len(self.limits)]}."
                )
            self.limits.append((low, high))

        mujoco.mj_forward(self.model, self.data)
        self.current_targets = [
            float(self.data.qpos[address]) for address in self.qpos_addresses
        ]
        for actuator_id, target in zip(self.actuator_ids, self.current_targets):
            self.data.ctrl[actuator_id] = target

        self.viewer = None
        if config.viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.camera_ids["agentview"]

        self.latest_observation: dict[str, Any] = {}
        self.latest_seq = 0
        self.last_command_at: float | None = None
        self.command_timed_out = False
        self.feedback_interval = 1.0 / config.fps

    def _read_joint_state(self) -> tuple[list[float], list[float]]:
        positions = [float(self.data.qpos[i]) for i in self.qpos_addresses]
        velocities = [float(self.data.qvel[i]) for i in self.dof_addresses]
        return positions, velocities

    def _render_jpeg(self, camera_id: int) -> str:
        self.renderer.update_scene(self.data, camera=camera_id)
        rgb = self.renderer.render()
        image = Image.fromarray(rgb)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def capture_observation(self) -> dict[str, Any]:
        self.latest_seq += 1
        positions, velocities = self._read_joint_state()
        image_time = float(self.data.time)
        images = {
            name: {
                "encoding": "jpeg",
                "width": self.config.width,
                "height": self.config.height,
                "timestamp": image_time,
                "data_base64": self._render_jpeg(self.camera_ids[name]),
            }
            for name in CAMERA_NAMES
        }
        return {
            "type": "observation",
            "seq": self.latest_seq,
            "sim_time": float(self.data.time),
            "timestamp_unix": time.time(),
            "joint_order": list(JOINT_ORDER),
            "command_mode": "absolute_position_rad",
            "joint_limits_rad": {
                name: list(bounds)
                for name, bounds in zip(JOINT_ORDER, self.limits)
            },
            "target_slew_rad_s": {
                name: speed
                for name, speed in zip(JOINT_ORDER, MAX_TARGET_SPEED_RAD_S)
            },
            "state": {
                "joint_position_rad": positions,
                "joint_velocity_rad_s": velocities,
                "commanded_position_rad": list(self.current_targets),
            },
            "images": images,
        }

    def validate_command(self, message: Any) -> tuple[int, list[float]]:
        if not isinstance(message, dict) or message.get("type") != "command":
            raise ValueError("Expected a JSON command with type='command'.")
        seq = message.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError("seq must be a non-negative integer.")
        targets = message.get("target_position_rad")
        if not isinstance(targets, list) or len(targets) != len(JOINT_ORDER):
            raise ValueError(
                f"target_position_rad must contain {len(JOINT_ORDER)} values "
                f"in this order: {list(JOINT_ORDER)}."
            )
        values: list[float] = []
        for index, raw in enumerate(targets):
            if isinstance(raw, bool):
                raise ValueError(f"Target {JOINT_ORDER[index]} must be numeric.")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"Target {JOINT_ORDER[index]} must be numeric.")
            if not math.isfinite(value):
                raise ValueError(f"Target {JOINT_ORDER[index]} must be finite.")
            low, high = self.limits[index]
            if value < low or value > high:
                raise ValueError(
                    f"Target {JOINT_ORDER[index]}={value} is outside actuator "
                    f"range [{low}, {high}] rad."
                )
            values.append(value)
        return seq, values

    async def run(self) -> None:
        timestep = float(self.model.opt.timestep)
        batch_duration = timestep * SIM_BATCH_STEPS
        next_feedback = time.perf_counter()
        next_step = time.perf_counter()

        while True:
            loop_start = time.perf_counter()
            if (
                self.last_command_at is not None
                and not self.command_timed_out
                and loop_start - self.last_command_at > COMMAND_TIMEOUT_S
            ):
                # Stop pursuing an old remote target and hold the measured pose.
                positions, _ = self._read_joint_state()
                self.current_targets = positions
                self.command_timed_out = True

            lock = self.viewer.lock() if self.viewer is not None else _NullLock()
            with lock:
                for _ in range(SIM_BATCH_STEPS):
                    for i, actuator_id in enumerate(self.actuator_ids):
                        current = float(self.data.ctrl[actuator_id])
                        target = self.current_targets[i]
                        max_delta = MAX_TARGET_SPEED_RAD_S[i] * timestep
                        delta = max(-max_delta, min(max_delta, target - current))
                        self.data.ctrl[actuator_id] = current + delta
                    mujoco.mj_step(self.model, self.data)

                now = time.perf_counter()
                if now >= next_feedback:
                    self.latest_observation = self.capture_observation()
                    next_feedback = now + self.feedback_interval

            if self.viewer is not None:
                self.viewer.sync()

            next_step += batch_duration
            await asyncio.sleep(max(0.0, next_step - time.perf_counter()))
            # Avoid a catch-up burst if rendering or the viewer fell behind.
            if time.perf_counter() - loop_start > batch_duration * 4:
                next_step = time.perf_counter()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def create_app(config: BridgeConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bridge = SimulationBridge(config)
        app.state.bridge = bridge
        bridge.latest_observation = bridge.capture_observation()
        worker = asyncio.create_task(bridge.run(), name="mujoco-simulation-loop")
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            bridge.close()

    app = FastAPI(title="MuJoCo Simulation Bridge", lifespan=lifespan)
    dashboard_html = (HERE / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return dashboard_html

    async def authenticate(websocket: WebSocket, bridge: SimulationBridge) -> bool:
        await websocket.accept()
        if bridge.config.token is None:
            return True
        try:
            auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            await websocket.close(code=4401, reason="Authentication required")
            return False
        supplied = auth_message.get("token") if isinstance(auth_message, dict) else None
        if not isinstance(supplied, str) or not hmac.compare_digest(
            supplied, bridge.config.token
        ):
            await websocket.close(code=4401, reason="Invalid token")
            return False
        await websocket.send_json({"type": "authenticated"})
        return True

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "scene": config.scene.name,
            "feedback_ws": "/ws/feedback",
            "commands_ws": "/ws/commands",
            "joint_order": list(JOINT_ORDER),
            "camera_names": list(CAMERA_NAMES),
            "limits_rad": {
                name: list(bounds)
                for name, bounds in zip(JOINT_ORDER, app.state.bridge.limits)
            },
            "target_slew_rad_s": {
                name: speed
                for name, speed in zip(JOINT_ORDER, MAX_TARGET_SPEED_RAD_S)
            },
        }

    @app.websocket("/ws/feedback")
    async def feedback(websocket: WebSocket):
        bridge: SimulationBridge = app.state.bridge
        if not await authenticate(websocket, bridge):
            return
        try:
            while True:
                observation = bridge.latest_observation
                if observation:
                    await asyncio.wait_for(websocket.send_json(observation), timeout=2.0)
                await asyncio.sleep(bridge.feedback_interval)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            await websocket.close()

    @app.websocket("/ws/commands")
    async def commands(websocket: WebSocket):
        bridge: SimulationBridge = app.state.bridge
        if not await authenticate(websocket, bridge):
            return
        last_seq = -1
        try:
            while True:
                message = await websocket.receive_json()
                try:
                    seq, targets = bridge.validate_command(message)
                    if seq <= last_seq:
                        raise ValueError("seq must increase for each command connection.")
                    bridge.current_targets = targets
                    bridge.last_command_at = time.perf_counter()
                    bridge.command_timed_out = False
                    last_seq = seq
                    await websocket.send_json(
                        {
                            "type": "ack",
                            "seq": seq,
                            "accepted": True,
                            "target_position_rad": targets,
                            "joint_order": list(JOINT_ORDER),
                        }
                    )
                except ValueError as error:
                    await websocket.send_json(
                        {
                            "type": "ack",
                            "seq": message.get("seq") if isinstance(message, dict) else None,
                            "accepted": False,
                            "error": str(error),
                        }
                    )
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(description="Run the local MuJoCo bridge.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if not args.scene.is_file():
        parser.error(f"Scene file does not exist: {args.scene}")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--fps, --width, and --height must be positive.")

    token = os.environ.get("MUJOCO_BRIDGE_TOKEN") or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and token is None:
        parser.error(
            "Remote binding requires MUJOCO_BRIDGE_TOKEN. Keep the bridge on "
            "127.0.0.1 unless you are using an authenticated tunnel."
        )

    return BridgeConfig(
        scene=args.scene.resolve(),
        host=args.host,
        port=args.port,
        fps=args.fps,
        width=args.width,
        height=args.height,
        viewer=not args.no_viewer,
        token=token,
    )


if __name__ == "__main__":
    bridge_config = parse_args()
    uvicorn.run(create_app(bridge_config), host=bridge_config.host, port=bridge_config.port)
