"""Dashboard and open HTTP/WebSocket adapter for the MuJoCo SmolVLA loop."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
JOINT_ORDER = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_motor_joint"]
STATE_NAMES = [f"joint_{i}.pos" for i in range(1, 6)] + ["gripper_motor_joint.pos"]
CAMERAS = ("agentview", "topview")
HTML = Path(__file__).with_name("dashboard.html")


class Runtime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.status: dict[str, Any] = {"state": "idle", "message": "Enter a task and press Start."}
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.preview: dict[str, Any] = {}

    def update(self, **values: Any) -> None:
        self.status.update(values)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Policy request failed: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("Policy response must be a JSON object.")
    return result


async def policy_prediction(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return await asyncio.to_thread(_post_json, url, payload, timeout)


def validate_action(action: Any, limits: dict[str, list[float]]) -> list[float]:
    if not isinstance(action, list) or len(action) != 6:
        raise RuntimeError("Policy 'action' must be six absolute revolute-joint targets in radians.")
    values: list[float] = []
    for name, raw in zip(JOINT_ORDER, action):
        if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw):
            raise RuntimeError(f"Policy action for {name} must be a finite number.")
        low, high = limits[name]
        if raw < low or raw > high:
            raise RuntimeError(f"Policy action for {name}={raw} is outside [{low}, {high}] rad.")
        values.append(float(raw))
    return values


def model_observation(observation: dict[str, Any], task: str) -> dict[str, Any]:
    """Map bridge feedback to the dataset's LeRobot feature names."""
    if observation.get("joint_order") != JOINT_ORDER:
        raise RuntimeError(f"Bridge joint order differs from expected order {JOINT_ORDER}.")
    state = observation.get("state", {}).get("joint_position_rad")
    if not isinstance(state, list) or len(state) != 6:
        raise RuntimeError("Bridge must provide six joint positions.")
    images = observation.get("images", {})
    result: dict[str, Any] = {
        "observation.state": [float(v) for v in state],
        "observation.state.names": STATE_NAMES,
        "task": task,
        "timestamp": observation.get("timestamp_unix"),
        "seq": observation.get("seq"),
        "observation.images": {},
    }
    for camera in CAMERAS:
        image = images.get(camera)
        if not isinstance(image, dict) or image.get("encoding") != "jpeg" or not image.get("data_base64"):
            raise RuntimeError(f"Bridge is missing a JPEG image for camera {camera}.")
        # Validate that this is usable image data before forwarding it.
        try:
            base64.b64decode(image["data_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Bridge sent invalid base64 for {camera}.") from error
        result["observation.images"][f"observation.images.{camera}"] = image
    return result


async def run_controller(runtime: Runtime, task_text: str) -> None:
    args = runtime.args
    token = os.environ.get("MUJOCO_BRIDGE_TOKEN")
    headers = {}
    try:
        async with websockets.connect(args.feedback_ws, max_size=20_000_000) as feedback, \
                websockets.connect(args.commands_ws, max_size=2_000_000) as commands:
            if token:
                await feedback.send(json.dumps({"token": token}))
                await commands.send(json.dumps({"token": token}))
                await feedback.recv()
                await commands.recv()
            runtime.update(state="running", message="Connected. Waiting for simulator observation.", task=task_text)
            seq = 0
            last_observation: dict[str, Any] | None = None
            while True:
                if runtime.stop_event.is_set():
                    break
                try:
                    raw = await asyncio.wait_for(feedback.recv(), timeout=min(args.feedback_timeout, 0.5))
                except asyncio.TimeoutError:
                    continue
                observation = json.loads(raw)
                last_observation = observation
                runtime.preview = observation.get("images", {})
                runtime.update(bridge_seq=observation.get("seq"), sim_time=observation.get("sim_time"))
                payload = {
                    "schema": "lerobot.smolvla.observation.v1",
                    **model_observation(observation, task_text),
                    # Auxiliary simulator context is not a trained model feature.
                    # The mock policy uses it to issue a safe hold-position action.
                    "simulator": {
                        "joint_order": observation["joint_order"],
                        "joint_position_rad": observation["state"]["joint_position_rad"],
                        "commanded_position_rad": observation["state"]["commanded_position_rad"],
                    },
                    "action_contract": {
                        "feature": "action",
                        "names": [f"{name}.pos" for name in JOINT_ORDER],
                        "shape": [6],
                        "dtype": "float32",
                        "units": "radian",
                        "representation": "absolute_joint_target_rad",
                    },
                }
                runtime.update(state="predicting", message="Sending current camera images, state, and task to policy.")
                prediction = await policy_prediction(args.policy_url, payload, args.policy_timeout)
                if runtime.stop_event.is_set():
                    break
                if prediction.get("schema") not in (None, "lerobot.smolvla.prediction.v1"):
                    raise RuntimeError("Unexpected policy response schema.")
                action = validate_action(prediction.get("action"), observation["joint_limits_rad"])
                seq += 1
                await commands.send(json.dumps({"type": "command", "seq": seq, "target_position_rad": action}))
                ack = json.loads(await asyncio.wait_for(commands.recv(), timeout=3.0))
                if ack.get("type") != "ack" or ack.get("seq") != seq or not ack.get("accepted"):
                    raise RuntimeError(f"Simulator rejected action: {ack.get('error', ack)}")
                runtime.update(
                    state="running", message=prediction.get("message", "Action accepted by simulator."),
                    action=action, action_seq=seq, policy_latency_ms=prediction.get("latency_ms"),
                    done=bool(prediction.get("done", False)), success=prediction.get("success"),
                )
                if prediction.get("done"):
                    runtime.update(state="completed", message=prediction.get("message", "Policy marked task complete."))
                    return
            if last_observation is not None:
                # Stop new motion by holding the latest measured pose. The bridge's
                # watchdog is still the fallback if this hold command is rejected.
                seq += 1
                hold = last_observation["state"]["joint_position_rad"]
                await commands.send(json.dumps({"type": "command", "seq": seq, "target_position_rad": hold}))
                await asyncio.wait_for(commands.recv(), timeout=3.0)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        runtime.update(state="error", message=str(error))
    finally:
        if runtime.status.get("state") in ("running", "predicting"):
            runtime.update(state="stopped", message="Controller stopped.")


def make_app(args: argparse.Namespace) -> FastAPI:
    runtime = Runtime(args)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if runtime.task and not runtime.task.done():
            runtime.task.cancel()

    app = FastAPI(title="SmolVLA Simulator Feedback", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTML.read_text(encoding="utf-8")

    @app.get("/api/config")
    async def config():
        return {"policy_url": args.policy_url, "feedback_ws": args.feedback_ws,
                "command_contract": "six absolute position targets in rad, ordered joint_1..joint_5,gripper_motor_joint",
                "observation_schema": "LeRobot: observation.images.agentview/topview, observation.state, task"}

    @app.get("/api/status")
    async def status():
        return runtime.status

    @app.get("/api/preview")
    async def preview():
        return runtime.preview

    @app.post("/api/start")
    async def start(body: dict[str, Any]):
        task_text = body.get("task")
        if not isinstance(task_text, str) or not task_text.strip():
            return {"ok": False, "error": "Enter a task instruction."}
        if runtime.task and not runtime.task.done():
            return {"ok": False, "error": "A control run is already active."}
        runtime.stop_event = asyncio.Event()
        runtime.task = asyncio.create_task(run_controller(runtime, task_text.strip()))
        return {"ok": True, "state": "starting"}

    @app.post("/api/stop")
    async def stop():
        runtime.stop_event.set()
        runtime.update(state="stopped", message="Stopped by operator.")
        return {"ok": True}

    @app.websocket("/ws/status")
    async def status_stream(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                await socket.send_json(runtime.status)
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    app.state.runtime = runtime
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dashboard and simulator-to-SmolVLA control adapter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8765)
    parser.add_argument("--policy-url", default="http://127.0.0.1:8781/predict")
    parser.add_argument("--policy-timeout", type=float, default=20.0)
    parser.add_argument("--feedback-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.port, args.bridge_port) < 1 or max(args.port, args.bridge_port) > 65535:
        parser.error("Ports must be between 1 and 65535.")
    args.feedback_ws = f"ws://{args.bridge_host}:{args.bridge_port}/ws/feedback"
    args.commands_ws = f"ws://{args.bridge_host}:{args.bridge_port}/ws/commands"
    return args


if __name__ == "__main__":
    options = parse_args()
    uvicorn.run(make_app(options), host=options.host, port=options.port, log_level="info")
