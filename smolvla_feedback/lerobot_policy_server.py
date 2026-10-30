"""Optional local SmolVLA inference endpoint using LeRobot's saved processors."""

from __future__ import annotations

import argparse
import base64
import io
import time
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

CAMERAS = ("agentview", "topview")
STATE_NAMES = [f"joint_{i}.pos" for i in range(1, 6)] + ["gripper_motor_joint.pos"]


def decode_image(image: dict[str, Any], key: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image["data_base64"], validate=True)
        with Image.open(io.BytesIO(raw)) as frame:
            result = np.array(frame.convert("RGB"), dtype=np.uint8, copy=True)
    except Exception as error:
        raise ValueError(f"Invalid encoded image for {key}: {error}") from error
    return result


def build_server(model_path: str, device_name: str) -> FastAPI:
    device = torch.device(device_name)
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(policy.config, model_path)

    app = FastAPI(title="Fine-tuned SmolVLA inference")
    app.state.policy = policy
    app.state.preprocess = preprocess
    app.state.postprocess = postprocess

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_path, "device": str(device),
                "state_dim": 6, "action_dim": 6, "cameras": list(CAMERAS)}

    @app.post("/predict")
    async def predict(body: dict[str, Any]):
        start = time.perf_counter()
        if body.get("schema") != "lerobot.smolvla.observation.v1":
            raise HTTPException(400, "Expected lerobot.smolvla.observation.v1")
        task = body.get("task")
        state = body.get("observation.state")
        images = body.get("observation.images")
        if not isinstance(task, str) or not task.strip():
            raise HTTPException(400, "task must be a non-empty string")
        if not isinstance(state, list) or len(state) != len(STATE_NAMES):
            raise HTTPException(400, "observation.state must have six values")
        if not isinstance(images, dict):
            raise HTTPException(400, "observation.images must be an object")
        if body.get("observation.state.names") != STATE_NAMES:
            raise HTTPException(400, "State feature names do not match the trained schema")

        raw_observation: dict[str, Any] = dict(zip(STATE_NAMES, state))
        features: dict[str, dict[str, Any]] = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
        }
        try:
            for camera in CAMERAS:
                feature_key = f"observation.images.{camera}"
                frame = decode_image(images[feature_key], feature_key)
                raw_observation[camera] = frame
                height, width, channels = frame.shape
                features[feature_key] = {
                    "dtype": "video", "shape": (height, width, channels),
                    "names": ["height", "width", "channels"],
                }
            observation_frame = build_inference_frame(
                observation=raw_observation, ds_features=features, device=device,
                task=task, robot_type="mujoco_manipulator",
            )
            model_input = app.state.preprocess(observation_frame)
            with torch.inference_mode():
                action_tensor = app.state.policy.select_action(model_input)
                action_tensor = app.state.postprocess(action_tensor)
            action = action_tensor.detach().to("cpu").reshape(-1).tolist()
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        if len(action) != 6:
            raise HTTPException(500, f"Fine-tuned model returned {len(action)} values; expected 6 actions")
        return {
            "schema": "lerobot.smolvla.prediction.v1",
            "action": action,
            "done": False,
            "message": "Fine-tuned SmolVLA action returned.",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a fine-tuned local SmolVLA policy at POST /predict.")
    parser.add_argument("--model", required=True, help="Local LeRobot checkpoint directory or Hub model id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    uvicorn.run(build_server(args.model, args.device), host=args.host, port=args.port, log_level="info")
