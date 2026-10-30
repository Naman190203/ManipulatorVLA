"""No-motion policy endpoint for checking the complete feedback/control pipe."""

from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI(title="SmolVLA Contract Mock")


@app.post("/predict")
async def predict(body: dict[str, Any]):
    started = time.perf_counter()
    if body.get("schema") != "lerobot.smolvla.observation.v1":
        raise HTTPException(400, "Expected lerobot.smolvla.observation.v1")
    if not isinstance(body.get("task"), str) or not body["task"].strip():
        raise HTTPException(400, "Missing task string")
    state = body.get("observation.state")
    images = body.get("observation.images")
    if not isinstance(state, list) or len(state) != 6:
        raise HTTPException(400, "observation.state must have 6 float values")
    if not isinstance(images, dict) or set(images) != {
        "observation.images.agentview", "observation.images.topview"
    }:
        raise HTTPException(400, "Expected agentview and topview camera images")
    simulator = body.get("simulator", {})
    action = simulator.get("commanded_position_rad")
    if not isinstance(action, list) or len(action) != 6:
        raise HTTPException(400, "Missing six-value simulator hold target")
    return {
        "schema": "lerobot.smolvla.prediction.v1",
        "action": action,
        "done": False,
        "message": "Mock policy echoed the current target; simulator should hold position.",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the no-motion policy contract checker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
