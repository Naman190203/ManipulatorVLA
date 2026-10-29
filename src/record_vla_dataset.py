"""Interactive VLA episode recorder for the local MuJoCo bridge.

It resets each randomized task, then records the dashboard-controlled robot at
the feedback rate. The resulting raw episodes keep both camera streams, state,
absolute commanded actions, derived action deltas, task language, and exact
initial placements. This is intentionally a demonstration recorder: mark an
episode successful only after visually checking the completed task.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import websockets


CAMERAS = ("agentview", "gripperview")


async def authenticate(websocket: Any, token: str | None) -> None:
    if token is not None:
        await websocket.send(json.dumps({"token": token}))
        reply = json.loads(await websocket.recv())
        if reply.get("type") != "authenticated":
            raise RuntimeError("Bridge rejected the supplied token.")


def save_frame(
    episode_dir: Path,
    frame_index: int,
    observation: dict[str, Any],
    previous_action: list[float] | None,
    video_writers: dict[str, Any],
) -> list[float]:
    images = observation["images"]
    for camera in CAMERAS:
        encoded = images[camera]["data_base64"]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            video_writers[camera].append_data(np.asarray(image.convert("RGB")))
    state = observation["state"]
    action = [float(value) for value in state["commanded_position_rad"]]
    action_delta = [0.0] * len(action) if previous_action is None else [
        current - previous for current, previous in zip(action, previous_action)
    ]
    record = {
        "frame_index": frame_index,
        "observation_seq": observation["seq"],
        "sim_time": observation["sim_time"],
        "video_frame_index": frame_index,
        "agentview_video": "agentview.mp4",
        "gripperview_video": "gripperview.mp4",
        "camera_timestamps": {camera: images[camera]["timestamp"] for camera in CAMERAS},
        "camera_resolution": {camera: [images[camera]["width"], images[camera]["height"]] for camera in CAMERAS},
        "joint_position_rad": state["joint_position_rad"],
        "joint_velocity_rad_s": state["joint_velocity_rad_s"],
        "action_absolute_joint_target_rad": action,
        "action_delta_joint_target_rad": action_delta,
    }
    with (episode_dir / "frames.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, separators=(",", ":")) + "\n")
    return action


async def record_episode(
    feedback: Any, commands: Any, sequence: int, task: dict[str, Any], root: Path,
    duration_s: float, video_fps: float,
) -> tuple[int, dict[str, Any]]:
    await commands.send(json.dumps({
        "type": "reset", "seq": sequence, "object_positions": task["object_positions"],
    }))
    acknowledgement = json.loads(await commands.recv())
    if not acknowledgement.get("accepted"):
        raise RuntimeError(f"Episode reset failed: {acknowledgement.get('error')}")
    sequence += 1

    episode_dir = root / "episodes" / task["episode_id"]
    if episode_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing {episode_dir}.")
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode.json").write_text(json.dumps({
        **task,
        "schema_version": 1,
        "joint_order": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_gear_joint"],
        "camera_names": list(CAMERAS),
        "camera_video_fps": video_fps,
        "started_at_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")

    input(f"\n{task['episode_id']}: {task['instruction']}\n"
          "Scene reset. Position the dashboard controls, then press Enter to start recording.")
    print(f"Recording for {duration_s:.1f} seconds. Demonstrate the full task now.")
    deadline = time.monotonic() + duration_s
    frame_index = 0
    last_observation_seq = -1
    previous_action: list[float] | None = None
    video_writers = {
        camera: imageio.get_writer(episode_dir / f"{camera}.mp4", fps=video_fps, codec="libx264")
        for camera in CAMERAS
    }
    try:
        while time.monotonic() < deadline:
            observation = json.loads(await asyncio.wait_for(feedback.recv(), timeout=3.0))
            if observation.get("type") != "observation" or observation.get("seq") == last_observation_seq:
                continue
            last_observation_seq = observation["seq"]
            previous_action = save_frame(
                episode_dir, frame_index, observation, previous_action, video_writers
            )
            frame_index += 1
    finally:
        for writer in video_writers.values():
            writer.close()
    completed = input("Is the task visibly complete and successful? [y/N] ").strip().lower() == "y"
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    metadata.update({"frame_count": frame_index, "success": completed, "finished_at_unix": time.time()})
    (episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sequence, {"episode_id": task["episode_id"], "success": completed, "frame_count": frame_index}


async def main_async(args: argparse.Namespace) -> None:
    suite = json.loads(args.tasks.read_text(encoding="utf-8"))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene_path = Path(suite["scene"])
    scene_sha256 = hashlib.sha256(scene_path.read_bytes()).hexdigest() if scene_path.is_file() else None
    (root / "dataset_info.json").write_text(json.dumps({
        "schema_version": 1,
        "source_task_suite": str(args.tasks.resolve()),
        "scene": suite["scene"],
        "scene_sha256": scene_sha256,
        "camera_names": suite["camera_names"],
        "action_representation": suite["action_representation"],
    }, indent=2) + "\n", encoding="utf-8")
    base = args.bridge.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    async with websockets.connect(f"{base}/ws/feedback", max_size=None) as feedback, \
            websockets.connect(f"{base}/ws/commands", max_size=None) as commands:
        await authenticate(feedback, args.token)
        await authenticate(commands, args.token)
        sequence = 0
        for task in suite["episodes"]:
            sequence, summary = await record_episode(
                feedback, commands, sequence, task, root, args.duration, args.video_fps
            )
            with (root / "index.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(summary) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record randomized VLA demonstrations from the bridge.")
    parser.add_argument("--tasks", type=Path, default=Path("dataset_tasks.json"))
    parser.add_argument("--output", type=Path, default=Path("vla_dataset_raw"))
    parser.add_argument("--bridge", default="ws://127.0.0.1:8765")
    parser.add_argument("--token", default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    if not args.tasks.is_file():
        parser.error(f"Task suite not found: {args.tasks}")
    if args.duration <= 0 or args.video_fps <= 0:
        parser.error("--duration and --video-fps must be positive.")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
