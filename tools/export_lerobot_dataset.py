"""Export successful raw recordings as a LeRobot video dataset.

Raw episode files are retained as the source of truth. This
exporter maps them to LeRobot's standard observation.images.*, observation.state,
action, and task fields. Run it in an environment with LeRobot installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from lerobot.datasets import LeRobotDataset


CAMERAS = ("agentview", "topview")
JOINT_FEATURE_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "gripper_motor_joint.pos",
)
JOINT_ORDER = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_motor_joint")
ACTION_REPRESENTATION = "absolute_joint_target_rad"


def read_episode(episode_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    if not metadata.get("success", False):
        raise ValueError(f"Episode is not marked successful: {episode_dir}")
    frame_path = episode_dir / "frames.jsonl"
    frames = [json.loads(line) for line in frame_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not frames:
        raise ValueError(f"Episode has no frames: {episode_dir}")
    state_sizes = {len(frame.get("joint_position_rad", [])) for frame in frames}
    action_sizes = {len(frame.get("action_absolute_joint_target_rad", [])) for frame in frames}
    if len(state_sizes) != 1 or next(iter(state_sizes)) != 6:
        raise ValueError(f"Expected six measured joints in the new gripper format: {episode_dir}")
    if action_sizes != {6}:
        raise ValueError(f"Expected six commanded joint targets in {episode_dir}")
    return metadata, frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert successful raw bridge episodes into LeRobot format.")
    parser.add_argument("--raw", type=Path, default=Path("vla_dataset_raw"), help="Raw recorder output directory")
    parser.add_argument("--output", type=Path, default=Path("vla_dataset_lerobot"), help="New LeRobot dataset directory; must not exist")
    parser.add_argument("--repo-id", default="local/manipulator_vla", help="LeRobot dataset identifier stored in metadata")
    parser.add_argument("--fps", type=int, default=10, help="Recorded frame rate")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.raw.is_dir():
        parser.error(f"Raw dataset directory not found: {args.raw}")
    if args.output.exists():
        parser.error(f"Output path already exists; choose a new --output: {args.output}")

    summary_path = args.raw / "index.json"
    if summary_path.is_file():
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary_path = args.raw / "index.jsonl"
        if not summary_path.is_file():
            parser.error(f"Missing recorder index: {summary_path}")
        summaries = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful_dirs = []
    for summary in summaries:
        if not summary.get("success", False):
            continue
        episode_dir = args.raw / "episodes" / summary["episode_id"]
        metadata, frames = read_episode(episode_dir)
        if not metadata.get("instruction"):
            raise ValueError(f"Missing language instruction: {episode_dir}")
        if metadata.get("schema_version") != 4:
            raise ValueError(f"Episode is not in the URDF-gripper dataset schema: {episode_dir}")
        if metadata.get("action_representation") != ACTION_REPRESENTATION:
            raise ValueError(f"Episode uses an old or unsupported action representation: {episode_dir}")
        if metadata.get("joint_order") != list(JOINT_ORDER):
            raise ValueError(f"Episode joint order does not match the URDF-derived gripper: {episode_dir}")
        successful_dirs.append((episode_dir, metadata, frames))
    if not successful_dirs:
        parser.error("No successful episodes found in the raw dataset")

    first_frame = successful_dirs[0][2][0]
    first_images = {}
    for camera in CAMERAS:
        image_path = successful_dirs[0][0] / f"{camera}.mp4"
        if not image_path.is_file():
            parser.error(f"Missing {camera} video in {successful_dirs[0][0]}")
        reader = imageio.get_reader(image_path)
        try:
            first_images[camera] = np.asarray(reader.get_data(0), dtype=np.uint8)
        finally:
            reader.close()
        if first_images[camera].ndim != 3 or first_images[camera].shape[2] != 3:
            parser.error(f"Expected RGB video for {camera}; got shape {first_images[camera].shape}")

    state_size = len(first_frame["joint_position_rad"])
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_size,),
            "names": list(JOINT_FEATURE_NAMES[:state_size]),
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(JOINT_FEATURE_NAMES),
        },
    }
    for camera in CAMERAS:
        height, width, channels = first_images[camera].shape
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="mujoco_manipulator",
        fps=args.fps,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )
    try:
        for episode_dir, metadata, frames in successful_dirs:
            if len(frames[0]["joint_position_rad"]) != state_size:
                raise ValueError("Cannot mix episodes with different observation.state dimensions")
            readers = {camera: imageio.get_reader(episode_dir / f"{camera}.mp4") for camera in CAMERAS}
            try:
                for index, record in enumerate(frames):
                    frame = {
                        "observation.state": np.asarray(record["joint_position_rad"], dtype=np.float32),
                        "action": np.asarray(record["action_absolute_joint_target_rad"], dtype=np.float32),
                        "task": metadata["instruction"],
                    }
                    for camera in CAMERAS:
                        key = f"observation.images.{camera}"
                        image = np.asarray(readers[camera].get_data(index), dtype=np.uint8)
                        if image.shape != features[key]["shape"]:
                            raise ValueError(f"Unexpected {camera} frame shape in {episode_dir}: {image.shape}")
                        frame[key] = image
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for reader in readers.values():
                    reader.close()
        print(f"Exported {len(successful_dirs)} successful episodes to {args.output}")
    finally:
        dataset.finalize()


if __name__ == "__main__":
    main()
