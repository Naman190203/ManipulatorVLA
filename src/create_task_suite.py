"""Create reproducible, collision-spaced VLA data-collection episodes.

This script creates task and placement metadata only. Run record_vla_dataset.py
while the bridge and dashboard are open to capture a successful demonstration
for every generated episode.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


MOVABLE_OBJECTS = ("red_ball", "green_ball", "blue_ball", "orange_cube", "purple_cube")
TASKS = (
    ("red_ball", "Put the red ball on the plate."),
    ("green_ball", "Put the green ball on the plate."),
    ("blue_ball", "Put the blue ball on the plate."),
    ("orange_cube", "Pick up the orange cube and place it on the plate."),
    ("purple_cube", "Pick up the purple cube and place it on the plate."),
)
OBJECT_Z = {"red_ball": 0.020, "green_ball": 0.020, "blue_ball": 0.020,
            "orange_cube": 0.018, "purple_cube": 0.018}
OBJECT_RADIUS = {"red_ball": 0.042, "green_ball": 0.042, "blue_ball": 0.042,
                 "orange_cube": 0.045, "purple_cube": 0.045}
PLATE_CENTER = (0.22, -0.26)


def sample_position(rng: random.Random, accepted: list[tuple[float, float]], radius: float) -> tuple[float, float]:
    """Sample a tabletop point away from the plate and other objects."""
    for _ in range(500):
        x = rng.uniform(0.15, 0.31)
        y = rng.uniform(-0.20, 0.12)
        if math.dist((x, y), PLATE_CENTER) < radius + 0.10:
            continue
        if all(math.dist((x, y), other) >= radius + 0.045 for other in accepted):
            accepted.append((x, y))
            return x, y
    raise RuntimeError("Could not sample a collision-free object placement.")


def make_episode(rng: random.Random, index: int) -> dict:
    target_object, instruction = TASKS[index % len(TASKS)]
    ordered_objects = (target_object,) + tuple(name for name in MOVABLE_OBJECTS if name != target_object)
    accepted: list[tuple[float, float]] = []
    placements = {}
    for name in ordered_objects:
        x, y = sample_position(rng, accepted, OBJECT_RADIUS[name])
        placements[name] = [round(x, 4), round(y, 4), OBJECT_Z[name]]
    return {
        "episode_id": f"episode_{index:05d}",
        "task_id": f"place_{target_object}_on_plate",
        "instruction": instruction,
        "target_object": target_object,
        "target_receptacle": "placement_plate",
        "object_positions": placements,
        "collection_note": "Record only after the object is placed on the plate and the task is visibly complete.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create randomized VLA collection tasks.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output", type=Path, default=Path("dataset_tasks.json"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")

    rng = random.Random(args.seed)
    suite = {
        "schema_version": 1,
        "seed": args.seed,
        "scene": "assets/scene_with_balls_cameras.xml",
        "camera_names": ["agentview", "gripperview"],
        "action_representation": "absolute_joint_target_rad",
        "episodes": [make_episode(rng, index) for index in range(args.episodes)],
    }
    args.output.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.episodes} randomized tasks to {args.output}")


if __name__ == "__main__":
    main()
