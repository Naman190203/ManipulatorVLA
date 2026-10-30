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
# Keep randomized grasp targets in the central, previously successful IK
# workspace, but spread them in both tabletop axes. The old list changed only
# x by 2 cm and left y fixed, so episodes looked nearly identical.
AUTONOMOUS_PICK_POSITIONS = tuple(
    (x, y)
    for y in (-0.12, -0.08, -0.04)
    for x in (0.20, 0.22, 0.24, 0.26, 0.28)
)
PLACEMENT_GRID_X = (0.12, 0.18, 0.24, 0.30, 0.36)
PLACEMENT_GRID_Y = (-0.16, -0.10, -0.04, 0.02, 0.08, 0.14, 0.20, 0.26)
MIN_OBJECT_SEPARATION = 0.055
MIN_PLATE_CLEARANCE = 0.115


def sample_positions(
    rng: random.Random,
    target_object: str,
    object_names: tuple[str, ...],
    target_position: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Vary the pick target across tested poses and randomize distractors."""
    candidates = [
        (x + rng.uniform(-0.001, 0.001), y + rng.uniform(-0.001, 0.001))
        for x in PLACEMENT_GRID_X
        for y in PLACEMENT_GRID_Y
        if abs(x - target_position[0]) >= 0.095
        and math.hypot(x, y) >= 0.18
    ]
    rng.shuffle(candidates)
    distractors = tuple(name for name in object_names if name != target_object)
    ordered_names = tuple(sorted(distractors, key=lambda name: OBJECT_RADIUS[name], reverse=True))
    chosen: dict[str, tuple[float, float]] = {target_object: target_position}

    def place(index: int) -> bool:
        if index == len(ordered_names):
            return True
        name = ordered_names[index]
        for point in candidates:
            if math.dist(point, PLATE_CENTER) < MIN_PLATE_CLEARANCE:
                continue
            if any(math.dist(point, other) < MIN_OBJECT_SEPARATION for other in chosen.values()):
                continue
            chosen[name] = point
            if place(index + 1):
                return True
            del chosen[name]
        return False

    if not place(0):
        raise RuntimeError("Could not find five collision-free object placements on the tabletop.")
    return chosen


def make_episode(rng: random.Random, index: int) -> dict:
    target_object, instruction = TASKS[index % len(TASKS)]
    ordered_objects = (target_object,) + tuple(name for name in MOVABLE_OBJECTS if name != target_object)
    # Cycle through a broad set of reachable, well-spaced source locations;
    # adding the task index gives reproducible variation under --seed.
    position_index = (index // len(TASKS)) % len(AUTONOMOUS_PICK_POSITIONS)
    target_position = AUTONOMOUS_PICK_POSITIONS[position_index]
    positions = sample_positions(rng, target_object, ordered_objects, target_position)
    placements = {}
    for name in ordered_objects:
        x, y = positions[name]
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
        "schema_version": 4,
        "seed": args.seed,
        "scene": "assets/scene_with_balls_cameras.xml",
        "camera_names": ["agentview", "topview"],
        "action_representation": "absolute_joint_target_rad",
        "joint_order": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_motor_joint"],
        "gripper_model": "Images/metal_2dof_gripper.urdf",
        "gripper_scale": 0.8,
        "gripper_mechanism": "single_actuated_revolute_joint_with_mirrored_revolute_jaws",
        "gripper_motor_limits_rad": [-0.41, 0.0],
        "episodes": [make_episode(rng, index) for index in range(args.episodes)],
    }
    args.output.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.episodes} randomized tasks to {args.output}")


if __name__ == "__main__":
    main()
