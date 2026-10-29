# VLA dataset collection

This project now records one self-contained raw episode per demonstration.
It is designed for high-quality VLA fine-tuning rather than only saving camera
videos.

## Required data

Every successful training sample needs:

- A natural-language instruction and stable task identifier.
- Synchronized `agentview.mp4` and `gripperview.mp4` RGB videos, plus a frame
  index that maps every video frame to the exact simulation timestamp.
- Robot proprioception: measured joint position and velocity.
- The action actually commanded at that frame. This bridge records absolute
  joint targets and also stores a frame-to-frame delta for model conversion.
- Initial poses for every movable object, a reproducible random seed, scene
  name, camera names, and joint order.
- A terminal success label. Failed or interrupted demonstrations should be
  retained separately for debugging, not silently mixed into positive training.

The recorder writes these items in `episode.json`, `frames.jsonl`, one MP4 per
camera, and the dataset-level `index.jsonl`. Video avoids the file-count and
storage overhead of JPEG sequences; `frames.jsonl` remains the source of truth
for timestamps and actions.

## Generate diverse tasks

```powershell
python src/create_task_suite.py --episodes 100 --seed 42 --output dataset_tasks.json
```

The generator balances five plate-placement instructions and samples collision-
spaced positions for all movable balls and cubes. The shallow plate stays fixed
because it is a static part of the current scene.

## Record demonstrations

Start the bridge and dashboard in separate terminals:

```powershell
python src/sim_bridge.py
python src/record_vla_dataset.py --tasks dataset_tasks.json --output vla_dataset_raw --duration 20 --video-fps 10
```

For each task the recorder resets the robot and all movable objects to the
manifested positions. Demonstrate the task from the dashboard, start capture
when ready, then mark the result successful only if the target is visibly on
the plate. The recorder captures a demonstration; it does not autonomously
solve the manipulation task. The bridge records the dashboard's current
commanded joint targets, so no second control client is needed.

## Before training

Review at least a sample of both camera streams and exclude episodes where the
target is occluded, the object is not placed correctly, or the action stream is
mostly stationary. Keep train/validation/test splits disjoint by randomized
placement seed. Convert only successful raw episodes into the final trainer
format (for example LeRobot video + Parquet), preserving the two camera names,
language, joint order, and action convention.
