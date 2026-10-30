# SmolVLA simulator feedback loop

This folder adds a local task dashboard and an adapter between the existing MuJoCo bridge and a SmolVLA inference endpoint. It is separate from the data recorder and does not change the existing scene, bridge, or dataset files.

## Data contract

The adapter sends one JSON object to `POST /predict` for each simulator observation:

```json
{
  "schema": "lerobot.smolvla.observation.v1",
  "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "observation.state.names": ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "gripper_motor_joint.pos"],
  "observation.images": {
    "observation.images.agentview": {"encoding": "jpeg", "width": 640, "height": 480, "data_base64": "..."},
    "observation.images.topview": {"encoding": "jpeg", "width": 640, "height": 480, "data_base64": "..."}
  },
  "task": "Put the red ball on the plate.",
  "action_contract": {
    "feature": "action", "names": ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "gripper_motor_joint.pos"],
    "shape": [6], "dtype": "float32", "units": "radian", "representation": "absolute_joint_target_rad"
  }
}
```

The six state values, action names, and camera names match this project's LeRobot exporter (`observation.state`, `observation.images.agentview`, `observation.images.topview`). The gripper actuator is one revolute motor joint; an equality constraint mirrors the other revolute jaw. Its allowed range is `[-0.41, 0.0]` radians. The extra `simulator` object in the request is adapter context and is **not** a model feature. The real policy server decodes the JPEGs, calls LeRobot's inference preprocessing and postprocessing pipelines, then returns a postprocessed action:

```json
{
  "schema": "lerobot.smolvla.prediction.v1",
  "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "done": false,
  "success": null,
  "message": "Optional operator-facing status"
}
```

`action` must contain six finite, in-range **absolute joint targets in radians** in bridge order: `joint_1`, `joint_2`, `joint_3`, `joint_4`, `joint_5`, `gripper_motor_joint`. It is sent to `/ws/commands` as `target_position_rad`. Negative gripper motor angles close the jaws; zero opens them. Return `done: true` when the policy considers the task finished. Otherwise the loop continues until the user presses Stop. The controller rejects the prediction if its shape, values, or joint limits do not match.

## Manual no-motion pipeline check (PowerShell)

Open three PowerShell windows in `D:\Naman_VLA`. Use the same Python environment in all windows; `requirements-bridge.txt` contains the bridge dependencies.

**Window 1 — start the simulator bridge:**

```powershell
cd D:\Naman_VLA
python src\sim_bridge.py
```

**Window 2 — start a contract-checking mock policy:**

```powershell
cd D:\Naman_VLA
python smolvla_feedback\mock_policy.py
```

This mock validates the incoming feature keys and dimensions, then echoes the simulator's current commanded position. It will not attempt to pick or move an object.

**Window 3 — start the dashboard/controller:**

```powershell
cd D:\Naman_VLA
python smolvla_feedback\server.py
```

Open `http://127.0.0.1:8780`, enter a task such as `Put the red ball on the plate.`, and click **Start control loop**. Verify that both camera panels show images, the feedback sequence advances, policy latency appears, and the six-value last action is shown. The mock status should say it echoed the current target. Click **Stop**; the controller sends a hold target at the latest measured joint pose.

The bridge health endpoint can be checked separately:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health | ConvertTo-Json -Depth 5
```

## Run a local fine-tuned SmolVLA checkpoint

Install the LeRobot checkout and SmolVLA dependencies in the Python environment used for the policy server. Use the checkpoint directory produced by fine-tuning:

```powershell
python -m pip install -e ".\lerobot[smolvla]"
python smolvla_feedback\lerobot_policy_server.py --model .\outputs\smolvla_finetuned --device cuda
```

If running CPU inference, use `--device cpu`. Keep the simulator and controller windows running; reload the dashboard and start the task. To use a different model-server port, launch the controller with `--policy-url http://127.0.0.1:PORT/predict`.

The inference endpoint returns `done: false` because the current exported observations contain joint state and images but no simulator object pose or completion labels available to this runtime. For automatic stopping, have the inference server or a future task evaluator return `done: true` and optionally `success: true/false`. Until then use the dashboard's Stop button after visually confirming completion.

## Files

- `server.py`: Dashboard API, bridge feedback consumer, policy request, action validation, and command sender.
- `dashboard.html`: Task entry, simulator views, status, and last action.
- `mock_policy.py`: Safe, no-motion endpoint for validating the data contract and command loop.
- `lerobot_policy_server.py`: Local model adapter using `SmolVLAPolicy`, `build_inference_frame`, and the saved LeRobot pre/postprocessors.
