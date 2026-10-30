# MuJoCo feedback and command bridge

Run commands from the project root (`D:\Naman_VLA`). The bridge runs the simulation locally and exposes timestamped observations and
bounded joint-position targets over two WebSocket endpoints. MuJoCo state is
owned by the simulation loop; clients cannot edit model state directly.

## Start it

From PowerShell in this folder:

```powershell
python -m pip install -r requirements-bridge.txt
python src/sim_bridge.py
```

Open the browser dashboard at [http://127.0.0.1:8765/](http://127.0.0.1:8765/)
or `/dashboard`. It shows both live camera frames and lets you select a joint
and jog it with the arrow keys. The default jog is 0.25 degrees per step; use
the step-size and hold-repeat fields to adjust it. Click **Hold current
measured pose** to stop the selected motion and hold the measured arm pose.
If `MUJOCO_BRIDGE_TOKEN` is set, enter that token under **Connection token**.
Camera streams default to 640×480; `--width` and `--height` can override this.

The MuJoCo viewer opens on `agentview`; use its camera selector to switch to
`topview`. The API listens on `127.0.0.1:8765` by default. To run without a
viewer window, add `--no-viewer`.

The optional desktop monitor and command panel can also be opened in a second
PowerShell window:

```powershell
python src/bridge_client.py
```

It displays both camera streams and measured joint state. The six target fields
start at the current joint positions. Send a target to move; **Hold current
pose** sends the latest measured positions. The bridge validates all targets
and applies its speed limits. Each field shows that joint's effective position
range and target slew rate.

## Endpoints

- `GET http://127.0.0.1:8765/health` reports joint order, target limits, and
  endpoint names.
- `ws://127.0.0.1:8765/ws/feedback` streams observations at 10 Hz by default.
- `ws://127.0.0.1:8765/ws/commands` accepts target positions and returns an
  acknowledgement for each command.

Feedback JSON contains `sim_time`, `timestamp_unix`, six revolute joint
positions in radians and velocities in radians/second, current commanded
targets, and both camera frames as base64-encoded JPEG images. The gripper's
revolute motor drives two linked slider fingers that move in parallel. Decode
each image from
`images.agentview.data_base64` or `images.topview.data_base64`.

Commands must contain exactly six target positions in this order:

```json
{
  "type": "command",
  "seq": 1,
  "target_position_rad": [0.0, 0.2, 0.4, -0.2, 0.0, -0.2]
}
```

These are absolute position targets, in the order `joint_1`, `joint_2`,
`joint_3`, `joint_4`, `joint_5`, `gripper_motor_joint`. Current arm limits are:

| Joint | Position range |
|---|---:|
| joint 1 | 0° to 360° |
| joint 2 | 0° to +180° (the upper semicircle from horizontal to opposite horizontal) |
| joint 3 | 0° to +115.16° |
| joint 4 | −119.7° to 0° |
| joint 5 | 0° to 360° |

Joint 3 and 4 zero positions remain aligned with their preceding links. Joint
3 bends positive and joint 4 bends negative, per the specified opposite
directions. The gripper motor interval is `[-0.48, 0.0]` radians; `0.0` is
open and negative values close it, moving the fingers together while they
remain parallel. Each allowed interval is enforced both by
the joint limit and the motor position actuator. Out-of-range values are
rejected. Arm targets are slewed at no more than 0.75 rad/s and the gripper at
1 rad/s; these are command slew limits, not measured guarantees of physical
joint speed. The arm joints add reflected armature and passive damping to the
1.3-ratio position servos. The implicit-fast integrator helps the joints settle
at requested positions.
Commands must have a strictly increasing `seq` on each connection. If
commands stop for 12 seconds, the bridge eases to a hold at the measured pose.

## Dataset episode reset

The command socket also accepts a bounded `reset` message for automated VLA
collection. It restores the initial robot pose and moves only the five named
free objects inside the safe tabletop region:

```json
{
  "type": "reset",
  "seq": 0,
  "object_positions": {
    "red_ball": [0.22, -0.10, 0.02],
    "green_ball": [0.25, 0.00, 0.02]
  }
}
```

See `docs/VLA_DATASET.md` for task generation and synchronized demonstration
recording. Reset is intended for local episode setup; it cannot alter robot
state, cameras, object identity, physics, or arbitrary MuJoCo state.

## Remote inference

Keep the bridge on localhost for local clients. For Colab, use an authenticated
tunnel that forwards to `127.0.0.1:8765`; do not bind an unauthenticated bridge
to a public interface. Set a long random token in `MUJOCO_BRIDGE_TOKEN` before
starting the server. Each WebSocket connection must first send
`{"token":"..."}`. The feedback and command WebSockets are separate so a
Colab client can independently receive frames and submit action chunks.

The predicted action must be mapped to the six joint targets above. SmolVLA's
input keys, feature shape, units, normalization, and action ordering must match
the data used to fine-tune the policy for this manipulator before using it for
closed-loop control.
