# Parallel translating gripper driven by a revolute motor

The active model uses one commanded revolute joint, `gripper_motor_joint`, to
drive the gripper. A second gear joint counter-rotates, and two equality
constraints convert the motor angle into matching slider motion on the left
and right fingers. The opposed slider axes make the long jaws translate
together without changing orientation.

The motor range is `0` radians (open) to `-0.48` radians (maximum close). The
autonomous collector uses object-specific closing angles for the balls and
cubes. The bridge, manual dashboard, collector, and SmolVLA features all use
the motor angle in radians. Internal slider coordinates are mechanism details
and are not exposed as policy actions.

## Run and inspect

From PowerShell in `D:\Naman_VLA`:

```powershell
python src\sim_bridge.py
```

This opens the MuJoCo viewer and browser dashboard at
`http://127.0.0.1:8765`. Select **Gripper motor · revolute open / close** and
jog it: negative angle closes the parallel fingers. To run the same scene
with an automatic gripper motion focused for inspection:

```powershell
python tools\preview_parallel_gripper.py
```

## Arm motion settling

The arm joints now include reflected armature and passive damping alongside
their position servos. A single-joint trajectory check showed the largest
uncommanded joint displacement fall from about `0.0059 rad` to `0.0015 rad`
(about `0.34°` to `0.08°`) over a 0.45-radian move. Joint coupling remains in
the dynamics, but visible oscillation is strongly reduced.

Record new demonstrations before training. The exporter checks the six-joint
order and rejects recordings that lack the new `gripper_motor_joint` metadata.
