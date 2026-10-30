# URDF-derived metal gripper

The active MuJoCo scene now uses the geometry in
`Images/metal_2dof_gripper.urdf`, uniformly scaled to 0.80× to suit the
manipulator tool and the 36–40 mm objects. The source has two mirrored
revolute jaw joints. Only `gripper_motor_joint` is actuated; the opposite jaw
is constrained to the negative of that angle. The public state/action remains
six absolute joint positions in radians, with this motor as the sixth value.

The motor is bounded to `[-0.41, 0.0]` rad: zero is open and negative values
close. Both revolute joints have hard joint stops. The imported URDF collision
shapes are active, with additional collision pads at the jaw tips. The large
overlapping pivot discs stay visual-only; the moving jaw bars and tips collide
with each other, the arm, and objects. At the close limit, the tip pads meet.
The dashboard and bridge clamp commands to the actuator and joint ranges.

The source URDF, scaled preview scene and rendered review video are preserved
under `Images/` and `assets/`. `tools/render_urdf_gripper_review.py` rebuilds
the preview, while passing `--output assets\scene_with_balls_cameras.xml
--gravity 0 0 -9.81 --no-render` rebuilds the production scene.

## Start the manipulator and dashboard

From PowerShell:

```powershell
cd D:\Naman_VLA
python src\sim_bridge.py
```

Open `http://127.0.0.1:8765/`. Choose **Gripper motor · mirrored revolute
jaws**. Tap or hold **↓/←** to close; use **↑/→** to open. The limit readout
shows the current simulator range.

To open the MuJoCo viewer and animate the gripper automatically:

```powershell
python tools\preview_parallel_gripper.py
```

## Arm motion settling

The arm joints retain the damping and armature settings used to reduce
uncommanded movement while a single joint is moving.
