from pathlib import Path
import mujoco

# Convert the original URDF to MJCF using the MuJoCo installation on your PC.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "assets" / "urdf_configuration.xml"
OUTPUT = PROJECT_ROOT / "assets" / "urdf_configuration_stabilized_mjcf.xml"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Could not find input URDF: {SOURCE}")

print(f"Using MuJoCo {mujoco.__version__}")
spec = mujoco.MjSpec.from_file(str(SOURCE))

# Tune only the five arm joints. Do not overwrite the imported body/root pose.
arm_joint_names = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5"}
updated = []
for joint in spec.joints:
    if joint.name in arm_joint_names:
        # MjSpec JointSpec uses damping and frictionloss for MJCF joint dynamics.
        joint.damping = 1.0
        joint.frictionloss = 0.05
        updated.append(joint.name)

# Compile once to validate the model before writing the generated MJCF.
model = spec.compile()
OUTPUT.write_text(spec.to_xml(), encoding="utf-8")

print(f"Converted MJCF written to: {OUTPUT}")
print(f"Arm joints with damping applied: {', '.join(updated) if updated else 'none found'}")
print(f"Compiled model validation passed ({model.njnt} joints, {model.nbody} bodies).")
print("Launch with:")
print(f'py -m mujoco.viewer --mjcf "{OUTPUT}"')
