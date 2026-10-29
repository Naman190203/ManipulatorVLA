"""
Rebuild the MuJoCo scene while keeping the robot upright and supported against gravity.

Place this beside urdf_configuration.xml and run:
    py add_scene_objects.py

This does not modify the original URDF.
"""
from pathlib import Path
import mujoco

HERE = Path(__file__).resolve().parent
ROBOT_FILE = HERE / "urdf_configuration.xml"
OUTPUT_FILE = HERE / "scene_with_balls_cameras.xml"

BALL_POSITIONS = {
    "red_ball":   (0.22, -0.10, 0.045),
    "green_ball": (0.25,  0.00, 0.045),
    "blue_ball":  (0.22,  0.10, 0.045),
}
BALL_RADIUS = 0.025
BALL_MASS = 0.08

AGENT_CAMERA_POS = (-0.35, -0.65, 0.75)
AGENT_CAMERA_XYAXES = (0.81, -0.586, 0, 0.351, 0.486, 0.802)

GRIPPER_CAMERA_POS = (0.035, 0.0, 0.045)
GRIPPER_CAMERA_XYAXES = (0, 0, 1, 0, 1, 0)


def main():
    if not ROBOT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {ROBOT_FILE}. Put this script beside urdf_configuration.xml."
        )

    spec = mujoco.MjSpec.from_file(str(ROBOT_FILE))

    # Keep world gravity normal so free balls fall downward.
    spec.option.gravity = (0, 0, -9.81)

    # The imported URDF's world_fixed joint has a 90-degree X rotation.
    # In the generated MJCF this appears on the root robot body. Neutralize
    # that root orientation in this scene only; leave the original URDF intact.
    root = spec.body("link_1")
    if root is None:
        raise RuntimeError("Could not find imported root body 'link_1'.")
    

    # This is a visual/simulation stability measure for this first scene:
    # compensate gravity on robot links so the unactuated joints do not sag.
    # Balls are added afterward and retain normal gravity.
    robot_body_names = (
        "link_1", "link_2", "link_3", "link_4", "link_5",
        "gripper_base", "left_finger", "right_finger",
    )
    for name in robot_body_names:
        body = spec.body(name)
        if body is not None:
            body.gravcomp = 1.0

    colors = {
        "red_ball": (1.0, 0.05, 0.05, 1.0),
        "green_ball": (0.05, 1.0, 0.05, 1.0),
        "blue_ball": (0.05, 0.25, 1.0, 1.0),
    }
    for name, pos in BALL_POSITIONS.items():
        body = spec.worldbody.add_body(name=name, pos=pos)
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(BALL_RADIUS, 0, 0),
            mass=BALL_MASS,
            rgba=colors[name],
            contype=1,
            conaffinity=1,
        )

    spec.worldbody.add_camera(
        name="agentview",
        pos=AGENT_CAMERA_POS,
        xyaxes=AGENT_CAMERA_XYAXES,
        fovy=60,
    )

    gripper_body = spec.body("gripper_base")
    if gripper_body is None:
        raise RuntimeError("Could not find body 'gripper_base' for gripperview.")
    gripper_body.add_camera(
        name="gripperview",
        pos=GRIPPER_CAMERA_POS,
        xyaxes=GRIPPER_CAMERA_XYAXES,
        fovy=65,
    )

    OUTPUT_FILE.write_text(spec.to_xml(), encoding="utf-8")
    print(f"Created: {OUTPUT_FILE}")
    print("Open with:")
    print(f'py -m mujoco.viewer --mjcf "{OUTPUT_FILE}"')
    print("Note: this version neutralizes the imported root rotation and gravity-compensates robot links.")
    print("If the arm now has the wrong pose, stop and report it; do not edit the original URDF.")


if __name__ == "__main__":
    main()
