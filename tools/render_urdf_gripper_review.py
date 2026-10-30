"""Build and render a review-only MuJoCo scene from the supplied URDF.

This deliberately does not modify the production scene, controller, or data
schemas. It imports the URDF's primitive geometry, keeps one actuated revolute
joint plus the mirrored jaw joint, applies a uniform dimensional scale, and
renders an open/close review clip.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "Images" / "metal_2dof_gripper.urdf"
SOURCE_SCENE = ROOT / "assets" / "scene_with_balls_cameras.xml"
REVIEW_SCENE = ROOT / "assets" / "scene_with_urdf_gripper_review.xml"
REVIEW_VIDEO = ROOT / "Images" / "metal_2dof_gripper_review.mp4"
SCALE = 0.80
MOTOR_MIN = -0.41
MOTOR_MAX = 0.0


def values(text: str | None, count: int, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return fallback
    result = tuple(float(value) for value in text.split())
    if len(result) != count:
        raise ValueError(f"Expected {count} values, got {text!r}")
    return result


def xyz_origin(element: ET.Element) -> tuple[tuple[float, ...], tuple[float, ...]]:
    origin = element.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return values(origin.get("xyz"), 3, (0.0, 0.0, 0.0)), values(origin.get("rpy"), 3, (0.0, 0.0, 0.0))


def add_link_geometries(body: ET.Element, link: ET.Element, link_name: str) -> None:
    inertial = link.find("inertial")
    if inertial is not None:
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is not None and inertia_el is not None:
            pos, rpy = xyz_origin(inertial)
            inertia = [float(inertia_el.get(key, "0")) * SCALE**5 for key in ("ixx", "iyy", "izz")]
            ET.SubElement(body, "inertial", {
                "pos": " ".join(f"{v*SCALE:.8g}" for v in pos),
                "euler": " ".join(f"{v:.8g}" for v in rpy),
                "mass": f"{float(mass_el.get('value', '0')) * SCALE**3:.8g}",
                "diaginertia": " ".join(f"{v:.8g}" for v in inertia),
            })

    for kind in ("visual", "collision"):
        for index, block in enumerate(link.findall(kind)):
            geom_data = block.find("geometry")
            if geom_data is None or len(geom_data) != 1:
                continue
            shape = geom_data[0]
            pos, rpy = xyz_origin(block)
            attrs = {
                "name": f"urdf_{link_name}_{kind}_{index}",
                "pos": " ".join(f"{v*SCALE:.8g}" for v in pos),
                "euler": " ".join(f"{v:.8g}" for v in rpy),
                "group": "0" if kind == "visual" else "3",
            }
            if kind == "visual":
                mat = block.find("material")
                if mat is not None and mat.get("name"):
                    attrs["material"] = f"urdf_{mat.get('name')}"
                attrs.update(contype="0", conaffinity="0")
            else:
                if link_name in {"left_moving_jaw", "right_moving_jaw"}:
                    # The URDF's large pivot discs visually mesh and overlap
                    # by design, so keep those two shapes visual-only. The
                    # bars and fingertip pads collide with the opposite jaw,
                    # the arm, and grasped objects.
                    if index == 0:
                        attrs.update(contype="0", conaffinity="0")
                    else:
                        attrs.update(contype="3", conaffinity="3", condim="4", friction="1.3 0.01 0.001")
                else:
                    attrs.update(contype="2", conaffinity="1", condim="4", friction="1.3 0.01 0.001")

            if shape.tag == "box":
                full = values(shape.get("size"), 3, (0.01, 0.01, 0.01))
                attrs.update(type="box", size=" ".join(f"{v*SCALE/2:.8g}" for v in full))
            elif shape.tag == "cylinder":
                radius = float(shape.get("radius", "0.01")) * SCALE
                half_length = float(shape.get("length", "0.01")) * SCALE / 2
                attrs.update(type="cylinder", size=f"{radius:.8g} {half_length:.8g}")
            elif shape.tag == "sphere":
                attrs.update(type="sphere", size=f"{float(shape.get('radius', '0.01'))*SCALE:.8g}")
            else:
                continue
            ET.SubElement(body, "geom", attrs)

    # The supplied URDF intentionally has simplified collision on the long
    # jaw bars. Add tip-pad collision surfaces at the matching authored visual
    # locations so the jaws physically stop on each other and contact objects.
    if link_name in {"left_moving_jaw", "right_moving_jaw"}:
        sign = -1.0 if link_name == "left_moving_jaw" else 1.0
        ET.SubElement(body, "geom", {
                "name": f"urdf_{link_name}_tip_collision_pad",
                "type": "box",
                "pos": f"{sign*0.049*SCALE:.8g} {0.139*SCALE:.8g} 0",
                "euler": f"0 0 {(-0.13963 if sign < 0 else 0.13963):.8g}",
                "size": " ".join(f"{v*SCALE/2:.8g}" for v in (0.023, 0.035, 0.009)),
                "group": "3", "contype": "3", "conaffinity": "3",
                "condim": "4", "friction": "1.3 0.01 0.001",
            })


def build_body(link_name: str, links: dict[str, ET.Element], child_joints: dict[str, ET.Element]) -> ET.Element:
    body = ET.Element("body", {"name": f"urdf_{link_name}", "gravcomp": "1"})
    add_link_geometries(body, links[link_name], link_name)
    joint = child_joints.get(link_name)
    if joint is not None:
        pos, rpy = xyz_origin(joint)
        body.set("pos", " ".join(f"{v*SCALE:.8g}" for v in pos))
        body.set("euler", " ".join(f"{v:.8g}" for v in rpy))
        joint_name = "gripper_motor_joint" if joint.get("name") == "left_gear_joint" else "gripper_mirror_joint"
        limit = joint.find("limit")
        lower = MOTOR_MIN if joint_name == "gripper_motor_joint" else 0.0
        upper = MOTOR_MAX if joint_name == "gripper_motor_joint" else -MOTOR_MIN
        original_axis = joint.find("axis")
        axis = values(original_axis.get("xyz") if original_axis is not None else None, 3, (0.0, 0.0, 1.0))
        dyn = joint.find("dynamics")
        ET.SubElement(body, "joint", {
            "name": joint_name,
            "type": "hinge",
            "axis": " ".join(f"{v:.8g}" for v in axis),
            "limited": "true",
            "range": f"{lower:.8g} {upper:.8g}",
            "damping": f"{float(dyn.get('damping', '0.08')) if dyn is not None else 0.08:.8g}",
            "frictionloss": f"{float(dyn.get('friction', '0.01')) if dyn is not None else 0.01:.8g}",
            "armature": "0.001",
            "solreflimit": "0.001 1",
            "solimplimit": "0.99 0.999 0.001",
        })

    for child_joint in child_joints.values():
        parent = child_joint.find("parent").get("link")
        child = child_joint.find("child").get("link")
        if parent == link_name:
            body.append(build_body(child, links, child_joints))
    return body


def convert_scene(output_path: Path = REVIEW_SCENE, gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    source = ET.parse(SOURCE_SCENE)
    root = source.getroot()
    urdf_root = ET.parse(URDF_PATH).getroot()
    links = {element.get("name"): element for element in urdf_root.findall("link")}
    child_joints = {joint.find("child").get("link"): joint for joint in urdf_root.findall("joint")}

    link5 = next(element for element in root.iter("body") if element.get("name") == "link_5")
    tool_frame = next(element for element in link5.iter("frame") if element.get("pos") == "0 0 0.08")
    for child in list(tool_frame):
        tool_frame.remove(child)

    asset = root.find("asset")
    for material in list(asset.findall("material")):
        if material.get("name", "").startswith("urdf_"):
            asset.remove(material)
    ET.SubElement(asset, "material", {"name": "urdf_aluminium", "rgba": "0.72 0.75 0.78 1", "specular": "0.65", "shininess": "0.45"})
    ET.SubElement(asset, "material", {"name": "urdf_steel", "rgba": "0.42 0.45 0.48 1", "specular": "0.65", "shininess": "0.45"})
    ET.SubElement(asset, "material", {"name": "urdf_dark_steel", "rgba": "0.18 0.20 0.22 1", "specular": "0.45", "shininess": "0.30"})
    ET.SubElement(asset, "material", {"name": "urdf_bright_steel", "rgba": "0.88 0.89 0.90 1", "specular": "0.70", "shininess": "0.50"})

    # URDF +Y is the fingertip direction. Rotate it into the arm's forward +X
    # direction while keeping the two jaws separated along the tool's Y axis.
    base = build_body("gripper_base", links, child_joints)
    base.set("euler", f"0 0 {-math.pi/2:.8g}")
    tool_frame.append(base)
    ET.SubElement(base, "site", {"name": "grasp_site", "pos": f"0 {0.157*SCALE:.8g} 0", "size": "0.004", "rgba": "0 1 0 0"})

    contact = root.find("contact")
    for pair in list(contact.findall("pair")):
        if pair.get("geom1") in {"gripper_palm_collision", "left_finger_geom", "right_finger_geom"} or pair.get("geom2") in {"gripper_palm_collision", "left_finger_geom", "right_finger_geom"}:
            contact.remove(pair)
    equality = root.find("equality")
    for child in list(equality):
        equality.remove(child)
    ET.SubElement(equality, "joint", {
        "name": "urdf_mirrored_jaw_constraint",
        "joint1": "gripper_mirror_joint", "joint2": "gripper_motor_joint",
        "polycoef": "0 -1 0 0 0", "solref": "0.001 1", "solimp": "0.99 0.999 0.001",
    })
    actuator = next(element for element in root.find("actuator") if element.get("name") == "gripper_open_close")
    actuator.set("ctrlrange", f"{MOTOR_MIN} {MOTOR_MAX}")
    actuator.set("forcerange", "-2 2")
    actuator.set("kp", "8")
    actuator.set("dampratio", "1.2")

    root.set("model", "custom_5dof_manipulator_urdf_gripper")
    # Keep the demonstration object in the jaws during the long review intro;
    # this is a kinematic appearance review, while contact and joint stops
    # remain enabled.
    root.find("option").set("gravity", " ".join(f"{v:.8g}" for v in gravity))
    if root.find("visual") is None:
        visual = ET.Element("visual")
        ET.SubElement(visual, "global", {"offwidth": "960", "offheight": "720"})
        root.insert(2, visual)
    ET.indent(source, space="  ")
    source.write(output_path, encoding="utf-8", xml_declaration=True)


def frame_with_label(renderer: mujoco.Renderer, data: mujoco.MjData, camera, label: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    frame = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 46), fill=(18, 24, 30))
    draw.text((14, 13), label, fill=(245, 247, 250))
    return np.asarray(frame)


def render_review() -> None:
    model = mujoco.MjModel.from_xml_path(str(REVIEW_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_open_close")
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_ball_free")
    if min(grip_site, motor, ball_joint) < 0:
        raise RuntimeError("Preview scene is missing the gripper site, actuator, or red ball.")
    ball_addr = int(model.jnt_qposadr[ball_joint])
    ball_center = data.site_xpos[grip_site].copy()
    data.qpos[ball_addr:ball_addr + 3] = ball_center
    data.qpos[ball_addr + 3:ball_addr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = ball_center
    camera.distance = 0.48
    camera.azimuth = 150
    camera.elevation = -20
    renderer = mujoco.Renderer(model, height=720, width=960)
    fps = 24
    with imageio.get_writer(REVIEW_VIDEO, fps=fps, codec="libx264", quality=8) as video:
        # First give a few seconds of whole-arm context using the authored
        # agentview camera, then a close shot of the actual URDF geometry.
        wide_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        for _ in range(fps * 2):
            video.append_data(frame_with_label(renderer, data, wide_cam, "URDF GRIPPER REVIEW  |  whole manipulator"))
        for _ in range(fps * 2):
            video.append_data(frame_with_label(renderer, data, camera, "Scaled URDF geometry  |  0.80x"))
        # Show open, close around a 40 mm red ball, hold, and reopen. Hard
        # revolute limits plus real jaw collision geoms constrain movement.
        segments = ((0.0, 1.2), (MOTOR_MIN, 2.0), (MOTOR_MIN, 1.0), (0.0, 2.0))
        current = 0.0
        for target, seconds in segments:
            count = int(fps * seconds)
            start = current
            for index in range(count):
                fraction = (index + 1) / count
                command = start + (target - start) * fraction
                data.ctrl[motor] = command
                mujoco.mj_step(model, data)
                video.append_data(frame_with_label(renderer, data, camera, f"Gripper motor {command:+.3f} rad  |  limited [{MOTOR_MIN:.2f}, {MOTOR_MAX:.2f}]"))
            current = target
    renderer.close()
    print(f"Review scene: {REVIEW_SCENE}")
    print(f"Review video: {REVIEW_VIDEO}")
    print(f"Model: {model.nq} qpos, {model.nv} dofs, {model.nu} actuators, {model.neq} equality constraints")


def main() -> None:
    global SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--output", type=Path, default=REVIEW_SCENE)
    parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--no-render", action="store_true", help="Only write the generated MJCF scene")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")
    SCALE = args.scale
    convert_scene(args.output, tuple(args.gravity))
    if not args.no_render:
        render_review()


if __name__ == "__main__":
    main()
