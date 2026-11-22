"""Open both releases on a detachable cookware handle and put it beside the pan."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, as_numpy, on_reset, on_step, plate, rotation, segment
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.5

ORIGIN = np.array([0.335, 0.091, TOP + 0.053])
RIM_R = 0.103
BUTTON_X = -0.049
BUTTON_Z = 0.019
PAN = (RIM_R, 0.0, 0.0)
PIVOT = (-0.034, 0.0, -0.0035)
LAY_ASIDE = (0.197, 0.237)
CHARCOAL = (0.095, 0.110, 0.123)
GRAY = (0.24, 0.267, 0.285)
DARK = (0.065, 0.072, 0.078)
BOTTOM = (0.19, 0.22, 0.24)
STEEL = (0.70, 0.73, 0.75)
RED = (0.62, 0.13, 0.12)


def _world(p):
    return tuple(ORIGIN + np.asarray(p))


pan = [
    CylShape((RIM_R, 0, -0.0491), 0.080, 0.0078, BOTTOM),
    CylShape((RIM_R, 0, -0.0449), 0.077, 0.0013, CHARCOAL, collide=False),
    CylShape((RIM_R, 0, -0.0441), 0.024, 0.0003, (0.37, 0.12, 0.13), collide=False),
]
for angle in np.linspace(0, math.tau, 40, endpoint=False):
    x, y = math.cos(angle), math.sin(angle)
    inner = np.array([RIM_R + 0.0785 * x, 0.0785 * y, -0.047])
    outer = np.array([RIM_R + (RIM_R - 0.0015) * x, (RIM_R - 0.0015) * y, -0.0007])
    pan.append(plate(inner, outer, 0.0165, 0.003, (x, y, 0.0), GRAY))
    a, b = angle - math.pi / 40, angle + math.pi / 40
    pan.append(segment((RIM_R + (RIM_R - 0.0015) * math.cos(a), (RIM_R - 0.0015) * math.sin(a), 0.0),
                       (RIM_R + (RIM_R - 0.0015) * math.cos(b), (RIM_R - 0.0015) * math.sin(b), 0.0),
                       0.0033, STEEL))

handle = [
    BoxShape((-0.098, 0, 0.022), (0.124, 0.035, 0.030), DARK),
    CylShape((-0.159, 0, 0.022), 0.0175, 0.029, DARK),
    BoxShape((-0.025, 0, 0.020), (0.046, 0.035, 0.028), CHARCOAL),
    BoxShape((-0.004, 0, 0.016), (0.055, 0.028, 0.016), CHARCOAL),
    BoxShape((0.0125, 0, 0.0048), (0.017, 0.022, 0.0075), STEEL),
    BoxShape((0.0085, 0, -0.0030), (0.009, 0.022, 0.011), GRAY),
    BoxShape((-0.109, 0, 0.0372), (0.056, 0.016, 0.0015), GRAY, collide=False),
]
handle_links = [
    Link("jaw", (
        BoxShape((-0.009, 0, -0.0064), (0.010, 0.023, 0.016), STEEL),
        BoxShape((-0.0030, 0, -0.0088), (0.016, 0.022, 0.005), STEEL),
        BoxShape((-0.024, 0, -0.0050), (0.022, 0.021, 0.009), GRAY),
    ), 0.029, _world((-0.014, 0, -0.004)), axis="X", slide=True,
         limits=(-0.015, 0.0005), stiffness=780, spring_damping=8.0),
    Link("lever", (
        BoxShape((-0.087, 0, 0.0010), (0.111, 0.027, 0.009), CHARCOAL),
        BoxShape((-0.123, 0, -0.0034), (0.036, 0.024, 0.003), GRAY),
    ), 0.034, _world(PIVOT), axis="Y", limits=(-25.0, 0.2), stiffness=0.21, spring_damping=0.021),
]
for side, sign in (("left", 1), ("right", -1)):
    handle_links.append(Link(
        side + "_button",
        (
            BoxShape((BUTTON_X, sign * 0.0225, BUTTON_Z), (0.0165, 0.010, 0.014), RED),
            BoxShape((BUTTON_X, sign * 0.0277, BUTTON_Z), (0.010, 0.0005, 0.004), STEEL, collide=False),
        ),
        0.006, _world((BUTTON_X, 0, BUTTON_Z)), axis="Y", slide=True,
        limits=(-0.006, 0.0003) if sign > 0 else (-0.0003, 0.006),
        stiffness=330, spring_damping=2.6,
    ))

OBJECTS = [
    Body.build("pan", tuple(ORIGIN), 0.53, pan, color=GRAY, friction=(0.61, 0.50),
               contact_offset=0.00065, damping=(0.0, 0.10), solver_iterations=40),
    Linkage.build("handle", tuple(ORIGIN), 0.137, handle, handle_links, floating=True,
                  friction=(0.85, 0.72), color=DARK, contact_offset=0.00050, solver_iterations=48),
]

def _box_bottom(shapes, height, axis):
    return min(float(height + axis @ shape.centre - np.abs(axis) @ (np.asarray(shape.size) / 2.0))
               for shape in shapes if isinstance(shape, BoxShape) and shape.collide)


_state = {}

@on_reset
def _reset():
    _state.clear()
    _state.update(ids={}, arms={}, together=0, unlocked=False, unsafe=False,
                  detached=False, score=0.0, step=0, max_pan_raise=0.0)

_reset()


@on_step
def _update(env):
    import torch

    base = getattr(env, "unwrapped", env)
    grip = base.scene["handle"]
    pan_body = base.scene["pan"]
    if not _state["ids"]:
        _state["ids"] = {key: grip.joint_names.index(f"{key}_turn")
                         for key in ("jaw", "lever", "left_button", "right_button")}
    idx = _state["ids"]
    q = as_numpy(grip.data.joint_pos)[0]
    button_l = -float(q[idx["left_button"]])
    button_r = float(q[idx["right_button"]])
    pan_pose = as_numpy(pan_body.data.root_link_pose_w)[0]
    pan_vel = as_numpy(pan_body.data.root_link_vel_w)[0]
    pan_rot = rotation(pan_pose[3:7])
    pan_centre = pan_pose[:3] + pan_rot @ PAN
    pan_bottom = pan_centre[2] - 0.053 * pan_rot[2, 2]
    raise_centre = pan_bottom - TOP
    tilt = math.acos(float(np.clip(pan_rot[2, 2], -1.0, 1.0)))
    supported = abs(raise_centre) <= 0.007 and tilt <= math.radians(9.0)

    shape = as_numpy(grip.data.root_link_pose_w)[0]
    handle_vel = as_numpy(grip.data.root_link_vel_w)[0]
    rot = rotation(shape[3:7])
    front = shape[:3] + rot @ [0.020, 0.0, 0.0]
    tail = shape[:3] + rot @ [-0.172, 0.0, 0.020]
    span = front - tail
    pan_delta = pan_centre[:2] - tail[:2]
    nearest = tail[:2] + span[:2] * np.clip(pan_delta @ span[:2] / (span[:2] @ span[:2]), 0.0, 1.0)
    handle_off = np.linalg.norm(nearest - pan_centre[:2]) >= RIM_R + 0.045

    if not _state["detached"]:
        _state["max_pan_raise"] = max(_state["max_pan_raise"], raise_centre)
        _state["unsafe"] |= not supported
    _state["together"] = _state["together"] + 1 if min(button_l, button_r) >= 0.0035 and supported else 0
    if _state["together"] >= 2 and not _state["unlocked"]:
        _state["unlocked"] = True
        target = torch.tensor([[-0.014, math.radians(-25.0)]], dtype=torch.float32, device=grip.device)
        grip.set_joint_position_target_index(target=target, joint_ids=[idx["jaw"], idx["lever"]])
        grip.write_data_to_sim()
    open_now = q[idx["jaw"]] <= -0.010 and q[idx["lever"]] <= math.radians(-19)
    if _state["unlocked"] and open_now and handle_off:
        _state["detached"] = True

    arm_pos = {}
    for side, anchor in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["arms"]:
            _state["arms"][side] = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        joints = as_numpy(arm.data.joint_pos)[0][_state["arms"][side]]
        arm_pos[side], _ = ik.tcp_pose(joints, anchor)
    clear = True
    for hand in arm_pos.values():
        projected = tail + span * np.clip((hand - tail) @ span / (span @ span), 0.0, 1.0)
        pan_radial = max(0.0, np.linalg.norm(hand[:2] - pan_centre[:2]) - RIM_R)
        pan_height = max(0.0, hand[2] - pan_centre[2], pan_bottom - hand[2])
        clear &= np.linalg.norm(hand - projected) >= 0.065 and math.hypot(pan_radial, pan_height) >= 0.036
    lever = float(q[idx["lever"]])
    lever_rot = np.array([[math.cos(lever), 0.0, math.sin(lever)], [0.0, 1.0, 0.0],
                          [-math.sin(lever), 0.0, math.cos(lever)]])
    pivot = np.asarray(PIVOT)
    lever_base = shape[:3] + rot @ (pivot - lever_rot @ pivot)
    lower = min(
        _box_bottom(handle, shape[2], rot[2]),
        _box_bottom(handle_links[0].shapes, shape[2] + rot[2, 0] * q[idx["jaw"]], rot[2]),
        _box_bottom(handle_links[1].shapes, lever_base[2], (rot @ lever_rot)[2]),
    )
    resting = abs(lower - TOP) <= 0.004 and np.linalg.norm(handle_vel[:3]) < 0.018
    settled = np.linalg.norm(pan_vel[:3]) < 0.010 and np.linalg.norm(handle_vel[3:]) < 0.30
    _state["score"] = 0.0
    if not _state["unsafe"] and _state["unlocked"]:
        if open_now:
            _state["score"] = 0.25
        if _state["detached"] and handle_off:
            _state["score"] = 0.65
            if resting and supported and settled and clear:
                _state["score"] = 1.0
    _state["step"] += 1
    if os.getenv("PAN_HANDLE_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"DETACH {_state['step']:03} b={button_l*1000:4.1f}/{button_r*1000:4.1f} jaw={q[idx['jaw']]*1000:6.2f} "
              f"lever={math.degrees(q[idx['lever']]):6.2f} lift={raise_centre*1000:6.2f} tilt={math.degrees(tilt):.2f} "
              f"open={open_now} detached={_state['detached']} unsafe={_state['unsafe']} "
              f"bottom={(lower-TOP)*1000:.2f} rest={resting} clear={clear} score={_state['score']:.2f} "
              f"h={np.round(shape[:3]-[0,0,TOP], 4)} btn={np.round(shape[:3]+rot@[BUTTON_X,0,BUTTON_Z]-[0,0,TOP],4)} l={np.round(arm_pos['left']-[0,0,TOP], 4)}", flush=True)


def score(objects):
    return min(_state["score"], 0.65)


def success(objects):
    return _state["score"] == 1.0


def demo():
    x, y, z = _world((BUTTON_X, 0.0, BUTTON_Z + 0.006))
    z -= 0.008
    open_grip, squeezed = -0.040, -0.014
    if os.getenv("PAN_HANDLE_SINGLE") == "1":
        return {"left": [
            Waypoint(2.4, (x, y + 0.050, TOP + 0.139), -0.021, "move above one button"),
            Waypoint(1.2, (x, y + 0.050, z), -0.021, "lower beside the left button"),
            Waypoint(0.85, (x, y + 0.037, z), -0.021, "push the left button only"),
            Waypoint(0.65, (x, y + 0.037, z), -0.021, "hold the single release"),
            Waypoint(1.1, (x, y + 0.061, TOP + 0.145), -0.021, "leave the locked handle"),
        ]}
    pos = (x, y, z)
    px, py = LAY_ASIDE
    return {"left": [
        Waypoint(2.4, (x, y, TOP + 0.145), open_grip, "reach over the side buttons"),
        Waypoint(1.2, pos, open_grip, "line up one finger with each button"),
        Waypoint(0.95, pos, squeezed, "squeeze both buttons together"),
        Waypoint(0.55, pos, squeezed, "leave room for the underside lever to spring open"),
        Waypoint(0.72, (x - 0.008, y, z + 0.023), squeezed, "raise the opened jaw off the rim"),
        Waypoint(0.89, (x - 0.076, y, z + 0.035), squeezed, "pull the handle clear"),
        Waypoint(1.1, (px, py, TOP + 0.126), squeezed, "move the handle beside the pan"),
        Waypoint(1.1, (px, py, TOP + 0.052), squeezed, "lower it onto the bench"),
        Waypoint(0.60, (px, py, TOP + 0.052), open_grip, "release the handle"),
        Waypoint(1.2, (px, py + 0.047, TOP + 0.143), open_grip, "withdraw"),
        Waypoint(2.2, (px, py + 0.047, TOP + 0.143), open_grip, "leave both pieces at rest"),
    ]}
