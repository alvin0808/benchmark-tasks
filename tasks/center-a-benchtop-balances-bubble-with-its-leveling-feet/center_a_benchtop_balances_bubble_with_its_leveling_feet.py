"""Level a benchtop precision balance using both of its threaded front feet."""

from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import BoxShape, CylShape, Link, Linkage, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"

ORIGIN = (0.352, 0.0, TOP + 0.001)
FRONT_X = -0.116
REAR_X = 0.087
FOOT_Y = {"left": 0.118, "right": -0.118}
FOOT_SHORT = {"left": 0.0031, "right": 0.0054}
KNOB_Z = 0.019
PITCH = 0.003
BUBBLE_GAIN = 0.40
LEVEL_TILT = math.radians(0.16)
DARK = (0.105, 0.136, 0.155)
GRAY = (0.45, 0.49, 0.52)
STEEL = (0.72, 0.74, 0.76)
WHITE = (0.90, 0.92, 0.90)
PANEL = (0.18, 0.25, 0.30)
KNURLED = (0.16, 0.17, 0.18)
LIME = (0.61, 0.76, 0.18)
RING = (0.22, 0.32, 0.05)
TIP = (0.97, 0.98, 0.83)


def _world(x, y, z):
    return (ORIGIN[0] + x, ORIGIN[1] + y, ORIGIN[2] + z)


def _dial_ring(x, y, z, radius, color):
    return [
        BoxShape((x + radius * math.cos(a), y + radius * math.sin(a), z),
                 (0.0011, 0.00145, 0.0003), color, yaw=a, collide=False)
        for a in np.linspace(0, math.tau, 32, endpoint=False)
    ]


case = [
    BoxShape((0.016, 0, 0.043), (0.211, 0.173, 0.035), WHITE),
    BoxShape((0.019, 0, 0.031), (0.218, 0.179, 0.012), DARK),
    BoxShape((-0.062, 0, 0.061), (0.053, 0.153, 0.004), PANEL),
    BoxShape((-0.065, 0.031, 0.0634), (0.027, 0.052, 0.001), KNURLED, collide=False),
    BoxShape((-0.065, 0.031, 0.0640), (0.022, 0.046, 0.001), (0.44, 0.53, 0.44), collide=False),
    BoxShape((0.042, 0, 0.065), (0.134, 0.143, 0.013), GRAY),
    BoxShape((0.042, 0, 0.073), (0.144, 0.153, 0.004), STEEL),
    CylShape((REAR_X, 0, 0.0135), 0.014, 0.027, KNURLED),
    CylShape((-0.067, -0.044, 0.064), 0.015, 0.003, DARK),
    CylShape((-0.067, -0.044, 0.0657), 0.0135, 0.0006, LIME, collide=False),
    *_dial_ring(-0.067, -0.044, 0.0662, 0.00435, RING),
    *_dial_ring(-0.067, -0.044, 0.0662, 0.0113, RING),
]

links = []
for side, y in FOOT_Y.items():
    short = FOOT_SHORT[side]
    anchor = _world(FRONT_X, y, 0)
    inside = math.copysign(1.0, y)
    case.append(BoxShape((-0.097, y - inside * 0.020, 0.030),
                         (0.041, 0.010, 0.014), DARK, yaw=inside * -0.71))
    links.append(Link(
        f"{side}_pad",
        (
            CylShape((FRONT_X, y, 0.0035 + short), 0.012, 0.007, KNURLED),
            CylShape((FRONT_X, y, 0.019 + short), 0.006, 0.024, GRAY),
        ),
        0.045, anchor, axis="Z", slide=True, limits=(-0.010, 0.0025),
        stiffness=12000.0, spring_damping=38.0,
    ))
    ribs = []
    for i in range(24):
        a = i * math.tau / 24
        marker = STEEL if i in (0, 8, 16) else GRAY
        ribs.append(BoxShape((FRONT_X + 0.0217 * math.cos(a), y + 0.0217 * math.sin(a), KNOB_Z + short),
                             (0.0017, 0.0018, 0.0105), marker, yaw=a, collide=False))
    links.append(Link(
        f"{side}_wheel",
        (
            CylShape((FRONT_X, y, KNOB_Z + short), 0.0225, 0.0125, KNURLED),
            CylShape((FRONT_X, y, KNOB_Z + short + 0.0067), 0.010, 0.0009, GRAY, collide=False),
            *ribs,
        ),
        0.040, anchor, axis="Z", parent=f"{side}_pad", friction=0.009, damping=0.0008,
    ))

# Two low-mass orthogonal slides move the visible air bubble across the level vial.
# Their targets follow gravity in the vial's local plane, independently of the screw positions.
vial = _world(-0.067, -0.044, 0.0668)
links.extend([
    Link("vial_x", (), 0.006, vial, axis="X", slide=True, limits=(-0.009, 0.009),
         stiffness=1.0, spring_damping=0.15),
    Link("bubble", (CylShape((-0.067, -0.044, 0.0668), 0.0029, 0.0007, TIP, collide=False),),
         0.002, vial, axis="Y", parent="vial_x", slide=True, limits=(-0.009, 0.009),
         stiffness=1.0, spring_damping=0.10),
])
OBJECTS = [
    Linkage.build("balance", ORIGIN, 2.1, case, links, floating=True,
                  color=WHITE, friction=(0.85, 0.70), contact_offset=0.00065, solver_iterations=48),
]

_state = {}


@on_reset
def _reset():
    _state.clear()
    _state.update(indices={}, previous={}, turns=dict.fromkeys(FOOT_Y, 0.0), arms={}, tilt=math.inf,
                  bubble=math.inf, supported=False, clear=False, settled=False, step=0, poses=deque(maxlen=6))


_reset()


@on_step
def _update(env):
    import torch

    base = getattr(env, "unwrapped", env)
    balance = base.scene["balance"]
    if not _state["indices"]:
        _state["indices"] = {name: balance.joint_names.index(f"{name}_turn")
                             for name in ("left_pad", "right_pad", "left_wheel", "right_wheel", "vial_x", "bubble")}
    idx = _state["indices"]
    q = as_numpy(balance.data.joint_pos)[0]
    pose = as_numpy(balance.data.root_link_pose_w)[0]
    rot = rotation(pose[3:7])
    up = rot[2]
    bubble_xy = np.array([q[idx["vial_x"]], q[idx["bubble"]]])
    targets = []
    target_ids = []
    for side in FOOT_Y:
        wheel = float(q[idx[f"{side}_wheel"]])
        prev = _state["previous"].get(side, 0.0)
        delta = (wheel - prev + math.pi) % math.tau - math.pi
        _state["previous"][side] = wheel
        # An anticlockwise turn of the left-hand thread extends the foot.
        # A high-stiffness prismatic drive carries the balance on the physical pad.
        travel = np.clip(_state["turns"][side] + PITCH * delta / math.tau, -0.0024, 0.0098)
        _state["turns"][side] = float(travel)
        targets.append(-travel)
        target_ids.append(idx[f"{side}_pad"])
    targets.extend(np.clip(BUBBLE_GAIN * up[:2], -0.008, 0.008))
    target_ids.extend([idx["vial_x"], idx["bubble"]])
    balance.set_joint_position_target_index(
        target=torch.tensor([targets], dtype=torch.float32, device=balance.device), joint_ids=target_ids,
    )
    balance.write_data_to_sim()

    feet = [[REAR_X, 0.0, 0.0]]
    for side, y in FOOT_Y.items():
        feet.append([FRONT_X, y, FOOT_SHORT[side] + q[idx[f"{side}_pad"]]])
    foot_z = pose[2] + np.asarray(feet) @ up
    _state["tilt"] = math.acos(float(np.clip(up[2], -1.0, 1.0)))
    _state["bubble"] = float(np.linalg.norm(bubble_xy))
    _state["supported"] = bool(np.max(np.abs(foot_z - TOP)) <= 0.0016)
    poses = _state["poses"]
    poses.append((pose[:3].copy(), rot.copy()))
    _state["settled"] = len(poses) == poses.maxlen and all(
        np.linalg.norm(p - pose[:3]) <= 0.0006
        and np.trace(r.T @ rot) >= 1.0 + 2.0 * math.cos(math.radians(0.13))
        for p, r in poses
    )

    nearest = math.inf
    diagnostics = {}
    lo = np.array([-0.143, -0.143, 0.0])
    hi = np.array([0.126, 0.143, 0.075])
    for side, anchor in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["arms"]:
            _state["arms"][side] = (arm.find_joints(ARM_JOINTS, preserve_order=True)[0],
                                    arm.find_joints(FINGER_JOINTS, preserve_order=True)[0])
        arm_ids, finger_ids = _state["arms"][side]
        joints = as_numpy(arm.data.joint_pos)[0]
        hand, hand_rot = ik.tcp_pose(joints[arm_ids], anchor)
        pads = [
            hand + hand_rot @ [0.0, sign * (abs(finger) + 0.0015), z]
            for sign, finger in zip((-1, 1), joints[finger_ids], strict=True)
            for z in (-0.026, 0.013)
        ]
        for point in [hand, *pads]:
            local = rot.T @ (point - pose[:3])
            nearest = min(nearest, float(np.linalg.norm(local - np.clip(local, lo, hi))))
        diagnostics[side] = (np.round(hand - [0, 0, TOP], 4), np.round(joints[finger_ids], 4))
    _state["clear"] = nearest >= 0.038
    _state["step"] += 1
    if os.getenv("BALANCE_DEBUG") == "1" and _state["step"] % 10 == 0:
        print(f"LEVEL {_state['step']:03} tilt={math.degrees(_state['tilt']):.3f} "
              f"feet={np.round(1e3 * (foot_z - TOP), 2)} "
              f"delta={np.round(1e3 * np.array(list(_state['turns'].values())), 2)} "
              f"bubble={np.round(1e3 * bubble_xy, 2)} ok={_state['supported']}/{_state['clear']}/{_state['settled']} near={nearest:.3f} l={diagnostics['left']} r={diagnostics['right']}", flush=True)


def score(objects):
    if not (_state["supported"] and _state["clear"] and _state["settled"]):
        return 0.0
    if _state["tilt"] <= LEVEL_TILT and _state["bubble"] <= 0.00145:
        return 1.0
    if _state["tilt"] <= math.radians(0.40) and _state["bubble"] <= 0.0030:
        return 0.5
    return 0.0


def success(objects):
    return score(objects) >= 1.0


def demo():
    def adjustments(side):
        x, y, _ = _world(FRONT_X, FOOT_Y[side], 0)
        z = TOP + KNOB_Z + 0.004
        contact = (x, y, z)
        hover = (x, y, TOP + 0.096)
        before, after = (-1.35, 0.09) if side == "left" else (-0.09, 1.43)
        open_grip, clamp = -0.030, -0.018
        count = int(os.getenv(f"BALANCE_{side.upper()}", "5" if side == "left" else "14"))
        path = [Waypoint(2.3, hover, open_grip, "approach the leveling wheel", before)]
        for i in range(count):
            yaw = after if i < count - 1 else float(os.getenv(
                f"BALANCE_{side.upper()}_FINAL", str(after if side == "left" else 0.84)
            ))
            path += [
                Waypoint(0.60, contact, open_grip, "lower the open gripper", before),
                Waypoint(0.52, contact, clamp, "pinch the outer edge of the wheel", before),
                Waypoint(0.62, contact, clamp, "turn the leveling wheel anticlockwise", yaw),
                Waypoint(0.35, contact, open_grip, "release the wheel", yaw),
                Waypoint(0.45, hover, open_grip, "move above the wheel", yaw),
                Waypoint(0.46, hover, open_grip, "orient for the next grip", before),
            ]
        withdrawn = (x - 0.081, y, TOP + 0.155)
        path.extend((
            Waypoint(1.4, withdrawn, open_grip, "move clear of the balance", before),
            Waypoint(2.5, withdrawn, open_grip, "leave it resting on its three feet", before),
        ))
        return path

    return {side: adjustments(side) for side in FOOT_Y}
