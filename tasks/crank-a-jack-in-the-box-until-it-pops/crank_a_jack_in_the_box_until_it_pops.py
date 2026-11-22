"""Turn a spring toy's offset crank until the figure pops up."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import BoxShape, CylShape, Link, Linkage, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_JOINTS, FINGER_OPEN
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
allow_matrix_grips()

ORIGIN = (0.304, 0.046, TOP)
BODY_W = 0.138
H = 0.137
AXLE_Z = 0.090
OFFSET = 0.026
HANDLE_Y = -0.115
RELEASE = 2.85 * math.tau
OVER_TURN = math.pi / 2
RED = (0.73, 0.13, 0.13)
YELLOW = (0.96, 0.70, 0.12)
BLUE = (0.20, 0.39, 0.64)
WHITE = (0.92, 0.87, 0.73)
STEEL = (0.53, 0.55, 0.57)
DARK = (0.06, 0.065, 0.075)


def _world(x, y, z):
    return tuple(np.asarray(ORIGIN) + [x, y, z])


def _knob(angle: float) -> tuple[float, float, float]:
    return (-OFFSET * math.sin(angle), HANDLE_Y, AXLE_Z - OFFSET * math.cos(angle))


case = [
    BoxShape((0, 0, 0.004), (BODY_W, BODY_W, 0.008), DARK),
    BoxShape((-BODY_W / 2 + 0.002, 0, H / 2), (0.004, BODY_W, H), RED),
    BoxShape((BODY_W / 2 - 0.002, 0, H / 2), (0.004, BODY_W, H), RED),
    BoxShape((0, -BODY_W / 2 + 0.002, H / 2), (BODY_W, 0.004, H), BLUE),
    BoxShape((0, BODY_W / 2 - 0.002, H / 2), (BODY_W, 0.004, H), BLUE),
    BoxShape((-BODY_W / 2 - 0.0005, 0, 0.071), (0.001, 0.087, 0.084), YELLOW, collide=False),
]
for angle in (0, math.pi / 3, 2 * math.pi / 3):
    case.append(BoxShape((-BODY_W / 2 - 0.0012, 0, 0.071), (0.001, 0.009, 0.059),
                         RED, collide=False,
                         quat_xyzw=(math.sin(angle / 2), 0, 0, math.cos(angle / 2))))

crank = [
    CylShape((0, -0.081, AXLE_Z), 0.005, 0.029, STEEL, axis="Y"),
    CylShape((0, -0.095, AXLE_Z - OFFSET / 2), 0.003, OFFSET + 0.004, STEEL),
    CylShape((0, -0.101, AXLE_Z - OFFSET), 0.0035, 0.016, STEEL, axis="Y"),
]
spinner = [
    CylShape((0, HANDLE_Y, AXLE_Z - OFFSET), 0.008, 0.024, DARK, axis="Y"),
    CylShape((0, HANDLE_Y - 0.012, AXLE_Z - OFFSET), 0.0085, 0.002, YELLOW, axis="Y", collide=False),
]
lid = [
    BoxShape((0, 0, H + 0.004), (BODY_W + 0.003, BODY_W + 0.003, 0.008), YELLOW),
    BoxShape((0, 0, H + 0.0085), (0.100, 0.100, 0.001), RED, collide=False),
]
figure = [
    CylShape((0, 0, 0.038), 0.021, 0.066, STEEL, collide=False),
    SphereShape((0, 0, 0.079), 0.027, RED),
    SphereShape((-0.004, 0, 0.103), 0.026, WHITE),
    SphereShape((-0.008, -0.016, 0.124), 0.012, BLUE),
    SphereShape((-0.008, 0.016, 0.124), 0.012, BLUE),
]
for y in (-0.010, 0.010):
    figure.append(SphereShape((-0.026, y, 0.108), 0.004, DARK, collide=False))

OBJECTS = [
    Linkage.build("toy", ORIGIN, 0.48, case, [
        Link("crank", tuple(crank), 0.028, _world(0, 0, AXLE_Z),
             damping=0.003),
        Link("knob", tuple(spinner), 0.008, _world(*_knob(0.0)),
             parent="crank", damping=0.0002),
        Link("lid", tuple(lid), 0.034, _world(BODY_W / 2, 0, H),
             limits=(0, 112), stiffness=0.31, spring_damping=0.043),
        Link("figure", tuple(figure), 0.043, _world(0, 0, 0.035), axis="Z", slide=True,
             limits=(0.0, 0.102), stiffness=45.0, spring_damping=2.2),
    ], floating=True, color=RED,
       friction=(0.84, 0.70), contact_offset=0.001, solver_iterations=48),
]

_state = {}


@on_reset
def _reset() -> None:
    _state.clear()
    _state.update(last=0.0, net=0.0, overshoot=0.0, released=False, opened=False, step=0,
                  indices={}, arms={}, clear=False, settled=False)


_reset()


@on_step
def _track(env) -> None:
    import torch

    base = getattr(env, "unwrapped", env)
    toy = base.scene["toy"]
    q = as_numpy(toy.data.joint_pos)[0]
    if not _state["indices"]:
        _state["indices"] = {key: toy.joint_names.index(f"{key}_turn") for key in ("crank", "lid", "figure")}
    idx = _state["indices"]
    crank_pos = float(q[idx["crank"]])
    change = (crank_pos - _state["last"] + math.pi) % math.tau - math.pi
    _state["last"] = crank_pos
    _state["net"] += change
    _state["step"] += 1
    if _state["opened"]:
        _state["overshoot"] += abs(change)
    if not _state["released"] and _state["net"] >= RELEASE:
        # The physical crank turns a cam. At release its catch opens and the loaded springs expand.
        _state["released"] = True
        toy.set_joint_position_target_index(
            target=torch.tensor([[math.radians(109.0), 0.108]], device=toy.device),
            joint_ids=[idx["lid"], idx["figure"]],
        )
        toy.write_data_to_sim()
    if _state["released"] and q[idx["lid"]] >= math.radians(5.0):
        _state["opened"] = True
    pose = as_numpy(toy.data.root_link_pose_w)[0]
    velocity = as_numpy(toy.data.root_link_vel_w)[0]
    _state["settled"] = np.linalg.norm(velocity[:3]) <= 0.015 and np.linalg.norm(velocity[3:]) <= 0.12
    knob = pose[:3] + rotation(pose[3:7]) @ _knob(crank_pos)
    hands = {}
    for side, arm_base in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["arms"]:
            _state["arms"][side] = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        joints = as_numpy(arm.data.joint_pos)[0][_state["arms"][side]]
        hands[side], _ = ik.tcp_pose(joints, arm_base)
    _state["clear"] = all(np.linalg.norm(knob - hand) > 0.065 for hand in hands.values())
    if os.getenv("JACK_DEBUG") and _state["step"] % 20 == 0:
        arm = base.scene["right_robot"]
        finger_ids = arm.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        jaw = as_numpy(arm.data.joint_pos)[0][finger_ids]
        print(f"jack step={_state['step']} crank_turns={_state['net'] / math.tau:.3f} "
              f"past={math.degrees(_state['overshoot']):.1f} lid={math.degrees(q[idx['lid']]):.1f} "
              f"figure={q[idx['figure']]:.3f} clear={_state['clear']} "
              f"tcp={np.round(np.asarray(hands['right']) - [0, 0, TOP], 3)} "
              f"knob={np.round(knob - [0, 0, TOP], 3)} fingers={np.round(jaw, 4)} "
              f"case={np.round(pose[:3] - [0, 0, TOP], 3)}", flush=True)


def _popped(objects) -> bool:
    toy = objects["toy"]
    return _state["released"] and toy.joints["lid"] >= math.radians(70) and toy.joints["figure"] >= 0.06


def score(objects) -> float:
    if _popped(objects):
        return 0.75
    return 0.25 if _state["net"] >= math.tau else 0.0


def success(objects) -> bool:
    toy = objects["toy"]
    return bool(
        _popped(objects)
        and _state["overshoot"] <= OVER_TURN
        and _state["clear"]
        and np.linalg.norm(toy.pos[:2] - ORIGIN[:2]) <= 0.025
        and abs(toy.pos[2] - TOP) <= 0.010
        and _state["settled"]
    )


def demo():
    def grip(dt, theta, height, fingers, label):
        x, y, z = _knob(theta)
        pos = _world(x, y, z + height)
        return Waypoint(dt, pos, fingers, label, math.pi / 2)

    sequence = [
        grip(2.7, 0, 0.048, FINGER_OPEN, "hover above crank knob"),
        grip(1.2, 0, 0.004, FINGER_OPEN, "lower to the crank knob"),
        grip(0.8, 0, 0.004, FINGER_CLOSED, "grasp the knob"),
    ]
    count = 8 if os.getenv("JACK_SHORT") == "1" else 36
    angle = 0.0
    for k in range(1, count + 1):
        angle = math.tau * k / 12
        sequence.append(grip(0.31, angle, 0.004, FINGER_CLOSED, "rotate the crank"))
    sequence += [
        grip(0.70, angle, 0.004, FINGER_CLOSED, "stop the crank in the grip"),
        grip(0.55, angle, 0.004, FINGER_OPEN, "release the crank"),
        grip(1.3, angle, 0.085, FINGER_OPEN, "clear the toy"),
        grip(1.3, angle, 0.085, FINGER_OPEN, "wait clear of the toy"),
    ]
    brace_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) @ grip_rotation()
    approach = (0.171, 0.093, TOP + 0.078)
    press = (0.232, 0.093, TOP + 0.070)
    left = [
        Waypoint(2.6, approach, FINGER_CLOSED, "approach the front face", brace_rotation),
        Waypoint(1.6, press, FINGER_CLOSED, "brace the front face of the tin case", brace_rotation),
        Waypoint(12.4, press, FINGER_CLOSED, "steady it through the crank turns", brace_rotation),
        Waypoint(1.5, approach, FINGER_OPEN, "move clear", brace_rotation),
    ]
    return {"left": left, "right": sequence}
