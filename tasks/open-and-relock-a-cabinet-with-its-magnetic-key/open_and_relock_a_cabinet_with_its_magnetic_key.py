"""Operate and verify a concealed spring latch using its external magnetic key."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, MeshShape, allow_matrix_grips, as_numpy, on_reset, on_step, plate, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
allow_matrix_grips()

HINGE = np.array([0.406, -0.183, TOP + 0.003])
DOOR_W = 0.354
HANDLE = np.array([-0.035, 0.320, 0.086])
LATCH = np.array([0.013, 0.188, 0.218])
ARMATURE = np.array([0.039, 0.188, 0.257])
AIM = np.array([-0.011, 0.188, 0.257])
KEY_REST = (0.238, -0.151, TOP + 0.0004)
KEY_STEM = 0.0225
OAK = (0.74, 0.55, 0.34)
EDGE = (0.57, 0.36, 0.19)
WHITE = (0.85, 0.87, 0.86)
BACK = (0.71, 0.76, 0.75)
METAL = (0.46, 0.49, 0.52)
BLACK = (0.11, 0.13, 0.15)
BLUE = (0.18, 0.42, 0.63)


def _world(p):
    return tuple(HINGE + np.asarray(p))


def _z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _hook_tooth():
    yz = LATCH[1]
    # Closing faces the long bevel. Pulling faces the shorter undercut,
    # whose reaction drives the hook towards its engaged stop.
    surface = [(0.053, 0.222), (0.057, 0.222), (0.069, 0.239), (0.057, 0.239)]
    vertices = np.array([(x, yz + d, z) for d in (-0.009, 0.009) for x, z in surface])
    faces = np.array([(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                      (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                      (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)])
    return MeshShape(vertices, faces, color=WHITE, approximation="convexHull")


case = [
    BoxShape((0.124, 0.176, 0.0055), (0.247, 0.380, 0.016), BACK),
    BoxShape((0.128, -0.008, 0.151), (0.242, 0.016, 0.285), WHITE),
    BoxShape((0.128, 0.361, 0.151), (0.242, 0.016, 0.285), WHITE),
    BoxShape((0.245, 0.176, 0.147), (0.012, 0.368, 0.284), BACK),
    BoxShape((0.125, 0.176, 0.284), (0.246, 0.380, 0.016), WHITE),
    BoxShape((0.027, 0.176, 0.265), (0.021, 0.348, 0.025), WHITE),
    BoxShape((0.003, 0.325, 0.055), (0.011, 0.025, 0.048), BLACK, collide=False),
]
door = [
    BoxShape((-0.001, DOOR_W / 2, 0.1475), (0.018, DOOR_W - 0.003, 0.268), OAK),
    BoxShape((-0.0102, DOOR_W / 2, 0.1475), (0.0006, DOOR_W - 0.004, 0.267), OAK, collide=False),
    BoxShape((-0.0108, DOOR_W / 2, 0.265), (0.001, DOOR_W - 0.028, 0.0025), EDGE, collide=False),
    BoxShape((-0.0108, DOOR_W / 2, 0.030), (0.001, DOOR_W - 0.028, 0.0025), EDGE, collide=False),
    CylShape((-0.020, HANDLE[1], HANDLE[2]), 0.0073, 0.024, METAL, axis="X"),
    CylShape(tuple(HANDLE), 0.0136, 0.0175, BLACK, axis="X"),
    BoxShape((0.010, LATCH[1], 0.234), (0.004, 0.046, 0.046), WHITE),
]

latch = [
    BoxShape((0.036, LATCH[1], 0.239), (0.038, 0.018, 0.005), WHITE),
    BoxShape((0.023, LATCH[1], 0.219), (0.021, 0.014, 0.005), WHITE),
    plate(LATCH, ARMATURE, 0.014, 0.004, (1.0, 0.0, 0.0), WHITE),
    BoxShape(tuple(ARMATURE), (0.005, 0.013, 0.006), METAL),
    _hook_tooth(),
]

OBJECTS = [
    Linkage.build("cabinet", tuple(HINGE), 4.0, case, [
        Link("door", tuple(door), 1.08, tuple(HINGE), axis="Z", limits=(0.0, 92.0), damping=0.024),
        Link("hook", tuple(latch), 0.007, _world(LATCH), axis="Y", parent="door",
             limits=(-30.0, 1.0), stiffness=0.039, spring_damping=0.0013, report_contacts=True),
    ], color=WHITE, friction=(0.70, 0.56), contact_offset=0.0005, solver_iterations=48),
    Body.build("keeper", tuple(HINGE), 0.016, [
        BoxShape((0.0450, LATCH[1], 0.2205), (0.008, 0.036, 0.026), WHITE),
        BoxShape((0.047, LATCH[1] - 0.021, 0.2493), (0.009, 0.007, 0.0555), WHITE),
        BoxShape((0.047, LATCH[1] + 0.021, 0.2493), (0.009, 0.007, 0.0555), WHITE),
    ], fixed=True, friction=(0.35, 0.24), contact_offset=0.0005, color=WHITE),
    Body.build("key", KEY_REST, 0.054, [
        CylShape((0, 0, 0.005), 0.021, 0.010, WHITE),
        CylShape((0, 0, 0.023), 0.0116, 0.026, BLUE),
        CylShape((0, 0, 0.037), 0.0091, 0.002, METAL, collide=False),
        CylShape((0, 0, 0.0005), 0.0125, 0.001, BLACK, collide=False),
    ], friction=(0.67, 0.51), color=WHITE, contact_offset=0.0005, solver_iterations=32),
]

_state = {}


@on_reset
def _reset():
    _state.clear()
    _state.update(ids={}, arms={}, view=None, armed=False, opened=False, shut_frames=0,
                  tug_frames=0, verified=False, damaged=False, clear=False, score=0.0, step=0)


_reset()


def _keeper_load(base):
    from isaaclab_physx.physics import PhysxManager

    if _state["view"] is None:
        sim = PhysxManager.get_physics_sim_view()
        env = base.scene.env_prim_paths[0]
        _state["view"] = sim.create_rigid_contact_view(
            [f"{env}/cabinet/hook"], filter_patterns=[[f"{env}/keeper/body"]], max_contact_data_count=64,
        )
    forces = _state["view"].get_contact_force_matrix(dt=float(base.physics_dt))
    return float(np.linalg.norm(forces.numpy().reshape(-1, 3), axis=1).max())


@on_step
def _update(env):
    import torch

    base = getattr(env, "unwrapped", env)
    cabinet = base.scene["cabinet"]
    if not _state["ids"]:
        _state["ids"] = {name: cabinet.joint_names.index(f"{name}_turn") for name in ("door", "hook")}
    ids = _state["ids"]
    q = as_numpy(cabinet.data.joint_pos)[0]
    speed = float(as_numpy(cabinet.data.joint_vel)[0][ids["door"]])
    angle, hook = float(q[ids["door"]]), float(q[ids["hook"]])
    door_rot, hook_rot = _z(angle), _y(hook)
    latch_xyz = HINGE + door_rot @ LATCH
    armature = latch_xyz + door_rot @ hook_rot @ (ARMATURE - LATCH)

    magnet = base.scene["key"]
    key_pose = as_numpy(magnet.data.root_link_pose_w)[0]
    key_rot = rotation(key_pose[3:7])
    face = key_pose[:3] + 0.001 * key_rot[:, 2]
    out = door_rot[:, 0]
    alignment = max(0.0, float(-key_rot[:, 2] @ out)) ** 2
    r = face - armature
    distance = float(np.linalg.norm(r))
    field = min(0.75, 0.725 * (0.041 / max(distance, 0.018)) ** 4) * alignment
    if float(r @ out) > -0.012:
        field = 0.0
    torque = float(np.cross(armature - latch_xyz, field * r / max(distance, 1e-5)) @ door_rot[:, 1])
    cabinet.set_joint_effort_target_index(
        target=torch.tensor([[torque]], dtype=torch.float32, device=cabinet.device),
        joint_ids=[ids["hook"]],
    )
    cabinet.write_data_to_sim()

    tooth = latch_xyz + door_rot @ hook_rot @ (np.array([0.055, LATCH[1], 0.222]) - LATCH)
    near_keeper = abs(tooth[1] - (HINGE[1] + LATCH[1])) <= 0.024
    load = _keeper_load(base)
    if angle <= 0.16 and hook < -0.25 and field > 0.21:
        _state["armed"] = True
    if not _state["opened"] and angle > 0.22 and not _state["armed"]:
        _state["damaged"] = True
    if _state["armed"] and angle >= math.radians(28):
        _state["opened"] = True
    key_parked = np.linalg.norm(key_pose[:2] - KEY_REST[:2]) <= 0.048 and abs(key_pose[2] - TOP) < 0.012
    latched = (
        angle <= math.radians(2.3) and hook >= -0.13 and near_keeper
        and tooth[0] >= HINGE[0] + 0.0503 and tooth[2] <= HINGE[2] + 0.229
    )
    _state["shut_frames"] = _state["shut_frames"] + 1 if latched and abs(speed) < 0.15 else 0
    tug = latched and distance >= 0.08 and _state["opened"] and _state["shut_frames"] >= 7 and 1.6 <= load <= 35.0
    _state["tug_frames"] = _state["tug_frames"] + 1 if tug else 0
    _state["verified"] |= _state["tug_frames"] >= 4
    _state["step"] += 1

    knob = HINGE + door_rot @ HANDLE
    hands = {}
    for side, anchor in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["arms"]:
            _state["arms"][side] = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q_arm = as_numpy(arm.data.joint_pos)[0][_state["arms"][side]]
        hands[side], _ = ik.tcp_pose(q_arm, anchor)
    _state["clear"] = all(
        np.linalg.norm(hand - knob) >= 0.075 and np.linalg.norm(hand - face) >= 0.065
        for hand in hands.values()
    )

    _state["score"] = 0.0
    if not _state["damaged"] and _state["opened"]:
        _state["score"] = 0.5
        if latched and key_parked and _state["clear"] and abs(speed) < 0.05:
            _state["score"] = 1.0 if _state["verified"] else 0.75
    if os.getenv("MAGKEY_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"KEY {_state['step']:03} door={math.degrees(angle):6.2f} hook={math.degrees(hook):6.2f} "
              f"load={load:5.2f} field={field:4.2f} opened={_state['opened']} locked={latched} "
              f"verified={_state['verified']} parked={key_parked} kp={np.round(key_pose[:3] - [0, 0, TOP], 3)} "
              f"L={np.round(hands['left'] - [0, 0, TOP], 3)} R={np.round(hands['right'] - [0, 0, TOP], 3)}", flush=True)


def score(objects):
    return _state["score"]


def success(objects):
    return _state["score"] == 1.0


def demo():
    forward = _y(-math.pi / 2) @ grip_rotation()

    def right(dt, p, fingers, label, grip=forward):
        return Waypoint(dt, tuple(p), fingers, label, grip)

    key_start = np.array(KEY_REST) + [0, 0, KEY_STEM + 0.002]
    lift = key_start + [0, 0, 0.085]
    aim = HINGE + AIM - [KEY_STEM, 0, 0]
    if os.getenv("MAGKEY_MISS") == "1":
        aim[1] -= 0.063
    approach = aim - [0.069, 0, 0]
    if os.getenv("MAGKEY_Y"):
        aim[1] += float(os.environ["MAGKEY_Y"])
        approach[1] = aim[1]
    def on_door(angle, retreat=0.0):
        target = aim - HINGE - [retreat, 0, 0]
        return HINGE + _z(angle) @ target

    a7 = math.radians(7)
    a10 = math.radians(10)
    right_arm = [
        right(2.3, lift, -0.029, "reach over the magnetic key", grip_rotation()),
        right(1.3, key_start, -0.029, "reach the key's stem", grip_rotation()),
        right(0.65, key_start, -0.008, "grasp the key", grip_rotation()),
        right(0.85, lift, -0.008, "lift the key", grip_rotation()),
        right(1.65, approach, -0.008, "face the key toward the door"),
        right(1.1, aim, -0.008, "bring the key up to the door"),
        right(0.70, aim, -0.008, "wait for the other gripper to take the knob"),
        right(1.0, on_door(a7), -0.008, "follow the door while the hook clears", _z(a7) @ forward),
        right(0.65, on_door(a10, 0.031), -0.008, "withdraw the key after the hook clears", _z(a10) @ forward),
        right(0.64, on_door(a10, 0.085), -0.008, "move the key away from the door", _z(a10) @ forward),
        right(1.45, lift, -0.008, "bring the key over its resting spot", grip_rotation()),
        right(1.15, key_start, -0.008, "lower the key", grip_rotation()),
        right(0.55, key_start, -0.029, "release the key", grip_rotation()),
        right(1.3, lift, -0.029, "clear the key", grip_rotation()),
    ]

    def hold(dt, angle, fingers, label, distance=0.0):
        d = np.asarray(HANDLE) + [-0.006 - distance, 0, 0.001]
        approach_angle = min(angle, math.radians(14.0))
        grip = _z(approach_angle) @ _y(math.radians(16.0)) @ forward
        return Waypoint(dt, _world(_z(angle) @ d), fingers, label, grip)

    def press(dt, x, label):
        grip = _y(math.radians(16.0)) @ forward
        pos = _world((x, HANDLE[1] - 0.013, HANDLE[2] + 0.067))
        return Waypoint(dt, pos, -0.002, label, grip)

    opening = math.radians(32.0)
    left_arm = [
        hold(3.0, 0.0, -0.027, "approach the knob", 0.066),
        hold(2.7, 0.0, -0.027, "wait for the key to be brought up", 0.066),
        hold(1.7, 0.0, -0.027, "reach the knob"),
        hold(1.1, 0.0, -0.010, "grasp the knob as the latch releases"),
        hold(1.0, a7, -0.010, "open until the hook clears the catch"),
        hold(2.2, opening, -0.010, "swing the cabinet door open"),
        hold(3.1, opening, -0.010, "wait until the key is away"),
        hold(2.6, 0.0, -0.010, "swing the door back to its frame"),
        hold(0.45, 0.0, -0.027, "release the knob"),
        hold(0.80, 0.0, -0.027, "move in front of the face", 0.070),
        press(0.90, -0.097, "approach the door face"),
        press(1.20, 0.003, "push the face until the latch clicks"),
        press(0.60, 0.003, "seat the latch on the keeper"),
        press(0.95, -0.094, "withdraw after the click"),
        hold(0.80, 0.0, -0.027, "line up for a final pull", 0.070),
        hold(1.10, 0.0, -0.027, "reach the closed door's knob"),
        hold(0.65, 0.0, -0.010, "grasp the knob for the final pull"),
        hold(0.55, 0.0, -0.010, "take up the play at the locked catch", 0.012),
        hold(0.70, 0.0, -0.010, "test the relocked handle with one short tug", 0.012),
        hold(0.45, 0.0, -0.027, "open fingers after testing the latch", 0.005),
        hold(1.40, 0.0, -0.027, "move away from the latched cabinet", 0.097),
    ]
    return {"left": left_arm, "right": right_arm}
