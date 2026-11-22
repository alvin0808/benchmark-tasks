"""Adjust a manual tonearm until its stylus reads two grams on a tracking force gauge."""

from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, MeshShape, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step, segment
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.5
allow_matrix_grips()

PIVOT = np.array([0.372, -0.086, TOP + 0.110])
STYLUS = np.array([-0.2245, 0.052, -0.022])
GAUGE_TOP = STYLUS[2] - 0.0029
WEIGHT_START = 0.0545
WEIGHT_LENGTH = 0.031
PITCH = 0.010
G = 9.81
COAL = (0.08, 0.095, 0.107)
BLACK = (0.045, 0.049, 0.054)
RUBBER = (0.095, 0.105, 0.114)
SILVER = (0.66, 0.69, 0.72)
WHITE = (0.92, 0.90, 0.85)
BLUE = (0.18, 0.34, 0.56)
LIME = (0.58, 0.81, 0.32)
DISPLAY = (0.064, 0.135, 0.086)
TIP_RADIUS = 0.00042

def _world(p):
    return tuple(PIVOT + np.asarray(p))


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _disc(x, y, z, radius, depth, color):
    count = 72
    angles = np.linspace(0.0, math.tau, count, endpoint=False)
    vertices = np.asarray([(x + radius * math.cos(a), y + radius * math.sin(a), height)
                           for height in (z - depth / 2, z + depth / 2) for a in angles])
    faces = [(0, i + 1, i) for i in range(1, count - 1)]
    faces.extend((count, count + i, count + i + 1) for i in range(1, count - 1))
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, count + j), (i, count + j, count + i)))
    return MeshShape(vertices, np.asarray(faces), color=color, approximation="convexHull")


deck = [
    BoxShape((-0.081, 0.108, -0.081), (0.352, 0.452, 0.045), COAL),
    BoxShape((-0.081, 0.108, -0.057), (0.352, 0.452, 0.004), SILVER),
    _disc(-0.108, 0.145, -0.048, 0.150, 0.014, SILVER),
    _disc(-0.108, 0.145, -0.0398, 0.1475, 0.0026, RUBBER),
    CylShape((-0.108, 0.145, -0.036), 0.0036, 0.006, SILVER),
    CylShape((0, 0, -0.032), 0.019, 0.052, COAL),
    CylShape((0, 0, -0.008), 0.014, 0.016, SILVER),
    BoxShape((-0.186, -0.046, -0.046), (0.010, 0.019, 0.020), BLACK),
    BoxShape((-0.186, -0.046, -0.034), (0.013, 0.022, 0.007), BLACK),
    CylShape((-0.208, 0.294, -0.055), 0.013, 0.006, COAL),
    BoxShape((-0.216, 0.268, -0.053), (0.028, 0.018, 0.004), COAL),
]
for x in (-0.222, 0.063):
    for y in (-0.085, 0.300):
        deck.append(CylShape((x, y, -0.105), 0.020, 0.010, BLACK))

tube = (
    CylShape((0.030, 0, 0), 0.0035, 0.099, SILVER, axis="X"),
    CylShape((-0.065, 0, -0.001), 0.0039, 0.120, SILVER, axis="X"),
    segment((-0.125, 0.0, -0.001), (-0.181, 0.038, -0.001), 0.007, SILVER),
    segment((-0.181, 0.038, -0.001), (-0.209, 0.049, -0.001), 0.007, SILVER),
    BoxShape((-0.214, 0.052, -0.0035), (0.040, 0.016, 0.005), BLACK),
    BoxShape((-0.218, 0.052, -0.0118), (0.026, 0.015, 0.011), BLUE),
    BoxShape((-0.209, 0.065, -0.001), (0.009, 0.015, 0.0045), SILVER),
    SphereShape(tuple(STYLUS + [0, 0, TIP_RADIUS]), TIP_RADIUS, SILVER),
)
weight = [
    CylShape((WEIGHT_START, 0, 0), 0.0185, WEIGHT_LENGTH, COAL, axis="X"),
    CylShape((WEIGHT_START - 0.016, 0, 0), 0.0165, 0.002, SILVER, axis="X", collide=False),
]
for i in range(24):
    angle = math.tau * i / 24
    q = (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))
    weight.append(BoxShape((WEIGHT_START, -0.0185 * math.sin(angle), 0.0185 * math.cos(angle)),
                           (0.026, 0.0011, 0.0010), WHITE if i in (0, 8, 16) else RUBBER,
                           collide=False, quat_xyzw=q))

# A low-profile gauge rests on the platter with a replaceable soft pad below the tip.
gauge_shapes = [
    BoxShape((-0.221, 0.083, -0.0325), (0.026, 0.083, 0.011), COAL),
    BoxShape((STYLUS[0], STYLUS[1], GAUGE_TOP - 0.0017), (0.016, 0.016, 0.0034), SILVER),
    BoxShape((-0.222, 0.102, -0.0268), (0.021, 0.030, 0.0012), DISPLAY),
    BoxShape((-0.215, 0.126, -0.0265), (0.008, 0.006, 0.0016), BLUE),
]
# The gauge digits are visual pieces. They are moved by the measured contact force only.
SEGMENTS = {
    "a": (0.0, 1.0, True), "b": (1.0, 0.5, False), "c": (1.0, -0.5, False),
    "d": (0.0, -1.0, True), "e": (-1.0, -0.5, False), "f": (-1.0, 0.5, False), "g": (0.0, 0.0, True),
}
NUMBERS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
DISPLAY_PARTS = []
for column, offset in enumerate((0.006, -0.004)):
    for name, (y, x, flat) in SEGMENTS.items():
        point = np.asarray(_world((-0.221 + 0.0055 * x, 0.102 + offset - 0.0030 * y, -0.02605)))
        size = (0.0007, 0.0055, 0.0002) if flat else (0.0052, 0.0007, 0.0002)
        display_name = f"digit_{column}_{name}"
        shown = name in NUMBERS[(2, 9)[column]]
        parked = point if shown else point - [0.0, 0.0, 0.11]
        body = Body.build(display_name, parked, 0.001, [BoxShape((0, 0, 0), size, LIME, collide=False)],
                          fixed=True, color=LIME)
        DISPLAY_PARTS.append((column, name, point, body))
gauge_shapes.append(BoxShape((-0.2265, 0.103, -0.02605), (0.0010, 0.0010, 0.0002), LIME, collide=False))

OBJECTS = [
    Linkage.build("turntable", tuple(PIVOT), 4.1, deck, [
        Link("gimbal", (), 0.019, tuple(PIVOT), axis="Z", limits=(-23.0, 32.0),
             friction=0.001, damping=0.009),
        Link("tube", tube[:-1], 0.045, tuple(PIVOT), axis="Y", parent="gimbal", limits=(-7.0, 8.0),
             damping=0.008),
        Link("stylus", (tube[-1],), 0.00032, _world(STYLUS), axis="Y", parent="tube",
             limits=(-0.03, 0.03), stiffness=7.0, spring_damping=0.01, report_contacts=True),
        Link("travel", (), 0.001, tuple(PIVOT), axis="X", parent="tube", slide=True, limits=(-0.018, 0.027),
             stiffness=2500.0, spring_damping=28.0),
        Link("weight", tuple(weight), 0.092, _world((WEIGHT_START, 0, 0)), axis="X", parent="travel",
             friction=0.0025, damping=0.0008),
    ], color=COAL, friction=(0.75, 0.60), contact_offset=0.00023, solver_iterations=48),
    Body.build("gauge", tuple(PIVOT), 0.060, gauge_shapes, fixed=True, color=COAL,
               friction=(0.40, 0.31), contact_offset=0.00022),
    *(part for _, _, _, part in DISPLAY_PARTS),
]

_state = {}

@on_reset
def _reset():
    _state.clear()
    _state.update(joints={}, hands={}, view=None, previous=0.0, travel=0.0, filtered=deque(maxlen=10),
                  display=29, score=0.0, step=0, damaged=False, heavy_frames=0)

_reset()


def _force(base):
    from isaaclab_physx.physics import PhysxManager

    if _state["view"] is None:
        sim = PhysxManager.get_physics_sim_view()
        env = base.scene.env_prim_paths[0]
        _state["view"] = sim.create_rigid_contact_view([f"{env}/turntable/stylus"],
            filter_patterns=[[f"{env}/gauge/body"]], max_contact_data_count=16)
    values = _state["view"].get_contact_force_matrix(dt=float(base.physics_dt)).numpy()
    return max(0.0, float(values.reshape(-1, 3)[:, 2].sum()))


def _display(base, load_grams):
    import torch

    value = max(0, min(99, round(load_grams * 10)))
    old = _state["display"]
    if old == value:
        return
    _state["display"] = value
    previous, current = divmod(old, 10), divmod(value, 10)
    for col, name, point, part in DISPLAY_PARTS:
        was_shown = name in NUMBERS[previous[col]]
        show = name in NUMBERS[current[col]]
        if show == was_shown:
            continue
        xyz = point if show else point - [0.0, 0.0, 0.11]
        pose = torch.tensor([[*xyz, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=base.device)
        base.scene[part.name].write_root_link_pose_to_sim_index(root_pose=pose)


@on_step
def _update(env):
    import torch

    base = getattr(env, "unwrapped", env)
    turntable = base.scene["turntable"]
    if not _state["joints"]:
        _state["joints"] = {key: turntable.joint_names.index(f"{key}_turn")
                            for key in ("gimbal", "tube", "weight", "travel")}
    idx = _state["joints"]
    q = as_numpy(turntable.data.joint_pos)[0]
    qd = as_numpy(turntable.data.joint_vel)[0]
    weight_angle = float(q[idx["weight"]])
    delta = (weight_angle - _state["previous"] + math.pi) % math.tau - math.pi
    _state["previous"] = weight_angle
    _state["travel"] = float(np.clip(_state["travel"] + PITCH * delta / math.tau, -0.016, 0.025))
    turntable.set_joint_position_target_index(
        target=torch.tensor([[_state["travel"]]], dtype=torch.float32, device=turntable.device),
        joint_ids=[idx["travel"]],
    )
    turntable.write_data_to_sim()

    pitch, yaw = float(q[idx["tube"]]), float(q[idx["gimbal"]])
    frame = _rz(yaw) @ _ry(pitch)
    tip = PIVOT + frame @ STYLUS
    counterweight = PIVOT + frame @ [WEIGHT_START + q[idx["travel"]], 0.0, 0.0]
    raw = _force(base) / G * 1000.0
    _state["filtered"].append(raw)
    measured = float(np.mean(_state["filtered"]))
    if len(_state["filtered"]) == _state["filtered"].maxlen:
        _display(base, measured)
    pred = PIVOT + STYLUS
    on_pad = abs(tip[0] - pred[0]) <= 0.0065 and abs(tip[1] - pred[1]) <= 0.0065
    off_mat = tip[2] >= PIVOT[2] - 0.0375
    heavy = raw > 15.0 and on_pad
    _state["heavy_frames"] = _state["heavy_frames"] + 1 if heavy else 0
    _state["damaged"] |= not off_mat or _state["heavy_frames"] >= 2
    clear = True
    hand_pos = {}
    for side, anchor in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["hands"]:
            _state["hands"][side] = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        armq = as_numpy(arm.data.joint_pos)[0][_state["hands"][side]]
        hand, _ = ik.tcp_pose(armq, anchor)
        hand_pos[side] = hand
        clear &= min(np.linalg.norm(hand - counterweight), np.linalg.norm(hand - tip)) >= 0.063
    _state["score"] = 0.0
    if not _state["damaged"] and clear and on_pad and abs(qd[idx["tube"]]) < 0.014:
        if 1.8 <= measured <= 2.2:
            _state["score"] = 1.0
        elif 1.6 <= measured <= 2.4:
            _state["score"] = 0.5
    _state["step"] += 1
    if os.getenv("TONE_DEBUG") == "1" and _state["step"] % 10 == 0:
        print(f"TONE {_state['step']:03} pitch={math.degrees(pitch):6.2f} yaw={math.degrees(yaw):5.2f} "
              f"twist={math.degrees(weight_angle):6.2f} x={q[idx['travel']]*1000:6.2f} "
              f"force={measured:4.2f}/{raw:4.2f} off={(tip[2]-PIVOT[2]-GAUGE_TOP)*1000:.2f} "
              f"clear={clear} damage={_state['damaged']} "
              f"r={np.round(hand_pos['right'] - [0,0,TOP], 4)}", flush=True)


def score(objects):
    return min(_state["score"], 0.5)


def success(objects):
    return _state["score"] == 1.0


def demo():
    yaw = math.pi / 2
    grab_height = 0.019
    open_grip, clamp = -0.026, -0.013
    start, finish = 0.28, 1.10
    outward = 1 if os.getenv("TONE_SHORT") == "1" else 2
    path = []
    travel = 0.0

    def on_collar(dt, theta, z, fingers, label):
        rim = _rx(theta)
        point = np.array([WEIGHT_START + travel, 0.0, 0.0]) + rim @ [0, 0, z]
        return Waypoint(dt, _world(point), fingers, label, rim @ grip_rotation(yaw))

    def turn(before, after, label):
        nonlocal travel
        if not path:
            path.append(on_collar(2.5, before, grab_height + 0.057, open_grip, "approach the counterweight"))
        path.extend((
            on_collar(0.52, before, grab_height + 0.045, open_grip, "orient over the collar"),
            on_collar(0.63, before, grab_height, open_grip, "lower the open gripper"),
            on_collar(0.57, before, grab_height, clamp, "grip the collar across its faces"),
        ))
        travel += (after - before) * PITCH / math.tau
        path.extend((
            on_collar(0.62, after, grab_height, clamp, label),
            on_collar(0.35, after, grab_height, open_grip, "release the collar"),
            on_collar(0.54, after, grab_height + 0.055, open_grip, "clear the counterweight"),
        ))

    end = finish
    if os.getenv("TONE_NO_ADJUST") == "1":
        path.append(on_collar(5.0, finish, grab_height + 0.072, open_grip, "keep clear of the counterweight"))
    else:
        for i in range(outward):
            end = finish if i == 0 else float(os.getenv("TONE_LAST", "0.77"))
            turn(start, end, "turn the counterweight a little away from the bearing")
    adjustment_end = sum(w.duration for w in path)
    path.append(on_collar(2.8, end, grab_height + 0.072, open_grip, "take the gauge reading hands clear"))

    tab_xy = (-0.209, 0.066)
    tab = _world((*tab_xy, 0.007))
    up = _world((*tab_xy, 0.011))
    landing = _world((*tab_xy, 0.0080))
    above = _world((*tab_xy, 0.079))
    withdrawn = _world((-0.185, 0.160, 0.082))
    left_grip = -math.pi / 2
    left_open, left_clamp = -0.015, -0.0025
    left = [
        Waypoint(1.0, above, left_open, "reach above the finger lift", left_grip),
        Waypoint(0.7, tab, left_open, "approach the finger lift", left_grip),
        Waypoint(0.55, tab, left_clamp, "steady the tonearm by its finger lift", left_grip),
        Waypoint(0.65, up, left_clamp, "lift the stylus off the gauge", left_grip),
        Waypoint(adjustment_end - 2.9, up, left_clamp, "support the stylus during adjustment", left_grip),
        Waypoint(1.0, landing, left_clamp, "lower the stylus just above the gauge", left_grip),
        Waypoint(0.62, landing, left_open, "let the stylus settle under its own weight", left_grip),
        Waypoint(0.85, withdrawn, left_open, "clear the gauge and tonearm", left_grip),
        Waypoint(2.0, withdrawn, left_open, "leave the gauge measuring the stylus", left_grip),
    ]
    return {"left": left, "right": path}
