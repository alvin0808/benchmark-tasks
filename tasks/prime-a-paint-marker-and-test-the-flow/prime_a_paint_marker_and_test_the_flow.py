"""Prime a pre-shaken paint marker with its spring nib, then test its flow."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, allow_matrix_grips, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.1
allow_matrix_grips()

START = np.array([0.320, -0.190, TOP + 0.0254])
AXIAL_END = 0.074
PAPER = np.array([0.277, -0.035, TOP + 0.0019])
PAPER_SIZE = np.array([0.155, 0.199])
TEST_X = 0.278
TEST_LOW = -0.071
TEST_HIGH = 0.004
BLUE = (0.10, 0.30, 0.70)
DARK = (0.082, 0.087, 0.094)
WHITE = (0.91, 0.89, 0.81)
INK = (0.04, 0.18, 0.63)
NIB_TRAVEL = 0.0035
HOLD_Y = -0.255
NIB_MIN = 0.00265


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


OBJECTS = [
    Body.build("pen_tray", (START[0], START[1], TOP), 0.13, [
        BoxShape((0.0, 0.0, 0.003), (0.153, 0.053, 0.006), (0.23, 0.30, 0.32)),
        *[BoxShape((x, 0.0, 0.011), (0.017, 0.037, 0.010), (0.23, 0.30, 0.32))
          for x in (0.052, -0.044)],
        *[BoxShape((x, y, 0.017), (0.013, 0.006, 0.014), (0.23, 0.30, 0.32))
          for x in (0.052, -0.044) for y in (-0.013, 0.013)],
    ], fixed=True, color=(0.23, 0.30, 0.32), friction=(0.81, 0.67), contact_offset=0.00025),
    Body.build("test_paper", tuple(PAPER), 0.014, [
        BoxShape((0, 0, -0.0008), (*PAPER_SIZE, 0.0016), WHITE),
        BoxShape((0, -0.071, 0.0002), (0.044, 0.001, 0.0004), (0.53, 0.56, 0.62), collide=False),
        *[BoxShape((TEST_X-PAPER[0], y-PAPER[1], 0.0002), (0.015, 0.001, 0.0004),
                   (0.45, 0.50, 0.55), collide=False) for y in (TEST_LOW, TEST_HIGH)],
    ], fixed=True, color=WHITE, friction=(0.62, 0.48), contact_offset=0.00035),
    Linkage.build("paint_marker", tuple(START), 0.026, [
        CylShape((0.013, 0.0, 0.0), 0.0095, 0.104, DARK, axis="X"),
        CylShape((0.045, 0.0, 0.0), 0.00955, 0.026, BLUE, axis="X"),
        CylShape((0.059, 0.0, 0.0), 0.0087, 0.010, BLUE, axis="X"),
        BoxShape((0.052, 0, -0.0085), (0.023, 0.012, 0.0038), BLUE),
        CylShape((-0.046, 0.0, 0.0), 0.0063, 0.016, DARK, axis="X"),
        CylShape((-0.057, 0.0, 0.0), 0.0040, 0.010, DARK, axis="X"),
        CylShape((0.013, 0.0, 0.0), 0.0096, 0.023, (0.75, 0.75, 0.71), collide=False, axis="X"),
    ], [
        Link("nib", (CylShape((-0.065, 0, 0), 0.0015, 0.015, (0.76, 0.76, 0.71), axis="X"),
                     CylShape((-0.0725, 0, 0), 0.0014, 0.003, (0.78, 0.77, 0.75), axis="X")),
             0.002, tuple(START + [-0.057, 0.0, 0.0]), axis="X", slide=True,
             limits=(-0.0001, NIB_TRAVEL), stiffness=375.0, spring_damping=1.10, report_contacts=True),
    ], floating=True, color=DARK, friction=(0.80, 0.55), contact_offset=0.00035, solver_iterations=40),
]

_state = {}


@on_reset
def _reset():
    if _state.get("stage") is not None:
        _state["stage"].RemovePrim("/World/PaintTest")
    _state.clear()
    _state.update(frames=0, compressed=False, pumps=0, score=0.0, marks=set(),
                  joint=None, contacts=None, previous=None, arms={}, drawn=[], stage=None)


_reset()


def paint_pixel(point, name, radius):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    if _state["stage"] is None:
        _state["stage"] = get_current_stage()
    dot = UsdGeom.Cylinder.Define(_state["stage"], f"/World/PaintTest/{name}")
    dot.CreateRadiusAttr(float(radius))
    dot.CreateHeightAttr(0.0003)
    dot.CreateDisplayColorAttr([Gf.Vec3f(*INK)])
    dot.AddTranslateOp().Set(Gf.Vec3d(float(point[0]), float(point[1]), PAPER[2] + 0.00015))


def paper_force(base):
    if _state["contacts"] is None:
        from isaaclab_physx.physics import PhysxManager

        view = PhysxManager.get_physics_sim_view()
        env = base.scene.env_prim_paths[0]
        _state["contacts"] = view.create_rigid_contact_view(
            [f"{env}/paint_marker/nib"], filter_patterns=[[f"{env}/test_paper/body"]],
            max_contact_data_count=32)
    v = _state["contacts"].get_contact_force_matrix(dt=base.physics_dt)
    forces = v.numpy() if hasattr(v, "numpy") else as_numpy(v)
    return float(np.linalg.norm(forces.reshape(-1, 3).sum(axis=0)))


def mark(tip):
    previous = _state["previous"]
    if previous is None or np.linalg.norm(previous - tip[:2]) > 0.012:
        previous = tip[:2]
    span = float(np.linalg.norm(previous - tip[:2]))
    for point in np.linspace(previous, tip[:2], 1 + int(math.ceil(span / 0.0017))):
        cell = tuple(np.rint(point / 0.0017).astype(int))
        if cell not in _state["marks"]:
            _state["marks"].add(cell)
            paint_pixel(point, f"p{len(_state['marks'])}", 0.0015)
    _state["previous"] = tip[:2].copy()
    if abs(tip[0] - TEST_X) <= 0.018 and TEST_LOW - 0.003 <= tip[1] <= TEST_HIGH + 0.003:
        _state["drawn"].append(tip[:2].copy())


@on_step
def _step(environment):
    base = getattr(environment, "unwrapped", environment)
    marker = base.scene["paint_marker"]
    if _state["joint"] is None:
        _state["joint"] = marker.joint_names.index("nib_turn")
    stroke = float(as_numpy(marker.data.joint_pos)[0, _state["joint"]])
    pose = as_numpy(marker.data.root_link_pose_w)[0]
    rr = rotation(pose[3:7])
    nib_dir = -rr[:, 0]
    tip = pose[:3] + (AXIAL_END - stroke) * nib_dir
    over_paper = np.all(np.abs(tip[:2] - PAPER[:2]) < PAPER_SIZE / 2)
    upright = -nib_dir[2] >= math.cos(math.radians(25))
    touch = over_paper and abs(tip[2] - PAPER[2]) <= 0.004 and paper_force(base) >= 0.035

    if touch and upright and stroke >= NIB_MIN and not _state["compressed"]:
        _state["pumps"] += 1
        _state["compressed"] = True
    if stroke <= 0.0008:
        _state["compressed"] = False
    # The controller samples at 20 Hz. A compressed nib whose point is still in
    # the paper contact band also records contact between physics samples.
    writing_contact = touch or (over_paper and -0.002 <= tip[2]-PAPER[2] <= 0.00065
                                and stroke >= 0.00035)
    writing_angle = -nib_dir[2] >= math.cos(math.radians(52))
    if writing_contact and writing_angle and _state["pumps"] >= 4:
        mark(tip)
    else:
        _state["previous"] = None

    if _state["drawn"]:
        occupied = set(int(round((xy[1] - TEST_LOW) / 0.005)) for xy in _state["drawn"])
    else:
        occupied = set()
    covered = streak = 0
    for index in range(16):
        streak = streak + 1 if index in occupied else 0
        covered = max(covered, streak)
    score = 0.0
    if covered >= 7:
        score = 0.5
    elif _state["pumps"] >= 4 and _state["marks"]:
        score = 0.25
    if covered >= 14:
        vel = as_numpy(marker.data.root_link_vel_w)[0]
        settled = np.linalg.norm(vel[:3]) <= 0.014 and abs(nib_dir[2]) <= 0.23 and pose[2] <= TOP + 0.014
        if settled:
            clear = True
            for side, anchor in station.BASES.items():
                robot = base.scene[f"{side}_robot"]
                if side not in _state["arms"]:
                    _state["arms"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
                q = as_numpy(robot.data.joint_pos)[0][_state["arms"][side]]
                hand, _ = ik.tcp_pose(q, anchor)
                relative = hand - pose[:3]
                along = float(np.clip(np.dot(relative, nib_dir), -0.067, AXIAL_END))
                clear &= np.linalg.norm(relative - along*nib_dir) >= 0.065
            if clear:
                score = 1.0
    _state["score"] = score
    _state["frames"] += 1
    if os.getenv("MARKER_DEBUG") == "1" and _state["frames"] % 5 == 0:
        print(f"MARKER {_state['frames']:03} stroke={stroke*1000:.2f}mm contact={touch} "
              f"pumps={_state['pumps']} paint={len(_state['marks'])} span={covered} score={score:.2f} "
              f"tip={np.round(tip-[0,0,TOP],4)} origin={np.round(pose[:3]-[0,0,TOP],4)} "
              f"axis={np.round(nib_dir,3)}", flush=True)


def score(objects):
    return min(_state["score"], 0.5)


def success(objects):
    return _state["score"] == 1.0


def demo():
    q = -0.0008
    tipward_grasp = 0.025
    pitch = 1.47
    turn = ry(-pitch) @ grip_rotation()
    nose = (AXIAL_END - tipward_grasp) * math.sin(pitch)
    level = float(PAPER[2])
    def down(dt, x, y, offset, label):
        return Waypoint(dt, (x + 0.012, y, level + nose + offset), q, label, turn)

    start_grasp = START + [-tipward_grasp, 0.002, 0.002]
    path = [
        Waypoint(2.3, tuple(START + [-tipward_grasp,0,0.070]), -0.028, "approach the paint marker"),
        Waypoint(0.9, tuple(start_grasp), -0.028, "reach the barrel"),
        Waypoint(0.63, tuple(start_grasp), q, "hold the barrel"),
        Waypoint(0.7, (0.326, START[1], TOP + 0.150), q, "lift the marker"),
        down(1.85, TEST_X, -0.103, 0.091, "turn the nib down above the test paper"),
    ]
    for _ in range(1 if os.getenv("MARKER_ONCE") == "1" else 5):
        path.extend([down(0.73, TEST_X, -0.103, -0.006, "depress the valve spring against the paper"),
                     down(0.58, TEST_X, -0.103, 0.029, "let the nib spring all the way back")])
    path.extend([
        down(0.75, TEST_X, TEST_LOW, 0.007, "move to the start of the test stroke"),
        down(0.65, TEST_X, TEST_LOW, -0.0082, "make light contact with the rough sheet"),
        down(2.9, TEST_X, TEST_HIGH, -0.0082, "draw a continuous blue test stroke"),
        down(0.68, TEST_X, TEST_HIGH, 0.065, "lift the nib off the paper"),
        Waypoint(1.42, (0.287, HOLD_Y, TOP + 0.130), q, "turn the marker horizontal"),
        Waypoint(1.15, (0.287, HOLD_Y, TOP + 0.012), q, "return it to the bench"),
        Waypoint(0.51, (0.287, HOLD_Y, TOP + 0.012), -0.028, "release the marker"),
        Waypoint(4.25, (0.287, HOLD_Y, TOP + 0.160), -0.028, "withdraw"),
    ])
    return {"right": path}
