"""Pick up a free-turning adhesive roller and remove loose fibers from a chair cushion."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Joint, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

CUSHION_X, CUSHION_Y = 0.325, 0.062
SIDE, HEIGHT = 0.300, 0.038
RADIUS, ROLL_WIDTH = 0.027, 0.102
HANDLE_OFFSET = 0.110
PARK = (0.265, -0.250, TOP + RADIUS + 0.002)
NO_CONTACT_ROLL_Z = TOP + HEIGHT + RADIUS + 0.030
MIN_FORCE = 0.35
FIBERS = tuple(
    (x, y, HEIGHT / 2 + 0.0005)
    for y in (-0.067, -0.028, 0.009, 0.052, 0.075)
    for x in (-0.073, -0.027, 0.034)
)

BLACK = (0.12, 0.13, 0.14)
TRIM = (0.08, 0.09, 0.10)
LINT = (0.92, 0.91, 0.82)
TAPE = (0.88, 0.91, 0.90)
PLASTIC = (0.20, 0.29, 0.34)
PAD = (0.28, 0.38, 0.40)

fabric = [BoxShape((0, 0, 0), (SIDE, SIDE, HEIGHT), BLACK)]
for x in (-SIDE / 2, SIDE / 2):
    fabric.append(CylShape((x, 0, 0), 0.003, SIDE - 0.005, TRIM, axis="Y", collide=False))
for y in (-SIDE / 2, SIDE / 2):
    fabric.append(CylShape((0, y, 0), 0.003, SIDE - 0.005, TRIM, axis="X", collide=False))
fiber_start = len(fabric)
for index, p in enumerate(FIBERS):
    fabric.append(BoxShape(p, (0.012 + 0.002 * (index % 3), 0.002, 0.0005), LINT,
                           yaw=math.radians((index * 41) % 180), collide=False))

ROLLER_BASE = [
    CylShape((0.0, 0.0, 0.0), RADIUS, ROLL_WIDTH, TAPE, axis="Y"),
    CylShape((0.0, ROLL_WIDTH / 2, 0.0), 0.020, 0.001, PLASTIC, axis="Y", collide=False),
]
ROLLER_BASE += [BoxShape((0, 0, 0), (0.012, 0.002, 0.0005), LINT, collide=False)] * len(FIBERS)
ROLLER_POS = (PARK[0], PARK[1] + HANDLE_OFFSET, PARK[2])
OBJECTS = [
    Body.build("cushion", (CUSHION_X, CUSHION_Y, TOP + HEIGHT / 2), 0.42, shapes=fabric,
               friction=(0.82, 0.69), damping=(0.5, 0.1)),
    Body.build("parking_mat", (PARK[0], PARK[1] + 0.072, TOP + 0.001), 0.15, fixed=True,
               shapes=[BoxShape((0, 0, 0), (0.095, 0.220, 0.002), PAD)]),
    Body.build("handle", PARK, 0.047, friction=(0.9, 0.73), shapes=[
        CylShape((0, -0.002, 0), 0.015, 0.100, PLASTIC, axis="Y"),
        CylShape((0, -0.053, 0), 0.022, 0.008, PLASTIC, axis="Y"),
        CylShape((0, 0.056, 0), 0.009, 0.055, PLASTIC, axis="Y"),
        CylShape((0, 0.101, 0), 0.010, 0.073, PLASTIC, axis="Y"),
    ]),
    Body.build("roller", ROLLER_POS, 0.065, shapes=ROLLER_BASE, friction=(0.97, 0.88),
               joint=Joint(anchor=ROLLER_POS, parent="handle", axis="Y", damping=0.00011)),
]

_trial = {}


@on_reset
def _reset() -> None:
    _trial.clear()
    _trial.update(cleaned=set(), step=0, pressure=0.0, shift=0.0, clear=False, joints={}, view=None, visuals=None)


_reset()


def _read(body):
    pose = as_numpy(body.data.root_link_pose_w)[0]
    return np.asarray(pose[:3], float), rotation(pose[3:7])


def _init_views(env_path: str) -> None:
    from isaaclab_physx.physics import PhysxManager
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom

    roller = f"{env_path}/roller/body"
    cushion = f"{env_path}/cushion/body"
    view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        [roller], filter_patterns=[[cushion]], max_contact_data_count=12
    )
    if view.sensor_count != 1 or view.filter_count != 1:
        raise RuntimeError("Could not resolve roller–cushion normal contact")
    stage = get_current_stage()
    source = [UsdGeom.Imageable(stage.GetPrimAtPath(f"{cushion}/s{i + fiber_start}")) for i in range(len(FIBERS))]
    picked = [UsdGeom.Xformable(stage.GetPrimAtPath(f"{roller}/s{i + 2}")) for i in range(len(FIBERS))]
    for fiber in source:
        fiber.MakeVisible()
    for fiber in picked:
        UsdGeom.Imageable(fiber.GetPrim()).MakeInvisible()
    _trial.update(view=view, visuals=(source, picked))


def _show_pickup(index: int, local: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    source, picked = _trial["visuals"]
    source[index].MakeInvisible()
    radial = np.asarray([local[0], 0.0, local[2]])
    radial *= RADIUS / max(np.linalg.norm(radial), 0.001)
    radial[1] = local[1]
    translate = picked[index].GetOrderedXformOps()[0]
    translate.Set(Gf.Vec3d(*map(float, radial)))
    UsdGeom.Imageable(picked[index].GetPrim()).MakeVisible()


@on_step
def _track(env) -> None:
    base = getattr(env, "unwrapped", env)
    if _trial["view"] is None:
        _init_views(base.scene.env_prim_paths[0])
    _trial["step"] += 1
    view = _trial["view"]
    force = view.get_contact_force_matrix(dt=float(base.physics_dt))
    force_array = force.numpy() if hasattr(force, "numpy") else as_numpy(force)
    pressure = float(np.linalg.norm(force_array.reshape(-1, 3)[0]))
    roller, cushion = base.scene["roller"], base.scene["cushion"]
    r_pos, r_rotation = _read(roller)
    c_pos, c_rotation = _read(cushion)
    handle_pos, _ = _read(base.scene["handle"])
    clear = True
    for side in station.BASES:
        robot = base.scene[f"{side}_robot"]
        if side not in _trial["joints"]:
            _trial["joints"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q = as_numpy(robot.data.joint_pos)[0][_trial["joints"][side]]
        tcp, _ = ik.tcp_pose(q, station.BASES[side])
        clear &= min(np.linalg.norm(tcp - handle_pos), np.linalg.norm(tcp - r_pos)) > 0.065
    r_vel = as_numpy(roller.data.root_link_vel_w)[0]
    c_vel = as_numpy(cushion.data.root_link_vel_w)[0]
    axis = r_rotation[:, 1]
    up = c_rotation[:, 2]
    across = np.cross(axis, up)
    speed = float(np.dot(r_vel[:3] - c_vel[:3], across))
    rim_speed = float(np.dot(r_vel[3:], axis) * RADIUS)
    rolling = abs(speed) >= 0.018 and abs(speed - rim_speed) <= max(0.02, abs(speed) * 0.52)
    shift = float(np.linalg.norm(c_pos[:2] - [CUSHION_X, CUSHION_Y]))
    _trial.update(pressure=pressure, shift=shift, clear=clear)
    if pressure >= MIN_FORCE and rolling and abs(float(np.dot(axis, up))) <= 0.13:
        for index, local in enumerate(FIBERS):
            if index in _trial["cleaned"]:
                continue
            offset = c_pos + c_rotation @ local - r_pos
            if abs(float(np.dot(offset, axis))) < ROLL_WIDTH / 2 - 0.002 and abs(float(np.dot(offset, across))) < 0.009:
                _trial["cleaned"].add(index)
                _show_pickup(index, r_rotation.T @ offset)
    if os.getenv("LINT_DEBUG") and _trial["step"] % 10 == 0:
        print(f"lint step={_trial['step']} collected={sorted(_trial['cleaned'])} force={pressure:.3f} "
              f"rolling={rolling} speed={speed:.3f}/{rim_speed:.3f} "
              f"roller={np.round(r_pos - [0, 0, TOP], 3)} handle={np.round(handle_pos - [0, 0, TOP], 3)} "
              f"tilt={np.dot(axis, up):.3f} cushion_shift={shift:.3f} clear={clear}", flush=True)


def score(objects) -> float:
    return 0.9 * len(_trial["cleaned"]) / len(FIBERS)


def success(objects) -> bool:
    handle, roller = objects["handle"], objects["roller"]
    parked = roller.low[2] <= TOP + 0.010
    stable = handle.resting(max_speed=0.02) and roller.resting(max_speed=0.02, max_spin=0.7)
    if os.getenv("LINT_DEBUG") and _trial["step"] % 10 == 0 and len(_trial["cleaned"]) == len(FIBERS):
        print(f"lint finish parked={parked} stable={stable} low_cm={(roller.low[2] - TOP) * 100:.2f}",
              flush=True)
    return bool(
        len(_trial["cleaned"]) == len(FIBERS)
        and _trial["shift"] <= 0.03
        and _trial["clear"]
        and parked
        and stable
    )


def demo():
    x, y, z = PARK
    yaw = math.pi / 2
    working_z = TOP + HEIGHT + RADIUS
    if os.getenv("LINT_HOVER") == "1":
        working_z = NO_CONTACT_ROLL_Z

    def pose(seconds, x, y, z, grip, label):
        return Waypoint(seconds, (x, y, z), grip, label, yaw)

    left_lane = 0.036 - HANDLE_OFFSET
    right_lane = 0.127 - HANDLE_OFFSET
    lift_z = TOP + 0.120
    return {
        "left": [],
        "right": [
            pose(2.0, x, y, lift_z, FINGER_OPEN, "over the roller handle"),
            pose(1.2, x, y, z + 0.004, FINGER_OPEN, "reach handle"),
            pose(0.8, x, y, z + 0.004, FINGER_CLOSED, "grip handle"),
            pose(1.2, x, y, lift_z, FINGER_CLOSED, "lift roller clear"),
            pose(1.4, 0.230, left_lane, lift_z, FINGER_CLOSED, "over first row"),
            pose(0.8, 0.230, left_lane, working_z, FINGER_CLOSED, "press onto cushion"),
            pose(2.0, 0.391, left_lane, working_z, FINGER_CLOSED, "roll first pass"),
            pose(0.7, 0.393, left_lane, lift_z, FINGER_CLOSED, "lift at end"),
            pose(1.2, 0.395, right_lane, lift_z, FINGER_CLOSED, "over second row"),
            pose(0.8, 0.395, right_lane, working_z, FINGER_CLOSED, "contact second row"),
            pose(2.0, 0.221, right_lane, working_z, FINGER_CLOSED, "roll second pass"),
            pose(0.8, 0.221, right_lane, lift_z, FINGER_CLOSED, "lift roller away"),
            pose(1.4, x, y, lift_z, FINGER_CLOSED, "return over mat"),
            pose(1.1, x, y, z + 0.006, FINGER_CLOSED, "lower roller"),
            pose(0.8, x, y, z + 0.006, FINGER_OPEN, "release handle"),
            pose(1.2, x, y, lift_z, FINGER_OPEN, "withdraw"),
            pose(3.8, x, y, lift_z, FINGER_OPEN, "wait for the roller to settle"),
        ],
    }
