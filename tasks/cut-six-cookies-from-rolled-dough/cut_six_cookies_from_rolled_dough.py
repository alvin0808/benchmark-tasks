"""Use a 52 mm round cutter to fit six complete circles on one piece of cookie dough."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, allow_matrix_grips, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, axis_angle_matrix, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
START = np.array([0.306, -0.163, TOP + 0.001])
BOARD_HEIGHT = 0.006
BOTTOM = TOP + BOARD_HEIGHT
THICKNESS = 0.0042
DOUGH = np.array([0.334, -0.001, BOTTOM + THICKNESS])
SIZE = np.array([0.198, 0.137])
RIM_WIDTH = 0.0008
NUMERIC_GAP = 0.00022
MAX_STAMP_DRIFT = 0.0025
# Raw control short dough from Laguna et al., Figure 1 (mm and N).
# Scale the representative profile by Table 1's measured mean plateau.
WIRE_DEPTH_MM = np.array([0.0, 0.25, 0.75, 1.25, 2.25, 3.25, 4.2])
WIRE_LOAD_N = np.array([0.0, 0.049, 0.095, 0.137, 0.181, 0.203, 0.224]) * (0.26 / 0.239)
WIRE_LENGTH_M = 0.030
RADIUS = 0.026
HEIGHT = 0.03
STEEL = (0.55, 0.62, 0.69)
DOUGH_COLOR = (0.79, 0.66, 0.48)
OUTLINE = (0.37, 0.26, 0.16)
RIM_SEGMENTS = 32
RIM_RADIUS = math.hypot(RADIUS + RIM_WIDTH / 2, RADIUS * math.tau / (2*RIM_SEGMENTS) * 1.01)
RIM_SAMPLES = 64
RIM_ANGLES = np.arange(RIM_SAMPLES) * math.tau / RIM_SAMPLES
RIM_POINTS = np.column_stack((RIM_RADIUS*np.cos(RIM_ANGLES),
                             RIM_RADIUS*np.sin(RIM_ANGLES), np.zeros(RIM_SAMPLES)))


allow_matrix_grips()


def ring(z, radius, width, height, color, count=RIM_SEGMENTS):
    shell = []
    for i in range(count):
        angle = (i + 0.5) * math.tau / count
        phi = math.tau / count
        shell.append(BoxShape((radius*math.cos(angle), radius*math.sin(angle), z),
                              (width, radius * phi * 1.01, height), color, angle))
    return shell


OBJECTS = [
    Body.build("board", (DOUGH[0], DOUGH[1], TOP), 0.3, [
        BoxShape((0, 0, BOARD_HEIGHT / 2), (0.207, 0.164, BOARD_HEIGHT), (0.50, 0.34, 0.20)),
    ], fixed=True, color=(0.50, 0.34, 0.20), friction=(0.52, 0.43), contact_offset=0.00035),
    Body.build("dough", tuple(DOUGH), 0.089, [
        BoxShape((0, 0, -THICKNESS/2), (*SIZE, THICKNESS), DOUGH_COLOR, collide=False),
    ], fixed=True, color=DOUGH_COLOR, friction=(0.63, 0.54), contact_offset=0.00030),
    Body.build("cutter", tuple(START), 0.027, [
        *ring(0.015, RADIUS, RIM_WIDTH, 0.030, STEEL),
        *ring(0.0285, RADIUS + 0.0013, 0.0024, 0.0035, STEEL),
    ], color=STEEL, friction=(0.82, 0.65), contact_offset=0.00028,
       report_contacts=True, solver_iterations=40),
]

_state = {}


@on_reset
def _reset():
    if _state.get("stage") is not None:
        _state["stage"].RemovePrim("/World/CookieCuts")
    _state.clear()
    _state.update(joints={}, view=None, stage=None, cut=[], pressed=False, stamp_center=None,
                  through=np.zeros(RIM_SAMPLES, dtype=bool), damaged=False, step=0, score=0.0)


_reset()


def board_contact(base):
    from isaaclab_physx.physics import PhysxManager

    if _state["view"] is None:
        view = PhysxManager.get_physics_sim_view()
        root = base.scene.env_prim_paths[0]
        _state["view"] = view.create_rigid_contact_view(
            [f"{root}/cutter/body"], filter_patterns=[[f"{root}/board/body"]], max_contact_data_count=128)
    contact = _state["view"].get_contact_force_matrix(dt=float(base.physics_dt))
    return abs(float(contact.numpy().reshape(-1, 3).sum(axis=0)[2]))


def dough_load(cutter, points, coord, enabled):
    """Apply the distributed vertical cutting load at the measured rim positions."""
    import torch

    depth_mm = (DOUGH[2] - points[:, 2]) * 1000
    inside = np.all(np.abs(points[:, :2] - DOUGH[:2]) <= SIZE / 2, axis=1)
    width = math.tau * RIM_RADIUS / RIM_SAMPLES
    load = (np.interp(depth_mm, WIRE_DEPTH_MM, WIRE_LOAD_N) * inside
            * width / WIRE_LENGTH_M) if enabled else np.zeros(RIM_SAMPLES)
    total = float(load.sum())
    moment_x, moment_y = (points[:, :2] - coord[:2]).T @ load
    device = cutter.device
    force = torch.tensor([[[0.0, 0.0, total]]], dtype=torch.float32, device=device)
    torque = torch.tensor([[[moment_y, -moment_x, 0.0]]], dtype=torch.float32, device=device)
    cutter.permanent_wrench_composer.set_forces_and_torques_index(force, torque, is_global=True)
    cutter.write_data_to_sim()
    return total


def show_cut(center, identifier):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    if _state["stage"] is None:
        _state["stage"] = get_current_stage()
    points = []
    segments = 48
    for k in range(segments):
        angle = math.tau * k / segments
        for radius in (RADIUS-0.00060, RADIUS+0.00060):
            points.append(Gf.Vec3f(float(center[0]+radius*math.cos(angle)),
                                  float(center[1]+radius*math.sin(angle)), float(DOUGH[2]+0.00024)))
    quads = []
    for k in range(segments):
        j = 2 * ((k+1) % segments)
        quads.extend([2*k, 2*k+1, j+1, j])
    seam = UsdGeom.Mesh.Define(_state["stage"], f"/World/CookieCuts/c{identifier}")
    seam.CreatePointsAttr(points)
    seam.CreateFaceVertexCountsAttr([4] * segments)
    seam.CreateFaceVertexIndicesAttr(quads)
    seam.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    seam.CreateDoubleSidedAttr(True)
    seam.CreateDisplayColorAttr([Gf.Vec3f(*OUTLINE)])


def _hand(tcp_model, side):
    robot = tcp_model.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    q = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(q, station.BASES[side])[0]


@on_step
def _step(environment):
    base = getattr(environment, "unwrapped", environment)
    cutter = base.scene["cutter"]
    pose = as_numpy(cutter.data.root_link_pose_w)[0]
    coord = pose[:3]
    rr = rotation(pose[3:7])
    face_up = rr[2, 2] >= 0.0
    height = coord[2] if face_up else coord[2] + HEIGHT*rr[2, 2]
    rim_tilt = math.sqrt(max(0.0, 1.0 - rr[2, 2]**2))
    rim = coord + RIM_POINTS @ rr.T
    furthest = coord[2] + RIM_RADIUS*rim_tilt
    overlaps_sheet = np.all(np.abs(coord[:2] - DOUGH[:2]) <= SIZE/2 + RIM_RADIUS)
    same_cut = any(np.linalg.norm(prev - coord[:2]) <= 0.003 for prev in _state["cut"])
    load = dough_load(cutter, rim, coord, face_up and overlaps_sheet and not same_cut)
    bottom_force = board_contact(base) if face_up and abs(height-BOTTOM) <= 0.002 else 0.0
    clears_dough = height - RIM_RADIUS*rim_tilt >= DOUGH[2] + 0.0005
    if not face_up or clears_dough:
        _state["pressed"] = False
        _state["stamp_center"] = None
        _state["through"].fill(False)
    if overlaps_sheet and bottom_force >= 0.10 and not _state["pressed"]:
        center = _state["stamp_center"]
        if center is None or np.linalg.norm(coord[:2] - center) > MAX_STAMP_DRIFT:
            _state["stamp_center"] = coord[:2].copy()
            _state["through"].fill(False)
        _state["through"] |= rim[:, 2] <= BOTTOM + NUMERIC_GAP
    press = bool(_state["through"].all())
    if press and not _state["pressed"]:
        xy = coord[:2].copy()
        room = np.all(np.abs(xy - DOUGH[:2]) + RADIUS + 0.001 <= SIZE/2)
        distinct = all(np.linalg.norm(prev - xy) >= 2*RADIUS + 0.001 for prev in _state["cut"])
        stutter = any(np.linalg.norm(prev - xy) <= 0.003 for prev in _state["cut"])
        if room and distinct:
            _state["cut"].append(xy)
            show_cut(xy, len(_state["cut"]))
        elif not stutter:
            _state["damaged"] = True
        _state["pressed"] = True

    count = min(len(_state["cut"]), 6)
    score = count / 8.0
    if count == 6:
        still = np.linalg.norm(as_numpy(cutter.data.root_link_vel_w)[0, :3]) <= 0.014
        outside = np.any(np.abs(coord[:2] - DOUGH[:2]) >= SIZE/2 + RADIUS + 0.007)
        if outside and abs(height - TOP) <= 0.004 and still:
            if all(np.linalg.norm(_hand(base, s) - coord) >= 0.074 for s in ("left", "right")):
                score = 1.0
    _state["score"] = score if not _state["damaged"] else 0.0
    _state["step"] += 1
    if os.getenv("COOKIE_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"COOKIE {_state['step']:03} n={count} press={press} damaged={_state['damaged']} score={_state['score']:.3f} "
              f"rim={np.count_nonzero(_state['through'])}/{RIM_SAMPLES} top_rim_gap={(furthest-BOTTOM)*1000:.3f}mm "
              f"load={load:.2f}N board={bottom_force:.2f}N up={np.round(rr[:2,2],4)} "
              f"pos={np.round(coord-[0,0,TOP],4)} hand={np.round(_hand(base,'right')-[0,0,TOP],4)}", flush=True)


def score(objects):
    return min(_state["score"], 0.75)


def success(objects):
    return _state["score"] == 1.0


def demo():
    pickup = START + [0,0,0.022]
    closed, opened = -0.015, -0.042
    path = [
        Waypoint(2.0, tuple(pickup + [0,0,0.073]), opened, "approach the cookie cutter"),
        Waypoint(0.85, tuple(pickup), opened, "reach the rolled metal rim"),
        Waypoint(0.58, tuple(pickup), closed, "take the cutter around its upper rim"),
        Waypoint(0.75, tuple(pickup+[0,0,0.055]), closed, "lift the cutter"),
    ]
    centers = [(DOUGH[0]+x, DOUGH[1]+y) for x in (0.062, 0., -0.062) for y in (-0.030, 0.030)]
    if os.getenv("COOKIE_OVERLAP") == "1":
        centers[2] = (centers[1][0]-0.030, centers[1][1])
    shallow = os.getenv("COOKIE_SHALLOW") == "1"
    press_height = TOP + (0.0280 if shallow else 0.006)
    # Compensate the repeatable wrist deflection under load at each station position.
    corrections = [(0, 0), (-0.010, 0), (-0.011, -0.014), (-0.007, -0.010), (0, 0), (-0.012, -0.004)]
    for (x, y), (roll, pitch) in zip(centers, corrections):
        level_under_load = (axis_angle_matrix(np.array([1, 0, 0]), roll)
                            @ axis_angle_matrix(np.array([0, 1, 0]), pitch) @ grip_rotation())
        path += [
            Waypoint(1.0, (x, y, TOP+0.071), closed, "lay out the next unused circle"),
            Waypoint(1.15, (x, y, press_height), closed, "press the cutter through the dough", level_under_load),
            Waypoint(0.8, (x, y, press_height), closed, "settle the rim", level_under_load),
            Waypoint(0.76, (x, y, TOP+0.071), closed, "lift the cutter straight out"),
        ]
    park = (0.271, -0.164, TOP+0.023)
    path += [
        Waypoint(0.9, (park[0],park[1],TOP+0.083), closed, "carry the cutter outside the dough"),
        Waypoint(0.85, park, closed, "set it down"),
        Waypoint(0.6, park, opened, "release"),
        Waypoint(1.8, (park[0],park[1],TOP+0.139), opened, "withdraw"),
    ]
    return {"right": path}
