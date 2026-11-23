"""Use a 52 mm round cutter to fit six complete circles on one piece of cookie dough."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, as_numpy, on_reset, on_step, rotation, segment
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
START = np.array([0.306, -0.163, TOP + 0.001])
DOUGH = np.array([0.334, -0.001, TOP + 0.0058])
SIZE = np.array([0.198, 0.137])
BASE_CUT = TOP + 0.0017
RADIUS = 0.026
HEIGHT = 0.03
STEEL = (0.55, 0.62, 0.69)
DOUGH_COLOR = (0.79, 0.66, 0.48)
OUTLINE = (0.37, 0.26, 0.16)


def ring(z, radius, width, height, color, count=32):
    shell = []
    for i in range(count):
        angle = (i + 0.5) * math.tau / count
        phi = math.tau / count
        shell.append(BoxShape((radius*math.cos(angle), radius*math.sin(angle), z),
                              (width, radius * phi * 1.01, height), color, angle))
    return shell


OBJECTS = [
    Body.build("board", (DOUGH[0], DOUGH[1], TOP), 0.3, [
        BoxShape((0, 0, 0.0007), (0.207, 0.164, 0.0014), (0.50, 0.34, 0.20)),
    ], fixed=True, color=(0.50, 0.34, 0.20), friction=(0.52, 0.43), contact_offset=0.00035),
    Body.build("dough", tuple(DOUGH), 0.089, [
        BoxShape((0, 0, -0.0021), (*SIZE, 0.0042), DOUGH_COLOR, collide=False),
        BoxShape((0, 0, BASE_CUT-DOUGH[2]-0.0002), (*SIZE, 0.0004), DOUGH_COLOR),
    ], fixed=True, color=DOUGH_COLOR, friction=(0.63, 0.54), contact_offset=0.00030),
    Body.build("cutter", tuple(START), 0.027, [
        *ring(0.015, RADIUS, 0.0008, 0.030, STEEL),
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
    _state.update(joints={}, view=None, stage=None, cut=[], pressed=False, damaged=False, step=0, score=0.0)


_reset()


def contact_force(base):
    from isaaclab_physx.physics import PhysxManager

    if _state["view"] is None:
        view = PhysxManager.get_physics_sim_view()
        root = base.scene.env_prim_paths[0]
        _state["view"] = view.create_rigid_contact_view(
            [f"{root}/cutter/body"], filter_patterns=[[f"{root}/dough/body"]], max_contact_data_count=128)
    contact = _state["view"].get_contact_force_matrix(dt=float(base.physics_dt))
    return abs(float(contact.numpy().reshape(-1, 3).sum(axis=0)[2]))


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
    overlaps_sheet = np.all(np.abs(coord[:2] - DOUGH[:2]) <= SIZE/2 + RADIUS)
    at_depth = face_up and height + RADIUS*rim_tilt <= BASE_CUT + 0.0022
    press = at_depth and overlaps_sheet and contact_force(base) >= 0.8
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
    if height - RADIUS*rim_tilt >= DOUGH[2] + 0.005:
        _state["pressed"] = False

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
              f"tilt={rim_tilt:.3f} rim={(height-TOP)*1000:.1f} "
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
    for x, y in centers:
        path += [
            Waypoint(1.0, (x, y, TOP+0.071), closed, "lay out the next unused circle"),
            Waypoint(0.86, (x, y, TOP+0.016), closed, "push the entire cutting edge down through the dough"),
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
