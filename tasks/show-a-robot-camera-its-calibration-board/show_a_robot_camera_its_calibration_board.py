"""On the station table: a USB webcam on a post at the far side of the table, looking back at the arms, and a printed
checkerboard calibration board leaning in a sloped stand (its rack) in front of the arms, its printed face up toward the
camera.

Task (instruction): show the camera on the post the calibration board in all the views its calibration needs, then put
the board back in its rack. The instruction does not say which views or how. The views the task checks are the ones the
ROS camera_calibration tool's tutorial ("Tutorial: Monocular Calibration", "Moving the Checkerboard") asks for:
"checkerboard on the camera's left, right, top and bottom of field of view", "checkerboard filling the whole field of
view", "checkerboard tilted to the left, right, top and bottom", and "At each step, hold the checkerboard still".
OpenCV's chessboard detector needs every inner corner and "white space (like a square-thick border, the wider the
better) around the board", so the board may be held only by its plain grey ends: held by one end it twists down in the
grip (its middle is 14 cm away), and a grip on its top edge covers the white border, so it takes both grippers, one at
each top corner.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- board: 300 x 160 x 3 mm aluminium composite, 0.18 kg, loose. Printed on its front: a checkerboard of 9 x 6 squares of
  20 mm (8 x 5 = 40 inner corners) inside a white border 20 mm wide (the printed area 220 x 160 mm), and a plain grey
  band 40 mm wide at each end. In the rack it leans back 60 degrees from upright, its bottom edge on the stand's base
  against a lip (the edge's middle at x 0.29), its back on a sloped rest.
- rack: fixed, a wooden stand: base x 0.135 to 0.310, y -0.13 to 0.13, 8 mm thick; a lip 10 mm tall in front of the
  board's bottom edge; a sloped rest under the board's back, 18 cm wide, from 3 to 10 cm up the board.
- camera: fixed, a USB webcam (90 x 30 x 30 mm) with its lens at (0.555, 0, 0.27 above the table), looking at (0.28, 0,
  0.11 above the table), 30 degrees down; 1280 x 720 pixels, focal length 1100 pixels in both directions (60 degrees
  across), principal point at the image centre (640, 360), no lens distortion. It stands on a 15 mm square post at x
  0.585.

What the camera sees is computed from the board's pose and those intrinsics (a pinhole camera). A view counts when the
board's 9 x 6 squares all lie at least 8 pixels inside the image, the board faces the camera (within 50 degrees), no
part of either arm (links, gripper, fingers) is in front of the checkerboard or its white border as seen from the camera
(a line-of-sight test to points 1 cm apart over them), the squares span at least 30 % of the image width, the board is
clear of its rack (5 mm or more from it) and of the table (2 cm or more), it has been held still for 0.5 s (its corners
moved under 1 mm), and OpenCV's findChessboardCorners finds all 40 inner corners in the board's image as the camera sees
it (drawn from the board's pose with those intrinsics). Nine targets, from the ROS tutorial's list, with the ROS
calibrator's own measures where it has them: on the left, right, top and bottom of the image (ROS X or Y measure 0.25 or
less / 0.75 or more), filling the view (ROS size measure 0.5 or more: the corner grid covers a quarter of the image),
and turned by 20 degrees or more so that its left, right, top or bottom edge is farther from the camera. One view can
meet several targets.

Score: 0 for the rest of the run once the board touches the table outside its rack (dropped, swung down or put down),
falls off the table, or touches the camera or its post. Otherwise 0.08 per target met (0.72 for all nine); 0.8 once all
nine are met and OpenCV's calibrateCamera, run on the corners found in the views that met them, recovers the focal
length within 2 %. Full score (1.0, success): that, with the board back in its rack (its middle within 1.5 cm and its
tilt within 10 degrees of the start), both grippers 5 cm or more from it, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps what happened over the run itself (views taken, a dropped board), so the score is read at the end.
SCORE_OVER = "end"

# -- the board ------------------------------------------------------------------------------------------------------
BOARD_W, BOARD_H, BOARD_T, BOARD_KG = 0.300, 0.160, 0.003, 0.18  # 3 mm aluminium composite, about 3.8 kg/m^2
SQ, NX, NY = 0.020, 9, 6  # squares: 9 x 6 of 20 mm (8 x 5 inner corners)
BORDER = 0.020  # white border round the squares, one square wide
PRINT_HW = NX * SQ / 2 + BORDER  # the printed area (squares and border): |u| <= 0.11, the full height
BAND = BOARD_W / 2 - PRINT_HW  # the plain grey band at each end (40 mm)

# -- the rack (board seated in it) ----------------------------------------------------------------------------------
RACK_LEAN = 60.0  # degrees back from upright
RACK_X = 0.29  # the board's bottom edge (middle of its thickness) in the rack
BASE_T = 0.008
BASE_X, BASE_HW = (0.135, 0.310), 0.13
LIP_H, LIP_T, LIP_GAP = 0.010, 0.008, 0.002
REST_HW, REST_V, REST_T = 0.09, (-0.05, 0.02), 0.020  # sloped rest: half width, extent up the board (from its middle)

# -- the camera -----------------------------------------------------------------------------------------------------
IMG_W, IMG_H = 1280, 720
FX = FY = 1100.0
CX, CY = 640.0, 360.0
CAM_POS = np.array([0.555, 0.0, TOP + 0.27])
CAM_AIM = np.array([0.28, 0.0, TOP + 0.11])
POST_X, POST_SIDE = 0.585, 0.015
CAM_BODY = (0.030, 0.090, 0.030)  # depth (along the view), width, height

WHITE, BLACK, GREY, BACK = (0.96, 0.96, 0.95), (0.04, 0.04, 0.05), (0.62, 0.64, 0.66), (0.80, 0.81, 0.82)
WOOD, WOOD_DARK, CAM_DARK, STEEL = (0.62, 0.45, 0.28), (0.45, 0.31, 0.18), (0.13, 0.13, 0.14), (0.55, 0.57, 0.60)


def board_axes(lean_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """The board's rotation (columns n, u, v: its face normal, its long axis to the robot's left, its up axis) when it
    leans back ``lean_deg`` from upright facing +x and is then turned ``yaw_deg`` about the vertical."""
    a, b = math.radians(lean_deg), math.radians(yaw_deg)
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    n = rz @ np.array([math.cos(a), 0.0, math.sin(a)])
    u = rz @ np.array([0.0, 1.0, 0.0])
    return np.column_stack([n, u, np.cross(n, u)])


def _quat(r: np.ndarray) -> tuple:
    from bodykit import _quat_from_matrix

    return tuple(float(v) for v in _quat_from_matrix(np.asarray(r, float)))


def _cam_frame() -> np.ndarray:
    """Columns: the camera's image right, image down and viewing direction (OpenCV's x, y, z) in the world."""
    z = CAM_AIM - CAM_POS
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


CAM_R = _cam_frame()

RACK_R = board_axes(RACK_LEAN)
SEAT_BOTTOM = np.array([RACK_X, 0.0, TOP + BASE_T + BOARD_T / 2 * RACK_R[2, 0]])  # the bottom edge's middle, seated
SEAT_C = SEAT_BOTTOM + BOARD_H / 2 * RACK_R[:, 2]  # the board's middle, seated


def _board_shapes() -> list:
    """The board in its own frame (x: out of the printed face, y: along its length, z: up its face): the plate, the
    grey back, the black squares, and the two grey bands at its ends (all printing is visual only)."""
    t = BOARD_T / 2
    out = [BoxShape((0.0, 0.0, 0.0), (BOARD_T, BOARD_W, BOARD_H), WHITE),
           BoxShape((-t - 0.0002, 0.0, 0.0), (0.0004, BOARD_W - 0.002, BOARD_H - 0.002), BACK, collide=False)]
    for i in range(NX):
        for j in range(NY):
            if (i + j) % 2 == 0:
                out.append(BoxShape((t + 0.0002, (i - (NX - 1) / 2) * SQ, (j - (NY - 1) / 2) * SQ),
                                    (0.0004, SQ, SQ), BLACK, collide=False))
    for s in (-1.0, 1.0):
        out.append(BoxShape((t + 0.0002, s * (PRINT_HW + BAND / 2), 0.0), (0.0004, BAND - 0.001, BOARD_H - 0.002),
                            GREY, collide=False))
    return out


def _rack_shapes() -> list:
    """The stand in a world-aligned frame at the middle of its base's top: base, lip, sloped rest and the block under
    the rest's upper end."""
    o = np.array([(BASE_X[0] + BASE_X[1]) / 2, 0.0, TOP + BASE_T])
    out = [BoxShape(tuple((np.array([0.0, 0.0, -BASE_T / 2])).tolist()), (BASE_X[1] - BASE_X[0], 2 * BASE_HW, BASE_T),
                    WOOD)]
    front = SEAT_BOTTOM[0] + BOARD_T / 2 * RACK_R[0, 0] + LIP_GAP  # the lip's back face
    out.append(BoxShape((front + LIP_T / 2 - o[0], 0.0, LIP_H / 2), (LIP_T, 2 * BASE_HW, LIP_H), WOOD_DARK))
    vmid = (REST_V[0] + REST_V[1]) / 2
    centre = SEAT_C + RACK_R @ np.array([-BOARD_T / 2 - REST_T / 2, 0.0, vmid])
    out.append(BoxShape(tuple((centre - o).tolist()), (REST_T, 2 * REST_HW, REST_V[1] - REST_V[0]), WOOD,
                        quat_xyzw=_quat(RACK_R)))
    top_end = SEAT_C + RACK_R @ np.array([-BOARD_T / 2 - REST_T, 0.0, REST_V[1]])  # the rest's upper back edge
    bx0, bx1 = top_end[0] - 0.012, top_end[0] + 0.016
    out.append(BoxShape(((bx0 + bx1) / 2 - o[0], 0.0, (0.045 - 0.008) / 2), (bx1 - bx0, 2 * REST_HW, 0.045 - 0.008),
                        WOOD_DARK))
    return out


RACK_O = np.array([(BASE_X[0] + BASE_X[1]) / 2, 0.0, TOP + BASE_T])


def _camera_shapes() -> list:
    """The webcam in its own frame (x: along its view, y: to its left, z: up), origin at the lens."""
    d, w, h = CAM_BODY
    return [BoxShape((-d / 2, 0.0, 0.0), (d, w, h), CAM_DARK),
            CylShape((0.001, 0.0, 0.0), 0.011, 0.002, BLACK, collide=False, axis="X"),
            CylShape((0.0022, 0.0, 0.0), 0.007, 0.001, (0.10, 0.16, 0.28), collide=False, axis="X"),
            BoxShape((0.0006, 0.030, 0.008), (0.001, 0.004, 0.003), (0.2, 0.8, 0.3), collide=False),
            BoxShape((-d - 0.004, 0.0, -h / 2 + 0.006), (0.008, 0.02, 0.012), STEEL)]


CAM_BODY_R = np.column_stack([CAM_R[:, 2], -CAM_R[:, 0], -CAM_R[:, 1]])  # view, left, up


def _post_shapes() -> tuple[list, float]:
    """The post in a world-aligned frame at its foot: a foot plate, a square post and a head under the camera."""
    back = CAM_POS + CAM_BODY_R @ np.array([-CAM_BODY[0] - 0.004, 0.0, -CAM_BODY[2] / 2])  # under the camera's mount
    top = float(back[2] - TOP) - 0.006
    return [BoxShape((0.0, 0.0, 0.005), (0.09, 0.09, 0.010), CAM_DARK),
            BoxShape((0.0, 0.0, top / 2), (POST_SIDE, POST_SIDE, top), STEEL),
            BoxShape((float(back[0] - POST_X), 0.0, top + 0.002), (0.03, 0.02, 0.012), CAM_DARK)], top


_POST_SHAPES, POST_TOP = _post_shapes()

# the board starts 1 mm off its rest and the base, as loose objects spawn
BOARD = Body.build("calibration_board", pos=tuple(float(v) for v in SEAT_C + 0.001 * RACK_R[:, 0] + [0, 0, 0.001]),
                   mass=BOARD_KG, color=WHITE, init_quat=_quat(RACK_R), friction=(0.8, 0.7), solver_iterations=32,
                   shapes=_board_shapes())
RACK = Body.build("board_rack", pos=tuple(float(v) for v in RACK_O), mass=1.0, color=WOOD, fixed=True,
                  friction=(0.6, 0.5), shapes=_rack_shapes())
CAMERA = Body.build("webcam", pos=tuple(float(v) for v in CAM_POS), mass=0.1, color=CAM_DARK, fixed=True,
                    init_quat=_quat(CAM_BODY_R), shapes=_camera_shapes())
POST = Body.build("camera_post", pos=(POST_X, 0.0, TOP), mass=1.0, color=STEEL, fixed=True, shapes=_POST_SHAPES)
OBJECTS = [RACK, POST, CAMERA, BOARD]

# -- geometry of what the camera sees -------------------------------------------------------------------------------
_IC = np.array([[0.0, (i - (NX - 2) / 2) * SQ, ((NY - 2) / 2 - j) * SQ] for j in range(NY - 1) for i in range(NX - 1)])
INNER = _IC + [BOARD_T / 2, 0.0, 0.0]  # the 40 inner corners on the printed face, row by row from the top left
_a, _b = NX * SQ / 2, NY * SQ / 2
OUTLINE = np.array([[BOARD_T / 2, -_a, _b], [BOARD_T / 2, _a, _b], [BOARD_T / 2, _a, -_b], [BOARD_T / 2, -_a, -_b]])
_us = np.linspace(-PRINT_HW, PRINT_HW, 23)
_vs = np.linspace(-BOARD_H / 2, BOARD_H / 2, 17)
PRINTED = np.array([[BOARD_T / 2, u, v] for u in _us for v in _vs])  # the squares and their border, every 1 cm
_pu = np.linspace(-BOARD_W / 2, BOARD_W / 2, 16)
_pv = np.linspace(-BOARD_H / 2, BOARD_H / 2, 9)
SHELL = np.array([[s * BOARD_T / 2, u, v] for s in (-1.0, 1.0) for u in _pu for v in _pv])  # both faces, every 2 cm
CORNERS8 = np.array([[sx * BOARD_T / 2, sy * BOARD_W / 2, sz * BOARD_H / 2] for sx in (-1, 1) for sy in (-1, 1)
                     for sz in (-1, 1)])

EDGE_PX = 8.0  # the ROS calibrator rejects a detection with a corner within 8 pixels of the image edge
MAX_FACING = 50.0  # degrees between the board's face and the line to the camera
MIN_WIDTH = 0.30  # the squares span at least this share of the image width
CLEAR_RACK, CLEAR_TABLE = 0.005, 0.02
STILL_STEPS, STILL_MM = 10, 1.0  # 0.5 s at 20 Hz
POS_LO, POS_HI = 0.25, 0.75  # ROS X / Y measure: left / right, top / bottom
CLOSE_SIZE = 0.5  # ROS size measure: sqrt(corner grid area / image area)
TURN_DEG = 20.0
F_TOL = 0.02
TARGETS = ("left", "right", "top", "bottom", "close", "left edge away", "right edge away", "top edge away",
           "bottom edge away")
TARGET_CREDIT, CAL_CREDIT = 0.08, 0.08
SEAT_POS, SEAT_DEG, CLEAR_GRIPPER = 0.015, 10.0, 0.05
TOUCH_TABLE, TOUCH_CAM = 0.003, 0.003


def project(points_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and depths of world points in the webcam."""
    pc = (np.asarray(points_w, float) - CAM_POS) @ CAM_R
    z = pc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([FX * pc[:, 0] / safe + CX, FY * pc[:, 1] / safe + CY], 1), z


def ros_measures(px: np.ndarray) -> tuple[float, float, float]:
    """The ROS calibrator's X, Y and size for the 40 inner corners in pixels (row by row): where the board sits across
    the image, shrunk by half its size, and the square root of its outside corners' area over the image's."""
    ul, ur, dr, dl = px[0], px[NX - 2], px[-1], px[-(NX - 1)]
    a, b, c = ur - ul, dr - ur, dl - dr
    p, q = b + c, a + b
    area = abs(p[0] * q[1] - p[1] * q[0]) / 2.0
    border = math.sqrt(area)
    x = min(1.0, max(0.0, (float(px[:, 0].mean()) - border / 2) / (IMG_W - border)))
    y = min(1.0, max(0.0, (float(px[:, 1].mean()) - border / 2) / (IMG_H - border)))
    return x, y, math.sqrt(area / (IMG_W * IMG_H))


def view_of(pos: np.ndarray, rot: np.ndarray) -> dict:
    """What the camera sees of the board at ``pos``/``rot`` (its middle; columns n, u, v), apart from the arms."""
    corners = pos + INNER @ rot.T
    px, depth = project(corners)
    opx, odepth = project(pos + OUTLINE @ rot.T)
    inside = bool((odepth > 0.05).all() and (opx[:, 0] >= EDGE_PX).all() and (opx[:, 0] <= IMG_W - EDGE_PX).all()
                  and (opx[:, 1] >= EDGE_PX).all() and (opx[:, 1] <= IMG_H - EDGE_PX).all())
    to_cam = CAM_POS - pos
    facing = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[:, 0] @ to_cam / np.linalg.norm(to_cam))))))
    nc = CAM_R.T @ rot[:, 0]
    turn = math.degrees(math.atan2(nc[0], -nc[2])) if nc[2] < 0 else 180.0
    tilt = math.degrees(math.atan2(nc[1], -nc[2])) if nc[2] < 0 else 180.0
    x, y, size = ros_measures(px) if inside else (0.5, 0.5, 0.0)
    width = float(opx[:, 0].max() - opx[:, 0].min()) / IMG_W
    return dict(px=px, inside=inside, facing=facing, turn=turn, tilt=tilt, X=x, Y=y, size=size, width=width)


def targets_met(v: dict) -> set:
    """The targets a valid view meets. turn > 0: the board's face points to the image's right, so its right edge is
    farther; tilt > 0: its face points down the image, so its bottom edge is farther."""
    out = set()
    if v["X"] <= POS_LO:
        out.add("left")
    if v["X"] >= POS_HI:
        out.add("right")
    if v["Y"] <= POS_LO:
        out.add("top")
    if v["Y"] >= POS_HI:
        out.add("bottom")
    if v["size"] >= CLOSE_SIZE:
        out.add("close")
    if v["turn"] <= -TURN_DEG:
        out.add("left edge away")
    if v["turn"] >= TURN_DEG:
        out.add("right edge away")
    if v["tilt"] <= -TURN_DEG:
        out.add("top edge away")
    if v["tilt"] >= TURN_DEG:
        out.add("bottom edge away")
    return out


# -- the arms as boxes (each in its link's frame), from the YAM's collision meshes -----------------------------------
LINK_BOXES = {
    "link2": ((-0.031, -0.0668, -0.0309), (0.031, -0.0001, 0.295)),
    "link3": ((-0.0912, -0.0799, -0.2735), (0.0315, 0.01, 0.0315)),
    "link4": ((-0.0982, -0.013, -0.1023), (0.0175, 0.073, 0.0175)),
    "link5": ((0.0, -0.0285, -0.0356), (0.069, 0.0285, 0.0335)),
    "gripper": ((-0.051, -0.0325, -0.0105), (0.051, 0.0325, 0.0635)),
}
# Each finger in the gripper frame (z: toward the tips; the TCP is at z 0.134), its inner face at x = q (the finger
# joint, <= 0) for the left finger and x = -q for the right: (z from, z to, thickness along x, half width along y).
FINGER_PARTS = ((0.131, 0.1468, 0.0064, 0.0137), (0.109, 0.131, 0.0100, 0.0201), (0.0635, 0.109, 0.0238, 0.034))


def arm_boxes(q_arm, q_fingers, base) -> list:
    """(centre, rotation, half size) of each box of one arm in the world, for its joint positions."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.urdf import forward_kinematics

    q = {n: float(v) for n, v in zip(ARM_JOINTS, q_arm)}
    q.update({n: float(v) for n, v in zip(FINGER_JOINTS, q_fingers)})
    fk = forward_kinematics(YAM_URDF, q)
    out = []
    b = np.asarray(base, float)
    for link, (lo, hi) in LINK_BOXES.items():
        t = fk[link]
        lo, hi = np.array(lo), np.array(hi)
        out.append((t[:3, :3] @ ((lo + hi) / 2) + t[:3, 3] + b, t[:3, :3], (hi - lo) / 2))
    g = fk["gripper"]
    for k, sign in ((0, -1.0), (1, 1.0)):
        qf = float(q_fingers[k])
        for z0, z1, th, hw in FINGER_PARTS:
            inner = qf if sign < 0 else -qf
            x0, x1 = (inner - th, inner) if sign < 0 else (inner, inner + th)
            c = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
            out.append((g[:3, :3] @ c + g[:3, 3] + b, g[:3, :3], np.array([(x1 - x0) / 2, hw, (z1 - z0) / 2])))
    return out


def blocked(points_w: np.ndarray, boxes: list) -> np.ndarray:
    """Which points are hidden from the camera by any box (slab test on the segment camera -> point)."""
    hit = np.zeros(len(points_w), bool)
    for c, r, h in boxes:
        a = r.T @ (CAM_POS - c)
        b = (np.asarray(points_w, float) - c) @ r
        d = b - a[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h[None] - a[None]) / d
            t2 = (h[None] - a[None]) / d
        lo = np.where(np.abs(d) < 1e-12, np.where(np.abs(a)[None] <= h[None], -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(np.abs(d) < 1e-12, np.where(np.abs(a)[None] <= h[None], np.inf, -np.inf), np.maximum(t1, t2))
        tlo, thi = lo.max(1), hi.min(1)
        hit |= (tlo <= thi) & (thi >= 0.0) & (tlo <= 1.0 - 1e-6)
    return hit


def _box_dist(points, c, r, h) -> np.ndarray:
    p = (np.asarray(points, float) - c) @ r
    return np.linalg.norm(np.maximum(np.abs(p) - h, 0.0), axis=1)


def _world_boxes(spec: Body) -> list:
    """(centre, rotation, half size) of a fixed body's colliding boxes in the world."""
    r0 = rotation(spec.start_quat())
    out = []
    for s in spec.shapes:
        if isinstance(s, BoxShape) and s.collide:
            centre = np.asarray(spec.pos) + r0 @ np.asarray(s.centre)
            out.append((centre, r0 @ rotation(s.quat()), np.array(s.size) / 2))
    return out


RACK_BOXES = _world_boxes(RACK)
CAM_BOXES = _world_boxes(CAMERA) + _world_boxes(POST)


SUPERSAMPLE = 4  # the frame is drawn 4 times finer and then averaged down, like a camera's pixels


def webcam_frame(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """The board as the webcam sees it (1280 x 720, 8-bit grey): its white face, grey ends and black squares drawn with
    the webcam's intrinsics over a mid-grey background (the arms are handled by the line-of-sight test)."""
    import cv2

    k = SUPERSAMPLE
    img = np.full((IMG_H * k, IMG_W * k), 110, np.uint8)
    t = BOARD_T / 2

    def fill(face_pts, value):
        pc = (pos + np.asarray(face_pts, float) @ rot.T - CAM_POS) @ CAM_R
        px = np.stack([FX * pc[:, 0] / pc[:, 2] + CX, FY * pc[:, 1] / pc[:, 2] + CY], 1)
        pts = np.round(((px + 0.5) * k - 0.5) * 256).astype(np.int32)
        cv2.fillPoly(img, [pts], int(value), lineType=cv2.LINE_AA, shift=8)

    def rect(u0, u1, v0, v1):
        return [[t, u0, v1], [t, u1, v1], [t, u1, v0], [t, u0, v0]]

    fill(rect(-BOARD_W / 2, BOARD_W / 2, -BOARD_H / 2, BOARD_H / 2), 240)
    for s in (-1.0, 1.0):
        u0, u1 = sorted((s * PRINT_HW, s * BOARD_W / 2))
        fill(rect(u0, u1, -BOARD_H / 2, BOARD_H / 2), 165)
    h = SQ / 2
    for i in range(NX):
        for j in range(NY):
            if (i + j) % 2 == 0:
                u, v = (i - (NX - 1) / 2) * SQ, (j - (NY - 1) / 2) * SQ
                fill(rect(u - h, u + h, v - h, v + h), 15)
    return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def find_corners(img: np.ndarray):
    """OpenCV's chessboard detector with the flags the ROS calibrator uses, refined to sub-pixel: the 40 inner corners
    (pixels) or None when it does not find them all."""
    import cv2

    ok, corners = cv2.findChessboardCorners(img, (NX - 1, NY - 1),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners[:, 0, :].astype(np.float32)


def calibrate(views: list) -> tuple[float, float]:
    """Relative focal-length errors (fx, fy) of OpenCV's calibrateCamera (default model, as in OpenCV's tutorial) on
    the corners found in the views."""
    import cv2

    obj = np.zeros((len(INNER), 3), np.float32)
    obj[:, 0] = _IC[:, 1]
    obj[:, 1] = -_IC[:, 2]
    _, k_est, _, _, _ = cv2.calibrateCamera([obj] * len(views), [np.asarray(v, np.float32) for v in views],
                                            (IMG_W, IMG_H), None, None)
    return float(k_est[0, 0] / FX - 1.0), float(k_est[1, 1] / FY - 1.0)


# -- tracking -------------------------------------------------------------------------------------------------------
_DEBUG = os.environ.get("CALIB_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("CALIB_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, failed="", hist=[], targets=set(), views=[], cal=None, view=None, valid=False,
               why="", hidden=0, seated=False, near=1.0, pos=SEAT_C.copy(), rot=RACK_R.copy(), low=0.0, tcp={},
               no_find=None)


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("CALIB_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _arms(base) -> dict:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if (side, "arm") not in _IDS:
            _IDS[(side, "arm")] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[(side, "fingers")] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        qa, qf = q[_IDS[(side, "arm")]], q[_IDS[(side, "fingers")]]
        tcp = np.asarray(ik.tcp_pose(qa, station.BASES[side])[0], float)
        out[side] = (qa, qf, tcp, station.BASES[side])
    return out


def board_gap(points, pos, rot) -> np.ndarray:
    """Distance from points to the board (a 300 x 160 x 3 mm plate)."""
    p = (np.asarray(points, float) - pos) @ rot
    h = np.array([BOARD_T / 2, BOARD_W / 2, BOARD_H / 2])
    return np.linalg.norm(np.maximum(np.abs(p) - h, 0.0), axis=1)


def _track_inner(env) -> None:
    """The board's pose, what the camera sees of it, which targets it meets, whether it touched the table or the
    camera, and whether it is back in its rack with the grippers clear."""
    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    bp = as_numpy(base.scene["calibration_board"].data.root_link_pose_w)[0]
    pos, rot = bp[:3] - origin, rotation(bp[3:7])
    arms = _arms(base)
    shell = pos + SHELL @ rot.T
    low = float((pos + CORNERS8 @ rot.T)[:, 2].min() - TOP)
    rack_gap = min(float(_box_dist(shell, c, r, h).min()) for c, r, h in RACK_BOXES)
    cam_gap = min(float(_box_dist(shell, c, r, h).min()) for c, r, h in CAM_BOXES)
    seated_pos = float(np.linalg.norm(pos - SEAT_C))
    seated_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(RACK_R.T @ rot) - 1.0) / 2.0))))
    in_rack = seated_pos < 0.04 and seated_deg < 20.0  # near its seat (for the table rule)
    if not _EP["failed"]:
        if low < TOUCH_TABLE and not in_rack:
            _EP["failed"] = f"the board touched the table outside its rack at step {_EP['tick']}"
        elif low < -0.05:
            _EP["failed"] = f"the board fell off the table at step {_EP['tick']}"
        elif cam_gap < TOUCH_CAM:
            _EP["failed"] = f"the board touched the camera or its post at step {_EP['tick']}"
    # what the camera sees
    view = view_of(pos, rot)
    hidden = 0
    if view["inside"]:
        boxes = [b for qa, qf, _, bs in arms.values() for b in arm_boxes(qa, qf, bs)] + RACK_BOXES
        hidden = int(blocked(pos + PRINTED @ rot.T, boxes).sum())
    outline = pos + OUTLINE @ rot.T
    _EP["hist"].append(outline)
    del _EP["hist"][:-STILL_STEPS]
    still = (len(_EP["hist"]) == STILL_STEPS
             and max(float(np.linalg.norm(h - outline, axis=1).max()) for h in _EP["hist"]) * 1000 < STILL_MM)
    why = ("outside the image" if not view["inside"] else
           "not facing the camera" if view["facing"] > MAX_FACING else
           f"{hidden} points of the print hidden" if hidden else
           "too far" if view["width"] < MIN_WIDTH else
           "touching the rack" if rack_gap < CLEAR_RACK else
           "too low" if low < CLEAR_TABLE else
           "moving" if not still else "")
    valid = not why and not _EP["failed"]
    if valid:
        new = targets_met(view) - _EP["targets"]
        tried = _EP["no_find"]
        if new and (tried is None or float(np.linalg.norm(tried - outline, axis=1).max()) > 0.002):
            found = find_corners(webcam_frame(pos, rot))
            if found is None:  # OpenCV does not find the board here: wait until it has moved before trying again
                _EP["no_find"] = outline.copy()
                new = set()
                why = "OpenCV found no checkerboard"
                valid = False
                if _DEBUG:
                    print(f"DBG step {_EP['tick']}: OpenCV found no checkerboard in the frame", flush=True)
            else:
                _EP["no_find"] = None
                _EP["views"].append(found)
        elif new:
            new = set()
        if new:
            _EP["targets"] |= new
            if _DEBUG:
                print(f"DBG view {len(_EP['views'])} at step {_EP['tick']}: {sorted(new)} X={view['X']:.2f} "
                      f"Y={view['Y']:.2f} size={view['size']:.2f} turn={view['turn']:.1f} tilt={view['tilt']:.1f}",
                      flush=True)
            if len(_EP["targets"]) == len(TARGETS) and _EP["cal"] is None:
                _EP["cal"] = calibrate(_EP["views"])
                if _DEBUG:
                    print(f"DBG calibration on {len(_EP['views'])} views: fx error {_EP['cal'][0] * 100:+.2f} %, "
                          f"fy error {_EP['cal'][1] * 100:+.2f} %", flush=True)
    near = min(float(board_gap(tcp[None], pos, rot)[0]) for _, _, tcp, _ in arms.values())
    seated = bool(seated_pos <= SEAT_POS and seated_deg <= SEAT_DEG)
    _EP.update(pos=pos, rot=rot, view=view, valid=valid, why=why, hidden=hidden, still=still, low=low,
               rack_gap=rack_gap, cam_gap=cam_gap, seated=seated, seated_pos=seated_pos, seated_deg=seated_deg,
               near=near, tcp={k: v[2] for k, v in arms.items()}, fingers={k: v[1] for k, v in arms.items()})


def _calibrated() -> bool:
    c = _EP["cal"]
    return c is not None and abs(c[0]) <= F_TOL and abs(c[1]) <= F_TOL


def _done() -> bool:
    return bool(not _EP["failed"] and _calibrated() and _EP["seated"] and _EP["near"] >= CLEAR_GRIPPER
                and _EP.get("still", False))


def _debug() -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p, v = _EP["pos"], _EP["view"]
    lean = math.degrees(math.atan2(_EP["rot"][2, 0], math.hypot(_EP["rot"][0, 0], _EP["rot"][1, 0])))
    if v is None:
        return
    cal = "-" if _EP["cal"] is None else f"{_EP['cal'][0] * 100:+.2f}%/{_EP['cal'][1] * 100:+.2f}%"
    print(f"DBG step={_EP['step']} board=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f}) lean={lean:.1f} low={_EP['low']:.3f} "
          f"X={v['X']:.2f} Y={v['Y']:.2f} size={v['size']:.2f} width={v['width']:.2f} turn={v['turn']:.1f} "
          f"tilt={v['tilt']:.1f} facing={v['facing']:.1f} hidden={_EP['hidden']} rack_gap={_EP.get('rack_gap', 0):.3f} "
          f"still={_EP.get('still')} valid={_EP['valid']} why={_EP['why'] or '-'} targets={len(_EP['targets'])} "
          f"cal={cal} seated={_EP['seated']} near={_EP['near']:.3f} failed={_EP['failed'] or '-'}", flush=True)
    if _EP["tcp"]:
        print("DBG   " + " | ".join(f"{k} tcp=({t[0]:.3f},{t[1]:.3f},{t[2] - TOP:.3f}) "
                                    f"fingers={_EP['fingers'][k][0]:.4f}" for k, t in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """0 once the board has touched the table outside its rack, fallen off the table or touched the camera or its post.
    Otherwise 0.08 per target met (0.72 for all nine), 0.8 once all nine are met and the calibration on those views
    recovers the focal length within 2 %. Full score (1.0) is success()."""
    if _DEBUG:
        _debug()
    if _EP["failed"]:
        return 0.0
    s = TARGET_CREDIT * len(_EP["targets"])
    if len(_EP["targets"]) == len(TARGETS) and _calibrated():
        s += CAL_CREDIT
    return round(s, 4)


def success(objects) -> bool:
    """All nine targets met and the calibration passed, the board back in its rack, both grippers 5 cm or more from
    it, still (held for 1 s by SUCCESS_HOLD_S)."""
    return _done()


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Both grippers take the board by its two top corners (each on a grey end band, pointing down the board's face,
# fingers closing through it), lift it out of the rack and raise it to face the camera; they then show it at the left,
# right, top and bottom of the camera's image, close up, and turned and tilted each way, holding each view still for
# about a second; then they lean it back into the rack, let go and move away.
# CALIB_ONE=1 runs it the wrong way: only the right gripper takes the board, by its right top corner, and plays the
# same moves.

ONE = os.environ.get("CALIB_ONE") == "1"
GRIP_U = BOARD_W / 2 - 0.013  # the TCP 13 mm in from each end...
GRIP_V = BOARD_H / 2 - 0.002  # ...and 2 mm below the top edge (the fingertips reach 15 mm down the face)
OPEN, CLOSED = -0.025, 0.0
NOMINAL_LEAN = 30.0
NOMINAL_C = np.array([0.279, 0.0, TOP + 0.1125])  # facing the camera, 32 cm from it
LIFT = 0.03


def grip_pose(arm: str, c, rot) -> tuple[np.ndarray, np.ndarray]:
    """The TCP pose that holds the board (middle c, rotation rot) by its top corner on the given side: pointing down
    the face, fingers closing through the board (the left gripper's x axis against the face normal)."""
    n, u, v = rot[:, 0], rot[:, 1], rot[:, 2]
    s = 1.0 if arm == "left" else -1.0
    z = -v
    x = -n if arm == "left" else n
    return np.asarray(c, float) + s * GRIP_U * u + GRIP_V * v, np.column_stack([x, np.cross(z, x), z])


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    for sd in itertools.product((-0.4, 0.0, 0.4), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2),
                                (-1.5, 0.0, 1.5)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=station.BASES[arm], iters=150)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
                if room > 0.3:
                    break
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    q0, q1 = np.array(READY, float), _good_ik(arm, pos, rot)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)


def _pose(dy=0.0, dv=0.0, closer=0.0, lean=NOMINAL_LEAN, yaw=0.0):
    """A board pose near the nominal one: moved dy sideways, dv up the face, ``closer`` toward the camera, then
    leaned and turned."""
    r0 = board_axes(NOMINAL_LEAN)
    toward = (CAM_POS - NOMINAL_C) / np.linalg.norm(CAM_POS - NOMINAL_C)
    return NOMINAL_C + np.array([0.0, dy, 0.0]) + dv * r0[:, 2] + closer * toward, board_axes(lean, yaw)


# the views, in order: (label, dy, dv, closer, lean, yaw)
VIEWS = (("on the image's left", -0.085, 0.0, 0.0, NOMINAL_LEAN, 0.0),
         ("on its right", 0.085, 0.0, 0.0, NOMINAL_LEAN, 0.0),
         ("at its top", 0.0, 0.031, 0.0, NOMINAL_LEAN, 0.0),
         ("at its bottom", 0.0, -0.027, 0.03, NOMINAL_LEAN, 0.0),
         ("close up", 0.0, 0.0, 0.09, NOMINAL_LEAN, 0.0),
         ("turned, its left edge away", 0.0, 0.0, 0.0, NOMINAL_LEAN, -30.0),
         ("turned, its right edge away", 0.0, 0.0, 0.0, NOMINAL_LEAN, 30.0),
         ("tilted, its top edge away", 0.0, 0.0, 0.0, NOMINAL_LEAN + 25.0, 0.0),
         ("tilted, its bottom edge away", 0.0, 0.0, 0.0, NOMINAL_LEAN - 25.0, 0.0))
MOVE_S, HOLD_S = 1.6, 1.3


def _slerp_rot(r0, r1, s):
    from yam_arena.ik import slerp_matrix

    return slerp_matrix(r0, r1, s)


def _board_plan() -> list:
    """(seconds, board middle, board rotation, fingers, label) segments after the grip, split so that no segment turns
    the board by more than 10 degrees (each gripper moves in a straight line within a segment)."""
    seat_up = SEAT_C + [0.0, 0.0, LIFT]
    plan = [(1.2, seat_up, RACK_R, CLOSED, "lift it off the rack")]
    c, r = _pose()
    plan.append((2.2, c, r, CLOSED, "raise it to face the camera"))
    for label, dy, dv, closer, lean, yaw in VIEWS:
        c, r = _pose(dy, dv, closer, lean, yaw)
        plan.append((MOVE_S, c, r, CLOSED, f"show it {label}"))
        plan.append((HOLD_S, c, r, CLOSED, f"hold it still {label}"))
    c, r = _pose()
    plan.append((1.4, c, r, CLOSED, "back to facing the camera"))
    plan.append((2.4, seat_up, RACK_R, CLOSED, "lean it back over the rack"))
    plan.append((1.2, SEAT_C + [0.0, 0.0, 0.001], RACK_R, CLOSED, "lower it into the rack"))
    plan.append((0.5, SEAT_C + [0.0, 0.0, 0.001], RACK_R, CLOSED, "settle"))
    plan.append((0.8, SEAT_C + [0.0, 0.0, 0.001], RACK_R, OPEN, "let go"))
    out, prev_c, prev_r = [], SEAT_C.copy(), RACK_R.copy()
    for t, c, r, finger, label in plan:
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(prev_r.T @ r) - 1.0) / 2.0))))
        k = max(1, int(math.ceil(ang / 10.0)))
        for i in range(1, k + 1):
            s = i / k
            out.append((t / k, prev_c + s * (np.asarray(c) - prev_c), _slerp_rot(prev_r, r, s), finger, label))
        prev_c, prev_r = np.asarray(c, float), r
    return out


def _arm_plan(arm: str) -> list:
    pre_c = SEAT_C + 0.06 * RACK_R[:, 2]  # the grips 6 cm up the face from their places, behind the top edge
    p0, r0 = grip_pose(arm, pre_c, RACK_R)
    out = _lead_in(arm, p0, r0, OPEN, "behind the board's top corner")
    p1, _ = grip_pose(arm, SEAT_C, RACK_R)
    out += [_w(1.2, p1, OPEN, "fingers either side of the top corner", r0),
            _w(0.4, p1, OPEN, "settle", r0),
            _w(1.0, p1, CLOSED, "take the corner", r0),
            _w(0.6, p1, CLOSED, "settle", r0)]
    for t, c, r, finger, label in _board_plan():
        p, rr = grip_pose(arm, c, r)
        out.append(_w(t, p, finger, label, rr))
    p_end, r_end = grip_pose(arm, SEAT_C, RACK_R)
    away = p_end + 0.07 * RACK_R[:, 2]
    out += [_w(1.0, away, OPEN, "draw back up the face", r_end),
            _w(1.2, away + [0.02, 0.0, 0.05], OPEN, "rise clear", r_end),
            _w(3.0, away + [0.02, 0.0, 0.05], OPEN, "stay clear", r_end)]
    return out


def demo():
    if ONE:
        from yam_arena import ik, station
        from yam_arena.constants import READY

        pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["left"])
        total = sum(w.duration for w in _arm_plan("right"))
        return {"left": [Waypoint(total, tuple(float(v) for v in pos), OPEN, "stay put", rot)],
                "right": _arm_plan("right")}
    return {"left": _arm_plan("left"), "right": _arm_plan("right")}
