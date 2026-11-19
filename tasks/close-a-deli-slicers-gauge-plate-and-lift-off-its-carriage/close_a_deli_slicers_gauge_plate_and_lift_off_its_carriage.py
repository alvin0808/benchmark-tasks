"""On the station table: a 12-inch (305 mm knife) manual deli slicer, switched off and unplugged for its cleaning, with
its product tray (the carriage tray the meat rides on) parked in the home position, pulled toward the operator (the
robot). Beside it on the bench, to the robot's right, a quarter-size sheet pan to carry the removed parts to the sink.

The procedure is the slicer maker's (Hobart HS7 instruction manual, "Cleaning": "Unplug the machine power cord and turn
the index knob fully clockwise to close the gauge plate before cleaning the slicer"; "Product Tray: to clean the
Product Tray, make sure the Gauge Plate is closed. Pull the Product Tray toward the operator until it stops. Loosen the
Knob on the Support Arm ... it can be removed by grasping with both hands and lifting straight up"; its Gauge Plate
Interlock "prevents the Product Tray from being tilted or removed unless the Product Tray is in the home position and
the Gauge Plate is closed"). Globe's G-series manual (G10/G12/G14) gives the same first step: "Turn slice thickness
dial clockwise past zero (0) until it stops so table covers knife edge."

World frame (m): x forward from the arm bases, y to the robot's left, z up; heights are above the table top (z 0.7475).
- Slicer base (fixed): x 0.09 to 0.66, y -0.03 to 0.40; on its left a pedestal (y 0.214 to 0.40, 12 cm tall) with a
  45-degree front slope, and behind the knife the knife housing (x 0.36 to 0.64, y 0.31 to 0.40, up to 38 cm).
- Knife: a 305 mm steel disc, 3 mm thick, its slicing face toward -y (the product side), leaning back 12 degrees (top
  toward +y) about a line along x through y 0.20, 10 cm up; upright, its hub would be at x 0.46, y 0.20, 20 cm up. A
  knife cover (centre plate, 230 mm across) covers its middle on the product side; a ring guard (r 150-165 mm) wraps its
  rim everywhere except the slicing arc, 165 to 275 degrees (in the knife's plane, from forward-horizontal toward up:
  the front-lower quarter, where the meat meets the edge), which stays bare all the time.
- Gauge plate: aluminium plate 220 x 140 x 8 mm (x 0.13 to 0.35, 10 to 24 cm up) in front of the knife in its plane.
  Its product face sits 3 mm proud of the knife's edge when closed, covering the edge at the front; at the start it is
  open to 9 mm. The index knob drives it: 12 degrees of knob per mm, closed by turning clockwise (seen from in front of
  the knob); the knob stops 6 degrees past zero (as on the Globe). Index knob: black fluted knob 55 mm across flats, 3
  cm tall, with a red pointer, on the pedestal's front slope (facing the robot and up), its foot centred at x 0.155, y
  0.30, 9 cm up; it turns against 0.10 N m. At the start it reads 9 mm (108 degrees open).
- Product tray (loose, 2.2 kg): aluminium tray floor 195 x 210 mm (x 0.14 to 0.335, y -0.02 to 0.19, top 10 cm up),
  a 12 mm front lip 3.5 cm tall (the operator's end, x 0.14 to 0.152) and a 12 mm side fence 5.5 cm tall along its
  right edge (y -0.02 to -0.008). Its support arm (x 0.19 to 0.31, y 0.05 to 0.11, 4.5 to 9 cm up) sits on the
  carriage: two 16 mm hinge pins, 1.2 cm tall, on the carriage (at x 0.205 and 0.295, y 0.08) stand in two 22 mm
  bores in the arm, so the tray only comes off by lifting it straight up 1.2 cm or more.
- Support-arm knob: black knob 40 mm across at x 0.095, y 0.09 on the carriage, 4.5 to 9.5 cm up; its clamp lug reaches
  over a tab at the front of the support arm and holds the tray down. Turned counter-clockwise (seen from above) 90
  degrees or more against 0.15 N m (it stops at 110), the lug clears the tab.
- Gauge plate interlock: a pawl on the carriage, under the tray, over a tab on the knife side of the support arm; it
  swings clear only while the gauge plate is closed (the index knob within 0.5 mm of zero) and swings back over the tab
  when it is opened again.
- Sheet pan (loose, 0.45 kg): aluminium quarter-size sheet pan, 23 x 33 cm outside with a 2 cm rim (x 0.13 to 0.36,
  y -0.37 to -0.04).

Score: 0 for the rest of the run once any part of either arm (any vertex of any collision mesh of any link, grippers
and finger pads included) has entered the bare edge band of the knife: within the slicing arc (165 to 275 degrees), 2
cm inside to 1.5 cm outside the edge's radius, from 3 cm in front of the knife's face to 1.2 cm behind it (the part
the gauge plate covers, x 0.35 or less, counts only while the plate is open). Otherwise the best stage reached, each
needing the ones before: 0.25 the gauge plate closed (index knob within 0.5 mm of zero); 0.5 also the support-arm knob
loosened 90 degrees or more; 0.75 also the tray off its hinge pins (its support arm lifted 1.5 cm or more, or moved
3 cm or more from its seat). Full score (1.0, success): the tray resting in the sheet pan (support arm's underside
within 8 mm of the pan's floor, all four of its corners inside the pan's rim, tilted 10 degrees or less), the gauge plate still
closed, both grippers 3 cm or more clear of the tray, the tray and the pan still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Link, Linkage, MeshShape, allow_matrix_grips, as_numpy,  # noqa: E402
                     on_reset, on_step, rotation)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

ALU, ALU2, STEEL, DARK, RED = (0.78, 0.79, 0.80), (0.68, 0.69, 0.71), (0.86, 0.87, 0.90), (0.12, 0.12, 0.13), \
    (0.80, 0.14, 0.12)
WHITE, PAN_C, BASE_C = (0.93, 0.93, 0.91), (0.72, 0.73, 0.74), (0.60, 0.61, 0.63)

# -- layout (world x, y; heights above the table) ---------------------------------------------------------------------
YK = 0.200  # the knife's slicing face (product side); the disc is 3 mm thick toward +y
KC = np.array([0.46, YK + 0.0015, 0.20])  # the knife's hub (x, y, height), before the lean
LEAN = 12.0  # the knife and the gauge plate lean back (top toward +y, away from the tray) about a line along x ...
LEAN_AT = np.array([0.0, YK, 0.10])  # ... through the gauge plate's foot
KR = 0.1525  # 305 mm (12 in) knife
EXPOSED = (165.0, 275.0)  # the bare slicing arc (degrees in the x-z plane, from +x toward +z)
PLATE_X = (0.13, 0.35)
PLATE_Z = (0.10, 0.24)
PLATE_T = 0.008
FACE_CLOSED = YK - 0.003  # the plate's product face when closed: 3 mm proud of the edge
OPEN0_MM = 9.0  # the gauge at the start
DEG_PER_MM = 12.0
PAST_ZERO_DEG = 6.0  # the knob stops this far past zero
OPEN_MAX_MM = 15.0
CLOSED_MM = 0.5  # the gauge counts as closed at this reading or less
PLATE_PIVOT_H = 0.48  # the plate's (hidden) swing arm: its pivot this far below the plate's middle
N_KNOB = np.array([-math.sqrt(0.5), 0.0, math.sqrt(0.5)])  # the index knob's axis (out of the base's front slope)
KNOB_FACE = np.array([0.155, 0.30, 0.09])  # the slope under the knob's middle
KNOB_AF, KNOB_H, KNOB_SKIRT = 0.055, 0.030, 0.003  # across flats, height along its axis, the dial skirt's thickness
INDEX_FRICTION = 0.10  # N m
# carriage, tray and its support arm
CARR_TOP = 0.045  # the carriage's top: the support arm sits on it
PIN_XS, PIN_Y, PIN_W, PIN_H, BORE_W = (0.205, 0.295), 0.08, 0.016, 0.012, 0.022
ARM_X, ARM_Y, ARM_Z1 = (0.19, 0.31), (0.05, 0.11), 0.09  # support arm (its underside at CARR_TOP)
FLOOR_X, FLOOR_Y, FLOOR_Z = (0.14, 0.335), (-0.02, 0.19), (0.09, 0.10)
LIP_X1, LIP_TOP = 0.152, 0.135  # front lip: x 0.14 to 0.152
FENCE_Y1, FENCE_TOP = -0.008, 0.155  # side fence: y -0.02 to -0.008
TAB_F = ((0.175, 0.19), (0.06, 0.10), 0.057)  # the support arm's front tab (x, y, top)
TAB_K = ((0.23, 0.27), (0.11, 0.135), 0.057)  # its knife-side tab, under the interlock pawl
TRAY_KG = 2.2
CLAMP_C, CLAMP_R, CLAMP_Z = (0.095, 0.09), 0.020, (0.045, 0.095)  # the support-arm knob
CLAMP_FREE_DEG, CLAMP_STOP_DEG, CLAMP_FRICTION = 90.0, 110.0, 0.15
PAWL_C, PAWL_FREE_DEG = (0.25, 0.165), 90.0
# the sheet pan
PAN_X, PAN_Y, PAN_RIM, PAN_FLOOR, PAN_WALL = (0.13, 0.36), (-0.37, -0.04), 0.020, 0.0015, 0.004
PAN_KG = 0.45

SLICER_O = np.array([0.38, 0.18, 0.0])  # the slicer linkage's origin (on the table)
TRAY_O = np.array([0.25, 0.08, CARR_TOP])  # the tray's body origin: under its support arm, on the carriage
PAN_O = np.array([(PAN_X[0] + PAN_X[1]) / 2, (PAN_Y[0] + PAN_Y[1]) / 2, 0.001])


def _bx(x0, x1, y0, y1, z0, z1, color, origin=SLICER_O, collide=True) -> BoxShape:
    o = np.asarray(origin, float)
    return BoxShape(((x0 + x1) / 2 - o[0], (y0 + y1) / 2 - o[1], (z0 + z1) / 2 - o[2]), (x1 - x0, y1 - y0, z1 - z0),
                    color, collide=collide)


def _quat(m) -> tuple:
    from bodykit import _quat_from_matrix

    return _quat_from_matrix(np.asarray(m, float))


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot(axis, a: float) -> np.ndarray:
    """Rotation by ``a`` (rad) about the unit vector ``axis``."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * kx + (1.0 - math.cos(a)) * kx @ kx


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


R_LEAN = _rx(math.radians(-LEAN))


def lean(p) -> np.ndarray:
    """A point of the upright knife/plate geometry, leaned back."""
    return LEAN_AT + R_LEAN @ (np.asarray(p, float) - LEAN_AT)


def _disc_y(centre, r, y0, y1, color, n=48, collide=True, origin=SLICER_O) -> list:
    """A leaned disc (upright: in an x-z plane, its axis along y from y0 to y1): an n-sided prism mesh."""
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([centre[0] + r * np.cos(ang), np.zeros(n), centre[2] + r * np.sin(ang)], axis=1)
    pts = np.concatenate([ring + [0.0, y0, 0.0], ring + [0.0, y1, 0.0],
                          [[centre[0], y0, centre[2]], [centre[0], y1, centre[2]]]])
    pts = LEAN_AT + (pts - LEAN_AT) @ R_LEAN.T - np.asarray(origin, float)
    a, b = 2 * n, 2 * n + 1
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces += [[k, k1, n + k1], [k, n + k1, n + k], [a, k1, k], [b, n + k, n + k1]]
    faces = np.array(faces, dtype=np.int64)
    # outward faces: flip any whose normal points toward the prism's middle
    mid = pts.mean(axis=0)
    for i, f in enumerate(faces):
        v = pts[f]
        nrm = np.cross(v[1] - v[0], v[2] - v[0])
        if np.dot(nrm, v.mean(axis=0) - mid) < 0:
            faces[i] = f[::-1]
    return [MeshShape(points=pts, faces=faces, normals=None, color=color, collide=collide,
                      approximation="convexHull")]


def _ring_y(centre, r0, r1, y0, y1, arcs, color, n=96, origin=SLICER_O) -> list:
    """Ring segments about an axis along y through ``centre`` (x-z plane), kept where their angle is in ``arcs``."""
    out = []
    w = 2.0 * r1 * math.tan(math.pi / n) * 1.05
    for k in range(n):
        a = 2 * math.pi * k / n
        deg = math.degrees(a) % 360.0
        if not any(lo <= deg <= hi for lo, hi in arcs):
            continue
        rc = (r0 + r1) / 2
        p = np.array([centre[0] + rc * math.cos(a), (y0 + y1) / 2, centre[2] + rc * math.sin(a)])
        m = R_LEAN @ _ry(-a)  # its x axis radial in the (leaned) knife plane
        out.append(BoxShape(tuple((lean(p) - origin).tolist()), (r1 - r0, y1 - y0, w), color, quat_xyzw=_quat(m)))
    return out


def _knob_frame() -> np.ndarray:
    """Columns: two directions across the index knob's face and its axis."""
    e1 = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    e2 = np.array([0.0, 1.0, 0.0])
    return np.column_stack([e1, e2, N_KNOB])


def _base_shapes() -> list:
    s = []
    # cast base: slab under the carriage, the pedestal on the knife side with its front slope
    s.append(_bx(0.09, 0.66, -0.03, 0.40, 0.0, 0.015, BASE_C))
    s.append(_bx(0.17, 0.66, 0.214, 0.40, 0.015, 0.12, BASE_C))
    slope = _knob_frame()
    c = KNOB_FACE - N_KNOB * 0.025 + slope[:, 0] * 0.0
    s.append(BoxShape(tuple((c + [0.0, 0.292 - KNOB_FACE[1], 0.0] - SLICER_O).tolist()), (0.11, 0.156, 0.05),
                      BASE_C, quat_xyzw=_quat(slope)))
    # dial marks on the slope around the knob (visual)
    for k in range(0, 9):
        a = math.radians(90.0 + k * 24.0)
        p = KNOB_FACE + 0.0005 * N_KNOB + 0.036 * (slope[:, 0] * math.cos(a) + slope[:, 1] * math.sin(a))
        s.append(BoxShape(tuple((p - SLICER_O).tolist()), (0.006, 0.002, 0.001), DARK if k else RED, collide=False,
                          quat_xyzw=_quat(slope @ _rz(a))))
    # knife housing and its top cover
    s.append(_bx(0.36, 0.64, 0.31, 0.40, 0.12, 0.38, BASE_C))
    s += _disc_y(KC, 0.06, YK + 0.003, YK + 0.09, ALU2, n=24)  # the hub boss behind the knife
    # carriage: a slab on the transport, its two hinge pins and the boss under the support-arm knob
    s.append(_bx(0.16, 0.32, 0.03, 0.13, 0.015, CARR_TOP, ALU2))
    s.append(_bx(0.075, 0.16, 0.07, 0.11, 0.015, CARR_TOP, ALU2))
    for px in PIN_XS:
        s.append(_bx(px - PIN_W / 2, px + PIN_W / 2, PIN_Y - PIN_W / 2, PIN_Y + PIN_W / 2, CARR_TOP,
                     CARR_TOP + PIN_H, STEEL))
    s.append(_bx(0.24, 0.26, 0.15, 0.18, 0.015, 0.059, DARK))  # the interlock pawl's post
    # knife, knife cover, ring guard
    s += _disc_y(KC, KR, YK, YK + 0.003, STEEL)
    s += _disc_y(KC, 0.115, YK - 0.012, YK, ALU)
    s += _disc_y(KC, 0.022, YK - 0.020, YK - 0.012, DARK, n=24)
    s += _ring_y(KC, 0.150, 0.165, YK - 0.006, YK + 0.026, ((0.0, EXPOSED[0]), (EXPOSED[1], 360.0)), ALU)
    return s


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _leaned_box(x0, x1, y0, y1, z0, z1, color) -> BoxShape:
    c = lean([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]) - SLICER_O
    return BoxShape(tuple(c.tolist()), (x1 - x0, y1 - y0, z1 - z0), color, quat_xyzw=_quat(R_LEAN))


def _plate_shapes() -> list:
    face = FACE_CLOSED + OPEN0_MM / 1000.0
    return [_leaned_box(PLATE_X[0], PLATE_X[1], face, face + PLATE_T, PLATE_Z[0], PLATE_Z[1], ALU),
            _leaned_box(PLATE_X[0] - 0.012, PLATE_X[0], face, face + 0.016, PLATE_Z[0], PLATE_Z[1] - 0.02, ALU2)]


def _index_shapes() -> list:
    f = _knob_frame()
    out = [BoxShape(tuple((KNOB_FACE + N_KNOB * KNOB_SKIRT / 2 - SLICER_O).tolist()), (0.07, 0.07, KNOB_SKIRT), DARK,
                    quat_xyzw=_quat(f @ _rz(math.radians(22.5))))]
    c = KNOB_FACE + N_KNOB * (KNOB_SKIRT + KNOB_H / 2)
    for k in range(4):  # fluted: four squares 22.5 degrees apart
        out.append(BoxShape(tuple((c - SLICER_O).tolist()), (KNOB_AF, KNOB_AF, KNOB_H), DARK,
                            quat_xyzw=_quat(f @ _rz(math.radians(22.5 * k)))))
    tip = KNOB_FACE + N_KNOB * (KNOB_SKIRT + KNOB_H + 0.0006) + f[:, 0] * 0.014
    out.append(BoxShape(tuple((tip - SLICER_O).tolist()), (0.026, 0.005, 0.001), RED, collide=False,
                        quat_xyzw=_quat(f)))
    return out


def _clamp_shapes() -> list:
    cx, cy = CLAMP_C
    o = SLICER_O
    return [CylShape((cx - o[0], cy - o[1], (CLAMP_Z[0] + CLAMP_Z[1]) / 2), CLAMP_R, CLAMP_Z[1] - CLAMP_Z[0], DARK),
            _bx(cx, 0.188, cy - 0.008, cy + 0.008, 0.059, 0.067, STEEL),
            _bx(cx - 0.003, cx + 0.003, cy - 0.016, cy + 0.016, CLAMP_Z[1], CLAMP_Z[1] + 0.001, RED, collide=False)]


def _pawl_shapes() -> list:
    px, py = PAWL_C
    return [_bx(px - 0.007, px + 0.007, 0.118, py + 0.007, 0.059, 0.067, RED)]


SLICER = Linkage.build(
    "slicer", pos=tuple((SLICER_O + [0.0, 0.0, TOP]).tolist()), mass=25.0, shapes=_base_shapes(),
    links=[
        Link("gauge_plate", tuple(_plate_shapes()), 0.6,
             anchor=tuple((lean([0.25, YK + 0.01, (PLATE_Z[0] + PLATE_Z[1]) / 2 - PLATE_PIVOT_H]) + [0, 0, TOP]).tolist()),
             axis="X",
             stiffness=3000.0, spring_damping=60.0),
        Link("index_knob", tuple(_index_shapes()), 0.08, anchor=tuple((KNOB_FACE + [0.0, 0.0, TOP]).tolist()),
             axis=tuple(N_KNOB.tolist()), friction=INDEX_FRICTION, damping=0.01,
             limits=(-(OPEN0_MM * DEG_PER_MM + PAST_ZERO_DEG), (OPEN_MAX_MM - OPEN0_MM) * DEG_PER_MM)),
        Link("clamp_knob", tuple(_clamp_shapes()), 0.06, anchor=(CLAMP_C[0], CLAMP_C[1], TOP + CARR_TOP), axis="Z",
             friction=CLAMP_FRICTION, damping=0.01, limits=(0.0, CLAMP_STOP_DEG)),
        Link("pawl", tuple(_pawl_shapes()), 0.05, anchor=(PAWL_C[0], PAWL_C[1], TOP + 0.063), axis="Z",
             stiffness=60.0, spring_damping=2.0, limits=(-1.0, PAWL_FREE_DEG + 1.0)),
    ],
    friction=(0.6, 0.5), solver_iterations=32)


def _tray_shapes() -> list:
    o = TRAY_O
    s = [_bx(FLOOR_X[0], FLOOR_X[1], FLOOR_Y[0], FLOOR_Y[1], FLOOR_Z[0], FLOOR_Z[1], ALU, origin=o),
         _bx(FLOOR_X[0], LIP_X1, FLOOR_Y[0], FLOOR_Y[1], FLOOR_Z[1], LIP_TOP, ALU, origin=o),
         _bx(FLOOR_X[0], FLOOR_X[1], FLOOR_Y[0], FENCE_Y1, FLOOR_Z[1], FENCE_TOP, ALU, origin=o)]
    # the support arm, with a bore over each hinge pin (built round the bores)
    x0, x1 = ARM_X
    y0, y1 = ARM_Y
    z0, z1 = CARR_TOP, ARM_Z1
    b = BORE_W / 2
    s.append(_bx(x0, x1, y0, y1, z0 + PIN_H + 0.004, z1, ALU2, origin=o))  # above the bores
    xs = [x0]
    for px in PIN_XS:
        xs += [px - b, px + b]
    xs.append(x1)
    for k in range(0, len(xs), 2):  # solid between the bores
        s.append(_bx(xs[k], xs[k + 1], y0, y1, z0, z0 + PIN_H + 0.004, ALU2, origin=o))
    for px in PIN_XS:  # either side of each bore
        s.append(_bx(px - b, px + b, y0, PIN_Y - b, z0, z0 + PIN_H + 0.004, ALU2, origin=o))
        s.append(_bx(px - b, px + b, PIN_Y + b, y1, z0, z0 + PIN_H + 0.004, ALU2, origin=o))
    for (tx, ty, tz) in (TAB_F, TAB_K):
        s.append(_bx(tx[0], tx[1], ty[0], ty[1], CARR_TOP, tz, ALU2, origin=o))
    return s


TRAY = Body.build("product_tray", pos=tuple((TRAY_O + [0.0, 0.0, TOP]).tolist()), mass=TRAY_KG,
                  shapes=_tray_shapes(), friction=(0.7, 0.6), solver_iterations=24)


def _pan_shapes() -> list:
    o = PAN_O
    x0, x1 = PAN_X
    y0, y1 = PAN_Y
    z = 0.001
    w = PAN_WALL
    return [_bx(x0, x1, y0, y1, z, z + PAN_FLOOR, PAN_C, origin=o),
            _bx(x0, x0 + w, y0, y1, z, z + PAN_RIM, PAN_C, origin=o),
            _bx(x1 - w, x1, y0, y1, z, z + PAN_RIM, PAN_C, origin=o),
            _bx(x0, x1, y0, y0 + w, z, z + PAN_RIM, PAN_C, origin=o),
            _bx(x0, x1, y1 - w, y1, z, z + PAN_RIM, PAN_C, origin=o)]


PAN = Body.build("sheet_pan", pos=tuple((PAN_O + [0.0, 0.0, TOP]).tolist()), mass=PAN_KG, shapes=_pan_shapes(),
                 friction=(0.5, 0.4))

OBJECTS = [SLICER, TRAY, PAN]

# -- tracking ---------------------------------------------------------------------------------------------------------
BAND_IN, BAND_OUT, BAND_FRONT, BAND_BACK = 0.020, 0.015, 0.030, 0.012
LIFT_OFF, MOVED_OFF = 0.015, 0.03
PAN_GAP, PAN_TILT = 0.008, 10.0
CLEAR = 0.03
STILL_STEPS, STILL_MM, STILL_DEG = 20, 1.0, 0.5
_DEBUG = os.environ.get("SLICER_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("SLICER_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], tcp={}, gauge=OPEN0_MM, clamp=0.0,
               pawl=0.0, plate_face=FACE_CLOSED + OPEN0_MM / 1000, tray=TRAY_O.copy(), tray_tilt=0.0, lifted=False,
               off=False, in_pan=False, pan=PAN_O.copy(), near=1.0, band_pts=0, tray_q=np.array([0, 0, 0, 1.0]),
               pan_gap=1.0, pan_inside=False)


_reset()


def _arm_points() -> dict:
    """Every link's collision-mesh vertices (link frame) from the YAM URDF."""
    import struct
    import xml.etree.ElementTree as ET

    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = pathlib.Path(YAM_URDF).parent
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            mesh = col.find("geometry").find("mesh")
            if mesh is None:
                continue
            b = (root / mesh.get("filename")).read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            tri = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            v = tri["v"].reshape(-1, 3).astype(float)
            pts.append(np.unique(np.round(v, 3), axis=0) @ rpy_matrix(rpy).T + xyz)
        if pts:
            out[link.get("name")] = np.concatenate(pts)
    return out


def in_band(pts: np.ndarray, gauge_closed: bool) -> np.ndarray:
    """Which points (world x, y, height) lie in the knife's bare edge band."""
    pts = LEAN_AT + (np.asarray(pts, float) - LEAN_AT) @ R_LEAN  # into the upright knife's frame
    u = pts[:, 0] - KC[0]
    v = pts[:, 2] - KC[2]
    r = np.hypot(u, v)
    ang = np.degrees(np.arctan2(v, u)) % 360.0
    side = YK - pts[:, 1]  # in front of the slicing face (product side) is positive
    hit = ((ang >= EXPOSED[0]) & (ang <= EXPOSED[1]) & (r >= KR - BAND_IN) & (r <= KR + BAND_OUT)
           & (side <= BAND_FRONT) & (side >= -BAND_BACK))
    if gauge_closed:
        hit &= pts[:, 0] > PLATE_X[1]
    return hit


def _tray_local_box():
    lo, hi = TRAY.local_bounds()
    return lo, hi


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("SLICER_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    import torch

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "mesh" not in _IDS:
        _IDS["mesh"] = _arm_points()
    sl = base.scene["slicer"]
    if "joints" not in _IDS:
        jn = list(sl.joint_names)
        _IDS["joints"] = {k: jn.index(f"{k}_turn") for k in ("gauge_plate", "index_knob", "clamp_knob", "pawl")}
    q = as_numpy(sl.data.joint_pos)[0]
    ji = _IDS["joints"]
    knob = math.degrees(float(q[ji["index_knob"]]))
    gauge = (knob + OPEN0_MM * DEG_PER_MM) / DEG_PER_MM  # mm (negative past zero)
    closed = gauge <= CLOSED_MM
    clamp = math.degrees(float(q[ji["clamp_knob"]]))
    pawl = math.degrees(float(q[ji["pawl"]]))
    # the index knob drives the gauge plate (its screw is not modelled) and the plate's cam drives the interlock pawl
    opening = min(OPEN_MAX_MM, max(0.0, gauge)) / 1000.0
    tgt = torch.tensor([[(OPEN0_MM / 1000.0 - opening) / PLATE_PIVOT_H, math.radians(PAWL_FREE_DEG if closed else 0.0)]],
                       dtype=torch.float32, device=base.device)
    sl.set_joint_position_target(tgt, joint_ids=[ji["gauge_plate"], ji["pawl"]])
    plate_face = FACE_CLOSED + OPEN0_MM / 1000.0 - PLATE_PIVOT_H * math.sin(float(q[ji["gauge_plate"]]))

    pts_all = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            names = list(robot.body_names)
            _IDS[side + "_links"] = [(n, names.index(n)) for n in _IDS["mesh"] if n in names]
        qa = as_numpy(robot.data.joint_pos)[0]
        _EP["tcp"][side] = np.asarray(ik.tcp_pose(qa[_IDS[side]], station.BASES[side])[0], float) - [0, 0, TOP]
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        for n, i in _IDS[side + "_links"]:
            pts_all.append(_IDS["mesh"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin)
    pts = np.concatenate(pts_all) - [0.0, 0.0, TOP]
    hits = in_band(pts, closed)

    tp = as_numpy(base.scene["product_tray"].data.root_link_pose_w)[0]
    tray = tp[:3] - origin - [0.0, 0.0, TOP]
    tr = rotation(tp[3:7])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(tr[2, 2])))))
    pp = as_numpy(base.scene["sheet_pan"].data.root_link_pose_w)[0]
    pan = pp[:3] - origin - [0.0, 0.0, TOP]
    prr = rotation(pp[3:7])
    lifted = float(tray[2] - TRAY_O[2]) >= LIFT_OFF
    moved = float(np.hypot(tray[0] - TRAY_O[0], tray[1] - TRAY_O[1])) >= MOVED_OFF
    # the support arm's underside corners, in the pan's frame
    corners = np.array([[x, y, 0.0] for x in (ARM_X[0] - TRAY_O[0], ARM_X[1] - TRAY_O[0])
                        for y in (ARM_Y[0] - TRAY_O[1], ARM_Y[1] - TRAY_O[1])])
    cw = tray + corners @ tr.T
    cp = (cw - pan) @ prr
    floor_top = PAN_FLOOR  # the pan's floor top over the pan's origin (its origin 1 mm over the table)
    pan_gap = float(cp[:, 2].max() - floor_top)
    lo_t, hi_t = _tray_local_box()
    hx = (PAN_X[1] - PAN_X[0]) / 2 - PAN_WALL
    hy = (PAN_Y[1] - PAN_Y[0]) / 2 - PAN_WALL
    inside = bool(np.all(np.abs(cp[:, 0]) <= hx) and np.all(np.abs(cp[:, 1]) <= hy))  # the support arm's footprint
    in_pan = inside and -0.003 <= pan_gap <= PAN_GAP and tilt <= PAN_TILT
    # grippers' clearance from the tray (its bounding box, in the tray's frame)
    near = 1.0
    for v in _EP["tcp"].values():
        loc = tr.T @ (v - tray)
        d = np.maximum(np.maximum(lo_t - loc, loc - hi_t), 0.0)
        near = min(near, float(np.linalg.norm(d)))

    if not _EP["failed"]:
        if hits.any():
            p = pts[hits][0]
            _EP["failed"] = f"an arm entered the knife's bare edge band at ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
        elif tray[2] < -0.05:
            _EP["failed"] = "the tray fell off the bench"
    stage = 0.0
    if closed:
        stage = 0.25
        if clamp >= CLAMP_FREE_DEG:
            stage = 0.5
            if lifted or moved:
                stage = 0.75
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)
    _EP.update(gauge=gauge, closed=closed, clamp=clamp, pawl=pawl, plate_face=plate_face, tray=tray, tray_tilt=tilt,
               tray_q=tp[3:7], lifted=lifted, off=lifted or moved, in_pan=in_pan, pan=pan, pan_gap=pan_gap,
               pan_inside=inside, near=near, band_pts=int(hits.sum()))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.r_[_EP["tray"], _EP["pan"], _EP["tray_q"]])
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(
        np.abs(p[:6] - recent[-1][:6]).max() * 1000 < STILL_MM
        and abs(float(np.dot(p[6:10], recent[-1][6:10]))) > math.cos(math.radians(STILL_DEG / 2)) for p in recent)
    done = bool(_EP.get("closed") and _EP["clamp"] >= CLAMP_FREE_DEG and _EP["in_pan"] and _EP["near"] >= CLEAR
                and still)
    return dict(still=still, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    tr = _EP["tray"]
    print(f"DBG step={_EP['step']} gauge={_EP['gauge']:+.2f}mm face={(_EP['plate_face'] - YK) * 1000:+.1f}mm "
          f"clamp={_EP['clamp']:.1f} pawl={_EP['pawl']:.1f} tray=({tr[0]:.3f},{tr[1]:.3f},{tr[2]:.4f}) "
          f"tilt={_EP['tray_tilt']:.1f} off={_EP['off']} pan_gap={_EP['pan_gap'] * 1000:.1f}mm "
          f"inside={_EP['pan_inside']} in_pan={_EP['in_pan']} near={_EP['near']:.3f} band={_EP['band_pts']} "
          f"still={t['still']} best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_EP["tcp"]) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})" for k, v in _EP["tcp"].items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once an arm has entered the knife's bare edge band or the
    tray fell off the bench): 0.25 the gauge plate closed; 0.5 also the support-arm knob loosened 90 degrees; 0.75 also
    the tray off its hinge pins. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if _EP["failed"]:
        return 0.0
    return 1.0 if t["done"] else _EP["best"]


def success(objects) -> bool:
    """The tray resting in the sheet pan, the gauge plate closed, the support-arm knob loose, both grippers 3 cm or
    more clear of the tray, the tray and the pan still for 1 s, and no arm ever in the knife's bare edge band."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm: takes the index knob on the base's front slope (gripper pointing along the knob's axis), turns it
# clockwise in two strokes until it stops past zero (the gauge plate closes over the edge and the interlock pawl swings
# clear), lets go and takes the tray's front lip. Right arm, meanwhile: takes the support-arm knob from above and turns
# it counter-clockwise 110 degrees in two strokes (its lug clears the support arm's tab), lets go and takes the tray's
# side fence. Both lift the tray straight up off its hinge pins, carry it to the right and lower it into the sheet pan,
# let go and move away.
# SLICER_WRONG=1: the gauge plate is left open (the left arm goes straight to the tray): the interlock pawl stays over
# the support arm and the tray does not come off. SLICER_ONE=1: the right arm alone lifts the tray by its fence.

WRONG = os.environ.get("SLICER_WRONG") == "1"
BACK = os.environ.get("SLICER_BACK") == "1"
ONE_ARM = os.environ.get("SLICER_ONE") == "1"
OPEN, CLOSED = -0.040, 0.0
KNOB_PINCH = -(KNOB_AF / 2 - 0.004)
KNOB_OPEN = -0.045
CLAMP_PINCH = -(CLAMP_R - 0.004)
CLAMP_OPEN = -0.036
LIP_PINCH = -0.0005  # 12 mm lip / fence: each finger 5.5 mm inside its faces (a firm pinch)
LIP_OPEN = -0.030
T45 = _ry(math.radians(-45.0)) @ grip_rotation(0.0)  # pointing along -N_KNOB (forward and down, 45 degrees)
KNOB_GRIP_D = KNOB_SKIRT + KNOB_H - 0.011  # the TCP this far out along the knob's axis
STROKE = 64.0  # degrees per stroke on the index knob
CLAMP_STROKE = 60.0
LIP_GRIP = np.array([(FLOOR_X[0] + LIP_X1) / 2, 0.150, 0.123])
if BACK:  # the wrong way: the tray taken by its back end, beside the knife (a finger goes behind it, at the bare edge)
    LIP_GRIP = np.array([FLOOR_X[1] - 0.002, 0.160, 0.100])
FENCE_GRIP = np.array([0.235, (FLOOR_Y[0] + FENCE_Y1) / 2, 0.140])
LIFT = 0.07  # commanded: each loaded arm sags 2-3 cm below its target
CARRY_LIFT = 0.05
TRANSIT = (np.array([0.075, 0.27, 0.15]), np.array([0.075, 0.15, 0.15]))  # the left gripper's way round the plate
CARRY = np.array([0.012, -0.24, 0.0])  # aimed 12 mm forward: the loaded arms lag behind
SET_DOWN = (PAN_FLOOR + 0.001 + 0.002) - CARR_TOP  # the tray's drop from home to 2 mm over the pan's floor


def _knob_rot(deg: float) -> np.ndarray:
    return _rot(N_KNOB, math.radians(deg)) @ T45


def _down(yaw_deg: float) -> np.ndarray:
    return grip_rotation(math.radians(yaw_deg))


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


def _joint_move(arm: str, q0, pos, rot, finger: float, label: str, seconds: float, steps: int = 15) -> list:
    from yam_arena import ik, station

    q1 = _good_ik(arm, pos, rot)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
    from yam_arena.constants import READY

    return _joint_move(arm, np.array(READY, float), pos, rot, finger, label, seconds, steps)


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in np.asarray(p, float) + [0.0, 0.0, TOP]), finger, label, rot)


def _dur(plan) -> float:
    return sum(w.duration for w in plan)


def _left_knob() -> list:
    grip = KNOB_FACE + N_KNOB * KNOB_GRIP_D
    out = _lead_in("left", np.array([0.10, 0.32, 0.19 + TOP]), _knob_rot(STROKE / 2), KNOB_OPEN,
                   "back and down in front of the slicer's base")
    out.append(_w(1.2, grip + N_KNOB * 0.06, KNOB_OPEN, "in front of the index knob, along its axis",
                  _knob_rot(STROKE / 2)))
    for k in range(2):
        a0, a1 = STROKE / 2, -STROKE / 2
        out += [_w(1.0 if k == 0 else 0.6, grip + N_KNOB * (0.06 if k == 0 else 0.015), KNOB_OPEN, "line up",
                   _knob_rot(a0)),
                _w(1.0, grip, KNOB_OPEN, "onto the knob", _knob_rot(a0)),
                _w(0.7, grip, KNOB_PINCH, "take the index knob", _knob_rot(a0)),
                _w(0.4, grip, KNOB_PINCH, "wait for the fingers", _knob_rot(a0))]
        for j in range(1, 7):
            out.append(_w(0.45, grip, KNOB_PINCH, "turn it clockwise (gauge plate closing)",
                          _knob_rot(a0 + (a1 - a0 - (6.0 if k else 0.0)) * j / 6)))
        out += [_w(0.4, grip, KNOB_PINCH, "hold", _knob_rot(a1 - (6.0 if k else 0.0))),
                _w(0.5, grip, KNOB_OPEN, "let go", _knob_rot(a1 - (6.0 if k else 0.0))),
                _w(0.6, grip + N_KNOB * 0.015, KNOB_OPEN, "back off", _knob_rot(a1 - (6.0 if k else 0.0)))]
    out.append(_w(0.8, grip + N_KNOB * 0.07, KNOB_OPEN, "back off the knob", _knob_rot(-STROKE / 2)))
    return out


def _left(t_fence: float) -> list:
    if ONE_ARM:
        out = _left_knob()
        aside = np.array([0.14, 0.32, 0.20])
        out.append(_w(1.5, aside, KNOB_OPEN, "move aside", T45))
        out.append(_w(40.0, aside, KNOB_OPEN, "stay put", T45))
        return out
    lip_rot = _down(-90.0)
    if WRONG:
        out = _lead_in("left", LIP_GRIP + [0.0, 0.0, 0.08 + TOP], lip_rot, LIP_OPEN, "over the tray's front lip")
    else:
        out = _left_knob()
        # round the front of the gauge plate (not over it): in front of the base, then over the tray
        q_now = _good_ik("left", KNOB_FACE + N_KNOB * (KNOB_GRIP_D + 0.07) + [0.0, 0.0, TOP], _knob_rot(-STROKE / 2))
        out += _joint_move("left", q_now, TRANSIT[0] + [0.0, 0.0, TOP], lip_rot, CLOSED,
                           "down in front of the base", 2.0, steps=10)
        out += [_w(1.6, TRANSIT[1], CLOSED, "past the front of the gauge plate", lip_rot),
                _w(1.4, LIP_GRIP + [0.0, 0.0, 0.08], LIP_OPEN, "over the tray's front lip", lip_rot)]
    t = _dur(out)
    out += [_w(max(0.3, t_fence - 2.2 - t), LIP_GRIP + [0.0, 0.0, 0.08], LIP_OPEN, "wait", lip_rot),
            _w(1.4, LIP_GRIP, LIP_OPEN, "fingers either side of the front lip", lip_rot),
            _w(0.8, LIP_GRIP, LIP_PINCH, "take the front lip", lip_rot)]
    return out + _carry("left", LIP_GRIP, lip_rot, max(0.0, t_fence + 0.8 - _dur(out)))


def _right_clamp() -> list:
    grip = np.array([CLAMP_C[0], CLAMP_C[1], CLAMP_Z[1] - 0.010])
    out = _lead_in("right", grip + [0.0, 0.0, 0.08 + TOP], _down(-30.0), CLAMP_OPEN, "over the support-arm knob")
    for k in range(2):
        a0 = -30.0
        out += [_w(1.2 if k == 0 else 0.5, grip + [0.0, 0.0, 0.0 if k else 0.0], CLAMP_OPEN, "onto the knob",
                   _down(a0)),
                _w(0.6, grip, CLAMP_PINCH, "take the support-arm knob", _down(a0)),
                _w(0.4, grip, CLAMP_PINCH, "wait for the fingers", _down(a0))]
        for j in range(1, 7):
            out.append(_w(0.4, grip, CLAMP_PINCH, "turn it counter-clockwise (loosen)",
                          _down(a0 + CLAMP_STROKE * j / 6)))
        out += [_w(0.5, grip, CLAMP_OPEN, "let go", _down(a0 + CLAMP_STROKE)),
                _w(0.8 if k == 0 else 0.4, grip, CLAMP_OPEN, "turn back", _down(a0))]
    out.append(_w(0.8, grip + [0.0, 0.0, 0.07], CLAMP_OPEN, "rise", _down(-30.0)))
    return out


def _right(t_left_knob: float) -> tuple[list, float]:
    fence_rot = _down(0.0)
    out = _right_clamp()
    out += [_w(2.0, FENCE_GRIP + [0.0, 0.0, 0.035], LIP_OPEN, "over the tray's side fence", fence_rot)]
    t = _dur(out)
    t_fence = max(t, t_left_knob + 4.5) + 1.4
    out += [_w(max(0.3, t_fence - 1.4 - t), FENCE_GRIP + [0.0, 0.0, 0.035], LIP_OPEN, "wait", fence_rot),
            _w(1.4, FENCE_GRIP, LIP_OPEN, "fingers either side of the fence", fence_rot),
            _w(0.8, FENCE_GRIP, LIP_PINCH, "take the fence", fence_rot)]
    return out, t_fence


def _carry(arm: str, grip, rot, wait: float) -> list:
    finger = LIP_PINCH
    up = grip + [0.0, -0.05 if arm == "right" else -0.03, LIFT]
    out = [_w(0.5 + wait, grip, finger, "wait for the fingers", rot),
           _w(3.0, up, finger, "lift the tray straight up off its hinge pins", rot),
           _w(0.8, up, finger, "settle", rot),
           _w(3.0, grip + CARRY + [0.0, 0.0, CARRY_LIFT], finger, "carry it over the sheet pan", rot),
           _w(0.6, grip + CARRY + [0.0, 0.0, CARRY_LIFT], finger, "settle", rot),
           _w(2.2, grip + CARRY + [0.0, 0.0, SET_DOWN + 0.004], finger, "lower it into the pan", rot),
           _w(0.8, grip + CARRY + [0.0, 0.0, SET_DOWN - 0.004], finger, "set it down", rot),
           _w(0.6, grip + CARRY + [0.0, 0.0, SET_DOWN - 0.004], LIP_OPEN, "let go", rot),
           _w(1.2, grip + CARRY + [0.0, 0.0, SET_DOWN + 0.08], LIP_OPEN, "rise", rot),
           _w(4.0, grip + CARRY + [0.0, 0.0, SET_DOWN + 0.08], LIP_OPEN, "stay clear", rot)]
    return out


def demo():
    t_left = 0.0 if WRONG else _dur(_left_knob())
    if ONE_ARM:
        right, t_fence = _right(0.0)
        right += _carry("right", FENCE_GRIP, _down(0.0), 0.0)
        return {"left": _left(t_fence), "right": right}
    right, t_fence = _right(t_left)
    left = _left(t_fence)
    right += _carry("right", FENCE_GRIP, _down(0.0), 0.0)
    # line the two carries up: both arms start lifting together
    lift_l = next(i for i, w in enumerate(left) if w.label.startswith("lift the tray"))
    lift_r = next(i for i, w in enumerate(right) if w.label.startswith("lift the tray"))
    tl, tr = _dur(left[:lift_l]), _dur(right[:lift_r])
    if tl > tr:
        w = right[lift_r - 1]
        right[lift_r - 1] = Waypoint(w.duration + tl - tr, w.pos, w.finger, w.label, w.yaw)
    elif tr > tl:
        w = left[lift_l - 1]
        left[lift_l - 1] = Waypoint(w.duration + tr - tl, w.pos, w.finger, w.label, w.yaw)
    return {"left": left, "right": right}
