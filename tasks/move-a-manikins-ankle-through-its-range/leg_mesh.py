"""The care-training leg's moulded surfaces: the lower leg (knee to ankle) and the foot, each one smooth triangle mesh,
built from adult women's mean measurements. NumPy only.

Leg frame (m), the leg lying on its back (supine) with the ankle in neutral (foot at a right angle to the leg):
origin at the ankle's hinge axis, midway between the ankle bones (malleoli); y along the lower leg toward the knee;
z up, out of the front of the shin (the toes point +z); x across the leg (the leg's outer side, toward the little
toe, is -x: a right leg seen from the robot, the patient's head toward +y).

Measurements: ANSUR II (2012 US Army anthropometric survey, public data, 1,986 women; means): foot length 246 mm,
foot breadth 93 mm, ball-of-foot length 182 mm, ball-of-foot circumference 228 mm, heel breadth 67 mm, bimalleolar
breadth 67 mm, lateral malleolus height 63 mm, ankle circumference (smallest, above the malleoli) 216 mm, calf
circumference 373 mm, tibial height (floor to the knee joint line) 438 mm. The ankle's hinge axis is taken 50 mm above
the sole (just under the malleoli's tips, where the talocrural axis runs) and 49 mm in front of the back of the heel
(0.2 of foot length, as in gait-analysis foot models).

Each surface is lofted: rounded-rectangle (superellipse) cross-sections whose width, top and bottom follow the
tables below, closed by domed ends.
"""

from __future__ import annotations

import math

import numpy as np

CM = 0.01
FOOT_LEN = 24.6 * CM
AXIS_UP = 5.0 * CM  # the hinge axis above the sole (standing)
AXIS_FWD = 4.9 * CM  # ... and in front of the back of the heel
KNEE_LINE = 43.8 * CM - AXIS_UP  # the knee joint line above the axis (tibial height less the axis height)
LEG_END = KNEE_LINE + 2.7 * CM  # the trainer's knee end (it stops just above the joint line)

# The foot, standing on its sole: along its length fx (cm from the back of the heel), its width and its plantar (s)
# and dorsal (t) surfaces' heights (cm above the floor). Breadth 9.3 at the ball (18.2 cm), heel 6.7; the heel's
# back-and-under corner is rounded to about 3 cm radius.
FOOT_TABLE = np.array([
    # fx    width  s     t
    [0.0,   2.6,  3.0,  5.8],
    [0.5,   4.6,  2.2,  7.0],
    [1.5,   6.0,  1.1,  7.8],
    [3.0,   6.7,  0.35, 8.3],
    [4.5,   6.7,  0.05, 8.5],
    [5.0,   6.7,  0.0,  8.6],
    [7.5,   7.0,  0.0,  8.0],
    [10.0,  7.7,  0.0,  6.7],
    [13.0,  8.5,  0.0,  5.6],
    [16.0,  9.1,  0.0,  4.6],
    [18.2,  9.3,  0.0,  4.0],
    [20.5,  9.0,  0.05, 3.2],
    [22.5,  7.8,  0.15, 2.6],
    [23.8,  5.6,  0.35, 2.1],
    [24.6,  2.2,  0.8,  1.5],
])

# The lower leg, lying on its back: along y (cm from the axis toward the knee), its width (x) and how far its front
# (shin, top) and back (calf, bottom) stand from the axis line (cm). Smallest girth 21.6 cm just above the malleoli,
# calf 37.3 cm about 13 cm below the knee joint line.
LEG_TABLE = np.array([
    # y      width  top   bottom
    [-1.5,   6.0,  2.6,  3.4],
    [0.0,    6.7,  2.9,  3.8],
    [3.0,    6.4,  2.9,  4.1],
    [6.0,    6.6,  2.9,  4.3],
    [10.0,   7.2,  3.0,  4.9],
    [15.0,   8.2,  3.1,  5.8],
    [20.0,   9.6,  3.2,  7.0],
    [25.0,  10.8,  3.4,  8.3],
    [28.0,  11.0,  3.5,  8.7],
    [31.0,  10.6,  3.6,  8.5],
    [35.0,   9.8,  4.2,  7.4],
    [38.0,  10.0,  5.0,  6.4],
    [LEG_END / CM, 9.6, 4.8, 5.8],
])

EXP = 2.4  # superellipse exponent of the cross-sections (2 = ellipse; larger = squarer)


def _section(cx, cz, half_w, top, bottom, n):
    """Points of one cross-section in the (x, z) plane: a superellipse ``half_w`` across, ``top`` above and ``bottom``
    below its centre (cx, cz)."""
    phi = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    x = cx + half_w * np.sign(c) * np.abs(c) ** (2.0 / EXP)
    up = np.where(s >= 0, top, bottom)
    z = cz + up * np.sign(s) * np.abs(s) ** (2.0 / EXP)
    return x, z


def _loft(stations, n=40, cap_a=None, cap_b=None):
    """A closed tube through cross-sections: ``stations`` = list of (axial coord, half width, top, bottom) with the
    section centred on the axis line; the tube runs along y; capped at both ends by a fan to ``cap_a`` / ``cap_b``
    (points; default: the end sections' centres)."""
    rings = []
    for yv, hw, top, bot in stations:
        x, z = _section(0.0, 0.0, hw, top, bot, n)
        rings.append(np.stack([x, np.full(n, yv), z], axis=1))
    pts = np.concatenate(rings)
    faces = []
    m = len(rings)
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [(a, c, b), (b, c, d)]
    first = rings[0].mean(0) if cap_a is None else np.asarray(cap_a, float)
    last = rings[-1].mean(0) if cap_b is None else np.asarray(cap_b, float)
    ia, ib = len(pts), len(pts) + 1
    pts = np.concatenate([pts, first[None], last[None]])
    for j in range(n):
        faces.append((ia, j, (j + 1) % n))
        k = (m - 1) * n
        faces.append((ib, k + (j + 1) % n, k + j))
    return pts, np.asarray(faces, dtype=np.int64)


def _orient(pts, faces):
    """Flip faces so that their normals point away from the centroid (the shapes are nearly convex)."""
    c = pts.mean(0)
    tri = pts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = (n * (tri.mean(1) - c)).sum(1) < 0
    faces = faces.copy()
    faces[out] = faces[out][:, [0, 2, 1]]
    return faces


def _normals(pts, faces):
    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def shank_mesh(n: int = 44):
    """The lower leg from just below the malleoli to the knee end (leg frame, m): points, faces, normals."""
    st = []
    for y, w, top, bot in LEG_TABLE:
        st.append((y * CM, w / 2 * CM, top * CM, bot * CM))
    # round the knee end: two smaller rings beyond it
    y, hw, top, bot = st[-1]
    st += [(y + 0.6 * CM, hw * 0.9, top * 0.9, bot * 0.9), (y + 1.0 * CM, hw * 0.65, top * 0.65, bot * 0.65)]
    pts, faces = _loft(st, n)
    faces = _orient(pts, faces)
    return pts, faces, _normals(pts, faces)


def foot_mesh(n: int = 44):
    """The foot (leg frame, m, ankle in neutral): points, faces, normals. Lofted along its length with the sections in
    the (width, height) plane, then turned so that the toes point up (+z) and the sole faces -y."""
    rings = []
    for fx, w, s, t in FOOT_TABLE:
        phi = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        c, sn = np.cos(phi), np.sin(phi)
        hw, mid, hh = w / 2, (s + t) / 2, (t - s) / 2
        fy = hw * np.sign(c) * np.abs(c) ** (2.0 / EXP)
        fz = mid + hh * np.sign(sn) * np.abs(sn) ** (2.0 / EXP)
        rings.append(np.stack([np.full(n, fx), fy, fz], axis=1) * CM)
    pts = np.concatenate(rings)
    m = len(rings)
    faces = []
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c2, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [(a, c2, b), (b, c2, d)]
    heel = np.array([-0.15, 0.0, 4.0]) * CM
    toe = np.array([FOOT_LEN / CM + 0.15, 0.0, 1.2]) * CM
    ia, ib = len(pts), len(pts) + 1
    pts = np.concatenate([pts, heel[None], toe[None]])
    for j in range(n):
        faces.append((ia, j, (j + 1) % n))
        k = (m - 1) * n
        faces.append((ib, k + (j + 1) % n, k + j))
    faces = np.asarray(faces, dtype=np.int64)
    # standing (fx forward, fy left, fz up) -> leg frame: z = fx - AXIS_FWD, y = fz - AXIS_UP, x = fy
    leg = np.stack([pts[:, 1], pts[:, 2] - AXIS_UP, pts[:, 0] - AXIS_FWD], axis=1)
    faces = _orient(leg, faces)
    return leg, faces, _normals(leg, faces)


def foot_point(fx_cm: float, fz_cm: float, fy_cm: float = 0.0) -> np.ndarray:
    """A point given in standing foot measures (cm along from the heel, up from the floor, across) in the leg frame."""
    return np.array([fy_cm * CM, fz_cm * CM - AXIS_UP, fx_cm * CM - AXIS_FWD])


def foot_section(fx_cm: float):
    """(width, sole height, top height) in cm at ``fx_cm`` along the foot, interpolated from the table."""
    t = FOOT_TABLE
    return tuple(float(np.interp(fx_cm, t[:, 0], t[:, k])) for k in (1, 2, 3))


def leg_section(y_cm: float):
    """(width, top, bottom) in cm at ``y_cm`` along the lower leg."""
    t = LEG_TABLE
    return tuple(float(np.interp(y_cm, t[:, 0], t[:, k])) for k in (1, 2, 3))
