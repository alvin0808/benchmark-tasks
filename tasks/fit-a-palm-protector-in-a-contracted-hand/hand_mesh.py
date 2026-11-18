"""The care-training hand's moulded surfaces and finger sizes, built from adult women's mean measurements. NumPy only.

Hand frame (cm here; the task file converts to m), the right hand lying palm up: origin on the knuckle (MCP) axis,
under the middle finger's knuckle; x along the hand toward the wrist and forearm (the fingers point -x when
straight); y across the hand toward the thumb (the thumb side is +y for a right hand palm up with the fingers
pointing -x); z up, out of the palm. The back of the hand and forearm rest on the pad at z = -1.4.

Measurements: ANSUR II (2012 US Army anthropometric survey, public data, 1,986 women; means): hand length 181 mm,
palm length 109 mm (wrist crease to the middle finger's base crease), hand breadth 78 mm (across the knuckles),
hand circumference 187 mm, wrist circumference 155 mm, forearm-hand length 440 mm (so the forearm, elbow to wrist,
is about 259 mm). The middle finger is then 72 mm from its base crease to its tip, or about 90 mm from the knuckle
joint's centre (which sits about 18 mm inside the base crease). The other fingers are scaled by the usual ratios to
the middle finger (index 0.93, ring 0.95, little 0.77), and each finger is split into its three bones in the usual
proportions (about 0.49 : 0.29 : 0.22). The hand is 2.8 cm thick across the palm (back of the hand to palm surface),
its knuckle axis 1.4 cm above the back of the hand.

Each surface is lofted: rounded-rectangle (superellipse) cross-sections across the hand (y, z) whose half width,
bottom and top follow the tables below, closed by domed ends; the thenar (thumb) and hypothenar (little-finger) pads
of the palm are ellipsoids.
"""

from __future__ import annotations

import math

import numpy as np

CM = 0.01
EXP = 2.6  # superellipse exponent of the forearm's cross-sections (2 = ellipse; larger = squarer)
PALM_EXP = 4.0  # the palm's: flatter across, as over the knuckle heads
BACK_Z = -1.4  # the back of the hand and forearm (lying on the pad)
PALM_Z = 1.3  # the palm's surface over the middle of the palm

# Fingers: (y of the knuckle axis, bone lengths prox / middle / distal, radii prox / middle / distal) in cm, and the
# resting curl (knuckle MCP and middle-joint PIP flexion, deg) of the contracted hand: a fist, each fingertip curled
# down to about 1 cm above the palm (the end joint, DIP, is set at 30 deg). Lengths are from the knuckle joint's centre
# to the tip.
FINGERS = {
    "index": dict(y=2.6, L=(4.1, 2.4, 1.9), r=(0.85, 0.78, 0.70), mcp=90.0, pip=106.0),
    "middle": dict(y=0.8, L=(4.4, 2.7, 2.0), r=(0.88, 0.80, 0.72), mcp=90.0, pip=102.0),
    "ring": dict(y=-1.0, L=(4.1, 2.6, 1.9), r=(0.82, 0.75, 0.68), mcp=90.0, pip=98.0),
    "little": dict(y=-2.7, L=(3.3, 1.9, 1.7), r=(0.72, 0.66, 0.60), mcp=90.0, pip=88.0),
}
DIP = 30.0  # end-joint flexion, fixed
ORDER = ("index", "middle", "ring", "little")

# The thumb: lying along the index finger's outer side, as in a fist with the thumb tucked beside it. Its knuckle
# (MCP) sits at the outer edge of the thenar pad, 3.6 cm behind the finger knuckles, the thumb lying on its side at the
# palm's mid-thickness; two bones (proximal 3.0 cm, distal 2.5 cm). It swings about the vertical through its knuckle:
# out, away from the index finger, against a spring.
THUMB_BASE = np.array([3.6, 5.0, -0.1])
THUMB_DIR = np.array([-1.0, -0.03, 0.05])
THUMB_L = (3.0, 2.5)
THUMB_R = (0.95, 0.85)

# The palm (from the knuckles to the wrist crease, 9.1 cm): x, half width, bottom z, top z (cm).
PALM_TABLE = np.array([
    [-1.0, 3.40, -1.25, 0.70],
    [-0.5, 3.80, -1.40, 1.10],
    [0.0, 3.92, -1.40, 1.25],
    [2.0, 3.95, -1.40, 1.32],
    [4.5, 3.85, -1.40, 1.32],
    [7.0, 3.50, -1.40, 1.20],
    [9.1, 2.95, -1.40, 1.00],
])
# The forearm (from 0.6 cm inside the wrist crease to 20 cm up the forearm, where the training arm is cut off): x,
# half width, bottom z, top z. Wrist 155 mm round (5.9 x 3.6 cm), the forearm widening toward the elbow.
FOREARM_TABLE = np.array([
    [8.5, 2.95, -1.40, 1.00],
    [10.0, 2.95, -1.40, 1.05],
    [13.0, 3.10, -1.40, 1.45],
    [17.0, 3.40, -1.40, 2.10],
    [21.0, 3.70, -1.40, 2.80],
    [25.0, 3.85, -1.40, 3.30],
    [29.1, 3.85, -1.40, 3.50],
])
FOREARM_END = 29.1
THENAR = (np.array([5.6, 2.9, 1.0]), np.array([2.9, 2.2, 0.85]))  # centre, radii (cm)
HYPOTHENAR = (np.array([4.8, -2.7, 0.9]), np.array([3.4, 1.25, 0.70]))


def finger_points(name: str, mcp: float | None = None, pip: float | None = None):
    """(knuckle, middle joint, end joint, fingertip centre) of a finger (cm, hand frame) at knuckle and middle-joint
    flexion ``mcp`` and ``pip`` (deg; default: the resting curl). The fingertip centre is the centre of the end
    bone's rounded tip (its outer end lies one tip radius further on)."""
    f = FINGERS[name]
    a1 = f["mcp"] if mcp is None else mcp
    a2 = f["pip"] if pip is None else pip

    def d(phi):
        p = math.radians(phi)
        return np.array([-math.cos(p), 0.0, math.sin(p)])

    k = np.array([0.0, f["y"], 0.0])
    p = k + f["L"][0] * d(a1)
    q = p + f["L"][1] * d(a1 + a2)
    t = q + (f["L"][2] - f["r"][2]) * d(a1 + a2 + DIP)
    return k, p, q, t


def thumb_points():
    """(base, joint, tip centre) of the thumb (cm, hand frame)."""
    u = THUMB_DIR / np.linalg.norm(THUMB_DIR)
    j = THUMB_BASE + THUMB_L[0] * u
    u2 = u + np.array([0.0, -0.10, -0.04])
    u2 /= np.linalg.norm(u2)
    t = j + (THUMB_L[1] - THUMB_R[1]) * u2
    return THUMB_BASE.copy(), j, t


def _section(cy, cz, half_w, top, bottom, n, exp=EXP):
    phi = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    y = cy + half_w * np.sign(c) * np.abs(c) ** (2.0 / exp)
    up = np.where(s >= 0, top, bottom)
    z = cz + up * np.sign(s) * np.abs(s) ** (2.0 / exp)
    return y, z


def _loft(table, n=40, dome_a=0.0, dome_b=0.0, exp=EXP):
    """A closed tube along x through the table's cross-sections, closed at each end by a fan to a point ``dome_a`` /
    ``dome_b`` cm beyond the end section's centre."""
    rings = []
    for xv, hw, bot, top in table:
        cz = (top + bot) / 2
        half = (top - bot) / 2
        y, z = _section(0.0, cz, hw, half, half, n, exp)
        rings.append(np.stack([np.full(n, xv), y, z], axis=1))
    pts = np.concatenate(rings)
    faces = []
    m = len(rings)
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [(a, b, c), (b, d, c)]
    first = rings[0].mean(0) + np.array([-dome_a, 0.0, 0.0])
    last = rings[-1].mean(0) + np.array([dome_b, 0.0, 0.0])
    ia, ib = len(pts), len(pts) + 1
    pts = np.concatenate([pts, first[None], last[None]])
    for j in range(n):
        faces.append((ia, (j + 1) % n, j))
        k = (m - 1) * n
        faces.append((ib, k + j, k + (j + 1) % n))
    return pts, np.asarray(faces, dtype=np.int64)


def _ellipsoid(centre, radii, n_lat=12, n_lon=24):
    pts = [centre + radii * np.array([0.0, 0.0, -1.0])]
    for i in range(1, n_lat):
        th = -math.pi / 2 + math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            pts.append(centre + radii * np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph),
                                                  math.sin(th)]))
    pts.append(centre + radii * np.array([0.0, 0.0, 1.0]))
    pts = np.asarray(pts)
    faces = []
    for j in range(n_lon):
        faces.append((0, 1 + (j + 1) % n_lon, 1 + j))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            faces += [(a, b, c), (b, d, c)]
    top = len(pts) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append((top, base + j, base + (j + 1) % n_lon))
    return pts, np.asarray(faces, dtype=np.int64)


def _orient(pts, faces):
    """Flip faces so that their normals point away from the centroid (the shapes are nearly convex)."""
    c = pts.mean(0)
    tri = pts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = ((tri.mean(1) - c) * nrm).sum(1) < 0
    faces = faces.copy()
    faces[out] = faces[out][:, ::-1]
    return faces


def _normals(pts, faces):
    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return vn


def _finish(pts, faces):
    faces = _orient(pts, faces)
    return pts * CM, faces, _normals(pts, faces)


def palm_mesh():
    """The palm, from the knuckles to the wrist crease (m, hand frame): points, faces, vertex normals."""
    return _finish(*_loft(PALM_TABLE, dome_a=0.4, dome_b=0.0, exp=PALM_EXP))


def forearm_mesh():
    """The forearm, from the wrist to its cut end 20 cm up (m, hand frame)."""
    return _finish(*_loft(FOREARM_TABLE, dome_a=0.0, dome_b=0.0))


def thenar_mesh():
    return _finish(*_ellipsoid(*THENAR))


def hypothenar_mesh():
    return _finish(*_ellipsoid(*HYPOTHENAR))


def palm_top(x: float, y: float) -> float:
    """The palm's surface height (cm, hand frame) at (x, y), from the palm table and the two pads (for checks)."""
    xs = PALM_TABLE[:, 0]
    top = float(np.interp(x, xs, PALM_TABLE[:, 3]))
    hw = float(np.interp(x, xs, PALM_TABLE[:, 1]))
    if abs(y) > hw:
        return -99.0
    for c, r in (THENAR, HYPOTHENAR):
        q = ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2
        if q < 1.0:
            top = max(top, float(c[2] + r[2] * math.sqrt(1.0 - q)))
    return top
