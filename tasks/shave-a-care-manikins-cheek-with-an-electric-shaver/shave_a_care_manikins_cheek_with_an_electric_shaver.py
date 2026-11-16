"""On the station table: a resident's bedside. A life-size adult male care-training head and neck (rigid vinyl) stands
seated upright on a weighted base, its face toward the robot's left so that its left cheek faces the robot. The left
side of the face has a stubble-shaded beard area. The skin over the left cheekbone is a soft insert that slides up
toward the temple when a fingertip drags it and springs back when let go. A three-head rotary electric shaver lies on
its back in a white tray in front of the face, its heads toward the head.

Task (instruction): shave the beard on the manikin's left cheek and jaw with the electric shaver without pressing hard
or touching its lips, nose, eye or ear, then put the shaver back on its tray. The instruction does not say how. The
procedure the task checks is the nurse-aide skills checklist's (Reuter, Nursing Assistant, 5.30 "Skills Checklist:
Shaving With an Electric Razor", CC BY 4.0): "Hold the shaver at a right angle to the resident's face, using your free
hand to pull their skin taught as you shave." Philips's Series 3000 shaver manual: "Stretch your skin with your free
hand so the blades can lift and cut hair" and "Do not press too hard as too much pressure can cause shaving heads to
pinch the skin." One hand holds the soft cheek skin pulled up while the other keeps the shaver flat on the curved face
with a light press.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. The head
frame: x the face's forward direction (world +y), y the manikin's left (world -x), z up; its origin, midway between the
ear canals, is at (0.300, 0.000) and 0.190 above the table.
- head: 2.0 kg on a sprung neck (a spring hinge about the world y axis at the base's top, 20 N m/rad: 1 N pushed into
  the middle of the beard area moves it about 0.8 mm). 15.2 cm wide, 23.6 cm chin to crown, the jaw 11.6 cm wide at its
  angles, the neck 11.4 cm across. Its surface is the implicit model below, shown on a 1.5-degree grid of rays from
  inside it and colliding through about 130 convex pieces that follow it (within 1.2 mm over the beard area).
- beard area (left side): from 1.8 cm in front of the ear canals to 2 cm left of the chin's midline, from the cheek line
  down to where the skin under the jaw faces the table at 30 degrees, at least 2.4 cm from the lips, 2.2 cm from the
  nostril and 1.6 cm from the eye and the ear: 38 cm2 (x 0.229 to 0.279, y 0.018 to 0.083, 0.098 to 0.181 above the
  table), sampled at 593 points about 3 mm apart. The soft cheek: its 372 points behind a line 7.2 cm in front of the
  ear canals and above one 6.2 cm below them.
- skin insert: 30 x 22 mm over the left cheekbone, centred at (0.227, 0.040) 0.211 above the table; it slides up along
  the head's z axis from 0 to 8 mm on a 0.12 N/mm spring; taut when pulled 3 mm or more.
- shaver: loose, 16.1 x 5.8 x 4.5 cm, 170 g (body 145 g, head unit 25 g); three heads 20 mm across, their centres 23 mm
  apart, each with a 1.2 mm domed face; the head unit pivots up to 20 degrees either way and presses in up to 6 mm on a
  1 N/mm spring. It lies on its back along +x in the tray, its body's top end at (0.1945, 0.200).
- tray: fixed, white, 20 x 9 cm with 1 cm walls, centred at (0.140, 0.200).

Score: 0 for the rest of the run once the shaver's spring reads over 5 N, a head comes within 1 mm of the lips, the
left nostril, the left eye or the left ear, a gripper's finger comes within 1 mm of the left eye, the head is pushed 8
mm, or the shaver falls 3 cm with no gripper holding it. A point is shaved when a head passes over it (inside its
20 mm circle, within 1.5 mm of its face), the head unit within 20 degrees of flat to the skin, pressing 0.3 to 3 N; on
the soft cheek only while the insert is pulled 3 mm or more. At the end: 1.0 (success, held 1 s) at least 95 % shaved
and the shaver resting on its tray, let go and still; 0.75 at least 95 % shaved; 0.5 at least 70 %; 0.25 at least 30 %.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Joint, MeshShape, allow_matrix_grips, as_numpy  # noqa: E402
from bodykit import on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
SKIN = (0.86, 0.69, 0.58)  # light vinyl skin
STUBBLE = (0.45, 0.39, 0.36)  # the beard's stubble shadow
LIPS = (0.72, 0.42, 0.40)
SCLERA = (0.93, 0.92, 0.89)
IRIS = (0.33, 0.24, 0.17)
PUPIL = (0.05, 0.04, 0.04)
NOSTRIL = (0.30, 0.18, 0.16)
BASE_GREY = (0.22, 0.23, 0.25)
TRAY_WHITE = (0.90, 0.91, 0.92)
SHAVER_BODY = (0.12, 0.13, 0.15)
SHAVER_TRIM = (0.55, 0.57, 0.60)
STEEL = (0.74, 0.75, 0.77)
HEAD_FRAME = (0.20, 0.22, 0.26)
LED = (0.30, 0.65, 0.95)


# ======================================================================================================================
# The head. Head frame: x forward (the face), y the manikin's left, z up; origin midway between the ear canals.
# ======================================================================================================================
def _len(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def _ell(p, c, r):
    """Approximate signed distance from an ellipsoid (centre c, radii r)."""
    q = (p - np.asarray(c, float)) / np.asarray(r, float)
    k0 = _len(q)
    k1 = _len(q / np.asarray(r, float))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def _cap(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip(np.sum(pa * ba, axis=-1) / (ba @ ba), 0.0, 1.0)
    return _len(pa - h[..., None] * ba) - r


def _sph(p, c, r):
    return _len(p - np.asarray(c, float)) - r


def _plane(p, p0, n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    return np.sum((p - np.asarray(p0, float)) * n, axis=-1)


def _smin(a, b, k):
    """Smooth union (blend radius k)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _mirror(p):
    q = np.array(p, float, copy=True)
    q[..., 1] = np.abs(q[..., 1])
    return q


EYE_C, EYE_R = np.array([0.064, 0.033, 0.010]), 0.0125  # left eyeball (the right one mirrored)
LIP_U, LIP_U_R = (0.091, 0.0, -0.0505), (0.008, 0.024, 0.006)
LIP_L, LIP_L_R = (0.089, 0.0, -0.0625), (0.008, 0.022, 0.0065)
NOSE_TIP = (0.104, 0.0, -0.023)
ALA = (0.092, 0.014, -0.029)  # nose wing
NOSTRIL_C = np.array([0.097, 0.0085, -0.0325])  # left nostril opening, on the underside of the nose


def head_sdf(p):
    """Signed distance (approximate, m) from the head and neck (ears apart), head frame. An adult male head: 19.5 cm
    from brow to back, 15.2 cm wide, 23.3 cm from chin to crown; jaw 11.4 cm wide at its angles; neck 11.4 cm
    across."""
    p = np.asarray(p, float)
    m = _mirror(p)
    d = _ell(p, (-0.012, 0.0, 0.036), (0.097, 0.076, 0.098))  # cranium
    d = _smin(d, _ell(p, (0.030, 0.0, 0.042), (0.060, 0.066, 0.068)), 0.025)  # forehead
    d = _smin(d, _ell(p, (0.034, 0.0, -0.012), (0.062, 0.066, 0.062)), 0.025)  # midface
    jaw = _ell(p, (0.030, 0.0, -0.052), (0.060, 0.061, 0.052))  # mandible, cut along the jaw line below
    jaw = _smax(jaw, _plane(m, (0.004, 0.0, -0.071), (-0.36, 0.0, -0.93)), 0.012)
    d = _smin(d, jaw, 0.02)
    d = _smin(d, _ell(m, (0.048, 0.050, 0.000), (0.026, 0.020, 0.016)), 0.015)  # cheekbones
    d = _smin(d, _ell(p, (0.071, 0.0, -0.086), (0.018, 0.024, 0.016)), 0.014)  # chin
    d = _smin(d, _ell(p, (0.072, 0.0, -0.047), (0.026, 0.032, 0.028)), 0.014)  # muzzle
    d = _smin(d, _ell(p, LIP_U, LIP_U_R), 0.004)  # upper lip
    d = _smin(d, _ell(p, LIP_L, LIP_L_R), 0.004)  # lower lip
    d = _smin(d, _cap(p, (0.090, 0.0, 0.018), (0.106, 0.0, -0.020), 0.007), 0.008)  # nose bridge
    d = _smin(d, _sph(p, NOSE_TIP, 0.010), 0.008)  # nose tip
    d = _smin(d, _sph(m, ALA, 0.008), 0.006)  # nose wings
    d = _smax(d, -_ell(m, NOSTRIL_C, (0.0045, 0.0035, 0.0025)), 0.001)  # nostrils
    d = _smin(d, _ell(p, (0.080, 0.0, 0.028), (0.014, 0.050, 0.010)), 0.012)  # brow
    d = _smax(d, -_ell(m, (0.088, 0.033, 0.010), (0.016, 0.019, 0.012)), 0.006)  # eye sockets
    d = _smin(d, _sph(m, EYE_C, EYE_R), 0.003)  # eyeballs
    d = _smin(d, _cap(p, (-0.008, 0.0, -0.030), (-0.008, 0.0, -0.200), 0.057), 0.012)  # neck
    return d


def _normals(sdf, p, h=2e-4):
    p = np.asarray(p, float)
    g = np.zeros_like(p)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        g[..., i] = (sdf(p + e) - sdf(p - e)) / (2 * h)
    return g / np.maximum(_len(g)[..., None], 1e-12)


# The left ear: a pinna 63 x 33 mm leaning back 15 degrees, its back edge standing 18 degrees off the head.
EAR_C = np.array([-0.011, 0.079, 0.004])


def _ear_frame() -> np.ndarray:
    t, f = math.radians(15.0), math.radians(18.0)
    tilt = np.array([[math.cos(t), 0.0, -math.sin(t)], [0.0, 1.0, 0.0], [math.sin(t), 0.0, math.cos(t)]])
    flare = np.array([[math.cos(f), -math.sin(f), 0.0], [math.sin(f), math.cos(f), 0.0], [0.0, 0.0, 1.0]])
    return flare @ tilt


EAR_R = _ear_frame()


def ear_sdf(p):
    q = (np.asarray(p, float) - EAR_C) @ EAR_R
    d = _ell(q, (0.0, 0.0, 0.0), (0.0165, 0.0045, 0.0315))
    return _smax(d, -_ell(q, (0.004, 0.0045, -0.004), (0.0085, 0.004, 0.0105)), 0.002)  # the bowl (concha)


def ray_surface(sdf, centre, dirs, t_max, samples=100, iters=16, t_min=0.0):
    """The outermost point where ``sdf`` changes sign along each ray centre + t * dir (t_min < t < t_max)."""
    centre = np.asarray(centre, float)
    ts = np.linspace(t_min, t_max, samples)
    inside = np.stack([sdf(centre + t * dirs) < 0 for t in ts], axis=1)
    last = np.where(inside.any(1), samples - 1 - np.argmax(inside[:, ::-1], axis=1), 0)
    a, b = ts[last], ts[np.minimum(last + 1, samples - 1)]
    for _ in range(iters):
        mid = 0.5 * (a + b)
        neg = sdf(centre + mid[:, None] * dirs) < 0
        a, b = np.where(neg, mid, a), np.where(neg, b, mid)
    return centre + (0.5 * (a + b))[:, None] * dirs


def sphere_dirs(th, ph):
    """Unit directions, shape (len(ph), len(th), 3): azimuth ``th`` from the face toward the head's left,
    elevation ``ph``."""
    T, P = np.meshgrid(th, ph)
    return np.stack([np.cos(P) * np.cos(T), np.cos(P) * np.sin(T), np.sin(P)], axis=-1)


HEAD_CENTRE = np.array([0.005, 0.0, -0.012])  # every ray of the head's surface starts here (it sees all of it)
GRID_DEG = 1.5
NECK_CUT = -0.168  # the neck ends in the base below this height (head frame)


def _head_grid():
    th = np.radians(np.arange(-180.0, 180.0, GRID_DEG))
    ph = np.radians(np.concatenate([np.arange(-80.0, 89.9, GRID_DEG), [89.9]]))
    D = sphere_dirs(th, ph)
    P = ray_surface(head_sdf, HEAD_CENTRE, D.reshape(-1, 3), 0.25, samples=74, t_min=0.03).reshape(D.shape)
    return th, ph, P


# -- the beard area, the no-touch zones and the skin patch (left side; the beard is mirrored for the look) ----------
def beard_top(f):
    """The upper edge of the beard (head frame z) at forward position f: the cheek line from the sideburn's top
    down toward the corner of the mouth."""
    return np.interp(f, [0.0, 0.02, 0.05, 0.075, 0.1], [-0.007, -0.009, -0.017, -0.028, -0.04])


BEARD_F0 = 0.018  # the sideburn's back edge, 1.8 cm in front of the ear canals
BEARD_NZ = -0.5  # its lower edge: where the skin under the jaw turns to face the table at 30 degrees
BEARD_Z0 = -0.115
NECK_GAP = 0.002  # skin within this of the neck's cylinder is the neck, not the beard
LIP_GAP, NOSTRIL_GAP, EYE_GAP, EAR_GAP = 0.024, 0.022, 0.016, 0.016  # the beard keeps this far from each zone
BEARD_L0 = 0.020  # its front edge: 2 cm left of the chin's midline
CELL = 0.003  # the beard area is sampled at points about this far apart

PATCH_F = (0.025, 0.055)  # the skin patch over the cheekbone (head frame x range, z range), left side
PATCH_Z = (0.010, 0.032)
PATCH_TRAVEL, PATCH_K = 0.008, 120.0  # it slides up toward the temple 0 to 8 mm on a 0.12 N/mm spring
TAUT = 0.003  # pulled at least this far: the cheek is taut
SOFT_F, SOFT_Z = 0.072, -0.062  # the soft cheek (needs the skin pulled): behind and above these lines


def _zone_points() -> dict:
    """Points over each no-touch zone's surface (head frame, left side)."""
    th = np.radians(np.arange(-40.0, 40.0, 0.6))
    ph = np.radians(np.arange(-40.0, 0.0, 0.6))
    P = ray_surface(head_sdf, HEAD_CENTRE, sphere_dirs(th, ph).reshape(-1, 3), 0.15, samples=34, t_min=0.05)
    grow = 0.002
    lips = P[((_ell(P, LIP_U, np.asarray(LIP_U_R) + grow) < 0) | (_ell(P, LIP_L, np.asarray(LIP_L_R) + grow) < 0))
             & (P[:, 1] >= -0.002)]
    nostril = P[np.linalg.norm(P - np.array([0.094, 0.011, -0.031]), axis=1) < 0.010]
    th = np.radians(np.arange(0.0, 60.0, 0.6))
    ph = np.radians(np.arange(-10.0, 30.0, 0.6))
    P = ray_surface(head_sdf, HEAD_CENTRE, sphere_dirs(th, ph).reshape(-1, 3), 0.15, samples=34, t_min=0.05)
    eye = P[np.linalg.norm(P - np.array([0.074, 0.033, 0.010]), axis=1) < 0.019]  # the eye and its lids' rim
    ed = sphere_dirs(np.radians(np.arange(-180.0, 180.0, 6.0)), np.radians(np.arange(-84.0, 90.0, 6.0)))
    ear = ray_surface(ear_sdf, EAR_C, ed.reshape(-1, 3), 0.05, samples=60)
    return {"lips": lips, "nostril": nostril, "eye": eye, "ear": ear}


def _min_dist(q, s) -> np.ndarray:
    out = np.full(len(q), 1.0)
    for i in range(0, len(s), 400):
        out = np.minimum(out, _len(q[:, None, :] - s[None, i:i + 400, :]).min(1))
    return out


def _beard_mask(p, n, zones) -> np.ndarray:
    """Which surface points (head frame, left side y >= 0) lie in the beard area."""
    on_neck = _cap(p, (-0.008, 0.0, -0.030), (-0.008, 0.0, -0.200), 0.057) < NECK_GAP
    m = ((p[:, 1] >= BEARD_L0) & (p[:, 0] >= BEARD_F0) & (p[:, 2] <= beard_top(p[:, 0])) & (n[:, 2] >= BEARD_NZ)
         & (p[:, 2] >= BEARD_Z0) & ~on_neck)
    idx = np.where(m)[0]
    if len(idx):
        ok = ((_min_dist(p[idx], zones["lips"]) >= LIP_GAP) & (_min_dist(p[idx], zones["nostril"]) >= NOSTRIL_GAP)
              & (_min_dist(p[idx], zones["eye"]) >= EYE_GAP) & (_min_dist(p[idx], zones["ear"]) >= EAR_GAP))
        m[idx[~ok]] = False
    return m


def _cells(zones):
    """The beard area's sample points (one per 3 mm cube of its surface), their normals and whether each lies on the
    soft cheek; and the area (m^2)."""
    th = np.radians(np.arange(-4.0, 108.0, 0.5))
    ph = np.radians(np.arange(-70.0, 16.0, 0.5))
    D = sphere_dirs(th, ph)
    P = ray_surface(head_sdf, HEAD_CENTRE, D.reshape(-1, 3), 0.14, samples=30, t_min=0.05).reshape(D.shape)
    dA = _len(np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))).reshape(-1)
    Q = P.reshape(-1, 3)
    N = _normals(head_sdf, Q)
    m = _beard_mask(Q, N, zones)
    area = float(dA[m].sum())
    B, BN = Q[m], N[m]
    _, first = np.unique(np.floor(B / CELL).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(first)
    cells, normals = B[keep], BN[keep]
    soft = (cells[:, 0] <= SOFT_F) & (cells[:, 2] >= SOFT_Z)
    return cells, normals, soft, area


def _grid_triangles(rows, cols, wrap=True):
    idx = np.arange(rows * cols).reshape(rows, cols)
    nxt = np.roll(idx, -1, axis=1) if wrap else idx[:, 1:]
    cur = idx if wrap else idx[:, :-1]
    a, b, c, d = cur[:-1], nxt[:-1], cur[1:], nxt[1:]
    return np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])


def _submesh(P, N, tri, keep, offset=0.0, colors=None):
    """The triangles ``tri[keep]`` as their own mesh (points re-indexed), pushed ``offset`` along the normals; with
    ``colors``, also the kept points' colours."""
    t = tri[keep]
    used, inv = np.unique(t.reshape(-1), return_inverse=True)
    out = (P[used] + offset * N[used], inv.reshape(-1, 3), N[used])
    return out if colors is None else out + (colors[used],)


def _slab_faces(nr: int, nc: int) -> np.ndarray:
    """Triangles closing an (nr x nc) surface grid (points 0..) and the same grid pulled inward (points nr*nc..)."""
    top = _grid_triangles(nr, nc, wrap=False)
    bot = top[:, ::-1] + nr * nc
    idx = np.arange(nr * nc).reshape(nr, nc)
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    a, b = ring, np.roll(ring, -1)
    side = np.concatenate([np.stack([a, b + nr * nc, b], -1), np.stack([a, a + nr * nc, b + nr * nc], -1)])
    return np.concatenate([top, bot, side])


def _blocks(lo: int, hi: int, step: int) -> list:
    b = list(range(lo, hi, step)) + [hi]
    if len(b) > 2 and b[-1] - b[-2] < step // 2:
        del b[-2]
    return list(zip(b[:-1], b[1:]))


def _tile(P, rr, cc, depth, z_min):
    """One convex collider: the surface points P[rr x cc] and the same pulled ``depth`` toward the head's centre."""
    nc = P.shape[1]
    surf = P[np.ix_(rr, [c % nc for c in cc])].copy()
    if surf[..., 2].max() < z_min + 0.002:
        return None
    surf[..., 2] = np.maximum(surf[..., 2], z_min)
    flat = surf.reshape(-1, 3)
    if np.ptp(flat, axis=0).max() < 0.002:
        return None
    dirs = flat - HEAD_CENTRE
    inner = HEAD_CENTRE + dirs * (1.0 - depth / np.maximum(_len(dirs), 1e-6))[:, None]
    return np.concatenate([flat, inner]), _slab_faces(len(rr), len(cc))


def _tiles(P, th, ph, fine_th, fine_ph, fine, coarse, depth, z_min):
    """Convex collider tiles over the head's surface grid (rows: elevation, columns: azimuth, wrapping round the
    back): ``fine`` grid steps square over the face's left side, ``coarse`` elsewhere. Neighbouring tiles share their
    edge points, so they leave no gaps; each is the hull of a patch of the surface and its inward copy."""
    nr, nc = P.shape[:2]
    th_deg, ph_deg = np.degrees(th), np.degrees(ph)
    fr = np.where((ph_deg >= fine_ph[0]) & (ph_deg <= fine_ph[1]))[0]
    fc = np.where((th_deg >= fine_th[0]) & (th_deg <= fine_th[1]))[0]
    r0, r1, c0, c1 = int(fr[0]), int(fr[-1]), int(fc[0]), int(fc[-1])
    out = []

    def sub(a, b, k):
        return sorted(set(list(range(a, b, k)) + [b]))

    for a, b in _blocks(r0, r1, fine):  # the face's left side
        for c, d in _blocks(c0, c1, fine):
            out.append((_tile(P, sub(a, b, 2), sub(c, d, 2), depth, z_min), True))
    for a, b in _blocks(r0, r1, coarse):  # the rest of that band, round the back of the head
        for c, d in _blocks(c1, c0 + nc, coarse):
            out.append((_tile(P, sub(a, b, 3), sub(c, d, 3), depth, z_min), False))
    for lo, hi in ((0, r0), (r1, nr - 1)):  # below and above the band
        for a, b in _blocks(lo, hi, coarse):
            for c, d in _blocks(0, nc, coarse):
                out.append((_tile(P, sub(a, b, 3), sub(c, d, 3), depth, z_min), False))
    return [(t, f) for t, f in out if t is not None]


def _build_head():
    """Everything about the head that the scene and the scorer need, computed once."""
    th, ph, P = _head_grid()
    rows, cols = P.shape[:2]
    Q = P.reshape(-1, 3)
    N = _normals(head_sdf, Q)
    tri = _grid_triangles(rows, cols)
    zones = _zone_points()
    cells, cell_n, soft, area = _cells(zones)
    # the eyes: their triangles coloured by what their middles lie on (sclera, iris, pupil)
    mid = Q[tri].mean(axis=1)
    eyes = np.abs(_len(_mirror(mid) - EYE_C) - EYE_R) < 0.0008
    look = _mirror(mid) - EYE_C
    look_cos = look[:, 0] / np.maximum(_len(look), 1e-9)
    pupil = eyes & (look_cos > math.cos(math.radians(13.0)))
    iris = eyes & ~pupil & (look_cos > math.cos(math.radians(30.0)))
    below = Q[tri][:, :, 2].max(axis=1) < NECK_CUT
    groups = {}
    taken = below.copy()
    for name, mask, color in (("pupil", pupil, PUPIL), ("iris", iris, IRIS), ("sclera", eyes, SCLERA)):
        m = mask & ~taken
        taken |= m
        groups[name] = (m, color)
    groups["skin"] = (~taken, SKIN)
    # the skin's colour per vertex, blended over a few millimetres at the edges: stubble on the beard area (both
    # sides), the lips, the nostrils
    def blur(m):
        g = m.reshape(rows, cols).astype(float)
        out = g.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    out += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
        return (out / 9.0).reshape(-1)

    stubble = blur(_beard_mask(_mirror(Q), N, zones))
    lips = blur(((_ell(Q, LIP_U, np.asarray(LIP_U_R) + 0.0006) < 0) | (_ell(Q, LIP_L, np.asarray(LIP_L_R) + 0.0006) < 0))
                & (Q[:, 0] > 0.080))
    nostril = blur((_ell(_mirror(Q), NOSTRIL_C, (0.0058, 0.0046, 0.0036)) < 0) & (N[:, 2] < 0.2))
    base = np.array(SKIN)
    colors = (base[None, :] + stubble[:, None] * (np.array(STUBBLE) - base) + lips[:, None] * (np.array(LIPS) - base)
              + nostril[:, None] * (np.array(NOSTRIL) - base))
    colors = np.clip(colors, 0.0, 1.0)
    tiles = _tiles(P, th, ph, fine_th=(-8.0, 112.0), fine_ph=(-64.0, 30.0), fine=8, coarse=24, depth=0.016,
                   z_min=NECK_CUT + 0.006)
    return dict(th=th, ph=ph, P=P, Q=Q, N=N, tri=tri, groups=groups, colors=colors, tiles=tiles, zones=zones,
                cells=cells, cell_n=cell_n, soft=soft, area=area)


def _build_ear(side: float, step: float = 6.0):
    ed = sphere_dirs(np.radians(np.arange(-180.0, 180.0, step)), np.radians(np.arange(-90.0 + step / 2, 90.0, step)))
    rows, cols = ed.shape[:2]
    P = ray_surface(ear_sdf, EAR_C, ed.reshape(-1, 3), 0.05, samples=60)
    N = _normals(ear_sdf, P)
    tri = _grid_triangles(rows, cols)
    if side < 0:
        P = P * np.array([1.0, -1.0, 1.0])
        N = N * np.array([1.0, -1.0, 1.0])
        tri = tri[:, ::-1]
    return P, N, tri


GEO = _build_head()


# ======================================================================================================================
# Layout (world frame: x forward from the arm bases, y to the robot's left, z up; heights above the table)
# ======================================================================================================================
HEAD_O = np.array([0.300, 0.000, TOP + 0.190])  # the head frame's origin (midway between the ear canals)
R_HEAD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # face toward +y, left side toward the robot
Q_HEAD = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
HEAD_KG = 2.0
NECK_AXIS = (HEAD_O[0], HEAD_O[1] - 0.008)  # the neck's axis (world x, y)
BASE_R, BASE_H = 0.070, 0.025  # the stand's base: a weighted disc the neck rises from
NECK_K, NECK_C = 20.0, 1.5  # the neck's spring hinge (N m/rad, N m s/rad), about the world y axis at the base


def to_world(p_head) -> np.ndarray:
    return HEAD_O + np.asarray(p_head, float) @ R_HEAD.T


def _w(p) -> tuple:
    return tuple(float(v) for v in p)


def _head_shapes() -> list:
    G = GEO
    out = []
    for name, (mask, color) in G["groups"].items():
        if not mask.any():
            continue
        if name == "skin":
            pts, faces, nrm, col = _submesh(G["Q"], G["N"], G["tri"], mask, colors=G["colors"])
            out.append(MeshShape(pts, faces, color, normals=nrm, colors=col))
        else:
            pts, faces, nrm = _submesh(G["Q"], G["N"], G["tri"], mask)
            out.append(MeshShape(pts, faces, color, normals=nrm))
    for side in (1.0, -1.0):
        P, N, tri = _build_ear(side)
        out.append(MeshShape(P, tri, SKIN, normals=N))
        P, _, tri = _build_ear(side, step=22.5)
        out.append(MeshShape(P, tri, SKIN, collide=True, visible=False))
    for pts, faces in (t for t, _ in G["tiles"]):
        out.append(MeshShape(pts, faces, SKIN, collide=True, visible=False))
    return out


# -- the skin patch over the left cheekbone --------------------------------------------------------------------------
def _patch_shapes() -> list:
    G = GEO
    Q, N, tri = G["Q"], G["N"], G["tri"]
    inside = ((Q[:, 0] >= PATCH_F[0]) & (Q[:, 0] <= PATCH_F[1]) & (Q[:, 2] >= PATCH_Z[0]) & (Q[:, 2] <= PATCH_Z[1])
              & (Q[:, 1] > 0.03))
    keep = inside[tri].all(axis=1)
    pts, faces, nrm = _submesh(Q, N, tri, keep, offset=0.0006)
    # its collider: the patch's surface on a 5 x 5 grid and the same 10 mm deeper
    f = np.linspace(PATCH_F[0], PATCH_F[1], 5)
    z = np.linspace(PATCH_Z[0], PATCH_Z[1], 5)
    F, Z = np.meshgrid(f, z)
    starts = np.stack([F.reshape(-1), np.zeros(F.size), Z.reshape(-1)], -1)
    surf = np.array([ray_surface(head_sdf, s, np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0] for s in starts])
    n = _normals(head_sdf, surf)
    surf = surf + 0.0006 * n
    inner = surf - 0.010 * n
    return [MeshShape(pts, faces, PATCH_SKIN, normals=nrm),
            MeshShape(np.concatenate([surf, inner]), _slab_faces(5, 5), PATCH_SKIN, collide=True, visible=False)]


PATCH_SKIN = (0.87, 0.67, 0.57)
PATCH_C = np.array([0.5 * (PATCH_F[0] + PATCH_F[1]), 0.0727, 0.5 * (PATCH_Z[0] + PATCH_Z[1])])  # its middle, head frame

# -- the shaver: a three-head rotary shaver, 16.1 x 5.8 x 4.5 cm, 170 g ------------------------------------------------
# Shaver frame: x along its axis toward the heads (origin at the top of the body), y across its width, z out of its
# front (the power button). Head-unit frame: the same axes, origin at the pivot.
BODY_L, BODY_W, BODY_T = 0.135, 0.058, 0.045
PIVOT_X = 0.012  # the head unit's pivot, above the body's top
HEAD_R = 0.010  # each rotary head's cutting face: 20 mm across
HEAD_RING = 0.0133  # the three heads' centres lie on this circle, 120 degrees apart
HEAD_ANG = (90.0, 210.0, 330.0)
HEAD_RIM_X, HEAD_TOP_X = 0.0130, 0.0142  # each head's face: its rim and the top of its 1.2 mm dome (unit frame)
SHAVER_KG, UNIT_KG = 0.145, 0.025
UNIT_K, UNIT_C, UNIT_TRAVEL = 1000.0, 3.0, 0.006  # the head unit presses in on a 1 N/mm spring, 6 mm at most
TILT_MAX, TILT_K, TILT_C = 20.0, 0.05, 0.0015  # it pivots +-20 degrees (both ways) on a soft centring spring


def head_centres() -> np.ndarray:
    a = np.radians(HEAD_ANG)
    return np.stack([np.full(3, HEAD_RIM_X), HEAD_RING * np.cos(a), HEAD_RING * np.sin(a)], -1)


def _dome() -> tuple:
    """One head's domed face as a convex collider (head-centre frame): rings of 16 points from its base to the top
    of its dome, closed by fans."""
    rings = ((HEAD_R, HEAD_RIM_X - 0.004), (HEAD_R, HEAD_RIM_X), (0.0068, HEAD_RIM_X + 0.0007),
             (0.0035, HEAD_TOP_X - 0.0002))
    pts = [(x, r * math.cos(2 * math.pi * k / 16), r * math.sin(2 * math.pi * k / 16)) for r, x in rings
           for k in range(16)]
    pts += [(HEAD_TOP_X, 0.0, 0.0), (HEAD_RIM_X - 0.004, 0.0, 0.0)]
    faces = []
    for i in range(len(rings) - 1):
        for k in range(16):
            a, b = i * 16 + k, i * 16 + (k + 1) % 16
            faces += [(a, b, b + 16), (a, b + 16, a + 16)]
    top, bottom = len(pts) - 2, len(pts) - 1
    last = (len(rings) - 1) * 16
    faces += [(last + k, last + (k + 1) % 16, top) for k in range(16)]
    faces += [((k + 1) % 16, k, bottom) for k in range(16)]
    return np.array(pts), np.array(faces)


def _body_shapes() -> list:
    c = -BODY_L / 2
    out = [BoxShape((c, 0.0, 0.0), (BODY_L, BODY_W - 0.012, BODY_T), SHAVER_BODY),
           BoxShape((c, 0.0, 0.0), (BODY_L, BODY_W, BODY_T - 0.012), SHAVER_BODY)]
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            out.append(CylShape((c, sy * (BODY_W / 2 - 0.006), sz * (BODY_T / 2 - 0.006)), 0.006, BODY_L, SHAVER_BODY,
                                axis="X"))
    out += [CylShape((PIVOT_X / 2, 0.0, 0.0), 0.0135, PIVOT_X, SHAVER_TRIM, axis="X"),  # the neck under the heads
            BoxShape((-0.004, 0.0, 0.0), (0.008, BODY_W + 0.001, BODY_T - 0.004), SHAVER_TRIM, collide=False),
            CylShape((-0.052, 0.0, BODY_T / 2), 0.0065, 0.002, SHAVER_TRIM, collide=False),  # power button
            CylShape((-0.052, 0.0, BODY_T / 2 + 0.0005), 0.0035, 0.002, LED, collide=False),
            BoxShape((-0.078, 0.0, BODY_T / 2), (0.016, 0.022, 0.001), (0.05, 0.05, 0.06), collide=False)]
    return out


def _unit_shapes() -> list:
    out = [CylShape((-0.004, 0.0, 0.0), 0.008, 0.008, HEAD_FRAME, collide=False, axis="X"),
           CylShape((0.0045, 0.0, 0.0), 0.011, 0.007, HEAD_FRAME, axis="X")]
    dome, dome_faces = _dome()
    for c in head_centres():
        out.append(CylShape((0.0045, c[1], c[2]), 0.0125, 0.007, HEAD_FRAME, axis="X"))
        out.append(MeshShape(dome + np.array([0.0, c[1], c[2]]), dome_faces, STEEL, collide=True, visible=False))
        out.append(CylShape((HEAD_RIM_X - 0.002 + 0.0007, c[1], c[2]), HEAD_R, 0.004 + 0.0014, STEEL, collide=False,
                            axis="X"))
        out.append(CylShape((HEAD_TOP_X - 0.0002, c[1], c[2]), 0.0045, 0.0006, (0.35, 0.36, 0.38), collide=False,
                            axis="X"))
    return out


# -- the tray it lies in, on its back, heads toward the head -----------------------------------------------------------
TRAY_C = (0.140, 0.200)
TRAY_L, TRAY_W, TRAY_FLOOR, TRAY_WALL = 0.200, 0.090, 0.004, 0.010
SHAVER_O = np.array([TRAY_C[0] + 0.0545, TRAY_C[1], TOP + TRAY_FLOOR + BODY_T / 2 + 0.001])  # its body's top, at rest


def _tray_shapes() -> list:
    hx, hy, t = TRAY_L / 2, TRAY_W / 2, 0.003
    h = TRAY_FLOOR + TRAY_WALL
    return [BoxShape((0.0, 0.0, TRAY_FLOOR / 2), (TRAY_L, TRAY_W, TRAY_FLOOR), TRAY_WHITE),
            BoxShape((0.0, hy - t / 2, h / 2), (TRAY_L, t, h), TRAY_WHITE),
            BoxShape((0.0, -hy + t / 2, h / 2), (TRAY_L, t, h), TRAY_WHITE),
            BoxShape((hx - t / 2, 0.0, h / 2), (t, TRAY_W, h), TRAY_WHITE),
            BoxShape((-hx + t / 2, 0.0, h / 2), (t, TRAY_W, h), TRAY_WHITE)]


BASE = Body.build("manikin_base", pos=_w((NECK_AXIS[0], NECK_AXIS[1], TOP)), mass=3.0, color=BASE_GREY, fixed=True,
                  shapes=[CylShape((0.0, 0.0, BASE_H / 2), BASE_R, BASE_H, BASE_GREY, collide=False),
                          CylShape((0.0, 0.0, BASE_H + 0.004), 0.060, 0.008, BASE_GREY, collide=False)])
HEAD = Body.build("manikin_head", pos=_w(HEAD_O), mass=HEAD_KG, color=SKIN, init_quat=Q_HEAD, friction=(0.8, 0.7),
                  damping=(0.0, 0.0), solver_iterations=32, shapes=_head_shapes(),
                  joint=Joint(anchor=_w((NECK_AXIS[0], NECK_AXIS[1], TOP + BASE_H)), kind="revolute", axis="Y",
                              limits=(-8.0, 8.0), stiffness=NECK_K, damping=NECK_C))
PATCH = Body.build("skin_patch", pos=_w(HEAD_O), mass=0.005, color=PATCH_SKIN, init_quat=Q_HEAD, friction=(1.2, 1.1),
                   solver_iterations=32, shapes=_patch_shapes(),
                   joint=Joint(anchor=_w(to_world(PATCH_C)), kind="slide", axis="Z", parent="manikin_head",
                               axial=(PATCH_K, 1.5, PATCH_TRAVEL), slide_range=(0.0, PATCH_TRAVEL)))
TRAY = Body.build("shaver_tray", pos=_w((TRAY_C[0], TRAY_C[1], TOP)), mass=0.2, color=TRAY_WHITE, fixed=True,
                  friction=(0.5, 0.45), shapes=_tray_shapes())
SHAVER = Body.build("shaver", pos=_w(SHAVER_O), mass=SHAVER_KG, color=SHAVER_BODY, friction=(0.9, 0.8),
                    solver_iterations=32, shapes=_body_shapes())
UNIT = Body.build("shaver_heads", pos=_w(SHAVER_O + [PIVOT_X, 0.0, 0.0]), mass=UNIT_KG, color=STEEL,
                  friction=(0.35, 0.3), contact_offset=0.0015, solver_iterations=32, shapes=_unit_shapes(),
                  joint=Joint(anchor=_w(SHAVER_O + [PIVOT_X, 0.0, 0.0]), kind="d6", parent="shaver",
                              d6_trans=(("X", -UNIT_TRAVEL, 0.0, UNIT_K, UNIT_C, 0.0),),
                              d6_rot=(("Y", -TILT_MAX, TILT_MAX, TILT_K, TILT_C, 0.0),
                                      ("Z", -TILT_MAX, TILT_MAX, TILT_K, TILT_C, 0.0))))
OBJECTS = [BASE, HEAD, PATCH, TRAY, SHAVER, UNIT]

# ======================================================================================================================
# Scoring
# ======================================================================================================================
CELLS, CELL_N, SOFT = GEO["cells"], GEO["cell_n"], GEO["soft"]
N_CELLS = len(CELLS)
FLAT_COS = math.cos(math.radians(20.0))  # a head lies flat on the skin: within 20 degrees of its normal
ON_FACE = 0.0015  # a cell is under a head's face: within 1.5 mm of the face (either side), inside its 20 mm circle
F_MIN, F_MAX, F_HARD = 0.3, 3.0, 5.0  # the shaving force window (N); harder than F_HARD: the skin is pinched
PUSHED = 0.008  # the head pushed this far (at its cheek) moves a patient's head: 0
DROP = 0.03  # the shaver falls this far with no hand on it: dropped
ZONE_TOUCH = 0.001  # a shaving head this close to the lips, a nostril, the eye or the ear touches it
STAGES = ((0.95, 0.75), (0.70, 0.5), (0.30, 0.25))  # (fraction shaved, score)
STILL_STEPS, STILL_MM = 20, 1.5
CLEAR = 0.05  # both gripper centres this far from the shaver's body: let go
G = 9.81
_DEBUG = os.environ.get("SHAVE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("SHAVE_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}
_HEADS = head_centres()  # the three heads' centres on their rims (unit frame)
_ZONES = {k: v for k, v in GEO["zones"].items()}
_ZONES["ear"] = _ZONES["ear"][:: max(1, len(_ZONES["ear"]) // 600)]
_BEARD_MID = CELLS.mean(axis=0)  # where the head's movement is measured (the middle of the beard area)


def _dome(rho):
    """Height of a head's domed face above its rim plane at distance rho from its centre."""
    return (HEAD_TOP_X - HEAD_RIM_X) * np.clip(1.0 - (rho / HEAD_R) ** 2, 0.0, 1.0)


def under_heads(unit_p, unit_r) -> np.ndarray:
    """Which cells lie under one of the three heads' faces, flat enough: ``unit_p``/``unit_r`` are the head unit's
    pivot and rotation in the head frame."""
    a = unit_r[:, 0]
    hit = np.zeros(N_CELLS, bool)
    flat = (-(CELL_N @ a)) >= FLAT_COS
    for c in _HEADS:
        v = CELLS - (unit_p + unit_r @ c)
        h = v @ a
        rho = _len(v - h[:, None] * a[None, :])
        gap = h - _dome(rho)
        hit |= (rho <= HEAD_R) & (np.abs(gap) <= ON_FACE) & flat
    return hit


def heads_dist(q) -> np.ndarray:
    """Distance (m) of points q (head-unit frame) from the three heads: discs 20 mm across, from 4 mm below their
    rims to the tops of their domes."""
    out = np.full(len(q), 1.0)
    for c in _HEADS:
        dx = np.maximum(np.maximum(HEAD_RIM_X - 0.004 - q[:, 0], q[:, 0] - HEAD_TOP_X), 0.0)
        dr = np.maximum(np.hypot(q[:, 1] - c[1], q[:, 2] - c[2]) - HEAD_R, 0.0)
        out = np.minimum(out, np.hypot(dx, dr))
    return out


def _body_dist(q) -> np.ndarray:
    """Distance of points q (shaver body frame) from the shaver's body (a box round it)."""
    lo = np.array([-BODY_L, -BODY_W / 2, -BODY_T / 2])
    hi = np.array([PIVOT_X, BODY_W / 2, BODY_T / 2])
    return _len(np.maximum(np.maximum(lo - q, q - hi), 0.0))


@on_reset
def _reset() -> None:
    _EP.clear()
    _EP.update(tick=0, counted=-1, shaved=np.zeros(N_CELLS, bool), failed="", hist=[], held_z=None, force=0.0,
               fmax=0.0, pull=0.0, pushed=0.0, cutting=False, prev_unit=None, near=1.0, on_tray=False, held=False,
               zone_d={k: 1.0 for k in ("lips", "nostril", "eye", "ear")})
    _IDS.pop("arms", None)


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("SHAVE_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _pose(base, name, origin):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - origin, rotation(p[3:7])


def _grippers(base) -> list:
    """Each gripper's TCP (position, rotation) and finger opening (the finger joint, m), from its joints."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if (side, "arm") not in _IDS:
            _IDS[(side, "arm")] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[(side, "fingers")] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        pos, rot = ik.tcp_pose(q[_IDS[(side, "arm")]], station.BASES[side])
        out.append((np.asarray(pos, float), np.asarray(rot, float), float(q[_IDS[(side, "fingers")]][0])))
    return out


def _finger_pts(tcp, rot, q) -> np.ndarray:
    """Points over the closed fingers' last 3 cm (a box per finger), world."""
    pts = []
    for x0, x1 in ((q - 0.010, q), (-q, -q + 0.010)):
        for gx in np.linspace(x0, x1, 3):
            for gy in np.linspace(-0.013, 0.013, 4):
                for gz in np.linspace(-0.016, 0.014, 5):
                    pts.append((gx, gy, gz))
    return tcp + np.array(pts) @ rot.T


def _track_inner(env) -> None:
    """Each control step: the head's pose and how far it was pushed, the skin patch's pull, the shaver's pose, its
    head unit's spring (the force) and tilt, which cells pass under a head flat and pressed in the force window (with
    the skin pulled taut where it needs it), what touches the no-touch zones, and where the grippers are."""
    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    hp, hr = _pose(base, "manikin_head", origin)
    pp, _ = _pose(base, "skin_patch", origin)
    sp, sr = _pose(base, "shaver", origin)
    up, ur = _pose(base, "shaver_heads", origin)
    arms = _grippers(base)
    pull = float((hr.T @ (pp - hp))[2])
    pushed = float(_len(to_world(_BEARD_MID) - (hp + hr @ _BEARD_MID)))
    rel = sr.T @ (up - sp)
    comp = max(0.0, PIVOT_X - float(rel[0]))
    axis = ur[:, 0]
    force = UNIT_K * comp - UNIT_KG * G * float(axis[2]) if comp > 0.00005 else 0.0
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(sr[:, 0] @ ur[:, 0])))))
    # the unit in the head frame
    u_h, r_h = hr.T @ (up - hp), hr.T @ ur
    cutting = F_MIN <= force <= F_MAX
    if cutting:
        poses = [(u_h, r_h)]
        prev = _EP["prev_unit"]
        if prev is not None:  # also the poses in between since the last step (a head moving fast still sweeps)
            n = int(min(6, max(1, math.ceil(float(_len(u_h - prev[0])) / 0.003))))
            for k in range(1, n):
                t = k / n
                rr = (1 - t) * prev[1] + t * r_h
                u, _s, vt = np.linalg.svd(rr)
                poses.append(((1 - t) * prev[0] + t * u_h, u @ vt))
        hit = np.zeros(N_CELLS, bool)
        for p_, r_ in poses:
            hit |= under_heads(p_, r_)
        ok = hit & (~SOFT | (pull >= TAUT))
        _EP["shaved"] |= ok
        _EP["prev_unit"] = (u_h, r_h)
    else:
        _EP["prev_unit"] = None
    # rules broken (0 for the rest of the run)
    if not _EP["failed"]:
        if force > F_HARD:
            _EP["failed"] = f"pressed {force:.1f} N (over {F_HARD:.0f} N) at step {_EP['tick']}"
        elif pushed > PUSHED:
            _EP["failed"] = f"the manikin's head was pushed {pushed * 1000:.1f} mm at step {_EP['tick']}"
        else:
            for name, pts in _ZONES.items():
                d = float(heads_dist((hp + pts @ hr.T - up) @ ur).min())
                _EP["zone_d"][name] = min(_EP["zone_d"][name], d)
                if d < ZONE_TOUCH:
                    _EP["failed"] = f"the shaving heads touched the {name} at step {_EP['tick']}"
                    break
            if not _EP["failed"]:
                eye = hp + _ZONES["eye"] @ hr.T
                for side, (tcp, rot, q) in zip(("left", "right"), arms):
                    if float(_min_dist(_finger_pts(tcp, rot, q), eye).min()) < ZONE_TOUCH:
                        _EP["failed"] = f"the {side} gripper touched the eye at step {_EP['tick']}"
    # held / dropped / on the tray / let go
    tcp_in = [(sr.T @ (tcp - sp)) for tcp, _, _ in arms]
    held = any(bool(np.all(np.abs(t - np.array([-BODY_L / 2, 0.0, 0.0])) <= np.array([BODY_L / 2 + 0.006, BODY_W / 2
                                                                                        + 0.006, BODY_T / 2 + 0.006])))
               and q > -0.032 for t, (_, _, q) in zip(tcp_in, arms))
    corners = np.array([[x, y, z] for x in (-BODY_L, PIVOT_X + HEAD_TOP_X) for y in (-BODY_W / 2, BODY_W / 2)
                        for z in (-BODY_T / 2, BODY_T / 2)])
    low = float((sp + corners @ sr.T)[:, 2].min()) - TOP
    if held or _EP["held_z"] is None:
        _EP["held_z"] = low
    elif low < _EP["held_z"] - DROP and not _EP["failed"]:
        _EP["failed"] = f"the shaver was dropped (fell {(_EP['held_z'] - low) * 100:.0f} cm) at step {_EP['tick']}"
    mid = sp + sr @ np.array([-BODY_L / 2 + 0.01, 0.0, 0.0])
    on_tray = bool(abs(mid[0] - TRAY_C[0]) <= TRAY_L / 2 and abs(mid[1] - TRAY_C[1]) <= TRAY_W / 2
                   and low <= TRAY_FLOOR + 0.004)
    near = min(float(_body_dist(sr.T @ (tcp - sp)).min()) for tcp, _, _ in arms)
    _EP["hist"].append(np.concatenate([sp, up]))
    del _EP["hist"][:-STILL_STEPS]
    _EP.update(force=force, fmax=max(_EP["fmax"], force), pull=pull, pushed=max(_EP["pushed"], pushed), tilt=tilt,
               comp=comp, held=held, low=low, on_tray=on_tray, near=near, arms=arms, unit_h=(u_h, r_h), sp=sp,
               cutting=cutting)


def _still() -> bool:
    h = _EP["hist"]
    return len(h) == STILL_STEPS and max(float(np.abs(v - h[-1]).max()) for v in h) * 1000 < STILL_MM


def coverage() -> float:
    return float(_EP["shaved"].mean())


def _stage() -> float:
    if _EP["failed"]:
        return 0.0
    cov = coverage()
    for frac, val in STAGES:
        if cov >= frac:
            return val
    return 0.0


def _done() -> bool:
    return bool(not _EP["failed"] and coverage() >= STAGES[0][0] and _EP["on_tray"] and not _EP["held"]
                and _EP["near"] >= CLEAR and _still())


def _debug() -> None:
    if (_EP["tick"] - 1) % _DEBUG_EVERY:
        return
    u, r = _EP.get("unit_h", (np.zeros(3), np.eye(3)))
    print(f"DBG step={_EP['tick']} cov={coverage():.3f} (soft {float(_EP['shaved'][SOFT].mean()):.3f}, firm "
          f"{float(_EP['shaved'][~SOFT].mean()):.3f}) F={_EP['force']:.2f}N fmax={_EP['fmax']:.2f} "
          f"comp={_EP.get('comp', 0) * 1000:.2f}mm tilt={_EP.get('tilt', 0):.1f} pull={_EP['pull'] * 1000:.2f}mm "
          f"pushed={_EP['pushed'] * 1000:.2f}mm unit_h=({u[0]:.4f},{u[1]:.4f},{u[2]:.4f}) "
          f"ax_h=({r[0, 0]:+.2f},{r[1, 0]:+.2f},{r[2, 0]:+.2f}) held={_EP['held']} low={_EP.get('low', 0) * 1000:.1f}mm "
          f"tray={_EP['on_tray']} near={_EP['near']:.3f} still={_still()} stage={_stage()} "
          f"zone_mm={ {k: round(v * 1000, 1) for k, v in _EP['zone_d'].items()} } "
          f"failed={_EP['failed'] or '-'}", flush=True)
    arms = _EP.get("arms")
    if arms:
        print("DBG   " + " | ".join(f"{s} tcp=({t[0]:.4f},{t[1]:.4f},{t[2] - TOP:.4f}) q={q * 1000:.1f}mm"
                                    for s, (t, _, q) in zip(("left", "right"), arms)), flush=True)
    if "unit_h" in _EP:  # why cells under the heads count or not
        a = r[:, 0]
        near = np.zeros(N_CELLS, bool)
        on = np.zeros(N_CELLS, bool)
        best_gap = 1.0
        for c in _HEADS:
            v = CELLS - (u + r @ c)
            h = v @ a
            rho = _len(v - h[:, None] * a[None, :])
            gap = h - _dome(rho)
            near |= rho <= HEAD_R
            on |= (rho <= HEAD_R) & (np.abs(gap) <= ON_FACE)
            if (rho <= HEAD_R).any():
                best_gap = min(best_gap, float(np.abs(gap[rho <= HEAD_R]).min()))
        flat = (-(CELL_N @ a)) >= FLAT_COS
        print(f"DBG   cells under a head {int(near.sum())}, on its face {int(on.sum())}, flat {int((on & flat).sum())}, "
              f"closest face gap {best_gap * 1000:.2f} mm", flush=True)


def _tally() -> None:
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        if _DEBUG:
            _debug()
            dump = os.environ.get("SHAVE_DUMP")
            if dump and _EP["tick"] % 50 == 0:  # debug only: which points are shaved so far
                np.save(dump, _EP["shaved"])


def score(objects) -> float:
    """0 for the rest of the run once a rule is broken (pressed over 5 N, the head pushed 8 mm, the shaver touching
    the lips, a nostril, the eye or the ear, a gripper touching the eye, the shaver dropped). Otherwise by the
    fraction of the beard area shaved: 0.75 at 95 % or more, 0.5 at 70 %, 0.25 at 30 %; full score (1.0) is
    success()."""
    _tally()
    return _stage()


def success(objects) -> bool:
    """95 % or more shaved, no rule broken, the shaver resting on its tray, let go, both grippers 5 cm or more from
    it, still for 1 s."""
    _tally()
    return _done()


# ======================================================================================================================
# Scripted demo (open loop, from the known layout; it reads nothing from the simulation)
# ======================================================================================================================
# The left gripper takes the shaver from its tray straight down across its body and turns it to face the left side of
# the face. It first shaves the firm skin over the jaw and beside the mouth, where no pull is needed, in four passes;
# then the right gripper's fingertips press the skin patch over the cheekbone and drag it 7 mm up toward the temple
# and hold it there while the left gripper shaves the soft cheek below in four more passes. The right gripper lets go
# of the skin, and the left gripper lays the shaver back in its tray, lets go and moves away. Each pass follows the
# face's surface: the shaver's axis along the skin's inward normal, its heads pressed a few millimetres past the
# skin (the head unit's spring, the arm and the manikin's neck give, which sets the force).
# SHAVE_ONE=1: the same, but the right arm never pulls the skin (it stays at its start pose).
# SHAVE_HARD=1: the jaw and chin are shaved pressing far too hard (the heads aimed 40 mm deeper).
WRONG_ONE = os.environ.get("SHAVE_ONE") == "1"
WRONG_HARD = os.environ.get("SHAVE_HARD") == "1"
GRIP_ROT = grip_rotation(0.0)  # left: straight down, fingers closing across the shaver's width (along world y)
GRIP_X, GRIP_Z = -0.085, 0.004  # where the TCP holds the shaver (shaver frame): 11 cm behind the heads' faces
MAX_UP = 12.0  # the shaver's body is tilted up at most this much (degrees); its head unit pivots the rest
OPEN_SHAVER, GRIP_SHAVER = -0.040, -(BODY_W / 2 - 0.0035)  # finger values: open; 3.5 mm inside each side
HARD_EXTRA = 0.040  # SHAVE_HARD: the jaw and chin's heads aimed this much deeper (m)
APPROACH = 0.025  # passes start and end this far off the skin, along its normal
# The passes, as points on the face seen from the head's centre (azimuth from the face toward the left ear and
# elevation, degrees), the head unit turned so the gripper comes from above (roll 0) or from the robot's left (300),
# and how far past the skin the heads are aimed (m): the head unit's spring, the arm and the neck give, so the force on
# the skin comes out at about 0.5 to 1.5 N.
# Where the arm actually puts the head unit when it shaves is lower and a little further forward than commanded (its
# joints give under the shaver's weight and the push; more with the gripper turned for the cheek): measured once in
# debug runs of this plan, as (azimuth, forward mm, up mm) at the two ends of a pass, and taken off its targets (only
# partly where taking it off would bring the heads nearer the ear or the eye).
FIRM_PASSES = (("jaw and chin", ((20.0, -44.0), (31.0, -44.0), (50.0, -42.0), (70.0, -45.0), (80.0, -40.0),
                                  (26.0, -38.0)), 0.0, 0.013, ((24.0, 7.0, -9.5), (80.0, 0.0, -9.0))),
               ("beside the mouth", ((38.0, -10.0), (37.0, -17.0), (36.0, -24.0)), 0.0, 0.012,
                ((36.0, 5.0, -6.0), (38.0, 5.0, -6.0))))
SOFT_PASSES = (("cheek", ((46.0, -8.0), (68.0, -8.0), (78.0, -14.0), (42.0, -14.0), (37.0, -20.0), (78.0, -20.0),
                          (78.0, -26.0), (35.0, -26.0), (34.0, -32.0), (78.0, -32.0)), 300.0, 0.012,
                ((34.0, 4.0, -15.0), (78.0, 3.0, -20.5))),)
PASS_SPEED = 0.022  # m/s along the skin
WAIT_PULL = 6.5  # s: the left arm waits off the skin after its last firm pass while the right pulls the cheek taut
PARK_SHIFT = np.array([0.0, 0.05, -0.01])  # where it waits: moved this far from the end of that pass (m)
RIGHT_LEAVES = 4.0  # s: after the cheek, the left arm waits while the right lets the skin go and moves away
# the skin patch: the right gripper's closed fingertips press it 12 mm past its surface (along its normal) and drag it
# 11 mm up (the arm sags a few millimetres at that height, and the patch stops at 8 mm)
PATCH_PRESS_AT = np.array([0.040, 0.0, 0.021])  # head frame (the surface point is found along +y from here)
PATCH_DRAG = 0.011
PATCH_PRESS = 0.012
PATCH_TILT, PATCH_ROLL = 50.0, 180.0  # the right gripper's approach: tilted down from the patch's inward normal
PATCH_VIA = np.array([0.140, -0.030, TOP + 0.210])  # the right TCP comes up beside the head here first


def _good_ik(arm: str, pos, rot):
    import itertools

    from yam_arena import ik, station

    best = None
    for sd in itertools.product((-0.4, 0.0, 0.4), (0.8, 1.3, 1.8), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2),
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


def _wp(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(float(t), tuple(float(v) for v in p), float(finger), label, np.asarray(rot, float))


def _ready(arm: str) -> tuple:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return np.asarray(pos, float), rot


def _t(plan: list) -> float:
    return float(sum(w.duration for w in plan))


def _hold(plan: list, until: float, label: str) -> None:
    w = plan[-1]
    if until > _t(plan) + 1e-6:
        plan.append(Waypoint(until - _t(plan), w.pos, w.finger, label, w.yaw))


def _surface(th: float, ph: float) -> tuple:
    """The skin's point and outward normal (head frame) seen from the head's centre at (azimuth, elevation)."""
    d = sphere_dirs(np.radians([th]), np.radians([ph])).reshape(-1, 3)
    p = ray_surface(head_sdf, HEAD_CENTRE, d, 0.16, samples=60, t_min=0.04)[0]
    return p, _normals(head_sdf, p[None])[0]


def _dome_samples() -> np.ndarray:
    pts = []
    for c in _HEADS:
        for r in np.linspace(0.0, HEAD_R, 5):
            for k in range(12 if r > 0 else 1):
                a = 2 * math.pi * k / 12
                pts.append((HEAD_RIM_X + float(_dome(np.array([r]))[0]), c[1] + r * math.cos(a),
                            c[2] + r * math.sin(a)))
    return np.array(pts)


_DOMES = _dome_samples()


def _shaver_rot(s_w: np.ndarray, roll: float) -> np.ndarray:
    """Shaver rotation (world) with its axis along s_w, turned about it so that the gripper (which comes at the
    shaver's front face) approaches from the most downward direction turned ``roll`` degrees about the axis."""
    down = np.array([0.0, 0.0, -1.0])
    a0 = down - (down @ s_w) * s_w
    a0 /= np.linalg.norm(a0)
    b0 = np.cross(s_w, a0)
    a = math.cos(math.radians(roll)) * a0 + math.sin(math.radians(roll)) * b0
    z = -a
    return np.column_stack([s_w, np.cross(z, s_w), z])


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector u onto unit vector v."""
    c = float(np.clip(u @ v, -1.0, 1.0))
    axis = np.cross(u, v)
    sn = float(np.linalg.norm(axis))
    if sn < 1e-9:
        return np.eye(3)
    k = axis / sn
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sn * kx + (1.0 - c) * (kx @ kx)


def shaver_pose(th: float, ph: float, roll: float, depth: float) -> dict:
    """Where the shaver goes to shave the skin at (th, ph): the head unit flat on the skin there (its axis along the
    skin's inward normal), the body tilted up at most MAX_UP (the unit's pivot takes the rest, up to 20 degrees, so
    the shaver's tail stays clear of the table), the heads ``depth`` past the skin along the body's axis (negative:
    off it). Returns the body's origin and rotation and the TCP pose (world), and the unit's pivot and rotation in
    the head frame (as the scorer sees them when the arm follows exactly)."""
    p, n = _surface(th, ph)
    s_unit = R_HEAD @ -n
    el = math.asin(max(-1.0, min(1.0, float(s_unit[2]))))
    lim = math.radians(MAX_UP)
    if el > lim:
        hz = s_unit[:2] / max(1e-9, float(np.linalg.norm(s_unit[:2])))
        s_body = np.array([hz[0] * math.cos(lim), hz[1] * math.cos(lim), math.sin(lim)])
    else:
        s_body = s_unit
    r_body = _shaver_rot(s_body, roll)
    ang = math.acos(max(-1.0, min(1.0, float(s_body @ s_unit))))
    if ang > math.radians(TILT_MAX - 1.0):  # beyond the pivot's reach: the unit tilts as far as it can
        k = np.cross(s_body, s_unit)
        k /= np.linalg.norm(k)
        a = math.radians(TILT_MAX - 1.0)
        s_unit = s_body * math.cos(a) + np.cross(k, s_body) * math.sin(a)
    r_unit = _rot_between(s_body, s_unit) @ r_body
    r_unit_h = R_HEAD.T @ r_unit
    s_h = R_HEAD.T @ s_unit

    def gap(t):
        return float(head_sdf((p - s_h * t) + _DOMES @ r_unit_h.T).min())

    lo, hi = 0.0, 0.05  # the pivot's distance behind p (along the unit's axis) at first contact of the heads
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if gap(mid) >= 0 else (mid, hi)
    pivot_h = p - s_h * hi
    pivot = to_world(pivot_h) + s_body * depth
    origin = pivot - s_body * PIVOT_X
    tcp = origin + r_body @ np.array([GRIP_X, 0.0, GRIP_Z])
    return dict(origin=origin, r_body=r_body, tcp=tcp, r_tcp=r_body @ GRIP_ROT, pivot_h=pivot_h, r_unit_h=r_unit_h,
                n=n, p=p)


def _shaver_tcp(th: float, ph: float, roll: float, depth: float) -> tuple:
    ps = shaver_pose(th, ph, roll, depth)
    return ps["tcp"], ps["r_tcp"], ps["n"]


def _sag(th: float, sag) -> np.ndarray:
    """The measured offset of the head unit from its target (world, m) at azimuth th, linear between the ends."""
    (t0, f0, u0), (t1, f1, u1) = sag
    w = min(1.0, max(0.0, (th - t0) / (t1 - t0)))
    return R_HEAD @ (np.array([f0 + (f1 - f0) * w, 0.0, u0 + (u1 - u0) * w]) * MM)


def _pass(plan: list, knots, roll: float, press: float, label: str, sag) -> None:
    """One pass along the skin: in from off the skin at its start, along it pressed, out at its end."""
    pts = []
    for (t0, p0), (t1, p1) in zip(knots[:-1], knots[1:]):
        k = max(1, int(round(max(abs(t1 - t0), abs(p1 - p0)) / 4.0)))
        pts += [(t0 + (t1 - t0) * i / k, p0 + (p1 - p0) * i / k) for i in range(k)]
    pts.append(knots[-1])
    first, rot0, _ = _shaver_tcp(*pts[0], roll, -APPROACH)
    plan.append(_wp(2.5, first - _sag(pts[0][0], sag), GRIP_SHAVER, f"off the skin at the start of the {label} pass",
                    rot0))
    prev = None
    for i, (th, ph) in enumerate(pts):
        tcp, rot, _ = _shaver_tcp(th, ph, roll, press)
        tcp = tcp - _sag(th, sag)
        dt = 1.2 if prev is None else max(0.25, float(np.linalg.norm(tcp - prev)) / PASS_SPEED)
        plan.append(_wp(dt, tcp, GRIP_SHAVER, f"{label} pass" + (": heads on the skin" if prev is None else ""), rot))
        if prev is None:
            plan.append(_wp(0.4, tcp, GRIP_SHAVER, f"{label} pass: settle", rot))
        prev = tcp
    last, rot1, _ = _shaver_tcp(*pts[-1], roll, -APPROACH)
    plan.append(_wp(1.0, last - _sag(pts[-1][0], sag), GRIP_SHAVER, f"off the skin at the end of the {label} pass",
                    rot1))


def _shaver_plan() -> tuple:
    """The left arm: take the shaver, shave the firm skin, wait for the pull, shave the cheek, lay it back. Returns
    the plan and the times the cheek passes start and end."""
    grip = SHAVER_O + np.array([GRIP_X, 0.0, GRIP_Z - 0.001])
    out = _lead_in("left", grip + [0.0, 0.0, 0.06], GRIP_ROT, OPEN_SHAVER, "above the shaver")
    out += [_wp(1.2, grip, OPEN_SHAVER, "fingers either side of the shaver", GRIP_ROT),
            _wp(0.3, grip, OPEN_SHAVER, "settle", GRIP_ROT),
            _wp(1.0, grip, GRIP_SHAVER, "take the shaver", GRIP_ROT),
            _wp(0.6, grip, GRIP_SHAVER, "settle", GRIP_ROT),
            _wp(1.2, grip + [0.0, 0.0, 0.07], GRIP_SHAVER, "lift it out of the tray", GRIP_ROT)]
    for i, (label, knots, roll, press, sag) in enumerate(FIRM_PASSES):
        _pass(out, knots, roll, press + (HARD_EXTRA if WRONG_HARD and i == 0 else 0.0), label, sag)
    t_firm = _t(out)
    w = out[-1]
    out.append(_wp(1.5, np.asarray(w.pos) + PARK_SHIFT, GRIP_SHAVER, "move clear of the other arm", w.yaw))
    _hold(out, t_firm + WAIT_PULL, "wait while the cheek is pulled taut")
    for label, knots, roll, press, sag in SOFT_PASSES:
        _pass(out, knots, roll, press, label, sag)
    w = out[-1]
    out.append(_wp(1.5, np.asarray(w.pos) + PARK_SHIFT, GRIP_SHAVER, "move clear of the other arm", w.yaw))
    t_done = _t(out)
    _hold(out, t_done + RIGHT_LEAVES, "wait for the other arm to move away")
    up = SHAVER_O + np.array([GRIP_X, 0.0, GRIP_Z + 0.07])
    put = SHAVER_O + np.array([GRIP_X, 0.0, GRIP_Z + 0.0005])
    out += [_wp(3.0, up, GRIP_SHAVER, "carry it back over the tray", GRIP_ROT),
            _wp(1.5, put, GRIP_SHAVER, "lower it into the tray", GRIP_ROT),
            _wp(0.5, put, GRIP_SHAVER, "settle", GRIP_ROT),
            _wp(0.8, put, OPEN_SHAVER, "let go", GRIP_ROT),
            _wp(1.0, put + [0.0, 0.0, 0.07], OPEN_SHAVER, "rise", GRIP_ROT),
            _wp(1.5, np.array([0.16, 0.25, TOP + 0.16]), OPEN_SHAVER, "move away", GRIP_ROT)]
    return out, t_firm, t_done


def _patch_frame(n_w: np.ndarray) -> np.ndarray:
    """The right gripper's rotation pressing the patch: its approach along the inward normal tilted down."""
    a = -n_w / np.linalg.norm(n_w)
    down = np.array([0.0, 0.0, -1.0])
    t = down - (down @ a) * a
    t /= np.linalg.norm(t)
    a2 = math.cos(math.radians(PATCH_TILT)) * a + math.sin(math.radians(PATCH_TILT)) * t
    ref = np.array([0.0, 1.0, 0.0])
    c0 = np.cross(ref, a2)
    c0 /= np.linalg.norm(c0)
    w0 = np.cross(a2, c0)
    r = math.radians(PATCH_ROLL)
    c = math.cos(r) * c0 + math.sin(r) * w0
    return np.column_stack([c, np.cross(a2, c), a2])


def _patch_plan(t_firm: float, t_end: float) -> list:
    """The right arm: wait, press the patch's upper part with the closed fingertips, drag it up (done 2 s after the
    left arm has left the skin from its last firm pass), hold it until the cheek is done, let go."""
    p = ray_surface(head_sdf, PATCH_PRESS_AT, np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0]
    n = R_HEAD @ _normals(head_sdf, p[None])[0]
    rot = _patch_frame(n)
    a = rot[:, 2]
    n_in = -np.array([n[0], n[1], 0.0]) / max(1e-9, float(np.hypot(n[0], n[1])))  # into the face, level
    touch = to_world(p) - a * 0.013  # the pad tips 13 mm past the TCP, on the skin
    pressed = touch + n_in * PATCH_PRESS  # pushed into the skin along its normal
    up = np.array([0.0, 0.0, PATCH_DRAG])
    pos, r0 = _ready("right")
    out = [Waypoint(max(0.5, t_firm - 1.0), tuple(float(v) for v in pos), 0.0, "wait with the fingers closed", r0)]
    off = touch - n_in * 0.03 - [0.0, 0.0, 0.01]
    out += _lead_in("right", PATCH_VIA, rot, 0.0, "up beside the head", seconds=3.0)
    out += [_wp(1.5, off, 0.0, "toward the skin over the cheekbone", rot),
            _wp(1.5, touch, 0.0, "fingertips on the skin", rot),
            _wp(1.0, pressed, 0.0, "press", rot),
            _wp(1.5, pressed + up, 0.0, "drag the skin up toward the temple", rot)]
    _hold(out, t_end, "hold the skin taut")
    out += [_wp(1.0, off + up, 0.0, "let the skin go", rot),
            _wp(1.5, PATCH_VIA, 0.0, "away from the head", rot),
            _wp(2.0, pos, 0.0, "back", r0)]
    return out


def demo():
    left, t_firm, t_done = _shaver_plan()
    if WRONG_ONE:
        pos, rot = _ready("right")
        return {"left": left, "right": [Waypoint(_t(left), tuple(float(v) for v in pos), 0.0, "stay put", rot)]}
    right = _patch_plan(t_firm, t_done)
    _hold(right, _t(left), "stay clear")
    return {"left": left, "right": right}
