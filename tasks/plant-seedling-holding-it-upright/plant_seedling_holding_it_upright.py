"""On the station table, in front of the arms: a low wooden potting bench with a square planting hole in it. A seedling
stands in a small nursery cell on the bench, and an open-fronted tray of soil crumbs lies on the bench with its front at
the hole's edge.

Task (instruction): plant the seedling in the middle of the hole, upright, its root plug on the bottom of the hole; fill
the hole with the tray's soil up to the top of the plug all round; leave the seedling standing on its own. The
instruction does not say how. Gardening guides: "Dig each planting hole a little wider than the roots of the new
plant." "Plant at the same depth that they were in the container." (UMass Extension); "Refill with your excavated soil"
and "Take care not to bury the base of the stems." (RHS). The seedling's root plug has a round bottom: on the hole's flat
floor the seedling falls over unless it is held, and it stands by itself only once soil fills the hole around the plug.
One hand holds it upright while the other tips the soil in.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- bench: fixed, 5 cm tall, x 0.16 to 0.44, y -0.20 to 0.16, with a planting hole 8 x 8 cm and 4.5 cm deep (its floor
  5 mm above the table) centred at (0.32, 0.03).
- nursery cell: fixed, 4.4 x 4.4 cm inside and 4.5 cm deep, on the bench at (0.24, 0.08).
- seedling: a root plug 4 cm across and 4.5 cm tall (as tall as the hole is deep) with a round bottom, a stem 0.8 cm
  square up to 15 cm, and three leaves 5 cm long at its top; 80 g.
- tray: 9 cm wide, 13 cm deep, 3.5 cm walls, open at the front, with ribs 1 mm high across its floor (one in front of
  each row of crumbs and one along the open front) and a flat handle 2 x 4 x 1.2 cm out of the top of its back wall;
  80 g; its open front lies on the bench 3 mm short of the hole's near (right) edge.
- soil: 140 crumbs, 1.2 cm balls of 1.09 g (153 g, 127 cm3 of crumbs), in four layers at the back of the tray. The hole
  holds 288 cm3 and the plug displaces about 48 cm3. The settled surface at four points determines coverage.

Soil level: measured 1 cm in from the middle of each of the hole's four sides (front, back, left and right of the plug),
as the height at which a flat-ended rod 1 cm across, set down on the soil, rests (the top of the highest crumb under its
end; crumbs still moving, in the tray or within 3 cm of a gripper do not count). Full score: the seedling upright (within
10 degrees), its plug's axis within 1 cm of the hole's centre and its bottom on the hole's floor (within 5 mm); the soil
at all four spots at most 1 cm below the rim; at most 1.5 cm of soil on the plug's top (measured 1.2 cm from the stem, on
four sides); both grippers 8 cm or more from the seedling's axis; the seedling still (moved under 1.5 mm and turned under
1 degree over 1 s), for 1 s. Partial credit, the best reached during the run, held or not: 0.5 planted and filled as
above; 0.25 planted as above with the soil at all four spots at least halfway up the hole (2.25 cm).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step,  # noqa: E402
                     rotation)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
BENCH_X, BENCH_Y, BENCH_H = (0.16, 0.44), (-0.20, 0.16), 0.05
HOLE_XY, HOLE_HALF, HOLE_FLOOR = (0.32, 0.03), 0.04, 0.005  # 8 x 8 cm, 4.5 cm deep
CELL_XY, CELL_IN, CELL_T, CELL_H = (0.24, 0.08), 0.044, 0.003, 0.045
BENCH_TOP = TOP + BENCH_H  # the hole's rim
FLOOR_TOP = TOP + HOLE_FLOOR
DEPTH = BENCH_H - HOLE_FLOOR

# -- the seedling (origin: the bottom of its root plug) ---------------------------------------------------------------
PLUG_R, PLUG_H = 0.020, 0.045  # a round bottom (a sphere) under a short cylinder: the plug is as tall as the hole is deep
STEM_W, STEM_TOP = 0.008, 0.150
LEAF = (0.050, 0.018, 0.003)
LEAF_Z = 0.147
SEEDLING_KG = 0.08
PLUG_C, STEM_C, LEAF_C = (0.36, 0.25, 0.15), (0.33, 0.55, 0.22), (0.30, 0.64, 0.26)

# -- the tray (origin: the middle of its floor's underside; its open front towards +y) --------------------------------
TRAY_W, TRAY_L, TRAY_WALL, TRAY_FLOOR, TRAY_T = 0.090, 0.130, 0.035, 0.004, 0.003
TRAY_TOP = TRAY_FLOOR + TRAY_WALL
RIB = (0.002, 0.001)  # low ribs across the floor, one in front of each row of crumbs and one along the open front:
# they keep the crumbs from rolling out while the tray lies flat and let them all roll out once it is tipped ~34 degrees
TAB = (0.020, 0.040, 0.012)  # the handle: a flat tab out of the top of the back wall (x, y, z)
TAB_C = (0.0, -TRAY_L / 2 - TAB[1] / 2, TRAY_TOP - TAB[2] / 2)
TRAY_GRIP = (0.0, -TRAY_L / 2 - 0.025, TRAY_TOP - TAB[2] / 2)  # the TCP on the tab: 2.5 cm behind the wall
TRAY_KG = 0.08
SHORT = 0.003  # the tray's open front lies this far short of the hole's near edge, on the bench
TRAY_XY = (HOLE_XY[0], HOLE_XY[1] - HOLE_HALF - SHORT - TRAY_L / 2)
TRAY_C, HANDLE_C = (0.20, 0.42, 0.62), (0.15, 0.15, 0.16)

# -- the soil ---------------------------------------------------------------------------------------------------------
CRUMB, CRUMB_KG = 0.012, 0.00109  # 1.2 cm balls, 1.2 g/cm3
ROW = CRUMB + 0.0015  # the rows' spacing along the tray
LAYERS = ((6, 8, 0.0), (5, 7, 0.5), (6, 7, 0.0), (5, 3, 0.5))  # (across, rows, row offset) from the floor up: 140
N_CRUMBS = int(os.environ.get("PLANT_CRUMBS", "140"))  # fewer: a loophole check (too little soil)
CRUMB_MU = 0.3
CRUMB_ROLL = 3.0  # angular damping: crumbs roll a little and stop, they do not run on like marbles
SOIL_C, BENCH_C, FLOOR_C, CELL_C = (0.40, 0.29, 0.19), (0.66, 0.52, 0.36), (0.28, 0.20, 0.14), (0.14, 0.14, 0.15)


def _bench() -> list:
    (x0, x1), (y0, y1) = BENCH_X, BENCH_Y
    hx, hy = HOLE_XY
    hx0, hx1, hy0, hy1 = hx - HOLE_HALF, hx + HOLE_HALF, hy - HOLE_HALF, hy + HOLE_HALF
    z = TOP + BENCH_H / 2

    def block(name, xa, xb, ya, yb):
        return Box(name, pos=((xa + xb) / 2, (ya + yb) / 2, z), size=(xb - xa, yb - ya, BENCH_H), color=BENCH_C)

    return [block("bench_near", x0, hx0, y0, y1), block("bench_far", hx1, x1, y0, y1),
            block("bench_right", hx0, hx1, y0, hy0), block("bench_left", hx0, hx1, hy1, y1),
            Box("hole_floor", pos=(hx, hy, TOP + HOLE_FLOOR / 2), size=(2 * HOLE_HALF, 2 * HOLE_HALF, HOLE_FLOOR),
                color=FLOOR_C)]


def _cell() -> list:
    cx, cy = CELL_XY
    a, t, z = CELL_IN / 2 + CELL_T / 2, CELL_T, BENCH_TOP + CELL_H / 2
    long = CELL_IN + 2 * CELL_T
    return [Box("cell_front", pos=(cx + a, cy, z), size=(t, long, CELL_H), color=CELL_C),
            Box("cell_back", pos=(cx - a, cy, z), size=(t, long, CELL_H), color=CELL_C),
            Box("cell_left", pos=(cx, cy + a, z), size=(CELL_IN, t, CELL_H), color=CELL_C),
            Box("cell_right", pos=(cx, cy - a, z), size=(CELL_IN, t, CELL_H), color=CELL_C)]


def _seedling() -> Body:
    shapes = [SphereShape((0.0, 0.0, PLUG_R), PLUG_R, PLUG_C),
              CylShape((0.0, 0.0, (PLUG_R + PLUG_H) / 2), PLUG_R, PLUG_H - PLUG_R, PLUG_C),
              BoxShape((0.0, 0.0, (PLUG_H + STEM_TOP) / 2), (STEM_W, STEM_W, STEM_TOP - PLUG_H), STEM_C)]
    shapes += [BoxShape((0.0, 0.0, LEAF_Z), LEAF, LEAF_C, yaw=k * math.pi / 3) for k in range(3)]
    return Body.build("seedling", pos=(*CELL_XY, BENCH_TOP + 0.0005), mass=SEEDLING_KG, shapes=shapes,
                      friction=(0.8, 0.7), damping=(0.05, 0.2), color=STEM_C)


def _tray() -> Body:
    hw, hl = TRAY_W / 2, TRAY_L / 2
    shapes = [BoxShape((0.0, 0.0, TRAY_FLOOR / 2), (TRAY_W, TRAY_L, TRAY_FLOOR), TRAY_C),
              BoxShape((hw - TRAY_T / 2, 0.0, TRAY_FLOOR + TRAY_WALL / 2), (TRAY_T, TRAY_L, TRAY_WALL), TRAY_C),
              BoxShape((-hw + TRAY_T / 2, 0.0, TRAY_FLOOR + TRAY_WALL / 2), (TRAY_T, TRAY_L, TRAY_WALL), TRAY_C),
              BoxShape((0.0, -hl + TRAY_T / 2, TRAY_FLOOR + TRAY_WALL / 2), (TRAY_W, TRAY_T, TRAY_WALL), TRAY_C),
              BoxShape(TAB_C, TAB, HANDLE_C)]
    y = -hl + TRAY_T + 0.0015 + CRUMB / 2 + ROW / 2
    while y < hl - 0.006:
        shapes.append(BoxShape((0.0, y, TRAY_FLOOR + RIB[1] / 2), (TRAY_W - 2 * TRAY_T, RIB[0], RIB[1]), TRAY_C))
        y += ROW
    shapes.append(BoxShape((0.0, hl - RIB[0] / 2, TRAY_FLOOR + RIB[1] / 2), (TRAY_W, RIB[0], RIB[1]), TRAY_C))
    return Body.build("tray", pos=(*TRAY_XY, BENCH_TOP + 0.0005), mass=TRAY_KG, shapes=shapes, friction=(0.7, 0.6),
                      color=TRAY_C)


def _crumb_spots() -> list[tuple[float, float, float]]:
    """Where the crumbs start, in the tray's frame: rows on the floor between the ribs from the back wall forward
    (1.9 cm of the floor left free at the front), and every crumb above resting in a hollow between four below."""
    px = (TRAY_W - 2 * TRAY_T - 0.002) / LAYERS[0][0]
    rise = math.sqrt(CRUMB ** 2 - (px / 2) ** 2 - (ROW / 2) ** 2) + 0.0003
    y0 = -TRAY_L / 2 + TRAY_T + 0.0015 + CRUMB / 2
    z0 = TRAY_FLOOR + 0.0005 + CRUMB / 2
    return [((i - (nx - 1) / 2) * px, y0 + (j + off) * ROW, z0 + k * rise)
            for k, (nx, ny, off) in enumerate(LAYERS) for j in range(ny) for i in range(nx)]


def _crumbs() -> list[Body]:
    tx, ty = TRAY_XY
    shape = SphereShape((0.0, 0.0, 0.0), CRUMB / 2, SOIL_C)
    return [Body.build(f"crumb_{i:03d}", pos=(tx + x, ty + y, BENCH_TOP + 0.0005 + z), mass=CRUMB_KG, color=SOIL_C,
                       shapes=[shape], friction=(CRUMB_MU, CRUMB_MU - 0.05), damping=(0.05, CRUMB_ROLL))
            for i, (x, y, z) in enumerate(_crumb_spots()[:N_CRUMBS])]


SEEDLING, TRAY = _seedling(), _tray()
CRUMBS = _crumbs()
CRUMB_NAMES = [c.name for c in CRUMBS]
OBJECTS = [*_bench(), *_cell(), SEEDLING, TRAY, *CRUMBS]

# -- scoring ----------------------------------------------------------------------------------------------------------
UPRIGHT = math.radians(10)  # the seedling's axis within 10 degrees of the vertical
CENTRED = 0.01  # its plug's axis within 1 cm of the hole's centre
DOWN = 0.005  # the plug's lowest point at most 5 mm above the hole's floor
SPOT_IN = 0.01  # the four measuring spots: 1 cm in from the middle of each side of the hole
PROBE_R = 0.005  # the rod set down on the soil is 1 cm across
FILLED = DEPTH - 0.01  # full: the soil at most 1 cm below the rim at every spot (3.5 cm above the floor)
HALF = DEPTH / 2  # 0.25: the soil at least halfway up at every spot (2.25 cm above the floor)
COVER, COVER_R = 0.015, 0.012  # at most 1.5 cm of soil on the plug's top, measured 1.2 cm from the stem
CLEAR = 0.08  # on its own: both grippers' centres 8 cm or more from the seedling's axis (open fingers reach ~5.5 cm)
LYING = 0.05  # a crumb moving slower than this (m/s), outside the tray and 3 cm or more from a gripper, is soil
STILL_STEPS, STILL_MM, STILL_DEG = 20, 1.5, 1.0  # still: over the last 1 s, moved < 1.5 mm and turned < 1 degree
_SPOTS = np.array([[HOLE_XY[0] + s * (HOLE_HALF - SPOT_IN), HOLE_XY[1]] for s in (1, -1)]
                  + [[HOLE_XY[0], HOLE_XY[1] + s * (HOLE_HALF - SPOT_IN)] for s in (1, -1)])  # front, back, left, right
_COVER_AT = np.array([[COVER_R, 0.0], [-COVER_R, 0.0], [0.0, COVER_R], [0.0, -COVER_R]])
_DEBUG = os.environ.get("PLANT_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("PLANT_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict[str, np.ndarray] = {}
_ARM_IDS: dict[str, list[int]] = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, hist=[], tally={})
    _TCP.clear()


_reset()


@on_step
def _track_grippers(env) -> None:
    """Each gripper's centre (TCP) from its arm's joint positions (yam_arena's own URDF kinematics)."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0, _ARM_IDS[side]]
        _TCP[side] = np.asarray(ik.tcp_pose(q, station.BASES[side])[0], float)


def _in_tray(o, pos: np.ndarray) -> np.ndarray:
    """Which of the points lie inside the tray (over its floor, between its walls, below their tops)."""
    t = o["tray"]
    loc = (pos - np.asarray(t.pos, float)) @ rotation(t.quat_xyzw)
    return ((np.abs(loc[:, 0]) < TRAY_W / 2) & (np.abs(loc[:, 1]) < TRAY_L / 2) & (loc[:, 2] > 0.0)
            & (loc[:, 2] < TRAY_TOP + 0.01))


def _lying_soil(o) -> np.ndarray:
    """The centres of the crumbs that lie in the hole's footprint as soil: slower than LYING, not in the tray, 3 cm or
    more from both grippers."""
    pos = np.array([np.asarray(o[n].pos, float) for n in CRUMB_NAMES])
    speed = np.array([float(np.linalg.norm(o[n].lin_vel)) for n in CRUMB_NAMES])
    keep = ((np.abs(pos[:, 0] - HOLE_XY[0]) < HOLE_HALF) & (np.abs(pos[:, 1] - HOLE_XY[1]) < HOLE_HALF)
            & (speed < LYING) & ~_in_tray(o, pos))
    for tcp in _TCP.values():
        keep &= np.linalg.norm(pos - tcp, axis=1) >= 0.03
    return pos[keep]


def _soil_height(soil: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The soil surface's height above the hole's floor at each point of ``xy`` (K x 2): where a rod 1 cm across,
    set down there, rests on the highest crumb under its end; 0 where there is none."""
    if len(soil) == 0:
        return np.zeros(len(xy))
    r = CRUMB / 2
    centre_distance = np.linalg.norm(xy[:, None, :] - soil[None, :, :2], axis=2)
    distance_to_probe = np.maximum(centre_distance - PROBE_R, 0.0)
    height_sq = r * r - distance_to_probe ** 2
    top = np.where(height_sq >= 0, soil[None, :, 2] + np.sqrt(np.maximum(height_sq, 0.0)), -np.inf)
    return np.maximum(top.max(axis=1) - FLOOR_TOP, 0.0)


def _axis_distance(point, a, b) -> float:
    t = float(np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * (b - a))))


def _tally(o) -> dict:
    """Everything the score needs, worked out once per control step; also keeps the best stage reached."""
    if _EP["counted"] == _EP["tick"]:
        return _EP["tally"]
    _EP["counted"] = _EP["tick"]
    s = o["seedling"]
    p, r = np.asarray(s.pos, float), rotation(s.quat_xyzw)
    tilt = math.acos(max(-1.0, min(1.0, float(r[2, 2]))))
    ball = p + r[:, 2] * PLUG_R  # the round bottom's centre
    low = float(ball[2] - PLUG_R - FLOOR_TOP)  # the plug's lowest point above the floor
    off = math.hypot(ball[0] - HOLE_XY[0], ball[1] - HOLE_XY[1])
    planted = bool(tilt <= UPRIGHT and off <= CENTRED and low <= DOWN)
    soil = _lying_soil(o)
    levels = _soil_height(soil, _SPOTS)
    top = p + r[:, 2] * PLUG_H  # the middle of the plug's top
    cover = max(0.0, float(_soil_height(soil, top[:2] + _COVER_AT).max()) - float(top[2] - FLOOR_TOP))
    filled = bool(levels.min() >= FILLED and cover <= COVER)
    half = bool(levels.min() >= HALF)
    tip = p + r[:, 2] * STEM_TOP
    near = min((_axis_distance(t, ball, tip) for t in _TCP.values()), default=0.0)
    alone = bool(len(_TCP) == 2 and near >= CLEAR)
    hist = _EP["hist"]
    hist.append((p.copy(), r[:, 2].copy()))
    del hist[:-STILL_STEPS]
    still = bool(len(hist) == STILL_STEPS and all(
        np.linalg.norm(q - p) * 1000 < STILL_MM
        and math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, r[:, 2])))))) < STILL_DEG for q, z in hist))
    stage = 0.5 if planted and filled else 0.25 if planted and half else 0.0
    _EP["best"] = max(_EP["best"], stage)
    t = dict(tilt=tilt, low=low, off=off, planted=planted, levels=levels, cover=cover, filled=filled, half=half,
             near=near, alone=alone, still=still, stage=stage, n_soil=len(soil),
             done=bool(planted and filled and alone and still))
    _EP["tally"] = t
    return t


def _debug(o, t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    tray = o["tray"]
    ttilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation(tray.quat_xyzw)[2, 2])))))
    pos = np.array([np.asarray(o[n].pos, float) for n in CRUMB_NAMES])
    in_tray = int(_in_tray(o, pos).sum())
    print(f"DBG step={_EP['step']} tilt={math.degrees(t['tilt']):.1f} low={t['low'] * 1000:.0f}mm "
          f"off={t['off'] * 1000:.0f}mm planted={t['planted']} levels(F,B,L,R)={np.round(t['levels'] * 1000).astype(int).tolist()}mm "
          f"cover={t['cover'] * 1000:.0f}mm soil_in_hole={t['n_soil']} in_tray={in_tray} tray_tilt={ttilt:.0f} "
          f"near={t['near']:.3f} alone={t['alone']} still={t['still']} stage={t['stage']} best={_EP['best']} "
          f"done={t['done']}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)
    if (_EP["step"] - 1) % (5 * _DEBUG_EVERY) == 0:  # the soil surface over the hole, 1 cm grid, outside the plug
        g = np.arange(-HOLE_HALF + 0.005, HOLE_HALF, 0.01)
        pts = np.array([[HOLE_XY[0] + a, HOLE_XY[1] + b] for a in g[::-1] for b in g[::-1]])
        h = _soil_height(_lying_soil(o), pts)
        sd = np.asarray(o["seedling"].pos, float)
        out = np.hypot(pts[:, 0] - sd[0], pts[:, 1] - sd[1]) > PLUG_R + 0.003
        print(f"DBG   soil surface outside the plug: mean {h[out].mean() * 1000:.0f} mm, min {h[out].min() * 1000:.0f} mm "
              f"(rows far to near, columns left to right):", flush=True)
        for row in (h * 1000).round().astype(int).reshape(len(g), len(g)):
            print("DBG     " + " ".join(f"{v:3d}" for v in row), flush=True)


def score(objects) -> float:
    """The best stage reached during the run, held or not: 0.25 the seedling upright in the middle of the hole with its
    plug on the floor and soil at least halfway up all round; 0.5 that with the soil within 1 cm of the rim all round
    and at most 1.5 cm of soil on the plug's top. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(objects, t)
    return _EP["best"]


def success(objects) -> bool:
    """Planted and filled as for 0.5, both grippers 8 cm or more from the seedling, the seedling still for 1 s."""
    return _tally(objects)["done"]


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand takes the seedling's stem from its own side, level, lifts it out of the cell, carries it over the hole,
# lowers it until the plug sits on the hole's floor and holds it upright there. The right hand takes the tray's handle
# from above and lifts the tray's back, so that the tray turns about its front edge at the hole's rim and the crumbs
# roll into the hole all round the plug; it lowers the tray and lets go. Then the left hand lets go and draws back.
# PLANT_LET_GO=1 runs it the wrong way: the left hand lets go as soon as the plug is on the floor, before the soil.
# PLANT_HOVER=1 holds the plug 1.5 cm above the floor while the soil goes in (a loophole check: planted too high).
# PLANT_TILT=<degrees> tips the tray less (a loophole check: some soil in the hole, not filled all round).

WRONG = os.environ.get("PLANT_LET_GO") == "1"
HOVER = 0.015 if os.environ.get("PLANT_HOVER") == "1" else 0.0


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


STEM_ROT = _rx(-math.pi / 2) @ grip_rotation(-math.pi / 2)  # left, level, pointing -y, fingers closing front and back
TRAY_ROT = grip_rotation(math.pi / 2)  # right, from above, fingers closing along x (the tray turns between them)
OPEN = -0.02
STEM_GRIP = -0.002  # 4 mm between the fingers round the 8 mm stem
TAB_GRIP = -0.006  # 1.2 cm between the fingers round the 2 cm tab
STEM_UP = 0.115  # grip the stem this far above the plug's bottom
LIFT = 0.055  # lift the seedling this much: the plug clears the cell's 4.5 cm walls
TILT = math.radians(float(os.environ.get("PLANT_TILT", "45")))  # commanded; the tray reaches ~37 degrees
PIVOT = np.array([HOLE_XY[0], TRAY_XY[1] + TRAY_L / 2, BENCH_TOP])  # the tray's front edge, which it turns about
HANDLE_AT = np.array([TRAY_XY[0], TRAY_XY[1] + TRAY_GRIP[1], BENCH_TOP + 0.0005 + TRAY_GRIP[2]])
BACK = 0.10  # after letting go the left hand draws back this far along +y


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    for sd in itertools.product((0.0,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2), (-1.5, 0.0, 1.5)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=station.BASES[arm], iters=150)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
    """From READY along a straight line in joint space (the big turn of the wrist stays on one IK branch)."""
    from yam_arena import ik, station
    from yam_arena.constants import READY

    q0, q1 = np.array(READY, float), _good_ik(arm, pos, rot)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, pos, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)


LEFT_TIMES = [1.5, 1.0, 0.4, 1.5, 2.0, 1.5, 0.6]  # slide in, grip, steady, lift, carry, lower, settle
RIGHT_TIMES = [1.0, 1.0, 0.4]  # down onto the handle, grip, steady
TIP_S, HOLD_S, BACK_S = 2.0, 3.0, 1.5


def _tray_handle(theta: float) -> np.ndarray:
    """The TCP on the handle with the tray's back lifted so that the tray turns ``theta`` about its front edge."""
    rel = HANDLE_AT - PIVOT
    c, s = math.cos(theta), math.sin(theta)
    return PIVOT + np.array([rel[0], rel[1] * c + rel[2] * s, -rel[1] * s + rel[2] * c])


def _right() -> list[Waypoint]:
    tcp = _tray_handle
    out = _lead_in("right", tcp(0.0) + [0.0, 0.0, 0.05], TRAY_ROT, OPEN, "over the tray's handle")
    out += [_w(RIGHT_TIMES[0], tcp(0.0), OPEN, "down round the handle", TRAY_ROT),
            _w(RIGHT_TIMES[1], tcp(0.0), TAB_GRIP, "grip the handle", TRAY_ROT),
            _w(RIGHT_TIMES[2] + sum(LEFT_TIMES) + 3.0 - sum(RIGHT_TIMES), tcp(0.0), TAB_GRIP,
               "wait until the seedling is held upright in the hole", TRAY_ROT)]
    n = 8
    out += [_w(TIP_S / n, tcp(TILT * k / n), TAB_GRIP, "lift the tray's back: the soil rolls into the hole", TRAY_ROT)
            for k in range(1, n + 1)]
    out.append(_w(HOLD_S, tcp(TILT), TAB_GRIP, "hold it tipped until the tray is empty", TRAY_ROT))
    out += [_w(BACK_S / n, tcp(TILT * (n - k) / n), TAB_GRIP, "lower the tray", TRAY_ROT) for k in range(1, n + 1)]
    out += [_w(0.8, tcp(0.0), OPEN, "let go", TRAY_ROT), _w(1.0, tcp(0.0) + [0.0, 0.0, 0.06], OPEN, "rise", TRAY_ROT),
            _w(4.0, tcp(0.0) + [0.0, 0.0, 0.06], OPEN, "wait", TRAY_ROT)]
    return out


def _left() -> list[Waypoint]:
    in_cell = np.array([CELL_XY[0], CELL_XY[1], BENCH_TOP + STEM_UP])
    beside = in_cell + [0.0, 0.03, 0.0]
    lifted = in_cell + [0.0, 0.0, LIFT]
    over = np.array([HOLE_XY[0], HOLE_XY[1], lifted[2]])
    planted = np.array([HOLE_XY[0], HOLE_XY[1], FLOOR_TOP + STEM_UP + 0.002 + HOVER])
    away = planted + [0.0, BACK, 0.0]
    out = _lead_in("left", beside, STEM_ROT, OPEN, "level, beside the seedling's stem")
    out += [_w(LEFT_TIMES[0], in_cell, OPEN, "slide in round the stem", STEM_ROT),
            _w(LEFT_TIMES[1], in_cell, STEM_GRIP, "grip the stem", STEM_ROT),
            _w(LEFT_TIMES[2], in_cell, STEM_GRIP, "steady", STEM_ROT),
            _w(LEFT_TIMES[3], lifted, STEM_GRIP, "lift it out of its cell", STEM_ROT),
            _w(LEFT_TIMES[4], over, STEM_GRIP, "carry it over the hole", STEM_ROT),
            _w(LEFT_TIMES[5], planted, STEM_GRIP, "lower it into the hole, upright", STEM_ROT)]
    soil_s = LEFT_TIMES[6] + 3.0 + TIP_S + HOLD_S + BACK_S + 0.5
    if WRONG:
        return out + [_w(0.8, planted, OPEN, "the wrong way: let go before the soil is in", STEM_ROT),
                      _w(1.5, away, OPEN, "draw back", STEM_ROT),
                      _w(1.0, away + [0.0, 0.0, 0.06], OPEN, "rise", STEM_ROT),
                      _w(soil_s + 3.0, away + [0.0, 0.0, 0.06], OPEN, "wait", STEM_ROT)]
    return out + [_w(soil_s, planted, STEM_GRIP, "hold it upright while the soil goes in", STEM_ROT),
                  _w(0.8, planted, OPEN, "let go: it stands in the soil", STEM_ROT),
                  _w(1.5, away, OPEN, "draw back", STEM_ROT),
                  _w(1.0, away + [0.0, 0.0, 0.06], OPEN, "rise", STEM_ROT),
                  _w(3.0, away + [0.0, 0.0, 0.06], OPEN, "wait", STEM_ROT)]


def demo():
    return {"left": _left(), "right": _right()}
