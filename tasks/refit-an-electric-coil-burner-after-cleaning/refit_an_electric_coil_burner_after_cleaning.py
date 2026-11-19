"""On the station table: a true-size section of an electric coil range's cooktop round one 8-inch burner well (an
enamelled steel top 75 mm above the table). The well's chrome drip bowl has just been washed and dropped back
into its recess turned a quarter turn the wrong way, so the opening in its side faces the back instead of the
plug-in receptacle, which is hidden under the cooktop's edge on the robot's right of the well. The 8-inch plug-in
coil element lies on the cooktop to the robot's left of the well. Task (instruction): put the coil burner back so
it is plugged in and lies flat in its bowl. The instruction does not say how.

How it is done (GE electric range owner's manuals, "Plug-in surface units", "Replacing a surface unit"): "Replace
the drip pan into the recess in the cooktop. Make sure opening in the pan lines up with the receptacle." "Insert
the terminals of the surface unit through the opening in the drip pan and into the receptacle." "Guide the surface
unit into place so it rests evenly." On removal: "Lift the surface unit about 1 inch above the drip pan and pull it
out. Do not lift the surface unit more than 1 inch."

The hand skill: the receptacle cannot be seen; the terminals must go in sideways, under the cooktop's edge, with the
coil held tipped up a few degrees (its far side about 1.5 cm up, so that it clears the recess while still 3 cm back
from home), pushed along the terminals' line past two spring contacts until they are home, and then the coil is let
down about its terminals until it lies flat on its legs in the bowl. With the bowl's opening turned away, the leads
and terminal bracket hit the bowl's side and never reach the receptacle. Set down from above, the terminals land on
the cooktop's edge and the coil is not plugged in.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- cooktop section: fixed; steel top 3 mm thick, its top at TOP + 75 mm, x 0.06-0.40, y -0.25-0.33, with a round
  recess hole 226 mm across centred at W = (0.205, -0.050); its housing down to the table.
- drip bowl: loose, chrome steel, 0.20 kg: flange 248 mm across resting on the cooktop round the hole; a sloped wall
  from 222 mm across at the top to 170 mm at the bottom, 40 mm deep; a 50 mm centre hole with a collar. The opening:
  a notch 43 mm wide and 26 mm deep in the side wall and flange. At the start it points along +x (the back).
- receptacle: fixed (a PhysX articulation), under the cooktop's edge on the -y side of the well, its face 122 mm from
  the well's centre, the terminals' axis 13 mm below the cooktop's top; two slots 6.5 mm wide, 10 mm tall and 32 mm
  deep, 22 mm apart, their floors 1 mm under a seated coil's terminals, each with a mouth flaring 3 mm all round over
  its first 5 mm; in each slot a spring contact: a leaf along the slot's +x side, hinged 8 mm in on a 0.4 N m/rad
  torsion spring, slanting in so that its deep end (20 mm in) stands 4.2 mm from the slot's far side; a 4.8 mm
  terminal pushes it about 0.9 mm aside (about 2.5 N).
- coil element: loose, 0.55 kg: five turns of 8 mm tube (outer turn 208 mm across, a 52 mm opening in the middle) on
  a three-legged support 43 mm tall; two leads drop 17 mm to a terminal bracket (32 x 8 x 12 mm) with two terminals
  4.8 mm across, 22 mm apart, 29 mm long with rounded tips. 252 mm from the far side to the terminals' tips.

Score (judged at the end of the run):
- 0: the bowl's opening more than 15 deg from the receptacle, or the bowl not resting flat in its recess.
- 0.25: the bowl right, the coil not plugged in.
- 0.5: the bowl right and both terminals in the receptacle (tips 20 mm or more past its face and each spring contact's
  deep end pushed aside 0.4 mm or more), but the coil not flat on its legs in the bowl, still held, or not still.
- 1.0 (success, held 1 s): the bowl right, plugged in, the coil flat (tilt under 2 deg) and resting on its legs in
  the bowl (its centre within 2 mm of its seated height), every finger point 20 mm or more from the coil, and the coil
  still (under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, Link, Linkage, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, capsule, on_reset, on_step, plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001


def _f(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


# -- the layout ----------------------------------------------------------------------------------------------------
W = np.array([0.205, -0.050])  # the burner well's centre
CT = TOP + 0.075  # the cooktop's top surface
TOP_T = 0.003  # the steel top's thickness
HOLE_R = 0.113  # the recess hole in the top
SLAB_X, SLAB_Y = (0.06, 0.40), (-0.25, 0.33)

# drip bowl (body frame: origin on the well's axis at the cooktop's top surface; its opening along +x)
B_FLANGE_R = (0.111, 0.124)
B_WALL_TOP_R, B_WALL_BOT_R, B_DEPTH = 0.111, 0.085, 0.040
B_HOLE_R, B_COLLAR_H = 0.025, 0.006
B_T = 0.0012  # sheet thickness (collision)
B_SEG = 48
NOTCH_SEGS = 3  # the opening: 3 of 48 segments (22.5 deg, 43 mm at the top of the wall)
NOTCH_DEPTH = 0.026
BOWL_YAW0 = math.radians(_f("COIL_BOWL_YAW0", "0"))  # the opening's direction at the start (world)
BOWL_TARGET = -math.pi / 2  # toward the receptacle (-y)
BOWL_MASS = 0.20

# coil element (body frame: origin at the coil's centre, in the plane of the tube's centre line; terminals along -y)
TUBE_R = 0.004
RINGS = (0.030, 0.0475, 0.065, 0.0825, 0.100)
RING_SEGS = (12, 16, 18, 20, 24)
LEAD_X = 0.011  # the terminals' half spacing
TERM_R = 0.0024
TERM_Z = -0.017  # the bracket and terminals below the tube's centre plane
BRACKET_Y = (-0.119, -0.111)
TERM_Y = (-0.119, -0.1460)  # the terminals' straight part (the tips are rounded, to -0.1484)
TIP_Y = TERM_Y[1] - TERM_R
LEG_R, LEG_ANG = 0.055, (90.0, 210.0, 330.0)
SEAT_Z = CT + 0.004  # the coil's origin height when it lies seated in the bowl
LEG_LEN = SEAT_Z - (CT - B_DEPTH + B_T)  # from the tube plane down to the bowl's floor
COIL_MASS = _f("COIL_MASS", "0.55")
COIL_MU = _f("COIL_MU", "0.8")
COIL0 = np.array([0.205, 0.235])  # where it lies on the cooktop at the start
COIL0_Z = CT + LEG_LEN + 0.001

# receptacle (world): its face across y, 122 mm from the well's centre, on the -y side
REC_FACE_Y = W[1] - 0.122
TERM_ZW = SEAT_Z + TERM_Z  # the terminals' centre height when seated
SLOT_W, SLOT_H, SLOT_D = 0.0065, 0.010, 0.032
SLOT_Z0 = -TERM_R - 0.001  # the slots' floor, from the terminals' axis when seated (1 mm under them)
MOUTH_X, MOUTH_D = 0.003, 0.005  # each mouth flares 3 mm out round its slot over the first 5 mm
REC_MU = _f("COIL_REC_MU", "0.2")  # terminal on the receptacle's ceramic and brass (estimate)
LEAF_K = _f("COIL_LEAF_K", "0.4")  # N m/rad: about 2.5 N at the leaf's deep end for the 0.9 mm a terminal pushes it
CLIP_GAP = _f("COIL_CLIP_GAP", "0.0042")  # the slot left between a spring contact and the far side of the slot (a terminal is 4.8 mm)
CLIP_Y = (0.008, 0.020)  # the spring contact's length, measured into the slot from the face

STEEL_TOP, ENAMEL_EDGE, CHROME, COIL_GREY, TERM_METAL, CERAMIC, DARK, BRACKET_C = (
    (0.93, 0.93, 0.91), (0.78, 0.78, 0.76), (0.62, 0.64, 0.68), (0.22, 0.22, 0.23), (0.70, 0.66, 0.55),
    (0.92, 0.90, 0.84), (0.15, 0.15, 0.16), (0.55, 0.55, 0.58))


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _cooktop() -> Body:
    """The top (a steel plate with the recess hole), its front edge and the burner box under the well (visual)."""
    sh = []
    zc = -TOP_T / 2  # body origin at the top surface over the well's centre
    n = 48
    q = 0.140  # the ring round the hole fills the square of this half side round the well
    for k in range(n):  # the ring round the hole, as 48 radial plates out to the square
        a = 2 * math.pi * (k + 0.5) / n
        r0, r1 = HOLE_R, q / max(abs(math.cos(a)), abs(math.sin(a))) + 0.003
        w = 2 * r1 * math.tan(math.pi / n) + 0.002
        sh.append(BoxShape(((r0 + r1) / 2 * math.cos(a), (r0 + r1) / 2 * math.sin(a), zc), (r1 - r0, w, TOP_T),
                           STEEL_TOP, yaw=a))
    for cx in (-1, 1):  # the square's corners, which the radial plates leave open
        for cy in (-1, 1):
            sh.append(BoxShape((cx * (q - 0.02), cy * (q - 0.02), zc), (0.04, 0.04, TOP_T), STEEL_TOP))
    x0, x1 = SLAB_X[0] - W[0], SLAB_X[1] - W[0]
    y0, y1 = SLAB_Y[0] - W[1], SLAB_Y[1] - W[1]
    sh += [BoxShape(((x0 + x1) / 2, (q + y1) / 2, zc), (x1 - x0, y1 - q, TOP_T), STEEL_TOP),
           BoxShape(((x0 + x1) / 2, (y0 - q) / 2, zc), (x1 - x0, -q - y0, TOP_T), STEEL_TOP),
           BoxShape(((x0 - q) / 2, 0.0, zc), (-q - x0, 2 * q, TOP_T), STEEL_TOP),
           BoxShape(((x1 + q) / 2, 0.0, zc), (x1 - q, 2 * q, TOP_T), STEEL_TOP)]
    h = CT - TOP - TOP_T  # the housing under the top, down to the table
    for (cx, cy, sx, sy) in (((x0 + x1) / 2, y0 + 0.002, x1 - x0, 0.004), ((x0 + x1) / 2, y1 - 0.002, x1 - x0, 0.004),
                             (x0 + 0.002, (y0 + y1) / 2, 0.004, y1 - y0), (x1 - 0.002, (y0 + y1) / 2, 0.004, y1 - y0)):
        sh.append(BoxShape((cx, cy, -TOP_T - h / 2), (sx, sy, h), ENAMEL_EDGE))
    # the burner box under the well (seen through the hole while the bowl is out): floor and walls, visual only
    sh.append(BoxShape((0.0, 0.0, -0.065), (0.27, 0.27, 0.002), DARK, collide=False))
    # a mounting bracket for the receptacle, from the burner box's side (visual)
    sh.append(BoxShape((0.0, REC_FACE_Y - W[1] - SLOT_D - 0.004, (TERM_ZW - CT) - 0.026),
                       (0.06, 0.004, 0.03), BRACKET_C, collide=False))
    return Body.build("cooktop", pos=(float(W[0]), float(W[1]), CT), mass=5.0, color=STEEL_TOP, fixed=True,
                      friction=(0.35, 0.3), shapes=sh, contact_offset=0.001)


def _bowl() -> Body:
    sh = []
    da = 2 * math.pi / B_SEG
    notch = set(range(-(NOTCH_SEGS // 2), NOTCH_SEGS // 2 + 1))
    slope = (B_WALL_TOP_R - B_WALL_BOT_R) / B_DEPTH
    for k in range(B_SEG):
        kk = k if k <= B_SEG // 2 else k - B_SEG
        a = da * kk  # segment centres at 0, 7.5, ... deg; the opening's segments centred on 0
        c, s = math.cos(a), math.sin(a)
        is_notch = kk in notch
        if not is_notch:  # flange
            r0, r1 = B_FLANGE_R
            w = 2 * r1 * math.tan(da / 2) + 0.001
            sh.append(BoxShape(((r0 + r1) / 2 * c, (r0 + r1) / 2 * s, B_T / 2), (r1 - r0 + 0.002, w, B_T), CHROME,
                               yaw=a))
        # the wall: upper part (absent in the opening) and lower part
        z_cut = -NOTCH_DEPTH
        r_cut = B_WALL_TOP_R - slope * NOTCH_DEPTH
        wtop = 2 * B_WALL_TOP_R * math.tan(da / 2) + 0.001
        wbot = 2 * B_WALL_BOT_R * math.tan(da / 2) + 0.001
        if not is_notch:
            sh.append(plate((B_WALL_TOP_R * c, B_WALL_TOP_R * s, B_T), (r_cut * c, r_cut * s, z_cut), wtop, B_T,
                            (c, s, slope), CHROME))
        sh.append(plate((r_cut * c, r_cut * s, z_cut), (B_WALL_BOT_R * c, B_WALL_BOT_R * s, -B_DEPTH), wbot, B_T,
                        (c, s, slope), CHROME))
        # the floor
        r0, r1 = B_HOLE_R, B_WALL_BOT_R + 0.002
        w = 2 * r1 * math.tan(da / 2) + 0.001
        sh.append(BoxShape(((r0 + r1) / 2 * c, (r0 + r1) / 2 * s, -B_DEPTH + B_T / 2), (r1 - r0, w, B_T), CHROME,
                           yaw=a))
    for k in range(16):  # the centre hole's collar
        a = 2 * math.pi * k / 16
        sh.append(BoxShape((B_HOLE_R * math.cos(a), B_HOLE_R * math.sin(a), -B_DEPTH + B_COLLAR_H / 2),
                           (B_T, 2 * B_HOLE_R * math.tan(math.pi / 16) + 0.001, B_COLLAR_H), CHROME, yaw=a))
    return Body.build("drip_bowl", pos=(float(W[0]), float(W[1]), CT + 0.0008), mass=BOWL_MASS, color=CHROME,
                      yaw=BOWL_YAW0, friction=(0.3, 0.25), shapes=sh, contact_offset=0.001, solver_iterations=32)


def _coil() -> Body:
    sh = []
    for r, n in zip(RINGS, RING_SEGS):
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            sh.append(capsule((r * math.cos(a0), r * math.sin(a0), 0.0), (r * math.cos(a1), r * math.sin(a1), 0.0),
                              TUBE_R, COIL_GREY))
    for sx in (-1, 1):  # the leads, the bracket and the terminals
        x = sx * LEAD_X
        sh.append(capsule((x, -RINGS[-1] + 0.002, 0.0), (x, BRACKET_Y[1], TERM_Z), 0.003, COIL_GREY))
        sh.append(capsule((x, BRACKET_Y[0], TERM_Z), (x, TERM_Y[1], TERM_Z), TERM_R, TERM_METAL))
    sh.append(BoxShape((0.0, (BRACKET_Y[0] + BRACKET_Y[1]) / 2, TERM_Z), (0.032, BRACKET_Y[1] - BRACKET_Y[0], 0.012),
                       BRACKET_C))
    for ang in LEG_ANG:  # the three-legged support: an arm under the turns and a leg down to the bowl's floor
        a = math.radians(ang)
        c, s = math.cos(a), math.sin(a)
        sh.append(BoxShape((0.065 * c, 0.065 * s, -TUBE_R - 0.002), (0.075, 0.006, 0.004), BRACKET_C, yaw=a))
        sh.append(BoxShape((LEG_R * c, LEG_R * s, -(LEG_LEN + TUBE_R + 0.002) / 2 - 0.0), (0.006, 0.010,
                           LEG_LEN - TUBE_R - 0.002), BRACKET_C, yaw=a))
    return Body.build("coil", pos=(float(COIL0[0]), float(COIL0[1]), COIL0_Z), mass=COIL_MASS, color=COIL_GREY,
                      friction=(COIL_MU, COIL_MU - 0.1), shapes=sh, contact_offset=0.001, solver_iterations=32)


def _receptacle() -> Body:
    """A ceramic block with two slots, each with a lead-in mouth; its origin at the face, on the terminals' axis (as
    they lie when the coil is seated). Each slot's floor is level with the bottom of its terminal, so a seated coil's
    terminals rest on it."""
    sh = []
    zb, zt = -0.009, 0.009  # block bottom and top, from the terminals' axis (its top stays under the steel top)
    s0, s1 = SLOT_Z0, SLOT_Z0 + SLOT_H  # the slots' floor and roof
    m0, m1 = s0 - MOUTH_X, min(s1 + MOUTH_X, zt)  # the mouths' bottom and top at the face
    xb = 0.028
    d = SLOT_D
    yc = -d / 2 - MOUTH_D / 2  # the block behind the mouths, from y -MOUTH_D to -d
    edges = [-xb, -LEAD_X - SLOT_W / 2, -LEAD_X + SLOT_W / 2, LEAD_X - SLOT_W / 2, LEAD_X + SLOT_W / 2, xb]
    for a, b in ((edges[0], edges[1]), (edges[2], edges[3]), (edges[4], edges[5])):
        sh.append(BoxShape(((a + b) / 2, yc, (zb + zt) / 2), (b - a, d - MOUTH_D, zt - zb), CERAMIC))
    sh.append(BoxShape((0.0, yc, (zb + s0) / 2), (2 * xb, d - MOUTH_D, s0 - zb), CERAMIC))
    sh.append(BoxShape((0.0, yc, (zt + s1) / 2), (2 * xb, d - MOUTH_D, zt - s1), CERAMIC))
    sh.append(BoxShape((0.0, -d - 0.002, (zb + zt) / 2), (2 * xb, 0.004, zt - zb), CERAMIC))
    zm = (s0 + s1) / 2
    for sx in (-1, 1):  # the lead-in mouths: four sloped plates round each slot
        x = sx * LEAD_X
        for side in (-1, 1):
            sh.append(plate((x + side * (SLOT_W / 2 + MOUTH_X), 0.0, zm), (x + side * SLOT_W / 2, -MOUTH_D, zm),
                            (m1 - m0) + 0.002, 0.0015, (-side, 0.0, 0.0), CERAMIC))
        sh.append(plate((x, 0.0, m0), (x, -MOUTH_D, s0), SLOT_W + 2 * MOUTH_X + 0.003, 0.0015, (0.0, 0.0, 1.0),
                        CERAMIC))
        sh.append(plate((x, 0.0, m1), (x, -MOUTH_D, s1), SLOT_W + 2 * MOUTH_X + 0.003, 0.0015, (0.0, 0.0, -1.0),
                        CERAMIC))
    return Linkage.build("receptacle", pos=(float(W[0]), float(REC_FACE_Y), float(TERM_ZW)), mass=0.1,
                         color=CERAMIC, friction=(REC_MU, REC_MU - 0.05), shapes=sh, links=[_leaf(0, -1), _leaf(1, 1)],
                         contact_offset=0.001, solver_iterations=32)


LEAF_T = 0.0015
LEAF_L = CLIP_Y[1] - CLIP_Y[0]  # from its hinge (8 mm in) to its deep end


def _leaf(k: int, sx: int) -> Link:
    """The spring contact in one slot: a leaf along the slot's +x side, as tall as the slot, hinged at its front end
    (8 mm in) about a vertical axis on a torsion spring and slanting into the slot, so that its deep end (20 mm in)
    stands 4.2 mm from the slot's far side; a terminal (4.8 mm) pushes it aside as it goes in (relative to the
    receptacle's origin: the face, on the terminals' axis)."""
    wall = sx * LEAD_X + SLOT_W / 2  # the slot's +x side
    x_deep = sx * LEAD_X - SLOT_W / 2 + CLIP_GAP  # the leaf's face at its deep end
    x_front = wall - 0.0003
    zc = SLOT_Z0 + SLOT_H / 2
    hh = SLOT_H - 0.001
    off = LEAF_T / 2  # the plate's middle line, behind its face
    shapes = (plate((x_front + off, -CLIP_Y[0], zc), (x_deep + off, -CLIP_Y[1], zc), hh, LEAF_T, (-1.0, 0.0, 0.0),
                    TERM_METAL),)
    anchor = (float(W[0] + x_front + off), float(REC_FACE_Y - CLIP_Y[0]), float(TERM_ZW + zc))
    return Link(f"contact_{k}", shapes, mass=0.003, anchor=anchor, axis="Z", limits=(0.0, 15.0),
                stiffness=LEAF_K, spring_damping=0.002)


COOKTOP = _cooktop()
RECEPTACLE = _receptacle()
BOWL = _bowl()
COIL = _coil()
OBJECTS = [COOKTOP, RECEPTACLE, BOWL, COIL]

# -- tracking and scoring -------------------------------------------------------------------------------------------
BOWL_DEG = 15.0
BOWL_TILT, BOWL_DZ = 2.0, 0.002
DEPTH_MIN = 0.020
CLIP_MIN = 0.0004
COIL_TILT = 2.0
COIL_DZ = 0.002
CLEAR = 0.020
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("COIL_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("COIL_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
TIPS = np.array([[sx * LEAD_X, TIP_Y, TERM_Z] for sx in (-1, 1)])
COIL_SAMPLES = np.array([[r * math.cos(a), r * math.sin(a), 0.0] for r in RINGS
                         for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
                        + [[sx * LEAD_X, y, TERM_Z] for sx in (-1, 1) for y in np.linspace(-0.11, TIP_Y, 5)])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, recent=[], near=1.0, tcp={}, gaps={})


_reset()


def _finger_points() -> dict:
    import struct
    import xml.etree.ElementTree as ET

    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = pathlib.Path(YAM_URDF).parent
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        name = link.get("name")
        if name not in ("tip_left", "tip_right"):
            continue
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            b = (root / col.find("geometry").find("mesh").get("filename")).read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            tri = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            v = tri["v"].reshape(-1, 3).astype(float)
            pts.append(np.unique(np.round(v, 4), axis=0) @ rpy_matrix(rpy).T + xyz)
        out[name] = np.concatenate(pts)
    return out


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("COIL_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "fingers" not in _IDS:
        _IDS["fingers"] = _finger_points()
    fingers = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[side + "_f"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
            _IDS[side + "_tips"] = [robot.body_names.index(n) for n in ("tip_left", "tip_right")]
        q = as_numpy(robot.data.joint_pos)[0]
        _EP["tcp"][side] = np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float)
        _EP["gaps"][side] = float(-q[_IDS[side + "_f"]].sum())
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        fingers[side] = np.concatenate([_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                                        for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])])
    cp = as_numpy(base.scene["coil"].data.root_link_pose_w)[0]
    pts = cp[:3] - origin + COIL_SAMPLES @ rotation(cp[3:7]).T
    tree = cKDTree(pts)
    _EP["near"] = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())


def _state(o) -> dict:
    bowl, coil = o["drip_bowl"], o["coil"]
    br = bowl.rotation()
    bowl_ang = math.degrees(math.atan2(br[1, 0], br[0, 0]))
    bowl_off = abs((bowl_ang - math.degrees(BOWL_TARGET) + 180.0) % 360.0 - 180.0)
    bowl_tilt = math.degrees(bowl.tilt())
    bowl_dz = float(bowl.pos[2] - CT)
    bowl_ok = bool(bowl_off <= BOWL_DEG and bowl_tilt < BOWL_TILT and abs(bowl_dz) < BOWL_DZ
                   and np.linalg.norm(bowl.pos[:2] - W) < 0.006)
    cr = coil.rotation()
    tips = coil.pos + TIPS @ cr.T
    depth = [float(REC_FACE_Y - t[1]) for t in tips]
    clips = [float(o["receptacle"].joints[f"contact_{k}"] * LEAF_L) for k in range(2)]  # deep end pushed aside
    plugged = bool(min(depth) >= DEPTH_MIN and min(clips) >= CLIP_MIN)
    coil_tilt = math.degrees(coil.tilt())
    coil_dz = float(coil.pos[2] - SEAT_Z)
    flat = bool(coil_tilt < COIL_TILT and abs(coil_dz) < COIL_DZ)
    tip_off = [(float(t[0] - (W[0] + sx * LEAD_X)) * 1000, float(t[2] - TERM_ZW) * 1000) for t, sx in zip(tips, (-1, 1))]
    return dict(tip_off=tip_off, bowl_ang=bowl_ang, bowl_off=bowl_off, bowl_tilt=bowl_tilt, bowl_dz=bowl_dz, bowl_ok=bowl_ok,
                depth=depth, clips=clips, plugged=plugged, coil_tilt=coil_tilt, coil_dz=coil_dz, flat=flat,
                coil_xy=coil.pos[:2] - W)


def _tally(o) -> dict:
    s = _state(o)
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(o["coil"].pos, float).copy())
        del recent[:-STILL_STEPS]
    s["still"] = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    s["clear"] = _EP["near"] >= CLEAR
    s["done"] = bool(s["bowl_ok"] and s["plugged"] and s["flat"] and s["clear"] and s["still"])
    return s


def _debug(s, o) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    c = o["coil"].pos
    print(f"DBG step={_EP['step']} bowl ang={s['bowl_ang']:+.1f} off={s['bowl_off']:.1f} tilt={s['bowl_tilt']:.2f} "
          f"dz={s['bowl_dz'] * 1000:+.1f}mm ok={int(s['bowl_ok'])} | coil=({c[0]:.3f},{c[1]:.3f},{(c[2] - CT) * 1000:+.1f}mm)"
          f" xy=({s['coil_xy'][0] * 1000:+.1f},{s['coil_xy'][1] * 1000:+.1f})mm tilt={s['coil_tilt']:.2f} "
          f"dz={s['coil_dz'] * 1000:+.1f}mm depth=({s['depth'][0] * 1000:.1f},{s['depth'][1] * 1000:.1f})mm "
          f"clips=({s['clips'][0] * 1000:.2f},{s['clips'][1] * 1000:.2f})mm plugged={int(s['plugged'])} "
          f"tips(dx,dz)=({s['tip_off'][0][0]:+.1f},{s['tip_off'][0][1]:+.1f};{s['tip_off'][1][0]:+.1f},{s['tip_off'][1][1]:+.1f}) "
          f"flat={int(s['flat'])} near={_EP['near'] * 1000:.0f}mm still={int(s['still'])} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{(v[2] - CT) * 1000:+.0f}mm) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    s = _tally(objects)
    if _DEBUG:
        _debug(s, objects)
    if not s["bowl_ok"]:
        return 0.0
    if not s["plugged"]:
        return 0.25
    if s["done"]:
        return 1.0
    return 0.5


def success(objects) -> bool:
    return bool(_tally(objects)["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
OPEN_C = -0.047
MODE = os.environ.get("COIL_MODE", "demo")  # demo | noturn (bowl left as it is) | drop (coil set down from above)


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    b = station.BASES[arm]
    j1 = math.atan2(float(pos[1]) - b[1], float(pos[0]) - b[0])
    for sd in itertools.product((j1,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (0.0,), (-1.8, -0.9, 0.0, 0.9, 1.8)):
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


def _ready(arm: str):
    from yam_arena import ik, station
    from yam_arena.constants import READY

    return ik.tcp_pose(np.array(READY, float), station.BASES[arm])


def _rot_axis(axis, a: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


# Right arm: turn the bowl. Its closed fingertips go down into the bowl's opening and carry it round a quarter turn.
TURN_R = _f("COIL_TURN_R", "0.100")  # the fingertips' radius from the well's axis
TURN_DZ = _f("COIL_TURN_DZ", "-0.008")  # the TCP's height from the cooktop's top while turning
TURN_END = _f("COIL_TURN_END", "-106")  # deg: carried round to here (aimed a little past)
TURN_S = _f("COIL_TURN_S", "8")


def _turn_at(a_deg: float, dz: float) -> np.ndarray:
    a = math.radians(a_deg)
    return np.array([W[0] + TURN_R * math.cos(a), W[1] + TURN_R * math.sin(a), CT + dz])


def _turn_rot(a_deg: float) -> np.ndarray:
    return grip_rotation(math.radians(a_deg) + math.pi / 2)  # fingers closing along the radius


def _right_path() -> list:
    a0 = math.degrees(BOWL_YAW0)
    out = _lead_in("right", _turn_at(a0, 0.06), _turn_rot(a0), 0.0, "over the bowl's opening, fingers closed")
    out += [_w(2.0, _turn_at(a0, TURN_DZ), 0.0, "fingertips down into the opening", _turn_rot(a0))]
    n = 18
    for k in range(1, n + 1):
        a = a0 + (TURN_END - a0) * k / n
        out.append(_w(TURN_S / n, _turn_at(a, TURN_DZ), 0.0, "carry the opening round toward the receptacle",
                      _turn_rot(a)))
    a1 = TURN_END + 4.0
    out += [_w(1.0, _turn_at(a1, TURN_DZ), 0.0, "ease back off the opening's edge", _turn_rot(a1)),
            _w(1.5, _turn_at(a1, 0.07), 0.0, "rise out of the bowl", _turn_rot(a1))]
    pos, rot = _ready("right")
    out += [_w(3.0, pos, OPEN_C, "back to the ready pose", rot), _w(120.0, pos, OPEN_C, "stay put", rot)]
    return out


# Left arm: the coil. Open loop, from the known layout. It waits while the bowl is turned, then its closed fingers go
# down into the coil's 52 mm middle and open against the inner turn (fingers opening along y), and it lifts the coil
# 7 cm. It carries the coil over the well tipped up 3 deg at the far side, 6.5 cm above its seat, lowers it with the
# terminals in line with the bowl's opening, pushes it along the terminals' line (aimed 2.5 cm past home: the arm is
# soft sideways and stops short), holds it there, lets the far side down about the terminals to 1 deg below level,
# rests it (still pressed toward home), closes its fingers off the inner turn, rises and moves away.
HOLD = _f("COIL_HOLD", "-0.027")  # fingers told 7.5 mm past the inner turn: a firm hold (about 37 N a finger, under
# the 50 N limit), so a push along the fingers is passed on (with both at their limit the hand slides inside the coil)
GRIP_DZ = _f("COIL_GRIP_DZ", "0.006")  # the TCP above the coil's tube plane
TILT = math.radians(_f("COIL_TILT", "3"))  # the coil tipped up, far side (+y) high, while the terminals go in
PIVOT = np.array([0.0, _f("COIL_PIVOT_Y", "-0.140"), TERM_Z])  # coil frame: the point it is tipped about
SHIFT0 = _f("COIL_SHIFT0", "0.032")  # the coil starts the push this far back (+y) from home
PUSH_PAST = _f("COIL_PUSH_PAST", "0.025")  # and the hand is aimed this far past home (the arm is soft sideways)
LIFT = _f("COIL_LIFT", "0.0")  # extra height for the approach
CARRY_DZ = _f("COIL_CARRY_DZ", "0.065")  # carried this high over its seat (its legs are 43 mm long)
SAG = _f("COIL_SAG", "0.018")  # the arm sags under the coil and the coil sits low in the fingers: aim this much higher
DOWN_PAST = math.radians(_f("COIL_DOWN_PAST", "1.0"))  # the far side let down to this far below level
REST_PAST = _f("COIL_REST_PAST", "0.012")  # while it rests, the hand still aimed this far past home
# the push runs a little downward (4 deg more than the tilt), which keeps the far side from rising as it is pushed
PUSH_A = TILT + math.radians(_f("COIL_PUSH_EXTRA", "4"))
L_ROT = grip_rotation(float(os.environ.get("COIL_LYAW", "0")))  # left arm, fingers opening along y


def _coil_pose(shift: float, tilt: float, dz: float = 0.0):
    """Coil origin (world) and rotation: tipped ``tilt`` about the x axis through the pivot, the pivot ``shift``
    back (+y) from its seated place and ``dz`` higher."""
    r = _rot_axis((1.0, 0.0, 0.0), tilt)
    piv_home = np.array([W[0], W[1], SEAT_Z]) + PIVOT
    piv = piv_home + np.array([0.0, shift * math.cos(PUSH_A), shift * math.sin(PUSH_A) + dz])
    return piv - r @ PIVOT, r


def _hand(o, r, dz=GRIP_DZ) -> np.ndarray:
    return o + r @ np.array([0.0, 0.0, dz])


def _left_path() -> list:
    c0 = np.array([COIL0[0], COIL0[1], COIL0_Z])
    out = [_w(9.0 if MODE != "noturn" else 0.5, _ready("left")[0], 0.0, "wait for the bowl", _ready("left")[1])]
    out += _lead_in("left", c0 + np.array([0.0, 0.0, 0.06]), L_ROT, 0.0, "over the coil's middle, fingers closed")
    out += [_w(2.0, c0 + np.array([0.0, 0.0, GRIP_DZ]), 0.0, "fingers down into the coil's middle", L_ROT),
            _w(1.2, c0 + np.array([0.0, 0.0, GRIP_DZ]), HOLD, "open against the inner turn", L_ROT),
            _w(0.6, c0 + np.array([0.0, 0.0, GRIP_DZ]), HOLD, "settle", L_ROT),
            _w(1.5, c0 + np.array([0.0, 0.0, 0.07]), HOLD, "lift the coil", L_ROT)]
    sag = np.array([0.0, 0.0, SAG])
    if MODE == "drop":  # wrong: carried over the bowl level and set straight down in its seated place
        o, r = _coil_pose(0.0, 0.0, CARRY_DZ)
        out += [_w(5.0, _hand(o, r) + sag, HOLD, "carry it over the bowl, level", L_ROT),
                _w(2.0, _hand(o, r) + sag, HOLD, "let the arm catch up", L_ROT)]
        o, r = _coil_pose(0.0, 0.0, 0.0)
        out += [_w(3.0, _hand(o, r) + sag, HOLD, "set it straight down", L_ROT)]
    else:
        o, r = _coil_pose(SHIFT0, TILT, CARRY_DZ + LIFT)
        out += [_w(5.0, _hand(o, r) + sag, HOLD, "carry it over the bowl, tipped up at the far side", r @ L_ROT),
                _w(2.0, _hand(o, r) + sag, HOLD, "let the arm catch up", r @ L_ROT)]
        o, r = _coil_pose(SHIFT0, TILT, LIFT)
        out += [_w(2.0, _hand(o, r) + sag, HOLD, "lower it, terminals in line with the opening", r @ L_ROT),
                _w(0.6, _hand(o, r) + sag, HOLD, "settle", r @ L_ROT)]
        n = 10
        for k in range(1, n + 1):
            sh = SHIFT0 + (-PUSH_PAST - SHIFT0) * k / n
            o, r = _coil_pose(sh, TILT, LIFT)
            out.append(_w(3.0 / n, _hand(o, r) + sag, HOLD, "push the terminals in under the cooktop's edge",
                          r @ L_ROT))
        o, r = _coil_pose(-PUSH_PAST, TILT, LIFT)
        out.append(_w(1.0, _hand(o, r) + sag, HOLD, "hold them home", r @ L_ROT))
        for k in range(1, n + 1):
            t = TILT + (-DOWN_PAST - TILT) * k / n
            o, r = _coil_pose(-PUSH_PAST, t, 0.0)
            out.append(_w(2.5 / n, _hand(o, r) + sag, HOLD, "let the far side down about the terminals",
                          r @ L_ROT))
    o, r = _coil_pose(-REST_PAST if MODE != "drop" else 0.0, 0.0, 0.0)
    h = _hand(o, r)
    out += [_w(1.0, h, HOLD, "rest it, still pressed home", L_ROT),
            _w(1.2, h, 0.0, "close the fingers off the inner turn", L_ROT),
            _w(2.0, h + np.array([0.0, 0.0, 0.07]), 0.0, "rise out of the coil", L_ROT),
            _w(2.0, h + np.array([-0.03, 0.10, 0.10]), 0.0, "move away", L_ROT),
            _w(120.0, h + np.array([-0.03, 0.10, 0.10]), 0.0, "stay clear", L_ROT)]
    return out


def demo():
    left = _left_path()
    if MODE == "noturn":
        pos, rot = _ready("right")
        right = [_w(120.0, pos, OPEN_C, "stay put", rot)]
    else:
        right = _right_path()
    return {"left": left, "right": right}
