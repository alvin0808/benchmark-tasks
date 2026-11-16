"""On the station table: a server rack's vertical mounting rail (an EIA-310 19-inch rail, 2 mm steel, real 9.53 mm
square holes), standing in a steel foot clamped to the bench, with its front (the side equipment mounts on) facing away
from the robot. One hole is marked with a yellow paint-pen square. An M6 cage nut lies face down on the bench behind
the rail, and a small flat-blade screwdriver lies on the bench to the right.

Task (instruction): fit the cage nut into the marked hole on the rack rail. The instruction does not say how. The
procedure the task checks is the rack makers': from inside the rack, lips facing out, "Hook one lip of the cage nut in
the square rail hole" (Oracle, SPARC M8/M7 Servers Installation Guide, "Insert Cage Nuts Into Rail Holes"), then, with a
flat-blade screwdriver, "Hold the cage nut in one hand, and compress the cage-nut clip with a flat-blade screwdriver.
With the clip compressed, push the edge of the cage nut fully into the hole from the inside of the rack-mounting
flange. Release the screwdriver pressure on the clip to lock the cage nut into place." (IBM, "Installing the cage nuts
in a 7014-S25 or 7014-S11 rack"). Turned home with nothing on its far clip, the clip's end swings out past the hole's
edge and catches behind the flange: the nut jams half in and the hooked clip is levered back out.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- rail: fixed. Mounting flange x 0.215 (its back face, toward the robot) to 0.217, y -0.030 to 0.0075, from 6 mm to
  210 mm above the table; side flange (web) along its -y edge, 30 mm deep. Holes: 9.53 mm square, three per U
  (U = 44.45 mm, U1 starting 25 mm above the table; holes 6.35, 22.225 and 38.1 mm above each U line), centred on
  y 0; U lines and numbers 1-4 on both faces. The marked hole is U3's top hole, its middle 152.2 mm above the table.
- cage nut, loose, 5.5 g, face down at (0.15, 0.10): an M6 square nut (10 x 10 x 4.8 mm, 5 mm bore) in a spring-steel
  cage (10.1 x 11.2 x 6.0 mm) with two lips on opposite sides. Each lip is a 5 mm wide strip folded over (inner leg
  4.75 mm long, outer leg flaring out to 13.72 mm across both lips, its end 2.3 mm in front of the cage's face) on a
  sprung hinge at its root (0.02 N m/rad, from 50 deg in to 10 deg out): pressed in it passes the 9.53 mm hole, let
  go it springs out over the flange's front face.
- screwdriver, loose, 40 g, lying along x with its tip at (0.215, -0.11), pointing toward the robot: tip 0.6 x 3.5 mm,
  60 mm shank, fluted handle 25 mm across.

Score: 0 for the rest of the run once the nut has fallen onto the bench, a lip has sprung out at another hole, or a lip
has been forced 30 deg open (its clip bent). Otherwise 0.5 once the nut sits in the marked hole by itself with at least
one lip sprung out over the front face, both grippers' centres 3 cm or more from it and the screwdriver out of the
hole. Full score: seated (both lips sprung out over the front face around the marked hole, the nut flat against the
flange's back face and centred in the hole), released like that, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import _quat_from_matrix, plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- the rail (world frame, m; heights above the table) ---------------------------------------------------------------
XB = 0.215  # back face of the rail's mounting flange (toward the robot); its front faces +x
FL_T = 0.002  # flange thickness (steel)
HOLE = 0.00953  # EIA-310 square hole (0.375 in)
HH = HOLE / 2
HY = 0.0  # the rail's hole column
IN_EDGE = 0.0075  # flange edge on the rack-opening side, from the hole centre (EIA-310: 450 mm clear opening)
OUT_EDGE = 0.030  # flange edge on the outer side, where the side flange (web) runs back
WEB_D = 0.030  # how far the side flange runs back from the mounting flange
U = 0.04445
Z_U0 = 0.025  # bottom of U1 above the table
N_U = 4
Z_FL0, Z_FL1 = 0.006, 0.210  # the rail's bottom (on its foot) and top
HOLE_Z = [Z_U0 + n * U + d for n in range(N_U) for d in (0.00635, 0.022225, 0.0381)]
MARK_U, MARK_K = 3, 2  # the marked hole: U3, top hole
MARK = (MARK_U - 1) * 3 + MARK_K
HZ = HOLE_Z[MARK]
STEEL, DARKSTEEL, WHITE, YELLOW, BLACK = (0.32, 0.33, 0.35), (0.20, 0.21, 0.22), (0.92, 0.92, 0.90), (0.98, 0.80, 0.08), (
    0.04, 0.04, 0.05)
RAIL_O = (XB, HY, TOP)


def _box(x0, x1, y0, y1, z0, z1, color, origin, collide=True) -> BoxShape:
    """A box from world corners (x, y; heights above the table) in the frame of a body whose origin is ``origin``."""
    return BoxShape(((x0 + x1) / 2 - origin[0], (y0 + y1) / 2 - origin[1], (z0 + z1) / 2 + TOP - origin[2]),
                    (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


_SEG = {"1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc"}


def _digit(ch, x, yc, zc, o, mirror: bool, h=0.0045, w=0.0028, t=0.0006) -> list:
    """A 7-segment digit painted on a face at x (visual): segments a (top) .. g (middle); ``mirror`` for the back face,
    so that it reads the right way round from behind."""
    out = []
    for s in _SEG[ch]:
        if s in "adg":
            z = {"a": zc + h / 2, "g": zc, "d": zc - h / 2}[s]
            out.append(_box(x - 0.00015, x + 0.00015, yc - w / 2, yc + w / 2, z - t / 2, z + t / 2, WHITE, o, False))
        else:
            right = (s in "bc") != mirror  # seen from the front, +y is to the right
            y = yc + w / 2 if right else yc - w / 2
            z0, z1 = (zc, zc + h / 2) if s in "bf" else (zc - h / 2, zc)
            out.append(_box(x - 0.00015, x + 0.00015, y - t / 2, y + t / 2, z0, z1, WHITE, o, False))
    return out


def _rail_shapes() -> list:
    """The rail in the frame of its marked hole's column at the flange's back face (table height): the mounting
    flange with real square holes (bars between the holes, strips beside them), the side flange, the foot, and paint."""
    o = RAIL_O
    out = []
    y_out, y_in = HY - OUT_EDGE, HY + IN_EDGE
    out.append(_box(XB, XB + FL_T, y_out, HY - HH, Z_FL0, Z_FL1, STEEL, o))
    out.append(_box(XB, XB + FL_T, HY + HH, y_in, Z_FL0, Z_FL1, STEEL, o))
    edges = [Z_FL0] + [v for z in HOLE_Z for v in (z - HH, z + HH)] + [Z_FL1]
    for z0, z1 in zip(edges[0::2], edges[1::2]):
        out.append(_box(XB, XB + FL_T, HY - HH, HY + HH, z0, z1, STEEL, o))
    out.append(_box(XB - WEB_D, XB + FL_T, y_out, y_out + FL_T, Z_FL0, Z_FL1, STEEL, o))
    # the steel foot the rail stands in, clamped to the bench: a base plate and an angle bolted to the side flange
    out.append(_box(XB - 0.060, XB + 0.045, y_out - 0.025, y_in + 0.020, 0.0, Z_FL0, DARKSTEEL, o))
    out.append(_box(XB - WEB_D, XB - 0.002, y_out - 0.004, y_out, Z_FL0, 0.045, DARKSTEEL, o))
    for zb in (0.016, 0.034):  # bolt heads on the angle
        out.append(CylShape((XB - 0.016 - o[0], y_out - 0.0045 - o[1], zb + TOP - o[2]), 0.003, 0.002, BLACK, False,
                            "Y"))
    # U lines and numbers on both faces of the flange's outer strip (visual)
    for xf, mirror in ((XB - 0.00025, True), (XB + FL_T + 0.00025, False)):
        y0, y1, yc = y_out + 0.002, HY - HH - 0.002, HY - OUT_EDGE / 2 - 0.004
        for n in range(N_U + 1):
            z = Z_U0 + n * U
            out.append(_box(xf - 0.00015, xf + 0.00015, y0, y1, z - 0.0003, z + 0.0003, WHITE, o, False))
        for n in range(N_U):
            out += _digit(str(n + 1), xf, yc, Z_U0 + (n + 0.5) * U, o, mirror)
    # the marked hole: a yellow paint-pen square round it on both faces and a stripe on the outer strip
    for xf in (XB - 0.0003, XB + FL_T + 0.0003):
        r0, r1 = HH + 0.0006, HH + 0.0022
        for (y0, y1, z0, z1) in ((HY - r1, HY + r1, HZ + r0, HZ + r1), (HY - r1, HY + r1, HZ - r1, HZ - r0),
                                 (HY - r1, HY - r0, HZ - r0, HZ + r0), (HY + r0, HY + r1, HZ - r0, HZ + r0)):
            out.append(_box(xf - 0.0001, xf + 0.0001, y0, y1, z0, z1, YELLOW, o, False))
        out.append(_box(xf - 0.0001, xf + 0.0001, y_out + 0.003, HY - r1 - 0.001, HZ - 0.0015, HZ + 0.0015, YELLOW, o,
                        False))
    return out


RAIL = Body.build("rack", pos=RAIL_O, mass=8.0, color=STEEL, fixed=True, friction=(0.45, 0.4), contact_offset=0.001,
                  shapes=_rail_shapes())

# -- the cage nut (its frame: a along its axis, out of the face that goes against the flange; w toward a lip; s = a x w)
NUT_S, NUT_T = 0.0100, 0.0048  # M6 square nut: 10 mm across flats, 4.8 mm thick
SHEET = 0.0006  # spring-steel cage
CAGE_W, CAGE_S = 0.0101, NUT_S + 2 * SHEET  # the cage: 10.1 mm toward the lips, 11.2 mm across
CAGE_D = NUT_T + 2 * SHEET  # back plate + nut + front frame
BORE = 0.0050  # M6 minor diameter (the passage); thread major 6 mm (visual)
LIP_W_IN = 0.0039  # the lips' inner legs: outer face 3.9 mm from the axis (7.8 mm neck inside the 9.53 mm hole)
LIP_WIDTH = 0.0050  # each lip is 5 mm wide (along s)
LIP_FOLD = (0.00475, -0.00015, 0.00045)  # fold centre (a, w from the root) and radius
LIP_P1 = (0.00475, 0.00015)  # outer leg centreline from the fold ...
LIP_END = (0.0026, 0.00266)  # ... to its end (end radius 0.3 mm): outer face 6.86 mm from the axis (13.72 mm across),
# the end's back 2.3 mm in front of the cage's face (the clamp gap: panels 1.83-2.64 mm)
LIP_KG, NUT_KG = 0.0003, 0.0049  # 5.5 g in all
LIP_K, LIP_C = 0.02, 0.0003  # a lip's spring about its root: N m/rad, N m s/rad
LIP_OPEN, LIP_CLOSE = 10.0, 50.0  # degrees the lip may open / close from relaxed
NUT_CO, NUT_IT = 0.0005, 48  # contact offset (m) and solver iterations of the nut and its lips
ZINC, NUTC, BOREC = (0.78, 0.79, 0.80), (0.55, 0.56, 0.58), (0.12, 0.12, 0.13)
NUT_START = (0.15, 0.10, TOP + 0.0052 + 0.0006 + 0.001)  # face down on the bench, resting on its lips' folds


def _cage_shapes() -> list:
    """The nut in its cage, in the frame of the middle of the cage's face (a, w, s): a block 10.1 (w) x 11.2 (s) x 6.0
    (a) mm with the M6 bore through it (5 mm square passage), the nut's face and the threaded bore (visual)."""
    hw, hs, p = CAGE_W / 2, CAGE_S / 2, BORE / 2
    a0, a1 = -CAGE_D, 0.0
    ac, ad = (a0 + a1) / 2, a1 - a0
    out = [BoxShape((ac, 0.0, (p + hs) / 2), (ad, CAGE_W, hs - p), ZINC),
           BoxShape((ac, 0.0, -(p + hs) / 2), (ad, CAGE_W, hs - p), ZINC),
           BoxShape((ac, (p + hw) / 2, 0.0), (ad, hw - p, 2 * p), ZINC),
           BoxShape((ac, -(p + hw) / 2, 0.0), (ad, hw - p, 2 * p), ZINC)]
    out.append(BoxShape((0.0001, 0.0, 0.0), (0.0002, 2 * (LIP_W_IN - SHEET), 0.0060), NUTC, collide=False))
    out.append(CylShape((-CAGE_D / 2 + 0.0002, 0.0, 0.0), 0.0030, CAGE_D + 0.0004, BOREC, collide=False, axis="X"))
    return out


def _lip_shapes(sign: float) -> list:
    """A lip in its root's frame (root on the cage's face at w = +-3.9 mm): inner leg, fold, outer leg, rounded end."""
    s = sign
    fa, fw, fr = LIP_FOLD
    out = [BoxShape((fa / 2, -s * SHEET / 2, 0.0), (fa, SHEET, LIP_WIDTH), ZINC),
           CylShape((fa, s * fw, 0.0), fr, LIP_WIDTH, ZINC, axis="Z")]
    p1 = (LIP_P1[0], s * LIP_P1[1], 0.0)
    p2 = (LIP_END[0], s * LIP_END[1], 0.0)
    d = np.subtract(p2, p1)
    out.append(plate(p1, p2, LIP_WIDTH, SHEET, (d[1], -d[0], 0.0), ZINC))
    out.append(CylShape(p2, SHEET / 2, LIP_WIDTH, ZINC, axis="Z"))
    return out


def _cage_nut(face_centre, A, W) -> list:
    """The cage nut as three bodies: the nut in its cage, and two lips (a: +w, b: -w) on sprung hinges at their roots.
    ``face_centre`` is the middle of the cage's face, ``A`` its axis and ``W`` the direction of lip a (world)."""
    A, W = np.asarray(A, float), np.asarray(W, float)
    S = np.cross(A, W)
    q = tuple(float(v) for v in _quat_from_matrix(np.column_stack([A, W, S])))
    c = np.asarray(face_centre, float)
    out = [Body.build("cage_nut", pos=tuple(float(v) for v in c), mass=NUT_KG, color=ZINC, init_quat=q,
                      friction=(0.5, 0.45), contact_offset=NUT_CO, solver_iterations=NUT_IT, shapes=_cage_shapes())]
    i = int(np.argmax(np.abs(S)))
    ax, sgn = "XYZ"[i], float(np.sign(S[i]))  # the hinges turn about S (a world axis at the start)
    for sign, lab in ((1.0, "a"), (-1.0, "b")):
        root = c + sign * LIP_W_IN * W
        lo, hi = sorted((sgn * sign * -LIP_CLOSE, sgn * sign * LIP_OPEN))
        out.append(Body.build(f"lip_{lab}", pos=tuple(float(v) for v in root), mass=LIP_KG, color=ZINC,
                              init_quat=q, friction=(0.5, 0.45), contact_offset=NUT_CO, solver_iterations=NUT_IT,
                              shapes=_lip_shapes(sign),
                              joint=Joint(anchor=tuple(float(v) for v in root), kind="revolute", axis=ax,
                                          parent="cage_nut", limits=(lo, hi), stiffness=LIP_K, damping=LIP_C,
                                          target=0.0)))
    return out


# -- the flat-blade screwdriver (its frame: x from the blade's tip toward the handle, y across the blade, z along it) -
SD_TIP_T, SD_TIP_W, SD_TIP_L = 0.0006, 0.0035, 0.009  # tip 0.6 x 3.5 mm (ISO 2380-1), its thin part 9 mm long
SD_WEDGE_T, SD_WEDGE_L = 0.0012, 0.004  # then 1.2 mm thick for 4 mm, then the round shank
SD_SHANK_R, SD_SHANK_END = 0.00175, 0.060  # 3.5 mm shank, 60 mm from the tip
SD_HANDLE = (0.066, 0.161, 0.025)  # fluted handle: from, to, 25 mm across
SD_KG = 0.040
SD_START = (0.215, -0.11)  # the tip's (x, y) where it lies on the bench, pointing toward the robot (-x)
HANDLE_C, BLADE_C, SHANKC = (0.85, 0.55, 0.10), (0.60, 0.61, 0.63), (0.70, 0.71, 0.73)


def _screwdriver_shapes() -> list:
    out = [BoxShape((SD_TIP_L / 2, 0.0, 0.0), (SD_TIP_L, SD_TIP_T, SD_TIP_W), BLADE_C),
           BoxShape((SD_TIP_L + SD_WEDGE_L / 2, 0.0, 0.0), (SD_WEDGE_L, SD_WEDGE_T, SD_TIP_W), BLADE_C)]
    x0 = SD_TIP_L + SD_WEDGE_L
    out.append(CylShape(((x0 + SD_SHANK_END) / 2, 0.0, 0.0), SD_SHANK_R, SD_SHANK_END - x0, SHANKC, axis="X"))
    out.append(CylShape(((SD_SHANK_END + SD_HANDLE[0]) / 2, 0.0, 0.0), 0.0055, SD_HANDLE[0] - SD_SHANK_END, SHANKC,
                        axis="X"))
    h0, h1, d = SD_HANDLE
    for k in range(4):  # four flutes round the handle: flats at +-y (where it is gripped) and +-z (it rests on one)
        a = k * math.pi / 4
        out.append(BoxShape(((h0 + h1) / 2, 0.0, 0.0), (h1 - h0, d, 0.58 * d), HANDLE_C,
                            quat_xyzw=(math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))))
    return out


SCREWDRIVER = Body.build("screwdriver", pos=(SD_START[0], SD_START[1], TOP + SD_HANDLE[2] / 2 + 0.001), mass=SD_KG,
                         color=HANDLE_C, friction=(0.4, 0.35), contact_offset=0.001, solver_iterations=32,
                         shapes=_screwdriver_shapes())

OBJECTS = [RAIL, *_cage_nut(NUT_START, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), SCREWDRIVER]

# -- scoring ----------------------------------------------------------------------------------------------------------
LIFTED_Z, BENCH_Z = 0.03, 0.02  # the nut lifted: its middle 3 cm above the bench; fallen: back below 2 cm, unheld
HELD_R = 0.06  # a gripper this close to the nut's middle may be holding it (it has not fallen then)
CLEAR_R = 0.03  # released: both grippers' centres at least this far from the nut's middle (fingertips 1.7 cm)
SD_OUT = 0.010  # the screwdriver's tip at least this far in front of the flange (or CLEAR_R from the nut's middle): out of the hole
FLAT_DEG = 8.0  # seated: the nut's axis within 8 deg of the flange's normal ...
FACE_BACK, FACE_FWD = 0.0012, 0.0005  # ... its face from 1.2 mm behind to 0.5 mm into the flange's back face ...
CENTRE_Y, CENTRE_Z = 0.0015, 0.0035  # ... its middle within 1.5 mm (across) and 3.5 mm (up/down) of the hole's middle
OUT_PAST = 0.0002  # a lip is out when its end's middle is in front of the front face and 0.2 mm past the hole's edge
BENT_DEG = 30.0  # a lip forced 30 deg open (20 deg past its stop) has been bent: the nut is ruined
STILL_STEPS, STILL_MM = 20, 0.5
_DEBUG = os.environ.get("CAGE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("CAGE_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", poses={}, lifted=False, seated=False, one_out=False,
               free=False, near=1.0, recent=[], outs=(None, None), angles=(0.0, 0.0), stage=0.0)
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("CAGE_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _pose(base, name, origin):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - origin, rotation(p[3:7])


def _lip_angle(pn, pl, sign) -> float:
    """A lip's opening angle (deg) relative to the nut: its a-axis turned toward its own side (+) or in (-)."""
    a_l = pn[1].T @ pl[1][:, 0]
    return math.degrees(math.atan2(sign * a_l[1], a_l[0]))


def _lip_end(pl, sign) -> np.ndarray:
    """World position of a lip's rounded end (its middle)."""
    return pl[0] + pl[1] @ np.array([LIP_END[0], sign * LIP_END[1], 0.0])


def _out_at(e, sign):
    """The hole (index into HOLE_Z) over whose edge this lip's end (world) has sprung out, or None."""
    if e[0] < XB + FL_T or sign * (e[1] - HY) < HH + OUT_PAST:
        return None
    for k, z in enumerate(HOLE_Z):
        if abs(e[2] - TOP - z) < HH:
            return k
    return None


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the nut's, its lips' and the screwdriver's poses; whether the nut
    is seated, hangs by one lip or is free of both grippers and the screwdriver; the rules it may break."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = np.asarray(ik.tcp_pose(q[_ARM_IDS[side]], station.BASES[side])[0], float)
    origin = as_numpy(base.scene.env_origins[0])
    P = {k: _pose(base, k, origin) for k in ("cage_nut", "lip_a", "lip_b", "screwdriver")}
    pn, rn = P["cage_nut"]
    A = rn[:, 0]
    mid = pn - (CAGE_D / 2) * A
    outs, angles, ends = [], [], []
    for lab, sign in (("a", 1.0), ("b", -1.0)):
        e = _lip_end(P[f"lip_{lab}"], sign)
        ends.append(e)
        outs.append(_out_at(e, sign))
        angles.append(_lip_angle(P["cage_nut"], P[f"lip_{lab}"], sign))
    near = min(float(np.linalg.norm(v - mid)) for v in _TCP.values())
    sp = P["screwdriver"][0]
    sd_clear = sp[0] >= XB + FL_T + SD_OUT or float(np.linalg.norm(sp - mid)) >= CLEAR_R
    z = mid[2] - TOP
    if z > LIFTED_Z:
        _EP["lifted"] = True
    if not _EP["failed"]:
        if _EP["lifted"] and z < BENCH_Z and near > HELD_R:
            _EP["failed"] = "the nut fell onto the bench"
        elif any(o is not None and o != MARK for o in outs):
            _EP["failed"] = "a lip sprang out at another hole"
        elif max(angles) > BENT_DEG:
            _EP["failed"] = "a lip was forced open: the nut's clip is bent"
    flat = A[0] >= math.cos(math.radians(FLAT_DEG))
    face_ok = XB - FACE_BACK <= pn[0] <= XB + FACE_FWD
    centred = abs(mid[1] - HY) <= CENTRE_Y and abs(z - HZ) <= CENTRE_Z
    seated = outs[0] == MARK and outs[1] == MARK and flat and face_ok and centred
    one_out = (outs[0] == MARK) != (outs[1] == MARK)
    free = near >= CLEAR_R and sd_clear
    stage = 0.5 if free and (seated or one_out) else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)
    _EP.update(poses=P, seated=seated, one_out=one_out, free=free, near=near, outs=tuple(outs), angles=tuple(angles),
               ends=ends, stage=stage, mid=mid)


def _tally() -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "mid" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["mid"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    return dict(still=still, done=bool(_EP["seated"] and _EP["free"] and still and not _EP["failed"]))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or "mid" not in _EP:
        return
    pn, rn = _EP["poses"]["cage_nut"]
    ax = rn[:, 0]
    ends = ""
    for (lab, sign), e in zip((("A", 1.0), ("B", -1.0)), _EP["ends"]):
        ends += (f" end{lab}(front+{(e[0] - XB - FL_T) * 1000:.2f} past+{(sign * (e[1] - HY) - HH) * 1000:.2f} "
                 f"dz{(e[2] - TOP - HZ) * 1000:.1f})")
    print(f"DBG step={_EP['step']} nut face=({pn[0]:.4f},{pn[1]:.4f},{pn[2] - TOP:.4f}) "
          f"axis=({ax[0]:.3f},{ax[1]:.3f},{ax[2]:.3f}) lipA={_EP['angles'][0]:.1f} lipB={_EP['angles'][1]:.1f}{ends}",
          flush=True)
    sp, sr = _EP["poses"]["screwdriver"]
    print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.4f},{v[1]:.4f},{v[2] - TOP:.4f})" for k, v in _TCP.items())
          + f" | sd tip=({sp[0]:.4f},{sp[1]:.4f},{sp[2] - TOP:.4f})", flush=True)
    print(f"DBG   stage={_EP['stage']} best={_EP['best']} seated={_EP['seated']} one_out={_EP['one_out']} "
          f"free={_EP['free']} near={_EP['near']:.3f} outs={_EP['outs']} still={t['still']} "
          f"failed={_EP['failed'] or '-'}", flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the nut has fallen onto the bench, a lip has sprung out
    at another hole or been forced open): 0.5 the nut sits in the marked hole by itself, at least one lip sprung out
    over the front face, nothing touching it. Full score (1.0) is success()."""
    t = _tally()
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """Seated in the marked hole (both lips sprung out over the front face, the nut flat against the flange's back
    face and centred in the hole), both grippers' centres 3 cm or more from it, the screwdriver out of the hole, still
    for 1 s, no rule broken."""
    return _tally()["done"]


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the rack makers' guides say: the left gripper takes the cage nut from the bench, turns it to face the rail and,
# from behind the rail, tilts it 45 deg and hooks its +y lip through the marked hole (the lip's end over the front face
# past the hole's +y edge). The right gripper brings the screwdriver and slides its blade into the hole along the hole's
# -y side, 6.8 mm past the flange, where it keeps the far lip in. The left turns the nut flat about its hooked lip and
# presses it home: the far lip, kept in line by the blade and then pressed in by the hole's edge, passes the hole. The
# right draws the blade out and the far lip springs out over the front face; the left lets go and both move away.
# CAGE_ONE=1 runs it the wrong way: the blade stays out and the left hand turns the nut home alone.

WRONG_ONE = os.environ.get("CAGE_ONE") == "1"
OPEN_C = -0.030
PINCH, PINCH_LIGHT = -0.0028, -0.0048  # fingers 2.8 mm inside each face of the 11.2 mm cage; 0.8 mm while turning it
FACE_OUT = 0.002  # the nut's face stands this far past the fingertips
TIP = 0.013  # fingertips past the TCP
HOOK_PHI = math.radians(45.0)  # the nut turned this far (its +y side toward the rail) to hook its +y lip
HOOK_PAST = 0.0015  # the hooked lip's end this far past the hole's +y edge ...
HOOK_FRONT = 0.0008  # ... and its middle this far in front of the front face (its back 0.5 mm clear)
ENTRY_IN = 0.0004  # on the way in, that end this far inside the +y edge
PULL_SNUG = 0.0004  # while the nut is turned flat, the hooked lip's end is pulled this far against the front face ...
PRESS_HOOK = 0.0025  # ... and the nut pressed this far toward the hooked side
T_TURN = 4.0
T_HOOKED = 14.9  # the left gripper has the nut hooked by its +y lip at this time (s): checked in _left()
T_SD_IN = T_HOOKED + 0.5  # the screwdriver goes in ...
T_PUSH = T_SD_IN + 2.5  # the nut is turned home ...
T_SD_OUT = T_PUSH + 3.6  # the screwdriver comes out, near the end of the turn ...
T_LET_GO = T_SD_OUT + 1.5  # the left gripper lets go
SEAT = np.array([XB - 0.0006, HY + 0.0004, TOP + HZ])  # pressed home: its face aimed 0.6 mm into the flange


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


DOWN_L = grip_rotation(-math.pi / 2)  # left arm straight down, fingers closing along x
LEVEL_L = _ry(-math.pi / 2) @ DOWN_L  # the same grip turned to point +x (fingers closing along z)
DOWN_SD = grip_rotation(0.0)  # right arm straight down, fingers closing along y (across the screwdriver's handle)


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


def _hook_centres():
    """The nut's axis and two face centres (world), turned HOOK_PHI about z with its +y side toward the rail: hooked
    (its +y lip's end over the front face past the hole's +y edge) and entry (that end just inside the edge)."""
    A = np.array([math.cos(HOOK_PHI), -math.sin(HOOK_PHI), 0.0])
    W = np.array([math.sin(HOOK_PHI), math.cos(HOOK_PHI), 0.0])
    end = LIP_END[0] * A + (LIP_W_IN + LIP_END[1]) * W
    hooked = np.array([XB + FL_T + HOOK_FRONT, HY + HH + HOOK_PAST, TOP + HZ]) - end
    entry = np.array([XB + FL_T + HOOK_FRONT, HY + HH - SHEET / 2 - ENTRY_IN, TOP + HZ]) - end
    return A, W, hooked, entry


def _left() -> list:
    rot_dn = DOWN_L
    grab = np.array([NUT_START[0], NUT_START[1], TOP + 0.0052 + FACE_OUT + TIP])
    out = _lead_in("left", grab + [0.0, 0.0, 0.06], rot_dn, OPEN_C, "above the cage nut")
    out += [_w(1.2, grab, OPEN_C, "fingers either side of the nut", rot_dn),
            _w(0.4, grab, OPEN_C, "settle", rot_dn),
            _w(0.8, grab, PINCH, "take the nut", rot_dn),
            _w(0.6, grab, PINCH, "settle", rot_dn),
            _w(1.0, grab + [0.0, 0.0, 0.07], PINCH, "lift it", rot_dn)]
    A, W, hooked, entry = _hook_centres()
    rot_h = _rz(-HOOK_PHI) @ LEVEL_L
    off = (TIP + FACE_OUT) * A  # face centre = TCP + off
    pre = entry - off - 0.012 * A
    out += [_w(1.5, np.array([0.16, 0.05, TOP + HZ + 0.02]), PINCH, "turn it to face the rail", _ry(-math.pi / 4) @ DOWN_L),
            _w(1.5, np.array([0.18, HY + 0.004, TOP + HZ]), PINCH, "turn it to face the rail", LEVEL_L),
            _w(1.5, pre, PINCH, "behind the marked hole, tilted 45 deg", rot_h),
            _w(0.5, pre, PINCH, "settle", rot_h),
            _w(1.5, entry - off, PINCH, "the +y lip into the hole", rot_h),
            _w(0.4, entry - off, PINCH, "settle", rot_h),
            _w(1.0, hooked - off, PINCH, "slide it toward +y: the lip's end over the front face (hooked)", rot_h)]
    assert abs(sum(w.duration for w in out) - T_HOOKED) < 1e-6, sum(w.duration for w in out)
    out += [_w(T_PUSH - T_HOOKED, hooked - off, PINCH, "hold it hooked while the blade goes in", rot_h)]
    # turn it flat about the hooked lip's end (pulled snug onto the front face and pressed toward that side)
    end = hooked + LIP_END[0] * A + (LIP_W_IN + LIP_END[1]) * W - [HOOK_FRONT + PULL_SNUG, -PRESS_HOOK, 0.0]
    n = 4
    for k in range(1, n + 1):
        phi = HOOK_PHI * (1.0 - k / n)
        Ak = np.array([math.cos(phi), -math.sin(phi), 0.0])
        Wk = np.array([math.sin(phi), math.cos(phi), 0.0])
        ck = end - LIP_END[0] * Ak - (LIP_W_IN + LIP_END[1]) * Wk
        out.append(_w(T_TURN / n, ck - (TIP + FACE_OUT) * Ak, PINCH_LIGHT,
                      "turn it flat about the hooked lip, pushing it in", _rz(-phi) @ LEVEL_L))
    seat_tcp = SEAT - (TIP + FACE_OUT) * np.array([1.0, 0.0, 0.0])
    away = seat_tcp + [-0.016, 0.03, 0.01]
    out += [_w(0.6, seat_tcp, PINCH_LIGHT, "press it home, flat against the flange", LEVEL_L),
            _w(T_LET_GO - T_PUSH - T_TURN - 0.6, seat_tcp, PINCH_LIGHT, "hold it there", LEVEL_L),
            _w(0.8, seat_tcp, OPEN_C, "let go", LEVEL_L),
            _w(1.0, away, OPEN_C, "back away", LEVEL_L),
            _w(4.0, away, OPEN_C, "stay clear", LEVEL_L)]
    return out


PINCH_SD = -(SD_HANDLE[2] / 2 - 0.003)
SD_GRIP_L = 0.075  # the right gripper takes the handle here (from the tip)
SD_AXIS_BELOW = 0.004  # the screwdriver's axis this far below the TCP in the grip
SD_LINE_Y = HY - HH + SD_TIP_T / 2 + 0.0010  # the blade enters 1 mm inside the hole's -y side ...
SD_WALL_Y = HY - HH + SD_TIP_T / 2 - 0.0025  # ... then is pressed against that side (aimed 2.5 mm into it)
SD_DEPTH = 0.0068  # the tip this far behind the flange's back face


def _right() -> list:
    rot = DOWN_SD
    grab = np.array([SD_START[0] + SD_GRIP_L, SD_START[1], TOP + SD_HANDLE[2] / 2 + SD_AXIS_BELOW])
    lift = grab + [0.0, 0.0, 0.10]
    front = np.array([XB + FL_T + 0.010 + SD_GRIP_L, SD_LINE_Y, TOP + HZ + SD_AXIS_BELOW])
    entered = front - [0.010 + 0.001, 0.0, 0.0]  # tip 1 mm into the hole
    at_wall = np.array([entered[0], SD_WALL_Y, entered[2]])
    inside = at_wall - [0.001 + FL_T + SD_DEPTH, 0.0, 0.0]
    away = front + [-0.01, -0.05, -0.02]
    out = _lead_in("right", grab + [0.0, 0.0, 0.06], rot, OPEN_C, "above the screwdriver's handle")
    out += [_w(1.2, grab, OPEN_C, "fingers either side of the handle", rot),
            _w(0.4, grab, OPEN_C, "settle", rot),
            _w(0.8, grab, PINCH_SD, "take the screwdriver", rot),
            _w(0.6, grab, PINCH_SD, "settle", rot),
            _w(1.5, lift, PINCH_SD, "lift it", rot),
            _w(2.0, front + [0.0, 0.0, 0.012], PINCH_SD, "in front of the marked hole", rot),
            _w(1.5, front, PINCH_SD, "blade level with the hole, near its -y side", rot)]
    t = sum(w.duration for w in out)
    if WRONG_ONE:
        return out + [_w(T_SD_OUT - t, front, PINCH_SD, "the blade stays out of the hole", rot),
                      _w(1.5, away, PINCH_SD, "move away", rot),
                      _w(6.0, away, PINCH_SD, "stay clear", rot)]
    out += [_w(T_SD_IN - t, front, PINCH_SD, "wait for the nut to be hooked", rot),
            _w(0.6, entered, PINCH_SD, "blade tip into the hole", rot),
            _w(0.5, at_wall, PINCH_SD, "against the hole's -y side", rot),
            _w(1.4, inside, PINCH_SD, "blade in along that side, past the far lip", rot),
            _w(T_SD_OUT - T_SD_IN - 2.5, inside, PINCH_SD, "keep the far lip in line while the nut is turned home",
               rot),
            _w(1.2, front, PINCH_SD, "draw the blade out: the far lip springs out over the front face", rot),
            _w(1.5, away, PINCH_SD, "move away", rot),
            _w(6.0, away, PINCH_SD, "stay clear", rot)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
