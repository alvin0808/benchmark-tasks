"""On the station table: a 14-inch children's bike standing upright, front wheel toward the robot, its rear wheel in a
slot-type floor stand. Its front wheel is held by a V-brake (linear-pull brake): two arms on pivots on the fork legs, sprung
apart, joined across the top of the tyre by the brake cable. On the bike's left arm (the robot's right, -y) a link holds
the end of the cable's curved metal "noodle" in a cup; the bare cable runs from the noodle across to its anchor bolt on
the right arm. The arms' springs keep the cable tight, so the noodle's end is pressed into the cup and cannot be lifted
out. Squeezed together, the arms slacken the cable and the cup moves off the noodle's end, which can then be lifted out
of the link; let go, the arms spring wide apart and the brake pads clear the tyre, so the wheel can come out.

Task (instruction): open the front brake so the wheel can be taken out. The instruction does not say how. The procedure
the task checks is the one in Park Tool's wheel removal guide ("Squeeze linear-pull calipers together and disconnect
cable 'noodle' from linkage") and in a bike maker's help page ("If you squeeze the brake arms together, you can pull the
noodle out of the black or silver tab that it is hooked into"). One gripper squeezes the two arms; the other lifts the
noodle out. Pulled without the squeeze, the noodle stays in its cup (the arms only swing together as one, keeping the
cable tight); squeezed with nobody at the noodle, the arms spring back shut onto it when let go.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. The bike's
front faces the robot (-x), so its left side is at -y. Its fork leans back 20 degrees; "along the arm" means along the
fork, 20 degrees back from vertical.
- bike: fixed; wheelbase 53 cm, front axle at x 0.217, 17.15 cm up, rear axle at x 0.747 in the stand; handlebar
  50 cm wide, grips 59 cm up; saddle top 53 cm up. Front wheel: ISO 254 rim (14 inch), 14 x 1.75 tyre, 343 mm across.
- brake arms: pivots 80 mm apart, 88 mm (along the fork) below the cable; arm tops 6 mm below it; pads 37 mm up
  the arm, 3 mm off the rim each side. Each arm is on a spring that pushes its top outward (7.1 N at the cable when
  closed); an arm turns at most 4.6 degrees inward (pad on the rim) and 18 degrees outward (open; pads 54 mm apart).
- cable: crosses 1.1 cm above the tyre, its centre at x 0.26, 34.8 cm up. Noodle: a 90-degree aluminium tube, 6 mm
  across, end ferrule 8 mm across, seated 4 mm deep in the link's cup (its end at y -0.010); it rises 7.5 cm to a
  rubber boot and the housing.

Score: 0 for the rest of the run once the noodle has been forced up out of its cup (the link bent). Otherwise the best
stage reached: 0.25 the cup cleared off the noodle's end (the brake squeezed far enough, 4.5 mm or more); 0.5 the noodle
lifted out of the link (its end 12 mm or more above the cup); 0.75 at the end, the noodle out of the link and the brake
sprung open, pads 46 mm or more apart (wider than the 44 mm tyre). Full score: that, with both grippers 5 cm or more
from the noodle and the arms, everything still, for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, capsule, on_reset,  # noqa: E402
                     on_step, rotation)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout -----------------------------------------------------------------------------------------------------------
LEAN = math.radians(20.0)  # the fork leans back (toward +x) this much from vertical
U = np.array([math.sin(LEAN), 0.0, math.cos(LEAN)])  # along the fork / the brake arms, upward
LX = np.array([math.cos(LEAN), 0.0, -math.sin(LEAN)])  # across the fork, toward the back of the bike
YY = np.array([0.0, 1.0, 0.0])
R_LEAN = np.column_stack([LX, YY, U])

TYRE_R, TYRE_HW = 0.1715, 0.022  # 14 x 1.75 tyre: outer radius, half width
RIM_R, RIM_HW = 0.134, 0.0125  # rim's outer edge radius, half width (25 mm)
CABLE_D = 0.181  # the cable's centre from the axle, along the fork line
BRAKE_FWD = 0.020  # the brake (arms, cable) sits this far in front of the fork legs' centre line
AXLE = np.array([0.217, 0.0, TOP + TYRE_R])
C = AXLE + CABLE_D * U - BRAKE_FWD * LX  # the middle of the cable, between the arm tops
WHEELBASE = 0.53
REAR_AXLE = AXLE + np.array([WHEELBASE, 0.0, 0.0])

ARM_L = 0.092  # from an arm's pivot to its top face (along the fork); the cable is 88 mm up
PIVOT_C, TOP_C = -0.088, -0.006  # pivot and arm top, along U from the cable's line
PIVOT_Y, TOP_Y = 0.040, 0.036  # arm centre lines' distance from the bike's mid plane, at the pivot and at the top
ARM_T, ARM_W = 0.008, 0.012  # arm thickness (y) and width (across the fork)
PAD_C = PIVOT_C + 0.037  # pad centre, along U
PAD_GAP = 0.003  # each pad's clearance to the rim
PAD_FACE = RIM_HW + PAD_GAP  # pad face's distance from the mid plane
SQUEEZE_DEG = math.degrees(math.atan(PAD_GAP / (PAD_C - PIVOT_C)))  # an arm turns this far in before its pad hits
OPEN_DEG = 18.0  # and this far out, unhooked
SPRING_K, SPRING_TARGET = 0.8, 45.0  # N m/rad, degrees outward: 0.63 N m (7.1 N at the cable) when closed
Y_TIP = -0.010  # the noodle's end (the ferrule's face) and the cup's seat
CUP_D, CUP_IN, CUP_WALL = 0.004, 0.0045, 0.003  # the cup: depth, inner half size, wall
FERRULE_R, FERRULE_L, TUBE_R = 0.004, 0.006, 0.003
BEND_Y, BEND_R, LEG_Y = -0.018, 0.010, -0.028  # the noodle's bend: start (y), radius; its upright leg (y)
LEG_TOP, BOOT_TOP, HOUSING_TOP = 0.075, 0.088, 0.115  # along U from the cable's line
ANCHOR_Y = TOP_Y  # the cable's anchor bolt, on the right arm's top
SLACK_MAX = 0.025  # the noodle can come this far toward the anchor when the cable is slack

ALU, DARK, RUBBER, RED, PAD, STEEL, CABLE = ((0.72, 0.73, 0.75), (0.12, 0.12, 0.13), (0.08, 0.08, 0.08),
                                             (0.78, 0.14, 0.12), (0.20, 0.20, 0.22), (0.55, 0.56, 0.58),
                                             (0.80, 0.80, 0.82))
FRAME = (0.13, 0.42, 0.72)  # a blue children's bike


def L(a: float, b: float, c: float) -> np.ndarray:
    """A point in the brake's frame: a across the fork (+back), b sideways (+y), c along the fork (+up), from C."""
    return C + a * LX + b * YY + c * U


def _quat(m) -> tuple:
    from bodykit import _quat_from_matrix

    return _quat_from_matrix(np.asarray(m, float))


Q_LEAN = _quat(R_LEAN)


def lbox(centre_world, origin, size, color, collide=True) -> BoxShape:
    """A box turned with the fork (size along across, sideways, along), centred at a world point."""
    return BoxShape(tuple((np.asarray(centre_world) - origin).tolist()), tuple(size), color, collide=collide,
                    quat_xyzw=Q_LEAN)


def cap(p1, p2, r, origin, color, collide=True):
    return capsule(tuple((np.asarray(p1) - origin).tolist()), tuple((np.asarray(p2) - origin).tolist()), r, color,
                   collide)


# -- the bike (fixed) -------------------------------------------------------------------------------------------------
BIKE_ORIGIN = np.array([AXLE[0], 0.0, TOP])


def _wheel(axle, out, n=28, collide_tyre=True, hub_y=0.05):
    o = BIKE_ORIGIN
    ring_r = TYRE_R - TYRE_HW
    pts = [axle + ring_r * np.array([math.cos(2 * math.pi * k / n), 0.0, math.sin(2 * math.pi * k / n)])
           for k in range(n)]
    for k in range(n):
        out.append(cap(pts[k], pts[(k + 1) % n], TYRE_HW, o, RUBBER, collide_tyre))
    m = 24
    rim = [axle + (RIM_R - 0.006) * np.array([math.cos(2 * math.pi * k / m), 0.0, math.sin(2 * math.pi * k / m)])
           for k in range(m)]
    for k in range(m):
        a, b = rim[k], rim[(k + 1) % m]
        d = b - a
        out.append(BoxShape(tuple(((a + b) / 2 - o).tolist()), (0.012, 2 * RIM_HW, float(np.linalg.norm(d))), ALU,
                            collide=False, quat_xyzw=_quat(np.column_stack([np.cross(YY, d / np.linalg.norm(d)), YY,
                                                                            d / np.linalg.norm(d)]))))
    out.append(CylShape(tuple((axle - o).tolist()), 0.016, 2 * hub_y - 0.01, ALU, collide=False, axis="Y"))
    out.append(CylShape(tuple((axle - o).tolist()), 0.005, 2 * hub_y + 0.01, STEEL, collide=False, axis="Y"))
    for k in range(16):
        ang = 2 * math.pi * k / 16
        side = 1.0 if k % 2 else -1.0
        a = axle + np.array([0.0, side * 0.025, 0.0]) + 0.014 * np.array([math.cos(ang), 0.0, math.sin(ang)])
        b = axle + (RIM_R - 0.012) * np.array([math.cos(ang + 0.2), 0.0, math.sin(ang + 0.2)])
        out.append(cap(a, b, 0.0011, o, STEEL, False))


def _bike_shapes() -> list:
    o = BIKE_ORIGIN
    out: list = []
    _wheel(AXLE, out)
    _wheel(REAR_AXLE, out, hub_y=0.06)
    fork_c = AXLE + 0.245 * U  # crown
    for s in (-1.0, 1.0):  # fork legs (behind the brake), dropouts to crown; not colliding (the arms ride on them)
        out.append(cap(AXLE + np.array([0.0, s * 0.048, 0.0]), fork_c + np.array([0.0, s * 0.046, 0.0]) - 0.01 * U,
                       0.010, o, FRAME, False))
        piv = L(0.0, s * PIVOT_Y, PIVOT_C)  # the brake bosses: posts from the fork leg forward to the arm's pivot
        out.append(cap(piv + BRAKE_FWD * LX, piv + 0.006 * LX, 0.005, o, STEEL, False))
    out.append(lbox(fork_c, o, (0.03, 0.115, 0.02), FRAME, False))  # crown
    ht0, ht1 = fork_c + 0.012 * U, fork_c + 0.115 * U  # head tube
    out.append(cap(ht0, ht1, 0.017, o, FRAME))
    stem = ht1 + 0.035 * U
    out.append(cap(ht1, stem, 0.012, o, DARK))
    bar_c = stem - 0.015 * LX
    grips = {}
    for s in (-1.0, 1.0):  # a riser bar, 50 cm wide: up 4 cm and back 3 cm to the grips
        bend = bar_c + np.array([0.0, s * 0.09, 0.0])
        g0 = bend + np.array([0.03, s * 0.07, 0.04])
        g1 = g0 + np.array([0.0, s * 0.09, 0.0])
        out.append(cap(bar_c, bend, 0.011, o, DARK))
        out.append(cap(bend, g0, 0.011, o, DARK))
        out.append(cap(g0, g1, 0.016, o, (0.10, 0.10, 0.10)))
        lever0 = g0 + np.array([-0.01, s * 0.01, 0.0])
        out.append(cap(lever0, lever0 + np.array([-0.035, s * 0.055, -0.012]), 0.006, o, DARK, False))
        grips[s] = lever0
    bb = AXLE + np.array([0.27, 0.0, -0.0]) + np.array([0.0, 0.0, 0.18 - TYRE_R])
    seat_dir = np.array([math.sin(math.radians(20)), 0.0, math.cos(math.radians(20))])
    st_top = bb + 0.25 * seat_dir
    out.append(cap(fork_c + 0.03 * U, bb, 0.019, o, FRAME))  # down tube
    out.append(cap(ht0 + 0.085 * U, st_top, 0.016, o, FRAME))  # top tube
    out.append(cap(bb, st_top, 0.016, o, FRAME))  # seat tube
    post = st_top + 0.08 * seat_dir
    out.append(cap(st_top, post, 0.011, o, ALU))
    out.append(BoxShape(tuple((post + np.array([0.0, 0.0, 0.02]) - o).tolist()), (0.20, 0.11, 0.04), DARK))
    for s in (-1.0, 1.0):
        ra = REAR_AXLE + np.array([0.0, s * 0.058, 0.0])
        out.append(cap(bb + np.array([0.0, s * 0.03, 0.0]), ra, 0.009, o, FRAME))  # chain stays
        out.append(cap(st_top + np.array([0.0, s * 0.02, -0.02]), ra, 0.008, o, FRAME))  # seat stays
        crank_end = bb + np.array([s * 0.06, s * 0.07, -s * 0.07]) * np.array([1, 1, 1])
        out.append(cap(bb + np.array([0.0, s * 0.06, 0.0]), crank_end, 0.007, o, ALU, False))
        out.append(BoxShape(tuple((crank_end + np.array([0.0, s * 0.04, 0.0]) - o).tolist()), (0.03, 0.07, 0.015),
                            DARK, collide=False))
    out.append(CylShape(tuple((bb + np.array([0.0, -0.05, 0.0]) - o).tolist()), 0.065, 0.004, STEEL, collide=False,
                        axis="Y"))
    # the rear wheel's slot-type floor stand: a base plate and two side bars either side of the tyre
    sx = REAR_AXLE[0] - 0.10
    out.append(BoxShape((sx - o[0], 0.0, 0.004), (0.20, 0.16, 0.008), DARK))
    for s in (-1.0, 1.0):
        out.append(BoxShape((sx - o[0], s * 0.031, 0.06), (0.20, 0.012, 0.10), DARK))
    # the brake housing from above the noodle's boot to the left lever (drawn fixed; the noodle carries its own piece)
    h0 = L(0.0, LEG_Y, HOUSING_TOP)
    h1 = h0 + np.array([0.02, -0.015, 0.06])
    h2 = grips[-1.0] + np.array([-0.06, 0.07, 0.02])
    for a, b in ((h0, h1), (h1, h2), (h2, grips[-1.0])):
        out.append(cap(a, b, 0.0025, o, DARK, False))
    return out


BIKE = Body.build("bike", pos=tuple(float(v) for v in BIKE_ORIGIN), mass=7.0, color=FRAME, fixed=True,
                  friction=(0.8, 0.7), shapes=_bike_shapes())


# -- brake arms -------------------------------------------------------------------------------------------------------
def _arm_shapes(s: float) -> list:
    """An arm (s = -1: the bike's left, at -y; +1: right) in its pivot's frame: the arm, its pivot boss, the pad on its
    holder; on the left arm the link with its cup and seat for the noodle's end, on the right the cable's anchor bolt."""
    o = L(0.0, s * PIVOT_Y, PIVOT_C)
    p_bot, p_top = L(0.0, s * PIVOT_Y, PIVOT_C - 0.006), L(0.0, s * TOP_Y, TOP_C - ARM_W / 2 * 0.0)
    out = []
    d = p_top - p_bot
    n = d / np.linalg.norm(d)
    across = np.cross(YY, n)
    across /= np.linalg.norm(across)
    side = np.cross(n, across)
    out.append(BoxShape(tuple(((p_bot + p_top) / 2 - o).tolist()), (ARM_W, ARM_T, float(np.linalg.norm(d))), DARK,
                        quat_xyzw=_quat(np.column_stack([across, side, n]))))
    out.append(cap(L(-0.008, s * PIVOT_Y, PIVOT_C), L(0.008, s * PIVOT_Y, PIVOT_C), 0.008, o, DARK, False))
    y_in = s * (PIVOT_Y + (TOP_Y - PIVOT_Y) * (PAD_C - PIVOT_C) / (TOP_C - PIVOT_C)) - s * ARM_T / 2
    holder_c = (y_in + s * (PAD_FACE + 0.006)) / 2
    out.append(lbox(L(0.0, holder_c, PAD_C), o, (0.008, abs(y_in - s * (PAD_FACE + 0.006)), 0.008), STEEL, False))
    out.append(lbox(L(0.0, s * (PAD_FACE + 0.003), PAD_C), o, (0.035, 0.006, 0.010), PAD, False))
    if s < 0:  # the link: a bracket from the arm's top inward under the noodle, ending in the cup and its seat
        y_arm = -(TOP_Y - ARM_T / 2)
        y_mouth = Y_TIP - CUP_D
        out.append(lbox(L(0.0, (y_arm + y_mouth) / 2, -CUP_IN - CUP_WALL / 2), o,
                        (0.010, abs(y_mouth - y_arm) + 0.002, CUP_WALL), ALU))
        yc = Y_TIP - CUP_D / 2
        w = 2 * (CUP_IN + CUP_WALL)
        for cc in (-1.0, 1.0):
            out.append(lbox(L(0.0, yc, cc * (CUP_IN + CUP_WALL / 2)), o, (w, CUP_D, CUP_WALL), ALU))
            out.append(lbox(L(cc * (CUP_IN + CUP_WALL / 2), yc, 0.0), o, (CUP_WALL, CUP_D, 2 * CUP_IN), ALU))
        out.append(lbox(L(0.0, Y_TIP + 0.0015, 0.0), o, (w, 0.003, w), ALU))
    else:  # the anchor bolt and its clamp washer on the outside of the arm's top
        out.append(CylShape(tuple((L(0.0, TOP_Y + ARM_T / 2 + 0.003, -0.008) - o).tolist()), 0.0055, 0.006, STEEL,
                            collide=False, axis="Y"))
    return out


def _arm(s: float, name: str) -> Body:
    piv = L(0.0, s * PIVOT_Y, PIVOT_C)
    lim = (-SQUEEZE_DEG, OPEN_DEG) if s < 0 else (-OPEN_DEG, SQUEEZE_DEG)
    return Body.build(name, pos=tuple(float(v) for v in piv), mass=0.06, color=DARK, friction=(0.7, 0.6),
                      solver_iterations=32, shapes=_arm_shapes(s),
                      joint=Joint(anchor=tuple(float(v) for v in piv), kind="revolute", axis="X", limits=lim,
                                  stiffness=SPRING_K, target=(SPRING_TARGET if s < 0 else -SPRING_TARGET),
                                  damping=0.002))


LEFT_ARM = _arm(-1.0, "left_brake_arm")
RIGHT_ARM = _arm(1.0, "right_brake_arm")

# -- the cable: a stand-in for the inner wire, which only pulls -------------------------------------------------------
# The noodle hangs on the right arm's anchor bolt by a joint that lets it move up and down and turn freely about the
# cable's crossing direction (the housing bends), and slide along the cable toward the anchor, but never farther from
# it than where it starts, held in its cup: the wire cannot stretch, but goes slack.
ANCHOR = L(0.0, ANCHOR_Y, -0.008)
TIP0 = L(0.0, Y_TIP, 0.0)


def _noodle_shapes() -> list:
    """The noodle in the frame of its end's face (the ferrule's face, pressed on the seat): ferrule, straight tube, the
    90-degree bend (four short pieces), the upright leg, the rubber boot and a piece of housing; the bare cable (drawn
    only) runs from its end to the anchor bolt."""
    o = TIP0
    out = [CylShape(tuple((L(0.0, Y_TIP - FERRULE_L / 2, 0.0) - o).tolist()), FERRULE_R, FERRULE_L, ALU, axis="Y")]
    y0 = Y_TIP - FERRULE_L
    out.append(CylShape(tuple((L(0.0, (y0 + BEND_Y) / 2, 0.0) - o).tolist()), TUBE_R, abs(BEND_Y - y0), ALU,
                        axis="Y"))
    centre = (BEND_Y, BEND_R)
    prev = L(0.0, BEND_Y, 0.0)
    for k in range(1, 5):
        a = math.pi / 2 * k / 4
        p = L(0.0, centre[0] - BEND_R * math.sin(a), centre[1] - BEND_R * math.cos(a))
        out.append(cap(prev, p, TUBE_R, o, ALU))
        prev = p
    out.append(cap(prev, L(0.0, LEG_Y, LEG_TOP), TUBE_R, o, ALU))
    out.append(cap(L(0.0, LEG_Y, LEG_TOP), L(0.0, LEG_Y, BOOT_TOP), 0.0048, o, RUBBER))
    out.append(cap(L(0.0, LEG_Y, BOOT_TOP), L(0.0, LEG_Y, HOUSING_TOP), 0.0025, o, DARK, False))
    out.append(cap(L(0.0, Y_TIP + 0.001, 0.0), L(0.0, ANCHOR_Y + 0.004, 0.0), 0.0008, o, CABLE, False))
    return out


NOODLE = Body.build("noodle", pos=tuple(float(v) for v in TIP0), mass=0.02, color=ALU, gravity=False,
                    friction=(0.6, 0.5), solver_iterations=32, shapes=_noodle_shapes(),
                    joint=Joint(anchor=tuple(float(v) for v in TIP0), kind="d6", parent="right_brake_arm",
                                free=(("transY", 0.0, SLACK_MAX, 0.3), ("transZ", -0.004, 0.06, 0.3),
                                      ("rotX", -60.0, 60.0, 0.0005))))
OBJECTS = [BIKE, LEFT_ARM, RIGHT_ARM, NOODLE]
if os.environ.get("VBRAKE_NONOODLE") == "1":  # debugging: the arms on their springs alone
    OBJECTS = [BIKE, LEFT_ARM, RIGHT_ARM]

# -- scoring ----------------------------------------------------------------------------------------------------------
CLEAR_MM = (CUP_D + 0.0005) * 1000  # the cup off the noodle's end: the end this far back from the seat
OUT_MM = 12.0  # lifted out: the end this far above the cup's middle (its ferrule above the cup's top)
FORCED_RISE_MM, FORCED_DEPTH_MM = 2.5, CUP_D * 1000 - 1.0  # risen this far while still this deep in the cup: forced
OPEN_GAP_MM = 46.0  # pad faces this far apart: the 44 mm tyre passes between them
FAR = 0.05
STILL_STEPS, STILL_MM, STILL_DEG = 20, 2.0, 0.5
SEAT_LOCAL = L(0.0, Y_TIP, 0.0) - L(0.0, -PIVOT_Y, PIVOT_C)  # the seat's middle in the left arm's frame
PAD_LOCAL = {s: L(0.0, s * PAD_FACE, PAD_C) - L(0.0, s * PIVOT_Y, PIVOT_C) for s in (-1.0, 1.0)}
SQZ_C = -0.008  # where the squeezing gripper holds the arms (along U from the cable's line)
GRAB_C = 0.060  # where the other gripper holds the noodle's upright leg
GRAB_LOCAL = L(0.0, LEG_Y, GRAB_C) - TIP0
_DEBUG = os.environ.get("VBRAKE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("VBRAKE_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], back=0.0, rise=0.0, gap=31.0, angL=0.0,
               angR=0.0, tip=TIP0.copy(), grab=TIP0 + GRAB_LOCAL, points=[], out_now=False, open_now=False)
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("VBRAKE_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _pose(base, name, origin):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - origin, rotation(p[3:7])


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the arms' angles and the pads' spacing; where the noodle's end is
    in the cup's frame (how far back from the seat, how far up)."""
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
    pl, rl = _pose(base, "left_brake_arm", origin)
    pr, rr = _pose(base, "right_brake_arm", origin)
    if "noodle" not in [o.name for o in OBJECTS]:
        pn, rn = TIP0.copy(), np.eye(3)
    else:
        pn, rn = _pose(base, "noodle", origin)
    ang_l = math.degrees(math.atan2(rl[2, 1], rl[1, 1]))  # + opens the left arm
    ang_r = -math.degrees(math.atan2(rr[2, 1], rr[1, 1]))  # + opens the right arm
    seat = pl + rl @ SEAT_LOCAL
    rel = rl.T @ (pn - seat)  # the noodle's end in the left arm's (start-aligned) frame, from the seat
    back = -float(rel[1])  # how far the end is back from the seat (toward -y)
    rise = float(rel @ U)  # how far it is up (along the arm)
    gap = float((pr + rr @ PAD_LOCAL[1.0])[1] - (pl + rl @ PAD_LOCAL[-1.0])[1])
    if rise * 1000 > FORCED_RISE_MM and back * 1000 < FORCED_DEPTH_MM and abs(rel @ LX) < 0.006 \
            and not _EP["failed"]:
        _EP["failed"] = "the noodle was forced up out of its cup (the link bent)"
    out_now = rise * 1000 >= OUT_MM
    open_now = out_now and gap * 1000 >= OPEN_GAP_MM
    stage = 0.75 if open_now else 0.5 if out_now else 0.25 if back * 1000 >= CLEAR_MM else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage if stage < 0.75 else 0.5)
    tops = [pl + rl @ (L(0.0, -TOP_Y, SQZ_C) - L(0.0, -PIVOT_Y, PIVOT_C)),
            pr + rr @ (L(0.0, TOP_Y, SQZ_C) - L(0.0, PIVOT_Y, PIVOT_C)),
            pl + rl @ (L(0.0, -TOP_Y, TOP_C) - L(0.0, -PIVOT_Y, PIVOT_C)),
            pr + rr @ (L(0.0, TOP_Y, TOP_C) - L(0.0, PIVOT_Y, PIVOT_C))]
    _EP.update(back=back, rise=rise, gap=gap, angL=ang_l, angR=ang_r, tip=pn, grab=pn + rn @ GRAB_LOCAL,
               points=[pn, pn + rn @ GRAB_LOCAL] + tops, out_now=out_now, open_now=open_now)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append((np.asarray(_EP["tip"], float).copy(), _EP["angL"], _EP["angR"]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(
        np.abs(p - recent[-1][0]).max() * 1000 < STILL_MM and abs(a - recent[-1][1]) < STILL_DEG
        and abs(b - recent[-1][2]) < STILL_DEG for p, a, b in recent)
    near = 0.0
    if len(_TCP) == 2 and _EP["points"]:
        near = min(float(np.linalg.norm(v - p)) for v in _TCP.values() for p in _EP["points"])
    stage = 0.75 if _EP["open_now"] else 0.0
    return dict(still=still, near=float(near), stage=stage,
                done=bool(_EP["open_now"] and near >= FAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    print(f"DBG step={_EP['step']} angL={_EP['angL']:.1f} angR={_EP['angR']:.1f} gap={_EP['gap'] * 1000:.1f}mm "
          f"back={_EP['back'] * 1000:.1f}mm rise={_EP['rise'] * 1000:.1f}mm out={_EP['out_now']} "
          f"open={_EP['open_now']} tip=({_EP['tip'][0]:.4f},{_EP['tip'][1]:.4f},{_EP['tip'][2] - TOP:.4f}) near={t['near']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """0 for the rest of the run once the noodle has been forced out of its cup. Otherwise the best of: 0.25 the cup
    cleared off the noodle's end; 0.5 the noodle lifted out of the link; and, at the moment, 0.75 the noodle out and the
    pads 46 mm or more apart. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else max(_EP["best"], t["stage"])


def success(objects) -> bool:
    """The noodle out of the link, the pads 46 mm or more apart, both grippers 5 cm or more from the noodle and the
    arms, everything still, for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the guides say: the left gripper, pointing forward and tipped down 30 degrees, comes in over the front tyre with its
# fingers either side of the two brake arms, at the top of the arms, and squeezes them until the pads touch the rim; the
# cup moves off the noodle's end. The right gripper, coming from the bike's left side, takes the noodle's upright leg,
# draws it back 7 mm and lifts it 3 cm, out of the link, lets go and moves away. The left gripper then opens and backs
# away: the arms spring wide apart.
# VBRAKE_ONE=1 runs it the wrong way: nobody squeezes the arms; the right gripper takes the noodle, pulls it back 6 mm
# and lifts it.

WRONG = os.environ.get("VBRAKE_ONE") == "1"
M_FWD = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) @ grip_rotation(0.0)  # level, +x, fingers on y


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


SQ_TILT = math.radians(float(os.environ.get("VBRAKE_SQ_TILT", "30")))
SQ_ROT = _ry(SQ_TILT) @ M_FWD  # pointing forward (+x), tipped down, fingers closing along y
GR_ROT = _rz(math.pi / 2) @ _ry(math.radians(15.0)) @ M_FWD  # pointing +y, tipped down 15, fingers closing along x
SQ_DIR = SQ_ROT[:, 2]
GR_DIR = GR_ROT[:, 2]
OPEN_SQ, OPEN_GR = -0.047, -0.022
ARM_OUT_AT_SQZ = PIVOT_Y + (TOP_Y - PIVOT_Y) * (SQZ_C - PIVOT_C) / (TOP_C - PIVOT_C) + ARM_T / 2
SQ_CLOSE = -(ARM_OUT_AT_SQZ - (SQZ_C - PIVOT_C) * math.tan(math.radians(SQUEEZE_DEG))
             - float(os.environ.get("VBRAKE_SQ_PAST", "0.004")))
GR_CLOSE = 0.0
FINGERTIP = 0.005


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


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
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


SQ_AT = L(0.0, 0.0, SQZ_C)  # between the arms, at the top of the arms
GR_AT = L(0.0, LEG_Y, GRAB_C) + np.array([0.0, 0.002, 0.0])
SQ_AWAY = SQ_AT + np.array([-0.07, 0.06, 0.01])
GR_AWAY = GR_AT + np.array([0.0, -0.07, 0.05])
T_SQUEEZED = 3.0 + 1.5 + 0.5 + 1.0
T_LIFTED = T_SQUEEZED + 0.3 + 1.2 + 0.8 + 0.4 + 0.5 + 1.2 + 0.5
T_GR_GONE = T_LIFTED + 0.6 + 1.0


def _left() -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    if WRONG:
        pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["left"])
        return [Waypoint(20.0, tuple(float(v) for v in pos), 0.0, "stay put", rot)]
    pre = SQ_AT + np.array([0.0, 0.0, 0.02])
    high = SQ_AT + np.array([-0.03, 0.0, 0.05])
    out = _lead_in("left", high, SQ_ROT, OPEN_SQ, "over the front tyre, in front of the brake, fingers open",
                   seconds=2.4, steps=12)
    out += [_w(0.6, pre, OPEN_SQ, "above the arms' tops", SQ_ROT),
            _w(1.5, SQ_AT, OPEN_SQ, "down, fingers either side of the two brake arms", SQ_ROT),
            _w(0.5, SQ_AT, OPEN_SQ, "settle", SQ_ROT),
            _w(1.0, SQ_AT, SQ_CLOSE, "squeeze the arms together until the pads touch the rim", SQ_ROT),
            _w(T_GR_GONE - T_SQUEEZED + 0.3, SQ_AT, SQ_CLOSE, "hold them squeezed", SQ_ROT),
            _w(0.6, SQ_AT, OPEN_SQ, "let them go: they spring apart", SQ_ROT),
            _w(1.2, SQ_AT + np.array([-0.05, 0.0, 0.005]), OPEN_SQ, "back away", SQ_ROT),
            _w(1.5, SQ_AWAY, OPEN_SQ, "move away", SQ_ROT),
            _w(5.0, SQ_AWAY, OPEN_SQ, "stay clear", SQ_ROT)]
    return out


def _right() -> list:
    pre = GR_AT - 0.07 * GR_DIR
    back = 0.006 if WRONG else 0.007
    out = _lead_in("right", pre, GR_ROT, OPEN_GR, "beside the noodle, fingers open")
    out += [_w(T_SQUEEZED + 0.3 - 3.0, pre, OPEN_GR, "wait for the squeeze" if not WRONG else "wait", GR_ROT),
            _w(1.2, GR_AT, OPEN_GR, "fingers either side of the noodle's upright leg", GR_ROT),
            _w(0.8, GR_AT, GR_CLOSE, "take the noodle", GR_ROT),
            _w(0.4, GR_AT, GR_CLOSE, "settle", GR_ROT),
            _w(0.5, GR_AT + np.array([0.0, -back, 0.0]), GR_CLOSE, "draw it back off the seat", GR_ROT),
            _w(1.2, GR_AT + np.array([0.0, -back, 0.03]), GR_CLOSE, "lift it out of the link", GR_ROT),
            _w(0.5, GR_AT + np.array([0.0, -back, 0.03]), GR_CLOSE, "settle", GR_ROT),
            _w(0.6, GR_AT + np.array([0.0, -back, 0.03]), OPEN_GR, "let go", GR_ROT),
            _w(1.0, GR_AT + np.array([0.0, -0.07, 0.05]), OPEN_GR, "back away", GR_ROT),
            _w(1.5, GR_AWAY, OPEN_GR, "move away", GR_ROT),
            _w(8.0, GR_AWAY, OPEN_GR, "stay clear", GR_ROT)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
