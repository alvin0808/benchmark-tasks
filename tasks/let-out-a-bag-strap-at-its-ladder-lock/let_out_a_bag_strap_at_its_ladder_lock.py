"""On the station table: an empty school daypack lies face down, its back panel and shoulder straps up, its top toward
the robot's left and its bottom toward the robot's right. Each shoulder strap is a padded strap sewn to the top of the
back panel; at its lower end a short webbing tab is sewn round the end bar of a black 25 mm ladder-lock buckle. A
25 mm nylon webbing sewn to the panel's bottom corner runs up the panel to that buckle, goes through it and over its
centre bar, and comes back out under the buckle's lift tab; its loose end (the tail, its tip folded and stitched)
lies back down along the webbing. The near strap (nearer the robot) is the one to adjust; its tail sticks out 15 cm
past the buckle.

Task (instruction): lengthen the near shoulder strap until its loose end sticks out only 7 cm past the buckle, and
leave the buckle holding. The instruction does not say how. A ladder lock lets webbing through in the lengthening
direction only while its lift tab is raised: "Created to hold under tension, then release by lifting the tab"
(Country Brook Design, 1 1/2 inch ladderlock buckles); pulled without the tab lifted, the strap does not give and
the bag comes along. So one hand holds the buckle tipped up off the webbing while the other draws the webbing
through it from the bag side, and then the buckle is laid flat again. Let go, the buckle springs back flat (the
stiff webbing bent through its bars pushes it down) and grips again.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- backpack: loose, 0.45 kg, a 30 x 44 x 8.5 cm body (x 0.11 to 0.41, y -0.255 to 0.185) lying face down, its back
  panel (grey mesh) 8.5 cm above the table; two red padded shoulder straps (6 cm wide, 12 mm thick) lie on the
  panel along y from its top edge down to y 0.07, at x 0.20 (near) and x 0.32 (far).
- near ladder lock: 35 x 26 mm, 6.5 mm tall, black acetal, 5 g; its centre bar (a round bar across the webbing) at
  x 0.20, y 0.030, 5.4 mm above the panel; the tab end toward -y. It tips about its centre bar.
- near webbing: 25 mm wide, 1.6 mm thick, 18 g/m (rigid 15 mm links on springy hinges across its width); sewn
  over its last 15 mm at y -0.225 to -0.210 on the panel, 24 cm from there up to the buckle's centre bar along
  x 0.20, then back out of the buckle, the tail lying on top of it from y 0.017 (the tab's edge) to y -0.133.

Modelled as a rule (README): the webbing's wrap over the centre bar is not simulated. Under the buckle both runs go
on, out of sight under the padded strap, to two hidden carriers on slides along y; every control step the tail's
carrier is sprung to go in as far as the strap's has come out. The buckle's grip is the strap slide's friction, set
every control step from the buckle's angle: 300 N (and a spring holding the slide) within 30 degrees of the webbing,
1.5 N past 50 degrees, falling in between. A spring returns the buckle flat.

Score: 0 for the rest of the run once the backpack has fallen off the table (its middle 3 cm or more below where it
lay); sliding it on the table costs nothing. Otherwise the best stage reached: 0.25 the strap let out by 2 cm or more (webbing drawn through the buckle); 0.5 let out to the
target, its loose end 6 to 8 cm past the buckle; 0.75 at the target with the buckle laid back flat (within 15
degrees of the webbing), so that it grips. Full score (1.0): at the target, the buckle flat, both grippers 5 cm or
more from the buckle and from the webbing, everything still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step, plate  # noqa: E402
from strapkit import Loop, Part, Strap, ids, set_drive, set_friction, set_target  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and the off-the-table rule latches 0, so the score is read at the end.
SCORE_OVER = "end"

# -- colours --------------------------------------------------------------------------------------------------------
RED = (0.66, 0.12, 0.11)  # pack fabric (600D polyester)
RED_DARK = (0.45, 0.07, 0.07)
MESH = (0.43, 0.44, 0.46)  # back panel air mesh
BLACK = (0.035, 0.035, 0.04)  # acetal buckle
WEB = (0.09, 0.09, 0.10)  # black nylon webbing
STITCH = (0.75, 0.75, 0.72)
ZIP = (0.08, 0.08, 0.09)

# -- layout (world frame) -------------------------------------------------------------------------------------------
PACK_X0, PACK_X1, PACK_Y0, PACK_Y1, PACK_H = 0.11, 0.41, -0.255, 0.185, 0.085
PACK_KG = 0.45
PACK_O = ((PACK_X0 + PACK_X1) / 2, (PACK_Y0 + PACK_Y1) / 2, TOP)  # the pack's origin: its footprint's middle
PANEL = TOP + PACK_H  # the back panel's surface
XS, XF = 0.20, 0.32  # the near and far straps' webbing centre lines
YB = 0.030  # the near buckle's centre bar
PAD_Y0 = YB + 0.040  # the padded strap's lower end (foam)
PAD_W, PAD_T = 0.060, 0.012

WEB_W, WEB_T = 0.025, 0.0016  # 25 mm (1 inch) webbing, 1.6 mm thick
WEB_KG_PER_M = 0.018
LINK = 0.015  # webbing link length
Z_A = PANEL + 0.0002 + WEB_T / 2  # the strap's webbing (from the corner) lies on the panel
Z_T = Z_A + WEB_T + 0.0002  # the tail lies on top of it
BAR_R = 0.0016  # the buckle's round centre bar
Z_K = Z_T + WEB_T / 2 + 0.0002 + BAR_R  # the centre bar's axis: the buckle tips about it
Y_SEWN = -0.225  # the webbing's end, sewn to the panel over its last link (to Y_SEWN + LINK)
N_A = 27  # links from the hidden carrier in the padded strap to the sewn end
H_A = N_A * LINK - (YB - Y_SEWN)  # how far into the padded strap the strap's webbing reaches
N_T = 12  # tail links
TAIL0 = 0.150  # the tail's length past the buckle's tab edge at the start
TAB_EDGE = YB - 0.013  # the buckle's tab edge (y)
H_T = N_T * LINK - TAIL0 - (YB - TAB_EDGE)  # how far into the padded strap the tail's webbing reaches
SLIDE_MAX = TAIL0 - LINK  # the folded, stitched tip stops at the buckle: the most the strap can be let out
SLIDE_TIGHT = 0.005  # the most it can be tightened from the start (the webbing to the corner is then taut)
TARGET, TOL = 0.080, 0.010  # let out 8 cm +- 1 cm: the loose end 7 cm +- 1 cm past the buckle
PARTIAL_LET = 0.020

# The ladder lock's grip on the webbing (see _friction): locked flat, released once tipped up.
LOCK_N = 300.0  # the slide's friction with the buckle gripping: more than the arms can pull
FREE_N = 1.5  # the slide's friction with the buckle tipped up past its release angle
LOCK_K, LOCK_D = 2000.0, 15.0  # gripping, a spring also holds the slide where it was gripped (N/m, N s/m):
# PhysX's joint friction alone lets a slide creep slowly
TAIL_K, TAIL_D = 2000.0, 10.0  # the tail's slide is sprung to follow the strap's, in as far as it comes out
ANGLE_GRIP, ANGLE_FREE = 30.0, 50.0  # degrees off the webbing: grips below, slides freely above
FLAT = 15.0  # laid back flat: within this of the webbing
BUCKLE_K, BUCKLE_D = 0.030, 0.0015  # its spring back to flat (N m/rad) and damping
BEND_K, BEND_D, BEND_ARM = 0.006, 0.0004, 2.0e-6  # webbing hinges: N m/rad, N m s/rad, kg m^2


def _box(x0, x1, y0, y1, z0, z1, color, origin, collide=True) -> BoxShape:
    """A box from world corners, in the frame of a part whose origin is ``origin``."""
    return BoxShape(((x0 + x1) / 2 - origin[0], (y0 + y1) / 2 - origin[1], (z0 + z1) / 2 - origin[2]),
                    (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


# -- the backpack (the articulation's loose root) -----------------------------------------------------------------
def _strap_dress(x: float, o, collide: bool) -> list:
    """A padded shoulder strap lying on the panel along y from its lower end to the top edge, at webbing line x:
    foam with a tapered lower end, black binding along its edges, a stitched bar tack at its end. Its lower end's
    webbing tab and buckle are added separately."""
    out = [_box(x - PAD_W / 2, x + PAD_W / 2, PAD_Y0 + 0.020, PACK_Y1, PANEL, PANEL + PAD_T, RED, o, collide),
           _box(x - PAD_W / 2 + 0.008, x + PAD_W / 2 - 0.008, PAD_Y0, PAD_Y0 + 0.020, PANEL, PANEL + PAD_T - 0.002,
                RED, o, collide)]
    for s in (-1.0, 1.0):
        out.append(_box(x + s * PAD_W / 2 - 0.002, x + s * PAD_W / 2 + 0.0005, PAD_Y0 + 0.020, PACK_Y1, PANEL,
                        PANEL + PAD_T + 0.0005, BLACK, o, False))
    out.append(_box(x - 0.016, x + 0.016, PAD_Y0 + 0.004, PAD_Y0 + 0.010, PANEL + PAD_T - 0.002,
                    PANEL + PAD_T - 0.0012, STITCH, o, False))
    return out


def _pack_shapes() -> list:
    o = (PACK_O[0], PACK_O[1], 0.0)  # heights in this function are above the table (the pack's origin)
    out = [_box(PACK_X0, PACK_X1, PACK_Y0, PACK_Y1, 0.004, PACK_H - 0.004, RED, o),
           _box(PACK_X0 + 0.006, PACK_X1 - 0.006, PACK_Y0 + 0.006, PACK_Y1 - 0.006, 0.0, 0.004, RED_DARK, o),
           _box(PACK_X0 + 0.004, PACK_X1 - 0.004, PACK_Y0 + 0.004, PACK_Y1 - 0.004, PACK_H - 0.004, PACK_H, MESH,
                o)]
    # the zip round the pack's sides, halfway up (shown only)
    z0, z1 = 0.040, 0.046
    out += [_box(PACK_X0 - 0.0006, PACK_X0, PACK_Y0, PACK_Y1, z0, z1, ZIP, o, False),
            _box(PACK_X1, PACK_X1 + 0.0006, PACK_Y0, PACK_Y1, z0, z1, ZIP, o, False),
            _box(PACK_X0, PACK_X1, PACK_Y0 - 0.0006, PACK_Y0, z0, z1, ZIP, o, False),
            _box(PACK_X0, PACK_X1, PACK_Y1, PACK_Y1 + 0.0006, z0, z1, ZIP, o, False)]
    # the haul loop at the top, between the straps
    xm = (XS + XF) / 2
    out.append(_box(xm - 0.0125, xm + 0.0125, PACK_Y1 - 0.004, PACK_Y1 + 0.022, PACK_H - 0.002, PACK_H + 0.003,
                    WEB, o, False))
    # mesh bottle pockets on the two sides, and the zip's pullers at the top (shown only)
    for x0, x1 in ((PACK_X0 - 0.012, PACK_X0), (PACK_X1, PACK_X1 + 0.012)):
        out.append(_box(x0, x1, PACK_Y0 + 0.02, PACK_Y0 + 0.17, 0.008, 0.062, MESH, o, False))
    for x in (PACK_X0 + 0.02, PACK_X1 - 0.02):
        out.append(_box(x - 0.004, x + 0.004, PACK_Y1, PACK_Y1 + 0.018, 0.040, 0.046, ZIP, o, False))
    # the sternum strap across the two padded straps, its side-release buckle in the middle (shown only)
    ys, zs = 0.150, PACK_H + PAD_T
    out += [_box(XS + 0.02, XF - 0.02, ys - 0.0095, ys + 0.0095, zs, zs + 0.0014, WEB, o, False),
            _box((XS + XF) / 2 - 0.022, (XS + XF) / 2 + 0.022, ys - 0.012, ys + 0.012, zs + 0.0014, zs + 0.0064,
                 BLACK, o, False)]
    # reinforcement patches at the two bottom corners, where the webbings are sewn (shown only)
    for x in (XS, XF):
        out.append(_box(x - 0.020, x + 0.020, PACK_Y0 + 0.010, Y_SEWN + LINK + 0.006, PACK_H, PACK_H + 0.0003,
                        RED_DARK, o, False))
    # the far strap, as it lies (its buckle, webbing and tail are part of the pack: not used in the task)
    out += _strap_dress(XF, PACK_O, collide=True)
    zb = Z_K - TOP
    out += [_box(XF - 0.0125, XF + 0.0125, Y_SEWN, YB, PACK_H + 0.0002, PACK_H + 0.0018, WEB, o, False),
            _box(XF - 0.0125, XF + 0.0125, TAB_EDGE - 0.11, YB, PACK_H + 0.0020, PACK_H + 0.0036, WEB, o, False),
            _box(XF - 0.0175, XF + 0.0175, YB - 0.013, YB + 0.013, zb - 0.004, zb + 0.0012, BLACK, o, False),
            _box(XF - 0.0125, XF + 0.0125, YB + 0.010, PAD_Y0 + 0.005, zb + 0.0004, zb + 0.0020, WEB, o, False)]
    return out


# -- the near strap's parts -----------------------------------------------------------------------------------------
def _pad_shapes() -> list:
    """The near padded strap and its webbing tab, from the buckle's end bar (sewn round it) to under the padded
    strap's end (the tab is shown only). The webbing that goes on into the padded strap passes through it unseen."""
    o = (XS, YB, PANEL)
    zt = Z_K - PANEL
    return _strap_dress(XS, o, collide=True) + [
            _box(XS - 0.0125, XS + 0.0125, YB + 0.009, PAD_Y0 + 0.004, PANEL + zt + 0.0005, PANEL + zt + 0.0021, WEB,
                 o, False),
            _box(XS - 0.013, XS + 0.013, YB + 0.030, YB + 0.034, PANEL + zt + 0.0021, PANEL + zt + 0.0025, STITCH,
                 o, False)]


def _keeper_shapes() -> list:
    """The buckle's round centre bar, which the webbing wraps (its axis is the buckle's tipping axis, so the bar
    stays where it is), and the webbing's wrap round it (shown)."""
    return [CylShape((0.0, 0.0, 0.0), BAR_R, 0.029, BLACK, axis="X"),
            CylShape((0.0, 0.0, -0.0008), BAR_R + WEB_T + 0.0002, WEB_W, WEB, collide=False, axis="X")]


def _buckle_shapes() -> list:
    """The ladder lock in the frame of its centre bar's axis: two side rails, the end bar the strap's tab is sewn
    round (+y), the bar at the tab end (-y) and the lift tab curving up from it."""
    zb = PANEL + 0.0004 - Z_K  # the rails' bottom (on the panel)
    out = []
    for s in (-1.0, 1.0):
        out.append(BoxShape((s * 0.016, 0.0, (zb + 0.0012) / 2), (0.003, 0.026, 0.0012 - zb), BLACK))
    out.append(CylShape((0.0, 0.0105, 0.0), 0.0018, 0.032, BLACK, axis="X"))
    out.append(BoxShape((0.0, -0.0105, -0.0002), (0.032, 0.003, 0.0028), BLACK))
    out.append(plate((0.0, -0.0115, 0.0010), (0.0, -0.0160, 0.0042), 0.035, 0.0018, (0.0, -0.6, 1.0), BLACK))
    return out


def _link_shapes(tail_tip: bool = False, sewn: bool = False) -> list:
    out = [BoxShape((0.0, 0.0, 0.0), (WEB_W, LINK + 0.0004, WEB_T), WEB)]
    if tail_tip:  # the tip folded under and stitched (shown)
        out.append(BoxShape((0.0, -0.002, -0.0001), (WEB_W - 0.001, LINK - 0.006, WEB_T + 0.0004), WEB,
                            collide=False))
        out.append(BoxShape((0.0, -0.004, WEB_T / 2 + 0.0002), (WEB_W - 0.004, 0.0015, 0.0003), STITCH,
                            collide=False))
    if sewn:  # a box-and-X stitch through the webbing into the panel (shown)
        for d in (-0.005, 0.005):
            out.append(BoxShape((0.0, d, WEB_T / 2 + 0.0002), (WEB_W - 0.004, 0.0008, 0.0003), STITCH,
                                collide=False))
        for s in (-1.0, 1.0):
            out.append(BoxShape((s * 0.0105, 0.0, WEB_T / 2 + 0.0002), (0.0008, 0.010, 0.0003), STITCH,
                                collide=False))
    return out


def _w3(p) -> tuple:
    return tuple(float(v) for v in p)


LINK_KG = WEB_KG_PER_M * LINK
LINK_I = (LINK_KG * (LINK ** 2 + WEB_T ** 2) / 12, LINK_KG * (WEB_W ** 2 + WEB_T ** 2) / 12,
          LINK_KG * (WEB_W ** 2 + LINK ** 2) / 12)


def _strap_parts() -> list:
    parts = [Part("backpack", tuple(_pack_shapes()), PACK_KG, _w3(PACK_O), material=(0.5, 0.45)),
             Part("pad", tuple(_pad_shapes()), 0.005, _w3((XS, YB, PANEL)), parent="backpack", joint="fixed",
                  inertia=(1e-6, 1e-6, 1e-6)),
             Part("keeper", tuple(_keeper_shapes()), 0.002, _w3((XS, YB, Z_K)), parent="backpack", joint="fixed",
                  material=(0.3, 0.25)),
             Part("buckle", tuple(_buckle_shapes()), 0.005, _w3((XS, YB, Z_K)), parent="backpack", joint="revolute",
                  axis="X", anchor=_w3((XS, YB, Z_K)), limits=(-95.0, 2.0), stiffness=BUCKLE_K, damping=BUCKLE_D,
                  armature=1e-6, material=(0.35, 0.3)),
             # the webbing's U-turn over the centre bar, hidden in the padded strap: two carriers on slides along y;
             # the tail's is sprung to follow the strap's every control step, in as far as the strap's comes out
             Part("carrier_a", (), 0.002, _w3((XS, YB + H_A, Z_A)), parent="backpack", joint="prismatic", axis="Y",
                  limits=(-SLIDE_MAX, SLIDE_TIGHT), friction=LOCK_N, damping=0.05, armature=0.005,
                  inertia=(1e-7, 1e-7, 1e-7)),
             Part("carrier_t", (), 0.002, _w3((XS, YB + H_T, Z_T)), parent="backpack", joint="prismatic", axis="Y",
                  limits=(-SLIDE_TIGHT, SLIDE_MAX), stiffness=TAIL_K, damping=TAIL_D, armature=0.005,
                  inertia=(1e-7, 1e-7, 1e-7))]
    parent = "carrier_a"
    for i in range(N_A):
        y = YB + H_A - (i + 0.5) * LINK
        name = f"a{i:02d}"
        parts.append(Part(name, tuple(_link_shapes(sewn=(i == N_A - 1))), LINK_KG, _w3((XS, y, Z_A)), parent=parent,
                          joint="revolute", axis="X", anchor=_w3((XS, y + LINK / 2, Z_A)), limits=(-120.0, 120.0),
                          stiffness=BEND_K, damping=BEND_D, armature=BEND_ARM, inertia=LINK_I, material=(0.6, 0.5),
                          linear_damping=0.2, angular_damping=0.5))
        parent = name
    parent = "carrier_t"
    for i in range(N_T):
        y = YB + H_T - (i + 0.5) * LINK
        name = f"t{i:02d}"
        parts.append(Part(name, tuple(_link_shapes(tail_tip=(i == N_T - 1))), LINK_KG, _w3((XS, y, Z_T)),
                          parent=parent, joint="revolute", axis="X", anchor=_w3((XS, y + LINK / 2, Z_T)),
                          limits=(-120.0, 120.0), stiffness=BEND_K, damping=BEND_D, armature=BEND_ARM,
                          inertia=LINK_I, material=(0.6, 0.5), linear_damping=0.2, angular_damping=0.5))
        parent = name
    return parts


A_LINKS = [f"a{i:02d}" for i in range(N_A)]
T_LINKS = [f"t{i:02d}" for i in range(N_T)]
_FILTERS = ([("pad", n) for n in A_LINKS + T_LINKS] + [("buckle", n) for n in A_LINKS + T_LINKS]
            + [("keeper", n) for n in T_LINKS]
            + [("buckle", "keeper"), ("buckle", "pad"), ("carrier_a", "carrier_t")])
STRAP = Strap.build("backpack", parts=_strap_parts(), loops=[Loop(A_LINKS[-1], "backpack")], filters=_FILTERS,
                    color=RED, floating=True, self_collision=True, solver_iterations=48, velocity_iterations=4,
                    contact_offset=0.0015)
OBJECTS = [STRAP]

# -- tracking and scoring ---------------------------------------------------------------------------------------------
FELL = 0.030  # the backpack's middle this far below where it lay: pulled off the table
CLEAR = 0.050  # both grippers this far from the buckle and the webbing: let go
STILL_STEPS, STILL_MM, STILL_DEG = 20, 1.5, 2.0
_DEBUG = os.environ.get("STRAP_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("STRAP_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", let=0.0, tilt=0.0, fric=None, pack0=None, hold=None,
               moved=0.0, near=1.0, hist=[], recent=[], arms=[], max_let=0.0, max_tilt=0.0)
    _IDS.pop("arm", None)


_reset()


def _friction(tilt_deg: float) -> float:
    """The slide's friction (N) with the buckle tipped ``tilt_deg`` off the webbing: LOCK_N at ANGLE_GRIP and flatter
    (the buckle grips), FREE_N past ANGLE_FREE, falling geometrically in between (the webbing's wrap round the bars
    opens up as the tab rises)."""
    if tilt_deg <= ANGLE_GRIP:
        return LOCK_N
    if tilt_deg >= ANGLE_FREE:
        return FREE_N
    s = (tilt_deg - ANGLE_GRIP) / (ANGLE_FREE - ANGLE_GRIP)
    return float(LOCK_N * (FREE_N / LOCK_N) ** s)


def _grippers(base) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        out.append(np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float))
    return out


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("STRAP_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """The buckle's tilt and how far the webbing has been drawn through it (the slide), the slide's friction for
    that tilt (written to the simulation), how far the backpack has moved, where the grippers are."""
    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    asset = base.scene["backpack"]
    idx = ids(asset, STRAP)
    origin = as_numpy(base.scene.env_origins[0])
    q = as_numpy(asset.data.joint_pos)[0]
    bp = as_numpy(asset.data.body_pos_w)[0] - origin
    tilt = -math.degrees(float(q[idx["joint"]["buckle"]]))
    let = -float(q[idx["joint"]["carrier_a"]])
    fric = _friction(tilt)
    if _EP["fric"] is None or abs(fric - _EP["fric"]) > 0.01 * max(fric, 1.0):
        set_friction(env, STRAP, "carrier_a", fric)
        _EP["fric"] = fric
    # gripping (buckle at ANGLE_GRIP or flatter): besides its friction, a spring holds the slide where it was gripped
    q_a = float(q[idx["joint"]["carrier_a"]])
    if tilt <= ANGLE_GRIP:
        if _EP["hold"] is None:
            set_drive(env, STRAP, "carrier_a", LOCK_K, LOCK_D, q_a)
            _EP["hold"] = q_a
    elif _EP["hold"] is not None:
        set_drive(env, STRAP, "carrier_a", 0.0, 0.05, 0.0)
        _EP["hold"] = None
    set_target(env, STRAP, "carrier_t", -q_a)  # the tail goes in as far as the strap's webbing has come out
    pack = bp[idx["body"]["backpack"]]
    if _EP["pack0"] is None:
        _EP["pack0"] = pack.copy()
    moved = float(np.linalg.norm(pack - _EP["pack0"]))
    if float(_EP["pack0"][2] - pack[2]) >= FELL and not _EP["failed"]:
        _EP["failed"] = f"the backpack fell off the table at step {_EP['tick']}"
    arms = _grippers(base)
    pts = [bp[idx["body"]["buckle"]]] + [bp[idx["body"][n]] for n in A_LINKS + T_LINKS
                                          if bp[idx["body"][n]][1] < PAD_Y0]
    near = min(float(np.linalg.norm(t - p)) for t in arms for p in pts)
    _EP["links"] = {n: bp[idx["body"][n]] for n in ("a00", "a10", A_LINKS[-1], "t00", "t05", T_LINKS[-1])}
    _EP["q_t"] = float(q[idx["joint"]["carrier_t"]])
    _EP.update(tilt=tilt, let=let, moved=moved, near=near, arms=arms, pack=pack,
               max_let=max(_EP["max_let"], let), max_tilt=max(_EP["max_tilt"], tilt))
    _EP["hist"].append(np.array([let * 1000, tilt / STILL_DEG * STILL_MM, *(pack * 1000)]))
    del _EP["hist"][:-STILL_STEPS]
    if not (np.isfinite(let) and np.isfinite(tilt)):  # a broken strap state never earns a stage
        return
    at = abs(let - TARGET) <= TOL
    flat = tilt <= FLAT
    stage = 0.75 if at and flat else 0.5 if at else 0.25 if let >= PARTIAL_LET else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _still() -> bool:
    h = _EP["hist"]
    return len(h) == STILL_STEPS and max(float(np.abs(v - h[-1]).max()) for v in h) < STILL_MM


def _done() -> bool:
    return bool(not _EP["failed"] and abs(_EP["let"] - TARGET) <= TOL and _EP["tilt"] <= FLAT and _EP["near"] >= CLEAR
                and _still())


def _debug() -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    print(f"DBG step={_EP['step']} let={_EP['let'] * 100:.2f}cm tail={(TAIL0 - _EP['let']) * 100:.1f}cm "
          f"tilt={_EP['tilt']:.1f}deg fric={_EP['fric'] or 0.0:.1f}N moved={_EP['moved'] * 100:.2f}cm "
          f"near={_EP['near']:.3f} still={_still()} best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if _EP.get("pack") is not None:
        pk = _EP["pack"]
        print(f"DBG   pack=({pk[0]:.3f},{pk[1]:.3f},{(pk[2] - TOP) * 1000:+.1f}mm) "
              f"tail_slide={_EP.get('q_t', 0.0) * 100:.2f}cm", flush=True)
    if _EP.get("links"):
        print("DBG   " + " ".join(f"{k}=({v[0]:.3f},{v[1]:.3f},{(v[2] - PANEL) * 1000:+.1f}mm)"
                                  for k, v in _EP["links"].items()), flush=True)
    if _EP["arms"]:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - PANEL:+.3f})"
                                    for k, v in zip(("left", "right"), _EP["arms"])), flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the backpack has fallen off the table): 0.25 let out 2 cm
    or more; 0.5 let out to the target (loose end 6 to 8 cm past the buckle); 0.75 that with the buckle laid back
    flat. Full score (1.0) is success()."""
    if _DEBUG and _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        _debug()
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """At the target, the buckle flat, both grippers 5 cm or more from the buckle and the webbing, all still 1 s."""
    return _done()


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left gripper takes the buckle by its two side rails at the tab end and tips it up about its centre bar to 60
# degrees, turning with it; the right gripper pinches the webbing by its edges 6 cm from the sewn corner and draws it
# toward the corner and up, so 8 cm of webbing runs through the buckle (the tail shortens by as much); the left lays
# the buckle flat again; both let go and move away. STRAP_ONE=1 runs it the wrong way: the left arm stays put and
# the right pulls the webbing with the buckle lying flat.

WRONG = os.environ.get("STRAP_ONE") == "1"
TILT_TO = math.radians(float(os.environ.get("STRAP_TILT", "68")))
PULL_TO_Y = float(os.environ.get("STRAP_PULL_Y", "-0.240"))
PULL_TO_Z = float(os.environ.get("STRAP_PULL_Z", "0.030"))
GRIP_L = grip_rotation(-math.pi / 2)  # left: straight down, fingers closing along x
GRIP_R = grip_rotation(math.pi / 2)  # right: straight down, fingers closing along x
OPEN = -0.040
RAIL_PINCH = -(0.0175 - 0.003)
WEB_PINCH = -(WEB_W / 2 - 0.003)
TIP_PAST_TCP = 0.013
B_GRIP_Y = YB - 0.014  # the left gripper's centre over the buckle's tab end
B_TCP = np.array([XS, B_GRIP_Y, PANEL + TIP_PAST_TCP + 0.0008])
W_GRIP_Y = -0.155  # the right gripper's centre on the webbing
W_TCP = np.array([XS, W_GRIP_Y, PANEL + TIP_PAST_TCP + 0.0006])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _tipped(a: float):
    """The left TCP's pose with the buckle tipped ``a`` (rad, tab end up): turned with it about its centre bar."""
    r = _rx(-a)
    axis = np.array([XS, YB, Z_K])
    return axis + r @ (B_TCP - axis), r @ GRIP_L


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


T_HELD = 3.0 + 1.4 + 0.4 + 0.8  # both grippers have closed
T_TIPPED = T_HELD + 2.4 + 0.4
T_PULLED = T_TIPPED + 3.0 + 0.5
T_FLAT = T_PULLED + 1.6 + 0.4


def _left() -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    if WRONG:
        pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["left"])
        return [Waypoint(T_FLAT + 6.0, tuple(float(v) for v in pos), OPEN, "stay put", rot)]
    out = _lead_in("left", B_TCP + [0.0, 0.0, 0.05], GRIP_L, OPEN, "above the buckle's tab end")
    out += [_w(1.4, B_TCP, OPEN, "fingers either side of the buckle's rails", GRIP_L),
            _w(0.4, B_TCP, OPEN, "settle", GRIP_L),
            _w(0.8, B_TCP, RAIL_PINCH, "take the buckle by its rails", GRIP_L)]
    steps = 8
    for k in range(1, steps + 1):
        p, r = _tipped(TILT_TO * k / steps)
        out.append(_w(2.4 / steps, p, RAIL_PINCH, "tip the buckle up about its centre bar", r))
    p_up, r_up = _tipped(TILT_TO)
    out += [_w(0.4, p_up, RAIL_PINCH, "settle", r_up),
            _w(T_PULLED - T_TIPPED, p_up, RAIL_PINCH, "hold it tipped up while the webbing is drawn through", r_up)]
    for k in range(steps - 1, -1, -1):
        p, r = _tipped(TILT_TO * k / steps)
        out.append(_w(1.6 / steps, p, RAIL_PINCH, "lay the buckle flat again", r))
    out += [_w(0.4, B_TCP, RAIL_PINCH, "settle", GRIP_L),
            _w(0.8, B_TCP, OPEN, "let go", GRIP_L),
            _w(1.0, B_TCP + [0.0, 0.03, 0.08], OPEN, "rise", GRIP_L),
            _w(1.2, np.array([0.17, 0.20, PANEL + 0.10]), OPEN, "move away", GRIP_L),
            _w(4.0, np.array([0.17, 0.20, PANEL + 0.10]), OPEN, "stay clear", GRIP_L)]
    return out


def _right() -> list:
    out = _lead_in("right", W_TCP + [0.0, 0.0, 0.05], GRIP_R, OPEN, "above the webbing near the corner")
    pulled = np.array([XS, PULL_TO_Y, PANEL + PULL_TO_Z])
    out += [_w(1.4, W_TCP, OPEN, "fingers either side of the webbing", GRIP_R),
            _w(0.4, W_TCP, OPEN, "settle", GRIP_R),
            _w(0.8, W_TCP, WEB_PINCH, "pinch the webbing by its edges", GRIP_R),
            _w(T_TIPPED - T_HELD, W_TCP + [0.0, 0.0, 0.002], WEB_PINCH, "wait for the buckle to be tipped up",
               GRIP_R),
            _w(3.0, pulled, WEB_PINCH, "draw the webbing through the buckle toward the corner", GRIP_R),
            _w(0.5, pulled, WEB_PINCH, "settle", GRIP_R),
            _w(T_FLAT - T_PULLED, pulled, WEB_PINCH, "hold while the buckle is laid flat", GRIP_R),
            _w(0.8, pulled, OPEN, "let go", GRIP_R),
            _w(1.0, pulled + [0.0, 0.0, 0.06], OPEN, "rise", GRIP_R),
            _w(1.2, np.array([0.17, -0.20, PANEL + 0.12]), OPEN, "move away", GRIP_R),
            _w(4.0, np.array([0.17, -0.20, PANEL + 0.12]), OPEN, "stay clear", GRIP_R)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
