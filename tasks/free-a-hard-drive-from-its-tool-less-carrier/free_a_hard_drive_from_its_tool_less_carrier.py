"""On the station table: a 2.5-inch hard drive held in a tool-less drive carrier, and an anti-static mat to the right
of the carrier. The carrier's base is fixed to the table; the drive lies flat on it, label up, its front end toward the
robot. Along each side of the drive runs one of the carrier's blue plastic rails; each rail carries two pins that sit in
the drive's two side screw holes, and each rail is sprung inward, so that it springs back into the holes as soon as it
is let go. A tab stands up at the front end of each rail, in front of the drive.

Task (instruction): take the hard drive out of its carrier and put it on the anti-static mat. The instruction does not
say how. The procedure the task checks is the computer maker's (Dell OptiPlex 5060 Tower Service Manual, "Removing the
2.5-inch drive from the drive bracket"): "Pull one side of the drive bracket to disengage the pins", then lift the
drive out. With one rail pulled out and held, the drive is lifted on that side, so that it pivots on the other rail's
pins, slid sideways off those pins and lifted clear. One hand cannot hold a rail out and lift the drive at once: let go,
the rail springs back into the holes. Pulled straight up with both rails in place, the drive is held by the pins in
shear; pulled hard, a rail breaks. A drive must not be dropped or knocked: it is set down gently.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- carrier base: fixed, x 0.184 to 0.316, y 0.009 to 0.071 (a slab 1.2 cm tall the drive lies on).
- hard drive: a 2.5-inch drive, 100.2 x 69.85 x 9.5 mm (SFF-8201), 0.115 kg, middle at x 0.26, y 0.04, lying on the
  base; two side holes on each side, 3.0 mm above its bottom and 76.6 mm apart (x 0.2239 and 0.3005), 4 mm square here.
- rails: 0.02 kg each, along the drive's sides, each hinged along x 7 mm below the drive's bottom and sprung inward
  (0.2 N m/rad, preloaded 1 degree onto its stop); its two pins (2.5 mm, 3 mm long) stand 2.7 mm into the holes; it can
  open 40 degrees; its joint breaks under 10 N or 0.5 N m. The tab at its front end stands 8 cm above the table, in
  front of the drive. Opening a rail about 17 degrees draws its pins out of the holes.
- anti-static mat: fixed, 20 x 15 cm, 2 mm thick, middle at x 0.26, y -0.165.

Score: 0 for the rest of the run once the drive has been dropped or knocked (it moved down faster than 0.5 m/s, or its
speed changed by more than 0.6 m/s within one control step) or a rail has broken. Otherwise the best stage reached: 0.25
one side of the drive lifted off its rail's pins (that side's two holes off the pins and risen 4 mm or more); 0.5 the
drive out of the carrier (all four holes off the pins and its lowest point 1 cm or more above the carrier's base, or
its middle 6 cm or more from where it started); 0.75 lying flat on the mat (label up, tilted 10 degrees or less, its
middle over the mat at least 5 mm in from its edges, its lowest point within 4 mm of the mat's top), at rest, not held
(no gripper's centre within 7 cm of the drive's middle and keeping the same offset to it). Full score: that, with both
rails back home (within 2 degrees of shut) and both grippers 6 cm or more from the drive and from the rails' tabs,
still for 1 s.
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
from bodykit import rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame; heights above the table) --------------------------------------------------------------------
D_L, D_W, D_H, D_KG = 0.1002, 0.06985, 0.0095, 0.115  # 2.5-inch drive (SFF-8201): length (x), width (y), height
X_C, Y_C = 0.26, 0.04  # the drive's middle at the start
X_F = X_C - D_L / 2  # its front end (toward the robot)
HOLE_X = (-D_L / 2 + 0.014, -D_L / 2 + 0.0906)  # side holes along the drive (its frame): 76.6 mm apart
HOLE_Z, HOLE, HOLE_D = 0.003, 0.004, 0.004  # hole centre above the drive's bottom; square side; depth
Z_F = 0.012  # the carrier base's top: the drive lies on it
BASE_X = (X_F - 0.026, X_F + D_L + 0.006)
BASE_HW = 0.031  # the base slab's half width (the drive overhangs it; the rails' bars run beside it)
HINGE_Y, HINGE_Z = 0.033, 0.005  # each rail's hinge: |y| from the drive's middle line, height above the table
RAIL_Y = (0.0352, 0.0392)  # posts and tab: |y| from the middle line (0.275 mm off the drive's side)
BAR_Y, BAR_Z = (0.033, 0.0392), (0.004, Z_F - 0.0005)  # the bar under the drive's side edge joining posts and tab
POST_HW, POST_TOP = 0.00175, Z_F + 0.0055  # posts round the pins: half length along x, top height
PIN_R, PIN_L = 0.00125, 0.003  # pins: radius, length (they stand 2.7 mm into the holes)
PIN_Z = Z_F + HOLE_Z
TAB_X, TAB_TOP = (X_F - 0.016, X_F - 0.003), 0.080  # the tab at each rail's front end
RAIL_KG = 0.02
RAIL_K = 0.2  # N m/rad: the rail's spring
RAIL_C = 0.003
PRELOAD_DEG = 1.0  # the spring aims this far past shut, so the rail rests on its stop
OPEN_LIMIT = 40.0  # degrees the rail can open
BREAK_N, BREAK_NM = 10.0, 0.5  # a rail's joint breaks under this force (N) or torque (N m)
MAT_X, MAT_Y, MAT_SX, MAT_SY, MAT_H = 0.26, -0.165, 0.20, 0.15, 0.002
HINGE_X = X_F + 0.036  # any point along the hinge axis

ALU, LABEL, LABEL_INK, PCB, CONN = ((0.72, 0.73, 0.75), (0.90, 0.90, 0.88), (0.25, 0.27, 0.32), (0.10, 0.35, 0.18),
                                    (0.08, 0.08, 0.09))
BASEC, BLUE, MATC, SNAP = (0.20, 0.21, 0.23), (0.16, 0.38, 0.78), (0.33, 0.45, 0.55), (0.80, 0.80, 0.82)


def _box(x0, x1, y0, y1, z0, z1, color, origin, collide=True) -> BoxShape:
    """A box from world corners (heights above the table), in the frame of a body whose origin is ``origin`` (its
    height also above the table)."""
    return BoxShape(((x0 + x1) / 2 - origin[0], (y0 + y1) / 2 - origin[1], (z0 + z1) / 2 - origin[2]),
                    (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


def _drive_shapes() -> list:
    """The drive in the frame of its bottom middle: a core and, along each side, a 4 mm strip built from boxes that
    leave the two square screw holes open; visual only: the label on the top cover, the circuit board underneath and
    the SATA connector at its back end."""
    core_hw = D_W / 2 - HOLE_D
    out = [BoxShape((0.0, 0.0, D_H / 2), (D_L, 2 * core_hw, D_H), ALU)]
    lo, hi = HOLE_Z - HOLE / 2, HOLE_Z + HOLE / 2
    edges = [-D_L / 2, HOLE_X[0] - HOLE / 2, HOLE_X[0] + HOLE / 2, HOLE_X[1] - HOLE / 2, HOLE_X[1] + HOLE / 2, D_L / 2]
    for s in (-1.0, 1.0):
        yc = s * (D_W / 2 - HOLE_D / 2)
        out.append(BoxShape((0.0, yc, lo / 2), (D_L, HOLE_D, lo), ALU))
        out.append(BoxShape((0.0, yc, (hi + D_H) / 2), (D_L, HOLE_D, D_H - hi), ALU))
        for a, b in ((edges[0], edges[1]), (edges[2], edges[3]), (edges[4], edges[5])):
            out.append(BoxShape(((a + b) / 2, yc, (lo + hi) / 2), (b - a, HOLE_D, hi - lo), ALU))
    out.append(BoxShape((-0.006, 0.0, D_H + 0.0003), (0.062, 0.052, 0.0006), LABEL, collide=False))
    for k, w in enumerate((0.040, 0.030, 0.036)):
        out.append(BoxShape((-0.026 + 0.008 * k, -0.026 + w / 2 + 0.002, D_H + 0.0007), (0.0025, w, 0.0002), LABEL_INK,
                            collide=False))
    out.append(BoxShape((0.006, 0.0, -0.0003), (0.080, 0.056, 0.0006), PCB, collide=False))
    out.append(BoxShape((D_L / 2 + 0.0004, -0.012, 0.0025), (0.0008, 0.032, 0.004), CONN, collide=False))
    return out


BASE_O = (X_C, Y_C, 0.0)


def _base_shapes() -> list:
    """The carrier's base in the frame of the middle of its bottom: the slab the drive lies on, and a seam line round
    its top (visual)."""
    o = BASE_O
    out = [_box(BASE_X[0], BASE_X[1], Y_C - BASE_HW, Y_C + BASE_HW, 0.0, Z_F, BASEC, o)]
    out.append(_box(BASE_X[0] + 0.002, BASE_X[0] + 0.004, Y_C - BASE_HW + 0.003, Y_C + BASE_HW - 0.003, Z_F,
                    Z_F + 0.0004, (0.30, 0.31, 0.33), o, collide=False))
    return out


def _rail_origin(s: float) -> tuple:
    return (HINGE_X, Y_C + s * HINGE_Y, HINGE_Z)


def _rail_shapes(s: float) -> list:
    """A rail (s = 1 on the drive's left, -1 on its right) in the frame of its hinge: the bar under the drive's side
    edge, two posts each holding a pin that stands into a screw hole, and the tab standing up at its front end."""
    o = _rail_origin(s)

    def ys(a, b):
        return (Y_C + s * a, Y_C + s * b) if s > 0 else (Y_C + s * b, Y_C + s * a)

    out = [_box(TAB_X[0], X_F + HOLE_X[1] + D_L / 2 + POST_HW, *ys(*BAR_Y), *BAR_Z, BLUE, o)]
    for hx in HOLE_X:
        x = X_C + hx
        out.append(_box(x - POST_HW, x + POST_HW, *ys(*RAIL_Y), BAR_Z[1], POST_TOP, BLUE, o))
        out.append(CylShape((x - o[0], Y_C + s * (RAIL_Y[0] - PIN_L / 2) - o[1], PIN_Z - o[2]), PIN_R, PIN_L,
                            (0.85, 0.86, 0.88), axis="Y"))
    out.append(_box(TAB_X[0], TAB_X[1], *ys(*RAIL_Y), BAR_Z[1], TAB_TOP, BLUE, o))
    out.append(_box(TAB_X[0] + 0.002, TAB_X[1] - 0.002, *ys(RAIL_Y[1], RAIL_Y[1] + 0.0006), TAB_TOP - 0.022,
                    TAB_TOP - 0.004, (0.10, 0.25, 0.55), o, collide=False))  # the tab's ribbed finger grip
    return out


def _mat_shapes() -> list:
    """The anti-static mat in the frame of the middle of its bottom, with its grounding snap in the far left corner."""
    return [BoxShape((0.0, 0.0, MAT_H / 2), (MAT_SX, MAT_SY, MAT_H), MATC),
            CylShape((MAT_SX / 2 - 0.015, MAT_SY / 2 - 0.015, MAT_H + 0.001), 0.005, 0.002, SNAP, collide=False)]


def _wp(p) -> tuple:
    return (float(p[0]), float(p[1]), float(TOP + p[2]))


DRIVE_START = np.array([X_C, Y_C, TOP + Z_F + 0.0005])
BASE = Body.build("carrier_base", pos=(X_C, Y_C, TOP), mass=1.0, color=BASEC, fixed=True, friction=(0.5, 0.45),
                  shapes=_base_shapes())
MAT = Body.build("antistatic_mat", pos=(MAT_X, MAT_Y, TOP), mass=0.5, color=MATC, fixed=True, friction=(0.9, 0.8),
                 shapes=_mat_shapes())
RAILS = []
for _name, _s in (("rail_left", 1.0), ("rail_right", -1.0)):
    _o = _rail_origin(_s)
    RAILS.append(Body.build(_name, pos=_wp(_o), mass=RAIL_KG, color=BLUE, friction=(0.5, 0.45), solver_iterations=32,
                            shapes=_rail_shapes(_s),
                            joint=Joint(anchor=_wp(_o), kind="revolute", axis="X", parent="carrier_base",
                                        limits=(-OPEN_LIMIT, 0.0) if _s > 0 else (0.0, OPEN_LIMIT),
                                        stiffness=RAIL_K, damping=RAIL_C, target=_s * PRELOAD_DEG,
                                        break_force=BREAK_N, break_torque=BREAK_NM)))
DRIVE = Body.build("hard_drive", pos=tuple(float(v) for v in DRIVE_START), mass=D_KG, color=ALU, friction=(0.4, 0.35),
                   solver_iterations=32, shapes=_drive_shapes())
OBJECTS = [BASE, MAT, *RAILS, DRIVE]

# -- scoring ----------------------------------------------------------------------------------------------------------
LIFT_MM, OUT_Z, OUT_R = 4.0, 0.010, 0.06  # one side lifted; out: lowest point above the base, or moved this far
FLAT_DEG, ON_MM, MAT_EDGE = 10.0, 4.0, 0.005
FALL_V, BANG_DV = 0.5, 0.6  # m/s: dropped (moving down this fast) or knocked (speed changed this much in one step)
HOME_DEG = 2.0
BROKEN_MM = 2.0  # a rail's hinge point this far from where it belongs: the joint broke
HELD_R, WITH_MM = 0.07, 2.0
REST_MM, REST_STEPS = 2.0, 3
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
DT = 0.05  # control step (20 Hz)
_DEBUG = os.environ.get("HDD_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("HDD_DEBUG_EVERY", "10"))
_SIDES = {"rail_left": 1.0, "rail_right": -1.0}
_HOLES = {n: [np.array([hx, s * D_W / 2, HOLE_Z]) for hx in HOLE_X] for n, s in _SIDES.items()}  # drive frame
HOLE_START_Z = float(DRIVE_START[2] + HOLE_Z)
_CORNERS = np.array([[sx * D_L / 2, sy * D_W / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, D_H)])
_TABS = {n: np.array([(TAB_X[0] + TAB_X[1]) / 2, Y_C + s * (RAIL_Y[0] + RAIL_Y[1]) / 2, TOP + (TAB_TOP + Z_F) / 2])
         for n, s in _SIDES.items()}
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


def _pin_tip_local(n: str, k: int) -> np.ndarray:
    """Pin k's tip in the rail's frame (the frame's origin is the hinge point)."""
    s = _SIDES[n]
    return np.array([X_C + HOLE_X[k] - HINGE_X, s * (RAIL_Y[0] - PIN_L) - s * HINGE_Y, PIN_Z - HINGE_Z])


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], hist=[], offsets={},
               pos=DRIVE_START.copy(), rot=np.eye(3), middle=DRIVE_START + [0.0, 0.0, D_H / 2], prev=None, vel=None,
               max_fall=0.0, max_dv=0.0, engaged={n: [True, True] for n in _SIDES}, rise={n: 0.0 for n in _SIDES},
               open_deg={n: 0.0 for n in _SIDES}, hinge_off={n: 0.0 for n in _SIDES}, low=Z_F, tilt=0.0,
               held=False, on_mat=False, lifted=False, out=False, tabs={n: _TABS[n].copy() for n in _SIDES})
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("HDD_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the drive's pose; each rail's angle and whether its joint broke;
    whether each pin still stands in its hole; whether the drive is lifted on one side, out, held, lying on the mat;
    whether it fell or was knocked."""
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
    dp = as_numpy(base.scene["hard_drive"].data.root_link_pose_w)[0]
    pos, rot = dp[:3] - origin, rotation(dp[3:7])
    middle = pos + rot @ np.array([0.0, 0.0, D_H / 2])
    corners = pos + _CORNERS @ rot.T
    low = float(corners[:, 2].min())
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))

    engaged, rise, open_deg, hinge_off, tabs = {}, {}, {}, {}, {}
    for n, s in _SIDES.items():
        rp = as_numpy(base.scene[n].data.root_link_pose_w)[0]
        rpos, rrot = rp[:3] - origin, rotation(rp[3:7])
        anchor = np.array(_wp(_rail_origin(s)))
        hinge_off[n] = float(np.linalg.norm(rpos - anchor))
        open_deg[n] = float(-s * math.degrees(math.atan2(rrot[2, 1], rrot[1, 1])))
        tabs[n] = rpos + rrot @ (_TABS[n] - anchor)
        flags = []
        for k in range(2):
            tip = rot.T @ (rpos + rrot @ _pin_tip_local(n, k) - pos)
            inside = (abs(tip[0] - HOLE_X[k]) < HOLE / 2 + 0.0005 and abs(tip[2] - HOLE_Z) < HOLE / 2 + 0.0005
                      and abs(tip[1]) < D_W / 2 + 0.0002)
            flags.append(bool(inside))
        engaged[n] = flags
        rise[n] = min(float((pos + rot @ _HOLES[n][k])[2]) for k in range(2)) - HOLE_START_Z
        if hinge_off[n] * 1000 > BROKEN_MM and not _EP["failed"]:
            _EP["failed"] = f"the {n.replace('_', ' ')} broke: it was forced past what it takes"

    # dropped or knocked: the drive's middle moving down fast, or its velocity jumping within one step
    if _EP["prev"] is not None and _EP["tick"] > 3:
        vel = (middle - _EP["prev"]) / DT
        if _EP["vel"] is not None:
            _EP["max_dv"] = max(_EP["max_dv"], float(np.linalg.norm(vel - _EP["vel"])))
        _EP["max_fall"] = max(_EP["max_fall"], float(-vel[2]))
        _EP["vel"] = vel
        if not _EP["failed"]:
            if -vel[2] > FALL_V:
                _EP["failed"] = f"the drive was dropped (it fell at {-vel[2]:.2f} m/s)"
            elif _EP["max_dv"] > BANG_DV:
                _EP["failed"] = f"the drive was knocked (its speed jumped {_EP['max_dv']:.2f} m/s in one step)"
    _EP["prev"] = middle.copy()

    held = False
    for side, tcp in _TCP.items():
        off = middle - tcp
        old = _EP["offsets"].get(side)
        if np.linalg.norm(off) < HELD_R and old is not None and np.linalg.norm(off - old) * 1000 < WITH_MM:
            held = True
        _EP["offsets"][side] = off
    _EP["hist"].append(middle.copy())
    del _EP["hist"][:-REST_STEPS]
    resting = (len(_EP["hist"]) == REST_STEPS
               and max(np.linalg.norm(h - _EP["hist"][-1]) for h in _EP["hist"]) * 1000 < REST_MM)

    lifted = any(not any(engaged[n]) and rise[n] * 1000 >= LIFT_MM for n in _SIDES)
    free = not any(any(f) for f in engaged.values())
    moved = float(np.hypot(middle[0] - X_C, middle[1] - Y_C))
    out = free and (low - TOP >= Z_F + OUT_Z or moved >= OUT_R)
    over_mat = (abs(middle[0] - MAT_X) < MAT_SX / 2 - MAT_EDGE and abs(middle[1] - MAT_Y) < MAT_SY / 2 - MAT_EDGE)
    on_mat = bool(over_mat and tilt <= FLAT_DEG and low - TOP <= MAT_H + ON_MM / 1000 and resting and not held)
    _EP.update(pos=pos, rot=rot, middle=middle, low=low, tilt=tilt, engaged=engaged, rise=rise, open_deg=open_deg,
               hinge_off=hinge_off, tabs=tabs, held=held, lifted=lifted, out=out, on_mat=on_mat)
    stage = 0.75 if on_mat else 0.5 if out else 0.25 if lifted else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["middle"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = 0.0
    if len(_TCP) == 2:
        targets = [_EP["middle"], *_EP["tabs"].values()]
        near = min(min(np.linalg.norm(v - t) for t in targets) for v in _TCP.values())
    home = all(abs(a) <= HOME_DEG for a in _EP["open_deg"].values())
    return dict(still=still, near=float(near), home=home,
                done=bool(_EP["on_mat"] and home and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    m = _EP["middle"]
    pins = " ".join("".join("I" if f else "o" for f in _EP["engaged"][n]) for n in _SIDES)
    print(f"DBG step={_EP['step']} middle=({m[0]:.4f},{m[1]:.4f},{m[2] - TOP:.4f}) "
          f"low={(_EP['low'] - TOP) * 1000:.1f}mm "
          f"tilt={_EP['tilt']:.1f} pins(L R)={pins} rise(L R)={_EP['rise']['rail_left'] * 1000:.1f} "
          f"{_EP['rise']['rail_right'] * 1000:.1f}mm open(L R)={_EP['open_deg']['rail_left']:.1f} "
          f"{_EP['open_deg']['rail_right']:.1f}deg hinge_off={max(_EP['hinge_off'].values()) * 1000:.2f}mm "
          f"held={_EP['held']} lifted={_EP['lifted']} out={_EP['out']} on_mat={_EP['on_mat']} home={t['home']} "
          f"near={t['near']:.3f} still={t['still']} max_fall={_EP['max_fall']:.3f} max_dv={_EP['max_dv']:.3f} "
          f"best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.4f},{v[1]:.4f},{v[2] - TOP:.4f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the drive was dropped or knocked or a rail broke): 0.25
    one side of the drive lifted off its rail's pins; 0.5 the drive out of the carrier; 0.75 lying flat on the mat, not
    held. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The drive lying flat on the mat, not held, both rails back home, both grippers 6 cm or more from the drive and
    the tabs, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the manual says: the right gripper takes the drive by its two sides, between the rails' posts; the left gripper,
# level and fingers closed, comes in front of the drive inside the left rail's tab and pushes the tab out, pulling the
# rail's pins out of the holes, and holds it there. The right gripper lifts the drive's left side, so that it pivots
# on the right rail's pins, slides it sideways off those pins and lifts it clear. The left gripper lets the rail spring
# back and moves away; the right carries the drive over the mat, lowers it until it nearly touches, lets go and moves
# away.
# The demo works from the known layout only. HDD_YANK=1 runs it the wrong way: nobody pulls a rail and the right
# gripper lifts the drive straight up.

WRONG = os.environ.get("HDD_YANK") == "1"


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


GRIP = grip_rotation(0.0)  # straight down, fingers closing along y (across the drive's width)
PUSH_PITCH = math.radians(10.0)
PUSH = _ry(PUSH_PITCH) @ _ry(-math.pi / 2) @ grip_rotation(0.0)  # level, pointing forward, tipped down 10 degrees
OPEN_W, CLOSED = -0.045, 0.0
PINCH = -(D_W / 2 - 0.004)  # each pad 4 mm inside the drive's side
FINGERTIP = 0.013  # the fingertips stand this far past the gripper's centre, along its pointing direction
TILT = math.radians(float(os.environ.get("HDD_TILT", "12")))
SLIDE = float(os.environ.get("HDD_SLIDE", "0.012"))  # commanded past the 3 mm needed: the arm is soft sideways
PUSH_OUT = float(os.environ.get("HDD_PUSH", "0.045"))  # how far the left gripper is sent out past touching the tab
B_X = X_C + (HOLE_X[0] + HOLE_X[1]) / 2  # the right gripper's centre: midway between the posts
B_TCP_Z = Z_F + 0.002 + FINGERTIP  # fingertips 2 mm above the drive's bottom
PIVOT = np.array([B_X, Y_C - (RAIL_Y[0] - PIN_L / 2), PIN_Z])  # the right rail's pins (heights above the table)
A_X = X_F - 0.003 - FINGERTIP * math.cos(PUSH_PITCH)  # fingertips 3 mm in front of the drive's front end
A_Y = Y_C + RAIL_Y[0] - 0.012  # the closed fingertips' outer face about 6 mm inside the tab
A_Z = float(os.environ.get("HDD_A_Z", "0.050"))  # the fingertips against the tab's middle, 5 cm above the table
CARRY_Z = 0.10
DROP_Z = MAT_H + 0.0015 + 0.002 + FINGERTIP  # the drive's bottom 1.5 mm above the mat

T_GRIP = 3.0 + 1.2 + 0.4 + 1.0 + 0.4  # the right gripper holds the drive
T_FLEXED = T_GRIP + 1.6 + 0.6  # the left rail held open
T_TILTED = T_FLEXED + 1.6 + 0.3
T_SLID = T_TILTED + 0.8 + 0.3
T_CLEAR = T_SLID + 1.2 + 0.3  # lifted clear of the carrier


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
    return Waypoint(t, _wp(p), finger, label, rot)


def _stay(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), CLOSED, "stay put", rot)]


def _tilted(phi: float) -> tuple:
    """The right gripper's centre and rotation with the drive turned ``phi`` about the right rail's pins (its left side
    up)."""
    rel = np.array([B_X, Y_C, B_TCP_Z]) - PIVOT
    return PIVOT + _rx(phi) @ rel, _rx(phi) @ GRIP


def _right() -> list:
    grab = np.array([B_X, Y_C, B_TCP_Z])
    out = _lead_in("right", _wp(grab + [0.0, 0.0, 0.05]), GRIP, OPEN_W, "above the drive, fingers wide open")
    out += [_w(1.2, grab, OPEN_W, "fingers down either side of the drive, between the rails' posts", GRIP),
            _w(0.4, grab, OPEN_W, "settle", GRIP),
            _w(1.0, grab, PINCH, "take the drive by its sides", GRIP),
            _w(0.4, grab, PINCH, "settle", GRIP)]
    if WRONG:
        return out + [_w(1.5, grab + [0.0, 0.0, 0.04], PINCH, "lift it straight up, nobody pulling a rail", GRIP),
                      _w(4.0, grab + [0.0, 0.0, 0.04], PINCH, "hold", GRIP)]
    out += [_w(T_FLEXED - T_GRIP, grab, PINCH, "wait for the left rail to be pulled out", GRIP)]
    n = 4
    for k in range(1, n + 1):
        p, r = _tilted(TILT * k / n)
        out.append(_w(1.6 / n, p, PINCH, "lift its left side: it pivots on the right rail's pins", r))
    p, r = _tilted(TILT)
    out.append(_w(0.3, p, PINCH, "settle", r))
    slid = p + _rx(TILT) @ np.array([0.0, SLIDE, 0.0])
    out += [_w(0.8, slid, PINCH, "slide it sideways off the right rail's pins", r),
            _w(0.3, slid, PINCH, "settle", r),
            _w(1.2, slid + [0.0, 0.0, 0.03], PINCH, "lift it clear of the carrier", r),
            _w(0.3, slid + [0.0, 0.0, 0.03], PINCH, "settle", r)]
    up = np.array([slid[0], slid[1], CARRY_Z])
    over = np.array([MAT_X + (B_X - X_C), MAT_Y, CARRY_Z])
    low = np.array([over[0], over[1], DROP_Z])
    away = np.array([0.14, -0.26, 0.20])
    out += [_w(1.0, up, PINCH, "level it and raise it", GRIP),
            _w(2.5, over, PINCH, "carry it over the mat", GRIP),
            _w(0.4, over, PINCH, "settle", GRIP),
            _w(1.6, low, PINCH, "lower it until it nearly touches the mat", GRIP),
            _w(0.5, low, PINCH, "settle", GRIP),
            _w(0.8, low, OPEN_W, "let go", GRIP),
            _w(1.0, over, OPEN_W, "rise", GRIP),
            _w(1.4, away, OPEN_W, "move away", GRIP),
            _w(4.0, away, OPEN_W, "stay clear", GRIP)]
    return out


def _left() -> list:
    if WRONG:
        return _stay("left", 14.0)
    touch = np.array([A_X, A_Y, A_Z])
    pushed = touch + [0.0, PUSH_OUT, 0.0]
    out = _lead_in("left", _wp(touch + [0.0, 0.0, 0.04]), PUSH, CLOSED, "above the inside of the left rail's tab")
    out += [_w(1.2, touch, CLOSED, "down inside the tab, in front of the drive", PUSH),
            _w(T_GRIP - (3.0 + 1.2), touch, CLOSED, "wait for the right gripper to hold the drive", PUSH),
            _w(1.6, pushed, CLOSED, "push the tab out: the rail's pins leave the drive's holes", PUSH),
            _w(T_CLEAR - T_GRIP - 1.6 + 0.4, pushed, CLOSED, "hold the rail open while the drive comes out", PUSH),
            _w(1.2, touch - [0.0, 0.006, 0.0], CLOSED, "let the rail spring back", PUSH),
            _w(1.0, touch + [-0.03, -0.01, 0.06], CLOSED, "rise", PUSH),
            _w(1.4, np.array([0.18, 0.24, 0.16]), CLOSED, "move away", PUSH),
            _w(10.0, np.array([0.18, 0.24, 0.16]), CLOSED, "stay clear", PUSH)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
