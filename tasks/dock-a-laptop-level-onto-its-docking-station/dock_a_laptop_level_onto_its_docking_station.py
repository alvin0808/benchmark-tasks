"""On the station table: a closed 13-inch business laptop lies on a small wooden laptop stand near the robot, its left
and right sides overhanging the stand, and a mechanical docking station (a port replicator, like Dell's E-Port)
stands behind it, its top 1 cm higher than the stand's. The laptop's back (hinge) edge faces the docking station. In
the laptop's underside, near its back edge, is the docking socket, with a latch slot on each side of it. On the
docking station's top stand the docking connector, a spring latch hook on each side of it, and two guide posts at its
back corners.

Task (instruction): dock the laptop. The instruction does not say how. The procedure the task checks is the docking
station maker's (Dell E-Port User's Guide, "Docking Your Laptop"): "Center the laptop with the E-Port, then push down
on the laptop until it clicks". The laptop is 31 cm wide, more than three times as wide as a gripper opens, and has no
handle: it is taken by both side edges, carried level onto the docking station, set back against the guide posts and
lowered straight down, so that the connector enters the socket; then both sides are pressed down until both latches
click. Held by one edge, it tips about the other and cannot be lifted level; the docking station stands higher than
the stand, so the laptop cannot be pushed across onto it.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- laptop: 1.3 kg, 21.5 cm deep (x) x 31 cm wide (y) x 1.9 cm thick, lying flat on the stand with its middle at x 0.12,
  y 0 (x 0.0125 to 0.2275), its back edge toward +x. Its frame: origin at the middle of its underside. The socket: a
  recess 22 x 50 mm (x, y), 12 mm deep, centred 3.25 cm in front of its back edge. Each latch slot (16 mm wide, at y
  -6.5 and +6.5 cm, as long as the socket) has its catch at its back: the bottom cover's 3 mm thick, 4 mm deep lip,
  with 9 mm of room above it.
- laptop stand: fixed, x 0.02 to 0.22, y -0.10 to 0.10, 3 cm tall: the laptop's sides overhang it by 5.5 cm.
- docking station: fixed, x 0.2375 to 0.385, y -0.105 to 0.105, its top 4 cm above the table. Guide posts at its back
  corners (y 0.075 to 0.105 each side), 1.5 cm tall, front faces at x 0.373. Connector: 16 x 44 mm (x, y), 7 mm tall,
  x 0.332 to 0.348 (3 mm clear of the socket all round). Latch hooks at y -0.065 and 0.065: each a steel stem with a
  rounded nose (a 3.5 mm radius cylinder along y, 7 mm above the top) at its back, on a slide along x with a 5000 N/m
  spring. Docked (the laptop's middle at x 0.265, its back edge against the posts), each nose overlaps its catch by
  2 mm. Pressed down onto the hooks, the catches push the noses forward (and the noses push the laptop back against
  the posts) until they pass below them, and the noses spring back over the catches: the latches click.

Score: 0 for the rest of the run once the laptop has been dropped (its middle falling faster than 0.6 m/s), turned
over, knocked off the table, or pulled off the docking station after it clicked in (its socket's mouth 3 mm or more
above the connector's base after both latches clicked). Otherwise the best stage reached: 0.25 lifted clear of the
stand (its lowest corner 1.5 cm or more above the stand's top, tilted under 10 degrees); 0.5 put down on the docking
station against the guide posts and let go (its back edge within 5 mm of both posts, its middle within 1 cm of the
station's centre line, the middle of its underside within 1 cm above the station's top, tilted under 6 degrees, at
rest, both grippers 6 cm or more from it); 0.75 docked (within 3 mm of its docked place, its socket's mouth within
1.5 mm of the connector's base, level within 2 degrees, both latch hooks back over their catches). Full score: docked,
both grippers 6 cm or more from the laptop, still for 1 s.
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

# -- layout (world frame, heights above the table) --------------------------------------------------------------------
# The laptop, in its own frame: origin at the middle of its underside, x toward its back (hinge) edge, y to its left.
LT_D, LT_W, LT_T = 0.215, 0.310, 0.019  # depth (x), width (y), thickness (z)
LT_KG = 1.3
LT_SPLIT = 0.012  # the recesses in the underside are this deep; the rest of the laptop is solid above
X_START, X_DOCK = 0.12, 0.265  # the laptop's middle (x): on the stand, and docked
STAND_X, STAND_HW, STAND_H = (X_START - 0.10, X_START + 0.10), 0.10, 0.03  # the laptop stand
DOCK_H = 0.04  # the docking station's top above the table
GUIDE_X = X_DOCK + LT_D / 2 + 0.0005  # the guide posts' front faces: the docked laptop's back edge rests against them
DOCK_X = (X_DOCK - 0.0275, GUIDE_X + 0.012)  # the docking station's front and back
DOCK_HW = 0.105
GUIDE_Y, GUIDE_UP = (0.075, 0.105), 0.015  # each guide post spans this |y|; it stands this far above the top
# The socket band in the laptop's underside (laptop frame): the socket and the two latch slots run across it.
BAND_X = (0.064, 0.086)
SOCK_HY = 0.025  # the socket's half length (y)
HOOK_Y, SLOT_HY = 0.065, 0.008  # the latch slots' middles (y -HOOK_Y and +HOOK_Y) and half widths
CATCH_X0, SKIN = 0.082, 0.003  # each slot's catch: the bottom cover's 3 mm lip from CATCH_X0 to BAND_X[1]
# On the docking station (x in the docked laptop's frame, z above the station's top):
CONN_X, CONN_HY, CONN_H = (0.067, 0.083), 0.022, 0.007  # the connector: 3 mm clear of the socket all round
STEM_X, STEM_HY, STEM_Z = (0.0715, 0.0805), 0.004, (-0.006, 0.0105)  # a latch hook's stem
NOSE_X, NOSE_Z, NOSE_R = 0.0805, 0.007, 0.0035  # its nose: a cylinder along y at the stem's back, over the catch
HOOK_K = float(os.environ.get("DOCK_HOOK_K", "5000"))  # N/m: the latch spring
HOOK_C, HOOK_TRAVEL = 2.0, 0.004
HOOK_KG = 0.01

LID, BASE, SEAM, LOGO = (0.74, 0.75, 0.77), (0.42, 0.43, 0.45), (0.10, 0.10, 0.11), (0.55, 0.56, 0.58)
DOCK_C, DOCK_TOP, RUBBER, STEEL, GOLD = ((0.09, 0.09, 0.10), (0.17, 0.17, 0.18), (0.05, 0.05, 0.05),
                                         (0.62, 0.63, 0.65), (0.80, 0.62, 0.22))
WOOD, MARK, EJECT = (0.36, 0.20, 0.10), (0.92, 0.92, 0.90), (0.55, 0.12, 0.10)


def _box(x0, x1, y0, y1, z0, z1, color, origin=(0.0, 0.0, 0.0), collide=True) -> BoxShape:
    """A box from its corners, in the frame of a body whose origin is ``origin``."""
    return BoxShape(((x0 + x1) / 2 - origin[0], (y0 + y1) / 2 - origin[1], (z0 + z1) / 2 - origin[2]),
                    (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


def _laptop_shapes() -> list:
    """The closed laptop in its own frame: the solid upper part, the bottom part round the socket and the two latch
    slots, each slot's catch; visual only: the seam between lid and base, the hinge, the lid's logo."""
    hd, hw = LT_D / 2, LT_W / 2
    out = [_box(-hd, hd, -hw, hw, LT_SPLIT, LT_T, LID),
           _box(-hd, BAND_X[0], -hw, hw, 0.0, LT_SPLIT, BASE),
           _box(BAND_X[1], hd, -hw, hw, 0.0, LT_SPLIT, BASE)]
    ys = [-hw, -HOOK_Y - SLOT_HY, -HOOK_Y + SLOT_HY, -SOCK_HY, SOCK_HY, HOOK_Y - SLOT_HY, HOOK_Y + SLOT_HY, hw]
    for a, b in ((ys[0], ys[1]), (ys[2], ys[3]), (ys[4], ys[5]), (ys[6], ys[7])):
        out.append(_box(BAND_X[0], BAND_X[1], a, b, 0.0, LT_SPLIT, BASE))
    for s in (-1.0, 1.0):
        out.append(_box(CATCH_X0, BAND_X[1], s * HOOK_Y - SLOT_HY, s * HOOK_Y + SLOT_HY, 0.0, SKIN, BASE))
    # visual: a dark seam round the sides where the lid meets the base, the hinge along the back, the logo
    z0, z1 = 0.0115, 0.0125  # the seam
    out += [_box(-hd - 0.0004, -hd, -hw, hw, z0, z1, SEAM, collide=False),
            _box(hd, hd + 0.0004, -hw + 0.03, hw - 0.03, z0, z1, SEAM, collide=False),
            _box(-hd, hd, hw, hw + 0.0004, z0, z1, SEAM, collide=False),
            _box(-hd, hd, -hw - 0.0004, -hw, z0, z1, SEAM, collide=False),
            _box(hd - 0.012, hd + 0.0004, -hw + 0.045, hw - 0.045, LT_T - 0.001, LT_T + 0.0004, SEAM, collide=False),
            _box(-0.02, 0.02, -0.02, 0.02, LT_T, LT_T + 0.0004, LOGO, collide=False),
            _box(-hd - 0.0004, -hd, -0.025, 0.025, 0.006, 0.0075, SEAM, collide=False)]
    return out


DOCK_O = ((DOCK_X[0] + DOCK_X[1]) / 2, 0.0, 0.0)


def _dock_shapes() -> list:
    """The docking station in the frame of the middle of its base: the body, the two guide posts at its back corners
    and the docking connector; visual only: a rubber pad on the top, slots round the latch hooks, the connector's
    contacts, an alignment mark at the front, the eject button, ports on the back."""
    o, xd = DOCK_O, X_DOCK
    out = [_box(DOCK_X[0], DOCK_X[1], -DOCK_HW, DOCK_HW, 0.0, DOCK_H, DOCK_C, o)]
    for s in (-1.0, 1.0):
        y0, y1 = sorted((s * GUIDE_Y[0], s * GUIDE_Y[1]))
        out.append(_box(GUIDE_X, DOCK_X[1], y0, y1, DOCK_H, DOCK_H + GUIDE_UP, DOCK_C, o))
    out.append(_box(xd + CONN_X[0], xd + CONN_X[1], -CONN_HY, CONN_HY, DOCK_H, DOCK_H + CONN_H, RUBBER, o))
    # visual
    out.append(_box(DOCK_X[0] + 0.008, xd + 0.040, -DOCK_HW + 0.012, DOCK_HW - 0.012, DOCK_H, DOCK_H + 0.0005,
                    DOCK_TOP, o, collide=False))
    for s in (-1.0, 1.0):
        y = s * HOOK_Y
        out.append(_box(xd + STEM_X[0] - 0.005, xd + STEM_X[1] + 0.006, y - 0.007, y + 0.007, DOCK_H,
                        DOCK_H + 0.0004, RUBBER, o, collide=False))
    out.append(_box(xd + CONN_X[0] + 0.003, xd + CONN_X[1] - 0.003, -CONN_HY + 0.003, CONN_HY - 0.003,
                    DOCK_H + CONN_H, DOCK_H + CONN_H + 0.0004, GOLD, o, collide=False))
    out.append(_box(DOCK_X[0] - 0.0004, DOCK_X[0], -0.012, 0.012, DOCK_H - 0.012, DOCK_H - 0.008, MARK, o,
                    collide=False))
    out.append(_box(DOCK_X[0] + 0.002, DOCK_X[0] + 0.010, -0.0015, 0.0015, DOCK_H, DOCK_H + 0.0005, MARK, o,
                    collide=False))
    out.append(_box(DOCK_X[0] - 0.0005, DOCK_X[0], DOCK_HW - 0.040, DOCK_HW - 0.015, 0.012, 0.024, EJECT, o,
                    collide=False))
    for k in range(5):
        y = -0.08 + 0.04 * k
        out.append(_box(DOCK_X[1], DOCK_X[1] + 0.0005, y - 0.008, y + 0.008, 0.012, 0.024, RUBBER, o,
                        collide=False))
    return out


def _hook_shapes() -> list:
    """A latch hook in the frame of its stem's middle on the station's top: the stem (standing in the station; it does
    not touch it) and the rounded nose at its back, over the catch when docked."""
    cx = (STEM_X[0] + STEM_X[1]) / 2
    return [BoxShape((0.0, 0.0, (STEM_Z[0] + STEM_Z[1]) / 2), (STEM_X[1] - STEM_X[0], 2 * STEM_HY,
                                                              STEM_Z[1] - STEM_Z[0]), STEEL),
            CylShape((NOSE_X - cx, 0.0, NOSE_Z), NOSE_R, 2 * STEM_HY, STEEL, axis="Y")]


def _hook_pos(s: float) -> tuple:
    return (X_DOCK + (STEM_X[0] + STEM_X[1]) / 2, s * HOOK_Y, TOP + DOCK_H)


START = np.array([X_START, 0.0, TOP + STAND_H + 0.001])
STAND = Body.build("laptop_stand", pos=((STAND_X[0] + STAND_X[1]) / 2, 0.0, TOP), mass=1.0, color=WOOD, fixed=True,
                   friction=(0.6, 0.5),
                   shapes=[BoxShape((0.0, 0.0, STAND_H / 2), (STAND_X[1] - STAND_X[0], 2 * STAND_HW, STAND_H), WOOD),
                           BoxShape((0.0, 0.0, STAND_H + 0.0002), (STAND_X[1] - STAND_X[0] - 0.012,
                                                                   2 * STAND_HW - 0.012, 0.0004), (0.30, 0.16, 0.08),
                                    collide=False)])
DOCK = Body.build("docking_station", pos=(DOCK_O[0], 0.0, TOP), mass=1.0, color=DOCK_C, fixed=True,
                  friction=(0.5, 0.4), shapes=_dock_shapes())
HOOKS = [Body.build(f"latch_{name}", pos=_hook_pos(s), mass=HOOK_KG, color=STEEL, friction=(0.2, 0.15),
                    solver_iterations=32, shapes=_hook_shapes(),
                    joint=Joint(anchor=_hook_pos(s), kind="slide", axis="X", parent="docking_station",
                                axial=(HOOK_K, HOOK_C, HOOK_TRAVEL)))
         for name, s in (("left", 1.0), ("right", -1.0))]
LAPTOP = Body.build("laptop", pos=tuple(float(v) for v in START), mass=LT_KG, color=LID, friction=(0.5, 0.4),
                    solver_iterations=32, shapes=_laptop_shapes())
OBJECTS = [STAND, DOCK, *HOOKS, LAPTOP]

# -- scoring ----------------------------------------------------------------------------------------------------------
LIFT_CLEAR, LIFT_TILT = 0.015, 10.0  # lifted: lowest corner this far above the stand, tilted under this (degrees)
GUIDE_MM, CENTRE_MM, ON_DOCK_Z, ON_DOCK_TILT = 5.0, 10.0, 0.010, 6.0
RELEASED = 0.06  # let go: both grippers' centres this far or more from the laptop
PLACE_MM, DOWN_MM, LEVEL_DEG = 3.0, 1.5, 2.0  # docked: within this of its place, down on the top, level
CATCH_MM, HOOK_BACK_MM = 0.5, 0.8  # a hook over its catch: overlapping it by this much, its spring back within this
CATCH_GAP_MM = 1.5  # ... and the catch's top no more than this below the hook's nose
DROP_V = 0.6  # m/s: the laptop's middle falling faster than this has been dropped
FORCED_MM = 3.0  # after both latches clicked, the socket's mouth this far above the connector's base: pulled off
REST_MM, REST_STEPS = 3.0, 3
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("DOCK_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("DOCK_DEBUG_EVERY", "10"))
_CORNERS = np.array([[sx * LT_D / 2, sy * LT_W / 2, z] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                     for z in (0.0, LT_T)])
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], hist=[], pos=START.copy(), rot=np.eye(3),
               low=STAND_H, tilt=0.0, fall=0.0, lifted=False, on_dock=False, docked=False, latched=[False, False],
               hook_dx=[0.0, 0.0], overlap=[0.0, 0.0], back_gap=[0.0, 0.0], prev_z=None, max_fall=0.0,
               clicked=False, sock_up=0.0, near=0.0, resting=False)
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("DOCK_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _world(p) -> np.ndarray:
    """A point of the laptop (its own frame) in the world."""
    return _EP["pos"] + _EP["rot"] @ np.asarray(p, float)


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the laptop's pose, lowest corner, tilt and fall speed; how far each
    latch hook is pushed forward and whether it is back over its catch; the stage reached."""
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
    lp = as_numpy(base.scene["laptop"].data.root_link_pose_w)[0]
    pos, rot = lp[:3] - origin, rotation(lp[3:7])
    _EP.update(pos=pos, rot=rot)
    corners = pos + _CORNERS @ rot.T
    low = float(corners[:, 2].min() - TOP)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))
    mid = _world([0.0, 0.0, LT_T / 2])
    dt = float(getattr(base, "step_dt", 0.05))
    fall = 0.0 if _EP["prev_z"] is None else (_EP["prev_z"] - float(mid[2])) / dt
    _EP["prev_z"] = float(mid[2])
    _EP["max_fall"] = max(_EP["max_fall"], fall)
    if not _EP["failed"]:
        if fall > DROP_V:
            _EP["failed"] = f"the laptop was dropped (its middle fell at {fall:.2f} m/s)"
        elif rot[2, 2] < 0.0:
            _EP["failed"] = "the laptop was turned over"
        elif mid[2] < TOP - 0.01:
            _EP["failed"] = "the laptop fell off the table"
    # the latch hooks: pushed forward (-x) how far; back over the catch (its top below the nose, overlapping it)?
    latched, dxs, overlaps = [], [], []
    for k, (name, s) in enumerate((("latch_left", 1.0), ("latch_right", -1.0))):
        hp = as_numpy(base.scene[name].data.root_link_pose_w)[0][:3] - origin
        dx = float(hp[0] - _hook_pos(s)[0])
        nose_back = X_DOCK + NOSE_X + NOSE_R + dx
        catch_front = _world([CATCH_X0, s * HOOK_Y, 0.0])
        catch_top = _world([CATCH_X0, s * HOOK_Y, SKIN])
        overlap = float(nose_back - catch_front[0])
        gap = (NOSE_Z - NOSE_R) - float(catch_top[2] - TOP - DOCK_H)  # the catch's top below the nose's underside
        under = -0.0002 < gap < CATCH_GAP_MM / 1000
        in_y = abs(float(catch_front[1]) - s * HOOK_Y) < SLOT_HY - STEM_HY
        latched.append(bool(abs(dx) * 1000 < HOOK_BACK_MM and CATCH_MM <= overlap * 1000 <= CATCH_MM + 4.5
                            and under and in_y))
        dxs.append(dx)
        overlaps.append(overlap)
    sock_up = float(_world([(CONN_X[0] + CONN_X[1]) / 2, 0.0, 0.0])[2] - TOP - DOCK_H)  # the socket's mouth
    if all(latched):
        _EP["clicked"] = True
    elif _EP["clicked"] and sock_up * 1000 > FORCED_MM and not _EP["failed"]:
        _EP["failed"] = "the laptop was pulled off the docking station after it had clicked in (latches forced)"
    near = _near()
    lifted = low - STAND_H >= LIFT_CLEAR and tilt < LIFT_TILT
    backs = [float(GUIDE_X - _world([LT_D / 2, s * (GUIDE_Y[0] + GUIDE_Y[1]) / 2, LT_T / 2])[0]) for s in (1.0, -1.0)]
    _EP["hist"].append(pos.copy())
    del _EP["hist"][:-REST_STEPS]
    resting = (len(_EP["hist"]) == REST_STEPS
               and max(np.linalg.norm(h - _EP["hist"][-1]) for h in _EP["hist"]) * 1000 < REST_MM)
    under_mid = float(pos[2] - TOP - DOCK_H)  # the middle of its underside above the station's top
    on_dock = bool(max(backs) * 1000 < GUIDE_MM and min(backs) > -0.002 and abs(mid[1]) * 1000 < CENTRE_MM
                   and -0.003 < under_mid < ON_DOCK_Z and tilt < ON_DOCK_TILT and resting and near >= RELEASED)
    docked = bool(abs(pos[0] - X_DOCK) * 1000 < PLACE_MM and abs(pos[1]) * 1000 < PLACE_MM
                  and sock_up * 1000 < DOWN_MM and tilt < LEVEL_DEG and all(latched))
    _EP.update(low=low, tilt=tilt, fall=fall, lifted=lifted, on_dock=on_dock, docked=docked, latched=latched,
               hook_dx=dxs, overlap=overlaps, back_gap=backs, resting=resting, sock_up=sock_up, near=near)
    stage = 0.75 if docked else 0.5 if on_dock else 0.25 if lifted else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _near() -> float:
    """The nearest gripper centre's distance from the laptop (from its surface)."""
    if len(_TCP) < 2:
        return 0.0
    half = np.array([LT_D / 2, LT_W / 2, LT_T / 2])
    out = []
    for v in _TCP.values():
        local = _EP["rot"].T @ (v - _EP["pos"]) - np.array([0.0, 0.0, LT_T / 2])
        out.append(float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0))))
    return min(out)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["pos"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = _EP["near"]
    return dict(still=still, near=near, done=bool(_EP["docked"] and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p = _EP["pos"]
    print(f"DBG step={_EP['step']} pos=({p[0]:.4f},{p[1]:.4f},{p[2] - TOP:.4f}) low={_EP['low'] * 100:.2f}cm "
          f"tilt={_EP['tilt']:.2f}deg fall={_EP['fall']:.2f}m/s sock_up={_EP['sock_up'] * 1000:.1f}mm back_gap_mm(L R)="
          f"{_EP['back_gap'][0] * 1000:.1f} {_EP['back_gap'][1] * 1000:.1f} hook_dx_mm(L R)="
          f"{_EP['hook_dx'][0] * 1000:.2f} {_EP['hook_dx'][1] * 1000:.2f} overlap_mm(L R)="
          f"{_EP['overlap'][0] * 1000:.2f} {_EP['overlap'][1] * 1000:.2f} latched={_EP['latched']} "
          f"clicked={_EP['clicked']} lifted={_EP['lifted']} on_dock={_EP['on_dock']} docked={_EP['docked']} "
          f"near={t['near']:.3f} max_fall={_EP['max_fall']:.2f} "
          f"still={t['still']} best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the laptop has been dropped, turned over, knocked off
    the table, or pulled off the docking station after it clicked in): 0.25 lifted clear of the stand; 0.5 put down on
    the docking station against its guide posts and let go; 0.75 docked (socket on the connector, level, both latches
    clicked). Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """Docked, both grippers 6 cm or more from the laptop, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the manual says, with the laptop kept level and square. Each gripper, tilted so that it points in and down over
# the laptop's side edge (which overhangs the stand), comes in with its fingers open, one above the lid and one below
# the edge, and pinches the edge between them; together they lift the laptop off the stand, carry it back over the
# docking station, lower it to just above the connector, set its back edge against the guide posts and lower it
# straight down onto the connector. They let go and draw out sideways, come over the lid near its left and right
# edges, straight down, and press both sides down together until both latches click, then rise and move away. Open
# loop: every target comes from the layout above. DOCK_ONE=1 runs it the wrong way: the right arm alone does the same.

WRONG = os.environ.get("DOCK_ONE") == "1"
TILT = math.radians(float(os.environ.get("DOCK_TILT", "75")))  # the pinching grippers' tilt from straight down
CLOSED = 0.0
DOWN = grip_rotation(0.0)  # straight down, fingers closing along y: for pressing the lid
PINCH_L = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(TILT), math.sin(TILT)],
                    [0.0, -math.sin(TILT), math.cos(TILT)]]) @ DOWN  # left: pointing in (-y) and down
PINCH_R = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(TILT), -math.sin(TILT)],
                    [0.0, math.sin(TILT), math.cos(TILT)]]) @ DOWN  # right: pointing in (+y) and down
FINGERTIP = 0.013  # the fingertips stand this far past the gripper's centre, along its pointing direction
BEAR = float(os.environ.get("DOCK_BEAR", "-0.003"))  # the laptop's bottom edge bears on the lower pad this far toward
# the fingertips from the gripper's centre (the pads run from 5 mm behind it to 13 mm past it)
GRIP_IN = float(os.environ.get("DOCK_GRIP_IN", "0.006"))  # each finger commanded this far past the face it pinches
OPEN_MORE = 0.010  # opened: each finger this far clear of the face it pinches
SAG = float(os.environ.get("DOCK_SAG", "0.015"))  # the arms sag this much under the laptop: commanded this much higher
PUSH = float(os.environ.get("DOCK_PUSH", "0.004"))  # set back: commanded this far past the guide posts
PRESS_X, PRESS_Y = 0.05, 0.12  # where each gripper presses the lid (laptop frame): behind its middle, near each side
PRESS = float(os.environ.get("DOCK_PRESS", "0.03"))  # pressed: commanded this far below the lid's top


def _pinch() -> np.ndarray:
    """(in, up, gap): the gripper's centre this far inside the laptop's side edge and above its underside, and the gap
    between the finger pads, when the upper fingertip is on the lid and the bottom edge bears on the lower pad."""
    sa, ca = math.sin(TILT), math.cos(TILT)
    a = np.array([[0.0, 1.0, sa / 2], [ca, sa, -0.5], [-sa, ca, 0.0]])
    return np.linalg.solve(a, np.array([LT_T + FINGERTIP * ca, 0.0, BEAR]))


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


def _stay(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), CLOSED, "stay put", rot)]


def _dock(arm: str) -> list:
    """Pinch the laptop's side edge on this arm's side and, with the other arm, carry it onto the docking station, set
    it back against the guide posts and lower it onto the connector; let go, press the lid down on this side, move
    away."""
    s = 1.0 if arm == "left" else -1.0
    rot = PINCH_L if arm == "left" else PINCH_R
    d_in, up, gap = _pinch()
    pinch, opened = -(gap / 2 - GRIP_IN), -(gap / 2 + OPEN_MORE)

    def at(x, bottom, out=0.0):
        """The gripper's centre pinching the edge when the laptop's underside is at ``bottom`` (out: drawn out)."""
        return np.array([x, s * (LT_W / 2 - d_in + out), bottom + up])

    b0 = TOP + STAND_H + 0.001  # the laptop's underside on the stand
    carry = TOP + DOCK_H + 0.020 + SAG  # 2 cm above the station's top (over the guide posts)
    near = TOP + DOCK_H + 0.013 + SAG  # 1.3 cm above it: clear of the connector and the hooks
    seat = TOP + DOCK_H + 0.003  # resting on the hooks' noses, before it clicks
    lid = TOP + DOCK_H + LT_T  # the docked laptop's top
    xb = X_DOCK + PUSH
    press = np.array([X_DOCK + PRESS_X, s * PRESS_Y, lid + FINGERTIP])
    out = _lead_in(arm, at(X_START, b0 + 0.04, 0.05), rot, opened, "beside the laptop's side, tilted in, open")
    out += [_w(1.0, at(X_START, b0, 0.05), opened, "level with its side edge", rot),
            _w(1.2, at(X_START, b0), opened, "in over the edge: one finger above the lid, one below the edge", rot),
            _w(0.4, at(X_START, b0), opened, "settle", rot),
            _w(1.0, at(X_START, b0), pinch, "pinch the edge", rot),
            _w(0.5, at(X_START, b0), pinch, "settle", rot),
            _w(1.5, at(X_START, carry), pinch, "lift it level off the stand, together", rot),
            _w(3.0, at(X_DOCK - 0.005, carry), pinch, "carry it back over the docking station", rot),
            _w(0.5, at(X_DOCK - 0.005, carry), pinch, "settle", rot),
            _w(1.0, at(X_DOCK - 0.005, near), pinch, "lower it to just above the connector", rot),
            _w(1.0, at(xb, near), pinch, "set its back edge against the guide posts", rot),
            _w(0.4, at(xb, near), pinch, "settle", rot),
            _w(1.5, at(xb, seat), pinch, "lower it straight down onto the connector", rot),
            _w(0.5, at(xb, seat), pinch, "settle", rot),
            _w(0.8, at(X_DOCK, seat - SAG / 2), opened, "let go", rot),
            _w(1.0, at(X_DOCK, seat - SAG / 2, 0.05), opened, "draw out sideways", rot),
            _w(0.8, at(X_DOCK, lid + 0.05, 0.05), opened, "rise", rot),
            _w(1.2, press + [0.0, 0.0, 0.03], CLOSED, "over the lid near its side, straight down", DOWN),
            _w(0.8, press + [0.0, 0.0, 0.012], CLOSED, "down onto the lid", DOWN),
            _w(1.0, press - [0.0, 0.0, PRESS], CLOSED, "press both sides down together until it clicks", DOWN),
            _w(0.6, press - [0.0, 0.0, PRESS], CLOSED, "hold", DOWN),
            _w(0.8, press + [0.0, 0.0, 0.06], CLOSED, "rise", DOWN),
            _w(1.2, [0.16, s * 0.24, TOP + 0.16], CLOSED, "move away", DOWN),
            _w(3.0, [0.16, s * 0.24, TOP + 0.16], CLOSED, "stay clear", DOWN)]
    return out


def demo():
    if WRONG:
        right = _dock("right")
        return {"left": _stay("left", sum(w.duration for w in right)), "right": right}
    return {"left": _dock("left"), "right": _dock("right")}
