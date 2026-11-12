"""On the station table: a car's rear seat, scaled down, with its head restraint in place, and an open parts box beside
the seat cushion. The head restraint's two poles stand in two holes in the top of the seatback. A lock plate in the
seatback's top catches a stop at the foot of the left pole when the head restraint is pulled up as far as it goes; it
slides back (away from the robot) and lets the stop pass only while its red release tab, standing out of the
seatback's front on the left, is pushed in.

Task (instruction): take the head restraint off and put it in the box. The instruction does not say how. The
procedure the task checks is the car maker's (Hyundai owner's manual, "Head restraint", removal from a rear seat):
"Raise the head restraint as far as it can go." "Press the head restraint release button (1) while pulling up the head
restraint (2)." Pulled up without the button pressed, the head restraint stops at the top; pulled harder, it forces the
lock and breaks it. One hand cannot press the button and pull the head restraint at once.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- seatback: fixed, x 0.24 to 0.30, y -0.09 to 0.09, 5 cm tall; two 12 mm square holes, 3.5 cm deep, at x 0.27,
  y -0.035 and 0.035. Seat cushion in front of it (x 0.12 to 0.24, 2.5 cm tall).
- head restraint: 0.15 kg, a 10 x 4 x 4 cm cushion whose bottom is 1.5 cm above the seatback, on two 8 mm poles 5 cm
  long standing on the floors of the holes; a stop at the foot of the left pole, on its inner (-y) side.
- lock plate: in the seatback's top, on a 400 N/m spring sliding along x; its tongue covers the stop's way up; its red
  tab stands out of the seatback's front at y 0.052 to 0.068, 1.8 cm above the top; pushed in 6 mm or more it lets the
  stop pass. Its joint breaks under 15 N (the lock forced).
- box: open, fixed, x 0.14 to 0.27, y -0.23 to -0.11, walls 5 cm tall.

Score: 0 for the rest of the run once the lock has been forced. Otherwise the best stage reached: 0.25 the head
restraint raised as far as it goes (2 cm or more); 0.5 taken off (its poles out of the holes); 0.75 in the box (its
middle inside the walls and its lowest point 1 cm or more below their tops, at rest, no gripper holding it). Full score: that, with both grippers
6 cm or more from it and from the tab, still for 1 s.
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
SB_X0, SB_X1, SB_HW, SB_H = 0.24, 0.30, 0.09, 0.05  # seatback: front, back (x), half width (y), height
POLE_X, POLE_Y, POLE_R, POLE_L = 0.27, 0.035, 0.004, 0.05  # poles at y -POLE_Y and +POLE_Y
HOLE, HOLE_D = 0.012, 0.035  # hole: square side, depth (the poles stand on the holes' floors)
SLOT = 0.004  # the left hole is widened this much toward -y for the stop to travel in
CUSH = (0.04, 0.10, 0.04)  # cushion: x, y, z
GAP = 0.015  # seated: the cushion's bottom above the seatback
HR_KG = 0.15
STOP = (0.004, 0.0045, 0.004)  # the stop on the left pole's inner (-y) side
PLATE_T = 0.006  # lock plate thickness: it lies in the seatback's top, z SB_H - PLATE_T .. SB_H
PRESS = 0.0075  # pushing the tab in this far lets the stop pass (the tongue clears it at 6 mm)
TAB_X0, TAB_Y = 0.229, (0.052, 0.068)  # the tab's front face (x) and its sides (y)
TAB_UP = 0.018  # how far the tab stands above the seatback's top
SPRING, BREAK_N = 400.0, 15.0
BOX_X, BOX_Y, BOX_H, BOX_T = (0.14, 0.27), (-0.23, -0.11), 0.05, 0.006
CUSHION_SEAT = (0.12, 0.24, 0.025)  # the seat cushion: x from, x to, height

SEATED_Z = TOP + SB_H + GAP  # the head restraint's origin (its cushion's bottom middle) when seated
HR_START = np.array([POLE_X, 0.0, SEATED_Z])
STOP_Y = POLE_Y - POLE_R - STOP[1] / 2 + 0.0005  # the stop's middle (y, world)
STOP_Z = -POLE_L + STOP[2] / 2 + 0.0005  # the stop's middle, at the left pole's foot (head restraint frame)
TOP_RISE = (SB_H - PLATE_T) - (SB_H + GAP + STOP_Z + STOP[2] / 2)  # the rise at which the stop meets the plate
TONGUE_X = (POLE_X - 0.004, POLE_X + 0.012)  # the lock plate's tongue at rest: over the stop's way up
TONGUE_Y = (POLE_Y - POLE_R - 0.0075, POLE_Y - POLE_R - 0.0008)
ARM_Y = (TONGUE_Y[0] - 0.008, TONGUE_Y[0] + 0.0005)  # the plate's arm from the tongue to the front strip
FRONT_X = (0.2375, 0.2475)  # the plate's front strip, running along y to the tab

FABRIC, DARK, CHROME, RED, CARD = (0.30, 0.31, 0.34), (0.10, 0.10, 0.11), (0.78, 0.79, 0.80), (0.75, 0.12, 0.10), (
    0.66, 0.52, 0.34)


def _seatback_shapes() -> list:
    """The seatback in the frame of the middle of its base: the block with two holes in its top (boxes round them)
    and a seam line (visual)."""
    cx, w = (SB_X0 + SB_X1) / 2, 2 * SB_HW
    hx = POLE_X - cx  # the holes' x in this frame
    out = []
    lo = SB_H - HOLE_D  # the holes' floor
    out.append(BoxShape((0.0, 0.0, lo / 2), (SB_X1 - SB_X0, w, lo), FABRIC))
    h2 = HOLE / 2
    # y bands: -SB_HW .. left hole, between the holes, right hole .. SB_HW; the left (+y) hole widened toward -y
    holes = [(-POLE_Y - h2, -POLE_Y + h2), (POLE_Y - h2 - SLOT, POLE_Y + h2)]
    edges = [-SB_HW, holes[0][0], holes[0][1], holes[1][0], holes[1][1], SB_HW]
    z, hz = lo + HOLE_D / 2, HOLE_D
    for a, b in ((edges[0], edges[1]), (edges[2], edges[3]), (edges[4], edges[5])):
        out.append(BoxShape((0.0, (a + b) / 2, z), (SB_X1 - SB_X0, b - a, hz), FABRIC))
    for a, b in holes:  # in front of and behind each hole
        fx0, fx1 = SB_X0 - cx, hx - h2
        bx0, bx1 = hx + h2, SB_X1 - cx
        out.append(BoxShape(((fx0 + fx1) / 2, (a + b) / 2, z), (fx1 - fx0, b - a, hz), FABRIC))
        out.append(BoxShape(((bx0 + bx1) / 2, (a + b) / 2, z), (bx1 - bx0, b - a, hz), FABRIC))
    out.append(BoxShape((SB_X0 - cx - 0.0004, 0.0, SB_H * 0.55), (0.0008, w - 0.01, 0.002), DARK, collide=False))
    return out


def _cushion_shapes() -> list:
    """The seat cushion in front of the seatback, in the frame of the middle of its base."""
    x0, x1, h = CUSHION_SEAT
    return [BoxShape((0.0, 0.0, h / 2), (x1 - x0, 2 * SB_HW, h), FABRIC),
            BoxShape((0.0, 0.0, h + 0.0004), (x1 - x0 - 0.02, 2 * SB_HW - 0.02, 0.0008), DARK, collide=False)]


def _box_shapes() -> list:
    """The open parts box, in the frame of the middle of its floor: floor and four walls."""
    sx, sy = BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0]
    out = [BoxShape((0.0, 0.0, 0.0015), (sx, sy, 0.003), CARD)]
    for s in (-1.0, 1.0):
        out.append(BoxShape((s * (sx - BOX_T) / 2, 0.0, BOX_H / 2), (BOX_T, sy, BOX_H), CARD))
        out.append(BoxShape((0.0, s * (sy - BOX_T) / 2, BOX_H / 2), (sx - 2 * BOX_T, BOX_T, BOX_H), CARD))
    return out


def _restraint_shapes() -> list:
    """The head restraint in the frame of its cushion's bottom middle: cushion (with a stitched band), two chrome poles
    and the stop on the left pole's inner side."""
    out = [BoxShape((0.0, 0.0, CUSH[2] / 2), CUSH, FABRIC),
           BoxShape((-CUSH[0] / 2 - 0.0004, 0.0, CUSH[2] * 0.6), (0.0008, CUSH[1] - 0.01, 0.003), DARK, collide=False)]
    for s in (-1.0, 1.0):
        out.append(CylShape((0.0, s * POLE_Y, -POLE_L / 2), POLE_R, POLE_L, CHROME))
    out.append(BoxShape((0.0, STOP_Y, STOP_Z), STOP, CHROME))
    return out


def _plate_shapes() -> list:
    """The lock plate in the frame of its tongue's middle: the tongue over the stop's way up, an arm toward the robot on
    the stop's inner side, a strip along the seatback's front edge, and the red tab standing out of the front."""
    cx, cy, top = (TONGUE_X[0] + TONGUE_X[1]) / 2, (TONGUE_Y[0] + TONGUE_Y[1]) / 2, SB_H - PLATE_T / 2

    def box(x0, x1, y0, y1, z0, z1, color=DARK):
        return BoxShape(((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy, (z0 + z1) / 2 - top), (x1 - x0, y1 - y0, z1 - z0), color)

    z0, z1 = SB_H - PLATE_T, SB_H
    return [box(TONGUE_X[0], TONGUE_X[1], TONGUE_Y[0], TONGUE_Y[1], z0, z1),
            box(FRONT_X[0], TONGUE_X[0] + 0.002, ARM_Y[0], ARM_Y[1], z0, z1),
            box(FRONT_X[0], FRONT_X[1], ARM_Y[0], TAB_Y[1], z0, z1),
            box(TAB_X0, FRONT_X[0] + 0.0005, TAB_Y[0], TAB_Y[1], z0, SB_H + TAB_UP, RED)]


PLATE_C = np.array([(TONGUE_X[0] + TONGUE_X[1]) / 2, (TONGUE_Y[0] + TONGUE_Y[1]) / 2, TOP + SB_H - PLATE_T / 2])
TAB_C = np.array([TAB_X0, (TAB_Y[0] + TAB_Y[1]) / 2, TOP + SB_H + TAB_UP / 2])  # the tab's front face, middle
SEATBACK = Body.build("seatback", pos=((SB_X0 + SB_X1) / 2, 0.0, TOP), mass=5.0, color=FABRIC, fixed=True,
                      friction=(0.7, 0.6), shapes=_seatback_shapes())
CUSHION = Body.build("seat_cushion", pos=((CUSHION_SEAT[0] + CUSHION_SEAT[1]) / 2, 0.0, TOP), mass=3.0, color=FABRIC,
                     fixed=True, friction=(0.8, 0.7), shapes=_cushion_shapes())
BOX = Body.build("parts_box", pos=((BOX_X[0] + BOX_X[1]) / 2, (BOX_Y[0] + BOX_Y[1]) / 2, TOP), mass=1.0, color=CARD,
                 fixed=True, friction=(0.7, 0.6), shapes=_box_shapes())
PLATE = Body.build("lock_plate", pos=tuple(float(v) for v in PLATE_C), mass=0.01, color=DARK, friction=(0.3, 0.2),
                   solver_iterations=32, shapes=_plate_shapes(),
                   joint=Joint(anchor=tuple(float(v) for v in PLATE_C), kind="slide", axis="X", parent="seatback",
                               axial=(SPRING, 3.0, 0.015), break_force=BREAK_N))
RESTRAINT = Body.build("head_restraint", pos=tuple(float(v) for v in HR_START), mass=HR_KG, color=FABRIC,
                       friction=(0.7, 0.6), solver_iterations=32, shapes=_restraint_shapes())
OBJECTS = [SEATBACK, CUSHION, BOX, PLATE, RESTRAINT]

# -- scoring ----------------------------------------------------------------------------------------------------------
RAISED, OFF = TOP_RISE - 0.005, POLE_L - GAP + 0.005  # risen this far: as far as it goes; poles out of the holes
FORCED_MM = 2.0  # the plate moved this far off its slide: its joint broke
HELD_R, WITH_MM = 0.06, 2.0
REST_MM, REST_STEPS = 3.0, 3
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("HEADREST_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("HEADREST_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], pos=HR_START.copy(), rise=0.0,
               centre=HR_START.copy(), press=0.0, off_slide=0.0, held=False, offsets={}, hist=[], boxed=False,
               button=TAB_C.copy())
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("HEADREST_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _in_box(p, low: float) -> bool:
    """Its middle inside the box's walls and its lowest point (a pole's foot) below their tops."""
    return (BOX_X[0] + BOX_T < p[0] < BOX_X[1] - BOX_T and BOX_Y[0] + BOX_T < p[1] < BOX_Y[1] - BOX_T
            and low - TOP < BOX_H - 0.01)


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; how far the head restraint has risen, whether a gripper holds it,
    whether it is in the box; how far the tab is pushed in and whether the lock plate has left its slide."""
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
    hp = as_numpy(base.scene["head_restraint"].data.root_link_pose_w)[0]
    pp = as_numpy(base.scene["lock_plate"].data.root_link_pose_w)[0]
    pos, rot = hp[:3] - origin, rotation(hp[3:7])
    plate = pp[:3] - origin
    centre = pos + rot @ np.array([0.0, 0.0, CUSH[2] / 2])
    low = min(float((pos + rot @ np.array([0.0, sy * POLE_Y, -POLE_L]))[2]) for sy in (-1.0, 1.0))
    press = float(plate[0] - PLATE_C[0])  # how far the tab is pushed in (+x)
    off_slide = float(np.hypot(plate[1] - PLATE_C[1], plate[2] - PLATE_C[2]))
    if off_slide * 1000 > FORCED_MM and not _EP["failed"]:
        _EP["failed"] = "the lock was forced: the head restraint was pulled against it until it broke"
    rise = float(pos[2] - SEATED_Z)
    held = False
    for side, tcp in _TCP.items():
        off = centre - tcp
        old = _EP["offsets"].get(side)
        if np.linalg.norm(off) < HELD_R and old is not None and np.linalg.norm(off - old) * 1000 < WITH_MM:
            held = True
        _EP["offsets"][side] = off
    _EP["hist"].append(pos.copy())
    del _EP["hist"][:-REST_STEPS]
    resting = (len(_EP["hist"]) == REST_STEPS
               and max(np.linalg.norm(h - _EP["hist"][-1]) for h in _EP["hist"]) * 1000 < REST_MM)
    boxed = bool(_in_box(centre, low) and resting and not held)
    _EP.update(pos=pos, rise=rise, centre=centre, press=press, off_slide=off_slide, held=held, boxed=boxed,
               button=TAB_C + (plate - PLATE_C))
    out_of_holes = rise >= OFF or _in_box(centre, low)
    stage = 0.75 if boxed else 0.5 if out_of_holes else 0.25 if rise >= RAISED else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["pos"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = 0.0
    if len(_TCP) == 2:
        near = min(min(np.linalg.norm(v - _EP["centre"]), np.linalg.norm(v - _EP["button"])) for v in _TCP.values())
    return dict(still=still, near=float(near), done=bool(_EP["boxed"] and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    c = _EP["centre"]
    print(f"DBG step={_EP['step']} rise={_EP['rise'] * 100:.1f}cm centre=({c[0]:.3f},{c[1]:.3f},{c[2] - TOP:.3f}) "
          f"press={_EP['press'] * 1000:.1f}mm off_slide={_EP['off_slide'] * 1000:.1f}mm held={_EP['held']} "
          f"boxed={_EP['boxed']} near={t['near']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the lock has been forced): 0.25 the head restraint raised
    as far as it goes; 0.5 taken off; 0.75 in the box. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The head restraint in the box, both grippers 6 cm or more from it and from the tab, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the manual says: the right gripper takes the head restraint's cushion and raises it as far as it goes; the left
# gripper, level and fingers closed, comes in front of the release tab and pushes it in with its fingertips; with the
# tab held in, the right gripper pulls the head restraint up out of the holes; the left lets the tab go and moves away;
# the right carries the head restraint over the box, lowers it in until its poles nearly touch the floor and lets go.
# HEADREST_YANK=1 runs it the wrong way: nobody pushes the tab and the right gripper pulls the head restraint
# straight out.

WRONG = os.environ.get("HEADREST_YANK") == "1"
GRIP = grip_rotation(math.pi / 2)  # straight down, fingers closing along x (across the cushion's 4 cm)
PUSH_PITCH = math.radians(float(os.environ.get("HEADREST_PITCH", "10")))
PUSH = np.array([[math.cos(PUSH_PITCH), 0.0, math.sin(PUSH_PITCH)], [0.0, 1.0, 0.0],
                 [-math.sin(PUSH_PITCH), 0.0, math.cos(PUSH_PITCH)]]) @ np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
                                                                    [1.0, 0.0, 0.0]]) @ grip_rotation(0.0)
# PUSH: level, pointing forward (+x) and tipped down PUSH_PITCH, fingers closed: their tips push the tab
OPEN_C, PINCH_C = -0.030, -(CUSH[0] / 2 - 0.003)
CLOSED = 0.0
FINGERTIP = 0.013  # the fingertips stand this far past the gripper's centre, along its pointing direction
PUSH_EXTRA = float(os.environ.get("HEADREST_PUSH_EXTRA", "0.006"))  # past the tab's travel: the arm gives under load


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


TCP0 = HR_START + np.array([0.0, -0.015, CUSH[2] - 0.006])  # the right gripper's centre on the cushion (right of middle)
T_GRIP = 3.0 + 1.2 + 0.5 + 0.6
T_AT_TOP = T_GRIP + 1.2 + 0.4  # raised as far as it goes
T_PRESSED = T_AT_TOP + 0.9  # the left gripper has the tab in
T_OUT = T_PRESSED + 1.4  # pulled out of the holes
BOX_C = np.array([(BOX_X[0] + BOX_X[1]) / 2, (BOX_Y[0] + BOX_Y[1]) / 2, TOP])
CARRY_Z = TOP + 0.15


def _right() -> list:
    out = _lead_in("right", TCP0 + [0.0, 0.0, 0.05], GRIP, OPEN_C, "above the head restraint")
    out += [_w(1.2, TCP0, OPEN_C, "fingers either side of the cushion", GRIP),
            _w(0.5, TCP0, OPEN_C, "settle", GRIP),
            _w(0.6, TCP0, PINCH_C, "take the cushion", GRIP),
            _w(1.2, TCP0 + [0.0, 0.0, TOP_RISE - 0.004], PINCH_C, "raise it nearly as far as it goes", GRIP),
            _w(0.4, TCP0 + [0.0, 0.0, TOP_RISE - 0.004], PINCH_C, "settle", GRIP)]
    out_z = TOP_RISE + 0.022
    out += [_w(T_PRESSED - T_AT_TOP, TCP0 + [0.0, 0.0, TOP_RISE - 0.004], PINCH_C,
               "wait for the tab" if not WRONG else "hold", GRIP),
            _w(1.4, TCP0 + [0.0, 0.0, out_z], PINCH_C,
               "pull it up out of the holes" if not WRONG else "pull it straight out, nobody pushing the tab", GRIP),
            _w(0.6, TCP0 + [0.0, 0.0, out_z + 0.01], PINCH_C, "clear of the seatback", GRIP)]
    hold_at = TCP0 + [0.0, 0.0, out_z + 0.01]
    above = np.array([BOX_C[0], BOX_C[1], CARRY_Z])
    low = np.array([BOX_C[0], BOX_C[1], TOP + 0.006 + POLE_L + CUSH[2] - 0.006])
    out += [_w(1.0, [hold_at[0], hold_at[1], CARRY_Z], PINCH_C, "lift it", GRIP),
            _w(2.0, above, PINCH_C, "carry it over the box", GRIP),
            _w(0.4, above, PINCH_C, "settle", GRIP),
            _w(1.4, low, PINCH_C, "lower it into the box", GRIP),
            _w(0.4, low, PINCH_C, "settle", GRIP),
            _w(0.6, low, OPEN_C, "let go", GRIP),
            _w(1.0, above, OPEN_C, "rise", GRIP),
            _w(1.4, np.array([0.14, -0.26, TOP + 0.20]), OPEN_C, "move away", GRIP),
            _w(3.0, np.array([0.14, -0.26, TOP + 0.20]), OPEN_C, "stay clear", GRIP)]
    return out


def _left() -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    if WRONG:
        pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["left"])
        return [Waypoint(16.0, tuple(float(v) for v in pos), CLOSED, "stay put", rot)]
    tip = FINGERTIP * np.array([math.cos(PUSH_PITCH), 0.0, -math.sin(PUSH_PITCH)])
    touch = np.array([TAB_X0 - 0.003, TAB_C[1], TOP + SB_H + 0.010]) - tip  # fingertips 3 mm off the tab's face
    pressed = touch + [0.003 + PRESS + PUSH_EXTRA, 0.0, 0.0]
    out = _lead_in("left", touch + [0.0, 0.0, 0.06], PUSH, CLOSED, "above the tab, in front of it, fingers closed")
    out += [_w(1.2, touch, CLOSED, "down in front of the tab", PUSH),
            _w(T_AT_TOP - (3.0 + 1.2), touch, CLOSED, "wait for the head restraint to be up", PUSH),
            _w(0.9, pressed, CLOSED, "push the tab in", PUSH),
            _w(T_OUT - T_PRESSED + 0.4, pressed, CLOSED, "hold the tab in while the head restraint comes out", PUSH),
            _w(0.6, touch - [0.01, 0.0, 0.0], CLOSED, "let the tab go", PUSH),
            _w(1.0, touch + [-0.03, 0.02, 0.07], CLOSED, "rise", PUSH),
            _w(1.4, np.array([0.18, 0.22, TOP + 0.16]), CLOSED, "move away", PUSH),
            _w(6.0, np.array([0.18, 0.22, TOP + 0.16]), CLOSED, "stay clear", PUSH)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
