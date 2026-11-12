"""On the station table: a grand piano, scaled down, with its lid closed. The lid is hinged along the piano's far side,
so it opens by lifting its front edge (a small grip block on the lid stands for the edge a hand lifts). The lid prop lies
folded down on its bracket at the front of the case, pointing at the robot; the prop cup hangs under the lid.

Task (instruction): open the piano lid. The instruction does not say how. The procedure the task checks is the piano
maker's (Yamaha, Grand Piano / Upright Piano Owner's Manual, "Opening/closing the lid of the Grand Piano"): "While
supporting the lifted two stacked Lids with your right hand, lift the lid prop with your left hand and then insert the
end into the hole on the main lid." The manual's warning: "Open and close the lid carefully as described in the
operation procedure. Otherwise, the lid could fall down and injure your hands and fingers." One hand cannot do it:
the lid falls as soon as it is let go before its prop stands in the cup.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- piano case: fixed, x 0.21 to 0.42, y -0.13 to 0.13, 4 cm tall, the keyboard along its right end (y -0.165 to -0.13).
- lid: 21 x 26 cm, 8 mm thick, 0.35 kg, hinged about the y axis along its far edge (x 0.42); the grip block (2.5 cm
  long, 8 mm thick, 2 cm tall) stands on it at its front edge at y -0.07; the prop cup hangs under it 19.7 cm from the
  hinge at y 0.06.
- prop: 7.3 cm long, 8 mm square, on a hinge about the y axis at (0.205, 0.06), 6 cm above the table, in front of the
  case; folded down it points at the robot. Raised 119 degrees (leaning back 29 degrees from upright), its end stands
  in the cup and holds the lid open at 25 degrees.

The lid moves with a gripper while its grip block is within 3.5 cm of the gripper's centre and keeps the same offset
from it (to 2 mm a step). The prop's end is in the cup when it is within 7 mm of the cup's centre. Score: 0 for the
rest of the run once the lid has fallen: after being raised 8 degrees or more, it dropped 6 degrees or more within 0.5 s
while no gripper held it. Otherwise the best stage reached: 0.25 the lid raised 15 degrees or more; 0.5 the prop's end
in the cup; 0.75 the lid resting on its prop (the prop's end in the cup, no gripper holding the lid, the lid at rest).
Full score: that, with both grippers 6 cm or more from the grip block, the cup and the prop, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Joint, SphereShape, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
                     on_step, rotation)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
CASE_X0, CASE_X1, CASE_HW, CASE_H = 0.21, 0.42, 0.13, 0.04  # the case's front and back (x), half width (y), height
CASE_C = np.array([(CASE_X0 + CASE_X1) / 2, 0.0, TOP])  # the case body's origin: the middle of its base
LID_D, LID_T, LID_KG = CASE_X1 - CASE_X0, 0.008, 0.35  # from the hinge to the front edge, thickness, mass
HINGE = np.array([CASE_X1, 0.0, TOP + CASE_H + LID_T / 2])  # the lid's hinge axis runs along y through this point
GRIP_Y, GRIP_L, GRIP_T, GRIP_H = -0.07, 0.025, 0.008, 0.02  # the grip block: y, length (x), thickness (y), height
GRIP_D = LID_D - GRIP_L / 2 - 0.002  # its middle's distance from the hinge
PROP_Y, PROP_W, PROP_KG = 0.06, 0.008, 0.03
R_G, KNOB_R = 0.02, 0.006  # a round knob on the prop, this far from its hinge, where a hand takes it
PIVOT = np.array([0.205, PROP_Y, TOP + 0.06])  # the prop's hinge (about y), in front of the case
SET_DEG = 25.0  # the lid's angle when it rests on its prop
CUP_D = 0.197  # the cup's distance from the hinge, along the lid
CUP_IN, CUP_WALL, CUP_DEPTH = 0.018, 0.004, 0.006  # the cup's inside (square), wall thickness, depth
NOTCH_D = 0.009  # the notch in the case's top edge that the closed lid's cup sits in


def _ry(a: float) -> np.ndarray:
    """Rotation by ``a`` (rad) about the world y axis (positive raises a lid that reaches out along -x)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


CUP_LOCAL = np.array([-CUP_D, PROP_Y, -LID_T / 2])  # the cup's centre on the lid's underside, from the hinge (closed)
GRIP_LOCAL = np.array([-GRIP_D, GRIP_Y, LID_T / 2 + GRIP_H - 0.004])  # where a gripper's centre takes the grip block
_CUP_SET = HINGE + _ry(math.radians(SET_DEG)) @ CUP_LOCAL  # the cup's centre with the lid on its prop
PROP_L = float(np.linalg.norm(_CUP_SET - PIVOT))  # the prop: from its hinge to its end
_U_SET = (_CUP_SET - PIVOT) / PROP_L
PROP_SET_DEG = math.degrees(math.atan2(_U_SET[2], -_U_SET[0]))  # its angle then (0 folded down, 90 upright)

BLACK, EBONY, GOLD, BRASS = (0.03, 0.03, 0.035), (0.08, 0.075, 0.08), (0.78, 0.62, 0.28), (0.80, 0.66, 0.32)
IVORY, STRING, FELT = (0.95, 0.94, 0.90), (0.86, 0.86, 0.82), (0.55, 0.10, 0.12)


def _case_shapes() -> list:
    """The case in the frame of the middle of its base: the rim (the collision box) and, inside it, under the lid, the
    gold iron frame and the strings (visual); the keyboard along its right end (keybed, white and black keys); the prop's
    bracket on its front face (visual)."""
    w = 2 * CASE_HW
    nx0, nx1 = -LID_D / 2, CUP_LOCAL[0] + CASE_X1 - CASE_C[0] + CUP_IN / 2 + CUP_WALL + 0.008  # the notch, x (local)
    ny0, ny1 = PROP_Y - CUP_IN / 2 - CUP_WALL - 0.008, PROP_Y + CUP_IN / 2 + CUP_WALL + 0.008  # and y
    lo = CASE_H - NOTCH_D
    out = [BoxShape((0.0, 0.0, lo / 2), (LID_D, w, lo), BLACK),  # the rim below the notch's floor, then round it:
           BoxShape(((nx1 + LID_D / 2) / 2, 0.0, lo + NOTCH_D / 2), (LID_D / 2 - nx1, w, NOTCH_D), BLACK),
           BoxShape(((nx0 + nx1) / 2, (-CASE_HW + ny0) / 2, lo + NOTCH_D / 2), (nx1 - nx0, ny0 + CASE_HW, NOTCH_D), BLACK),
           BoxShape(((nx0 + nx1) / 2, (ny1 + CASE_HW) / 2, lo + NOTCH_D / 2), (nx1 - nx0, CASE_HW - ny1, NOTCH_D), BLACK),
           BoxShape((0.003, 0.0, CASE_H + 0.0003), (LID_D - 0.03, w - 0.03, 0.0006), GOLD, collide=False)]
    for k in range(13):  # strings, along y
        out.append(BoxShape((-0.08 + k * 0.0135, 0.004, CASE_H + 0.0009), (0.0012, w - 0.05, 0.0005), STRING,
                            collide=False))
    kb_y, kb_w = -CASE_HW - 0.0175, 0.035
    out.append(BoxShape((0.0, kb_y, 0.012), (LID_D - 0.01, kb_w, 0.024), BLACK))  # keybed
    out.append(BoxShape((0.0, kb_y - 0.0015, 0.0245), (LID_D - 0.03, kb_w - 0.006, 0.001), IVORY, collide=False))
    n_white, x0 = 16, -(LID_D - 0.03) / 2
    step = (LID_D - 0.03) / n_white
    for k in range(1, n_white):  # gaps between the white keys
        out.append(BoxShape((x0 + k * step, kb_y - 0.0015, 0.0252), (0.0007, kb_w - 0.006, 0.0004), EBONY,
                            collide=False))
    for k in range(n_white - 1):  # black keys: two, then three, per octave of seven white keys
        if k % 7 in (2, 6):
            continue
        out.append(BoxShape((x0 + (k + 1) * step, kb_y + 0.005, 0.0262), (0.0065, 0.017, 0.0026), EBONY,
                            collide=False))
    out.append(BoxShape((PIVOT[0] - CASE_C[0], PROP_Y, 0.0395), (0.008, 0.008, 0.029), EBONY, collide=False))  # bracket
    out.append(CylShape((PIVOT[0] - CASE_C[0], PROP_Y, PIVOT[2] - TOP), 0.0035, 0.012, BRASS, collide=False, axis="Y"))
    return out


def _lid_shapes() -> list:
    """The lid in the frame of its hinge line (closed): the board, the grip block at its front edge and the brass cup
    under it (four walls round a square hole)."""
    out = [BoxShape((-LID_D / 2, 0.0, 0.0), (LID_D, 2 * CASE_HW, LID_T), BLACK),
           BoxShape((-LID_D + 0.004, 0.0, LID_T / 2 + 0.0003), (0.006, 2 * CASE_HW - 0.01, 0.0006), EBONY,
                    collide=False),
           BoxShape((-GRIP_D, GRIP_Y, LID_T / 2 + GRIP_H / 2), (GRIP_L, GRIP_T, GRIP_H), BLACK)]
    cx, cy, cz = CUP_LOCAL
    off = CUP_IN / 2 + CUP_WALL / 2
    z = cz - CUP_DEPTH / 2
    for s in (-1.0, 1.0):
        out.append(BoxShape((cx + s * off, cy, z), (CUP_WALL, CUP_IN + 2 * CUP_WALL, CUP_DEPTH), BRASS))
        out.append(BoxShape((cx, cy + s * off, z), (CUP_IN, CUP_WALL, CUP_DEPTH), BRASS))
    out.append(BoxShape((cx, cy, cz - 0.0003), (CUP_IN, CUP_IN, 0.0006), FELT, collide=False))
    return out


def _prop_shapes() -> list:
    """The prop in the frame of its hinge, folded down (pointing at the robot, along -x): the stick, its brass end cap."""
    return [BoxShape((-PROP_L / 2, 0.0, 0.0), (PROP_L, PROP_W, PROP_W), BLACK),
            BoxShape((-PROP_L + 0.003, 0.0, 0.0), (0.006, PROP_W + 0.0006, PROP_W + 0.0006), BRASS, collide=False),
            SphereShape((-R_G, 0.0, 0.0), KNOB_R, BLACK)]


CASE = Body.build("piano_case", pos=tuple(float(v) for v in CASE_C), mass=40.0, color=BLACK, fixed=True,
                  friction=(0.6, 0.5), shapes=_case_shapes())
LID = Body.build("piano_lid", pos=tuple(float(v) for v in HINGE), mass=LID_KG, color=BLACK, friction=(0.6, 0.5),
                 damping=(0.0, 0.05), solver_iterations=32, shapes=_lid_shapes(),
                 joint=Joint(anchor=tuple(float(v) for v in HINGE), kind="revolute", axis="Y", parent=None,
                             limits=(-2.0, 70.0), damping=0.01))
PROP = Body.build("lid_prop", pos=tuple(float(v) for v in PIVOT), mass=PROP_KG, color=BLACK, friction=(0.6, 0.5),
                  solver_iterations=32, shapes=_prop_shapes(),
                  joint=Joint(anchor=tuple(float(v) for v in PIVOT), kind="revolute", axis="Y", parent="piano_case",
                              limits=(-1.0, 135.0), damping=0.002))
OBJECTS = [CASE, LID, PROP]

# -- scoring ----------------------------------------------------------------------------------------------------------
RAISED_DEG, FELL_FROM_DEG, FALL_DEG, FALL_STEPS = 15.0, 8.0, 6.0, 10
IN_CUP = 0.007
HELD_R = 0.035  # the lid moves with a gripper when its grip block is this close to the gripper's centre ...
WITH_MM = 2.0  # ... and its offset from the gripper's centre changes by less than this from one step to the next
REST_DEG, REST_STEPS = 0.3, 3
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("PIANO_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("PIANO_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], lid=0.0, prop=0.0, top=0.0, hist=[],
               held=False, offsets={}, tip=PIVOT + PROP_L * np.array([-1.0, 0.0, 0.0]), cup=HINGE + CUP_LOCAL,
               grip=HINGE + GRIP_LOCAL, in_cup=False, propped=False)
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("PIANO_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _angle(rot) -> float:
    """How far a part that starts reaching out along -x has turned up about y (degrees)."""
    v = rot @ np.array([-1.0, 0.0, 0.0])
    return math.degrees(math.atan2(v[2], -v[0]))


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the lid's and the prop's angles, the prop's end and the cup; whether
    the lid moves with a gripper, has fallen, rests on its prop."""
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
    lp = as_numpy(base.scene["piano_lid"].data.root_link_pose_w)[0]
    pp = as_numpy(base.scene["lid_prop"].data.root_link_pose_w)[0]
    l_pos, l_rot = lp[:3] - origin, rotation(lp[3:7])
    p_pos, p_rot = pp[:3] - origin, rotation(pp[3:7])
    lid, prop = _angle(l_rot), _angle(p_rot)
    tip = p_pos + PROP_L * (p_rot @ np.array([-1.0, 0.0, 0.0]))
    cup = l_pos + l_rot @ CUP_LOCAL
    grip = l_pos + l_rot @ GRIP_LOCAL
    held = False  # the lid moving with a gripper: its grip block close to the gripper's centre, keeping its offset
    for side, tcp in _TCP.items():
        off = grip - tcp
        old = _EP["offsets"].get(side)
        if np.linalg.norm(off) < HELD_R and old is not None and np.linalg.norm(off - old) * 1000 < WITH_MM:
            held = True
        _EP["offsets"][side] = off
    _EP["hist"].append(lid)
    del _EP["hist"][:-FALL_STEPS]
    _EP["top"] = max(_EP["top"], lid)
    if (not held and not _EP["failed"] and _EP["top"] >= FELL_FROM_DEG
            and max(_EP["hist"]) - lid >= FALL_DEG):
        _EP["failed"] = f"the lid fell from {max(_EP['hist']):.0f} to {lid:.0f} degrees with no gripper holding it"
    in_cup = bool(np.linalg.norm(tip - cup) < IN_CUP)
    rest = _EP["hist"][-REST_STEPS:]
    resting = len(rest) == REST_STEPS and max(rest) - min(rest) < REST_DEG
    propped = bool(in_cup and not held and resting and lid >= RAISED_DEG)
    _EP.update(lid=lid, prop=prop, tip=tip, cup=cup, grip=grip, held=held, in_cup=in_cup, propped=propped)
    stage = 0.75 if propped else 0.5 if in_cup else 0.25 if _EP["top"] >= RAISED_DEG else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([_EP["grip"], _EP["tip"]]).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = 0.0
    if len(_TCP) == 2:
        near = min(min(np.linalg.norm(v - _EP["grip"]), np.linalg.norm(v - _EP["cup"]),
                       _seg_dist(v, PIVOT, _EP["tip"])) for v in _TCP.values())
    return dict(still=still, near=float(near), done=bool(_EP["propped"] and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    tip, cup = _EP["tip"], _EP["cup"]
    print(f"DBG step={_EP['step']} lid={_EP['lid']:.1f} top={_EP['top']:.1f} prop={_EP['prop']:.1f} "
          f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2] - TOP:.3f}) cup=({cup[0]:.3f},{cup[1]:.3f},{cup[2] - TOP:.3f}) "
          f"gap={np.linalg.norm(tip - cup) * 1000:.0f}mm held={_EP['held']} in_cup={_EP['in_cup']} "
          f"propped={_EP['propped']} near={t['near']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the lid has fallen): 0.25 the lid raised 15 degrees;
    0.5 the prop's end in the cup; 0.75 the lid resting on its prop. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The lid resting on its prop, the prop's end in the cup, both grippers 6 cm or more from the grip block, the cup
    and the prop, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the manual says: the right gripper takes the lid's grip block from above and raises the lid to 30 degrees, the
# block turning between its fingers, and holds it there; meanwhile the left gripper comes down level from above, takes
# the folded prop near its hinge and swings it up until its end stands under the cup; the right gripper then lowers the
# lid onto the prop's end; the left gripper lets go of the prop and backs out, and only then does the right gripper let
# go of the lid and rise. PIANO_DROP=1 runs it the wrong way: the right gripper raises the lid and lets go of it without
# the prop; the lid falls.

WRONG = os.environ.get("PIANO_DROP") == "1"
DOWN = grip_rotation(0.0)  # straight down, fingers closing along y
SIDE = _ry(math.radians(float(os.environ.get("PIANO_PITCH", "20")))) @ _ry(-math.pi / 2) @ grip_rotation(0.0)
# SIDE: pointing forward (+x), fingers closing along y; PIANO_PITCH tips it down (degrees)
OPEN, PINCH, PINCH_KNOB = -0.015, -0.001, -0.003  # finger joint targets round the 8 mm grip block, the 12 mm knob
UP_DEG = 30.0  # the lid is held here while the prop goes in under the cup


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


def _grip_at(deg: float) -> np.ndarray:
    return HINGE + _ry(math.radians(deg)) @ GRIP_LOCAL


def _prop_grip(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return PIVOT + R_G * np.array([-math.cos(a), 0.0, math.sin(a)])


T_LID_UP = 3.0 + 1.2 + 0.5 + 0.6 + 6 * 0.6 + 0.5  # when the right gripper has the lid at UP_DEG
SWING = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, PROP_SET_DEG)
T_PROP_UP = T_LID_UP + len(SWING) * 0.5 + 0.6  # when the prop's end stands under the cup
T_LID_DOWN = T_PROP_UP + 4 * 0.5 + 0.5  # when the lid rests on the prop's end
T_LEFT_GONE = T_LID_DOWN + 0.6 + 1.0 + 1.4  # when the left gripper has let go of the prop and moved away


def _right() -> list:
    g0 = _grip_at(0.0)
    out = _lead_in("right", g0 + [0.0, 0.0, 0.06], DOWN, OPEN, "above the lid's grip block")
    out += [_w(1.2, g0, OPEN, "fingers either side of the grip block", DOWN),
            _w(0.5, g0, OPEN, "settle", DOWN),
            _w(0.6, g0, PINCH, "take the grip block", DOWN)]
    for deg in (5.0, 10.0, 15.0, 20.0, 25.0, UP_DEG):
        out.append(_w(0.6, _grip_at(deg), PINCH, f"raise the lid ({deg:.0f} degrees)", DOWN))
    out.append(_w(0.5, _grip_at(UP_DEG), PINCH, "settle", DOWN))
    if WRONG:
        g = _grip_at(UP_DEG)
        return out + [_w(1.0, g, PINCH, "hold it up", DOWN),
                      _w(0.6, g, OPEN, "let go of the lid without the prop", DOWN),
                      _w(1.0, g + [-0.06, 0.0, 0.04], OPEN, "move away", DOWN),
                      _w(3.0, g + [-0.06, 0.0, 0.04], OPEN, "stay clear", DOWN)]
    out.append(_w(T_PROP_UP - T_LID_UP, _grip_at(UP_DEG), PINCH, "hold the lid up while the prop goes in", DOWN))
    for deg in (28.0, 26.5, 25.5, SET_DEG + 0.3):
        out.append(_w(0.5, _grip_at(deg), PINCH, f"lower the lid onto the prop ({deg:.1f} degrees)", DOWN))
    g = _grip_at(SET_DEG + 0.3)
    out += [_w(0.5, g, PINCH, "settle", DOWN),
            _w(T_LEFT_GONE - T_LID_DOWN, g, PINCH, "keep holding it while the left gripper lets go of the prop", DOWN),
            _w(0.6, g, OPEN, "let go of the lid", DOWN),
            _w(1.0, g + [0.0, 0.0, 0.07], OPEN, "rise", DOWN),
            _w(1.4, np.array([0.12, -0.20, TOP + 0.20]), OPEN, "move away", DOWN),
            _w(2.0, np.array([0.12, -0.20, TOP + 0.20]), OPEN, "stay clear", DOWN)]
    return out


def _left() -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    if WRONG:
        pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["left"])
        return [Waypoint(12.0, tuple(float(v) for v in pos), OPEN, "stay put", rot)]
    p0 = _prop_grip(0.0)
    out = _lead_in("left", p0 + [0.0, 0.0, 0.05], SIDE, OPEN, "above the folded prop, level")
    out += [_w(1.2, p0, OPEN, "down, fingers either side of the prop", SIDE),
            _w(0.5, p0, OPEN, "settle", SIDE),
            _w(0.6, p0, PINCH_KNOB, "take the prop by its knob", SIDE),
            _w(T_LID_UP - (3.0 + 1.2 + 0.5 + 0.6), p0, PINCH_KNOB, "wait for the lid to be up", SIDE)]
    for deg in SWING:
        out.append(_w(0.5, _prop_grip(deg), PINCH_KNOB, f"swing the prop up ({deg:.0f} degrees)", SIDE))
    p = _prop_grip(PROP_SET_DEG)
    out += [_w(0.6, p, PINCH_KNOB, "settle, its end under the cup", SIDE),
            _w(T_LID_DOWN - T_PROP_UP, p, PINCH_KNOB, "hold it while the lid comes down onto it", SIDE),
            _w(0.6, p, OPEN, "let go of the prop", SIDE),
            _w(1.0, p + [-0.04, 0.0, 0.0], OPEN, "back out", SIDE),
            _w(1.4, p + [-0.04, 0.05, 0.03], OPEN, "move away", SIDE),
            _w(4.0, p + [-0.04, 0.05, 0.03], OPEN, "stay clear", SIDE)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
