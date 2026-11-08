"""On the station table, a steering tie-rod end whose castle nut has just been torqued onto its stud, and a cotter pin
lying in a holder beside it. The stud has a cross hole for the pin, but none of the nut's six slots lines up with it:
the nearest slot is 25 degrees short of the hole in the loosening direction, the next one 35 degrees away in the
tightening direction.

Task (instruction): line one of the nut's slots up with the hole and put the cotter pin through. The instruction does
not say which way to turn the nut. The rule this task checks is the one manufacturers give for torqued castle nuts on
steering and suspension joints (Meritor's tie-rod end procedure, for example): if no slot lines up at the specified
torque, tighten the nut further to the next slot, never loosen it to line up the hole, because backing it off leaves
the joint under its specified clamp load. Turning the nut 25 degrees back is the shorter move and the wrong one.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- tie-rod end: fixed, forged steel. A steering arm 10 x 5 x 4.5 cm centred at (0.29, -0.07), the tie rod running from it
  to the robot's right, and the stud (22 mm) rising from its top, through a washer, to 7.8 cm above the table, with a
  square 8 mm cross hole along x 6.8 cm above the table.
- castle nut: loose on the stud (it can only turn about it), steel, 32 mm across the flats, 2.6 cm tall: a hex 1.6 cm
  tall and six castellations 1 cm tall above it with slots 7 mm wide or more between them, through the middle of each
  flat. It turns
  with some resistance (damped); it starts 25 degrees clockwise (seen from above) past a slot lining up.
- cotter pin: loose, steel, a 5 mm shank 7.5 cm long with a flat eye at one end, lying along x in a holder at
  (0.21, -0.17), eye toward the robot, its shank 2 cm above the table.

Score: 0 for the rest of the run once the nut has been turned more than 5 degrees in the loosening direction (backed
off). Otherwise the best stage reached: 0.25 the nut turned 10 degrees or more in the tightening direction; 0.5 a slot
lined up with the hole (within 4 degrees) by tightening; 0.75 the pin through both slots and the stud. Full score: the
pin through, a slot lined up, both grippers 5 cm or more from the nut and the pin, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
                     on_step, plate)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
BX, BY = 0.29, -0.07  # the stud's axis
ARM_TOP = 0.045  # the steering arm's top, above the table
WASHER_T = 0.002
NUT_Z0 = ARM_TOP + WASHER_T  # the nut's underside, above the table
HEX_H, CAST_H = 0.016, 0.010
NUT_TOP = NUT_Z0 + HEX_H + CAST_H
STUD_R, STUD_TOP = 0.011, 0.078
HOLE_Z, HOLE_HALF = NUT_Z0 + HEX_H + CAST_H / 2, 0.004  # the cross hole's axis height, half its size
AF, BORE_R = 0.032, 0.0115  # across flats; the nut's (hexagonal) bore
CAST_W = 0.0045  # each castellation is this wide, leaving slots 7 mm wide at the bore and wider outside
START_DEG = -25.0  # the nut's slots start here (0 is a slot lined up with the hole along x)
NUT_KG = 0.06
STEEL_C, FORGED_C, NUT_C, PIN_C = (0.62, 0.64, 0.67), (0.25, 0.26, 0.28), (0.72, 0.70, 0.62), (0.78, 0.76, 0.70)


def _fixture_shapes() -> list:
    """Body frame: origin on the stud's axis at table height."""
    out = [BoxShape((0.0, 0.0, ARM_TOP / 2), (0.10, 0.05, ARM_TOP), FORGED_C),  # steering arm
           CylShape((0.0, -0.075, 0.022), 0.008, 0.10, STEEL_C, axis="Y"),  # tie rod, to the right
           CylShape((0.0, 0.0, ARM_TOP + WASHER_T / 2), 0.0175, WASHER_T, STEEL_C)]  # washer
    lo, hi = HOLE_Z - HOLE_HALF, HOLE_Z + HOLE_HALF
    out.append(CylShape((0.0, 0.0, (ARM_TOP + lo) / 2), STUD_R, lo - ARM_TOP, STEEL_C))  # stud below the hole
    out.append(CylShape((0.0, 0.0, (hi + STUD_TOP) / 2), STUD_R, STUD_TOP - hi, STEEL_C))  # and above it
    side = STUD_R - HOLE_HALF
    for s in (1, -1):  # the stud either side of the square cross hole (along x)
        out.append(BoxShape((0.0, s * (HOLE_HALF + side / 2), HOLE_Z), (2 * STUD_R * 0.9, side, 2 * HOLE_HALF),
                            STEEL_C))
    return out


def _nut_shapes() -> list:
    """Body frame: origin on the stud's axis at table height; the slots at START_DEG + 60k."""
    out = []
    apothem = AF / 2
    side = AF / math.sqrt(3.0)
    for k in range(6):  # the hex: six plates, each a flat, slot k through the middle of flat k
        a = math.radians(START_DEG + 60.0 * k)
        u = np.array([math.cos(a), math.sin(a), 0.0])
        rc = (BORE_R + apothem) / 2
        out.append(BoxShape(tuple(rc * u + [0.0, 0.0, NUT_Z0 + HEX_H / 2]), (apothem - BORE_R, side + 0.002, HEX_H),
                            NUT_C, yaw=a))
    for k in range(6):  # castellations at the corners, between the slots
        a = math.radians(START_DEG + 30.0 + 60.0 * k)
        rc = (BORE_R + apothem) / 2 + 0.001
        u = np.array([math.cos(a), math.sin(a), 0.0])
        out.append(BoxShape(tuple(rc * u + [0.0, 0.0, NUT_Z0 + HEX_H + CAST_H / 2]),
                            (apothem - BORE_R + 0.002, CAST_W, CAST_H), NUT_C, yaw=a))
    return out


FIXTURE = Body.build("tie_rod_end", pos=(BX, BY, TOP), mass=1.0, fixed=True, friction=(0.5, 0.45), color=FORGED_C,
                     contact_offset=0.001, shapes=_fixture_shapes())
NUT = Body.build("castle_nut", pos=(BX, BY, TOP), mass=NUT_KG, friction=(0.6, 0.5), color=NUT_C,
                 contact_offset=0.001, solver_iterations=32, shapes=_nut_shapes(),
                 joint=Joint(anchor=(BX, BY, TOP + NUT_Z0), kind="revolute", axis="Z", parent="tie_rod_end",
                             limits=(-100.0, 100.0), damping=0.4))

PIN_L, PIN_R = 0.075, 0.0025
PIN_START = np.array([0.21, -0.17, TOP + 0.020])  # the shank's middle


def _pin_shapes() -> list:
    """Body frame: x along the shank toward its point, origin at the shank's middle; the eye at -x."""
    return [CylShape((0.0, 0.0, 0.0), PIN_R, PIN_L, PIN_C, axis="X"),
            BoxShape((-PIN_L / 2 - 0.004, 0.0, 0.0), (0.008, 0.004, 0.013), PIN_C)]  # the eye, flat, upright


def _holder_shapes() -> list:
    out = []
    for dx in (-0.008, 0.028):  # two posts with a notch the shank rests in
        out.append(BoxShape((dx, 0.0, (0.020 - PIN_R) / 2), (0.012, 0.016, 0.020 - PIN_R), FORGED_C))
        for s in (1, -1):
            out.append(BoxShape((dx, s * 0.0055, 0.020), (0.012, 0.005, 0.006), FORGED_C))
    return out


PIN = Body.build("cotter_pin", pos=tuple(PIN_START + [0.0, 0.0, 0.0003]), mass=0.012, friction=(0.6, 0.5),
                 color=PIN_C, damping=(0.0, 0.3), contact_offset=0.0008, solver_iterations=32, shapes=_pin_shapes())
HOLDER = Body.build("pin_holder", pos=(PIN_START[0], PIN_START[1], TOP), mass=1.0, fixed=True, friction=(0.5, 0.45),
                    color=FORGED_C, shapes=_holder_shapes())
OBJECTS = [FIXTURE, NUT, HOLDER, PIN]

# -- scoring ----------------------------------------------------------------------------------------------------------
BACKED_DEG, TURNED_DEG, ALIGN_DEG = 5.0, 10.0, 4.0
CLEAR_HANDS = 0.05
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("NUT_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("NUT_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[])
    _TCP.clear()


_reset()


@on_step
def _track_grippers(env) -> None:
    """Each gripper's centre (TCP) from its arm's joint positions: "let go" needs it."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1  # one control step: success() and score() both tally it, the history takes it once
    base = getattr(env, "unwrapped", env)
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = np.asarray(ik.tcp_pose(q[_ARM_IDS[side]], station.BASES[side])[0], float)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _tally(o) -> dict:
    nut, pin = o["castle_nut"], o["cotter_pin"]
    r = nut.rotation()
    turned = math.degrees(math.atan2(r[1, 0], r[0, 0]))  # about +z from the start: positive loosens
    if turned > BACKED_DEG:
        _EP["failed"] = _EP["failed"] or f"the nut was backed off ({turned:.0f} degrees)"
    # A slot lines up when one sits within ALIGN_DEG of the hole's line (along x); slots repeat every 60 degrees.
    off_line = min(abs(_wrap(START_DEG + turned + 60.0 * k)) for k in range(6))
    off_line = min(off_line, min(abs(_wrap(START_DEG + turned + 60.0 * k - 180.0)) for k in range(6)))
    aligned = off_line <= ALIGN_DEG and turned <= -(60.0 + START_DEG) + ALIGN_DEG
    pr = pin.rotation()
    axis = pr[:, 0]
    tip = pin.pos + axis * PIN_L / 2
    eye = pin.pos - axis * PIN_L / 2
    # The pin is through when its shank crosses the stud's axis inside the hole and sticks out beyond the far flat.
    through = False
    if abs(axis[0]) > math.cos(math.radians(12.0)):
        t = (BX - pin.pos[0]) / axis[0]
        at = pin.pos + axis * t
        inside = abs(at[1] - BY) < HOLE_HALF and abs(at[2] - (TOP + HOLE_Z)) < HOLE_HALF
        ends = sorted([tip[0], eye[0]])
        through = inside and ends[0] < BX - AF / 2 - 0.002 and ends[1] > BX + AF / 2 + 0.003
    stage = 0.75 if through else 0.5 if aligned else 0.25 if turned <= -TURNED_DEG else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([pin.pos, [turned / 1000.0]]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    nut_c = np.array([BX, BY, TOP + NUT_Z0 + (HEX_H + CAST_H) / 2])
    gaps = [min(float(np.linalg.norm(v - nut_c)) - AF / 2, float(np.linalg.norm(v - eye))) for v in _TCP.values()]
    hands = min(gaps) if len(_TCP) == 2 else 0.0
    done = through and aligned and hands >= CLEAR_HANDS and still
    return dict(turned=turned, off_line=off_line, aligned=aligned, through=through, tip=tip, stage=stage,
                still=still, hands=hands, done=bool(done))


def _debug(o, t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p = o["cotter_pin"].pos
    print(f"DBG step={_EP['step']} nut={t['turned']:+.1f}deg off_line={t['off_line']:.1f} aligned={t['aligned']} "
          f"pin=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.4f}) tip_x={t['tip'][0]:.3f} through={t['through']} "
          f"hands={t['hands']:.3f} still={t['still']} stage={t['stage']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the nut has been turned more than 5 degrees in the
    loosening direction): 0.25 the nut turned 10 degrees or more in the tightening direction; 0.5 a slot lined up with
    the hole by tightening; 0.75 the pin through both slots and the stud. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(objects, t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The pin through, a slot lined up by tightening, both grippers 5 cm or more from the nut and the pin's eye, still
    for 1 s, the nut never backed off."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right gripper comes down on the nut with its fingers on two opposite flats, closes, and turns the nut 35 degrees
# clockwise (tightening) to the next slot, lets go and rises. It then takes the cotter pin by its shank just in front
# of the eye, lifts it out of the holder, lines it up with the hole and pushes it straight through the slot, the stud
# and the slot opposite, lets go and moves away. The left arm stays where it is. NUT_BACK_OFF=1 turns the nut 25 degrees
# back to the nearest slot instead (the wrong way), and the run scores 0.

OPEN = -0.030
NUT_PINCH = -(AF / 2 - 0.003)
PIN_PINCH = -(PIN_R - 0.0015)
SAG = float(os.environ.get("NUT_SAG", "0.0015"))
GRIP_BACK = 0.0095  # the pin is held this far in front of the shank's back end (the eye is behind it)
TURN_DEG = 25.0 if os.environ.get("NUT_BACK_OFF") == "1" else -(60.0 + START_DEG)  # +25 loosens, -35 tightens


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
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
    """From READY along a straight line in joint space (the wrist's big turn stays on one IK branch)."""
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


def _idle(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), OPEN, "stay put", rot)]


def _right() -> list[Waypoint]:
    # Fingers on the flats at START_DEG + 120 and + 300: closing along that line.
    yaw0 = math.radians(START_DEG + 120.0 - 90.0)
    on_nut = np.array([BX, BY, TOP + NUT_Z0 + 0.015])
    turned = [grip_rotation(yaw0 + math.radians(TURN_DEG) * f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    out = _lead_in("right", on_nut + [0.0, 0.0, 0.06], turned[0], OPEN, "above the nut")
    out += [_w(1.5, on_nut, OPEN, "down, fingers either side of two flats", turned[0]),
            _w(1.0, on_nut, NUT_PINCH, "grip the nut", turned[0])]
    for k in range(1, 5):
        out.append(_w(0.8, on_nut, NUT_PINCH, "turn it clockwise, tightening, to the next slot", turned[k]))
    out += [_w(0.8, on_nut, OPEN, "let go", turned[-1]),
            _w(1.2, on_nut + [0.0, 0.0, 0.06], OPEN, "rise", turned[-1])]
    if os.environ.get("NUT_BACK_OFF") == "1":
        return out + [_w(3.0, on_nut + [0.0, 0.0, 0.06], OPEN, "wait", turned[-1])]
    down = grip_rotation(0.0)  # fingers closing along y, across the shank
    grip = PIN_START + [-PIN_L / 2 + GRIP_BACK, 0.0, SAG]
    line_z = TOP + HOLE_Z + SAG
    tip_x = BX - AF / 2 - 0.015  # the point 15 mm short of the nut
    start = np.array([tip_x - PIN_L + GRIP_BACK, BY, line_z])
    home = start + [AF + 0.015 + 0.015, 0.0, 0.0]  # the point 15 mm past the far flat
    out += [_w(2.5, grip + [0.0, 0.0, 0.05], OPEN, "above the pin, just in front of its eye", down),
            _w(1.5, grip, OPEN, "down either side of the shank", down),
            _w(1.0, grip, PIN_PINCH, "pick the pin up", down),
            _w(1.5, grip + [0.0, 0.0, 0.04], PIN_PINCH, "lift it out of the holder", down),
            _w(2.5, start + [0.0, 0.0, 0.03], PIN_PINCH, "carry it over in line with the hole", down),
            _w(1.5, start, PIN_PINCH, "down to the hole's height, point toward the slot", down),
            _w(0.6, start, PIN_PINCH, "steady", down),
            _w(3.0, home, PIN_PINCH, "push it through the slot, the stud and the far slot", down),
            _w(0.8, home, OPEN, "let go", down),
            _w(1.5, home + [-0.04, 0.0, 0.07], OPEN, "move away", down),
            _w(3.0, home + [-0.04, 0.0, 0.07], OPEN, "wait", down)]
    return out


def demo():
    right = _right()
    return {"left": _idle("left", sum(w.duration for w in right)), "right": right}
