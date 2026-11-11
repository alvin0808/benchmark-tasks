"""On the station table, which stands for a restaurant kitchen's pass: a T-bone steak just off the grill on a wooden
board, and an instant-read probe thermometer standing in its holder.

Task (instruction): check the steak's temperature with the thermometer. The instruction does not say where. The rule
this task checks is the one every food-safety course teaches about probe thermometers (Clemson Cooperative Extension,
Home & Garden Information Center, HGIC 3587, Food Thermometers: A Key to Food Safety: "The food thermometer should be
placed in the thickest part of the food and should not be touching bone, fat or gristle."). Bone and fat heat
differently from the meat, so a probe against them reads wrong, and the thickest part is the last to cook. A T-bone's
bone runs through the middle of the steak, with the thick strip loin on one side and the thin tenderloin on the other:
the quick way, the probe pushed in near the middle, meets the bone; the probe's tip in the middle of the strip, clear
of bone and fat, keeps the rule.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- board: fixed at (0.22, 0.02), 20 x 22 cm, 1.5 cm thick.
- steak (shown only: the probe passes into it): on the board, centred at (0.22, 0.02). The bone's crossbar runs along y
  at its far edge (x 0.275); its stem runs along x down the middle (y 0.02, 0.8 cm wide). The strip loin lies on the
  +y side (3.6 cm thick in the middle, a fat cap along its outer edge), the tenderloin on the -y side (2.2 cm thick).
- thermometer: a loose body (40 g): a head (2.6 x 2.2 x 3.6 cm) on a probe 5 cm long and 4 mm across; it stands in a
  holder at (0.18, -0.20), its head resting on the holder's rim.

The probe's tip is in the meat when it is inside the steak's outline and below its top. A reading is taken when the tip
has stayed in the meat, moving less than 0.5 mm a step, for 1.5 s; it counts when the tip is in the strip's thick middle
(within 1.8 cm of its centre and 1 to 2.6 cm above the board) for 3 s. Score: 0 for the rest of the run once a reading
is taken with the probe within 6 mm of the bone or the fat, or with its tip outside the strip's thick middle. Otherwise
the best stage reached: 0.25 the thermometer lifted out of its holder; 0.5 its tip in the meat; 0.75 a reading in the
strip's thick middle. Full score: that, and the thermometer standing back in its holder, both grippers 6 cm or more from
it, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
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
BOARD_XY, BOARD_SIZE, BOARD_T = np.array([0.22, 0.02]), (0.20, 0.22), 0.015
# The steak, in the frame of the board's top under its centre: boxes (x0, x1, y0, y1, height).
STRIP = (-0.055, 0.049, 0.004, 0.070, 0.028)
STRIP_TOP = (-0.032, 0.030, 0.014, 0.060, 0.036)  # the strip's thicker middle (its top at 3.6 cm)
FAT = (-0.050, 0.049, 0.070, 0.080, 0.024)
TENDER = (-0.035, 0.049, -0.050, -0.004, 0.022)
BONE_CROSS = (0.049, 0.061, -0.055, 0.080, 0.032)
BONE_STEM = (-0.045, 0.049, -0.004, 0.004, 0.030)
CORE_C, CORE_R, CORE_Z = np.array([0.0, 0.037]), 0.018, (0.010, 0.026)  # the strip's thick middle
HOLDER_XY = np.array([0.18, -0.20])
HOLDER_R_IN, HOLDER_R_OUT, HOLDER_H = 0.009, 0.016, 0.060
HEAD = (0.026, 0.022, 0.036)
PROBE_L, PROBE_R = 0.050, 0.002

MEAT, SEAR, FATC, BONE, WOOD, STEEL, FACE, DIAL, GREEN, GREYL = (
    (0.47, 0.28, 0.18), (0.20, 0.11, 0.07), (0.93, 0.88, 0.74), (0.96, 0.94, 0.88), (0.62, 0.44, 0.26),
    (0.78, 0.79, 0.81), (0.95, 0.95, 0.95), (0.15, 0.15, 0.16), (0.15, 0.85, 0.30), (0.45, 0.45, 0.47))
HIDE = np.array([0.22, 0.0, TOP - 0.30])


def _box(b, color, collide=False):
    x0, x1, y0, y1, h = b
    return BoxShape(((x0 + x1) / 2, (y0 + y1) / 2, h / 2), (x1 - x0, y1 - y0, h), color, collide=collide)


def _board_shapes() -> list:
    """The board (it collides) and the steak on it (shown only), in the frame of the board's centre on the table."""
    out = [BoxShape((0.0, 0.0, BOARD_T / 2), (BOARD_SIZE[0], BOARD_SIZE[1], BOARD_T), WOOD)]
    meat = [_box(STRIP, MEAT), _box(STRIP_TOP, MEAT), _box(FAT, FATC), _box(TENDER, MEAT), _box(BONE_CROSS, BONE),
            _box(BONE_STEM, BONE)]
    # grill marks on the strip and the tenderloin (shown only)
    for dx in (-0.03, -0.005, 0.02):
        meat.append(BoxShape((dx, 0.037, STRIP_TOP[4] + 0.0004), (0.006, 0.045, 0.0008), SEAR, collide=False))
        meat.append(BoxShape((dx + 0.005, -0.027, TENDER[4] + 0.0004), (0.006, 0.040, 0.0008), SEAR, collide=False))
    for s in meat:
        c = s.centre
        out.append(BoxShape((c[0], c[1], c[2] + BOARD_T), s.size, s.color, collide=False))
    return out


def _holder_shapes() -> list:
    """A holder: a base and a tube the probe stands in (16 panels), in the frame of its centre on the table."""
    out = [CylShape((0.0, 0.0, 0.002), HOLDER_R_OUT + 0.006, 0.004, STEEL)]
    n = 16
    width = 2 * HOLDER_R_OUT * math.tan(math.pi / n) + 0.0005
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        r = (HOLDER_R_IN + HOLDER_R_OUT) / 2
        h = math.radians(math.degrees(a)) / 2
        out.append(BoxShape((r * math.cos(a), r * math.sin(a), 0.004 + (HOLDER_H - 0.004) / 2),
                            (HOLDER_R_OUT - HOLDER_R_IN, width, HOLDER_H - 0.004), DIAL,
                            quat_xyzw=(0.0, 0.0, math.sin(h), math.cos(h))))
    return out


def _thermo_shapes() -> list:
    """The thermometer in the frame of its probe's tip: the probe up to the head, the head, its dial (shown only)."""
    hx, hy, hz = HEAD
    return [CylShape((0.0, 0.0, PROBE_L / 2), PROBE_R, PROBE_L, STEEL),
            BoxShape((0.0, 0.0, PROBE_L + hz / 2), HEAD, DIAL),
            CylShape((-hx / 2 - 0.0005, 0.0, PROBE_L + hz * 0.55), 0.009, 0.001, FACE, collide=False, axis="X")]


BOARD = Body.build("board_and_steak", pos=(float(BOARD_XY[0]), float(BOARD_XY[1]), TOP), mass=1.0, color=WOOD,
                   fixed=True, friction=(0.7, 0.6), shapes=_board_shapes())
HOLDER = Body.build("thermometer_holder", pos=(float(HOLDER_XY[0]), float(HOLDER_XY[1]), TOP), mass=0.3, color=DIAL,
                    fixed=True, shapes=_holder_shapes())
THERMO_Z0 = TOP + HOLDER_H + 0.001 - PROBE_L  # its tip's height when its head rests on the holder's rim
THERMO = Body.build("thermometer", pos=(float(HOLDER_XY[0]), float(HOLDER_XY[1]), THERMO_Z0), mass=0.04, color=DIAL,
                    friction=(0.8, 0.7), damping=(0.05, 0.5), solver_iterations=24, shapes=_thermo_shapes())
LIGHT = Body.build("reading_light", pos=tuple(HIDE), mass=0.01, color=GREEN, gravity=False, damping=(5.0, 5.0),
                   shapes=[SphereShape((0.0, 0.0, 0.0), 0.004, GREEN, collide=False)])
OBJECTS = [BOARD, HOLDER, THERMO, LIGHT]

# -- scoring ----------------------------------------------------------------------------------------------------------
TOUCH = 0.006  # the probe this close to bone or fat touches it
STILL_PROBE_MM, ATTEMPT_S, READ_S = 0.5, 1.5, 3.0
OUT_OF_HOLDER = 0.010  # lifted: the tip this far above where it stood
HOME_TOL = 0.006
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("STEAK_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("STEAK_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], tip=np.zeros(3), up=np.array([0, 0, 1.0]),
               prev_tip=None, still_s=0.0, in_meat=False, touch="", core=False, read=False, lifted=False, home=False)
    _TCP.clear()


_reset()


def _place(base, name: str, pos) -> None:
    import torch

    origin = as_numpy(base.scene.env_origins[0])
    pose = np.array([[pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2], 0.0, 0.0, 0.0, 1.0]])
    obj = base.scene[name]
    obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=base.device))
    obj.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32,
                                                                       device=base.device))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("STEAK_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _fail(why: str) -> None:
    if not _EP["failed"]:
        _EP["failed"] = why


def _inside(local, b, margin: float = 0.0) -> bool:
    x0, x1, y0, y1, h = b
    return (x0 - margin <= local[0] <= x1 + margin and y0 - margin <= local[1] <= y1 + margin
            and local[2] <= h + margin)


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the thermometer's tip (in the meat, touching bone or fat, in the
    strip's thick middle, still); readings; the stage."""
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
    dt = float(getattr(base, "step_dt", 0.05))
    origin = as_numpy(base.scene.env_origins[0])
    tp = as_numpy(base.scene["thermometer"].data.root_link_pose_w)[0]
    tip, r = tp[:3] - origin, rotation(tp[3:7])
    up = r[:, 2]
    _EP.update(tip=tip, up=up)
    local = tip - np.array([BOARD_XY[0], BOARD_XY[1], TOP + BOARD_T])
    meat_boxes = (STRIP, STRIP_TOP, TENDER)
    in_meat = any(_inside(local, b) for b in meat_boxes) and local[2] > 0.0
    # the probe's part inside the steak: sample it from the tip up to the steak's top
    touch = ""
    for f in np.linspace(0.0, 0.04, 9):
        pt = local + up * f
        if pt[2] > 0.036:
            break
        if any(_inside(pt, b, TOUCH) for b in (BONE_CROSS, BONE_STEM)):
            touch = "the bone"
            break
        if _inside(pt, FAT, TOUCH):
            touch = "the fat"
            break
    core = (math.hypot(local[0] - CORE_C[0], local[1] - CORE_C[1]) <= CORE_R and CORE_Z[0] <= local[2] <= CORE_Z[1])
    moved = 1e9 if _EP["prev_tip"] is None else float(np.linalg.norm(tip - _EP["prev_tip"])) * 1000
    _EP["prev_tip"] = tip.copy()
    _EP["still_s"] = _EP["still_s"] + dt if (in_meat and moved < STILL_PROBE_MM) else 0.0
    _EP.update(in_meat=in_meat, touch=touch, core=core)
    if _EP["still_s"] >= ATTEMPT_S:
        if touch:
            _fail(f"a reading was taken with the probe against {touch}")
        elif not core:
            _fail("a reading was taken outside the strip's thick middle")
    if _EP["still_s"] >= READ_S and core and not touch:
        _EP["read"] = True
    hl = tip - np.array([HOLDER_XY[0], HOLDER_XY[1], THERMO_Z0])
    if hl[2] > OUT_OF_HOLDER:
        _EP["lifted"] = True
    _EP["home"] = bool(math.hypot(hl[0], hl[1]) < HOME_TOL and abs(hl[2]) < HOME_TOL and up[2] > 0.99)
    show = _EP["read"] and not _EP["failed"]
    _place(base, "reading_light", tip + up * (PROBE_L + HEAD[2] + 0.004) if show else HIDE)
    stage = 0.0
    if _EP["lifted"]:
        stage = 0.25
    if in_meat and not touch:
        stage = 0.5
    if _EP["read"]:
        stage = 0.75
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(_EP["tip"].copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    head = _EP["tip"] + _EP["up"] * (PROBE_L + HEAD[2] / 2)
    near = min(np.linalg.norm(v - head) for v in _TCP.values()) if len(_TCP) == 2 else 0.0
    done = _EP["read"] and _EP["home"] and near >= CLEAR and still
    return dict(still=still, near=float(near), done=bool(done))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    tp = _EP["tip"]
    local = tp - np.array([BOARD_XY[0], BOARD_XY[1], TOP + BOARD_T])
    print(f"DBG step={_EP['step']} tip=({tp[0]:.3f},{tp[1]:.3f},{tp[2] - TOP:.3f}) on_board=({local[0]:.3f},"
          f"{local[1]:.3f},{local[2]:.3f}) in_meat={_EP['in_meat']} touch={_EP['touch'] or '-'} core={_EP['core']} "
          f"still={_EP['still_s']:.2f}s read={_EP['read']} lifted={_EP['lifted']} home={_EP['home']} "
          f"near={t['near']:.3f} best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once a reading is taken against bone or fat, or outside the
    strip's thick middle): 0.25 the thermometer lifted out of its holder; 0.5 its tip in the meat; 0.75 a reading in the
    strip's thick middle. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """A reading taken in the strip's thick middle, the thermometer standing back in its holder, both grippers 6 cm or
    more from its head, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right gripper takes the thermometer by its head, lifts it out of its holder, pushes the probe straight down into
# the middle of the strip loin until the tip is halfway down, holds it there for the reading, draws it out, stands it
# back in its holder and moves away. STEAK_BONE=1 runs it the wrong way: the probe goes in beside the bone, near the
# steak's middle; the run scores 0 when the reading is taken.

BY_BONE = os.environ.get("STEAK_BONE") == "1"
OPEN = -0.030
PINCH = -(HEAD[1] / 2 - 0.002)
GRIP_UP = PROBE_L + HEAD[2] / 2 + 0.004  # the TCP this far above the tip, on the head


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


def _idle(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), OPEN, "stay put", rot)]


def demo():
    rot = grip_rotation(0.0)  # fingers closing along y, across the head
    home = np.array([HOLDER_XY[0], HOLDER_XY[1], THERMO_Z0 + GRIP_UP])
    spot = (np.array([0.0, -0.007]) if BY_BONE else CORE_C) + BOARD_XY  # where the probe goes in
    tip_z = TOP + BOARD_T + (CORE_Z[0] + CORE_Z[1]) / 2
    into = np.array([spot[0], spot[1], tip_z + GRIP_UP])
    high = TOP + 0.15
    right = _lead_in("right", home + [0.0, 0.0, 0.05], rot, OPEN, "above the thermometer")
    right += [_w(1.0, home, OPEN, "fingers either side of its head", rot),
              _w(0.6, home, PINCH, "take it by the head", rot),
              _w(1.4, np.array([home[0], home[1], high]), PINCH, "lift it out of its holder", rot),
              _w(1.8, np.array([into[0], into[1], high]), PINCH, "over the strip loin's middle" if not BY_BONE
                 else "over the steak's middle, by the bone", rot),
              _w(1.6, into + [0.0, 0.0, 0.03], PINCH, "the tip to the meat", rot),
              _w(1.2, into, PINCH, "push it in until the tip is halfway down", rot),
              _w(4.0, into, PINCH, "hold it still for the reading", rot),
              _w(1.4, np.array([into[0], into[1], high]), PINCH, "draw it out", rot),
              _w(1.8, np.array([home[0], home[1], high]), PINCH, "back over its holder", rot),
              _w(1.4, home + [0.0, 0.0, 0.003], PINCH, "stand it back in the holder", rot),
              _w(0.6, home + [0.0, 0.0, 0.003], OPEN, "let go", rot),
              _w(0.8, home + [0.0, 0.0, 0.06], OPEN, "rise", rot),
              _w(1.2, np.array([0.20, -0.26, TOP + 0.18]), OPEN, "move away", rot)]
    t = sum(w.duration for w in right)
    right.append(_w(2.0, np.array([0.20, -0.26, TOP + 0.18]), OPEN, "stay clear", rot))
    return {"left": _idle("left", t + 2.0), "right": right}
