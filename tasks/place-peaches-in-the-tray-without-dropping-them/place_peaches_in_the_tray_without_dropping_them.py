"""On the station table, which stands for a packing table at a fruit farm: four ripe peaches just picked, two on each
side, each resting in a foam ring, and a cardboard packing tray in the middle with four cups.

Task (instruction): pack the peaches into the tray. The instruction does not say how. The rule this task checks is the
one fruit growers train pickers and packers in: a ripe peach bruises when it is dropped, even a short way, so it is
placed, not dropped (University of Maryland Extension, Maximizing Apple and Peach Profits: Preventative Bruising
Practices: peaches are placed "gently into the basket", apples handled "like an egg"). The quick way is to let each
peach go over its cup and let it fall in; lowering it until it rests in the cup, and only then letting go, keeps the
rule.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- peaches: loose spheres (6.8 cm across, 0.15 kg), at (0.17, +0.21), (0.31, +0.21), (0.17, -0.21) and (0.31, -0.21),
  each resting in a foam ring on the table.
- tray: a loose body (0.1 kg) at (0.24, 0.0), 22 x 26 cm, with four cups (rings 6.4 cm across inside, 2.6 cm high,
  that a peach rests on) at (0.19, +0.075), (0.29, +0.075), (0.19, -0.075) and (0.29, -0.075).

A peach moves with a gripper while its centre is within 4.5 cm of the gripper's centre and keeps the same offset from
it (to 2 mm a step); it was let go at the height where it last moved with a gripper, and comes to rest when it has
moved less than 3 mm in 0.15 s. Score: 0 for the rest of the run once a peach has come to rest 1.5 cm or
more below the height at which it was released (it was dropped, or rolled off and fell). Otherwise the best stage
reached: 0.25 one peach resting in a cup; 0.5 two; 0.75 three. Full score: all four resting in the tray's cups, both
grippers 6 cm or more from every peach, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, SphereShape, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
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
PEACH_R = 0.034
PEACHES = {"peach_1": (0.17, 0.21), "peach_2": (0.31, 0.21), "peach_3": (0.17, -0.21), "peach_4": (0.31, -0.21)}
TRAY_XY = np.array([0.24, 0.0])
TRAY_SIZE, TRAY_T = (0.22, 0.26), 0.004
CUPS = [(-0.05, 0.075), (0.05, 0.075), (-0.05, -0.075), (0.05, -0.075)]  # cup centres, from the tray's centre
CUP_R, CUP_WALL, CUP_H = 0.032, 0.004, 0.026  # inside radius, wall, height: a peach rests on the rim
RING_R, RING_H = 0.026, 0.014  # the foam rings the peaches start in (inside radius, height)
SEAT = math.sqrt(PEACH_R ** 2 - CUP_R ** 2)  # a peach's centre above a cup's rim when it rests in the cup

PEACH_COLOR, BLUSH, CARD, FOAM = (0.98, 0.62, 0.25), (0.88, 0.30, 0.20), (0.80, 0.68, 0.50), (0.92, 0.92, 0.88)


def _quat_z(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _ring(cx: float, cy: float, r_in: float, thick: float, height: float, z0: float, n: int, color) -> list:
    """A ring of ``n`` wall panels round (cx, cy), its inside radius ``r_in``, from ``z0`` up ``height``."""
    out = []
    r_mid = r_in + thick / 2
    width = 2 * (r_in + thick) * math.tan(math.pi / n) + 0.0005
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        out.append(BoxShape((cx + r_mid * math.cos(a), cy + r_mid * math.sin(a), z0 + height / 2),
                            (thick, width, height), color, quat_xyzw=_quat_z(math.degrees(a))))
    return out


def _tray_shapes() -> list:
    """The tray in the frame of its base's centre: a flat card with four cups (rings) on it."""
    out = [BoxShape((0.0, 0.0, TRAY_T / 2), (TRAY_SIZE[0], TRAY_SIZE[1], TRAY_T), CARD)]
    for cx, cy in CUPS:
        out += _ring(cx, cy, CUP_R, CUP_WALL, CUP_H, TRAY_T, 16, CARD)
    return out


def _peach_shapes() -> list:
    """A peach: a sphere with a blush on one side and a stem scar on top (visual)."""
    return [SphereShape((0.0, 0.0, 0.0), PEACH_R, PEACH_COLOR),
            SphereShape((0.012, 0.010, 0.006), PEACH_R - 0.009, BLUSH, collide=False),
            SphereShape((0.0, 0.0, PEACH_R - 0.003), 0.004, (0.45, 0.30, 0.15), collide=False)]


def _ring_body(name: str, xy) -> Body:
    return Body.build(name, pos=(xy[0], xy[1], TOP + 0.001), mass=0.02, color=FOAM, friction=(0.9, 0.8),
                      shapes=_ring(0.0, 0.0, RING_R, 0.006, RING_H, 0.0, 12, FOAM))


RING_SEAT = math.sqrt(PEACH_R ** 2 - RING_R ** 2)
TRAY = Body.build("packing_tray", pos=(float(TRAY_XY[0]), float(TRAY_XY[1]), TOP + 0.001), mass=0.1, color=CARD,
                  friction=(0.8, 0.7), shapes=_tray_shapes())
RINGS = [_ring_body(f"{k}_ring", xy) for k, xy in PEACHES.items()]
PEACH_BODIES = [Body.build(k, pos=(xy[0], xy[1], TOP + 0.001 + RING_H + RING_SEAT + 0.001), mass=0.15,
                           color=PEACH_COLOR, friction=(0.9, 0.8), damping=(0.05, 0.4), shapes=_peach_shapes())
                for k, xy in PEACHES.items()]
OBJECTS = [TRAY] + RINGS + PEACH_BODIES

# -- scoring ----------------------------------------------------------------------------------------------------------
HELD_R = 0.045  # a peach moves with a gripper when its centre is this close to the gripper's centre ...
WITH_MM = 2.0  # ... and its offset from the gripper's centre changes by less than this from one step to the next
DROP_LIMIT = 0.015
REST_MM, REST_STEPS = 3.0, 3
CLEAR = 0.06
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("PEACH_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("PEACH_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


def _fresh() -> dict:
    return dict(pos=np.zeros(3), held=False, release_z=None, offset=None, hist=[], in_cup=False, fall=0.0)


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], peaches={k: _fresh() for k in PEACHES},
               tray_pos=np.zeros(3), tray_rot=np.eye(3))
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("PEACH_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; each peach: held or not, where it was let go, where it came to
    rest; the tray's cups; the rule."""
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
    tp = as_numpy(base.scene["packing_tray"].data.root_link_pose_w)[0]
    t_pos, t_rot = tp[:3] - origin, rotation(tp[3:7])
    _EP.update(tray_pos=t_pos, tray_rot=t_rot)
    for k in PEACHES:
        s = _EP["peaches"][k]
        p = as_numpy(base.scene[k].data.root_link_pose_w)[0][:3] - origin
        s["pos"] = p
        # moving with a gripper: close to its centre and keeping the same offset from it (held, or resting under it)
        near = min(_TCP, key=lambda a: np.linalg.norm(_TCP[a] - p)) if _TCP else None
        offset = None if near is None else p - _TCP[near]
        held = bool(near is not None and np.linalg.norm(offset) < HELD_R and s["offset"] is not None
                    and np.linalg.norm(offset - s["offset"]) * 1000 < WITH_MM)
        s["offset"] = offset
        if held:
            s["release_z"] = float(p[2])  # where it last moved with a gripper: where it was let go
        s["held"] = held
        s["hist"].append(p.copy())
        del s["hist"][:-REST_STEPS]
        resting = (len(s["hist"]) == REST_STEPS
                   and max(np.linalg.norm(h - s["hist"][-1]) for h in s["hist"]) * 1000 < REST_MM)
        if not held and resting and s["release_z"] is not None:
            s["fall"] = s["release_z"] - float(p[2])
            if s["fall"] >= DROP_LIMIT and not _EP["failed"]:
                _EP["failed"] = (f"{k.replace('_', ' ')} was dropped: it came to rest {s['fall'] * 100:.1f} cm below "
                                 "where it was let go")
            s["release_z"] = None
        local = t_rot.T @ (p - t_pos)
        s["in_cup"] = (not held and resting
                       and any(math.hypot(local[0] - cx, local[1] - cy) < 0.008 for cx, cy in CUPS)
                       and abs(local[2] - (TRAY_T + CUP_H + SEAT)) < 0.006)
    n = sum(s["in_cup"] for s in _EP["peaches"].values())
    stage = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75}.get(n, 0.75)
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([s["pos"] for s in _EP["peaches"].values()] + [_EP["tray_pos"]]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = (min(np.linalg.norm(v - s["pos"]) for v in _TCP.values() for s in _EP["peaches"].values())
            if len(_TCP) == 2 else 0.0)
    done = all(s["in_cup"] for s in _EP["peaches"].values()) and near >= CLEAR and still
    return dict(still=still, near=float(near), done=bool(done))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    parts = []
    for k, s in _EP["peaches"].items():
        p = s["pos"]
        rz = "-" if s["release_z"] is None else f"{s['release_z'] - TOP:.3f}"
        parts.append(f"{k}=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f}) held={s['held']} rel_z={rz} "
                     f"fall={s['fall'] * 100:.1f}cm cup={s['in_cup']}")
    print(f"DBG step={_EP['step']} near={t['near']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    print("DBG   " + " | ".join(parts), flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once a peach has come to rest 1.5 cm or more below where it was
    let go): 0.25 one peach resting in a cup; 0.5 two; 0.75 three. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """All four peaches resting in the tray's cups, both grippers 6 cm or more from every peach, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left gripper takes its two peaches, one at a time, across the middle from above, carries each over a cup on its
# side of the tray, lowers it until it rests in the cup, and only then opens; the right gripper does the same with its
# two, after the left has moved away. PEACH_DROP=1 runs it the wrong way: each peach is let go 6 cm above its cup.

DROP = os.environ.get("PEACH_DROP") == "1"
OPEN = -0.045
PINCH = -(PEACH_R - 0.004)
CARRY_Z = 0.12


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


def _away(arm: str) -> np.ndarray:
    s = 1.0 if arm == "left" else -1.0
    return np.array([0.20, s * 0.28, TOP + 0.18])


def _pack(arm: str) -> list[Waypoint]:
    s = 1.0 if arm == "left" else -1.0
    rot = grip_rotation(0.0)  # the fingers close along y
    names = ["peach_1", "peach_2"] if arm == "left" else ["peach_3", "peach_4"]
    cups = [c for c in CUPS if c[1] * s > 0]
    out = []
    for i, (k, (cx, cy)) in enumerate(zip(names, cups)):
        x, y = PEACHES[k]
        at = np.array([x, y, TOP + 0.001 + RING_H + RING_SEAT])
        cup = np.array([TRAY_XY[0] + cx, TRAY_XY[1] + cy, TOP + 0.001 + TRAY_T + CUP_H + SEAT])
        let_go = cup + [0.0, 0.0, 0.06 if DROP else 0.002]
        if i == 0:
            out += _lead_in(arm, at + [0.0, 0.0, 0.08], rot, OPEN, f"above {k.replace('_', ' ')}")
        else:
            out.append(_w(1.6, at + [0.0, 0.0, 0.08], OPEN, f"above {k.replace('_', ' ')}", rot))
        out += [_w(1.0, at, OPEN, "fingers either side of it", rot),
                _w(0.5, at, PINCH, "take it round the middle", rot),
                _w(0.9, at + [0.0, 0.0, CARRY_Z - (at[2] - TOP)], PINCH, "lift it", rot),
                _w(1.4, np.array([cup[0], cup[1], TOP + CARRY_Z]), PINCH, "over its cup", rot),
                _w(1.2, let_go, PINCH, "let it fall" if DROP else "lower it until it rests in the cup", rot),
                _w(0.5, let_go, OPEN, "let go", rot),
                _w(0.8, let_go + [0.0, 0.0, 0.08], OPEN, "rise", rot)]
    out.append(_w(1.2, _away(arm), OPEN, "move away", rot))
    return out


def demo():
    left = _pack("left")
    t_left = sum(w.duration for w in left)
    from yam_arena import ik, station
    from yam_arena.constants import READY

    rpos, rrot = ik.tcp_pose(np.array(READY, float), station.BASES["right"])
    right = [Waypoint(t_left - 3.0, tuple(float(v) for v in rpos), OPEN, "wait for the left arm", rrot)]
    right += _pack("right")
    t_right = sum(w.duration for w in right)
    left.append(_w(t_right - t_left + 2.0, _away("left"), OPEN, "stay clear", grip_rotation(0.0)))
    right.append(_w(2.0, _away("right"), OPEN, "stay clear", grip_rotation(0.0)))
    return {"left": left, "right": right}
