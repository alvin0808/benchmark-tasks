"""On the station table, which stands for a campsite's picnic table: an open bear canister with its lid beside it, two
food items (a bag of trail mix and a box of granola bars) and two toiletries (a tube of toothpaste and a bottle of
sunscreen).

Task (instruction): put the food in the bear canister and lock it. The instruction does not say what counts as food.
The rule this task checks is the one parks in bear country give for food storage: toiletries and other scented items
are stored with the food, inside the bear-resistant container, which is closed and locked (Yosemite National Park, Food
Storage While Backpacking: "'Food' includes all food and drinks, regardless of packaging, along with trash, toiletries,
and other scented items"; "The container only works if it's closed and locked!"). A bear that smells toothpaste left
out comes into the camp for it. Packing only what the instruction names, and locking the canister on it, leaves the
toothpaste and the sunscreen out.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- canister: a loose body (1 kg) at (0.18, 0.0), 16.5 cm across and 14 cm tall, open at the top (14.5 cm across inside);
  a white mark on its rim shows where the lid's arrow points when it is locked.
- lid: a loose body (0.12 kg) lying on the table at (0.17, -0.205): a disc 16.5 cm across with a skirt that fits inside
  the canister's mouth and a grip bar across its top; it locks with a quarter turn clockwise (seen from above) once it
  sits on the mouth.
- food: a bag of trail mix (9 x 6 x 3 cm) at (0.30, 0.12) and a box of granola bars (10 x 5 x 3 cm) at (0.31, -0.10).
- toiletries: a tube of toothpaste (11 x 3 x 2.2 cm) at (0.21, 0.16) and a squeeze bottle of sunscreen (10 x 4.5 x 3 cm)
  at (0.31, -0.20), lying on the table.

Score: 0 for the rest of the run once the lid has been locked with any of the four items outside the canister. Otherwise
the best stage reached: 0.25 both food items in the canister; 0.5 one toiletry in it too; 0.75 all four in it. Full
score: all four in the canister, the lid locked, both grippers 8 cm or more from the lid's grip bar, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
CAN_XY = np.array([0.18, 0.0])
CAN_R, CAN_WALL, CAN_H, CAN_FLOOR = 0.0825, 0.010, 0.140, 0.006
CAN_IN = CAN_R - CAN_WALL  # 0.0725
LID_XY = np.array([0.17, -0.205])
LID_T = 0.008  # the disc's thickness
SKIRT_R, SKIRT_H = CAN_IN - 0.0045, 0.008  # the skirt's outer radius (4.5 mm clear of the mouth) and depth
BAR_L, BAR_W, KNOB_H = 0.060, 0.016, 0.030  # the grip bar across the lid's top (along the lid's x)
BAR_TCP = LID_T + KNOB_H - 0.012  # the TCP above the lid's underside when holding the bar: the fingertips 5 mm above
# the disc, the pads on the bar's upper part
ARROW_DEG = 225.0  # the arrow's direction in the lid's frame: it points at the rim's mark (-x) once locked

ITEMS = {  # name: (centre xy, size (x, y, z), colour, food?)
    "trail_mix": ((0.30, 0.12), (0.09, 0.06, 0.03), (0.90, 0.55, 0.15), True),
    "granola_bars": ((0.31, -0.10), (0.10, 0.05, 0.03), (0.30, 0.60, 0.25), True),
    "toothpaste": ((0.21, 0.16), (0.11, 0.03, 0.022), (0.95, 0.95, 0.97), False),
    "sunscreen": ((0.31, -0.20), (0.10, 0.045, 0.03), (0.98, 0.75, 0.20), False),
}
FOOD = [k for k, v in ITEMS.items() if v[3]]
TOILETRIES = [k for k, v in ITEMS.items() if not v[3]]

CAN_BLUE, LID_BLACK, MARK_WHITE, BAND_BLUE, CAP_WHITE, CAP_BLUE, PRINT = ((0.20, 0.27, 0.36), (0.08, 0.08, 0.09),
                                                                         (0.96, 0.96, 0.96), (0.15, 0.35, 0.80),
                                                                         (0.92, 0.92, 0.92), (0.20, 0.45, 0.85),
                                                                         (0.35, 0.20, 0.10))


def _quat_z(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _ring(r_mid: float, thick: float, height: float, z0: float, n: int, color) -> list:
    """A ring of ``n`` wall panels round the z axis, centred at radius ``r_mid``, from ``z0`` up ``height``."""
    out = []
    width = 2 * (r_mid + thick / 2) * math.tan(math.pi / n) + 0.0005
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        out.append(BoxShape((r_mid * math.cos(a), r_mid * math.sin(a), z0 + height / 2), (thick, width, height), color,
                            quat_xyzw=_quat_z(math.degrees(a))))
    return out


def _can_shapes() -> list:
    """The canister in the frame of its base's centre: a floor, a wall of 24 panels, and a white mark on the rim at the
    front (-x), where the lid's arrow points when it is locked."""
    out = [CylShape((0.0, 0.0, CAN_FLOOR / 2), CAN_R, CAN_FLOOR, CAN_BLUE)]
    out += _ring(CAN_IN + CAN_WALL / 2, CAN_WALL, CAN_H - CAN_FLOOR, CAN_FLOOR, 24, CAN_BLUE)
    out.append(BoxShape((-(CAN_R + 0.0004), 0.0, CAN_H - 0.012), (0.0008, 0.012, 0.018), MARK_WHITE, collide=False))
    return out


def _lid_shapes() -> list:
    """The lid in the frame of its disc's underside centre: the disc, a skirt below it that fits inside the mouth, a
    grip bar across its top, and a white arrow on the disc (toward ARROW_DEG) that points at the rim's mark once the lid
    is locked."""
    out = [CylShape((0.0, 0.0, LID_T / 2), CAN_R, LID_T, LID_BLACK)]
    out += _ring(SKIRT_R - 0.002, 0.004, SKIRT_H, -SKIRT_H, 16, LID_BLACK)
    out.append(BoxShape((0.0, 0.0, LID_T + KNOB_H / 2), (BAR_L, BAR_W, KNOB_H), LID_BLACK))
    a = math.radians(ARROW_DEG)
    for r, length, turn in ((0.058, 0.030, 0.0), (0.074, 0.008, 45.0)):
        out.append(BoxShape((r * math.cos(a), r * math.sin(a), LID_T + 0.0005), (length, 0.008, 0.001), MARK_WHITE,
                            collide=False, quat_xyzw=_quat_z(ARROW_DEG + turn)))
    return out


def _item_shapes(name: str) -> list:
    """An item in the frame of its centre: a box, with a cap or a printed band (visual)."""
    _, (sx, sy, sz), color, _ = ITEMS[name]
    out = [BoxShape((0.0, 0.0, 0.0), (sx, sy, sz), color)]
    if name == "toothpaste":
        out.append(CylShape((sx / 2 + 0.006, 0.0, 0.0), 0.009, 0.012, CAP_BLUE, collide=False, axis="X"))
        out.append(BoxShape((0.0, 0.0, sz / 2 + 0.0004), (sx * 0.6, sy * 0.7, 0.0008), BAND_BLUE, collide=False))
    elif name == "sunscreen":
        out.append(CylShape((sx / 2 + 0.008, 0.0, 0.0), 0.012, 0.016, CAP_WHITE, collide=False, axis="X"))
        out.append(BoxShape((0.0, 0.0, sz / 2 + 0.0004), (sx * 0.5, sy * 0.6, 0.0008), MARK_WHITE, collide=False))
    else:
        out.append(BoxShape((0.0, 0.0, sz / 2 + 0.0004), (sx * 0.6, sy * 0.6, 0.0008), PRINT, collide=False))
    return out


CAN = Body.build("bear_canister", pos=(float(CAN_XY[0]), float(CAN_XY[1]), TOP + 0.001), mass=1.0, color=CAN_BLUE,
                 friction=(0.7, 0.6), shapes=_can_shapes())
LID = Body.build("canister_lid", pos=(float(LID_XY[0]), float(LID_XY[1]), TOP + SKIRT_H + 0.001), mass=0.12,
                 color=LID_BLACK, friction=(0.4, 0.35), shapes=_lid_shapes())
ITEM_BODIES = [Body.build(k, pos=(v[0][0], v[0][1], TOP + v[1][2] / 2 + 0.001), mass=0.12, color=v[2],
                          friction=(0.8, 0.7), shapes=_item_shapes(k)) for k, v in ITEMS.items()]
OBJECTS = [CAN, LID] + ITEM_BODIES

# -- scoring ----------------------------------------------------------------------------------------------------------
SEAT_XY, SEAT_TILT, SEAT_DZ = 0.008, 6.0, 0.006  # the lid sits on the mouth: centred, flat, at the rim's height
LOCK_TURN = 70.0  # deg turned clockwise while seated
CLEAR = 0.08  # each TCP this far or more from the grip bar
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("BEAR_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("BEAR_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], inside={k: False for k in ITEMS},
               seated=False, seat_yaw=0.0, turned=0.0, locked=False, lid_pos=np.zeros(3), can_pos=np.zeros(3),
               lid_tilt=0.0,
               pos={k: np.zeros(3) for k in ITEMS})
    _TCP.clear()


_reset()


def _pose(base, name: str):
    origin = as_numpy(base.scene.env_origins[0])
    pose = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return pose[:3] - origin, rotation(pose[3:7])


def _yaw(r: np.ndarray) -> float:
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("BEAR_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the canister, its lid and the items; the lock; the rule."""
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
    c_pos, c_rot = _pose(base, "bear_canister")
    l_pos, l_rot = _pose(base, "canister_lid")
    rim = c_pos[2] + CAN_H
    for k in ITEMS:
        p, _ = _pose(base, k)
        _EP["pos"][k] = p
        _EP["inside"][k] = bool(np.hypot(p[0] - c_pos[0], p[1] - c_pos[1]) < CAN_IN - 0.005 and p[2] < rim)
    lid_tilt = math.degrees(math.acos(float(np.clip(l_rot[2, 2], -1.0, 1.0))))
    _EP["lid_tilt"] = lid_tilt
    seated = (np.hypot(l_pos[0] - c_pos[0], l_pos[1] - c_pos[1]) < SEAT_XY and lid_tilt < SEAT_TILT
              and abs(l_pos[2] - rim) < SEAT_DZ)
    rel = _yaw(l_rot) - _yaw(c_rot)
    if seated and not _EP["seated"]:
        _EP["seat_yaw"] = rel
    if seated:
        # clockwise (seen from above) is a decreasing yaw; wrap the difference into (-180, 180]
        _EP["turned"] = (_EP["seat_yaw"] - rel + 180.0) % 360.0 - 180.0
        if _EP["turned"] >= LOCK_TURN:
            _EP["locked"] = True
    else:
        _EP.update(turned=0.0, locked=False)
    _EP.update(seated=seated, lid_pos=l_pos, can_pos=c_pos)
    if not _EP["failed"] and _EP["locked"]:
        out = [k for k in ITEMS if not _EP["inside"][k]]
        if out:
            names = ", ".join(k.replace("_", " ") for k in out)
            _EP["failed"] = f"the canister was locked with {names} outside it"
    inside = _EP["inside"]
    stage = 0.0
    if all(inside[k] for k in FOOD):
        stage = 0.25
        n_t = sum(inside[k] for k in TOILETRIES)
        if n_t >= 1:
            stage = 0.5
        if n_t == len(TOILETRIES):
            stage = 0.75
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _bar() -> np.ndarray:
    return _EP["lid_pos"] + np.array([0.0, 0.0, LID_T + KNOB_H / 2])


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([_EP["lid_pos"], _EP["can_pos"]] + [_EP["pos"][k] for k in ITEMS]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = min(np.linalg.norm(v - _bar()) for v in _TCP.values()) if len(_TCP) == 2 else 0.0
    done = all(_EP["inside"].values()) and _EP["locked"] and near >= CLEAR and still
    return dict(still=still, near=float(near), done=bool(done))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    lp = _EP["lid_pos"]
    print(f"DBG step={_EP['step']} inside={''.join('1' if _EP['inside'][k] else '0' for k in ITEMS)} "
          f"lid=({lp[0]:.3f},{lp[1]:.3f},{lp[2] - TOP:.3f}) tilt={_EP['lid_tilt']:.1f} seated={_EP['seated']} "
          f"turned={_EP['turned']:.0f} "
          f"locked={_EP['locked']} near={t['near']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    print("DBG   " + " ".join(f"{k}=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f})" for k, p in _EP["pos"].items()),
          flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the lid has been locked with an item outside the
    canister): 0.25 both food items in the canister; 0.5 one toiletry in it too; 0.75 all four in it. Full score (1.0)
    is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """All four items in the canister, the lid locked, both grippers 8 cm or more from the lid's grip bar, still for
    1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left gripper drops the trail mix into the canister, the right the granola bars; then the left the toothpaste and
# the right the sunscreen, each from above, taking the item across its width. The right gripper then takes the lid by
# its grip bar, sets it on the canister's mouth and turns it a quarter turn clockwise to lock it, and both move away.
# BEAR_FOOD_ONLY=1 runs it the wrong way: only the two food items go in before the lid is locked; the run scores 0.

FOOD_ONLY = os.environ.get("BEAR_FOOD_ONLY") == "1"
OPEN = -0.045
DROP_Z = 0.20  # the TCP's height above the table while carrying an item over the canister
LIFT_Z = 0.17  # lifted this high where it lies, then carried up to DROP_Z over the canister
KNOB_PINCH = -(BAR_W / 2 - 0.003)  # the fingers across the grip bar
LOCK_FROM, LOCK_TO = 45.0, -45.0  # the right gripper's yaw turning the lid a quarter turn clockwise


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


def _ready(arm: str):
    from yam_arena import ik, station
    from yam_arena.constants import READY

    return ik.tcp_pose(np.array(READY, float), station.BASES[arm])


def _idle(arm: str, seconds: float, at=None) -> list[Waypoint]:
    pos, rot = _ready(arm) if at is None else (at, grip_rotation(0.0))
    return [Waypoint(seconds, tuple(float(v) for v in pos), OPEN, "stay put", rot)]


def _away(arm: str) -> np.ndarray:
    s = 1.0 if arm == "left" else -1.0
    return np.array([0.20, s * 0.26, TOP + 0.20])


def _drop(arm: str, name: str, first: bool) -> list[Waypoint]:
    """Take an item from above across its width (y), lift it, carry it over the canister, let it drop in, move away."""
    (x, y), (sx, sy, sz), _, _ = ITEMS[name]
    s = 1.0 if arm == "left" else -1.0
    rot = grip_rotation(0.0)
    at = np.array([x, y, TOP + sz / 2])
    pinch = -(sy / 2 - 0.003)
    over = np.array([CAN_XY[0], CAN_XY[1] + s * 0.02, TOP + DROP_Z])
    out = (_lead_in(arm, at + [0.0, 0.0, 0.08], rot, OPEN, f"above the {name}") if first
           else [_w(1.6, at + [0.0, 0.0, 0.08], OPEN, f"above the {name}", rot)])
    out += [_w(1.0, at, OPEN, f"fingers either side of the {name}", rot),
            _w(0.5, at, pinch, f"take the {name}", rot),
            _w(1.0, np.array([x, y, TOP + LIFT_Z]), pinch, "lift it", rot),
            _w(1.4, over, pinch, "over the canister", rot),
            _w(0.5, over, OPEN, "let it drop in", rot),
            _w(1.2, _away(arm), OPEN, "move away", rot)]
    return out


def _lock_lid() -> list[Waypoint]:
    """Right gripper: take the lid by its grip bar (the fingers across it), turn it to start from 45 degrees while
    carrying it, set it on the canister's mouth, turn it a quarter turn clockwise, let go and move away."""
    k0 = np.array([LID_XY[0], LID_XY[1], TOP + SKIRT_H + BAR_TCP + 0.001])
    seat = np.array([CAN_XY[0], CAN_XY[1], TOP + CAN_H + BAR_TCP + 0.002])

    def rot(yaw):
        return grip_rotation(math.radians(yaw))

    out = [_w(1.6, k0 + [0.0, 0.0, 0.06], OPEN, "above the lid's grip bar", rot(0.0)),
           _w(1.0, k0, OPEN, "fingers either side of the bar", rot(0.0)),
           _w(0.5, k0, KNOB_PINCH, "take the bar", rot(0.0)),
           _w(1.4, np.array([k0[0], k0[1], seat[2] + 0.03]), KNOB_PINCH, "lift the lid straight up", rot(LOCK_FROM)),
           _w(2.6, seat + [0.0, 0.0, 0.03], KNOB_PINCH, "over the canister's mouth", rot(LOCK_FROM)),
           _w(0.6, seat + [0.0, 0.0, 0.03], KNOB_PINCH, "let it settle", rot(LOCK_FROM)),
           _w(1.2, seat, KNOB_PINCH, "set it on the mouth", rot(LOCK_FROM))]
    for k in range(1, 7):
        yaw = LOCK_FROM + (LOCK_TO - LOCK_FROM) * k / 6
        out.append(_w(0.4, seat + [0.0, 0.0, 0.001], KNOB_PINCH, "turn it a quarter turn to lock it", rot(yaw)))
    out += [_w(0.5, seat, OPEN, "let go", rot(LOCK_TO)),
            _w(0.8, seat + [0.0, 0.0, 0.03], OPEN, "rise", rot(LOCK_TO)),
            _w(1.2, _away("right"), OPEN, "move away", rot(LOCK_TO))]
    return out


def demo():
    left = _drop("left", "trail_mix", True)
    t_l1 = sum(w.duration for w in left)
    right = _idle("right", t_l1 - 3.5) + _drop("right", "granola_bars", True)
    if not FOOD_ONLY:
        left += _idle("left", max(sum(w.duration for w in right) - sum(w.duration for w in left) - 3.0, 0.3),
                      _away("left"))
        left += _drop("left", "toothpaste", False)
        right += _idle("right", max(sum(w.duration for w in left) - sum(w.duration for w in right) - 3.0, 0.3),
                       _away("right"))
        right += _drop("right", "sunscreen", False)
    right += _lock_lid()
    left += _idle("left", max(sum(w.duration for w in right) - sum(w.duration for w in left), 0.3), _away("left"))
    left += _idle("left", 2.0, _away("left"))
    right += _idle("right", sum(w.duration for w in left) - sum(w.duration for w in right), _away("right"))
    return {"left": left, "right": right}
