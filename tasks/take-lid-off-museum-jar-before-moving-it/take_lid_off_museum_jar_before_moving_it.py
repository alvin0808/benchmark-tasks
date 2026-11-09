"""On the station table, which stands for a work table in a museum's collection store: a lidded porcelain jar (blue and
white, square, with a round neck) stands on a green felt mat, and a padded tray (foam in a black frame) waits on the
left.

Task (instruction): move the jar onto the padded tray. The instruction does not say how. The rule this task checks is
one every museum's handling guidelines give: take a lid (or any loose part) off an object and move it separately
before moving the object. A lid that rides along can slide off and break, or chip the rim it sits on. US National
Park Service, "Guidelines for Handling Collections": "If an object is made in separable sections, take it apart before
moving it." National Museums Scotland, "How to handle ceramics and glass": remove any loose parts (most commonly lids)
and treat them as a separate object.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- jar: square body 7 x 7 x 8.5 cm with a round neck (6 cm across, 1.2 cm tall) on top, 0.35 kg, standing on the felt
  mat at (0.29, -0.02). Its lid (0.06 kg) is a cap 7.1 cm across with a small knob on top; its skirt (1 cm deep) fits
  round the neck with 2.5 mm to spare, so it stays on unless lifted.
- padded tray: fixed, its foam floor 13 x 24 cm (1.2 cm thick) inside a frame 2.5 cm tall, centred at (0.17, 0.13).

Score: 0 for the rest of the run once the jar has been moved (2 cm along the table, lifted 1 cm or tipped 10 degrees)
with its lid still on it (the lid's skirt round the neck), or knocked over (tipped 30 degrees). Otherwise the best stage
reached: 0.25 the lid lifted off the jar; 0.5 the jar standing on the tray's foam; 0.75 the jar on the tray and the lid
too (lying flat on the foam, or seated back on the jar). Full score: that, with both grippers 4 cm or more from the
lid's knob and 8 cm or more from the jar's centre, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
JAR0 = np.array([0.29, -0.02, TOP])  # the jar's origin: the centre of its base
BODY_W, BODY_H = 0.070, 0.085
NECK_R, NECK_WALL, NECK_H = 0.026, 0.004, 0.012  # the neck's inner radius, wall, height above the body
NECK_TOP = BODY_H + NECK_H
LID_R, LID_T = 0.0355, 0.005  # the lid's cap: outer radius, top's thickness (the top's underside is the lid's origin)
SKIRT_IN, SKIRT_H = 0.0325, 0.010  # the skirt hanging from the top round the neck: inner radius, depth
KNOB = (0.014, 0.010)  # the knob on top: width, height
LID_GRIP = np.array([0.0, 0.0, 0.007])  # where the lid is gripped (by its side, the pads' tips 6 mm above its rim)
LID_KNOB = np.array([0.0, 0.0, LID_T + KNOB[1] / 2])
JAR_GRIP = np.array([0.0, 0.0, 0.055])  # where the body is gripped, in the jar's frame
LID0 = JAR0 + [0.0, 0.0, NECK_TOP]
TRAY_C = np.array([0.17, 0.13, TOP])  # the centre of the tray's foam, on the table
TRAY_IN = (0.13, 0.24)  # the foam's size inside the frame (x, y)
PAD_H, FRAME_T, FRAME_H = 0.012, 0.010, 0.025
PAD_TOP = TOP + PAD_H
JAR_DROP = np.array([0.172, 0.185])  # where the demo sets the jar down on the foam
LID_DROP = np.array([0.170, 0.062])  # and the lid
PORCELAIN, COBALT, FELT, FOAM, FRAME = ((0.93, 0.93, 0.90), (0.10, 0.20, 0.55), (0.16, 0.36, 0.20),
                                        (0.80, 0.80, 0.78), (0.10, 0.10, 0.11))


def _quat_z(deg: float) -> tuple:
    a = math.radians(deg)
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def _ring(cx, cy, z0, r_in, thick, height, color, n=16, collide=True) -> list:
    """A tube wall of ``n`` flat boxes around a vertical axis at (cx, cy), from z0 up ``height``; its inner faces touch a
    circle of radius ``r_in``."""
    r_mid = r_in + thick / 2
    chord = 2 * (r_in + thick) * math.tan(math.pi / n) + 0.001
    return [BoxShape((cx + r_mid * math.cos(2 * math.pi * k / n), cy + r_mid * math.sin(2 * math.pi * k / n),
                      z0 + height / 2), (thick, chord, height), color, collide=collide,
                     quat_xyzw=_quat_z(math.degrees(2 * math.pi * k / n))) for k in range(n)]


def _jar_shapes() -> list:
    """The body, the neck, and the blue decoration (visual only) on each face: a band at the top and at the foot and a
    diamond between them, in the frame of the base's centre."""
    out = [BoxShape((0.0, 0.0, BODY_H / 2), (BODY_W, BODY_W, BODY_H), PORCELAIN)]
    out += _ring(0.0, 0.0, BODY_H, NECK_R, NECK_WALL, NECK_H, PORCELAIN, n=16)
    out.append(CylShape((0.0, 0.0, BODY_H + 0.0004), NECK_R + NECK_WALL + 0.004, 0.0008, COBALT, collide=False))
    h = BODY_W / 2 + 0.0004
    for k in range(4):
        a = math.pi / 2 * k
        c, s = math.cos(a), math.sin(a)
        q = _quat_z(math.degrees(a))
        for z, t in ((0.010, 0.006), (BODY_H - 0.010, 0.006)):
            out.append(BoxShape((h * c, h * s, z), (0.0008, BODY_W - 0.006, t), COBALT, collide=False, quat_xyzw=q))
        dia = 0.026
        out.append(BoxShape((h * c, h * s, BODY_H / 2), (0.0008, dia, dia), COBALT, collide=False,
                            quat_xyzw=_mul(q, (math.sin(math.pi / 8), 0.0, 0.0, math.cos(math.pi / 8)))))
        out.append(BoxShape(((h + 0.0003) * c, (h + 0.0003) * s, BODY_H / 2), (0.0008, dia * 0.55, dia * 0.55),
                            PORCELAIN, collide=False,
                            quat_xyzw=_mul(q, (math.sin(math.pi / 8), 0.0, 0.0, math.cos(math.pi / 8)))))
    return out


def _mul(a, b) -> tuple:
    """Quaternion product a * b (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _lid_shapes() -> list:
    """The cap's top, the skirt under its edge (round the neck), the knob and a blue ring on top, in the frame of the
    top's underside at its centre."""
    return ([CylShape((0.0, 0.0, LID_T / 2), LID_R, LID_T, PORCELAIN)]
            + _ring(0.0, 0.0, -SKIRT_H, SKIRT_IN, LID_R - SKIRT_IN, SKIRT_H, PORCELAIN, n=20)
            + [BoxShape((0.0, 0.0, LID_T + KNOB[1] / 2), (KNOB[0], KNOB[0], KNOB[1]), PORCELAIN),
               BoxShape((0.0, 0.0, LID_T + KNOB[1] + 0.0004), (KNOB[0] - 0.004, KNOB[0] - 0.004, 0.0008), COBALT,
                        collide=False),
               CylShape((0.0, 0.0, LID_T + 0.0004), LID_R - 0.003, 0.0008, COBALT, collide=False),
               CylShape((0.0, 0.0, LID_T + 0.0008), LID_R - 0.008, 0.0008, PORCELAIN, collide=False)]
            + _ring(0.0, 0.0, -SKIRT_H + 0.003, LID_R + 0.0002, 0.0006, 0.003, COBALT, n=20, collide=False))


def _tray_shapes() -> list:
    """The foam floor and the frame round it, in the frame of the foam's centre on the table."""
    ix, iy = TRAY_IN
    out = [BoxShape((0.0, 0.0, PAD_H / 2), (ix, iy, PAD_H), FOAM)]
    for sx in (-1, 1):
        out.append(BoxShape((sx * (ix + FRAME_T) / 2, 0.0, FRAME_H / 2), (FRAME_T, iy + 2 * FRAME_T, FRAME_H), FRAME))
    for sy in (-1, 1):
        out.append(BoxShape((0.0, sy * (iy + FRAME_T) / 2, FRAME_H / 2), (ix, FRAME_T, FRAME_H), FRAME))
    return out


JAR = Body.build("lidded_jar", pos=tuple(JAR0 + [0.0, 0.0, 0.0005]), mass=0.35, color=PORCELAIN, friction=(0.6, 0.5),
                 damping=(0.0, 0.5), solver_iterations=24, shapes=_jar_shapes())
LID = Body.build("jar_lid", pos=tuple(LID0 + [0.0, 0.0, 0.0008]), mass=0.06, color=PORCELAIN, friction=(0.6, 0.5),
                 solver_iterations=24, shapes=_lid_shapes())
TRAY = Body.build("padded_tray", pos=tuple(TRAY_C), mass=1.0, color=FOAM, fixed=True, friction=(0.8, 0.7),
                  shapes=_tray_shapes())
MAT = Body.build("felt_mat", pos=(float(JAR0[0]), float(JAR0[1]), TOP), mass=0.05, color=FELT, fixed=True,
                 shapes=[BoxShape((0.0, 0.0, 0.0004), (0.11, 0.11, 0.0008), FELT, collide=False)])
OBJECTS = [JAR, LID, TRAY, MAT]

# -- scoring ----------------------------------------------------------------------------------------------------------
MOVED, LIFTED, JAR_TIP, FELL = 0.02, 0.01, math.radians(10.0), math.radians(30.0)
IN_NECK_R, IN_NECK_Z = 0.015, SKIRT_H - 0.001  # the skirt still round the neck
SEAT_R, SEAT_Z, SEAT_TILT = 0.006, (-0.002, 0.004), math.radians(12.0)
ON_FOAM_Z, UPRIGHT, LID_FLAT = (-0.003, 0.012), math.radians(10.0), math.radians(20.0)
JAR_MARGIN, LID_MARGIN = BODY_W / 2 + 0.003, LID_R - 0.005
CLEAR_KNOB, CLEAR_JAR = 0.04, 0.08
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("JAR_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("JAR_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], in_neck=True, seated=True, ref=None,
               jar_pos=JAR0.copy(), jar_quat=np.array([0.0, 0.0, 0.0, 1.0]), lid_pos=LID0.copy(),
               lid_quat=np.array([0.0, 0.0, 0.0, 1.0]), jar_on=False, lid_on_tray=False, lid_r=0.0, lid_dz=0.0,
               moved=0.0)
    _TCP.clear()


_reset()


def _rot(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _tilt(quat) -> float:
    return math.acos(max(-1.0, min(1.0, _rot(quat)[2, 2])))


def _on_foam(pos, quat, margin: float, base_dz: float, max_tilt: float) -> bool:
    """A body whose origin ``base_dz`` above the foam's top when it rests on it lies inside the frame (its centre
    ``margin`` or more in from the frame's inner faces) and flat on the foam."""
    ix, iy = TRAY_IN
    inside = abs(pos[0] - TRAY_C[0]) <= ix / 2 - margin and abs(pos[1] - TRAY_C[1]) <= iy / 2 - margin
    dz = pos[2] - (PAD_TOP + base_dz)
    return bool(inside and ON_FOAM_Z[0] <= dz <= ON_FOAM_Z[1] and _tilt(quat) <= max_tilt)


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("JAR_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the jar and its lid; the rule."""
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
    jar = as_numpy(base.scene["lidded_jar"].data.root_link_pose_w)[0]
    lid = as_numpy(base.scene["jar_lid"].data.root_link_pose_w)[0]
    jpos, jquat = jar[:3] - origin, jar[3:7]
    lpos, lquat = lid[:3] - origin, lid[3:7]
    rj = _rot(jquat)
    rel = rj.T @ (lpos - (jpos + rj @ np.array([0.0, 0.0, NECK_TOP])))  # the lid's origin over the neck's top
    lid_r, lid_dz = float(np.hypot(rel[0], rel[1])), float(rel[2])
    in_neck = lid_r <= IN_NECK_R and lid_dz <= IN_NECK_Z
    rel_tilt = math.acos(max(-1.0, min(1.0, float((rj.T @ _rot(lquat))[2, 2]))))
    seated = lid_r <= SEAT_R and SEAT_Z[0] <= lid_dz <= SEAT_Z[1] and rel_tilt <= SEAT_TILT
    jar_tilt = _tilt(jquat)
    if in_neck and (_EP["ref"] is None or not _EP["in_neck"]):
        _EP["ref"] = (jpos.copy(), jar_tilt)  # where the jar stood when its lid went (back) on
    moved = 0.0
    if not _EP["failed"]:
        if jar_tilt > FELL:
            _EP["failed"] = f"the jar was knocked over (tipped {math.degrees(jar_tilt):.0f} deg)"
        elif in_neck and _EP["ref"] is not None:
            p0, t0 = _EP["ref"]
            moved = float(np.hypot(*(jpos[:2] - p0[:2])))
            if moved > MOVED or jpos[2] - p0[2] > LIFTED or jar_tilt - t0 > JAR_TIP:
                _EP["failed"] = (f"the jar was moved with its lid on ({moved * 100:.1f} cm along, "
                                 f"{(jpos[2] - p0[2]) * 100:.1f} cm up, tipped {math.degrees(jar_tilt):.0f} deg)")
    jar_on = _on_foam(jpos, jquat, JAR_MARGIN, 0.0, UPRIGHT)
    lid_on_tray = _on_foam(lpos, lquat, LID_MARGIN, SKIRT_H, LID_FLAT)
    _EP.update(in_neck=in_neck, seated=seated, jar_pos=jpos, jar_quat=jquat, lid_pos=lpos, lid_quat=lquat,
               jar_on=jar_on, lid_on_tray=lid_on_tray, lid_r=lid_r, lid_dz=lid_dz, moved=moved)
    stage = 0.25 if not in_neck else 0.0
    if jar_on:
        stage = 0.5
        if lid_on_tray or seated:
            stage = 0.75
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    jar, lid = np.asarray(o["lidded_jar"].pos, float), np.asarray(o["jar_lid"].pos, float)
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([jar, lid]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    knob = _EP["lid_pos"] + _rot(_EP["lid_quat"]) @ LID_KNOB
    body = _EP["jar_pos"] + _rot(_EP["jar_quat"]) @ np.array([0.0, 0.0, BODY_H / 2])
    if len(_TCP) == 2:
        to_knob = min(np.linalg.norm(v - knob) for v in _TCP.values())
        to_jar = min(np.linalg.norm(v - body) for v in _TCP.values())
    else:
        to_knob = to_jar = 0.0
    both = _EP["jar_on"] and (_EP["lid_on_tray"] or _EP["seated"])
    done = both and to_knob >= CLEAR_KNOB and to_jar >= CLEAR_JAR and still
    return dict(still=still, to_knob=float(to_knob), to_jar=float(to_jar), done=bool(done))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    j, lp = _EP["jar_pos"], _EP["lid_pos"]
    print(f"DBG step={_EP['step']} jar=({j[0]:.3f},{j[1]:.3f},{(j[2] - TOP) * 1000:+.1f}mm tilt "
          f"{math.degrees(_tilt(_EP['jar_quat'])):.1f}) lid=({lp[0]:.3f},{lp[1]:.3f},{(lp[2] - TOP) * 1000:.1f}mm tilt "
          f"{math.degrees(_tilt(_EP['lid_quat'])):.1f}) lid_r={_EP['lid_r'] * 1000:.1f}mm lid_dz={_EP['lid_dz'] * 1000:+.1f}mm "
          f"in_neck={_EP['in_neck']} seated={_EP['seated']} moved={_EP['moved'] * 1000:.1f}mm jar_on={_EP['jar_on']} "
          f"lid_on_tray={_EP['lid_on_tray']} knob={t['to_knob']:.3f} jar={t['to_jar']:.3f} still={t['still']} "
          f"best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the jar has been moved with its lid on, or knocked over):
    0.25 the lid lifted off the jar; 0.5 the jar standing on the tray's foam; 0.75 the jar and the lid both on the tray
    (the lid flat on the foam or seated on the jar). Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The jar standing on the tray's foam and its lid on the tray too (flat on the foam or seated on the jar), both
    grippers 4 cm or more from the knob and 8 cm or more from the jar's centre, still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right gripper lifts the lid off by its side (not by the knob: handling guides warn against lifting by knobs,
# handles or rims), carries it to the near end of the tray and sets it down on the foam,
# lets go and moves away; then the left gripper takes the jar by its body, carries it onto the tray, sets it down,
# lets go and moves away. JAR_WITH_LID=1 runs it the wrong way: the left gripper moves the jar with its lid on (the
# right arm stays put); the run scores 0 as soon as the jar is lifted.

WITH_LID = os.environ.get("JAR_WITH_LID") == "1"
OPEN = -0.030
WIDE = -0.0465  # nearly fully open (9.3 cm), round the jar's 7 cm body
PINCH_LID = -(LID_R - 0.003)
PINCH_JAR = -(BODY_W / 2 - 0.003)


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


def _jar_moves(down) -> list[Waypoint]:
    """The left gripper takes the jar by its body, carries it onto the tray and sets it down, lets go, moves away."""
    grip0 = JAR0 + JAR_GRIP
    carry_z = TOP + 0.115
    drop = np.array([JAR_DROP[0], JAR_DROP[1], PAD_TOP + JAR_GRIP[2] + 0.002])
    out = _lead_in("left", grip0 + [0.0, 0.0, 0.07], down, WIDE, "above the jar")
    out += [_w(2.0, grip0, WIDE, "fingers either side of the body", down),
            _w(0.8, grip0, PINCH_JAR, "take the jar by its body", down),
            _w(1.5, np.array([grip0[0], grip0[1], carry_z]), PINCH_JAR, "lift it", down),
            _w(3.0, np.array([drop[0], drop[1], carry_z]), PINCH_JAR, "carry it over the tray", down),
            _w(0.6, np.array([drop[0], drop[1], carry_z]), PINCH_JAR, "let it stop swinging", down),
            _w(1.8, drop, PINCH_JAR, "set it down on the foam", down),
            _w(0.8, drop, WIDE, "let go", down),
            _w(1.5, drop + [0.0, 0.0, 0.08], WIDE, "rise", down),
            _w(1.5, np.array([0.20, 0.27, TOP + 0.16]), OPEN, "move away", down),
            _w(2.0, np.array([0.20, 0.27, TOP + 0.16]), OPEN, "stay clear", down)]
    return out


def demo():
    down = grip_rotation(0.0)  # fingers closing along y
    if WITH_LID:
        left = _jar_moves(down)
        return {"left": left, "right": _idle("right", sum(w.duration for w in left))}

    side0 = LID0 + LID_GRIP + [0.0, 0.0, 0.001]
    up = side0 + [0.0, 0.0, 0.035]  # the skirt (1 cm) well clear of the neck
    drop = np.array([LID_DROP[0], LID_DROP[1], PAD_TOP + SKIRT_H + LID_GRIP[2] + 0.002])
    right = _lead_in("right", side0 + [0.0, 0.0, 0.04], down, WIDE, "above the lid")
    right += [_w(1.5, side0, WIDE, "fingers either side of the lid", down),
              _w(0.8, side0, PINCH_LID, "take the lid by its side", down),
              _w(1.5, up, PINCH_LID, "lift it straight off", down),
              _w(2.5, np.array([drop[0], drop[1], up[2]]), PINCH_LID, "carry it over the tray", down),
              _w(0.5, np.array([drop[0], drop[1], up[2]]), PINCH_LID, "let it stop swinging", down),
              _w(1.5, drop, PINCH_LID, "set it down on the foam", down),
              _w(0.6, drop, WIDE, "let go", down),
              _w(1.0, drop + [0.0, 0.0, 0.05], WIDE, "rise", down),
              _w(1.5, np.array([0.22, -0.20, TOP + 0.15]), OPEN, "move away", down)]
    t_lid = sum(w.duration for w in right)
    left = _idle("left", t_lid) + _jar_moves(down)
    return {"left": left, "right": right}
