"""On the station table, in front of the arms: a cake stand (a pale blue pedestal 10 cm tall) with a round cake pan on it.
The pan holds a plain sponge cake, top (crust) up, and a flat plate lies upside down on the pan, resting on its narrow
rim. The pan has two flat tab handles riveted to its rim, one towards each arm.

Task: turn the cake out onto the plate without damaging it. Hold the plate and the pan together, one hand on each side,
turn them over together, set them down on the stand plate side down, and lift the pan off: the cake ends upside down on
the plate. Plate, pan and cake are loose. Held on one side only the stack hangs off one grip and comes apart, and the
plate and the pan are too wide for one hand: the two hands clamp it from both sides while they turn it.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- stand: fixed pale blue cylinder, radius 6.5 cm, 10 cm tall, centred at (0.30, 0).
- pan: a round pan 14 cm across inside and 3.6 cm deep, 200 g, with a narrow flat rim 7 mm wide (out to 8 cm from its
  centre), as on ordinary round cake tins, and two flat tab handles 3 cm wide on the rim, reaching 13 cm out along the
  left-right line.
- cake: a sponge disc 12.7 cm across and 2.8 cm tall, 100 g, a dark crust on its top. It is two rigid halves, each
  1.4 cm thick, held together by a sprung slide joint along the cake's axis (6,000 N/m: the whole cake squashes 3 mm
  under 18 N spread over its face), free to squash 10 mm and to stretch 4 mm, which breaks for good when the halves
  are sheared, bent or twisted, or pushed against those stops, with more than 40 N or 0.3 N m.
- plate: a flat white disc 21 cm across and 6 mm thick, 220 g, lying on the pan's rim.

The cake is damaged, and the score is 0 for the rest of the run, as soon as (a) any part of either arm presses on it
with more than 3 N (a gripper denting or pinching the sponge), (b) it is squashed by more than 3 mm (its two halves
pressed together), or (c) it breaks (the halves sheared more than 2 mm, bent or twisted more than 3 degrees, or pulled
more than 3 mm apart). Otherwise the best stage reached: 0.25 once the cake has lain upside down on the plate (crust
down, turned more than 160 degrees, its middle within 2.5 cm of the plate's, touching it); 0.5 once it lies like that
on the plate with the plate flat on the stand (its middle within 3 cm of the stand's) and everything still. Full score:
that, with the pan lifted off and away (its middle 15 cm or more from the cake's, so no part of it is over the cake),
both grippers 4 cm or more from the cake and the plate, the cake undamaged and everything still for 2 s.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, on_reset, on_step,  # noqa: E402
                     plate, rotation)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Cylinder  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
# The file keeps the best stage reached itself and damage latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- objects (each body's origin is the centre of its underside, axis up) --------------------------------------------
STAND_XY, STAND_R, STAND_H = (0.30, 0.0), 0.065, 0.10
PAN_R_IN, PAN_T, PAN_FLOOR, PAN_H = 0.070, 0.003, 0.004, 0.036  # inside radius, wall, floor, height to the rim's top
RIM_IN, RIM_OUT, RIM_T = PAN_R_IN + PAN_T, 0.080, 0.003  # the narrow flat rim at the top of the wall (7 mm wide)
EAR_OUT, EAR_W = 0.130, 0.030  # two flat tab handles ("ears") riveted to the rim, along +-y
PAN_KG = 0.20  # 1.9 mm aluminium sheet: floor, wall, rim and handles
CAKE_R, CAKE_H, CAKE_KG = 0.0635, 0.028, 0.10  # a 12.7 cm (5 in) disc cut from a sponge layer and levelled to 2.8 cm
HALF = CAKE_H / 2
# The two halves' joint: a spring along the cake's axis (a sponge about 13 kPa stiff over the cake's 127 cm2: 6,000 N/m),
# well damped, free to squash 10 mm and to stretch 4 mm; sheared, bent or twisted (the locked directions) or pushed
# against those limits harder than this, it breaks for good: the cake has split.
CAKE_K, CAKE_DAMP = 6000.0, 10.0
SQUASH_TRAVEL, STRETCH_TRAVEL = 0.010, 0.004
BREAK_N, BREAK_NM = 40.0, 0.3
PLATE_R, PLATE_T, PLATE_KG = 0.105, 0.006, 0.22

PAN_GREY, CRUST, CAKE_C, WHITE, STAND_C = (0.62, 0.63, 0.66), (0.36, 0.20, 0.10), (0.80, 0.62, 0.38), (0.94, 0.94, 0.92), \
    (0.52, 0.62, 0.72)

STAND_TOP = TOP + STAND_H
CAKE_Z0 = STAND_TOP + PAN_FLOOR + 0.001  # the cake's underside, 1 mm above the pan's floor


def _pan() -> Body:
    n = 24
    shapes = [CylShape((0.0, 0.0, PAN_FLOOR / 2), PAN_R_IN + PAN_T, PAN_FLOOR, PAN_GREY)]
    r_wall = PAN_R_IN + PAN_T / 2
    z_rim = PAN_H - RIM_T / 2
    for k in range(n):
        a = 2 * math.pi * k / n
        u = (math.cos(a), math.sin(a), 0.0)
        shapes.append(plate((r_wall * u[0], r_wall * u[1], PAN_FLOOR), (r_wall * u[0], r_wall * u[1], PAN_H - RIM_T),
                            2 * r_wall * math.tan(math.pi / n) + 0.001, PAN_T, u, PAN_GREY))
        shapes.append(plate((RIM_IN * u[0], RIM_IN * u[1], z_rim), (RIM_OUT * u[0], RIM_OUT * u[1], z_rim),
                            2 * RIM_OUT * math.tan(math.pi / n) + 0.001, RIM_T, (0.0, 0.0, 1.0), PAN_GREY))
    for s in (1, -1):
        shapes.append(BoxShape((0.0, s * (RIM_OUT + EAR_OUT) / 2, z_rim), (EAR_W, EAR_OUT - RIM_OUT + 0.004, RIM_T),
                               PAN_GREY))
    return Body.build("pan", pos=(*STAND_XY, STAND_TOP + 0.0005), mass=PAN_KG, shapes=shapes, friction=(0.6, 0.5),
                      color=PAN_GREY)


def _cake_half(name: str, z: float, crust: bool, joint: Joint | None = None) -> Body:
    shapes = [CylShape((0.0, 0.0, HALF / 2), CAKE_R, HALF, CAKE_C)]
    if crust:
        shapes.append(CylShape((0.0, 0.0, HALF - 0.0015), CAKE_R - 0.001, 0.0035, CRUST, collide=False))
    return Body.build(name, pos=(*STAND_XY, z), mass=CAKE_KG / 2, shapes=shapes, friction=(0.7, 0.6), color=CAKE_C,
                      joint=joint)


def _plate() -> Body:
    shapes = [CylShape((0.0, 0.0, PLATE_T / 2), PLATE_R, PLATE_T, WHITE)]
    return Body.build("plate", pos=(*STAND_XY, STAND_TOP + PAN_H + 0.001), mass=PLATE_KG, shapes=shapes,
                      friction=(0.6, 0.5), color=WHITE)


PAN = _pan()
CAKE = _cake_half("cake", CAKE_Z0, crust=False)  # the lower half, as baked
CAKE_TOP = _cake_half("cake_top", CAKE_Z0 + HALF, crust=True,  # the upper half, with the crust
                      joint=Joint(anchor=(*STAND_XY, CAKE_Z0 + HALF), kind="slide", axis="Z", parent="cake",
                                  axial=(CAKE_K, CAKE_DAMP, (-SQUASH_TRAVEL, STRETCH_TRAVEL)), break_force=BREAK_N,
                                  break_torque=BREAK_NM))
PLATE = _plate()
OBJECTS = [Cylinder("cake_stand", pos=(*STAND_XY, TOP + STAND_H / 2), radius=STAND_R, height=STAND_H, color=STAND_C),
           PAN, CAKE, CAKE_TOP, PLATE]

# -- scoring --------------------------------------------------------------------------------------------------------
TURNED = math.radians(160)  # the cake's axis turned at least this far from up: crust down
FLAT = math.radians(10)
ON_PLATE_XY = 0.025
ON_STAND_XY = 0.03
TOUCH = (-0.002, 0.003)  # resting on a surface: its face this close to that surface (m)
PAN_AWAY = 0.15  # cake radius 6.35 cm + pan rim radius 8 cm = 14.35 cm: no part of the pan is over the cake
HANDS_CLEAR = 0.04
PRESS_N = 3.0  # an arm pressing on the cake harder than this dents the sponge
SQUASH_MM = 3.0
SHEAR_MM, BEND_DEG, TORN_MM = 2.0, 3.0, 3.0  # the halves this far out of line: the joint has broken
ARM_PRIMS = ("LeftArm", "RightArm")  # the two arms' prims in the env (yam_arena.embodiment.SIDES)

_DEBUG = os.environ.get("CAKE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("CAKE_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}
_PRESS: dict = {}  # the contact view that measures how hard the arms press on the cake, and this step's force


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, failed="", best=0.0, peak_press=0.0, peak_squash=0.0, done=False, t={})
    _TCP.clear()
    _PRESS["now"] = 0.0


_reset()


def _press(base) -> float:
    """The largest contact force (N) with which any link of either arm presses on either half of the cake, from
    PhysX's pair-wise contact forces in the last physics step (yam_arena spawns the arms with contact reporting on)."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    view = _PRESS.get("view") if _PRESS.get("sim") is sim else None
    if view is None:
        from yam_arena.robot import asset_link_paths

        env = base.scene.env_prim_paths[0]
        links = [f"{env}/{arm}{path}" for arm in ARM_PRIMS for path in asset_link_paths().values()]
        halves = [f"{env}/cake/body", f"{env}/cake_top/body"]
        view = sim.create_rigid_contact_view(links, filter_patterns=[halves] * len(links),
                                             max_contact_data_count=8 * len(links) * len(halves))
        if view.sensor_count != len(links) or view.filter_count != len(halves):
            raise RuntimeError(f"cake press: contact view has {view.sensor_count} links x {view.filter_count} "
                               f"halves, expected {len(links)} x {len(halves)}")
        _PRESS.update(sim=sim, view=view, dt=float(base.physics_dt))
        if _DEBUG:
            print(f"DBG press view: {view.sensor_count} arm links x {view.filter_count} cake halves, "
                  f"dt {_PRESS['dt']:.5f} s", flush=True)
    forces = view.get_contact_force_matrix(dt=_PRESS["dt"])
    a = forces.numpy() if hasattr(forces, "numpy") else as_numpy(forces)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).max())


@on_step
def _track(env) -> None:
    """Each gripper's centre (TCP) from its arm's joint positions, and how hard the arms press on the cake: more than
    PRESS_N dents it, for the rest of the episode."""
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
    f = _press(base)
    _PRESS["now"] = f
    _EP["peak_press"] = max(_EP["peak_press"], f)
    if f > PRESS_N and not _EP["failed"]:
        _EP["failed"] = f"an arm pressed on the cake with {f:.1f} N (more than {PRESS_N:.0f} N): it is dented"


def _cake_joint(o) -> dict:
    """The two halves seen from each other: squash (mm, the halves pressed together), stretch (mm), shear (mm), bend and
    twist (deg), from their poses."""
    b, t = o["cake"], o["cake_top"]
    rb, rt = rotation(b.quat_xyzw), rotation(t.quat_xyzw)
    rel = rb.T @ (np.asarray(t.pos) - np.asarray(b.pos)) - np.array([0.0, 0.0, HALF])
    rr = rb.T @ rt
    return dict(squash=max(0.0, -rel[2]) * 1000, stretch=max(0.0, rel[2]) * 1000,
                shear=float(math.hypot(rel[0], rel[1])) * 1000,
                bend=math.degrees(math.acos(max(-1.0, min(1.0, rr[2, 2])))),
                twist=abs(math.degrees(math.atan2(rr[1, 0], rr[0, 0]))))


def _on(face_point, s) -> tuple[float, float]:
    """(distance from the plate ``s``'s axis, gap to its nearer flat face) of a point, in the plate's own frame."""
    loc = rotation(s.quat_xyzw).T @ (np.asarray(face_point, float) - np.asarray(s.pos))
    return float(math.hypot(loc[0], loc[1])), abs(loc[2] - PLATE_T / 2) - PLATE_T / 2


def _clear(tcp, centre, r_axis, radius: float, half_t: float) -> float:
    """Distance from a point to a flat cylinder (centre, axis, radius, half thickness)."""
    d = np.asarray(tcp, float) - centre
    along = float(np.dot(d, r_axis))
    radial = float(np.linalg.norm(d - along * r_axis))
    return float(math.hypot(max(0.0, radial - radius), max(0.0, abs(along) - half_t)))


def _tally(o) -> dict:
    b, t, p, pan = o["cake"], o["cake_top"], o["plate"], o["pan"]
    rb, rp = rotation(b.quat_xyzw), rotation(p.quat_xyzw)
    j = _cake_joint(o)
    _EP["peak_squash"] = max(_EP["peak_squash"], j["squash"])
    if not _EP["failed"]:
        if j["shear"] > SHEAR_MM or j["bend"] > BEND_DEG or j["twist"] > BEND_DEG or j["stretch"] > TORN_MM \
                or j["squash"] > SQUASH_TRAVEL * 1000 + 1.0:
            _EP["failed"] = (f"the cake broke (halves sheared {j['shear']:.1f} mm, bent {j['bend']:.1f} deg, twisted "
                             f"{j['twist']:.1f} deg, pulled {j['stretch']:.1f} mm apart)")
        elif j["squash"] > SQUASH_MM:
            _EP["failed"] = f"the cake was squashed by {j['squash']:.1f} mm (more than {SQUASH_MM:.0f} mm)"
    mid = np.asarray(b.pos) + rb @ np.array([0.0, 0.0, HALF])  # the middle of the cake (between the halves)
    axis = rb[:, 2]
    crust = np.asarray(t.pos) + rotation(t.quat_xyzw) @ np.array([0.0, 0.0, HALF])  # the centre of the crust face
    upside_down = axis[2] <= math.cos(TURNED)
    off_axis, gap = _on(crust, p)
    on_plate = bool(upside_down and off_axis <= ON_PLATE_XY and TOUCH[0] <= gap <= TOUCH[1])
    plate_on_stand = bool(abs(rp[2, 2]) >= math.cos(FLAT)
                          and math.hypot(p.pos[0] - STAND_XY[0], p.pos[1] - STAND_XY[1]) <= ON_STAND_XY
                          and STAND_TOP + TOUCH[0] <= p.low[2] <= STAND_TOP + TOUCH[1])
    still = all(o[n].resting() for n in ("pan", "cake", "cake_top", "plate"))
    pan_away = math.hypot(pan.pos[0] - mid[0], pan.pos[1] - mid[1]) >= PAN_AWAY
    hands = min((min(_clear(v, mid, axis, CAKE_R, HALF),
                     _clear(v, np.asarray(p.pos) + rp[:, 2] * PLATE_T / 2, rp[:, 2], PLATE_R, PLATE_T / 2))
                 for v in _TCP.values()), default=0.0) if len(_TCP) == 2 else 0.0
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        now = 0.25 * on_plate + 0.25 * (on_plate and plate_on_stand and still)
        _EP["best"] = max(_EP["best"], now)
    done = on_plate and plate_on_stand and still and pan_away and hands >= HANDS_CLEAR and not _EP["failed"]
    out = dict(j=j, mid=mid, axis=axis, on_plate=on_plate, gap=gap, off_axis=off_axis, plate_on_stand=plate_on_stand,
               still=still, pan_away=pan_away, hands=hands, done=bool(done))
    _EP["t"] = out
    return out


def _debug(o, t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    parts = []
    for n in ("pan", "cake", "cake_top", "plate"):
        s = o[n]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, rotation(s.quat_xyzw)[2, 2]))))
        parts.append(f"{n}=({s.pos[0]:.3f},{s.pos[1]:.3f},{s.pos[2] - TOP:.3f}) tilt={tilt:.0f}")
    j = t["j"]
    print(f"DBG step={_EP['step']} {' | '.join(parts)}", flush=True)
    print(f"DBG   on_plate={t['on_plate']} (off_axis={t['off_axis'] * 100:.1f}cm gap={t['gap'] * 1000:.1f}mm) "
          f"on_stand={t['plate_on_stand']} still={t['still']} pan_away={t['pan_away']} hands={t['hands']:.3f} "
          f"best={_EP['best']} done={t['done']} failed={_EP['failed'] or '-'}", flush=True)
    print(f"DBG   cake squash={j['squash']:.2f}mm (peak {_EP['peak_squash']:.2f}) stretch={j['stretch']:.2f} "
          f"shear={j['shear']:.2f}mm bend={j['bend']:.2f} twist={j['twist']:.2f} press={_PRESS['now']:.2f}N "
          f"(peak {_EP['peak_press']:.2f})", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def success(objects) -> bool:
    """The cake upside down on the plate on the stand, undamaged, the pan off and away, both grippers clear, all still
    (held 2 s)."""
    return bool(_tally(objects)["done"])


def score(objects) -> float:
    """0 for the rest of the run once the cake is damaged (pressed by an arm with more than 3 N, squashed by more than
    3 mm, or broken). Otherwise the best stage reached: 0.25 once the cake has lain upside down on the plate, 0.5 once
    it lies like that on the plate flat on the stand, all still. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(objects, t)
    return 0.0 if _EP["failed"] else _EP["best"]


# -- scripted demo --------------------------------------------------------------------------------------------------
# Both hands come in level from their own sides with the fingers upright and clamp the plate and the pan's handle
# under it together, one hand on each side; they lift the stack, turn it over together about the line between them, lower it
# plate side down onto the stand and let go. Then each takes one of the pan's ears from above, they lift the pan off
# the cake together, carry it towards the arms and set it down upside down on the table.


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


SIDE = {"left": _rx(-math.pi / 2) @ grip_rotation(-math.pi / 2),  # pointing -y, fingers closing front and back
        "right": _rx(math.pi / 2) @ grip_rotation(math.pi / 2)}  # pointing +y
SIDE_ARM = {"left": 1, "right": -1}
TOPDOWN = {"left": grip_rotation(-math.pi / 2), "right": grip_rotation(math.pi / 2)}  # fingers closing along x


def _clamp_rot(arm: str, turn: float) -> np.ndarray:
    """Side grip with the fingers upright (closing along z), turned ``turn`` about world y: -90 deg at the start,
    +90 deg once the stack is upside down."""
    return _ry(turn) @ SIDE[arm]


GRIP_Y = float(os.environ.get("CAKE_GRIP_Y", "0.105"))  # TCPs this far either side of the centre line
OUT_Y = 0.12  # beyond the plate (radius 10.5 cm); a side grip reaches no further out than ~13 cm
BESIDE_X = 0.035  # wait this far in front of the ear (3 cm wide), so the fingers come down clear of it
STACK_MID = PAN_H + 0.0015  # the plate / handle interface, above the pan's underside
CLAMP = float(os.environ.get("CAKE_CLAMP", "-0.0025"))  # 5 mm between the fingers round the 9 mm stack
OPEN_SIDE = -0.02  # 4 cm apart: enough round the stack, small sweep
FLIP_Z = STAND_TOP + STACK_MID + float(os.environ.get("CAKE_LIFT", "0.08"))  # turn it over here: clear of the stand
# Turned over, the plate is below the stack's middle (6 mm thick, the middle 1.5 mm above its top): set down when its
# underside reaches the stand, pressing 1 mm.
SET_Z = STAND_TOP + 0.0045 - 0.001 + float(os.environ.get("CAKE_SET_DZ", "0.0"))
PLATE_TOP_DOWN = STAND_TOP + PLATE_T  # the plate's top once it lies on the stand: the pan's rim and ears rest on it
EAR_Y = 0.118  # grip the ears here: outside the plate (radius 10.5 cm)
EAR_GRIP = float(os.environ.get("CAKE_EAR_GRIP", "-0.011"))  # 2.2 cm between the fingers round the 3 cm ear
TIP = 0.013
# Set the pan down this far forward: its rim (8 cm) then passes 3.5 cm short of the plate's edge on the way down.
PARK_X = float(os.environ.get("CAKE_PARK_X", "0.08"))


def _good_ik(arm: str, pos, rot, seed=None):
    import itertools

    from yam_arena import ik, station

    seeds = [seed] if seed is not None else []
    seeds += list(itertools.product((0.0,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2), (-1.5, 0.0, 1.5)))
    best = None
    for sd in seeds:
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=station.BASES[arm], iters=150)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
            if seed is not None and best is not None:
                break
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _joint_move(arm: str, q0, q1, seconds: float, steps: int, finger: float, label: str) -> list[Waypoint]:
    """Waypoints along a straight line in joint space (big reorientations stay on one IK branch)."""
    from yam_arena import ik, station

    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, pos, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)


def _arm_plan(arm: str) -> list[Waypoint]:
    from yam_arena.constants import READY

    s = SIDE_ARM[arm]
    x0, _ = STAND_XY
    grip = np.array([x0, s * GRIP_Y, STAND_TOP + STACK_MID])
    beside = np.array([x0, s * OUT_Y, STAND_TOP + STACK_MID])
    wait = beside + [BESIDE_X, 0.0, 0.0]
    up = np.array([x0, s * GRIP_Y, FLIP_Z])
    down = np.array([x0, s * GRIP_Y, SET_Z])
    start = _clamp_rot(arm, -math.pi / 2)
    turned = _clamp_rot(arm, math.pi / 2)
    plan = _joint_move(arm, np.array(READY, float), _good_ik(arm, wait, start), 3.0, 15, OPEN_SIDE,
                       "level, beside the stack, fingers upright")
    plan += [
        _w(1.2, beside, OPEN_SIDE, "slide along, the ear between the fingers", start),
        _w(1.2, grip, OPEN_SIDE, "slide in round the plate and the pan's handle", start),
        _w(1.0, grip, CLAMP, "clamp them together", start),
        _w(0.4, grip, CLAMP, "steady", start),
        _w(1.5, up, CLAMP, "lift the stack off the stand", start),
    ]
    for k, deg in enumerate(range(-90 + 15, 91, 15)):
        plan.append(_w(4.0 / 12, up, CLAMP, "turn it over together", _clamp_rot(arm, math.radians(deg))))
    plan += [
        _w(0.6, up, CLAMP, "steady", turned),
        _w(2.0, down + [0.0, 0.0, 0.02], CLAMP, "lower it over the stand, plate side down", turned),
        _w(1.2, down, CLAMP, "set it down", turned),
        _w(0.6, down, CLAMP, "settle", turned),
        _w(0.8, down, OPEN_SIDE, "let go", turned),
        _w(1.0, down + [0.0, s * (OUT_Y - GRIP_Y), 0.0], OPEN_SIDE, "slide out", turned),
        _w(1.0, down + [BESIDE_X, s * (OUT_Y - GRIP_Y), 0.0], OPEN_SIDE, "slide along, clear of the ear", turned),
        _w(1.0, down + [BESIDE_X, s * (OUT_Y - GRIP_Y), 0.06], OPEN_SIDE, "rise", turned),
    ]
    # over the ear, from above: a joint-space move from the side grip to a downward grip
    ear_above = np.array([x0, s * EAR_Y, PLATE_TOP_DOWN + 0.07])
    q_out = _good_ik(arm, down + [BESIDE_X, s * (OUT_Y - GRIP_Y), 0.06], turned)
    q_ear = _good_ik(arm, ear_above, TOPDOWN[arm])
    plan += _joint_move(arm, q_out, q_ear, 2.4, 12, FINGER_OPEN, "turn the hand to come down on the ear")
    ear_z = PLATE_TOP_DOWN - 0.002 + TIP  # the pads' tips 2 mm below the ear (it overhangs the plate's edge)
    ear = np.array([x0, s * EAR_Y, ear_z])
    lifted = ear + [0.0, 0.0, 0.07]
    park = np.array([PARK_X, s * EAR_Y, TOP + 0.010 - 0.002 + TIP])  # the rim 1 cm above the table, then let go
    plan += [
        _w(1.2, ear, FINGER_OPEN, "down round the ear", TOPDOWN[arm]),
        _w(1.0, ear, EAR_GRIP, "grip the ear", TOPDOWN[arm]),
        _w(0.4, ear, EAR_GRIP, "steady", TOPDOWN[arm]),
        _w(1.5, lifted, EAR_GRIP, "lift the pan off the cake", TOPDOWN[arm]),
        _w(2.0, [PARK_X, s * EAR_Y, lifted[2]], EAR_GRIP, "carry it towards the arms, over the plate", TOPDOWN[arm]),
        _w(1.5, park, EAR_GRIP, "lower it past the plate's edge", TOPDOWN[arm]),
        _w(0.8, park, FINGER_OPEN, "let go", TOPDOWN[arm]),
        _w(1.0, park + [0.0, 0.0, 0.06], FINGER_OPEN, "rise", TOPDOWN[arm]),
    ]
    return plan


# CAKE_SQUASH=1 (a debug run for the damage check): once the pan is down, the right gripper lets go of it, comes back
# over the turned-out cake with its fingers closed, presses down on it 2.5 cm past its top, holds 1 s and rises, while
# the left gripper keeps the pan sliding to and fro on the table (so the run does not end on success first) and then
# lets go: the cake ends where full score needs it, but it was pressed and squashed, so the run scores 0.
# CAKE_ONE=1 (the wrong way): only the left arm works; the right arm stays at its ready pose. The left gripper clamps the
# plate and the pan's handle on its side alone and tries the same moves: held on one side, the stack hangs off that grip
# and comes apart.
SQUASH = os.environ.get("CAKE_SQUASH") == "1"
ONE_ARM = os.environ.get("CAKE_ONE") == "1"
SQUASH_DEPTH = float(os.environ.get("CAKE_SQUASH_DEPTH", "0.025"))


def _squash_plan(arm: str, last: Waypoint) -> list[Waypoint]:
    """From the end of the arm's plan: over the turned-out cake, fingers closed, press down on it, rise."""
    x0, y0 = STAND_XY[0] - 0.02, STAND_XY[1] - 0.02  # 2.8 cm from the cake's middle, towards the right arm
    top = STAND_TOP + PLATE_T + CAKE_H  # the turned-out cake's top
    above = np.array([x0, y0, top + TIP + 0.03])
    press = np.array([x0, y0, top + TIP - SQUASH_DEPTH])
    q_last = _good_ik(arm, last.pos, np.asarray(last.yaw, float))
    plan = _joint_move(arm, q_last, _good_ik(arm, above, TOPDOWN[arm]), 2.0, 10, FINGER_CLOSED,
                       "over the cake, fingers closed")
    plan += [
        _w(1.5, above + [0.0, 0.0, -0.03], FINGER_CLOSED, "down to the cake", TOPDOWN[arm]),
        _w(1.5, press, FINGER_CLOSED, "press down on it", TOPDOWN[arm]),
        _w(1.0, press, FINGER_CLOSED, "hold", TOPDOWN[arm]),
        _w(1.5, above, FINGER_CLOSED, "rise", TOPDOWN[arm]),
        _w(1.0, above + [-0.12, -0.10, 0.0], FINGER_CLOSED, "away", TOPDOWN[arm]),
    ]
    return plan


def _stay(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [_w(seconds, pos, FINGER_OPEN, "stay at the ready pose", rot)]


def _keep_pan_moving(arm: str, plan: list[Waypoint], seconds: float) -> list[Waypoint]:
    """Instead of letting go of the pan once it is down, keep holding its ear and slide it 1.5 cm to and fro on the
    table for ``seconds`` (so the scene is not yet still and the run goes on), then let go and rise."""
    park = np.asarray(plan[-1].pos, float) - [0.0, 0.0, 0.06]  # the plan ends rising 6 cm from the park
    out = plan[:-2]
    legs = max(2, int(round(seconds)))
    for k in range(legs):
        out.append(_w(1.0, park + [0.0, 0.015 * (1 if k % 2 == 0 else -1), 0.0], EAR_GRIP, "slide the pan to and fro",
                      TOPDOWN[arm]))
    out += [_w(0.6, park, EAR_GRIP, "steady", TOPDOWN[arm]),
            _w(0.8, park, FINGER_OPEN, "let go", TOPDOWN[arm]),
            _w(1.0, park + [0.0, 0.0, 0.06], FINGER_OPEN, "rise", TOPDOWN[arm])]
    return out


def demo():
    left, right = _arm_plan("left"), _arm_plan("right")
    if SQUASH:
        right += _squash_plan("right", right[-1])
        left = _keep_pan_moving("left", left, sum(w.duration for w in right) - sum(w.duration for w in left) + 1.8)
    if ONE_ARM:
        right = _stay("right", sum(w.duration for w in left))
    return {"left": left, "right": right}
