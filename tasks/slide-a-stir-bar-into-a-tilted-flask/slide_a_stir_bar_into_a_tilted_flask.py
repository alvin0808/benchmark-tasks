"""On the station table: an empty 250 mL round-bottom flask standing in a cork ring, and a PTFE magnetic stir bar
lying on the table to the left of it.

Task (instruction): put the stir bar into the round-bottom flask and leave the flask standing in its cork ring. The
instruction does not say how. A stir bar dropped down the neck of the upright flask falls about 10 cm onto the flask's
thin round bottom; chemistry teaching labs say to slide it in instead (University of York, Chemistry Teaching Labs,
"Stirring & heating": "Gently slide the magnetic stirrer bar into the glassware", "dropping it in may cause damage";
its photo shows a round-bottom flask held on its side while the bar is guided into the neck). A round-bottom flask
cannot stand outside its cork ring and rolls if laid on the table, so one hand holds it by the neck, lifted out of the
ring and turned well over toward the other hand, which lays the bar in the neck; the bar slides down the neck and the
inside wall; the flask is turned upright slowly and set back in its ring. One hand cannot hold the flask turned over
and lay the bar in at once.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- cork ring: fixed, 10 cm across, 3.4 cm tall, centred at x 0.28, y -0.05; the flask's bulb rests on its sloping
  inner seat, the bulb's middle 5.9 cm above the table.
- flask: 0.12 kg, a borosilicate round-bottom flask 14 cm tall: an 8.5 cm bulb (2 mm glass, modelled as a shell of
  flat plates) and a neck 2.8 cm across with a 2.2 cm bore, its top 4 cm the ground-glass 24/40 joint. Loose.
- stir bar: 5 g, 25 mm long and 8 mm across with rounded ends (PTFE), lying on the table at x 0.27, y 0.12, along y.

Score: 0 for the rest of the run once the bar has met the bulb's inside wall at 1 m/s or faster (dropped in rather
than slid in), the flask has touched the table (its lowest point within 4 mm of it), dropped (moved down faster than
0.6 m/s while not held) or tipped over (tilted more than 45 degrees while not held), or the bar, once wholly inside,
has come back out. Otherwise the best stage reached: 0.25 the flask lifted out of its ring (its lowest point 3.9 cm or
more above the table) and tilted 60 degrees or more; 0.5 the bar wholly inside the flask; 0.75 the flask upright
(tilted less than 10 degrees) in its cork ring with the bar in its bulb, at rest, not held. Full score: that, with
both grippers' centres 6 cm or more from the flask's glass, still for 1 s. Held: a gripper's centre within 1.5 cm of
the neck's axis (or of the bulb's middle), its fingers closed on it (their gap from 4 mm less to 6 mm more than the
neck's or the bulb's width), the flask moving with it.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CapsuleShape, CylShape, allow_matrix_grips, as_numpy  # noqa: E402
from bodykit import on_reset, on_step, plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

# -- the flask (frame: the bulb's middle, neck along +z) ------------------------------------------------------------
R_OUT, GLASS = 0.0425, 0.002  # 250 mL round-bottom flask: 85 mm across, 2 mm wall
R_IN = R_OUT - GLASS
NECK_RO, NECK_RI = 0.014, 0.011  # neck 28 mm across, 22 mm bore (the 24/40 joint's bore narrows from 24 to 20 mm)
NECK_TOP = 0.0975  # the mouth, above the bulb's middle: the flask is 140 mm tall
JOINT_L = 0.040  # the ground-glass joint: the neck's top 40 mm
N_NECK = 12
Z_J = math.sqrt(R_IN ** 2 - NECK_RI ** 2)  # where the bore meets the bulb's inside
FLASK_KG = 0.12
# -- the stir bar (frame: its middle, lying along x) -----------------------------------------------------------------
BAR_R, BAR_L, BAR_KG = 0.004, 0.025, 0.005
BAR_SEG = BAR_L / 2 - BAR_R  # its two end spheres' middles, along its axis
# -- the cork ring (frame: its base's middle) -------------------------------------------------------------------------
RING_RI, RING_RO, RING_H, N_RING = 0.022, 0.050, 0.034, 16
SEAT = ((0.022, 0.020), (0.036, 0.032))  # the sloping seat, (r, z) at its inner and outer edges
# -- layout ----------------------------------------------------------------------------------------------------------
X_F, Y_F = 0.28, -0.05  # the ring and the flask
SEATED_Z = 0.0587  # the bulb's middle above the table when the flask rests in its ring (the plates' corners)
X_B, Y_B = 0.27, 0.12  # the stir bar

GLASS_C, FROST_C, PTFE, CORK = (0.78, 0.88, 0.96), (0.90, 0.92, 0.93), (0.96, 0.96, 0.94), (0.50, 0.37, 0.25)


def _bands():
    """The bulb's latitude bands (polar angle from the neck, rad) and the number of plates in each."""
    edges = [math.degrees(math.asin(NECK_RI / R_IN))] + list(range(30, 166, 15))
    out = []
    for t1, t2 in zip(edges[:-1], edges[1:]):
        s = max(math.sin(math.radians(t1)), math.sin(math.radians(t2)))
        out.append((math.radians(t1), math.radians(t2), max(8, min(18, round(2 * math.pi * R_IN * s / 0.0135)))))
    return out


def _flask_shapes() -> list:
    """The flask in the frame of its bulb's middle: the bulb as a shell of flat glass plates whose inner faces touch an
    81 mm sphere (no gaps: they overlap behind each other's inner faces), a round plate at the bottom, and the neck as
    a ring of 12 plates whose top 40 mm is frosted (the ground joint)."""
    out = []
    ov = 0.0016
    for i, (t1, t2, n) in enumerate(_bands()):
        tm, dphi = (t1 + t2) / 2, 2 * math.pi / n
        half = R_IN * math.tan((t2 - t1) / 2)
        l_up, l_down = half + (0.0 if i == 0 else ov), half + ov  # the top band stops at the bore
        width = 2 * R_IN * max(math.sin(t1), math.sin(t2)) * math.tan(dphi / 2) / math.cos((t2 - t1) / 2) + 2 * ov
        for k in range(n):
            ph = (k + 0.5) * dphi
            d = np.array([math.sin(tm) * math.cos(ph), math.sin(tm) * math.sin(ph), math.cos(tm)])
            tang = np.array([math.cos(tm) * math.cos(ph), math.cos(tm) * math.sin(ph), -math.sin(tm)])
            c = d * (R_IN + GLASS / 2)
            out.append(plate(c - tang * l_up, c + tang * l_down, width, GLASS, d, GLASS_C, look="glass"))
    out.append(CylShape((0.0, 0.0, -(R_IN + GLASS / 2)), R_IN * math.sin(math.radians(15)) + 0.0025, GLASS, GLASS_C,
                        look="glass"))
    t = NECK_RO - NECK_RI
    w = 2 * (NECK_RI + t / 2) * math.tan(math.pi / N_NECK) + 0.0012
    z0, zj = Z_J - 0.0015, NECK_TOP - JOINT_L
    for k in range(N_NECK):
        ph = (k + 0.5) * 2 * math.pi / N_NECK
        d = np.array([math.cos(ph), math.sin(ph), 0.0])
        c = d * (NECK_RI + t / 2)
        out.append(plate(c + [0, 0, z0], c + [0, 0, zj], w, t, d, GLASS_C, look="glass"))
        out.append(plate(c + [0, 0, zj], c + [0, 0, NECK_TOP], w, t, d, FROST_C, look="frosted"))
    return out


def _ring_shapes() -> list:
    """The cork ring in the frame of its base's middle: in each of 16 segments a foot, the sloping seat the bulb rests
    on, and the outer top."""
    out = []
    (r0, z0), (r1, z1) = SEAT
    along = np.array([r1 - r0, z1 - z0]) / math.hypot(r1 - r0, z1 - z0)
    for k in range(N_RING):
        ph = (k + 0.5) * 2 * math.pi / N_RING
        d = np.array([math.cos(ph), math.sin(ph), 0.0])
        q = (0.0, 0.0, math.sin(ph / 2), math.cos(ph / 2))

        def wide(r):
            return 2 * r * math.tan(math.pi / N_RING) + 0.002

        out.append(BoxShape(tuple((d * (RING_RI + RING_RO) / 2 + [0, 0, z0 / 2]).tolist()),
                            (RING_RO - RING_RI, wide(RING_RO), z0), CORK, quat_xyzw=q))
        out.append(BoxShape(tuple((d * (r1 + RING_RO) / 2 + [0, 0, (z0 + RING_H) / 2]).tolist()),
                            (RING_RO - r1, wide(RING_RO), RING_H - z0), CORK, quat_xyzw=q))
        n3 = -along[1] * d + np.array([0.0, 0.0, along[0]])  # the seat's normal: inward and up
        th = 0.008
        out.append(plate(d * r0 + [0, 0, z0] - n3 * th / 2, d * r1 + [0, 0, z1] - n3 * th / 2, wide(r1), th, n3, CORK))
    return out


RING = Body.build("cork_ring", pos=(X_F, Y_F, TOP), mass=0.05, color=CORK, fixed=True, friction=(0.8, 0.7),
                  shapes=_ring_shapes())
FLASK_START = np.array([X_F, Y_F, TOP + SEATED_Z + 0.0008])
FLASK = Body.build("flask", pos=tuple(float(v) for v in FLASK_START), mass=FLASK_KG, color=GLASS_C,
                   friction=(0.12, 0.10), friction_combine="min", solver_iterations=32, shapes=_flask_shapes())
BAR_START = np.array([X_B, Y_B, TOP + BAR_R + 0.0005])
BAR = Body.build("stir_bar", pos=tuple(float(v) for v in BAR_START), mass=BAR_KG, color=PTFE, yaw=math.pi / 2,
                 friction=(0.6, 0.5), solver_iterations=32, contact_offset=0.008,
                 shapes=[CapsuleShape((0.0, 0.0, 0.0), BAR_R, BAR_L - 2 * BAR_R, PTFE, axis="X")])
OBJECTS = [RING, FLASK, BAR]

# -- scoring ----------------------------------------------------------------------------------------------------------
V_HARD = 1.0  # m/s: the bar meeting the bulb's inside wall this fast has been dropped in (a free fall of 5 cm)
NEAR = 0.012  # the bar's end within this of the inside wall: its speed toward the wall counts
LIFTED_Z = RING_H + 0.005  # lifted: the flask's lowest point this far above the table (clear of the ring)
TILTED = math.radians(60.0)
UPRIGHT = math.radians(10.0)
SEAT_XY, SEAT_DZ = 0.010, 0.008  # in its ring: the bulb's middle within 1 cm of the ring's axis and 8 mm of the
# height at which it rests there
DOWN_Z = 0.004  # fallen: the flask's lowest point this close to the table
DROP_V = 0.6  # dropped: moving down this fast while not held
TIPPED = math.radians(45.0)  # tipped over: tilted this far while not held
HELD_AT, HELD_SQUEEZE, HELD_SLACK, WITH_MM = 0.015, 0.004, 0.006, 5.0  # held: a gripper's centre within 1.5 cm of
# the neck's axis (or of the bulb's middle), its fingers closed on it (their gap from 4 mm less to 6 mm more than the
# neck's or the bulb's width), the flask moving with it (its offset from the gripper changing less than 5 mm a step)
REST_MM, REST_STEPS = 2.0, 3
CLEAR = 0.06  # full score: both grippers' centres this far from the flask's glass
STILL_STEPS, STILL_MM = 20, 1.5
_DEBUG = os.environ.get("STIRBAR_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("STIRBAR_DEBUG_EVERY", "10"))
_EP: dict = {}
_SUB: dict = {}
_TCP: dict = {}
_TCPR: dict = {}
_GAP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], hist=[], offsets={}, pos=FLASK_START.copy(),
               rot=np.eye(3), tilt=0.0, low=SEATED_Z - R_OUT, lifted=False, seated=False, held=False, was_in=False,
               bar_where="table", done=False, neck=(FLASK_START.copy(), FLASK_START.copy()), near=1.0)
    _SUB.update(prev={}, vmax=0.0, vmax_at=None, skip=1)
    _TCP.clear()


_reset()


def _pose(base, name: str):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - as_numpy(base.scene.env_origins[0]), rotation(p[3:7])


def _substep(base, dt: float) -> None:
    """Every physics substep (240 Hz): how fast each end of the bar moves toward the bulb's inside wall, relative to
    the flask, while within NEAR of it; the largest such speed is kept."""
    if _SUB["skip"] > 0:
        _SUB["skip"] -= 1
        _SUB["prev"] = {}
        return
    pf, rf = _pose(base, "flask")
    pb, rb = _pose(base, "stir_bar")
    for s in (-1.0, 1.0):
        e = rf.T @ (pb + rb @ np.array([s * BAR_SEG, 0.0, 0.0]) - pf)  # the end sphere's middle, flask frame
        d = float(np.linalg.norm(e))
        prev = _SUB["prev"].get(s)
        if prev is not None and e[2] < Z_J and -0.003 < R_IN - BAR_R - d < NEAR:
            v = (d - prev) / dt
            if v > _SUB["vmax"]:
                _SUB["vmax"], _SUB["vmax_at"] = v, e.copy()
        _SUB["prev"][s] = d


def _install(base) -> None:
    """Hook the scene's per-substep update (Isaac Lab calls it after every physics step) to run _substep."""
    if _SUB.get("scene") is base.scene:
        return
    orig = base.scene.update

    def update(dt, *args, **kwargs):
        out = orig(dt, *args, **kwargs)
        try:
            _substep(base, dt)
        except Exception:
            import traceback

            print("STIRBAR_TRACEBACK\n" + traceback.format_exc(), flush=True)
        return out

    base.scene.update = update
    _SUB["scene"] = base.scene
    _SUB["prev"] = {}


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("STIRBAR_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _bar_place(pf, rf, pb, rb) -> str:
    """Where the bar is: 'bulb' (both ends inside the bulb), 'neck' (inside the flask, partly in the neck), 'out'."""
    places = []
    for s in (-1.0, 1.0):
        e = rf.T @ (pb + rb @ np.array([s * BAR_SEG, 0.0, 0.0]) - pf)
        if e[2] < Z_J and np.linalg.norm(e) < R_IN - BAR_R + 0.002:
            places.append("bulb")
        elif Z_J - 0.004 <= e[2] < NECK_TOP - BAR_R and math.hypot(e[0], e[1]) < NECK_RI - BAR_R + 0.002:
            places.append("neck")
        else:
            places.append("out")
    if "out" in places:
        return "out"
    return "bulb" if places == ["bulb", "bulb"] else "neck"


def _inside(e) -> bool:
    """Whether a point (flask frame) is inside the flask: in the bulb or in the bore below the mouth."""
    if e[2] < Z_J:
        return bool(np.linalg.norm(e) < R_IN)
    return bool(e[2] <= NECK_TOP and math.hypot(e[0], e[1]) < NECK_RI)


def _glass_gap(p, pf, neck) -> float:
    """Distance from a gripper's centre to the flask's outside: to the neck's glass or to the bulb's (negative when
    the gripper's centre is inside the neck or the bulb, as when it holds it)."""
    return min(_neck_dist(p, *neck) - NECK_RO, float(np.linalg.norm(p - pf)) - R_OUT)


def _neck_dist(p, base_pt, top_pt) -> float:
    """Distance from p to the neck's axis, from the bulb's top to the mouth."""
    d = top_pt - base_pt
    s = float(np.clip(np.dot(p - base_pt, d) / np.dot(d, d), 0.0, 1.0))
    return float(np.linalg.norm(p - (base_pt + s * d)))


def _track_inner(env) -> None:
    """Each gripper's centre (TCP) from its joints; the flask's pose, tilt, lowest point, whether it is lifted, held,
    seated in its ring, fallen or dropped; where the bar is and how fast it has met the bulb's inside wall."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    base = getattr(env, "unwrapped", env)
    _install(base)
    _EP["tick"] += 1
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        if side + "_fingers" not in _ARM_IDS:
            _ARM_IDS[side + "_fingers"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
        _GAP[side] = -float(sum(q[i] for i in _ARM_IDS[side + "_fingers"]))  # the gap between the pads (m)
        tp, tr = ik.tcp_pose(q[_ARM_IDS[side]], station.BASES[side])
        _TCP[side] = np.asarray(tp, float)
        _TCPR[side] = np.asarray(tr, float)
    pf, rf = _pose(base, "flask")
    pb, rb = _pose(base, "stir_bar")
    _EP["bar"] = (pb, rb)
    data = base.scene["flask"].data
    vel = as_numpy(data.root_com_vel_w if hasattr(data, "root_com_vel_w") else data.root_vel_w)[0][:3]
    tilt = float(math.acos(max(-1.0, min(1.0, rf[2, 2]))))
    axis = rf[:, 2]
    mouth_low = float((pf + axis * NECK_TOP)[2]) - NECK_RO * math.sqrt(max(0.0, 1.0 - axis[2] ** 2))
    low = min(float(pf[2] - R_OUT), mouth_low) - TOP  # the flask's lowest point: its bulb's, or its mouth's if lower
    neck = (pf + axis * (R_OUT - 0.002), pf + axis * NECK_TOP)
    gaps = {side: _glass_gap(tcp, pf, neck) for side, tcp in _TCP.items()}
    near = min(gaps.values()) if gaps else 1.0
    held = False
    for side, tcp in _TCP.items():  # held: a gripper's fingers closed on the neck or the bulb, the flask moving with it
        on_neck = (_neck_dist(tcp, *neck) < HELD_AT
                   and 2 * NECK_RO - HELD_SQUEEZE < _GAP[side] < 2 * NECK_RO + HELD_SLACK)
        on_bulb = (float(np.linalg.norm(tcp - pf)) < HELD_AT
                   and 2 * R_OUT - HELD_SQUEEZE < _GAP[side] < 2 * R_OUT + HELD_SLACK)
        off = rf.T @ (tcp - pf)
        old = _EP["offsets"].get(side)
        if (on_neck or on_bulb) and old is not None and np.linalg.norm(off - old) * 1000 < WITH_MM:
            held = True
        _EP["offsets"][side] = off
    place = _bar_place(pf, rf, pb, rb)
    if place != "out":
        _EP["was_in"] = True
    mid_in = _inside(rf.T @ (pb - pf))
    if not _EP["failed"]:
        if low < DOWN_Z:
            _EP["failed"] = "the flask fell: it touched the table (a round-bottom flask can only stand in its ring)"
        elif vel[2] < -DROP_V and not held:
            _EP["failed"] = "the flask was dropped"
        elif tilt > TIPPED and not held:
            _EP["failed"] = "the flask tipped over"
        elif _SUB["vmax"] >= V_HARD:
            _EP["failed"] = (f"the stir bar struck the flask's inside wall at {_SUB['vmax']:.2f} m/s (dropped in, not "
                             f"slid in)")
        elif _EP["was_in"] and not mid_in:
            _EP["failed"] = "the stir bar came back out of the flask"
    seated = (math.hypot(pf[0] - X_F, pf[1] - Y_F) < SEAT_XY and abs(pf[2] - TOP - SEATED_Z) < SEAT_DZ
              and tilt < UPRIGHT)
    _EP["hist"].append(np.concatenate([pf, pb]))
    del _EP["hist"][:-REST_STEPS]
    resting = (len(_EP["hist"]) == REST_STEPS
               and max(np.abs(h - _EP["hist"][-1]).max() for h in _EP["hist"]) * 1000 < REST_MM)
    lifted = low > LIFTED_Z
    done = bool(seated and place == "bulb" and resting and not held)
    _EP.update(pos=pf, rot=rf, tilt=tilt, low=low, lifted=lifted, seated=seated, held=held, bar_where=place,
               done=done, neck=neck, near=near)
    stage = 0.75 if done else 0.5 if place != "out" else 0.25 if (lifted and tilt >= TILTED) else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["pos"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    near = float(_EP["near"]) if len(_TCP) == 2 else 0.0
    return dict(still=still, near=near, done=bool(_EP["done"] and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p = _EP["pos"]
    at = _SUB["vmax_at"]
    at_s = "-" if at is None else f"({at[0] * 1000:.0f},{at[1] * 1000:.0f},{at[2] * 1000:.0f})mm"
    print(f"DBG step={_EP['step']} flask=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f}) tilt={math.degrees(_EP['tilt']):.1f} "
          f"low={_EP['low'] * 100:.1f}cm lifted={_EP['lifted']} held={_EP['held']} seated={_EP['seated']} "
          f"bar={_EP['bar_where']} vmax={_SUB['vmax']:.2f}m/s at {at_s} near={t['near']:.3f} still={t['still']} "
          f"best={_EP['best']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)
    if "bar" in _EP and "left" in _TCPR:
        pb, rb = _EP["bar"]
        rl = _TCPR["left"]
        mid_g = rl.T @ (pb - _TCP["left"])
        ax_g = rl.T @ rb[:, 0]
        mid_f = _EP["rot"].T @ (pb - _EP["pos"])
        ax_f = _EP["rot"].T @ rb[:, 0]
        ends = [mid_f + s * BAR_SEG * ax_f for s in (-1.0, 1.0)]
        tip = max(ends, key=lambda e: -e[2])  # the end deeper along the neck toward the bulb (smaller z)
        print(f"DBG   tip(flask)=({tip[0] * 1000:.1f},{tip[1] * 1000:.1f},{tip[2] * 1000:.1f})mm "
              f"axis(flask)=({ax_f[0]:.3f},{ax_f[1]:.3f},{ax_f[2]:.3f}) finger gaps: left "
              f"{_GAP.get('left', 0) * 1000:.1f} right {_GAP.get('right', 0) * 1000:.1f} mm", flush=True)
        print(f"DBG   bar=({pb[0]:.3f},{pb[1]:.3f},{pb[2] - TOP:.3f}) in left gripper: mid=({mid_g[0] * 1000:.1f},"
              f"{mid_g[1] * 1000:.1f},{mid_g[2] * 1000:.1f})mm axis=({ax_g[0]:.2f},{ax_g[1]:.2f},{ax_g[2]:.2f}) "
              f"in flask: ({mid_f[0] * 1000:.1f},{mid_f[1] * 1000:.1f},{mid_f[2] * 1000:.1f})mm", flush=True)


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the bar has met the bulb's inside wall at 1 m/s or
    faster, the flask has touched the table, dropped or tipped over, or the bar has come back out): 0.25 the flask
    lifted out of its ring and tilted 60 degrees or more; 0.5 the bar wholly inside the flask; 0.75 the flask upright
    in its ring with the bar in its bulb, at rest, not held. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The flask upright in its cork ring with the bar in its bulb, not held, both grippers' centres 6 cm or more from
    its glass, still for 1 s, and no rule broken."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the York teaching lab shows: the right gripper, level and pointing forward, takes the flask by its neck, lifts it
# out of the cork ring and turns it 70 degrees over, the neck toward the left arm; the left gripper takes the stir bar
# by one end, pinched slantwise near its fingertips so that most of the bar sticks out, lays it in the mouth along the
# neck and lets go: the bar slides down the neck into the bulb; the left gripper moves away; the right gripper turns
# the flask upright slowly (the bar slides along the inside wall to the bottom), sets it back in the cork ring, lets go
# and moves away. STIRBAR_DROP=1 runs it the wrong way: the flask stays upright in its ring and the left gripper takes
# the bar by one end, lowers its other end into the mouth and lets go: it drops down the neck onto the bottom.

WRONG = os.environ.get("STIRBAR_DROP") == "1"


def _rx(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _frame(closing, approach) -> np.ndarray:
    c, a = np.asarray(closing, float), np.asarray(approach, float)
    return np.column_stack([c, np.cross(a, c), a])


ROLL = 70.0  # how far the right gripper turns the flask over (degrees from upright)
PINCH = 45.0  # the left gripper's tilt below level when it takes the bar
HOLD_UP = _frame((0, -1, 0), (1, 0, 0))  # right: level, pointing forward (+x), fingers closing along y
HOLD_SIDE = _rx(-ROLL) @ HOLD_UP  # the same turned about x: the neck tips toward +y
_LEVEL_L = _frame((-1, 0, 0), (0, -1, 0))  # left: level, pointing toward -y, fingers closing along x
FEED_PICK = _rx(PINCH) @ _LEVEL_L  # tilted PINCH below level
# The bar as the left gripper holds it once lifted (the pinch settles about 15 degrees under the bar's weight; seen in
# the simulator): its middle and its front end's direction in the gripper's frame (closing, width, approach).
GRASP_MID = np.array([-0.0001, -0.0049, 0.0169])
GRASP_DIR = np.array([0.0, -0.87, 0.50]) / np.linalg.norm([0.0, -0.87, 0.50])
GRIP_H = 0.066  # the right gripper's centre on the neck, above the bulb's middle
OPEN_R, NECK_PINCH = -0.030, -(NECK_RO - 0.006)
OPEN_L, BAR_PINCH, BAR_LET = -0.012, -(BAR_R - 0.0035), -0.008
# The slantwise pinch (from the fingers' collision shapes): at PINCH 45 degrees, with the fingertips 1.5 mm above the
# table, the bar's axis lies 11 mm below the gripper's centre; the pads hold its last 5.5 mm, so its middle is
# 15.2 mm ahead (-y) of the gripper's centre.
PINCH_DY, PINCH_DZ = 0.0152, 0.011
LIFT = 0.045
LAY_IN = 0.014  # how far the bar's front end goes past the mouth (the fingertips then nearly touch the rim)
LAY_UP = 0.003  # aim the bar this far above the neck's floor
LAY_DX = 0.0015  # the left arm holds this pose about 1.5 mm short in x (seen in the simulator): aim that much further


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
    return [Waypoint(seconds, tuple(float(v) for v in pos), OPEN_R, "stay put", rot)]


GRIP0 = np.array([X_F, Y_F, TOP + SEATED_Z + GRIP_H])  # the right gripper's centre on the neck, flask in its ring
GRIP1 = GRIP0 + [0.0, 0.0, LIFT]  # lifted out of the ring
NECK_DIR = _rx(-ROLL) @ np.array([0.0, 0.0, 1.0])  # the neck's direction, turned over
FLOOR_DIR = _rx(-ROLL) @ np.array([0.0, 1.0, 0.0])  # from the neck's axis toward its floor (down)
SAG = 0.0045  # the right arm holds the turned flask this much below its target (seen in the simulator)
PLACE_DX, PLACE_DY = 0.002, 0.005  # and sets it down about this much short in x and to its right (seen in the
# simulator), so it aims that much further
MOUTH = GRIP1 + NECK_DIR * (NECK_TOP - GRIP_H) - [0.0, 0.0, SAG]
BAR_PICK = BAR_START + [0.0, 0.0, -0.0005]  # the bar's middle on the table
TCP_PICK = BAR_PICK + [0.0, PINCH_DY, PINCH_DZ]
_FRONT = _rx(-ROLL) @ np.array([0.0, 0.0, -1.0])  # into the turned neck
_LAY_PITCH = min(np.arange(30.0, 90.0, 0.25), key=lambda b: -float((_rx(b) @ _LEVEL_L @ GRASP_DIR) @ _FRONT))
FEED_LAY = _rx(_LAY_PITCH) @ _LEVEL_L  # the left gripper's tilt that lays the held bar along the turned neck
T_GRAB = 3.0 + 1.5 + 1.5 + 0.5 + 1.0 + 1.0  # right: holding the neck
T_TURNED = T_GRAB + 2.0 + 3.0 + 0.5  # lifted and turned over


def _bar_at(depth: float) -> np.ndarray:
    """The bar's middle when its front end is ``depth`` past the mouth, lying along the turned neck's floor."""
    return MOUTH + FLOOR_DIR * (NECK_RI - BAR_R - LAY_UP) - NECK_DIR * (depth - BAR_L / 2) + [LAY_DX, 0.0, 0.0]


def _right(t_clear: float) -> list:
    """The right arm's plan; ``t_clear`` is when the left gripper is clear of the mouth."""
    if WRONG:
        return _stay("right", 30.0)
    rot0, rot1 = HOLD_UP, HOLD_SIDE
    out = _lead_in("right", np.array([0.20, -0.14, TOP + 0.185]), rot0, OPEN_R, "level, to the right of the flask")
    out += [_w(1.5, GRIP0 + [-0.075, 0.0, 0.0], OPEN_R, "behind the neck, fingers open", rot0),
            _w(1.5, GRIP0, OPEN_R, "fingers either side of the neck", rot0),
            _w(0.5, GRIP0, OPEN_R, "settle", rot0),
            _w(1.0, GRIP0, NECK_PINCH, "take the neck", rot0),
            _w(1.0, GRIP0, NECK_PINCH, "settle", rot0),
            _w(2.0, GRIP1, NECK_PINCH, "lift the flask out of the cork ring", rot0),
            _w(3.0, GRIP1, NECK_PINCH, "turn it on its side, the neck toward the left arm", rot1),
            _w(0.5, GRIP1, NECK_PINCH, "settle", rot1),
            _w(t_clear - T_TURNED + 0.3, GRIP1, NECK_PINCH, "hold it while the bar is laid in", rot1)]
    mid = _rx(-30.0) @ HOLD_UP
    place = np.array([PLACE_DX, PLACE_DY, 0.0])  # aim off the arm's offset so the bulb lands in the ring's middle
    down = GRIP0 + place + [0.0, 0.0, 0.001]  # the bulb just on the seat: let go without pressing it into the ring
    out += [_w(4.0, GRIP1, NECK_PINCH, "turn it slowly toward upright: the bar slides down the neck", mid),
            _w(2.5, GRIP1 + place, NECK_PINCH, "upright: the bar slides down the inside wall to the bottom", rot0),
            _w(0.5, GRIP1 + place, NECK_PINCH, "settle", rot0),
            _w(2.5, down, NECK_PINCH, "lower it into the cork ring", rot0),
            _w(0.5, down, NECK_PINCH, "settle", rot0),
            _w(1.0, down, OPEN_R, "let go", rot0),
            _w(0.5, down, OPEN_R, "settle", rot0),
            _w(1.5, GRIP0 + [-0.075, 0.0, 0.0], OPEN_R, "draw back off the neck", rot0),
            _w(1.5, np.array([0.18, -0.12, TOP + 0.17]), OPEN_R, "move away", rot0),
            _w(4.0, np.array([0.18, -0.12, TOP + 0.17]), OPEN_R, "stay clear", rot0)]
    return out


def _left_drop() -> list:
    """The wrong way: take the bar by one end from above, turn the gripper level so the bar hangs down, lower its end
    into the mouth of the upright flask and let it drop."""
    pick_rot, hang_rot = _rx(90.0) @ _frame((-1, 0, 0), (0, -1, 0)), _frame((-1, 0, 0), (0, -1, 0))
    tcp_pick = BAR_PICK + [0.0, -(BAR_L / 2 - 0.008) - 0.0085, 0.0103]  # the pads on the bar's last 8 mm (-y end)
    mouth_z = TOP + SEATED_Z + NECK_TOP
    hang = np.array([X_F, Y_F + 0.0103, mouth_z - 0.008 + 0.0255])  # the bar's lower end 8 mm down the mouth
    out = _lead_in("left", tcp_pick + [0.0, 0.0, 0.04], pick_rot, OPEN_L, "above the stir bar's end, fingers open")
    return out + [_w(1.5, tcp_pick, OPEN_L, "fingers either side of the bar's end", pick_rot),
                  _w(0.5, tcp_pick, OPEN_L, "settle", pick_rot),
                  _w(1.0, tcp_pick, BAR_PINCH, "take the bar's end", pick_rot),
                  _w(1.0, tcp_pick, BAR_PINCH, "settle", pick_rot),
                  _w(1.5, tcp_pick + [0.0, 0.0, 0.08], BAR_PINCH, "lift it", pick_rot),
                  _w(3.0, hang + [0.0, 0.0, 0.05], BAR_PINCH, "over the upright flask, the bar hanging down", hang_rot),
                  _w(0.5, hang + [0.0, 0.0, 0.05], BAR_PINCH, "settle", hang_rot),
                  _w(1.5, hang, BAR_PINCH, "the bar's end in the mouth", hang_rot),
                  _w(0.5, hang, BAR_PINCH, "settle", hang_rot),
                  _w(0.6, hang, OPEN_L, "let the bar drop down the neck", hang_rot),
                  _w(1.0, hang + [0.0, 0.03, 0.05], OPEN_L, "rise", hang_rot),
                  _w(1.5, np.array([0.20, 0.18, TOP + 0.16]), OPEN_L, "move away", pick_rot),
                  _w(8.0, np.array([0.20, 0.18, TOP + 0.16]), OPEN_L, "stay clear", pick_rot)]


def _left() -> tuple:
    if WRONG:
        return _left_drop(), 0.0
    rot = FEED_PICK
    above = TCP_PICK + [0.0, 0.0, 0.04]
    out = _lead_in("left", above, rot, OPEN_L, "above the stir bar's end, tilted, fingers open")
    out += [_w(1.5, TCP_PICK, OPEN_L, "fingers either side of the bar's end", rot),
            _w(0.5, TCP_PICK, OPEN_L, "settle", rot),
            _w(1.0, TCP_PICK, BAR_PINCH, "pinch the bar's end", rot),
            _w(1.0, TCP_PICK, BAR_PINCH, "settle", rot),
            _w(1.5, TCP_PICK + [0.0, 0.0, 0.06], BAR_PINCH, "lift it", rot)]
    lay = FEED_LAY
    t_used = sum(w.duration for w in out)
    ready = _bar_at(-0.020) - lay @ GRASP_MID  # the bar's front end 2 cm short of the mouth, in line with the neck
    laid = _bar_at(LAY_IN) - lay @ GRASP_MID
    up = -FLOOR_DIR
    out += [_w(max(1.5, T_TURNED - t_used - 0.5), ready + [0.0, 0.02, 0.012], BAR_PINCH, "toward the flask", lay),
            _w(0.5, ready, BAR_PINCH, "in line with the neck", lay),
            _w(1.5, laid, BAR_PINCH, "lay the bar in the mouth along the neck", lay),
            _w(0.5, laid, BAR_PINCH, "settle", lay),
            _w(0.8, laid, BAR_LET, "let go: the bar slides down the neck", lay),
            _w(0.6, laid, BAR_LET, "settle", lay),
            _w(0.8, laid + up * 0.016, BAR_LET, "lift the fingers clear", lay),
            _w(1.0, laid + up * 0.016 + NECK_DIR * 0.04, BAR_LET, "back away from the mouth", lay)]
    t_clear = sum(w.duration for w in out)
    out += [_w(1.5, np.array([0.16, 0.24, TOP + 0.16]), BAR_LET, "move away", lay),
            _w(20.0, np.array([0.16, 0.24, TOP + 0.16]), BAR_LET, "stay clear", lay)]
    return out, t_clear


def demo():
    left, t_clear = _left()
    return {"left": left, "right": _right(t_clear)}
