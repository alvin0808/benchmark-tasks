"""On the station table: a plier-style cherry and olive pitter (two die-cast handles hinged at one end, a cup ring at
the other end of the lower handle and a plunger on the upper handle above it, a return spring holding it open, 14.6 cm
long and 100 g, like the OXO Good Grips Cherry & Olive Pitter) lies on its side; three large green olives lie on a
saucer; a ramekin for the stones and a bowl for the pitted olives stand by. Task (instruction): pit the three olives
with the pitter, the pitted olives into the bowl and the stones into the ramekin.

How it is done (as the pitter's retailers describe it: the fruit goes in the front chamber, the handle is squeezed,
and the stone drops out through the splatter shield while the fruit stays whole): the pitter is held in one hand over
a dish, the olive is set in the cup with the other hand, end up and centred, the handles are squeezed all the way so
the plunger goes through the olive and pushes the stone out of the bottom of the cup; the hand eases off, and the
pitted olive is tipped out of the cup.

The hand skill: the pitter must be picked up from the table and turned in the hand so the cup faces up; it is held by
its handles alone (its spring pushes them apart against the fingers), and squeezed shut against that spring (1.9 N m:
about 20-30 N across the handles) without dropping it; each olive (22 x 18 mm, slick) is taken from the saucer,
turned end up and set into the 19.6 mm cup under the raised plunger, without knocking the pitter; and each pitted
olive is tipped out over the bowl.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- pitter: loose, 100 g, 14.6 cm long. In its own frame (H: the hinge pin at the origin, its axis along H-x, the handles
  along +H-y, the cup up = +H-z): lower handle 28 mm wide, 8 mm thick, from the hinge to y 112 mm; the cup at y 125
  mm: a cone seat from a 12 mm hole (where the stone leaves) to a 19.6 mm bore 8 mm deep, a funnel to 25.6 mm at the
  rim, in a ring 28 mm across; upper handle 28 x 8 mm to y 100 mm, its front arm and the 4 mm plunger over the cup. The
  return spring (3.6 N m/rad, 1.9 N m at full stroke) holds the handles 30 deg open. It starts lying on its side
  (hinge axis vertical), the hinge pin at (0.12, -0.20), the cup toward +y, the upper handle toward +x.
- olives: three, each 22 mm long and 18 mm across, 5.5 g, with a stone 14 x 8 mm (1.5 g) inside (a separate body
  held to the flesh by a fixed joint, never colliding with it), lying on a saucer (14 cm across) along y at x 0.16,
  0.20, 0.24, y 0.14. Wet olives are slick: friction 0.25 against anything.
- ramekin for the stones: 9 cm across inside, 3.5 cm tall, at (0.28, 0.00). Bowl for the pitted olives: 9.6 cm
  across inside, 4.5 cm tall, at (0.165, 0.00).

Pitting rule (a stand-in for the plunger cutting through the olive; stated in the fidelity note): the plunger and the
upper handle's front arm do not collide with the olive. When the handles are squeezed to full stroke (the handle
within 1.5 deg of shut, which takes 1.9 N m against the spring: about 20-30 N across the handles), an olive in the
cup is pitted if it sits in the seat (its centre within 4 mm of the seated height), its centre within 3 mm of the
cup's axis and its long axis within 25 deg of the plunger; then its stone's joint is let go and the stone drops out of
the cup's hole. An olive in the cup that is not seated like that
is torn (the plunger goes through its side): its stone stays in for good.

Score (sum over the three olives, 1/6 each part):
- 1/6: its stone has been pushed out and lies in the ramekin.
- 1/6: it was pitted (not torn) and lies in the bowl, released (every finger point 1 cm or more from it), still.
1.0 (success, held 1 s): all six parts.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CapsuleShape, CylShape, Joint, Link, Linkage, _quat_from_matrix  # noqa: E402
from bodykit import allow_matrix_grips, as_numpy, on_reset, on_step, plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
_E = os.environ.get

# -- the pitter, in its own frame H (m): the hinge pin at the origin, axis H-x, handles along +H-y, cup up (+H-z) --
HW = 0.028  # handle width (along H-x)
HT = 0.008  # handle thickness
LOW_Z, UP_Z = -0.010, 0.010  # the handles' centre lines when shut
CUP_Y = 0.125
SEAT_R0, SEAT_Z0, SEAT_Z1 = 0.006, -0.010, -0.004  # the cone seat: radius at its bottom; bottom and top
BORE_R, BORE_Z = 0.0098, 0.004  # the bore above the seat (the olive is 18 mm across)
FUNNEL_R, RIM_Z = 0.0128, 0.008  # the funnel at the cup's mouth, up to the rim
RING_R = 0.014
OPEN_DEG = 30.0
SPRING = float(_E("PIT_SPRING", "3.6"))  # N m/rad toward open
FULL_DEG = OPEN_DEG - 1.5  # full stroke: the plunger through the olive and the stone's hole
PIT_MASS, UPPER_MASS = 0.085, 0.015
# where it starts: lying on its side (hinge axis vertical, the upper handle toward +x)
PIN0 = np.array([0.12, -0.20, TOP + 0.0145])
R_LIE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # R_y(+90 deg): H -> world at the start

# -- olives, dishes ---------------------------------------------------------------------------------------------------
OLIVE_R, OLIVE_L = 0.009, 0.004  # capsule: 18 mm across, 22 mm long
STONE_R, STONE_L = 0.004, 0.006  # 8 mm across, 14 mm long
OLIVE_M, STONE_M = 0.0055, 0.0015
SAUCER = np.array([0.20, 0.14])
SAUCER_T = 0.004
OLIVE_XY = [np.array([x, 0.14]) for x in (0.16, 0.20, 0.24)]
OLIVE_Z = TOP + SAUCER_T + OLIVE_R + 0.001
RAMEKIN, RAM_R, RAM_H = np.array([0.28, 0.00]), 0.045, 0.035
BOWL, BOWL_R, BOWL_H = np.array([0.165, 0.00]), 0.048, 0.045
SEATED_Z = -0.0007  # the olive's centre (H) when it sits upright in the seat
SEAT_DXY, SEAT_DZ, SEAT_TILT = 0.003, 0.004, 25.0

GREEN, STONE_C, METAL, BLACK, WHITE, CREAM, CLEAR_C = ((0.42, 0.52, 0.12), (0.55, 0.40, 0.25), (0.62, 0.63, 0.65),
                                                     (0.10, 0.10, 0.11), (0.95, 0.95, 0.93), (0.92, 0.88, 0.78),
                                                     (0.80, 0.85, 0.88))


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def _box(c, size, color, R=np.eye(3), collide=True) -> BoxShape:
    return BoxShape(_v(R @ np.asarray(c, float)), _v(size), color, collide=collide, quat_xyzw=_quat_from_matrix(R))


def _xf(s: BoxShape, R) -> BoxShape:
    """A box shape turned by R about the frame origin."""
    return BoxShape(_v(R @ np.asarray(s.centre)), s.size, s.color, collide=s.collide,
                    quat_xyzw=_quat_from_matrix(R @ rotation(s.quat())))


def _ring(cy, r0, z0, r1, z1, thick, color, n=16, collide=True) -> list:
    """A cone (or a cylinder when r0 == r1) of n plates about the H-z axis through (0, cy)."""
    out = []
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        d = np.array([math.cos(a), math.sin(a), 0.0])
        p1 = np.array([0.0, cy, z0]) + d * r0
        p2 = np.array([0.0, cy, z1]) + d * r1
        w = 2 * math.pi * max(r0, r1) / n * 1.15
        out.append(plate(p1, p2, w, thick, d, color, collide=collide))
    return out


def _pitter() -> Linkage:
    base = [_box((0.0, 0.054, LOW_Z), (HW, 0.116, HT), BLACK),  # lower handle (soft grip)
            _box((0.0, -0.002, 0.0), (HW, 0.010, 0.022), METAL),  # the hinge knuckle
            _box((0.0, 0.112, LOW_Z), (0.016, 0.010, HT), METAL)]  # neck to the cup ring
    base += _ring(CUP_Y, SEAT_R0, SEAT_Z0, BORE_R, SEAT_Z1, 0.002, METAL)  # the cone seat
    base += _ring(CUP_Y, BORE_R + 0.001, SEAT_Z1, BORE_R + 0.001, BORE_Z, 0.002, METAL)  # the bore
    base += _ring(CUP_Y, BORE_R + 0.001, BORE_Z, FUNNEL_R + 0.001, RIM_Z, 0.002, METAL)  # the funnel at its mouth
    base += _ring(CUP_Y, RING_R - 0.0015, -0.014, RING_R - 0.0015, BORE_Z, 0.003, METAL)  # the ring's wall
    base += _ring(CUP_Y, RING_R - 0.002, -0.015, 0.0125, -0.030, 0.0012, CLEAR_C, n=12, collide=False)  # shield
    up = [_box((0.0, 0.050, UP_Z), (HW, 0.100, HT), BLACK),  # upper handle
          _box((0.0, 0.113, UP_Z), (0.016, 0.026, 0.006), METAL, collide=False),  # front arm over the cup
          _box((0.0, CUP_Y, UP_Z - 0.012), (0.004, 0.004, 0.026), METAL, collide=False),  # the plunger
          _box((0.0, CUP_Y, -0.003), (0.0065, 0.0065, 0.0012), METAL, collide=False)]  # its cross tip
    ro = _rx(OPEN_DEG)  # the upper handle starts open
    up = [_xf(_xf(s, ro), R_LIE) for s in up]
    base = [_xf(s, R_LIE) for s in base]
    links = [Link("upper", tuple(up), mass=UPPER_MASS, anchor=_v(PIN0), axis="Z", limits=(-0.5, OPEN_DEG),
                  stiffness=SPRING, spring_damping=0.02)]
    return Linkage.build("pitter", pos=_v(PIN0), mass=PIT_MASS, color=BLACK, floating=True, friction=(0.9, 0.8),
                         shapes=base, links=links, contact_offset=0.0015, solver_iterations=48)


def _dish(name, xy, r_in, h, color, mass) -> Body:
    sh = [CylShape((0.0, 0.0, 0.0025), r_in + 0.004, 0.005, color)]
    n = 20
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        d = np.array([math.cos(a), math.sin(a), 0.0])
        c = d * (r_in + 0.002)
        sh.append(plate((c[0], c[1], 0.004), (c[0], c[1], h), 2 * math.pi * (r_in + 0.004) / n * 1.15, 0.004, d,
                        color))
    return Body.build(name, pos=(float(xy[0]), float(xy[1]), TOP + 0.0005), mass=mass, color=color,
                      friction=(0.7, 0.6), shapes=sh)


SAUCER_BODY = Body.build("saucer", pos=(float(SAUCER[0]), float(SAUCER[1]), TOP + 0.0005), mass=0.25, color=WHITE,
                         friction=(0.6, 0.5), shapes=[CylShape((0.0, 0.0, SAUCER_T / 2), 0.070, SAUCER_T, WHITE)])
RAMEKIN_BODY = _dish("ramekin", RAMEKIN, RAM_R, RAM_H, WHITE, 0.25)
BOWL_BODY = _dish("bowl", BOWL, BOWL_R, BOWL_H, CREAM, 0.30)
PITTER = _pitter()
OLIVES, STONES = [], []
for _i, _xy in enumerate(OLIVE_XY, 1):
    _c = (float(_xy[0]), float(_xy[1]), float(OLIVE_Z))
    OLIVES.append(Body.build(f"olive{_i}", pos=_c, mass=OLIVE_M, color=GREEN, friction=(0.25, 0.2), friction_combine="min",
                             shapes=[CapsuleShape((0.0, 0.0, 0.0), OLIVE_R, OLIVE_L, GREEN, axis="Y")],
                             contact_offset=0.0015, solver_iterations=32))
    STONES.append(Body.build(f"stone{_i}", pos=_c, mass=STONE_M, color=STONE_C, friction=(0.3, 0.25), friction_combine="min",
                             shapes=[CapsuleShape((0.0, 0.0, 0.0), STONE_R, STONE_L, STONE_C, axis="Y")],
                             joint=Joint(anchor=_c, kind="fixed", parent=f"olive{_i}", break_force=50.0,
                                         break_torque=5.0),
                             no_collide_with=(f"olive{_i}",), contact_offset=0.001, solver_iterations=32))
OBJECTS = [SAUCER_BODY, RAMEKIN_BODY, BOWL_BODY, PITTER] + OLIVES + STONES

# -- tracking and scoring --------------------------------------------------------------------------------------------
CLEAR = 0.010
STILL = 0.02  # m/s
_DEBUG = _E("PIT_DEBUG") == "1"
_DEBUG_EVERY = int(_E("PIT_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
_CAP_PTS = np.array([[r * math.cos(a), y, r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
                     for y in (-0.002, 0.002) for r in (OLIVE_R,)]
                    + [[0.0, s * (OLIVE_L / 2 + OLIVE_R), 0.0] for s in (1, -1)])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, angle=0.0, pitted=[False] * 3, torn=[False] * 3, release=[0] * 3, squeezes=0,
               shut=False, olive={}, stone={}, near=[1.0] * 3, tcp={}, gaps={}, pin=PIN0.copy(), cup=None,
               olive_in_bowl=[False] * 3, stone_in_dish=[False] * 3, seat=[None] * 3)


_reset()


def _finger_points() -> dict:
    import struct
    import xml.etree.ElementTree as ET

    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = pathlib.Path(YAM_URDF).parent
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        name = link.get("name")
        if name not in ("tip_left", "tip_right"):
            continue
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            b = (root / col.find("geometry").find("mesh").get("filename")).read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            tri = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            v = tri["v"].reshape(-1, 3).astype(float)
            pts.append(np.unique(np.round(v, 4), axis=0) @ rpy_matrix(rpy).T + xyz)
        out[name] = np.concatenate(pts)
    return out


def _pose(base, name, origin):
    pose = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return pose[:3] - origin, rotation(pose[3:7])


def _speed(base, name) -> float:
    return float(np.linalg.norm(as_numpy(base.scene[name].data.root_com_vel_w)[0][:3]))


def _release_stone(i: int) -> None:
    """Let go of the stone's joint: set its break force to almost nothing, so the stone's own weight breaks it."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.endswith(f"/stone{i + 1}/joint"):
            UsdPhysics.Joint(prim).GetBreakForceAttr().Set(1.0e-4)
            UsdPhysics.Joint(prim).GetBreakTorqueAttr().Set(1.0e-6)
            _EP.setdefault("released_paths", []).append(p)


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("PIT_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _in_dish(p, centre, r_in, h, margin=0.0) -> bool:
    return bool(np.hypot(p[0] - centre[0], p[1] - centre[1]) <= r_in - margin and p[2] <= TOP + h)


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "fingers" not in _IDS:
        _IDS["fingers"] = _finger_points()
    fingers = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[side + "_f"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
            _IDS[side + "_tips"] = [robot.body_names.index(n) for n in ("tip_left", "tip_right")]
        q = as_numpy(robot.data.joint_pos)[0]
        _EP["tcp"][side] = np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float)
        _EP["gaps"][side] = float(-q[_IDS[side + "_f"]].sum())
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        fingers[side] = np.concatenate([_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                                        for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])])
    allf = np.concatenate(list(fingers.values()))
    pt = base.scene["pitter"]
    if "upper" not in _IDS:
        _IDS["upper"] = list(pt.joint_names).index("upper_turn")
    ang = math.degrees(float(as_numpy(pt.data.joint_pos)[0][_IDS["upper"]]))
    pp, pr = _pose(base, "pitter", origin)
    hr = pr @ R_LIE  # the H frame's axes in the world
    cup = pp + hr @ np.array([0.0, CUP_Y, SEATED_Z])
    shut = ang >= FULL_DEG
    new_squeeze = shut and not _EP["shut"]
    _EP.update(angle=ang, pin=pp, cup=cup, cup_axis=hr[:, 2])
    if new_squeeze:
        _EP["squeezes"] += 1
    for i in range(3):
        op, orr = _pose(base, f"olive{i + 1}", origin)
        sp, _ = _pose(base, f"stone{i + 1}", origin)
        loc = hr.T @ (op - pp) - np.array([0.0, CUP_Y, SEATED_Z])  # the olive's centre from the seated spot (H)
        tilt = math.degrees(math.acos(min(1.0, abs(float(orr[:, 1] @ hr[:, 2])))))
        dxy = float(np.hypot(loc[0], loc[1]))
        in_cup = dxy < 0.012 and -0.010 < loc[2] < 0.016
        _EP["seat"][i] = (dxy, float(loc[2]), tilt, in_cup)
        if new_squeeze and in_cup and not (_EP["pitted"][i] or _EP["torn"][i]):
            if dxy <= SEAT_DXY and abs(loc[2]) <= SEAT_DZ and tilt <= SEAT_TILT:
                _EP["pitted"][i] = True
                _EP["release"][i] = _EP["tick"]
                _release_stone(i)
            else:
                _EP["torn"][i] = True
        pts = op + _CAP_PTS @ orr.T
        _EP["near"][i] = float(cKDTree(pts).query(allf, k=1)[0].min())
        low = float(pts[:, 2].min())
        _EP["olive"][i] = (op, low)
        _EP["stone"][i] = sp
        still = _speed(base, f"olive{i + 1}") < STILL
        _EP["olive_in_bowl"][i] = bool(_in_dish((op[0], op[1], low), BOWL, BOWL_R, BOWL_H) and still
                                       and _EP["near"][i] >= CLEAR)
        _EP["stone_in_dish"][i] = bool(_EP["pitted"][i] and np.linalg.norm(sp - op) > 0.02
                                       and _in_dish(sp, RAMEKIN, RAM_R, RAM_H) and _speed(base, f"stone{i + 1}") < STILL)
    _EP["shut"] = shut


def _parts() -> list:
    out = []
    for i in range(3):
        out.append(int(_EP["stone_in_dish"][i]) + int(_EP["pitted"][i] and not _EP["torn"][i]
                                                      and _EP["olive_in_bowl"][i]))
    return out


def _debug() -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    s = []
    for i in range(3):
        d = _EP["seat"][i]
        o = _EP["olive"].get(i, (np.zeros(3), 0.0))[0]
        st = _EP["stone"].get(i, np.zeros(3))
        s.append(f"o{i + 1}=({o[0]:.3f},{o[1]:.3f},{(o[2] - TOP) * 1000:.0f}mm) st=({st[0]:.3f},{st[1]:.3f},"
                 f"{(st[2] - TOP) * 1000:.0f}) " + (f"seat dxy={d[0] * 1000:.1f} dz={d[1] * 1000:.1f} tilt={d[2]:.0f} "
                                                    f"in={int(d[3])} " if d else "")
                 + f"P{int(_EP['pitted'][i])}T{int(_EP['torn'][i])}B{int(_EP['olive_in_bowl'][i])}"
                   f"D{int(_EP['stone_in_dish'][i])} near={_EP['near'][i] * 1000:.0f}")
    pin, cup = _EP["pin"], _EP["cup"] if _EP["cup"] is not None else np.zeros(3)
    ca = _EP.get("cup_axis", np.zeros(3))
    print(f"DBG step={_EP['step']} ang={_EP['angle']:.1f} sq={_EP['squeezes']} cupax=({ca[0]:.2f},{ca[1]:.2f},{ca[2]:.2f}) pin=({pin[0]:.3f},{pin[1]:.3f},"
          f"{(pin[2] - TOP) * 1000:.0f}mm) cup=({cup[0]:.3f},{cup[1]:.3f},{(cup[2] - TOP) * 1000:.0f}mm) "
          + " | ".join(s) + " | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{(v[2] - TOP) * 1000:.0f}mm) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    if _DEBUG:
        _debug()
    return round(sum(_parts()) / 6.0, 4)


def success(objects) -> bool:
    return bool(sum(_parts()) == 6)


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Open loop, from the known layout. Right arm: straight down over the lying pitter, fingers either side of its handles,
# closes on them to 12 deg short of open (the spring then presses the handles out against the fingers, about 8 N),
# lifts it and turns it cup up (the hand level, pointing +x, pitched 8 deg nose down) over the ramekin. Left arm, for
# each olive: pinches it on the saucer from above, lifts it and turns it end up (the hand level, pointing -y), carries
# it in over the cup under the raised plunger and lets it drop 24 mm into the cup's funnel, then backs out. The right
# hand squeezes the handles shut (the plunger through: the stone drops into the ramekin), eases off, carries the pitter
# over the bowl, tips it nose down to 118 deg so the pitted olive slides out into the bowl, and turns back. After the
# third olive it lays the pitter back on the table.
# PIT_SOFT=1 (the wrong way): the same, but each squeeze is light (the gripper closes to 50 mm): the handles close only
# to 11 deg of their 30, the plunger never reaches full stroke, no stone comes out, and the unpitted olives go into the
# bowl: score 0.
OPEN_C = -0.047
SOFT = _E("PIT_SOFT") == "1"
N_OLIVES = int(_E("PIT_N", "3"))


def _R(c1, c3) -> np.ndarray:
    c1, c3 = np.asarray(c1, float), np.asarray(c3, float)
    c1, c3 = c1 / np.linalg.norm(c1), c3 / np.linalg.norm(c3)
    return np.column_stack([c1, np.cross(c3, c1), c3])


R_DOWN = _R((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))  # right: straight down, fingers closing along x
R_LEVEL = _ry(-90.0) @ R_DOWN  # right: level, pointing +x, fingers closing along z (the pitter turned cup up)
HOLD_TILT = float(_E("PIT_HOLD_TILT", "8.0"))  # deg: the hand pitched a little nose down (the wrist off its limit)
R_HOLD = _ry(HOLD_TILT) @ R_LEVEL
L_DOWN = _R((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))  # left: straight down, fingers closing along x
L_LOAD = _rx(-90.0) @ L_DOWN  # left: level, pointing -y, fingers closing along x (the olive turned end up)

P = {k: float(_E("PIT_" + k.upper(), v)) for k, v in dict(
    squeeze="0.0", soft_close="-0.025", grasp_dz="0.002", tcp_x="0.28", tcp_z="0.155",
    olive_grip="-0.0045", olive_open="-0.019", olive_dz="0.004", drop="0.024", tip="118", tip_x="0.17",
    tip_z="0.12").items()}


def _grasp_offset(open_deg: float) -> np.ndarray:
    """The TCP (between the fingers, over the handles' middle at y 60 mm) in H, the handles open by open_deg."""
    a = math.radians(open_deg)
    z_up = 0.094 * math.sin(a) + (UP_Z + HT / 2) * math.cos(a)  # the upper handle's top under the pad's front edge
    return np.array([0.0, 0.060, (LOW_Z - HT / 2 + z_up) / 2])


HOLD_DEG = float(_E("PIT_HOLD_DEG", "18.0"))  # the handles held this far open (12 deg squeezed: ~8 N on them)


def _gap(open_deg: float) -> float:
    a = math.radians(open_deg)
    return 0.094 * math.sin(a) + (UP_Z + HT / 2) * math.cos(a) - (LOW_Z - HT / 2)


P["hold"] = -_gap(HOLD_DEG) / 2 + 0.001


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    b = station.BASES[arm]
    j1 = math.atan2(float(pos[1]) - b[1], float(pos[0]) - b[0])
    for sd in itertools.product((j1,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (0.0,), (-1.8, -0.9, 0.0, 0.9, 1.8)):
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


def _joint_path(arm: str, q0, q1, finger: float, label: str, seconds: float, steps: int = 15) -> list:
    from yam_arena import ik, station

    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _ik_chain(arm: str, pos, rots) -> list:
    """Joint solutions for the TCP at ``pos`` turning through ``rots``, each solved from the one before (one branch)."""
    from yam_arena import ik, station

    qs = [_good_ik(arm, pos, rots[0])]
    for r in rots[1:]:
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(r), qs[-1], base_pos=station.BASES[arm],
                             iters=200)
        if pe > 0.001 or re > math.radians(0.5):
            raise ValueError(f"no continuous IK for the {arm} arm at {pos}")
        qs.append(q)
    return qs


def _chain_path(arm: str, qs, finger: float, label: str, seconds: float) -> list:
    from yam_arena import ik, station

    out = []
    for q in qs[1:]:
        p, r = ik.tcp_pose(q, station.BASES[arm])
        out.append(Waypoint(seconds / (len(qs) - 1), tuple(float(v) for v in p), finger, label, r))
    return out


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0) -> list:
    from yam_arena.constants import READY

    return _joint_path(arm, np.array(READY, float), _good_ik(arm, pos, rot), finger, label, seconds)


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)


def _ready(arm: str):
    from yam_arena import ik, station
    from yam_arena.constants import READY

    return ik.tcp_pose(np.array(READY, float), station.BASES[arm])


def _dur(wps) -> float:
    return float(sum(w.duration for w in wps))


def _turn(p0, p1, r0, r1, finger, seconds, label, n=8) -> list:
    """Move from (p0, r0) to (p1, r1) in n steps, the rotation interpolated about its own axis."""
    from scipy.spatial.transform import Rotation, Slerp

    sl = Slerp([0.0, 1.0], Rotation.from_matrix([r0, r1]))
    out = []
    for k in range(1, n + 1):
        f = k / n
        out.append(_w(seconds / n, np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * f, finger, label,
                      sl([f]).as_matrix()[0]))
    return out


class _Arm:
    def __init__(self, name):
        self.name, self.wps = name, []

    def add(self, wps):
        self.wps += list(wps)

    def t(self):
        return _dur(self.wps)

    def wait_until(self, t, label="wait"):
        if self.t() < t - 1e-6:
            last = self.wps[-1]
            self.wps.append(Waypoint(t - self.t(), last.pos, last.finger, label, last.yaw))


def demo():
    hold_tcp = np.array([P["tcp_x"], -0.065, TOP + P["tcp_z"]])  # right TCP holding the pitter cup up
    g = _grasp_offset(HOLD_DEG)
    rh = _ry(HOLD_TILT)  # the H axes in the world when held
    pin_held = hold_tcp - rh @ g
    cup_seat = pin_held + rh @ np.array([0.0, CUP_Y, SEATED_Z])
    grasp0 = PIN0 + R_LIE @ _grasp_offset(OPEN_DEG) + np.array([0.0, 0.0, P["grasp_dz"]])
    R, L = _Arm("right"), _Arm("left")
    # right: pick the pitter up by its handles and turn it cup up over the ramekin
    R.add(_lead_in("right", grasp0 + [0, 0, 0.07], R_DOWN, OPEN_C, "over the pitter's handles"))
    R.add([_w(1.5, grasp0, OPEN_C, "fingers either side of the handles", R_DOWN),
           _w(1.0, grasp0, P["hold"], "close on the handles", R_DOWN),
           _w(0.6, grasp0, P["hold"], "settle", R_DOWN),
           _w(1.2, grasp0 + [0, 0, 0.06], P["hold"], "lift", R_DOWN)])
    R.add(_turn(grasp0 + [0, 0, 0.06], hold_tcp, R_DOWN, R_HOLD, P["hold"], 3.0, "turn it cup up over the ramekin",
                n=10))
    R.add([_w(0.8, hold_tcp, P["hold"], "settle", R_HOLD)])
    # left: meanwhile, to the first olive
    lp, lq = _ready("left")
    first = True
    sq = P["soft_close"] if SOFT else P["squeeze"]
    for k in range(N_OLIVES):
        o = np.array([OLIVE_XY[k][0], OLIVE_XY[k][1], OLIVE_Z + P["olive_dz"]])
        above = o + [0, 0, 0.06]
        if first:
            L.add(_lead_in("left", above, L_DOWN, P["olive_open"], "over an olive"))
            first = False
        else:
            L.add([_w(2.0, above, P["olive_open"], "over the next olive", L_DOWN)])
        L.add([_w(1.5, o, P["olive_open"], "fingers either side of the olive", L_DOWN),
               _w(1.0, o, P["olive_grip"], "pinch it", L_DOWN),
               _w(0.6, o, P["olive_grip"], "settle", L_DOWN),
               _w(1.0, above, P["olive_grip"], "lift", L_DOWN)])
        drop_c = cup_seat + rh @ np.array([0.0, 0.0, P["drop"]])  # the olive's centre over the seat
        tcp_drop = drop_c + [0, P["olive_dz"], 0]  # the olive sits olive_dz ahead of the TCP (approach -y)
        pre = tcp_drop + [0, 0.05, 0.0]
        L.add(_turn(above, pre, L_DOWN, L_LOAD, P["olive_grip"], 2.5, "turn it end up, in line with the cup", n=10))
        L.wait_until(R.t(), "wait for the pitter")
        L.add([_w(1.5, tcp_drop, P["olive_grip"], "in over the cup", L_LOAD),
               _w(0.4, tcp_drop, P["olive_grip"], "settle", L_LOAD),
               _w(0.8, tcp_drop, P["olive_open"], "let it drop into the seat", L_LOAD),
               _w(1.2, pre, P["olive_open"], "back out", L_LOAD),
               _w(1.0, pre + [0, 0.02, 0.05], P["olive_open"], "clear", L_LOAD)])
        R.wait_until(L.t(), "hold the pitter still")
        R.add([_w(1.2, hold_tcp, sq, "squeeze the handles shut", R_HOLD),
               _w(0.8, hold_tcp, sq, "hold: the stone drops", R_HOLD),
               _w(1.0, hold_tcp, P["hold"], "ease off", R_HOLD)])
        # tip the pitted olive out over the bowl
        tip_tcp = np.array([P["tip_x"], -0.065, TOP + P["tip_z"]])
        r_tip = _ry(P["tip"]) @ R_LEVEL
        R.add([_w(1.5, tip_tcp, P["hold"], "over the bowl", R_HOLD)])
        chain = _ik_chain("right", tip_tcp, [_ry(HOLD_TILT + (P["tip"] - HOLD_TILT) * k / 12) @ R_LEVEL
                                                for k in range(13)])
        R.add(_chain_path("right", chain, P["hold"], "tip the olive out", 1.5))
        R.add([_w(1.2, tip_tcp, P["hold"], "let it fall", r_tip)])
        R.add(_chain_path("right", chain[::-1], P["hold"], "turn back cup up", 1.5))
        R.add([_w(1.5, hold_tcp, P["hold"], "back over the ramekin", R_HOLD)])
        if k == N_OLIVES - 1:
            L.add(_joint_path("left", _good_ik("left", L.wps[-1].pos, L.wps[-1].yaw),
                              np.array(__import__("yam_arena.constants", fromlist=["READY"]).READY, float),
                              OPEN_C, "back to the ready pose", 3.0))
    # right: put the pitter down? It is left held cup up over the ramekin, then set down on the table.
    down = np.array([0.12, -0.20, TOP + 0.0145]) + R_LIE @ _grasp_offset(HOLD_DEG) + [0, 0, P["grasp_dz"]]
    R.add(_turn(hold_tcp, down + [0, 0, 0.06], R_HOLD, R_DOWN, P["hold"], 3.0, "turn it back on its side", n=10))
    R.add([_w(1.5, down + [0, 0, 0.004], P["hold"], "lay it on the table", R_DOWN),
           _w(1.0, down + [0, 0, 0.004], OPEN_C, "let go", R_DOWN),
           _w(1.2, down + [0, 0, 0.08], OPEN_C, "rise", R_DOWN)])
    rp, rq = _ready("right")
    R.add([_w(3.0, rp, OPEN_C, "back to the ready pose", rq)])
    T = max(R.t(), L.t()) + 3.0
    R.wait_until(T, "stay clear")
    L.wait_until(T, "stay clear")
    return {"left": L.wps, "right": R.wps}
