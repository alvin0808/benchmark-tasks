"""On the station table: a true-size adult care-training forearm and right hand lies palm up on a firm foam pad, the
fingers pointing toward the robot and curled into a fist by sprung joints (a contracted hand: the fingertips curled
down to within 1 cm of the palm). Beside it on the pad lies a firm palm cone (13 cm long, 2.5 cm across at its narrow
end, 4.2 cm at its wide end). Task (instruction): fit the cone into the hand, lying across the palm under the fingers
with its narrow end at the thumb side and all four fingertips resting on it, without forcing the fingers and with the
hand kept on its pad, and let go. The instruction does not say how.

Procedure. Oxford Health NHS Care Home Support Service, "Palm protectors: advice for care home staff": "Fit the palm
protector by gently prising open the fingers and thumb. Place the thumb through first and then slide the sheepskin
pad under the fingers". Nursing Assistant (Chippewa Valley Technical College, open textbook, 9.4): "The first type of
hand device is a cone that slides into the palm of the hand ... The cone should not be forced into the fingers but
placed gently." The Royal Wolverhampton NHS Trust, "Palm Cone": "Place the cone inside the palm of the hand, the
narrower end of the cone should be towards the thumb and the wider end towards the little finger".

Two hands, because of the hand itself: the fingers spring shut the moment they are let go, and the fist leaves no
room for the cone (the gap under the curled fingers is about 1.5 cm wide and 2 cm high, the cone 2.5 to 4.2 cm
across). One hand holds the fingers open while the other lays the cone in the palm; the fingers are then eased back
down onto it.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. hand_mesh.py
builds the moulded surfaces, the finger sizes and the resting curl from ANSUR II women's means (hand 181 mm long,
78 mm across the knuckles, palm 109 mm).
- pad: fixed, 46 x 39 x 2 cm, x 0.10-0.56, y -0.25-0.14 (firm foam, wipe-clean cover).
- training hand: loose; forearm and palm one hollow shell, 0.50 kg (the forearm cut 20 cm above the wrist), 0.58 kg
  in all. Its knuckle (MCP) axis runs along y at x 0.240, z = table + 0.035 (1.4 cm above the back of the hand); the
  middle finger's knuckle at y 0.048, the thumb toward +y. The four fingers turn together at the knuckles on one
  sprung joint (the fingers of a contracted hand flex together), each with its own sprung play about it, and each
  finger's middle joint (PIP) is sprung on its own; the end joints are fixed at 30 deg. The thumb lies along the index
  finger on its own sprung joint. All springs pull toward the resting fist (knuckles 90 deg, middle joints 106, 102,
  98 and 88 deg from the index to the little finger).
- palm cone: loose, 0.07 kg, lying on the pad with its axis along y, wide end toward -y, centre (0.260, -0.155).

Score (0 for the rest of the run once a rule is broken; otherwise the best stage reached):
- 0: a finger jerked open (a knuckle opening faster than 90 deg/s) or let snap shut (any finger joint closing faster
  than 150 deg/s); a finger opened past straight at its knuckle, or its middle joint straightened to under 30 deg of
  bend; the thumb pushed out more than 35 deg; a finger pushed open more than 10 deg at the knuckle by the cone pressing
  on it (the cone forced into the fingers); the hand lifted more than 1 cm off the pad or moved more than 2 cm along it.
- 0.25: all four fingers held open at once (each knuckle bent 45 deg or less).
- 0.5: the cone seated in the palm: lying across it (axis within 25 deg of the knuckle line) with its narrow end at
  the thumb side (from 3 cm inside to 2.5 cm beyond the index finger's outer edge), its middle no more than 4 cm
  behind the knuckle line, its lowest point over the palm within 1 cm of the palm's surface.
- 0.75: the cone seated and all four fingertips resting on it (each fingertip within 2.5 mm of the cone).
- 1.0 (success, held 1 s): that, with every gripper finger point 3 cm or more from the hand and the cone, and the
  cone and the fingers still (under 1 mm and 0.5 deg over 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hand_mesh as hm  # noqa: E402
from bodykit import Body, BoxShape, Link, Linkage, MeshShape, SphereShape, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, capsule, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the stages reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"
CM = 0.01

# -- the layout ----------------------------------------------------------------------------------------------------
PAD_T = 0.02
PAD_X, PAD_Y = (0.10, 0.56), (-0.25, 0.14)
X_M, Y_H = 0.240, 0.040  # the knuckle axis (x) and the hand's middle line (y) (the middle finger's knuckle at +0.8 cm)
Z_M = TOP + PAD_T + (-hm.BACK_Z) * CM + 0.001  # the knuckle axis: 1.4 cm above the back of the hand, 1 mm spawn gap
ORIGIN = np.array([X_M, Y_H, Z_M])
SKIN, NAIL, PAD_BLUE, TERRY, BAND = (0.90, 0.76, 0.66), (0.93, 0.80, 0.76), (0.60, 0.72, 0.84), (0.80, 0.70, 0.55), \
    (0.97, 0.97, 0.95)

BASE_KG = 0.50  # forearm and palm, a hollow moulded shell
PROX_KG, DIST_KG, THUMB_KG, KNUCKLE_KG = 0.008, 0.008, 0.012, 0.010
SKIN_FRICTION = (1.0, 0.9)  # soft vinyl skin

# Springs (N m/rad) and damping (N m s/rad). Knuckles: the four fingers' common joint; play: each finger's own give
# at the knuckle; middle joints (PIP); thumb.
K_KNUCKLE, C_KNUCKLE = 0.080, 0.030
K_PLAY, C_PLAY = 0.15, 0.002
K_PIP, C_PIP = 0.020, 0.002
K_THUMB, C_THUMB = 0.030, 0.0015
J_FRICTION, J_DAMPING = 0.001, 0.0005
KNUCKLE_LIM = (-100.0, 2.0)  # deg from the fist (negative opens the fingers)
PLAY_LIM = (-8.0, 6.0)
PIP_LIM = (-85.0, 3.0)
THUMB_LIM = (-40.0, 2.0)  # negative swings the thumb out, away from the index finger

# The cone: Posey palm cone (REF 5608), firm terry-covered plastic, 13 cm long; end diameters 3.2 and 5.0 cm.
CONE_L, CONE_RN, CONE_RW = 0.130, 0.0125, 0.021
CONE_KG = 0.07
CONE_ALPHA = math.atan2(CONE_RW - CONE_RN, CONE_L)  # its half angle: lying on its side its axis tilts this much


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat(m: np.ndarray):
    from bodykit import _quat_from_matrix

    return _quat_from_matrix(np.asarray(m, float))


def cone_mesh(n: int = 40):
    """The cone (m, cone frame): axis along y, narrow end at +y, centre at the origin; the rims rounded 3 mm."""
    rings = []
    ch = 0.003
    for y, r in ((CONE_L / 2, CONE_RN - ch), (CONE_L / 2 - ch * 0.4, CONE_RN - ch * 0.3),
                 (CONE_L / 2 - ch, CONE_RN + (CONE_RW - CONE_RN) * ch / CONE_L),
                 (-CONE_L / 2 + ch, CONE_RW - (CONE_RW - CONE_RN) * ch / CONE_L),
                 (-CONE_L / 2 + ch * 0.4, CONE_RW - ch * 0.3), (-CONE_L / 2, CONE_RW - ch)):
        phi = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        rings.append(np.stack([r * np.cos(phi), np.full(n, y), r * np.sin(phi)], axis=1))
    pts = np.concatenate(rings + [np.array([[0.0, CONE_L / 2, 0.0], [0.0, -CONE_L / 2, 0.0]])])
    faces = []
    m = len(rings)
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [(a, b, c), (b, d, c)]
    ia, ib = m * n, m * n + 1
    for j in range(n):
        faces.append((ia, (j + 1) % n, j))
        faces.append((ib, (m - 1) * n + j, (m - 1) * n + (j + 1) % n))
    faces = hm._orient(pts, np.asarray(faces, dtype=np.int64))
    return pts, faces, hm._normals(pts, faces)


CONE_P, CONE_F, CONE_N = cone_mesh()
R_CONE0 = _rx(-CONE_ALPHA)  # lying on its side: the wide end up by the half angle, so a side line lies flat
CONE_XY0 = (X_M + 0.020, -0.155)
CONE_POS0 = np.array([CONE_XY0[0], CONE_XY0[1],
                      TOP + PAD_T + 0.001 - float((CONE_P @ R_CONE0.T)[:, 2].min())])
CONE_GRIP_Y = -CONE_L / 2 + 0.017  # where the right gripper holds it: 1.7 cm from the wide end (cone frame)


def cone_radius(s):
    """The cone's radius (m) at ``s`` along its axis (cone frame y, narrow end at +L/2)."""
    return CONE_RN + (CONE_L / 2 - np.asarray(s, float)) / CONE_L * (CONE_RW - CONE_RN)


# -- bodies ---------------------------------------------------------------------------------------------------------
def _m(p) -> tuple:
    return tuple(float(v) for v in np.asarray(p, float) * CM)


def _nail(q, t, r, color=NAIL):
    """A thin nail plate on the back of an end bone (from its end joint ``q`` to its tip centre ``t``, cm)."""
    d = (t - q) / np.linalg.norm(t - q)
    back = np.cross(d, np.array([0.0, 1.0, 0.0]))  # the bone's back (dorsal) side, for a finger curling about +y
    c = q + (t - q) * 0.62 + back * (r * 0.92)
    m = np.column_stack([back, np.cross(d, back), d])
    return BoxShape(_m(c), (0.0006, r * 1.3 * CM, 0.9 * np.linalg.norm(t - q) * CM), color, collide=False,
                    quat_xyzw=_quat(m))


def _hand() -> Linkage:
    base = []
    for mesh in (hm.palm_mesh(), hm.forearm_mesh(), hm.thenar_mesh(), hm.hypothenar_mesh()):
        base.append(MeshShape(*mesh, SKIN))
    links = [Link("knuckles", (SphereShape((0.0, 0.0, 0.0), 0.009, SKIN),), mass=KNUCKLE_KG,
                  anchor=tuple(float(v) for v in ORIGIN), axis="Y", limits=KNUCKLE_LIM, friction=J_FRICTION,
                  damping=J_DAMPING, stiffness=K_KNUCKLE, spring_damping=C_KNUCKLE)]
    for name in hm.ORDER:
        f = hm.FINGERS[name]
        k, p, q, t = hm.finger_points(name)
        links.append(Link(f"p_{name}", (capsule(_m(k), _m(p), f["r"][0] * CM, SKIN),),
                          mass=PROX_KG * (0.75 if name == "little" else 1.0), parent="knuckles",
                          anchor=tuple(float(v) for v in ORIGIN + np.asarray(_m(k))), axis="Y", limits=PLAY_LIM,
                          friction=J_FRICTION, damping=J_DAMPING, stiffness=K_PLAY, spring_damping=C_PLAY))
        links.append(Link(f"d_{name}", (capsule(_m(p), _m(q), f["r"][1] * CM, SKIN),
                                         capsule(_m(q), _m(t), f["r"][2] * CM, SKIN), _nail(q, t, f["r"][2])),
                          mass=DIST_KG * (0.75 if name == "little" else 1.0), parent=f"p_{name}",
                          anchor=tuple(float(v) for v in ORIGIN + np.asarray(_m(p))), axis="Y", limits=PIP_LIM,
                          friction=J_FRICTION, damping=J_DAMPING, stiffness=K_PIP, spring_damping=C_PIP))
    b, j, t = hm.thumb_points()
    links.append(Link("thumb", (capsule(_m(b), _m(j), hm.THUMB_R[0] * CM, SKIN),
                                capsule(_m(j), _m(t), hm.THUMB_R[1] * CM, SKIN)),
                      mass=THUMB_KG, anchor=tuple(float(v) for v in ORIGIN + np.asarray(_m(b))), axis="Z",
                      limits=THUMB_LIM, friction=J_FRICTION, damping=J_DAMPING, stiffness=K_THUMB,
                      spring_damping=C_THUMB))
    return Linkage.build("care_hand", pos=tuple(float(v) for v in ORIGIN), mass=BASE_KG, color=SKIN, floating=True,
                         friction=SKIN_FRICTION, shapes=base, links=links)


def _cone() -> Body:
    shapes = [MeshShape(CONE_P, CONE_F, CONE_N, TERRY, approximation="convexHull")]
    # the elastic band that holds it in the hand, sewn along one side (white; not modelled as a strap)
    s0, s1 = CONE_L / 2 - 0.01, -CONE_L / 2 + 0.01
    r0, r1 = float(cone_radius(s0)), float(cone_radius(s1))
    a = np.array([0.0, s0, r0 + 0.0006])  # along the top as it lies
    b = np.array([0.0, s1, r1 + 0.0006])
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([0.0, 0.0, 1.0]) - d * d[2]
    n /= np.linalg.norm(n)
    shapes.append(BoxShape(tuple(((a + b) / 2).tolist()), (0.0008, 0.016, float(np.linalg.norm(b - a))), BAND,
                           collide=False, quat_xyzw=_quat(np.column_stack([n, np.cross(d, n), d]))))
    return Body.build("palm_cone", pos=tuple(float(v) for v in CONE_POS0), mass=CONE_KG, color=TERRY,
                      friction=(0.9, 0.8), init_quat=_quat(R_CONE0), shapes=shapes)


PAD = Body.build("hand_pad", pos=((PAD_X[0] + PAD_X[1]) / 2, (PAD_Y[0] + PAD_Y[1]) / 2, TOP + PAD_T / 2), mass=1.0,
                 color=PAD_BLUE, fixed=True, friction=(0.8, 0.7),
                 shapes=[BoxShape((0.0, 0.0, 0.0), (PAD_X[1] - PAD_X[0], PAD_Y[1] - PAD_Y[0], PAD_T), PAD_BLUE)])
HAND = _hand()
CONE = _cone()
OBJECTS = [PAD, HAND, CONE]

# -- scoring --------------------------------------------------------------------------------------------------------
MAX_OPEN = 90.0  # deg/s between control steps: a knuckle opening faster than this was jerked open
MAX_CLOSE = 150.0  # deg/s: any finger joint closing faster than this was let snap shut
OPEN_KNUCKLE = 45.0  # stage 1: every knuckle bent this much or less, at once
KNUCKLE_MIN = 0.0  # a knuckle opened past straight: forced
PIP_MIN = 30.0  # a middle joint straightened to under this bend: forced
THUMB_MAX = 35.0  # the thumb pushed out more than this: forced
CONE_FORCE = 10.0  # a finger touching the cone pushed open more than this since it touched: the cone was forced in
LIFT_MAX, MOVE_MAX = 0.010, 0.020
AXIS_TOL = 25.0  # deg between the cone's axis and the knuckle line
NARROW_OUT, NARROW_IN = 0.025, 0.030  # the narrow end: at most this far beyond / inside the index finger's outer edge
SEAT_BACK = 0.040  # the cone's middle at most this far behind the knuckle line (and not in front of it)
ON_PALM = 0.010  # its lowest point within this of the palm's surface
TOUCH = 0.0025  # a fingertip this close to the cone rests on it
CLEAR = 0.03
STILL_STEPS, STILL_MM, STILL_DEG = 20, 1.0, 0.5
INDEX_EDGE = (hm.FINGERS["index"]["y"] + hm.FINGERS["index"]["r"][0]) * CM  # the index finger's outer edge (y)
PALM_HALF, PALM_LEN = 0.035, 0.08  # the palm (hand frame): within 3.5 cm of its middle line, 0 to 8 cm behind the knuckles

_DEBUG = os.environ.get("PALM_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("PALM_DEBUG_EVERY", "5"))
_EP: dict = {}
_TCP: dict = {}
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, failed="", stage=0.0, prev=None, settle=0, ref=None, recent=[],
               touch_ref={}, angles={}, speed=0.0, lift=0.0, move=0.0, near=1.0, tips={}, seated=False, seat={},
               peak_speed=0.0, min_knuckle=90.0)
    _TCP.clear()


_reset()


def _finger_points() -> dict:
    """Each finger's collision hull vertices in the finger's own frame, from the robot description the simulator
    loads."""
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


# Surface samples of the hand (start pose, relative to ORIGIN, m) per body, for the grippers' clearance check.
def _capsule_samples(a, b, r, n=6, m=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    u = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(d, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    out = []
    for i in range(n + 1):
        c = a + (b - a) * i / n
        for k in range(m):
            ph = 2 * math.pi * k / m
            out.append(c + r * (math.cos(ph) * u + math.sin(ph) * v))
    out += [a - d * r, b + d * r]
    return np.asarray(out)


def _samples() -> dict:
    out = {"base": np.concatenate([np.asarray(s.points)[:: 2] for s in HAND.shapes])}
    for link in HAND.links:
        pts = [_capsule_samples(np.asarray(s.centre) - rotation(s.quat_xyzw)[:, 0] * s.height / 2,
                                np.asarray(s.centre) + rotation(s.quat_xyzw)[:, 0] * s.height / 2, s.radius)
               for s in link.shapes if hasattr(s, "height") and s.collide]
        out[link.name] = np.concatenate(pts) if pts else np.zeros((1, 3))
    return out


SAMPLES = _samples()
CONE_SAMPLES = np.array([(float(cone_radius(sv)) * math.cos(ph), sv, float(cone_radius(sv)) * math.sin(ph))
                         for sv in np.linspace(-CONE_L / 2, CONE_L / 2, 27)
                         for ph in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)])  # its surface (cone frame)
# each fingertip: points along the end bone's axis (start pose, relative to ORIGIN, m), and its radius
TIP_AXIS = {}
for _n in hm.ORDER:
    _k, _p, _q, _t = hm.finger_points(_n)
    TIP_AXIS[_n] = (np.array([(_q + (_t - _q) * s) * CM for s in np.linspace(0.3, 1.0, 6)]),
                    hm.FINGERS[_n]["r"][2] * CM)


def cone_distance(points, cone_pos, cone_rot) -> np.ndarray:
    """Distance (m) from each point to the cone's surface (negative inside), from its axis distance and radius."""
    q = (np.asarray(points, float) - cone_pos) @ cone_rot
    s = q[:, 1]
    rho = np.hypot(q[:, 0], q[:, 2])
    sc = np.clip(s, -CONE_L / 2, CONE_L / 2)
    side = (rho - cone_radius(sc)) * math.cos(CONE_ALPHA)
    beyond = np.abs(s) - CONE_L / 2
    return np.where(beyond > 0, np.maximum(side, beyond), side)


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("PALM_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _angles(q: dict) -> dict:
    """Joint angles (deg): each finger's knuckle (MCP) and middle-joint (PIP) bend, and how far the thumb is out."""
    out = {}
    kn = math.degrees(q["knuckles_turn"])
    for name in hm.ORDER:
        f = hm.FINGERS[name]
        out[f"mcp_{name}"] = f["mcp"] + kn + math.degrees(q[f"p_{name}_turn"])
        out[f"pip_{name}"] = f["pip"] + math.degrees(q[f"d_{name}_turn"])
    out["thumb_out"] = -math.degrees(q["thumb_turn"])
    return out


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "fingers" not in _IDS:
        _IDS["fingers"] = _finger_points()
    fingers, gaps = {}, {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[side + "_f"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
            _IDS[side + "_tips"] = [robot.body_names.index(n) for n in ("tip_left", "tip_right")]
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float)
        gaps[side] = float(-q[_IDS[side + "_f"]].sum())
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        fingers[side] = np.concatenate([_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                                        for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])])
    hand = base.scene["care_hand"]
    if "bodies" not in _IDS:
        _IDS["bodies"] = {k: i for i, k in enumerate(hand.body_names)}
        _IDS["joints"] = list(hand.joint_names)
    bp = as_numpy(hand.data.body_link_pose_w)[0]
    jq = as_numpy(hand.data.joint_pos)[0]
    q = dict(zip(_IDS["joints"], (float(v) for v in jq)))
    ang = _angles(q)
    ib = _IDS["bodies"]
    pose = {k: (bp[i][:3] - origin, rotation(bp[i][3:7])) for k, i in ib.items()}
    hp, hr = pose["base"]
    to_hand = lambda p: (np.asarray(p, float) - hp) @ hr  # world -> hand frame (m, relative to the knuckle axis)  # noqa: E731
    cone = base.scene["palm_cone"]
    cpose = as_numpy(cone.data.root_link_pose_w)[0]
    cp, cr = cpose[:3] - origin, rotation(cpose[3:7])

    # the hand on its pad: its knuckle end and its forearm end
    ends = np.array([hp, hp + hr @ np.array([hm.FOREARM_END * CM, 0.0, 0.0])])
    if _EP["ref"] is None:
        _EP["settle"] += 1
        if _EP["settle"] >= 10:  # read where everything rests once it has settled (the first 10 control steps)
            _EP["ref"] = dict(ends=ends.copy(), angles=dict(ang))
    lift = move = 0.0
    if _EP["ref"] is not None:
        d = ends - _EP["ref"]["ends"]
        lift = float(d[:, 2].max())
        move = float(np.linalg.norm(d[:, :2], axis=1).max())
    prev = _EP["prev"]
    knuckles = [f"mcp_{n}" for n in hm.ORDER]
    opening = 0.0 if prev is None else max((prev[k] - ang[k]) * 20.0 for k in knuckles)
    closing = 0.0 if prev is None else max((ang[k] - prev[k]) * (-20.0 if k == "thumb_out" else 20.0) for k in ang)
    speed = max(opening, closing)
    _EP["prev"] = dict(ang)

    # fingertips on the cone
    tips = {}
    for name in hm.ORDER:
        axis, r = TIP_AXIS[name]
        dp, dr = pose[f"d_{name}"]
        world = dp + axis @ dr.T
        tips[name] = float(cone_distance(world, cp, cr).min() - r)
    # where the cone lies in the hand
    a_w = cr[:, 1]  # its axis, toward the narrow end
    a_h = hr.T @ a_w
    c_h = to_hand(cp)
    narrow_h = to_hand(cp + a_w * CONE_L / 2)
    surf = to_hand(cp + CONE_SAMPLES @ cr.T)
    over = surf[(np.abs(surf[:, 1]) <= PALM_HALF) & (surf[:, 0] >= 0.0) & (surf[:, 0] <= PALM_LEN)]
    low_h = float(over[:, 2].min()) if len(over) else 1.0  # its lowest point over the palm
    axis_deg = math.degrees(math.acos(max(-1.0, min(1.0, a_h[1]))))
    palm_z = hm.PALM_Z * CM  # the palm's surface (hand frame)
    seat = dict(axis=axis_deg, back=c_h[0], narrow=narrow_h[1] - INDEX_EDGE, low=low_h - palm_z)
    seated = bool(axis_deg <= AXIS_TOL and 0.0 <= c_h[0] <= SEAT_BACK
                  and -NARROW_IN <= narrow_h[1] - INDEX_EDGE <= NARROW_OUT and low_h - palm_z <= ON_PALM)

    # the gripper fingers' clearance from the hand and the cone
    near = 1.0
    world_pts = [cp + CONE_SAMPLES @ cr.T]
    for k, pts in SAMPLES.items():
        bpos, brot = pose[k]
        world_pts.append(bpos + pts @ brot.T)
    tree = cKDTree(np.concatenate(world_pts))
    near_side = {side: float(tree.query(f, k=1)[0].min()) for side, f in fingers.items()}
    near = min(near_side.values())

    # rules
    if not _EP["failed"] and _EP["ref"] is not None:
        tr = _EP["touch_ref"]
        for name in hm.ORDER:
            if tips[name] <= TOUCH:
                ref = tr.setdefault(name, ang[f"mcp_{name}"])
                if ang[f"mcp_{name}"] < ref - CONE_FORCE:
                    _EP["failed"] = f"the {name} finger was pushed open by the cone (forced into the fingers)"
            else:
                tr.pop(name, None)
        bad_mcp = [n for n in hm.ORDER if ang[f"mcp_{n}"] < KNUCKLE_MIN]
        bad_pip = [n for n in hm.ORDER if ang[f"pip_{n}"] < PIP_MIN]
        if _EP["failed"]:
            pass
        elif opening > MAX_OPEN:
            _EP["failed"] = f"a finger was jerked open ({opening:.0f} deg/s at the knuckle)"
        elif closing > MAX_CLOSE:
            fast = max(ang, key=lambda k: (ang[k] - prev[k]) * (-1 if k == "thumb_out" else 1))
            _EP["failed"] = f"a finger was let snap shut ({fast}, {closing:.0f} deg/s)"
        elif bad_mcp:
            _EP["failed"] = f"the {bad_mcp[0]} finger was opened past straight at its knuckle"
        elif bad_pip:
            _EP["failed"] = f"the {bad_pip[0]} finger's middle joint was straightened too far"
        elif ang["thumb_out"] > THUMB_MAX:
            _EP["failed"] = "the thumb was pushed out too far"
        elif lift > LIFT_MAX:
            _EP["failed"] = "the hand was lifted off the pad"
        elif move > MOVE_MAX:
            _EP["failed"] = "the hand was moved along the pad"
    if not _EP["failed"] and _EP["ref"] is not None:
        if all(ang[f"mcp_{n}"] <= OPEN_KNUCKLE for n in hm.ORDER):
            _EP["stage"] = max(_EP["stage"], 0.25)
        if seated:
            _EP["stage"] = max(_EP["stage"], 0.5)
            if all(tips[n] <= TOUCH for n in hm.ORDER):
                _EP["stage"] = max(_EP["stage"], 0.75)
    _EP.update(q=q, angles=ang, speed=speed, lift=lift, move=move, near=near, near_side=near_side, tips=tips,
               seated=seated, seat=seat, gaps=gaps, cone=cp.copy(), peak_speed=max(_EP["peak_speed"], speed),
               min_knuckle=min(_EP["min_knuckle"], min(ang[f"mcp_{n}"] for n in hm.ORDER)))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "cone" in _EP and _EP["angles"]:
        _EP["counted"] = _EP["tick"]
        recent.append((np.asarray(_EP["cone"], float).copy(), dict(_EP["angles"])))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(
        np.abs(p - recent[-1][0]).max() * 1000 < STILL_MM
        and max(abs(a[k] - recent[-1][1][k]) for k in a) < STILL_DEG for p, a in recent)
    resting = bool(_EP["seated"] and _EP["tips"] and all(v <= TOUCH for v in _EP["tips"].values()))
    done = bool(resting and _EP["near"] >= CLEAR and still)
    return dict(still=still, resting=resting, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or not _EP["angles"]:
        return
    a = _EP["angles"]
    print(f"DBG step={_EP['step']} stage={_EP['stage']} mcp=" + ",".join(f"{a['mcp_' + n]:.0f}" for n in hm.ORDER)
          + " pip=" + ",".join(f"{a['pip_' + n]:.0f}" for n in hm.ORDER) + f" thumb={a['thumb_out']:.1f}"
          f" speed={_EP['speed']:.0f} lift={_EP['lift'] * 1000:.1f}mm move={_EP['move'] * 1000:.1f}mm"
          f" tips=" + ",".join(f"{v * 1000:.1f}" for v in _EP["tips"].values())
          + f" seated={int(_EP['seated'])} seat=" + ",".join(f"{k}:{v * (1 if k == 'axis' else 1000):.0f}"
                                                           for k, v in _EP["seat"].items())
          + f" near={_EP['near'] * 1000:.0f}mm still={int(t['still'])} failed={_EP['failed'] or '-'} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     f" near={_EP['near_side'][k] * 1000:.0f}" for k, v in _TCP.items()), flush=True)
    if "q" in _EP:
        print("DBGQ " + " ".join(f"{k.replace('_turn', '')}={math.degrees(v):+.1f}" for k, v in _EP["q"].items()),
              flush=True)


def score(objects) -> float:
    """0 for good once a rule is broken; otherwise the best stage reached (0.25 fingers held open, 0.5 cone seated,
    0.75 fingertips resting on it). Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if _EP["failed"]:
        return 0.0
    return float(_EP["stage"])


def success(objects) -> bool:
    """Cone seated, all four fingertips on it, both grippers 3 cm or more away, cone and fingers still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper comes down over the fist, jaws across it front to back, and
# closes on the middle and ring fingers' curl (front jaw on the backs of their first bones, rear jaw on their end
# joints); it turns the curled fingers open about the knuckle axis, 80 deg in 6 s, and holds them there (the other
# two follow on the common knuckle spring). The right gripper, its wrist leaning 30 deg to the robot's right, picks the
# cone up by its wide end, carries it over and lays it across the palm, narrow end at the thumb, 2.2 cm behind the
# knuckles. The left gripper then turns the fingers
# back down until their tips are just over the cone, loosens its grip so that they lean on its rear jaw and eases them
# on down onto the cone; the right gripper lets go of the cone and rises; the left opens slowly, so that the fingers
# follow its rear jaw down and settle on the cone, turns its open jaws on, away from the fingers, and lifts clear.
# PALM_ONE=1 runs the wrong way: the right gripper alone, the left arm stays where it is. PALM_LETGO=1 shows what one
# gripper meets the other way round: the left gripper opens the fingers and then lets go of them (to fetch the cone),
# the right arm stays where it is.

ONE = os.environ.get("PALM_ONE") == "1"
LETGO = os.environ.get("PALM_LETGO") == "1"
OPEN_C = -0.047
ROT_A = grip_rotation(-math.pi / 2)  # left gripper: jaws closing along x
ROT_B = _rx(math.radians(30.0)) @ grip_rotation(math.pi / 2)  # right gripper: jaws closing along x, its wrist
# leaning 30 deg toward the robot's right (clear of the left gripper over the hand)
# the pinch on the middle and ring fingers' curl (hand frame, cm): centre, and the jaws' span across it
PINCH = np.array([1.4, -0.5, 4.0])
PINCH_SPAN = 4.6  # cm between the front of the first bones and the back of the end joints
PINCH_C = -(PINCH_SPAN * CM - 0.008) / 2  # finger value: 4 mm inside each face
OPEN_DEG = 80.0  # the fingers turned open this far (the knuckles from 90 to 10 deg)
OPEN_S = 6.0  # in this time
TOUCH_DEG = 46.0  # then turned back down to 44 deg at the knuckles: the fingertips just over the cone
DOWN_DEG = 43.0  # and, the grip loosened, on down to 47 deg at the knuckles as the fingertips settle on it
EASE_C = -0.026  # the loosened grip: the jaws 5.2 cm apart
AWAY_DEG = 30.0  # after letting go, the open jaws turn on to here (60 deg at the knuckles), clear of fingers and cone
NARROW_AT = 5.0  # the cone's narrow end laid 1.55 cm beyond the index finger's outer edge, over the thumb (y, cm)
CONE_AT = np.array([2.2, NARROW_AT - CONE_L / 2 / CM, 0.0])  # cone centre in the palm (hand frame x, y; cm)
CONE_DROP = 0.002  # the cone held this far above the palm when laid in
CONE_PINCH = -(2 * float(cone_radius(CONE_GRIP_Y)) - 0.007) / 2


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


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
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


def pinch_pose(turn_deg: float):
    """(TCP, rotation) of the left gripper holding the curl with the fingers turned ``turn_deg`` open about the
    knuckle axis."""
    r = _ry(-math.radians(turn_deg))  # opening the fingers is a negative turn about +y
    return ORIGIN + r @ (PINCH * CM), r @ ROT_A


def cone_target():
    """(TCP, rotation) of the right gripper with the cone laid in the palm."""
    c = ORIGIN + np.array([CONE_AT[0] * CM, CONE_AT[1] * CM, 0.0])
    low = float((CONE_P @ R_CONE0.T)[:, 2].min())
    palm = hm.PALM_Z * CM
    c[2] = ORIGIN[2] + palm - low + CONE_DROP
    return c + R_CONE0 @ np.array([0.0, CONE_GRIP_Y, 0.0]) + np.array([0.0, 0.0, -0.004]), ROT_B


def cone_grip0():
    """(TCP, rotation) of the right gripper on the cone where it lies."""
    return CONE_POS0 + R_CONE0 @ np.array([0.0, CONE_GRIP_Y, 0.0]) + np.array([0.0, 0.0, -0.004]), ROT_B


T_OPENED = 3.0 + 1.0 + 2.0 + 1.0 + 0.5 + OPEN_S  # the left gripper has the fingers open (s)
T_CONE_IN = T_OPENED + 5.5  # the right gripper has laid the cone in (s)


def _left() -> list:
    p0, r0 = pinch_pose(0.0)
    above = p0 + np.array([0.0, 0.0, 0.08])
    out = _lead_in("left", above, r0, OPEN_C, "above the fist, jaws across it")
    out += [_w(1.0, above, -0.033, "jaws 6.6 cm apart", r0),
            _w(2.0, p0, -0.033, "down round the middle and ring fingers' curl", r0),
            _w(1.0, p0, PINCH_C, "close on the curl", r0),
            _w(0.5, p0, PINCH_C, "settle", r0)]
    steps = 16
    for k in range(1, steps + 1):
        a = OPEN_DEG * (0.5 - 0.5 * math.cos(math.pi * k / steps))
        p, r = pinch_pose(a)
        out.append(_w(OPEN_S / steps, p, PINCH_C, "turn the curled fingers open about the knuckles", r))
    p, r = pinch_pose(OPEN_DEG)
    out.append(_w(T_CONE_IN - T_OPENED + 0.5, p, PINCH_C, "hold the fingers open", r))
    for k in range(1, steps + 1):
        a = OPEN_DEG - (OPEN_DEG - TOUCH_DEG) * (0.5 - 0.5 * math.cos(math.pi * k / steps))
        p, r = pinch_pose(a)
        out.append(_w(3.0 / steps, p, PINCH_C, "ease the fingers down until their tips touch the cone", r))
    p, r = pinch_pose(TOUCH_DEG)
    out += [_w(0.5, p, PINCH_C, "settle", r),
            _w(1.0, p, EASE_C, "loosen the grip: the fingers lean on the rear jaw", r)]
    for k in range(1, steps + 1):
        a = TOUCH_DEG - (TOUCH_DEG - DOWN_DEG) * (0.5 - 0.5 * math.cos(math.pi * k / steps))
        p, r = pinch_pose(a)
        out.append(_w(3.0 / steps, p, EASE_C, "keep easing them down as they fold over the cone", r))
    p, r = pinch_pose(DOWN_DEG)
    out += [_w(0.5, p, EASE_C, "settle", r),
            _w(3.0, p, EASE_C, HOLD_LABEL, r)]
    for k in range(1, 9):  # open slowly: the rear jaw draws back the way the fingers close, and they follow it down
        out.append(_w(3.0 / 8, p, EASE_C + (OPEN_C - EASE_C) * k / 8, "let the fingers down onto the cone", r))
    for k in range(1, 7):  # turn the open jaws on, the way the fingers close: away from the backs of the fingers
        p, r = pinch_pose(DOWN_DEG - (DOWN_DEG - AWAY_DEG) * k / 6)
        out.append(_w(1.5 / 6, p, OPEN_C, "turn the open jaws on, away from the fingers", r))
    up = p + np.array([-0.02, 0.0, 0.05])
    out += [_w(1.5, up, OPEN_C, "lift clear", r),
            _w(1.5, up + np.array([-0.05, 0.0, 0.04]), OPEN_C, "rise clear", r),
            _w(8.0, up + np.array([-0.05, 0.0, 0.04]), OPEN_C, "stay clear", r)]
    return out


HOLD_LABEL = "keep the fingers resting on the cone while the cone is let go"


def _right(t_let_go: float) -> list:
    g0, r = cone_grip0()
    gt, _ = cone_target()
    above0 = g0 + np.array([0.0, 0.0, 0.08])
    above_t = gt + np.array([0.0, 0.0, 0.07])
    out = _lead_in("right", above0, r, OPEN_C, "above the cone's wide end")
    out += [_w(1.5, g0, OPEN_C, "down round the wide end", r),
            _w(1.0, g0, CONE_PINCH, "take the cone by its wide end", r),
            _w(0.5, g0, CONE_PINCH, "settle", r),
            _w(1.5, above0, CONE_PINCH, "lift it", r)]
    t = sum(w.duration for w in out)
    if t < T_OPENED:
        out.append(_w(T_OPENED - t, above0, CONE_PINCH, "wait for the fingers to be open", r))
    out += [_w(2.0, above_t, CONE_PINCH, "carry it over the palm", r),
            _w(2.5, gt, CONE_PINCH, "lay it across the palm, narrow end at the thumb", r),
            _w(1.0, gt, CONE_PINCH, "hold it there", r)]
    t = sum(w.duration for w in out)
    out.append(_w(max(t_let_go - t, 0.5), gt, CONE_PINCH, "hold it while the fingers come down on it", r))
    out += [_w(1.0, gt, OPEN_C, "let go of the cone", r),
            _w(1.5, gt + np.array([0.0, 0.0, 0.08]), OPEN_C, "rise", r),
            _w(1.5, gt + np.array([0.0, -0.08, 0.10]), OPEN_C, "move clear", r),
            _w(6.0, gt + np.array([0.0, -0.08, 0.10]), OPEN_C, "stay clear", r)]
    return out


def _stay(arm: str) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(80.0, tuple(float(v) for v in pos), OPEN_C, "stay put", rot)]


def demo():
    left = _left()
    k = [w.label for w in left].index(HOLD_LABEL)
    right = _right(sum(w.duration for w in left[:k]))  # the cone is let go once the fingers rest on it
    if ONE:
        return {"left": _stay("left"), "right": right}
    if LETGO:
        i = [w.label for w in left].index("hold the fingers open")
        p, r = pinch_pose(OPEN_DEG)
        back = p - r[:, 2] * 0.06
        let_go = left[:i] + [_w(1.0, p, PINCH_C, "hold the fingers open", r),
                             _w(1.0, p, -0.035, "let go of the fingers to fetch the cone", r),
                             _w(1.5, back, -0.035, "back out", r), _w(10.0, back, -0.035, "stay", r)]
        return {"left": let_go, "right": _stay("right")}
    return {"left": left, "right": right}
