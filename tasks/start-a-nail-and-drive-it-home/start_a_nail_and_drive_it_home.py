"""On the station table: a 250 mm offcut of western red cedar 2x6 (38 x 140 mm) standing on its narrow edge, held to
the table by two steel angle brackets, with a pencil cross on its top edge; a 4d finishing nail (1-1/2 in.) lying
loose on the table beyond the board's left end; a 7 oz claw hammer lying on its side on the table to the right.

Task (instruction): drive the nail into the board at the pencil mark. The instruction does not say how. The procedure
the task checks is the US Army and Air Force hand-tool manual (TM 9-243 / TO 32-1-101, "Using a carpenter's hammer"):
"Support a common nail between thumb and first finger about halfway up the nail." "Tap nail head with hammer face until
the nail will remain standing by itself." "Remove fingers and drive nail flush. Make sure the hand supporting the work
is not in direct line with the hammer blows." A nail stood on its point falls over when it is let go, so one hand has
to hold it upright while the other taps it in until it stands; then the holding hand must get out of the hammer's way
before the nail is driven home.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- board: fixed, a 250 mm length of 2x6 (38 x 140 mm) on edge: x 0.191 to 0.229, y -0.078 to 0.172, its top edge 14 cm
  above the table. A pencil cross on the top edge marks where the nail goes: x 0.210, y 0.047.
- nail: loose, a 4d finishing nail (38.1 mm under the head, 1.83 mm shank, 2.4 mm head, 0.78 g) lying on the table
  along x, its point toward the robot at x 0.170, y 0.242.
- hammer: loose, a 7 oz claw hammer (Stanley 51-613: 198 g head, 23 mm face, 111 mm head, 286 mm long, 0.34 kg in all)
  lying on its side on the table, its head at x 0.101, y -0.107, its handle running back to the right.

The nail in the wood (the model, acting in a 20 mm square under the mark, where the board's collision shape is open
under its drawn top face): the nail's point is held where it first touches the wood there; it sinks only when the
hammer's face strikes its head at 0.3 m/s or more (a new blow once the face has lifted 8 mm off the head). The wood
resists it with R(d) = 5.7 N/mm x (d + 2 mm) at depth d (Wood Handbook eq. 8-1 for this 1.83 mm shank in wood of
specific gravity 0.32; the extra 2 mm stands for the point), and each blow drives it until that resistance has taken
0.7 x the hammer's kinetic energy. A steady push never drives it. Until it is 5.5 mm in (three shank diameters) it has
no side support and falls over if nothing holds it (tipped past 25 deg it pops out); from then on it stands by itself
at the lean it has. A blow more than 15 deg off the nail's axis (the face's path or its tilt) bends the nail.

Score: 0 for the rest of the run once the nail is bent, or once the hammer has struck a gripper other than the one
holding it (the one holding the nail, say). Otherwise the best stage reached: 0.25 the nail started at the mark (2 mm
or more in, its point within 3 mm of the mark's middle); 0.5 standing on its own at the mark (8 mm or more in, upright
within 5 deg, no gripper touching it); 0.75 driven home (its head flush with the board, within 1 mm, upright within
5 deg, at the mark, no gripper touching it). Full score: that, with the hammer let go (no gripper touching it, at rest)
and both grippers 5 cm or more from the nail, still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Joint, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"

RIGHT_BASE = np.array([0.0, -0.305])
FACE_OFF = np.array([-0.0025, -0.003])  # where the face's middle is sent, from the nail; a stroke lands it about 5 mm
# further along +x and 3 mm along +y (the arm swings out as it strikes, and the strokes are sent nose-up), so that the
# face lands 2 to 5 mm beyond the nail, the whole head under it, its rim clear of the fingers holding the nail

# -- board: a 2x6 on edge, with a pencil cross on its top edge ---------------------------------------------------------
MX, MY = 0.210, 0.047  # the pencil mark
BT, BHT = 0.038, 0.140  # board thickness (x) and height (z): a 2x6 on edge
BY0, BY1 = MY - 0.125, MY + 0.125  # its ends (250 mm)
SURF = TOP + BHT  # its top edge
CEDAR, END_GRAIN, PENCIL, BRACKET = (0.80, 0.56, 0.42), (0.70, 0.46, 0.33), (0.20, 0.20, 0.22), (0.55, 0.57, 0.60)
BC = (MX, (BY0 + BY1) / 2, TOP)  # the board body's origin: middle of its bottom


def _box(x0, x1, y0, y1, z0, z1, color, origin, collide=True) -> BoxShape:
    """A box from world corners, in the frame of a body whose origin is ``origin``."""
    return BoxShape(((x0 + x1) / 2 - origin[0], (y0 + y1) / 2 - origin[1], (z0 + z1) / 2 - origin[2]),
                    (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


HOLE, HOLE_D = 0.020, 0.045  # the wood under the mark: a 20 mm square, 45 mm deep, where the nail model acts


def _board_shapes() -> list:
    """The board. Under the pencil mark its collision shape leaves a 20 mm square, 45 mm deep, open (its top face is
    drawn over it): there the nail meets the wood through the model (the joint below), not through contact."""
    o, x0, x1 = BC, MX - BT / 2, MX + BT / 2
    h0, h1, hy0, hy1, z0 = MX - HOLE / 2, MX + HOLE / 2, MY - HOLE / 2, MY + HOLE / 2, SURF - HOLE_D
    out = [_box(x0, x1, BY0, BY1, TOP, z0, CEDAR, o),
           _box(x0, x1, BY0, hy0, z0, SURF, CEDAR, o), _box(x0, x1, hy1, BY1, z0, SURF, CEDAR, o),
           _box(x0, h0, hy0, hy1, z0, SURF, CEDAR, o), _box(h1, x1, hy0, hy1, z0, SURF, CEDAR, o),
           _box(h0, h1, hy0, hy1, SURF - 0.0004, SURF, CEDAR, o, collide=False)]
    for y in (BY0, BY1):  # end grain, a shade darker (visual only)
        out.append(_box(x0, x1, y - 0.0004, y + 0.0004, TOP, SURF, END_GRAIN, o, collide=False))
    # the pencil cross: two strokes 14 mm long, 0.7 mm wide, drawn on the top edge (visual only)
    out.append(_box(MX - 0.007, MX + 0.007, MY - 0.00035, MY + 0.00035, SURF, SURF + 0.0002, PENCIL, o, collide=False))
    out.append(_box(MX - 0.00035, MX + 0.00035, MY - 0.007, MY + 0.007, SURF, SURF + 0.0002, PENCIL, o, collide=False))
    # two steel angle brackets on the back face, screwed to the board and the table
    for yc in (BY0 + 0.03, BY1 - 0.03):
        out.append(_box(x1, x1 + 0.0025, yc - 0.019, yc + 0.019, TOP, TOP + 0.038, BRACKET, o))
        out.append(_box(x1, x1 + 0.038, yc - 0.019, yc + 0.019, TOP, TOP + 0.0025, BRACKET, o))
    return out


# -- nail: a 4d finishing nail (1-1/2 in. long, 15-gauge 0.072 in. = 1.83 mm shank, 3/32 in. = 2.4 mm head, 0.78 g of
# steel). Body frame: origin at the point, z along the nail toward the head ------------------------------------------
N_LEN, N_R, N_HEAD_R, N_HEAD_T, N_KG = 0.0381, 0.000914, 0.00119, 0.0008, 0.00078
N_POINT = 0.0025  # the point, modelled as a thin stub and a ball at its tip
STEEL = (0.66, 0.67, 0.70)


SHANK_MU, STEEL_MU = (1.0, 0.9), (0.4, 0.35)  # a bright nail's shank in rubber finger pads; steel on steel


def _nail_shapes() -> list:
    return [SphereShape((0.0, 0.0, 0.0003), 0.0003, STEEL),
            CylShape((0.0, 0.0, 0.0003 + (N_POINT - 0.0003) / 2), 0.0006, N_POINT - 0.0003, STEEL),
            CylShape((0.0, 0.0, (N_POINT + N_LEN) / 2), N_R, N_LEN - N_POINT, STEEL, friction=SHANK_MU),
            CylShape((0.0, 0.0, N_LEN + N_HEAD_T / 2), N_HEAD_R, N_HEAD_T, STEEL, friction=STEEL_MU)]


NAIL_TIP0 = np.array([0.170, 0.242])  # where its point lies at the start (left of the board); the nail runs along +x
QY90 = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))  # about y by -90 deg: a joint frame's x axis along the nail
BIG = 1.0  # m: a limit this wide does not hold anything on the table
SWING_FREE = 170.0  # deg: the nail's tilt is free up to this (a nail lying on the table is at 90)
# The nail is jointed to the board from the start by a D6 joint whose frames are the mark (x axis up) and the nail's
# point (x axis along the nail). Its translations are limited to +-1 m and its tilt to 170 deg (nothing on the table);
# while the simulation runs the task file narrows them to hold the point where it pricked the wood, at its depth, and,
# once it is set, the nail's tilt. (PhysX only takes new limit values at runtime, so every limit exists from the start.)
NAIL = Body.build("nail", pos=(float(NAIL_TIP0[0]), float(NAIL_TIP0[1]), TOP + N_HEAD_R + 0.0008), mass=N_KG,
                  color=STEEL, shapes=_nail_shapes(), init_quat=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
                  friction=(0.5, 0.45), solver_iterations=255, sleeps=False,
                  joint=Joint(anchor=(MX, MY, SURF), kind="free", parent="board",
                              limits=(("transX", -BIG, BIG), ("transY", -BIG, BIG), ("transZ", -BIG, BIG),
                                      ("rotY", -SWING_FREE, SWING_FREE), ("rotZ", -SWING_FREE, SWING_FREE)),
                              frames=(((0.0, MY - BC[1], BHT), QY90), ((0.0, 0.0, 0.0), QY90))))

# -- hammer: a 7 oz claw hammer (Stanley 51-613). Body frame: origin at the eye (the head's middle on the handle's
# axis), z up the head toward the claw (the face down at z = -FACE_D), the handle along -y. ------------------------
HEAD_KG, HANDLE_KG = 0.198, 0.142
FACE_D, FACE_R, CLAW_UP, HEAD_W = 0.050, 0.0115, 0.061, 0.022
HANDLE_END, HANDLE_H, HANDLE_W = 0.273, 0.028, 0.019  # handle: to 273 mm from the eye, 28 x 19 mm
HEAD_C, WOODC, HEADC = (0.18, 0.19, 0.21), (0.78, 0.62, 0.38), (0.24, 0.25, 0.27)


def _box_inertia(m, a, b, c) -> np.ndarray:
    return np.array([m / 12 * (b * b + c * c), m / 12 * (a * a + c * c), m / 12 * (a * a + b * b)])


def _hammer_mass():
    """Mass, centre of mass and principal moments (body axes) of a steel head on a hickory handle."""
    parts = [(HEAD_KG, np.array([0.0, 0.0, -0.006]), _box_inertia(HEAD_KG, HEAD_W, HEAD_W + 0.004, FACE_D + CLAW_UP)),
             (HANDLE_KG, np.array([0.0, -(HEAD_W / 2 + HANDLE_END) / 2, 0.0]),
              _box_inertia(HANDLE_KG, HANDLE_W, HANDLE_END - HEAD_W / 2, HANDLE_H))]
    m = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m
    inertia = np.zeros(3)
    for mi, ci, ii in parts:
        d = ci - com
        inertia += ii + mi * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return m, com, inertia


H_KG, H_COM, H_I = _hammer_mass()
# The right gripper takes the handle choked up, 31 mm from the head's middle (its fingers 0.6 mm clear of the head), and
# the blows are straight-down strokes of the whole arm. A two-finger pinch holds little turning load (about 50 N of grip
# on pads 2.6 cm wide: about 1 N m), so it cannot swing the hammer from the wrist, and a blow, stopping the face, turns
# the hammer in the fingers by more the further the grip is from the face: held at its balance point, 59 mm from the
# head, a full stroke turned it about 6 deg and slid it 2 to 3 mm along its handle (measured in this simulation).
L_H = 0.031

# -- the hammer's striking posture: level, fingers closing vertically, pointing along heading H, in the arm's plane --
_FACE_XY = np.array([MX, MY]) + FACE_OFF
_R_GRIP = math.sqrt(float((_FACE_XY - RIGHT_BASE) @ (_FACE_XY - RIGHT_BASE)) - L_H ** 2)
H = math.atan2(_FACE_XY[1] - RIGHT_BASE[1], _FACE_XY[0] - RIGHT_BASE[0]) - math.atan2(L_H, _R_GRIP)
AP = np.array([math.cos(H), math.sin(H), 0.0])  # the right gripper's heading while striking
U = np.array([-math.sin(H), math.cos(H), 0.0])  # along the handle, from the grip toward the head
STRIKE_XY = RIGHT_BASE + _R_GRIP * AP[:2]  # the grip while striking: the face's middle over the nail


def _hammer_shapes() -> list:
    mu = STEEL_MU  # the steel head; the hickory handle takes the body's friction
    out = [CylShape((0.0, 0.0, -FACE_D + 0.007), FACE_R, 0.014, HEADC, friction=mu),  # the face end
           BoxShape((0.0, 0.0, (-FACE_D + 0.014 + 0.014) / 2), (HEAD_W - 0.004, HEAD_W - 0.002, FACE_D - 0.014 + 0.014),
                    HEAD_C, friction=mu),  # the neck up to the eye
           BoxShape((0.0, 0.0, 0.0), (HEAD_W, HEAD_W + 0.006, 0.030), HEAD_C, friction=mu)]  # the eye
    for s in (-1.0, 1.0):  # the claw: two prongs curving back over the handle
        out.append(BoxShape((s * 0.0045, -0.006, 0.030), (0.006, 0.016, 0.022), HEAD_C, friction=mu))
        out.append(BoxShape((s * 0.0045, -0.016, 0.050), (0.006, 0.012, 0.024), HEAD_C, friction=mu, quat_xyzw=(
            math.sin(math.radians(-25) / 2), 0.0, 0.0, math.cos(math.radians(-25) / 2))))
    out.append(BoxShape((0.0, -(HEAD_W / 2 + 0.003 + HANDLE_END) / 2, 0.0),
                        (HANDLE_W, HANDLE_END - HEAD_W / 2 - 0.003, HANDLE_H), WOODC))
    return out


def _quat(m) -> tuple:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def _axis_rot(axis, deg) -> np.ndarray:
    a = math.radians(deg)
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


R_HAM_USE = np.column_stack([AP, U, [0.0, 0.0, 1.0]])  # face down, handle back along -U from the head
R_HAM_LYING = _axis_rot(U, 90.0) @ R_HAM_USE  # on its side, face toward the robot
PICK_R = 0.22  # the grip's distance from the right arm's base axis where the hammer lies
HAMMER_GRIP0 = np.array([RIGHT_BASE[0] + PICK_R * AP[0], RIGHT_BASE[1] + PICK_R * AP[1]])
HEAD0 = HAMMER_GRIP0 + L_H * U[:2]
HAMMER = Body.build("hammer", pos=(float(HEAD0[0]), float(HEAD0[1]), TOP + FACE_R + 0.001),
                    mass=float(H_KG), color=HEAD_C, shapes=_hammer_shapes(), init_quat=_quat(R_HAM_LYING),
                    com=tuple(float(v) for v in H_COM), inertia=tuple(float(v) for v in H_I), friction=(0.8, 0.7),
                    ccd=True, solver_iterations=32)

BOARD = Body.build("board", pos=BC, mass=1.2, fixed=True, color=CEDAR, shapes=_board_shapes(), friction=(0.6, 0.5))
OBJECTS = [BOARD, NAIL, HAMMER]

# -- the nail in the wood: numbers -----------------------------------------------------------------------------------
SG = 0.32  # specific gravity of western red cedar at 12 % moisture (Western Red Cedar designer handbook)
P_N_PER_M = 54.12 * SG ** 2.5 * (2 * N_R * 1000) * 1000  # Wood Handbook eq. 8-1: 5.7 N per mm of depth for this nail
D0 = 0.002  # the point's resistance, as 2 mm more shank in the wood
ETA = 0.7  # share of the hammer's energy that drives the nail (assumed; to be measured on the real board)
# From this depth the nail stands by itself: its point (about 2 shank diameters long) and as much shank again in the
# wood, three shank diameters, 5.5 mm (to be measured on the real board: the depth at which a tapped nail stays up).
D_SET = 3 * 2 * N_R
V_MIN = 0.3  # m/s: slower contact is a push, which never drives the nail
REARM = 0.008  # m: a new blow counts only after the face has lifted this far off the head (a rebound is part of a blow)
BEND_DEG = 15.0
POP_DEG = 25.0  # a nail not yet set that tips this far pops out of its shallow hole
FLUSH = N_LEN  # driven home: the head's underside on the wood
EPS = 1e-5

# -- scoring numbers ---------------------------------------------------------------------------------------------------
STARTED, STAND_D = 0.002, 0.008
AT_MARK = 0.003
UPRIGHT_DEG = 5.0
FLUSH_TOL = 0.001
TOUCH = 0.0015  # a gripper this close to the nail or the hammer touches it
CLEAR = 0.05
STILL_MM, STILL_STEPS = 1.0, 20
_DEBUG = os.environ.get("NAIL_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("NAIL_DEBUG_EVERY", "10"))

_EP: dict = {}
_JOINT: list = []
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, sub=0, wrapped=False, mode="loose", d=0.0, set=False, pin=None, failed="", best=0.0,
               blows=[], armed=True, prev=None, coll_on_at=None, tilt=90.0, lat=1.0, tip=None, head=None,
               nail_touch=True, ham_touch=True, ham_rest=False, near=0.0, hist=[], recent=[], counted=-1,
               driven=False, stand=False, started=False, last_blow=None, hit_v=0.0)
    _JOINT.clear()


_reset()


# -- runtime access to the joint ---------------------------------------------------------------------------------
def _joint_prim():
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    if not _JOINT:
        for prim in stage.Traverse():
            if str(prim.GetPath()).endswith("/nail/joint"):
                _JOINT.append(str(prim.GetPath()))
                break
    return stage.GetPrimAtPath(_JOINT[0])


_LIM: dict = {}


def _limit(dof: str, lo: float, hi: float) -> None:
    """Set a D6 translation limit to [lo, hi], writing the two ends in an order that never leaves lo > hi."""
    prim = _joint_prim()
    cur = _LIM.get(dof, (-BIG, BIG))
    first_hi = lo > cur[1]
    for name, v in ((("high", hi), ("low", lo)) if first_hi else (("low", lo), ("high", hi))):
        prim.GetAttribute(f"limit:{dof}:physics:{name}").Set(float(v))
    _LIM[dof] = (lo, hi)


_R_JOINT = rotation(QY90)  # the joint's frame in the board's (world-aligned) frame and in the nail's own frame


def _swing_deg(r_nail) -> tuple[float, float]:
    """The joint's two swing angles (deg) as PhysX measures them: the nail's frame relative to the board's, split into
    swing and twist about the joint's x axis (the nail's axis), each swing angle 4 atan(s / (1 + w))."""
    q = np.array(_quat(_R_JOINT.T @ r_nail @ _R_JOINT))  # x, y, z, w
    q = q if q[3] >= 0 else -q
    tw = np.array([q[0], 0.0, 0.0, q[3]]) / max(1e-12, math.hypot(q[0], q[3]))  # the twist; swing = q * twist^-1
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = -tw[0], 0.0, 0.0, tw[3]
    sy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    sz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    sw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return math.degrees(4 * math.atan2(sy, 1 + sw)), math.degrees(4 * math.atan2(sz, 1 + sw))


def _hold_tilt(r_nail) -> None:
    """Hold the nail's tilt where it is: each of the joint's two swing angles within 0.5 deg of its present value."""
    prim = _joint_prim()
    for dof, mid in zip(("rotY", "rotZ"), _swing_deg(r_nail)):
        prim.GetAttribute(f"limit:{dof}:physics:low").Set(float(mid - 0.5))
        prim.GetAttribute(f"limit:{dof}:physics:high").Set(float(mid + 0.5))
    _EP["held_swing"] = _swing_deg(r_nail)


# -- state readers -----------------------------------------------------------------------------------------------
def _pose(base, name):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - as_numpy(base.scene.env_origins[0]), rotation(p[3:7])


def _nail_geom(base):
    tip, r = _pose(base, "nail")
    axis = r[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(axis[2])))))
    head = tip + (N_LEN + N_HEAD_T) * axis  # middle of the head's top
    return tip, axis, tilt, head


def _hammer_state(base):
    data = base.scene["hammer"].data
    pos, r = _pose(base, "hammer")
    vel = as_numpy(data.root_link_vel_w)[0]
    face_l = np.array([0.0, 0.0, -FACE_D])
    face = pos + r @ face_l
    vface = vel[:3] + np.cross(vel[3:6], r @ face_l)
    vcom = vel[:3] + np.cross(vel[3:6], r @ H_COM)
    w_b = r.T @ vel[3:6]
    ke = 0.5 * H_KG * float(vcom @ vcom) + 0.5 * float(w_b @ (H_I * w_b))
    return dict(pos=pos, r=r, face=face, normal=-r[:, 2], vface=vface, ke=ke, speed=float(np.linalg.norm(vel[:3])),
                spin=float(np.linalg.norm(vel[3:6])))


def _gripper_hulls(base) -> dict:
    """World convex hulls of each gripper (body and both fingers), from the arms' joints."""
    from yam_arena import analysis, station
    from yam_arena.constants import ALL_JOINTS

    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ALL_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0][_IDS[side]]
        poses = analysis.fk_link_poses_world(dict(zip(ALL_JOINTS, q)), station.BASES[side])
        hulls = analysis.world_hulls(poses, ["tip_left", "tip_right", "gripper"])
        out[side] = [(h, h.vertices.min(0), h.vertices.max(0)) for _, h in hulls]
    return out


def _distance(points: np.ndarray, hulls) -> float:
    """Smallest signed distance (m) from the points to the hulls (the largest plane distance per hull: a lower bound
    of the true distance outside, the depth inside)."""
    from yam_arena import analysis

    lo, hi = points.min(0), points.max(0)
    best = 1.0
    for h, hlo, hhi in hulls:
        if np.maximum(hlo - hi, lo - hhi).max() > 0.02:
            continue
        best = min(best, float(analysis._signed(points, h).min()))
    return best


_RIM = np.array([[FACE_R * math.cos(a), FACE_R * math.sin(a), -FACE_D] for a in np.linspace(0, 2 * math.pi, 12,
                                                                                               endpoint=False)])
_HEAD_PTS = np.vstack([_RIM, [[0.0, 0.0, -FACE_D]],
                       [[sx * HEAD_W / 2, sy * (HEAD_W + 0.006) / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                        for z in (-FACE_D + 0.014, -0.015, 0.015, 0.04, CLAW_UP)]])
_HANDLE_PTS = np.array([[sx * HANDLE_W / 2, -y, sz * HANDLE_H / 2] for y in np.linspace(0.02, HANDLE_END, 14)
                        for sx in (-1, 1) for sz in (-1, 1)])


def _nail_points(tip, axis) -> np.ndarray:
    """Points on the nail's surface above the wood (the shank's sides and the head's rim)."""
    side = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(side) < 0.1:
        side = np.cross(axis, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    side2 = np.cross(axis, side)
    pts = []
    for s in np.arange(0.0, N_LEN + 0.0001, 0.003):
        c = tip + s * axis
        if c[2] < SURF - 0.0005 and _EP["mode"] == "in":
            continue
        for v in (side, -side, side2, -side2):
            pts.append(c + N_R * v)
    hc = tip + (N_LEN + N_HEAD_T / 2) * axis
    for v in (side, -side, side2, -side2):
        pts.append(hc + N_HEAD_R * v)
    return np.array(pts)


# -- the model, run at every physics step (240 Hz) --------------------------------------------------------------------
def _over_mark(p) -> bool:
    """The point is over the wood the model covers: the 20 mm square under the mark."""
    return abs(p[0] - MX) < HOLE / 2 - N_R and abs(p[1] - MY) < HOLE / 2 - N_R


def _substep(base) -> None:
    _EP["sub"] += 1
    tip, axis, tilt, head = _nail_geom(base)
    ham = _hammer_state(base)
    mode = _EP["mode"]
    # the point meets the wood: it pricks it and stays where it is (pinned sideways); lifted off or fallen flat, free
    if mode == "loose" and _over_mark(tip) and tip[2] - SURF < 0.0005 and tilt < 30.0:
        off = tip[:2] - (MX, MY)
        _limit("transY", off[1] - EPS, off[1] + EPS)
        _limit("transZ", -off[0] - EPS, -off[0] + EPS)
        _limit("transX", 0.0, BIG)
        _EP.update(mode="point", pin=off.copy())
    elif mode == "point" and (tip[2] - SURF > 0.0015 or tilt > 75.0):
        for dof in ("transX", "transY", "transZ"):
            _limit(dof, -BIG, BIG)
        _EP.update(mode="loose", pin=None)
    elif mode == "in" and not _EP["set"] and tilt > POP_DEG:  # tipped over before it was set: it pops out
        _limit("transX", 0.0, BIG)
        _EP.update(mode="point", d=0.0)
    # blows: the face reaches the head moving toward it at V_MIN or more
    rel = head - ham["face"]
    sep = float(rel @ ham["normal"])  # how far the head's top is in front of the face (negative: pressed into it)
    lateral = float(np.linalg.norm(rel - sep * ham["normal"]))
    if sep > REARM or lateral > FACE_R + N_HEAD_R + 0.01:
        _EP["armed"] = True
    prev = _EP["prev"]
    if (_EP["armed"] and -0.01 < sep < 0.0005 and lateral < FACE_R + N_HEAD_R and prev is not None
            and _EP["mode"] in ("point", "in") and not _EP["failed"]):
        _EP["armed"] = False
        v = prev["vface"]
        v_in = float(-(v @ axis))  # approach speed along the nail, toward its point
        if v_in >= V_MIN:
            off_path = math.degrees(math.acos(max(-1.0, min(1.0, v_in / max(1e-9, float(np.linalg.norm(v)))))))
            off_face = math.degrees(math.acos(max(-1.0, min(1.0, float(-(prev["normal"] @ axis))))))
            energy = prev["ke"] * math.cos(math.radians(off_path)) ** 2
            d0 = _EP["d"]
            n = prev["normal"]
            r_s = R_STRIKE  # the face's tilt in the striking frame: about U (pitch) and about AP (roll), deg
            blow = dict(step=_EP["step"], v=v_in, ke=prev["ke"], e=energy, path=off_path, face=off_face, d0=d0,
                        off=(ham["face"] - head)[:2] * 1000,
                        face_pitch=math.degrees(math.asin(max(-1.0, min(1.0, float(n @ AP))))),
                        face_roll=math.degrees(math.asin(max(-1.0, min(1.0, float(n @ U))))),
                        nail_lean=np.degrees(axis[:2]), vel=np.round(v, 2))
            if max(off_path, off_face) > BEND_DEG:
                _EP["failed"] = (f"the nail bent: a blow {max(off_path, off_face):.0f} deg off its axis "
                                 f"(path {off_path:.0f}, face {off_face:.0f})")
            elif d0 < FLUSH:
                # the wood's resistance, rising with depth, takes ETA of the blow's energy: integral of R over the step
                d1 = min(FLUSH, math.sqrt((d0 + D0) ** 2 + 2.0 * ETA * energy / P_N_PER_M) - D0)
                _limit("transX", -d1 - EPS, -d1 + EPS)
                _EP.update(mode="in", d=d1)
                if d1 >= D_SET and not _EP["set"]:  # it now stands: the wood holds its tilt where it is
                    _EP["set"] = True
                    _EP["set_tilt"] = tilt
                    _hold_tilt(_pose(base, "nail")[1])
                blow["d1"] = d1
            _EP["blows"].append(blow)
            _EP["last_blow"] = blow
            if _DEBUG:
                print(f"DBG BLOW {len(_EP['blows'])} step={_EP['step']} v={v_in:.2f}m/s ke={prev['ke']:.3f}J "
                      f"e={energy:.3f}J path={off_path:.1f} face={off_face:.1f} tilt={tilt:.1f} "
                      f"d {d0 * 1000:.2f}->{blow.get('d1', d0) * 1000:.2f}mm face-off-head(mm)={np.round(blow['off'], 1)} "
                      f"face_normal_along_AP={blow['face_pitch']:.1f} along_U={blow['face_roll']:.1f} "
                      f"nail_lean_xy(deg)={np.round(blow['nail_lean'], 1)} swing={np.round(_swing_deg(_pose(base, 'nail')[1]), 2)} "
                      f"held={np.round(_EP.get('held_swing', (0, 0)), 2)} vface={blow['vel']} "
                      f"mode={_EP['mode']} "
                      f"failed={_EP['failed'] or '-'}", flush=True)
    # the hammer must not strike a gripper other than the one holding it (for example the one holding the nail): its head
    # moving fast into that gripper's fingers or body
    if ham["speed"] > 0.2 or float(np.linalg.norm(ham["vface"])) > 0.3:
        pts = ham["pos"] + _HEAD_PTS @ ham["r"].T
        hulls = None
        for side in ("left", "right"):
            tcp = _EP.get("tcp", {}).get(side)
            if tcp is None or np.linalg.norm(tcp - ham["face"]) > 0.12 or side in _EP.get("ham_holders", ()):
                continue
            if hulls is None:
                hulls = _gripper_hulls(base)
            if _distance(pts, hulls[side]) < 0.0005 and not _EP["failed"]:
                _EP["failed"] = f"the hammer struck the {side} gripper at {np.linalg.norm(ham['vface']):.2f} m/s"
                if _DEBUG:
                    from yam_arena import analysis
                    for h, hlo, hhi in hulls[side]:
                        sd = analysis._signed(pts, h)
                        i = int(np.argmin(sd))
                        if sd[i] < 0.0005:
                            nv = float(np.min(np.linalg.norm(h.vertices - pts[i], axis=1)))
                            print(f"DBG HIT point {i} at {np.round((pts[i] - [MX, MY, SURF]) * 1000, 1)} mm, plane "
                                  f"distance {sd[i] * 1000:.2f} mm, nearest hull vertex {nv * 1000:.1f} mm, hull box "
                                  f"{np.round((hlo - [MX, MY, SURF]) * 1000, 1)}..{np.round((hhi - [MX, MY, SURF]) * 1000, 1)}",
                                  flush=True)
    _EP["prev"] = ham
    if _DEBUG:
        down = float(-ham["vface"][2])
        if down > _EP.get("vmax", 0.0):
            _EP["vmax"] = down
            _EP["vmax_ke"] = ham["ke"]


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("NAIL_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    base = getattr(env, "unwrapped", env)
    _EP["step"] += 1
    if not _EP["wrapped"]:  # run the nail model after every physics step from now on
        orig = base.scene.update

        def update(dt, _orig=orig, _base=base):
            _orig(dt)
            try:
                _substep(_base)
            except Exception:
                import traceback

                print("NAIL_TRACEBACK\n" + traceback.format_exc(), flush=True)
                raise

        base.scene.update = update
        _EP["wrapped"] = True
    tcps, tcp_rots = {}, {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        ids = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0][ids]
        p_tcp, r_tcp = ik.tcp_pose(q, station.BASES[side])
        tcps[side] = np.asarray(p_tcp, float)
        tcp_rots[side] = np.asarray(r_tcp, float)
        if side == "right":
            _EP["q_right"] = q
            if _DEBUG:  # the right arm's fastest joint speeds (rad/s), joints 1-3 and 4-6
                qd = np.abs(as_numpy(robot.data.joint_vel)[0][ids])
                for key, v in (("qd13", float(qd[:3].max())), ("qd46", float(qd[3:6].max()))):
                    _EP[key] = max(_EP.get(key, 0.0), v)
                    _EP[key + "_run"] = max(_EP.get(key + "_run", 0.0), v)
    _EP["tcp"] = tcps
    tip, axis, tilt, head = _nail_geom(base)
    ham = _hammer_state(base)
    hulls = _gripper_hulls(base)
    nail_pts = _nail_points(tip, axis)
    ham_pts = ham["pos"] + np.vstack([_HEAD_PTS, _HANDLE_PTS]) @ ham["r"].T
    nail_touch = min(_distance(nail_pts, hulls[s]) for s in hulls) < TOUCH
    ham_touch = min(_distance(ham_pts, hulls[s]) for s in hulls) < TOUCH
    handle_pts = ham["pos"] + _HANDLE_PTS @ ham["r"].T
    _EP["ham_holders"] = tuple(s for s in hulls if _distance(handle_pts, hulls[s]) < TOUCH)  # grippers on the handle
    lat = float(np.linalg.norm(tip[:2] - (MX, MY))) if _EP["mode"] == "in" else 1.0
    near = min(float(np.linalg.norm(t - head)) for t in tcps.values())
    _EP["hist"].append(ham["pos"].copy())
    del _EP["hist"][:-STILL_STEPS]
    ham_rest = (len(_EP["hist"]) == STILL_STEPS
                and max(float(np.abs(h - _EP["hist"][-1]).max()) for h in _EP["hist"]) * 1000 < STILL_MM)
    d = _EP["d"]
    started = _EP["mode"] == "in" and d >= STARTED and lat <= AT_MARK
    upright = tilt <= UPRIGHT_DEG
    stand = started and d >= STAND_D and _EP["set"] and upright and not nail_touch
    driven = started and d >= FLUSH - FLUSH_TOL and upright and not nail_touch
    _EP.update(tilt=tilt, lat=lat, tip=tip, head=head, nail_touch=nail_touch, ham_touch=ham_touch, ham_rest=ham_rest,
               near=near, started=started, stand=stand, driven=driven)
    if _DEBUG:
        _EP["face"] = ham["face"] - [MX, MY, SURF]
        _EP["grip_in_hammer"] = ham["r"].T @ (tcps["right"] - ham["pos"])
        rel = tcp_rots["right"].T @ ham["r"]  # the hammer's turn in the fingers ...
        planned = R_STRIKE.T @ R_HAM_USE  # ... against the planned one
        d = planned.T @ rel
        _EP["ham_turn"] = np.degrees([math.atan2(d[2, 1], d[2, 2]), math.asin(max(-1, min(1, -d[2, 0]))),
                                      math.atan2(d[1, 0], d[0, 0])])
        a = tcp_rots["right"][:, 2]
        _EP["grip_heading"] = math.degrees(math.atan2(a[1], a[0])) - math.degrees(H)
        _EP["grip_pitch"] = math.degrees(math.asin(max(-1.0, min(1.0, float(a[2])))))
        _EP["face_tilt"] = math.degrees(math.acos(max(-1.0, min(1.0, float(-ham["normal"][2])))))
    stage = 0.75 if driven else 0.5 if stand else 0.25 if started else 0.0
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)


def _tally(o) -> dict:
    done = bool(_EP["driven"] and not _EP["ham_touch"] and _EP["ham_rest"] and _EP["near"] >= CLEAR)
    return dict(done=done)


def _debug() -> None:
    _EP["dbg"] = _EP.get("dbg", 0) + 1
    if (_EP["dbg"] - 1) % _DEBUG_EVERY:
        return
    tip = _EP["tip"]
    tcp = _EP.get("tcp", {})
    b = _EP["last_blow"]
    bl = (f"last_blow(step {b['step']} v={b['v']:.2f} ke={b['ke']:.3f} e={b['e']:.3f} path={b['path']:.1f} "
          f"face={b['face']:.1f} d {b['d0'] * 1000:.1f}->{b.get('d1', b['d0']) * 1000:.1f}mm)") if b else "no blow"
    print(f"DBG step={_EP['step']} mode={_EP['mode']} d={_EP['d'] * 1000:.2f}mm set={_EP['set']} "
          f"tip=({(tip[0] - MX) * 1000:.1f},{(tip[1] - MY) * 1000:.1f},{(tip[2] - SURF) * 1000:.1f})mm "
          f"tilt={_EP['tilt']:.1f} lat={_EP['lat'] * 1000:.1f}mm nail_touch={_EP['nail_touch']} "
          f"ham_touch={_EP['ham_touch']} ham_rest={_EP['ham_rest']} near={_EP['near']:.3f} blows={len(_EP['blows'])} "
          f"best={_EP['best']} failed={_EP['failed'] or '-'} | {bl}", flush=True)
    if tcp:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in tcp.items())
              + f" | fastest face (down) since last: {_EP.get('vmax', 0.0):.2f} m/s, ke {_EP.get('vmax_ke', 0.0):.3f} J"
              + f" | face rel mark (mm) {np.round(_EP.get('face', np.zeros(3)) * 1000, 1)} tilt {_EP.get('face_tilt', 0):.1f}"
              + f" | right tcp in hammer frame (mm) {np.round(_EP.get('grip_in_hammer', np.zeros(3)) * 1000, 1)}"
              + f" | hammer turn in grip (deg, about its x y z) {np.round(_EP.get('ham_turn', np.zeros(3)), 1)}"
              + f" | grip heading err {_EP.get('grip_heading', 0):.1f} pitch {_EP.get('grip_pitch', 0):.1f}"
              + f" | q_right {np.round(_EP.get('q_right', np.zeros(6)), 2)}"
              + f" | right joint speed peak (rad/s) j1-3 {_EP.get('qd13', 0.0):.2f} j4-6 {_EP.get('qd46', 0.0):.2f}"
              + f" (run {_EP.get('qd13_run', 0.0):.2f} / {_EP.get('qd46_run', 0.0):.2f})",
              flush=True)
        _EP["vmax"] = 0.0
        _EP["qd13"] = _EP["qd46"] = 0.0


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once the nail is bent or the hammer has struck a gripper
    other than the one holding it):
    0.25 started at the mark; 0.5 standing on its own at the mark; 0.75 driven home. Full score (1.0) is success()."""
    if _DEBUG:
        _debug()
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """The nail driven home at the mark, upright, no gripper touching it, the hammer let go and at rest, both grippers
    5 cm or more from the nail, still for 1 s."""
    return bool(_tally(objects)["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As TM 9-243 says: the left gripper takes the nail from the table, turns it point-down and holds it upright with its
# point on the mark, pinching it low on its shank; the right takes the hammer by its handle close to its head,
# turns it face-down over the nail and taps the head with short downward strokes until the nail stands by
# itself; the left lets go and rises clear; the right drives the nail flush with full strokes from 6 cm up, then lays
# the hammer back on its side on the table and draws back. NAIL_ONE=1 runs it the wrong way, one hand at a time: the
# left stands the nail on its point on the mark and lets go before the hammer comes.

WRONG = os.environ.get("NAIL_ONE") == "1"
# NAIL_PUSH=1: instead of tapping, the hammer's face is lowered slowly onto the held nail's head and pressed down for
# 2 s (no blow), then lifted; the left gripper then lets go. A push never drives the nail, so it falls over: score 0.
PUSH = os.environ.get("NAIL_PUSH") == "1"
PUSH_S = 5.0
OPEN, CLOSED = -0.040, 0.0
NAIL_PINCH = 0.015  # the left gripper's centre this far up the nail from its point
HOLD_R = None  # set in _plan()


def _level(approach, closing) -> np.ndarray:
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.asarray(closing, float) - np.dot(closing, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


R_STRIKE = _level(AP, (0.0, 0.0, -1.0))  # level, pointing along H, fingers closing vertically (J6 near +90 deg)
R_HAMMER_PICK = _axis_rot(U, 90.0) @ R_STRIKE  # straight down, fingers closing along H
# Holding the hammer the wrist sags: the face meets the nail tipped about 6 deg about the grip's sideways axis and 2 deg
# about its pointing axis unless the strokes are sent turned the other way, so they are (measured in this simulation).
PITCH_UP, ROLL = 5.8, 2.0
R_STROKE = _axis_rot(AP, ROLL) @ _axis_rot(U, -PITCH_UP) @ R_STRIKE  # the grip the strokes are sent with
R_HOLD = _level((1.0, 0.0, 0.0), (0.0, -1.0, 0.0))  # level, pointing +x, fingers closing along y
R_NAIL_PICK = _axis_rot((0.0, 1.0, 0.0), 90.0) @ R_HOLD  # straight down, fingers closing along y


SAG = 0.021  # m: holding the hammer in this posture the arm settles this far below where it is sent (its joints'
# position gains against the arm's and the hammer's weight), so the demo sends every striking height this much higher


def _grip_z(head_top_z: float) -> float:
    """The height to send the grip to for the face to rest on a nail head at ``head_top_z`` (handle level)."""
    return head_top_z + FACE_D + SAG


def _good_ik(arm, pos, rot, seeds=None):
    from yam_arena import ik, station

    best = None
    for sd in seeds or itertools.product((-0.4, 0.0, 0.4), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2),
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


def _joint_path(arm, q0, q1, finger, label, seconds, steps=15) -> list:
    """Waypoints along a straight line in joint space from q0 to q1 (to change the arm's configuration)."""
    from yam_arena import ik, station

    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _lead_in(arm, pos, rot, finger, label, seconds=3.0, steps=15, seeds=None) -> list:
    from yam_arena.constants import READY

    return _joint_path(arm, np.array(READY, float), _good_ik(arm, pos, rot, seeds), finger, label, seconds, steps)


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)


def _stay(wps: list, seconds: float, label: str = "wait") -> list:
    last = wps[-1]
    return [Waypoint(seconds, last.pos, last.finger, label, last.yaw)]


_SEEDS_HOLD = [(j1, j2, j3, j4, j5, j6) for j1 in (-1.4, -1.0) for j2 in (1.4, 2.0) for j3 in (0.4, 0.8)
               for j4 in (0.8, 1.3) for j5 in (-1.4, -1.0) for j6 in (-0.3, 0.0, 0.3)]
_SEEDS_PICK = [(H, j2, j3, j4, 0.0, 1.57) for j2 in (0.6, 1.0, 1.4) for j3 in (0.6, 1.0, 1.4) for j4 in (-1.2, -0.6, 0.0)]

T_PLACED = 12.0  # by then the nail stands on the mark, held
HOLD_TRIM = (-2.5, -0.4)  # deg: the holding grip turned about the world's y and x axes (see _left; -3 is the most
# the left arm reaches there, its fourth joint then at its stop)
PARK_DY = 0.12  # the left gripper parks this far along the board (+y) from the mark, high
T_TAPS = T_PLACED - 0.5 + 1.6 + 1.2 + 1.0  # the first tap starts then
# Taps: (lift above the head, drive target below the head) in m: light taps while the nail is held, then full strokes
# from 6 cm up driven toward a point well below the head (the arm is still speeding up when the face meets the head).
TAPS = [(0.045, 0.035)] * 3
TAP_LIFT = 0.15  # s: a tap's lift-off
# Full strokes start with the grip at FULL_TOP_Z and are driven toward STRIKE_LOW_Z, the lowest the arm reaches with the
# face level (lower, the wrist's pitch joint is at its stop and the face tips over).
FULL_TOP_Z, STRIKE_LOW_Z = SURF + 0.170, SURF + 0.030
N_BLOWS = 16
TAP_S, BLOW_S = 0.95, 0.5
E_PLAN = 0.45  # J: the hammer's energy per full stroke, as measured in this simulation (for the demo's stroke heights)
D_TAPPED = 0.0069  # m: about how deep the three taps leave the nail
# Where the full strokes are sent. A full stroke lands further out along the grip's heading than a tap does, and at
# every full blow the hammer slips 1.4 to 2.6 mm along its handle in the two fingers (the fingers part for a few ms and
# the wrist gives), turning a little too, so that each blow lands about 2 mm further along the handle than the one
# before (measured in this simulation). The demo is open loop: it sends blow k 2 mm x k further back.
FULL_AIM_AP, FULL_AIM_U0, FULL_SLIP = -0.0131, 0.0012, 0.002
# Halfway, after REGRIP_AFTER full blows, the grip has slipped about REGRIP_SLIDE along the handle (measured in this
# simulation: about 2.6 mm a blow), so the hammer is laid on the table and taken again that much nearer its head.
REGRIP_AFTER, REGRIP_SLIDE = 6, 0.016


def _full_xy(k: int) -> np.ndarray:
    return STRIKE_XY + FULL_AIM_AP * AP[:2] + (FULL_AIM_U0 - FULL_SLIP * k) * U[:2]


def _left() -> list:
    from yam_arena import ik, station

    tip0 = np.array([NAIL_TIP0[0], NAIL_TIP0[1], TOP])
    pick = tip0 + [NAIL_PINCH, 0.0, 0.0133]
    out = _lead_in("left", pick + [0, 0, 0.06], R_NAIL_PICK, OPEN, "above the nail")
    out += [_w(1.0, pick, OPEN, "fingers either side of the nail", R_NAIL_PICK),
            _w(0.8, pick, CLOSED, "pinch the nail low on its shank", R_NAIL_PICK),
            _w(0.4, pick, CLOSED, "settle", R_NAIL_PICK),
            _w(1.2, pick + [0, 0, 0.17], CLOSED, "lift it clear of the board", R_NAIL_PICK)]
    # Where the nail sits in the fingers follows from how it lies on the table: its point on the table, its shank
    # rising toward its head (which props it up). Turn the grip so that the nail stands upright over the mark.
    rise = math.atan2(N_HEAD_R - 0.0003, N_LEN + N_HEAD_T / 2)  # about 4.3 deg
    tip_lying = tip0 + [0.0, 0.0, 0.0003]
    axis_lying = np.array([math.cos(rise), 0.0, math.sin(rise)])
    tip_in_grip = R_NAIL_PICK.T @ (tip_lying - pick)
    axis_in_grip = R_NAIL_PICK.T @ axis_lying
    r_hold = _axis_rot((0.0, 1.0, 0.0), math.degrees(rise)) @ R_HOLD
    assert np.allclose(r_hold @ axis_in_grip, [0.0, 0.0, 1.0], atol=1e-6)
    # Lifted and turned, the nail settles in the fingers leaning about 3 deg toward +x and 0.4 deg toward -y (measured
    # in this simulation), and the taps lean it about 1.5 deg further toward +x: the grip is turned back (HOLD_TRIM).
    r_hold = _axis_rot((1.0, 0.0, 0.0), HOLD_TRIM[1]) @ _axis_rot((0.0, 1.0, 0.0), HOLD_TRIM[0]) @ r_hold
    hold = np.array([MX, MY, SURF]) - r_hold @ tip_in_grip
    above = hold + [0.0, 0.0, 0.03]
    q_lift = _good_ik("left", pick + [0, 0, 0.17], R_NAIL_PICK)
    q_above = _good_ik("left", above, r_hold, _SEEDS_HOLD)
    out += _joint_path("left", q_lift, q_above, CLOSED, "turn it point-down over the mark", 2.6, 20)
    out += [_w(0.5, above, CLOSED, "settle", r_hold),
            _w(1.2, hold - [0, 0, 0.001], CLOSED, "point down onto the mark", r_hold),
            _w(0.5, hold - [0, 0, 0.001], CLOSED, "hold it upright", r_hold)]
    t = sum(w.duration for w in out)
    if t < T_PLACED:
        out += _stay(out, T_PLACED - t, "hold it upright")
    if not WRONG:  # the right way: hold it while it is tapped in; the wrong way lets go before the hammer comes
        out += _stay(out, _taps_seconds(), "hold it while it is tapped")
    # The hammer is over the nail by then: draw the open fingers back off the nail, below its face, before rising.
    back = hold + [-0.035, 0.0, 0.004]
    up = back + [0.0, 0.0, 0.07]
    out += [_w(0.4, hold - [0, 0, 0.001], OPEN, "let go", r_hold),
            _w(0.8, back, OPEN, "draw back off the nail, under the hammer", r_hold),
            _w(1.0, up, OPEN, "rise clear", r_hold)]
    out += [_w(1.2, up + [0.0, PARK_DY, 0.0], OPEN, "out of the hammer's way, along the board", r_hold)]
    return out + _stay(out, 60.0, "stay clear")


def _taps_seconds() -> float:
    return T_TAPS - T_PLACED + (PUSH_S if PUSH else len(TAPS) * TAP_S)


def _stroke(grip_xy, head_z, raise_m, low_z, seconds, label, follow=True) -> list:
    """One blow: lift the hammer ``raise_m`` over the nail's head, then drive the grip toward ``low_z`` (below the
    head, so the arm is still speeding up when the face meets the head), for one more step if ``follow``, lift off."""
    top = np.array([grip_xy[0], grip_xy[1], _grip_z(head_z) + raise_m])
    low = np.array([grip_xy[0], grip_xy[1], low_z])
    rest = np.array([grip_xy[0], grip_xy[1], _grip_z(head_z) + 0.02])
    hold = 0.20 if seconds > 0.6 else 0.05
    lift = 0.10 if follow else TAP_LIFT
    r = R_STROKE
    out = [_w(seconds - hold - 0.05 - (0.05 if follow else 0.0) - lift, top, CLOSED, f"raise ({label})", r),
           _w(hold, top, CLOSED, "hold", r), _w(0.05, low, CLOSED, f"strike ({label})", r)]
    if follow:
        out.append(_w(0.05, low, CLOSED, "follow through", r))
    return out + [_w(lift, rest, CLOSED, "lift off", r)]


def _depths() -> list:
    """The depth the model gives after each full stroke of E_PLAN, from D_TAPPED (for the strokes' start heights)."""
    d, out = D_TAPPED, []
    for _ in range(N_BLOWS):
        d = min(FLUSH, math.sqrt((d + D0) ** 2 + 2.0 * ETA * E_PLAN / P_N_PER_M) - D0)
        out.append(d)
    return out


def _right() -> list:
    head_top0 = SURF + N_LEN + N_HEAD_T  # the nail's head standing on the mark, not yet in
    pick = np.array([HAMMER_GRIP0[0], HAMMER_GRIP0[1], TOP + HANDLE_W / 2 + 0.0005 + 0.005])
    out = _lead_in("right", pick + [0, 0, 0.07], R_HAMMER_PICK, OPEN, "above the hammer's handle",
                   seeds=_SEEDS_PICK)
    out += [_w(1.0, pick, OPEN, "fingers either side of the handle", R_HAMMER_PICK),
            _w(0.8, pick, CLOSED, "take the handle close to the head", R_HAMMER_PICK),
            _w(0.4, pick, CLOSED, "settle", R_HAMMER_PICK),
            _w(1.0, pick + [0, 0, 0.10], CLOSED, "lift it", R_HAMMER_PICK)]
    high = np.array([STRIKE_XY[0], STRIKE_XY[1], _grip_z(head_top0) + 0.08])
    stage = high - 0.10 * AP  # the head high over the table in front of the board
    t = sum(w.duration for w in out)
    if t < T_PLACED - 0.5:  # keep the hammer low on this side while the left gripper brings the nail over
        out += _stay(out, T_PLACED - 0.5 - t, "wait for the nail to be placed")
    out += [_w(1.6, stage, CLOSED, "turn it face-down", R_STRIKE),
            _w(1.2, high, CLOSED, "over the nail, high", R_STROKE),
            _w(1.0, np.array([STRIKE_XY[0], STRIKE_XY[1], _grip_z(head_top0) + 0.025]), CLOSED,
               "face over the nail's head", R_STROKE)]
    assert abs(sum(w.duration for w in out) - T_TAPS) < 1e-6, sum(w.duration for w in out)
    if PUSH:  # press the face down on the head slowly (well under 0.3 m/s), hold the push, lift it again
        over = np.array([STRIKE_XY[0], STRIKE_XY[1], _grip_z(head_top0) + 0.025])
        press = np.array([STRIKE_XY[0], STRIKE_XY[1], _grip_z(head_top0) - 0.015])
        out += [_w(2.0, press, CLOSED, "lower the face onto the head slowly and press", R_STROKE),
                _w(2.0, press, CLOSED, "keep pressing", R_STROKE), _w(1.0, over, CLOSED, "lift it", R_STROKE)]
        return out + _stay(out, 6.0, "wait")
    for k, (raise_m, below) in enumerate(TAPS):  # light taps while the left gripper holds the nail
        head_z = head_top0 - 0.004
        out += _stroke(STRIKE_XY, head_z, raise_m, _grip_z(head_z) - below, TAP_S, f"tap {k + 1}", follow=False)
    if WRONG:
        return out + _stay(out, 3.0, "wait")
    out += _stay(out, 3.6, "wait for the left gripper to get clear")
    depth = D_TAPPED
    for k, depth_after in enumerate(_depths()):  # full strokes; each starts from over where the head now is
        if k == REGRIP_AFTER:  # put the hammer down and take it again, nearer the head
            pick2 = pick + REGRIP_SLIDE * U
            out += [_w(0.8, high, CLOSED, "lift the hammer off the nail", R_STRIKE),
                    _w(1.6, pick + [0, 0, 0.10], CLOSED, "turn it onto its side over the table", R_HAMMER_PICK),
                    _w(1.0, pick + [0, 0, 0.004], CLOSED, "lower it onto the table", R_HAMMER_PICK),
                    _w(0.5, pick + [0, 0, 0.004], OPEN, "let go", R_HAMMER_PICK),
                    _w(0.6, pick + [0, 0, 0.05], OPEN, "rise", R_HAMMER_PICK),
                    _w(0.6, pick2 + [0, 0, 0.05], OPEN, "over the handle, nearer the head", R_HAMMER_PICK),
                    _w(1.0, pick2, OPEN, "fingers either side of the handle", R_HAMMER_PICK),
                    _w(0.8, pick2, CLOSED, "take the handle close to the head again", R_HAMMER_PICK),
                    _w(0.4, pick2, CLOSED, "settle", R_HAMMER_PICK),
                    _w(1.0, pick2 + [0, 0, 0.10], CLOSED, "lift it", R_HAMMER_PICK),
                    _w(1.6, stage, CLOSED, "turn it face-down", R_STRIKE),
                    _w(1.2, high, CLOSED, "over the nail, high", R_STROKE)]
        kk = k if k < REGRIP_AFTER else k - REGRIP_AFTER  # the aim starts again after the re-grip
        head_z = head_top0 - depth
        out += _stroke(_full_xy(kk), head_z, FULL_TOP_Z - _grip_z(head_z), STRIKE_LOW_Z, BLOW_S, f"blow {k + 1}")
        depth = depth_after
    # lay it back down on its side where it was taken from (the reverse of picking it up)
    out += [_w(0.8, high, CLOSED, "lift the hammer off the nail", R_STRIKE),
            _w(1.6, pick + [0, 0, 0.10], CLOSED, "turn it back onto its side over the table", R_HAMMER_PICK),
            _w(1.0, pick + [0, 0, 0.004], CLOSED, "lower it onto the table", R_HAMMER_PICK),
            _w(0.5, pick + [0, 0, 0.004], OPEN, "let go", R_HAMMER_PICK),
            _w(0.8, pick + [0, 0, 0.08], OPEN, "rise clear", R_HAMMER_PICK),
            _w(1.0, np.array([0.16, -0.25, TOP + 0.16]), OPEN, "away", R_HAMMER_PICK)]
    return out + _stay(out, 4.0, "stay clear")


def demo():
    return {"left": _left(), "right": _right()}
