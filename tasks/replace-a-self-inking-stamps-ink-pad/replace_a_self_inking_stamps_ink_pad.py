"""On the station table: an office self-inking stamp of the Trodat Printy 4913 kind stands on the desk with its dry ink
pad in it; a new pad lies on the desk to its right, ink side down, and an empty parts tray stands in front of that.
Task (instruction): replace the stamp's dry ink pad with the new one, put the dry pad in the tray and leave the stamp
standing ready to use. The instruction does not say how.

How it is done (thestampmaker.com, "How to Re-Ink a Self-Inking Stamp", Trodat self-inking date stamps): "Push stamp
down slightly and push buttons on side to lock into position. Pull out ink pad." 2000 Plus: "Push down slightly on the
stamp - about 1/8" ... Lock the stamp in position using the locks on both sides of the stamp." stempel.eu.com ("How to
change your ink pad"): "Press down the stamp slightly (approx. 3mm)", "Lock stamping mechanism", "Pull or push out old
ink pad", "Insert new ink pad", "Release 'Lock-mode'" (by pressing the stamp down onto paper).

Why the lock is needed: at rest the stamp's spring holds the text plate (rubber die, face up) up in the pad's pocket
from below, so the pad cannot slide out (pulled, it catches on the die and the light stamp slides along the desk);
pressed down about 3 mm, the text plate drops clear of the pad, and the two lock buttons on the stamp's ends, pushed
in, catch the body there. Pressing the body down again (as when stamping) frees the buttons, which spring back out,
and the body returns up, the text plate rising into the new pad's pocket.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. The stamp is
drawn in its own frame (x from its front to its back, y along its length, z up) and stands turned so that its front
(where the pad's grip tab is) faces the robot's right (-y) and its length runs along x.
- stamp frame (chassis, loose, 40 g): 38 mm deep x 74 mm long x 68 mm tall, standing on two foot rails, its middle at
  (0.240, 0.070). The pad slot runs front to back between its end walls, 43-53.5 mm above the desk (grooves in the end
  walls hold the pad's ends), with a back stop; each end wall has a lock notch 52-59 mm up.
- stamp body (cap + two end skirts + text-plate mount, 45 g): on a slide joint along the frame's vertical, 0-6 mm down,
  on a spring (1500 N/m: 6 N at rest, 9.75 N at 2.5 mm, 15 N at 6 mm). Cap 40 x 76 x 13 mm, top 85 mm above the desk.
  Text plate 58 x 22 mm, top 44.4 mm up at rest (1.4 mm up in the pad's 1.5 mm pocket).
- two lock buttons (20 g each), in the skirts at the stamp's two ends, 60 mm up: on slide joints, 2.5 mm inward travel,
  light outward springs (0.4 N out, 0.8 N in). Their inner tips stop on the frame's end walls unless the body is down
  2.5 mm or more, when they line up with the lock notches; pushed in, the notch's top edge holds the body 2.5 mm down
  (the spring pressing the tip up against it holds the button in by friction). Pressed down past 3.5 mm, the tips come
  free and the buttons spring out.
- ink pads (old in the slot, new on the desk at (0.270, -0.100), 15 g each): 68 x 31 x 10 mm (Trodat 6/4913), a plastic
  case on top, foam below with a 60 x 24 x 1.5 mm pocket where the text plate sits, and a grip tab (2.5 mm thick, 15 mm
  long, 18 mm tall) on the front edge. A spring detent ball (2 mm radius, 0.8 N at rest) in the slot's ceiling clicks
  into a groove across the pad's top when the pad is fully home. The slot's first 5 mm flare 2 mm up, down and to
  each side (lead-in faces).
- parts tray (fixed, open, 90 x 120 mm, 12 mm rims), its middle at (0.140, -0.110).

Score (best stage reached; full score is success()):
- 0.1: the stamp locked: body 2.0 mm or more down with both lock buttons pushed in 1.5 mm or more.
- 0.3: the dry pad out of the slot (its rear edge more than 1 mm in front of the frame's front face).
- 0.45: the dry pad in the tray (all of it inside the rims, its lowest point within 15 mm of the floor), released.
- 0.6: the new pad fully home in the slot (rear edge within 1.0 mm of the back stop: the detent in its groove), ink
  side down (case up, within 15 deg), the dry pad in the tray.
- 0.8: the stamp unlocked with the new pad home: both buttons out (less than 0.5 mm in) and the body back up (less
  than 0.5 mm down), so the text plate stands in the new pad's pocket.
- 1.0 (success, held 1 s): all of that at once, the stamp standing upright on its foot rails (within 10 deg of
  vertical, on the desk), every finger 10 mm or more from the stamp and both pads, and everything still (moving less
  than 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, Joint, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# -- the layout ----------------------------------------------------------------------------------------------------
SX, SY = float(os.environ.get("STAMP_SX", "0.240")), float(os.environ.get("STAMP_SY", "0.070"))
S0 = np.array([SX, SY, TOP + 0.0005])  # the stamp's origin: the middle of its footprint, on the desk
HD, HW = 19 * MM, 37 * MM  # half depth (stamp x), half length (stamp y) of the frame
SLOT_Z0, SLOT_Z1 = 43 * MM, 53.5 * MM  # the pad's grooves (pad 10 mm thick, 0.5 mm play)
WALL_IN, GROOVE = 31 * MM, 34.5 * MM  # end walls' inner face, the grooves' depth into the walls (|y|)
NOTCH_Z0, NOTCH_Z1 = 52 * MM, 59 * MM  # the lock notches in the end walls' outer part (|y| 34.5-37 mm)
FRAME_TOP = 68 * MM
MOUTH_X = -14 * MM  # the slot's flared mouth runs from the front face (x -19 mm) to here
BACK_STOP = 12 * MM  # the slot's back stop (x): a seated pad spans x -19 .. 12 mm
PAD_L, PAD_W, PAD_T = 31 * MM, 68 * MM, 10 * MM  # along x (slides), along y, thick
PAD_SEAT = np.array([BACK_STOP - PAD_L / 2, 0.0, SLOT_Z0 + PAD_T / 2])  # the pad's middle when seated (stamp frame)
PLATE_X0, PLATE_X1, PLATE_HW = -13 * MM, 9 * MM, 29 * MM  # the text plate (58 x 22 mm), x range and half length
PLATE_TOP = 44.4 * MM  # its top at rest: 1.4 mm up into the pad's 1.5 mm pocket
POCKET = (-14 * MM, 10 * MM, 30 * MM, 1.5 * MM)  # the pad's pocket in stamp x (seated): x0, x1, half length, depth
TAB_L, TAB_T, TAB_H = 15 * MM, 2.5 * MM, 18 * MM  # grip tab: protrudes, thick, tall (from the pad's underside)
BTN_Z = 60 * MM  # lock buttons' middle height
SKIRT_IN, SKIRT_OUT = 37.3 * MM, 40 * MM
TIP_IN = 37.2 * MM  # the button tip's inner face at rest (|y|); the frame's end wall is at 37 mm
BTN_TRAVEL = 2.5 * MM  # the tip stops 0.2 mm short of the notch's floor
CAP_Z0, CAP_Z1 = 72 * MM, 85 * MM
BODY_K = float(os.environ.get("STAMP_K", "1500"))  # N/m
BODY_PRE = 4 * MM  # spring target 4 mm above the rest stop: 6 N preload
BODY_TRAVEL = 6 * MM
BTN_K = 150.0  # N/m: the buttons' outward springs
DET_R, DET_PRE = 2 * MM, 1.0 * MM  # detent ball radius, how far it pokes below the slot ceiling
DET_K = 800.0  # N/m (0.8 N at rest)
DET_GROOVE = 3.2 * MM  # the groove across the pad's top for the ball (x width)
DET_X = 6 * MM  # the ball's x (stamp frame)

NEW_PAD_C = np.array([float(v) for v in os.environ.get("STAMP_NEWPAD", "0.270,-0.100").split(",")])
TRAY_C = np.array([float(v) for v in os.environ.get("STAMP_TRAY", "0.140,-0.110").split(",")])
TRAY_L, TRAY_W, TRAY_H, TRAY_FLOOR = 0.090, 0.120, 0.012, 0.003  # x, y, rim height, floor thickness

BLUE, DARKBLUE, GREY, BLACK, RUBBER, FOAM, CASE, TRAYC, WHITE = (
    (0.15, 0.35, 0.75), (0.10, 0.20, 0.45), (0.55, 0.57, 0.60), (0.08, 0.08, 0.09), (0.55, 0.12, 0.10),
    (0.12, 0.12, 0.14), (0.30, 0.30, 0.32), (0.85, 0.85, 0.82), (0.92, 0.92, 0.90))


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


# The stamp and the pads are drawn in the stamp's own frame and stand turned 90 deg on the desk.
M = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # stamp frame -> world (a turn of +90 deg)


def _box(x0, x1, y0, y1, z0, z1, color, collide=True) -> BoxShape:
    """A box given by its extents in the stamp's frame, placed in the (world-aligned) body frame."""
    return BoxShape((-(y0 + y1) / 2, (x0 + x1) / 2, (z0 + z1) / 2), (y1 - y0, x1 - x0, z1 - z0), color,
                    collide=collide)


def _wbox(x0, x1, y0, y1, z0, z1, color, collide=True) -> BoxShape:
    """A box given in the world-aligned frame."""
    return BoxShape(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


def _ramp(p1, p2, width, normal) -> BoxShape:
    """A 1 mm plate (a lead-in face) whose middle line runs from ``p1`` to ``p2`` (stamp frame)."""
    return plate(M @ np.asarray(p1, float), M @ np.asarray(p2, float), width, 1 * MM, M @ np.asarray(normal, float),
                 GREY)


def _frame() -> Body:
    mx = MOUTH_X
    sh = [_box(-HD, -HD + 4 * MM, -WALL_IN, WALL_IN, 0.0, 6 * MM, GREY),  # foot rails, front and back
          _box(HD - 4 * MM, HD, -WALL_IN, WALL_IN, 0.0, 6 * MM, GREY),
          _box(mx, HD, -WALL_IN, WALL_IN, SLOT_Z1, SLOT_Z1 + 3 * MM, GREY),  # the slot's ceiling
          _box(BACK_STOP, HD, -WALL_IN, WALL_IN, SLOT_Z0, SLOT_Z1, GREY),  # back stop
          _box(-HD, -HD + 2 * MM, -WALL_IN, WALL_IN, SLOT_Z1 + 3 * MM, FRAME_TOP, GREY),  # front plate over the slot
          # the slot's mouth (its first 5 mm) flares 2 mm up, down and to each side: lead-in faces
          _ramp((-HD, 0.0, SLOT_Z1 + 2 * MM + 0.5 * MM), (mx, 0.0, SLOT_Z1 + 0.5 * MM), 2 * HW, (0.0, 0.0, 1.0))]
    for s in (1.0, -1.0):
        def ys(a, b):
            return (min(s * a, s * b), max(s * a, s * b))
        # inner part of the end wall, with the groove for the pad's end
        sh += [_box(mx, HD, *ys(WALL_IN, GROOVE), 0.0, SLOT_Z0, GREY),
               _box(mx, HD, *ys(WALL_IN, GROOVE), SLOT_Z1, FRAME_TOP, GREY),
               # outer part, with the lock notch
               _box(mx, HD, *ys(GROOVE, HW), 0.0, NOTCH_Z0, GREY),
               _box(mx, HD, *ys(GROOVE, HW), NOTCH_Z1, FRAME_TOP, GREY),
               _box(mx, -6 * MM, *ys(GROOVE, HW), NOTCH_Z0, NOTCH_Z1, GREY),
               _box(6 * MM, HD, *ys(GROOVE, HW), NOTCH_Z0, NOTCH_Z1, GREY),
               # the end wall at the mouth, cut back round the flared opening
               _box(-HD, mx, *ys(WALL_IN, HW), 0.0, SLOT_Z0 - 2 * MM, GREY),
               _box(-HD, mx, *ys(WALL_IN, HW), SLOT_Z1 + 2 * MM, FRAME_TOP, GREY),
               _ramp((-HD, s * 33.9 * MM, SLOT_Z0 - 2 * MM - 0.5 * MM), (mx, s * 33.9 * MM, SLOT_Z0 - 0.5 * MM),
                     6 * MM, (0.0, 0.0, 1.0)),
               _ramp((-HD, s * (HW + 0.3 * MM), (SLOT_Z0 + SLOT_Z1) / 2), (mx, s * (GROOVE + 0.5 * MM),
                     (SLOT_Z0 + SLOT_Z1) / 2), SLOT_Z1 - SLOT_Z0 + 4 * MM, (0.0, s, 0.0))]
    return Body.build("stamp_frame", pos=_v(S0), mass=0.040, color=GREY, friction=(0.6, 0.5), shapes=sh,
                      solver_iterations=32, contact_offset=0.001)


def _body() -> Body:
    sh = [_box(-20 * MM, 20 * MM, -38 * MM, 38 * MM, CAP_Z0, CAP_Z1, BLUE)]
    for s in (1.0, -1.0):
        y0, y1 = sorted((s * SKIRT_IN, s * SKIRT_OUT))
        sh += [_box(-17 * MM, 17 * MM, y0, y1, 22 * MM, CAP_Z0, BLUE)]
    # the text plate on its mount, face up under the pad (it turns face down only on a full stroke, not modelled)
    sh += [_box(PLATE_X0, PLATE_X1, -PLATE_HW, PLATE_HW, PLATE_TOP - 6 * MM, PLATE_TOP - 1.5 * MM, DARKBLUE),
           _box(PLATE_X0, PLATE_X1, -PLATE_HW, PLATE_HW, PLATE_TOP - 1.5 * MM, PLATE_TOP, RUBBER)]
    return Body.build("stamp_body", pos=_v(S0), mass=0.045, color=BLUE, friction=(0.7, 0.6), shapes=sh,
                      solver_iterations=32, contact_offset=0.001,
                      joint=Joint(anchor=_v(S0), kind="slide", axis="Z", parent="stamp_frame",
                                  axial=(BODY_K, 8.0, BODY_TRAVEL), slide_range=(-BODY_TRAVEL, 0.0),
                                  slide_target=BODY_PRE))


def _button(side: str) -> Body:
    s = 1.0 if side == "near" else -1.0  # the stamp's +y end is the world's -x (nearer the robot)
    y = lambda a, b: tuple(sorted((s * a, s * b)))  # noqa: E731
    sh = [_box(-5 * MM, 5 * MM, *y(SKIRT_OUT, SKIRT_OUT + 2.5 * MM), BTN_Z - 4 * MM, BTN_Z + 4 * MM, BLACK),
          _box(-3 * MM, 3 * MM, *y(TIP_IN + 1.8 * MM, SKIRT_OUT), BTN_Z - 1.5 * MM, BTN_Z + 1.5 * MM, BLACK),
          _box(-4 * MM, 4 * MM, *y(TIP_IN, TIP_IN + 1.8 * MM), BTN_Z - 1.5 * MM, BTN_Z + 1.5 * MM, BLACK)]
    rng = (0.0, BTN_TRAVEL) if s > 0 else (-BTN_TRAVEL, 0.0)  # stamp +y is world -x: in is +x for the near one
    return Body.build(f"lock_{side}", pos=_v(S0), mass=0.02, color=BLACK, friction=(0.6, 0.5), shapes=sh,
                      solver_iterations=32, contact_offset=0.0008,
                      joint=Joint(anchor=_v(S0), kind="slide", axis="X", parent="stamp_body",
                                  axial=(BTN_K, 0.5, BTN_TRAVEL), slide_range=rng, slide_target=-s * 2.7 * MM))


def _detent() -> Body:
    c = M @ np.array([DET_X, 0.0, SLOT_Z1 + DET_R - DET_PRE])
    return Body.build("detent", pos=_v(S0 + c), mass=0.002, color=GREY, friction=(0.2, 0.15),
                      shapes=[SphereShape((0.0, 0.0, 0.0), DET_R, GREY)], solver_iterations=32, contact_offset=0.0008,
                      joint=Joint(anchor=_v(S0 + c), kind="slide", axis="Z", parent="stamp_frame",
                                  axial=(DET_K, 0.3, 0.0015), slide_range=(0.0, 0.0015), slide_target=-DET_PRE))


def _pad_shapes(dry: bool) -> list:
    """A pad in its own frame: middle of its 68 x 31 x 10 mm block, x along its sliding direction (tab at -x)."""
    hx, hy, hz = PAD_L / 2, PAD_W / 2, PAD_T / 2
    px0, px1, phy, pd = POCKET
    px0, px1 = px0 - PAD_SEAT[0], px1 - PAD_SEAT[0]  # the pocket in the pad's frame
    gx = DET_X - PAD_SEAT[0]  # the detent groove across its top
    foam = FOAM if not dry else (0.35, 0.33, 0.33)
    z_case = -hz + 4 * MM  # the case from here up, foam below
    sh = [_box(-hx, gx - DET_GROOVE / 2, -hy, hy, z_case, hz, CASE),
          _box(gx + DET_GROOVE / 2, hx, -hy, hy, z_case, hz, CASE),
          _box(gx - DET_GROOVE / 2, gx + DET_GROOVE / 2, -hy, hy, z_case, hz - 1.0 * MM, CASE),
          # the foam (ink side, below) around the pocket for the text plate
          _box(-hx, px0, -hy, hy, -hz, z_case, foam), _box(px1, hx, -hy, hy, -hz, z_case, foam),
          _box(px0, px1, -hy, -phy, -hz, z_case, foam), _box(px0, px1, phy, hy, -hz, z_case, foam),
          _box(px0, px1, -phy, phy, -hz + pd, z_case, foam),
          # the grip tab on the front edge
          _box(-hx - TAB_L, -hx, -TAB_T / 2, TAB_T / 2, -hz, -hz + TAB_H, CASE),
          _box(-hx - TAB_L + 2 * MM, -hx - 2 * MM, -TAB_T / 2 - 0.2 * MM, TAB_T / 2 + 0.2 * MM, -hz + 3 * MM,
               -hz + TAB_H - 3 * MM, WHITE if dry else BLUE, collide=False)]
    return sh


OLD_PAD = Body.build("old_pad", pos=_v(S0 + M @ PAD_SEAT + np.array([0.0, 0.0, 0.0002])), mass=0.015, color=CASE,
                     friction=(0.4, 0.35), shapes=_pad_shapes(True), solver_iterations=32, contact_offset=0.0008)
NEW_PAD = Body.build("new_pad", pos=(float(NEW_PAD_C[0]), float(NEW_PAD_C[1]), TOP + PAD_T / 2 + 0.0008),
                     mass=0.015, color=CASE, friction=(0.4, 0.35), shapes=_pad_shapes(False), solver_iterations=32,
                     contact_offset=0.0008)


def _tray() -> Body:
    hx, hy, t = TRAY_L / 2, TRAY_W / 2, 0.003
    sh = [_wbox(-hx, hx, -hy, hy, 0.0, TRAY_FLOOR, TRAYC),
          _wbox(-hx, -hx + t, -hy, hy, 0.0, TRAY_H, TRAYC), _wbox(hx - t, hx, -hy, hy, 0.0, TRAY_H, TRAYC),
          _wbox(-hx, hx, -hy, -hy + t, 0.0, TRAY_H, TRAYC), _wbox(-hx, hx, hy - t, hy, 0.0, TRAY_H, TRAYC)]
    return Body.build("tray", pos=(float(TRAY_C[0]), float(TRAY_C[1]), TOP + 0.0002), mass=0.1, color=TRAYC,
                      fixed=True, friction=(0.5, 0.4), shapes=sh)


FRAME, BODY, LOCK_N, LOCK_F, DETENT, TRAY = _frame(), _body(), _button("near"), _button("far"), _detent(), _tray()
OBJECTS = [FRAME, BODY, LOCK_N, LOCK_F, DETENT, OLD_PAD, NEW_PAD, TRAY]

# -- tracking and scoring -------------------------------------------------------------------------------------------
LOCK_D, LOCK_B = 2.0 * MM, 1.5 * MM  # locked: body this far down and both buttons this far in
UP_D, OUT_B = 0.5 * MM, 0.5 * MM  # unlocked: body back up within this, buttons out within this
HOME = 1.0 * MM  # the pad's rear edge within this of the back stop
FLAT_DEG, UPRIGHT_DEG = 15.0, 10.0
CLEAR = 0.010
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("STAMP_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("STAMP_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, best=0.0, recent=[], gaps={}, tcp={}, st={})


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


def _corners(lo, hi) -> np.ndarray:
    return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("STAMP_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _pose(base, name, origin):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - origin, rotation(p[3:7])


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "fingers" not in _IDS:
        _IDS["fingers"] = _finger_points()
    fingers = []
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
        fingers += [_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                    for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])]
    tree = cKDTree(np.concatenate(fingers))
    fp, fr = _pose(base, "stamp_frame", origin)
    bp, _ = _pose(base, "stamp_body", origin)
    np_, _ = _pose(base, "lock_near", origin)
    fp_, _ = _pose(base, "lock_far", origin)
    to_f = lambda p: M.T @ fr.T @ (np.asarray(p) - fp)  # noqa: E731  (in the stamp's own frame)
    d = float(-to_f(bp)[2])  # how far the body is down
    b_n = float(-(to_f(np_)[1] - to_f(bp)[1]))  # how far each button is in
    b_f = float(to_f(fp_)[1] - to_f(bp)[1])
    st = dict(d=d, bn=b_n, bf=b_f, tilt=math.degrees(math.acos(max(-1.0, min(1.0, fr[2, 2])))),
              fz=float(fp[2] - TOP), fxy=fp[:2].copy())
    for name, spec in (("old", OLD_PAD), ("new", NEW_PAD)):
        pp, pr = _pose(base, f"{name}_pad", origin)
        loc = to_f(pp) - PAD_SEAT  # its middle from the seated place, in the stamp frame
        lo, hi = spec.local_bounds()
        cw = pp + _corners(lo, hi) @ pr.T
        rear = float(to_f(pp)[0] + PAD_L / 2)  # its rear edge's x in the stamp frame (along the slot)
        flat = math.degrees(math.acos(max(-1.0, min(1.0, float((fr.T @ pr)[2, 2])))))  # case up, in the stamp
        in_slot = abs(loc[1]) < 3 * MM and abs(loc[2]) < 3 * MM and rear > -HD
        out = rear < -HD - 1 * MM or not (abs(loc[1]) < 10 * MM and abs(loc[2]) < 10 * MM)
        home = in_slot and abs(rear - BACK_STOP) <= HOME and flat <= FLAT_DEG
        tl = cw - np.array([TRAY_C[0], TRAY_C[1], TOP])
        in_tray = bool(np.all(np.abs(tl[:, 0]) <= TRAY_L / 2) and np.all(np.abs(tl[:, 1]) <= TRAY_W / 2)
                       and tl[:, 2].min() <= TRAY_FLOOR + 0.015)
        near = float(tree.query(cw, k=1)[0].min())
        # also points along the pad's faces (corners alone miss a finger pinching its tab)
        g = np.array([[x, y, z] for x in np.linspace(lo[0], hi[0], 5) for y in np.linspace(lo[1], hi[1], 5)
                      for z in (lo[2], hi[2])])
        near = min(near, float(tree.query(pp + g @ pr.T, k=1)[0].min()))
        st[name] = dict(loc=loc, rear=rear, flat=flat, in_slot=in_slot, out=out, home=home, in_tray=in_tray,
                        near=near, p=pp.copy())
    # the stamp's own clearance: points over its cap, ends and sides
    g = np.array([[x, y, z] for x in np.linspace(-HD, HD, 4) for y in np.linspace(-SKIRT_OUT - 2.5 * MM,
                  SKIRT_OUT + 2.5 * MM, 7) for z in (0.0, 0.03, 0.06, CAP_Z1)])
    st["near_stamp"] = float(tree.query(fp + g @ (fr @ M).T, k=1)[0].min())
    _EP["st"] = st
    locked = d >= LOCK_D and b_n >= LOCK_B and b_f >= LOCK_B
    unlocked = d <= UP_D and b_n <= OUT_B and b_f <= OUT_B
    o, n = st["old"], st["new"]
    stage = 0.0
    if locked:
        stage = 0.1
    if _EP["best"] >= 0.1 and o["out"]:
        stage = max(stage, 0.3)
    if _EP["best"] >= 0.3 and o["in_tray"] and o["near"] >= CLEAR:
        stage = max(stage, 0.45)
    if _EP["best"] >= 0.45 and n["home"] and o["in_tray"]:
        stage = max(stage, 0.6)
    if _EP["best"] >= 0.6 and n["home"] and unlocked and o["in_tray"]:
        stage = max(stage, 0.8)
    _EP["best"] = max(_EP["best"], stage)
    st["stage"] = stage


def _tally(o) -> dict:
    st = _EP.get("st", {})
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and st:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([st["fxy"], [st["fz"]], st["new"]["p"], st["old"]["p"]]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    if not st:
        return dict(still=False, done=False)
    upright = st["tilt"] <= UPRIGHT_DEG and st["fz"] < 0.004
    clear = st["near_stamp"] >= CLEAR and st["old"]["near"] >= CLEAR and st["new"]["near"] >= CLEAR
    done = bool(_EP["best"] >= 0.8 and st.get("stage", 0.0) >= 0.8 and upright and clear and still)
    return dict(still=still, upright=upright, clear=clear, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or not _EP.get("st"):
        return
    st = _EP["st"]
    o, n = st["old"], st["new"]
    pad = lambda s: (f"loc=({s['loc'][0] * 1000:+.1f},{s['loc'][1] * 1000:+.1f},{s['loc'][2] * 1000:+.1f})mm "  # noqa
                     f"rear={s['rear'] * 1000:+.1f} flat={s['flat']:.0f} slot={int(s['in_slot'])} out={int(s['out'])} "
                     f"home={int(s['home'])} tray={int(s['in_tray'])} near={s['near'] * 1000:.0f}")
    print(f"DBG step={_EP['step']} d={st['d'] * 1000:.2f}mm bn={st['bn'] * 1000:.2f} bf={st['bf'] * 1000:.2f} "
          f"tilt={st['tilt']:.1f} fz={st['fz'] * 1000:.1f} fxy=({st['fxy'][0]:.3f},{st['fxy'][1]:.3f}) "
          f"stage={st.get('stage', 0)} best={_EP['best']} near_stamp={st['near_stamp'] * 1000:.0f} "
          f"still={int(t.get('still', 0))} | OLD {pad(o)} | NEW {pad(n)} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """The best stage reached (0.1, 0.3, 0.45, 0.6, 0.8); full score is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if t["done"]:
        return 1.0
    return float(_EP["best"])


def success(objects) -> bool:
    return bool(_tally(objects)["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper, straight down with its fingers closed, presses the stamp's cap
# down near its back edge until the body stops (6 mm, past the 2.5 mm the locks need; the arm stops short under the
# spring's load, so it is told to go 30 mm down). The right gripper, tipped 25 deg down and reaching in from the
# stamp's front, fingers closing across the stamp's ends, squeezes the two lock buttons in. The left hand lets the cap
# up and the right hand opens: the body rises 3.5 mm onto the buttons' tips and stays locked 2.5 mm down. The left
# hand then holds the stamp by its two ends (steadying it). The right gripper, now tipped 35 deg, pinches the dry
# pad's grip tab, draws the pad straight out, lays it in the tray, takes the new pad by its tab on the desk, carries
# it level to the slot and slides it home (told 20 mm past the stop: the arm is soft sideways and the detent resists)
# and backs away. The
# left hand lets go, closes, and presses the cap down past the lock (as when stamping): the buttons spring out, and
# as the hand rises the body returns up. STAMP_MODE=nolock: nobody presses the stamp or its buttons; the left hand
# only steadies the stamp by its ends while the right one pulls the dry pad by its tab.
OPEN_C = -0.047
MODE = os.environ.get("STAMP_MODE", "demo")
SQUEEZE = float(os.environ.get("STAMP_SQUEEZE", "-0.038"))  # right fingers on the buttons (76 mm commanded)
STEADY = float(os.environ.get("STAMP_STEADY", "-0.039"))  # left fingers on the stamp's ends (78 mm)
PRESS = float(os.environ.get("STAMP_PRESS", "0.030"))  # the left hand told this far below the cap's top
TAB_PINCH = float(os.environ.get("STAMP_TABPINCH", "0.0"))
TILT = math.radians(float(os.environ.get("STAMP_TILT", "35")))
SQZ_TILT = math.radians(float(os.environ.get("STAMP_SQZTILT", "25")))
SQZ_X = float(os.environ.get("STAMP_SQZX", "-10")) * MM  # the right TCP this far in front of the buttons (stamp x)
TAB_BACK = float(os.environ.get("STAMP_TABBACK", "14")) * MM  # the TCP this far in front of the tab's root (stamp x)
TAB_DZ = float(os.environ.get("STAMP_TABDZ", "15")) * MM  # the TCP this far above the tab's bottom
PUSH_OVER = float(os.environ.get("STAMP_PUSHOVER", "20")) * MM
UNLOCK = float(os.environ.get("STAMP_UNLOCK", "0.025"))  # the cap pressed this far below its locked height
INS_DX = float(os.environ.get("STAMP_INSDX", "2.8")) * MM  # the new pad's aim, corrected (world x)
INS_DZ = float(os.environ.get("STAMP_INSDZ", "3.0")) * MM  # and raised
PRESS_X = 0.016  # the left fingertips on the cap this far behind its middle (stamp x)


def _rot(z, x) -> np.ndarray:
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - np.dot(x, z) * z / np.dot(z, z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def W(p) -> np.ndarray:
    """A point given in the stamp's frame (m), in the world."""
    return S0 + M @ np.asarray(p, float)


GRIP_L = grip_rotation(-math.pi / 2)  # left: straight down, fingers closing along x
APPROACH = M @ np.array([math.cos(TILT), 0.0, -math.sin(TILT)])  # right: into the slot (+y), tipped down 35 deg
TAB_GRIP = _rot(APPROACH, (1.0, 0.0, 0.0))  # fingers closing along x, across the grip tab
SQZ_APPROACH = M @ np.array([math.cos(SQZ_TILT), 0.0, -math.sin(SQZ_TILT)])
SQZ_GRIP = _rot(SQZ_APPROACH, (1.0, 0.0, 0.0))  # 25 deg down, fingers closing across the stamp's ends


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


T_LOCKED = 12.0  # s: the stamp is locked and the right hand has backed away
T_RIGHT_DONE = 0.0  # filled in by _right()


def _left() -> list:
    cap = W([PRESS_X, 0.0, CAP_Z1 + 0.0128])  # closed fingertips on the cap's back part
    ends = W([0.0, 0.0, BTN_Z])  # the TCP between the stamp's ends
    out = _lead_in("left", cap + np.array([0.0, 0.0, 0.05]), GRIP_L, 0.0, "above the cap, fingers closed")
    if MODE == "nolock":
        out = _lead_in("left", ends + np.array([0.0, 0.0, 0.06]), GRIP_L, OPEN_C, "above the stamp, open")
        out += [_w(2.0, ends, OPEN_C, "fingers either side of the stamp's ends", GRIP_L),
                _w(1.0, ends, STEADY, "hold the stamp by its ends", GRIP_L),
                _w(60.0, ends, STEADY, "steady the stamp", GRIP_L)]
        return out
    out += [_w(1.5, cap + np.array([0.0, 0.0, 0.004]), 0.0, "down onto the cap", GRIP_L),
            _w(1.5, cap - np.array([0.0, 0.0, PRESS]), 0.0, "press the body down", GRIP_L),
            _w(2.5, cap - np.array([0.0, 0.0, PRESS]), 0.0, "hold it down while the buttons go in", GRIP_L),
            _w(1.0, cap + np.array([0.0, 0.0, 0.03]), 0.0, "let the cap up", GRIP_L),
            _w(0.8, cap + np.array([0.0, 0.0, 0.03]), OPEN_C, "open", GRIP_L),
            _w(T_LOCKED - 10.3, cap + np.array([0.0, 0.0, 0.03]), OPEN_C, "wait for the right hand to back away",
               GRIP_L),
            _w(1.5, ends + np.array([0.0, 0.0, 0.035]), OPEN_C, "over the stamp's ends", GRIP_L),
            _w(2.0, ends, OPEN_C, "fingers either side of the stamp's ends", GRIP_L),
            _w(1.0, ends, STEADY, "hold the stamp by its ends", GRIP_L),
            _w(T_RIGHT_DONE - T_LOCKED - 4.5, ends, STEADY, "steady the stamp while the pads change", GRIP_L)]
    top = W([0.0, 0.0, CAP_Z1 + 0.0128])
    out += [_w(0.8, ends, OPEN_C, "let go of the ends", GRIP_L),
            _w(1.5, top + np.array([0.0, 0.0, 0.04]), OPEN_C, "rise over the cap", GRIP_L),
            _w(0.8, top + np.array([0.0, 0.0, 0.04]), 0.0, "close the fingers", GRIP_L),
            _w(1.5, top + np.array([0.0, 0.0, 0.004]), 0.0, "down onto the cap", GRIP_L),
            _w(1.2, top - np.array([0.0, 0.0, 0.0025 + UNLOCK]), 0.0, "press the body down: the locks free",
               GRIP_L),
            _w(1.0, top - np.array([0.0, 0.0, 0.0025 + UNLOCK]), 0.0, "hold", GRIP_L),
            _w(1.5, top + np.array([0.0, 0.0, 0.01]), 0.0, "let the body come up", GRIP_L),
            _w(2.0, top + np.array([0.0, 0.0, 0.07]), 0.0, "rise clear", GRIP_L),
            _w(10.0, top + np.array([0.0, 0.0, 0.07]), 0.0, "stay clear", GRIP_L)]
    return out


def _right() -> list:
    global T_RIGHT_DONE
    out_dir = M @ np.array([-1.0, 0.0, 0.0])  # out of the slot, toward the stamp's front (world -y)
    sq = W([SQZ_X, 0.0, BTN_Z])  # the TCP between the two lock buttons
    sq_back = sq - SQZ_APPROACH * 0.04 + np.array([0.0, 0.0, 0.02])
    out = _lead_in("right", sq_back, SQZ_GRIP, OPEN_C, "in front of the stamp, open, its ends between the fingers",
                   seconds=4.0)
    face = PAD_SEAT[0] - PAD_L / 2  # the seated pad's front face (stamp x)
    pinch = W([face - TAB_BACK, 0.0, SLOT_Z0 + TAB_DZ])  # the TCP on the dry pad's tab
    back = pinch - APPROACH * 0.03
    if MODE != "nolock":
        out += [_w(1.5, sq, OPEN_C, "fingers either side of the lock buttons", SQZ_GRIP),
                _w(1.0, sq, OPEN_C, "wait for the body to go down", SQZ_GRIP),
                _w(1.0, sq, SQUEEZE, "squeeze both lock buttons in", SQZ_GRIP),
                _w(2.3, sq, SQUEEZE, "hold them in while the cap comes up", SQZ_GRIP),
                _w(0.8, sq, OPEN_C, "let go: the body stays locked", SQZ_GRIP),
                _w(1.4, sq - SQZ_APPROACH * 0.05 + np.array([0.0, 0.0, 0.03]), OPEN_C, "back away", SQZ_GRIP)]
        # 4 + 1.5 + 1.0 + 1.0 + 2.3 + 0.8 + 1.4 = 12.0 s
    else:
        out += [_w(8.0, sq_back, OPEN_C, "wait", SQZ_GRIP)]
    out += [_w(1.5, back + np.array([0.0, 0.0, 0.02]), -0.012, "in front of the dry pad's tab", TAB_GRIP),
            _w(1.2, back, -0.012, "line up with the tab", TAB_GRIP),
            _w(1.5, pinch, -0.012, "fingers either side of the tab", TAB_GRIP),
            _w(1.0, pinch, TAB_PINCH, "pinch the tab", TAB_GRIP),
            _w(3.0, pinch + out_dir * 0.040, TAB_PINCH, "draw the pad straight out", TAB_GRIP),
            _w(1.0, pinch + out_dir * 0.045 + np.array([0.0, 0.0, 0.002]), TAB_PINCH, "clear of the slot", TAB_GRIP)]
    # the dry pad into the tray: the TCP sits this far from the pad's middle
    tcp_from_mid = M @ np.array([-(PAD_L / 2 + TAB_BACK), 0.0, TAB_DZ - PAD_T / 2])
    tray = np.array([TRAY_C[0], TRAY_C[1], TOP + TRAY_FLOOR + PAD_T / 2])
    over = tray + tcp_from_mid + np.array([0.0, 0.0, 0.04])
    lay = tray + tcp_from_mid + np.array([0.0, 0.0, 0.006])
    out += [_w(1.5, pinch + out_dir * 0.045 + np.array([0.0, 0.0, 0.04]), TAB_PINCH, "lift it", TAB_GRIP),
            _w(2.5, over, TAB_PINCH, "carry it over the tray", TAB_GRIP),
            _w(1.5, lay, TAB_PINCH, "lower it into the tray", TAB_GRIP),
            _w(0.8, lay, -0.012, "let go", TAB_GRIP),
            _w(1.2, lay - APPROACH * 0.03 + np.array([0.0, 0.0, 0.03]), -0.012, "back away", TAB_GRIP)]
    # the new pad from the desk
    nmid = np.array([NEW_PAD_C[0], NEW_PAD_C[1], TOP + PAD_T / 2])
    npinch = nmid + tcp_from_mid
    nback = npinch - APPROACH * 0.03
    out += [_w(2.0, nback + np.array([0.0, 0.0, 0.03]), -0.012, "over the new pad's tab", TAB_GRIP),
            _w(1.2, nback, -0.012, "line up with its tab", TAB_GRIP),
            _w(1.2, npinch, -0.012, "fingers either side of the tab", TAB_GRIP),
            _w(1.0, npinch, TAB_PINCH, "pinch the tab", TAB_GRIP),
            _w(1.5, npinch + np.array([0.0, 0.0, 0.05]), TAB_PINCH, "lift the new pad", TAB_GRIP)]
    # the new pad hangs a little in the pinch (it pitches down 8-12 deg and sits 2.8 mm off along the stamp's length,
    # found by trial): aim it 2.8 mm over and 3 mm high at the mouth, and 2 mm high while it slides in
    lift = np.array([INS_DX, 0.0, INS_DZ])
    start = pinch + out_dir * 0.026 + lift  # the pad's rear 5 mm inside the slot's mouth
    out += [_w(2.5, start + out_dir * 0.015 + np.array([0.0, 0.0, 0.03]), TAB_PINCH, "carry it level to the slot",
               TAB_GRIP),
            _w(1.5, start + out_dir * 0.012, TAB_PINCH, "line it up with the slot", TAB_GRIP),
            _w(1.0, start, TAB_PINCH, "into the slot's mouth", TAB_GRIP),
            _w(3.0, pinch - out_dir * PUSH_OVER + lift * np.array([1.0, 1.0, 0.6]), TAB_PINCH,
               "slide it home until it clicks", TAB_GRIP),
            _w(0.8, pinch - out_dir * PUSH_OVER + lift * np.array([1.0, 1.0, 0.6]), TAB_PINCH, "hold", TAB_GRIP),
            _w(0.8, pinch + np.array([0.0, 0.0, 0.0006]), -0.012, "let go of the tab", TAB_GRIP),
            _w(1.5, pinch - APPROACH * 0.04, -0.012, "back away", TAB_GRIP)]
    park = pinch - APPROACH * 0.04 + out_dir * 0.05 + np.array([0.0, 0.0, 0.08])
    out += [_w(2.0, park, -0.012, "rise clear", TAB_GRIP)]
    T_RIGHT_DONE = sum(w.duration for w in out)
    out += [_w(40.0, park, -0.012, "stay clear", TAB_GRIP)]
    return out


def demo():
    right = _right()
    return {"left": _left(), "right": right}
