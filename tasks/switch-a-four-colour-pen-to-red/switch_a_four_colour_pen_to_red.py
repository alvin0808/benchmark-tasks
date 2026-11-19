"""On the station table: a four-colour retractable ballpoint pen (the size of a BIC 4 Colours: 145 mm long, 12 mm
across, 11.5 g) lies on a desk pad with its black tip out. Task (instruction): switch the pen to red and leave it lying
on the pad. The instruction does not say how.

How it is done (Pilot's instructions for its four-colour ballpoint pens, which work the same way): "Slide the lever of
the colour you wish to use and the writing tip comes out, and is set in place"; "To retract the writing tip, slide
other lever." Each colour has its own slider in a slot round the top of the barrel; sliding one down toward the tip
until it catches pushes its refill out and lets go of the one that was out.

The hand skill: the slider's thumb tab is 4.5 mm wide and stands 3 mm proud of the barrel, 12.7 mm from the next
colour's tab; it takes 2 N to start it and 4 N at the end of its 10 mm stroke, and a stroke that stops short springs
back. The pen is a light round body lying loose: pushed along its length on its own it just slides over the pad (it
weighs 0.11 N), so it has to be held (the other hand pinches the barrel near its tip) while one fingertip pushes the
red tab all the way down without touching the black one beside it, and then both let go.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- desk pad: loose, 24 x 34 cm (long side along y), 3 mm thick, 0.30 kg, middle at (0.20, -0.01).
- pen: loose, lying on the pad along y, its axis at x 0.19, the cap end at y -0.085 and the point at y +0.060. Seen
  from the cap end, the four sliders sit 90 deg apart round the barrel; angles below are measured from straight up
  toward +x: red 45 deg (top, far side), black 315 deg (top, near side), blue 135 deg and green 225 deg (the pen rests
  on these two and on its lower barrel); the clip at 270 deg (toward the robot). From the cap end: the end cap
  0-10 mm, the upper barrel 10-62 mm (blue), the slots 20-44 mm, the lower barrel 62-125 mm (white), the cone and
  point 125-145 mm. Each slider's tab is 8 mm long, its back (cap-side) face 22 mm from the cap end when up and 32 mm
  when down. 8.5 g barrel and four 0.75 g sliders (with their refills): 11.5 g.
- each slider is a separate part on a slide along the barrel: 0 to 10 mm of travel, a spring (200 N/m) that pulls it
  back up with 2.0 N at the top and 4.0 N at the bottom. The catch (a rule on the sliders' positions): a slider pushed
  down to 9.5 mm is held there (its spring's anchor moves past the bottom, pressing it onto its stop with 4 N) and a
  slider pushed down past 6 mm lets go of the one held (as on the real pen, sliding another lever retracts the tip that
  was out). A slider that was let go springs back up and acts on the catch again only once it is above 5 mm. Its
  refill's tip is out of the point when the slider is past 8 mm (2 mm out when down). At the start black is held down
  (its tip 2 mm out), the others are up.

Score (at the end of the run):
- 0.3 when the three colours other than red are retracted (their sliders within 1 mm of the top: no tip out).
- 0.7 more when red is held out by its catch (red slider 9.5 mm or more down and the catch holding it: its tip 1.5 mm
  or more out of the point), counted only with the pen let go (every finger point 2 cm or more from it), lying on the
  pad (its middle over the pad, its axis at most 3 mm higher above the pad than where it rests on its two tabs) and
  still (under 1 mm in 1 s): a tip held out by a finger is not set.
- 1.0 (success, held 1 s) = both.
A different colour latched leaves the 0.3 out (it is not retracted) and the 0.7 out (red is not out); so does black
left out. Nothing done scores 0.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, SphereShape, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# -- the layout ----------------------------------------------------------------------------------------------------
PAD_T = 0.003
PAD_C = (0.20, -0.01)
PAD_X, PAD_Y = 0.24, 0.34
PAD_TOP = TOP + PAD_T

PX = 0.19  # the pen's axis
CAP_Y = -0.085  # the cap end
PEN_L = 0.145
R = 0.006  # barrel radius
TAB_W, TAB_L, TAB_R0, TAB_R1 = 4.5 * MM, 8.0 * MM, 5.0 * MM, 9.0 * MM  # tab: width, length, radial from-to
TAB_S0 = 22.0 * MM  # the tab's back face from the cap end, retracted
TRAVEL = 10.0 * MM
LATCH = 9.5 * MM  # pushed this far, the catch holds it
RELEASE = 6.0 * MM  # pushed this far, another slider lets go of the held one
REARM = 5.0 * MM  # a slider that was let go acts on the catch again once it is back above this
TIP_HIDE = 8.0 * MM  # travel at which the tip reaches the point
K_SPRING = float(os.environ.get("PEN4_K", "200"))  # N/m
PRE = 10.0 * MM  # the spring's preload at the top (2.0 N with 200 N/m)
HOLD_PRE = 20.0 * MM  # held: the spring's anchor this far past the top, i.e. pressing down against the bottom stop
# The pen's linkage origin: on the axis, 72.5 mm from the cap end (the middle), at the height where it rests.
LOW = TAB_R1 * math.cos(math.radians(45)) + TAB_W / 2 * math.sin(math.radians(45))  # axis above the pad on two tabs
AXIS_Z = PAD_TOP + LOW + 0.0008
ORIGIN = (PX, CAP_Y + PEN_L / 2, AXIS_Z)
COLOURS = ("red", "blue", "green", "black")
ANGLE = {"red": 45.0, "blue": 135.0, "green": 225.0, "black": 315.0}
START_DOWN = "black"

RED_C, BLUE_C, GREEN_C, BLACK_C = (0.85, 0.08, 0.08), (0.1, 0.25, 0.85), (0.05, 0.6, 0.2), (0.06, 0.06, 0.06)
INK = {"red": RED_C, "blue": BLUE_C, "green": GREEN_C, "black": BLACK_C}
BARREL_BLUE, WHITE, SLOT, STEEL, PAD_COL = (0.2, 0.42, 0.85), (0.93, 0.93, 0.9), (0.1, 0.1, 0.12), (0.75, 0.76, 0.78), \
    (0.32, 0.38, 0.42)


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _s(s: float) -> float:
    """y of a point ``s`` from the cap end, relative to the linkage origin."""
    return CAP_Y + s - ORIGIN[1]


def _roty(phi_deg: float) -> tuple:
    a = math.radians(phi_deg) / 2
    return (0.0, math.sin(a), 0.0, math.cos(a))


def _radial(phi_deg: float, r: float) -> tuple:
    a = math.radians(phi_deg)
    return (r * math.sin(a), r * math.cos(a))  # (x, z) of a point at radius r, angle phi from up toward +x


def _slider(colour: str) -> Link:
    phi = ANGLE[colour]
    s0 = TAB_S0 + (TRAVEL if colour == START_DOWN else 0.0)
    rm = (TAB_R0 + TAB_R1) / 2
    x, z = _radial(phi, rm)
    tab = BoxShape((x, _s(s0 + TAB_L / 2), z), (TAB_W, TAB_L, TAB_R1 - TAB_R0), INK[colour], quat_xyzw=_roty(phi))
    # a ridge across the tab's back end (the thumb's grip), drawn only
    xr, zr = _radial(phi, TAB_R1 + 0.0002)
    ridge = BoxShape((xr, _s(s0 + 1.0 * MM), zr), (TAB_W, 1.6 * MM, 0.0004), (0.9, 0.9, 0.9), collide=False,
                     quat_xyzw=_roty(phi))
    # the refill's writing tip (ink colour), inside the point until the slider is past 8 mm; drawn only
    tip_end = PEN_L - TIP_HIDE + (TRAVEL if colour == START_DOWN else 0.0)  # its end, from the cap end
    tip = [CylShape((0.0, _s(tip_end - 2.5 * MM), 0.0), 0.0009, 5.0 * MM, INK[colour], collide=False, axis="Y"),
           SphereShape((0.0, _s(tip_end - 0.4 * MM), 0.0), 0.0006, STEEL, collide=False)]
    lim = (-TRAVEL, 0.0) if colour == START_DOWN else (0.0, TRAVEL)
    return Link(colour, (tab, ridge, *tip), mass=0.00075, anchor=ORIGIN, axis="Y", kind="prismatic", limits=lim,
                stiffness=K_SPRING, spring_damping=0.5)


def _pen() -> Linkage:
    sh = [CylShape((0.0, _s(0.005), 0.0), R - 0.0004, 0.010, WHITE, axis="Y"),  # end cap
          SphereShape((0.0, _s(0.0025), 0.0), R - 0.0012, WHITE),
          CylShape((0.0, _s(0.036), 0.0), R, 0.052, BARREL_BLUE, axis="Y"),  # upper barrel 10-62 mm
          CylShape((0.0, _s(0.0935), 0.0), R, 0.063, WHITE, axis="Y"),  # lower barrel 62-125 mm
          CylShape((0.0, _s(0.129), 0.0), 0.0052, 0.008, WHITE, axis="Y"),  # cone and point, in steps
          CylShape((0.0, _s(0.136), 0.0), 0.0040, 0.006, WHITE, axis="Y"),
          CylShape((0.0, _s(0.142), 0.0), 0.0026, 0.006, STEEL, axis="Y"),
          CylShape((0.0, _s(0.1449), 0.0), 0.0012, 0.0002, SLOT, collide=False, axis="Y")]  # the hole in the point
    for c in COLOURS:  # the slots (drawn only)
        x, z = _radial(ANGLE[c], R + 0.0001)
        sh.append(BoxShape((x, _s(0.032), z), (3.0 * MM, 0.024, 0.0004), SLOT, collide=False,
                           quat_xyzw=_roty(ANGLE[c])))
    # the clip, at 270 deg: a strip 3 mm wide standing 1.5 mm off the barrel, from the cap down to 48 mm
    xc, zc = _radial(270.0, R + 0.0021)
    sh.append(BoxShape((xc, _s(0.027), zc), (3.0 * MM, 0.042, 1.2 * MM), STEEL, quat_xyzw=_roty(270.0)))
    xb, zb = _radial(270.0, R + 0.0010)
    sh.append(BoxShape((xb, _s(0.008), zb), (3.0 * MM, 0.008, 3.0 * MM), STEEL, quat_xyzw=_roty(270.0)))
    links = [_slider(c) for c in COLOURS]
    return Linkage.build("pen", pos=ORIGIN, mass=0.0085, color=WHITE, floating=True, friction=(0.5, 0.4),
                         shapes=sh, links=links, contact_offset=0.0008, solver_iterations=48)


PAD = Body.build("desk_pad", pos=(PAD_C[0], PAD_C[1], TOP + PAD_T / 2 + 0.0003), mass=0.30, color=PAD_COL,
                 friction=(0.6, 0.5), shapes=[BoxShape((0.0, 0.0, 0.0), (PAD_X, PAD_Y, PAD_T), PAD_COL)])
PEN = _pen()
OBJECTS = [PAD, PEN]

# -- tracking and scoring -------------------------------------------------------------------------------------------
RETRACTED = 1.0 * MM  # a slider within this of the top: its tip is in
CLEAR = 0.02  # every finger point this far from the pen: let go
REST = 3.0 * MM  # its lowest point within this of the pad's top
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("PEN4_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("PEN4_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}

# points on the pen (linkage frame, at the start) for the finger-clearance check: the barrel's surface and the tabs
_PTS = np.array([[R * math.sin(a), _s(s), R * math.cos(a)] for s in np.linspace(0.0, PEN_L, 30)
                 for a in np.radians(np.arange(0, 360, 30))]
                + [[*(_radial(ANGLE[c], TAB_R1)[:1]), _s(TAB_S0 + d), _radial(ANGLE[c], TAB_R1)[1]]
                   for c in COLOURS for d in np.linspace(0.0, TAB_L + TRAVEL, 6)])


@on_reset
def _reset() -> None:
    _EP.clear()
    _EP.update(tick=0, step=0, counted=-1, held=START_DOWN, peak={c: 0.0 for c in COLOURS},
               armed={c: c != START_DOWN for c in COLOURS}, recent=[], near=1.0, gaps={}, tcp={}, events=[])


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


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("PEN4_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _set_targets(pen, device) -> None:
    """The springs' anchors: the held slider presses down on its bottom stop, the others are pulled to the top."""
    import torch

    vals = []
    for c in COLOURS:
        off = TRAVEL if c == START_DOWN else 0.0  # joint 0 is where the slider started
        vals.append((TRAVEL + HOLD_PRE if c == _EP["held"] else -PRE) - off)
    pen.actuators.target_command.set_position_index(
        value=torch.tensor([vals], dtype=torch.float32, device=device), joint_ids=_IDS["joints"])


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
    pen = base.scene["pen"]
    if "joints" not in _IDS:
        names = list(pen.joint_names)
        _IDS["joints"] = [names.index(f"{c}_turn") for c in COLOURS]
        _IDS["base"] = list(pen.body_names).index("base")
    q = as_numpy(pen.data.joint_pos)[0]
    trav = {c: float(q[j]) + (TRAVEL if c == START_DOWN else 0.0) for c, j in zip(COLOURS, _IDS["joints"])}
    # the catch. A slider acts on it only on its way down: once let go it must spring back above 5 mm first.
    armed = _EP["armed"]
    for c in COLOURS:
        if c != _EP["held"] and trav[c] < REARM:
            armed[c] = True
    for c in COLOURS:
        if c == _EP["held"] or not armed[c]:
            continue
        if trav[c] >= RELEASE and _EP["held"] is not None:
            _EP["events"].append((_EP["tick"], f"{_EP['held']} let go by {c}"))
            armed[_EP["held"]] = False
            _EP["held"] = None
        if trav[c] >= LATCH:
            _EP["held"] = c
            armed[c] = False
            _EP["events"].append((_EP["tick"], f"{c} caught"))
    _set_targets(pen, base.device)
    for c in COLOURS:
        _EP["peak"][c] = max(_EP["peak"][c], trav[c])
    bp = as_numpy(pen.data.body_link_pose_w)[0][_IDS["base"]]
    p, r = bp[:3] - origin, rotation(bp[3:7])
    o = np.asarray(ORIGIN, float)
    pts = p + (_PTS) @ r.T
    tree = cKDTree(pts)
    near = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    axis = r[:, 1]
    tilt = math.degrees(math.asin(min(1.0, abs(float(axis[2])))))
    low = float(pts[:, 2].min()) - (PAD_TOP - R)  # the barrel's lowest sample above the pad (0: barrel on the pad)
    pad = as_numpy(base.scene["desk_pad"].data.root_link_pose_w)[0][:3] - origin
    if _DEBUG:  # the red tab's centre and the right fingers' points near it, in the pen's frame (x, s, z) in mm
        names = list(pen.body_names)
        rp = as_numpy(pen.data.body_link_pose_w)[0][names.index("red")]
        rx, rz = _radial(ANGLE["red"], (TAB_R0 + TAB_R1) / 2)
        tab = rp[:3] - origin + rotation(rp[3:7]) @ np.array([rx, _s(TAB_S0 + TAB_L / 2), rz])
        to_pen = lambda w: (np.asarray(w) - p) @ r - np.array([0.0, _s(0.0), 0.0])  # noqa: E731
        fr = to_pen(fingers["right"])
        fr = fr[(np.abs(fr[:, 0]) < 0.015) & (fr[:, 2] < 0.012)]
        _EP["dbg_tab"] = to_pen(tab) * 1000
        _EP["dbg_fr"] = (fr.min(0) * 1000, fr.max(0) * 1000) if len(fr) else None
        _EP["roll"] = math.degrees(math.atan2(float(r[0, 2]), float(r[2, 2])))
    _EP.update(trav=trav, p=p, near=near, tilt=tilt, low=low, pad_z=float(pad[2] + PAD_T / 2),
               on_pad=bool(abs(p[0] - pad[0]) <= PAD_X / 2 and abs(p[1] - pad[1]) <= PAD_Y / 2),
               cap=p + r @ (np.array([0.0, _s(0.0), 0.0])), moved=float(np.linalg.norm(p[:2] - o[:2])))
    del o


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "p" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["p"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    trav = _EP.get("trav", {c: (TRAVEL if c == START_DOWN else 0.0) for c in COLOURS})
    others_in = all(trav[c] <= RETRACTED for c in COLOURS if c != "red")
    red_set = _EP["held"] == "red" and trav["red"] >= LATCH
    # lying on the pad: its middle over the pad, its axis at most REST above where it rests on its two tabs
    resting = "p" in _EP and _EP["on_pad"] and float(_EP["p"][2]) - _EP["pad_z"] <= LOW + REST
    free = _EP["near"] >= CLEAR
    red_counts = red_set and free and resting and still
    s = (0.3 if others_in else 0.0) + (0.7 if red_counts else 0.0)
    return dict(still=still, others_in=others_in, red_set=red_set, resting=resting, free=free, score=s)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p = _EP.get("p", np.zeros(3))
    tr = _EP.get("trav", {})
    print(f"DBG step={_EP['step']} " + " ".join(f"{c}={tr.get(c, 0) * 1000:.1f}" for c in COLOURS)
          + f" held={_EP['held']} pen=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.4f}) tilt={_EP.get('tilt', 0):.1f} "
          f"moved={_EP.get('moved', 0) * 1000:.1f}mm near={_EP['near'] * 1000:.0f}mm still={int(t['still'])} "
          f"rest={int(t['resting'])} score={t['score']:.2f} ev={_EP['events'][-3:]} roll={_EP.get('roll', 0):.1f} "
          f"tab={np.round(_EP.get('dbg_tab', np.zeros(3)), 1)} "
          f"fr={_EP.get('dbg_fr') and [np.round(a, 1) for a in _EP['dbg_fr']]} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return float(t["score"])


def success(objects) -> bool:
    return bool(_tally(objects)["score"] >= 0.999)


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper comes down over the barrel 105 mm from the cap end, tilted 40 deg
# about x so its wrist leans away (toward +y) from the right one, fingers closing across the barrel, and pinches it
# (6 mm commanded gap on the 12 mm barrel). The right gripper, closed, points at the pen's axis from the red tab's side
# (turned 45 deg about y, tilted 15 deg about x), its fingertips coming down 0.5 mm off the barrel 3 mm behind the red
# tab, and pushes along the barrel toward the point, 25 mm past the tab's 10 mm stroke (the arm, soft sideways, stops
# short under load), holds 1 s, lifts off and backs away; the left one then opens and rises.
# PEN4_ONE=1 (the wrong way): the right arm alone does the same push, with nothing holding the pen.
OPEN_C = -0.047
ONE = os.environ.get("PEN4_ONE") == "1"
TILT_L = float(os.environ.get("PEN4_TILT_L", "-40"))  # deg about x (negative: wrist toward +y)
TILT_R = float(os.environ.get("PEN4_TILT_R", "15"))
HOLD_S = float(os.environ.get("PEN4_HOLD_S", "0.105"))  # where the left fingers pinch, from the cap end
HOLD_DZ = float(os.environ.get("PEN4_HOLD_DZ", "0.008"))  # TCP above the axis while pinching
PINCH = float(os.environ.get("PEN4_PINCH", "-0.003"))  # finger value (minus half the gap): 6 mm on a 12 mm barrel
PSI = float(os.environ.get("PEN4_PSI", "45"))  # the right hand points at the axis from this angle (the red tab's)
TIP_R = float(os.environ.get("PEN4_TIP_R", "0.0065"))  # its fingertips this far from the axis (barrel: 6 mm)
TIP_DEPTH = 0.0136  # the closed fingertips' deepest point past the TCP along the hand's pointing direction
TIP_FRONT = 0.0098  # and their front (toward the pen's point) this far ahead of the TCP along the pen
PUSH_GAP = float(os.environ.get("PEN4_PUSH_GAP", "0.003"))  # the fingertips this far behind the tab at the start
PUSH_OVER = float(os.environ.get("PEN4_PUSH_OVER", "0.025"))  # pushed this far past the tab's 10 mm of travel
CLOSED = 0.0


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


ROT_L = _rx(TILT_L) @ grip_rotation(-math.pi / 2)
def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


ROT_R = _rx(TILT_R) @ _ry(PSI) @ grip_rotation(math.pi / 2)


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


def _stay(arm: str, t: float = 120.0) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(t, tuple(float(v) for v in pos), OPEN_C, "stay put", rot)]


SETTLE = 0.00165  # the axis settles this much lower than spawned (onto its two tabs, tip end on the barrel)


def _on_axis(s: float, dx: float = 0.0, dz: float = 0.0) -> np.ndarray:
    return np.array([PX + dx, CAP_Y + s, AXIS_Z - SETTLE + dz])


def _left() -> list:
    hold = _on_axis(HOLD_S, 0.0, HOLD_DZ)
    out = _lead_in("left", hold + np.array([0.0, 0.0, 0.06]), ROT_L, OPEN_C, "above the barrel near the point, open")
    out += [_w(2.0, hold + np.array([0.0, 0.0, 0.015]), OPEN_C, "lower round the barrel", ROT_L),
            _w(1.0, hold, OPEN_C, "lower round the barrel", ROT_L),
            _w(1.2, hold, PINCH, "pinch the barrel", ROT_L),
            _w(11.0, hold, PINCH, "hold the pen while the red slider is pushed", ROT_L),
            _w(1.0, hold, OPEN_C, "let go", ROT_L),
            _w(2.0, hold + np.array([0.0, 0.0, 0.06]), OPEN_C, "rise clear", ROT_L),
            _w(10.0, hold + np.array([0.0, 0.0, 0.08]), OPEN_C, "stay clear", ROT_L)]
    return out


def _push_at(s: float, lift: float = 0.0) -> np.ndarray:
    """The right TCP with its fingertips at ``s`` (their front, from the cap end), ``TIP_R`` from the axis on the red
    side, ``lift`` farther out."""
    u = np.array([math.sin(math.radians(PSI)), 0.0, math.cos(math.radians(PSI))])
    return _on_axis(s - TIP_FRONT) + u * (TIP_R + TIP_DEPTH + lift)


UP4 = np.array([0.0, 0.0, 0.04])
AWAY = _push_at(TAB_S0 + TRAVEL, 0.02) + np.array([-0.03, -0.02, 0.035])  # backed away, up and toward the robot


def _right() -> list:
    start = _push_at(TAB_S0 - PUSH_GAP)
    end = _push_at(TAB_S0 + TRAVEL + PUSH_OVER)
    wait = 0.0 if ONE else 4.5
    out = _lead_in("right", _push_at(TAB_S0 - PUSH_GAP, 0.006) + UP4, ROT_R, CLOSED,
                   "above the red tab's back, pointing at the barrel, closed")
    out += [_w(wait + 0.5, _push_at(TAB_S0 - PUSH_GAP, 0.006) + UP4, CLOSED, "wait for the pen to be held", ROT_R),
            _w(2.0, _push_at(TAB_S0 - PUSH_GAP, 0.006), CLOSED, "come down behind the red tab", ROT_R),
            _w(1.0, start, CLOSED, "fingertips on the barrel behind the red tab", ROT_R),
            _w(2.5, end, CLOSED, "push the red slider down the barrel to its catch", ROT_R),
            _w(1.0, end, CLOSED, "hold it there a moment", ROT_R),
            _w(1.5, _push_at(TAB_S0 + TRAVEL + PUSH_OVER, 0.03), CLOSED, "lift off the pen", ROT_R),
            _w(2.0, AWAY, CLOSED, "back away", ROT_R),
            _w(12.0, AWAY, CLOSED, "stay clear", ROT_R)]
    return out


def demo():
    if ONE:
        return {"right": _right(), "left": _stay("left")}
    return {"left": _left(), "right": _right()}
