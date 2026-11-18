"""On the station table: a true-size adult care-training lower leg (knee to foot) lies on its back on a thin foam bed
pad, the calf cradled in a foam leg trough, the heel resting on the pad; its ankle is a hinge with a sprung, stiffening
passive moment, like a relaxed ankle. Task (instruction): exercise the ankle through its range three times, foot up
toward the shin and toes pointed down, gently, without forcing it, the leg kept on the bed; finish with the foot back
at rest and let go. The instruction does not say how.

Procedure (NNAAP nurse aide skills exam, Skill 15, Credentia candidate handbook): "While supporting the foot and ankle
close to the bed, pushes/pulls foot toward head (dorsiflexion), and pushes/pulls foot down, toes point down (plantar
flexion) (AT LEAST 3 TIMES unless pain is verbalized). Moves joints gently, slowly, and smoothly through the range of
motion." Two hands, because of the leg itself: it is loose and light, so a hand that only moves the foot moves the leg
too. Pointing the toes takes about 0.9 N m at 30 deg, more than the 0.7 N m by which the lower leg's own weight holds
its knee end down, so a hand on the foot alone levers the leg up off the bed about the ankle (and drags it down the
bed). One hand holds the leg just above the ankle; the other turns the foot.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. leg_mesh.py
builds the two moulded surfaces from ANSUR II women's means (foot 246 mm, ankle 216 mm round, calf 373 mm round).
- bed pad: fixed, 24 x 64 x 1.2 cm, x 0.095-0.335, y -0.215-0.425; a foam leg trough on it: two walls 3 cm thick,
  7 cm tall, 11.9 cm long, their inner faces 5.53 cm either side of the leg's midline (x 0.215), from 22 to 34 cm up
  the leg (they keep the calf from rolling outward; a lying leg rolls out unless something holds it).
- training leg: loose. Lower leg 0.30 kg (hollow shell; centre of mass 24 cm up from the ankle), its knee end 41.5 cm
  from the ankle axis; it lies along +y, knee toward the robot's left, resting on the heel and the calf, tilted 7.2 deg
  (knee end up). Ankle axis (through the malleoli, 50 mm above the sole, 49 mm in front of the back of the heel) at
  (0.215, -0.080, table + 0.064), along x. Foot 0.15 kg, toes up (top of the toes 25.7 cm above the table).
- ankle: a hinge about x. Its restoring moment rises in steps toward each end (N m/rad): dorsiflexion 1.8 to 10 deg,
  5.4 to 20 deg, 18 beyond; plantarflexion 1.2 to 20 deg, 3.0 to 45 deg, 15 beyond (0.79 N m at 15 deg up, 1.26 at
  20, 0.94 at 30 deg down, 1.73 at 45); hard stops at 30 deg up and 55 deg down. At rest on its heel the foot sits
  about 3 deg toward plantarflexion. Skin friction 1.2 static.

Score (0 for the rest of the run once a rule is broken; otherwise repetitions counted over the run):
- 0: the foot pushed past the end of its range (dorsiflexion over 20 deg or plantarflexion over 45 deg); the ankle
  jerked (its angle changing faster than 90 deg/s between control steps); the heel lifted more than 1 cm above where it
  lay; the knee end lifted more than 1 cm (the leg levered up off the bed); the leg dragged more than 2 cm along the
  bed (its knee end).
- 0.25 per repetition, at most 0.75: a repetition is dorsiflexion to 15 deg or more and plantarflexion to 30 deg or
  more, one after the other (each counted only after the other has been reached since).
- 1.0 (success, held 1 s): three repetitions, the foot back within 5 deg of its resting angle, the heel and the knee
  end back down (within 5 mm of where they lay), every finger point 3 cm or more from the leg and foot, and the leg
  and foot still (under 0.5 deg and 1 mm over 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import leg_mesh as lm  # noqa: E402
from bodykit import Body, BoxShape, Link, Linkage, MeshShape, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the repetitions counted itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"
CM = 0.01

# -- the layout ----------------------------------------------------------------------------------------------------
PAD_T = 0.012  # the bed pad's thickness
PAD_X, PAD_Y = (0.095, 0.335), (-0.215, 0.425)  # its extent on the table
LEG_X = 0.215  # the leg's midline
AXIS_Y = -0.080  # the ankle's hinge axis (y)
SKIN, PAD_BLUE, TAPE, TROUGH_BLUE = (0.90, 0.76, 0.66), (0.60, 0.72, 0.84), (0.10, 0.10, 0.12), (0.42, 0.56, 0.72)

SHANK_KG, FOOT_KG = 0.30, 0.15  # a hollow moulded lower leg and foot
LEG_FRICTION = (1.2, 1.0)  # soft vinyl skin


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


SHANK_P, SHANK_F, SHANK_N = lm.shank_mesh()
FOOT_P, FOOT_F, FOOT_N = lm.foot_mesh()


def _rest_tilt() -> float:
    """The leg's tilt (rad, knee end up) when it lies on a flat surface on its heel and its calf."""
    lo, hi = 0.0, math.radians(14.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        r = _rx(mid)
        heel = float((FOOT_P @ r.T)[:, 2].min())
        calf = float((SHANK_P @ r.T)[:, 2].min())
        if calf < heel:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


TILT = _rest_tilt()  # about 7 deg
R_TILT = _rx(TILT)
AXIS_H = -float((FOOT_P @ R_TILT.T)[:, 2].min())  # the axis above the surface the leg lies on
AXIS = np.array([LEG_X, AXIS_Y, TOP + PAD_T + AXIS_H + 0.001])  # world, at the start (1 mm above the pad)


# -- the ankle's passive moment ------------------------------------------------------------------------------------
# Restoring moment (N m) against the angle from neutral, piecewise linear (slopes in N m/rad between breakpoints in
# degrees): soft in mid-range, stiffening toward each end, a hard stop past it.
DF_SEGS = [(0.0, 1.8), (10.0, 5.4), (20.0, 18.0)]  # dorsiflexion side: from 0 deg 1.8 N m/rad, from 10 deg 5.4 ...
PF_SEGS = [(0.0, 1.2), (20.0, 3.0), (45.0, 15.0)]  # plantarflexion side
DF_STOP, PF_STOP = 30.0, 55.0  # hard stops (deg)
DF_END, PF_END = 20.0, 45.0  # the end of the trainer's range: past here it is being forced
SPRING_C, JOINT_C, JOINT_F = 0.02, 0.03, 0.02  # drive damping (N m s/rad), viscous (N m s/rad), Coulomb (N m)


def _segments(segs):
    """[(start_deg, k, c_rad)] with moment = k * (angle - c) on each segment, continuous from 0 at 0 deg."""
    out, m0 = [], 0.0
    for i, (a0, k) in enumerate(segs):
        c = math.radians(a0) - m0 / k
        out.append((a0, k, c))
        if i + 1 < len(segs):
            m0 += k * math.radians(segs[i + 1][0] - a0)
    return out


DF_LIN, PF_LIN = _segments(DF_SEGS), _segments(PF_SEGS)


def ankle_moment(theta_deg: float) -> float:
    """The restoring moment (N m) at ``theta_deg`` (dorsiflexion positive), positive pushing back toward plantar."""
    segs = DF_LIN if theta_deg >= 0 else PF_LIN
    a = abs(theta_deg)
    for a0, k, c in reversed(segs):
        if a >= a0:
            m = k * (math.radians(a) - c)
            return m if theta_deg >= 0 else -m
    return 0.0


def drive_for(q: float):
    """(stiffness N m/rad, target rad) for the ankle joint's drive at joint angle ``q`` (rad, + = plantarflexion):
    the tangent line of the moment curve on the segment the angle is in."""
    theta = -math.degrees(q)
    segs = DF_LIN if theta >= 0 else PF_LIN
    a = abs(theta)
    for a0, k, c in reversed(segs):
        if a >= a0:
            return (k, -c) if theta >= 0 else (k, c)
    return (segs[0][1], 0.0)


# -- bodies ---------------------------------------------------------------------------------------------------------
def _quat(m: np.ndarray):
    from bodykit import _quat_from_matrix

    return _quat_from_matrix(np.asarray(m, float))


def _tape_marks() -> list:
    """A black tape strip along the outer side of the shin (2 to 14 cm above the ankle), for reading the ankle angle on
    a side photo against the one on the foot."""
    w, top, bot = lm.leg_section(8.0)
    c = np.array([-w / 2 * CM - 0.0004, 8.0 * CM, 0.0])
    return [BoxShape(tuple((R_TILT @ c).tolist()), (0.001, 0.12, 0.006), TAPE, collide=False, quat_xyzw=_quat(R_TILT))]


def _foot_tape() -> list:
    """A black tape strip along the outer side of the foot (3 to 13 cm along it from the ankle's axis), at right angles
    to the shin's strip with the ankle in neutral."""
    x = -max(lm.foot_section(8.0)[0], lm.foot_section(18.0)[0]) / 2 * CM - 0.0004
    c = np.array([x, -2.5 * CM, 8.0 * CM])
    return [BoxShape(tuple((R_TILT @ c).tolist()), (0.001, 0.006, 0.10), TAPE, collide=False, quat_xyzw=_quat(R_TILT))]


def _leg() -> Linkage:
    shank_pts = SHANK_P @ R_TILT.T
    foot_pts = FOOT_P @ R_TILT.T
    shank = [MeshShape(shank_pts, SHANK_F, SHANK_N @ R_TILT.T, SKIN)] + _tape_marks()
    foot = [MeshShape(foot_pts, FOOT_F, FOOT_N @ R_TILT.T, SKIN)] + _foot_tape()
    k0, _ = drive_for(0.0)
    links = [Link("foot", tuple(foot), mass=FOOT_KG, anchor=tuple(float(v) for v in AXIS), axis="X",
                  limits=(-DF_STOP, PF_STOP), friction=JOINT_F, damping=JOINT_C, stiffness=k0,
                  spring_damping=SPRING_C)]
    return Linkage.build("care_leg", pos=tuple(float(v) for v in AXIS), mass=SHANK_KG, color=SKIN, floating=True,
                         friction=LEG_FRICTION, shapes=shank, links=links)


TROUGH_Y = (22.0, 34.0)  # the leg trough's side walls: from/to (cm up the leg from the axis)
TROUGH_GAP, TROUGH_H, TROUGH_T = 0.0553, 0.070, 0.030  # inner faces this far either side of the midline; height; thick


def _pad_shapes() -> list:
    cx, cy = (PAD_X[0] + PAD_X[1]) / 2, (PAD_Y[0] + PAD_Y[1]) / 2
    out = [BoxShape((0.0, 0.0, 0.0), (PAD_X[1] - PAD_X[0], PAD_Y[1] - PAD_Y[0], PAD_T), PAD_BLUE)]
    y0 = AXIS_Y + TROUGH_Y[0] * CM * math.cos(TILT)
    y1 = AXIS_Y + TROUGH_Y[1] * CM * math.cos(TILT)
    for s in (1.0, -1.0):
        out.append(BoxShape((LEG_X + s * (TROUGH_GAP + TROUGH_T / 2) - cx, (y0 + y1) / 2 - cy,
                             PAD_T / 2 + TROUGH_H / 2), (TROUGH_T, y1 - y0, TROUGH_H), TROUGH_BLUE))
    return out


PAD = Body.build("bed_pad", pos=((PAD_X[0] + PAD_X[1]) / 2, (PAD_Y[0] + PAD_Y[1]) / 2, TOP + PAD_T / 2), mass=1.0,
                 color=PAD_BLUE, fixed=True, friction=(0.7, 0.6), shapes=_pad_shapes())
LEG = _leg()
OBJECTS = [PAD, LEG]

# -- scoring --------------------------------------------------------------------------------------------------------
DF_GOAL, PF_GOAL = 15.0, 30.0  # a repetition reaches both
REPS_FULL = 3
MAX_SPEED = 90.0  # deg/s, between control steps: faster is a jerk
HEEL_MAX = 0.010  # the heel more than this above where it lay: the foot and ankle were not kept close to the bed
LIFT_MAX = 0.010  # the knee end more than this above where it lay: the leg was lifted off the bed
DRAG_MAX = 0.020  # the leg slid more than this along the bed (its knee end)
BACK_DEG = 5.0  # at the end, the foot within this of where it rested
ON_PAD = 0.005  # at the end, the heel and the leg back on the bed (within 5 mm)
CLEAR = 0.03  # released: every finger point this far or more from the leg and foot
STILL_STEPS, STILL_DEG, STILL_MM = 20, 0.5, 1.0
_DEBUG = os.environ.get("ANKLE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("ANKLE_DEBUG_EVERY", "5"))
_EP: dict = {}
_TCP: dict = {}
_IDS: dict = {}

HEEL_PTS = FOOT_P[FOOT_P[:, 2] < -0.01]  # the heel's surface (leg frame)
KNEE_PTS = SHANK_P[SHANK_P[:, 1] > lm.LEG_END - 0.03]  # the knee end
SAMPLE_FOOT = FOOT_P[:: 3]
SAMPLE_SHANK = SHANK_P[:: 3]


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, failed="", events=[], theta=0.0, prev=None, rest=None, settle=0,
               heel=0.0, knee=0.0, drag=0.0, near=1.0, recent=[], speed=0.0, peak_df=0.0, peak_pf=0.0)
    _TCP.clear()
    _IDS.pop("drive", None)


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


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("ANKLE_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _set_drive(leg, q: float) -> None:
    """Keep the ankle's spring on the tangent of its moment curve at the present angle (stiffer toward the ends)."""
    import torch

    k, target = drive_for(q)
    j = _IDS["ankle"]
    if _IDS.get("drive") != (k, target):
        if _IDS.get("drive") is None or _IDS["drive"][0] != k:
            leg.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[k]], dtype=torch.float32, device=leg.device),
                                                   joint_ids=[j])
        leg.actuators.target_command.set_position_index(
            value=torch.tensor([[target]], dtype=torch.float32, device=leg.device), joint_ids=[j])
        _IDS["drive"] = (k, target)


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
    leg = base.scene["care_leg"]
    if "ankle" not in _IDS:
        names, jn = list(leg.body_names), list(leg.joint_names)
        _IDS["bodies"] = {k: names.index(k) for k in ("base", "foot")}
        _IDS["ankle"] = jn.index("foot_turn")
    bp = as_numpy(leg.data.body_link_pose_w)[0]
    jq = float(as_numpy(leg.data.joint_pos)[0][_IDS["ankle"]])
    _set_drive(leg, jq)
    ib = _IDS["bodies"]
    sp, sr = bp[ib["base"]][:3] - origin, rotation(bp[ib["base"]][3:7])
    fp, fr = bp[ib["foot"]][:3] - origin, rotation(bp[ib["foot"]][3:7])
    # body frames are world-aligned at the start, with their origin at the axis; the meshes were turned by R_TILT
    shank_w = lambda p: sp + (np.asarray(p) @ R_TILT.T) @ sr.T  # noqa: E731
    foot_w = lambda p: fp + (np.asarray(p) @ R_TILT.T) @ fr.T  # noqa: E731
    theta = -math.degrees(jq)  # dorsiflexion positive
    pad_top = TOP + PAD_T
    heel = float(foot_w(HEEL_PTS)[:, 2].min() - pad_top)
    knee_pts = shank_w(KNEE_PTS)
    knee_low = float(knee_pts[:, 2].min())
    knee_mid = knee_pts.mean(0)
    if _EP["prev"] is None:
        _EP.update(knee0=knee_low, knee_mid0=knee_mid.copy(), heel0=heel)
    knee = knee_low - _EP["knee0"]
    drag = float(np.linalg.norm((knee_mid - _EP["knee_mid0"])[:2]))
    speed = 0.0 if _EP["prev"] is None else abs(theta - _EP["prev"]) * 20.0
    _EP["prev"] = theta
    if _EP["rest"] is None:  # where the foot rests: read once it has settled (the first 10 control steps)
        _EP["settle"] += 1
        if _EP["settle"] >= 10:
            _EP["rest"] = theta
    pts = np.concatenate([shank_w(SAMPLE_SHANK), foot_w(SAMPLE_FOOT)])
    tree = cKDTree(pts)
    near = {side: float(tree.query(f, k=1)[0].min()) for side, f in fingers.items()}
    if not _EP["failed"] and _EP["rest"] is not None:
        if theta > DF_END:
            _EP["failed"] = f"the foot was pushed past the end of its range (dorsiflexion {theta:.1f} deg)"
        elif theta < -PF_END:
            _EP["failed"] = f"the foot was pushed past the end of its range (plantarflexion {-theta:.1f} deg)"
        elif speed > MAX_SPEED:
            _EP["failed"] = f"the ankle was jerked ({speed:.0f} deg/s)"
        elif heel > _EP["heel0"] + HEEL_MAX:
            _EP["failed"] = "the foot and ankle were lifted more than 1 cm off the bed"
        elif knee > LIFT_MAX:
            _EP["failed"] = "the leg was lifted off the bed at the knee end"
        elif drag > DRAG_MAX:
            _EP["failed"] = "the leg was dragged along the bed"
    ev = _EP["events"]
    if not _EP["failed"]:
        if theta >= DF_GOAL and (not ev or ev[-1] != "DF"):
            ev.append("DF")
        elif theta <= -PF_GOAL and (not ev or ev[-1] != "PF"):
            ev.append("PF")
    if _DEBUG:
        yaw = math.degrees(math.atan2(sr[0, 1], sr[1, 1]))
        roll = math.degrees(math.atan2(sr[2, 0], sr[2, 2]))
        _EP["dbg"] = dict(knee_dxy_mm=tuple(np.round((knee_mid - _EP["knee_mid0"])[:2] * 1000, 1).tolist()),
                          yaw=round(yaw, 2), roll=round(roll, 2),
                          axis_moved_mm=tuple(np.round((sp - AXIS) * 1000, 1).tolist()))
    _EP.update(theta=theta, heel=heel, knee=knee, drag=drag, near=min(near.values()), speed=speed, gaps=gaps,
               peak_df=max(_EP["peak_df"], theta), peak_pf=max(_EP["peak_pf"], -theta), knee_mid=knee_mid)


def _reps() -> int:
    return len(_EP["events"]) // 2


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "knee_mid" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append((_EP["theta"], np.asarray(_EP["knee_mid"], float).copy()))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(abs(t - recent[-1][0]) < STILL_DEG
                                               and np.abs(p - recent[-1][1]).max() * 1000 < STILL_MM
                                               for t, p in recent)
    back = _EP["rest"] is not None and abs(_EP["theta"] - _EP["rest"]) <= BACK_DEG
    lying = _EP["heel"] <= _EP.get("heel0", 0.0) + ON_PAD and abs(_EP["knee"]) <= ON_PAD
    done = bool(_reps() >= REPS_FULL and back and lying and _EP["near"] >= CLEAR and still)
    return dict(still=still, back=back, lying=lying, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    print(f"DBG step={_EP['step']} theta={_EP['theta']:+.1f} rest={_EP['rest']} M={ankle_moment(_EP['theta']):+.2f}Nm "
          f"speed={_EP['speed']:.0f} heel={_EP['heel'] * 1000:.1f}mm knee={_EP['knee'] * 1000:+.1f}mm "
          f"drag={_EP['drag'] * 1000:.1f}mm reps={_reps()} ev={''.join(e[0] for e in _EP['events'])} "
          f"peak=({_EP['peak_df']:.1f},{_EP['peak_pf']:.1f}) near={_EP['near'] * 1000:.0f}mm back={int(t['back'])} "
          f"lying={int(t['lying'])} still={int(t['still'])} drive={_IDS.get('drive')} failed={_EP['failed'] or '-'} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _TCP.items()), flush=True)
    if "dbg" in _EP:
        print(f"DBG   {_EP['dbg']}", flush=True)


def score(objects) -> float:
    """0 for good once a rule is broken; otherwise 0.25 per full repetition (dorsiflexion to 15 deg and
    plantarflexion to 30 deg), at most 0.75. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if _EP["failed"]:
        return 0.0
    return 0.25 * min(_reps(), REPS_FULL)


def success(objects) -> bool:
    """Three repetitions, the foot back at rest, the leg lying on the bed, both grippers 3 cm or more away, still 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper takes the lower leg 10 cm above the ankle from above, fingers
# closing across it at its widest, its wrist leaning 20 deg back toward the knee (clear of the other wrist), and holds
# it where it lies. The right gripper comes round from the robot's right, clear of the toes, to the side of the
# forefoot, and slides in from the robot's side at the forefoot's height (pointing 35 deg down), one finger against the
# sole and one over the top of the foot, 17 cm along the foot from the heel; it closes and turns the foot about the
# ankle's axis: toward the shin (dorsiflexion), then down (plantarflexion), three times, 2.2 s each way, then back to
# where the foot rests; it lets go and backs out the way it came, and then the left gripper lets go of the leg.
# ANKLE_ONE=1 runs the wrong way: the right gripper alone, the left arm stays where it is.

ONE = os.environ.get("ANKLE_ONE") == "1"
OPEN_C = -0.047
FF_FX = 17.0  # the forefoot grip: cm along the foot from the back of the heel
FF_W, FF_S, FF_T = lm.foot_section(FF_FX)
FF_PT = R_TILT @ lm.foot_point(FF_FX, (FF_S + FF_T) / 2)  # grip point relative to the axis (world-aligned)
FF_PINCH = -((FF_T - FF_S) * CM - 0.010) / 2  # finger value: closed 5 mm inside each face
SUP_Y = 10.0  # the support grip: cm up the leg from the axis
SUP_W, SUP_TOP, SUP_BOT = lm.leg_section(SUP_Y)
# the pads (TCP +5 to -13 mm) straddle the leg's widest line, so the squeeze does not push the leg out upward
SUP_PT = R_TILT @ np.array([0.0, SUP_Y * CM, (SUP_TOP - SUP_BOT) / 2 * CM + 0.004])
SUP_PINCH = -((SUP_W * CM) - 0.004) / 2  # closed 2 mm inside each side
SUP_LEAN = math.radians(20.0)
ROT_SUP = _rx(-SUP_LEAN) @ grip_rotation(-math.pi / 2)  # fingers close along x; wrist leaning toward the knee
# The grip's commanded turn (deg, about the ankle axis; + plantarflexion). Squeezed between parallel pads, the tapering
# forefoot settles about 3.5 deg further toward plantarflexion than the grip, so the grip is set 3.5 deg the other
# way to take the foot as it rests; and the arm gives a little under the spring's moment, so it is told to go past
# the goals.
GRIP_Q0 = -3.5
DF_CMD, PF_CMD = 26.5, 31.5
HALF_S = 2.2  # seconds per half cycle
FF_PITCH = math.radians(35.0)  # the forefoot grip comes from the robot's side, pointing 35 deg down


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def ff_pose(q_deg: float):
    """(TCP, rotation) of the forefoot grip with the foot turned ``q_deg`` (+ plantarflexion) about the ankle axis:
    fingers closing along the foot's thickness (sole and top), pointing in from the robot's side and 35 deg down."""
    r = _rx(math.radians(q_deg))
    p = AXIS + r @ FF_PT
    return p, r @ _rx(TILT) @ _ry(-math.pi / 2 + FF_PITCH) @ grip_rotation(0.0)


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


T_HOLD = 13.0  # when (s) both grips are on: the right gripper starts turning the foot then


def _right() -> list:
    p0, r0 = ff_pose(GRIP_Q0)
    out_x = np.array([-0.08, 0.0, 0.0])  # toward the robot: the fingers clear of the foot's outer side
    safe = p0 + np.array([-0.06, -0.15, 0.0])  # off to the right of the foot, clear of the toes
    out = _lead_in("right", safe, r0, 0.0, "off to the side of the foot, fingers closed")
    out += [_w(1.0, safe, OPEN_C, "open", r0),
            _w(2.0, p0 + out_x, OPEN_C, "beside the forefoot, at its height", r0),
            _w(2.5, p0 + out_x, OPEN_C, "wait for the leg to be held", r0),
            _w(2.0, p0, OPEN_C, "slide in: one finger at the sole, one over the top of the foot", r0),
            _w(0.5, p0, OPEN_C, "settle", r0),
            _w(1.0, p0, FF_PINCH, "hold the forefoot", r0),
            _w(T_HOLD - 12.0, p0, FF_PINCH, "steady", r0)]
    steps = 8
    for rep in range(REPS_FULL):
        for target, label in ((-DF_CMD, "foot toward the shin (dorsiflexion)"), (PF_CMD, "toes down (plantarflexion)")):
            q_from = GRIP_Q0 if (rep == 0 and target < 0) else (-DF_CMD if target > 0 else PF_CMD)
            for k in range(1, steps + 1):
                q = q_from + (target - q_from) * (0.5 - 0.5 * math.cos(math.pi * k / steps))
                p, r = ff_pose(q)
                out.append(_w(HALF_S / steps, p, FF_PINCH, label, r))
            p, r = ff_pose(target)
            out.append(_w(0.4, p, FF_PINCH, "hold", r))
    for k in range(1, steps + 1):
        q = GRIP_Q0 + (PF_CMD - GRIP_Q0) * (0.5 + 0.5 * math.cos(math.pi * k / steps))
        p, r = ff_pose(q)
        out.append(_w(HALF_S / steps, p, FF_PINCH, "back to neutral", r))
    out += [_w(1.0, p0, FF_PINCH, "settle", r0),
            _w(1.0, p0, OPEN_C, "let go", r0),
            _w(2.0, p0 + out_x, OPEN_C, "back out the way it came", r0),
            _w(1.5, safe, OPEN_C, "move clear", r0),
            _w(4.0, safe, OPEN_C, "stay clear", r0)]
    return out


def _left(t_let_go: float) -> list:
    g = AXIS + SUP_PT
    above = g - 0.09 * ROT_SUP[:, 2]
    out = _lead_in("left", above + np.array([0.0, 0.0, 0.04]), ROT_SUP, OPEN_C, "above the lower leg, open")
    out += [_w(1.0, above, OPEN_C, "lower toward it", ROT_SUP),
            _w(1.5, g, OPEN_C, "fingers either side of the leg above the ankle", ROT_SUP),
            _w(0.5, g, OPEN_C, "settle", ROT_SUP),
            _w(1.0, g, SUP_PINCH, "hold the leg above the ankle", ROT_SUP),
            _w(1.0, g, SUP_PINCH, "settle", ROT_SUP)]
    out.append(_w(max(t_let_go - 8.0, 1.0), g, SUP_PINCH, "keep it steady, close to the bed", ROT_SUP))
    out += [_w(1.0, g, SUP_PINCH, "keep it steady", ROT_SUP),
            _w(0.5, g, SUP_PINCH, "settle", ROT_SUP),
            _w(1.0, g, OPEN_C, "let go", ROT_SUP),
            _w(1.5, above, OPEN_C, "rise along the fingers", ROT_SUP),
            _w(2.0, above + np.array([-0.03, 0.04, 0.04]), OPEN_C, "stay clear", ROT_SUP)]
    return out


def _stay(arm: str) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(60.0, tuple(float(v) for v in pos), OPEN_C, "stay put", rot)]


def demo():
    right = _right()
    if ONE:
        return {"left": _stay("left"), "right": right}
    t_out = sum(w.duration for w in right[:-2])  # the right gripper has let go and backed out
    return {"left": _left(t_out + 0.5), "right": right}
