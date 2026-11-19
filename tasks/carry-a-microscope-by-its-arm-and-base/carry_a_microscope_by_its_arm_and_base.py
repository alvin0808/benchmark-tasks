"""On the station table: a student compound microscope (monocular, LED, 1.8 kg, 30 cm tall: the size and weight of
a common school model such as the AmScope M150C, about 4 lb / 1.8 kg with a 3-1/2 in square stage) stands on a dark
grey rubber mat on the left, its arm toward the robot. A blue bench mat lies on the right. Task (instruction): move
the microscope onto the blue bench mat, standing upright with its arm toward you as it stands now, and let go of it.
The instruction does not say how.

How it is done (school and lab rules): "Carry the scope with both hands, one on the arm, one on the base" (Nassau
Community College, Microscope Use); "Remove the microscope from the cabinet by grasping the handle in the arm of the
scope with one hand and placing the other hand below the base" (University of Wisconsin, Botany 130); "use one hand
to hold the arm and the other to support the bottom of the base firmly" (Nikon Eclipse Ei guide, Carrying); "carry it
from the base and support arm ... if picked up by stage or eyepiece holder it can become misaligned" (Technology
Networks, How to care for your microscope); "never hold the stage, observation tube, or focusing knob when moving the
frame" (Evident/Olympus, How to properly and safely transport your microscope).

The hand skill: the 1.8 kg microscope is too heavy and too front-heavy for one gripper on its arm: the grip turns on
the arm and the microscope tips forward onto the front of its base and drags (the arm alone cannot lift it). The
other gripper has to get under the base, which stands flat on the mat. The demo's way: the arm hand rocks the
microscope back about 10 deg onto the back edge of its base, so the front half of the base rises 2 cm; the base hand,
pointing in from the right side, slides its lower finger under the raised side and pinches the base's edge; then the
two lift it level together (the arm behind its centre of mass, the base hand in front of it), carry it 18 cm to the
right and set it down square on the bench mat.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- start mat: dark grey rubber, 24 x 16 x 0.3 cm, fixed, centre (0.300, 0.075).
- bench mat: blue rubber, 24 x 16 x 0.3 cm, fixed, centre (0.300, -0.105).
- microscope: loose, 1.80 kg, base centre at (0.300, 0.075) on the start mat, arm toward -x (the robot). Base: white
  cast metal, 18 x 13 x 3.0 cm on four rubber feet (4 mm), 1.00 kg. Arm: a white C-shaped casting, 4.0 cm deep and
  3.6 cm wide, rising from the back of the base to 20 cm, then reaching 8 cm forward to carry the head; 0.32 kg. Focus
  knobs (coarse 38 mm, fine 24 mm) on both sides of the arm at 7.5 cm; 0.06 kg. Stage: black, 8.9 x 8.9 x 1.0 cm at
  12.1-13.1 cm with two slide clips, a slide, and the condenser under it; 0.20 kg. Head: white block on the arm's top
  with a revolving nosepiece and three objectives under it, and the eyepiece tube leaning 45 deg back toward the user,
  its top 30.5 cm up; 0.22 kg. Each part is its own rigid body, held to the arm (the arm to the base) by a joint
  locked at its start angle, so that the simulation can tell which part a gripper touches.

Score (0 for the rest of the run once a rule is broken):
- 0: any link of either arm touches the stage (with its clips and condenser), the head (with the nosepiece,
  objectives, eyepiece tube and eyepiece) or a focus knob with 0.5 N or more; the microscope tilts more than 15 deg
  from upright at any time; it drops (its base falls faster than 0.4 m/s, as after a free fall of 8 mm); its base's
  lowest corner is more than 3 cm above the table for 0.25 s without one gripper on the arm and the other on the base,
  each pressing with 1 N or more (carried with one hand).
- 0.25: carried properly: its base's lowest corner 3 cm or more above the table with one gripper on the arm and the
  other on the base.
- 0.5: then standing upright (within 5 deg) on the bench mat: base resting (lowest corner within 5 mm of the mat's
  top) with all four corners of the base over the mat, and turned no more than 15 deg from how it stood.
- 1.0 (success, held 1 s): that, released (no link of either arm touching any part of it, 0.1 N) and still (base
  moving under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, allow_matrix_grips, as_numpy  # noqa: E402
from bodykit import capsule, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- the layout ----------------------------------------------------------------------------------------------------
MAT_T = 0.003
MAT_L, MAT_W = 0.24, 0.16  # along x, along y
START = np.array([0.300, 0.075])
GOAL = np.array([0.300, -0.105])
BASE_L, BASE_W, BASE_T, FOOT = 0.180, 0.130, 0.030, 0.004
ARM_X0, ARM_X1, ARM_W = -0.075, -0.035, 0.036  # the arm post (microscope frame: x from the arm to the stage)
ARM_TOP = 0.200
STAGE_Z = 0.121
MASSES = dict(base=1.00, arm=0.32, knobs=0.06, stage=0.20, head=0.22)

CREAM, BLACK, SILVER, RUBBER, GREY, BLUE, GLASS = (
    (0.94, 0.94, 0.93), (0.10, 0.10, 0.11), (0.75, 0.76, 0.78), (0.06, 0.06, 0.06), (0.42, 0.43, 0.45),
    (0.20, 0.36, 0.62), (0.80, 0.88, 0.92))


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _mat(name: str, c, color) -> Body:
    return Body.build(name, pos=(float(c[0]), float(c[1]), TOP + MAT_T / 2), mass=0.2, fixed=True, color=color,
                      friction=(0.9, 0.8), shapes=[BoxShape((0.0, 0.0, 0.0), (MAT_L, MAT_W, MAT_T), color)])


def _scope() -> Linkage:
    """The microscope, a floating articulation: the base, and the arm on it; the knobs, stage and head on the arm.
    Shape centres are relative to the linkage origin, the middle of the base's underside (feet bottoms)."""
    fx, fy = BASE_L / 2 - 0.015, BASE_W / 2 - 0.015
    base = [CylShape((sx * fx, sy * fy, FOOT / 2), 0.008, FOOT, RUBBER) for sx in (-1, 1) for sy in (-1, 1)]
    base += [BoxShape((0.0, 0.0, FOOT + BASE_T / 2), (BASE_L, BASE_W, BASE_T), CREAM),
             CylShape((0.025, 0.0, FOOT + BASE_T + 0.0006), 0.013, 0.001, GLASS, collide=False),  # the LED window
             BoxShape((0.070, 0.0, FOOT + BASE_T / 2), (0.041, 0.040, 0.012), GREY, collide=False)]  # switch plate
    zb = FOOT + BASE_T
    ax = (ARM_X0 + ARM_X1) / 2
    arm = [BoxShape((ax, 0.0, (zb + ARM_TOP) / 2), (ARM_X1 - ARM_X0, ARM_W, ARM_TOP - zb), CREAM),
           BoxShape((-0.035, 0.0, ARM_TOP + 0.012), (0.080, ARM_W, 0.024), CREAM),  # the top limb, to the head
           BoxShape((-0.031, 0.0, 0.100), (0.010, 0.028, 0.080), SILVER)]  # the focus rack on the arm's front
    kz, kx = 0.075, -0.052
    knobs = []
    for s in (-1, 1):
        knobs += [CylShape((kx, s * (ARM_W / 2 + 0.007), kz), 0.019, 0.014, BLACK, axis="Y"),
                  CylShape((kx, s * (ARM_W / 2 + 0.019), kz), 0.012, 0.010, BLACK, axis="Y")]
    stage = [BoxShape((0.025, 0.0, STAGE_Z + 0.005), (0.089, 0.089, 0.010), BLACK),
             BoxShape((-0.0245, 0.0, STAGE_Z - 0.008), (0.010, 0.050, 0.030), BLACK),  # its bracket on the rack
             CylShape((0.025, 0.0, STAGE_Z - 0.016), 0.018, 0.030, BLACK),  # the condenser under it
             BoxShape((0.025, 0.0, STAGE_Z + 0.0105), (0.025, 0.075, 0.001), GLASS, collide=False),  # a slide
             BoxShape((0.010, 0.030, STAGE_Z + 0.0115), (0.040, 0.006, 0.002), SILVER, collide=False),  # clips
             BoxShape((0.010, -0.030, STAGE_Z + 0.0115), (0.040, 0.006, 0.002), SILVER, collide=False)]
    hz = ARM_TOP + 0.012
    head = [BoxShape((0.025, 0.0, hz), (0.060, 0.050, 0.036), CREAM),
            CylShape((0.025, 0.0, hz - 0.026), 0.028, 0.016, SILVER)]  # the revolving nosepiece
    for a in (90.0, 210.0, 330.0):
        r = math.radians(a)
        head.append(CylShape((0.025 + 0.014 * math.cos(r), 0.014 * math.sin(r), hz - 0.050), 0.009, 0.032, SILVER))
    t0, t1 = np.array([0.012, 0.0, hz + 0.016]), np.array([-0.035, 0.0, hz + 0.063])
    head += [capsule(_v(t0), _v(t1), 0.016, CREAM),  # the eyepiece tube, leaning back toward the user
             capsule(_v(t1), _v(t1 + (t1 - t0) * 0.30), 0.0165, BLACK)]  # the eyepiece
    o = np.array([START[0], START[1], TOP + MAT_T + 0.001])
    at = lambda p: _v(o + np.asarray(p))  # noqa: E731
    lk = dict(limits=(-0.05, 0.05), report_contacts=True)  # locked at the start angle; contacts reported
    links = [Link("arm", tuple(arm), MASSES["arm"], at((ax, 0.0, zb + 0.01)), "Y", None, **lk),
             Link("knobs", tuple(knobs), MASSES["knobs"], at((kx, 0.0, kz)), "Y", "arm", **lk),
             Link("stage", tuple(stage), MASSES["stage"], at((-0.032, 0.0, STAGE_Z)), "Y", "arm", **lk),
             Link("head", tuple(head), MASSES["head"], at((0.0, 0.0, hz)), "Y", "arm", **lk)]
    return Linkage.build("scope", pos=_v(o), mass=MASSES["base"], color=CREAM, floating=True, friction=(0.8, 0.7),
                         shapes=base, links=links, contact_offset=0.002, solver_iterations=32,
                         report_contacts=True)


START_MAT = _mat("start_mat", START, (0.22, 0.23, 0.25))
GOAL_MAT = _mat("bench_mat", GOAL, BLUE)
SCOPE = _scope()
OBJECTS = [START_MAT, GOAL_MAT, SCOPE]
PARTS = ("base", "arm", "knobs", "stage", "head")
FORBIDDEN = ("stage", "head", "knobs")

# -- tracking and scoring -------------------------------------------------------------------------------------------
TOUCH_N = 0.5  # a gripper touching a forbidden part with this much force breaks the rule
HOLD_N = 1.0  # a gripper holds a part when it presses on it with this much force
TILT_MAX = 15.0
TILT_OK = 5.0
LIFT = 0.03  # base's lowest point this far above the table: lifted
ONE_HAND_STEPS = 5  # 0.25 s at 20 Hz
REST = 0.005
DROP_V = 0.4  # m/s: the base falling faster than this (a free fall of 8 mm) is a drop
YAW_OK = 15.0
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("SCOPE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("SCOPE_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
_CV: dict = {}
ARM_PRIMS = ("LeftArm", "RightArm")
_CORNERS = np.array([[sx * BASE_L / 2, sy * BASE_W / 2, 0.0] for sx in (-1, 1) for sy in (-1, 1)])


@on_reset
def _reset() -> None:
    _EP.clear()
    _EP.update(tick=0, step=0, counted=-1, failed="", carried=False, one_hand=0, recent=[], forces={}, tilt=0.0,
               max_tilt=0.0, low=0.0, yaw=0.0, on_goal=False, held_by="", tcp={})
    _CV.clear()


_reset()


def _contacts(base) -> dict:
    """{part: (left arm's largest contact force, right arm's)} in N, from PhysX's pair-wise contact forces."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    if _CV.get("sim") is not sim:
        from yam_arena.robot import asset_link_paths

        env = base.scene.env_prim_paths[0]
        links = [f"{env}/{arm}{path}" for arm in ARM_PRIMS for path in asset_link_paths().values()]
        parts = [f"{env}/scope/{p}" for p in PARTS]
        view = sim.create_rigid_contact_view(parts, filter_patterns=[links] * len(parts),
                                             max_contact_data_count=16 * len(links) * len(parts))
        _CV.update(sim=sim, view=view, dt=float(base.physics_dt), nl=len(links))
    m = _CV["view"].get_contact_force_matrix(dt=_CV["dt"])
    a = np.asarray(m.numpy() if hasattr(m, "numpy") else as_numpy(m), float).reshape(len(PARTS), _CV["nl"], 3)
    f = np.linalg.norm(a, axis=-1)
    h = _CV["nl"] // 2
    return {p: (float(f[i, :h].max()), float(f[i, h:].max())) for i, p in enumerate(PARTS)}


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("SCOPE_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        _EP["tcp"][side] = np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float)
    sc = base.scene["scope"]
    if "base" not in _IDS:
        _IDS["base"] = list(sc.body_names).index("base")
    bp = as_numpy(sc.data.body_link_pose_w)[0][_IDS["base"]]
    p, r = bp[:3] - origin, rotation(bp[3:7])
    corners = p + _CORNERS @ r.T
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    _EP["lean"] = (math.degrees(math.asin(max(-1.0, min(1.0, float(r[0, 2]))))),
                   math.degrees(math.asin(max(-1.0, min(1.0, float(r[1, 2]))))))  # its top toward +x, toward +y
    low = float(corners[:, 2].min()) - TOP
    prev = _EP.get("p")
    fall = 0.0 if prev is None else (float(prev[2]) - float(p[2])) / float(getattr(base, "step_dt", 0.05))
    f = _contacts(base)
    arm_l, arm_r = f["arm"]
    base_l, base_r = f["base"]
    proper = (arm_l >= HOLD_N and base_r >= HOLD_N) or (arm_r >= HOLD_N and base_l >= HOLD_N)
    if not _EP["failed"]:
        for part in FORBIDDEN:
            if max(f[part]) >= TOUCH_N:
                _EP["failed"] = f"a gripper touched the {part} with {max(f[part]):.1f} N"
        if fall > DROP_V:
            _EP["failed"] = f"the microscope dropped ({fall:.2f} m/s)"
        if tilt > TILT_MAX:
            _EP["failed"] = f"the microscope tilted {tilt:.0f} deg"
        if low > LIFT and not proper:
            _EP["one_hand"] += 1
            if _EP["one_hand"] >= ONE_HAND_STEPS:
                _EP["failed"] = "the microscope was off the table without one hand on the arm and one on the base"
        else:
            _EP["one_hand"] = 0
    if low > LIFT and proper:
        _EP["carried"] = True
    loc = (corners[:, :2] - GOAL)
    on_goal = bool(np.all(np.abs(loc[:, 0]) <= MAT_L / 2) and np.all(np.abs(loc[:, 1]) <= MAT_W / 2))
    _EP.update(forces=f, fall=fall, max_fall=max(_EP.get("max_fall", 0.0), fall), tilt=tilt,
               max_tilt=max(_EP["max_tilt"], tilt), low=low, yaw=yaw, p=p, on_goal=on_goal,
               proper=proper, touching=max(max(v) for v in f.values()))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "p" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["p"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(q - recent[-1]).max() * 1000 < STILL_MM for q in recent)
    placed = bool(_EP["carried"] and _EP["on_goal"] and _EP["tilt"] <= TILT_OK
                  and _EP["low"] <= MAT_T + REST and abs(_EP["yaw"]) <= YAW_OK)
    released = _EP.get("touching", 1.0) < 0.1
    return dict(still=still, placed=placed, released=released, done=bool(placed and released and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    p = _EP.get("p", np.zeros(3))
    fs = " ".join(f"{k}={v[0]:.1f}/{v[1]:.1f}" for k, v in _EP["forces"].items())
    lean = _EP.get("lean", (0.0, 0.0))
    print(f"DBG step={_EP['step']} tilt={_EP['tilt']:.1f} lean=({lean[0]:+.1f},{lean[1]:+.1f}) "
          f"max={_EP['max_tilt']:.1f} low={_EP['low'] * 1000:.1f}mm fall={_EP.get('max_fall', 0.0):.3f} "
          f"yaw={_EP['yaw']:.1f} p=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f}) F[L/R] {fs} "
          f"proper={int(_EP.get('proper', 0))} "
          f"carried={int(_EP['carried'])} goal={int(_EP['on_goal'])} placed={int(t['placed'])} rel={int(t['released'])} "
          f"still={int(t['still'])} failed={_EP['failed'] or '-'} | "
          + " ".join(f"{k}=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if _EP["failed"]:
        return 0.0
    if t["done"]:
        return 1.0
    if t["placed"]:
        return 0.5
    return 0.25 if _EP["carried"] else 0.0


def success(objects) -> bool:
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper comes down behind the microscope, level and pointing forward,
# and closes on the arm 15 cm up from either side. It rocks the microscope back onto the back edge of its base (told
# 12 deg, it reaches about 10: the arm gives under the load), so the front half of the base rises about 2 cm. The
# right gripper, pointing in from the right side and tilted 75 deg down from vertical (fingers closing nearly
# vertically), comes level with the raised side 3 cm in front of the base's centre, slides its lower finger under it
# and pinches the base's edge. Together they lift it level (the right hand turned 9 deg about x and y against the
# weight off its side; both commanded 3 cm higher for the arms' sag), carry it 18 cm to the right and set it down on
# the bench mat; the right gripper lets go and draws out sideways, then the left lets go of the arm and backs away.
# SCOPE_MODE=one runs it the wrong way: the left gripper alone takes the arm and tries to lift and carry it.
OPEN_C = -0.047
MODE = os.environ.get("SCOPE_MODE", "demo")  # demo | one (wrong: the arm hand alone) | lift (arm hand lifts, waits)
TIP_CMD = float(os.environ.get("SCOPE_TIP_CMD", "12"))  # deg the arm hand is told to rock it back onto its back edge
TIP_EXP = float(os.environ.get("SCOPE_TIP_EXP", "10"))  # deg it is expected to reach (the arm gives under the load)
ARM_GRIP = float(os.environ.get("SCOPE_ARM_GRIP", "-0.0145"))  # closed on the 36 mm arm
BASE_GRIP = float(os.environ.get("SCOPE_BASE_GRIP", "-0.011"))  # closed on the 30 mm base
GRIP_Z = float(os.environ.get("SCOPE_GRIP_Z", "0.150"))  # the arm hand's TCP height on the arm
GRIP_X = float(os.environ.get("SCOPE_GRIP_X", "-0.066"))  # and how far along the arm (microscope frame)
PINCH_X = float(os.environ.get("SCOPE_PINCH_X", "0.030"))  # where along the base's side the base hand pinches
TILT_B = math.radians(float(os.environ.get("SCOPE_TILT_B", "75")))  # the base hand's tilt from straight down
IN_B = float(os.environ.get("SCOPE_IN_B", "0.000"))  # the base hand's TCP this far inside the base's side
SAG = float(os.environ.get("SCOPE_SAG", "0.030"))  # the arm hand sags this much under the load: commanded higher
SAG_B = float(os.environ.get("SCOPE_SAG_B", "0.030"))  # the base hand sags this much
CARRY_UP = float(os.environ.get("SCOPE_CARRY_UP", "0.060"))  # carried with its base this high
CARRY_PITCH = math.radians(float(os.environ.get("SCOPE_CARRY_PITCH", "-9")))  # the base hand turned about y and x
CARRY_ROLL = math.radians(float(os.environ.get("SCOPE_CARRY_ROLL", "9")))  # while carrying, to hold it level
DOWN = grip_rotation(0.0)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


FWD = _ry(-math.pi / 2) @ DOWN  # pointing forward (+x), level, fingers closing along y, pads upright
SIDE_R = _rx(TILT_B) @ DOWN  # the right arm pointing in (+y) and down, fingers closing near-vertically


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    b = station.BASES[arm]
    j1 = math.atan2(float(pos[1]) - b[1], float(pos[0]) - b[0])
    for sd in itertools.product((j1, 0.0), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2),
                                (-1.8, -0.9, 0.0, 0.9, 1.8)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=b, iters=150)
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


def _at(c, local) -> np.ndarray:
    """World point of a microscope-frame point with its base centred at ``c`` (x, y) on a mat."""
    return np.array([c[0] + local[0], c[1] + local[1], TOP + MAT_T + local[2]])


UP = np.array([0.0, 0.0, 1.0])


def _tipped(c, local, deg: float) -> np.ndarray:
    """World point of a microscope-frame point once the microscope, base centred at ``c``, is rocked back ``deg`` onto
    the back edge of its base (its front rising)."""
    pivot = np.array([-BASE_L / 2, 0.0, 0.0])
    return _at(c, pivot + _ry(-math.radians(deg)) @ (np.asarray(local, float) - pivot))


def demo():
    up = UP
    S, G = START, GOAL
    d = np.array([G[0] - S[0], G[1] - S[1], 0.0])
    # left: the arm hand, level from behind, fingers either side of the arm
    a_loc = np.array([GRIP_X, 0.0, GRIP_Z])
    a_s = _at(S, a_loc)
    back = a_s - np.array([0.05, 0.0, 0.0])
    left = _lead_in("left", np.array([0.22, 0.22, TOP + 0.42]), FWD, OPEN_C, "up, clear of the microscope")
    left += [_w(2.0, back + up * 0.17, OPEN_C, "high behind the microscope", FWD),
             _w(2.0, back, OPEN_C, "down behind its arm", FWD),
             _w(2.0, a_s, OPEN_C, "fingers either side of the arm", FWD),
             _w(1.0, a_s, ARM_GRIP, "take the arm", FWD),
             _w(1.0, a_s, ARM_GRIP, "settle", FWD)]
    if MODE == "one":
        g = _at(G, a_loc)
        left += [_w(3.0, a_s + up * (CARRY_UP + SAG), ARM_GRIP, "lift it by the arm alone", FWD),
                 _w(5.0, g + up * (CARRY_UP + SAG), ARM_GRIP, "carry it", FWD),
                 _w(3.0, g + up * 0.002, ARM_GRIP, "set it down", FWD),
                 _w(1.0, g, OPEN_C, "let go", FWD),
                 _w(10.0, g - np.array([0.06, 0.0, 0.0]), OPEN_C, "back away", FWD)]
        return {"left": left, "right": _stay("right")}
    n = 8
    for k in range(1, n + 1):  # rock it back onto the back edge of its base, so its front half rises
        a = TIP_CMD * k / n
        left.append(_w(2.5 / n, _tipped(S, a_loc, a), ARM_GRIP, "rock it back on its base's back edge", FWD))
    tip_a = _tipped(S, a_loc, TIP_CMD)
    tip_r = FWD  # the hand stays level (the arm turns a little between the fingers)
    left.append(_w(1.0, tip_a, ARM_GRIP, "hold it rocked back", tip_r))
    if MODE == "lift":
        left.append(_w(15.0, tip_a, ARM_GRIP, "hold it rocked back", tip_r))
        return {"left": left, "right": _stay("right")}
    # right: the base hand, pointing in from the right side; its lower finger slides under the raised base
    b_loc = np.array([PINCH_X, -BASE_W / 2 + IN_B, FOOT + BASE_T / 2])
    b_rot = _ry(-math.radians(TIP_EXP)) @ SIDE_R
    b_in = _tipped(S, b_loc, TIP_EXP)
    b_out = b_in - np.array([0.0, 0.07, 0.0])
    right = _lead_in("right", b_out + up * 0.05, b_rot, OPEN_C, "beside the base, open", seconds=4.0)
    pre = sum(w.duration for w in left) - 4.0
    if pre > 0:
        right = _stay("right", pre) + right
    right += [_w(1.5, b_out, OPEN_C, "level with the raised base's side", b_rot),
              _w(2.0, b_in, OPEN_C, "slide the lower finger under the base", b_rot),
              _w(1.0, b_in, BASE_GRIP, "pinch the base's side", b_rot),
              _w(1.0, b_in, BASE_GRIP, "settle", b_rot)]
    left.append(_w(sum(w.duration for w in right) - sum(w.duration for w in left), tip_a, ARM_GRIP, "hold", tip_r))
    # both: level it and lift it, carry it over the bench mat, set it down square, let go
    b_s = _at(S, b_loc)
    lift, lift_b = up * (CARRY_UP + SAG), up * (CARRY_UP + SAG_B)
    down = up * float(os.environ.get("SCOPE_DOWN", "0.0"))
    left += [_w(3.0, a_s + lift, ARM_GRIP, "lift it level together", FWD),
             _w(5.0, a_s + lift + d, ARM_GRIP, "carry it over the bench mat", FWD),
             _w(3.0, a_s + d + down, ARM_GRIP, "set it down", FWD),
             _w(1.0, a_s + d + down, ARM_GRIP, "settle", FWD)]
    b_c = _rx(CARRY_ROLL) @ _ry(CARRY_PITCH) @ SIDE_R
    right += [_w(3.0, b_s + lift_b, BASE_GRIP, "lift it level together", b_c),
              _w(5.0, b_s + lift_b + d, BASE_GRIP, "carry it over the bench mat", b_c),
              _w(3.0, b_s + d + down, BASE_GRIP, "set it down", SIDE_R),
              _w(1.0, b_s + d + down, BASE_GRIP, "settle", SIDE_R),
              _w(1.0, b_s + d + down, OPEN_C, "let go of the base", SIDE_R),
              _w(2.0, b_s + d - np.array([0.0, 0.07, 0.0]) + up * 0.005, OPEN_C, "draw out sideways", SIDE_R),
              _w(12.0, b_s + d - np.array([0.0, 0.08, 0.0]) + up * 0.08, OPEN_C, "stay clear", SIDE_R)]
    left += [_w(3.0, a_s + d + down, ARM_GRIP, "hold while the base hand draws out", FWD),
             _w(1.0, a_s + d + down, OPEN_C, "let go of the arm", FWD),
             _w(2.0, a_s + d - np.array([0.05, 0.0, 0.0]), OPEN_C, "back away from the arm", FWD),
             _w(10.0, a_s + d - np.array([0.05, 0.0, -0.06]), OPEN_C, "stay clear", FWD)]
    return {"left": left, "right": right}
