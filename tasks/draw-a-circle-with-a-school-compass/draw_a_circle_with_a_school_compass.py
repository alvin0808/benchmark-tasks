"""On the station table: an A4 writing pad lies loose on the desk with a centre mark pencilled and pricked on its top
sheet; a metal school compass holding a pencil stands on it, already set to a 40 mm radius, its steel point in the
pricked mark and the pencil's point on the paper. Task (instruction): draw the circle round the mark with the compass,
then lay the compass down on the desk. The instruction does not say how.

How it is done (bhanzu.com, "Compass drawing"): "hold only the handle (knurled knob) at the top and spin it between
finger and thumb, letting the legs hang and trace"; "let the pencil glide with almost no downward force"; "Don't do
this: push the pencil into the paper to get a heavier mark, which flexes the legs and widens the radius".

The hand skill: a gripper cannot roll the 8 mm handle, so it carries it round a 20 mm circle about the point while
turning with it, holding it lightly (the compass rocks between the fingers, both points stay down) and pressing
lightly (the pencil's point breaks above 5 N; the friction hinge opens under a hard push and the circle spirals; a
sideways shove or a lift skids the point out of its mark). A wrist turns only about 240 deg, so part-way round the hand
opens round the handle, turns back and takes it again while the 32 g compass stands on its two points.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- writing pad: loose, A4 (21.0 x 29.7 cm, long side along y), 6 mm thick, 0.30 kg; middle at (0.215, -0.070). The
  centre mark at (0.215, -0.100): a pencilled cross and a pricked hole 1.0 mm square, 1.5 mm deep.
- compass: loose. Head (14 x 9 x 12 mm) with an 8 mm knurled handle 22 mm long on top; the needle leg fixed to the
  head; the pencil leg on a friction hinge (0.10 N m Coulomb friction) about the axis across the legs (y); hinge to
  points 92 mm; set to 40 mm, the pencil's point 40 mm along +x from the steel point. 20 g head and needle leg, 12 g
  pencil leg with its pencil stub (7 mm, 55 mm long).
- the pencil line: 24 dark marks (visual only), each laid on the paper once its 15 deg of the circle has been drawn.

Score (0 for the rest of the run once a rule is broken):
- 0: the steel point skidded (more than 1 mm from the mark while its tip is in the paper or less than 0.3 mm above
  it); the pencil pressed on the paper with more than 5 N (its point breaks); a pencil line more than 3 mm off the
  40 mm circle (a spiral or a stray mark).
- 0.8 x (degrees of the circle drawn / 360): counted in 1-degree steps along the path of the pencil's point while it
  presses on the paper with the marking force, 0.3 N or more (read from the contact force every control step), at a
  radius within 1.5 mm of 40 mm (each degree once; a dot where it stood does not count). The pencil's point meets the
  pad through a compliant contact (3 N/mm, 10 N s/m: the give of the paper stack under it), so it presses steadily.
- 1.0 (success, held 1 s): at least 355 deg drawn, the compass lying on its side (hinge axis within 30 deg of
  vertical, head within 3 cm of the desk), every finger point 2 cm or more from it, and it still (head under 1 mm in
  1 s).
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
from bodykit import as_numpy, capsule, on_reset, on_step, rotation, segment  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# -- the layout ----------------------------------------------------------------------------------------------------
PAD_T = 0.006  # 50 sheets of 80 g/m2 paper (5 mm) on a 1 mm card back
PAD_W, PAD_L = 0.210, 0.297  # A4: short side along x, long side along y
CENTRE = np.array([float(os.environ.get("COMPASS_CX", "0.215")), float(os.environ.get("COMPASS_CY", "-0.100"))])
PAD_C = (CENTRE[0] + 0.0, CENTRE[1] + 0.03)  # the pad's middle (the mark is 3 cm right of it)
PAPER = TOP + PAD_T  # the top sheet
HOLE = float(os.environ.get("COMPASS_HOLE", "1.0")) * MM
HOLE_D = float(os.environ.get("COMPASS_HOLE_D", "1.5")) * MM  # the pricked centre mark: square side, depth

RADIUS = 0.040
LEG = 0.092  # hinge axis to the tips
HALF = math.asin(RADIUS / 2 / LEG)  # each leg's angle from vertical
HINGE_H = LEG * math.cos(HALF)  # the hinge axis above the paper
PSI0 = math.radians(float(os.environ.get("COMPASS_PSI0", "0")))  # the direction from the point to the pencil, at start
# The pencil's point on the pad: a compliant contact (the give of the paper stack under the graphite point), so the
# point presses on the paper steadily rather than bouncing on a rigid contact.
LEAD_K = float(os.environ.get("COMPASS_LEAD_K", "3000"))  # N/m
LEAD_C = float(os.environ.get("COMPASS_LEAD_C", "10"))  # N s/m
HINGE_NM = float(os.environ.get("COMPASS_HINGE", "0.10"))  # the hinge's friction torque (N m)
HANDLE_R, HANDLE_L = 0.004, 0.022  # the knurled handle
HEAD_H = 0.012  # the head block (the hinge in its middle)
HANDLE_Z0 = HINGE_H + HEAD_H / 2  # handle bottom above the paper
HANDLE_TOP = HANDLE_Z0 + HANDLE_L

PAPER_WHITE, CARD, STEEL, DARK, PENCIL_Y, WOOD, GRAPHITE, MARK = (
    (0.95, 0.95, 0.93), (0.55, 0.45, 0.35), (0.72, 0.74, 0.78), (0.18, 0.18, 0.2), (0.95, 0.75, 0.15),
    (0.86, 0.72, 0.52), (0.15, 0.15, 0.15), (0.1, 0.1, 0.1))

D0 = np.array([math.cos(PSI0), math.sin(PSI0), 0.0])  # point -> pencil, at start
N0 = np.cross([0.0, 0.0, 1.0], D0)  # across the legs (the hinge axis)
HINGE_AXIS = "Y" if abs(N0[1]) > 0.5 else "X"
POINT0 = np.array([CENTRE[0], CENTRE[1], PAPER])  # the needle's tip at the paper surface
MARK_REL = np.array([CENTRE[0] - PAD_C[0], CENTRE[1] - PAD_C[1], PAD_T / 2])  # the mark from the pad's body origin
HINGE0 = POINT0 + D0 * RADIUS / 2 + np.array([0.0, 0.0, HINGE_H])
PENCIL0 = POINT0 + D0 * RADIUS  # the pencil's tip


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _compass() -> Linkage:
    o = HINGE0  # the linkage origin: the hinge
    up_n = (POINT0 - o) / np.linalg.norm(POINT0 - o)  # hinge -> point (unit)
    up_p = (PENCIL0 - o) / np.linalg.norm(PENCIL0 - o)  # hinge -> pencil tip (unit)
    rel = lambda p: _v(np.asarray(p) - o)  # noqa: E731
    yaw = PSI0
    base = [BoxShape((0.0, 0.0, 0.0), (0.014, 0.009, HEAD_H), STEEL, yaw=yaw),
            CylShape((0.0, 0.0, HEAD_H / 2 + HANDLE_L / 2), HANDLE_R, HANDLE_L, DARK),
            # needle leg: a bar from the head to the needle holder, then the needle shank and its point
            segment(rel(o), rel(POINT0 - up_n * 0.016), 0.004, STEEL),
            segment(rel(POINT0 - up_n * 0.016), rel(POINT0 - up_n * 0.0005), 0.0012, STEEL),
            segment(rel(POINT0 - up_n * 0.0005), rel(POINT0 + np.array([0.0, 0.0, -HOLE_D + 0.0001])), 0.0007,
                    STEEL)]
    # pencil leg: a bar from the head to the ring, a pencil stub clamped in the ring, sharpened, its lead on the paper
    tip = PENCIL0
    pencil = [segment(rel(o), rel(tip - up_p * 0.045), 0.004, STEEL),
              segment(rel(tip - up_p * 0.040), rel(tip - up_p * 0.034), 0.0095, STEEL),
              capsule(rel(tip - up_p * 0.052), rel(tip - up_p * 0.018), 0.0035, PENCIL_Y),
              capsule(rel(tip - up_p * 0.016), rel(tip - up_p * 0.006), 0.0018, WOOD),
              capsule(rel(tip - up_p * 0.005), rel(tip - up_p * 0.0012), 0.0006, GRAPHITE),
              SphereShape(rel(tip + np.array([0.0, 0.0, 0.0006])), 0.0006, GRAPHITE)]
    links = [Link("pencil_leg", tuple(pencil), mass=0.012, anchor=_v(o), axis=HINGE_AXIS, friction=HINGE_NM,
                  damping=0.0005, limits=(-25.0, 25.0), report_contacts=True,
                  compliant=(LEAD_K, LEAD_C))]
    return Linkage.build("compass", pos=_v(o + np.array([0.0, 0.0, 0.0003])), mass=0.020, color=STEEL, floating=True,
                         friction=(0.5, 0.4), shapes=base, links=links, contact_offset=0.001, solver_iterations=48)


def _pad() -> Body:
    """The pad: the stack below the pricked top layer, and the top layer (1 mm) in four pieces round the hole."""
    cx, cy = PAD_C
    mx, my = CENTRE[0] - cx, CENTRE[1] - cy  # the mark, from the pad's middle
    zb, zt = -PAD_T / 2 + (PAD_T - HOLE_D) / 2, PAD_T / 2 - HOLE_D / 2
    hb = PAD_T - HOLE_D
    h = HOLE / 2
    x0, x1, y0, y1 = -PAD_W / 2, PAD_W / 2, -PAD_L / 2, PAD_L / 2
    sh = [BoxShape((0.0, 0.0, zb), (PAD_W, PAD_L, hb), PAPER_WHITE),
          BoxShape(((x0 + mx - h) / 2, 0.0, zt), (mx - h - x0, PAD_L, HOLE_D), PAPER_WHITE),
          BoxShape(((x1 + mx + h) / 2, 0.0, zt), (x1 - mx - h, PAD_L, HOLE_D), PAPER_WHITE),
          BoxShape((mx, (y0 + my - h) / 2, zt), (HOLE, my - h - y0, HOLE_D), PAPER_WHITE),
          BoxShape((mx, (y1 + my + h) / 2, zt), (HOLE, y1 - my - h, HOLE_D), PAPER_WHITE),
          # the card back's edge, and the pencilled centre mark (a small cross)
          BoxShape((0.0, 0.0, -PAD_T / 2 + 0.0005), (PAD_W + 0.001, PAD_L + 0.001, 0.001), CARD),
          BoxShape((mx, my, PAD_T / 2 + 0.0001), (0.006, 0.0006, 0.0002), MARK, collide=False),
          BoxShape((mx, my, PAD_T / 2 + 0.0001), (0.0006, 0.006, 0.0002), MARK, collide=False)]
    return Body.build("pad", pos=(cx, cy, TOP + PAD_T / 2 + 0.0005), mass=0.30, color=PAPER_WHITE,
                      friction=(0.45, 0.4), shapes=sh, contact_offset=0.001)


PAD = _pad()
COMPASS = _compass()
# The pencil line, shown (visual only, nothing collides with it): one dark mark per 15 deg of the circle, kept below
# the table until the pencil's point has drawn the whole of its 15 deg, then laid on the paper there.
LINE_N = 24
LINE_W = 0.0012  # drawn wider than a real pencil line (about 0.5 mm) so the cameras show it
HIDE = np.array([0.22, -0.10, TOP - 0.30])
LINES = [Body.build(f"line_{k}", pos=_v(HIDE + np.array([0.0, 0.0, -0.01 * k])), mass=0.001, color=MARK,
                    gravity=False, damping=(5.0, 5.0),
                    shapes=[BoxShape((0.0, 0.0, 0.0), (2 * RADIUS * math.sin(math.pi / LINE_N) + 0.0005, LINE_W,
                                                       0.0001), MARK, collide=False)]) for k in range(LINE_N)]
OBJECTS = [PAD, COMPASS] + LINES

# -- tracking and scoring -------------------------------------------------------------------------------------------
TOL = 1.5 * MM  # a line within 1.5 mm of the 40 mm radius counts
STRAY = 3.0 * MM  # a line further off than this spoils the drawing
SKID = 1.0 * MM  # the point more than this from the mark ...
ON_PAPER = 0.3 * MM  # ... while its tip is in the paper or less than this above it: it skidded
MARK_N = float(os.environ.get("COMPASS_MARK_N", "0.3"))  # N: the pencil marks the paper when it presses this hard
F_SNAP = 5.0  # N: pressing harder than this breaks the pencil's point
SECTORS = 360  # 1-degree steps of the circle
FULL_DEG = 355  # the whole circle: at least this many degrees of it drawn
CLEAR = 0.02
LIE_DEG = 30.0
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("COMPASS_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("COMPASS_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
_CV: dict = {}

# points on the compass (relative to the hinge at the start, world-aligned) for the finger-clearance check
_UPN = (POINT0 - HINGE0) / np.linalg.norm(POINT0 - HINGE0)
_UPP = (PENCIL0 - HINGE0) / np.linalg.norm(PENCIL0 - HINGE0)
POINT_REL = POINT0 + np.array([0.0, 0.0, -HOLE_D + 0.0001]) - HINGE0  # the needle's tip (base frame)
LEAD_REL = PENCIL0 - HINGE0  # the pencil's tip (pencil-leg frame)
BASE_SAMPLES = np.array([[0.0, 0.0, z] for z in np.linspace(0.0, HEAD_H / 2 + HANDLE_L, 8)]
                        + [list(_UPN * s) for s in np.linspace(0.0, LEG, 10)])
LEG_SAMPLES = np.array([list(_UPP * s) for s in np.linspace(0.0, LEG, 10)])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, failed="", sectors=set(), travel=0.0, radii={}, shown=set(), prev_ang=None, recent=[], force=0.0, peak_f=0.0, r=RADIUS,
               rmin=1.0, rmax=0.0, skid=0.0, point_h=0.0, lie=90.0, near=1.0, hinge=0.0, gaps={}, tcp={})
    _CV.pop("view", None)


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


def _lead_force(base) -> float:
    """The contact force (N) between the pencil leg and the pad, from PhysX's pair-wise contact forces."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("view") if _CV.get("sim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        view = sim.create_rigid_contact_view([f"{env}/compass/pencil_leg"], filter_patterns=[[f"{env}/pad/body"]],
                                             max_contact_data_count=64)
        _CV.update(sim=sim, view=view, dt=float(base.physics_dt))
    forces = view.get_contact_force_matrix(dt=_CV["dt"])
    a = forces.numpy() if hasattr(forces, "numpy") else as_numpy(forces)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).max())


def _show_lines(base, origin, mark, pr) -> None:
    """Lay each 15-deg mark of the pencil line on the paper once the pencil's point has drawn all of its 15 deg."""
    import torch

    step = 360 // LINE_N
    for k in range(LINE_N):
        if k in _EP["shown"] or not all(d in _EP["sectors"] for d in range(k * step, (k + 1) * step)):
            continue
        _EP["shown"].add(k)
        a = math.radians((k + 0.5) * step)
        rr = float(np.mean([_EP["radii"][d] for d in range(k * step, (k + 1) * step)]))
        p = mark + pr @ np.array([rr * math.cos(a), rr * math.sin(a), 0.00006]) + origin
        yaw = math.atan2(pr[1, 0], pr[0, 0]) + a + math.pi / 2
        pose = np.array([[p[0], p[1], p[2], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        obj = base.scene[f"line_{k}"]
        obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=base.device))
        obj.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32,
                                                                           device=base.device))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("COMPASS_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


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
    comp = base.scene["compass"]
    if "bodies" not in _IDS:
        names, jn = list(comp.body_names), list(comp.joint_names)
        _IDS["bodies"] = {k: names.index(k) for k in ("base", "pencil_leg")}
        _IDS["hinge"] = jn.index("pencil_leg_turn")
    bp = as_numpy(comp.data.body_link_pose_w)[0]
    ib = _IDS["bodies"]
    hp, hr = bp[ib["base"]][:3] - origin, rotation(bp[ib["base"]][3:7])
    lp, lr = bp[ib["pencil_leg"]][:3] - origin, rotation(bp[ib["pencil_leg"]][3:7])
    hinge = math.degrees(float(as_numpy(comp.data.joint_pos)[0][_IDS["hinge"]]))
    point = hp + hr @ POINT_REL
    lead = lp + lr @ LEAD_REL
    f = _lead_force(base)
    # in the pad's frame (it is loose): origin at the centre mark on the top sheet
    pd = as_numpy(base.scene["pad"].data.root_link_pose_w)[0]
    pr = rotation(pd[3:7])
    mark = pd[:3] - origin + pr @ MARK_REL
    to_pad = lambda p: pr.T @ (np.asarray(p) - mark)  # noqa: E731
    pt_p, ld_p = to_pad(point), to_pad(lead)
    skid = float(np.linalg.norm(pt_p[:2]))
    point_h = float(pt_p[2] + HOLE_D)
    r = float(np.linalg.norm(ld_p[:2]))
    on_pad = lambda q: abs(q[0] + MARK_REL[0]) <= PAD_W / 2 and abs(q[1] + MARK_REL[1]) <= PAD_L / 2  # noqa: E731
    axis_w = hr @ N0
    lie = math.degrees(math.acos(min(1.0, abs(float(axis_w[2])))))
    pts = np.concatenate([hp + BASE_SAMPLES @ hr.T, lp + LEG_SAMPLES @ lr.T])
    tree = cKDTree(pts)
    near = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    draws = f >= MARK_N  # the pencil leaves a line where it presses on the paper with the marking force
    if not _EP["failed"]:
        if f > F_SNAP:
            _EP["failed"] = f"the pencil pressed with {f:.1f} N (over {F_SNAP:.0f} N): its point breaks"
        elif skid > SKID and on_pad(pt_p) and pt_p[2] < ON_PAPER:
            _EP["failed"] = f"the point skidded {skid * 1000:.1f} mm off the centre mark"
        elif draws and abs(r - RADIUS) > STRAY:
            _EP["failed"] = f"a pencil line {abs(r - RADIUS) * 1000:.1f} mm off the 40 mm circle"
    if not _EP["failed"] and draws and abs(r - RADIUS) <= TOL:
        # the arc the pencil's point has just drawn along the circle, in 1-degree steps
        ang = math.degrees(math.atan2(ld_p[1], ld_p[0])) % 360.0
        prev = _EP["prev_ang"]
        if prev is not None:
            delta = (ang - prev + 180.0) % 360.0 - 180.0
            _EP["travel"] += abs(delta)
            if abs(delta) <= 20.0 and _EP["travel"] >= 2.0:  # it has moved (a dot where it stood is no arc)
                lo = prev if delta >= 0 else ang
                for k in range(int(math.floor(lo)), int(math.floor(lo + abs(delta))) + 1):
                    if k + 1 > lo and k < lo + abs(delta):
                        _EP["sectors"].add(k % 360)
                        _EP["radii"][k % 360] = r
        _EP["prev_ang"] = ang
    else:
        _EP["prev_ang"] = None
    _show_lines(base, origin, mark, pr)
    if draws:
        _EP["rmin"] = min(_EP["rmin"], r)
        _EP["rmax"] = max(_EP["rmax"], r)
    _EP.update(force=f, peak_f=max(_EP["peak_f"], f), r=r, skid=skid, point_h=point_h, lie=lie, near=near,
               hinge=hinge, hp=hp, lead=lead, lead_h=float(ld_p[2]), pad_moved=float(np.linalg.norm(mark[:2] - CENTRE)))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "hp" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["hp"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    full = len(_EP["sectors"]) >= FULL_DEG
    lying = _EP["lie"] <= LIE_DEG and float(_EP.get("hp", np.zeros(3))[2]) < TOP + 0.03
    done = bool(full and lying and _EP["near"] >= CLEAR and still)
    return dict(still=still, full=full, lying=lying, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    hp = _EP.get("hp", np.zeros(3))
    print(f"DBG step={_EP['step']} f={_EP['force']:.2f}N peak={_EP['peak_f']:.2f} r={_EP['r'] * 1000:.2f}mm "
          f"rng=({_EP['rmin'] * 1000:.2f},{_EP['rmax'] * 1000:.2f}) skid={_EP['skid'] * 1000:.2f}mm "
          f"pt_h={_EP['point_h'] * 1000:.2f}mm lead_h={_EP.get('lead_h', 0.0) * 1000:.2f}mm hinge={_EP['hinge']:+.2f}deg sectors={len(_EP['sectors'])} "
          f"lie={_EP['lie']:.0f} near={_EP['near'] * 1000:.0f}mm head=({hp[0]:.3f},{hp[1]:.3f},{hp[2] - TOP:.3f}) "
          f"pad_moved={_EP.get('pad_moved', 0) * 1000:.1f}mm still={int(t['still'])} failed={_EP['failed'] or '-'} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """0 once a rule is broken; otherwise 0.8 x the fraction of the 72 sectors drawn. Full score is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if _EP["failed"]:
        return 0.0
    return 0.8 * len(_EP["sectors"]) / SECTORS


def success(objects) -> bool:
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout; the left arm stays at the ready pose. The right gripper comes down over the
# handle, fingers closing across the legs' plane, holds it lightly (commanded 3 mm narrower than the handle, so the
# compass can still rock between the fingers) and presses 3 mm down. It carries the handle round the point on its 20 mm
# circle, the hand turning with the legs: back 50 deg, then forward 190 deg (aimed 12 deg ahead, waiting 1.5 s at each
# end). It opens to 14 mm round the handle, turns the open hand back 180 deg while the compass stands on its point,
# closes again, turns the hand 5 deg about the axis across the legs (keeps the pencil pressing) and carries it on
# 215 deg, past where the line began. It lifts the compass straight up, carries it to
# the desk beside the pad, stands it on its points, tips it over on them 60 deg, lets go and backs away.
# COMPASS_MODE=press presses 10 mm down instead of 3 mm; COMPASS_MODE=twist turns the hand on the spot over the handle.
OPEN_C = -0.047
ARM = os.environ.get("COMPASS_ARM", "right")
MODE = os.environ.get("COMPASS_MODE", "demo")  # demo | press, twist (wrong ways) | stand (let go at once)
PINCH = float(os.environ.get("COMPASS_PINCH", "-0.0025"))  # closed on the 8 mm handle
CAGE = float(os.environ.get("COMPASS_CAGE", "-0.007"))  # open round the handle
RADIAL = float(os.environ.get("COMPASS_RADIAL", "0.0")) * MM  # the handle target this far out toward the pencil
TURN_S = float(os.environ.get("COMPASS_TURN_S", "1.5"))  # the open hand turns back in this time
GRIP_DZ = HANDLE_TOP - 0.006  # the TCP this far above the paper on the handle
PRESS = float(os.environ.get("COMPASS_PRESS", "0.003"))  # how far below the handle's seated height the TCP is told
YAW_OFF = float(os.environ.get("COMPASS_YAWOFF", "0"))  # the grip's yaw minus the legs' direction (deg)


def handle_at(psi: float, dz: float = 0.0) -> np.ndarray:
    """The TCP on the handle with the legs turned to ``psi`` (rad) about the centre mark."""
    d = np.array([math.cos(psi), math.sin(psi), 0.0])
    return POINT0 + d * (RADIUS / 2 + RADIAL) + np.array([0.0, 0.0, GRIP_DZ + dz])


def grip_at(yaw: float) -> np.ndarray:
    return grip_rotation(yaw)


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


SWEEP1 = [float(v) for v in os.environ.get("COMPASS_SWEEP1", "-50,140").split(",")]  # legs' direction (deg)
REGRASP = float(os.environ.get("COMPASS_REGRASP", "180"))  # the open hand turns back this far round the handle
SWEEP2 = float(os.environ.get("COMPASS_SWEEP2", "215"))  # then the legs turn on this far
LEAD = float(os.environ.get("COMPASS_LEAD", "12"))  # deg the handle is aimed ahead of where the legs should be
# After the regrasp the hand is turned 180 deg and the arm's small give tips the compass the other way about the axis
# across its legs, easing the pencil off the paper; the hand is turned this far about that axis (sign as in _sweep) to
# keep the pencil marking (found by trial: at 0 the force fell below 0.3 N on about 12 % of the steps).
TILT2 = float(os.environ.get("COMPASS_TILT2", "-5.0"))  # deg
DWELL = 1.5  # s: at the end of each sweep the hand is held there while the arm catches up
SWEEP_S = float(os.environ.get("COMPASS_SWEEP_S", "14"))  # seconds per 190 deg


def _sweep(psi_a: float, psi_b: float, yaw_off: float, dz: float, seconds: float, label: str,
           lead: float = 0.0, tilt: float = 0.0) -> list:
    """The handle carried round the point from legs' direction ``psi_a`` to ``psi_b`` (deg), the hand turning with
    the legs (``yaw_off`` rad from their direction); the handle is aimed ``lead`` deg ahead along the way, since the
    arm, soft sideways, stays short of a target it pushes against."""
    out = []
    n = max(8, int(abs(psi_b - psi_a) / 5.0))
    for k in range(1, n + 1):
        psi = math.radians(psi_a + (psi_b - psi_a) * k / n)
        p = handle_at(psi + math.radians(lead), dz)
        if MODE == "twist":  # wrong: turn the wrist about the handle where it stands
            p = _EP.setdefault("twist_p", p)
        r = grip_at(psi + yaw_off)
        if tilt:  # the hand turned ``tilt`` deg about the axis across the legs
            r = _rot_axis((-math.sin(psi), math.cos(psi), 0.0), math.radians(tilt)) @ r
        out.append(_w(seconds / n, p, PINCH, label, r))
    return out


def _arm_path() -> list:
    off = math.radians(YAW_OFF)
    dz = -PRESS
    if MODE == "press":
        dz = -0.010
    a1, b1 = SWEEP1
    psi0 = math.degrees(PSI0)
    above = lambda psi, h: handle_at(math.radians(psi), 0.04 + h)  # noqa: E731
    r_start = grip_at(math.radians(psi0) + off)
    out = _lead_in(ARM, above(psi0, 0.0), r_start, OPEN_C, "above the compass's handle, open")
    out += [_w(2.0, handle_at(math.radians(psi0), 0.0), OPEN_C, "lower round the handle", r_start),
            _w(0.6, handle_at(math.radians(psi0), 0.0), OPEN_C, "settle", r_start),
            _w(1.0, handle_at(math.radians(psi0), 0.0), PINCH, "take the handle", r_start),
            _w(1.0, handle_at(math.radians(psi0), dz), PINCH, "press lightly down", r_start)]
    if MODE == "stand":
        out += [_w(1.0, handle_at(math.radians(psi0), 0.0), CAGE, "let go of it", r_start),
                _w(2.0, handle_at(math.radians(psi0), 0.03), CAGE, "rise", r_start),
                _w(15.0, handle_at(math.radians(psi0), 0.06), CAGE, "watch", r_start)]
        return out
    # to the start of the first sweep (turning back with the legs, pencil drawing)
    out += _sweep(psi0, a1, off, dz, SWEEP_S * abs(a1 - psi0) / 190.0, "draw back to the start of the first sweep",
                  -LEAD)
    out += _sweep(a1, a1, off, dz, DWELL, "let the hand catch up", -LEAD)
    out += _sweep(a1, a1, off, dz, 1.0, "turn", 0.0)
    out += _sweep(a1, b1, off, dz, SWEEP_S * abs(b1 - a1) / 190.0, "draw: carry the handle round the point", LEAD)
    out += _sweep(b1, b1, off, dz, DWELL, "let the hand catch up", LEAD)
    pb = handle_at(math.radians(b1), 0.0)
    out += [_w(0.6, pb, PINCH, "ease the press", grip_at(math.radians(b1) + off)),
            _w(0.8, pb, CAGE, "open round the handle", grip_at(math.radians(b1) + off))]
    n = 12
    for k in range(1, n + 1):  # turn the wrist back round the standing handle, fingers open
        yaw = math.radians(b1 - REGRASP * k / n) + off
        out.append(_w(TURN_S / n, pb, CAGE, "turn the open hand back round the handle", grip_at(yaw)))
    off2 = off - math.radians(REGRASP)
    ra = grip_at(math.radians(b1) + off2)
    out += [_w(0.5, pb, CAGE, "settle", ra), _w(1.0, pb, PINCH, "take the handle again", ra),
            _w(0.6, handle_at(math.radians(b1), dz), PINCH, "press lightly down", ra)]
    end = b1 + SWEEP2
    out += _sweep(b1, b1, off2, dz, 0.6, "lean the hand toward the pencil", 0.0, TILT2)
    out += _sweep(b1, end, off2, dz, SWEEP_S * SWEEP2 / 190.0, "draw the rest of the circle", LEAD, TILT2)
    out += _sweep(end, end, off2, dz, DWELL, "let the hand catch up", LEAD, TILT2)
    r_end = grip_at(math.radians(end) + off2)
    pe = handle_at(math.radians(end), 0.0)
    lift = pe + np.array([0.0, 0.0, 0.03])
    out += [_w(0.6, pe, PINCH, "ease the press", r_end),
            _w(3.0, lift, PINCH, "lift the compass straight up", r_end)]
    out += _lay_down(math.radians(YAW_PUT) - off2, off2)
    return out


PUT = np.array([float(v) for v in os.environ.get("COMPASS_PUT", "0.20,-0.255").split(",")])  # where it is laid down
YAW_PUT = float(os.environ.get("COMPASS_YAW_PUT", "60"))  # the hand's yaw (deg) while laying it down
TILT = float(os.environ.get("COMPASS_TILT", "60"))  # deg the hand tips it over on its points before letting go


def _rot_axis(axis, a: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def _lay_down(psi: float, off: float) -> list:
    """Carry it off the pad to the desk, stand it on its points there, tip it over on them, let go, move away."""
    d = np.array([math.cos(psi), math.sin(psi), 0.0])
    r0 = grip_at(psi + off)
    mid = np.array([PUT[0], PUT[1], TOP])  # the midpoint between its points on the desk
    up = np.array([0.0, 0.0, GRIP_DZ + 0.002])
    out = [_w(4.0, mid + up + np.array([0.0, 0.0, 0.03]), PINCH, "carry it off the pad, over the desk", r0),
           _w(2.0, mid + up, PINCH, "stand it on its points on the desk", r0)]
    sgn = 1.0 if TILT >= 0 else -1.0
    n = 8
    for k in range(1, n + 1):
        a = math.radians(abs(TILT)) * k / n * sgn
        rk = _rot_axis(d, a)
        out.append(_w(2.5 / n, mid + rk @ up, PINCH, "tip it over on its points", rk @ r0))
    rk = _rot_axis(d, math.radians(TILT))
    p = mid + rk @ up
    out += [_w(0.5, p, PINCH, "settle", rk @ r0),
            _w(1.0, p, OPEN_C, "let go", rk @ r0),
            _w(2.0, p + rk @ np.array([0.0, 0.0, 0.05]), OPEN_C, "back away along the fingers", rk @ r0),
            _w(8.0, p + rk @ np.array([0.0, 0.0, 0.05]) + np.array([0.02, 0.0, 0.02]), OPEN_C, "stay clear", rk @ r0)]
    return out


def demo():
    other = "left" if ARM == "right" else "right"
    return {ARM: _arm_path(), other: _stay(other)}
