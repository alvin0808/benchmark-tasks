"""On the station table: an adult chest trainer (a torso from the neck to the upper abdomen, true size) lies on its back
across the table, head to the robot's left (+y), its right side toward the robot, as a patient lies for a heart
examination with the nurse at the right side; a stethoscope's chestpiece lies diaphragm down in a steel tray beside it.
Task (instruction): listen to the heart at the aortic, pulmonic, tricuspid and mitral areas, in that order, with the
diaphragm flat and sealed on the skin and still for 2 s at each, then put the chestpiece back in its tray.

How it is done (Open RN, Nursing Skills, 2nd ed., ch. 9.3, CC BY 4.0): "The aortic area is the second intercostal
space to the right of the sternum. The pulmonic area is the second intercostal space to the left of the sternum. ...
The tricuspid (or parasternal) area is at the fourth intercostal space to the left of the sternum. The mitral ... is
the fifth intercostal space at the midclavicular line. Auscultation usually begins at the aortic area ... Move the
stethoscope sequentially to the pulmonic area ..." (Erb's point, a fifth area that text lists between them, is not
scored.) Pressure: the diaphragm's pressure "may (and sometimes should) actually leave a visible imprint of the chest
piece on the skin" (Shindler, Cardiac Auscultation Essays, Rutgers RWJMS); a bench study of stethoscopes loaded the
chestpiece with "a standardized weight of 227 gm corresponding to the typical pressure applied by a hand when
auscultating a chest" (Medical Devices: Evidence and Research 12, Dove Press), about 2.2 N.

The hand skill: the flat 43 mm diaphragm hears only when its whole rim is on the skin, and the chest is curved, so the
skin's slope differs from site to site: 9.7 deg head-to-foot at the 2nd intercostal spaces, 4.4 deg at the 4th, 1.8
deg at the 5th; 5 deg left-right at the sternal edges, 26 deg at the mitral area. The rim seals only with the disc
within about 2 deg of the skin there and pressed 2-10 N (the skin and rubber rim give about 0.3 mm per N). The
gripper pinches the head by its sides, so the head rocks a little between the fingers about the line joining them (the
left-right slope); the head-to-foot slope the wrist must match, and the arm is soft (about 1 N/mm), so the press is
set by how far past the skin the hand is aimed, and an off-centre press tips the disc off the skin.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. Patient
frame: u to the patient's left (= +x), v toward the head (= +y), from the sternal notch on the midline at
(X_MID, Y_NOTCH) = (0.200, 0.065); heights above the table.
- torso: fixed (it lies still), skin-coloured, built from superellipse cross-sections: 32 cm wide (x 0.04 to 0.36), 38
  cm at the shoulders; 21.5 cm deep at the 4th-5th ribs, 19.3 cm at the sternal notch, 18.5 cm at its lower end;
  from 24 cm below the notch (y -0.175) to the shoulders 5 cm above it (y 0.115), a neck stub (11.6 cm across) beyond.
- sites (u, v; cm): aortic (-2.8, -6.0), pulmonic (+2.8, -6.0), tricuspid (+2.8, -11.5), mitral (+9.5, -14.0); in
  world cm (x, y, height): (17.2, 0.5, 20.5), (22.8, 0.5, 20.5), (22.8, -5.0, 21.2), (29.5, -7.5, 19.6).
- chestpiece (a Littmann Classic III-type two-sided head): loose, 80 g. Steel head 44.5 mm across: adult side 12 mm
  tall (a 3 mm rubber rim, 44 mm across, round the 43 mm diaphragm), waist 30 mm, paediatric side 36 mm; 22 mm in
  all; stem 8 mm thick out of the waist 14 mm above the diaphragm, along +x to 62 mm from the head's axis. The rim is
  the only part that touches the skin: its contacts are compliant (the rubber rim and the trainer's skin, 1.2 N/mm per
  contact point; the face sinks about 1.3 mm at 4.5 N), and its contact force on the torso is read every step.
- tray: fixed steel tray, 13 x 10 cm inside (x by y), 1.5 cm walls, centred at (0.225, -0.255); the head lies in it
  at (0.195, -0.255), stem toward +x.

Score (best stage reached; nothing is latched):
- a site is heard when, for 2 s (40 steps) in a row: every one of 24 points round the diaphragm's 43 mm rim lies
  within 0.5 mm of the skin or pressed into it, the contact force on the torso is 2-10 N, the disc's centre is within
  20 mm of the site (along the skin) and moves less than 3 mm over the 2 s. Sites count only in order.
- 0.2 per site heard (0.8 for all four); 1.0 (success, held 1 s): all four heard, the chestpiece lying in its tray
  (its face within 4 mm of the tray floor, inside the walls, tilted under 10 deg), every finger point 2 cm or more
  from it, and still (under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, MeshShape, SphereShape, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001
_DEBUG = os.environ.get("STETH_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("STETH_DEBUG_EVERY", "5"))

# -- the torso -----------------------------------------------------------------------------------------------------
X_MID = float(os.environ.get("STETH_XMID", "0.200"))  # the midline (sternum) runs along y at this x
Y_NOTCH = float(os.environ.get("STETH_YNOTCH", "0.065"))  # the sternal notch
H_MAX = 0.215  # chest depth at the 4th-5th ribs (back on the table to the front of the chest)
H_DROP = 0.022  # the chest is this much lower at the sternal notch (the sternum rises toward the xiphoid)
V_LOW, V_TOP = -0.240, 0.050  # the torso from 24 cm below the notch to the top of the shoulders
A_CHEST = 0.160  # half breadth
P_UP, P_LOW = 2.2, 4.0  # cross-section: superellipse powers of the front and of the back
UP_FRAC = 0.55  # the front's curve spans this fraction of the depth


def depth(v: float) -> float:
    """Height of the chest's front on the midline at ``v`` (m above the table)."""
    if v > 0.0:
        return H_MAX - H_DROP - 17.0 * v * v
    if v > -0.16:
        return H_MAX - H_DROP * ((v + 0.16) / 0.16) ** 2
    if v > -0.18:
        return H_MAX
    w = min(1.0, (-0.18 - v) / 0.06)  # below the xiphoid the front falls 3 cm to the upper abdomen
    return H_MAX - 0.030 * w * w * (3 - 2 * w)


def half_breadth(v: float) -> float:
    return A_CHEST + (0.6 * v if v > 0.0 else 0.0)


def surface(u: float, v: float) -> float:
    """The skin's height (m above the table) at (u, v) on the front of the chest."""
    a, h = half_breadth(v), depth(v)
    s = min(abs(u) / a, 0.999999)
    zc, b = h * (1 - UP_FRAC), h * UP_FRAC
    return zc + b * (1.0 - s ** P_UP) ** (1.0 / P_UP)


def normal(u: float, v: float) -> np.ndarray:
    """Outward unit normal of the skin at (u, v) (world axes)."""
    e = 1e-4
    du = (surface(u + e, v) - surface(u - e, v)) / (2 * e)
    dv = (surface(u, v + e) - surface(u, v - e)) / (2 * e)
    n = np.array([-du, -dv, 1.0])
    return n / np.linalg.norm(n)


def to_world(u, v, z=None) -> np.ndarray:
    if z is None:
        z = surface(u, v)
    return np.array([X_MID + u, Y_NOTCH + v, TOP + z])


SITE_NAMES = ("aortic", "pulmonic", "tricuspid", "mitral")
SITES_UV = {"aortic": (-0.028, -0.060), "pulmonic": (0.028, -0.060), "tricuspid": (0.028, -0.115),
            "mitral": (0.095, -0.140)}
SITES = {k: to_world(*uv) for k, uv in SITES_UV.items()}
NORMALS = {k: normal(*uv) for k, uv in SITES_UV.items()}

SKIN, NIPPLE, STEEL, DARK, RIM, DIAPH, TRAY = (
    (0.86, 0.72, 0.62), (0.62, 0.40, 0.36), (0.78, 0.80, 0.83), (0.15, 0.15, 0.17), (0.12, 0.12, 0.13),
    (0.93, 0.93, 0.90), (0.70, 0.72, 0.75))


def _torso_mesh(nt: int = 120, nv: int = 72) -> MeshShape:
    """The torso as a closed triangle mesh: superellipse cross-sections along v (front P_UP, back P_LOW), end caps."""
    vs = np.linspace(V_LOW, V_TOP, nv)
    ts = np.linspace(0.0, 2 * math.pi, nt, endpoint=False)
    pts = []
    for v in vs:
        a, h = half_breadth(v), depth(v)
        zc, b = h * (1 - UP_FRAC), h * UP_FRAC
        for t in ts:
            c, s = math.cos(t), math.sin(t)
            u = a * math.copysign(abs(c) ** (2 / (P_UP if s >= 0 else P_LOW)), c)
            if s >= 0:
                z = surface(u, v)  # exactly on the analytic front
            else:
                z = zc - zc * abs(s) ** (2 / P_LOW)
            pts.append((X_MID + u, Y_NOTCH + v, TOP + max(z, 0.0005)))
    pts = np.array(pts)
    faces = []
    for i in range(nv - 1):
        for j in range(nt):
            a0, a1 = i * nt + j, i * nt + (j + 1) % nt
            b0, b1 = a0 + nt, a1 + nt
            faces += [(a0, a1, b1), (a0, b1, b0)]
    # end caps (fans round each end's centre)
    for i, flip in ((0, True), (nv - 1, False)):
        ring = pts[i * nt:(i + 1) * nt]
        c = len(pts)
        pts = np.vstack([pts, ring.mean(axis=0)])
        for j in range(nt):
            a0, a1 = i * nt + j, i * nt + (j + 1) % nt
            faces.append((c, a1, a0) if flip else (c, a0, a1))
    faces = np.array(faces)
    # fix the winding so normals point out (the mesh is centred on its own axis)
    ctr = np.array([X_MID, Y_NOTCH + (V_LOW + V_TOP) / 2, TOP + 0.09])
    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = (tri.mean(axis=1) - ctr)
    flip = np.einsum("ij,ij->i", fn, out) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    vn = np.zeros_like(pts)
    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return MeshShape(pts, faces, vn, SKIN)


def _torso() -> Body:
    mesh = _torso_mesh()
    o = np.array([X_MID, Y_NOTCH, TOP])  # body origin: the sternal notch's midline point, on the table
    shapes = [MeshShape(mesh.points - o, mesh.faces, mesh.normals, SKIN)]
    # neck stub: a cylinder along y above the shoulders
    shapes.append(CylShape((0.0, V_TOP + 0.045, 0.105), 0.058, 0.10, SKIN, axis="Y"))
    # nipples (visual): 4th intercostal space, 9.5 cm each side of the midline
    for u in (-0.095, 0.095):
        p = to_world(u, -0.115) - o
        shapes.append(SphereShape(tuple(float(x) for x in p), 0.006, NIPPLE, collide=False))
    return Body.build("torso", pos=tuple(float(x) for x in o), mass=7.0, color=SKIN, fixed=True,
                      friction=(0.8, 0.7), shapes=shapes, contact_offset=0.002)


# -- the chestpiece and its tray -------------------------------------------------------------------------------------
HEAD_R = 0.02225  # adult side, 44.5 mm across its rim (43 mm diaphragm)
DIAPH_R = 0.0215  # the rim's sealing edge (radius of the scored rim points)
RIM_R = 0.0220  # the rubber rim, just inside the steel ring the fingers hold
HEAD_H1 = 0.012  # adult side's height
WAIST_R, WAIST_H = 0.015, 0.004
PED_R, PED_H = 0.018, 0.006  # paediatric side
STEM_Z = 0.014  # the stem's axis above the diaphragm face
STEM_R = 0.004
STEM_X0, STEM_X1 = 0.012, 0.062  # the stem from inside the waist out to its end (from the head's axis, +x)
TRAY_C = np.array([float(v) for v in os.environ.get("STETH_TRAY", "0.225,-0.255").split(",")])
TRAY_IN = (0.13, 0.10)  # inside, x by y
TRAY_WALL, TRAY_FLOOR, TRAY_WALL_H = 0.003, 0.003, 0.015
TRAY_TOP = TOP + TRAY_FLOOR  # the tray floor's top
HEAD_C0 = np.array([TRAY_C[0] - 0.030, TRAY_C[1], TRAY_TOP + 0.0005])  # the diaphragm face's centre at the start
SKIN_K = float(os.environ.get("STETH_K", "1200"))  # N/m, compliant contact of the head (rim + trainer skin)
SKIN_C = float(os.environ.get("STETH_C", "20"))  # N s/m


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _chestpiece() -> Linkage:
    o = HEAD_C0  # the linkage origin: the diaphragm face's centre (shapes are relative to it)
    stem = [CylShape(((STEM_X0 + STEM_X1) / 2, 0.0, STEM_Z), STEM_R, STEM_X1 - STEM_X0, STEEL, axis="X"),
            CylShape((STEM_X1 - 0.004, 0.0, STEM_Z), STEM_R + 0.0012, 0.008, DARK, axis="X")]  # tube's collar
    # the steel head (gripped by its sides) on the base with the stem; the rubber rim and diaphragm on their own link,
    # the only part that touches the skin: its contacts are compliant and reported
    head = [CylShape((0.0, 0.0, 0.003 + (HEAD_H1 - 0.003) / 2), HEAD_R, HEAD_H1 - 0.003, STEEL),
            CylShape((0.0, 0.0, HEAD_H1 + WAIST_H / 2), WAIST_R, WAIST_H, STEEL),
            CylShape((0.0, 0.0, HEAD_H1 + WAIST_H + PED_H / 2), PED_R, PED_H, STEEL),
            CylShape((0.0, 0.0, HEAD_H1 + WAIST_H + PED_H - 0.0006), PED_R - 0.0015, 0.0012, DARK)]
    rim = [CylShape((0.0, 0.0, 0.0015), RIM_R, 0.003, RIM)]
    links = [Link("rim", tuple(rim), mass=0.008, anchor=_v(o + np.array([0.0, 0.0, 0.003])), axis="X",
                  friction=2.0, damping=0.01, limits=(-0.2, 0.2), report_contacts=True, compliant=(SKIN_K, SKIN_C))]
    return Linkage.build("chestpiece", pos=_v(o), mass=0.072, color=STEEL, floating=True, friction=(0.8, 0.7),
                         shapes=stem + head, links=links, contact_offset=0.0015, solver_iterations=48)


def _tray() -> Body:
    w, l = TRAY_IN
    t, h = TRAY_WALL, TRAY_WALL_H
    sh = [BoxShape((0.0, 0.0, TRAY_FLOOR / 2), (w + 2 * t, l + 2 * t, TRAY_FLOOR), TRAY),
          BoxShape(((w + t) / 2, 0.0, h / 2), (t, l + 2 * t, h), TRAY),
          BoxShape((-(w + t) / 2, 0.0, h / 2), (t, l + 2 * t, h), TRAY),
          BoxShape((0.0, (l + t) / 2, h / 2), (w, t, h), TRAY),
          BoxShape((0.0, -(l + t) / 2, h / 2), (w, t, h), TRAY)]
    return Body.build("tray", pos=(float(TRAY_C[0]), float(TRAY_C[1]), TOP), mass=0.3, color=TRAY, fixed=True,
                      friction=(0.5, 0.4), shapes=sh)


TORSO = _torso()
CHESTPIECE = _chestpiece()
TRAY_B = _tray()
OBJECTS = [TORSO, TRAY_B, CHESTPIECE]

# -- tracking and scoring -------------------------------------------------------------------------------------------
SEAL_GAP = float(os.environ.get("STETH_SEAL_GAP", "0.5")) * MM  # every rim point within this of the skin (or in it)
F_MIN, F_MAX = 2.0, 10.0  # N
SITE_TOL = 0.020
HOLD_STEPS = 40  # 2 s at 20 Hz
DRIFT = 0.003
RIM_N = 24
RIM_LOCAL = np.array([[DIAPH_R * math.cos(a), DIAPH_R * math.sin(a), 0.0]
                      for a in np.linspace(0, 2 * math.pi, RIM_N, endpoint=False)])
CLEAR = 0.02
STILL_STEPS, STILL_MM = 20, 1.0
_EP: dict = {}
_IDS: dict = {}
_CV: dict = {}
PIECE_SAMPLES = np.array([[r * math.cos(a), r * math.sin(a), z] for r in (0.0, HEAD_R) for z in (0.0, 0.011, 0.021)
                          for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
                         + [[x, 0.0, STEM_Z] for x in np.linspace(STEM_X0, STEM_X1, 8)])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, done=[], run=0, run_p=None, force=0.0, peak_f=0.0, gap=1.0, tilt=0.0,
               dist=1.0, near=1.0, recent=[], gaps={}, tcp={}, cp=None, in_tray=False, flat=90.0, site_seen={})
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


def _head_force(base) -> float:
    """The contact force (N) between the chestpiece's head and the torso, from PhysX's pair-wise contact forces."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("view") if _CV.get("sim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        view = sim.create_rigid_contact_view([f"{env}/chestpiece/rim"], filter_patterns=[[f"{env}/torso/body"]],
                                             max_contact_data_count=256)
        _CV.update(sim=sim, view=view, dt=float(base.physics_dt))
    forces = view.get_contact_force_matrix(dt=_CV["dt"])
    a = forces.numpy() if hasattr(forces, "numpy") else as_numpy(forces)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).max())


def rim_gaps(face_c, face_r) -> np.ndarray:
    """Height of each rim point above the skin (negative: pressed into it), along the skin's normal."""
    out = []
    for p in face_c + RIM_LOCAL @ face_r.T:
        u, v = p[0] - X_MID, p[1] - Y_NOTCH
        n = normal(u, v)
        out.append((p[2] - TOP - surface(u, v)) * n[2])
    return np.array(out)


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("STETH_TRACEBACK\n" + traceback.format_exc(), flush=True)
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
    cp = base.scene["chestpiece"]
    if "rim" not in _IDS:
        _IDS["rim"] = list(cp.body_names).index("rim")
    bp = as_numpy(cp.data.body_link_pose_w)[0][_IDS["rim"]]
    hr = rotation(bp[3:7])
    # every body frame sits at the linkage origin (the face centre at the start), world-aligned at the start
    face_c = bp[:3] - origin
    f = _head_force(base)
    gaps = rim_gaps(face_c, hr)
    u, v = face_c[0] - X_MID, face_c[1] - Y_NOTCH
    n = normal(u, v)
    tilt = math.degrees(math.acos(min(1.0, float(-(hr[:, 2] @ -n)))))  # face normal (-z_local) vs -n
    rn = piece_rot(n)  # the aligned pose there: tilt across the stem (about the stem axis) and along it
    zl = rn.T @ hr[:, 2]
    _EP["tilt_xy"] = (math.degrees(math.atan2(zl[0], zl[2])), math.degrees(math.atan2(zl[1], zl[2])))
    k = len(_EP["done"])
    dist = 1.0
    ok = False
    if k < len(SITE_NAMES):
        target = SITES[SITE_NAMES[k]]
        d = face_c - target
        dist = float(np.linalg.norm(d - (d @ NORMALS[SITE_NAMES[k]]) * NORMALS[SITE_NAMES[k]]))
        sealed = bool(gaps.max() <= SEAL_GAP)
        ok = sealed and F_MIN <= f <= F_MAX and dist <= SITE_TOL
        for name in SITE_NAMES:  # what the disc sealed on, for the debug log
            dd = face_c - SITES[name]
            if np.linalg.norm(dd) < SITE_TOL and sealed and F_MIN <= f <= F_MAX:
                _EP["site_seen"][name] = _EP["site_seen"].get(name, 0) + 1
    if ok:
        if _EP["run"] == 0 or np.linalg.norm(face_c - _EP["run_p"]) > DRIFT:
            _EP["run"], _EP["run_p"] = 1, face_c.copy()
        else:
            _EP["run"] += 1
        if _EP["run"] >= HOLD_STEPS:
            _EP["done"].append(SITE_NAMES[k])
            _EP["run"], _EP["run_p"] = 0, None
    else:
        _EP["run"], _EP["run_p"] = 0, None
    pts = face_c + PIECE_SAMPLES @ hr.T
    tree = cKDTree(pts)
    near = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    in_tray = (abs(face_c[0] - TRAY_C[0]) <= TRAY_IN[0] / 2 and abs(face_c[1] - TRAY_C[1]) <= TRAY_IN[1] / 2
               and face_c[2] - TRAY_TOP < 0.004)
    flat = math.degrees(math.acos(min(1.0, float(hr[2, 2]))))
    _EP.update(force=f, peak_f=max(_EP["peak_f"], f), gap=float(gaps.max()), gap_min=float(gaps.min()), tilt=tilt,
               dist=dist, near=near, cp=face_c, in_tray=in_tray, flat=flat)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and _EP["cp"] is not None:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["cp"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    back = bool(_EP["in_tray"] and _EP["flat"] < 10.0 and _EP["near"] >= CLEAR and still)
    done = bool(len(_EP["done"]) == len(SITE_NAMES) and back)
    return dict(still=still, back=back, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    cp = _EP["cp"] if _EP["cp"] is not None else np.zeros(3)
    print(f"DBG step={_EP['step']} f={_EP['force']:.2f}N peak={_EP['peak_f']:.1f} gap_max={_EP['gap'] * 1000:+.2f}mm "
          f"gap_min={_EP.get('gap_min', 0) * 1000:+.2f}mm tilt={_EP['tilt']:.1f}deg ({_EP.get('tilt_xy', (0, 0))[0]:+.1f},{_EP.get('tilt_xy', (0, 0))[1]:+.1f}) dist={_EP['dist'] * 1000:.0f}mm "
          f"run={_EP['run']} done={','.join(_EP['done']) or '-'} face=({cp[0]:.3f},{cp[1]:.3f},{cp[2] - TOP:.3f}) "
          f"near={_EP['near'] * 1000:.0f}mm tray={int(_EP['in_tray'])} flat={_EP['flat']:.0f} still={int(t['still'])} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if t["done"]:
        return 1.0
    return 0.2 * len(_EP["done"])


def success(objects) -> bool:
    return bool(_tally(objects)["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout; the left arm stays at the ready pose. The right gripper pinches the chestpiece's
# head by its sides (fingers closing along y), the hand leaned 42 deg back about that line from the diaphragm's axis (a
# straight-down hand cannot reach the chest's 20 cm height). It lifts the chestpiece 25 cm, and for each site in turn:
# it turns the diaphragm to the skin's normal there (aimed LEAN deg past it about the pinch line, which the press turns
# back), comes down along the normal from 2 cm, aims PRESS past the skin, holds for HOLD_S, and lifts back along the
# normal. Then it carries the chestpiece back over its tray, lays it down, lets go and backs away.
# STETH_MODE=level keeps the diaphragm level (as it was picked up) at every site: the wrong way.
OPEN_C = -0.047
PINCH = float(os.environ.get("STETH_PINCH", "-0.016"))  # finger value: closed 6 mm inside each side of the 44.5 mm head (a firm hold)
OFFSET = math.radians(float(os.environ.get("STETH_OFFSET", "42")))  # the hand's lean back from the diaphragm's axis
TCP_Z = float(os.environ.get("STETH_TCP_Z", "0.021"))  # the TCP on the head's axis this far above the diaphragm
PRESS = float(os.environ.get("STETH_PRESS", "0.008"))  # aim the face this far past the skin
HOLD_S = float(os.environ.get("STETH_HOLD_S", "3.5"))
MODE = os.environ.get("STETH_MODE", "demo")
LIFT = float(os.environ.get("STETH_LIFT", "0.245"))  # carried this high between the tray and the chest
LEAN = float(os.environ.get("STETH_LEAN", "4.0"))  # deg: under the press the head turns this far in the fingers about the closing axis
HOVER = float(os.environ.get("STETH_HOVER", "0.020"))  # come down along the normal from this far off the skin
ARM = "right"


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


R_GRIP0 = grip_rotation(0.0)  # fingers close along y
R_PICK = _ry(-OFFSET) @ R_GRIP0  # the hand leaned back (its fingers point forward and down), turned about the closing axis
# the TCP in the chestpiece's frame (face centre, z up out of its back, stem along +y)
TCP_LOCAL = np.array([0.0, 0.0, TCP_Z])
R_GRASP = R_PICK  # the hand's rotation relative to the chestpiece's frame (picked with the chestpiece level)


def piece_rot(n) -> np.ndarray:
    """The chestpiece's rotation with its back along ``n`` (its face on the skin) and its stem toward +x (the
    patient's left)."""
    z = np.asarray(n, float) / np.linalg.norm(n)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def hand_for(face_c, rp) -> tuple:
    return face_c + rp @ TCP_LOCAL, rp @ R_GRASP


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


def _stay(arm: str, t: float = 150.0) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(t, tuple(float(v) for v in pos), OPEN_C, "stay put", rot)]


def _arm_path() -> list:
    I3 = np.eye(3)
    p0, r0 = hand_for(HEAD_C0, I3)
    up = np.array([0.0, 0.0, 1.0])
    out = _lead_in(ARM, p0 + up * 0.05, r0, OPEN_C, "above the chestpiece's head, open")
    out += [_w(2.0, p0, OPEN_C, "lower round the head", r0), _w(0.5, p0, OPEN_C, "settle", r0),
            _w(1.2, p0, PINCH, "take the head by its sides", r0), _w(2.0, p0 + up * 0.06, PINCH, "lift the chestpiece", r0),
            _w(2.0, p0 + up * LIFT, PINCH, "lift it clear of the torso", r0)]
    prev = None
    for name in SITE_NAMES:
        n = NORMALS[name] if MODE != "level" else np.array([0.0, 0.0, 1.0])
        rp = piece_rot(n) @ _ry(-math.radians(LEAN if MODE != "level" else 0.0))  # aimed past the skin's normal: the head turns in the fingers
        s = SITES[name]
        ph, rh = hand_for(s + NORMALS[name] * HOVER, rp)
        pa, _ = hand_for(s + NORMALS[name] * (HOVER + 0.015), rp)
        pd, _ = hand_for(s - n * PRESS, rp)
        out += [_w(3.0 if prev is None else 2.5, pa, PINCH, f"over the {name} area", rh),
                _w(1.5, ph, PINCH, f"to just off the {name} area, diaphragm to the skin's normal", rh),
                _w(1.5, pd, PINCH, f"press the diaphragm on the {name} area", rh),
                _w(HOLD_S, pd, PINCH, f"hold it sealed on the {name} area", rh),
                _w(1.0, ph, PINCH, f"lift off the {name} area", rh)]
        prev = name
    pt, rt = hand_for(HEAD_C0 + np.array([0.0, 0.0, 0.002]), I3)
    out += [_w(3.0, pt + up * LIFT, PINCH, "over the tray", rt), _w(2.0, pt + up * 0.06, PINCH, "down to the tray", rt), _w(2.0, pt, PINCH, "lay it in the tray", rt),
            _w(0.5, pt, PINCH, "settle", rt), _w(1.0, pt, OPEN_C, "let go", rt),
            _w(1.5, pt - rt[:, 2] * 0.05, OPEN_C, "back the fingers off along themselves", rt),
            _w(10.0, pt - rt[:, 2] * 0.05 + up * 0.05, OPEN_C, "stay clear", rt)]
    return out


def demo():
    return {ARM: _arm_path(), "left": _stay("left")}
