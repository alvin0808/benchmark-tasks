"""On the station table: a single printed times-table worksheet (A4, 80 g/m2 paper) lies loose on the desk with five
answers pencilled in its answer boxes; one of them is wrong (7 x 8 = 54). A white plastic block eraser stands on its
end on the desk to the right of the sheet. Task (instruction): rub out the wrong answer with the eraser without
rubbing out the other answers, leaving the sheet where it is, and put the eraser down. The instruction does not say how.

How it is done: "Always erase in one direction while holding the paper firmly behind your erasing hand. This
prevents the paper from buckling or creasing." (wildlifeartstore.com, "How This Eraser Drawing Hack Transformed My
Art"); the Navy drafting manual (draftingmanuals.tpub.com, "Electric Erasers"): "Isolate the unwanted area or protect
surrounding areas with an erasing shield."

Why two hands: the eraser grips the paper far better than the paper grips the desk (eraser on paper about 1.0,
paper on a desk about 0.45), so an eraser rubbed on a sheet nobody holds drags the whole sheet along with it instead
of sliding over the pencil marks (a real single sheet then slides and creases); the other hand pressing the sheet to
the desk beside the answer keeps it still. The working hand rubs a 13 x 23 mm end face back and forth over a 10 mm
tall answer box with the neighbouring answers 6 mm above and below it, pressing hard enough to erase (1 N or more)
with strokes long enough to break the eraser's grip on the paper (the arm is soft sideways).

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- worksheet: loose, A4 (21.0 x 29.7 cm), lying landscape (its long side along y), 5.0 g (80 g/m2); middle at
  (0.205, 0.020). Five rows of printed sums, 16 mm apart, with a printed 30 x 10 mm answer box at the right of each
  (box middles at y -0.100; rows at x 0.237, 0.221, 0.205, 0.189, 0.173). Pencilled answers (graphite, two digits 7 mm
  tall): 35, 42, 54 (the wrong one, in the middle row at x 0.205), 63, 49.
- eraser: loose, 65 x 23 x 13 mm (Staedtler Mars plastic 526 50 size) with its card sleeve, 25 g, standing on its
  13 x 23 mm end at (0.170, -0.175), its 13 mm side along x.

Score (latched 0 for the rest of the run once a rule is broken):
- graphite: each pencilled answer is a set of 1 mm grid cells along its strokes. A cell is rubbed while the eraser's
  end face covers it, touching the paper (within 0.3 mm of the sheet), pressing on the sheet with 1 N or more (PhysX
  contact force, read every control step), by how far the face slides over the sheet that step. A cell of the wrong
  answer is erased after 60 mm of rubbing; a cell of another answer is damaged after 10 mm (visibly lightened).
- 0: more than 5 % of the other answers' cells damaged. (A sheet that slides along with the eraser is not penalised
  as such: the eraser does not slide over it, so nothing is erased.)
- 0.6 x the fraction of the wrong answer's cells erased.
- 0.75: at least 90 % of the wrong answer erased.
- 1.0 (success, held 1 s): at least 90 % erased, and the eraser put down (its lowest point within 3 mm of the desk or the sheet,
  every finger point of both grippers 2 cm or more from it) and still (under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, allow_matrix_grips  # noqa: E402
from bodykit import as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001
_E = os.environ.get

# -- the layout ----------------------------------------------------------------------------------------------------
SHEET_T = float(_E("ERASE_SHEET_T", "0.3")) * MM  # a sheet of 80 g/m2 paper is 0.1 mm; modelled 0.3 mm thick
SHEET_X, SHEET_Y = 0.210, 0.297  # A4 lying landscape: short side along x
SHEET_C = (0.205, 0.020)
SHEET_MASS = 0.005  # 80 g/m2 x 0.0624 m2
PAPER = TOP + SHEET_T  # the sheet's top
# friction: the eraser's material wins against everything it touches ("max"), the sheet's loses ("min")
MU_ERASER = tuple(float(v) for v in _E("ERASE_MU_E", "1.0,0.9").split(","))
MU_SHEET = tuple(float(v) for v in _E("ERASE_MU_S", "0.45,0.40").split(","))

ROW_DX = 0.016  # rows of sums 16 mm apart
ROWS = [("7x5=", "35"), ("7x6=", "42"), ("7x8=", "54"), ("7x9=", "63"), ("7x7=", "49")]
WRONG = 2  # the middle row: 7 x 8 = 54
BOX_Y = -0.100  # the answer boxes' middles (world y)
BOX_W, BOX_H = 0.030, 0.010  # along y, along x
ROW_X = [SHEET_C[0] + ROW_DX * (WRONG - i) for i in range(len(ROWS))]  # world x of each row's middle
DIG_W, DIG_H, DIG_GAP = 0.0045, 0.007, 0.002  # a handwritten digit's width (along -y), height (along +x)

ER_X, ER_Y, ER_Z = 0.013, 0.023, 0.065  # the eraser: thickness (x), width (y), length (standing: z)
ER_MASS = 0.025
ER0 = (float(_E("ERASE_EX", "0.170")), float(_E("ERASE_EY", "-0.175")))
ER_K = float(_E("ERASE_K", "20000"))  # N/m: the give of the eraser's soft end on the paper (compliant contact)
ER_C = float(_E("ERASE_C", "40"))

PAPER_WHITE, INK, BOX_INK, GRAPHITE, ER_WHITE, SLEEVE, SLEEVE_DARK = (
    (0.96, 0.96, 0.94), (0.08, 0.08, 0.1), (0.45, 0.55, 0.75), (0.22, 0.22, 0.24), (0.97, 0.97, 0.97),
    (0.15, 0.35, 0.75), (0.08, 0.12, 0.3))

# glyph strokes in a 4.5 x 7 unit box (u along the reading direction, v up the page), polylines
GLYPHS = {
    "2": [[(0, 5.5), (1, 7), (3.5, 7), (4.5, 5.5), (0, 0), (4.5, 0)]],
    "3": [[(0, 7), (4.5, 7), (2, 4.2), (4.5, 2.5), (3.5, 0), (0, 0.5)]],
    "4": [[(3.2, 0), (3.2, 7), (0, 2), (4.5, 2)]],
    "5": [[(4.5, 7), (0.5, 7), (0, 4), (3, 4.3), (4.5, 2.5), (3.5, 0.2), (0, 0.5)]],
    "6": [[(4, 7), (1, 5), (0, 2), (1, 0), (3.5, 0), (4.5, 1.8), (3.5, 3.5), (1, 3.5), (0, 2)]],
    "7": [[(0, 7), (4.5, 7), (1.5, 0)]],
    "8": [[(2.25, 3.5), (0.4, 5.3), (2.25, 7), (4.1, 5.3), (2.25, 3.5), (0, 1.7), (2.25, 0), (4.5, 1.7),
           (2.25, 3.5)]],
    "9": [[(4.5, 4.5), (1, 3.8), (0, 5.5), (1.2, 7), (3.5, 7), (4.5, 5.5), (4.5, 4.5), (3.5, 0)]],
    "x": [[(1, 1.5), (3.5, 4.5)], [(1, 4.5), (3.5, 1.5)]],
    "=": [[(0.5, 2.3), (4, 2.3)], [(0.5, 4.2), (4, 4.2)]],
}


def _glyph_world(ch: str, u0: float, xmid: float, w: float = DIG_W, h: float = DIG_H) -> list:
    """The glyph's polylines in world (x, y): ``u0`` is its left edge (world y, reading toward -y), ``xmid`` its row."""
    out = []
    for line in GLYPHS[ch]:
        out.append([(xmid - h / 2 + v / 7.0 * h, u0 - u / 4.5 * w) for (u, v) in line])
    return out


def _answer_lines(row: int) -> list:
    """The handwritten answer of a row: polylines in world (x, y)."""
    text = ROWS[row][1]
    span = len(text) * DIG_W + (len(text) - 1) * DIG_GAP
    u = BOX_Y + span / 2
    lines = []
    for ch in text:
        lines += _glyph_world(ch, u, ROW_X[row])
        u -= DIG_W + DIG_GAP
    return lines


def _stroke_box(a, b, width: float, color, z: float) -> BoxShape:
    """A flat stroke from ``a`` to ``b`` (sheet frame x, y) at height ``z``, visual only."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ln = float(np.linalg.norm(d))
    yaw = math.atan2(d[1], d[0])
    c = (a + b) / 2
    return BoxShape((float(c[0]), float(c[1]), z), (ln + width, width, 0.0001), color, yaw=yaw, collide=False)


def _to_sheet(p) -> tuple:
    return (p[0] - SHEET_C[0], p[1] - SHEET_C[1])


def _sheet() -> Body:
    zt = SHEET_T / 2 + 0.00005
    sh = [BoxShape((0.0, 0.0, 0.0), (SHEET_X, SHEET_Y, SHEET_T), PAPER_WHITE)]
    # printed: a title bar, a name line, the sums and the answer boxes
    sh.append(BoxShape((0.080, 0.02, zt), (0.006, 0.120, 0.0001), INK, collide=False))
    sh.append(BoxShape((0.080, -0.090, zt), (0.0006, 0.060, 0.0001), INK, collide=False))
    for i, (q, _) in enumerate(ROWS):
        u = BOX_Y + BOX_W / 2 + 0.008 + 4 * (DIG_W + 0.0025)  # the sum is printed left of its box
        for ch in q:
            for line in _glyph_world(ch, u, ROW_X[i], 0.0042, 0.0065):
                for a, b in zip(line[:-1], line[1:]):
                    sh.append(_stroke_box(_to_sheet(a), _to_sheet(b), 0.0008, INK, zt))
            u -= 0.0042 + 0.0025
        bx, by = _to_sheet((ROW_X[i], BOX_Y))
        for (dx, dy, sx, sy) in ((BOX_H / 2, 0, 0.0004, BOX_W), (-BOX_H / 2, 0, 0.0004, BOX_W),
                                 (0, BOX_W / 2, BOX_H, 0.0004), (0, -BOX_W / 2, BOX_H, 0.0004)):
            sh.append(BoxShape((bx + dx, by + dy, zt), (sx, sy, 0.0001), BOX_INK, collide=False))
        if i != WRONG:  # the other answers, pencilled (drawn on the sheet)
            for line in _answer_lines(i):
                for a, b in zip(line[:-1], line[1:]):
                    sh.append(_stroke_box(_to_sheet(a), _to_sheet(b), 0.0009, GRAPHITE, zt + 0.00002))
    return Body.build("sheet", pos=(SHEET_C[0], SHEET_C[1], TOP + SHEET_T / 2 + 0.0003), mass=SHEET_MASS,
                      color=PAPER_WHITE, friction=MU_SHEET, friction_combine="min", shapes=sh, contact_offset=0.001,
                      solver_iterations=32)


def _eraser() -> Body:
    sl0, sl1 = -0.012, 0.0245  # the card sleeve, along its length (from the middle)
    sh = [BoxShape((0.0, 0.0, 0.0), (ER_X, ER_Y, ER_Z), ER_WHITE),
          BoxShape((0.0, 0.0, (sl0 + sl1) / 2), (ER_X + 0.0006, ER_Y + 0.0006, sl1 - sl0), SLEEVE, collide=False),
          BoxShape((0.0, 0.0, sl1 - 0.004), (ER_X + 0.0008, ER_Y + 0.0008, 0.002), SLEEVE_DARK, collide=False)]
    return Body.build("eraser", pos=(ER0[0], ER0[1], TOP + ER_Z / 2 + 0.001), mass=ER_MASS, color=ER_WHITE,
                      friction=MU_ERASER, friction_combine="max", report_contacts=True, compliant=(ER_K, ER_C),
                      shapes=sh, contact_offset=0.002, solver_iterations=32)


SHEET = _sheet()
ERASER = _eraser()

# the wrong answer's pencil strokes: one visual body per piece (at most 3 mm long), hidden once its cells are erased
PIECE_MAX = 0.003
_PIECES = []  # (a, b) world xy at the start
for _line in _answer_lines(WRONG):
    for _a, _b in zip(_line[:-1], _line[1:]):
        _a, _b = np.array(_a), np.array(_b)
        _n = max(1, int(math.ceil(np.linalg.norm(_b - _a) / PIECE_MAX)))
        for _k in range(_n):
            _PIECES.append((_a + (_b - _a) * _k / _n, _a + (_b - _a) * (_k + 1) / _n))
MARK_Z = PAPER + 0.00007
HIDE = np.array([0.22, -0.10, TOP - 0.30])


def _piece_body(k: int, a, b) -> Body:
    d = np.asarray(b) - np.asarray(a)
    c = (np.asarray(a) + np.asarray(b)) / 2
    return Body.build(f"mark_{k}", pos=(float(c[0]), float(c[1]), MARK_Z), mass=0.0005, color=GRAPHITE,
                      yaw=float(math.atan2(d[1], d[0])), gravity=False, damping=(5.0, 5.0),
                      shapes=[BoxShape((0.0, 0.0, 0.0), (float(np.linalg.norm(d)) + 0.0009, 0.0009, 0.0001), GRAPHITE,
                                       collide=False)])


MARKS = [_piece_body(k, a, b) for k, (a, b) in enumerate(_PIECES)]
OBJECTS = [SHEET, ERASER] + MARKS


# -- graphite cells (1 mm grid along the strokes, in the sheet's frame) ------------------------------------------------
def _cells(lines) -> np.ndarray:
    pts = []
    for line in lines:
        for a, b in zip(line[:-1], line[1:]):
            a, b = np.array(a), np.array(b)
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / 0.0005)))
            pts += [a + (b - a) * t for t in np.linspace(0.0, 1.0, n + 1)]
    rel = np.array([_to_sheet(p) for p in pts])
    idx = np.unique(np.floor(rel / 0.001).astype(int), axis=0)
    return (idx + 0.5) * 0.001


CELLS_WRONG = _cells(_answer_lines(WRONG))
CELLS_OTHER = np.concatenate([_cells(_answer_lines(i)) for i in range(len(ROWS)) if i != WRONG])
PIECE_CELLS = []  # which wrong-answer cells each visual piece covers
for _a, _b in _PIECES:
    _sa, _sb = np.array(_to_sheet(_a)), np.array(_to_sheet(_b))
    _d = _sb - _sa
    _t = np.clip(((CELLS_WRONG - _sa) @ _d) / max(float(_d @ _d), 1e-12), 0.0, 1.0)
    _dist = np.linalg.norm(CELLS_WRONG - (_sa + _t[:, None] * _d), axis=1)
    PIECE_CELLS.append(np.nonzero(_dist <= 0.0008)[0])

# -- tracking and scoring -------------------------------------------------------------------------------------------
PRESS_N = float(_E("ERASE_PRESS_N", "1.0"))  # N: the eraser rubs graphite off at this contact force or more
ERASE_MM = float(_E("ERASE_MM", "60"))  # mm of rubbing that erase a cell of the wrong answer
DAMAGE_MM = float(_E("ERASE_DAMAGE_MM", "10"))  # mm of rubbing that visibly lighten a cell of another answer
TOUCH = 0.0003  # the face within this of the sheet's top touches it
FULL = 0.90  # erased fraction for full credit
INTACT = 0.95  # the other answers' cells undamaged, at least
CLEAR = 0.02
REST = 0.003
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = _E("ERASE_DEBUG") == "1"
_DEBUG_EVERY = int(_E("ERASE_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
_CV: dict = {}

CORNERS = np.array([[sx * SHEET_X / 2, sy * SHEET_Y / 2] for sx in (-1, 1) for sy in (-1, 1)])
_g = np.linspace(-0.5, 0.5, 5)
ER_SAMPLES = np.array([[ER_X * a, ER_Y * b, -ER_Z / 2 + c * ER_Z] for a in _g for b in _g for c in (0.0, 1.0)]
                      + [[sx * ER_X / 2, sy * ER_Y / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                         for z in np.linspace(-ER_Z / 2, ER_Z / 2, 9)])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, failed="", rub_w=np.zeros(len(CELLS_WRONG)),
               rub_o=np.zeros(len(CELLS_OTHER)), hidden=set(), placed=None, spell=None, prev_face=None,
               recent=[], force=0.0, peak_f=0.0, near=1.0, low=1.0, drag=0.0, moved=0.0, touch=0, slid=0.0,
               tcp={}, gaps={}, ep=None)
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


def _press_force(base) -> float:
    """The contact force (N) between the eraser and the sheet, from PhysX's pair-wise contact forces."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("view") if _CV.get("sim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        view = sim.create_rigid_contact_view([f"{env}/eraser/body"], filter_patterns=[[f"{env}/sheet/body"]],
                                             max_contact_data_count=64)
        _CV.update(sim=sim, view=view, dt=float(base.physics_dt))
    forces = view.get_contact_force_matrix(dt=_CV["dt"])
    a = forces.numpy() if hasattr(forces, "numpy") else as_numpy(forces)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).sum())


def _hold_force(base) -> float:
    """Debug only: the contact force (N) of the left arm's links on the sheet (the arms report contacts)."""
    from isaaclab_physx.physics import PhysxManager
    from yam_arena.robot import asset_link_paths

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("hview") if _CV.get("hsim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        links = [f"{env}/LeftArm{path}" for path in asset_link_paths().values()]
        view = sim.create_rigid_contact_view(links, filter_patterns=[[f"{env}/sheet/body"]] * len(links),
                                             max_contact_data_count=16 * len(links))
        _CV.update(hsim=sim, hview=view)
    a = view.get_contact_force_matrix(dt=float(base.physics_dt))
    a = a.numpy() if hasattr(a, "numpy") else as_numpy(a)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).sum())


def _arms_force(base) -> float:
    """Debug only: the contact force (N) between the two arms."""
    from isaaclab_physx.physics import PhysxManager
    from yam_arena.robot import asset_link_paths

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("aview") if _CV.get("asim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        links = [f"{env}/LeftArm{path}" for path in asset_link_paths().values()]
        others = [f"{env}/RightArm{path}" for path in asset_link_paths().values()]
        view = sim.create_rigid_contact_view(links, filter_patterns=[others] * len(links),
                                             max_contact_data_count=16 * len(links) * len(others))
        _CV.update(asim=sim, aview=view)
    a = view.get_contact_force_matrix(dt=float(base.physics_dt))
    a = a.numpy() if hasattr(a, "numpy") else as_numpy(a)
    return float(np.linalg.norm(np.asarray(a, float).reshape(-1, 3), axis=-1).sum())


def _place_marks(base, origin, sp, sr) -> None:
    """Keep the wrong answer's pencil pieces on the sheet as it moves; hide the erased ones below the table."""
    import torch

    yaw_s = math.atan2(sr[1, 0], sr[0, 0])
    key = (round(float(sp[0]) * 1e4), round(float(sp[1]) * 1e4), round(float(sp[2]) * 1e4), round(yaw_s * 2000),
           len(_EP["hidden"]))
    if key == _EP["placed"]:
        return
    _EP["placed"] = key
    for k, (a, b) in enumerate(_PIECES):
        if k in _EP["hidden"]:
            p, yaw = HIDE + np.array([0.0, 0.0, -0.002 * k]), 0.0
        else:
            c = (np.asarray(a) + np.asarray(b)) / 2
            rel = np.array([c[0] - SHEET_C[0], c[1] - SHEET_C[1], SHEET_T / 2 + 0.00007])
            p = sp + sr @ rel
            d = np.asarray(b) - np.asarray(a)
            yaw = math.atan2(d[1], d[0]) + yaw_s
        p = p + origin
        pose = np.array([[p[0], p[1], p[2], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        obj = base.scene[f"mark_{k}"]
        obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=base.device))
        obj.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32,
                                                                           device=base.device))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("ERASE_TRACEBACK\n" + traceback.format_exc(), flush=True)
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
    sd = as_numpy(base.scene["sheet"].data.root_link_pose_w)[0]
    sp, sr = sd[:3] - origin, rotation(sd[3:7])
    ed = as_numpy(base.scene["eraser"].data.root_link_pose_w)[0]
    ep, er = ed[:3] - origin, rotation(ed[3:7])
    f = _press_force(base)
    corners = (sp[:2] + CORNERS @ sr[:2, :2].T)
    corners0 = np.array(SHEET_C) + CORNERS
    moved = float(np.linalg.norm(corners - corners0, axis=1).max())
    ans = sp[:2] + sr[:2, :2] @ np.array(_to_sheet((ROW_X[WRONG], BOX_Y)))
    # the eraser's end faces (either end may be used): which points of the face touch the sheet
    to_sheet = lambda p: sr.T @ (np.asarray(p) - sp)  # noqa: E731
    best = None
    for sgn in (-1.0, 1.0):
        fc = ep + er @ np.array([0.0, 0.0, sgn * ER_Z / 2])
        h = float(to_sheet(fc)[2] - SHEET_T / 2)
        if best is None or h < best[0]:
            best = (h, sgn, fc)
    face_h, sgn, fc = best
    fc_s = to_sheet(fc)

    def covered(cells) -> np.ndarray:
        """Cells (sheet frame) under the end face where the face is within TOUCH of the sheet's top."""
        pw = sp[:, None] + sr @ np.c_[cells, np.full(len(cells), SHEET_T / 2)].T  # 3 x n, world
        ce = er.T @ (pw - ep[:, None])  # eraser frame
        inside = (np.abs(ce[0]) <= ER_X / 2) & (np.abs(ce[1]) <= ER_Y / 2)
        face = ep[:, None] + er @ np.vstack([ce[0], ce[1], np.full(ce.shape[1], sgn * ER_Z / 2)])
        fs = sr.T @ (face - sp[:, None])  # that point of the end face, in the sheet's frame
        over = np.linalg.norm(fs[:2] - cells.T, axis=0) <= 0.001  # the face point is over the cell (not tipped)
        return inside & over & (fs[2] - SHEET_T / 2 <= TOUCH)

    pressing = f >= PRESS_N
    slid = 0.0
    if _EP["prev_face"] is not None and pressing:
        slid = float(np.linalg.norm(fc_s[:2] - _EP["prev_face"][:2]))
        if slid < 0.02:  # one control step: anything bigger is a jump, not a rub
            _EP["rub_w"][covered(CELLS_WRONG)] += slid * 1000
            _EP["rub_o"][covered(CELLS_OTHER)] += slid * 1000
    _EP["prev_face"] = fc_s if pressing else None
    # debug only: how far the sheet is dragged along under the eraser in one press
    if f >= 0.5:
        # the wrong answer's middle (a point of the sheet) moving while the eraser presses: the sheet is dragged
        if _EP["spell"] is None:
            _EP["spell"] = ans.copy()
        _EP["drag"] = max(_EP["drag"], float(np.linalg.norm(ans - _EP["spell"])))
    else:
        _EP["spell"] = None
    damaged = float((_EP["rub_o"] >= DAMAGE_MM).mean())
    if not _EP["failed"]:
        if damaged > 1.0 - INTACT:
            _EP["failed"] = f"{damaged * 100:.0f} % of the other answers rubbed"
    erased = _EP["rub_w"] >= ERASE_MM
    for k, idx in enumerate(PIECE_CELLS):
        if k not in _EP["hidden"] and (len(idx) == 0 or erased[idx].all()):
            _EP["hidden"].add(k)
    _place_marks(base, origin, sp, sr)
    pts = ep + ER_SAMPLES @ er.T
    tree = cKDTree(pts)
    near = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    low = float(pts[:, 2].min())
    on_sheet = bool(np.all(np.abs(to_sheet(ep)[:2]) <= np.array([SHEET_X, SHEET_Y]) / 2))
    rest_h = low - (PAPER if on_sheet else TOP)
    if _DEBUG:
        _EP["hold_f"] = _hold_force(base)
        _EP["arms_f"] = _arms_force(base)
    _EP["sheet_d"] = (float((sp[0] - SHEET_C[0]) * 1000), float((sp[1] - SHEET_C[1]) * 1000),
                      math.degrees(math.atan2(sr[1, 0], sr[0, 0])))
    _EP.update(force=f, peak_f=max(_EP["peak_f"], f), near=near, rest_h=rest_h, moved=moved, face_h=face_h, ep=ep,
               slid=slid, erased=float(erased.mean()), damaged=damaged, fc=fc_s)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and _EP.get("ep") is not None:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["ep"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    erased = _EP.get("erased", 0.0)
    clean = erased >= FULL
    down = _EP.get("rest_h", 1.0) <= REST and _EP["near"] >= CLEAR
    return dict(still=still, erased=erased, clean=clean, down=down, done=bool(clean and down and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    ep = _EP.get("ep")
    ep = np.zeros(3) if ep is None else ep
    fc = _EP.get("fc", np.zeros(3))
    print(f"DBG step={_EP['step']} f={_EP['force']:.2f}N hold={_EP.get('hold_f', 0):.2f}N arms={_EP.get('arms_f', 0):.1f}N peak={_EP['peak_f']:.2f} face_h={_EP.get('face_h', 0) * 1000:.2f}mm "
          f"face=({fc[0] * 1000:.1f},{fc[1] * 1000:.1f})mm slid={_EP['slid'] * 1000:.2f}mm "
          f"erased={t['erased'] * 100:.0f}% dmg={_EP.get('damaged', 0) * 100:.1f}% hidden={len(_EP['hidden'])}/{len(_PIECES)} "
          f"sheet=({_EP.get('sheet_d', (0, 0, 0))[0]:.1f},{_EP.get('sheet_d', (0, 0, 0))[1]:.1f}mm,{_EP.get('sheet_d', (0, 0, 0))[2]:.2f}deg) "
          f"drag={_EP['drag'] * 1000:.1f}mm moved={_EP['moved'] * 1000:.1f}mm eraser=({ep[0]:.3f},{ep[1]:.3f},{ep[2] - TOP:.3f}) "
          f"rest_h={_EP.get('rest_h', 1) * 1000:.1f}mm near={_EP['near'] * 1000:.0f}mm still={int(t['still'])} "
          f"failed={_EP['failed'] or '-'} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def _debug_cells() -> None:
    if _EP["step"] % 100:
        return
    ctr = np.array(_to_sheet((ROW_X[WRONG], BOX_Y)))
    left = [(round((c[0] - ctr[0]) * 1000, 1), round((c[1] - ctr[1]) * 1000, 1), round(float(r), 0))
            for c, r in zip(CELLS_WRONG, _EP["rub_w"]) if r < ERASE_MM]
    print(f"DBGCELLS step={_EP['step']} left={left}", flush=True)


def score(objects) -> float:
    """0 once a rule is broken; 0.75 when the wrong answer is erased and the sheet is in place; else 0.6 x erased."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
        _debug_cells()
    if _EP["failed"]:
        return 0.0
    if t["clean"]:
        return 0.75
    return 0.6 * t["erased"]


def success(objects) -> bool:
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left gripper, open (fingertips 8 cm apart along x), comes down on the sheet in
# the wrong answer's row 21 cm to its left and presses it to the desk with both fingertips, 10 mm below their touch
# (about 7 N). The right gripper takes the eraser by its top across its 23 mm width (so the rubbing does not pivot it
# between the fingers), carries it over the wrong answer, presses its lower end on the paper and rubs it back and forth
# along the row, each stroke commanded well past where the eraser goes (the arm is soft sideways and has to build up
# the force that breaks the eraser's grip on the paper); it lifts the eraser, stands it back where it was, lets go and
# rises; then the left hand rises.
# ERASE_ONE=1: the left arm stays at its ready pose (one hand only).
OPEN_C = -0.047
ONE = _E("ERASE_ONE") == "1"
R_RIGHT = grip_rotation(math.radians(float(_E("ERASE_GRIP_YAW", "0"))))  # fingers closing along y, across the 23 mm
R_LEFT = grip_rotation(math.radians(float(_E("ERASE_HOLD_YAW", "-90"))))  # -90: fingertips apart along x
R_L0, R_L180 = grip_rotation(0.0), grip_rotation(math.pi)
PINCH = float(_E("ERASE_PINCH", "-0.008"))  # on the eraser's 23 mm width
GRIP_DOWN = float(_E("ERASE_GRIP_DOWN", "0.008"))  # the TCP this far below the eraser's top
HOLD = np.array([float(v) for v in _E("ERASE_HOLD", "0.205,0.110").split(",")])  # where the left hand presses
HOLD_PRESS = float(_E("ERASE_HOLD_PRESS", "0.010"))  # m below the fingertips' touch
HOLD_FINGER = float(_E("ERASE_HOLD_FINGER", "-0.040"))  # the left gripper open (fingertips 8 cm apart along x)
TIP_DZ = 0.0135  # the TCP above the paper when the closed fingertips touch it
TARGET = np.array([float(v) for v in _E("ERASE_TARGET", f"{ROW_X[WRONG]},{BOX_Y}").split(",")])
RUB_PRESS = float(_E("ERASE_RUB_PRESS", "0.004"))  # m below the eraser's touch
STROKE = float(_E("ERASE_STROKE", "0.040"))  # m each way along y
STROKES = int(_E("ERASE_STROKES", "12"))  # back-and-forth rubs
STROKE_OFF = float(_E("ERASE_STROKE_OFF", "0.004"))  # m: the strokes' middle this far toward +y of the answer's
HALF_S = float(_E("ERASE_HALF_S", "1.0"))  # seconds per half stroke


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


def _dur(wps) -> float:
    return float(sum(w.duration for w in wps))


def _right_path() -> tuple[list, float, float]:
    """The eraser hand; returns its waypoints, the time the rubbing starts and the time the eraser leaves the sheet."""
    top = np.array([ER0[0], ER0[1], TOP + ER_Z])
    grip = top - np.array([0.0, 0.0, GRIP_DOWN])
    out = _lead_in("right", grip + np.array([0.0, 0.0, 0.05]), R_RIGHT, OPEN_C, "above the eraser, open")
    out += [_w(2.0, grip, OPEN_C, "lower round the eraser's top", R_RIGHT),
            _w(0.5, grip, OPEN_C, "settle", R_RIGHT),
            _w(1.2, grip, PINCH, "take the eraser", R_RIGHT),
            _w(1.5, grip + np.array([0.0, 0.0, 0.04]), PINCH, "lift it", R_RIGHT)]
    # the TCP when the eraser's lower end touches the paper
    touch = np.array([TARGET[0], TARGET[1], PAPER + ER_Z - GRIP_DOWN])
    out += [_w(2.5, touch + np.array([0.0, 0.0, 0.02]), PINCH, "carry it over the wrong answer", R_RIGHT),
            _w(1.5, touch + np.array([0.0, 0.0, 0.003]), PINCH, "lower it", R_RIGHT),
            _w(1.0, touch - np.array([0.0, 0.0, RUB_PRESS]), PINCH, "press its end on the paper", R_RIGHT)]
    t_rub = _dur(out)
    p0 = touch - np.array([0.0, 0.0, RUB_PRESS])
    for k in range(STROKES):
        out += [_w(HALF_S, p0 + np.array([0.0, STROKE_OFF + STROKE, 0.0]), PINCH, f"rub {k + 1} (left)", R_RIGHT),
                _w(HALF_S, p0 + np.array([0.0, STROKE_OFF - STROKE, 0.0]), PINCH, f"rub {k + 1} (right)", R_RIGHT)]
    out += [_w(HALF_S, p0, PINCH, "back to the middle", R_RIGHT),
            _w(1.0, touch + np.array([0.0, 0.0, 0.02]), PINCH, "lift the eraser off", R_RIGHT)]
    t_off = _dur(out)
    put = np.array([ER0[0], ER0[1], TOP + ER_Z - GRIP_DOWN + 0.001])
    out += [_w(2.5, put + np.array([0.0, 0.0, 0.02]), PINCH, "carry it back", R_RIGHT),
            _w(1.5, put, PINCH, "stand it on the desk", R_RIGHT),
            _w(1.0, put, OPEN_C, "let go", R_RIGHT),
            _w(1.5, put + np.array([0.0, 0.0, 0.06]), OPEN_C, "rise", R_RIGHT),
            _w(10.0, put + np.array([0.0, 0.0, 0.06]), OPEN_C, "stay clear", R_RIGHT)]
    return out, t_rub, t_off


def _left_path(t_rub: float, t_off: float) -> list:
    tip = np.array([HOLD[0], HOLD[1], PAPER + TIP_DZ])
    out = _lead_in("left", tip + np.array([0.0, 0.0, 0.04]), R_LEFT, HOLD_FINGER, "above the sheet beside the answer")
    out += [_w(2.0, tip + np.array([0.0, 0.0, 0.004]), HOLD_FINGER, "lower the fingertips to the sheet", R_LEFT),
            _w(1.0, tip - np.array([0.0, 0.0, HOLD_PRESS]), HOLD_FINGER, "press the sheet to the desk", R_LEFT)]
    wait = max(0.5, t_off + 0.5 - _dur(out))
    out += [_w(wait, tip - np.array([0.0, 0.0, HOLD_PRESS]), HOLD_FINGER, "hold the sheet down", R_LEFT),
            _w(1.5, tip + np.array([0.0, 0.0, 0.05]), HOLD_FINGER, "rise", R_LEFT),
            _w(15.0, tip + np.array([0.0, 0.0, 0.05]), HOLD_FINGER, "stay clear", R_LEFT)]
    return out


def demo():
    right, t_rub, t_off = _right_path()
    left = _stay("left") if ONE else _left_path(t_rub, t_off)
    return {"right": right, "left": left}
