"""On the station table, a guillotine paper cutter with its blade raised, and a white card on its board.

Task (instruction): trim the card so that 8 cm of it is left, lining its left edge up with the red line on the board.
The instruction does not say how to hold it. Trimming a card, a photo or a sheet on a guillotine trimmer is an
everyday job at home, at school and in the office, and paper-cutter safety sheets (the University of Washington's, for
example) give the way to do it: maintain a firm grip on the blade handle, keep your fingers behind the guard at all
times, never let your hand go underneath the blade, and after cutting place the handle in the down position. One hand
holds the card flat on the board, well back from the blade, and the other brings the blade down by its handle.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- cutter: fixed. A board 24 x 22 cm, 1.5 cm thick, from x 0.20 to 0.44 and y -0.12 to 0.10, with a steel cutting edge
  along its right side (y -0.12) and a finger guard, a bar 4 mm wide raised 4 mm off the board, 1.5 cm inside it;
  thin grey lines every 2 cm parallel to the cutting edge and a red line 8 cm from it (y -0.04). The blade's arm is
  hinged at the back right corner (x 0.43, 1 cm above the board) and starts raised 35 degrees; the blade under it, its
  edge 2.2 cm below the arm's axis, passes just outside the cutting edge (1.2 cm below the board's top when the arm is
  level), so that it cuts from the hinge outward as it comes down. Its handle, a black plastic knob 2 cm across and
  4.5 cm long, is at the arm's free end, 21 cm from the hinge, just right of the cutting edge (y -0.145). The arm stays
  where it is left unless pushed with more than 0.25 N m at the hinge (its tension spring), and stops level with the
  board. The board is smooth laminate (friction 0.12).
- card: white 300 g/m2 cardstock, 10 x 10 cm, 0.38 mm thick, 3 g, square to the board, centred at x 0.32, y -0.02 (its
  right edge 5 cm left of the cutting edge).

The card's resistance to the cut (a rule standing in for the card, which the simulation does not cut in two): while
the blade's edge passes through card lying across the cutting edge, the hinge resists with the torque that cutting it
takes. By the work balance, that torque is the card's work of cutting per unit area of cut, J_C = 4.9 kJ/m2 (paper cut
with scissors: Mahvash et al. 2008, IEEE Trans. Biomed. Eng. 55(3):848-856, Table I), times its thickness, times how far
the cut moves along the cutting edge per radian the blade comes down (which grows as the blade nears the board). For the
card in place, 6 to 16 cm from the hinge, the blade cuts it between 11.7 and 4.3 degrees, and the torque rises from
0.5 to 3.9 N m (0.19 J in all), on top of the spring's 0.25 N m: about 3.6 to 20 N at the knob, at right angles to the
arm. Card already cut does not resist again.

Score: 0 for the rest of the run once, with the blade up (2 degrees or more), any part of a gripper that is not holding
the handle (its fingers, its body or the wrist link it turns on) has come within 4 cm of the cutting edge, on the board
or past the edge (y -0.16 to -0.08, x 0.18 to 0.46, up to 10 cm above the board). A gripper holds the handle when both
its fingers touch the knob (within 3 mm of it) from opposite sides of it, 120 degrees or more apart around it. Otherwise
the best stage reached: 0.25 the card in place (flat on the board: every corner of its underside within 2 mm of the
board and tilted 2 degrees at most; the midpoint of its left edge within 5 mm of the red line; square within 4 degrees;
its right edge past the cutting edge); 0.5 the cut: the blade brought down from 15 degrees or more to below 2 degrees
with the card in place all the way; 0.75 after the cut, the blade down with every part of both grippers 1 cm or more
from the knob (the handle left down). Full score: cut, the piece the cut leaves 8 cm wide within 5 mm at both ends,
the handle left down, that piece flat where it was cut (the midpoint of the card's left edge still within 5 mm of the
red line, square within 4 degrees), both grippers' centres 4 cm or more above the board (the card let go of), the card
(within 1 mm) and the blade (within 1 degree) still for 1 s. The simulated card is not cut in two: the piece is the
part of it between its left edge and the cutting edge when the blade comes down, its width read at the card's two ends.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, Link, Linkage, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
                     on_step, rotation)
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps the best stage reached itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"
# CUTTER_CARD_TILT=1 (a check of the scoring, not the task): a 5 mm strip lies on the board under the card's left edge
# and the card starts on it, already lined up with the red line but tilted (see the demo's notes).
CARD_TILT = os.environ.get("CUTTER_CARD_TILT") == "1"
# CUTTER_CARD_SKEW=1 (a check of the scoring, not the task): the card starts flat and passes the alignment checks (the
# midpoint of its left edge 3 mm from the red line, turned 3.5 degrees), but the piece a cut leaves is 7.4 cm wide at
# one end (see the demo's notes).
CARD_SKEW = os.environ.get("CUTTER_CARD_SKEW") == "1"

# -- layout (world frame) ---------------------------------------------------------------------------------------------
BX0, BX1, BY0, BY1, BT = 0.20, 0.44, -0.12, 0.10, 0.015  # the board: x, y extent and thickness
BOARD_TOP = TOP + BT
YB = BY0  # the cutting edge, along x
GUARD_Y, RED_Y = YB + 0.015, YB + 0.08
HINGE = np.array([0.43, YB - 0.005, BOARD_TOP + 0.010])
ARM_L, OPEN_DEG, KNOB_Y, KNOB_R, KNOB_LEN = 0.21, 35.0, YB - 0.025, 0.010, 0.045
KNOB_MU = (0.8, 0.7)  # the knob's surface: moulded plastic (static, dynamic friction)
T_BLADE = 0.25  # N m: the tension spring's hold, as Coulomb friction at the hinge
BLADE_D = 0.022  # m: the blade's edge below the arm's axis (12 mm below the board's top when the arm is level)
CARD_W, CARD_T, CARD_KG = 0.10, 0.00038, 0.003  # 300 g/m2 cardstock, 0.38 mm thick
# The card's resistance to the cut (a material rule: the simulated card is not cut in two). For a blade turning on a
# pivot past a fixed edge, as in scissors, the work balance gives the torque needed to cut a sheet: its work of cutting
# per unit area of cut (J_C) times its thickness times how far the cut advances per radian the blade turns. Here the cut
# is at X, where the blade's edge crosses the board's top at the cutting edge, measured from the hinge:
# X = (BLADE_D - h cos a) / sin a, h the hinge's height above the board, so the torque is
# J_C * CARD_T * (BLADE_D cos a - h) / sin(a)^2. It acts at the hinge, as added Coulomb friction, while X is inside the
# card lying on the board at the cutting edge, and only where that card is still uncut.
J_C = 4900.0  # J/m^2: paper, from the work of cutting it with scissors (Mahvash et al. 2008, Table I)
RESIST_SCALE = float(os.environ.get("CUTTER_RESIST_SCALE", "1"))  # CUTTER_RESIST_SCALE=2: the resistance doubled
LIFT_FREE = 0.5  # degrees: the blade lifted this far above the lowest it has cut to moves freely again
# CUTTER_PROBE=<N m>@<high>-<low> (a check of the arm, not the task): that much added hinge friction while the blade is
# between those angles (degrees), whatever the card does, to see how hard the right arm can turn the blade.
_PROBE = (tuple(float(v) for v in os.environ["CUTTER_PROBE"].replace("@", " ").replace("-", " ", 1).split())
          if os.environ.get("CUTTER_PROBE") else None)
REST_Z = BOARD_TOP + CARD_T / 2  # the card's centre when it lies flat on the board
CARD_START = np.array([0.32, -0.02, REST_Z])
STRIP_H, STRIP_Y = 0.005, (-0.050, -0.042)  # CUTTER_CARD_TILT only: the strip's height and its y extent
if CARD_TILT:  # lined up with the red line, resting on the strip's near edge and on the cutting edge
    _TILT = math.atan2(STRIP_H, STRIP_Y[0] - YB)
    _CY = RED_Y - CARD_W / 2
    CARD_C0 = np.array([0.32, _CY, BOARD_TOP + ((_CY - YB) * math.sin(_TILT) + CARD_T / 2) / math.cos(_TILT) + 0.001])
    CARD_QUAT = (math.sin(_TILT / 2), 0.0, 0.0, math.cos(_TILT / 2))
elif CARD_SKEW:  # flat on the board, the midpoint of its left edge 3 mm right of the red line, turned 3.5 degrees
    _SKEW = math.radians(3.5)
    CARD_C0 = np.array([0.32, RED_Y - 0.003 - CARD_W / 2 * math.cos(_SKEW), REST_Z])
    CARD_QUAT = (0.0, 0.0, math.sin(_SKEW / 2), math.cos(_SKEW / 2))
else:
    CARD_C0, CARD_QUAT = CARD_START, None
WOOD, STEEL, BLACK, GREY, RED, WHITE = ((0.55, 0.36, 0.22), (0.74, 0.75, 0.78), (0.08, 0.08, 0.09),
                                       (0.42, 0.40, 0.38), (0.80, 0.10, 0.08), (0.96, 0.95, 0.92))
ORIGIN = np.array([0.5 * (BX0 + BX1), 0.5 * (BY0 + BY1), TOP])  # the cutter's frame: board centre, table level


def _rel(p) -> tuple:
    return tuple(float(v) for v in np.asarray(p, float) - ORIGIN)


def _base_shapes() -> list:
    cx, cy = 0.5 * (BX0 + BX1), 0.5 * (BY0 + BY1)
    out = [BoxShape(_rel((cx, cy, TOP + BT / 2)), (BX1 - BX0, BY1 - BY0, BT), WOOD)]
    out.append(BoxShape(_rel((cx, YB + 0.002, BOARD_TOP + 0.0005)), (BX1 - BX0, 0.004, 0.001), STEEL, collide=False))
    for k in range(1, 11):  # grey lines every 2 cm from the cutting edge; the red one at 8 cm
        y = YB + 0.02 * k
        if y >= BY1 - 0.005:
            break
        col, w = (RED, 0.0025) if abs(y - RED_Y) < 1e-6 else (GREY, 0.001)
        out.append(BoxShape(_rel((cx, y, BOARD_TOP + 0.0003)), (BX1 - BX0 - 0.02, w, 0.0006), col, collide=False))
    gx0, gx1 = BX0 + 0.02, BX1 - 0.03
    out.append(BoxShape(_rel((0.5 * (gx0 + gx1), GUARD_Y, BOARD_TOP + 0.010)), (gx1 - gx0, 0.004, 0.012), GREY))
    for gx in (gx0, gx1):  # the guard's two posts (visual), leaving 4 mm under the bar for paper
        out.append(BoxShape(_rel((gx, GUARD_Y, BOARD_TOP + 0.002)), (0.006, 0.004, 0.004), GREY, collide=False))
    out.append(BoxShape(_rel((BX1 - 0.006, 0.5 * (BY0 + BY1), BOARD_TOP + 0.005)), (0.012, BY1 - BY0, 0.010), GREY))
    out.append(BoxShape(_rel((HINGE[0], HINGE[1], BOARD_TOP + 0.005)), (0.03, 0.02, 0.02), BLACK, collide=False))
    if CARD_TILT:
        out.append(BoxShape(_rel((0.32, 0.5 * sum(STRIP_Y), BOARD_TOP + STRIP_H / 2)),
                            (0.12, STRIP_Y[1] - STRIP_Y[0], STRIP_H), GREY))
    return out


def _blade_shapes() -> list:
    """The arm in its starting pose, raised OPEN_DEG: a bar along it and the blade under it, its edge BLADE_D below the
    arm's axis (visual only, so that it passes through the card), and the knob at its free end (the only part that
    collides, for gripping)."""
    a = math.radians(OPEN_DEG)
    along = np.array([-math.cos(a), 0.0, math.sin(a)])  # from the hinge toward the handle
    below = np.array([-math.sin(a), 0.0, -math.cos(a)])  # the arm's underside
    q = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    mid = HINGE + along * ARM_L / 2
    knob = HINGE + along * ARM_L + [0.0, KNOB_Y - HINGE[1], 0.0]
    return [BoxShape(_rel(mid), (ARM_L, 0.012, 0.012), BLACK, collide=False, quat_xyzw=q),
            BoxShape(_rel(mid + below * (0.006 + BLADE_D) / 2), (ARM_L - 0.02, 0.002, BLADE_D - 0.006), STEEL,
                     collide=False, quat_xyzw=q),
            CylShape(_rel(knob), KNOB_R, KNOB_LEN, BLACK, axis="Y")]


CUTTER = Linkage.build(
    "paper_cutter", pos=tuple(ORIGIN), mass=3.0, color=WOOD, shapes=_base_shapes(), friction=(0.12, 0.10),
    contact_offset=0.001,
    links=[Link("blade", _blade_shapes(), mass=0.05, anchor=tuple(HINGE), axis="Y", friction=T_BLADE, material=KNOB_MU,
                limits=(-OPEN_DEG - 0.5, 75.0 - OPEN_DEG))])
CARD = Body.build("card", pos=tuple(float(v) for v in CARD_C0), mass=CARD_KG, color=WHITE, friction=(0.8, 0.7),
                  contact_offset=0.001, solver_iterations=24, init_quat=CARD_QUAT,
                  shapes=[BoxShape((0.0, 0.0, 0.0), (CARD_W, CARD_W, CARD_T), WHITE)])
OBJECTS = [CUTTER, CARD]

# -- scoring ----------------------------------------------------------------------------------------------------------
# The rule: while the blade is up, every part of a gripper that is not holding the handle stays CLEAR from the cutting
# edge's vertical plane, on the board or past the edge, over the board's length +-2 cm, up to ZONE_TOP above the board.
CLEAR, ZONE_X, ZONE_TOP = 0.04, (BX0 - 0.02, BX1 + 0.02), 0.10
BLADE_DOWN = 2.0  # degrees: the blade is down below this
CUT_FROM = 15.0  # degrees: the cut is the blade coming down from at least this high (above where it meets the card) with
# the card in place all the way
# The parts of each arm checked: both fingers, the gripper's body and the wrist link it turns on. Points every 2 mm or
# less over the surfaces of their collision hulls (the hulls the arm's URDF lists), placed each step by the links'
# simulated poses; one point is kept per 1.5 mm cube on the fingers and per 3 mm cube on the body and the wrist.
PARTS = {"tip_left": ("left finger", 0.0015), "tip_right": ("right finger", 0.0015), "gripper": ("body", 0.003),
         "link5": ("wrist", 0.003)}
TOUCH = 0.003  # a finger within this of the knob's surface touches it
OPPOSITE = 120.0  # degrees around the knob between the two fingers' nearest points: a grip from opposite sides
LET_GO = 0.01  # every part of both grippers at least this far from the knob: the handle left alone
EDGE_TOL, SQUARE = 0.005, math.radians(4.0)
# The piece the cut leaves must be 8 cm wide within 5 mm at both ends. The simulated card is not cut in two, so the
# piece is a proxy: the part of the card between its left edge and the cutting edge when the blade comes down, its
# width read at the card's two ends (the two corners of its left edge). A real trial cuts the card and measures it.
PIECE_W, PIECE_TOL = 0.08, 0.005
FLAT_GAP, FLAT_TILT = 0.002, 2.0  # m, degrees
CLEAR_UP = 0.04
STILL_STEPS, STILL_MM = 20, 1.0
CARD_CORNERS = np.array([[sx * CARD_W / 2, sy * CARD_W / 2, -CARD_T / 2] for sx in (-1, 1) for sy in (-1, 1)])
# Printed only (CUTTER_DEBUG=1), never scored: the first version's checks, to show what they would have given. A
# gripper whose centre was within 4 cm of the knob counted as on the handle; the others' centres had to stay 5 cm from
# the cutting edge (or anywhere past it) below 10 cm while the blade was 2 to 30 degrees up; the card counted as in
# place when its centre was within 3 mm of its resting height (no flatness check), and cut at any moment it was in
# place with the blade below 2 degrees; full score needed both centres 4 cm above the board and 4 cm from the knob.
OLD_ON_KNOB, OLD_CLEAR, OLD_HIGH, OLD_LIFT = 0.04, 0.05, 30.0, 0.003
_DEBUG = os.environ.get("CUTTER_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("CUTTER_DEBUG_EVERY", "10"))
_BLADE_LOG = os.environ.get("CUTTER_BLADE_LOG") == "1"  # with CUTTER_DEBUG=1: the blade's line every 2 control steps
_EP: dict = {}
_TCP: dict = {}
_ARM: dict = {}
_IDS: dict = {}
_JOINT: dict = {}
_PTS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, best=0.0, failed="", recent=[], angle=OPEN_DEG, armed=False, cut=False,
               fric=None, low=90.0, low_pose=None, resist=0.0, chord=None, hinge_tq=0.0, peak_tq=0.0, peak_all=0.0,
               peak_resist=0.0,
               work=0.0, prev_angle=OPEN_DEG,
               card_ok=False, flat=False, let_go=False, knob=None, post_ok=False, piece_w=(0.0, 0.0), cut_w=None,
               old=dict(failed=False, cut=False, best=0.0, card_ok=False, exempt=[], near=99.0, held=0, success=False))
    _TCP.clear()
    _ARM.clear()


_reset()


def _yaw(q) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _knob(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return HINGE + ARM_L * np.array([-math.cos(a), 0.0, math.sin(a)]) + [0.0, KNOB_Y - HINGE[1], 0.0]


def _knob_dist(w: np.ndarray, knob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each point's distance (m) from the knob's surface (negative inside it), and its angle (degrees) around the
    knob's axis (along y)."""
    dx, dz = w[:, 0] - knob[0], w[:, 2] - knob[2]
    dr = np.hypot(dx, dz) - KNOB_R
    da = np.abs(w[:, 1] - knob[1]) - KNOB_LEN / 2
    out = np.where((dr > 0) | (da > 0), np.hypot(np.maximum(dr, 0.0), np.maximum(da, 0.0)), np.maximum(dr, da))
    return out, np.degrees(np.arctan2(dz, dx))


H_HINGE = float(HINGE[2] - BOARD_TOP)  # the hinge's axis above the board


def _cut_at(angle_deg: float) -> tuple[float, float]:
    """Where the blade's edge crosses the cutting edge (the board's top) with the blade at this angle: its distance X
    from the hinge along the cutting edge (m; inf when the edge does not reach the board), and dX/da (m per radian the
    blade comes down)."""
    a = math.radians(angle_deg)
    if a <= 1e-4 or BLADE_D - H_HINGE * math.cos(a) <= 0.0:
        return math.inf, 0.0
    s = math.sin(a)
    return (BLADE_D - H_HINGE * math.cos(a)) / s, (BLADE_D * math.cos(a) - H_HINGE) / (s * s)


def _angle_at(x: float) -> float:
    """The blade's angle (degrees) at which its edge crosses the cutting edge X from the hinge: X sin a + h cos a =
    BLADE_D."""
    return math.degrees(math.asin(BLADE_D / math.hypot(x, H_HINGE)) - math.atan2(H_HINGE, x))


def _chord(cpos: np.ndarray, rot: np.ndarray):
    """The part of the cutting edge's line (y = YB) the card covers while lying on the board there: its two ends'
    distances from the hinge (m, near and far); None when the card does not reach across the line or is not down on the
    board at it (its mid-plane more than 3 mm from where a card lying flat has it)."""
    lo, hi = -math.inf, math.inf
    for k in (0, 1):  # inside the card's two pairs of sides: |along its axis k| <= CARD_W / 2
        ax, ay = rot[0, k], rot[1, k]
        b = -cpos[0] * ax + (YB - cpos[1]) * ay
        if abs(ax) < 1e-9:
            if abs(b) > CARD_W / 2:
                return None
            continue
        s0, s1 = sorted(((-CARD_W / 2 - b) / ax, (CARD_W / 2 - b) / ax))
        lo, hi = max(lo, s0), min(hi, s1)
    if hi - lo < 0.002:
        return None
    for x in (lo, hi):  # the card's mid-plane at the chord's ends
        local = rot.T @ (np.array([x, YB, cpos[2]]) - cpos)
        z = float((cpos + rot @ np.array([local[0], local[1], 0.0]))[2])
        if abs(z - REST_Z) > 0.003:
            return None
    return HINGE[0] - hi, HINGE[0] - lo


def _resistance(angle: float, cpos: np.ndarray, rot: np.ndarray) -> float:
    """The card's resistance to the blade at this angle (N m at the hinge, see J_C), where the card is still uncut: the
    lowest the blade has come with the card where it lies now, plus LIFT_FREE, bounds the part already cut."""
    pose = np.array([cpos[0], cpos[1], math.degrees(math.atan2(rot[1, 0], rot[0, 0]))])
    if _EP["low_pose"] is None or np.abs(pose[:2] - _EP["low_pose"][:2]).max() > 0.002 \
            or abs(pose[2] - _EP["low_pose"][2]) > 1.0:
        _EP.update(low=90.0, low_pose=pose)  # the card has moved: the blade meets uncut card
    chord = _chord(cpos, rot)
    _EP["chord"] = chord
    out = 0.0
    if chord is not None:
        x, dxda = _cut_at(angle)
        if chord[0] <= x <= chord[1] and angle <= _EP["low"] + LIFT_FREE:
            out = RESIST_SCALE * J_C * CARD_T * dxda
        _EP["low"] = min(_EP["low"], angle)
    return out


def _set_friction(cutter, torque: float) -> None:
    """The hinge's Coulomb friction (static and dynamic, N m), written to the simulation when it changes."""
    if _EP["fric"] is not None and abs(_EP["fric"] - torque) < 1e-4:
        return
    import torch

    val = torch.tensor([[torque]], dtype=torch.float32, device=cutter.device)
    cutter.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=val, joint_dynamic_friction_coeff=val,
                                                         joint_ids=[_JOINT["blade"]])
    _EP["fric"] = torque


def _hinge_torque(cutter) -> float:
    """The torque the hinge carries about its axis (N m: its friction, or its stop), as PhysX reports it."""
    arr = np.asarray(cutter.root_view.get_dof_projected_joint_forces().numpy(), float).reshape(-1)
    return float(arr[_JOINT["blade"]])


def _link_points() -> dict:
    """Points over the surfaces of each checked link's collision hulls, in the link's frame (see PARTS)."""
    if not _PTS:
        import xml.etree.ElementTree as ET

        from yam_arena import ASSET_DIR, YAM_URDF
        from yam_arena.analysis import read_stl
        from yam_arena.urdf import rpy_matrix

        for link in ET.parse(YAM_URDF).getroot().iter("link"):
            name = link.get("name")
            if name not in PARTS:
                continue
            pts = []
            for col in link.findall("collision"):
                o = col.find("origin")
                rot = rpy_matrix(np.array([float(v) for v in o.get("rpy", "0 0 0").split()])) if o is not None \
                    else np.eye(3)
                shift = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
                for a, b, c in read_stl(ASSET_DIR / col.find("geometry/mesh").get("filename")) @ rot.T + shift:
                    n = max(1, math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
                                         / 0.002))
                    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
                    keep = (i + j) <= n
                    pts.append(a + np.outer(i[keep] / n, b - a) + np.outer(j[keep] / n, c - a))
            pts = np.concatenate(pts)
            _, keep = np.unique(np.floor(pts / PARTS[name][1]).astype(np.int64), axis=0, return_index=True)
            _PTS[name] = pts[np.sort(keep)]
        missing = set(PARTS) - set(_PTS)
        if missing:
            raise RuntimeError(f"no collision hulls for {sorted(missing)} in {YAM_URDF}")
    return _PTS


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("CUTTER_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    """The blade's angle; for each gripper, its checked parts from the links' simulated poses (whether it holds the
    knob, how near it comes to the cutting edge and to the knob); the card's place and flatness; the rule."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1  # one control step: success() and score() both tally it, the history takes it once
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    cutter = base.scene["paper_cutter"]
    if not _JOINT:
        _JOINT["blade"] = list(cutter.joint_names).index("blade_turn")
    angle = OPEN_DEG + math.degrees(float(as_numpy(cutter.data.joint_pos)[0][_JOINT["blade"]]))
    knob = _knob(angle)
    pts = _link_points()
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            names = list(robot.body_names)
            _IDS[side] = (list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0]),
                          {link: next(i for i, n in enumerate(names) if n == link or n.endswith("/" + link))
                           for link in PARTS})
        arm_ids, body_ids = _IDS[side]
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = np.asarray(ik.tcp_pose(q[arm_ids], station.BASES[side])[0], float)
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        world = {link: pts[link] @ rotation(poses[i][3:7]).T + (poses[i][:3] - origin) for link, i in body_ids.items()}
        touch = []
        for link in ("tip_left", "tip_right"):
            d, ang = _knob_dist(world[link], knob)
            i = int(np.argmin(d))
            touch.append((float(d[i]), float(ang[i])))
        sep = abs((touch[0][1] - touch[1][1] + 180.0) % 360.0 - 180.0)
        holding = touch[0][0] <= TOUCH and touch[1][0] <= TOUCH and sep >= OPPOSITE
        gap, near, part = math.inf, math.inf, "-"
        for link, w in world.items():
            gap = min(gap, float(_knob_dist(w, knob)[0].min()))
            m = (w[:, 0] >= ZONE_X[0]) & (w[:, 0] <= ZONE_X[1]) & (w[:, 2] - BOARD_TOP <= ZONE_TOP)
            if m.any():
                dy = float(np.abs(w[m, 1] - YB).min())
                if dy < near:
                    near, part = dy, PARTS[link][0]
        _ARM[side] = dict(holding=holding, touch=touch, sep=sep, gap=gap, near=near, part=part)
        if angle >= BLADE_DOWN and not holding and near < CLEAR and not _EP["failed"]:
            _EP["failed"] = (f"the {side} gripper's {part} {near * 100:.1f} cm from the cutting edge with the blade "
                             f"{angle:.0f} degrees up, that gripper not holding the handle")
    card = as_numpy(base.scene["card"].data.root_link_pose_w)[0]
    cpos, cquat = card[:3] - origin, card[3:7]
    rot = rotation(cquat)
    _EP["angle"] = angle
    resist = _resistance(angle, cpos, rot)  # the card's resistance to the blade, for the next control step
    if _PROBE is not None:
        resist = _PROBE[0] if _PROBE[2] <= angle <= _PROBE[1] else 0.0
    if _EP["prev_angle"] > angle:  # the work the blade has done against the card coming down (J)
        _EP["work"] += _EP["resist"] * math.radians(_EP["prev_angle"] - angle)
    _EP["prev_angle"] = angle
    _EP["resist"] = resist
    _EP["peak_resist"] = max(_EP["peak_resist"], resist)
    _set_friction(cutter, T_BLADE + resist)
    if _BLADE_LOG and _EP["tick"] % 2 == 0:
        print(f"BLD tick={_EP['tick']} {_blade_line()} right_tcp_z={(_TCP.get('right', np.zeros(3))[2] - BOARD_TOP) * 100:.2f}cm",
              flush=True)
    if _DEBUG:
        tq = _hinge_torque(cutter)
        _EP["hinge_tq"] = tq
        _EP["peak_all"] = max(_EP["peak_all"], abs(tq))
        if angle > BLADE_DOWN - 1.0:  # not pressed onto its stop
            _EP["peak_tq"] = max(_EP["peak_tq"], abs(tq))
    lift = (cpos + CARD_CORNERS @ rot.T)[:, 2] - BOARD_TOP  # the underside's corners above the board
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))
    left_edge = float((cpos + rot @ np.array([0.0, CARD_W / 2, 0.0]))[1])  # the midpoints of its left and right edges
    right_edge = float((cpos + rot @ np.array([0.0, -CARD_W / 2, 0.0]))[1])
    yaw = _yaw(cquat)
    flat = float(np.abs(lift).max()) <= FLAT_GAP and tilt <= FLAT_TILT
    card_ok = flat and abs(left_edge - RED_Y) <= EDGE_TOL and abs(yaw) <= SQUARE and right_edge < YB
    # after the cut: the piece lies flat where it was cut (the card's right edge past the cutting edge is no longer asked:
    # on a real cutter that part is cut off)
    post_ok = flat and abs(left_edge - RED_Y) <= EDGE_TOL and abs(yaw) <= SQUARE
    piece_w = tuple(float((cpos + rot @ np.array([sx * CARD_W / 2, CARD_W / 2, 0.0]))[1] - YB) for sx in (-1, 1))
    let_go = min(a["gap"] for a in _ARM.values()) >= LET_GO
    _EP.update(angle=angle, knob=knob, card_ok=card_ok, flat=flat, lift=lift, tilt=tilt, left_edge=left_edge,
               cyaw=yaw, let_go=let_go, post_ok=post_ok, piece_w=piece_w)
    if not card_ok:  # the card must stay in place from CUT_FROM all the way down
        _EP["armed"] = False
    elif angle >= CUT_FROM:
        _EP["armed"] = True
    elif angle < BLADE_DOWN and _EP["armed"]:
        if not _EP["cut"]:
            _EP["cut_w"] = piece_w  # the width of the piece this cut leaves, at both ends
        _EP["cut"] = True
    stage = 0.25 if card_ok else 0.0
    if _EP["cut"]:
        stage = 0.5
        if angle < BLADE_DOWN and let_go:
            stage = 0.75
    if not _EP["failed"]:
        _EP["best"] = max(_EP["best"], stage)
    if _DEBUG:
        _old_rules(angle, knob, cpos, yaw)


def _old_rules(angle: float, knob: np.ndarray, cpos: np.ndarray, yaw: float) -> None:
    """The first version's checks (see OLD_ON_KNOB), printed only."""
    old = _EP["old"]
    exempt = [s for s, p in _TCP.items() if np.linalg.norm(p - knob) <= OLD_ON_KNOB]
    near = min((p[1] - YB for s, p in _TCP.items() if s not in exempt and ZONE_X[0] <= p[0] <= ZONE_X[1]
                and p[2] <= BOARD_TOP + ZONE_TOP), default=99.0)
    if BLADE_DOWN <= angle <= OLD_HIGH and near < OLD_CLEAR:
        old["failed"] = True
    card_ok = (abs(cpos[1] + CARD_W / 2 - RED_Y) <= EDGE_TOL and abs(yaw) <= SQUARE and cpos[1] - CARD_W / 2 < YB
               and abs(cpos[2] - REST_Z) < OLD_LIFT)
    if angle < BLADE_DOWN and card_ok:
        old["cut"] = True
    stage = 0.25 if card_ok else 0.0
    if old["cut"]:
        stage = 0.75 if angle < BLADE_DOWN and not exempt else 0.5
    old.update(best=max(old["best"], stage), card_ok=card_ok, exempt=exempt, near=near)


def _tally(o) -> dict:
    card = np.asarray(o["card"].pos, float)
    recent = _EP["recent"]
    new_tick = _EP["counted"] != _EP["tick"]
    if new_tick:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([card, [_EP["angle"] / 1000.0]]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    up = min((p[2] - BOARD_TOP for p in _TCP.values()), default=0.0) if len(_TCP) == 2 else 0.0
    down = _EP["angle"] < BLADE_DOWN
    piece_ok = _EP["cut_w"] is not None and all(abs(w - PIECE_W) <= PIECE_TOL for w in _EP["cut_w"])
    done = _EP["cut"] and piece_ok and down and _EP["let_go"] and _EP["post_ok"] and up >= CLEAR_UP and still
    if _DEBUG and new_tick and _EP["knob"] is not None:
        old = _EP["old"]
        off_knob = min(float(np.linalg.norm(p - _EP["knob"])) for p in _TCP.values())
        old_done = old["cut"] and down and up >= CLEAR_UP and off_knob >= OLD_ON_KNOB and still
        old["held"] = old["held"] + 1 if old_done else 0
        old["success"] = old["success"] or old["held"] >= STILL_STEPS
    return dict(still=still, up=up, down=down, done=bool(done))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    old = _EP["old"]
    old_score = 0.0 if old["failed"] else (1.0 if old["success"] else old["best"])
    lift = _EP.get("lift", np.zeros(4)) * 1000
    print(f"DBG step={_EP['step']} blade={_EP['angle']:.1f}deg card_left_edge={_EP.get('left_edge', 0) * 100:+.2f}cm "
          f"(red {RED_Y * 100:+.1f}) yaw={math.degrees(_EP.get('cyaw', 0.0)):+.1f} tilt={_EP.get('tilt', 0.0):.1f} "
          f"corners_mm=[{' '.join(f'{v:+.1f}' for v in lift)}] flat={_EP['flat']} card_ok={_EP['card_ok']} "
          f"piece_cm=({_EP['piece_w'][0] * 100:.2f},{_EP['piece_w'][1] * 100:.2f}) cut={_EP['cut']} cut_piece_cm="
          f"{'-' if _EP['cut_w'] is None else '(%.2f,%.2f)' % (_EP['cut_w'][0] * 100, _EP['cut_w'][1] * 100)} "
          f"post_ok={_EP['post_ok']} let_go={_EP['let_go']} up={t['up']:.3f} still={t['still']} best={_EP['best']} "
          f"failed={_EP['failed'] or '-'}", flush=True)
    for side, a in _ARM.items():
        p = _TCP[side]
        print(f"DBG   {side}: holding={a['holding']} fingers_to_knob_mm=({a['touch'][0][0] * 1000:.1f}, "
              f"{a['touch'][1][0] * 1000:.1f}) apart={a['sep']:.0f}deg knob_gap_mm={a['gap'] * 1000:.1f} "
              f"near_edge_cm={a['near'] * 100:.1f} ({a['part']}) tcp=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f})",
              flush=True)
    print(f"DBG   first version's rules (not scored): exempt={old['exempt'] or '-'} free_centre_near_cm="
          f"{old['near'] * 100:.1f} broken={old['failed']} card_ok={old['card_ok']} cut={old['cut']} "
          f"success={old['success']} score={old_score}", flush=True)
    print(f"DBG   blade: {_blade_line()}", flush=True)


def _blade_line() -> str:
    ch = _EP["chord"]
    where = "-" if ch is None else (f"{ch[0] * 100:.1f}-{ch[1] * 100:.1f}cm from the hinge, cut from "
                                    f"{_angle_at(ch[0]):.2f} to {_angle_at(ch[1]):.2f}deg")
    return (f"angle={_EP['angle']:.2f}deg resist={_EP['resist']:.3f}Nm friction={_EP['fric'] or 0.0:.3f}Nm "
            f"hinge_torque={_EP['hinge_tq']:+.3f}Nm peak_above_1deg={_EP['peak_tq']:.3f}Nm peak_all={_EP['peak_all']:.3f}Nm "
            f"peak_resist={_EP['peak_resist']:.3f}Nm work={_EP['work']:.4f}J low={_EP['low']:.2f}deg card_at_edge={where}")


def score(objects) -> float:
    """The best stage reached (0 for the rest of the run once, with the blade up, a part of a gripper not holding the
    handle has come within 4 cm of the cutting edge): 0.25 the card in place and flat; 0.5 the blade brought down from
    15 degrees or more with the card in place all the way (the cut); 0.75 the handle then left down. Full score (1.0) is
    success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["best"]


def success(objects) -> bool:
    """Cut with the card in place, the piece the cut leaves 8 cm wide within 5 mm at both ends, the handle left down (no
    part of either gripper within 1 cm of the knob), the piece flat where it was cut, both grippers' centres 4 cm or
    more above the board, the rule kept, the card and the blade still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left gripper, fingers closed (closing along x), presses on the card at its left edge, its centre over the edge,
# slides it right until that edge is on the red line and keeps pressing there: every part of it stays 4.8 cm or more
# from the cutting edge while the blade is up. The right gripper closes its fingers on the knob and brings the blade
# down along its arc, its wrist turning with the handle: quickly to 14 degrees, then slowly through the card, pulling
# the knob down against the card's resistance (in the scripted run the hinge carries up to 3.9 N m while the blade cuts),
# and presses the handle onto its stop; it lets go with the handle down and moves away; then the left gripper lifts off
# and moves away.
# CUTTER_RESIST_SCALE=2 (a check of the margin): the card resists twice as hard; the blade comes through more slowly
# and the run still scores 1.0 (the hinge carries up to 7.6 N m).
# CUTTER_HAND_NEAR=1 (the wrong way) presses the card 3.5 cm from the cutting edge instead: score 0 (a finger comes
# within 4 cm of the cutting edge while the blade is up).
# CUTTER_HOVER=1 (the first version's loophole): the right gripper lets go of the knob with the blade 17 degrees up,
# hovers beside it with its fingers open, 1.5 cm above it, for 2.3 s, then takes it again and finishes the cut: score
# 0 (it was not holding the handle there); the first version's rule counted a gripper within 4 cm of the knob as on it.
# CUTTER_CARD_TILT=1 (the other loophole): the card starts lined up with the red line but lying on a 5 mm strip, tilted
# 4 degrees; the left gripper stays away and the right one makes the cut: score 0 (the card is not flat); the first
# version only checked that the card's centre was within 3 mm of its resting height.
# CUTTER_CARD_LATE=1 (a cut that is not one): the right gripper brings the blade down first and moves away; only then
# does the left gripper slide the card into place, under the lowered blade (the blade passes through it here), and
# let go: score 0.25 (the card in place, never cut); the first version credited the cut as soon as the card was in
# place with the blade down, and gave 1.0.
# CUTTER_CARD_SKEW=1 (a piece of the wrong size): the card starts flat, the midpoint of its left edge 3 mm right of
# the red line and turned 3.5 degrees, which the alignment checks pass; the left gripper stays away and the right one
# makes the cut. The piece it leaves is 7.4 cm wide at one end, so the run scores 0.75 (cut, handle down, no full
# score); the previous version, which asked only for the card still in place after the cut, gave 1.0.

HAND_NEAR = os.environ.get("CUTTER_HAND_NEAR") == "1"
HOVER = os.environ.get("CUTTER_HOVER") == "1"
CARD_LATE = os.environ.get("CUTTER_CARD_LATE") == "1"
PRESS = 0.006  # the closed fingertips are aimed this far into the card, so that they press on it
CLOSED, OPEN = 0.0, -0.030
PINCH_KNOB = CLOSED  # closed on the knob: the fingers squeeze it with the gripper's full force
PUSH_DEG = 5.0  # the handle is pushed toward 5 degrees below level, onto its stop (at -0.5)
CARD_GOAL_Y = RED_Y - CARD_W / 2
HOLD_AT = 0.035 if HAND_NEAR else 0.080  # how far from the cutting edge the left gripper's centre presses, in the end
DOWN_L = grip_rotation(-math.pi / 2)  # straight down, fingers closing along x
DOWN_R = grip_rotation(math.pi / 2)
PARK_L = np.array([0.20, 0.12, BOARD_TOP + 0.12])  # CUTTER_CARD_TILT, CUTTER_CARD_SKEW: where the left gripper waits


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
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
    """From READY along a straight line in joint space (the wrist's big turn stays on one IK branch)."""
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


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


def demo():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    press_z = BOARD_TOP + CARD_T + PAD_TIP_PAST_TCP - PRESS
    goal = np.array([CARD_START[0], YB + HOLD_AT, press_z])
    start = goal + [0.0, CARD_START[1] - CARD_GOAL_Y, 0.0]  # the same point on the card, before it is slid
    lead = _lead_in("left", start + [0.0, 0.0, 0.04], DOWN_L, CLOSED, "above the card's left edge")
    slide = [_w(1.0, start, CLOSED, "press on the card at its left edge", DOWN_L),
             _w(3.0, goal, CLOSED, "slide it right until its left edge is on the red line", DOWN_L),
             _w(0.5, goal, CLOSED, "keep pressing it", DOWN_L)]
    left = lead + slide
    t_hold = 0.0 if CARD_LATE else sum(w.duration for w in left)
    knob0 = _knob(OPEN_DEG)
    right = _lead_in("right", knob0 + [0.0, 0.0, 0.035], DOWN_R, OPEN, "above the cutter's knob", seconds=3.0)
    right += [_w(max(0.5, t_hold - 3.0), knob0 + [0.0, 0.0, 0.035], OPEN, "wait for the card to be in place", DOWN_R),
              _w(1.0, knob0, OPEN, "fingers either side of the knob", DOWN_R),
              _w(0.8, knob0, PINCH_KNOB, "grip the knob", DOWN_R)]
    # bring the blade down along its arc, the wrist turning with the handle: to 14 degrees, then slowly through the card
    # (the blade cuts it between about 12 and 4 degrees), then press the handle down onto its stop
    for angle, secs in ((29.0, 0.6), (23.0, 0.6), (17.0, 0.6), (14.0, 0.4), (10.0, 0.8), (6.0, 0.8), (3.0, 0.8),
                        (-1.0, 0.8)):
        turn = OPEN_DEG - angle
        at, rot = _knob(angle), _ry(-turn) @ DOWN_R
        right.append(_w(secs, at, PINCH_KNOB, "bring the blade down" if angle > 14.0 else "cut the card", rot))
        if HOVER and angle == 17.0:  # let go with the blade part-way down, hover beside the knob, then take it again
            right += [_w(0.6, at, OPEN, "let go of the knob, the blade part-way down", rot),
                      _w(0.8, at + [0.0, 0.0, 0.015], OPEN, "hover beside the knob", rot),
                      _w(1.5, at + [0.0, 0.0, 0.015], OPEN, "hover beside the knob", rot),
                      _w(0.8, at, OPEN, "fingers either side of the knob again", rot),
                      _w(0.8, at, PINCH_KNOB, "grip the knob again", rot)]
    end, knob1 = _ry(-(OPEN_DEG + 1.0)) @ DOWN_R, _knob(-1.0)
    right += [_w(0.8, _knob(-PUSH_DEG), PINCH_KNOB, "press the handle down onto its stop", end),
              _w(0.5, _knob(-PUSH_DEG), PINCH_KNOB, "press the handle down onto its stop", end),
              _w(0.4, knob1, PINCH_KNOB, "ease off", end),
              _w(0.6, knob1, OPEN, "let go, the handle down", end),
              _w(0.8, knob1 + [0.0, 0.0, 0.06], OPEN, "rise", end),
              _w(1.5, knob1 + [-0.02, -0.04, 0.10], OPEN, "move away", DOWN_R),
              _w(3.5, knob1 + [-0.02, -0.04, 0.10], OPEN, "wait", DOWN_R)]
    t_cut = sum(w.duration for w in right) - 3.5 - 1.5 - 0.8
    left += [_w(max(0.5, t_cut - t_hold), goal, CLOSED, "keep pressing while the blade comes down", DOWN_L),
             _w(0.8, goal + [0.0, 0.0, 0.06], CLOSED, "lift off the card", DOWN_L),
             _w(1.5, goal + [-0.04, 0.06, 0.10], CLOSED, "move away", DOWN_L),
             _w(3.0, goal + [-0.04, 0.06, 0.10], CLOSED, "wait", DOWN_L)]
    if CARD_LATE:  # the card is slid into place only once the blade is down and the right gripper has moved away
        left = lead + [_w(t_cut + 2.3 - 3.0, start + [0.0, 0.0, 0.04], CLOSED, "wait above the card", DOWN_L)] + slide
        left += [_w(1.0, goal, CLOSED, "keep pressing it", DOWN_L),
                 _w(0.8, goal + [0.0, 0.0, 0.06], CLOSED, "lift off the card", DOWN_L),
                 _w(1.5, goal + [-0.04, 0.06, 0.10], CLOSED, "move away", DOWN_L),
                 _w(3.0, goal + [-0.04, 0.06, 0.10], CLOSED, "wait", DOWN_L)]
    if CARD_TILT or CARD_SKEW:  # the left gripper waits away from the card, which it never touches
        total = sum(w.duration for w in left)
        left = _lead_in("left", PARK_L, DOWN_L, CLOSED, "wait away from the card")
        left.append(_w(total - 3.0, PARK_L, CLOSED, "wait away from the card", DOWN_L))
    return {"left": left, "right": right}
