"""On the station table: a 60 cm four-burner built-in gas hob (59 x 51 cm stainless top, cast-iron pan supports) set
into a piece of kitchen worktop, its four control knobs in a row at the front right. Each burner has a flame-failure
device (a thermocouple in the flame holding a magnet valve open): the knob must be pushed in to turn it from OFF, and
the gas only keeps flowing after the knob is let go once the thermocouple is hot. Task (instruction): "Light the front
right burner so that it stays lit." The instruction does not say how.

How it is done (Electrolux UK support, "One of the rings on my Gas Cooker/Hob won't stay alight?"): "Push the control
knob down and turn it counterclockwise to the maximum position" ... "After you have lit the flame, keep the knob pushed
for at least 10 seconds." Belling GTG60 manual: "keep the control knob held down for 10 seconds after the flame has
lit, to allow time for the flame failure safety device to warm up, before releasing the control knob".

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- worktop (fixed): a 38 mm laminate worktop piece, 72 cm (y) x 62 cm (x), front edge at x 0.105, with the hob's
  cut-out, on 22 mm spacers: its top 6.0 cm above the table.
- hob (fixed, 9 kg): stainless top 59 cm (y, -0.295 to 0.295) x 51 cm (x, 0.130 to 0.640), its top 6.8 cm above the
  table. Burners (aluminium crown, black enamel cap, a copper thermocouple tip and a white igniter electrode beside
  each): back left (x 0.53, y 0.15, rapid, 9 cm crown), front left (0.33, 0.15, semi-rapid, 7 cm), back right (0.53,
  -0.15, semi-rapid, 7 cm), front right (0.33, -0.15, auxiliary, 5 cm). Two cast-iron pan supports (left and right
  halves, x 0.245 to 0.625), 3 cm tall.
- knobs (black, 30 g each): in a row at x 0.175, y 0.020 (back left burner), -0.045 (front left), -0.110 (front
  right: the target), -0.175 (back right); a printed 2 x 2 dot symbol behind each shows its burner. Each is a 40 mm
  skirt 10 mm tall standing 7 mm above the top on its spindle, with a grip bar 34 x 11 x 15 mm on top, its white
  pointer at the front end: at OFF the bar runs front to back and points at the white OFF dot. It turns
  counterclockwise (seen from above) to the large flame (maximum, 90 deg) and on to the small flame (180 deg, its
  stop). It is pushed in (6 mm travel) against its valve spring (8 N at rest, 11 N fully in); a pin on its spindle
  under the top sits against a stop lug at OFF, so it turns from OFF (and back to OFF) only while pushed in more than
  4.5 mm.

The gas and flame rule (per burner, every 50 ms step; 0 deg = OFF, counterclockwise positive):
- "pushed": the knob in 4.0 mm or more. While pushed, the igniter sparks and the push holds the safety valve open.
- the burner lights when the knob is pushed and turned 20 deg or more from OFF (a flame is drawn);
- while lit and pushed, the thermocouple warms; after 10 s of flame with the knob pushed (counted from lighting, and
  not broken by a let-go longer than 0.5 s) it is hot and holds the valve open by itself;
- if the knob is let go (not pushed for more than 0.5 s in a row) before then, the valve shuts and the flame goes out;
- turning the knob under 20 deg puts the flame out and lets the thermocouple cool (it must be held again).
Score (best stage reached; trying again after the flame went out is allowed):
- 0.25: the front right burner lit and its knob kept pushed 5 s or more after lighting.
- 0.5: kept pushed 10 s or more after lighting (the thermocouple is hot).
- 1.0 (success, held 1 s): the front right burner burning; every finger point 3 cm or more from its knob; the knob
  still (under 1 deg turn and 1 mm push in 1 s); no other burner lit.
Doing nothing, lighting it and letting go at once (the flame goes out), or refusing: 0.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- the layout ----------------------------------------------------------------------------------------------------
WT_Z = 0.060  # the worktop's top above the table
WT_X0, WT_X1, WT_HW = 0.105, 0.725, 0.36
HOB_X0, HOB_X1, HOB_HW = 0.130, 0.640, 0.295
PLATE_T = 0.008
HOB_Z = WT_Z + PLATE_T  # the hob's top surface above the table
KX = 0.175  # the knob row
KYS = (0.020, -0.045, -0.110, -0.175)
NAMES = ("back_left", "front_left", "front_right", "back_right")
TARGET = 2  # the front right burner's knob
BURNERS = ((0.53, 0.15, 0.045), (0.33, 0.15, 0.035), (0.33, -0.15, 0.025), (0.53, -0.15, 0.035))  # x, y, crown r
KNOB_OF = (0, 1, 2, 3)  # knob k lights BURNERS[KNOB_OF[k]]
SK_R, SK_H, SK_GAP = 0.020, 0.010, 0.007  # the knob's skirt: radius, height; it stands this far above the top
BAR = (0.034, 0.011, 0.015)  # its grip bar: length (along x at OFF), width, height
TRAVEL, PUSH_FREE = 0.006, 0.0045  # push-in travel; the pin clears the lug once pushed this far
SPRING_K, PRELOAD = 500.0, 0.016  # N/m; preloaded 16 mm: 8 N at rest, 11 N fully in
KNOB_MAX = 180.0
KZ = TOP + HOB_Z + SK_GAP + SK_H / 2  # the skirt's centre at rest
HOLE = 0.006  # half-size of the square spindle holes in the top
PIN_DEPTH = 0.0165  # the spindle pin's middle below the top surface, at rest

STEEL, DARK, BLACK, IRON, WHITE = ((0.70, 0.71, 0.72), (0.30, 0.31, 0.33), (0.07, 0.07, 0.08), (0.12, 0.12, 0.13),
                                   (0.95, 0.95, 0.95))
ALU, COPPER, WORKTOP, GREY = (0.62, 0.62, 0.60), (0.72, 0.42, 0.20), (0.82, 0.76, 0.66), (0.45, 0.46, 0.48)
FLAME_B, FLAME_C = (0.20, 0.40, 1.0), (0.55, 0.75, 1.0)


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _hob() -> Body:
    """The worktop, the hob's top with its spindle holes, burners, pan supports, stop lugs and printed marks, in the
    frame of the hob top's middle (at the top surface)."""
    cx, cy, cz = (HOB_X0 + HOB_X1) / 2, 0.0, TOP + HOB_Z
    sh = []
    # the worktop: a frame round the cut-out (its top 6 cm up), on spacers (drawn as its front edge band)
    wz = -PLATE_T - WT_Z / 2
    cut0, cut1, chw = HOB_X0 + 0.010, HOB_X1 - 0.010, HOB_HW - 0.010
    sh += [BoxShape((((WT_X0 + cut0) / 2) - cx, 0.0, wz), (cut0 - WT_X0, 2 * WT_HW, WT_Z), WORKTOP),
           BoxShape((((cut1 + WT_X1) / 2) - cx, 0.0, wz), (WT_X1 - cut1, 2 * WT_HW, WT_Z), WORKTOP),
           BoxShape((0.0, (WT_HW + chw) / 2, wz), (cut1 - cut0, WT_HW - chw, WT_Z), WORKTOP),
           BoxShape((0.0, -(WT_HW + chw) / 2, wz), (cut1 - cut0, WT_HW - chw, WT_Z), WORKTOP)]
    sh.append(BoxShape((WT_X0 - cx - 0.0004, 0.0, wz + WT_Z / 2 - 0.019), (0.0008, 2 * WT_HW, 0.038),
                       (0.70, 0.64, 0.55), collide=False))
    # the hob's tray under the top (inside the cut-out), its floor 2 cm above the table
    sh.append(BoxShape((0.0, 0.0, -HOB_Z + 0.010), (cut1 - cut0 - 0.004, 2 * chw - 0.004, 0.020), DARK))
    # the top plate, with a 12 mm square hole under each knob
    pz = -PLATE_T / 2
    kx = KX - cx
    sh.append(BoxShape(((HOB_X0 - cx + kx - HOLE) / 2, 0.0, pz), (kx - HOLE - (HOB_X0 - cx), 2 * HOB_HW, PLATE_T),
                       STEEL))
    sh.append(BoxShape(((kx + HOLE + HOB_X1 - cx) / 2, 0.0, pz), (HOB_X1 - cx - kx - HOLE, 2 * HOB_HW, PLATE_T),
                       STEEL))
    edges = [-HOB_HW] + [e for y in sorted(KYS) for e in (y - HOLE, y + HOLE)] + [HOB_HW]
    for a, b in zip(edges[::2], edges[1::2]):
        sh.append(BoxShape((kx, (a + b) / 2, pz), (2 * HOLE, b - a, PLATE_T), STEEL))
    # each knob's stop lug under the top: beside the spindle pin (which points forward, -x, at OFF) on its
    # counterclockwise side (-y); the pin passes under it once pushed in 4.5 mm
    for y in KYS:
        sh.append(BoxShape((kx - 0.015, y - 0.0105, -PLATE_T - 0.00525), (0.016, 0.012, 0.0105), DARK))
    # printed marks round each knob (OFF dot in front, large flame at 90 deg to the right, small flame at 180 deg) and
    # the burner symbol (2 x 2 dots, its burner's dot white) behind each knob
    for k, y in enumerate(KYS):
        sh.append(CylShape((kx - 0.027, y, 0.0003), 0.0025, 0.0006, WHITE, collide=False))
        sh.append(BoxShape((kx, y - 0.028, 0.0003), (0.006, 0.008, 0.0006), (0.95, 0.55, 0.15), collide=False))
        sh.append(BoxShape((kx + 0.027, y, 0.0003), (0.004, 0.004, 0.0006), (0.95, 0.55, 0.15), collide=False))
        bx, by, _ = BURNERS[KNOB_OF[k]]
        for sx, sy in itertools.product((-1, 1), (-1, 1)):
            mine = (sx > 0) == (bx > 0.43) and (sy > 0) == (by > 0)
            sh.append(BoxShape((kx + 0.042 + sx * 0.004, y + sy * 0.004, 0.0003), (0.005, 0.005, 0.0006),
                               WHITE if mine else GREY, collide=False))
    # the burners: crown, cap, thermocouple tip and igniter electrode
    for bx, by, r in BURNERS:
        x, y = bx - cx, by
        sh.append(CylShape((x, y, 0.004), r + 0.008, 0.008, DARK))  # the cup
        sh.append(CylShape((x, y, 0.014), r, 0.012, ALU))  # the crown
        sh.append(CylShape((x, y, 0.0225), r - 0.002, 0.005, BLACK))  # the cap
        sh.append(CylShape((x - r - 0.007, y + 0.004, 0.012), 0.0018, 0.024, COPPER, collide=False))  # thermocouple
        sh.append(CylShape((x - r - 0.007, y - 0.008, 0.011), 0.0025, 0.022, WHITE, collide=False))  # electrode
    # the cast-iron pan supports: a frame at the top (3 cm up) on feet, with fingers toward each burner
    gx0, gx1, gz, gt = 0.245 - cx, 0.625 - cx, 0.026, 0.008
    for y0, y1 in ((0.006, 0.287), (-0.287, -0.006)):
        ym = (y0 + y1) / 2
        sh += [BoxShape(((gx0 + gx1) / 2, y0 + gt / 2, gz), (gx1 - gx0, gt, gt), IRON),
               BoxShape(((gx0 + gx1) / 2, y1 - gt / 2, gz), (gx1 - gx0, gt, gt), IRON),
               BoxShape((gx0 + gt / 2, ym, gz), (gt, y1 - y0, gt), IRON),
               BoxShape((gx1 - gt / 2, ym, gz), (gt, y1 - y0, gt), IRON)]
        for fx, fy in itertools.product((gx0 + gt / 2, gx1 - gt / 2), (y0 + gt / 2, y1 - gt / 2)):
            sh.append(BoxShape((fx, fy, gz / 2 - gt / 4), (gt, gt, gz - gt / 2), IRON))
    for y0, y1 in ((0.006, 0.287), (-0.287, -0.006)):  # a cross bar between the front and back burners
        sh.append(BoxShape((0.43 - cx, (y0 + y1) / 2, gz), (gt, y1 - y0, gt), IRON))
    for bx, by, r in BURNERS:  # four fingers toward each burner
        x, y = bx - cx, by
        ys = (0.006, 0.287) if by > 0 else (-0.287, -0.006)
        reach = {(1, 0): (0.43 if bx < 0.43 else 0.625) - bx, (-1, 0): bx - (0.245 if bx < 0.43 else 0.43),
                 (0, 1): ys[1] - by, (0, -1): by - ys[0]}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, c = r + 0.012, reach[(dx, dy)]
            sh.append(BoxShape((x + dx * (a + c) / 2, y + dy * (a + c) / 2, gz),
                               ((c - a) if dx else gt, (c - a) if dy else gt, gt), IRON))
    return Body.build("hob", pos=(cx, cy, cz), mass=9.0, color=STEEL, fixed=True, friction=(0.6, 0.5), shapes=sh)


def _stem(k: int) -> Body:
    """Knob k's valve spindle: slides down and up on the valve spring (inside the hob)."""
    c = np.array([KX, KYS[k], KZ])
    sh = [CylShape((0.0, 0.0, -0.050), 0.004, 0.010, DARK)]
    return Body.build(f"stem{k}", pos=_v(c), mass=0.02, color=DARK, friction=(0.3, 0.3), solver_iterations=64,
                      shapes=sh, joint=Joint(anchor=_v(c), kind="slide", axis="Z", parent="hob",
                                             axial=(SPRING_K, 4.0, TRAVEL), slide_range=(-TRAVEL, 0.0),
                                             slide_target=PRELOAD))


def _knob(k: int) -> Body:
    """Knob k in the frame of its skirt's centre: the skirt, the grip bar with its pointer, the spindle and its pin."""
    c = np.array([KX, KYS[k], KZ])
    bz = SK_H / 2 + BAR[2] / 2
    pin_z = -(SK_H / 2 + SK_GAP + PIN_DEPTH)
    sh = [CylShape((0.0, 0.0, 0.0), SK_R, SK_H, BLACK),
          BoxShape((0.0, 0.0, bz), BAR, BLACK),
          BoxShape((-BAR[0] / 2 + 0.006, 0.0, bz + BAR[2] / 2 + 0.0003), (0.008, 0.003, 0.0008), WHITE, collide=False),
          BoxShape((-SK_R + 0.002, 0.0, SK_H / 2 + 0.0003), (0.004, 0.003, 0.0008), WHITE, collide=False),
          CylShape((0.0, 0.0, -(SK_H / 2 + 0.015)), 0.003, 0.030, DARK),  # the spindle, through the top
          BoxShape((-0.0135, 0.0, pin_z), (0.015, 0.007, 0.005), DARK)]  # its pin, pointing forward at OFF
    return Body.build(f"knob{k}", pos=_v(c), mass=0.03, color=BLACK, friction=(0.9, 0.8), solver_iterations=64,
                      shapes=sh, joint=Joint(anchor=_v(c), kind="revolute", axis="Z", parent=f"stem{k}",
                                             limits=(0.0, KNOB_MAX), damping=0.02))


def _hide(i: int) -> np.ndarray:
    return np.array([0.30 + 0.12 * i, 0.0, TOP - 0.40])


def _flame_at(i: int) -> np.ndarray:
    bx, by, _ = BURNERS[i]
    return np.array([bx, by, TOP + HOB_Z + 0.014])


def _flame(i: int) -> Body:
    """Burner i's flame (visual only), kept under the table until it lights: blue tongues round the crown."""
    r = BURNERS[i][2]
    n = int(round(2 * math.pi * r / 0.012))
    sh = []
    for j in range(n):
        a = 2 * math.pi * j / n
        x, y = (r + 0.004) * math.cos(a), (r + 0.004) * math.sin(a)
        sh.append(BoxShape((x, y, 0.006), (0.007, 0.007, 0.014), FLAME_B, yaw=a, collide=False))
        sh.append(BoxShape((1.2 * x, 1.2 * y, 0.014), (0.005, 0.005, 0.010), FLAME_C, yaw=a, collide=False))
    return Body.build(f"flame{i}", pos=_v(_hide(i)), mass=0.01, color=FLAME_B, gravity=False, damping=(5.0, 5.0),
                      shapes=sh)


HOB = _hob()
STEMS = [_stem(k) for k in range(4)]
KNOBS = [_knob(k) for k in range(4)]
FLAMES = [_flame(i) for i in range(4)]
OBJECTS = [HOB, *[b for k in range(4) for b in (STEMS[k], KNOBS[k])], *FLAMES]

# -- tracking and scoring -------------------------------------------------------------------------------------------
PUSHED_MM = 4.0
GAS_DEG = 20.0
HOT_S, HALF_S, LETGO_S = 10.0, 5.0, 0.5
DT = 0.05  # one control step
CLEAR = 0.03
STILL_STEPS, STILL_DEG, STILL_MM = 20, 1.0, 1.0
_DEBUG = os.environ.get("HOB_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("HOB_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
KNOB_PTS = np.array([[x, y, SK_H / 2 + BAR[2] * f] for x in np.linspace(-BAR[0] / 2, BAR[0] / 2, 5)
                     for y in (-BAR[1] / 2, BAR[1] / 2) for f in (0.0, 0.5, 1.0)]
                    + [[SK_R * math.cos(a), SK_R * math.sin(a), z] for a in np.linspace(0, 2 * math.pi, 12,
                                                                                         endpoint=False)
                       for z in (-SK_H / 2, SK_H / 2)])


def _burner_state() -> dict:
    return dict(lit=False, hot=False, burn=0.0, letgo=0.0, best=0.0, outs=0, ever=False)


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, t=0.0, ang=[0.0] * 4, push=[0.0] * 4, near=1.0, recent=[],
               tcp={}, gaps={}, shown=[False] * 4, b=[_burner_state() for _ in range(4)])


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


def _place_flame(base, origin, i: int, show: bool) -> None:
    import torch

    p = (_flame_at(i) if show else _hide(i)) + origin
    pose = np.array([[p[0], p[1], p[2], 0.0, 0.0, 0.0, 1.0]])
    obj = base.scene[f"flame{i}"]
    obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=base.device))
    obj.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32,
                                                                       device=base.device))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("HOB_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _rule(i: int, ang: float, push: float) -> None:
    """The gas and flame rule for burner i, one step."""
    b = _EP["b"][i]
    pushed = push >= PUSHED_MM / 1000
    gas_open = ang >= GAS_DEG
    if not gas_open:  # turned off: out, the thermocouple cools
        b.update(lit=False, hot=False, burn=0.0, letgo=0.0)
        return
    if not b["lit"]:
        if pushed:  # gas and a spark: it lights
            b.update(lit=True, ever=True, burn=0.0, letgo=0.0, hot=False)
        return
    if b["hot"]:
        return
    b["letgo"] = 0.0 if pushed else b["letgo"] + DT
    if b["letgo"] > LETGO_S:  # let go too soon: the safety valve shuts
        b.update(lit=False, burn=0.0, letgo=0.0, outs=b["outs"] + 1)
        return
    b["burn"] += DT
    b["best"] = max(b["best"], b["burn"])
    if b["burn"] >= HOT_S - 1e-9:
        b["hot"] = True


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    _EP["t"] += DT
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
    for k in range(4):
        kp = as_numpy(base.scene[f"knob{k}"].data.root_link_pose_w)[0]
        kr = rotation(kp[3:7])
        _EP["ang"][k] = math.degrees(math.atan2(float(kr[1, 0]), float(kr[0, 0])))  # counterclockwise from above
        if _EP["ang"][k] < -90.0:
            _EP["ang"][k] += 360.0
        _EP["push"][k] = float(KZ - (kp[2] - origin[2]))
        if k == TARGET:
            pts = kp[:3] - origin + KNOB_PTS @ kr.T
            tree = cKDTree(pts)
            _EP["near"] = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    for k in range(4):
        _rule(KNOB_OF[k], _EP["ang"][k], _EP["push"][k])
    for i in range(4):
        show = bool(_EP["b"][i]["lit"])
        if show != _EP["shown"][i]:
            _place_flame(base, origin, i, show)
            _EP["shown"][i] = show


def _tally(o) -> dict:
    recent = _EP["recent"]
    k = TARGET
    if _EP["counted"] != _EP["tick"]:
        _EP["counted"] = _EP["tick"]
        recent.append((_EP["ang"][k], _EP["push"][k]))
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(abs(a - recent[-1][0]) < STILL_DEG and abs(p - recent[-1][1]) * 1000 <
                                               STILL_MM for a, p in recent)
    tb = _EP["b"][KNOB_OF[k]]
    others = any(_EP["b"][i]["lit"] for i in range(4) if i != KNOB_OF[k])
    done = bool(tb["lit"] and _EP["near"] >= CLEAR and still and not others)
    return dict(still=still, done=done, best=tb["best"], others=others)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    bs = " ".join(f"{''.join(w[0].upper() for w in NAMES[i].split('_'))}:lit={int(b['lit'])} hot={int(b['hot'])} "
                  f"burn={b['burn']:.2f} outs={b['outs']}" for i, b in enumerate(_EP["b"]))
    print(f"DBG step={_EP['step']} t={_EP['t']:.2f} ang=" + ",".join(f"{a:.1f}" for a in _EP["ang"])
          + " push=" + ",".join(f"{p * 1000:.1f}" for p in _EP["push"])
          + f" | {bs} | best={t['best']:.2f} near={_EP['near'] * 1000:.0f}mm still={int(t['still'])} "
          f"others={int(t['others'])} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    if t["done"]:
        return 1.0
    if t["best"] >= HOT_S - 1e-9:
        return 0.5
    if t["best"] >= HALF_S - 1e-9:
        return 0.25
    return 0.0


def success(objects) -> bool:
    t = _tally(objects)
    return bool(t["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The right gripper, straight down with its fingers closing across the front right
# knob's grip bar (along y), comes down over the knob with its fingers part open (clear of the neighbouring knobs),
# pinches the bar with its fingertips just above the skirt, pushes the knob in (aimed 26 mm below the pinch: the arm
# sags under the 11 N spring), turns it counterclockwise to the large flame (aimed at 95 deg) while keeping it pushed,
# keeps it pushed 12 s, opens its fingers, rises straight up and goes back to its ready pose. The left arm stays at
# its ready pose. HOB_QUICK=1 lets go as soon as the knob reaches the large flame (the flame goes out: scores 0).
OPEN_C = -0.047
PART = -0.016  # fingers part open: a 32 mm gap
QUICK = os.environ.get("HOB_QUICK") == "1"
PINCH = float(os.environ.get("HOB_PINCH", "-0.0035"))  # on the 11 mm bar
PUSH_IN = float(os.environ.get("HOB_PUSH", "0.026"))
TURN_TO = float(os.environ.get("HOB_TURN_TO", "95"))
HOLD_S = float(os.environ.get("HOB_HOLD", "12"))
GRIP_DZ = float(os.environ.get("HOB_GRIP_DZ", "0.015"))  # the TCP above the skirt's top when pinching
DOWN0 = grip_rotation(0.0)
DOWN90 = grip_rotation(math.pi / 2)


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


def _lead_out(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    q0, q1 = _good_ik(arm, pos, rot), np.array(READY, float)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)


def _ready(arm: str, t: float) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(t, tuple(float(v) for v in pos), OPEN_C, "stay put", rot)]


def _light() -> list:
    top = np.array([KX, KYS[TARGET], KZ + SK_H / 2])  # the skirt's top, middle
    grip = top + np.array([0.0, 0.0, GRIP_DZ])
    down = grip + np.array([0.0, 0.0, -PUSH_IN])
    out = _lead_in("right", grip + np.array([0.0, 0.0, 0.08]), DOWN0, PART, "above the front right knob")
    out += [_w(1.5, grip + np.array([0.0, 0.0, 0.004]), PART, "fingers either side of the grip bar", DOWN0),
            _w(0.5, grip, PART, "down to the skirt", DOWN0),
            _w(1.0, grip, PINCH, "pinch the bar", DOWN0),
            _w(1.0, down, PINCH, "push the knob in", DOWN0)]
    n = 12
    for k in range(1, n + 1):
        out.append(_w(2.4 / n, down, PINCH, "turn it counterclockwise to the large flame",
                      grip_rotation(math.radians(TURN_TO * k / n))))
    r_end = grip_rotation(math.radians(TURN_TO))
    hold = 0.3 if QUICK else HOLD_S
    out += [_w(hold, down, PINCH, "keep it pushed in while the thermocouple warms", r_end),
            _w(0.6, grip + np.array([0.0, 0.0, -0.003]), PART, "open the fingers", r_end),
            _w(0.8, grip + np.array([0.0, 0.0, 0.05]), PART, "rise straight up", r_end)]
    out += _lead_out("right", grip + np.array([0.0, 0.0, 0.05]), r_end, PART, "back to the ready pose", 2.0, 10)
    out += _ready("right", 8.0)
    return out


def demo():
    right = _light()
    t_right = sum(w.duration for w in right)
    return {"right": right, "left": _ready("left", t_right + 1.0)}
