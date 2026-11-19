"""On the station table: a countertop immersion-element deep fryer (Presto ProFry style, 3 qt of oil), taken apart for
washing. Its stainless base stands fixed on the table with its back toward the robot; the washed, empty enamelled pot
stands on the table in front of the robot, and the control assembly (control unit with its immersion heating element)
stands on the table to the robot's left. Task (instruction): put the fryer back together so that it is ready to use.
The instruction does not say how.

How it is done (Presto Stainless Steel ProFry 05461 instructions, "How To Use for Frying"): "2. Place removable pot in
base. 3. Attach the control assembly to the base by aligning the two guides on the front of the control unit with the
two tracks on the back of the base. Slide the control assembly down as far as it goes making sure both guides are in
the tracks. Note: The appliance will only operate when both guides are properly positioned in the tracks." The pot is
lifted "grasping two opposite corners of the removable pot" (Care and Cleaning, 2).

Why the order is forced: the element hangs 11 cm down inside the pot and runs along its floor, so once the control
assembly is on the base the pot can no longer go in under it (FRYER_WRONG=ctrl_first). Why two hands for the pot: the
pot (27 x 17.5 cm, 0.65 kg) fits its base with 3 mm to spare all round, so it must go down level; pinched by one end
wall it tips and falls (FRYER_WRONG=one_arm).

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- base: fixed. Outer 20.5 (x) x 30.0 (y) cm, its rim 16.0 cm above the table, walls 1.2 cm, on 1 cm feet; back wall's
  outer face at x 0.300, centred on y 0. Black handles on both ends. On the outside of the back wall two vertical
  tracks (channels 8 mm wide between rails 5 mm wide and 6 mm deep, y centres +-0.025, from 6.5 to 15.0 cm above the
  table, the rails' top 1 cm leaning out at 45 deg as a funnel), each closed at its foot by a stop (the guides stand
  on it: the interlock). The wall's top outer edge is chamfered 7 mm.
- pot: loose, 0.65 kg. Outer 17.5 (x) x 27.0 (y) cm at the top, 1 cm less each way at the bottom (draft), 12.5 cm
  deep, walls 4 mm, a 9 mm flange round the top that rests on the base's rim (seated: centred at x 0.4025, y 0,
  3 mm from the base's walls all round). Starts on the table centred at (0.135, 0.000).
- control assembly: loose, 0.75 kg, centre of mass 4 mm in front of the guides. Control unit 5.8 (x) x 7.0 x 11.6 cm
  behind the base's back wall, under a head 11.2 x 7.0 x 2.9 cm that reaches 4.6 cm forward over the rim (thermostat
  knob and signal light on top, underside 3 mm above the pot's flange when seated); two guides (5 mm wide, 7.5 mm
  deep, 5 cm long, the lowest 1 cm narrowed to 3 mm) on the unit's front face; the element: two 9 mm tubes down from
  the head 4 cm in front of the wall, 1.6 cm inside the pot's back wall, and a 12.4 x 10.9 cm loop lying 8 mm above the
  pot's floor when seated. Starts standing on the table on its loop and the unit's foot, element pointing forward, the
  unit's back face at x 0.200, centred on y 0.260.

Score (best stage at the end of the run; SCORE_OVER = "end"):
- 0.25: the pot seated in the base (flange on the rim: within 3 mm of its place across, 2 mm in height, tilted < 2 deg).
- 0.5: and both guides of the control assembly in their tracks (inside the channels, below the rails' tops).
- 0.75: and the control assembly fully down (both guides within 2 mm of their stops, the assembly resting on the base
  with 3 N or more, PhysX contact force: the interlock is closed) with the whole element loop inside the pot.
- 1.0 (success, held 1 s): all of that, with no part of either gripper within 1 cm of the pot or the control assembly,
  both still (under 1 mm in 1 s), the element never having struck the pot (contact force over 5 N between the control
  assembly and the pot at any step caps the score at 0.75) and the control assembly never dropped (a fall faster than
  0.7 m/s or a tip over 45 deg caps the score at 0.5).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step, plate, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

STEEL, DARK, ENAMEL, ENAMEL_IN, GREY, RED, ELEM, WOOD = (
    (0.78, 0.79, 0.80), (0.10, 0.10, 0.11), (0.12, 0.12, 0.13), (0.30, 0.30, 0.32), (0.55, 0.56, 0.58),
    (0.85, 0.10, 0.08), (0.62, 0.60, 0.56), (0.55, 0.42, 0.30))

# -- the fryer base (fixed) ----------------------------------------------------------------------------------------
XB = 0.300  # the back wall's outer face
BASE_D, BASE_W, WALL, RIM, FOOT = 0.205, 0.300, 0.012, 0.160, 0.010
XBC = XB + BASE_D / 2  # the base's middle (x)
TRACK_Y = (0.025, -0.025)
GAP = 0.008  # each track's channel between its rails
RAIL_W, RAIL_D = 0.005, 0.006  # rail width (y), how far it stands off the wall (x)
STOP_Z, RAIL_TOP = 0.065, 0.150
FLARE = 0.010  # the rails' top 1 cm flares out at 45 deg  # the stops' tops and the rails' tops, above the table


def _base() -> Body:
    lx = lambda x: x - XBC  # noqa: E731  world x -> base frame
    h = RIM - FOOT
    zc = FOOT + h / 2
    sh = [BoxShape((0.0, 0.0, FOOT + 0.006), (BASE_D, BASE_W, 0.012), STEEL),  # floor
          # back wall: full thickness up to 1 cm below the rim, then its top 1 cm with the outer edge chamfered
          BoxShape((lx(XB + WALL / 2), 0.0, FOOT + (h - 0.010) / 2), (WALL, BASE_W, h - 0.010), STEEL),
          BoxShape((lx(XB + 0.007 + (WALL - 0.007) / 2), 0.0, RIM - 0.005), (WALL - 0.007, BASE_W, 0.010), STEEL),
          plate((lx(XB + 0.0014), 0.0, RIM - 0.010 + 0.0), (lx(XB + 0.0084), 0.0, RIM - 0.0014), BASE_W, 0.004,
                (1.0, 0.0, -1.0), STEEL),
          BoxShape((lx(XB + BASE_D - WALL / 2), 0.0, zc), (WALL, BASE_W, h), STEEL),  # front wall
          BoxShape((0.0, BASE_W / 2 - WALL / 2, zc), (BASE_D - 2 * WALL, WALL, h), STEEL),  # end walls
          BoxShape((0.0, -BASE_W / 2 + WALL / 2, zc), (BASE_D - 2 * WALL, WALL, h), STEEL),
          # black handles on the ends, a dark rim band
          BoxShape((0.0, BASE_W / 2 + 0.013, 0.125), (0.10, 0.026, 0.024), DARK),
          BoxShape((0.0, -BASE_W / 2 - 0.013, 0.125), (0.10, 0.026, 0.024), DARK)]
    for sx, sy in itertools.product((-1, 1), (-1, 1)):  # feet
        sh.append(CylShape((sx * (BASE_D / 2 - 0.02), sy * (BASE_W / 2 - 0.03), FOOT / 2), 0.012, FOOT, DARK))
    rl = (RAIL_TOP - STOP_Z)
    for ty in TRACK_Y:  # the two tracks on the back wall's outside: rails, a stop, a flared mouth
        for s in (-1, 1):
            yc = ty + s * (GAP / 2 + RAIL_W / 2)
            rl1 = rl - FLARE
            sh.append(BoxShape((lx(XB - RAIL_D / 2), yc, STOP_Z + rl1 / 2), (RAIL_D, RAIL_W, rl1), GREY))
            # the mouth: the rail's top 1 cm leans out at 45 deg (its inner face from the channel's edge), a funnel
            o = RAIL_W / (2 * math.sqrt(2))
            z0 = RAIL_TOP - FLARE
            sh.append(plate((lx(XB - RAIL_D / 2), ty + s * (GAP / 2 + o), z0 - o),
                            (lx(XB - RAIL_D / 2), ty + s * (GAP / 2 + o + FLARE), z0 + FLARE - o), RAIL_D, RAIL_W,
                            (0.0, -s, 1.0), GREY))
        sh.append(BoxShape((lx(XB - RAIL_D / 2), ty, STOP_Z - 0.004), (RAIL_D, GAP, 0.008), RED))  # the stop
    return Body.build("base", pos=(XBC, 0.0, TOP), mass=2.0, color=STEEL, fixed=True, shapes=sh,
                      friction=(0.4, 0.35))


# -- the pot (loose) -----------------------------------------------------------------------------------------------
POT_D, POT_W, POT_H, POT_T, FLANGE, POT_DRAFT = 0.175, 0.270, 0.125, 0.004, 0.009, 0.005
POT_SEAT = np.array([XB + WALL + 0.003 + POT_D / 2, 0.0, TOP + RIM - (POT_H - 0.003)])  # origin: outer bottom centre
POT_START = np.array([0.135, 0.0, TOP + 0.0005])
POT_MASS = 0.65


def _pot() -> Body:
    """Drawn with a 5 mm draft on every side (the bottom is 1 cm smaller each way than the top), so it centres itself
    as it goes down into the base."""
    d, w, hgt, t, dr = POT_D, POT_W, POT_H, POT_T, POT_DRAFT
    sh = [BoxShape((0.0, 0.0, t / 2), (d - 2 * dr, w - 2 * dr, t), ENAMEL)]
    for s in (-1, 1):
        sh.append(plate((s * (d / 2 - dr - t / 2), 0.0, 0.0), (s * (d / 2 - t / 2), 0.0, hgt), w - dr, t,
                        (1.0, 0.0, 0.0), ENAMEL))
        sh.append(plate((0.0, s * (w / 2 - dr - t / 2), 0.0), (0.0, s * (w / 2 - t / 2), hgt), d - dr - 2 * t, t,
                        (0.0, 1.0, 0.0), ENAMEL))
    sh += [  # the flange round the top (3 mm thick, 9 mm wide)
          BoxShape((-d / 2 - FLANGE / 2, 0.0, hgt - 0.0015), (FLANGE, w + 2 * FLANGE, 0.003), ENAMEL),
          BoxShape((d / 2 + FLANGE / 2, 0.0, hgt - 0.0015), (FLANGE, w + 2 * FLANGE, 0.003), ENAMEL),
          BoxShape((0.0, w / 2 + FLANGE / 2, hgt - 0.0015), (d, FLANGE, 0.003), ENAMEL),
          BoxShape((0.0, -w / 2 - FLANGE / 2, hgt - 0.0015), (d, FLANGE, 0.003), ENAMEL),
          # MIN / MAX marks inside the back wall (visual)
          BoxShape((-d / 2 + t + 0.004, 0.08, 0.045), (0.001, 0.03, 0.002), STEEL, collide=False),
          BoxShape((-d / 2 + t + 0.003, 0.08, 0.085), (0.001, 0.03, 0.002), STEEL, collide=False)]
    return Body.build("pot", pos=tuple(float(v) for v in POT_START), mass=POT_MASS, color=ENAMEL, shapes=sh,
                      friction=(0.5, 0.45), report_contacts=True, solver_iterations=24)


# -- the control assembly (loose) ------------------------------------------------------------------------------------
# Body frame = world when seated, shifted by (XB, 0, TOP): x 0 is the back wall's outer face, z 0 the table.
EL_X0, EL_X1, EL_Y, EL_YT, EL_R = 0.040, 0.155, 0.050, 0.020, 0.0045
EL_Z = RIM + 0.003 - POT_H + POT_T + 0.008 + EL_R  # the loop's tubes: 8 mm above the pot's floor when seated
NECK_Z0 = RIM + 0.006  # the head's underside, 3 mm above the pot's flange
UNIT_X0, UNIT_X1, UNIT_W, UNIT_Z0, UNIT_Z1 = -0.066, -0.008, 0.070, EL_Z - EL_R, NECK_Z0 + 0.029
GUIDE_W, GUIDE_X0, GUIDE_X1, GUIDE_Z1 = 0.005, -0.008, -0.0005, STOP_Z + 0.050
TIP_W, TIP_L = 0.003, 0.010
NECK_X1 = 0.046  # the head reaches forward over the rim
CTRL_MASS = 0.75
CTRL_COM = (0.004, 0.0, 0.120)  # just in front of the guides: seated, it leans on the rails
CTRL_START = np.array([0.200 - UNIT_X0, 0.260, TOP - UNIT_Z0 + 0.0005])  # the body origin at the start
CTRL_SEAT = np.array([XB, 0.0, TOP])
GUIDE_BOTTOMS = [np.array([-0.003, ty, STOP_Z]) for ty in TRACK_Y]
LOOP_PTS = [np.array([x, y, EL_Z - EL_R]) for x in (EL_X0 - EL_R, EL_X1 + EL_R) for y in (-EL_Y - EL_R, EL_Y + EL_R)]


def _ctrl() -> Body:
    ux = (UNIT_X0 + UNIT_X1) / 2
    bh = NECK_Z0 - UNIT_Z0
    sh = [BoxShape((ux, 0.0, UNIT_Z0 + bh / 2), (UNIT_X1 - UNIT_X0, UNIT_W, bh), DARK),  # the body behind the wall
          # the head: reaches forward over the rim and carries the element
          BoxShape(((UNIT_X0 + NECK_X1) / 2, 0.0, (NECK_Z0 + UNIT_Z1) / 2), (NECK_X1 - UNIT_X0, UNIT_W,
                                                                              UNIT_Z1 - NECK_Z0), DARK),
          # thermostat knob and signal light on top (visual), a vent band
          CylShape((ux, -0.012, UNIT_Z1 + 0.006), 0.014, 0.012, GREY, collide=False),
          CylShape((ux, 0.022, UNIT_Z1 + 0.002), 0.004, 0.004, RED, collide=False),
          BoxShape((UNIT_X0 - 0.0005, 0.0, 0.16), (0.001, 0.05, 0.004), GREY, collide=False)]
    for ty in TRACK_Y:  # the two guides on the unit's front face, their lower 10 mm narrowed to 3 mm (a lead-in)
        sh.append(BoxShape(((GUIDE_X0 + GUIDE_X1) / 2, ty, (STOP_Z + TIP_L + GUIDE_Z1) / 2),
                           (GUIDE_X1 - GUIDE_X0, GUIDE_W, GUIDE_Z1 - STOP_Z - TIP_L), GREY))
        sh.append(BoxShape(((GUIDE_X0 + GUIDE_X1) / 2, ty, STOP_Z + TIP_L / 2),
                           (GUIDE_X1 - GUIDE_X0, TIP_W, TIP_L), GREY))
    # the element: two tubes down from the neck, then a loop lying on the pot's floor
    tz = (EL_Z + NECK_Z0) / 2
    for y in (-EL_YT, EL_YT):
        sh.append(CylShape((EL_X0, y, tz), EL_R, NECK_Z0 - EL_Z, ELEM))
    for y in (-EL_Y, EL_Y):
        sh.append(CylShape(((EL_X0 + EL_X1) / 2, y, EL_Z), EL_R, EL_X1 - EL_X0, ELEM, axis="X"))
    for x in (EL_X0, EL_X1):
        sh.append(CylShape((x, 0.0, EL_Z), EL_R, 2 * EL_Y + 2 * EL_R, ELEM, axis="Y"))
    return Body.build("ctrl", pos=tuple(float(v) for v in CTRL_START), mass=CTRL_MASS, color=DARK, shapes=sh,
                      friction=(0.6, 0.5), com=CTRL_COM, report_contacts=True, solver_iterations=24)


BASE = _base()
POT = _pot()
CTRL = _ctrl()
OBJECTS = [BASE, POT, CTRL]

# -- tracking and scoring -------------------------------------------------------------------------------------------
POT_XY_MM, POT_Z_MM, POT_TILT = 3.0, 2.0, 2.0
DOWN_MM = 2.0
SEAT_N = 3.0  # N: the control assembly rests on the base this hard (its weight is 8.8 N)
STRIKE_N = float(os.environ.get("FRYER_STRIKE_N", "5.0"))
DROP_V, TIP_DEG = 0.7, 45.0
CLEAR = 0.01
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("FRYER_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("FRYER_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}
_CV: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, struck=False, dropped="", peak_pot=0.0, f_pot=0.0, f_base=0.0,
               recent=[], near=1.0, st={}, tcp={}, gaps={})
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


def _forces(base) -> tuple[float, float]:
    """Contact force magnitudes (N): control assembly - pot, control assembly - base (PhysX pair-wise forces)."""
    from isaaclab_physx.physics import PhysxManager

    sim = PhysxManager.get_physics_sim_view()
    view = _CV.get("view") if _CV.get("sim") is sim else None
    if view is None:
        env = base.scene.env_prim_paths[0]
        view = sim.create_rigid_contact_view([f"{env}/ctrl/body"],
                                             filter_patterns=[[f"{env}/pot/body", f"{env}/base/body"]],
                                             max_contact_data_count=256)
        _CV.update(sim=sim, view=view, dt=float(base.physics_dt))
    forces = view.get_contact_force_matrix(dt=_CV["dt"])
    a = np.asarray(forces.numpy() if hasattr(forces, "numpy") else as_numpy(forces), float).reshape(-1, 3)
    return float(np.linalg.norm(a[0])), float(np.linalg.norm(a[1]))


def _boxes(spec: Body, margin: float) -> list:
    out = []
    for s in spec.shapes:
        if getattr(s, "collide", True):
            c = np.array(s.centre, float)
            h = s.half_extent() + margin
            out.append((c - h, c + h))
    return out


_CTRL_BOXES = _boxes(CTRL, CLEAR)
_POT_BOXES = _boxes(POT, CLEAR)


def _near(points_local: np.ndarray, boxes: list) -> bool:
    for lo, hi in boxes:
        if np.any(np.all((points_local >= lo) & (points_local <= hi), axis=1)):
            return True
    return False


def _tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("FRYER_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
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
        fingers.append(np.concatenate([_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                                       for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])]))
    fingers = np.concatenate(fingers)
    pd = as_numpy(base.scene["pot"].data.root_link_pose_w)[0]
    cd = as_numpy(base.scene["ctrl"].data.root_link_pose_w)[0]
    cv = as_numpy(base.scene["ctrl"].data.root_com_lin_vel_w)[0]
    pp, pr = pd[:3] - origin, rotation(pd[3:7])
    cp, cr = cd[:3] - origin, rotation(cd[3:7])
    f_pot, f_base = _forces(base)
    if f_pot > STRIKE_N:
        _EP["struck"] = True
    _EP["peak_pot"] = max(_EP["peak_pot"], f_pot)
    ctilt = _tilt_deg(cr)
    if not _EP["dropped"]:
        if float(cv[2]) < -DROP_V:
            _EP["dropped"] = f"fell at {-float(cv[2]):.2f} m/s"
        elif ctilt > TIP_DEG:
            _EP["dropped"] = f"tipped {ctilt:.0f} deg"
    # the pot in its seat
    dpot = pp - POT_SEAT
    pot_seated = (max(abs(dpot[0]), abs(dpot[1])) * 1000 < POT_XY_MM and abs(dpot[2]) * 1000 < POT_Z_MM
                  and _tilt_deg(pr) < POT_TILT)
    # the guides in their tracks, and down on their stops
    gb = [cp + cr @ g for g in GUIDE_BOTTOMS]
    engaged = all(abs(g[1] - ty) < GAP / 2 and XB - RAIL_D - 0.001 < g[0] < XB + 0.001 and g[2] - TOP < RAIL_TOP
                  for g, ty in zip(gb, TRACK_Y))
    down_mm = max(g[2] - TOP - STOP_Z for g in gb) * 1000
    down = engaged and down_mm < DOWN_MM and f_base >= SEAT_N
    # the element loop inside the pot (pot frame: inside the walls, above the floor, below the flange)
    loop = [pr.T @ (cp + cr @ q - pp) for q in LOOP_PTS]
    inside = all(abs(q[0]) < POT_D / 2 - POT_T and abs(q[1]) < POT_W / 2 - POT_T and POT_T - 0.001 < q[2] < POT_H
                 for q in loop)
    near = _near((fingers - cp) @ cr, _CTRL_BOXES) or _near((fingers - pp) @ pr, _POT_BOXES)
    _EP.update(f_pot=f_pot, f_base=f_base, pp=pp, cp=cp, pot_seated=pot_seated, engaged=engaged, down=down,
               down_mm=down_mm, inside=inside, near=near, ctilt=ctilt, dpot=dpot, gb=gb)


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and "pp" in _EP:
        _EP["counted"] = _EP["tick"]
        recent.append(np.concatenate([_EP["pp"], _EP["cp"]]).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    s = 0.0
    if _EP.get("pot_seated"):
        s = 0.25
        if _EP.get("engaged"):
            s = 0.5
            if _EP.get("down") and _EP.get("inside"):
                s = 0.75
                if still and not _EP.get("near"):
                    s = 1.0
    if _EP["struck"]:
        s = min(s, 0.75)
    if _EP["dropped"]:
        s = min(s, 0.5)
    return dict(still=still, score=s)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or "pp" not in _EP:
        return
    pp, cp, dp = _EP["pp"], _EP["cp"], _EP["dpot"]
    gb = _EP["gb"]
    print(f"DBG step={_EP['step']} score={t['score']:.2f} pot=({pp[0]:.3f},{pp[1]:.3f},{pp[2] - TOP:.3f}) "
          f"dpot=({dp[0] * 1000:.1f},{dp[1] * 1000:.1f},{dp[2] * 1000:.1f})mm seated={int(_EP['pot_seated'])} "
          f"ctrl=({cp[0]:.3f},{cp[1]:.3f},{cp[2] - TOP:.3f}) tilt={_EP['ctilt']:.1f} "
          f"guides=({gb[0][0]:.4f},{gb[0][1]:.4f},{gb[0][2] - TOP:.4f})/({gb[1][0]:.4f},{gb[1][1]:.4f},"
          f"{gb[1][2] - TOP:.4f}) eng={int(_EP['engaged'])} down_mm={_EP['down_mm']:.1f} down={int(_EP['down'])} "
          f"inside={int(_EP['inside'])} f_pot={_EP['f_pot']:.1f} peak={_EP['peak_pot']:.1f} f_base={_EP['f_base']:.1f} "
          f"near={int(_EP['near'])} still={int(t['still'])} struck={int(_EP['struck'])} drop={_EP['dropped'] or '-'} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return t["score"]


def success(objects) -> bool:
    return _tally(objects)["score"] >= 1.0


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. Both arms take the pot by its two end walls (pitched grips, fingers either side of
# each wall), lift it clear of the base's back wall, carry it forward over the base, lower it level into the base and
# let go just above its seat (the fingers outside the end walls cannot follow it below the rim; it drops the last few
# centimetres down the 3 mm clearance onto its flange). The right arm then moves away. The left arm takes the control
# unit by its sides, lifts the assembly until the element loop is above the pot's rim, carries it over the base's back,
# lowers it straight down (the loop into the pot, then both guides into their tracks) and pushes 6 mm past the seat,
# lets go and backs away. FRYER_WRONG=ctrl_first: the control assembly goes on first, then the pot is lowered onto it.
# FRYER_WRONG=one_arm: the right arm alone carries the pot by its right end wall along the same path.
OPEN_C = -0.047
WRONG = os.environ.get("FRYER_WRONG", "")
POT_PITCH = float(os.environ.get("FRYER_POT_PITCH", "45"))
CTRL_PITCH = float(os.environ.get("FRYER_CTRL_PITCH", "45"))
POT_PINCH = float(os.environ.get("FRYER_POT_PINCH", "0.0"))
UNIT_PINCH = float(os.environ.get("FRYER_UNIT_PINCH", "-0.010"))  # 25 mm inside each side: a firm hold
POT_GRIP_DZ = 0.010  # the TCP this far below the pot's rim
POT_RELEASE = float(os.environ.get("FRYER_POT_RELEASE", "0.040"))  # let go with the flange this far above the rim
POT_AIM_DX = float(os.environ.get("FRYER_POT_AIM", "0.010"))
POT_LIFT = 0.200  # the pot's bottom this far above the table while carried (the base's rim is at 0.170)
CTRL_GRIP = np.array([float(os.environ.get("FRYER_GRIP_X", "0.0")), 0.0, UNIT_Z1 - 0.013])  # on the head's sides  # the TCP on the control unit (body frame)
CTRL_LIFT = float(os.environ.get("FRYER_CTRL_LIFT", "0.160"))  # above the seat while carried over
PUSH = float(os.environ.get("FRYER_PUSH", "0.006"))
CTRL_TILT = float(os.environ.get("FRYER_CTRL_TILT", "6.5"))  # deg the assembly droops in the grip when lifted
CTRL_AIM_X = float(os.environ.get("FRYER_CTRL_AIM", "0.009"))
CTRL_AIM_Y = float(os.environ.get("FRYER_CTRL_AIM_Y", "0.004"))


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


FORWARD = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) @ grip_rotation(0.0)  # level, along +x
POT_ROT = _ry(POT_PITCH) @ FORWARD  # pointing forward and down, fingers closing along y
CTRL_ROT = _ry(CTRL_PITCH) @ FORWARD
F45 = _ry(45.0) @ FORWARD
F60 = _ry(60.0) @ FORWARD
F70 = _ry(70.0) @ FORWARD


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


def _ready(arm: str):
    from yam_arena import ik, station
    from yam_arena.constants import READY

    return ik.tcp_pose(np.array(READY, float), station.BASES[arm])


def _pot_wall(arm: str, centre: np.ndarray, dz: float = 0.0) -> np.ndarray:
    """The TCP on the pot's left (+y) or right (-y) end wall, the pot's outer bottom centre at ``centre``."""
    sy = 1.0 if arm == "left" else -1.0
    return centre + np.array([0.0, sy * (POT_W / 2 - POT_T / 2), POT_H - POT_GRIP_DZ + dz])


def _pot_path(arm: str, t0: float = 0.0, from_ready: bool = True) -> list:
    rot = POT_ROT
    start, seat = POT_START.copy(), POT_SEAT.copy()
    lifted = np.array([start[0], start[1], TOP + 0.160])
    mid = np.array([0.215, start[1], TOP + POT_LIFT])
    aim = np.array([POT_AIM_DX, 0.0, 0.0])  # aimed past the seat: the pot slips back in the fingers and the arm lags
    over = np.array([seat[0], seat[1], TOP + POT_LIFT]) + aim
    release = seat + np.array([0.0, 0.0, POT_RELEASE]) + aim
    out = []
    if t0 > 0:
        p, r = _ready(arm)
        out.append(Waypoint(t0, tuple(float(v) for v in p), OPEN_C, "wait", r))
    if from_ready:
        out += _lead_in(arm, _pot_wall(arm, start, 0.06), rot, OPEN_C, "above the pot's end wall, open")
    else:
        out += [_w(3.0, _pot_wall(arm, start, 0.06), OPEN_C, "above the pot's end wall, open", rot)]
    out += [_w(2.0, _pot_wall(arm, start), OPEN_C, "fingers either side of the end wall", rot),
            _w(0.5, _pot_wall(arm, start), OPEN_C, "settle", rot),
            _w(1.2, _pot_wall(arm, start), POT_PINCH, "pinch the end wall", rot),
            _w(0.6, _pot_wall(arm, start), POT_PINCH, "settle", rot),
            _w(3.0, _pot_wall(arm, lifted), POT_PINCH, "lift the pot level", rot),
            _w(1.5, _pot_wall(arm, mid), POT_PINCH, "up and forward, clear of the base's back", rot),
            _w(3.0, _pot_wall(arm, over), POT_PINCH, "carry it over the base", rot),
            _w(0.8, _pot_wall(arm, over), POT_PINCH, "settle", rot),
            _w(3.0, _pot_wall(arm, release), POT_PINCH, "lower it level into the base", rot),
            _w(0.8, _pot_wall(arm, release), POT_PINCH, "settle", rot),
            _w(1.0, _pot_wall(arm, release), OPEN_C, "let go", rot),
            _w(0.6, _pot_wall(arm, release), OPEN_C, "wait", rot)]
    sy = 1.0 if arm == "left" else -1.0
    out_p = _pot_wall(arm, release) + np.array([0.0, sy * 0.025, 0.0])
    out += [_w(1.0, out_p, OPEN_C, "move out, clear of the flange", rot),
            _w(1.5, out_p + np.array([-0.03, 0.0, 0.07]), OPEN_C, "and up", rot)]
    return out


def _ctrl_tcp(origin: np.ndarray) -> np.ndarray:
    return origin + CTRL_GRIP


def _ctrl_path(arm: str) -> list:
    rot = CTRL_ROT
    carry = _ry(-CTRL_TILT) @ CTRL_ROT  # the hand tipped back, so the assembly (drooping in the grip) hangs level
    start = CTRL_START.copy()
    seat = CTRL_SEAT.copy()
    g0 = _ctrl_tcp(start)
    lifted = g0 + np.array([0.0, 0.0, (seat[2] + CTRL_LIFT) - start[2]])
    aim = np.array([CTRL_AIM_X, CTRL_AIM_Y, 0.0])  # the loaded arm stops short, and the unit sits off-centre in the grip
    over = _ctrl_tcp(seat) + np.array([0.0, 0.0, CTRL_LIFT]) + aim
    entry = _ctrl_tcp(seat) + np.array([0.0, 0.0, RAIL_TOP - STOP_Z + 0.008]) + aim
    seat_p = _ctrl_tcp(seat) + np.array([0.0, 0.0, -PUSH]) + aim
    out = [_w(2.5, g0 + np.array([-0.05, 0.0, 0.05]), OPEN_C, "behind and above the control unit, open", rot),
           _w(2.0, g0, OPEN_C, "fingers either side of the control unit", rot),
           _w(0.5, g0, OPEN_C, "settle", rot),
           _w(1.0, g0, UNIT_PINCH, "take the control unit by its sides", rot),
           _w(0.6, g0, UNIT_PINCH, "settle", rot),
           _w(3.0, lifted, UNIT_PINCH, "lift the assembly, element above the pot's rim", carry),
           _w(0.8, lifted, UNIT_PINCH, "settle", carry),
           _w(4.0, over, UNIT_PINCH, "carry it over the base's back", carry),
           _w(1.0, over, UNIT_PINCH, "settle", carry)]
    n = 8
    for k in range(1, n + 1):
        out.append(_w(4.0 / n, over + (entry - over) * k / n, UNIT_PINCH,
                      "lower it straight down: the loop into the pot, the guides to the tracks' mouths", carry))
    out += [_w(0.8, entry, UNIT_PINCH, "settle", carry)]
    for k in range(1, n + 1):
        out.append(_w(4.0 / n, entry + (seat_p - entry) * k / n, UNIT_PINCH,
                      "slide it down the tracks", carry))
    out += [_w(1.0, seat_p, UNIT_PINCH, "settle on the stops", carry),
            _w(1.0, seat_p, OPEN_C, "let go", carry),
            _w(0.5, seat_p, OPEN_C, "wait", carry)]
    d = carry @ np.array([0.0, 0.0, 1.0])  # the gripper's pointing direction
    out += [_w(1.5, seat_p - d * 0.05, OPEN_C, "draw the fingers back along themselves", carry),
            _w(1.5, seat_p - d * 0.05 + np.array([-0.03, 0.0, 0.06]), OPEN_C, "and up, clear", carry),
            _w(8.0, seat_p - d * 0.05 + np.array([-0.03, 0.0, 0.06]), OPEN_C, "stay clear", carry)]
    return out


def demo():
    if WRONG == "one_arm":  # the pot by one end wall only, the same path; the left arm stays at the ready pose
        p, r = _ready("left")
        return {"right": _pot_path("right"), "left": [Waypoint(60.0, tuple(float(v) for v in p), OPEN_C, "stay", r)]}
    if WRONG == "ctrl_first":
        left = _lead_in("left", _ctrl_tcp(CTRL_START) + np.array([-0.05, 0.0, 0.05]), CTRL_ROT, OPEN_C,
                        "to the control unit")
        left += _ctrl_path("left")
        t_ctrl = sum(w.duration for w in left)
        return {"left": left + _pot_path("left", from_ready=False), "right": _pot_path("right", t_ctrl)}
    left = _pot_path("left")
    right = _pot_path("right")
    p, r = _ready("right")
    right += [_w(3.0, p, OPEN_C, "back to the side", r), _w(60.0, p, OPEN_C, "stay", r)]
    left += _ctrl_path("left")
    return {"left": left, "right": right}
