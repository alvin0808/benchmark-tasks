"""On the station table: a full size D aluminium medical oxygen cylinder lying in a wooden cradle, its CGA 870
pin-index post valve pointing at the robot with the valve's outlet facing down, and a yoke oxygen regulator standing
upright on the table to the right, its sealing washer on its inlet.

Task (instruction): fit the regulator onto the cylinder's valve, hand-tight, and let go of it. The instruction does
not say how. The procedure the task checks is the regulator maker's (Graham-Field oxygen regulator with yoke
connector, JB0150-081/082/083 user manual, "Setup"): "Ensure the regulator inlet sealing washer is in place." "Slip
the yoke over the cylinder post and fit the pins into the holes on the cylinder valve. The pins should fit easily into
these holes." "Tighten the T-handle firmly by hand; turn it clockwise until the screw is seated in the dimple on the
cylinder valve." The valve and yoke follow ISO 407 (basic dimensions, Table 2) with the oxygen pins at positions 2
and 5. With the outlet facing down the regulator hangs under the valve: it stays on its pins only while it is held up,
until the T-screw is tight. Let go before that, it drops off its pins and shifts in the yoke's play, so the screw then
clamps it with its pins resting on the valve's face beside their holes, proud of the face.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475. Valve
frame: origin on the cylinder's axis level with the outlet's centre (x 0.255, y 0, 9.2 cm above the table), axes as
the world's.
- cylinder: loose in the cradle, 3.4 kg (2.5 kg empty, 0.6 kg oxygen, 0.3 kg valve), 111.3 mm across, 424 mm from
  neck to base (Catalina MD), its axis along x, 9.2 cm above the table; green shoulder, aluminium body, white label.
- post valve (brass, valve frame, mm): the post x -16 to 44.5, 22.2 wide (y), 25 thick (z), its free end's four
  edges chamfered 2.5 mm; the outlet port (7.2 mm square, 6 deep) in the bottom face at x 0; the two oxygen
  pin holes (4.75 mm square, 6 deep, a 0.55 mm 45-degree countersink to 5.85 mm) at x 13.6, y +-4.42 (radius 14.3
  mm, 18 degrees either side of the line toward the cylinder); the conical recess (a 6 mm four-sided pocket, 2.1 deep)
  in the top face at x 0; the spindle's square end and gland nut on the free end; a safety relief plug on the left.
- cradle: fixed, two V saddles at x 0.395 and 0.655 and an end board behind the cylinder's base.
- regulator: a floating articulation, 0.34 kg. Yoke (frame origin: the inlet's centre on the inlet face): window
  24 mm (y) by 36 mm (inlet face to the top bar), the inlet block x -12 to 20, 8 mm thick, carrying the 6.3 mm
  spigot (3.3 mm proud) and the two steel pins (4 mm, 5.5 mm proud, flat ends) at x 13.6, y +-4.42; below it the green
  body (38 mm across, 44 mm long), a black flow dial (40 mm across) at the bottom, a contents gauge facing +x and a
  hose barb. The T-screw (M12, 1.75 mm per turn) comes down through the top bar, its point a 2.5 mm ball, backed off
  so the point is 33.8 mm above the inlet face; a black T-handle (60 x 14 x 22 mm) on top, turned 50 degrees from x.
  It stands on its dial with the inlet's centre at x 0.205, y -0.10, 6.5 cm above the table.
- sealing washer: loose, 16 mm across (20 g here for contact stability; a real one is 2-3 g), 6.6 mm hole, 1.6 mm thick (ISO 407 g), lying on the inlet face round the
  spigot.

T-screw model (PhysX has no screw joint): the screw part slides along the yoke's clamp axis on a drive whose target
follows the T-handle's turn at 1.75 mm per turn (clockwise seen from above tightens); once the point is stopped the
drive's lead times its stiffness is the clamp force (capped at 120 N), and the T-handle's Coulomb friction is 0.05 N m
plus 0.019 m times the clamp force (0.75 N m at 37 N: hand-tight). It holds where it is left.

Score (from the state at the end of the trial; 0 for the rest of the run once the washer has come off the inlet, the
cylinder has left its cradle or the regulator has fallen off the table): 0.25 seated: the yoke's inlet flush with the
valve's face over the washer (inlet face 1.6 +- 0.5 mm below the outlet face, within 2 degrees of square), both pins
3 mm or more into their holes, the washer between the faces round the spigot; 0.5 also the screw's point down in the
recess; 0.75 also hand-tight (0.75 N m or more needed to turn the T-handle further). Full score: that, with both
grippers 6 cm or more from the T-handle and from the regulator's body, the regulator and the T-handle still for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, on_reset,  # noqa: E402
                     on_step, plate, rotation)
from linkkit import Assembly, Part, ids, set_friction, set_target  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file judges the end state itself and a broken rule latches 0, so the score is read at the end of the run.
SCORE_OVER = "end"
MM = 0.001

# -- layout ---------------------------------------------------------------------------------------------------------
XO, H = 0.255, 0.092  # the outlet's centre (x) on the cylinder's axis; the axis's height above the table
VALVE = np.array([XO, 0.0, TOP + H])  # the valve frame's origin
# post (valve frame, m)
P_X0, P_X1, P_HW, P_HT = -16 * MM, 44.5 * MM, 11.1 * MM, 12.5 * MM  # free end, end of the flat, half width, half thickness
CHAMF = 2.5 * MM
PORT, PORT_D = 7.2 * MM, 6.0 * MM
HOLE, HOLE_D, SINK = 4.75 * MM, 6.0 * MM, 0.55 * MM
PIN_DX, PIN_DY = 13.6 * MM, 4.42 * MM  # positions 2 and 5: radius 14.3 mm, 18 degrees either side
DIMPLE, DIMPLE_D = 6.0 * MM, 2.1 * MM
# cylinder (Catalina MD)
CYL_R, NECK_X, BASE_X = 55.65 * MM, 58 * MM, 482 * MM  # radius; neck start and base (valve frame x)
CYL_KG = 3.4
SADDLES = (XO + 0.14, XO + 0.40)
# regulator (yoke frame: origin at the inlet's centre on the inlet face)
WIN_HW, WIN_H, BAR_T = 12.0 * MM, 36 * MM, 3.5 * MM  # window half width (y), inlet face to top bar, side bar thickness
YK_HW = WIN_HW + BAR_T
INLET = ((-12 * MM, 20 * MM), (-8 * MM, 0.0))  # inlet block: x range, z range
SPIG_R, SPIG_L = 3.15 * MM, 3.3 * MM
PIN_R, PIN_L = 2.0 * MM, 5.5 * MM
BODY_R, BODY_Z = 19 * MM, (-52 * MM, -8 * MM)
DIAL_R, DIAL_Z = 20 * MM, (-64 * MM, -52 * MM)
TIP_R = 2.5 * MM
TIP0 = 33.8 * MM  # the screw point's lowest point above the inlet face at the start (backed off)
SHAFT_R = 6.0 * MM
WING = (60 * MM, 14 * MM, 22 * MM)  # T-handle: length, thickness, height
WING_Z0 = 60 * MM  # the T-handle's bottom above the inlet face at the start
PITCH = 1.75 * MM
REG_KG, SCREW_KG, TBAR_KG = 0.30, 0.02, 0.02
REG_START = np.array([0.205, -0.10, TOP - DIAL_Z[0] + 0.001])  # the yoke's origin at the start: the dial 1 mm up
YAW_A = float(os.environ.get("O2_YAW", "50"))  # the left gripper's turn at the start of each half turn (degrees)
TBAR_YAW = YAW_A  # the T-handle's direction (degrees from +x) at the start: across the fingers at that turn
# washer (ISO 407: 16 mm across, 1.6 mm thick)
W_RO, W_RI, W_T, W_KG = 8.0 * MM, 3.3 * MM, 1.6 * MM, 0.02
# screw model
K_SCREW, C_SCREW = 1.0e5, 150.0  # N/m, N s/m
TAU0, MU_EFF, F_MAX = 0.05, 0.019, 120.0  # N m, m, N
T_HAND = 0.75  # N m: turning the T-handle further takes this much when it is hand-tight
F_HAND = (T_HAND - TAU0) / MU_EFF  # the clamp force then (37 N)
S_MIN = -10.5 * MM  # the screw's travel (down)
# where it sits when fitted (valve frame): the inlet face 1.6 mm below the outlet face
SEAT_Z = -P_HT - W_T

BRASS, BRASS_D, STEEL, ALU, GREEN, WHITE, BLACK = ((0.80, 0.64, 0.30), (0.62, 0.48, 0.20), (0.62, 0.63, 0.66),
                                                   (0.80, 0.81, 0.83), (0.12, 0.52, 0.28), (0.93, 0.93, 0.92),
                                                   (0.06, 0.06, 0.07))
CHROME, WOOD, WOOD_D, RED, BLUE_GREEN = (0.86, 0.87, 0.89), (0.72, 0.56, 0.36), (0.55, 0.40, 0.24), (0.75, 0.12, 0.10), (
    0.10, 0.45, 0.30)


def _box(x0, x1, y0, y1, z0, z1, color, collide=True) -> BoxShape:
    return BoxShape(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, collide=collide)


def _funnel(cx, cy, z_face, inner, outer, depth, down: bool, color) -> list:
    """Four plates forming a square funnel round (cx, cy): ``outer`` wide at the face (z_face), ``inner`` wide
    ``depth`` in (a countersink, or a pyramid pocket when ``inner`` is ~0). ``down``: the pocket is in a bottom face
    and goes up. Each plate's top surface is the funnel's wall; the rest of it lies inside the material."""
    out = []
    s = -1.0 if down else 1.0
    t = 0.0006
    zi = z_face - s * depth
    for ax in (0, 1):
        for sg in (-1.0, 1.0):
            p_out = np.array([cx, cy, z_face])
            p_in = np.array([cx, cy, zi])
            p_out[ax] += sg * outer / 2
            p_in[ax] += sg * inner / 2
            n = np.zeros(3)
            n[ax] = -sg * depth
            n[2] = s * (outer - inner) / 2
            n /= np.linalg.norm(n)  # the wall's normal, out of the material into the pocket
            off = -n * t / 2
            out.append(plate((p_in + off).tolist(), (p_out + off).tolist(), outer, t, n.tolist(), color))
    return out


def _tile(x0, x1, y0, y1, z0, z1, holes, color) -> list:
    """Boxes filling the slab [x0, x1] x [y0, y1] x [z0, z1] except the rectangular ``holes`` (x0, x1, y0, y1)."""
    xs = sorted({x0, x1, *[h[0] for h in holes], *[h[1] for h in holes]})
    out = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa < 1e-9:
            continue
        cut = sorted((h[2], h[3]) for h in holes if h[0] < (xa + xb) / 2 < h[1])
        ya = y0
        for c0, c1 in cut + [(y1, y1)]:
            if c0 - ya > 1e-9:
                out.append(_box(xa, xb, ya, c0, z0, z1, color))
            ya = max(ya, c1)
    return out


# -- the cylinder with its valve (one loose body, valve frame) --------------------------------------------------------
def _post_shapes() -> list:
    """The CGA 870 post (ISO 407): the outlet port and the two oxygen pin holes (with countersinks) in its bottom face,
    the conical recess in its top face, the free end's edges chamfered; the spindle's end, gland nut and safety relief
    plug."""
    out = []
    hw, ht = P_HW, P_HT
    xa = P_X0 + CHAMF  # behind the free end's chamfers
    zb, zt = -ht + PORT_D, ht - DIMPLE_D  # the bottom pockets' floors, the recess's floor
    out.append(_box(xa, P_X1, -hw, hw, zb, zt, BRASS))  # middle layer, solid
    out.append(_box(P_X0, xa, -hw + CHAMF, hw - CHAMF, -ht + CHAMF, ht - CHAMF, BRASS))  # free end
    for sg in (-1.0, 1.0):  # 45-degree chamfers round the free end's four edges
        n = np.array([-1.0, 0.0, sg]) / math.sqrt(2)
        a = np.array([P_X0, 0.0, sg * (ht - CHAMF)]) - n * 0.0015
        b = np.array([xa, 0.0, sg * ht]) - n * 0.0015
        out.append(plate(a.tolist(), b.tolist(), 2 * hw - 0.001, 0.003, n.tolist(), BRASS))
        n = np.array([-1.0, sg, 0.0]) / math.sqrt(2)
        a = np.array([P_X0, sg * (hw - CHAMF), 0.0]) - n * 0.0015
        b = np.array([xa, sg * hw, 0.0]) - n * 0.0015
        out.append(plate(a.tolist(), b.tolist(), 2 * ht - 0.001, 0.003, n.tolist(), BRASS))
    port = (-PORT / 2, PORT / 2, -PORT / 2, PORT / 2)

    def holes(w):
        return [port] + [(PIN_DX - w / 2, PIN_DX + w / 2, sy * PIN_DY - w / 2, sy * PIN_DY + w / 2) for sy in (-1, 1)]

    out += _tile(xa, P_X1, -hw, hw, -ht + SINK, zb, holes(HOLE), BRASS)  # bottom layer, holes 4.75 mm
    out += _tile(xa, P_X1, -hw, hw, -ht, -ht + SINK, holes(HOLE + 2 * SINK), BRASS)  # face layer, countersinks
    for sy in (-1.0, 1.0):
        out += _funnel(PIN_DX, sy * PIN_DY, -ht, HOLE, HOLE + 2 * SINK, SINK, True, BRASS)
    out.append(_box(-PORT / 2 - 0.0008, PORT / 2 + 0.0008, -PORT / 2 - 0.0008, PORT / 2 + 0.0008, -ht - 0.0002,
                    -ht - 0.0001, BRASS_D, collide=False))  # the outlet's mouth (visual)
    d2 = DIMPLE / 2
    out += _tile(xa, P_X1, -hw, hw, zt, ht, [(-d2, d2, -d2, d2)], BRASS)  # top layer with the recess
    out += _funnel(0.0, 0.0, ht, 0.0004, DIMPLE, DIMPLE_D, False, BRASS)
    out.append(CylShape((P_X0 - 0.002, 0.0, 0.0), 0.0075, 0.004, BRASS_D, axis="X"))  # gland nut
    out.append(_box(P_X0 - 0.010, P_X0 - 0.004, -0.0037, 0.0037, -0.0037, 0.0037, STEEL))  # spindle's square end
    out.append(CylShape((0.030, hw + 0.0022, 0.0), 0.0055, 0.0044, BRASS_D, axis="Y"))  # safety relief plug
    out.append(CylShape((0.030, hw + 0.0049, 0.0), 0.0035, 0.0012, BRASS, axis="Y", collide=False))
    out.append(_box(0.008, 0.024, hw, hw + 0.0003, -0.004, 0.004, BRASS_D, collide=False))  # "O2" stamp
    return out


def _cylinder_shapes() -> list:
    out = _post_shapes()
    out.append(CylShape(((P_X1 + NECK_X) / 2, 0.0, 0.0), 0.014, NECK_X - P_X1, BRASS, axis="X"))  # valve's round body
    out.append(CylShape((NECK_X + 0.004, 0.0, 0.0), 0.017, 0.008, ALU, axis="X"))  # neck
    for x0, x1, r in ((0.066, 0.072, 0.030), (0.072, 0.078, 0.040), (0.078, 0.086, 0.047), (0.086, 0.096, 0.052),
                      (0.096, 0.108, 0.0545)):
        out.append(CylShape(((x0 + x1) / 2, 0.0, 0.0), r, x1 - x0, GREEN, axis="X"))  # shoulder (green crown)
    out.append(CylShape(((0.108 + BASE_X) / 2, 0.0, 0.0), CYL_R, BASE_X - 0.108, ALU, axis="X"))
    out.append(CylShape((BASE_X + 0.001, 0.0, 0.0), CYL_R - 0.003, 0.002, ALU, axis="X"))
    # label (visual): a white band with the gas name in green and the green and yellow hazard diamonds
    top = CYL_R + 0.0008
    out.append(CylShape((0.215, 0.0, 0.0), CYL_R + 0.0004, 0.11, WHITE, axis="X", collide=False))
    for k, (x0, w) in enumerate(((0.178, 0.075), (0.190, 0.050), (0.184, 0.062))):
        out.append(_box(x0, x0 + w, -0.026 + 0.009 * k, -0.021 + 0.009 * k, top - 0.0004, top + 0.0002, GREEN,
                        collide=False))
    for xc, col in ((0.240, GREEN), (0.240, (0.95, 0.80, 0.10))):
        yc = 0.016 if col == GREEN else -0.016 + 0.0
        if col != GREEN:
            xc = 0.240
        out.append(BoxShape((xc, yc, top), (0.016, 0.016, 0.0010), col, yaw=math.pi / 4, collide=False))
    return out


CYLINDER = Body.build("oxygen_cylinder", pos=tuple(float(v) for v in VALVE + [0.0, 0.0, 0.0005]), mass=CYL_KG,
                      color=ALU, friction=(0.5, 0.4), contact_offset=0.001, solver_iterations=64,
                      shapes=_cylinder_shapes())


def _saddle_shapes() -> list:
    """A V saddle in the frame of its base's middle: base, two boards at 45 degrees, two side posts."""
    out = [_box(-0.015, 0.015, -0.065, 0.065, 0.0, 0.020, WOOD_D)]
    apex = H - CYL_R * math.sqrt(2)  # where the two tangent planes meet, above the table
    for sg in (-1.0, 1.0):
        n = np.array([0.0, -sg, 1.0]) / math.sqrt(2)  # the board's top face normal (toward the cylinder's axis)
        a = np.array([0.0, sg * 0.012, apex + 0.012]) - n * 0.006
        b = np.array([0.0, sg * 0.062, apex + 0.062]) - n * 0.006
        out.append(plate(a.tolist(), b.tolist(), 0.030, 0.012, n.tolist(), WOOD))
        out.append(_box(-0.015, 0.015, sg * 0.052 - 0.008, sg * 0.052 + 0.008, 0.020, apex + 0.045, WOOD_D))
    return out


CRADLE_FRONT = Body.build("cradle_front", pos=(SADDLES[0], 0.0, TOP), mass=1.0, color=WOOD, fixed=True,
                          friction=(0.6, 0.5), shapes=_saddle_shapes())
CRADLE_BACK = Body.build("cradle_back", pos=(SADDLES[1], 0.0, TOP), mass=1.0, color=WOOD, fixed=True,
                         friction=(0.6, 0.5), shapes=_saddle_shapes())
END_X = XO + BASE_X + 0.003
END_BOARD = Body.build("cradle_end", pos=(END_X + 0.006, 0.0, TOP), mass=1.0, color=WOOD, fixed=True,
                       friction=(0.6, 0.5), shapes=[_box(-0.006, 0.006, -0.06, 0.06, 0.0, 0.13, WOOD),
                                                    _box(-0.03, 0.03, -0.06, 0.06, 0.0, 0.012, WOOD_D)])


# -- the regulator (one floating articulation, yoke frame) ------------------------------------------------------------
def _yoke_shapes() -> list:
    (ix0, ix1), (iz0, iz1) = INLET
    out = [_box(ix0, ix1, -YK_HW, YK_HW, iz0, iz1, CHROME)]
    for sg in (-1.0, 1.0):
        y0, y1 = sorted((sg * WIN_HW, sg * YK_HW))
        out.append(_box(-0.009, 0.009, y0, y1, iz1, WIN_H, CHROME))  # side bars
    out.append(_box(-0.009, 0.009, -YK_HW, YK_HW, WIN_H, WIN_H + 0.010, CHROME))
    out.append(CylShape((0.0, 0.0, WIN_H + 0.0115), 0.009, 0.003, CHROME, collide=False))  # threaded boss
    out.append(_box(-0.0081, 0.0081, -0.006, 0.006, WIN_H + 0.002, WIN_H + 0.008, BLACK, collide=False))  # "O2" plate
    out.append(CylShape((0.0, 0.0, SPIG_L / 2), SPIG_R, SPIG_L, BRASS))  # spigot
    for sy in (-1.0, 1.0):
        out.append(CylShape((PIN_DX, sy * PIN_DY, (PIN_L - 0.002) / 2), PIN_R, PIN_L + 0.002, STEEL))
    # body, flow dial (with its marks), contents gauge, hose barb
    out.append(CylShape((0.0, 0.0, (BODY_Z[0] + BODY_Z[1]) / 2), BODY_R, BODY_Z[1] - BODY_Z[0], GREEN))
    out.append(CylShape((0.0, 0.0, BODY_Z[1] - 0.0015), BODY_R + 0.0004, 0.003, CHROME, collide=False))
    out.append(CylShape((0.0, 0.0, (DIAL_Z[0] + DIAL_Z[1]) / 2), DIAL_R, DIAL_Z[1] - DIAL_Z[0], BLACK))
    for k in range(12):
        a = 2 * math.pi * k / 12
        out.append(BoxShape(((DIAL_R + 0.0003) * math.cos(a), (DIAL_R + 0.0003) * math.sin(a), DIAL_Z[1] - 0.004),
                            (0.0012, 0.0012, 0.004), WHITE, yaw=a, collide=False))
    out.append(_box(-0.016, -0.004, -0.004, 0.004, DIAL_Z[1] - 0.0002, DIAL_Z[1] + 0.0003, WHITE, collide=False))
    out.append(CylShape((0.0225, 0.0, -0.030), 0.004, 0.008, CHROME, axis="X"))  # gauge stem
    out.append(CylShape((0.032, 0.0, -0.030), 0.019, 0.012, CHROME, axis="X"))  # gauge case
    out.append(CylShape((0.0382, 0.0, -0.030), 0.016, 0.0004, WHITE, axis="X", collide=False))  # dial face
    out.append(_box(0.0385, 0.0389, -0.010, 0.010, -0.044, -0.040, GREEN, collide=False))  # full zone
    out.append(_box(0.0385, 0.0389, -0.0008, 0.0008, -0.030, -0.018, BLACK, collide=False))  # needle
    out.append(_box(0.0385, 0.0389, -0.013, -0.008, -0.022, -0.019, RED, collide=False))  # empty zone
    out.append(CylShape((0.031, 0.0, -0.058), 0.0035, 0.018, WHITE, axis="X"))  # barb
    out.append(CylShape((0.024, 0.0, -0.058), 0.0050, 0.004, WHITE, axis="X", collide=False))
    return out


def _screw_shapes() -> list:
    """The screw (it slides): the ball point and the threaded shaft up to the T-handle, yoke frame."""
    zc = TIP0 + TIP_R
    return [SphereShape((0.0, 0.0, zc), TIP_R, STEEL),
            CylShape((0.0, 0.0, (zc + WING_Z0) / 2), SHAFT_R, WING_Z0 - zc, STEEL)]


def _tbar_shapes() -> list:
    a = math.radians(TBAR_YAW)
    zc = WING_Z0 + WING[2] / 2
    return [BoxShape((0.0, 0.0, zc), WING, BLACK, yaw=a),
            BoxShape((0.0, 0.0, zc), (WING[1] + 0.002, WING[1] + 0.002, WING[2] + 0.002), BLACK, yaw=a, collide=False),
            CylShape((0.0, 0.0, WING_Z0 + WING[2] + 0.0004), 0.004, 0.0008, CHROME, collide=False)]


REG_POS = tuple(float(v) for v in REG_START)
REGULATOR = Assembly.build("regulator", color=GREEN, contact_offset=0.001, solver_iterations=64, parts=[
    Part("yoke", tuple(_yoke_shapes()), REG_KG, REG_POS, material=(0.5, 0.4)),
    Part("screw", tuple(_screw_shapes()), SCREW_KG, REG_POS, parent="yoke", joint="prismatic", axis="Z",
         anchor=REG_POS, limits=(S_MIN, 0.0005), stiffness=K_SCREW, damping=C_SCREW, material=(0.3, 0.25)),
    Part("tbar", tuple(_tbar_shapes()), TBAR_KG, REG_POS, parent="screw", joint="revolute", axis="Z", anchor=REG_POS,
         friction=TAU0, material=(0.9, 0.8)),
])


def _washer_shapes() -> list:
    out = []
    rm, rw = (W_RO + W_RI) / 2, W_RO - W_RI
    for k in range(8):
        a = 2 * math.pi * k / 8
        out.append(BoxShape((rm * math.cos(a), rm * math.sin(a), 0.0), (rw, 2 * W_RO * math.tan(math.pi / 8), W_T),
                            BLACK, yaw=a))
        out.append(BoxShape(((W_RO - 0.0006) * math.cos(a), (W_RO - 0.0006) * math.sin(a), 0.0),
                            (0.0014, 2 * W_RO * math.tan(math.pi / 8) + 0.0004, W_T + 0.0002), STEEL, yaw=a,
                            collide=False))
    return out


WASHER_START = REG_START + [0.0, 0.0, W_T / 2 + 0.0003]
WASHER = Body.build("sealing_washer", pos=tuple(float(v) for v in WASHER_START), mass=W_KG, color=BLACK,
                    friction=(0.9, 0.8), contact_offset=0.001, solver_iterations=64, shapes=_washer_shapes())

OBJECTS = [CRADLE_FRONT, CRADLE_BACK, END_BOARD, CYLINDER, REGULATOR, WASHER]


# -- tracking and scoring ---------------------------------------------------------------------------------------------
FLUSH_MM, SQUARE_DEG, PIN_IN_MM = 0.5, 2.0, 3.0
WASHER_OFF_MM, WASHER_TILT = 1.5, 10.0
LOST_R_MM, LOST_Z = 4.0, (-1.5 * MM, 4.5 * MM)
TIP_XY_MM, TIP_DOWN_MM = 1.5, 0.8
CYL_MOVE, CYL_TILT = 0.03, 10.0
CLEAR = 0.06
STILL_STEPS, STILL_MM, STILL_DEG = 20, 1.0, 2.0
_DEBUG = os.environ.get("O2_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("O2_DEBUG_EVERY", "10"))
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, counted=-1, failed="", recent=[], cyl0=None, slip=0.0, fric=None, target=0.0, force=0.0,
               s=0.0, turns=0.0, stage=0.0, info={}, tbar_c=REG_START.copy(), body_c=REG_START.copy(), reg=None,
               q_prev=None, q_turn=0.0)
    _TCP.clear()


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("O2_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _pose(asset_data, i, origin):
    p = as_numpy(asset_data.body_pos_w)[0][i] - origin
    q = as_numpy(asset_data.body_quat_w)[0][i]
    return p, q


def _screw_model(env, q_turn: float, s: float) -> None:
    """The thread: the screw's drive target follows the T-handle's turn; the clamp force is the drive's lead times its
    stiffness (the thread slips past F_MAX); the T-handle's friction rises with it."""
    turns = -q_turn / (2 * math.pi) - _EP["slip"]  # clockwise seen from above (negative about +z) tightens
    target = max(S_MIN, min(0.0, -PITCH * turns))
    lead = s - target
    if lead * K_SCREW > F_MAX:  # the thread slips: the target stays F_MAX ahead
        _EP["slip"] += (lead - F_MAX / K_SCREW) / PITCH
        target = s - F_MAX / K_SCREW
        lead = F_MAX / K_SCREW
    force = max(0.0, lead) * K_SCREW
    set_target(env, REGULATOR, "screw", target)
    fric = TAU0 + MU_EFF * force
    if _EP["fric"] is None or abs(fric - _EP["fric"]) > 0.005:
        set_friction(env, REGULATOR, "tbar", fric)
        _EP["fric"] = fric
    _EP.update(target=target, force=force, s=s, turns=turns)


def _track_inner(env) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = np.asarray(ik.tcp_pose(q[_ARM_IDS[side]], station.BASES[side])[0], float)
    origin = as_numpy(base.scene.env_origins[0])
    reg = base.scene["regulator"]
    idx = ids(reg, REGULATOR)
    jq = as_numpy(reg.data.joint_pos)[0]
    s, q_raw = float(jq[idx["joint"]["screw"]]), float(jq[idx["joint"]["tbar"]])
    if _EP["q_prev"] is not None:  # PhysX wraps a free revolute joint's angle: keep the turn continuous
        d = q_raw - _EP["q_prev"]
        _EP["q_turn"] += d - 2 * math.pi * round(d / (2 * math.pi))
    else:
        _EP["q_turn"] = q_raw
    _EP["q_prev"] = q_raw
    q_turn = _EP["q_turn"]
    _screw_model(env, q_turn, s)
    yp, yq = _pose(reg.data, idx["body"]["yoke"], origin)
    cyl = base.scene["oxygen_cylinder"].data
    cp = as_numpy(cyl.root_link_pose_w)[0]
    c_pos, c_rot = cp[:3] - origin, rotation(cp[3:7])
    wp = as_numpy(base.scene["sealing_washer"].data.root_link_pose_w)[0]
    w_pos, w_rot = wp[:3] - origin, rotation(wp[3:7])
    y_rot = rotation(yq)
    if _EP["cyl0"] is None:
        _EP["cyl0"] = (c_pos.copy(), c_rot.copy())
    # the yoke in the valve frame
    rel_r = c_rot.T @ y_rot
    rel_p = c_rot.T @ (yp - c_pos)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rel_r[2, 2]))))
    depths = []
    for sy in (-1.0, 1.0):
        top = rel_p + rel_r @ np.array([PIN_DX, sy * PIN_DY, PIN_L])
        inside = abs(top[0] - PIN_DX) < HOLE / 2 and abs(top[1] - sy * PIN_DY) < HOLE / 2
        depths.append(float(top[2] + P_HT) if inside else min(0.0, float(top[2] + P_HT)))
    gap = float(-P_HT - rel_p[2])  # the outlet face above the inlet face (the washer's room)
    # the washer in the yoke frame
    w_in_y = y_rot.T @ (w_pos - yp)
    w_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float((y_rot.T @ w_rot)[2, 2])))))
    w_off = float(np.hypot(w_in_y[0], w_in_y[1]))
    if not _EP["failed"] and (w_off > LOST_R_MM * MM or not LOST_Z[0] < w_in_y[2] < LOST_Z[1]):
        _EP["failed"] = "the sealing washer came off the regulator's inlet"
    # the screw's point in the valve frame
    tip = c_rot.T @ (yp + y_rot @ np.array([0.0, 0.0, TIP0 + s]) - c_pos)
    c0p, c0r = _EP["cyl0"]
    c_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(c_rot[:, 0], c0r[:, 0]))))))
    if not _EP["failed"] and (np.linalg.norm(c_pos - c0p) > CYL_MOVE or c_tilt > CYL_TILT):
        _EP["failed"] = "the cylinder left its cradle"
    if not _EP["failed"] and yp[2] < TOP - 0.03:
        _EP["failed"] = "the regulator fell off the table"
    seated = bool(abs(gap - W_T) < FLUSH_MM * MM and tilt < SQUARE_DEG and min(depths) >= PIN_IN_MM * MM
                  and abs(rel_p[0]) < 2 * MM and abs(rel_p[1]) < 2 * MM
                  and w_off < WASHER_OFF_MM * MM and abs(w_in_y[2] - W_T / 2) < 0.6 * MM and w_tilt < WASHER_TILT)
    in_recess = bool(np.hypot(tip[0], tip[1]) < TIP_XY_MM * MM and tip[2] < P_HT - TIP_DOWN_MM * MM)
    tight = bool(_EP["force"] >= F_HAND)
    stage = 0.75 if seated and in_recess and tight else 0.5 if seated and in_recess else 0.25 if seated else 0.0
    tbar_c = yp + y_rot @ np.array([0.0, 0.0, WING_Z0 + WING[2] / 2 + s])
    body_c = yp + y_rot @ np.array([0.0, 0.0, (BODY_Z[0] + BODY_Z[1]) / 2])
    _EP.update(stage=stage, tbar_c=tbar_c, body_c=body_c, reg=(yp.copy(), q_turn),
               info=dict(rel=rel_p, tilt=tilt, depths=depths, rpy=(math.degrees(math.atan2(rel_r[2, 1], rel_r[2, 2])),
                                                                  math.degrees(-math.asin(max(-1.0, min(1.0, rel_r[2, 0])))),
                                                                  math.degrees(math.atan2(rel_r[1, 0], rel_r[0, 0]))), gap=gap, w_off=w_off, w_z=float(w_in_y[2]),
                         w_tilt=w_tilt, tip=tip, seated=seated, in_recess=in_recess, tight=tight, c_tilt=c_tilt,
                         c_move=float(np.linalg.norm(c_pos - c0p))))


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and _EP["reg"] is not None:
        _EP["counted"] = _EP["tick"]
        recent.append((_EP["reg"][0].copy(), _EP["reg"][1]))
        del recent[:-STILL_STEPS]
    still = (len(recent) == STILL_STEPS
             and all(np.abs(p - recent[-1][0]).max() * 1000 < STILL_MM
                     and abs(math.degrees(a - recent[-1][1])) < STILL_DEG for p, a in recent))
    near = 0.0
    if len(_TCP) == 2:
        near = min(min(np.linalg.norm(v - _EP["tbar_c"]), np.linalg.norm(v - _EP["body_c"])) for v in _TCP.values())
    return dict(still=still, near=float(near), done=bool(_EP["stage"] >= 0.75 and near >= CLEAR and still))


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or not _EP["info"]:
        return
    i = _EP["info"]
    r = i["rel"] * 1000
    print(f"DBG step={_EP['step']} rel=({r[0]:.1f},{r[1]:.1f},{r[2]:.1f})mm tilt={i['tilt']:.1f} "
          f"rpy=({i['rpy'][0]:.1f},{i['rpy'][1]:.1f},{i['rpy'][2]:.1f}) "
          f"pins=({i['depths'][0] * 1000:.1f},{i['depths'][1] * 1000:.1f})mm gap={i['gap'] * 1000:.2f}mm "
          f"washer(off={i['w_off'] * 1000:.1f} z={i['w_z'] * 1000:.1f} tilt={i['w_tilt']:.0f}) "
          f"tip=({i['tip'][0] * 1000:.1f},{i['tip'][1] * 1000:.1f},{i['tip'][2] * 1000:.1f}) turns={_EP['turns']:.2f} "
          f"s={_EP['s'] * 1000:.2f} tgt={_EP['target'] * 1000:.2f} F={_EP['force']:.0f}N "
          f"tau={(_EP['fric'] or 0):.2f} seated={i['seated']} recess={i['in_recess']} tight={i['tight']} "
          f"cyl(move={i['c_move'] * 1000:.1f}mm tilt={i['c_tilt']:.1f}) near={t['near']:.3f} still={t['still']} "
          f"stage={_EP['stage']} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        print("DBG   " + " | ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f})" for k, v in _TCP.items()),
              flush=True)


def score(objects) -> float:
    """From the end state (0 for the rest of the run once the washer has come off the inlet, the cylinder has left its
    cradle or the regulator has fallen off the table): 0.25 seated (pins in, flush on the washer); 0.5 also the screw's
    point in the recess; 0.75 also hand-tight. Full score (1.0) is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return 0.0 if _EP["failed"] else _EP["stage"]


def success(objects) -> bool:
    """Seated, the point in the recess and hand-tight, both grippers 6 cm or more from the T-handle and the body, the
    regulator and the T-handle still for 1 s."""
    t = _tally(objects)
    return bool(t["done"] and not _EP["failed"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# As the manual says. The right gripper, level and pointing forward, takes the regulator's body from the robot's side
# and carries it upright (the washer stays on the inlet) to in front of the valve, the yoke's window level with the
# post. It slides the yoke over the post, lifts it until the pins nearly meet the valve's face, then presses up while
# moving it slowly back and forth along the post until the pins drop into their holes and the washer meets the face,
# and holds it there. The left gripper, pointing down, takes the T-handle and turns it clockwise half a turn at a time:
# it lets go, rises clear of the handle, turns back, comes down and takes it again. The tenth half turn runs the point
# into the recess and tightens it. The right gripper lets go and moves away, then the left.
# O2_ONE=1 runs it one hand at a time: the right gripper seats the regulator the same way, then lets go of it and moves
# away before the left gripper turns the T-handle (as a single hand would have to).

ONE_ARM = os.environ.get("O2_ONE") == "1"
OPEN_R, BODY_PINCH = -0.030, -(BODY_R - 4.5 * MM)
OPEN_L, WING_PINCH = -0.022, -(WING[1] / 2 - 6.0 * MM)
CLOSED = 0.0
LEVEL_PITCH = 5.0
GRIP_Z = -26 * MM  # the right gripper's centre on the body's axis, this far below the inlet face
LIFT_EXTRA = float(os.environ.get("O2_LIFT_EXTRA", "8.0")) * MM  # pressed up past the seat (the arm gives under load)
FEEL_EXTRA = float(os.environ.get("O2_FEEL_EXTRA", "3.0")) * MM  # pressed up lightly while feeling for the holes
SAG = float(os.environ.get("O2_SAG", "12.5")) * MM  # the right arm's sag holding the regulator at the valve (runs)
X_TRIM = float(os.environ.get("O2_XTRIM", "2.0")) * MM  # the right arm's lag along x carrying it (from our runs)
SLIDE_SAG = float(os.environ.get("O2_SLIDE_SAG", "2.0")) * MM  # more sag as the arm reaches over the post (runs)
Y_TRIM = float(os.environ.get("O2_YTRIM", "2.5")) * MM  # the right arm drifts this far to its right reaching in (runs)
SWEEP = float(os.environ.get("O2_SWEEP", "5.0")) * MM  # the feel along the post: this far either way
STROKES = 11
LAST_EXTRA = 35.0  # the last half turn goes this much further (degrees): the screw tightens
WING_GRIP = -1 * MM  # the turning gripper's centre above the T-handle's middle
T_TAKE, T_TURN, T_LETGO, T_RISE, T_BACK, T_DOWN = 0.6, 0.55, 0.4, 0.5, 0.3, 0.7  # one half turn's phases (s)


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


FORWARD = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) @ grip_rotation(0.0)  # level, along +x
LEVEL = _ry(LEVEL_PITCH) @ FORWARD  # pointing forward 5 degrees down, fingers closing along y
ROLL_FIX = float(os.environ.get("O2_ROLL", "4.0"))  # the arm rolls this much under the regulator's weight (runs)


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


CARRY = _rx(ROLL_FIX) @ LEVEL  # the same, turned back against that roll while it carries the regulator


def _down(yaw_deg: float) -> np.ndarray:
    return grip_rotation(math.radians(yaw_deg))


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


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
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


def _idle(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), CLOSED, "stay put", rot)]


def _seat_origin(dz: float = 0.0, dx: float = 0.0) -> np.ndarray:
    """The yoke's origin (the inlet's centre on its face) when fitted, world frame, shifted."""
    return VALVE + np.array([dx, 0.0, SEAT_Z + dz])


WING_MID0 = SEAT_Z + WING_Z0 + WING[2] / 2  # the T-handle's middle above the valve frame, yoke seated, screw backed off


def _half_turns(n: int, yaw0: float, label: str, last_holds: bool) -> list[Waypoint]:
    """The T-handle turned clockwise half a turn ``n`` times by a gripper pointing down (it starts above it, open, at
    ``yaw0``): down, take, turn 180 degrees in four steps (following the screw down), let go, rise clear, turn back.
    ``last_holds``: the last half turn ends holding the handle."""
    out: list[Waypoint] = []
    for k in range(n):
        z = VALVE[2] + WING_MID0 - k * PITCH / 2 + WING_GRIP
        grip = np.array([XO, 0.0, z])
        out += [_w(T_DOWN, grip, OPEN_L, f"{label}: down on the T-handle ({k + 1} of {n})", _down(yaw0)),
                _w(T_TAKE, grip, WING_PINCH, f"{label}: take it", _down(yaw0))]
        for j in range(1, 5):
            out.append(_w(T_TURN, grip - [0.0, 0.0, PITCH / 2 * j / 4], WING_PINCH, f"{label}: turn it clockwise",
                          _down(yaw0 - 45.0 * j)))
        if last_holds and k == n - 1:
            out.append(_w(T_TURN, grip - [0.0, 0.0, PITCH / 2], WING_PINCH, f"{label}: tighten it by hand",
                          _down(yaw0 - 180.0 - LAST_EXTRA)))
            break
        low = grip - [0.0, 0.0, PITCH / 2]
        clear = low + [0.0, 0.0, 0.027]
        out += [_w(T_LETGO, low, OPEN_L, f"{label}: let go", _down(yaw0 - 180.0)),
                _w(T_RISE, clear, OPEN_L, f"{label}: rise clear of the handle", _down(yaw0 - 180.0))]
        for j in range(1, 5):
            out.append(_w(T_BACK, clear, OPEN_L, f"{label}: turn back", _down(yaw0 - 180.0 + 45.0 * j)))
    return out


def _dur(wps) -> float:
    return float(sum(w.duration for w in wps))


def _right_seat() -> list[Waypoint]:
    """Pick the regulator, slide the yoke over the post and lift it onto its pins; ends holding it up."""
    grip_off = np.array([0.0, 0.0, GRIP_Z])
    pick = REG_START + grip_off
    front = pick - [0.045, 0.0, 0.0]
    high = np.array([front[0], front[1], TOP + 0.16])
    out = _lead_in("right", np.array([0.25, -0.22, TOP + 0.20]), LEVEL, OPEN_R, "level, high on the right")
    out += [_w(1.0, np.array([0.19, -0.15, TOP + 0.19]), OPEN_R, "over the table in front of the regulator", LEVEL),
            _w(0.8, high, OPEN_R, "over the table in front of the regulator", LEVEL),
            _w(1.5, front, OPEN_R, "down in front of the regulator's body", LEVEL),
            _w(2.0, pick, OPEN_R, "fingers either side of the body", LEVEL),
            _w(0.4, pick, OPEN_R, "settle", CARRY),
            _w(0.8, pick, BODY_PINCH, "take the body", LEVEL)]
    # the yoke's window level with the post: the inlet face midway between the pins' ends just clearing the post's
    # bottom face and the screw's point just clearing its top face
    slide = (-P_HT - PIN_L + P_HT - TIP0) / 2  # the inlet face's height in the valve frame
    low = SEAT_Z - slide
    pre = _seat_origin(dz=-low + SAG, dx=-0.055 + X_TRIM) + grip_off
    on = pre + [0.055, Y_TRIM, SLIDE_SAG]
    touch = on + [0.0, 0.0, (-P_HT - PIN_L - 1.0 * MM) - slide]  # the pins' ends 1 mm under the face
    up = _seat_origin(dz=SAG + SLIDE_SAG + LIFT_EXTRA, dx=X_TRIM) + grip_off  # held up firmly
    feel = _seat_origin(dz=SAG + SLIDE_SAG + FEEL_EXTRA, dx=X_TRIM) + grip_off  # pressed up lightly
    out += [_w(1.2, pick + [0.0, 0.0, 0.03], BODY_PINCH, "lift it, upright", CARRY),
            _w(2.5, pre, BODY_PINCH, "carry it in front of the valve, the window level with the post", CARRY),
            _w(0.5, pre, BODY_PINCH, "settle", CARRY),
            _w(4.0, on, BODY_PINCH, "slide the yoke over the post", CARRY),
            _w(0.5, on, BODY_PINCH, "settle", CARRY),
            _w(1.0, touch, BODY_PINCH, "lift it until the pins nearly meet the valve's face", CARRY),
            _w(0.8, feel, BODY_PINCH, "press up lightly: the pins drop into their holes", CARRY),
            _w(1.0, feel - [SWEEP, 0.0, 0.0], BODY_PINCH, "feel along the post", CARRY),
            _w(2.0, feel + [SWEEP, 0.0, 0.0], BODY_PINCH, "feel along the post", CARRY),
            _w(1.0, feel, BODY_PINCH, "feel along the post", CARRY),
            _w(0.8, up, BODY_PINCH, "pins in: hold it up against the valve", CARRY)]
    return out


def _right() -> list[Waypoint]:
    out = _right_seat()
    up = np.asarray(out[-1].pos, float)
    if ONE_ARM:  # one hand at a time: let go of the regulator before the T-handle is turned
        t_turn = _dur(_half_turns(STROKES, YAW_A, "x", True)) + 0.6 + 1.8
        return out + [_w(0.6, up, OPEN_R, "let go of it", CARRY),
                      _w(1.2, up - [0.07, 0.0, 0.0], OPEN_R, "back out", CARRY),
                      _w(1.4, np.array([0.16, -0.10, TOP + 0.09]), OPEN_R, "move away", CARRY),
                      _w(t_turn + 6.0, np.array([0.16, -0.10, TOP + 0.09]), OPEN_R, "stay clear", CARRY)]
    t_hold = _dur(_half_turns(STROKES, YAW_A, "x", True)) + 0.6
    out += [_w(t_hold, up, BODY_PINCH, "hold it up against the valve while the T-handle is turned", CARRY),
            _w(0.6, up, OPEN_R, "let go", CARRY),
            _w(1.2, up - [0.07, 0.0, 0.0], OPEN_R, "back out", CARRY),
            _w(1.4, np.array([0.16, -0.10, TOP + 0.09]), OPEN_R, "move away", CARRY),
            _w(6.0, np.array([0.16, -0.10, TOP + 0.09]), OPEN_R, "stay clear", CARRY)]
    return out


def _left() -> list[Waypoint]:
    t_seated = _dur(_right_seat()) + (3.2 if ONE_ARM else 0.0)  # one hand at a time: after the right has let go
    above = np.array([XO, 0.0, VALVE[2] + WING_MID0 + WING_GRIP + 0.027])  # fingertips clear over the T-handle
    aside = above + [0.0, 0.08, 0.0]  # waits to the left, clear of the regulator being carried in
    out = _lead_in("left", aside, _down(YAW_A), OPEN_L, "beside the valve, pointing down")
    out += [_w(t_seated - _dur(out) - 1.2, aside, OPEN_L, "wait for the regulator to be held on its pins", _down(YAW_A)),
            _w(1.2, above, OPEN_L, "over the T-handle", _down(YAW_A))]
    out += _half_turns(STROKES, YAW_A, "tighten", True)
    last = np.asarray(out[-1].pos, float)
    end_yaw = _down(YAW_A - 180.0 - LAST_EXTRA)
    out += [_w(0.6, last, WING_PINCH, "hold it tight", end_yaw),
            _w(0.6 + 1.2, last, WING_PINCH, "hold the T-handle while the right hand lets go", end_yaw),
            _w(0.5, last, OPEN_L, "let go of the T-handle", end_yaw),
            _w(0.8, last + [0.0, 0.0, 0.03], OPEN_L, "rise", end_yaw),
            _w(1.4, np.array([0.16, 0.20, TOP + 0.20]), OPEN_L, "move away", end_yaw),
            _w(6.0, np.array([0.16, 0.20, TOP + 0.20]), OPEN_L, "stay clear", end_yaw)]
    return out


def demo():
    return {"left": _left(), "right": _right()}
