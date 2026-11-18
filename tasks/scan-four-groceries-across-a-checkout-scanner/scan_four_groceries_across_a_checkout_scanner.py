"""A checkout counter on the station table: the end of the conveyor belt on the robot's left, a hands-free presentation
barcode scanner in the middle (a Zebra DS9308-SR, USD 385, standing on the counter, its window facing the robot), and
the bagging shelf on the robot's right. Four groceries stand or lie at the belt's end, each with a real UPC-A barcode
printed on one side, none facing the scanner.

Task (instruction): ring up the four groceries on the scanner, each read once, and put them on the bagging shelf. The
instruction does not say how. The belt's end and the bagging shelf are farther apart than one arm reaches: the right
arm's gripper reaches no further left than 45 cm from the middle and the groceries stand 45 to 64 cm to the left; the
left arm's gripper reaches no further right than 45 cm and the shelf starts 46 cm to the right. So a cashier's two-hand
relay is needed: one hand takes an item from the belt, turns its barcode to the scanner and presents it in front of the
window, the other takes it on to the bagging shelf.

The scanner model is disclosed and uses the DS9308's own numbers (Zebra's DS9300 Series spec sheet and DS9308 Product
Reference Guide). A code is read at a control step (20 Hz) when (1) its centre is inside the imager's 52 x 33 degree
field of view, which opens toward the robot from the window's middle, 0 to 22.4 cm in front of the window (the range
for a 100 % UPC-A); (2) it faces the window within 60 degrees (skew and pitch tolerance); (3) it moves slower than
3.05 m/s (swipe speed); and (4) nothing of either arm (links, gripper, fingers) is between the window's middle and any
of nine points spread over the code. After a read the same code is not read again until it has been out of view for
0.5 s, the scanner's default "Timeout Between Decodes, Same Symbol" ("The barcode must be out of the field of view for
the timeout period before the scanner reads the same consecutive symbol"); presenting it again after that rings it up a
second time (a double charge). In hands-free mode the scanner's LED is solid green and "turns off after decode": it
goes out for 0.5 s on each read.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top (the counter) is at z 0.7475.
- scanner, fixed: 14.5 cm tall, 8.6 cm wide, 8.3 cm deep, its oval base centred at (0.373, 0.150); the head's front face
  at x 0.335 carries the window (64 x 50 mm, its middle 10 cm above the counter); the LED lens on the head's top.
- belt end, fixed (flush black rubber): x 0.02 to 0.40, y 0.455 to 0.645.
- groceries, loose, each with a 12-digit UPC-A (100 % size: 0.33 mm modules, 22.85 mm bars), bars upright:
  spaghetti box 264 x 71 x 41 mm, 0.48 kg, lying along x at (0.20, 0.49), its code on its left face (+y) 7 cm from its
  middle away from the robot; soup can (condensed, 211 x 400 can) 68 mm x 102 mm, 0.35 kg, standing at (0.045, 0.605),
  code facing 160 deg (back toward the robot); glass jar (16 oz salsa) 75 mm x 127 mm, 0.65 kg, standing at
  (0.15, 0.605), code facing +y; water bottle (16.9 fl oz) 65 mm x 204 mm, 0.52 kg, standing at (0.26, 0.605), code
  facing +y.
- bagging shelf, fixed (stainless, flush): x -0.05 to 0.40, y -0.645 to -0.46 (its inner edge marked by a yellow line).

Score: 0 for the rest of the run once the jar falls faster than 1 m/s (dropped about 5 cm or more) or tumbles faster
than 5 rad/s (knocked over), or any grocery falls off the counter. Otherwise 0.25 per item checked out: read exactly
once and ending on the bagging shelf (its whole footprint past the line, resting on the shelf, still for 1 s, every
finger 2 cm or more from it). An item read but left on the counter earns nothing (it has not reached the customer); an
item read twice or more earns nothing (charged twice); an item that reaches the shelf before it has been read scores 0
for the rest of the run (it left unpaid). Full score (1.0, success): all four checked out, held for 1 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, on_reset, on_step  # noqa: E402
from bodykit import rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
# The file keeps what happened over the run itself (reads, an item bagged unpaid, a dropped jar): score read at the end.
SCORE_OVER = "end"
DT = 0.05  # one control step at 20 Hz

# -- UPC-A ------------------------------------------------------------------------------------------------------------
UPC_L = ("0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011")
UPC_R = tuple("".join("1" if c == "0" else "0" for c in s) for s in UPC_L)
MODULE = 0.00033  # 100 % size
BAR_H = 0.02285  # data bars at 100 %
GUARD_DROP = 5 * MODULE  # guard bars run 5 modules lower
QUIET = 9 * MODULE
CODE_W = 95 * MODULE + 2 * QUIET  # 37.29 mm with the quiet zones


def upc_check(d11: str) -> str:
    s = 3 * sum(int(c) for c in d11[0::2]) + sum(int(c) for c in d11[1::2])
    return str((10 - s % 10) % 10)


def upc_modules(d12: str) -> str:
    return ("101" + "".join(UPC_L[int(c)] for c in d12[:6]) + "01010" + "".join(UPC_R[int(c)] for c in d12[6:])
            + "101")


def upc_bars(d12: str) -> list:
    """(centre offset from the symbol's middle along its width (m), width (m), guard bar?) for each dark bar."""
    m = upc_modules(d12)
    guard = set(range(0, 3)) | set(range(45, 50)) | set(range(92, 95))
    out, i = [], 0
    while i < len(m):
        if m[i] == "1":
            j = i
            while j < len(m) and m[j] == "1":
                j += 1
            out.append((((i + j) / 2 - 47.5) * MODULE, (j - i) * MODULE, i in guard))
            i = j
        else:
            i += 1
    return out


def upc(d11: str) -> str:
    return d11 + upc_check(d11)


# -- colours ----------------------------------------------------------------------------------------------------------
INK, WHITE, PAPER = (0.03, 0.03, 0.04), (0.97, 0.97, 0.96), (0.93, 0.92, 0.88)
STEEL, TIN, DARK, GLASS = (0.70, 0.72, 0.74), (0.78, 0.79, 0.80), (0.10, 0.10, 0.11), (0.06, 0.09, 0.12)
RED, YELLOW, BLUE, CAP_BLUE = (0.78, 0.10, 0.08), (0.96, 0.80, 0.25), (0.18, 0.42, 0.82), (0.10, 0.30, 0.78)
SALSA, LID, WATER, PASTA_BLUE = (0.55, 0.16, 0.08), (0.12, 0.32, 0.16), (0.80, 0.88, 0.94), (0.10, 0.22, 0.55)
LED_ON, RUBBER, LINE = (0.35, 1.00, 0.35), (0.09, 0.09, 0.10), (0.95, 0.80, 0.10)
SHELF_STEEL = (0.50, 0.53, 0.57)  # brushed stainless, darker than the counter
LED_OFF, SCAN_GREY = (0.07, 0.10, 0.08), (0.30, 0.31, 0.33)


# -- the groceries ----------------------------------------------------------------------------------------------------
# Each item's frame: origin at the middle of its base (a lying box: the middle of its bottom face), z up, its code
# facing +x (local). ``start``: (x, y) of the origin and the yaw (deg) that turns the code to face that world direction.


def _cyl_code(d12: str, r: float, zc: float, panel: bool) -> list:
    """Upright bars of a UPC-A wrapped round a cylinder of radius ``r``, its middle facing +x at height ``zc``; on a
    white panel (quiet zones included) when ``panel``."""
    out = []
    rb = r + 0.0004
    if panel:
        half = (CODE_W / 2 + 0.002) / r
        n = 6
        for k in range(n):
            a0, a1 = -half + 2 * half * k / n, -half + 2 * half * (k + 1) / n
            a = (a0 + a1) / 2
            chord = 2 * r * math.sin((a1 - a0) / 2) + 0.0004
            out.append(BoxShape(((r + 0.0002) * math.cos(a), (r + 0.0002) * math.sin(a), zc - 0.0007),
                                (0.0003, chord, BAR_H + 0.0045), WHITE, yaw=a, collide=False))
    for s, w, guard in upc_bars(d12):
        a = s / rb
        h = BAR_H + (GUARD_DROP if guard else 0.0)
        zm = zc + BAR_H / 2 - h / 2
        out.append(BoxShape((rb * math.cos(a), rb * math.sin(a), zm), (0.0003, w, h), INK, yaw=a, collide=False))
    return out


def _flat_code(d12: str, x_face: float, zc: float, yc: float = 0.0) -> list:
    """Upright bars of a UPC-A on a flat face at x = ``x_face`` (facing +x), centred at (yc, zc)."""
    out = []
    for s, w, guard in upc_bars(d12):
        h = BAR_H + (GUARD_DROP if guard else 0.0)
        out.append(BoxShape((x_face + 0.00025, yc + s, zc + BAR_H / 2 - h / 2), (0.0003, w, h), INK, collide=False))
    return out


CAN_CODE, JAR_CODE, BOTTLE_CODE, BOX_CODE = upc("40010750305"), upc("40045416001"), upc("40050016900"), upc(
    "40045426402")


def _can_shapes() -> list:
    """A 211 x 400 can (2 11/16 x 4 in): steel body, rolled seams top and bottom, a paper label (red above, white
    below) with the code on the white part."""
    out = [CylShape((0.0, 0.0, 0.002), 0.0342, 0.004, TIN), CylShape((0.0, 0.0, 0.0508), 0.0330, 0.0936, TIN),
           CylShape((0.0, 0.0, 0.0996), 0.0342, 0.004, TIN),
           CylShape((0.0, 0.0, 0.0730), 0.03318, 0.0420, RED, collide=False),
           CylShape((0.0, 0.0, 0.0300), 0.03318, 0.0440, WHITE, collide=False)]
    return out + _cyl_code(CAN_CODE, 0.03318, 0.0300, panel=False)


def _jar_shapes() -> list:
    """A 16 oz glass salsa jar: body 75 mm across, a shoulder and neck, a 70 mm lug lid; a paper label round its
    middle with the code on a white panel."""
    out = [CylShape((0.0, 0.0, 0.0500), 0.0375, 0.1000, SALSA), CylShape((0.0, 0.0, 0.1030), 0.0350, 0.0060, SALSA),
           CylShape((0.0, 0.0, 0.1095), 0.0330, 0.0070, GLASS), CylShape((0.0, 0.0, 0.1200), 0.0345, 0.0140, LID),
           CylShape((0.0, 0.0, 0.0540), 0.03775, 0.0680, YELLOW, collide=False)]
    return out + _cyl_code(JAR_CODE, 0.03775, 0.0540, panel=True)


def _bottle_shapes() -> list:
    """A 16.9 fl oz (500 ml) PET water bottle, 65 mm across and 204 mm tall with its cap; a label with the code."""
    out = [CylShape((0.0, 0.0, 0.0750), 0.0325, 0.1500, WATER), CylShape((0.0, 0.0, 0.1550), 0.0285, 0.0100, WATER),
           CylShape((0.0, 0.0, 0.1650), 0.0225, 0.0100, WATER), CylShape((0.0, 0.0, 0.1750), 0.0160, 0.0100, WATER),
           CylShape((0.0, 0.0, 0.1815), 0.0170, 0.0030, WATER), CylShape((0.0, 0.0, 0.1870), 0.0135, 0.0080, WATER),
           CylShape((0.0, 0.0, 0.1975), 0.0150, 0.0130, CAP_BLUE),
           CylShape((0.0, 0.0, 0.0875), 0.03270, 0.0550, BLUE, collide=False)]
    return out + _cyl_code(BOTTLE_CODE, 0.03270, 0.0850, panel=True)


BOX_L, BOX_W, BOX_H = 0.264, 0.071, 0.041
BOX_CODE_Y = -0.070  # the code's middle, 7 cm from the box's middle along its length


def _box_shapes() -> list:
    """A 1 lb spaghetti box (264 x 71 x 41 mm) lying on its broad face, its length along local y; the code on its
    +x face 7 cm from the middle, on a white panel; a window panel on its top."""
    out = [BoxShape((0.0, 0.0, BOX_H / 2), (BOX_W, BOX_L, BOX_H), PASTA_BLUE),
           BoxShape((BOX_W / 2 + 0.0001, BOX_CODE_Y, BOX_H / 2), (0.0002, CODE_W + 0.004, BAR_H + 0.006), WHITE,
                    collide=False),
           BoxShape((0.0, 0.0, BOX_H + 0.0001), (BOX_W - 0.012, BOX_L - 0.05, 0.0002), (0.85, 0.75, 0.45),
                    collide=False)]
    return out + _flat_code(BOX_CODE, BOX_W / 2 + 0.0002, BOX_H / 2, BOX_CODE_Y)


class Item:
    """A grocery: its shapes, mass, how it stands, where its code is (local), its size for distance tests, and the
    demo's grips (TCP height above its base for each hand, finger targets). A grip's TCP is at least 5.3 cm below the
    item's top less 5 mm: the fingers' carriages, 5.4 to 7.4 cm above the TCP, meet over the gripper's middle."""

    def __init__(self, name, shapes, mass, start, yaw_deg, code, code_c, code_r, kind, size, grip_l, grip_r, width,
                 friction=(0.8, 0.7)):
        self.name, self.mass, self.start, self.yaw_deg, self.code = name, mass, start, yaw_deg, code
        self.code_c = np.asarray(code_c, float)  # code centre (local)
        self.code_r = code_r  # radius of the code's surface (cylinders) or None (flat)
        self.kind, self.size = kind, size  # "cyl": (r, height); "box": (half x, half y, height)
        self.grip_l, self.grip_r, self.width = grip_l, grip_r, width
        self.body = Body.build(name, pos=(float(start[0]), float(start[1]), TOP + 0.001), mass=mass,
                               shapes=shapes, yaw=math.radians(yaw_deg), friction=friction, solver_iterations=24)

    def code_points(self) -> np.ndarray:
        """Nine points spread over the code's bars (local)."""
        pts = []
        for u, v in itertools.product((-0.42, 0.0, 0.42), (-0.4, 0.0, 0.4)):
            s, z = u * 95 * MODULE, self.code_c[2] + v * BAR_H
            if self.code_r is None:
                pts.append([self.code_c[0], self.code_c[1] + s, z])
            else:
                a = s / self.code_r
                pts.append([self.code_r * math.cos(a), self.code_r * math.sin(a), z])
        return np.array(pts)


ITEMS = [
    Item("spaghetti_box", _box_shapes(), 0.48, (0.20, 0.49), 90.0, BOX_CODE, (BOX_W / 2, BOX_CODE_Y, BOX_H / 2), None,
         "box", (BOX_W / 2, BOX_L / 2, BOX_H), grip_l=0.0205, grip_r=0.0205, width=BOX_W, friction=(0.7, 0.6)),
    Item("soup_can", _can_shapes(), 0.35, (0.045, 0.605), 160.0, CAN_CODE, (0.03318, 0.0, 0.0300), 0.03318,
         "cyl", (0.0342, 0.1016), grip_l=0.075, grip_r=0.055, width=0.066),
    Item("salsa_jar", _jar_shapes(), 0.65, (0.15, 0.605), 90.0, JAR_CODE, (0.03775, 0.0, 0.0540), 0.03775,
         "cyl", (0.0375, 0.1270), grip_l=0.090, grip_r=0.080, width=0.075, friction=(0.75, 0.65)),
    Item("water_bottle", _bottle_shapes(), 0.52, (0.26, 0.605), 90.0, BOTTLE_CODE, (0.03270, 0.0, 0.0850), 0.03270,
         "cyl", (0.0325, 0.2040), grip_l=0.156, grip_r=0.156, width=0.065),
]
BY_NAME = {it.name: it for it in ITEMS}
JAR = "salsa_jar"

# -- the scanner (Zebra DS9308-SR presentation imager), the belt's end and the bagging shelf ---------------------------
# 14.5 cm tall, 8.6 cm wide, 8.3 cm deep; it stands on the counter upright, its window facing the robot. Its imager
# looks out of the window's middle (W): field of view 52 x 33 deg (half angles 26 and 16.5), 100 % UPC-A read from 0
# to 22.4 cm, skew and pitch tolerance 60 deg, swipe speed up to 3.05 m/s (Zebra DS9300 Series spec sheet).
SCAN_C = (0.373, 0.150)  # the middle of its base on the counter
HEAD_X, HEAD_D, HEAD_W, HEAD_Z = -0.008, 0.060, 0.086, (0.048, 0.145)  # head: centre x (from SCAN_C), depth, width, z
BASE_R, BASE_H = 0.042, 0.024
WIN_W, WIN_H, WIN_ZC = 0.064, 0.050, 0.100  # the window on the head's front face: width, height, centre height
XW = SCAN_C[0] + HEAD_X - HEAD_D / 2  # the window's plane (0.335)
W_LENS = np.array([XW, SCAN_C[1], TOP + WIN_ZC])  # where the imager looks out from
FOV_H, FOV_V = 26.0, 16.5  # half angles (deg)
READ_RANGE, FACING_DEG, MAX_SPEED = 0.224, 60.0, 3.05
SAME_SYMBOL_STEPS = 10  # 0.5 s out of view before the same code reads again
BLINK_STEPS = 10  # the solid green LED goes out for 0.5 s on each decode
LED_POS = np.array([XW + 0.010, SCAN_C[1], TOP + HEAD_Z[1]])  # the LED lens on the head's top, near its front
HIDE = np.array([0.15, 0.0, TOP - 0.30])  # where the lit lens waits, under the counter, while the LED is out

BELT_X, BELT_Y = (0.02, 0.40), (0.455, 0.645)
SHELF_X, SHELF_Y = (-0.05, 0.40), (-0.645, -0.46)
SHELF_LINE = SHELF_Y[1]


def _scanner_shapes() -> list:
    """The DS9308 in the frame of its base's middle on the counter: an oval base, a neck, the head with its window on
    the front (facing -x, the robot) and its LED lens on top (dark: the lit lens is a separate body)."""
    hx0 = HEAD_X - HEAD_D / 2
    return [CylShape((0.0, 0.0, BASE_H / 2), BASE_R, BASE_H, DARK),
            BoxShape((0.006, 0.0, (BASE_H + HEAD_Z[0]) / 2), (0.030, 0.040, HEAD_Z[0] - BASE_H), DARK),
            BoxShape((HEAD_X, 0.0, (HEAD_Z[0] + HEAD_Z[1]) / 2), (HEAD_D, HEAD_W, HEAD_Z[1] - HEAD_Z[0]), DARK),
            BoxShape((hx0 - 0.0003, 0.0, WIN_ZC), (0.0006, WIN_W, WIN_H), GLASS, collide=False),
            BoxShape((hx0 - 0.0003, 0.0, (HEAD_Z[0] + WIN_ZC - WIN_H / 2) / 2), (0.0006, WIN_W, WIN_ZC - WIN_H / 2 -
                                                                                   HEAD_Z[0] - 0.006), SCAN_GREY,
                     collide=False),
            BoxShape((LED_POS[0] - SCAN_C[0], 0.0, HEAD_Z[1] + 0.0006), (0.012, 0.030, 0.0012), LED_OFF,
                     collide=False)]


def _led_shapes() -> list:
    """The lit LED lens (solid green), slightly larger than the dark one, in the frame of LED_POS."""
    return [BoxShape((0.0, 0.0, 0.0014), (0.014, 0.032, 0.0016), LED_ON, collide=False)]


def _belt_shapes() -> list:
    """The belt's end: black rubber, flush with the counter, with a stainless end plate toward the scanner."""
    cx, cy = (BELT_X[0] + BELT_X[1]) / 2, (BELT_Y[0] + BELT_Y[1]) / 2
    return [BoxShape((0.0, 0.0, 0.0005), (BELT_X[1] - BELT_X[0], BELT_Y[1] - BELT_Y[0], 0.001), RUBBER, collide=False),
            BoxShape((0.0, BELT_Y[0] - cy - 0.012, 0.0005), (BELT_X[1] - BELT_X[0], 0.024, 0.001), STEEL,
                     collide=False)]


def _shelf_shapes() -> list:
    """The bagging shelf: a stainless plate flush with the counter, its inner edge marked with a yellow line."""
    cx, cy = (SHELF_X[0] + SHELF_X[1]) / 2, (SHELF_Y[0] + SHELF_Y[1]) / 2
    return [BoxShape((0.0, 0.0, 0.0005), (SHELF_X[1] - SHELF_X[0], SHELF_Y[1] - SHELF_Y[0], 0.001), SHELF_STEEL,
                     collide=False),
            BoxShape((0.0, SHELF_Y[1] - cy - 0.008, 0.0011), (SHELF_X[1] - SHELF_X[0], 0.016, 0.0004), LINE,
                     collide=False)]


SCANNER = Body.build("presentation_scanner", pos=(SCAN_C[0], SCAN_C[1], TOP), mass=0.318, color=DARK, fixed=True,
                     shapes=_scanner_shapes())
FLASH = Body.build("good_decode_led", pos=tuple(float(v) for v in LED_POS), mass=0.01, color=LED_ON, gravity=False,
                   damping=(5.0, 5.0), shapes=_led_shapes())
BELT = Body.build("belt_end", pos=((BELT_X[0] + BELT_X[1]) / 2, (BELT_Y[0] + BELT_Y[1]) / 2, TOP), mass=1.0,
                  color=RUBBER, fixed=True, shapes=_belt_shapes())
SHELF = Body.build("bagging_shelf", pos=((SHELF_X[0] + SHELF_X[1]) / 2, (SHELF_Y[0] + SHELF_Y[1]) / 2, TOP), mass=1.0,
                   color=STEEL, fixed=True, shapes=_shelf_shapes())
OBJECTS = [SCANNER, BELT, SHELF, FLASH] + [it.body for it in ITEMS]

# -- the arms as boxes (each in its link's frame), from the YAM's collision meshes -----------------------------------
LINK_BOXES = {
    "link2": ((-0.031, -0.0668, -0.0309), (0.031, -0.0001, 0.295)),
    "link3": ((-0.0912, -0.0799, -0.2735), (0.0315, 0.01, 0.0315)),
    "link4": ((-0.0982, -0.013, -0.1023), (0.0175, 0.073, 0.0175)),
    "link5": ((0.0, -0.0285, -0.0356), (0.069, 0.0285, 0.0335)),
    "gripper": ((-0.051, -0.0325, -0.0105), (0.051, 0.0325, 0.0635)),
}
# Each finger in the gripper frame (z toward the tips; the TCP at z 0.134), its inner face at x = q (the finger joint,
# <= 0) for the left finger and x = -q for the right: (z from, z to, thickness along x, half width along y). A negative
# thickness runs inward from the inner face: the finger's carriage, which reaches over the gripper's middle.
FINGER_PARTS = ((0.131, 0.1468, 0.0064, 0.0137), (0.109, 0.131, 0.0100, 0.0201), (0.0635, 0.109, 0.0238, 0.034),
                (0.053, 0.080, -0.048, 0.034))


def arm_boxes(q_arm, q_fingers, base) -> tuple[list, list]:
    """(centre, rotation, half size) of each box of one arm in the world, and the finger boxes alone."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.urdf import forward_kinematics

    q = {n: float(v) for n, v in zip(ARM_JOINTS, q_arm)}
    q.update({n: float(v) for n, v in zip(FINGER_JOINTS, q_fingers)})
    fk = forward_kinematics(YAM_URDF, q)
    out = []
    b = np.asarray(base, float)
    for link, (lo, hi) in LINK_BOXES.items():
        t = fk[link]
        lo, hi = np.array(lo), np.array(hi)
        out.append((t[:3, :3] @ ((lo + hi) / 2) + t[:3, 3] + b, t[:3, :3], (hi - lo) / 2))
    g = fk["gripper"]
    fingers = []
    for k, sign in ((0, -1.0), (1, 1.0)):
        qf = float(q_fingers[k])
        for z0, z1, th, hw in FINGER_PARTS:
            inner = qf if sign < 0 else -qf
            x0, x1 = (inner - th, inner) if sign < 0 else (inner, inner + th)
            x0, x1 = min(x0, x1), max(x0, x1)
            c = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
            fingers.append((g[:3, :3] @ c + g[:3, 3] + b, g[:3, :3], np.array([(x1 - x0) / 2, hw, (z1 - z0) / 2])))
    return out + fingers, fingers


def segments_blocked(starts: np.ndarray, ends: np.ndarray, boxes: list) -> np.ndarray:
    """Which segments start -> end pass through any box (slab test)."""
    hit = np.zeros(len(starts), bool)
    for c, r, h in boxes:
        a = (np.asarray(starts, float) - c) @ r
        d = (np.asarray(ends, float) - c) @ r - a
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h[None] - a) / d
            t2 = (h[None] - a) / d
        par = np.abs(d) < 1e-12
        inside = np.abs(a) <= h[None]
        lo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        tlo, thi = lo.max(1), hi.min(1)
        hit |= (tlo <= thi) & (thi >= 0.0) & (tlo <= 1.0)
    return hit


def box_corners(c, r, h) -> np.ndarray:
    return np.array([c + r @ (h * np.array(s)) for s in itertools.product((-1, 1), repeat=3)])


def item_distance(it: Item, pos, rot, points) -> np.ndarray:
    """Distance from points (world) to the item's outline (a cylinder, or a box)."""
    p = (np.asarray(points, float) - pos) @ rot
    if it.kind == "cyl":
        r, hgt = it.size
        dr = np.maximum(np.hypot(p[:, 0], p[:, 1]) - r, 0.0)
    else:
        hx, hy, hgt = it.size
        dr = np.hypot(np.maximum(np.abs(p[:, 0]) - hx, 0.0), np.maximum(np.abs(p[:, 1]) - hy, 0.0))
    dz = np.maximum(np.maximum(-p[:, 2], p[:, 2] - hgt), 0.0)
    return np.hypot(dr, dz)


def item_outline(it: Item) -> np.ndarray:
    """Points round the item's bottom and top (local), for its lowest point and its footprint."""
    if it.kind == "cyl":
        r, hgt = it.size
        ring = [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    else:
        hx, hy, hgt = it.size
        ring = [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array([[x, y, z] for x, y in ring for z in (0.0, hgt)])


OUTLINES = {it.name: item_outline(it) for it in ITEMS}
CODE_PTS = {it.name: it.code_points() for it in ITEMS}


def window_view(c: np.ndarray, n: np.ndarray) -> list:
    """["window"] when a code (centre c, outward normal n, world) is in the imager's view: inside its 52 x 33 deg field
    of view, 0 to 22.4 cm in front of the window, facing the window within 60 deg."""
    d = XW - c[0]
    if not 0.0 < d <= READ_RANGE:
        return []
    if abs(c[1] - W_LENS[1]) > d * math.tan(math.radians(FOV_H)) or abs(c[2] - W_LENS[2]) > d * math.tan(
            math.radians(FOV_V)):
        return []
    to_lens = W_LENS - c
    if float(n @ to_lens) / float(np.linalg.norm(to_lens)) < math.cos(math.radians(FACING_DEG)):
        return []
    return ["window"]


def window_targets(window: str, pts: np.ndarray) -> np.ndarray:
    """Each code point's line of sight ends at the imager (the window's middle)."""
    return np.repeat(W_LENS[None], len(pts), axis=0)


# -- tracking -------------------------------------------------------------------------------------------------------
_DEBUG = os.environ.get("SCAN_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("SCAN_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}
STILL_STEPS, STILL_MM = 20, 3.0
CLEAR_FINGER = 0.02
JAR_FALL, JAR_SPIN = 1.0, 5.0
ITEM_CREDIT = 0.25  # per item read exactly once and released on the bagging shelf


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, failed="", flash_until=-1, led_on=None, log=[],
               reads={n: 0 for n in BY_NAME}, armed={n: True for n in BY_NAME}, unseen={n: 99 for n in BY_NAME},
               unpaid={n: False for n in BY_NAME}, prev={n: None for n in BY_NAME}, hist={n: [] for n in BY_NAME},
               state={}, view={n: "" for n in BY_NAME}, why={n: "" for n in BY_NAME}, tcp={}, fingers={})


_reset()


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("SCAN_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _arms(base) -> dict:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if (side, "arm") not in _IDS:
            _IDS[(side, "arm")] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[(side, "fingers")] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        qa, qf = q[_IDS[(side, "arm")]], q[_IDS[(side, "fingers")]]
        tcp = np.asarray(ik.tcp_pose(qa, station.BASES[side])[0], float)
        out[side] = (qa, qf, tcp, station.BASES[side])
    return out


def _place(base, name: str, pos) -> None:
    """Put a gravity-free visual body at ``pos`` (world, relative to the env origin), at rest."""
    import torch

    origin = as_numpy(base.scene.env_origins[0])
    pose = np.array([[pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2], 0.0, 0.0, 0.0, 1.0]])
    obj = base.scene[name]
    obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=base.device))
    obj.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32,
                                                                       device=base.device))


def _on_shelf(name: str, pos, rot) -> bool:
    """Its whole footprint past the shelf's line and on the shelf, its lowest point within 1 cm of the counter."""
    pts = pos + OUTLINES[name] @ rot.T
    return bool(pts[:, 1].max() < SHELF_LINE and pts[:, 1].min() > SHELF_Y[0] - 0.01
                and pts[:, 0].min() > SHELF_X[0] - 0.01 and pts[:, 0].max() < SHELF_X[1] + 0.01
                and pts[:, 2].min() < TOP + 0.01)


def _track_inner(env) -> None:
    """Each item's pose and its code; which window sees the code, and the reads; the LED; broken rules; the shelf."""
    _EP["tick"] += 1
    tick = _EP["tick"]
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    arms = _arms(base)
    boxes, fingers = [], []
    for qa, qf, tcp, bs in arms.values():
        b, f = arm_boxes(qa, qf, bs)
        boxes += b
        fingers += f
    finger_pts = np.concatenate([box_corners(c, r, h) for c, r, h in fingers])
    _EP["tcp"] = {k: v[2] for k, v in arms.items()}
    _EP["fingers"] = {k: v[1] for k, v in arms.items()}
    for it in ITEMS:
        n = it.name
        data = base.scene[n].data
        pose = as_numpy(data.root_link_pose_w)[0]
        vel = as_numpy(data.root_com_vel_w)[0] if hasattr(data, "root_com_vel_w") else as_numpy(data.root_vel_w)[0]
        pos, rot = pose[:3] - origin, rotation(pose[3:7])
        outline = pos + OUTLINES[n] @ rot.T
        low = float(outline[:, 2].min())
        near = float(item_distance(it, pos, rot, finger_pts).min())
        st = dict(pos=pos, rot=rot, vel=vel[:3], ang=vel[3:6], low=low, near=near)
        # the code: centre, outward normal, sample points
        c = pos + rot @ it.code_c
        nrm = rot @ np.array([1.0, 0.0, 0.0])
        pts = pos + CODE_PTS[n] @ rot.T
        prev = _EP["prev"][n]
        speed = 0.0 if prev is None else float(np.linalg.norm(c - prev)) / DT
        _EP["prev"][n] = c
        views = window_view(c, nrm)
        seen, why = False, "not in view"
        for w in views:
            if speed > MAX_SPEED:
                why = f"too fast ({speed:.2f} m/s)"
                continue
            if segments_blocked(pts, window_targets(w, pts), boxes).any():
                why = f"covered (from the {w} window)"
                continue
            seen, why = True, w
            break
        if seen and _EP["armed"][n]:
            _EP["reads"][n] += 1
            _EP["armed"][n] = False
            _EP["flash_until"] = tick + BLINK_STEPS
            _EP["log"].append((tick, n, why))
            if _DEBUG:
                print(f"DBG READ {n} #{_EP['reads'][n]} at step {tick} through the {why}: code at "
                      f"({c[0]:.3f},{c[1]:.3f},{c[2] - TOP:.3f}) speed {speed:.2f} m/s", flush=True)
        if seen:
            _EP["unseen"][n] = 0
        else:
            _EP["unseen"][n] += 1
            if _EP["unseen"][n] >= SAME_SYMBOL_STEPS:
                _EP["armed"][n] = True
        _EP["view"][n], _EP["why"][n] = ",".join(views), why
        # rules
        if not _EP["failed"]:
            if low < TOP - 0.05:
                _EP["failed"] = f"the {n} fell off the counter at step {tick}"
            elif n == JAR and vel[2] < -JAR_FALL:
                _EP["failed"] = f"the glass jar fell at {-vel[2]:.2f} m/s at step {tick} (dropped)"
            elif n == JAR and float(np.linalg.norm(vel[3:6])) > JAR_SPIN:
                _EP["failed"] = (f"the glass jar tumbled at {float(np.linalg.norm(vel[3:6])):.1f} rad/s at step {tick} "
                                 f"(knocked over)")
            if _EP["failed"] and _DEBUG:
                print(f"DBG FAILED: {_EP['failed']}", flush=True)
        on = _on_shelf(n, pos, rot)
        if on and _EP["reads"][n] == 0 and not _EP["unpaid"][n]:
            _EP["unpaid"][n] = True
            if _DEBUG:
                print(f"DBG UNPAID: the {n} reached the bagging shelf unread at step {tick}", flush=True)
        hist = _EP["hist"][n]
        hist.append(pos.copy())
        del hist[:-STILL_STEPS]
        still = len(hist) == STILL_STEPS and max(float(np.linalg.norm(h - pos)) for h in hist) * 1000 < STILL_MM
        st.update(on=on, still=still, bagged=bool(on and still and near >= CLEAR_FINGER))
        _EP["state"][n] = st
    lit = tick >= _EP["flash_until"]  # solid green, out for 0.5 s after each decode
    _place(base, FLASH.name, LED_POS if lit else HIDE)
    _EP["led_on"] = lit


def _item_score(n: str) -> float:
    """0.25 when the item was read exactly once and rests, released, on the bagging shelf; otherwise 0 (left on the
    counter, read twice, or bagged unread)."""
    if _EP["unpaid"][n] or _EP["reads"][n] != 1:
        return 0.0
    st = _EP["state"].get(n)
    return ITEM_CREDIT if st and st["bagged"] else 0.0


def _done() -> bool:
    return bool(not _EP["failed"] and all(_item_score(n) >= ITEM_CREDIT - 1e-9 for n in BY_NAME))


def _debug() -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or not _EP["state"]:
        return
    parts = []
    for n, st in _EP["state"].items():
        p = st["pos"]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(st["rot"][2, 2])))))
        parts.append(f"{n}=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:.3f}) tilt={tilt:.0f} reads={_EP['reads'][n]} "
                     f"view={_EP['view'][n] or '-'} why={_EP['why'][n]} near={st['near']:.3f} on={st['on']} "
                     f"still={st['still']} unpaid={_EP['unpaid'][n]}")
    print(f"DBG step={_EP['step']} led={_EP['led_on']} failed={_EP['failed'] or '-'} score={score(None):.2f}",
          flush=True)
    for s in parts:
        print("DBG   " + s, flush=True)
    if _EP["tcp"]:
        print("DBG   " + " | ".join(f"{k} tcp=({t[0]:.3f},{t[1]:.3f},{t[2] - TOP:.3f}) "
                                    f"fingers={_EP['fingers'][k][0]:.4f}" for k, t in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """0 once the jar was dropped or knocked over or any item fell off the counter. Otherwise 0.25 per item read exactly
    once and resting, released, on the bagging shelf (still 1 s, fingers 2 cm or more away); an item left on the
    counter, read twice, or bagged unread earns nothing. Full score (1.0) is success()."""
    if objects is not None and _DEBUG:
        _debug()
    if _EP["failed"]:
        return 0.0
    return round(sum(_item_score(n) for n in BY_NAME), 4)


def success(objects) -> bool:
    """All four items read exactly once and resting on the bagging shelf, released and still (held for 1 s by
    SUCCESS_HOLD_S), nothing dropped or broken."""
    return _done()


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand takes each grocery from the belt's end with its fingers on two sides away from the code (the box 2.5 cm
# from its middle, on the side away from its code; the bottle, too tall to reach over, from the side), draws it clear
# of the others and turns it in the hand by a quarter or half turn so that its code faces the scanner's window (the
# box is lifted to turn), raises it until its code is level with the window and passes it across at about 0.2 m/s 14 to
# 16 cm in front of the window (one read, the LED goes out for 0.5 s) and sets it down on the counter past the
# scanner. The right hand takes it from there round to the bagging shelf. The spaghetti box goes first (it lies along
# the shelf's outer edge), then the can, the jar and the bottle (in a row along the shelf's inner side).
# SCAN_ONE=1 runs it the wrong way: the left hand alone, which passes every item across the scanner and then sets it
# down on the counter as far toward the shelf as it reaches; the right arm stays where it is. Nothing reaches the shelf,
# so it scores 0.

ONE = os.environ.get("SCAN_ONE", "") in ("1", "left")
OPEN = -0.047
PASS_X = 0.14
PASS_Y0 = 0.36  # the pass starts here (the code is still out of view)
HAND_Y = -0.07  # where the left hand sets each item down (the TCP's y)
PRESENT_Z = WIN_ZC  # each code is passed across at the window's height (10 cm above the counter)
SIDE = ("water_bottle",)  # taken from the side at its grip height: the arms cannot reach over its 20 cm
L_SAFE = (0.20, 0.25, TOP + 0.17)  # the left hand's start, over the scanner's left part, clear of the groceries
L_BACK = (0.16, 0.20, TOP + 0.17)
SHELF_SPOTS = {"spaghetti_box": (0.18, -0.605, 0.0), "salsa_jar": (0.06, -0.515, -90.0),
               "soup_can": (0.18, -0.515, -90.0), "water_bottle": (0.30, -0.515, -90.0)}
ONE_SPOTS = {"spaghetti_box": (0.05, -0.15), "soup_can": (0.135, -0.15), "salsa_jar": (0.215, -0.04),
             "water_bottle": (0.125, -0.075)}  # SCAN_ONE: where the left hand alone sets each item down (its TCP)
ONE_PASS_END = 0.0  # SCAN_ONE: the pass ends here, before the hand turns off to each item's spot
ORDER = ("spaghetti_box", "soup_can", "salsa_jar", "water_bottle")
R_WAIT = (0.24, -0.27, TOP + 0.19)
R_VIA = (0.23, -0.40, TOP + 0.19)


def _rot(yaw_deg: float) -> np.ndarray:
    from yam_arena.demo import grip_rotation

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


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    q0, q1 = np.array(READY, float), _good_ik(arm, pos, rot)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, p, finger, label, yaw_deg) -> Waypoint:
    return Waypoint(float(t), tuple(float(v) for v in p), float(finger), label, _rot(yaw_deg))


def _dur(ws) -> float:
    return sum(w.duration for w in ws)


def _close(it: Item) -> float:
    return -(it.width / 2 - 0.010)  # 1 cm inside each face: the finger drive's full force


def _open_for(it: Item, extra: float = 0.012) -> float:
    return max(OPEN, -(it.width / 2 + extra))


def _height(it: Item) -> float:
    return it.size[1] if it.kind == "cyl" else it.size[2]


def _approach(it: Item, x: float, y: float, zg: float, yaw: float, op: float, t_over: float) -> list:
    """Bring the open fingers either side of the item at its grip height: from above, or for an item too tall to reach
    over, from beside it (toward +y, along the fingers' gap)."""
    if it.name in SIDE:
        return [_w(t_over, (x, y - 0.09, TOP + zg + 0.025), op, f"beside the {it.name}", yaw),
                _w(0.7, (x, y - 0.09, TOP + zg), op, "down beside it", yaw),
                _w(1.3, (x, y, TOP + zg), op, f"fingers either side of the {it.name}", yaw)]
    hz = TOP + max(zg + 0.04, _height(it) + 0.025)
    return [_w(t_over, (x, y, hz), op, f"over the {it.name}", yaw),
            _w(1.0, (x, y, TOP + zg), op, f"down to the {it.name}", yaw)]


BOX_GRIP = 0.025  # the left hand takes the box 2.5 cm from its middle, on the side away from the code
L_TURN_BOX = (0.16, 0.385, TOP + 0.16)  # the box is turned here, lifted clear, its code out of the scanner's view
# (it leads the hand by 9.5 cm); the wrist's last joint is 3 deg from its limit there


def _left_item(it: Item, first: bool, waits: dict, set_spot=None) -> tuple[list, dict]:
    """The left hand's moves for one item; returns them and the times (from the start of this item's moves) when it has
    set the item down and when it is clear of it."""
    x, y = it.start
    zg = it.grip_l
    box = it.name == "spaghetti_box"
    if box:
        # its code is 7 cm from its middle away from the robot; the fingers take its long faces 2.5 cm the other way
        gx, gy, yaw0, yaw1 = x - BOX_GRIP, y, 0.0, -90.0
    else:
        gx, gy, yaw0, yaw1 = x, y, it.yaw_deg, 0.0  # fingers either side of the code
    grip = (gx, gy, TOP + zg)
    up = (gx, gy, TOP + zg + (0.01 if it.name in SIDE else 0.03))
    op, cl = _open_for(it, 0.006 if it.name == JAR else 0.012), _close(it)  # the jar: its neighbour is close
    out = []
    if first:
        out += _lead_in("left", L_SAFE, _rot(0.0), op, "clear of the groceries")
    out += _approach(it, gx, gy, zg, yaw0, op, 1.8 if first else 2.2)
    out += [_w(1.0, grip, cl, "close", yaw0),
            _w(0.6, grip, cl, "settle", yaw0),
            _w(0.8, up, cl, "lift", yaw0)]
    # turn the code toward the scanner (at most 85 deg per move, the wrist turning one way), away from the other items
    turn_at = L_TURN_BOX if box else (gx, 0.50, up[2])
    out.append(_w(1.4 if box else 1.0, turn_at, cl, "lift it clear" if box else "draw it out of the row", yaw0))
    turn = yaw1 - yaw0
    k = max(1, int(math.ceil(abs(turn) / 85.0)))
    for i in range(1, k + 1):
        out.append(_w(1.0, turn_at, cl, "turn the code toward the scanner", yaw0 + turn * i / k))
    zp = TOP + zg + (PRESENT_Z - float(it.code_c[2]))  # the item raised so that its code is level with the window
    hand = (PASS_X, HAND_Y)
    sx, sy = hand if set_spot is None else set_spot
    px, y_start = (L_TURN_BOX[0], L_TURN_BOX[1]) if box else (PASS_X, PASS_Y0)  # the box's code leads by 9.5 cm
    out.append(_w(1.4, (px, y_start, zp), cl, "down to the scanner's left end", yaw1))
    if waits.get(it.name, 0.0) > 0:
        out.append(_w(waits[it.name], (px, y_start, zp), cl, "wait for the counter to clear", yaw1))
    pass_end = hand[1] + 0.02 if set_spot is None else ONE_PASS_END
    out.append(_w(2.8, (px, pass_end, zp), cl, "pass it across the scanner", yaw1))
    out.append(_w(0.8, (sx, sy, zp), cl, "on past the scanner", yaw1))
    out.append(_w(1.0, (sx, sy, TOP + zg + 0.002), cl, "set it down", yaw1))
    t_set = _dur(out)
    out.append(_w(0.7, (sx, sy, TOP + zg + 0.002), op, "let go", yaw1))
    out.append(_w(0.5, (sx, sy, TOP + zg + 0.012), op, "lift off", yaw1))
    if box:
        back = (sx, sy + 0.06, TOP + zg + 0.06)  # the box lies along the hand's path: draw back along it
    else:
        back = (sx - 0.08, sy, TOP + zg + 0.012)  # slide the open fingers off sideways
    out.append(_w(0.8, back, op, "draw back", yaw1))
    out.append(_w(1.0, L_BACK, op, "back over the scanner", 0.0))
    return out, dict(t_set=t_set, t_clear=_dur(out))


def _right_item(it: Item) -> list:
    """The right hand's moves for one item, from its waiting pose: take it from the counter past the scanner, carry it
    round to the bagging shelf and set it down."""
    if it.name == "spaghetti_box":
        hx, hy = PASS_X, HAND_Y - BOX_GRIP  # the box's middle
    else:
        hx, hy = PASS_X, HAND_Y
    zg = it.grip_r
    sx, sy, syaw = SHELF_SPOTS[it.name]
    op, cl = _open_for(it), _close(it)
    carry_z = TOP + 0.19
    return _approach(it, hx, hy, zg, 90.0, op, 1.4) + [
        _w(1.0, (hx, hy, TOP + zg), cl, "close", 90.0),
        _w(0.6, (hx, hy, TOP + zg), cl, "settle", 90.0),
        _w(1.0, (hx, hy, carry_z), cl, "lift", 90.0),
        _w(1.4, R_WAIT, cl, "carry it round", 45.0),
        _w(1.2, R_VIA, cl, "toward the shelf", 0.0 if syaw < 0 else 20.0),
        _w(1.2, (sx, sy, carry_z), cl, "over its place", syaw),
        _w(1.2, (sx, sy, TOP + zg + 0.003), cl, "set it down", syaw),
        _w(0.7, (sx, sy, TOP + zg + 0.003), _open_for(it, 0.008), "let go", syaw),
        _w(0.9, (sx, sy, carry_z), _open_for(it, 0.008), "rise", syaw),
        _w(1.2, R_VIA, op, "back", 0.0),
        _w(1.3, R_WAIT, op, "wait", 90.0)]


def _plans() -> tuple[list, list]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    waits: dict = {}
    for _ in range(8):
        left, marks, t = [], {}, 0.0
        for i, name in enumerate(ORDER):
            it = BY_NAME[name]
            moves, m = _left_item(it, i == 0, waits, ONE_SPOTS[name] if ONE else None)
            marks[name] = dict(t_set=t + m["t_set"], t_clear=t + m["t_clear"])
            left += moves
            t += _dur(moves)
        if ONE:
            pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES["right"])
            right = [Waypoint(_dur(left) + 2.0, tuple(float(v) for v in pos), OPEN, "stay put", rot)]
            left.append(_w(2.0, L_BACK, OPEN, "stay clear", 0.0))
            return left, right
        right = _lead_in("right", R_WAIT, _rot(90.0), OPEN, "wait beside the scanner")
        t_r, ok = _dur(right), True
        for i, name in enumerate(ORDER):
            it = BY_NAME[name]
            start = marks[name]["t_clear"] + 0.2
            if start > t_r:
                right.append(_w(start - t_r, R_WAIT, OPEN, "wait", 90.0))
                t_r = start
            moves = _right_item(it)
            k_lift = next(j for j, w in enumerate(moves) if w.label == "carry it round")
            lifted = t_r + _dur(moves[:k_lift + 1]) + 0.3  # carried away from the counter past the scanner
            right += moves
            t_r += _dur(moves)
            if i + 1 < len(ORDER):
                nxt = ORDER[i + 1]
                gap = lifted - (marks[nxt]["t_set"] - 2.2)  # the next item nears the spot ~2 s before it is set down
                if gap > 0:
                    waits[nxt] = waits.get(nxt, 0.0) + gap + 0.3
                    ok = False
        if ok:
            break
    end = max(_dur(left), _dur(right)) + 2.0
    left.append(_w(end - _dur(left), L_BACK, OPEN, "stay clear", 0.0))
    right.append(_w(end - _dur(right), R_WAIT, OPEN, "stay clear", 90.0))
    return left, right


def demo():
    left, right = _plans()
    return {"left": left, "right": right}
