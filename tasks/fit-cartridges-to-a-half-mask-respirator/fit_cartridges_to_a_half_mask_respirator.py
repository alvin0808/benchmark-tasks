"""On the station table: a reusable half-mask respirator (a 3M 6000-series half facepiece, 82 g) lies face-down on its
soft seal, chin toward the robot, both cartridge ports empty; two new organic-vapour cartridges (3M 6001 type, 7.5 cm
across, 2.5 cm thick, 50 g each) lie face-up on two foam pads in front of it, one on each side. Task (instruction): fit
both cartridges to the mask and leave the mask lying on the bench. The instruction does not say how.

How it is done (3M Half Facepiece 7500 Series user instructions, the same 6000-series cartridges and bayonet): "Align
cartridge notch with arrow on facepiece, as shown, and push together (Fig. 1). Turn cartridge clockwise to stop (1/4
turn) (Fig. 2)." The figures show the facepiece held in one hand while the other fits the cartridge.

The bayonet (each port, in the port's frame: z out of the mask along the port's axis, origin on the gasket's face):
- the port is a boss on the mask's rigid frame, its axis tilted 42 deg out from vertical (the cheeks of a face-down
  mask face up and out). A foam gasket ring (r 15.5-21 mm, 2 mm thick, compliant contacts: 10 kN/m per contact, see
  GASKET_K) lies on a lip (r 15-23 mm, 3 mm thick); two slots 10 mm wide cut through both at the arrow
  and opposite it; under the lip, a groove (to r 19 mm, 7 mm deep) with two stops in it: one beside each slot that
  keeps the lugs from going counter-clockwise, one a quarter turn clockwise where they stop. Over the last lug-width
  before the stop, the lip's underside is 0.5 mm higher: a pocket the lug springs into (the click).
- the cartridge has a spigot (r 13 mm, 9 mm long) with two lugs (reaching r 18 mm, 6 mm wide, 3 mm tall) on it; its
  notch marks one lug. With the cartridge's face on the uncompressed gasket the lugs' tops are 0.7 mm above the lip's
  underside, so they pass under the lip only while the cartridge is pushed in, squeezing the gasket. Let go, the gasket pushes
  the cartridge back out until the lugs bear on the lip's underside (or sit in the pockets): it is locked.

The hand difficulty: pushing a cartridge onto a port tilted 42 deg pushes the 82 g mask sideways and over on its
soft seal, and turning it against the squeezed gasket turns the mask with it, unless the other hand holds the mask.
So one hand holds the mask (by the empty port's boss, then by the fitted cartridge) while the other lines the notch
up with the arrow, pushes and turns, then the hands swap roles.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- mask: loose, its seal's footprint 12 x 10 cm centred at (0.215, 0.0), 8 cm tall; ports at y +-5.6 cm, 5.0 cm up.
- cartridges: loose, each on a fixed 9 x 9 x 1.2 cm foam pad (a 4.6 cm square hole under the bayonet), left at
  (0.115, 0.135), right at (0.115, -0.135).

Score (end state; RESP_DEBUG=1 prints DBG lines; RESP_ONE=1 is the one-arm wrong way, RESP_MISALIGN=1 pushes the
cartridges on with the notch 45 deg off the arrow):
- per cartridge: 0.2 pushed onto its port (axis within 8 deg of the port's, centred within 3 mm, its face within
  0.5 mm of the gasket's face or squeezing it: the lugs down through the slots); 0.45 instead when also turned
  clockwise 80 deg or more (the stop is a quarter turn on; with the lugs' play they settle at 95-100 deg in the
  simulation: lugs under the lip, locked). The two add up (0.9 with both locked).
- 1.0 (success, held 1 s): both cartridges turned to the stop, the mask lying face-down on the bench (tilted 15 deg
  or less, its seal within 5 mm of the bench), every finger 15 mm or more from the mask and both cartridges, and the
  mask still (under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, MeshShape, allow_matrix_grips, as_numpy  # noqa: E402
from bodykit import on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001
_E = os.environ.get

# -- colours ---------------------------------------------------------------------------------------------------------
MASK_GREY, FRAME_GREY, SEAL_GREY, CART_MAG, CART_BLACK, LABEL_W, FOAM, GASKET, ARROW, STRAP = (
    (0.55, 0.60, 0.66), (0.32, 0.34, 0.38), (0.62, 0.66, 0.72), (0.78, 0.18, 0.36), (0.12, 0.12, 0.13),
    (0.93, 0.93, 0.90), (0.22, 0.24, 0.26), (0.10, 0.10, 0.10), (0.95, 0.85, 0.15), (0.20, 0.30, 0.55))

# -- the layout ------------------------------------------------------------------------------------------------------
MASK_C = np.array([float(_E("RESP_MX", "0.215")), 0.0])
TILT = math.radians(float(_E("RESP_TILT", "42")))  # each port's axis from vertical, outward
PORT_Y, PORT_Z = 0.056, 0.050  # the gasket face's centre, mask frame
MASK_M = 0.082
CART_M = 0.050
CART_R, CART_T = 0.0375, 0.025
SPIG_R, SPIG_L = 0.0130, 0.009
LUG_R0, LUG_R1, LUG_W, LUG_H = 0.0125, 0.0180, 0.006, 0.003
LUG_TOP = -float(_E("RESP_LUGTOP", "4.3")) * MM  # lug top, cartridge frame (face at 0): 1 mm above the lip's underside
GASKET_T, LIP_T = 0.002, 0.003  # gasket z -2..0, lip z -5..-2
LIP_R0, LIP_R1, GASKET_R0, GASKET_R1 = 0.0150, 0.023, 0.0155, 0.021
GROOVE_R, GROOVE_D = 0.019, 0.007  # groove wall radius; groove z -12..-5
POCKET = float(_E("RESP_POCKET", "0.5")) * MM
SLOT_W = 0.010
GASKET_K = float(_E("RESP_GK", "10000"))  # N/m per contact
GASKET_C = float(_E("RESP_GC", "40"))
PAD_C = {"left": np.array([0.115, 0.135]), "right": np.array([0.115, -0.135])}
PAD_T, PAD_W, PAD_HOLE = 0.012, 0.090, 0.046
SIDES = ("left", "right")
SGN = {"left": 1.0, "right": -1.0}


def _v(p) -> tuple:
    return tuple(float(x) for x in p)


def _rot_axis(axis, a: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def _quat(m: np.ndarray) -> tuple:
    from bodykit import _quat_from_matrix

    return _quat_from_matrix(np.asarray(m, float))


# port frames (mask frame = world-aligned at the start): z = the port's axis out of the mask, x = world x
PORT_N = {s: np.array([0.0, SGN[s] * math.sin(TILT), math.cos(TILT)]) for s in SIDES}
PORT_O = {s: np.array([0.0, SGN[s] * PORT_Y, PORT_Z]) for s in SIDES}
PORT_R = {s: np.column_stack([[1.0, 0.0, 0.0], np.cross(PORT_N[s], [1.0, 0.0, 0.0]), PORT_N[s]]) for s in SIDES}
# the slot (and the arrow) points up the mask, toward its middle: +y_p on the right, -y_p on the left
SLOT_ANG = {"left": -90.0, "right": 90.0}
SLOT_HALF = math.degrees(SLOT_W / 2 / 0.016)  # 17.9 deg at the lugs' radius
LUG_HALF = math.degrees(LUG_W / 2 / 0.0155)  # 11.1 deg
TURN = 90.0  # the stop


def _pbox(side, centre_p, size, ang_deg, color, collide=True) -> BoxShape:
    """A box given in the port's frame (centre, size along (radial, tangential, axial), turned ``ang_deg`` about the
    port's axis), as a shape in the mask's frame."""
    r = PORT_R[side] @ _rot_axis([0, 0, 1], math.radians(ang_deg))
    c = PORT_O[side] + PORT_R[side] @ np.asarray(centre_p, float)
    return BoxShape(_v(c), _v(size), color, collide=collide, quat_xyzw=_quat(r))


def _ring(side, r0, r1, z0, z1, a0, a1, color, step=6.0) -> list:
    """Chord boxes filling the ring r0..r1, z0..z1 (port frame) between angles a0 and a1 (deg)."""
    out = []
    n = max(1, int(math.ceil((a1 - a0) / step)))
    d = (a1 - a0) / n
    for k in range(n):
        a = a0 + d * (k + 0.5)
        w = 2 * r1 * math.tan(math.radians(d / 2)) + 0.0004
        rc = (r0 / math.cos(math.radians(d / 2)) + r1) / 2
        out.append(_pbox(side, (rc * math.cos(math.radians(a)), rc * math.sin(math.radians(a)), (z0 + z1) / 2),
                         (r1 - r0 / math.cos(math.radians(d / 2)), w, z1 - z0), a, color))
    return out


def _arcs_without_slots(side, extra=()) -> list:
    """Angular arcs (deg) of the lip outside the two slots, as (a0, a1, kind) with the pocket arcs split out."""
    s = SLOT_ANG[side]
    cuts = []
    for base in (s, s + 180.0):
        cuts.append((base + SLOT_HALF, base + 180.0 - SLOT_HALF))
    out = []
    for a0, a1 in cuts:
        # the pocket: the last lug-width before the stop, a quarter turn clockwise from the slot
        for base in (s, s + 180.0):
            p0, p1 = base - TURN - LUG_HALF - 1.5, base - TURN + LUG_HALF + 1.5
            for shift in (-360.0, 0.0, 360.0):
                q0, q1 = p0 + shift, p1 + shift
                if q0 >= a0 and q1 <= a1:
                    out += [(a0, q0, "lip"), (q0, q1, "pocket"), (q1, a1, "lip")]
                    break
            else:
                continue
            break
        else:
            out.append((a0, a1, "lip"))
    return out


def _port_shapes(side) -> tuple[list, list]:
    """(rigid shapes, gasket shapes) of one port, in the mask's frame."""
    rigid, gasket = [], []
    # the boss's wall (round the groove), from inside the mask up to the lip
    rigid += _ring(side, GROOVE_R, LIP_R1, -0.022, -GASKET_T - LIP_T, -180.0, 180.0, FRAME_GREY, step=10.0)
    # the floor of the groove (the inhalation valve's seat)
    rigid.append(_pbox(side, (0.0, 0.0, -0.013), (0.026, 0.026, 0.002), 0.0, FRAME_GREY))
    for a0, a1, kind in _arcs_without_slots(side):
        zb = -GASKET_T - LIP_T + (POCKET if kind == "pocket" else 0.0)
        rigid += _ring(side, LIP_R0, LIP_R1, zb, -GASKET_T, a0, a1, FRAME_GREY, step=5.0)
        gasket += _ring(side, GASKET_R0, GASKET_R1, -GASKET_T, 0.0, a0, a1, GASKET, step=8.0)
    # the stops in the groove: beside each slot (no turning counter-clockwise) and a quarter turn clockwise on
    s = SLOT_ANG[side]
    for base in (s, s + 180.0):
        for a in (base + SLOT_HALF + 7.0, base - TURN - LUG_HALF - 7.5):
            rigid.append(_pbox(side, (0.0168 * math.cos(math.radians(a)), 0.0168 * math.sin(math.radians(a)),
                                      -0.0085), (0.0040, 0.0040, 0.0070), a, FRAME_GREY))
    # the arrow on the frame beside the slot (visual): a yellow wedge on the boss's rim, pointing at the slot
    a = math.radians(s)
    rigid.append(_pbox(side, (0.0255 * math.cos(a), 0.0255 * math.sin(a), -GASKET_T - 0.0005), (0.006, 0.004, 0.001),
                       s, ARROW, collide=False))
    return rigid, gasket


def _shell_points() -> np.ndarray:
    """The facepiece's outer surface (mask frame): a convex hull from the seal's oval up to the nose ridge."""
    pts = []
    for z, a, b in ((0.0, 0.060, 0.050), (0.008, 0.060, 0.050), (0.026, 0.054, 0.042)):
        for k in range(32):
            t = 2 * math.pi * k / 32
            pts.append((a * math.cos(t), b * math.sin(t), z))
    for x, y, z in ((-0.045, 0.020, 0.046), (0.0, 0.020, 0.066), (0.040, 0.010, 0.078), (0.052, 0.004, 0.072),
                    (-0.052, 0.012, 0.040)):
        pts += [(x, y, z), (x, -y, z)]
    return np.array(pts)


def _hull_mesh(points: np.ndarray, color) -> MeshShape:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    remap = {int(v): i for i, v in enumerate(used)}
    pts = points[used]
    c = pts.mean(axis=0)
    faces = []
    for tri in hull.simplices:
        f = [remap[int(v)] for v in tri]
        nrm = np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]])
        if np.dot(nrm, pts[f[0]] - c) < 0:
            f = [f[0], f[2], f[1]]
        faces.append(f)
    normals = pts - c
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return MeshShape(pts, np.array(faces, dtype=np.int64), normals, color, approximation="convexHull")


def _check_ports(points: np.ndarray) -> None:
    """The shell stays below each port's floor round its axis, and below the cartridge's face under its footprint."""
    for s in SIDES:
        rel = (points - PORT_O[s]) @ PORT_R[s]
        r = np.linalg.norm(rel[:, :2], axis=1)
        bad = rel[(r < CART_R + 0.004) & (rel[:, 2] > -0.0015)]
        assert len(bad) == 0, f"the shell rises into the {s} cartridge's place: {bad}"


def _mask() -> Body:
    shell = _shell_points()
    _check_ports(shell)
    sh = [_hull_mesh(shell, MASK_GREY)]
    gaskets = []
    for s in SIDES:
        rigid, gasket = _port_shapes(s)
        sh += rigid
        gaskets += gasket
    # the exhalation valve cover at the chin, the frame's bar across the front, the seal's lighter rim (visual)
    sh += [BoxShape((-0.058, 0.0, 0.026), (0.018, 0.038, 0.030), FRAME_GREY),
           BoxShape((-0.002, 0.0, 0.058), (0.030, 0.050, 0.010), FRAME_GREY, collide=False),
           BoxShape((0.0, 0.0, 0.0015), (0.122, 0.064, 0.003), SEAL_GREY, collide=False),
           BoxShape((0.0, 0.0, 0.0015), (0.070, 0.102, 0.003), SEAL_GREY, collide=False)]
    # the head harness's straps, lying slack on the bench behind the mask (visual only: elastic, not modelled)
    for y in (-0.035, 0.035):
        sh.append(BoxShape((0.095, y, 0.0006), (0.075, 0.012, 0.0012), STRAP, collide=False))
    first = len(sh)
    sh += gaskets
    return Body.build("mask", pos=(float(MASK_C[0]), float(MASK_C[1]), TOP + 0.0008), mass=MASK_M, color=MASK_GREY,
                      friction=(0.7, 0.6), shapes=sh, contact_offset=0.001, solver_iterations=32,
                      compliant_shapes=tuple(range(first, len(sh))),
                      compliant=(GASKET_K, GASKET_C, 0.9, 0.8))


def _cartridge(side) -> Body:
    """Cartridge frame: origin at the centre of its back face (the face that meets the gasket), z out of the back
    toward its filter face; the lugs along +-x; the notch marks the +x lug."""
    lz = LUG_TOP - LUG_H / 2
    rc = (LUG_R0 + LUG_R1) / 2
    sh = [CylShape((0.0, 0.0, CART_T / 2), CART_R, CART_T, CART_BLACK),
          CylShape((0.0, 0.0, -SPIG_L / 2), SPIG_R, SPIG_L, CART_BLACK),
          BoxShape((rc, 0.0, lz), (LUG_R1 - LUG_R0, LUG_W, LUG_H), CART_BLACK),
          BoxShape((-rc, 0.0, lz), (LUG_R1 - LUG_R0, LUG_W, LUG_H), CART_BLACK),
          # the label on its face (magenta: organic vapour), and the notch at the +x lug (a white mark on the rim)
          CylShape((0.0, 0.0, CART_T + 0.0003), CART_R - 0.004, 0.0006, CART_MAG, collide=False),
          CylShape((0.0, 0.0, CART_T + 0.0007), 0.016, 0.0004, LABEL_W, collide=False),
          BoxShape((CART_R - 0.002, 0.0, CART_T / 2), (0.0045, 0.006, CART_T + 0.0016), LABEL_W, collide=False)]
    yaw = math.radians(CART_YAW[side])
    p = PAD_C[side]
    return Body.build(f"cartridge_{side}", pos=(float(p[0]), float(p[1]), TOP + PAD_T + 0.0008), mass=CART_M,
                      color=CART_BLACK, friction=(0.7, 0.6), shapes=sh, yaw=yaw, contact_offset=0.001,
                      solver_iterations=32)


def _pad(side) -> Body:
    h, w = PAD_HOLE / 2, PAD_W / 2
    side_w = w - h
    sh = [BoxShape((0.0, (h + w) / 2, PAD_T / 2), (PAD_W, side_w, PAD_T), FOAM),
          BoxShape((0.0, -(h + w) / 2, PAD_T / 2), (PAD_W, side_w, PAD_T), FOAM),
          BoxShape(((h + w) / 2, 0.0, PAD_T / 2), (side_w, PAD_HOLE, PAD_T), FOAM),
          BoxShape((-(h + w) / 2, 0.0, PAD_T / 2), (side_w, PAD_HOLE, PAD_T), FOAM)]
    p = PAD_C[side]
    return Body.build(f"pad_{side}", pos=(float(p[0]), float(p[1]), TOP), mass=0.02, color=FOAM, fixed=True,
                      friction=(0.8, 0.7), shapes=sh)


# -- the grips (needed by the cartridges' start yaw) ----------------------------------------------------------------
OPEN_C = -0.047
GRIP_YAW0 = math.radians(float(_E("RESP_GYAW", "45")))  # the hand turned about the port's axis at the start of a turn
PICK_YAW = {"left": math.radians(float(_E("RESP_PYAW_L", "0"))), "right": math.radians(float(_E("RESP_PYAW_R", "0")))}
MISALIGN = math.radians(-45.0) if _E("RESP_MISALIGN") == "1" else 0.0  # notch 45 deg clockwise of the arrow


def entry_rot(side, twist_deg: float = 0.0, mrot=None) -> np.ndarray:
    """The TCP rotation fitting the ``side`` cartridge: pointing down the port's axis, turned ``twist_deg``
    clockwise (seen from outside) from the start of the turn; ``mrot``: the mask turned so (default: as it starts)."""
    n = PORT_N[side]
    base = _rot_axis([1.0, 0.0, 0.0], -SGN[side] * TILT) @ grip_rotation(0.0)
    r = _rot_axis(-n, math.radians(twist_deg)) @ _rot_axis(n, GRIP_YAW0) @ base
    return r if mrot is None else mrot @ r


def _cart_yaw(side) -> float:
    """The cartridge's yaw on its pad such that, picked straight down with the hand at ``PICK_YAW`` and turned to
    ``entry_rot``, its notch lug lines up with the slot (the arrow)."""
    n = PORT_N[side]
    a = math.radians(SLOT_ANG[side]) + MISALIGN
    d = PORT_R[side] @ np.array([math.cos(a), math.sin(a), 0.0])
    rc_e = np.column_stack([d, np.cross(n, d), n])
    rc_in_g = entry_rot(side).T @ rc_e
    rc0 = grip_rotation(PICK_YAW[side]) @ rc_in_g
    assert abs(rc0[2, 2] - 1.0) < 1e-6
    return math.degrees(math.atan2(rc0[1, 0], rc0[0, 0]))


CART_YAW = {s: _cart_yaw(s) for s in SIDES}
MASK = _mask()
CARTS = {s: _cartridge(s) for s in SIDES}
PADS = {s: _pad(s) for s in SIDES}
OBJECTS = [PADS["left"], PADS["right"], MASK, CARTS["left"], CARTS["right"]]

# -- tracking and scoring -------------------------------------------------------------------------------------------
ON_AXIS_DEG = 8.0
ON_CENTRE = 3.0 * MM
ON_DEPTH = -0.5 * MM  # the cartridge's face at most this far out from the gasket's face (negative: out)
TURNED_DEG = 80.0
MASK_TILT_DEG = 15.0
MASK_UP = 5.0 * MM
CLEAR = 15.0 * MM
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = _E("RESP_DEBUG") == "1"
_DEBUG_EVERY = int(_E("RESP_DEBUG_EVERY", "5"))
_EP: dict = {}
_IDS: dict = {}


def _cart_samples() -> np.ndarray:
    pts = []
    for z in (0.0, CART_T / 2, CART_T):
        for k in range(16):
            t = 2 * math.pi * k / 16
            pts.append((CART_R * math.cos(t), CART_R * math.sin(t), z))
    pts += [(0, 0, CART_T), (0.02, 0, CART_T), (-0.02, 0, CART_T), (0, 0.02, CART_T), (0, -0.02, CART_T)]
    return np.array(pts)


CART_SAMPLES = _cart_samples()
MASK_SAMPLES = np.concatenate([_shell_points()] + [
    PORT_O[s] + (PORT_R[s] @ np.array([[0.023 * math.cos(t), 0.023 * math.sin(t), z]
                                       for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)
                                       for z in (-0.016, -0.002)]).T).T for s in SIDES])


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, recent=[], carts={}, mask_tilt=0.0, mask_up=0.0, near=1.0, gaps={},
               tcp={}, mask_p=np.array([MASK_C[0], MASK_C[1], TOP]), moved=0.0, peak_turn={s: 0.0 for s in SIDES})


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


def cartridge_on_port(mask_pos, mask_rot, cart_pos, cart_rot, side) -> dict:
    """Where a cartridge sits on its port: axis angle (deg), off-centre (m), depth (m, + = squeezing the gasket) and
    how far it is turned clockwise from the slot (deg, 0..180, the two lugs being alike)."""
    o = mask_pos + mask_rot @ PORT_O[side]
    pr = mask_rot @ PORT_R[side]
    rel = pr.T @ (np.asarray(cart_pos) - o)
    cr = pr.T @ np.asarray(cart_rot)
    axis = math.degrees(math.acos(max(-1.0, min(1.0, float(cr[2, 2])))))
    lug = cr[:, 0]
    ang = math.degrees(math.atan2(lug[1], lug[0]))
    turned = (SLOT_ANG[side] - ang) % 180.0  # clockwise seen from outside = negative about the port's axis
    if turned > 170.0:
        turned -= 180.0  # a few degrees counter-clockwise of the slot
    return dict(axis=axis, off=float(np.linalg.norm(rel[:2])), depth=float(-rel[2]), turned=float(turned))


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("RESP_TRACEBACK\n" + traceback.format_exc(), flush=True)
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
    fingers = []
    for side in SIDES:
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[side + "_f"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
            _IDS[side + "_tips"] = [robot.body_names.index(n) for n in ("tip_left", "tip_right")]
        q = as_numpy(robot.data.joint_pos)[0]
        tp, trot = ik.tcp_pose(q[_IDS[side]], station.BASES[side])
        _EP["tcp"][side] = np.asarray(tp, float)
        pr = PORT_R[side]
        hx, ex = pr.T @ np.asarray(trot)[:, 0], pr.T @ entry_rot(side)[:, 0]
        _EP.setdefault("hand", {})[side] = (math.degrees(math.atan2(ex[1], ex[0]) - math.atan2(hx[1], hx[0])) + 180) % 360 - 180
        _EP["gaps"][side] = float(-q[_IDS[side + "_f"]].sum())
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        fingers += [_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                    for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])]
    fingers = np.concatenate(fingers)
    md = as_numpy(base.scene["mask"].data.root_link_pose_w)[0]
    mp, mr = md[:3] - origin, rotation(md[3:7])
    pts = [mp + MASK_SAMPLES @ mr.T]
    carts = {}
    for side in SIDES:
        cd = as_numpy(base.scene[f"cartridge_{side}"].data.root_link_pose_w)[0]
        cp, cr = cd[:3] - origin, rotation(cd[3:7])
        carts[side] = cartridge_on_port(mp, mr, cp, cr, side)
        carts[side]["pos"] = cp
        pts.append(cp + CART_SAMPLES @ cr.T)
        _EP["peak_turn"][side] = max(_EP["peak_turn"][side], carts[side]["turned"])
    _EP["mask_q"] = md[3:7]
    _EP["lport"] = np.concatenate([mp + mr @ PORT_O["left"], mr @ PORT_N["left"]])
    near = float(cKDTree(np.concatenate(pts)).query(fingers, k=1)[0].min())
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(mr[2, 2])))))
    _EP.update(carts=carts, mask_tilt=tilt, mask_up=float(mp[2] - TOP), near=near, mask_p=mp,
               moved=float(np.linalg.norm(mp[:2] - MASK_C)))


def _stage(c: dict) -> float:
    on = c["axis"] <= ON_AXIS_DEG and c["off"] <= ON_CENTRE and c["depth"] >= ON_DEPTH
    if not on:
        return 0.0
    return 0.45 if c["turned"] >= TURNED_DEG else 0.2


def _tally(o) -> dict:
    recent = _EP["recent"]
    if _EP["counted"] != _EP["tick"] and _EP["tick"] > 0:
        _EP["counted"] = _EP["tick"]
        recent.append(np.asarray(_EP["mask_p"], float).copy())
        del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(np.abs(p - recent[-1]).max() * 1000 < STILL_MM for p in recent)
    stages = {s: _stage(_EP["carts"][s]) if s in _EP["carts"] else 0.0 for s in SIDES}
    lying = _EP["mask_tilt"] <= MASK_TILT_DEG and _EP["mask_up"] <= MASK_UP
    clear = _EP["near"] >= CLEAR
    done = bool(all(v >= 0.45 for v in stages.values()) and lying and clear and still)
    return dict(still=still, stages=stages, lying=lying, clear=clear, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    cs = " ".join(f"{s[0].upper()}:ax={c['axis']:.1f} off={c['off'] * 1000:.1f} dep={c['depth'] * 1000:+.2f} "
                  f"turn={c['turned']:.1f} st={t['stages'][s]:.2f}" for s, c in _EP["carts"].items())
    mp = _EP["mask_p"]
    lo = _EP.get("lport", np.zeros(6))
    mq = _EP.get("mask_q", np.zeros(4))
    print(f"DBG step={_EP['step']} {cs} maskpose={mp[0]:.5f},{mp[1]:.5f},{mp[2]:.5f},{mq[0]:.5f},{mq[1]:.5f},{mq[2]:.5f},{mq[3]:.5f} lport=({lo[0]:.4f},{lo[1]:.4f},{lo[2] - TOP:.4f}) ln=({lo[3]:.3f},{lo[4]:.3f},{lo[5]:.3f}) | mask=({mp[0]:.3f},{mp[1]:.3f}) moved={_EP['moved'] * 1000:.1f}mm "
          f"tilt={_EP['mask_tilt']:.1f} up={_EP['mask_up'] * 1000:.1f}mm near={_EP['near'] * 1000:.0f}mm "
          f"still={int(t['still'])} done={int(t['done'])} | "
          + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{v[2] - TOP:.3f}) gap={_EP['gaps'][k] * 1000:.0f} hand={_EP['hand'][k]:.0f}"
                     for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """0.2 per cartridge pushed onto its port, 0.45 per cartridge turned to the stop; full score is success()."""
    t = _tally(objects)
    if _DEBUG:
        _debug(t)
    return float(sum(t["stages"].values()))


def success(objects) -> bool:
    return bool(_tally(objects)["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left hand holds the mask by its empty left port's boss (pinched across it,
# coming down the port's axis) while the right hand takes the right cartridge off its pad, turns it to point down the
# right port's axis with its notch at the arrow, pushes it on (past the gasket's face) and turns it a quarter turn
# clockwise to the stop, and keeps holding it: now it holds the mask. The left hand lets go of the boss, takes the left
# cartridge and fits it the same way. Both hands let go and draw back along the ports' axes.
# RESP_ONE=1: the right hand alone fits the right cartridge, the left arm stays at the ready pose.
ONE = _E("RESP_ONE") == "1"
CART_PINCH = -(CART_R - float(_E("RESP_SQ", "5")) * MM)  # closed 3.5 mm inside each side of the 75 mm cartridge
BOSS_PINCH = -(LIP_R1 - float(_E("RESP_BSQ", "3.0")) * MM)
TCP_IN = float(_E("RESP_TCPIN", "10")) * MM  # the TCP this far in from the cartridge's filter face
PUSH = float(_E("RESP_PUSH", "12")) * MM  # how far past the gasket's face the cartridge's back face is told to go
HOLD_TCP = float(_E("RESP_HOLDZ", "-6")) * MM  # TCP on the boss, port frame z
TWIST_S = float(_E("RESP_TWIST_S", "3.0"))
UNWIND = float(_E("RESP_UNWIND", "9"))  # deg: the hand's turn eased back to this past the stop once there
EASE = float(_E("RESP_EASE", "-0.5")) * MM  # after the turn the hand is told to hold the cartridge's face this far out
OVERTURN = float(_E("RESP_OVER", "20"))  # deg past the stop the hand is told to turn (the wrist gives under load)


# Where the mask lies when the left cartridge is fitted (world position and quaternion xyzw of its origin): the right
# hand holds it by the fitted right cartridge, and the left hand letting go of the boss lets it settle a few mm from
# where it started. Measured in the simulation (deterministic) with RESP_DEBUG=1; the scripted demo is allowed to
# know the simulator's state.
# The same for the right cartridge: the left hand pinching the boss shifts the mask about 3 mm first.
MASK_AT = {"left": [float(v) for v in _E("RESP_MASK_AT_LEFT", "0.21518,-0.00064,0.74794,0.00244,-0.00317,0.00497,"
                                         "0.99998").split(",") if v.strip()],
           "right": [float(v) for v in _E("RESP_MASK_AT_RIGHT", "0.21825,0.00020,0.74750,-0.00003,0.00000,-0.01307,"
                                          "0.99991").split(",") if v.strip()]}


def mask_pose(side):
    """(position, rotation) of the mask when the ``side`` cartridge is fitted."""
    if len(MASK_AT[side]) == 7 and not (ONE and side == "right"):
        return np.array(MASK_AT[side][:3]), rotation(MASK_AT[side][3:])
    return np.array([MASK_C[0], MASK_C[1], TOP + 0.0008]), np.eye(3)


def port_start(side, z: float) -> np.ndarray:
    """A point on the ``side`` port's axis, ``z`` (m) out from the gasket's face, with the mask as it starts."""
    return np.array([MASK_C[0], MASK_C[1], TOP + 0.0008]) + PORT_O[side] + PORT_N[side] * z


def port_world(side, z: float) -> np.ndarray:
    """A point on the ``side`` port's axis, ``z`` (m) out from the gasket's face (world), with the mask lying as
    when that cartridge is fitted."""
    mp, mr = mask_pose(side)
    return mp + mr @ (PORT_O[side] + PORT_N[side] * z)


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


def _slerp_path(p0, r0, p1, r1, seconds, finger, label, n=8) -> list:
    from scipy.spatial.transform import Rotation, Slerp

    sl = Slerp([0.0, 1.0], Rotation.from_matrix([r0, r1]))
    out = []
    for k in range(1, n + 1):
        u = k / n
        out.append(_w(seconds / n, np.asarray(p0) + (np.asarray(p1) - p0) * u, finger, label,
                      sl([u]).as_matrix()[0]))
    return out


def _fit(side: str) -> list:
    """Take the ``side`` cartridge off its pad, push it onto its port with the notch at the arrow, turn it to the stop.
    Ends holding it."""
    p = PAD_C[side]
    rp = grip_rotation(PICK_YAW[side])
    top = TOP + PAD_T + 0.0008 + CART_T
    grip = np.array([p[0], p[1], top - TCP_IN])
    above = grip + np.array([0.0, 0.0, 0.06])
    mr = mask_pose(side)[1]
    r0 = entry_rot(side, 0.0, mr)
    tcp_out = lambda z: port_world(side, z + CART_T - TCP_IN)  # noqa: E731  TCP with the back face z out
    out = [_w(1.5, grip + np.array([0.0, 0.0, 0.02]), OPEN_C, "down beside the cartridge", rp),
           _w(1.0, grip, OPEN_C, "fingers round its rim", rp),
           _w(1.0, grip, CART_PINCH, "take the cartridge", rp),
           _w(1.5, above, CART_PINCH, "lift it off the pad", rp)]
    out += _slerp_path(above, rp, tcp_out(0.05), r0, 3.0, CART_PINCH, "turn it to point down the port, notch at the arrow")
    out += [_w(1.5, tcp_out(0.014), CART_PINCH, "bring it up to the port", r0),
            _w(1.0, tcp_out(0.014), CART_PINCH, "steady, notch at the arrow", r0),
            _w(1.5, tcp_out(0.0), CART_PINCH, "lugs into the slots", r0),
            _w(1.5, tcp_out(-PUSH), CART_PINCH, "push it on, squeezing the gasket", r0)]
    n = 10
    for k in range(1, n + 1):
        out.append(_w(TWIST_S / n, tcp_out(-PUSH), CART_PINCH, "turn it clockwise to the stop",
                      entry_rot(side, (TURN + OVERTURN) * k / n, mr)))
    r1 = entry_rot(side, TURN + OVERTURN, mr)
    r2 = entry_rot(side, TURN + UNWIND, mr)  # the wrist no longer twisting the cartridge (and the mask)
    out += [_w(0.8, tcp_out(-PUSH), CART_PINCH, "hold at the stop", r1),
            _w(1.0, tcp_out(EASE), CART_PINCH, "ease the push and the turn: the gasket seats the lugs", r2)]
    return out


def _let_go(side: str, finger_closed: float, z_tcp: float, rot) -> list:
    pin = port_world(side, z_tcp)
    return [_w(0.8, pin, OPEN_C, "let go", rot),
            _w(1.5, pin + PORT_N[side] * 0.06, OPEN_C, "draw back along the port's axis", rot)]


def demo():
    from yam_arena.constants import READY  # noqa: F401

    right = _lead_in("right", np.array([PAD_C["right"][0], PAD_C["right"][1], TOP + PAD_T + CART_T + 0.06]),
                     grip_rotation(PICK_YAW["right"]), OPEN_C, "above the right cartridge, open")
    right += _fit("right")
    t_right = sum(w.duration for w in right)
    rz = CART_T - TCP_IN + EASE
    r_end = entry_rot("right", TURN + UNWIND, mask_pose("right")[1])
    if ONE:
        right += _let_go("right", CART_PINCH, rz, r_end)
        pr, rr = right[-1].pos, right[-1].yaw
        right.append(_w(8.0, pr, OPEN_C, "stay clear", rr))
        p, r = _ready("left")
        return {"right": right, "left": [Waypoint(t_right + 12.0, tuple(float(v) for v in p), OPEN_C, "stay put", r)]}
    # the left hand: hold the mask by the empty left port's boss until the right cartridge is on
    hl = entry_rot("left")
    left = _lead_in("left", port_start("left", 0.06), hl, OPEN_C, "above the left port, open")
    left += [_w(2.0, port_start("left", HOLD_TCP), OPEN_C, "down round the port's boss", hl),
             _w(1.0, port_start("left", HOLD_TCP), BOSS_PINCH, "hold the mask by the boss", hl)]
    t_left = sum(w.duration for w in left)
    left.append(_w(max(0.5, t_right - t_left + 0.5), port_start("left", HOLD_TCP), BOSS_PINCH, "keep holding", hl))
    left += [_w(0.8, port_start("left", HOLD_TCP), OPEN_C, "let go of the boss", hl),
             _w(1.5, port_start("left", 0.06), OPEN_C, "draw back along the port's axis", hl),
             _w(3.0, np.array([PAD_C["left"][0], PAD_C["left"][1], TOP + PAD_T + CART_T + 0.06]), OPEN_C,
                "over the left cartridge", grip_rotation(PICK_YAW["left"]))]
    left += _fit("left")
    t_left = sum(w.duration for w in left)
    l_end = entry_rot("left", TURN + UNWIND, mask_pose("left")[1])
    lz = CART_T - TCP_IN + EASE
    right.append(_w(max(0.5, t_left - sum(w.duration for w in right) + 0.3), right[-1].pos, CART_PINCH,
                    "keep holding the fitted cartridge (and the mask)", r_end))
    left += _let_go("left", CART_PINCH, lz, l_end)
    left.append(_w(6.0, left[-1].pos, OPEN_C, "stay clear", l_end))
    right += [_w(2.5, right[-1].pos, CART_PINCH, "wait for the left hand to draw back", r_end)]
    right += _let_go("right", CART_PINCH, rz, r_end)
    right.append(_w(5.0, right[-1].pos, OPEN_C, "stay clear", r_end))
    return {"left": left, "right": right}
