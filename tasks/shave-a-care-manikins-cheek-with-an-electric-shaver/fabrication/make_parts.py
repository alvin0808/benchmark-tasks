"""Fabrication files for the shaving task: STL parts (mm) for the sprung cheek insert and the force-sensing shaver neck,
built from the task file's own head model (the insert's curved faces) and from extruded profiles (the printed frames).
Writes into <task>/fabrication/. Run: python make_parts.py (needs numpy and shapely >= 2.1)."""
import importlib.util
import pathlib
import struct

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.geometry.polygon import orient

TASK_DIR = pathlib.Path(__file__).resolve().parents[1]  # the task folder (this file is in its fabrication/)
OUT = TASK_DIR / "fabrication"
spec = importlib.util.spec_from_file_location("t", TASK_DIR / "shave_a_care_manikins_cheek_with_an_electric_shaver.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


# -- meshes -------------------------------------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.v, self.f = [], []

    def add(self, verts, faces):
        base = sum(len(v) for v in self.v)
        self.v.append(np.asarray(verts, float))
        self.f.append(np.asarray(faces, int) + base)
        return self

    def arrays(self):
        return np.concatenate(self.v), np.concatenate(self.f)

    def write(self, path):
        v, f = self.arrays()
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        with open(path, "wb") as fh:
            fh.write(b"shaving task fabrication part (mm)".ljust(80, b" "))
            fh.write(struct.pack("<I", len(f)))
            for k in range(len(f)):
                fh.write(struct.pack("<12fH", *n[k], *tri[k, 0], *tri[k, 1], *tri[k, 2], 0))
        return len(f)


def extrude(poly, z0, z1, axes=(0, 1, 2)):
    """A closed prism: the shapely polygon (with holes) in the plane of axes[0], axes[1], from z0 to z1 along axes[2]."""
    poly = orient(poly, sign=1.0)
    tris = shapely.constrained_delaunay_triangles(poly)
    verts, index, faces = [], {}, []

    def vid(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in index:
            index[key] = len(verts)
            p = [0.0, 0.0, 0.0]
            p[axes[0]], p[axes[1]], p[axes[2]] = x, y, z
            verts.append(p)
        return index[key]

    for t in tris.geoms:
        (x0, y0), (x1, y1), (x2, y2) = list(t.exterior.coords)[:3]
        if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        faces.append((vid(x0, y0, z1), vid(x1, y1, z1), vid(x2, y2, z1)))  # top, normal +
        faces.append((vid(x0, y0, z0), vid(x2, y2, z0), vid(x1, y1, z0)))  # bottom, normal -
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)[:-1]
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            a0, b0, a1, b1 = vid(ax, ay, z0), vid(bx, by, z0), vid(ax, ay, z1), vid(bx, by, z1)
            faces += [(a0, b0, b1), (a0, b1, a1)]
    v = np.array(verts)
    f = np.array(faces)
    if axes != (0, 1, 2) and np.linalg.det(np.eye(3)[list(axes)]) < 0:  # a mirrored axis order flips every face
        f = f[:, ::-1]
    return v, f


def rrect(w, h, r, cx=0.0, cy=0.0):
    return box(cx - w / 2 + r, cy - h / 2 + r, cx + w / 2 - r, cy + h / 2 - r).buffer(r, quad_segs=8)


def circle(d, cx=0.0, cy=0.0):
    return Point(cx, cy).buffer(d / 2, quad_segs=24)


def slab(outer, inner):
    """A closed solid between two (rows, cols, 3) grids of points (outer and inner surfaces)."""
    r, c = outer.shape[:2]
    idx = np.arange(r * c).reshape(r, c)
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    top = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, cc], -1).reshape(-1, 3)])
    bot = top[:, ::-1] + r * c
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    p, q = ring, np.roll(ring, -1)
    side = np.concatenate([np.stack([p, p + r * c, q + r * c], -1), np.stack([p, q + r * c, q], -1)])
    v = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    f = np.concatenate([top, bot, side])
    # make the faces point outward: the outer grid's normal must point along the outer surface's normal
    n0 = np.cross(v[top[0][1]] - v[top[0][0]], v[top[0][2]] - v[top[0][0]])
    if n0 @ (outer[0, 0] - inner[0, 0]) < 0:
        f = f[:, ::-1]
    return v, f


# -- the cheek insert, in the insert frame: x along the face (head forward), y up, z out of the cheek; mm, origin at
#    the insert's middle on the skin --------------------------------------------------------------------------------
F0, F1, U0, U1 = 25.0, 55.0, 10.0, 32.0  # the insert's outline (head frame, mm): T.PATCH_F, T.PATCH_Z
FC, UC = 40.0, 21.0  # its middle
SKIN_T, CARRIER_T = 2.0, 1.0
SHELL_T = 3.0  # the manikin's face shell under the recess
POST = 4.0  # the carrier's post through the slot
TRAVEL = 8.0


def face_grid(f0, f1, u0, u1, step=1.0):
    fs = np.arange(f0, f1 + 1e-9, step) / 1000.0
    us = np.arange(u0, u1 + 1e-9, step) / 1000.0
    pts = np.array([[[f, 0.0, u] for f in fs] for u in us])
    flat = pts.reshape(-1, 3)
    surf = np.array([T.ray_surface(T.head_sdf, s, np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0] for s in flat])
    nrm = T._normals(T.head_sdf, surf)
    return surf.reshape(pts.shape), nrm.reshape(pts.shape)


S, N = face_grid(F0, F1, U0, U1)
P0 = T.ray_surface(T.head_sdf, np.array([FC / 1000, 0.0, UC / 1000]), np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0]


def to_insert(p):
    """Head frame (m) -> insert frame (mm): x = forward, y = up, z = out of the cheek (head +y)."""
    d = (np.asarray(p) - P0) * 1000.0
    return np.stack([d[..., 0], d[..., 2], d[..., 1]], -1)


def offset(k_mm):
    return to_insert(S - N * (k_mm / 1000.0))


skin = Mesh().add(*slab(offset(0.0), offset(SKIN_T)))
carrier = Mesh().add(*slab(offset(SKIN_T), offset(SKIN_T + CARRIER_T)))
z_back = float(offset(SKIN_T + CARRIER_T)[..., 2].min())  # the carrier's back, lowest point
# the post through the slot in the face shell, and the tab that screws to the MGN5C block (2 x M2, 8 mm across)
Z_TAB1 = round(z_back - SHELL_T - 6.0, 1)  # the tab's front face, well behind the tray's floor (the slide sits deep)
TAB_T = 2.0
_o3 = offset(SKIN_T + CARRIER_T)
_foot = (np.abs(_o3[..., 0]) <= POST / 2 + 0.5) & (np.abs(_o3[..., 1]) <= POST / 2 + 0.5)
Z_POST1 = float(_o3[..., 2][_foot].mean()) + 0.5  # the post ends inside the carrier plate, behind the insert's middle
carrier.add(*extrude(box(-POST / 2, -POST / 2, POST / 2, POST / 2), Z_TAB1 - 0.2, Z_POST1))
tab = box(-6.0, -8.0, 6.0, 8.0).difference(circle(2.2, -4.0, 0.0)).difference(circle(2.2, 4.0, 0.0))
carrier.add(*extrude(tab, Z_TAB1 - TAB_T, Z_TAB1))

# the guide frame inside the head: a plate for the MGNR5R rail (40 mm, M2 holes at 15 mm pitch), and a cup that
# holds the return spring (Lee Spring LP 018E 03 S316, 9.14 mm OD, 19.05 mm free length) above the block
BLOCK_H, BLOCK_L, BLOCK_W = 6.0, 16.0, 12.0  # MGN5C: height over the rail's base, length along the rail, width
Z_RAIL = Z_TAB1 - TAB_T - BLOCK_H  # the rail's base (the frame plate's front face)
PLATE_T = 3.0
RAIL_Y0 = -15.0  # the rail's bottom end (insert frame): the block spans y -8..8 at rest, -8..16 over its travel
plate = box(-8.0, -22.0, 8.0, 40.0)
for y in (RAIL_Y0 + 5.0, RAIL_Y0 + 20.0, RAIL_Y0 + 35.0):
    plate = plate.difference(circle(2.2, 0.0, y))
for x in (-5.0, 5.0):
    plate = plate.difference(circle(3.4, x, -18.5))
frame = Mesh().add(*extrude(plate, Z_RAIL - PLATE_T, Z_RAIL))
RAIL_H = 3.6  # MGNR5R rail height
SPRING_Z = Z_RAIL + RAIL_H + 9.14 / 2 + 0.33  # the spring's axis: its coil clears the rail's top
SPRING_Y0 = BLOCK_L / 2  # the block's top end at rest
SPRING_FREE = 19.05
cup_y0 = SPRING_Y0 + SPRING_FREE - 8.0  # the cup's mouth: the spring's top 8 mm inside it
cup_profile = (box(-7.0, Z_RAIL - PLATE_T + 0.2, 7.0, SPRING_Z + 6.0).difference(circle(9.7, 0.0, SPRING_Z))
               .difference(box(-2.8, Z_RAIL, 2.8, Z_RAIL + RAIL_H + 0.3)))  # a channel for the rail's top end
# extrude along y (the slide's axis): axes map (profile x -> x, profile y -> z, extrusion -> y)
frame.add(*extrude(cup_profile, cup_y0, cup_y0 + 8.0, axes=(0, 2, 1)))
frame.add(*extrude(box(-7.0, Z_RAIL - PLATE_T + 0.2, 7.0, SPRING_Z + 6.0).difference(box(-2.8, Z_RAIL, 2.8, Z_RAIL + RAIL_H + 0.3)),
                   cup_y0 + 7.8, cup_y0 + 10.0, axes=(0, 2, 1)))

# the carrier's end wall over the tab's top end: with the block's end face it takes the whole spring end
carrier.add(*extrude(box(-5.5, Z_TAB1 - 0.2, 5.5, SPRING_Z + 5.0), BLOCK_L / 2 - 2.0, BLOCK_L / 2, axes=(0, 2, 1)))

# the window tray glued into a window cut in the manikin's face: its rim continues the face's surface, its 3 mm pocket
# holds the insert flush (0.3 mm clearance behind the carrier), its 3 mm floor carries the slot that guides the post and
# stops it at 0 and 8 mm, and two standoffs take the guide frame's M3 screws.
TRAY_F, TRAY_U = (20.0, 60.0), (-3.0, 45.0)  # outline (head frame, mm)
POCKET_F, POCKET_U = (24.0, 56.0), (9.0, 41.0)  # the insert at rest spans U 10..32, pulled 8 mm U 18..40
SLOT_F, SLOT_U = (FC - POST / 2 - 0.2, FC + POST / 2 + 0.2), (UC - POST / 2, UC + POST / 2 + TRAVEL)
GAP = 0.3


def patch(f0, f1, u0, u1, k0, k1, step=1.0):
    """A curved slab of the head's skin region F f0..f1, U u0..u1 (head frame, mm), from k0 to k1 mm under the skin."""
    fs = np.linspace(f0, f1, int(np.ceil((f1 - f0) / step)) + 1) / 1000.0
    us = np.linspace(u0, u1, int(np.ceil((u1 - u0) / step)) + 1) / 1000.0
    pts = np.array([[[f, 0.0, u] for f in fs] for u in us]).reshape(-1, 3)
    surf = np.array([T.ray_surface(T.head_sdf, s, np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0] for s in pts])
    nrm = T._normals(T.head_sdf, surf)
    shape = (len(us), len(fs), 3)
    return slab(to_insert((surf - nrm * (k0 / 1000.0)).reshape(shape)),
                to_insert((surf - nrm * (k1 / 1000.0)).reshape(shape)))


FLOOR_K = (SKIN_T + CARRIER_T + GAP, SKIN_T + CARRIER_T + GAP + SHELL_T)
tray = Mesh()
o = 0.2  # overlap between neighbouring pieces, so the shells join without shared faces
tray.add(*patch(TRAY_F[0], SLOT_F[0], *TRAY_U, *FLOOR_K))
tray.add(*patch(SLOT_F[1], TRAY_F[1], *TRAY_U, *FLOOR_K))
tray.add(*patch(SLOT_F[0] - o, SLOT_F[1] + o, TRAY_U[0], SLOT_U[0], *FLOOR_K))
tray.add(*patch(SLOT_F[0] - o, SLOT_F[1] + o, SLOT_U[1], TRAY_U[1], *FLOOR_K))
RIM_K = (0.0, FLOOR_K[0] + o)
tray.add(*patch(TRAY_F[0], POCKET_F[0] + o, *TRAY_U, *RIM_K))
tray.add(*patch(POCKET_F[1] - o, TRAY_F[1], *TRAY_U, *RIM_K))
tray.add(*patch(POCKET_F[0], POCKET_F[1], TRAY_U[0], POCKET_U[0] + o, *RIM_K))
tray.add(*patch(POCKET_F[0], POCKET_F[1], POCKET_U[1] - o, TRAY_U[1], *RIM_K))
# standoffs for the guide frame's two M3 screws (frame holes at x -5, 5, y -18.5 in the insert frame)
_fl = to_insert(np.array([T.ray_surface(T.head_sdf, np.array([(FC + x) / 1000, 0.0, (UC - 18.5) / 1000]),
                                        np.array([[0.0, 1.0, 0.0]]), 0.12, samples=60)[0] for x in (-5.0, 5.0)]))
for (x, y, zs) in _fl:
    tray.add(*extrude(circle(7.0, x, y).difference(circle(2.5, x, y)), Z_RAIL, zs - FLOOR_K[1] + 0.5))

# -- the force-sensing neck of the shaver: z along the shaver's axis from the top face of its body (mm) -------------
BODY_W, BODY_T, BODY_R = 58.0, 45.0, 6.0
SLEEVE_IN = (BODY_W + 0.6, BODY_T + 0.6, BODY_R + 0.3)
SLEEVE_OUT = (BODY_W + 4.6, BODY_T + 4.6, BODY_R + 2.3)
BOARD = (12.0, 1.6)  # the force sensor's carrier board (square side, thickness): Honeywell FMAMSDXX005WC2C3 on it
POCKET_Z0 = 0.4  # the board's pocket in the floor: its bottom; the board's top is then 2.0 mm up
SEAT_TOP = 7.3  # the spring rests on the seat's top
SEAT_D, SEAT_NUB, SEAT_BOSS = 12.0, (2.0, 0.5), (6.2, 1.5)  # the seat: disc, the nub on the sensor's sphere, spring boss
SEAT_T_NOMINAL = 1.8  # = 4.8 mm minus the sensor's height over its board (sphere included); the file's value
SPRING2 = dict(od=8.0, free=12.0, solid=4.9, rate=1.0)  # the neck's compliance spring
FLANGE_D, FLANGE_T, SHAFT_D = 16.4, 2.0, 12.0
STOP_Z = SEAT_TOP + SPRING2["free"] - 6.0  # the flange cannot go lower: 6 mm of travel
FLANGE_Z0 = SEAT_TOP + SPRING2["free"]  # the flange's bottom at rest (the spring free, no preload)
CAP_Z0 = FLANGE_Z0 + FLANGE_T
CAP_Z1 = CAP_Z0 + 4.0
neck = Mesh()
neck.add(*extrude(rrect(*SLEEVE_OUT[:2], SLEEVE_OUT[2]).difference(rrect(*SLEEVE_IN[:2], SLEEVE_IN[2])), -10.0, 0.0))
outline = rrect(*SLEEVE_OUT[:2], SLEEVE_OUT[2])
neck.add(*extrude(outline.difference(box(-3.0, -1.5, 3.0, 1.5)), -0.2, POCKET_Z0 + 0.2))  # a hole for the 4 wires
_bp = BOARD[0] / 2 + 0.2
neck.add(*extrude(outline.difference(box(-_bp, -_bp, _bp, _bp)), POCKET_Z0, 3.0))  # the board's pocket
neck.add(*extrude(circle(30.0).difference(circle(12.4)), 2.8, STOP_Z))  # the spring's bore, ending in the stop
neck.add(*extrude(circle(30.0).difference(circle(FLANGE_D + 0.6)), STOP_Z - 0.2, CAP_Z0))  # the flange's travel
neck.add(*extrude(circle(30.0).difference(circle(SHAFT_D + 0.4)), CAP_Z0 - 0.2, CAP_Z1))  # the cap guiding the shaft
plunger = Mesh()
plunger.add(*extrude(circle(FLANGE_D), FLANGE_Z0, FLANGE_Z0 + FLANGE_T))
plunger.add(*extrude(circle(SHAFT_D).difference(circle(2.6)), FLANGE_Z0 + FLANGE_T - 0.2, CAP_Z1 + 10.0))  # M3 bore
seat = Mesh()
seat.add(*extrude(circle(SEAT_D), SEAT_TOP - SEAT_T_NOMINAL, SEAT_TOP))
seat.add(*extrude(circle(SEAT_NUB[0]), SEAT_TOP - SEAT_T_NOMINAL - SEAT_NUB[1], SEAT_TOP - SEAT_T_NOMINAL + 0.2))
seat.add(*extrude(circle(SEAT_BOSS[0]), SEAT_TOP - 0.2, SEAT_TOP + SEAT_BOSS[1]))

DIMS = dict(RAIL_H=RAIL_H, Z_TAB1=Z_TAB1, TAB_T=TAB_T, Z_RAIL=Z_RAIL, PLATE_T=PLATE_T, RAIL_Y0=RAIL_Y0, SPRING_Z=SPRING_Z,
            SPRING_Y0=SPRING_Y0, cup_y0=cup_y0, z_back=z_back, STOP_Z=STOP_Z, FLANGE_Z0=FLANGE_Z0, CAP_Z0=CAP_Z0,
            CAP_Z1=CAP_Z1, Z_POST1=Z_POST1, standoffs=np.round(_fl, 2).tolist(), P0_mm=(P0 * 1000).round(1).tolist())

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, m in (("cheek_insert_skin.stl", skin), ("cheek_insert_carrier.stl", carrier),
                    ("cheek_insert_guide_frame.stl", frame), ("cheek_insert_window_tray.stl", tray),
                    ("shaver_neck_housing.stl", neck),
                    ("shaver_neck_plunger.stl", plunger), ("shaver_neck_spring_seat.stl", seat)):
        n = m.write(OUT / name)
        v, f = m.arrays()
        print(f"{name}: {n} triangles, extent {np.round(v.max(0) - v.min(0), 1)} mm")
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in DIMS.items()})
