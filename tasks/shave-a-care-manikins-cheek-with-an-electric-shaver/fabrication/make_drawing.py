"""Dimensioned drawing sheet for the shaving task's fabrication folder: where the cheek insert sits, the insert's slide in
section, and the force-sensing shaver neck in section. Sections are cut from the STL files themselves.
Writes <task>/fabrication/drawing.svg and drawing.png. Run: python make_drawing.py (needs numpy, shapely, trimesh, matplotlib)."""
import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPoly

import make_parts as fab

T = fab.T
OUT = fab.OUT
INK, GREY, BLUE, RED, GREEN = "#1b1b1b", "#8a8a8a", "#1f5fa8", "#b8322a", "#2c7a3f"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def section(name, origin, normal, drop):
    """2-D outlines of an STL cut by a plane, as lists of (N, 2) arrays of the kept coordinates (indices ``drop`` removed)."""
    m = trimesh.load(OUT / name)
    out = []
    for body in m.split(only_watertight=False):
        s = body.section(plane_origin=origin, plane_normal=normal)
        if s is None:
            continue
        for ent in s.entities:
            pts = s.vertices[ent.points]
            out.append(np.delete(pts, drop, axis=1))
    return out


def draw_paths(ax, paths, color=INK, fill=None, lw=0.9, swap=False):
    for p in paths:
        q = p[:, ::-1] if swap else p
        if fill:
            ax.add_patch(MPoly(q, closed=True, fc=fill, ec=color, lw=lw))
        else:
            ax.plot(q[:, 0], q[:, 1], color=color, lw=lw)


def dim(ax, a, b, text, off=(0, 0), color=BLUE, fs=7, ha="center"):
    """A dimension line between points a and b with its text at the middle (+ off)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax.annotate("", xy=a, xytext=b, arrowprops=dict(arrowstyle="<->", color=color, lw=0.7, shrinkA=0, shrinkB=0))
    m = (a + b) / 2 + np.asarray(off)
    ax.text(m[0], m[1], text, color=color, fontsize=fs, ha=ha, va="center",
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))


def spring(ax, y0, y1, zc, d, n=9, color=GREEN, along_x=False):
    ys = np.linspace(y0, y1, 2 * n + 1)
    zs = zc + np.array([(-1) ** i * d / 2 for i in range(len(ys))])
    zs[0] = zs[-1] = zc
    if along_x:
        ax.plot(zs, ys, color=color, lw=1.0)
    else:
        ax.plot(ys, zs, color=color, lw=1.0)


fig = plt.figure(figsize=(16.54, 11.69))  # A3 landscape
fig.text(0.015, 0.975, "Shaving task: sprung cheek insert and force-sensing shaver neck (all dimensions in mm)",
         fontsize=13, weight="bold", va="top")
fig.text(0.015, 0.952, "STL files in this folder: cheek_insert_skin, cheek_insert_carrier, cheek_insert_window_tray, "
         "cheek_insert_guide_frame, shaver_neck_housing, shaver_neck_plunger, shaver_neck_spring_seat. "
         "Sections below are cut from those files; bought parts are drawn in grey/green.", fontsize=8.5, va="top")

# -- panel A: where the insert goes on the face (head frame, seen from the manikin's left) ---------------------------
ax = fig.add_axes([0.03, 0.50, 0.31, 0.42])
cells = T.CELLS * 1000.0
soft = T.SOFT
ax.scatter(cells[~soft, 0], cells[~soft, 2], s=3, c=GREY, label=f"firm beard ({(~soft).sum()} squares)")
ax.scatter(cells[soft, 0], cells[soft, 2], s=3, c="#c98a3a", label=f"soft cheek ({soft.sum()} squares)")
F0, F1, U0, U1 = fab.TRAY_F + fab.TRAY_U
ax.add_patch(Rectangle((F0, U0), F1 - F0, U1 - U0, fill=False, ec=INK, lw=1.0))
ax.add_patch(Rectangle((fab.POCKET_F[0], fab.POCKET_U[0]), np.ptp(fab.POCKET_F), np.ptp(fab.POCKET_U), fill=False,
                       ec=INK, lw=0.8, ls="--"))
ax.add_patch(Rectangle((fab.F0, fab.U0), fab.F1 - fab.F0, fab.U1 - fab.U0, fc="#f2c9b3", ec=RED, lw=1.0))
ax.add_patch(Rectangle((fab.SLOT_F[0], fab.SLOT_U[0]), np.ptp(fab.SLOT_F), np.ptp(fab.SLOT_U), fc="white", ec=INK, lw=0.7))
ax.plot([0], [0], "k+", ms=10)
ax.text(1.5, -2.5, "ear canals' midpoint\n(head-frame origin)", fontsize=7, va="top")
ax.axvline(T.SOFT_F * 1000, color="#c98a3a", lw=0.6, ls=":")
ax.axhline(T.SOFT_Z * 1000, color="#c98a3a", lw=0.6, ls=":")
dim(ax, (fab.F0, fab.U0 - 1.5), (fab.F1, fab.U0 - 1.5), "insert 30", off=(0, -2.2))
dim(ax, (fab.F1 + 2.5, fab.U0), (fab.F1 + 2.5, fab.U1), "22", off=(2.5, 0))
dim(ax, (0, fab.U0), (fab.F0, fab.U0), "25", off=(0, 1.8))
dim(ax, (fab.F0 - 3, 0), (fab.F0 - 3, fab.U0), "10", off=(-2.5, 0))
dim(ax, (F0, U1 + 2.5), (F1, U1 + 2.5), "window and tray 40", off=(0, 2.0))
dim(ax, (F1 + 8, U0), (F1 + 8, U1), "48", off=(2.5, 0))
dim(ax, (0, T.SOFT_Z * 1000 - 3), (T.SOFT_F * 1000, T.SOFT_Z * 1000 - 3), f"{T.SOFT_F * 1000:.0f}", off=(0, -2))
ax.text(fab.FC + 24, fab.U1 - 2, "pocket 32 x 32\n(dashed): the\ninsert slides\nup 0-8 in it", ha="left", fontsize=6.5)
ax.text(fab.FC + 4, fab.SLOT_U[1] + 1.0, "slot 4.4 x 12\n(post stops)", fontsize=6.5)
ax.set_xlabel("forward from the ear canals, along the face (F, mm)")
ax.set_ylabel("up from the ear canals' level (U, mm)")
ax.set_aspect("equal")
ax.set_xlim(-12, 112)
ax.set_ylim(-95, 62)
ax.grid(lw=0.3, color="#dddddd")
ax.legend(loc="lower left", fontsize=7, markerscale=3, frameon=False)
ax.set_title("A. Where the insert goes (left side of the face)", loc="left", fontsize=9.5,
             weight="bold")

# -- panel B: section of the insert's slide on its middle plane (insert frame x = 0) ----------------------------------
ax = fig.add_axes([0.37, 0.50, 0.30, 0.42])
cut = dict(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), drop=0)  # keep (y, z)
for name, col, fill in (("cheek_insert_window_tray.stl", INK, "#e4e4e4"), ("cheek_insert_skin.stl", RED, "#f2c9b3"),
                        ("cheek_insert_carrier.stl", BLUE, "#cfe0f3"), ("cheek_insert_guide_frame.stl", INK, "#e4e4e4")):
    draw_paths(ax, section(name, cut["origin"], cut["normal"], cut["drop"]), color=col, fill=fill, swap=True)
# bought parts: the rail, the block, the spring (drawn here, not in the STLs)
zr = fab.Z_RAIL
ax.add_patch(Rectangle((zr, fab.RAIL_Y0), fab.RAIL_H, 40.0, fc="#bdbdbd", ec=INK, lw=0.7))
ax.add_patch(Rectangle((zr + 1.0, -fab.BLOCK_L / 2), fab.BLOCK_H - 1.0, fab.BLOCK_L, fc="#9e9e9e", ec=INK, lw=0.7))
spring(ax, fab.SPRING_Y0, fab.SPRING_Y0 + fab.SPRING_FREE, fab.SPRING_Z, 9.14, along_x=True)
ax.text(zr - 0.5, fab.RAIL_Y0 - 1.5, "HIWIN MGNR5R rail, 40 long", fontsize=6.5, ha="left", va="top", color=INK)
ax.text(zr - 9, 0, "MGN5C\nblock", fontsize=6.5, ha="right", va="center")
ax.annotate("", xy=(zr + 1, 0), xytext=(zr - 8.6, 0), arrowprops=dict(arrowstyle="-", lw=0.5))
ax.text(fab.SPRING_Z + 6.5, fab.SPRING_Y0 + 9, "Lee Spring\nLP 018E 03 S316\n0.12 N/mm, free 19.05", fontsize=6.5,
        color=GREEN, va="center")
ax.text(3.0, -15.5, "silicone skin 2\non carrier 1", fontsize=6.5, color=RED)
ax.text(-5.5, -14, "tray floor 3\n(0.3 behind\nthe carrier)", fontsize=6.5, ha="right")
ax.text(fab.Z_TAB1 - 1, -10.5, "tab: 2 x M2\nto the block", fontsize=6.5, ha="right", color=BLUE)
dim(ax, (fab.Z_TAB1 + 4, fab.SLOT_U[0] - fab.UC), (fab.Z_TAB1 + 4, fab.SLOT_U[1] - fab.UC),
    "post 4x4 in\nslot 12: travel 0-8", off=(9.5, 0), fs=6.5)
dim(ax, (fab.SPRING_Z - 6.5, fab.SPRING_Y0), (fab.SPRING_Z - 6.5, fab.cup_y0 + 8.0), "19.05", off=(-3.2, 0), fs=6.5)
dim(ax, (zr - fab.PLATE_T, 43), (0, 43), f"{-zr + fab.PLATE_T:.1f} from the skin's middle to the frame's back",
    off=(0, 2.2), fs=6.5)
ax.text(zr - fab.PLATE_T - 0.8, -18.5, "M3 standoffs\n(tray) to frame", fontsize=6.5, ha="right", va="center")
ax.set_xlabel("out of the cheek, z (mm; 0 = the skin at the insert's middle)")
ax.set_ylabel("up the slide, toward the temple, y (mm; 0 = insert middle at rest)")
ax.set_aspect("equal")
ax.set_xlim(-46, 20)
ax.set_ylim(-28, 50)
ax.grid(lw=0.3, color="#dddddd")
ax.set_title("B. Cheek insert slide, section on its middle (shown at rest)", loc="left", fontsize=9.5, weight="bold")

# -- panel C: the force-sensing neck in section (plane through the shaver's axis) ------------------------------------
ax = fig.add_axes([0.70, 0.50, 0.28, 0.42])
cut = dict(origin=(0.0, 0.0, 10.0), normal=(0.0, 1.0, 0.0), drop=1)  # keep (x, z)
for name, col, fill in (("shaver_neck_housing.stl", INK, "#e4e4e4"), ("shaver_neck_plunger.stl", BLUE, "#cfe0f3"),
                        ("shaver_neck_spring_seat.stl", BLUE, "#cfe0f3")):
    draw_paths(ax, section(name, cut["origin"], cut["normal"], cut["drop"]), color=col, fill=fill)
ax.add_patch(Rectangle((-6.0, fab.POCKET_Z0), 12.0, fab.BOARD[1], fc="#7fae7f", ec=INK, lw=0.6))
ax.add_patch(Rectangle((-2.5, fab.POCKET_Z0 + fab.BOARD[1]), 5.0, fab.SEAT_TOP - fab.SEAT_T_NOMINAL - fab.SEAT_NUB[1]
                       - fab.POCKET_Z0 - fab.BOARD[1], fc="#4d4d4d", ec=INK, lw=0.6))
spring(ax, fab.SEAT_TOP, fab.FLANGE_Z0, 0.0, 8.0, n=6, along_x=True)
ax.add_patch(Rectangle((-29.0, -10.0), 58.0, 10.0, fc="none", ec=GREY, lw=0.8, ls="--"))
ax.text(0, -6, "shaver body's top end (stock head holder removed)", fontsize=6.5, ha="center", color=GREY)
hu_z0 = fab.CAP_Z1 + 10.0 + 1.0
ax.add_patch(Rectangle((-28.0, hu_z0), 56.0, 10.0, fc="none", ec=GREY, lw=0.8, ls="--"))
ax.text(0, hu_z0 + 5, "stock head unit on its own +-20 deg pivot,\nscrewed to the plunger (M3)", fontsize=6.5,
        ha="center", va="center", color=GREY)
ax.text(8.5, 1.4, "force sensor on its 12x12 board\n(Honeywell FMAMSDXX005WC2C3)", fontsize=6.5, va="center")
ax.text(-9, (fab.SEAT_TOP + fab.FLANGE_Z0) / 2, "spring\n1 N/mm\nfree 12", fontsize=6.5, ha="right", va="center",
        color=GREEN)
dim(ax, (14.0, fab.STOP_Z), (14.0, fab.FLANGE_Z0), "travel 6", off=(5.0, 0), fs=6.5)
dim(ax, (-14.0, 0), (-14.0, fab.CAP_Z1 + 10.0), f"{fab.CAP_Z1 + 10.0:.1f}", off=(-3.5, 0), fs=6.5)
dim(ax, (-fab.SLEEVE_OUT[0] / 2, -12.5), (fab.SLEEVE_OUT[0] / 2, -12.5), "sleeve 62.6 x 49.6 outside, "
    "58.6 x 45.6 inside, 10 deep", off=(0, -2.3), fs=6.5)
ax.text(fab.FLANGE_D / 2 + 1, fab.FLANGE_Z0 + 1, "flange 16.4", fontsize=6.5, color=BLUE)
ax.text(fab.SHAFT_D / 2 + 1, fab.CAP_Z1 + 6, "shaft 12, M3 bore", fontsize=6.5, color=BLUE)
ax.text(-fab.SEAT_D / 2 - 1, fab.SEAT_TOP - 0.6, "seat", fontsize=6.5, color=BLUE, ha="right")
ax.set_xlabel("across the shaver (mm)")
ax.set_ylabel("along the shaver's axis from its body's top end (mm)")
ax.set_aspect("equal")
ax.set_xlim(-36, 36)
ax.set_ylim(-18, 50)
ax.grid(lw=0.3, color="#dddddd")
ax.set_title("C. Force-sensing neck, section through the axis (at rest)", loc="left", fontsize=9.5, weight="bold")

# -- parts list and checks ---------------------------------------------------------------------------------------------
parts = [
    ("1", "Rail", "HIWIN MGNR5R, cut to 40 mm (M2 holes at 15 mm pitch)", "bought"),
    ("1", "Carriage block", "HIWIN MGN5C (12 wide, 16 long, 6 high on the rail; 2 x M2, 8 apart)", "bought"),
    ("1", "Insert return spring", "Lee Spring LP 018E 03 S316: 0.12 N/mm, free 19.05, OD 9.14", "bought"),
    ("1", "Skin", "Smooth-On Ecoflex 00-30, 2 mm sheet, bonded with Sil-Poxy", "cast"),
    ("1 each", "Carrier, window tray, guide frame", "STL, PETG, 0.12 mm layers, 100 % infill", "printed"),
    ("1", "Force sensor", "Honeywell FMAMSDXX005WC2C3: 5 N, I2C, 3.3 V, +-2 % FSS", "bought"),
    ("1", "Neck spring", "stainless compression spring, OD 8, wire 0.7, free 12, about 1.0 N/mm", "bought"),
    ("1", "Logger", "Seeed Studio XIAO nRF52840 Sense (I2C, 6-axis IMU, Bluetooth) + 3.7 V LiPo", "bought"),
    ("1 each", "Housing, plunger, seat", "STL, PETG, 0.12 mm layers, 100 % infill", "printed"),
    ("1", "Shaver", "three-head rotary shaver, about 16 x 6 x 4.5 cm, 170 g", "bought"),
]
ax = fig.add_axes([0.03, 0.04, 0.56, 0.40])
ax.axis("off")
ax.set_title("Parts (full list, sources and assembly in README.md here)", loc="left", fontsize=9.5, weight="bold")
tab = ax.table(cellText=[list(p) for p in parts], colLabels=["Qty", "Part", "What", "Source"], loc="upper left",
               colWidths=[0.06, 0.22, 0.60, 0.10], cellLoc="left")
tab.auto_set_font_size(False)
tab.set_fontsize(7.5)
tab.scale(1, 1.45)

ax = fig.add_axes([0.61, 0.04, 0.37, 0.40])
ax.axis("off")
ax.set_title("Calibration (before each session; details in README.md here)", loc="left", fontsize=9.5, weight="bold")
checks = [
    "Insert spring: push the insert up with a 0-5 N force gauge;",
    "  0.36 N at 3 mm and 0.96 N at 8 mm (+-15 %), read on the mm scale",
    "  beside the insert; it must return to 0 when let go.",
    "Force sensor: shaver upright, heads up, in a vise; tare; add 20, 50,",
    "  100, 200 and 400 g on the head unit; straight-line fit; every",
    "  reading within 0.1 N of m x 9.81.",
    "Neck spring: with the sensor's reading, 1 N must press the head",
    "  unit in 1.0 +- 0.15 mm (feeler gauge under the flange stop).",
    "Head's neck: 5 N pushed level into the beard's middle (11.6 cm",
    "  above the neck's hinge) moves it 4.0 +- 0.5 mm along the push",
    "  (0.8 mm per newton; ruler on a stand).",
    "Gel: one 0.5 N pass of the shaver over a gel strip on a vinyl test",
    "  tile must leave a dark 20 mm track under the UV torch.",
]
ax.text(0.0, 0.93, "\n".join(checks), fontsize=8.2, va="top", family="DejaVu Sans Mono", transform=ax.transAxes)

fig.savefig(OUT / "drawing.svg")
fig.savefig(OUT / "drawing.png", dpi=110)
print("written", OUT / "drawing.svg", OUT / "drawing.png")
