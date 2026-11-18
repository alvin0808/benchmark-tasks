"""Fabrication files for the palm-cone task: the training hand's printed parts as STL (millimetres, closed surfaces,
the same shapes the simulation uses) and a drawing sheet (top and side views with the main dimensions, the joint
and spring layout, the springs' torque lines and the cone). Writes into <task>/fabrication/.
Run: python make_parts.py (needs numpy and matplotlib)."""
import math
import pathlib
import struct
import sys

import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parents[1]
OUT = TASK_DIR / "fabrication"
sys.path.insert(0, str(TASK_DIR))
import hand_mesh as hm  # noqa: E402

# The springs and joint ranges (the task file's values; copied so this script runs without the simulator).
K_KNUCKLE, K_PLAY, K_PIP, K_THUMB = 0.080, 0.15, 0.020, 0.030  # N m/rad
CONE_L, CONE_DN, CONE_DW = 130.0, 25.0, 42.0  # mm


def write_stl(path, pts_mm, faces, title):
    pts = np.asarray(pts_mm, float)
    tri = pts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as fh:
        fh.write(title.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(faces)))
        for k in range(len(faces)):
            fh.write(struct.pack("<12fH", *n[k], *tri[k, 0], *tri[k, 1], *tri[k, 2], 0))
    return len(faces)


def capsule_mesh(a, b, r, n=32, m=8):
    """A closed capsule (cm in, mm out) from a to b, radius r."""
    a, b = np.asarray(a, float) * 10, np.asarray(b, float) * 10
    r = r * 10
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    rings = []
    for k in range(m, 0, -1):  # the hemisphere at a
        th = math.pi / 2 * k / m
        rings.append((a - d * r * math.sin(th), r * math.cos(th)))
    rings.append((a, r))
    rings.append((b, r))
    for k in range(1, m + 1):
        th = math.pi / 2 * k / m
        rings.append((b + d * r * math.sin(th), r * math.cos(th)))
    pts, faces = [], []
    for c, rr in rings:
        for j in range(n):
            ph = 2 * math.pi * j / n
            pts.append(c + rr * (math.cos(ph) * u + math.sin(ph) * v))
    for i in range(len(rings) - 1):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            s, t = p + n, q + n
            faces += [(p, q, s), (q, t, s)]
    pts = np.asarray(pts)
    faces = np.asarray(faces)
    return pts, hm._orient(pts, faces)


def merged(parts):
    pts, faces, off = [], [], 0
    for p, f in parts:
        pts.append(p)
        faces.append(f + off)
        off += len(p)
    return np.concatenate(pts), np.concatenate(faces)


def main():
    base = []
    for mesh in (hm.palm_mesh(), hm.forearm_mesh(), hm.thenar_mesh(), hm.hypothenar_mesh()):
        base.append((mesh[0] * 1000.0, mesh[1]))
    p, f = merged(base)
    print("hand_base.stl", write_stl(OUT / "hand_base.stl", p, f, "palm cone task: hand base shell, palm up (mm)"))
    for name in hm.ORDER:
        fd = hm.FINGERS[name]
        k, pp, q, t = hm.finger_points(name)
        prox = capsule_mesh(k, pp, fd["r"][0])
        end = merged([capsule_mesh(pp, q, fd["r"][1]), capsule_mesh(q, t, fd["r"][2])])
        write_stl(OUT / f"finger_{name}_first.stl", *prox, f"palm cone task: {name} first segment (mm)")
        write_stl(OUT / f"finger_{name}_end.stl", *end, f"palm cone task: {name} middle+end segments (mm)")
        print(f"finger_{name}_first.stl, finger_{name}_end.stl")
    b, j, t = hm.thumb_points()
    th = merged([capsule_mesh(b, j, hm.THUMB_R[0]), capsule_mesh(j, t, hm.THUMB_R[1])])
    write_stl(OUT / "thumb.stl", *th, "palm cone task: thumb (mm)")
    print("thumb.stl")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    # A: top view, palm up
    ax = fig.add_axes([0.03, 0.42, 0.45, 0.54])
    for tab, col in ((hm.PALM_TABLE, "#e9cdb6"), (hm.FOREARM_TABLE, "#e9cdb6")):
        ax.fill_between(tab[:, 0] * 10, -tab[:, 1] * 10, tab[:, 1] * 10, color=col, ec="#8a6a55", lw=1.2)
    for c, r in (hm.THENAR, hm.HYPOTHENAR):
        ax.add_patch(plt.matplotlib.patches.Ellipse((c[0] * 10, c[1] * 10), 2 * r[0] * 10, 2 * r[1] * 10,
                                                    fc="#dfbca2", ec="#8a6a55"))
    for name in hm.ORDER:
        fd = hm.FINGERS[name]
        k, pp, q, t = hm.finger_points(name, mcp=0.0, pip=0.0)
        L = sum(fd["L"])
        ax.add_patch(plt.Rectangle((-L * 10, (fd["y"] - fd["r"][0]) * 10), L * 10, 2 * fd["r"][0] * 10, fc="#f2dccb",
                                   ec="#8a6a55", ls="--"))
        ax.text(-L * 10 - 4, fd["y"] * 10, f"{name}\n{fd['L'][0] * 10:.0f}/{fd['L'][1] * 10:.0f}/{fd['L'][2] * 10:.0f}",
                fontsize=7, ha="right", va="center")
    b, j, t = hm.thumb_points()
    ax.plot([b[0] * 10, j[0] * 10, t[0] * 10], [b[1] * 10, j[1] * 10, t[1] * 10], color="#8a6a55", lw=6, alpha=.5)
    ax.plot([0, 0], [-45, 45], "k-", lw=2)
    ax.text(-105, 52, "knuckle axle (3 mm steel) at x = 0, 14 mm above the back of the hand", fontsize=8)
    ax.axvline(91, color="#a33", lw=1, ls=":")
    ax.text(93, -48, "wrist crease", fontsize=8, color="#a33")
    ax.annotate("", (0, -55), (-90, -55), arrowprops=dict(arrowstyle="<->"))
    ax.text(-75, -62, "middle finger 90 mm (knuckle to tip)", fontsize=8)
    ax.annotate("", (0, -70), (91, -70), arrowprops=dict(arrowstyle="<->"))
    ax.text(10, -78, "knuckles to wrist crease 91 mm (palm length 109 mm)", fontsize=8)
    ax.text(150, -60, "forearm cut 200 mm above the wrist; hand breadth 78 mm,\nhand 181 mm long "
                     "(ANSUR II women's means)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-110, 300)
    ax.set_ylim(-90, 60)
    ax.set_xlabel("x: toward the forearm (mm)")
    ax.set_ylabel("y: toward the thumb (mm)")
    ax.set_title("A. Training hand, right, palm up (fingers drawn straight, dashed)")
    ax.grid(alpha=0.3)

    # B: side view of the middle finger at rest and opened, joints and springs
    bx = fig.add_axes([0.52, 0.42, 0.45, 0.54])
    xs = np.linspace(-1.0, 9.1, 60)
    bx.fill_between(xs * 10, hm.BACK_Z * 10, [hm.palm_top(x, 0.0) * 10 for x in xs], color="#e9cdb6",
                    ec="#8a6a55")
    rest = hm.FINGERS["middle"]["pip"]
    for mcp, pip, ls, lab in ((90.0, rest, "-", f"resting fist (knuckle 90, middle joint {rest:.0f} deg)"),
                              (10.0, rest, "--", "held open (knuckle 10 deg)"),
                              (60.0, rest - 2.0, ":", "resting on the cone (knuckle about 60 deg)")):
        k, pp, q, t = hm.finger_points("middle", mcp=mcp, pip=pip)
        pts = np.array([k, pp, q, t]) * 10
        bx.plot(pts[:, 0], pts[:, 2], ls, color="#8a6a55", lw=8, alpha=.45, solid_capstyle="round", label=lab)
        bx.plot(pts[:3, 0], pts[:3, 2], "ko", ms=4)
    bx.add_patch(plt.Circle((29, 13 + 15), 15, fill=False, ec="#c08040", lw=2))
    bx.text(46, 40, "cone in the palm, about 29 mm\nbehind the knuckles (25 to 42 mm\nacross; 30 mm here)", fontsize=8,
            color="#c08040")
    bx.text(-95, 75, "Knuckle: printed bar on the 3 mm axle carrying all four first segments;\n"
                     f"bar spring {K_KNUCKLE * 1000 * math.pi / 180:.2f} N mm/deg; each segment on the bar: "
                     f"spring {K_PLAY * 1000 * math.pi / 180:.1f} N mm/deg,\nstops 8 deg open / 6 deg closed. "
                     f"Middle joints: 2 mm pin, spring {K_PIP * 1000 * math.pi / 180:.2f} N mm/deg each,\n"
                     "printed stop at the resting bend. End joints printed solid at 30 deg.\n"
                     f"Thumb: 2 mm pin upright at its knuckle, spring {K_THUMB * 1000 * math.pi / 180:.2f} N mm/deg, "
                     "stop at rest.\nAll springs unloaded in the resting fist.", fontsize=8)
    bx.set_aspect("equal")
    bx.set_xlim(-100, 100)
    bx.set_ylim(-20, 110)
    bx.legend(fontsize=7, loc="lower right")
    bx.set_title("B. Middle finger, side view: joints (dots) and springs")
    bx.set_xlabel("x (mm)")
    bx.set_ylabel("up from the knuckle axle (mm)")
    bx.grid(alpha=0.3)

    # C: spring torque lines
    cx = fig.add_axes([0.05, 0.06, 0.40, 0.28])
    a = np.linspace(0, 90, 91)
    for k, lab in ((K_KNUCKLE, "knuckle bar (all four fingers)"), (K_PIP, "each middle joint"),
                   (K_THUMB, "thumb")):
        cx.plot(a, k * np.radians(a) * 1000, label=lab)
    cx.set_xlabel("opened from rest (deg)")
    cx.set_ylabel("closing moment (N mm)")
    cx.legend(fontsize=8)
    cx.grid(alpha=0.3)
    cx.set_title("C. Moments the springs must give (calibration in the README)")

    # D: the cone
    dx = fig.add_axes([0.52, 0.06, 0.42, 0.28])
    dx.plot([0, CONE_L, CONE_L, 0, 0], [CONE_DN / 2, CONE_DW / 2, -CONE_DW / 2, -CONE_DN / 2, CONE_DN / 2],
            color="#c08040", lw=2)
    dx.text(CONE_L / 2 - 30, 0, f"palm cone, {CONE_L:.0f} mm long\nfirm plastic, terry cover, 70 g", fontsize=9)
    dx.text(-4, CONE_DN / 2 + 4, f"narrow end {CONE_DN:.0f} mm\n(thumb side)", fontsize=8)
    dx.text(CONE_L - 30, CONE_DW / 2 + 4, f"wide end {CONE_DW:.0f} mm\n(little-finger side)", fontsize=8)
    dx.set_aspect("equal")
    dx.set_xlim(-10, 150)
    dx.set_ylim(-40, 45)
    dx.set_title("D. The cone (Posey Palm Cone, REF 5608, or any firm palm cone of this size)")
    dx.grid(alpha=0.3)
    fig.savefig(OUT / "drawing.png", dpi=100)
    print("drawing.png")


if __name__ == "__main__":
    main()
