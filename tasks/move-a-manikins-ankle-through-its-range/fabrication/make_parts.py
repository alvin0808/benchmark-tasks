"""Fabrication files for the ankle task: the training leg's two shells as STL (millimetres, closed surfaces, the same
meshes the simulation uses) and a drawing sheet (side view with the main dimensions, the ankle's spring layout and
its moment-angle curve). Writes into <task>/fabrication/. Run: python make_parts.py (needs numpy and matplotlib)."""
import importlib.util
import pathlib
import struct
import sys

import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parents[1]
OUT = TASK_DIR / "fabrication"
sys.path.insert(0, str(TASK_DIR))
import leg_mesh as lm  # noqa: E402


def write_stl(path, pts_m, faces, title):
    pts = np.asarray(pts_m, float) * 1000.0
    tri = pts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as fh:
        fh.write(title.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(faces)))
        for k in range(len(faces)):
            fh.write(struct.pack("<12fH", *n[k], *tri[k, 0], *tri[k, 1], *tri[k, 2], 0))
    return len(faces)


def main():
    sp, sf, _ = lm.shank_mesh()
    fp, ff, _ = lm.foot_mesh()
    print("leg_shank.stl", write_stl(OUT / "leg_shank.stl", sp, sf, "ankle task: training leg, lower leg shell (mm)"))
    print("leg_foot.stl", write_stl(OUT / "leg_foot.stl", fp, ff, "ankle task: training leg, foot shell (mm)"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    # A: side view (leg frame: y along the leg, z up) of both shells, the axle and the tape lines
    ax = fig.add_axes([0.03, 0.40, 0.62, 0.55])
    lt, ft = lm.LEG_TABLE, lm.FOOT_TABLE
    ax.fill_between(lt[:, 0] * 10, -lt[:, 3] * 10, lt[:, 2] * 10, color="#e9cdb6", lw=1.5, ec="#8a6a55")
    fz_up = (ft[:, 3] - lm.AXIS_UP / lm.CM) * 10  # dorsum, leg frame y (mm)
    fz_lo = (ft[:, 2] - lm.AXIS_UP / lm.CM) * 10  # sole
    zz = (ft[:, 0] - lm.AXIS_FWD / lm.CM) * 10
    ax.fill_betweenx(zz, fz_lo, fz_up, color="#dfbca2", lw=1.5, ec="#8a6a55")
    ax.plot([0], [0], "ko", ms=6)
    ax.annotate("ankle axle, 8 mm\n(axis 50 mm above the sole,\n49 mm in front of the heel)", (0, 0), (40, -95),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.plot([20, 140], [0, 0], color="k", lw=3)
    ax.text(60, 6, "tape line on the shin's outer side", fontsize=8)
    ax.plot([-25, -25], [30, 130], color="k", lw=3)
    ax.text(-20, 135, "tape line on the foot's outer side", fontsize=8)
    ax.annotate("", (lm.LEG_END * 1000, -75), (0, -75), arrowprops=dict(arrowstyle="<->"))
    ax.text(150, -72, f"axis to knee end {lm.LEG_END * 1000:.0f} mm", fontsize=9)
    ax.annotate("", (-55, 200), (-55, -50), arrowprops=dict(arrowstyle="<->"))
    ax.text(-52, 210, f"foot length {lm.FOOT_LEN * 1000:.0f} mm", fontsize=9)
    ax.text(250, 60, "calf: 110 mm wide, 373 mm round\nankle: 66 mm wide, 216 mm round\n"
                     "(ANSUR II women's means)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("along the leg from the ankle axis (mm)")
    ax.set_ylabel("up, lying on the back (mm)")
    ax.set_title("A. Training leg, outer side, ankle in neutral (lying on its back)")
    ax.grid(alpha=0.3)

    # B: the ankle spring layout (section across the ankle, schematic)
    bx = fig.add_axes([0.68, 0.45, 0.30, 0.48])
    bx.set_xlim(-60, 60)
    bx.set_ylim(-60, 60)
    bx.add_patch(plt.Circle((0, 0), 4, color="k"))
    bx.add_patch(plt.Circle((0, 0), 12, fill=False, lw=2))
    bx.text(14, 2, "torsion spring D\n(resists dorsiflexion)", fontsize=8)
    bx.add_patch(plt.Circle((0, 0), 18, fill=False, lw=2, ls="--"))
    bx.text(14, -26, "torsion spring P\n(resists plantarflexion)", fontsize=8)
    bx.add_patch(plt.Rectangle((-40, 30), 12, 8, color="#555"))
    bx.text(-58, 42, "bump stop D: met at 10 deg\ndorsiflexion", fontsize=8)
    bx.add_patch(plt.Rectangle((28, 30), 12, 8, color="#555"))
    bx.text(18, 42, "bump stop P: met at 20 deg\nplantarflexion", fontsize=8)
    bx.plot([0, 0], [0, 32], color="#a33", lw=3)
    bx.text(-27, 15, "foot's stop\nfinger", fontsize=8, color="#a33")
    bx.text(-58, -55, "Each spring's one leg bears on the shank's ankle block, the other on a pin\n"
                      "in the foot's yoke on its own side only, so it acts one way, from neutral.\n"
                      "Printed hard stops at 30 deg dorsiflexion and 55 deg plantarflexion.",
            fontsize=8)
    bx.set_aspect("equal")
    bx.axis("off")
    bx.set_title("B. Ankle springs and stops (schematic)")

    # C: moment-angle curve
    spec = importlib.util.spec_from_file_location("t", TASK_DIR / "move_a_manikins_ankle_through_its_range.py")
    try:
        t = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(t)
        th = np.linspace(-55, 30, 341)
        m = [t.ankle_moment(v) for v in th]
        cx = fig.add_axes([0.06, 0.06, 0.55, 0.27])
        cx.plot(th, m, color="k")
        for v, lab in ((15, "goal 15"), (20, "end of range 20"), (-30, "goal 30"), (-45, "end of range 45")):
            cx.axvline(v, color="#a33" if "end" in lab else "#36a", lw=1)
            cx.text(v + 0.5, max(m) * 0.8 if v > 0 else min(m) * 0.8, lab, fontsize=8)
        cx.set_xlabel("ankle angle (deg): dorsiflexion +, plantarflexion -")
        cx.set_ylabel("restoring moment (N m)")
        cx.set_title("C. Passive moment the ankle must give (calibration targets in the README)")
        cx.grid(alpha=0.3)
    except Exception as e:  # the task file needs yam_arena; the curve is skipped without it
        print("curve skipped:", e)
    fig.savefig(OUT / "drawing.png", dpi=110)
    print("drawing.png")


if __name__ == "__main__":
    main()
