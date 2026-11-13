"""On the station table, in front of the arms: a garden tub with two wooden battens laid across it, and on the battens a
garden sieve full of potting soil with six pebbles on top.

Task (instruction): sift the soil into the tub, keeping the pebbles in the sieve, and leave the sieve resting on the
battens. The instruction does not say how. The sieve is wider than a gripper can span, with a flat handle on each side.
Only held by both handles can it be lifted off the battens, shaken level over the tub, the way gardeners riddle soil
over a barrow, and set back; lifted by one handle it stays propped on its other side, and rocked that way it walks off
the battens into the tub (SIEVE_ONE=1).

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- tub: fixed, 28 x 28 cm inside, 6 cm tall, centred at (0.40, 0).
- battens: fixed, 2 x 32 x 1.2 cm, across the tub's rims under the sieve's front and back walls.
- sieve: loose, 0.6 kg; a wooden frame 20 cm square inside and 7 cm tall (front and back walls 2.4 cm thick, side
  walls 8 mm) over a steel mesh of 4 mm bars with 14 mm square holes; a flat handle 6 x 2.5 x 1 cm out of the middle of
  its left and right walls, 2.5 cm above the mesh. It starts on the battens, centred over the tub.
- soil: 72 crumbs, 2 x 1 x 1 cm, 3 g each (216 g), in two layers on the mesh. Each lies across the mesh until shaking
  tips it end-first through a hole (see the fidelity note in task.json); 4 drop through while the scene settles.
- pebbles: 6, 2.6 cm across, 30 g each, laid on the soil; they roll down onto the mesh while the scene settles.
- crumbs and pebbles: gentle contact settings in PhysX (see LOOSE_PHYSICS).

Score (read at the end of the run). Soil counts only once it has gone down through the mesh: each crumb is followed
from inside the sieve to below the mesh; a crumb that leaves the sieve over its walls instead is never credited.
0 for the rest of the run once a pebble has left the sieve (over its walls, onto the table or into the tub) or once
any soil has left it over its walls. Otherwise by the mass of soil that went through the mesh and is in the tub, as a
share of the 216 g at the start: 0.25 for 25 % or more (18 crumbs), 0.5 for 50 % or more (36 crumbs). Full score:
70 % or more (51 crumbs), every pebble in the sieve, and the sieve resting back on the battens (its middle within 2 cm
of where it started and within 1.5 cm of its starting height, tilted less than 6 degrees), let go (both grippers'
centres 3 cm or more from every part of it), still for 2 s.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, SphereShape, allow_matrix_grips, as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
# The file follows the soil and the pebbles over the run itself (where each crumb left the sieve; a pebble or soil out
# over the walls latches 0), so the score is read at the end of the run.
SCORE_OVER = "end"

# Debug switches (see README.md): SIEVE_DEBUG=1 prints DBG lines; SIEVE_ONE=1 is the wrong way (the left arm alone);
# SIEVE_PEBBLE_OUT=1 and SIEVE_POUR=1 replay the two loopholes the scoring closes; SIEVE_NO_PEBBLES=1 leaves the
# pebbles out of the scene, as if they had all been taken out to the table beforehand.
_DEBUG = os.environ.get("SIEVE_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("SIEVE_DEBUG_EVERY", "10"))  # steps between debug lines
_DEBUG_FAST = os.environ.get("SIEVE_DEBUG_FAST") == "1"  # also print anything thrown up inside the sieve, every step
ONE = os.environ.get("SIEVE_ONE") == "1"
PEBBLE_OUT = os.environ.get("SIEVE_PEBBLE_OUT") == "1"
POUR = os.environ.get("SIEVE_POUR") == "1"
NO_PEBBLES = os.environ.get("SIEVE_NO_PEBBLES") == "1"

# -- layout (world frame) -----------------------------------------------------------------------------------------------
TUB_XY, TUB_IN, TUB_H, TUB_T = (0.40, 0.0), 0.28, 0.06, 0.01
BATTEN = (0.020, TUB_IN + 2 * TUB_T + 0.02, 0.012)  # across the tub's left and right rims
TUB_C, BATTEN_C = (0.22, 0.42, 0.28), (0.66, 0.52, 0.36)

# -- the sieve (origin: the middle of the mesh's underside) -------------------------------------------------------------
INNER, WALL_T, WALL_H = 0.200, 0.008, 0.070
FB_T = 0.024  # front and back walls 2.4 cm thick (they rest on the battens), side walls 8 mm
PITCH, BAR, BAR_H = 0.018, 0.004, 0.006  # 14 mm holes between 4 mm bars
LUG = (0.060, 0.025, 0.010)  # x, y, z: a flat handle out of each side wall, at the loaded sieve's centre of mass
LUG_Z = 0.025  # at the loaded sieve's centre of mass: 4 cm up, it swung about the handles when shaken
LUG_Y = INNER / 2 + WALL_T + LUG[1] / 2
SIEVE_KG = 0.6
FRAME_C, MESH_C, LUG_C = (0.55, 0.40, 0.25), (0.30, 0.30, 0.32), (0.18, 0.18, 0.18)
BATTEN_X = INNER / 2 + FB_T / 2  # the battens under the front and back walls (+-1.5 cm of play), clear of the mesh
START = np.array([TUB_XY[0], TUB_XY[1], TOP + TUB_H + BATTEN[2]])  # the sieve's origin resting on the battens
SPAWN_DZ = 0.001  # spawned 1 mm up: the thick walls flush on the battens were shoved 6.7 cm sideways at reset

# -- soil and pebbles ---------------------------------------------------------------------------------------------------
CRUMB, CRUMB_KG, CRUMB_MU = (0.020, 0.010, 0.010), 0.003, (0.35, 0.25)
N_SIDE, LAYERS = 6, 2  # 72 crumbs
STONE_R, STONE_KG, N_STONES = 0.013, 0.030, 6
SOIL_C, STONE_C = (0.35, 0.25, 0.16), (0.62, 0.62, 0.64)
STONE_SPOTS = [(-0.05, -0.05), (0.05, -0.04), (0.0, 0.0), (-0.04, 0.05), (0.05, 0.05), (0.0, -0.075)]  # from the middle
# Crumbs and pebbles jammed between the thin bars, a pebble and a wall were thrown out of the sieve at up to 11 m/s by
# the physics engine (debug runs sieveS1, sieveS2); real soil and stones in a shaken sieve move far slower. They get
# PhysX's maximum linear velocity 0.8 m/s, push-out at most 0.3 m/s and 32 solver iterations: kicks of a few m/s still
# happen now and then, but in the runs since none has thrown a piece out of the sieve (sieveW1-W3, sieveF1-F5).
LOOSE_PHYSICS = dict(max_speed=0.8, max_depenetration=0.3, solver_iterations=32)


def _sieve() -> Body:
    shapes = []
    n = int(round(INNER / PITCH))
    for k in range(n + 1):
        c = -INNER / 2 + k * INNER / n
        shapes.append(BoxShape((c, 0.0, BAR_H / 2), (BAR, INNER, BAR_H), MESH_C))
        shapes.append(BoxShape((0.0, c, BAR_H / 2), (INNER, BAR, BAR_H), MESH_C))
    a, fb, long = INNER / 2 + WALL_T / 2, INNER / 2 + FB_T / 2, INNER + 2 * WALL_T
    shapes += [BoxShape((fb, 0.0, WALL_H / 2), (FB_T, long, WALL_H), FRAME_C),
               BoxShape((-fb, 0.0, WALL_H / 2), (FB_T, long, WALL_H), FRAME_C),
               BoxShape((0.0, a, WALL_H / 2), (INNER, WALL_T, WALL_H), FRAME_C),
               BoxShape((0.0, -a, WALL_H / 2), (INNER, WALL_T, WALL_H), FRAME_C)]
    shapes += [BoxShape((0.0, s * LUG_Y, LUG_Z), LUG, LUG_C) for s in (1, -1)]
    return Body.build("sieve", pos=tuple(START + [0.0, 0.0, SPAWN_DZ]), mass=SIEVE_KG, shapes=shapes, friction=(1.0, 0.9),
                      color=FRAME_C)


def _fixed() -> list:
    (tx, ty), h, t, w = TUB_XY, TUB_IN / 2 + TUB_T / 2, TUB_T, TUB_IN + 2 * TUB_T
    z = TOP + TUB_H / 2
    tub = [Box("tub_floor", pos=(tx, ty, TOP + 0.002), size=(w, w, 0.004), color=TUB_C),
           Box("tub_front", pos=(tx - h, ty, z), size=(t, w, TUB_H), color=TUB_C),
           Box("tub_back", pos=(tx + h, ty, z), size=(t, w, TUB_H), color=TUB_C),
           Box("tub_left", pos=(tx, ty + h, z), size=(TUB_IN, t, TUB_H), color=TUB_C),
           Box("tub_right", pos=(tx, ty - h, z), size=(TUB_IN, t, TUB_H), color=TUB_C)]
    battens = [Box(f"batten_{name}", pos=(tx + s * BATTEN_X, ty, TOP + TUB_H + BATTEN[2] / 2), size=BATTEN,
                   color=BATTEN_C) for name, s in (("front", -1), ("back", 1))]
    return tub + battens


def _soil() -> list[Body]:
    out, k = [], 0
    pitch = 0.026
    z0 = START[2] + SPAWN_DZ + BAR_H + CRUMB[2] / 2 + 0.0005
    for layer in range(LAYERS):
        for i in range(N_SIDE):
            for j in range(N_SIDE):
                x = START[0] + (i - (N_SIDE - 1) / 2) * pitch + (0.006 if layer else 0.0)
                y = START[1] + (j - (N_SIDE - 1) / 2) * pitch + (0.006 if layer else 0.0)
                yaw = (0.0, 0.5, 1.1, 1.57, 2.2)[k % 5]
                out.append(Body.build(f"crumb_{k:02d}", pos=(x, y, z0 + layer * (CRUMB[2] + 0.0025)), yaw=yaw,
                                      mass=CRUMB_KG, shapes=[BoxShape((0.0, 0.0, 0.0), CRUMB, SOIL_C)],
                                      friction=CRUMB_MU, damping=(0.05, 0.3), color=SOIL_C, **LOOSE_PHYSICS))
                k += 1
    return out


def _stones() -> list[Body]:
    if NO_PEBBLES:
        return []
    z = START[2] + SPAWN_DZ + BAR_H + LAYERS * (CRUMB[2] + 0.0025) + STONE_R + 0.002
    return [Body.build(f"pebble_{k}", pos=(START[0] + dx, START[1] + dy, z), mass=STONE_KG,
                       shapes=[SphereShape((0.0, 0.0, 0.0), STONE_R, STONE_C)], friction=(0.8, 0.7), color=STONE_C,
                       **LOOSE_PHYSICS)
            for k, (dx, dy) in enumerate(STONE_SPOTS[:N_STONES])]


SIEVE = _sieve()
SOIL, STONES = _soil(), _stones()
CRUMB_NAMES, STONE_NAMES = [c.name for c in SOIL], [s.name for s in STONES]
OBJECTS = [*_fixed(), SIEVE, *SOIL, *STONES]

# -- scoring --------------------------------------------------------------------------------------------------------
# Credit is the share of the starting soil's mass (216 g) that went down through the mesh and is in the tub. All crumbs
# weigh 3 g, so a share is a number of crumbs: 70 % (151.2 g) takes 51 of the 72, 50 % (108 g) 36 and 25 % (54 g) 18.
FULL, HALF, QUARTER = 0.70, 0.50, 0.25
SOIL_G = len(SOIL) * CRUMB_KG * 1000.0


def _crumbs_for(share: float) -> int:
    return math.ceil(share * len(SOIL) - 1e-9)


N_FULL, N_HALF, N_QUARTER = _crumbs_for(FULL), _crumbs_for(HALF), _crumbs_for(QUARTER)
BACK_XY, BACK_Z, LEVEL = 0.02, 0.015, math.radians(6)  # back on the battens (a crumb under a wall is fine)
CLEAR = 0.03  # let go: each gripper's centre this far from every part of the sieve (holding it puts it within ~1 cm)
SLACK = 0.03  # seen below the mesh up to this far outside the walls' inner faces: still through the mesh (it falls
# away from the moving sieve for up to one 50 ms step before it is seen)
# The sieve's parts as boxes in its own frame (lower corner, upper corner): the mesh, the four walls, the two handles.
_FB, _SW = INNER / 2 + FB_T, INNER / 2 + WALL_T
PARTS = [((-INNER / 2, -INNER / 2, 0.0), (INNER / 2, INNER / 2, BAR_H)),
         ((INNER / 2, -_SW, 0.0), (_FB, _SW, WALL_H)), ((-_FB, -_SW, 0.0), (-INNER / 2, _SW, WALL_H)),
         ((-INNER / 2, INNER / 2, 0.0), (INNER / 2, _SW, WALL_H)),
         ((-INNER / 2, -_SW, 0.0), (INNER / 2, -INNER / 2, WALL_H))]
PARTS += [((-LUG[0] / 2, s * LUG_Y - LUG[1] / 2, LUG_Z - LUG[2] / 2),
           (LUG[0] / 2, s * LUG_Y + LUG[1] / 2, LUG_Z + LUG[2] / 2)) for s in (1.0, -1.0)]
PARTS = [(np.array(lo), np.array(hi)) for lo, hi in PARTS]
_EP: dict = {}
_TCP: dict = {}
_ARM_IDS: dict = {}


@on_reset
def _reset() -> None:
    _EP.update(step=0, tick=0, seen=-1, crumb={}, failed="", pebble_out="", n_credit=0, n_through=0, n_rim=0, n_out=0,
               n_in_tub=0)
    _TCP.clear()


_reset()


@on_step
def _track_grippers(env) -> None:
    """Each gripper's centre (TCP) and finger opening, from its arm's joint positions, after every control step."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _ARM_IDS:
            _ARM_IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
        q = as_numpy(robot.data.joint_pos)[0]
        _TCP[side] = (np.asarray(ik.tcp_pose(q[_ARM_IDS[side]], station.BASES[side])[0], float), q[-2:])


def _in_tub(p) -> bool:
    x, y, z = p
    return bool(abs(x - TUB_XY[0]) < TUB_IN / 2 and abs(y - TUB_XY[1]) < TUB_IN / 2 and TOP < z < TOP + TUB_H)


def _local(s, p) -> np.ndarray:
    """A world point in the sieve's frame (origin: the middle of the mesh's underside, z up through the mesh)."""
    return rotation(s.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(s.pos, float))


def _within_walls(l) -> bool:
    return bool(abs(l[0]) <= INNER / 2 and abs(l[1]) <= INNER / 2)


def _update(o) -> None:
    """Once per control step: where each crumb is in its way out of the sieve, and the latched rules.

    A crumb is "in" while its centre is above the mesh's underside and within the walls' inner faces. From "in", going
    below the mesh's underside means it went down through the mesh ("through"); going outside the walls' inner faces
    means it left over a wall ("rim"), which latches 0 for the run, as does a pebble whose centre leaves the walls'
    inner faces or goes below the mesh. A crumb that went through and comes back up above the bars is "in" again. A
    crumb already below the mesh at the first step dropped through a hole while the scene settled ("through").
    """
    if _EP["seen"] == _EP["tick"]:
        return
    _EP["seen"] = _EP["tick"]
    s = o["sieve"]
    status = _EP["crumb"]
    for n in CRUMB_NAMES:
        l = _local(s, o[n].pos)
        st = status.get(n)
        if st is None:  # the first step of the run: every crumb starts on the mesh, and a few drop through the holes
            # while the scene settles before the run (they lie in the tub, as real soil spread on the mesh would)
            if l[2] >= 0.0 and _within_walls(l):
                st = "in"
            else:
                st = "through" if l[2] < 0.0 and max(abs(l[0]), abs(l[1])) <= INNER / 2 + SLACK else "out"
        elif st == "in":
            if l[2] < 0.0:  # below the mesh: from inside the walls this takes the way through a hole
                st = "through" if max(abs(l[0]), abs(l[1])) <= INNER / 2 + SLACK else "rim"
            elif not _within_walls(l):
                st = "rim"
        elif st == "through" and l[2] > BAR_H and _within_walls(l):
            st = "in"
        status[n] = st
    n_rim = sum(v == "rim" for v in status.values())
    if n_rim and not _EP["failed"]:
        _EP["failed"] = "soil left the sieve over its walls"
    for n in STONE_NAMES:
        l = _local(s, o[n].pos)
        if (l[2] < 0.0 or not _within_walls(l)) and not _EP["pebble_out"]:
            _EP["pebble_out"] = n
            if not _EP["failed"]:
                _EP["failed"] = f"{n} left the sieve"
    through = [n for n in CRUMB_NAMES if status[n] == "through"]
    _EP.update(n_through=len(through), n_credit=sum(_in_tub(o[n].pos) for n in through), n_rim=n_rim,
               n_out=sum(v == "out" for v in status.values()), n_in_tub=sum(_in_tub(o[n].pos) for n in CRUMB_NAMES))
    if _DEBUG_FAST:  # what throws soil or pebbles up: anything in the sieve rising fast or above the walls' tops
        r = rotation(s.quat_xyzw)
        for n in CRUMB_NAMES + STONE_NAMES:
            if n in status and status[n] != "in":
                continue
            l = _local(s, o[n].pos)
            v = np.asarray(o[n].lin_vel, float) - np.asarray(s.lin_vel, float)
            if v[2] > 0.35 or l[2] > WALL_H - 0.01:
                print(f"DBG fast step={_EP['tick']} {n} local=({l[0]:.3f},{l[1]:.3f},{l[2]:.3f}) "
                      f"v_rel=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f}) sieve_v=({s.lin_vel[0]:.2f},{s.lin_vel[1]:.2f},"
                      f"{s.lin_vel[2]:.2f}) sieve_w=({s.ang_vel[0]:.1f},{s.ang_vel[1]:.1f},{s.ang_vel[2]:.1f}) "
                      f"tilt={math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2])))):.1f}", flush=True)


def _tally(o) -> dict:
    _update(o)
    s = o["sieve"]
    tilt = float(math.acos(max(-1.0, min(1.0, rotation(s.quat_xyzw)[2, 2]))))
    back = bool(math.hypot(s.pos[0] - START[0], s.pos[1] - START[1]) <= BACK_XY and abs(s.pos[2] - START[2]) < BACK_Z
                and tilt < LEVEL)
    gaps = {}
    for side, (tcp, _) in _TCP.items():
        l = _local(s, tcp)
        gaps[side] = min(float(np.linalg.norm(l - np.clip(l, lo, hi))) for lo, hi in PARTS)
    let_go = len(gaps) == 2 and min(gaps.values()) >= CLEAR
    stones_in = all(0.0 <= l[2] <= WALL_H and _within_walls(l) for l in (_local(s, o[n].pos) for n in STONE_NAMES))
    still = s.resting()
    done = bool(not _EP["failed"] and _EP["n_credit"] >= N_FULL and stones_in and back and let_go and still)
    return dict(tilt=tilt, back=back, gaps=gaps, let_go=let_go, stones_in=stones_in, still=still, done=done)


def _stage() -> float:
    if _EP["failed"]:
        return 0.0
    n = _EP["n_credit"]
    return 0.5 if n >= N_HALF else 0.25 if n >= N_QUARTER else 0.0


def _debug(o, t) -> None:
    _EP["step"] += 1
    if _EP["step"] == 1:
        print("DBG start: " + " ".join(f"{n}=({o[n].pos[0]:.3f},{o[n].pos[1]:.3f},{o[n].pos[2] - TOP:.3f})"
                                       for n in STONE_NAMES), flush=True)
    if (_EP["step"] - 1) % _DEBUG_EVERY:
        return
    s = o["sieve"]
    gaps = " ".join(f"{k}={v:.3f}" for k, v in t["gaps"].items())
    print(f"DBG step={_EP['step']} sieve=({s.pos[0]:.3f},{s.pos[1]:.3f},{s.pos[2] - TOP:.3f}) "
          f"tilt={math.degrees(t['tilt']):.0f} through={_EP['n_through']} "
          f"credited={_EP['n_credit']}/{len(CRUMB_NAMES)} "
          f"({_EP['n_credit'] * CRUMB_KG * 1000:.0f} g of {SOIL_G:.0f} g) in_tub_any_way={_EP['n_in_tub']} "
          f"rim={_EP['n_rim']} out={_EP['n_out']} pebble_out={_EP['pebble_out'] or '-'} stones_in={t['stones_in']} "
          f"back={t['back']} gaps=[{gaps}] let_go={t['let_go']} still={t['still']} done={t['done']} "
          f"stage={_stage()} failed={_EP['failed'] or '-'}", flush=True)
    if len(_TCP) == 2:
        r = rotation(s.quat_xyzw)
        roll, pitch = math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(-math.asin(max(-1.0, min(1.0, r[2, 0]))))
        out = []
        for side, sy in (("left", 1.0), ("right", -1.0)):
            lug = np.asarray(s.pos) + r @ np.array([0.0, sy * LUG_Y, LUG_Z])
            tcp, fingers = _TCP[side]
            out.append(f"{side} tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2] - TOP:.3f}) "
                       f"d_lug={np.round(tcp - lug, 3).tolist()} "
                       f"f={np.round(fingers, 4).tolist()}")
        print(f"DBG   roll={roll:.1f} pitch={pitch:.1f} " + " | ".join(out), flush=True)


def score(objects) -> float:
    """Partial credit at the end of the run (success scores 1.0): 0 once a pebble or any soil has left the sieve over
    its walls; otherwise 0.5 with at least 50 % of the starting soil's mass (36 of the 72 crumbs) through the mesh and
    in the tub, 0.25 with at least 25 % (18 crumbs), else 0."""
    _tally(objects)
    return _stage()


def success(objects) -> bool:
    """At least 70 % of the soil (51 crumbs) through the mesh and in the tub, no pebble or soil ever out over the walls,
    every pebble in the sieve, the sieve back on the battens, both grippers' centres 3 cm or more from every part of it,
    the sieve still."""
    t = _tally(objects)
    if _DEBUG:
        _debug(objects, t)
    return t["done"]


# -- scripted demo --------------------------------------------------------------------------------------------------
# Each gripper comes in level from its own side, pointing inwards with its fingers upright, slides onto its handle from
# the front and clamps it from above and below (the lower finger carries the load). Together they lift the sieve off the
# battens, shake it round in small circles, level, until the soil has gone through, set it back on the battens, let go,
# slide off the handles to the front and rise clear. The demo is open-loop: it plays the same waypoints from the known
# layout and reads nothing from the simulation.


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


SIDE_ROT = {"left": _ry(-math.pi / 2) @ _rx(-math.pi / 2) @ grip_rotation(-math.pi / 2),  # level, pointing -y,
            "right": _ry(-math.pi / 2) @ _rx(math.pi / 2) @ grip_rotation(math.pi / 2)}  # fingers closing along z
DOWN_ROT = {"left": grip_rotation(-math.pi / 2), "right": grip_rotation(math.pi / 2)}  # straight down, closing along x
GRIP_IN = 0.007  # the TCP this far in from the handle's outer edge: the fingertips (1.3 cm on) stop 5 mm short of it
CLAMP = float(os.environ.get("SIEVE_CLAMP", "-0.001"))  # 2 mm between the fingers round the 1 cm handle: a firm clamp
OPEN = -0.020  # 4 cm apart
SLIDE = 0.055  # start this far in front of the handle's middle, then slide on
LIFT = float(os.environ.get("SIEVE_LIFT", "0.035"))  # lift the sieve this far off the battens to shake it
AMP = float(os.environ.get("SIEVE_AMP", "0.03"))  # radius (m) of the shaking circles, front to back
AMP_Y = float(os.environ.get("SIEVE_AMP_Y", str(AMP)))  # and side to side
HZ = float(os.environ.get("SIEVE_HZ", "3.0"))
SHAKE_S = float(os.environ.get("SIEVE_SHAKE_S", "20"))
RAMP_S = 1.0  # the circles grow from nothing over the first second and shrink to nothing over the last
PITCH_COMP = float(os.environ.get("SIEVE_PITCH_COMP", "4.5"))  # deg, see _arm()
REVERSE_S = float(os.environ.get("SIEVE_REVERSE_S", "2"))  # reverse the circles every 2 s
STEP_S = 0.05  # one control step at 20 Hz
# Shaken, the handles slip 1.3 to 1.9 cm back in the clamps (debug runs sieveW1-W3): the sieve is set down this much
# further forward, so that it lands where it started.
SET_DX = 0.015
OFF_FRONT, RISE = 0.085, 0.08  # after letting go: slide this far to the front, off the handle, then rise this far
# SIEVE_POUR=1: lift the sieve higher and turn it about the line between the handles, back wall down, so that the soil
# pours out over the back wall into the tub (the wall swings down between the battens), then turn it back level, set it
# back on the battens and let go.
POUR_LIFT = float(os.environ.get("SIEVE_POUR_LIFT", "0.10"))
POUR_DEG = float(os.environ.get("SIEVE_POUR_DEG", "120"))


def _grip_point(arm: str, origin) -> np.ndarray:
    s = 1.0 if arm == "left" else -1.0
    return np.asarray(origin, float) + np.array([0.0, s * (LUG_Y + LUG[1] / 2 - GRIP_IN), LUG_Z])


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    for sd in itertools.product((0.0,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (-1.2, 0.0, 1.2), (-1.5, 0.0, 1.5)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=station.BASES[arm], iters=150)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _joint_line(arm: str, q0, q1, finger: float, label: str, seconds: float, steps: int) -> list[Waypoint]:
    """TCP waypoints along a straight line in joint space (a big turn of the wrist stays on one IK branch)."""
    from yam_arena import ik, station

    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list[Waypoint]:
    """From READY to the pose, along a straight line in joint space."""
    from yam_arena.constants import READY

    return _joint_line(arm, np.array(READY, float), _good_ik(arm, pos, rot), finger, label, seconds, steps)


def _back_to_ready(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0,
                   steps: int = 15) -> list[Waypoint]:
    """From the pose back to READY, along a straight line in joint space."""
    from yam_arena.constants import READY

    return _joint_line(arm, _good_ik(arm, pos, rot), np.array(READY, float), finger, label, seconds, steps)


def _stay(arm: str, seconds: float) -> list[Waypoint]:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    pos, rot = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return [Waypoint(seconds, tuple(float(v) for v in pos), OPEN, "stay put", rot)]


def _shake_offsets() -> list[tuple[float, np.ndarray]]:
    """(duration, offset from the lifted pose) of each shaking waypoint: one per control step on a circle, so that the
    grips go round at an even speed (the player samples each step half-way between two waypoints, on the circle's
    chord); the circles reverse every REVERSE_S and grow from nothing over the first RAMP_S and shrink over the last."""
    out, phase = [], 0.0
    for k in range(1, int(round(SHAKE_S / STEP_S)) + 1):
        t = k * STEP_S
        turn = -1.0 if REVERSE_S > 0 and int(t / REVERSE_S) % 2 else 1.0
        phase += turn * 2.0 * math.pi * HZ * STEP_S
        r = max(0.0, min(1.0, t / RAMP_S, (SHAKE_S - t) / RAMP_S))
        out.append((STEP_S, np.array([r * AMP * math.cos(phase), r * AMP_Y * math.sin(phase), 0.0])))
    return out


def _arm(arm: str) -> list[Waypoint]:
    # Loaded, each wrist droops and the sieve pitches back ~4.5 deg about the line between the handles (debug runs
    # sfE): while it is held up, the grips are tipped forward by as much so that it hangs level.
    level = _ry(-math.radians(PITCH_COMP)) @ SIDE_ROT[arm]

    def w(t, origin, finger, label, dx=0.0, dz=0.0, rot=SIDE_ROT[arm]):
        p = _grip_point(arm, origin) + np.array([dx, 0.0, dz])
        return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)

    rot = SIDE_ROT[arm]

    lifted = START + [0.0, 0.0, LIFT]
    # Come down in front of the handle, never over it: an open lower finger passing over the handle lands on it.
    high = _grip_point(arm, START) + [-SLIDE, 0.0, 0.08]
    out = _lead_in(arm, high, rot, OPEN, "level, above the front of its handle, fingers upright")
    out += [w(1.5, START, OPEN, "down beside the handle's front end", dx=-SLIDE),
            w(1.5, START, OPEN, "slide onto the handle"),
            w(1.2, START, CLAMP, "clamp it from above and below"),
            w(0.5, START, CLAMP, "steady"),
            w(2.0, lifted, CLAMP, "lift the sieve off the battens, level", rot=level)]
    if POUR:
        high_up = START + [0.0, 0.0, POUR_LIFT]
        tipped = _ry(math.radians(POUR_DEG)) @ level
        out += [w(2.0, high_up, CLAMP, "(loophole) lift it higher", rot=level),
                w(3.0, high_up, CLAMP, "(loophole) turn it back wall down: the soil pours out over it", rot=tipped),
                w(4.0, high_up, CLAMP, "(loophole) hold it tipped", rot=tipped),
                w(3.0, high_up, CLAMP, "(loophole) turn it back level", rot=level),
                w(2.0, lifted, CLAMP, "(loophole) lower it", rot=level)]
    else:
        out.append(w(STEP_S / 2, lifted, CLAMP, "steady", rot=level))  # half a step: the player samples mid-waypoint
        out += [w(t, lifted + d, CLAMP, "shake it round in small circles", rot=level) for t, d in _shake_offsets()]
    out += [w(1.0, lifted, CLAMP, "hold it still: the last soil drops", rot=level),
            w(2.0, START + [SET_DX, 0.0, 0.002], CLAMP, "set it back on the battens"),
            w(1.0, START + [SET_DX, 0.0, 0.002], OPEN, "let go"),
            w(1.5, START, OPEN, "slide off the handle to the front", dx=SET_DX - OFF_FRONT),
            w(1.5, START, OPEN, "rise clear of the sieve", dx=SET_DX - OFF_FRONT, dz=RISE),
            w(4.0, START, OPEN, "wait", dx=SET_DX - OFF_FRONT, dz=RISE)]
    return out


# SIEVE_PEBBLE_OUT=1 (a loophole the scoring closes): before the sift, the left gripper, tilted 45 degrees forward
# (straight down it does not reach that far out), takes the pebble in the middle of the sieve, lifts it out over the
# left wall and lays it on the table beside the tub; then both arms sift the soil as in the demo. The right arm waits at
# READY meanwhile. Where the pebble lies once the scene has settled (the pebbles roll off the soil while the scene
# settles before the run) comes from a debug run (the DBG start line of sieveW1: pebble_2).
PEBBLE_AT = (0.388, 0.028, 0.101)  # pebble_2's centre once settled (x, y, height above the table)
PEBBLE_ROT = _ry(math.radians(-45.0)) @ grip_rotation(0.0)  # fingers closing along y, pointing forward and down
PEBBLE_OPEN, PEBBLE_SHUT = -0.025, -0.009  # 5 cm apart; 1.8 cm, 4 mm inside each side of the 2.6 cm pebble
PEBBLE_DROP = (0.30, 0.20)  # on the table, left of the tub


def _pebble_out(arm: str = "left") -> list[Waypoint]:
    rot = PEBBLE_ROT
    at = np.array([PEBBLE_AT[0], PEBBLE_AT[1], TOP + PEBBLE_AT[2]])
    before = at - 0.06 * (rot @ np.array([0.0, 0.0, 1.0]))  # 6 cm back along the fingers
    up = np.array([at[0], at[1], TOP + 0.15])
    over_wall = np.array([0.37, 0.07, TOP + 0.168])
    out_wall = np.array([0.34, 0.115, TOP + 0.168])  # the pebble's bottom 1.3 cm above the walls' tops
    out_over = np.array([PEBBLE_DROP[0], PEBBLE_DROP[1], TOP + 0.168])
    down = np.array([PEBBLE_DROP[0], PEBBLE_DROP[1], TOP + 0.02])

    def p(t, pos, finger, label):
        return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)

    out = _lead_in(arm, before, rot, PEBBLE_OPEN, "(loophole) behind a pebble, fingers open")
    out += [p(1.5, at, PEBBLE_OPEN, "(loophole) fingers either side of it"),
            p(1.0, at, PEBBLE_SHUT, "(loophole) take it"),
            p(0.5, at, PEBBLE_SHUT, "(loophole) settle"),
            p(1.5, up, PEBBLE_SHUT, "(loophole) lift it out of the soil"),
            p(1.0, over_wall, PEBBLE_SHUT, "(loophole) towards the left wall"),
            p(1.0, out_wall, PEBBLE_SHUT, "(loophole) over the wall"),
            p(1.5, out_over, PEBBLE_SHUT, "(loophole) out over the table"),
            p(1.5, down, PEBBLE_SHUT, "(loophole) down to the table"),
            p(0.8, down, PEBBLE_OPEN, "(loophole) lay it there"),
            p(1.0, down + [0.0, 0.0, 0.06], PEBBLE_OPEN, "(loophole) rise")]
    out += _back_to_ready(arm, down + [0.0, 0.0, 0.06], rot, OPEN, "(loophole) back to the start pose")
    return out


def _duration(wps) -> float:
    return sum(w.duration for w in wps)


def demo():
    if ONE:  # the wrong way: the left arm alone lifts the sieve by its handle and shakes it
        left = _arm("left")
        return {"left": left, "right": _stay("right", _duration(left))}
    if PEBBLE_OUT:
        first = _pebble_out("left")
        return {"left": first + _arm("left"), "right": _stay("right", _duration(first)) + _arm("right")}
    return {"left": _arm("left"), "right": _arm("right")}
