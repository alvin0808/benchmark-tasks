"""Activate an instant cold pack and mix the contents with the outer pouch intact."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, Joint, MeshShape, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CENTRE = np.array([.305, -.006, TOP + .067])
PACK_YAW = math.pi / 2
T = .017
WIDTH = .229
HEIGHT = .127
PACK = (.86, .90, .92)
SEAL = (.78, .86, .89)
BLUE = (.03, .38, .55)
BOX = (.27, .32, .37)
FACE_NAMES = ("front_wall", "back_wall")
STIFFNESS = 1500.0  # N/m, per face of the pouch
BURST_N = 12.0
BURST_TRAVEL = 2 * BURST_N / STIFFNESS
STROKES = 6
STROKE_LENGTH = .075


def retention_stand():
    pieces = [BoxShape((0, 0, .003), (.268, .085, .006), BOX)]
    for sign in (-1, 1):
        pieces.append(BoxShape((0, sign*.0235, .014), (.250, .007, .017), (.40,.48,.53)))
    return Body.build("tray", (*CENTRE[:2], TOP), .22, pieces, yaw=PACK_YAW,
                      fixed=True, contact_offset=.0003)


def pouch():
    pieces = [
        BoxShape((0., 0., -HEIGHT/2+.004), (WIDTH, .026, .008), SEAL),
        BoxShape((0., 0., HEIGHT/2-.003), (WIDTH, .006, .006), SEAL),
    ]
    for sign in (-1, 1):
        pieces.append(BoxShape((sign*(WIDTH/2-.007), 0., 0.), (.014, .012, HEIGHT), SEAL))
    # A foil perimeter weld encloses the bulging sidewalls in the physical product.
    return Body.build("pouch", CENTRE, .062, pieces, color=SEAL, yaw=PACK_YAW,
                      contact_offset=.0002, solver_iterations=42)


def wrapper_face(sign):
    """Round the visual film surface back toward the welded perimeter."""
    points = [(0., sign*.0038, 0.)]
    slices = 48
    for x_scale, z_scale, depth in ((.087, .049, .0038), (.110, .060, -.0084)):
        for angle in np.arange(slices)*math.tau/slices:
            a, b = math.cos(angle), math.sin(angle)
            points.append((x_scale*math.copysign(math.sqrt(abs(a)),a),
                           sign*depth, z_scale*math.copysign(math.sqrt(abs(b)),b)))
    faces = []
    for i in range(slices):
        first, second = i+1, (i+1)%slices+1
        faces.extend(((0, first, second), (first, first+slices, second),
                      (second, first+slices, second+slices)))
    if sign > 0:
        faces = [tuple(reversed(f)) for f in faces]
    return MeshShape(np.array(points), np.array(faces), color=PACK, collide=False)


GLYPHS = {
    "A": ("01110","10001","10001","11111","10001","10001","10001"),
    "C": ("01111","10000","10000","10000","10000","10000","01111"),
    "D": ("11110","10001","10001","10001","10001","10001","11110"),
    "K": ("10001","10010","10100","11000","10100","10010","10001"),
    "L": ("10000","10000","10000","10000","10000","10000","11111"),
    "O": ("01110","10001","10001","10001","10001","10001","01110"),
    "P": ("11110","10001","10001","11110","10000","10000","10000"),
}


def wrapper_label(sign):
    points, faces = [], []
    for i, ch in enumerate("COLD PACK"):
        for row, columns in enumerate(GLYPHS.get(ch, ())):
            for col, bit in enumerate(columns):
                if bit == "0":
                    continue
                x = (6*i+col-26)*.00145
                z = .020-row*.00165
                start = len(points)
                for dx, dz in ((0,0),(.00135,0),(.00135,-.00155),(0,-.00155)):
                    points.append((-sign*(x+dx), sign*.0039, z+dz))
                faces.extend(((start,start+2,start+1), (start,start+3,start+2)))
    return MeshShape(np.array(points), np.array(faces), color=BLUE, collide=False)


def sidewall(name, sign):
    pos = CENTRE + [-sign*(T-.002), 0, 0]
    shape = [
        BoxShape((0, 0, 0), (.174, .004, .111), PACK),
        wrapper_face(sign),
        wrapper_label(sign),
        BoxShape((0, sign*.0040, .033), (.105, .00016, .006), BLUE, collide=False),
        BoxShape((0, sign*.0040, -.032), (.105, .00016, .004), BLUE, collide=False),
    ]
    inward = (0.0, .012) if sign > 0 else (-.012, 0.0)
    joint = Joint(tuple(pos), parent="pouch", kind="slide", axis="X", axial=(STIFFNESS, 5., .012),
                  slide_range=inward)
    return Body.build(name, pos, .009, shape, color=PACK, yaw=PACK_YAW, joint=joint, friction=(.85,.70),
                      contact_offset=.0002, solver_iterations=42, report_contacts=True)


OBJECTS = [retention_stand(), pouch(), sidewall(FACE_NAMES[0], -1), sidewall(FACE_NAMES[1], 1)]
DEBUG = os.getenv("COLD_PACK_DEBUG") == "1"
_state = {}


@on_reset
def reset():
    _state.clear()
    _state.update(joints={}, opened=False, damage=False, strokes=0, extremum=None,
                  direction=None, burst_step=None, step=0, temperature=22., value=0., display=None)


reset()


def _display(root, is_cool):
    if _state["display"] == is_cool:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    # A passive temperature indicator is fixed to the outside of each wall.
    for name, sign in zip(FACE_NAMES, (-1, 1)):
        path = f"{root}/{name}/body/temperature_indicator"
        indicator = UsdGeom.Cube.Get(stage, path)
        if not indicator:
            indicator = UsdGeom.Cube.Define(stage, path)
            indicator.CreateSizeAttr(1.)
            indicator.AddTranslateOp().Set(Gf.Vec3d(.051, sign*.0041, -.005))
            indicator.AddScaleOp().Set(Gf.Vec3f(.012, .00020, .028))
        color = (.08, .67, .87) if is_cool else (.22, .23, .24)
        indicator.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _state["display"] = is_cool


def _hand(base, side):
    robot = base.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    joint_pos = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(joint_pos, station.BASES[side])[0]


def _hand_clear(base, side, position, axes):
    local = axes.T @ (_hand(base, side) - position)
    excess = np.maximum(np.abs(local) - [WIDTH/2, T, HEIGHT/2], 0.)
    return np.linalg.norm(excess) >= .035


def _mix(position):
    peak = _state["extremum"]
    if peak is None:
        _state["extremum"] = position.copy()
        return
    delta = position - peak
    distance = np.linalg.norm(delta)
    old_dir = _state["direction"]
    projection = float(delta @ old_dir) if old_dir is not None else 0.
    if old_dir is not None and projection > 0.:
        _state["extremum"] = position.copy()
    elif distance >= STROKE_LENGTH and (old_dir is None or projection <= -.6 * STROKE_LENGTH):
        _state["strokes"] += 1
        _state["direction"] = delta / distance
        _state["extremum"] = position.copy()


@on_step
def step(env):
    base = getattr(env, "unwrapped", env)
    root = base.scene.env_prim_paths[0]
    body = base.scene["pouch"]
    pose = as_numpy(body.data.root_link_pose_w)[0]
    axis = rotation(pose[3:])
    wall_pos = [as_numpy(base.scene[name].data.root_link_pose_w)[0, :3] for name in FACE_NAMES]
    indentation = 2*(T-.002) - float((wall_pos[1]-wall_pos[0]) @ axis[:, 1])
    lowest = pose[2] - np.abs(axis[2]) @ [WIDTH/2, T, HEIGHT/2]
    lifted = lowest >= TOP+.0325
    release_load = .5 * STIFFNESS * max(indentation, 0.)
    if not _state["opened"] and lifted and indentation >= BURST_TRAVEL:
        _state["opened"] = True
        _state["burst_step"] = _state["step"]
    if indentation > .022:
        _state["damage"] = True
    if lifted and _state["opened"]:
        _mix(pose[:3])
    if _state["opened"]:
        mixed = min(_state["strokes"] / STROKES, 1.)
        seconds = (_state["step"] - _state["burst_step"]) / 20.
        _state["temperature"] = 22. - 17. * mixed * (1. - np.exp(-seconds/5.))
    cold = _state["temperature"] <= 14.
    _display(root, cold)

    score_value = 0.
    if _state["opened"]:
        score_value = .25
    if cold and _state["strokes"] >= STROKES:
        score_value = .75
        still = np.linalg.norm(as_numpy(body.data.root_link_vel_w)[0, :3]) < .015
        upright = axis[2,2] > .94 and abs(pose[2]-CENTRE[2]-.003) < .012
        in_tray = np.all(np.abs(pose[:2]-CENTRE[:2]) <= [.014,.016])
        clear = all(_hand_clear(base, side, pose[:3], axis) for side in ("left","right"))
        if still and upright and in_tray and clear:
            score_value = 1.
    _state["value"] = score_value if not _state["damage"] else 0.
    _state["step"] += 1
    if DEBUG and _state["step"] % 10 == 0:
        print(f"COLD {_state['step']:03d} score={_state['value']:.2f} released={_state['opened']} "
              f"damaged={_state['damage']} strokes={_state['strokes']} temp={_state['temperature']:.1f} "
              f"position={np.round(pose[:3]-[0,0,TOP],4)} tilt={axis[2,2]:.3f} force={release_load:.2f} "
              f"hands={np.round([_hand(base,s)-[0,0,TOP] for s in ('left','right')],4).tolist()}", flush=True)


def score(objects):
    return min(_state["value"], .75)


def success(objects):
    return _state["value"] == 1.


def demo():
    open_wide, hold_left, hold_right = -.034, 0.0, -.013
    left_y, right_y = CENTRE[1]+.105, CENTRE[1]-.061
    x = CENTRE[0]
    z = TOP+.125
    press = -.005
    if os.getenv("COLD_PACK_GENTLE") == "1":
        press = hold_right
    bimanual = {}
    for side, y, hold in (("left",left_y,hold_left), ("right",right_y,hold_right)):
        grip = press if side == "right" else hold
        path = [
            Waypoint(1.8,(x,y,z+.073),open_wide,"approach pouch"),
            Waypoint(1.1,(x,y,z),open_wide,"straddle the outer pouch"),
            Waypoint(.7,(x,y,z),hold,"hold the pack above the tray lip"),
            Waypoint(1.2,(x,y,z+.097),hold,"lift from the tray together"),
            Waypoint(.9,(x,y,z+.097),grip,"compress the filled inner pouch"),
            Waypoint(.5,(x,y,z+.097),hold,"ease off after the chamber collapses"),
        ]
        for count in range(9):
            shake = 0.08 * (-1 if count % 2 else 1)
            path.append(Waypoint(.49,(x+shake,y,z+.097),hold,"shake the full pouch in both hands"))
        path += [
            Waypoint(1.3,(x,y,z+.073),hold,"return to the tray"),
            Waypoint(1.1,(x,y,z),hold,"lower upright into the tray"),
            Waypoint(.7,(x,y,z),open_wide,"release pouch"),
            Waypoint(1.8,(x,y,z+.113),open_wide,"withdraw from the cooled pack"),
        ]
        for waypoint in path:
            waypoint.yaw = -PACK_YAW if side == "left" else PACK_YAW
        bimanual[side] = path
    return bimanual
