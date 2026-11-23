"""Trim a nail-art swatch in a real two-jaw tabletop clipper."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, CylShape, Joint, _author_joint, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
MOUTH = np.array([.34,.026,TOP+.032])
LEVER_PIN = MOUTH + [0, .010, .019]
CUT_AT = -.066
SWATCH0 = np.array([.230,.194,TOP+.013])
NIP = .0044
POLISH = (.88,.31,.38)
NAIL_BASE = (.82,.58,.51)
STEEL = (.46,.52,.59)
PLATE = (.22,.39,.52)


def nail_pieces(a, b, color):
    parts = []
    width = .011
    for i in range(7):
        x = width * ((i+.5)/7-.5)
        z = -.0008 * (x/(width/2))**2
        parts.append(BoxShape((x,(a+b)/2,z),(width/7+.00010,b-a,.0007),color))
    return parts


FIXTURE = Body.build("base", MOUTH, .35, [
    BoxShape((0,-.051,-.028), (.067,.133,.008), PLATE),
    BoxShape((0,-.017,-.008), (.019,.033,.016), STEEL),
    BoxShape((0,-.003,-.002),( .017,.007,.004), STEEL),
    CylShape((0,.011,.019), .0035,.024,STEEL,axis="X",collide=False),
    BoxShape((-.013,.011,.008),(.006,.014,.021),STEEL),
    BoxShape((.013,.011,.008),(.006,.014,.021),STEEL),
], fixed=True, color=STEEL, contact_offset=.00018)

SLIDE = Body.build("upper_jaw", MOUTH, .022, [
    BoxShape((0,-.003,NIP+.002),(.017,.007,.004), STEEL),
    BoxShape((0,-.011,.010),(.020,.016,.0035),STEEL),
], joint=Joint(tuple(MOUTH), parent="base", kind="slide", axis="Z",
               axial=(900., 5., NIP), slide_range=(-NIP-.0004,0.)),
   color=STEEL, contact_offset=.00015, report_contacts=True, solver_iterations=32)

LEVER = Body.build("paddle", LEVER_PIN, .041, [
    BoxShape((0,-.044,.002),(.019,.088,.004),STEEL),
    BoxShape((0,-.084,.005),(.042,.037,.009),(.26,.42,.55)),
    BoxShape((0,-.020,-.004),(.018,.008,.008),STEEL),
], joint=Joint(tuple(LEVER_PIN), parent="base", axis="X", limits=(-0.5,16.), stiffness=.12, damping=.025),
   color=STEEL, contact_offset=.0002, solver_iterations=32)

REST = Body.build("pickup_rest", (*SWATCH0[:2],TOP), .12, [
    BoxShape((0,.016,.003),(.055,.088,.006),PLATE),
    BoxShape((0,.006,.008),(.020,.068,.006),(.33,.47,.53)),
], fixed=True, contact_offset=.00018)

HOLDER = Body.build("swatch", SWATCH0, .024, [
    BoxShape((0,.020,0),(.023,.063,.008),NAIL_BASE),
    BoxShape((0,-.021,.001),(.012,.025,.006),NAIL_BASE),
    *nail_pieces(CUT_AT,-.026,POLISH),
    BoxShape((0,CUT_AT+.0008,.00055),(.011,.0009,.00025),(.92,.83,.59),collide=False),
], color=POLISH, contact_offset=.00014, friction=(.85,.70), report_contacts=True)

OFFCUT = Body.build("offcut", SWATCH0, .00045, nail_pieces(CUT_AT-.004,CUT_AT,POLISH),
    joint=Joint(tuple(SWATCH0+[0,CUT_AT,0]), parent="swatch", kind="fixed"),
    color=POLISH, gravity=True, contact_offset=.00012, report_contacts=True)

OBJECTS = [FIXTURE, SLIDE, LEVER, REST, HOLDER, OFFCUT]
_state = {}


@on_reset
def reset():
    if _state.get("released"):
        root, stage = _state["released"]
        path = f"{root}/offcut"
        _author_joint(stage, path, f"{path}/body", OFFCUT)
    _state.clear()
    _state.update(contact=None, joints={}, released=None, cut=False, spoiled=False, score=0., step=0)


reset()


def _arm(base, side):
    robot = base.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    q = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(q, station.BASES[side])[0]


def _load(base, root):
    from isaaclab_physx.physics import PhysxManager
    if _state["contact"] is None:
        view = PhysxManager.get_physics_sim_view()
        _state["contact"] = view.create_rigid_contact_view(
            [f"{root}/upper_jaw/body"],
            filter_patterns=[[f"{root}/swatch/body",f"{root}/offcut/body"]],max_contact_data_count=0,
        )
    return np.linalg.norm(_state["contact"].get_contact_force_matrix(dt=float(base.physics_dt)).numpy(),axis=-1).sum()


@on_step
def step(env):
    from isaaclab.sim.utils.stage import get_current_stage
    base = getattr(env,"unwrapped",env)
    root = base.scene.env_prim_paths[0]
    tag = as_numpy(base.scene["swatch"].data.root_link_pose_w)[0]
    scrap = as_numpy(base.scene["offcut"].data.root_link_pose_w)[0]
    axis = rotation(tag[3:])
    line = tag[:3] + axis @ [0,CUT_AT,0]
    depth = NIP - (MOUTH[2]-float(as_numpy(base.scene["upper_jaw"].data.root_link_pose_w)[0,2]))
    contact = _load(base,root)
    delta = line-MOUTH
    straight = axis[1,1] >= .984 and axis[2,2] >= .980
    placed = abs(delta[0])<=.0028 and abs(delta[1])<=.0015 and abs(delta[2])<=.0015 and straight
    if contact >= .24 and depth <= .0014 and not _state["cut"]:
        if placed:
            stage = get_current_stage()
            stage.RemovePrim(f"{root}/offcut/joint")
            _state["released"] = (root,stage)
            _state["cut"] = True
        else:
            _state["spoiled"] = True
    separated = _state["cut"] and np.linalg.norm(tag[:3]-scrap[:3]) > .030
    score_value = .25 if placed else 0.
    if separated:
        score_value = .75
        parked = (np.linalg.norm(tag[:2]-SWATCH0[:2]) <= .020
                  and abs(tag[2]-SWATCH0[2]) <= .008
                  and np.linalg.norm(as_numpy(base.scene["swatch"].data.root_link_vel_w)[0,:3]) <= .014)
        clear = all(np.linalg.norm(_arm(base,s)-tag[:3]) >= .068 for s in ("left","right"))
        if parked and clear:
            score_value = 1.
    _state["score"] = score_value if not _state["spoiled"] else 0.
    _state["step"] += 1
    if os.getenv("CLIP_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"CLIP {_state['step']:03d} score={_state['score']:.2f} line={np.round(delta*1000,1)} "
              f"gap={depth*1000:.2f} force={contact:.3f} straight={straight} cut={_state['cut']} "
              f"valid={not _state['spoiled']} pose={np.round(tag[:3]-[0,0,TOP],4)} "
              f"hands={np.round([_arm(base,s)-[0,0,TOP] for s in ('left','right')],4).tolist()}",
              flush=True)


def score(objects):
    return min(_state["score"],.75)


def success(objects):
    return _state["score"] == 1.


def demo():
    open_, grip = -.025, -.007
    yaw = -math.pi/2
    buy = SWATCH0 + [0,0,.008]
    align = MOUTH + [0,-CUT_AT-.0026,.008]
    if os.getenv("CLIP_OFF_LINE") == "1":
        align[1] -= .006
    left = [
        Waypoint(1.6,tuple(buy+[0,0,.065]),open_,"approach nail swatch",yaw),
        Waypoint(.9,tuple(buy),open_,"straddle swatch handle",yaw),
        Waypoint(.7,tuple(buy),grip,"grasp swatch handle",yaw),
        Waypoint(.9,tuple(buy+[0,0,.044]),grip,"lift swatch",yaw),
        Waypoint(1.1,tuple(align+[0,.030,.047]),grip,"line up the marked nail with the jaws",yaw),
        Waypoint(.9,tuple(align+[0,.030,0]),grip,"lower to the jaw opening",yaw),
        Waypoint(.8,tuple(align),grip,"feed to the trim mark",yaw),
        Waypoint(2.8,tuple(align),grip,"hold the nail while the lever closes",yaw),
        Waypoint(.65,tuple(align+[0,.040,0]),grip,"retract from the jaw",yaw),
        Waypoint(.72,tuple(align+[0,.040,.046]),grip,"lift clear of the jaw",yaw),
        Waypoint(1.,tuple(buy+[0,0,.046]),grip,"carry back to the rest",yaw),
        Waypoint(.8,tuple(buy),grip,"set the swatch down",yaw),
        Waypoint(.6,tuple(buy),open_,"release",yaw),
        Waypoint(1.4,tuple(buy+[0,0,.110]),open_,"clear the swatch",yaw),
    ]
    press = (MOUTH[0],LEVER_PIN[1]-.084,TOP+.035)
    high = (press[0],press[1],TOP+.145)
    right = [
        Waypoint(6.7,high,0.,"stay above the lever while the swatch approaches"),
        Waypoint(1.0,press,0.,"press the clipper lever once"),
        Waypoint(.65,press,0.,"complete the trim"),
        Waypoint(1.1,high,0.,"release the lever and lift clear"),
    ]
    return {"left":left,"right":right}
