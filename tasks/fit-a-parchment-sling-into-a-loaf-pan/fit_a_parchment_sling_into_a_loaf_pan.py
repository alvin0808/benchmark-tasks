"""Press a flat baking-parchment sling into a loaf pan and leave both lift tabs accessible."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, Joint, as_numpy, on_reset, on_step, plate, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.
PAN = np.array([.369,0.,TOP])
FLOOR_HEIGHT = .006
RIM_HEIGHT = .059
START = PAN+[0,0,RIM_HEIGHT+.0012]
HALF_WIDTH = .047
HALF_LENGTH = .096
PANEL_LENGTH = .088
GREY = (.29,.34,.41)
LIP = (.47,.52,.57)
PAPER = (.83,.71,.52)


def make_pan():
    parts = [BoxShape((0,0,.003),(.101,.201,.006),GREY)]
    for sign in (-1,1):
        parts += [
            plate((sign*.050,0,.004),(sign*.060,0,.057),.222,.0025,(sign,0,0),GREY),
            plate((0,sign*.100,.004),(0,sign*.111,.057),.121,.0025,(0,sign,0),GREY),
            BoxShape((sign*.061,0,.057),(.006,.225,.004),LIP),
            BoxShape((0,sign*.112,.057),(.124,.005,.004),LIP),
        ]
    return Body.build("loaf_pan",PAN,.38,parts, fixed=True,contact_offset=.00018, solver_iterations=32)


def make_sling():
    thickness = .0006
    floor = Body.build("parchment",START,.006,[
        BoxShape((0,0,0),(HALF_WIDTH*2,HALF_LENGTH*2,thickness),PAPER),
    ],color=PAPER,contact_offset=.00012,friction=(.38,.30),solver_iterations=32)
    ends = []
    for sign,name in ((1,"left_end"),(-1,"right_end")):
        hinge = START+[0,sign*HALF_LENGTH,0]
        travel = (-1.,94.) if sign==1 else (-94.,1.)
        parts = [
            BoxShape((0,sign*PANEL_LENGTH/2,0),(HALF_WIDTH*2,PANEL_LENGTH,thickness),PAPER),
            BoxShape((0,sign*(PANEL_LENGTH-.007),.00032),(.080,.00045,.00006),(.72,.62,.44),collide=False),
        ]
        ends.append(Body.build(name,hinge,.0018,parts,color=PAPER,contact_offset=.00013,
                               joint=Joint(tuple(hinge),parent="parchment",axis="X",
                                           limits=travel,stiffness=.0015,damping=.004),
                               friction=(.40,.32),solver_iterations=32))
    return [floor,*ends]


OBJECTS = [make_pan(),*make_sling()]
_state = {}


@on_reset
def reset():
    _state.clear()
    _state.update(score=0.,step=0,joints={})


reset()


def _hand(base,side):
    robot = base.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS,preserve_order=True)[0]
    q = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(q,station.BASES[side])[0]


@on_step
def evaluate(env):
    base = getattr(env,"unwrapped",env)
    center = as_numpy(base.scene["parchment"].data.root_link_pose_w)[0]
    mat = rotation(center[3:])
    rel = center[:3]-PAN
    flat = (abs(rel[0]) <= .006 and abs(rel[1]) <= .006 and FLOOR_HEIGHT-.001 <= rel[2] <= FLOOR_HEIGHT+.005
            and mat[2,2] >= .998 and abs(mat[0,0]) >= .998)
    blocked = False
    located = []
    landing = []
    for sign,name in ((1,"left_end"),(-1,"right_end")):
        end = as_numpy(base.scene[name].data.root_link_pose_w)[0]
        axis = rotation(end[3:]) @ [0,sign,0]
        free_end = end[:3]-PAN + axis*PANEL_LENGTH
        lower_wall = end[:3]-PAN + axis*.038
        located.append(free_end)
        landing.append(axis[2] >= .90 and .095 <= sign * lower_wall[1] <= .116
                       and RIM_HEIGHT+.018 <= free_end[2] <= RIM_HEIGHT+.040
                       and .100 <= sign*free_end[1] <= .137)
        blocked |= free_end[2] < rel[2]-.007
    score_value = 0.
    if flat and not blocked:
        score_value = .25 + .25 * sum(landing)
    clear = all(_hand(base,side)[2] >= TOP+RIM_HEIGHT+.052 for side in ("left","right"))
    still = np.linalg.norm(as_numpy(base.scene["parchment"].data.root_link_vel_w)[0,:3]) <= .007
    _state["score"] = 1. if score_value == .75 and clear and still else score_value
    _state["step"] += 1
    if os.getenv("PARCHMENT_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"PARCHMENT {_state['step']:03d} score={_state['score']} center={np.round(rel,4)} "
              f"flat={mat[2,2]:.5f} ends={np.round(located,4).tolist()} walls={landing} "
              f"hands={np.round([_hand(base,s)-PAN for s in ('left','right')],4).tolist()}",flush=True)


def score(objects):
    return min(_state["score"],.75)


def success(objects):
    return _state["score"] == 1.


def demo():
    result = {}
    shallow = os.getenv("PARCHMENT_SHALLOW") == "1"
    for side,sgn in (("left",1),("right",-1)):
        yaw = -sgn*math.pi/2
        entry = PAN+[0,sgn*.079,RIM_HEIGHT+.009]
        low = PAN+[0,sgn*.084,.065 if shallow else .011]
        result[side] = [
            Waypoint(1.65,tuple(entry+[0,0,.080]),0.,"approach one end of the liner",yaw),
            Waypoint(.85,tuple(entry),0.,"contact the sheet inside the rim",yaw),
            Waypoint(1.9,tuple(low),0.,"press the parchment into the bottom corners",yaw),
            Waypoint(.85,tuple(low+[0,sgn*.004,0]),0.,"work each bottom crease up to the wall",yaw),
            Waypoint(.7,tuple(low+[0,sgn*.004,0]),0.,"allow the folds to settle",yaw),
            Waypoint(1.1,tuple(entry+[0,0,.095]),0.,"lift off without dragging the paper",yaw),
        ]
    return result
