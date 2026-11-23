"""Move a water-filled mug while leaving its condensation-stuck coaster flat on the table."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, CylShape, Joint, _author_joint, as_numpy, on_reset, on_step, plate, rotation, segment
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.
COASTER0 = np.array([.376,-.048,TOP+.005])
MUG0 = COASTER0+[0,.011,.004]
TRAY = np.array([.376,.075,TOP])
RADIUS = .040
DIR = np.array([.50,math.sqrt(3)/2,0.])
GRIP = DIR*.076 + [0,0,.055]
BLUE = (.19,.46,.62)
CERAMIC = (.75,.72,.56)
WATER = (.34,.65,.77)
WOOD = (.55,.39,.25)


def make_mug():
    mug = [
        CylShape((0,0,.003),.035,.006,BLUE),
        CylShape((0,0,.051),.0355,.0015,WATER,collide=False),
    ]
    for index in range(24):
        angle = index*math.tau/24
        n = np.array([math.cos(angle),math.sin(angle),0.])
        mug.append(plate(n*.033+[0,0,.006],n*RADIUS+[0,0,.084],.011,.003,n,BLUE))
        mug.append(BoxShape(tuple(n*RADIUS+[0,0,.083]),(.0045,.012,.0050),BLUE,yaw=angle))
    points = [DIR*r+[0,0,z] for r,z in (
        (.039,.068),(.057,.071),(.071,.068),(.077,.059),(.077,.044),
        (.070,.030),(.055,.028),(.039,.033),
    )]
    for a,b in zip(points,points[1:]):
        mug.append(segment(a,b,.011,BLUE))
    return Body.build("mug",MUG0,.29,mug,color=BLUE,
                      joint=Joint(tuple(MUG0),parent="coaster",kind="wet_film",axial=(155.,2.8,.022)),
                      contact_offset=.00026,friction=(.79,.65),solver_iterations=32)


COASTER = Body.build("coaster",COASTER0,.086,[
    BoxShape((0,0,-.0015),(.128,.128,.007),CERAMIC),
    BoxShape((0,0,.0020),(.122,.122,.002),(.83,.77,.59)),
    CylShape((0,.011,.0031),.031,.0005,(.49,.65,.65),collide=False),
    *[BoxShape((x,y,-.0045),(.019,.019,.001),( .32,.28,.21))
      for x in (-.047,.047) for y in (-.047,.047)],
],color=CERAMIC,contact_offset=.00019,friction=(.25,.20),solver_iterations=32)

TRAY_BODY = Body.build("tray",TRAY,.40,[
    BoxShape((0,0,.004),(.145,.116,.008),WOOD),
    BoxShape((0,0,.0082),(.140,.109,.00035),(.65,.48,.32),collide=False),
],fixed=True,contact_offset=.00019)

MUG = make_mug()
OBJECTS = [COASTER,MUG,TRAY_BODY]
_state = {}


@on_reset
def reset():
    if _state.get("detached"):
        root,stage = _state["detached"]
        path = f"{root}/mug"
        _author_joint(stage,path,f"{path}/body",MUG)
    _state.clear()
    _state.update(joints={}, step=0, score=0., detached=None, invalid=False, speed=None,
                  acceleration=np.zeros(3), motion_time=0)


reset()


def _hand(base,side):
    robot = base.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS,preserve_order=True)[0]
    q = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(q,station.BASES[side])[0]


@on_step
def evaluate(env):
    from isaaclab.sim.utils.stage import get_current_stage
    base = getattr(env,"unwrapped",env)
    root = base.scene.env_prim_paths[0]
    mug = as_numpy(base.scene["mug"].data.root_link_pose_w)[0]
    coaster = as_numpy(base.scene["coaster"].data.root_link_pose_w)[0]
    velocity = as_numpy(base.scene["mug"].data.root_link_vel_w)[0,:3]
    base_rotation = rotation(coaster[3:])
    cup_rotation = rotation(mug[3:])
    relative = base_rotation.T @ (mug[:3]-coaster[:3])-(MUG0-COASTER0)
    shifted = np.linalg.norm(relative[:2])
    coaster_move = coaster[:3]-COASTER0
    if coaster_move[2] > .004 or np.linalg.norm(coaster_move[:2]) > .009 or cup_rotation[2,2] < .9848:
        _state["motion_time"] += 1
        if _state["motion_time"] >= 2:
            _state["invalid"] = True
    else:
        _state["motion_time"] = 0
    if _state["speed"] is not None:
        accel = 20.*(velocity-_state["speed"])
        _state["acceleration"] = .72*_state["acceleration"] + .28*accel
    _state["speed"] = velocity.copy()
    apparent_gravity = cup_rotation.T @ ([0,0,9.81]+_state["acceleration"])
    freeboard = .032
    if apparent_gravity[2] < 1. or np.linalg.norm(apparent_gravity[:2])*.037 > freeboard*apparent_gravity[2]:
        _state["invalid"] = True
    if _state["detached"] is None and shifted >= .015 and abs(relative[2]) <= .002 and not _state["invalid"]:
        stage = get_current_stage()
        stage.RemovePrim(f"{root}/mug/joint")
        _state["detached"] = (root,stage)
    score_value = .25 if shifted >= .007 else 0.
    if _state["detached"] is not None and np.linalg.norm(mug[:2]-coaster[:2]) >= .107:
        score_value = .75
        parked = np.linalg.norm(mug[:2]-TRAY[:2]) <= .012 and abs(mug[2]-(TOP+.008)) <= .004
        still = np.linalg.norm(velocity) <= .010
        handle = mug[:3]+cup_rotation @ GRIP
        clear = True
        for side in ("left","right"):
            hand = _hand(base,side)
            clear &= (np.linalg.norm(hand-mug[:3]) > .095
                      and np.linalg.norm(hand-handle) >= .048
                      and np.linalg.norm(hand-coaster[:3]) >= .095)
        if parked and still and clear:
            score_value = 1.
    _state["score"] = score_value if not _state["invalid"] else 0.
    _state["step"] += 1
    if os.getenv("WET_DEBUG") == "1" and _state["step"]%5 == 0:
        print(f"WET {_state['step']:03d} score={_state['score']} coupled={not bool(_state['detached'])} "
              f"rel={np.round(relative,4)} mug={np.round(mug[:3]-[0,0,TOP],4)} "
              f"coaster={np.round(coaster_move,4)} spill={_state['invalid']} "
              f"hands={np.round([_hand(base,s)-[0,0,TOP] for s in ('left','right')],4).tolist()}",flush=True)


def score(objects):
    return min(_state["score"],.75)


def success(objects):
    return _state["score"] == 1.


def demo():
    open_, close = -.025,-.003
    yaw = -2*math.pi/3
    grasp = MUG0+GRIP+[0,0,.007]
    shear = 0 if os.getenv("WET_LIFT_ONLY")=="1" else .044
    freed = grasp + [0,shear,0]
    landing = TRAY+[0,0,.008]+GRIP+[0,.004,.007]
    left = [
        Waypoint(1.6,tuple(grasp+[0,0,.073]),open_,"approach the mug handle",yaw),
        Waypoint(.85,tuple(grasp),open_,"straddle the outside of the handle",yaw),
        Waypoint(.70,tuple(grasp),close,"grasp the mug handle",yaw),
        Waypoint(1.25,tuple(freed),close,"slide the mug sideways to break the wet film",yaw),
        Waypoint(1.9,tuple(landing),close,"slide the level mug onto the adjacent wooden pad",yaw),
        Waypoint(.75,tuple(landing),close,"settle the mug on the pad",yaw),
        Waypoint(.70,tuple(landing),open_,"release the mug",yaw),
        Waypoint(1.35,tuple(landing+[0,0,.115]),open_,"withdraw without rocking the mug",yaw),
    ]
    brace = COASTER0 + [-.047,-.052,-.001]
    right = [
        Waypoint(1.6,tuple(brace+[0,0,.075]),0.,"approach the exposed corner of the coaster",0.),
        Waypoint(.95,tuple(brace),0.,"keep the coaster on the table",0.),
        Waypoint(3.5,tuple(brace),0.,"hold the coaster while the mug is released",0.),
        Waypoint(1.35,tuple(brace+[0,-.035,.117]),0.,"lift off the coaster",0.),
    ]
    if os.getenv("WET_LIFT_ONLY") == "1":
        left[4:6] = [Waypoint(1.8,tuple(grasp+[0,0,.065]),close,"try lifting without shearing",yaw)]
    return {"left":left,"right":right}
