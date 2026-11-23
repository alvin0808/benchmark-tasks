"""Remove air from a reusable zipper food bag through its small one-way valve."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, CylShape, Joint, allow_matrix_grips, as_numpy, on_reset, on_step, plate, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, axis_angle_matrix, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.
BAG = np.array([.374,.022,TOP])
PORT = BAG + [0,-.083,.011]
PARK = np.array([.217,-.160,TOP+.001])
INITIAL_PUMP = PORT+[0,0,.0009]
CAP_Z = .128
EXTENSION = .057
BODY_GRIP = .041
CAP_WIDTH = .040
CAP_OFFSET = .058
ZIP = (.16,.46,.69)
BAG_FILM = (.68,.80,.79,.52)
CLEAR_SHEET = (.74,.86,.85,.26)
SEAM = (.42,.59,.59,.76)
FILM_DARK = (.34,.53,.50)
PUMP_PALE = (.87,.88,.83)
PUMP_DARK = (.16,.35,.47)
GASKET = (.16,.20,.21)


def make_bag():
    card = []
    for i in range(6):
        card.append(BoxShape((-.048+i*.019,.046,.007),(.016,.057,.010),(.82,.59,.33)))
    lower = Body.build("bag", BAG, .19, [
        BoxShape((0,0,.0018),(.218,.220,.003),BAG_FILM),
        BoxShape((0,.007,.0127),(.207,.181,.00065),CLEAR_SHEET,collide=False),
        BoxShape((-.106,0,.004),(.004,.208,.002),SEAM,collide=False),
        BoxShape((.106,0,.004),(.004,.208,.002),SEAM,collide=False),
        BoxShape((0,-.105,.004),(.211,.003,.002),SEAM,collide=False),
        BoxShape((0,.103,.005),(.205,.006,.005),ZIP),
        BoxShape((0,.109,.006),(.205,.005,.006),ZIP),
        CylShape((0,-.083,.006),.020,.010,ZIP),
        CylShape((0,-.083,.011),.014,.0015,GASKET),
        *card,
        CylShape((0,-.083,.0165),.0115,.013,ZIP),
    ], color=BAG_FILM[:3], contact_offset=.00024, friction=(.85,.76))
    initial = BAG + [0,0,.044]
    upper = Body.build("membrane", initial, .017, [
        BoxShape((0,.017,0),(.060,.060,.0018),BAG_FILM),
        plate((-.030,.010,0),(-.041,.010,-.014),.064,.0015,(-1,0,0),BAG_FILM),
        plate((.030,.010,0),(.041,.010,-.014),.064,.0015,(1,0,0),BAG_FILM),
        plate((0,-.013,0),(0,-.027,-.014),.064,.0015,(0,-1,0),CLEAR_SHEET,collide=False),
        plate((0,.047,0),(0,.064,-.014),.064,.0015,(0,1,0),CLEAR_SHEET,collide=False),
        BoxShape((0,.025,.00105),(.051,.0017,.0003),FILM_DARK,collide=False),
        BoxShape((0,.037,.00105),(.051,.0017,.0003),FILM_DARK,collide=False),
    ], color=BAG_FILM[:3], joint=Joint(tuple(initial),parent="bag",kind="slide",axis="Z",
                                   slide_range=(-.031,0),axial=(390.,7.,0.)),
        contact_offset=.00019,solver_iterations=32)
    return lower,upper


def make_pump():
    foot = []
    for index in range(16):
        angle = index*math.tau/16
        n = np.array([math.cos(angle),math.sin(angle),0.])
        center = n*.0150+[0,0,.008]
        foot.append(BoxShape(tuple(center),(.0053,.0064,.016),GASKET,yaw=angle))
    pump = Body.build("pump", INITIAL_PUMP, .090, [
        *foot,
        CylShape((0,0,.0270),.025,.008,PUMP_DARK),
        CylShape((0,0,.065),.0155,.086,PUMP_PALE),
        CylShape((0,0,.105),.017,.011,PUMP_DARK),
        BoxShape((0,0,BODY_GRIP),(.031,.029,.043),PUMP_DARK),
    ], color=PUMP_PALE,friction=(.93,.81),contact_offset=.00025,
       report_contacts=True,solver_iterations=32)
    cap = Body.build("plunger",INITIAL_PUMP,.026, [
        CylShape((0,0,.118),.0055,.070,PUMP_PALE),
        BoxShape((0,.014,CAP_Z),(.026,.081,.012),PUMP_DARK),
        BoxShape((0,CAP_OFFSET,CAP_Z),(.034,.027,.019),PUMP_DARK),
    ], joint=Joint(tuple(INITIAL_PUMP),parent="pump",kind="slide",axis="Z",
                   slide_range=(0.,EXTENSION),axial=(32.,3.2,EXTENSION)),
       contact_offset=.00022,friction=(.94,.83),solver_iterations=32)
    return pump,cap


OBJECTS = [*make_bag(),*make_pump()]
_state = {}


@on_reset
def reset():
    if _state.get("drive") is not None:
        _state["drive"].Set(0.)
    _state.clear()
    _state.update(strokes=0,stroke_open=False,ready=False,load=0.,drive=None,score=0.,step=0,joints={})


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
    from isaaclab_physx.physics import PhysxManager
    from pxr import UsdPhysics
    base = getattr(env,"unwrapped",env)
    root = base.scene.env_prim_paths[0]
    bag = as_numpy(base.scene["bag"].data.root_link_pose_w)[0]
    pump = as_numpy(base.scene["pump"].data.root_link_pose_w)[0]
    cap = as_numpy(base.scene["plunger"].data.root_link_pose_w)[0]
    membrane = as_numpy(base.scene["membrane"].data.root_link_pose_w)[0,:3]
    bag_axis = rotation(bag[3:])
    pump_axis = rotation(pump[3:])
    valve = bag[:3] + bag_axis @ (PORT-BAG)
    error = pump[:3]-valve
    travel = np.dot(cap[:3]-pump[:3],pump_axis[:,2])
    sealed = abs(error[2]) <= .005 and np.linalg.norm(error[:2]) <= .006 and pump_axis[2,2] >= .975
    if _state["drive"] is None:
        stage = get_current_stage()
        _state["drive"] = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{root}/membrane/joint"),
                                                  "transZ").GetTargetPositionAttr()
        view = PhysxManager.get_physics_sim_view()
        _state["contact"] = view.create_rigid_contact_view([f"{root}/pump/body"],
                             filter_patterns=[[f"{root}/bag/body"]],max_contact_data_count=0)
    force = _state["contact"].get_contact_force_matrix(dt=float(base.physics_dt)).numpy()
    load = abs(float(force[...,2].sum()))
    _state["load"] = .65*_state["load"] + .35*load
    seal_loaded = sealed and _state["load"] >= .38
    if not seal_loaded:
        _state["stroke_open"] = False
        _state["ready"] = False
    elif travel <= .013:
        if _state["ready"]:
            _state["strokes"] += 1
            _state["drive"].Set(-min(.029,_state["strokes"]*.0097))
        _state["stroke_open"] = True
        _state["ready"] = False
    elif _state["stroke_open"] and travel >= .045:
        _state["ready"] = True
    height = np.dot(membrane-bag[:3],bag_axis[:,2])
    score_value = 0.
    for threshold,required in ((.038,1),(.029,2),(.018,3)):
        if height <= threshold and _state["strokes"] >= required:
            score_value = .25 * required
    parked = np.linalg.norm(pump[:2]-PARK[:2]) <= .037 and abs(pump[2]-TOP) <= .007 and pump_axis[2,2] >= .985
    pump_axis_z = pump_axis[:,2]
    handle = cap[:3] + pump_axis @ [0,CAP_OFFSET,CAP_Z]
    hands_clear = True
    for side in ("left","right"):
        hand = _hand(base,side)
        from_base = hand-pump[:3]
        along = np.clip(np.dot(from_base,pump_axis_z),0.,CAP_Z)
        shaft_clearance = np.linalg.norm(from_base - along*pump_axis_z)
        hands_clear &= (shaft_clearance >= .036 and np.linalg.norm(hand-handle) >= .040
                        and np.linalg.norm(hand-membrane) >= .080)
    flat = bag_axis[2,2] >= .994 and bag[2]-TOP <= .005
    still = np.linalg.norm(as_numpy(base.scene["pump"].data.root_link_vel_w)[0,:3]) <= .014
    _state["score"] = 1. if score_value == .75 and parked and still and hands_clear and flat else score_value
    _state["step"] += 1
    if os.getenv("VAC_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"VAC {_state['step']:03d} score={_state['score']} strokes={_state['strokes']} stage={_state['ready']} "
              f"port_err={np.round(error,4)} contact={load:.2f} stroke={travel:.4f} height={height:.4f} "
              f"pump={np.round(pump[:3]-[0,0,TOP],4)} cap={np.round(cap[:3]+pump_axis@[0,CAP_OFFSET,CAP_Z]-[0,0,TOP],4)} "
              f"hands={np.round([_hand(base,s)-[0,0,TOP] for s in ('left','right')],4).tolist()}",flush=True)


def score(objects):
    return min(_state["score"],.75)


def success(objects):
    return _state["score"] == 1.


def demo():
    open_, close = -.025,-.006
    allow_matrix_grips()
    yaw_r = axis_angle_matrix(np.array([1.,0.,0.]),math.radians(82.)) @ grip_rotation(math.pi/2)
    yaw_l = axis_angle_matrix(np.array([1.,0.,0.]),math.radians(-31.)) @ grip_rotation(-math.pi/2)
    dock = PORT
    pickup = PARK + [0,-.004,BODY_GRIP+.009]
    setdown = dock + [0,-.004,BODY_GRIP+.005]
    side_clear = setdown+[0,-.075,0]
    brace = os.getenv("VAC_NO_BRACE") != "1"
    hold = setdown+[0,-.002,-.012] if brace else side_clear
    right = [
        Waypoint(1.6,tuple(side_clear+[0,0,.080]),open_,"approach the side of the pump",yaw_r),
        Waypoint(1.05,tuple(side_clear),open_,"lower to the pump body",yaw_r),
        Waypoint(1.1,tuple(setdown),open_,"straddle the pump body below the handle",yaw_r),
        Waypoint(.7,tuple(setdown if brace else side_clear),close if brace else open_,"hold the pump square against the valve",yaw_r),
        Waypoint(.65,tuple(hold),close if brace else open_,"preload the rubber gasket",yaw_r),
        Waypoint(10.75,tuple(hold),close if brace else open_,"maintain the gasket seal while pumping",yaw_r),
        Waypoint(.85,tuple(setdown+[0,0,.060]),close,"lift the pump clear of the valve",yaw_r),
        Waypoint(.92,tuple(pickup+[0,0,.061]),close,"carry it back to the starting spot",yaw_r),
        Waypoint(.8,tuple(pickup),close,"set its base down",yaw_r),
        Waypoint(.55,tuple(pickup),open_,"release the pump body",yaw_r),
        Waypoint(1.25,tuple(pickup+[0,-.045,.126]),open_,"move the right hand clear",yaw_r),
    ]
    cap_low = dock + [-.003,CAP_OFFSET+.009,CAP_Z+.010]
    cap_high = cap_low+[0,0,.066]
    cap_approach = cap_low+[0,0,.095]
    left = [
        Waypoint(4.35,tuple(cap_approach),open_,"wait on the clear side of the bag",yaw_l),
        Waypoint(.75,tuple(cap_low+[0,0,.050]),open_,"line up the jaw with the end of the handle",yaw_l),
        Waypoint(1.05,tuple(cap_low),open_,"straddle the plunger grip",yaw_l),
        Waypoint(.65,tuple(cap_low),-.004,"grip the plunger",yaw_l),
    ]
    for _ in range(4):
        left.extend([
            Waypoint(.95,tuple(cap_high),-.004,"draw one stroke without breaking the gasket seal",yaw_l),
            Waypoint(.95,tuple(cap_low),-.004,"push the plunger back down",yaw_l),
        ])
    left.extend([
        Waypoint(.6,tuple(cap_low),open_,"release the plunger",yaw_l),
        Waypoint(1.0,tuple(cap_approach),open_,"withdraw from the pump and valve",yaw_l),
    ])
    return {"left":left,"right":right}
