"""Heat copper braid on a solder pad and remove the iron and braid together."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, CylShape, Link, Linkage, allow_matrix_grips, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
allow_matrix_grips()

PAD = np.array([0.322, 0.013, TOP+0.0144])
IRON = np.array([0.322, -0.169, TOP+0.026])
WICK = np.array([0.322, 0.119, TOP+0.01475])
TIP = np.array([0.0, 0.110, -0.0010])
END = np.array([0.0, -0.111, 0.0])
MOSS = (0.11, 0.36, 0.27)
MAT = (0.21, 0.34, 0.38)
COPPER = (0.72, 0.35, 0.14)
SILVER = (0.66, 0.69, 0.69)
BLACK = (0.10, 0.118, 0.13)
YELLOW = (0.84, 0.62, 0.11)
OFF_WHITE = (0.86, 0.86, 0.77)
HEAT_SECONDS = 1.30


def _rx(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]])


board_parts = [
    BoxShape((0,0,0.005),(0.141,0.166,0.010),MAT),
    BoxShape((0,0.006,0.011),(0.096,0.117,0.002),MOSS),
    BoxShape((0,0,0.0122),(0.010,0.009,0.0004),COPPER),
    BoxShape((0.036,-0.039,0.018),(0.017,0.015,0.012),BLACK),
    BoxShape((-0.038,0.039,0.017),(0.012,0.020,0.009),BLACK),
    BoxShape((0.028,0.036,0.021),(0.018,0.019,0.016),SILVER),
    BoxShape((-0.009,0,0.01208),(0.016,0.0008,0.00016),COPPER,collide=False),
    BoxShape((0.002,0.016,0.01208),(0.00085,0.026,0.00016),COPPER,collide=False),
    BoxShape((0.014,-0.010,0.01208),(0.023,0.0008,0.00016),COPPER,collide=False),
    BoxShape((0.0,-0.0050,0.01208),(0.010,0.0007,0.00016),OFF_WHITE,collide=False),
]
for x,y in ((-0.026,0.037),(-0.019,0.037),(0.011,-0.036),(0.027,-0.036),(0.015,-0.004)):
    board_parts.append(BoxShape((x,y,0.014),(0.0038,0.0044,0.004),SILVER))
OBJECTS = [
    Body.build("board",(PAD[0],PAD[1],TOP),0.49,board_parts,fixed=True,color=MOSS,
               contact_offset=0.00021),
    Body.build("solder",(PAD[0],PAD[1],TOP),0.008,[
        BoxShape((0,0,0.01335),(0.008,0.0064,0.0021),SILVER),
    ],fixed=True,color=SILVER,contact_offset=0.00020),
    Body.build("iron_rest",(IRON[0],IRON[1],TOP),0.16,[
        BoxShape((0,0,0.006),(0.082,0.110,0.012),BLACK),
        BoxShape((0,-0.034,0.012),(0.033,0.015,0.006),SILVER),
        BoxShape((0,0.034,0.012),(0.033,0.015,0.006),SILVER),
        *[BoxShape((x,y,0.017),(0.005,0.015,0.014),SILVER)
          for x in (-0.016,0.016) for y in (-0.034,0.034)],
    ],fixed=True,color=BLACK,contact_offset=0.00028),
    Body.build("iron",tuple(IRON),0.074,[
        CylShape((0,-0.012,0),0.0115,0.082,BLACK,axis="Y"),
        CylShape((0,-0.044,0),0.0117,0.017,YELLOW,axis="Y"),
        CylShape((0,0.028,0),0.012,0.010,YELLOW,axis="Y"),
        CylShape((0,0.061,0),0.0045,0.061,SILVER,axis="Y"),
        CylShape((0,0.096,0),0.0030,0.016,SILVER,axis="Y"),
        BoxShape((0,0.1065,-0.0008),(0.0032,0.009,0.0024),SILVER),
    ],color=BLACK,contact_offset=0.00023,friction=(0.88,0.70),solver_iterations=40,report_contacts=True),
    Body.build("mat_extension",(WICK[0],WICK[1],TOP),0.07,[
        BoxShape((0,0,0.00555),(0.070,0.066,0.0111),MAT),
        *[BoxShape((sign*0.0295,0,0.013),(0.0045,0.016,0.0038),MAT)
          for sign in (-1,1)],
        *[BoxShape((side*0.018,sign*0.0248,0.013),(0.005,0.004,0.0038),MAT)
          for side in (-1,1) for sign in (-1,1)],
    ],fixed=True,color=MAT,contact_offset=0.00025),
    Linkage.build("wick",tuple(WICK),0.027,[
        CylShape((0,0,0),0.026,0.007,YELLOW),
        CylShape((0,0,0.0036),0.012,0.0006,BLACK),
        CylShape((0,0,-0.0036),0.0257,0.0007,BLACK),
    ],[
        Link("braid", (
            BoxShape((0,-0.067,0.0),(0.0056,0.088,0.0007),COPPER),
            *[BoxShape((0,-0.034-i*0.005,0.00037),(0.0050,0.00025,0.00015),
                        (0.36,0.17,0.07),0.65,False) for i in range(15)],
        ),0.002,tuple(WICK+[0,-0.022,0]),axis="X",limits=(-24,24),stiffness=0.085,
             spring_damping=0.008,report_contacts=True),
    ],floating=True,color=YELLOW,contact_offset=0.00020,friction=(0.83,0.67),solver_iterations=40),
]

_state = {}


@on_reset
def _reset():
    _state.clear()
    _state.update(view=None, hands={}, heat=0.0, wet=False, removed=False, damaged=False,
                  started_lift=None, cooled=0, step=0, score=0.0, appearance=False)


_reset()


def _force(base):
    from isaaclab_physx.physics import PhysxManager

    if _state["view"] is None:
        sim = PhysxManager.get_physics_sim_view()
        e = base.scene.env_prim_paths[0]
        _state["view"] = sim.create_rigid_contact_view(
            [f"{e}/wick/braid", f"{e}/iron/body"],
            filter_patterns=[[f"{e}/iron/body", f"{e}/solder/body", f"{e}/board/body"],
                             [f"{e}/wick/braid", f"{e}/solder/body", f"{e}/board/body"]],
            max_contact_data_count=96)
    f = _state["view"].get_contact_force_matrix(dt=float(base.physics_dt)).numpy()
    return np.linalg.norm(f.reshape(2,3,3),axis=-1)


def _display_absorption(base):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    e = base.scene.env_prim_paths[0]
    solder = UsdGeom.Imageable(stage.GetPrimAtPath(f"{e}/solder/body/s0"))
    solder.MakeInvisible()
    stain = UsdGeom.Cube.Define(stage,f"{e}/wick/braid/SolderStain")
    stain.CreateSizeAttr(1.0)
    stain.AddTranslateOp().Set(Gf.Vec3d(0.0,-0.105,0.00046))
    stain.AddScaleOp().Set(Gf.Vec3f(0.0054,0.0095,0.00025))
    stain.CreateDisplayColorAttr([Gf.Vec3f(*SILVER)])


def _clear(base, centers):
    for side, anchor in station.BASES.items():
        arm = base.scene[f"{side}_robot"]
        if side not in _state["hands"]:
            _state["hands"][side] = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q = as_numpy(arm.data.joint_pos)[0][_state["hands"][side]]
        hand,_ = ik.tcp_pose(q,anchor)
        if min(np.linalg.norm(hand-p) for p in centers) <= 0.069:
            return False
    return True


@on_step
def _step(env):
    base = getattr(env,"unwrapped",env)
    iron, wick = base.scene["iron"], base.scene["wick"]
    p = as_numpy(iron.data.root_link_pose_w)[0]
    bp = as_numpy(wick.data.body_link_pose_w)[0,wick.body_names.index("braid")]
    wp = as_numpy(wick.data.root_link_pose_w)[0]
    tip = p[:3] + rotation(p[3:7]) @ TIP
    end = bp[:3] + rotation(bp[3:7]) @ END
    f = _force(base)
    near = np.linalg.norm(tip[:2]-PAD[:2]) <= 0.009
    over_braid = near and f[0,0] >= 0.070 and f[0,1] >= 0.040
    braid_on_pad = f[0,1] >= 0.018
    if f[1,1] > 0.08 or f[1,2] > 0.08:
        _state["damaged"] = True
    if over_braid:
        _state["heat"] = min(2.8, _state["heat"] + 0.05)
    else:
        _state["heat"] = max(0.0, _state["heat"] - 0.083)
    if _state["heat"] >= HEAT_SECONDS:
        _state["wet"] = True
        if not _state["appearance"]:
            _display_absorption(base)
            _state["appearance"] = True
    if _state["wet"] and not _state["removed"]:
        # A thin solder bridge can still wet the braid when the force sample
        # has dropped to zero. Test the short gap where that film can persist.
        in_solder_film = (np.linalg.norm(end[:2]-PAD[:2]) <= 0.008
                          and abs(end[2]-PAD[2]) <= 0.0016)
        if (braid_on_pad or in_solder_film) and f[0,0] < 0.025:
            _state["cooled"] += 1
        else:
            _state["cooled"] = 0
        if _state["cooled"] >= 5:
            _state["damaged"] = True
        if min(end[2],tip[2]) >= PAD[2] + 0.0065 and not braid_on_pad:
            _state["removed"] = True

    score = 0.25 if _state["wet"] else 0.0
    if _state["removed"]:
        score = 0.6
        iron_parked = np.linalg.norm(p[:2]-IRON[:2]) <= 0.029 and abs(p[2]-IRON[2]) <= 0.010
        wick_path = np.linspace(wp[:2], end[:2], 9)
        over_mat = np.any(np.all(np.abs(wick_path - PAD[:2]) <= [0.071,0.084], axis=1))
        over_tray = np.any(np.all(np.abs(wick_path - WICK[:2]) <= [0.036,0.034], axis=1))
        spool_parked = abs(wp[2]-TOP) <= 0.007 and not (over_mat or over_tray)
        istill = np.linalg.norm(as_numpy(iron.data.root_link_vel_w)[0,:3]) < 0.021
        wstill = np.linalg.norm(as_numpy(wick.data.root_link_vel_w)[0,:3]) < 0.021
        if iron_parked and spool_parked and istill and wstill and _clear(base,(p[:3],wp[:3])):
            score = 1.0
    _state["score"] = score if not _state["damaged"] else 0.0
    _state["step"] += 1
    if os.getenv("WICK_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"WICK {_state['step']:03} force={np.round(f[0],2)}/{np.round(f[1],2)} "
              f"heat={_state['heat']:.2f} wet={_state['wet']} removed={_state['removed']} "
              f"damage={_state['damaged']} score={_state['score']:.2f} "
              f"tip={np.round(tip-[0,0,TOP],4)} braid={np.round(end-[0,0,TOP],4)} "
              f"i={np.round(p[:3]-[0,0,TOP],4)} w={np.round(wp[:3]-[0,0,TOP],4)}",flush=True)


def score(objects):
    return min(_state["score"],0.6)


def success(objects):
    return _state["score"] == 1.0


def demo():
    ix = IRON[0]
    iy = IRON[1]
    wx = WICK[0]
    wy = WICK[1]
    iron_grip = -0.004
    spool_grip = -0.0185
    iron_yaw = math.pi/2
    tilt = math.radians(-25)
    rot = _rx(tilt) @ grip_rotation(iron_yaw)
    ir_home = np.array([ix,iy,TOP+0.032])
    wick_home = np.array([wx,wy,TOP+0.0190])
    tip_at = PAD - _rx(tilt) @ TIP
    spool_at = wick_home
    spool_park = np.array([wx,0.274,TOP+0.0060])
    late = os.getenv("WICK_LATE") == "1"
    right = [
        Waypoint(2.2,tuple(ir_home+[0,0,0.066]),-0.029,"approach the iron's insulated handle",iron_yaw),
        Waypoint(0.9,tuple(ir_home),-0.029,"reach the handle",iron_yaw),
        Waypoint(0.65,tuple(ir_home),iron_grip,"grip the handle",iron_yaw),
        Waypoint(0.9,tuple(ir_home+[0,0,0.083]),iron_grip,"lift out of the rest",iron_yaw),
        Waypoint(1.45,tuple(tip_at+[0,0,0.044]),iron_grip,"position the chisel tip above the wick",rot),
        Waypoint(0.9,tuple(tip_at+[0,0,-0.0105]),iron_grip,"press the tip onto the braid",rot),
        Waypoint(2.33,tuple(tip_at+[0,0,-0.0105]),iron_grip,"hold heat through the copper",rot),
        Waypoint(0.75,tuple(tip_at+[0,0,0.039]),iron_grip,"lift the iron with the braid",rot),
        Waypoint(1.3,tuple(ir_home+[0,0,0.064]),iron_grip,"return to the rest",iron_yaw),
        Waypoint(1.02,tuple(ir_home),iron_grip,"lower the iron",iron_yaw),
        Waypoint(0.63,tuple(ir_home),-0.029,"release the iron",iron_yaw),
        Waypoint(2.85,tuple(ir_home+[0,0,0.144]),-0.029,"withdraw from the iron",iron_yaw),
    ]
    left = [
        Waypoint(2.2,tuple(wick_home+[0,0,0.068]),-0.035,"approach the copper braid reel"),
        Waypoint(0.9,tuple(wick_home),-0.035,"reach the reel"),
        Waypoint(0.65,tuple(wick_home),spool_grip,"grip the reel"),
        Waypoint(0.72,tuple(spool_at),spool_grip,"square the braid across the joint",-0.078),
        Waypoint(4.86+(0.48 if late else 0.0),tuple(spool_at),spool_grip,"keep the braid over the pad",-0.078),
        Waypoint(0.75,tuple(spool_at+[0,0,0.043]),spool_grip,"lift the braid with the iron",-0.078),
        Waypoint(1.08,tuple(spool_park+[0,0,0.057]),spool_grip,"carry away the used braid"),
        Waypoint(1.15,tuple(spool_park),spool_grip,"set the reel down"),
        Waypoint(0.65,tuple(spool_park),-0.038,"release the reel"),
        Waypoint(2.8,tuple(spool_park+[0,0,0.121]),-0.038,"withdraw"),
    ]
    return {"right":right,"left":left}
