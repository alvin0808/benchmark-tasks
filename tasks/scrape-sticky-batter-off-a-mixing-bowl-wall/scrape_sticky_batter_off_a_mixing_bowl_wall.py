"""Gather clinging batter from a mixing bowl wall with the flexible edge of a spatula."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, CylShape, Joint, _author_joint, allow_matrix_grips, as_numpy, on_reset, on_step, plate, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import ArmPlan, Waypoint, grip_rotation, axis_angle_matrix
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
BOWL_START = np.array([.349,-.042,TOP])
RIM = .076
RADIUS = .086
SUPPORT = np.array([.236,.184,TOP])
TOOL = SUPPORT+[0,0,.158]
TIP = np.array([0,0,-.145])
TINT = (.65,.80,.80)
SILICONE = (.32,.48,.72)
EDGE = (.53,.66,.85)
BATTER = (.78,.44,.19)
SMEARS = ((.81,.034),(0.,.052),(0.,.065))
PROFILE = [(.008,.041),(.022,.054),(.049,.075),(.072,.086)]


def radial(angle):
    return np.array([math.sin(angle),math.cos(angle),0.])


def radius_at(z):
    return float(np.interp(z, *zip(*PROFILE)))


def make_bowl():
    parts = [CylShape((0,0,.004),.044,.008,TINT)]
    for i in range(24):
        angle = i*math.tau/24
        normal = radial(angle)
        for (a,small),(b,large) in zip(PROFILE,PROFILE[1:]):
            inner = normal*small + [0,0,a]
            outer = normal*large + [0,0,b]
            span = 2*large*math.tan(math.pi/24)+.0005
            parts.append(plate(inner,outer,span,.003,normal,TINT))
        rim = normal*.085 + [0,0,.073]
        parts.append(BoxShape(tuple(rim),(.024,.010,.006),TINT,yaw=-angle))
    return Body.build("bowl",BOWL_START,.41,parts,friction=(.28,.22),contact_offset=.0003,
                      report_contacts=True,solver_iterations=32)


def make_tool():
    holder = Body.build("utensil_rest",SUPPORT,.22,[
        BoxShape((0,0,.003),(.065,.068,.006),(.30,.32,.35)),
        BoxShape((-.026,0,.042),(.007,.054,.074),(.30,.32,.35)),
        BoxShape((.026,0,.042),(.007,.054,.074),(.30,.32,.35)),
        BoxShape((0,-.010,.030),(.037,.006,.052),(.30,.32,.35)),
        BoxShape((0,.010,.030),(.037,.006,.052),(.30,.32,.35)),
    ],fixed=True,contact_offset=.00020)
    shaft = Body.build("handle",TOOL,.042,[
        BoxShape((0,0,.010),(.022,.022,.094),SILICONE),
        CylShape((0,0,-.064),.0048,.057,SILICONE),
    ],color=SILICONE,contact_offset=.00025,friction=(.87,.74))
    blade = Body.build("blade",TOOL,.023,[
        BoxShape((0,0,-.111),(.029,.009,.044),SILICONE),
        BoxShape((.002,0,-.135),(.033,.007,.011),EDGE),
        BoxShape((.004,0,-.142),(.024,.006,.006),EDGE),
    ],joint=Joint(tuple(TOOL+[0,0,-.093]),parent="handle",axis="X",limits=(-16.,16.),
                  stiffness=2.2,damping=.30), contact_offset=.00022,
       report_contacts=True,color=EDGE,solver_iterations=32)
    return holder,shaft,blade


def build_dabs():
    dabs = []
    for angle, z in SMEARS:
        n = radial(angle)
        pos = BOWL_START + n*(radius_at(z)-.012) + [0,0,z]
        shape = plate(-n*.004 + [0,0,-.010],n*.004 + [0,0,.010],
                      .025,.007,n,BATTER)
        dabs.append(Body.build(f"smear_{len(dabs)}",pos,.0026,[shape],
                               joint=Joint(tuple(pos),parent="bowl",kind="fixed",collide_connected=True),
                               friction=(.43,.34),damping=(12.,8.),max_linear_speed=.20,solver_iterations=32,
                               contact_offset=.00020,color=BATTER))
    return dabs


DABS = build_dabs()
OBJECTS = [make_bowl(),*make_tool(),*DABS]
_state = {}


@on_reset
def reset():
    for index,root,stage in _state.get("separated", {}).values():
        name = DABS[index].name
        path = f"{root}/{name}"
        _author_joint(stage,path,f"{path}/body",DABS[index])
    _state.clear()
    _state.update(separated={}, contact=None, joints={}, prior=None, score=0., step=0)


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

    base = getattr(env,"unwrapped",env)
    root = base.scene.env_prim_paths[0]
    bowl = as_numpy(base.scene["bowl"].data.root_link_pose_w)[0]
    blade = as_numpy(base.scene["blade"].data.root_link_pose_w)[0]
    bowl_rotation = rotation(bowl[3:])
    tip = bowl_rotation.T @ (blade[:3] + rotation(blade[3:]) @ TIP - bowl[:3])
    _state["blade_world"] = bowl[:3] + bowl_rotation @ tip
    _state["hand_left"] = _hand(base,"left")
    previous = _state["prior"]
    _state["prior"] = tip
    if _state["contact"] is None:
        view = PhysxManager.get_physics_sim_view()
        _state["contact"] = view.create_rigid_contact_view(
            [f"{root}/blade/body"],filter_patterns=[[f"{root}/{dab.name}/body" for dab in DABS]],
            max_contact_data_count=0,
        )
    contact = _state["contact"].get_contact_force_matrix(dt=float(base.physics_dt)).numpy()
    forces = np.linalg.norm(contact,axis=-1).reshape(-1)
    if previous is not None and tip[2]-previous[2] < -.00008:
        for i,force in enumerate(forces):
            if i not in _state["separated"] and force >= .42:
                stage = get_current_stage()
                stage.RemovePrim(f"{root}/{DABS[i].name}/joint")
                _state["separated"][i] = (i,root,stage)
    collected = 0
    collected_positions = {}
    for i in _state["separated"]:
        dab_world = as_numpy(base.scene[DABS[i].name].data.root_link_pose_w)[0,:3]
        local = bowl_rotation.T @ (dab_world-bowl[:3])
        collected_positions[i] = np.round(local,4).tolist()
        if np.linalg.norm(local[:2]) <= .063 and .005 <= local[2] <= .031:
            collected += 1
    score_value = .25 * collected
    tool_speed = as_numpy(base.scene["handle"].data.root_link_vel_w)[0,:3]
    bowl_speed = as_numpy(base.scene["bowl"].data.root_link_vel_w)[0,:3]
    blade_clear = (tip[2] >= .035 and (tip[2] >= RIM+.035
                   or np.linalg.norm(tip[:2]) >= RADIUS+.040))
    settled = (np.linalg.norm(tool_speed) <= .025 and np.linalg.norm(bowl_speed) <= .012
               and np.linalg.norm(bowl[:2]-BOWL_START[:2]) <= .030 and bowl_rotation[2,2] >= .990)
    hands_clear = True
    for side in ("left","right"):
        hand = bowl_rotation.T @ (_hand(base,side)-bowl[:3])
        hands_clear &= np.linalg.norm(hand[:2]) >= RADIUS+.040 or hand[2] >= RIM+.043
    _state["score"] = 1. if collected == len(DABS) and settled and blade_clear and hands_clear else score_value
    _state["step"] += 1
    if os.getenv("BATTER_DEBUG") == "1" and _state["step"] % 5 == 0:
        print(f"BATTER {_state['step']:03d} score={_state['score']} count={collected} "
              f"bowl={np.round(bowl[:3]-BOWL_START,4)} tip={np.round(tip,4)} "
              f"force={np.round(forces,2)} released={collected_positions} clear={blade_clear}/{hands_clear} settled={settled} hand={np.round(_hand(base,'left')-TOOL,4)} tool_speed={np.linalg.norm(tool_speed):.3f}",flush=True)


def score(objects):
    return min(_state["score"], .75)


def success(objects):
    return _state["score"] == 1.


class ScrapeWaypoint(Waypoint):
    def __init__(self, duration, tip, finger, label, orientation):
        super().__init__(duration, tuple(tip+[0,.025,.178]), finger, label, orientation)
        self.blade_target = np.asarray(tip)


def _allow_tip_feedback():
    if getattr(ArmPlan, "_spatula_tip_feedback", False):
        return
    original = ArmPlan.target

    def target(self, t):
        wrist, rot, finger, label = original(self,t)
        elapsed = t
        previous = None
        for index, waypoint in enumerate(self.waypoints):
            if elapsed >= waypoint.duration:
                elapsed -= waypoint.duration
                previous = waypoint
                continue
            measured = _state.get("blade_world")
            if not isinstance(waypoint,ScrapeWaypoint) or measured is None:
                return wrist,rot,finger,label
            if isinstance(previous,ScrapeWaypoint):
                origin = previous.blade_target
            else:
                if getattr(self,"_spatula_stage",None) != index:
                    self._spatula_start = measured.copy()
                    self._spatula_stage = index
                origin = self._spatula_start
            desired = origin + ik.smoothstep(elapsed / waypoint.duration) * (waypoint.blade_target-origin)
            wrist = _state["hand_left"] + np.clip(1.22*(desired-measured),-.033,.033)
            return wrist,rot,finger,label
        return wrist,rot,finger,label

    ArmPlan.target = target
    ArmPlan._spatula_tip_feedback = True


def demo():
    allow_matrix_grips()
    _allow_tip_feedback()
    yaw = 0.
    upright = axis_angle_matrix(np.array([1.,0.,0.]),math.radians(12.)) @ grip_rotation(yaw)
    open_, close = -.023,-.004
    pickup = TOOL+[0,0,.027]
    transit = np.array([.273,.110,TOP+.251])
    left = [
        Waypoint(1.5,tuple(pickup+[0,0,.06]),open_,"approach the spatula handle",yaw),
        Waypoint(.7,tuple(pickup),open_,"straddle the handle",yaw),
        Waypoint(.65,tuple(pickup),close,"grasp the handle",yaw),
        Waypoint(.9,tuple(pickup+[0,0,.080]),close,"lift clear of the holder",yaw),
        Waypoint(.85,tuple(transit),close,"carry the blade above the bowl",yaw),
    ]
    outward = radial(0.)
    lower_outward = radial(.81)
    offset = .026 if os.getenv("BATTER_INWARD_OFFSET") == "1" else 0.
    above = BOWL_START+outward*.044+[0,0,.095]
    start = BOWL_START+outward*(.069-offset)+[0,0,.070]
    middle = BOWL_START+outward*(.060-offset)+[0,0,.050]
    lower = BOWL_START+lower_outward*(.049-offset)+[0,0,.027]
    bottom = BOWL_START+lower_outward*(.031-offset)+[0,0,.014]
    left.extend([
        ScrapeWaypoint(1.35,above,close,"move the silicone edge above the wall",upright),
        ScrapeWaypoint(1.55,start,close,"align the working edge to the streak",upright),
        ScrapeWaypoint(1.65,middle,close,"press down along the upper wall",upright),
        ScrapeWaypoint(1.65,lower,close,"keep contact along the lower curve",upright),
        ScrapeWaypoint(1.15,bottom,close,"draw the batter down into the bowl",upright),
        ScrapeWaypoint(.95,BOWL_START+outward*.017+[0,0,.012],close,"clear the batter into the middle",upright),
        Waypoint(1.75,tuple(BOWL_START+[.028,.051,.328]),close,
                 "lift vertically above the mixing bowl",upright),
        Waypoint(1.55,tuple(BOWL_START+[.032,.215,.309]),close,
                 "hold the spatula steady and clear of the bowl",upright),
    ])
    brace = BOWL_START+[0,-.086,RIM+.009]
    high = brace+[0,0,.080]
    right = [
        Waypoint(2.5,tuple(high),0.,"approach the opposite rim"),
        Waypoint(1.1,tuple(brace),0.,"pin the bowl against the table"),
        Waypoint(12.85,tuple(brace),0.,"keep the bowl steady"),
        Waypoint(1.2,tuple(high),0.,"release the bowl"),
    ]
    return {"left":left,"right":right}
