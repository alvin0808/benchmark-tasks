"""Seat the elastic skirt of a reusable square silicone lid below a pan rim."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bodykit import Body, BoxShape, Joint, as_numpy, on_reset, on_step, rotation
from yam_arena import ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
PAN = np.array([.443, 0., TOP])
START = np.array([.211, 0., TOP + .022])
PAN_HALF = .102
RIM_HALF = .108
RIM_Z = .065
SKIRT = .018
FREE_END = .092
TAB = .114
GLASS = (.61, .72, .73)
EDGE = (.77, .84, .85)
SILICONE = (.32, .57, .54)
BEAD = (.19, .43, .40)


def make_pan():
    parts = [
        BoxShape((0, 0, .003), (.192, .192, .006), GLASS),
        BoxShape((0, 0, -.001), (.231, .231, .002), (.18, .20, .24), collide=False),
    ]
    for sign in (-1, 1):
        parts += [
            BoxShape((sign*.100, 0, .032), (.004, .199, .061), GLASS),
            BoxShape((0, sign*.100, .032), (.199, .004, .061), GLASS),
            BoxShape((sign*.102, 0, .062), (.012, .211, .006), EDGE),
            BoxShape((0, sign*.102, .062), (.193, .012, .006), EDGE),
        ]
    return Body.build("pan", PAN, .81, parts, fixed=True, contact_offset=.0003)


def make_lid():
    parts = [BoxShape((0, 0, .001), (.229, .178, .002), SILICONE)]
    for sign in (-1,1):
        parts.append(BoxShape((sign*.115, 0, -SKIRT/2), (.003, .166, SKIRT), BEAD))
    for index in range(-2,3):
        parts.append(BoxShape((index*.035,0,.00215),(.0014,.146,.0003),BEAD,collide=False))
    return Body.build("lid", START, .052, parts, contact_offset=.00026, friction=(.78,.68))


def make_end(name, sign):
    parts = [
        BoxShape((0, sign*.074, .0018), (.229, .040, .002), SILICONE),
        BoxShape((0, sign*FREE_END, -.008), (.231, .003, .016), BEAD),
        BoxShape((0, sign*(FREE_END-.0017), -.015), (.224, .004, .004), BEAD),
        BoxShape((0, sign*(TAB-.009), -.0005), (.030, .024, .0055), SILICONE),
        BoxShape((0, sign*TAB, .002), (.025, .012, .009), BEAD),
    ]
    for xsign in (-1,1):
        parts.append(BoxShape((xsign*.115, sign*.079, -.008),(.003,.029,.016),BEAD))
    travel = (0., .035) if sign == 1 else (-.035, 0.)
    return Body.build(name, START, .021, parts, contact_offset=.00026, friction=(.92,.81),
                      joint=Joint(tuple(START), parent="lid", kind="slide", axis="Y",
                                  slide_range=travel, axial=(70., 2., 0.)), solver_iterations=32)


OBJECTS = [make_pan(), make_lid(), make_end("left_skirt", 1), make_end("right_skirt", -1)]
_state = {}


@on_reset
def reset():
    _state.clear()
    _state.update(step=0, score=0., joints={})


reset()


def _hand(base, side):
    robot = base.scene[f"{side}_robot"]
    if side not in _state["joints"]:
        _state["joints"][side] = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    q = as_numpy(robot.data.joint_pos)[0][_state["joints"][side]]
    return ik.tcp_pose(q, station.BASES[side])[0]


@on_step
def evaluate(env):
    base = getattr(env, "unwrapped", env)
    lid = as_numpy(base.scene["lid"].data.root_link_pose_w)[0]
    matrix = rotation(lid[3:])
    offset = lid[:3] - PAN
    yaw_ok = max(abs(matrix[0,0]), abs(matrix[0,1])) >= .997
    level = matrix[2,2] >= .997
    centered = np.linalg.norm(offset[:2]) <= .008
    top_at_rim = -.002 <= offset[2] - RIM_Z <= .009
    positions = [as_numpy(base.scene[f"{side}_skirt"].data.root_link_pose_w)[0,:3]
                 for side in ("left", "right")]
    ends = [pose + matrix @ [0,sgn*FREE_END,-.015] for pose,sgn in zip(positions,(1,-1))]
    lateral = matrix[:,1]
    wrapped = []
    for sgn, end in zip((1,-1),ends):
        edge_distance = sgn * np.dot(end-PAN, lateral)
        skirt_depth = RIM_Z - (end[2]-TOP)
        wrapped.append(PAN_HALF-.002 <= edge_distance <= RIM_HALF+.006 and .006 <= skirt_depth <= .018)
    covered = centered and yaw_ok and level and top_at_rim
    corners = [end + matrix @ [xsign*.112,0,0] for end in ends for xsign in (-1,1)]
    corners_seated = True
    for point in corners:
        crosswise = abs(float(np.dot(point-PAN, matrix[:,0])))
        depth = RIM_Z - (point[2]-TOP)
        if not (PAN_HALF-.002 <= crosswise <= RIM_HALF+.010 and .006 <= depth <= .020):
            corners_seated = False
    partial = .25 * (1 + sum(wrapped)) if covered else 0.
    tab_points = [p + matrix @ [0,s*TAB,0] for p,s in zip(positions,(1,-1))]
    clear = all(min(np.linalg.norm(_hand(base, side)-p) for p in tab_points) >= .042
                for side in ("left","right"))
    still = np.linalg.norm(as_numpy(base.scene["lid"].data.root_link_vel_w)[0,:3]) <= .012
    _state["score"] = 1. if partial == .75 and corners_seated and clear and still else partial
    _state["step"] += 1
    if os.getenv("LID_DEBUG") == "1" and _state["step"] % 10 == 0:
        print(f"LID {_state['step']:03} score={_state['score']} lid={np.round(offset,4)} "
              f"level={matrix[2,2]:.5f}/{max(abs(matrix[0,0]), abs(matrix[0,1])):.5f} ends={np.round(np.asarray(ends)-PAN,4)} "
              f"hands={np.round([_hand(base,s)-PAN for s in ('left','right')],4).tolist()}",flush=True)


def score(objects):
    return min(_state["score"], .75)


def success(objects):
    return _state["score"] == 1.


def demo():
    spread = .033 if os.getenv("LID_NO_STRETCH") != "1" else 0.
    open_, pinch = -.025, -.010
    result = {}
    for side, sign in (("left",1),("right",-1)):
        yaw = -sign*math.pi/2
        pickup = START+[0,sign*(TAB-.002),.008]
        placement = PAN+[.001,sign*(TAB+spread-.002),RIM_Z+.004]
        held = pickup+[0,sign*spread,.078]
        high = placement+[0,0,.035]
        result[side] = [
            Waypoint(1.6,tuple(pickup+[0,0,.057]),open_,"approach the pull tab",yaw),
            Waypoint(.85,tuple(pickup),open_,"straddle the pull tab",yaw),
            Waypoint(.65,tuple(pickup),pinch,"grasp the pull tab",yaw),
            Waypoint(.95,tuple(pickup+[0,0,.078]),pinch,"lift the lid from the table",yaw),
            Waypoint(.9,tuple(held),pinch,"stretch the two ends past the rim width",yaw),
            Waypoint(1.5,tuple(high),pinch,"carry the stretched lid above the pan",yaw),
            Waypoint(1.3,tuple(placement),pinch,"lower the skirts below the lip",yaw),
            Waypoint(.65,tuple(placement-[0,sign*min(spread,.011),0]),pinch,"let the skirt tighten under the rim",yaw),
            Waypoint(.65,tuple(placement-[0,sign*min(spread,.011),0]),open_,"release both tabs",yaw),
            Waypoint(1.4,tuple(placement+[0,sign*.040,.110]),open_,"leave the pan covered",yaw),
        ]
    return result
