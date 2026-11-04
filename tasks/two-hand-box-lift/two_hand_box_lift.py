"""On the station table, in front of the two arms: a closed, handle-less cardboard box lies on the centre line
near the robot, and a fixed wooden riser (a low shelf) stands just beyond it. The box is wider and deeper than a
gripper can open, so no single gripper can pick it up: it has to be squeezed between both grippers, lifted and set
down on the riser.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- box: closed white corrugated cardboard box, 14 cm deep (x) x 20 cm wide (y) x 10 cm tall, 400 g, centred at
  (0.16, 0), on the table.
- riser: fixed dark wooden block, 16 cm deep x 30 cm wide x 5 cm tall, centred at (0.33, 0).

Full score: the box stands upright on the riser, centred on it within 3 cm, and stays still for 3 s.
"""

import numpy as np

from yam_arena.constants import FINGER_CLOSED
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 3.0  # long enough for a demo to let go and pull its grippers away

CARDBOARD = (0.92, 0.91, 0.88)  # white corrugated board, so it stands out on the light oak table
WOOD = (0.36, 0.20, 0.10)

BOX_SIZE = (0.14, 0.20, 0.10)  # x (front to back), y (side to side), z
BOX_X = 0.16
RISER_SIZE = (0.16, 0.30, 0.05)
RISER_X = 0.33

BOX = Box("box", pos=(BOX_X, 0.0, TOP + BOX_SIZE[2] / 2), size=BOX_SIZE, mass=0.4, color=CARDBOARD)
RISER = Box("riser", pos=(RISER_X, 0.0, TOP + RISER_SIZE[2] / 2), size=RISER_SIZE, color=WOOD)
OBJECTS = [BOX, RISER]

# Thresholds, as a reviewer would check them with a tape measure.
RISER_TOP = TOP + RISER_SIZE[2]
ON_TOL = 0.01  # the box's underside within 1 cm of the riser's top
CENTRE_TOL = 0.03  # full score: the box's centre within 3 cm of the riser's centre, front to back and side to side
TILT_MAX = np.radians(10)  # upright: the box's top face tilted less than 10 degrees


def _tilt(quat_xyzw) -> float:
    """Angle between the object's own z axis and the vertical (rad)."""
    x, y, _, _ = quat_xyzw
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def _upright_on_riser(box) -> bool:
    """Standing on its bottom face on top of the riser: upright, its underside at the riser's top, and its centre
    over the riser's top face."""
    upright = _tilt(box.quat_xyzw) < TILT_MAX
    level = abs(box.low[2] - RISER_TOP) < ON_TOL
    over = abs(box.pos[0] - RISER_X) < RISER_SIZE[0] / 2 and abs(box.pos[1]) < RISER_SIZE[1] / 2
    return bool(upright and level and over)


def success(objects):
    """The box stands still, upright and centred on the riser."""
    box = objects["box"]
    centred = abs(box.pos[0] - RISER_X) <= CENTRE_TOL and abs(box.pos[1]) <= CENTRE_TOL
    return _upright_on_riser(box) and centred and box.resting()


def distance(objects):
    box = objects["box"]
    goal = np.array([RISER_X, 0.0, RISER_TOP + BOX_SIZE[2] / 2])
    return float(np.linalg.norm(box.pos - goal))


def score(objects):
    """Partial credit (success scores 1.0): 0.5 when the box ends upright on the riser but more than 3 cm off
    centre; 0 for anything else (still on the table, on its side, dropped or fallen off)."""
    return 0.5 if _upright_on_riser(objects["box"]) else 0.0


# -- scripted demo -------------------------------------------------------------------------------------------------
# Both grippers stay closed and press the box's left and right faces with the outer faces of their fingers (turned
# 90 degrees, in opposite directions for the two arms, as joint6's range needs). The closed fingers reach 3.4 cm to
# each side of the grip point at this turn, so the grip point sits that far outside a face, less the squeeze.
FINGER_REACH = 0.034
# Each grip point is commanded this far inside the box; the arms' joint stiffness turns the shortfall into squeeze.
# Further out over the riser the arms are less stiff sideways, so they are commanded deeper there.
SQUEEZE = 0.05
SQUEEZE_FAR = 0.06
CLEAR = 0.04  # stand-off from the box's faces before squeezing and after letting go
GRIP_Z = 0.03  # grip point above the box's underside: fingertips 1.7 cm above it, the gripper body clear of its top
LIFT = 0.12  # lift target for the box's underside; it sags about 2 cm under the squeeze, still clearing the riser
LET_GO = 0.003  # set the box down from 3 mm above the riser


def demo():
    """Both arms together: each presses its closed fingers against one side of the box, both squeeze, lift it about
    10 cm, carry it forward over the riser, lower it onto the riser, let go sideways and rise. The two plans share
    every timing, so the box stays level and centred between them."""
    half = BOX_SIZE[1] / 2
    plans = {}
    for arm, side, yaw in (("left", 1.0, -np.pi / 2), ("right", -1.0, np.pi / 2)):
        y_touch = side * (half + FINGER_REACH)
        y_out = y_touch + side * CLEAR
        y_in, y_in_far = y_touch - side * SQUEEZE, y_touch - side * SQUEEZE_FAR
        z_grip, z_lift = TOP + GRIP_Z, TOP + LIFT + GRIP_Z
        z_place = RISER_TOP + LET_GO + GRIP_Z
        x0, x1 = BOX_X, RISER_X
        plans[arm] = [
            Waypoint(2.0, (x0, y_out, z_lift), FINGER_CLOSED, "beside the box", yaw),
            Waypoint(1.5, (x0, y_out, z_grip), FINGER_CLOSED, "down", yaw),
            Waypoint(1.5, (x0, y_in, z_grip), FINGER_CLOSED, "squeeze", yaw),
            Waypoint(0.5, (x0, y_in, z_grip), FINGER_CLOSED, "settle", yaw),
            Waypoint(2.0, (x0, y_in, z_lift), FINGER_CLOSED, "lift", yaw),
            Waypoint(4.0, (x1, y_in_far, z_lift), FINGER_CLOSED, "carry", yaw),
            Waypoint(1.5, (x1, y_in_far, z_place), FINGER_CLOSED, "lower", yaw),
            Waypoint(1.0, (x1, y_out, z_place), FINGER_CLOSED, "let go", yaw),
            Waypoint(1.5, (x1, y_out, z_lift), FINGER_CLOSED, "rise", yaw),
        ]
    return plans
