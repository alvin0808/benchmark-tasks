# Show a webcam its checkerboard in the views a calibration needs, then put the board back in its rack

Every camera on a robot is calibrated (its focal length, principal point and lens distortion are measured) by showing it
a printed checkerboard in a series of views, and again whenever the camera is moved, knocked or replaced. Today a person
does it, robot by robot. A USB webcam on a post at the far side of the table looks back at the arms. A checkerboard
calibration board (30 x 16 cm, 3 mm aluminium composite) leans in a sloped wooden stand, its rack, in front of the arms,
printed face up toward the webcam. The instruction asks for the board to be shown to the camera in all the views a
calibration needs and then put back in its rack; it does not say which views or how.

The views the task checks are the ones the ROS camera calibration tool asks for. ROS 2 `camera_calibration`, [Tutorial:
Monocular Calibration][ros-tut], "Moving the Checkerboard": "In order to get a good calibration you will need to move
the checkerboard around in the camera frame such that: checkerboard on the camera's left, right, top and bottom of field
of view [...] checkerboard filling the whole field of view; checkerboard tilted to the left, right, top and bottom
(Skew)", and "At each step, hold the checkerboard still until the image is highlighted in the calibration window." The
task uses the tool's own measures ([calibrator.py][ros-cal]): X and Y, where the board's corners sit across the image
(with the image shrunk by half the board's size, so that 0 and 1 mean touching an edge), and size, the square root of
the corner grid's area over the image's; the tool also rejects a detection with a corner within 8 pixels of the image's
edge. The board must be found by OpenCV's chessboard detector, which needs every inner corner and white space round the
squares ([findChessboardCorners][cv-find]): "The function requires white space (like a square-thick border, the wider
the better) around the board to make the detection more robust in various environments. Otherwise, if there is no border
and the background is dark, the outer black squares cannot be segmented properly and so the square grouping and ordering
algorithm fails." The calibration itself is OpenCV's `calibrateCamera`, as in OpenCV's [camera calibration
tutorial][cv-tut] ("Those images are taken from a static camera and chess boards are placed at different locations and
orientations"), which returns the camera matrix with the "focal length (fx,fy) and optical centers (cx, cy)".

Why two arms are needed, as seen in the scene: the checkerboard and its white border must stay uncovered, so the board
can only be held by its two plain grey ends. Held by one end, its middle is 14 cm from the grip, and a pinch on a 3 mm
board cannot hold that twist: the board swings down in the grip and its far end drops onto the table (`CALIB_ONE=1`,
below). Held at the middle of its top edge, the fingers cover the white border above the squares. Held by both top
corners, one gripper on each grey end, it hangs level between them and can be moved, turned and tilted as one. The task
is the task file [`show_a_robot_camera_its_calibration_board.py`](show_a_robot_camera_its_calibration_board.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The scorer also uses OpenCV (`cv2`, tested with 4.14), which Isaac Sim itself
installs (`isaacsim-core` depends on it). No third-party models or assets are used: every object is built in the task
file from boxes and cylinders.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them. `bodykit.py` adds bodies made of
several shapes (the board with its printing, the stand, the webcam, the post): importing it hooks `yam_arena.scene` when
the simulator starts, without changing yam_arena, and lets the task file follow the board and the arms after every
control step.

What is modelled and what is not:

- The board is a 300 x 160 x 3 mm plate of 0.18 kg (3 mm aluminium composite weighs about 3.8 kg per square metre).
  Printed on its front: a checkerboard of 9 x 6 squares of 20 mm (8 x 5 = 40 inner corners) inside a white border 20 mm
  wide (one square, as OpenCV asks), and a plain grey band 40 mm wide at each end to hold it by. Its back is grey. The
  printing is visual only.
- The rack is a wooden stand: an 8 mm base, a 10 mm lip in front of the board's bottom edge and a sloped rest under its
  back, holding it leaned back 60 degrees from upright.
- The webcam is 90 x 30 x 30 mm on a 15 mm square post, 32 cm from the board held facing it, 1280 x 720 pixels with a
  focal length of 1,100 pixels in both directions (60 degrees across) and its principal point at the image centre. The
  simulation does not render it. What it sees is computed from the board's pose and those intrinsics, as an ideal
  pinhole camera with no lens distortion: the board's image (its white face, grey ends and black squares, drawn at four
  times the resolution and averaged down) is what OpenCV's detector is run on, and whether an arm is in the way is a
  line-of-sight test from the lens to points 1 cm apart over the squares and their border, against boxes round each
  arm's links, gripper and fingers (from the YAM's collision meshes, placed by the arm's joints).

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
board's pose is read from the simulation after every control step, and each arm's links and fingers from its joints.

A view counts when all 9 x 6 squares lie 8 pixels or more inside the image, the board faces the webcam (within 50
degrees), no part of either arm is in front of the squares or their white border as seen from the webcam, the squares
span 30 % or more of the image width, the board is 5 mm or more from its rack and 2 cm or more above the table, its
corners have moved less than 1 mm in the last 0.5 s (10 control steps), and OpenCV's `findChessboardCorners` (with the
flags the ROS tool uses, refined by `cornerSubPix`) finds all 40 inner corners in the board's image. There are nine
targets, and one view may meet several:

| Target | Measure |
|---|---|
| left, right | ROS X 0.25 or less / 0.75 or more |
| top, bottom | ROS Y 0.25 or less / 0.75 or more |
| close | ROS size 0.5 or more (the corner grid covers a quarter of the image) |
| left edge away, right edge away | board turned 20 degrees or more about the image's vertical |
| top edge away, bottom edge away | board tilted 20 degrees or more about the image's horizontal |

(The ROS tool fills its X and Y bars at a spread of 0.7; here a spread of 0.5 is asked for, because a board this size
fills more than half of this webcam's image, which leaves it little room to move.)

Score: 0.08 per target met (0.72 for all nine). 0.8 once all nine are met and OpenCV's `calibrateCamera` (its default
model, as in OpenCV's tutorial), run on the corners found in the views that met them, gives the focal length within 2 %
of the webcam's (both fx and fy). Views that only move the board about, without turning or tilting it, cannot pass this
check: from them the focal length cannot be told apart from the board's distance (calibrated on the scripted demo's five
untilted views alone, as planned, the focal length comes out 17 times too long). Full score (1.0): that, with the board
back in its rack (its middle within 1.5 cm and its tilt within 10 degrees of where it started), both grippers 5 cm or
more from it, still for 1 s. The score is 0 for the rest of the run once the board touches the table outside its rack
(dropped, swung down or put down), falls off the table, or touches the webcam or its post. Doing nothing scores 0: in
its rack the board touches the rack (and the rack's lip hides its bottom edge from the webcam), so no view counts. `-E
object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=show_a_robot_camera_its_calibration_board.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Show the camera on the post the calibration board in all the views a calibration needs, then put the board back in its rack." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/calib-dream
```

The scripted demo, which reaches full score (both grippers take the board by its top corners, one on each grey end, lift
it off the rack and raise it to face the webcam; they show it at the left, right, top and bottom of the webcam's image,
close up, and turned and tilted each way by about 25 degrees, holding each view still for about a second; then they lean
it back into the rack, let go and draw back):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=show_a_robot_camera_its_calibration_board.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Show the camera on the post the calibration board in all the views a calibration needs, then put the board back in its rack." \
  --scorer task_score --max-steps 960 --store-frames --log-dir ~/yam-eval/calib-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/calib-dream/*.json; do inspect-robots video "$log"; done
```

`CALIB_DEBUG=1` prints, every 10 steps (`CALIB_DEBUG_EVERY=5` for every 5), where the board is and how far it leans, its
lowest point, the webcam's measures of it (X, Y, size, width, turn, tilt, angle to the webcam), how many of the print's
points an arm hides, its gap to the rack, whether it is still, whether the view counts and why not, the targets met, the
calibration's focal-length errors, whether it is back in its rack, how near the nearest gripper is, any broken rule, and
each gripper's centre and finger opening; and a line for each view that meets new targets and for the calibration.

`CALIB_ONE=1` runs the demo the wrong way: only the right gripper takes the board, by its right top corner, and plays
the same moves while the left arm stays back. As the gripper lifts its corner, the board's other end stays down on the
rack; carried toward the first view, the board swings down in the single grip and its far end hits the table (step 208),
so the run scores 0; no view had counted before that. A grip at the middle of the top edge fails the line-of-sight test
instead: held that way and facing the webcam, the fingers hide part of the white border above the squares (6 of the test
points), so no view counts.

## Runtime per trial

On one RTX PRO 6000 (Blackwell) that was shared with another user's training job and two other simulations at the time:
about 11 minutes for the scripted demo while rendering 1280 x 720 video (873 steps, 44 s of simulated time; the run ends
once success has held for 1 s), 9 to 11 minutes for the same demo at 320 x 176, and about 27 minutes for a Dream-1 trial
of 2,400 steps. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, the scripted demo is deterministic and the scorer draws no
random numbers.

## Building it for real

A 1280 x 720 USB webcam on a post or a small stand, calibrated once beforehand for its reference focal length; a
checkerboard of 9 x 6 squares of 20 mm with a 20 mm white border, printed on 3 mm aluminium composite, 300 x 160 mm,
with plain grey ends; and a sloped stand with a lip. The webcam records the trial. Afterwards OpenCV's
`findChessboardCorners` is run on its frames; a view is a stretch of at least 0.5 s in which the corners are found and
move less than about 3 pixels, with no finger or arm over the squares or their border; the views are sorted into the
nine targets with the measures above; and `calibrateCamera` on them must give the focal length within 2 % of the
reference. The board in its rack, the grippers clear of it and a drop onto the table are judged from the overhead and
wrist videos.

[ros-tut]: https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/doc/tutorial_mono.rst
[ros-cal]: https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/calibrator.py
[cv-find]: https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp
[cv-tut]: https://github.com/opencv/opencv/blob/4.x/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.markdown
