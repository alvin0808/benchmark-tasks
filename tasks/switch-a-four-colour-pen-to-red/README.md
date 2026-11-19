# Switch a four-colour pen from black to red: hold the light round pen and slide the red slider down until it latches

A four-colour retractable ballpoint pen (the size of a BIC 4 Colours: 145 mm long, 12 mm across, 11.5 g) lies on a
desk pad with its black tip out. The instruction asks for the pen to be switched to red (the red tip out, the other
tips in) and left lying on the pad.

Four-colour pens (BIC 4 Colours, Pilot, Zebra and others) are on desks in offices and schools everywhere, and
switching to red to mark a correction is an everyday gesture. Each colour has its own slider in a slot round the top
of the barrel; sliding one down toward the point until it catches pushes its refill out and lets go of the one that
was out. Pilot's instructions for its four-colour ballpoint pens ([instructions for use][pilot], "Ball Point Pen"):

> (1) Slide the lever of the colour you wish to use and the writing tip comes out, and is set in place.
> (2) To retract the writing tip, slide other lever.

The BIC 4 Colours works the same way: a retailer's listing ([Rapid Electronics][rapid]) gives a "choice of black,
blue, red and green ink from one pen", "which can be selected by pressing the appropriate coloured slider on the top
of the blue barrel".

## The hand skill, and why two hands

A person holds the pen in one hand and slides the colour's lever with the thumb. The robot has the same problem, and
it is visible in the scene:

- The pen weighs 11.5 g and lies loose on the pad. The red slider needs 2 N to start and 4 N at the end of its 10 mm
  stroke. Pushed along its length on its own, the pen just slides over the pad: with the right arm alone (below,
  `PEN4_ONE=1`) the pen slid 32 mm and the slider did not move. So one gripper pinches the barrel near the point while
  the other pushes the tab.
- The tab is small: 4.5 mm wide, standing 3 mm proud of a 12 mm barrel, 13 mm from the black tab on the other
  shoulder of the barrel. A fingertip has to come down behind it without landing on it, and push it along the barrel
  without riding over it or catching the black one.
- The stroke has to be finished: a slider that stops short of its catch springs back, and the black tip stays out.
- Both hands then let go and leave the pen lying on the pad.

Two arms are the natural way to do it, not a strict need: the task does not forbid other ways of holding the pen
(against something, or pressed down hard), and they were not tried.

The task is the task file [`switch_a_four_colour_pen_to_red.py`](switch_a_four_colour_pen_to_red.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the pen and the pad are built in these files from boxes, cylinders and spheres.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a joint. The pen
(a loose body with four sliders on prismatic joints along its barrel) is built with `bodykit.py`, which adds such
bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and
gives the task file each body's pose and each joint's position. This copy adds prismatic joints to its linkages
(`Link(kind="prismatic")`). The task file sets each slider's spring target every control step through Isaac Lab's
articulation API: that is how the catch is modelled (below).

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Desk pad, loose: 24 x 34 cm, 3 mm thick, 0.30 kg, grey, long side along y, centred at x 20, y -1.
- Pen, loose, lying on the pad along y: axis at x 19, the cap end at y -8.5 and the point at y +6.0. From the cap end:
  the white end cap (0-10 mm), the blue upper barrel (10-62 mm, 12 mm across) with the four slider slots (20-44 mm),
  the white lower barrel (62-125 mm), the cone and steel point (125-145 mm). The four sliders sit 90 degrees apart
  round the barrel. Seen from the cap end, measured from straight up toward +x: red at 45 degrees (uppermost, far
  side), black at 315 degrees (uppermost, near side), blue at 135 and green at 225 degrees (the pen rests on these two
  tabs and on its lower barrel), the steel clip at 270 degrees (toward the robot). Each tab is 4.5 mm wide, 8 mm long
  and stands 3 mm proud of the barrel; its back (cap-side) face is 22 mm from the cap end when the slider is up and
  32 mm when it is down. 8.5 g barrel and four 0.75 g sliders with their refills: 11.5 g.
- The black tip is out (its slider down and caught), the others in.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

What is modelled:

- Each slider is a separate part on a slide along the barrel, 0 to 10 mm of travel, on a spring (200 N/m) that pulls
  it back up with 2.0 N at the top and 4.0 N at the bottom (estimates for the real pen's return springs and catch).
- The catch is a rule on the sliders' travel, read every control step: a slider pushed down to 9.5 mm is held there
  (its spring's anchor moves past the bottom, so it presses onto its bottom stop with 4 N); a slider pushed down past
  6 mm lets go of the one that was held, which springs back up (as on the real pen, where sliding another lever
  retracts the tip that was out). A slider that was let go acts on the catch again only once it is back above 5 mm.
- A refill's tip is out of the point when its slider is more than 8 mm down (2 mm out when it is fully down). The
  tips are drawn only (they do not collide).
- Not modelled: the clip's release, the refills inside the barrel, the click.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the pen's pose, the four sliders' positions and the points of each finger's
collision hulls, and applies the catch rule.

- 0.3: the black, blue and green tips are in (each slider within 1 mm of its top).
- 0.7 more: the red tip is out and set by its catch (the red slider 9.5 mm or more down and held by the catch: its tip
  1.5 mm or more out of the point), counted only with the pen let go (every finger point 2 cm or more from it), lying
  on the pad (its middle over the pad, its axis at most 3 mm higher above the pad than where it rests on its two tabs)
  and still (moving less than 1 mm in 1 s). A red tip held out by a finger does not count.
- 1.0 (success, held 1 s): both.

Another colour left out (black never released, or blue or green pushed down to their catch instead) loses the 0.3;
red not set loses the 0.7. Doing nothing scores 0. A real trial is scored from a close photo of the point at the end
(which tips are out) and the trial video (the pen let go and lying on the pad at the end).

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=switch_a_four_colour_pen_to_red.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Switch the pen to red: the red tip out and the other tips in. Leave the pen lying on the desk pad." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pen4-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The left gripper comes down over the
barrel 105 mm from the cap end, tilted 40 degrees so its wrist leans away from the right arm, fingers closing across
the barrel, and pinches it (a 6 mm commanded gap on the 12 mm barrel). The right gripper, closed, points at the pen's
axis from the red tab's side (turned 45 degrees about the pen's axis and tilted 15 degrees back), brings its
fingertips down 0.5 mm off the barrel 3 mm behind the red tab, and pushes along the barrel toward the point, aimed
25 mm past the end of the tab's stroke (the arm, soft sideways, stops short under load). The red slider passes 6 mm
(black is let go and springs back in) and reaches its catch at 9.5 mm. The right gripper holds 1 s, lifts off and backs
away; the left gripper opens and rises. It reaches success at step 395 (19.75 s).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=switch_a_four_colour_pen_to_red.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Switch the pen to red: the red tip out and the other tips in. Leave the pen lying on the desk pad." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/pen4-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/pen4-dream/*.json; do inspect-robots video "$log"; done
```

`PEN4_DEBUG=1` prints, every 5 steps (`PEN4_DEBUG_EVERY=1` for every step), each slider's travel, which one the catch
holds, the pen's position, tilt and roll and how far it has moved, how close the nearest finger is, whether it is
still and lying on the pad, the score, the catch's last events, the red tab's and the right fingers' positions in the pen's
frame, and where each gripper is and how far open.

`PEN4_ONE=1` runs the demo the wrong way: the right arm alone makes the same push while the left arm stays at its
ready pose, so nothing holds the pen. The fingertips meet the red tab, and the pen slides 32 mm along the pad toward
its point; the red slider does not move (its 2 N spring is stiffer than the pen's friction on the pad), black stays
out. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell Server Edition shared with other simulation jobs: the scripted demo while
rendering 1280 x 720 video stepped 395 steps in 136 s; a Dream-1 trial of 2,400 steps at 320 x 176 stepped in 713 s
(about 12 minutes); debug runs of the demo at 320 x 176 without video stepped about 400 steps in 107 s, 4 to 5 minutes
from start to finish. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A BIC 4 Colours (Original) pen, or any four-colour retractable ballpoint pen with sliders, and a desk pad or a sheet of
grey card about 24 x 34 cm, laid loose on the table with its middle 20 cm in front of the midpoint between the arm
bases.
Before each trial push the black slider down until it catches (black tip out, the others in) and lay the pen on the
pad along two tape marks with its cap end toward the robot's right and the red slider uppermost on the far side.
Take a close photo of the point at the end of the trial.

[pilot]: https://www.pilot.co.jp/support/dr4_1_light_2018_eng.pdf
[rapid]: https://www.rapidonline.com/bic-4-color-8482-4-colour-ballpoint-pens-517588
