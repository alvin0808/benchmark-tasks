# Listen at the four heart sites on a manikin with a stethoscope: diaphragm flat, sealed and still at each, in order

Listening to the heart is part of every nursing physical assessment. An adult chest trainer (a torso from the neck to
the upper abdomen, true size) lies on its back across the table, its right side toward the robot, as a patient lies
with the nurse at the right side; a stethoscope's chestpiece lies diaphragm down in a steel tray beside it. The
instruction asks for the heart to be listened to at the aortic, pulmonic, tricuspid and mitral areas, in that order,
with the diaphragm flat and sealed on the skin and held still for 2 s at each, and the chestpiece to be put back in its
tray.

What makes it hard is a hand skill, and it comes from the objects. The diaphragm is a flat 43 mm disc and it hears
only while its whole rim is on the skin; the chest is curved, so the slope of the skin under the disc is different at
every site (head to foot: 9.7 deg at the 2nd intercostal spaces, 4.4 deg at the 4th, 1.8 deg at the 5th; left to
right: 5 deg at the edges of the sternum, 26 deg at the mitral area out at the midclavicular line). The rim seals only
with the disc within about 2 deg of the skin there and pressed with 2 to 10 N (the skin and the rubber rim give about
0.3 mm per newton, so a disc a few degrees off leaves a gap on its high side however hard it is pressed). The gripper
pinches the head by its sides, so the head rocks a little between the fingers about the line joining them; the arm is
soft (about 1 N/mm), so the press is set by how far past the skin the hand is aimed, and pressing deeper turns the head
in the fingers and lifts one side of the rim. The disc must then stay still for 2 s, and move to the next site in the
order nurses use. One arm does it; the other is not needed.

The task is the task file
[`listen_at_the_four_heart_sites_with_a_stethoscope.py`](listen_at_the_four_heart_sites_with_a_stethoscope.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters

Registered nurses ([O*NET 29-1141.00][rn], 3,465,400 US jobs in 2025: "Monitor, record, and report symptoms or changes
in patients' conditions") and licensed practical nurses ([O*NET 29-2061.00][lpn], 666,900: "Work as part of a
healthcare team to assess patient needs") listen to patients' hearts in their routine assessments. The open nursing
textbook [Open RN, Nursing Skills, 2nd ed., ch. 9.3 "Cardiovascular Assessment"][openrn] (CC BY 4.0) gives the sites
and the order: "The aortic area is the second intercostal space to the right of the sternum. The pulmonic area is the
second intercostal space to the left of the sternum. Erb's point is directly below the pulmonic area and located at
the third intercostal space to the left of the sternum. The tricuspid (or parasternal) area is at the fourth
intercostal space to the left of the sternum. The mitral (also called apical or left ventricular area) is the fifth
intercostal space at the midclavicular line. Auscultation usually begins at the aortic area (upper right sternal edge).
Use the diaphragm of the stethoscope to carefully identify the S1 and S2 sounds." and "Move the stethoscope
sequentially to the pulmonic area (upper left sternal edge), Erb's point (left third intercostal space at the sternal
border), and tricuspid area". The task scores the four valve areas (Erb's point, between the pulmonic and tricuspid
areas, is not scored).

On pressure: "The amount of pressure exerted with a diaphragm in order to hear a faint high frequency murmur may (and
sometimes should) actually leave a visible imprint of the chest piece on the skin" ([Shindler, Cardiac Auscultation
Essays, Rutgers RWJMS][shindler]); a bench comparison of stethoscopes placed the chestpiece with "a standardized weight
of 227 gm corresponding to the typical pressure applied by a hand when auscultating a chest" ([Medical Devices:
Evidence and Research 12, Dove Press][dove]), about 2.2 N. The task's window, 2 to 10 N, starts there. The chestpiece
is modelled on the Littmann Classic III ([3M product overview][3m]: "Tunable diaphragms on both the adult and pediatric
sides of the chestpiece"; tunable technology lets you shift between sounds by "adjusting how much you press on the
chestpiece").

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the torso, the chestpiece and the tray are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The torso (a triangle mesh made from superellipse
cross-sections), the chestpiece (a two-sided head and a stem, with its rubber rim on its own part) and the tray are
built with `bodykit.py`, which adds such bodies: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives the task file each body's pose. This copy of `bodykit.py` turns on PhysX contact
reporting for the rim, so the task file reads the force between the rim and the torso every control step, and gives
the rim PhysX's compliant (implicit spring-damper) contact through its own physics material: 1.2 N/mm per contact
point with 20 N s/m of damping, standing in for the give of the trainer's skin and the rubber rim together (pressed
with 4.5 N, the face sinks 1.3 mm into the skin), and the force read holds steady within 0.1 N while the disc is held
still.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Adult chest trainer, fixed (it lies still), skin-coloured firm vinyl: a torso from the neck to the upper abdomen,
  lying on its back across the table, head toward +y, its right side toward the robot. 32 cm wide (x 4 to 36; 38 cm at
  the shoulders), 21.5 cm deep at the 4th-5th ribs, 19.3 cm at the sternal notch, 18.5 cm at its lower end; from y
  -17.5 to 11.5, a neck stub (11.6 cm across) beyond. Its midline runs along x 20.0; the sternal notch is at y 6.5;
  nipples at the 4th intercostal space, 9.5 cm each side of the midline. The four sites (x, y, height): aortic (17.2,
  0.5, 20.5), pulmonic (22.8, 0.5, 20.5), tricuspid (22.8, -5.0, 21.2), mitral (29.5, -7.5, 19.6); that is 2.8 cm
  either side of the midline 6 cm below the notch, 2.8 cm left of it 11.5 cm below, and 9.5 cm left of it 14 cm below.
  The skin's slope there (head to foot, left to right): 9.7 and 5.1 deg, 9.7 and 5.1 deg, 4.4 and 5.2 deg, 1.8 and
  25.9 deg.
- Stethoscope chestpiece (Littmann Classic III type), loose, 80 g: a steel two-sided head 44.5 mm across and 22 mm
  tall (adult side 12 mm tall, with a 3 mm rubber rim 44 mm across round the 43 mm diaphragm; a 30 mm waist; the
  paediatric side 36 mm across), and an 8 mm steel stem out of the waist 14 mm above the diaphragm, reaching 62 mm from
  the head's axis, with a dark collar where the tube fits. It lies diaphragm down in the tray, the head's centre at
  x 19.5, y -25.5, the stem toward +x.
- Steel tray, fixed: 13 x 10 cm inside (x by y), 1.5 cm walls, 3 mm floor, centred at x 22.5, y -25.5.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the rim's pose, the contact force between the rim and the torso (PhysX pair-wise
contact forces) and the points of each finger's collision hulls.

- A site is heard when, for 2 s (40 control steps) in a row: every one of 24 points round the diaphragm's 43 mm rim
  lies within 0.5 mm of the skin or pressed into it (the rim sealed all round); the contact force on the torso is 2 to
  10 N; the disc's centre is within 20 mm of the site, measured along the skin; and the centre moves less than 3 mm
  over the 2 s. The sites count only in order: aortic, then pulmonic, then tricuspid, then mitral (a site heard out of
  order does not count).
- 0.2 for each site heard (0.8 for all four).
- 1.0 (success, held 1 s): all four heard, and the chestpiece back in its tray (its face within 4 mm of the tray floor,
  its centre inside the walls, tilted under 10 deg), every finger point 2 cm or more from it, and it still (moving
  under 1 mm in 1 s).
- Nothing is latched: no outcome in this task is a hazard. Doing nothing scores 0.

A real trial is scored from the video, a kitchen scale and a photo. The trainer lies on a tared digital kitchen scale
(10 kg, 1 g) in view of the camera, and before the trial the diaphragm's rim is pressed once on a washable stamp pad.
A site counts when the video shows the chestpiece resting there for 2 s with the scale reading 204 to 1020 g (2 to
10 N) throughout, and the photo after the trial shows there a complete ring (no gap: the rim touched all round) that
is not smeared or doubled (it did not slide), its centre inside the site's 20 mm circle on a clear acetate overlay
laid on the trainer by its sternal-notch and midline marks. The order and the end state (in the tray, let go, still)
are read from the video. A trainer with its own site detection (the Life/form SmartScope plays the sound for the area
under the chestpiece) can be used as a check of the position too.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=listen_at_the_four_heart_sites_with_a_stethoscope.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Listen to the manikin's heart with the stethoscope at the aortic, pulmonic, tricuspid and mitral areas, in that order, holding it on each for 2 seconds, then put the chestpiece back in its tray." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/steth-dream
```

The scripted demo, which reaches full score (open loop from the known layout; the left arm stays at the ready pose).
The right gripper pinches the chestpiece's head by its sides, closing 6 mm inside each side, with the hand leaned 42 deg
back from the diaphragm's axis about the line between its fingers (a straight-down hand cannot reach the chest's 20 cm
height). It lifts the chestpiece 25 cm, clear of the torso, and for each site in turn turns the diaphragm to the skin's
slope there (aimed 4 deg past it about the line between the fingers, which the press turns back), comes down along the
skin's normal from 2 cm, aims the face 8 mm past the skin, holds 3.5 s and lifts off along the normal. Then it carries
the chestpiece back over its tray, lays it down, lets go and backs away. On the four sites it pressed with 4.5, 3.7,
4.0 and 3.2 N (steady within 0.1 N over each 2 s), the disc 1.0, 0.8, 1.3 and 1.5 deg off the skin's slope, the
highest rim point 0.24, 0.46, 0.35 mm into the skin and 0.47 mm above it (within the 0.5 mm seal).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=listen_at_the_four_heart_sites_with_a_stethoscope.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Listen to the manikin's heart with the stethoscope at the aortic, pulmonic, tricuspid and mitral areas, in that order, holding it on each for 2 seconds, then put the chestpiece back in its tray." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/steth-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/steth-dream/*.json; do inspect-robots video "$log"; done
```

`STETH_DEBUG=1` prints, every 5 steps (`STETH_DEBUG_EVERY=1` for every step), the rim's contact force and its peak,
the highest and lowest rim points relative to the skin, how far the disc is off the skin's slope (in all, and about
the two axes), the distance to the next site, how long the current hold has lasted, the sites heard, where the face
is, how close the nearest finger is, whether the chestpiece is in its tray, and where each gripper is.

`STETH_MODE=level` runs the demo the wrong way: the same path and press, but the diaphragm kept level, as it was picked
up, at every site, as a naive straight-down press would. At the aortic area the disc sits 10.5 deg off the skin
(9.7 deg of it the head-to-foot slope), its high side 7.3 mm off the skin while it presses with 7.1 to 7.4 N: the rim
never seals, the aortic area is never heard, and since the sites count only in order the run scores 0. Two other
tries of the right way show how narrow the seal is: aimed 5.5 deg past the slope instead of 4, the disc sits 2.3 deg off
at the aortic area and its rim lifts 0.5 to 0.7 mm (not sealed); aimed 10 mm past the skin instead of 8, the head turns
2-4 deg further in the fingers and the aortic, tricuspid and mitral areas do not seal (score 0 in both tries).

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs, the scripted demo while rendering 1280 x 720
video took 285 s (1,227 steps, 61.4 s of simulated time, ending on success), running beside a Dream-1 trial of 2,400
steps at 320 x 176 that took 438 s. Debug runs of the demo without video (1,500 steps at 320 x 176) took about 10
minutes with three runs side by side. Isaac Sim's first start on a new machine adds a few minutes while it compiles
its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

- An adult chest trainer of about this size lying on its back: an auscultation trainer such as the Life/form
  Auscultation Trainer and SmartScope ([Anatomy Warehouse][aw]: "a three dimensional training torso"; with its
  SmartScope "hear different sounds depending on the location of the stethoscope"; 6,484 USD), or, much cheaper, an
  adult CPR-training torso (about 150 USD), whose firm skin gives a few millimetres under a press. Mark the sternal
  notch and the midline with two small dots of marker.
- A Littmann Classic III stethoscope (about 100 USD), its tubing laid out to the side.
- A steel instrument tray (13 x 10 cm inside or so), fixed to the table with double-sided tape.
- A digital kitchen scale under the trainer, a washable stamp pad, alcohol wipes, and a clear acetate overlay with the
  sternal-notch and midline marks and the four sites, each a circle of 20 mm radius, printed at the positions above.

[openrn]: https://wtcs.pressbooks.pub/nursingskills/chapter/9-3-cardiovascular-assessment/
[shindler]: https://rwjms1.rwjms.rutgers.edu/shindler/choice.htm
[dove]: https://www.dovepress.com/an-in-vitro-acoustic-analysis-and-comparison-of-popular-stethoscopes-peer-reviewed-fulltext-article-MDER
[3m]: https://www.life-assist.com/Content/Docs/3M_Littmann%20Classic%20III%20Overview.pdf
[rn]: https://www.onetonline.org/link/summary/29-1141.00
[lpn]: https://www.onetonline.org/link/summary/29-2061.00
[aw]: https://anatomywarehouse.com/deluxe-auscultation-training-station-a-101406
