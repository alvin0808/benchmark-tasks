# Carry a 1.8 kg school microscope to the bench mat by its arm and under its base, never by the tube, stage or knobs

Fetching the microscope to the bench is how every school biology practical begins, and nearly everyone who has done
one was taught how to carry it: one hand on the arm, the other under the base, never by the eyepiece tube or the
stage. Here a true-size student compound microscope (monocular, LED, 30 cm tall, 1.8 kg) stands on a dark grey rubber
mat, its arm toward the robot, and a blue bench mat lies 18 cm to its right. The instruction asks for the microscope
to be moved onto the blue mat, standing upright with its arm toward the robot as it stands now, and let go; it does
not say how.

What makes it hard is visible in the scene. The microscope is heavy and front-heavy: its weight acts about 6 cm in
front of the arm. One gripper on the arm cannot lift it: the grip turns on the arm and the microscope tips forward
onto the front of its base and drags (see the wrong way below). So the second gripper must hold the base, which stands
flat on the mat with no room under it. It has to be rocked back a little on the back edge of its base (about 10
degrees) so that a finger can slide under the raised side, then both grippers lift it level together, carry it and set
it down square on the other mat, and all the while the tempting handles a single top-down grasp goes for (the eyepiece
tube, the head, the stage, the focus knobs) must not be touched.

The task is the task file [`carry_a_microscope_by_its_arm_and_base.py`](carry_a_microscope_by_its_arm_and_base.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters, and the rule it follows

Carrying a microscope correctly is the first thing taught in a school or college biology lab, and the wording is the
same everywhere:

- "Carry the scope with both hands, one on the arm, one on the base." ([Nassau Community College, Microscope Use][ncc])
- "Remove the microscope from the cabinet by grasping the handle in the arm of the scope with one hand and placing the
  other hand below the base." ([University of Wisconsin, Botany 130, microscope procedure][wisc])
- "use one hand to hold the arm and the other to support the bottom of the base firmly" ([Nikon Eclipse Ei guide,
  Carrying][nikon])
- "If the microscope must be moved, carry it from the base and support arm," because "if picked up by stage or
  eyepiece holder it can become misaligned." ([Technology Networks, How to care for your microscope][tn])
- "never hold the stage, observation tube, or focusing knob when moving the frame" ([Evident (Olympus), How to
  properly and safely transport your microscope][evident])

Anyone watching a clip of a robot carrying a microscope this way knows at once what it is doing and why it is done that
way. The category is Societal relatability. (As a job it would fall short of the 500,000-worker bar: carrying a
microscope is a regular chore for lab technicians and science teachers, but not their core work.)

The microscope is the size and weight of a common school model such as the AmScope M150C (monocular, LED, 40X-1000X;
about 4 lb, 1.8 kg; "Stage Dimensions: 3-1/2" x 3-1/2"" ([AmScope M150C][amscope])). It is modelled at full size
and its full mass, not scaled down, because the weight is what makes one hand on the arm fail; and every hold and
every touch in the scorer is a contact force read from the simulator, not a gripper's position near a part.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the microscope and the mats are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The microscope is built with `bodykit.py`, which
adds bodies made of several shapes and joints between them: importing it hooks `yam_arena.scene` when the simulator
starts, without changing yam_arena. The microscope is one floating PhysX articulation: the base, and on it the arm;
on the arm the focus knobs, the stage and the head, each a rigid body held by a joint locked at its start angle (within
0.05 degrees), so that the simulation can tell which part each gripper touches. This copy of `bodykit.py` turns on
PhysX contact reporting for every part, the base included (`Linkage(report_contacts=True)`), and the task file reads
the contact force between each part and every link of both arms every control step.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Start mat: dark grey rubber, 24 x 16 x 0.3 cm, fixed, centre x 30.0, y 7.5.
- Bench mat: blue rubber, 24 x 16 x 0.3 cm, fixed, centre x 30.0, y -10.5 (2 cm gap between the mats).
- Microscope, loose, 1.80 kg, standing upright on the start mat with its base centred on it and its arm toward the
  robot (back edge of the base at x 21.0):
  - base: white cast metal, 18 x 13 x 3.0 cm (18 cm front to back), on four rubber feet 16 mm across and 4 mm tall;
    1.00 kg;
  - arm: white, 4.0 cm deep and 3.6 cm wide, rising from the back of the base to 20 cm, then reaching forward over the
    stage (to 0.5 cm in front of the base's centre) to carry the head; 0.32 kg;
  - focus knobs: black, coarse 38 mm and fine 24 mm across, on both sides of the arm 7.5 cm up; 0.06 kg;
  - stage: black, 8.9 x 8.9 x 1.0 cm at 12.1 to 13.1 cm, with two clips and a slide, the condenser under it; 0.20 kg;
  - head: white block on the arm's top with a revolving nosepiece and three objectives under it (their tips 2.2 cm
    above the stage), and the eyepiece tube leaning 45 degrees back toward the robot, its top 30.5 cm up; 0.22 kg.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the microscope base's pose (tilt from upright, turn about the vertical, the height of
its lowest corner) and the contact force between each part of the microscope and every link of both arms (PhysX
pair-wise contact forces, the largest per arm).

- 0 for the rest of the run once a rule is broken:
  - any link of either arm touches the stage (with its clips and condenser), the head (with the nosepiece,
    objectives, eyepiece tube and eyepiece) or a focus knob with 0.5 N or more (the parts the sources say never to
    hold);
  - the microscope tilts more than 15 degrees from upright at any time;
  - it drops: its base falls faster than 0.4 m/s (as after a free fall of 8 mm; set down by the arms it moves
    about ten times slower);
  - its base's lowest corner is more than 3 cm above the table for 0.25 s (5 control steps) without one gripper on the
    arm and the other gripper on the base, each pressing on it with 1 N or more: lifted by one hand.
- 0.25: it has been carried properly: its base's lowest corner 3 cm or more above the table while one gripper holds the
  arm and the other the base (each 1 N or more).
- 0.5: after that, it stands upright (within 5 degrees) on the bench mat: its base resting (lowest corner within 5 mm
  of the mat's top), all four corners of the base over the mat, and turned no more than 15 degrees from how it stood.
- 1.0 (success, held 1 s): that, and released (no link of either arm touching any part of it, under 0.1 N) and still
  (its base moving less than 1 mm over 1 s).

Doing nothing scores 0. A real trial is scored from the video and a ruler: a gripper touching the tube, head, stage or
a knob, a tilt past 15 degrees, a drop, or the base more than 3 cm up without one gripper on the arm and the other on
the base are judged from the video (side camera); the base over the mat is checked with a ruler, upright with a phone
level app on the stage, and the release on the video.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=carry_a_microscope_by_its_arm_and_base.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Move the microscope onto the blue bench mat, standing upright with its arm toward you as it stands now, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/scope-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The left gripper comes down behind the
microscope, level and pointing forward, and closes on the arm 15 cm up from either side. It rocks the microscope back
about 10 degrees onto the back edge of its base (it is told 12 degrees; the arm gives under the load), so the front
half of the base rises 2 cm. The right gripper, pointing in from the right side and tilted 75 degrees down from
vertical (its fingers closing nearly vertically), comes level with the raised side 3 cm in front of the base's centre,
slides its lower finger under it and pinches the base's edge. Together they lift it about 5 cm, level (the right hand
turned 9 degrees about each horizontal axis to hold the base level against the weight off its side), carry it 18 cm
to the right, and set it down on the bench mat. The right gripper lets go and draws out sideways, then the left lets
go of the arm and backs away.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=carry_a_microscope_by_its_arm_and_base.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Move the microscope onto the blue bench mat, standing upright with its arm toward you as it stands now, and let go of it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/scope-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/scope-dream/*.json; do inspect-robots video "$log"; done
```

`SCOPE_DEBUG=1` prints, every 5 steps (`SCOPE_DEBUG_EVERY=1` for every step), the microscope's tilt (and which way it
leans), the largest tilt so far, the fastest fall so far, the height of the base's lowest corner, its turn, its
position, the contact force on each part from the left and the right arm, whether it is held properly, carried, over
the bench mat, placed, released and still, any rule broken, and where each gripper is.

`SCOPE_MODE=one` runs it the wrong way: the left gripper alone takes the arm as in the demo and tries to lift it and
carry it to the bench mat. The arm alone cannot lift 1.8 kg held this far behind its centre of mass: the grip turns on
the arm (pressing on it with 17 to 24 N), the microscope tips forward and sideways onto the front corner of its base
and is dragged along the table, never lifted; by step 286 it passes 15 degrees (it reaches 22): 0 for the rest of the
run. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell, the scripted demo while rendering 1280 x 720 video took about 3 minutes (728
steps, 36.4 s of simulated time, ending on success), and a Dream-1 trial of 2,400 steps at 320 x 176 took about 8
minutes, each including Isaac Sim's start (about 20 s) and the video encoding, with another run sharing the GPU.
Debug runs of the demo without video took 4 to 6 minutes on the shared GPU. Isaac Sim's first start on a new machine
adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A student monocular compound microscope of about 1.8 kg (AmScope M150C, about 100 USD, or any similar school model
with an arm handle at the back), two rubber bench mats (24 x 16 cm, or larger mats with the two areas taped out), and
a phone level app. For each trial, stand the microscope on the grey mat, base centred and arm toward the robot, with
the stage lowered and the lowest-power objective in place. A side camera at table height helps to judge the touches,
the tilt and the hands while the microscope is up. Put a soft mat under the work area in case of a drop.

[ncc]: https://ncc.edu/programsandcourses/academic_departments/alliedhealthsciences/microscopeuse.shtml
[wisc]: https://botit.botany.wisc.edu/botany_130/microscope/procedure.html
[nikon]: https://www.microscope.healthcare.nikon.com/guides/ei/en/page/carrying.html
[tn]: https://www.technologynetworks.com/immunology/how-to-guides/how-to-care-for-your-microscope-313068
[evident]: https://lifescience.evidentscientific.com.cn/en/discovery/how-to-properly-and-safely-transport-your-microscope/
[amscope]: https://amscope.com/products/m150c
