# Draw a 40 mm circle with a school compass, its point kept in the centre mark, then lay the compass down

Drawing a circle with a pair of compasses is a piece of school geometry homework almost everyone has done. A metal
school compass holding a pencil stands on an A4 writing pad, already set to a 40 mm radius, its steel point in the
pricked centre mark and the pencil's point on the paper. The instruction asks for the circle to be drawn round the
mark and the compass then laid down on the desk; it does not say how.

What makes it hard is the hand skill, and it is visible in the scene. A person spins the knurled handle between finger
and thumb. A two-finger gripper cannot roll the handle, so it has to carry the 8 mm handle round a 20 mm circle about
the point while turning with it, holding it lightly enough that the compass can rock in its fingers and both points
stay on the paper, and pressing lightly: pressed hard, the pencil's point breaks or the friction hinge opens and the
circle becomes a spiral; pushed sideways or lifted, the point skids out of its mark. And a wrist turns only about
240 deg, so part-way round the hand has to open round the handle, turn back and take it again while the 32 g compass
balances on its two points. One arm does it; the other is not needed (the pad, 0.30 kg, does not slide under the
compass's light forces).

The task is the task file [`draw_a_circle_with_a_school_compass.py`](draw_a_circle_with_a_school_compass.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters

The compass circle is standard school geometry: the US Common Core standard for high-school geometry
([HSG.CO.D.12][ccss]) reads "Make formal geometric constructions with a variety of tools and methods", compass and
straightedge first among them. Anyone watching a clip knows at once what the robot is doing. School how-to guides describe the hand skill the task tests ([bhanzu.com, "Compass drawing"][howto]):
"hold only the handle (knurled knob) at the top and spin it between finger and thumb, letting the legs hang and
trace"; "let the pencil glide with almost no downward force; darken the line by going around a second time, not by
pressing"; "Don't do this: push the pencil into the paper to get a heavier mark, which flexes the legs and widens the
radius"; "If the hinge is loose, the gap widens as you turn, so the curve spirals outward and never closes"; and
"Don't do this: grip the legs and drag them around; this nudges the gap wider or narrower as you turn."

The compass is modelled on a traditional die-cast metal school compass that holds a pencil, such as the Helix
G05048 ([Rapid Online listing][helix]: "A traditional style die-cast metal compass", "Overall length: 107mm",
"Shortened safety point", "Self-centering handle mechanism", "Cam-lock pencil retaining ring").

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the pad and the compass are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a hinge. The
compass (a head with its handle and needle leg, and a pencil leg on a friction hinge) and the pad (a block with the
pricked hole in its top layer) are built with `bodykit.py`, which adds such bodies and joints: importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena, and gives the task file each body's pose and
the hinge's angle. This copy of `bodykit.py` also turns on PhysX contact reporting for a chosen part (here the pencil
leg), so the task file can read the force between the pencil and the paper, and can give a part PhysX's compliant
(implicit spring-damper) contact through its own physics material: the pencil's point meets the pad on a 3 N/mm
spring with 10 N s/m of damping, standing in for the give of the paper stack under a graphite point. On a rigid
contact the point bounced on and off the paper from one physics step to the next; on the compliant one it presses
steadily, and the force read every control step is the force with which it marks the paper.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- A4 writing pad, loose: 21.0 x 29.7 cm (long side along y), 6 mm thick (50 sheets of 80 g/m2 paper on a 1 mm card
  back), 0.30 kg, friction 0.45 on the desk; x 10.95 to 32.05, y -21.85 to 7.85. A pencilled cross (6 mm) marks the
  centre at x 21.5, y -10.0 (3 cm right of the pad's middle), and the compass point has pricked it: a hole 1.0 mm
  across and 1.5 mm deep in the top sheets.
- School compass, loose, already set to a 40 mm radius: a steel head (14 x 9 x 12 mm) with a dark knurled handle on top
  (8 mm across, 22 mm long; its top 12.4 cm above the desk); the needle leg (a 4 mm bar, then a 1.2 mm needle with a
  0.7 mm point) fixed to the head; the pencil leg on a friction hinge in the head (the hinge holds until 0.10 N m acts
  on it, 1.1 N at the pencil's point across the legs), with a ring clamping a pencil stub (7 mm, yellow, sharpened, its
  graphite point 1.2 mm across); 92 mm from the hinge to each point. Head and needle leg 20 g, pencil leg with pencil
  12 g. It stands on its two points: the steel point in the pricked mark, the pencil's point on the paper 4 cm forward
  of it (x 25.5, y -10.0), the legs in the x-z plane, the hinge's axis along y. The compass stands on its own there.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads where the steel point and the pencil's point are, in the pad's frame with the centre
mark as origin; the contact force between the pencil leg and the pad (PhysX pair-wise contact forces); the compass's
pose; and the points of each finger's collision hulls.

- The pencil draws only while its point presses on the paper with 0.3 N or more (the contact force between the
  pencil and the pad, read every control step). That is a light pencil stroke: Grade 4 children writing with a pencil
  press it on the page with 0.65 to 1.23 N on average ([Schwellnus et al. 2013][force], instrumented stylus on a
  force-sensing tablet), and a pencil at 1.5 N already leaves a full dark mark ([paper test][paper]); at 0.3 N an HB
  point still leaves a light, visible line. The drawn circle is counted in 1-degree steps along the path the pencil's
  point takes round the mark (each degree once; a dot where it stood does not count), wherever it presses with 0.3 N or
  more at a radius within 1.5 mm of 40 mm. The pencil line shown in the cameras follows the same rule.
- 0 for the rest of the run once a rule is broken: the steel point skidded (more than 1 mm from the centre mark while
  its tip is in the paper or less than 0.3 mm above it); the pencil pressed on the paper with more than 5 N (its point
  breaks); the pencil drew a line more than 3 mm off the 40 mm circle (a spiral or a stray mark).
- Otherwise 0.8 x the fraction of the circle drawn (degrees drawn / 360).
- 1.0 (success, held 1 s): at least 355 deg of the circle drawn; the compass lying on its side (the hinge's axis within
  30 deg of vertical, the head within 3 cm of the desk); every finger point 2 cm or more from its handle and legs; and it still (the
  head moving less than 1 mm over 1 s).

Doing nothing scores 0. A real trial is scored from the sheet and the video: lay a transparent overlay printed with
two circles of radius 38.5 and 41.5 mm (and a 3 mm band beyond them) on the drawn circle, centred on the prick; the
line inside the band is read off with the overlay's degree scale; a second prick or a scratch from the point, a broken
pencil point, or a line outside the 3 mm band each scores 0; the compass lying on the desk, let go, is checked on the
video.

The pencil line is shown in the cameras by dark marks (visual only; nothing collides with them), one for each 15 deg of
the circle, laid on the paper once the pencil's point has drawn the whole of that 15 deg. They are drawn 1.2 mm wide,
wider than a real pencil line, so the cameras show it.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=draw_a_circle_with_a_school_compass.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Draw the circle around the centre mark on the pad with the compass standing there (it is already set to a 40 mm radius), then lay the compass down on the desk." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/compass-dream
```

The scripted demo, which reaches full score (open loop from the known layout; the left arm stays at the ready pose).
The right gripper comes down over the handle, fingers closing across the legs' plane, and holds the handle lightly
(commanded 3 mm narrower than the handle), so the compass can still rock between the fingers and both points stay on
the paper; it presses 3 mm down, so the pencil marks with 0.3 to 1.8 N (0.8 N typical). It carries the handle round the point on the 20 mm circle the handle must follow, the
hand turning with the legs: first back 50 deg, then forward 190 deg, aiming the handle 12 deg ahead of where the legs
should be (the arm is soft sideways and stays short of a target it pushes against) and waiting 1.5 s at each end. It
opens to 14 mm round the handle, turns the open hand back 180 deg in 1.5 s (the compass stands on its point meanwhile,
the open fingers on either side of it most of the way), closes on the handle again, turns the hand 5 deg about the axis
across the legs (after the regrasp the arm's give otherwise eases the pencil off the paper) and carries it on another
215 deg, past where the line began. Then it lifts the compass straight up out of the mark, carries it to the desk beside
the pad (x 20, y -25.5), stands it on its points there, tips it over on them 60 deg, lets go, and backs away.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=draw_a_circle_with_a_school_compass.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Draw the circle around the centre mark on the pad with the compass standing there (it is already set to a 40 mm radius), then lay the compass down on the desk." \
  --scorer task_score --max-steps 1500 --store-frames --log-dir ~/yam-eval/compass-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/compass-dream/*.json; do inspect-robots video "$log"; done
```

`COMPASS_DEBUG=1` prints, every 5 steps (`COMPASS_DEBUG_EVERY=1` for every step), the pencil's force on the paper,
its radius and the range of radii drawn, how far the steel point is from the mark and how high it sits, the pencil
point's height above the paper, the hinge's angle, the degrees of circle drawn, how far the compass is from lying on
its side, how close the nearest finger is, how far the pad has moved, where each gripper is and any rule broken.

`COMPASS_MODE=press` runs the demo the wrong way: the same path, pressing 10 mm down instead of 3 mm. Pressed 10 mm down, the compass bears hard on both its points (the pencil up to 4.4 N, the hinge giving 0.3 deg) and the handle is squeezed down in the fingers; when the hand eases off to let go of the handle part-way round (step 606, 118 deg drawn), the compass springs out of its mark and the point skids 10.7 mm across the paper: 0 for the rest of the run. Score 0.
`COMPASS_MODE=twist` turns the hand on the spot over the handle instead of carrying it round the point. The handle turns in the grip and the compass twists and tips instead of going round: after 20 deg of line the pencil lifts off the paper (11 mm up by the end). Score 0.04.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs, the scripted demo while rendering 1280 x 720
video took 525 s (1,364 steps, 68.2 s of simulated time, ending on success), running beside a debug run of the task;
a Dream-1 trial of 2,400 steps at 320 x 176 took 914 s (0.38 s per step). Debug runs of the demo without video took 8
to 15 minutes on the busy GPU. Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

Any traditional metal school compass that holds a pencil (Helix G05048 or similar, about 5 USD) with a short sharpened
HB pencil, set to 40 mm against a ruler, the hinge screw tightened so the legs hold their angle firmly; an A4 writing
pad (50 sheets) with a cross pencilled 3 cm right of its middle and pricked once with the compass point; the overlay
for scoring printed on a transparency. For each trial put a fresh sheet on top (or tear off the drawn one), prick the
centre once, stand the compass with its point in the prick and its pencil 4 cm forward of it, and start.

[howto]: https://bhanzu.com/math/geometry/compass-drawing
[ccss]: https://www.thecorestandards.org/Math/Content/HSG/CO/
[force]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3722657
[paper]: https://bleistift.blog/2016/08/comparing-different-types-of-paper/
[helix]: https://rapidonline.com/helix-g05048-compass-graduate-pencil-34-2040
