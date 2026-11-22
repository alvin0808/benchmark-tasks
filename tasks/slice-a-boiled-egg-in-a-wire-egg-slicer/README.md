# Slice a peeled boiled egg in a hinged wire egg slicer, laid lengthwise in its cradle, and swing the frame back open

A hinged wire egg slicer (die-cast aluminium cradle 140 x 80 mm with ten stainless-steel cutting wires on a frame
hinged at one end, like the Westmark "Justus" egg slicer) stands on a non-slip mat with its frame swung open. A peeled
hard-boiled egg lies on a saucer beside it, crosswise to the cradle. The instruction asks for round slices, with the
frame swung back open and the slices left in the slicer.

Slicing a boiled egg in a wire slicer is an everyday kitchen job (egg salad, sandwiches, salads, garnish) that most
people have done or watched. The maker describes the tool as cutting "the hard-boiled egg quickly and cleanly into
round or oval slices without breaking the yolk" ([Westmark Justus][justus]); for round slices the egg lies lengthwise
in the cradle's egg-shaped hollow. The usual way: lay the egg in the hollow, swing the wire frame over and press it
down through the egg in one stroke until it closes on the cradle, then swing the frame open again to free the slices
(they are then taken out or tipped onto a plate).

## The hand skill

- The peeled egg is slick and rounded. It lies across the robot's reach on the saucer and has to be picked up and
  turned a quarter turn, then set lengthwise and centred in a hollow only 6 mm deep. A shallow contact can be
  corrected by lifting the wire frame and reseating the intact egg.
- The frame is taken by its tab (30 x 32 mm, 3 mm thick) and swung over through 95 deg along its arc, then driven
  down through the egg against the wires' cutting resistance, about 9 N at the tab, until it closes on the cradle;
  then it is swung back open past upright onto its stop, without knocking the eleven loose slices out of the
  cradle. The fingertips hold the tab by its two side edges, and the tab turns between them as the frame swings.

Two arms are not required (the demo uses the right arm for the egg and the left for the frame, because each is the one
that reaches its part comfortably).

The task is the task file
[`slice_a_boiled_egg_in_a_wire_egg_slicer.py`](slice_a_boiled_egg_in_a_wire_egg_slicer.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the slicer, the egg and the saucer are built in these files from boxes, cylinders and
triangle meshes computed in the task file.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a joint. The
cradle (a fixed body whose top, with the hollow, is a triangle mesh), the wire frame (a linkage with one revolute
joint), the egg's eleven slices (each a convex mesh, joined by fixed joints that can break) and the saucer are built
with `bodykit.py`, which adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives the task file each body's pose and each joint's position. This copy of
`bodykit.py` lets a sprung joint cap its drive's force (`effort_limit`). Every control step the task file sets the hinge drive's force limit and target through Isaac Lab's
articulation API and, when the egg is cut, changes the slices' joints through USD, as described below.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Egg slicer cradle, fixed (it stands on a non-slip mat): die-cast aluminium, 8 cm (x) by 14 cm (y), 2.2 cm tall,
  centred at x 21, y 4. In its top an egg-shaped hollow (the lower part of the egg, 1 mm clearance, 6 mm deep) at its
  centre, its long axis along y, and slots for the wires across the bed.
- Wire frame: hinged across the cradle's +y end (axis along x, 2.45 cm above the table, at y 10.4). Bars 4 mm square
  round an opening 8.4 cm wide and 12.6 cm long; a tab 3.0 cm wide, 3.2 cm long and 3 mm thick bent up at the free
  end; ten stainless-steel wires across it, 5.5 mm apart. Closed, the wires lie in the cradle's slots 1 mm below the
  hollow's floor, at y 4 + (-2.475, -1.925 ... 2.475) cm. 40 g. It starts swung open 95 deg, just past upright,
  resting on its stop. The hinge turns freely (0.03 N m s/rad of drag).
- Egg: peeled, hard-boiled, US large: 5.7 cm long, 4.3 cm across, 50 g. It lies on the saucer with its long axis
  along x, its centre at x 13, y -10.
- Saucer: 10 cm across, 6 mm thick, centred at x 13, y -10, loose.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

What is modelled:

- The egg is eleven slices, cut at the planes where the wires pass through an egg lying centred in the hollow, held
  together by fixed joints (they part only at 400 N) until it is cut. Each slice is the convex hull of its part of the
  egg (an ellipsoid), its cut faces set back 0.15 mm. A slice collides with every other slice except its two
  neighbours (PhysX does not collide a jointed pair).
- Cutting: the task tests each of the ten finite wire spans for intersection with the egg at its measured position
  and orientation. While a wire is in the egg, the hinge drive can apply up to 1.2 N m, about 9 N at the tab
  13.5 cm from the hinge. A round cut needs a seated egg, contact with all ten wires during the closing stroke and
  a frame angle within 1.5 deg of closed. That stroke weakens the ten slice joints to the light cling of moist cut
  faces (0.3 N or 0.8 mN m). Opening the frame 4 deg from its low point resets the stroke. Leaving the seat also
  resets the stroke.
- Integrity: before slicing, the scorer measures the positions of the two sides of each slice joint. An opening of
  at least 1.5 mm indicates actual separation of the modelled egg. Wire contact with a misaligned egg leaves its
  integrity score alone. The rigid model supports round cuts at the manufactured wire planes. It approximates
  resistance and omits small dents and off-axis oval cuts.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the frame's angle, each slice's pose and speed, and the points of each finger's
collision hulls. "In the cradle": a slice's centre over the cradle's 8 x 14 cm top and at most 4.5 cm above the table.

- 0.25: the intact egg has lain in the hollow, lengthwise and centred (its long axis within 20 deg of the hollow's,
  its horizontal centre within 6 mm of the hollow's centre and no more than 4 mm above its seated height).
- 0.5: a full stroke made all 11 round slices, including the two rounded end caps.
- 1.0 (success, held 1 s): as 0.5, with the frame swung back open (85 deg or more), all 11 slices in the cradle, every
  finger point 1 cm or more from every slice, and all of them still (under 2 cm/s).

The score is 0 if the egg comes apart before a valid cut. Doing nothing also scores 0.

For a real trial, film the cradle from overhead and from the side and use a ruler in each view. Compare the egg's
position and axis with the hollow in those recordings, and check the side view for complete frame closure. A
shallow nick is allowed if the egg remains in one piece and the final slices are intact. A detached piece or a
through-tear that separates the egg before the valid cut scores 0. A complete stroke earns cutting credit when
inspection shows nine intact round interior slices and two intact end caps (eleven pieces total, with no loose
fragments). For each interior slice, the widest diameter should be at most 1.1 times the narrowest. Use a close end
photo and take a face-on inspection photo with a ruler if the shape is unclear. For full credit, the frame is at least
85 deg open, all eleven pieces are left in the cradle, and both hands stay at least 1 cm clear with the slices settled
for the final second. Check that hold on video before moving any piece for inspection.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=slice_a_boiled_egg_in_a_wire_egg_slicer.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Slice the peeled egg into round slices, then swing the slicer's frame back open, leaving the slices in it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/eggslicer-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The right gripper, straight down with
its fingers across the egg, picks the egg up off the saucer, turns it a quarter turn on the way and sets it in the
hollow, lengthwise, lets go and goes back to its ready pose. The left gripper, straight down with its fingers closing
along x, takes the frame's tab by its two side edges with its fingertips and swings the frame over along its arc (the
tab turns between the fingertips) onto the egg and on to 8 deg past closed, so that the arm presses the wires through;
it holds it closed 1.5 s, swings it back past upright (it stops at 95 deg on its stop), lets go and goes back. It
reaches success at about step 785 (39 s).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=slice_a_boiled_egg_in_a_wire_egg_slicer.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Slice the peeled egg into round slices, then swing the slicer's frame back open, leaving the slices in it." \
  --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/eggslicer-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/eggslicer-dream/*.json; do inspect-robots video "$log"; done
```

`EGG_DEBUG=1` prints, every 5 steps (`EGG_DEBUG_EVERY=1` for every step), the frame's opening angle, whether the hinge
is resisting (the wires in the egg), whether the egg lies in the hollow, the largest opening between adjacent
uncut sections, how many wires have been tracked in the current stroke, and whether the egg has been cut. It also
reports the egg's centre and axis, the number of slices in the cradle and their spread, and the nearest finger
clearance.

`EGG_CROSS=1` finishes the trial with a crosswise stroke, which does not produce the round cuts. `EGG_RECOVER=1`
first contacts the wires with the egg crosswise, then reopens the frame, puts the same egg lengthwise in the
cradle and completes the normal stroke. Use `--max-steps 2200` for the recovery run.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell Server Edition shared with other simulation jobs: debug runs of the scripted demo
without video (ending on success at about step 785, 39 s of simulated time) took about 2 minutes from start to finish,
the revised demo rendered at 1280 x 720 took 2 min 8 s, and the revised Dream-1 trial of 2,400 steps at 320 x 176 took
5 min 37 s. The scripted demo scored 1.0. Dream-1 scored 0.0 after approaching the saucer and leaving the egg on it.
Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A hinged wire egg slicer with an egg-shaped hollow and about ten wires (for example the Westmark "Justus", 140 x 80 x
30 mm, die-cast aluminium; any hinged wire slicer of this kind will do), set on a thin non-slip mat with its hinge
toward the robot's left, the hollow's centre 21 cm in front of the midpoint between the arm bases and 4 cm to the
left; a saucer about 10 cm across at x 13, y -10. Eggs: US
large, hard-boiled (10 minutes in boiling water, then into iced water for 5 minutes), peeled, kept in the fridge (4 C)
and used within the day; one egg per trial, laid on the saucer lengthwise along x just before the trial. The egg is
moist: thin food-safe nitrile covers on the finger pads keep the pads clean; wipe them between trials.

Before collecting trials, check the particular slicer's resistance and stability with three eggs of the specified
size and temperature. On a kitchen scale, tare the slicer, mat and seated egg. Press the tab vertically about
13.5 cm from the hinge and record the largest scale reading before the frame hits its stop. The simulated
1.2 N m corresponds to about 9 N or 0.9 kg on the scale near closure. This is an estimate, so record the actual
readings. Set the mat directly on the trial table and mark the cradle's corners with removable tape. Repeat the
stroke and check that the cradle stays within 2 mm of the marks and does not tip. Adjust the mat if it slips.
Before each trial clear the slices out, wipe the cradle and swing the frame open past upright.

[justus]: https://www.culinaris.de/en/Westmark-Egg-divider-Justus/31305
