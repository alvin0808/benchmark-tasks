# Start a nail by holding it while tapping it in, then drive it home

Driving a nail is one of the most familiar household jobs: hanging a picture, fixing a loose board, putting up a
shelf or a garden-bed corner. On the table stands a 250 mm offcut of western red cedar 2x6 (38 x 140 mm) on its narrow
edge, held to the table by two steel angle brackets, with a pencil cross on its top edge. A 4d finishing nail
(1-1/2 in.) lies loose on the table beyond the board's left end, and a 7 oz claw hammer lies on its side on the table
to the right. The instruction asks for the nail to be driven into the board at the pencil mark; it does not say how.

The procedure the task checks is the US Army and Air Force hand-tool manual, TM 9-243 / TO 32-1-101 *Use and Care of
Hand Tools and Measuring Tools* ([2007, change 2][tm], section 21.4.1 "Using a Carpenter's Hammer"): "Support a common
nail between thumb and first finger about halfway up the nail." "Tap nail head with hammer face until the nail will
remain standing by itself." "Remove fingers and drive nail flush. Make sure the hand supporting the work is not in
direct line with the hammer blows." A nail stood on its point falls over as soon as it is let go (the wrong-way run
below shows it), so one gripper has to hold the nail upright on the mark while the other taps it in, until it stands by
itself; then the holding gripper has to get out of the hammer's way before the nail is driven home. The task is the
task file [`start_a_nail_and_drive_it_home.py`](start_a_nail_and_drive_it_home.py) for the `yam_arena` embodiment
(scene, nail model, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task uses no third-party assets: every object is built from boxes,
cylinders and spheres in the task file.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or change a joint while
the simulation runs. The board (with its brackets and pencil cross), the nail (point, shank and head) and the hammer
(face, neck, eye, claw and handle) are bodies built from several shapes, and the nail is joined to the board by a joint
the task file changes as it is driven, so `bodykit.py` adds such bodies and joints: importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena, and gives the task file each body's pose. This
copy of `bodykit.py` adds what the nail needs (a joint whose limits are changed at runtime, a body that never sleeps,
continuous collision detection for the hammer, and given centre of mass and inertia for a steel head on a wooden
handle); its docstrings describe them.

## What is modelled and what is not

The objects are true size. The nail is a 4d finishing nail: 1-1/2 in. (38.1 mm) long, 15-gauge shank (0.072 in.,
1.83 mm), 3/32 in. (2.4 mm) head, 0.78 g of steel ([a 4d finish nail's listing][fn]). The hammer is a 7 oz
claw hammer with a hickory handle (Stanley 51-613: 198 g head, 23 mm face, 111 mm head, 286 mm long, about 0.34 kg in
all); its centre of mass and inertia are those of the steel head on the wooden handle. The board is a 2x6 of western
red cedar (specific gravity 0.32 at 12 % moisture, [Western Red Cedar designer handbook][wrc]), on edge so that the
nail driven home stays well inside the wood. A slim finishing
nail in soft cedar is what lets the arm drive it home in about fifteen blows: the arm's strokes reach about 1.6 m/s, a
fraction of a person's swing, and a 4d common nail in spruce would need about forty, more blows than the two-finger
grip on the hammer stands (see below).

The nail's way into the wood is a model, not contact (rigid-body simulators cannot push a nail into wood). Under the
pencil mark the board's collision shape leaves a 20 mm square open, 45 mm deep, under its drawn top face, and the nail
is joined to the board by a joint whose limits the task file sets while the simulation runs:

- When the nail's point touches the wood there (point down, within 30 deg of upright), the point is held where it
  touched, sideways, as a nail's point catches the grain; the nail can still tip over about it and be lifted off.
- The nail goes in only when the hammer's face strikes its head moving toward it at 0.3 m/s or more (a steady push,
  however hard, never drives it); a new blow counts once the face has lifted 8 mm or more off the head (a rebound is
  part of the blow that caused it). The wood resists it with R(d) = 5.7 N/mm x (d + 2 mm) at depth d: 5.7 N per mm is
  the withdrawal resistance of this nail in wood of specific gravity 0.32 by the Wood Handbook's equation for smooth
  nails ([Wood Handbook, chapter 8, eq. 8-1][wh]: p = 54.12 G^2.5 D N per mm of penetration), and measured
  driving-force curves rise with depth at the same slope once the point is in ([Arima et al. 1989][arima]); the extra
  2 mm stands for the point. A blow drives the nail until that resistance has taken 0.7 of the hammer's kinetic energy
  just before the face meets the head (from the simulated hammer's mass, inertia and velocity; only the part of its
  motion along the nail counts). 0.7 is the share of the blow's energy assumed to go into the nail (the rest springs
  back). Driven flush from the surface, the nail takes 4.6 J (6.6 J of blows); a blow stops it at flush.
- Until the nail is 5.5 mm in it has no side support: its tilt is free, so it falls over unless something holds it,
  and if it tips past 25 deg it pops out (its point back at the surface). 5.5 mm is three shank diameters: the point,
  about two diameters long, and as much shank again. From 5.5 mm it stands by itself: its tilt is held where it was.
- A blow whose path or face is more than 15 deg off the nail's axis bends the nail.

These numbers are to be measured on the real board before trials (see Building it for real). The nail does not show
a bend, and the wood does not split. PhysX solves the nail with its largest number of iterations (255): with fewer, a
blow pushed the nail a few millimetres past the depth its joint holds and it sprang back up under the hammer's face,
which a nail in wood does not do.

The robot holds the hammer by its handle choked up, 31 mm from the head's middle (its fingers 0.6 mm clear of the head),
and strikes with straight downward strokes of the whole arm. A two-finger pinch holds little turning load (about 50 N
of grip on pads 2.6 cm wide holds about 1 N m), so it cannot swing the hammer from the wrist. And at each full blow the
hammer slips along its handle in the two fingers, as a real parallel gripper would: each finger already pushes at its
50 N limit, so for a few milliseconds after the face stops the fingers part and the wrist gives. Here the slip is about
1.4 to 2.6 mm a blow (more the closer the grip is to the head) and the face lands about 2 mm further along the handle
each time, so the scripted demo, which is open loop, sends each full blow 2 mm further back than the one before. After
six full blows, with the grip about 47 mm from the head's middle, it lays the hammer on the table and takes it again
16 mm nearer its head (one re-grip), then drives on; the grip is never more than 47 mm from the head's middle.
The arm's joint speed limits in yam_arena's configuration (from i2rt's YAM settings) are 5.5 rad/s for joints 1-3
(DM4340 motors) and 20 rad/s for joints 4-6 (DM4310). In the demo the right arm's joints reach at most 2.7 rad/s
(joints 1-3) and 7.4 rad/s (joints 4-6); its taps meet the head at 0.75 to 0.84 m/s (0.09 to 0.12 J of hammer energy)
and its full strokes at 1.35 to 1.75 m/s (0.32 to 0.54 J); fourteen full strokes drive the nail flush.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. A bent
nail, or the hammer striking a gripper other than the one holding it (its head touching that gripper's fingers or body
while moving faster than 0.2 m/s; for example the gripper holding the nail), sets the score to 0 for the rest of the run.
Otherwise the score is the best stage reached: 0.25 the nail started at the mark (2 mm or more in, its point within
3 mm of the cross's middle); 0.5 standing on its own at the mark (8 mm or more in, upright within 5 deg, no gripper
touching it); 0.75 driven home (its head flush with the wood, within 1 mm, upright within 5 deg, at the mark, no
gripper touching it). Full score: that, with the hammer let go (no gripper touching it, at rest) and both grippers 5 cm
or more from the nail's head, still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts
it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=start_a_nail_and_drive_it_home.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Drive the nail into the board at the pencil mark." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nail-dream
```

The scripted demo, which reaches full score (the left gripper takes the nail from the table, turns it point-down and
holds it upright on the cross, pinching it low on its shank; the right takes the hammer by its handle close to the
head and turns it face-down over the nail; it taps the head three times, until the nail stands by itself; the left
lets go, draws its fingers back off the nail under the hammer and rises clear; the right drives the nail with full
strokes, lays the hammer on the table halfway and takes it again nearer its head, drives the nail flush, lays the
hammer back on its side on the table and draws back):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=start_a_nail_and_drive_it_home.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Drive the nail into the board at the pencil mark." \
  --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/nail-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/nail-dream/*.json; do inspect-robots video "$log"; done
```

`NAIL_DEBUG=1` prints, every 10 steps (`NAIL_DEBUG_EVERY=5` for every 5), the nail's state (loose, on its point, or in
the wood), its depth, whether it is set, where its point is against the cross, its tilt, whether a gripper touches the
nail or the hammer, whether the hammer is at rest, how close the nearer gripper is, the stage reached and any failure,
each gripper's position, the fastest the hammer's face moved down since the last line, where the face is, how the
hammer sits in the right gripper and the right arm's fastest joint speeds; and a line for every blow (its speed,
energy, angles and the depth before and after).

`NAIL_ONE=1` runs the demo the wrong way, one hand at a time: the left gripper stands the nail on its point on the
cross and lets go before the hammer comes. Stood on its point and let go, the nail falls over and off the board at
once, and the hammer's taps then strike the bare mark: score 0.

`NAIL_PUSH=1` checks that a push does not drive the nail: instead of tapping, the right gripper lowers the hammer's
face slowly onto the held nail's head and presses it down for 2 s, then lifts it; the left then lets go.
The face rests on the head and the arm presses down on it for 2 s, but the face never moves toward the nail at
0.3 m/s, so there is no blow and the nail does not go in; when the left gripper lets go it falls over: score 0.

## Runtime per trial

The scripted demo reaches full score at step 930 (46.5 s at 20 Hz) and the run stops there; a Dream-1 trial runs the
full 2,400 steps (2 minutes of simulated time). On an NVIDIA RTX PRO 6000 Blackwell that another job was training on
at the same time, the scripted demo took about 26 minutes at 320 x 176 (no video) and 25 minutes with the 1280 x 720
cameras of the demo video, the `NAIL_ONE=1` and `NAIL_PUSH=1` runs (540 and 560 steps) about 20 minutes each, and
the Dream-1 run 58 minutes.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A 250 mm offcut of western red cedar 2x6 (38 x 140 mm), two steel angle brackets with wood screws to hold it on edge
on the table, a pencil and a square, a box of 4d (1-1/2 in.) finishing nails, and a 7 oz claw hammer with a wooden
handle (Stanley 51-613 or similar). Before every trial: a fresh offcut (or the next 25 mm along the same edge, with the
brackets moved to keep the cross at the same place on the table), a pencil cross drawn across the edge at its middle,
a new nail laid on the table as in the screenshot, and the hammer laid on its side as in the screenshot.

Every criterion is checked with a ruler and a square: the nail started (2 mm or more in) and its point's distance from
the cross's middle (3 mm); standing on its own: 8 mm or more in (its head's top 30.9 mm or less above the wood, of
38.9 mm), and its lean, a square stood on the wood beside it (5 deg is a 2.7 mm gap at the top of the 30.9 mm that
shows); driven home: a ruler laid across the head touches the wood on both sides of it (flush within 1 mm, no edge of
the head proud). Whether the hammer struck the gripper holding the nail, whether a blow glanced off (bending the nail)
and whether the grippers let go and drew back are judged from the overhead and wrist videos.

The nail model's numbers are to be measured on the board before trials, with the same hammer: hold a nail in a cedar
offcut, film taps from known heights at 240 frames per second to get the face's speed, and measure the depth after
each tap with a depth gauge. The depth per blow against the hammer's energy and the depth gives the share 0.7 and the
resistance 5.7 N/mm; the depth at which a tapped nail first stays up when let go gives the 5.5 mm. If they differ,
change `ETA`, `P_N_PER_M`, `D0` and `D_SET` in the task file.

[tm]: https://everyspec.com/USAF/USAF-Tech-Manuals/TO_32-1-101_CHG-2_14466/
[wh]: https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr282/chapter_08_fpl_gtr282.pdf
[wrc]: https://www.realcedar.com/static/fc582921da8c377027b05e5759830a6c/Real_Cedar-Western_Red_Cedar_Designer_Handbook-2016_11_07.pdf
[fn]: https://www.zoro.com/zoro-select-finish-nail-4d-x-1-12-in-l-pk630-4nfc9/i/G2329625/
[arima]: https://doi.org/10.2472/jsms.38.747
