# Pit three olives with a plier pitter: olive upright in the cup, squeezed through, stone to a ramekin, olive to a bowl

A plier-style cherry and olive pitter (like the OXO Good Grips Cherry & Olive Pitter; here two die-cast handles hinged at
one end, a cup ring at the other end of the lower handle, a plunger on the upper handle above it, a spring holding
the handles open) lies on its side on the table. Three large green olives with their stones lie on a saucer; a
ramekin for the stones and a bowl for the pitted olives stand by. The instruction asks for the three olives to be
pitted with the pitter, the pitted olives in the bowl and the stones in the ramekin.

Pitting olives or cherries with a hand pitter is a common kitchen job, at home and in restaurant prep. Retailers of
the OXO pitter ([Harrod Horticultural][harrod], [Lakeland][lakeland]) describe its use: the fruit goes in the front
chamber, the handle is squeezed, and the stone drops out through the splatter shield while the fruit stays whole. It
is 14.6-15.5 cm long, about 100 g, die-cast zinc ([John Lewis][jl]). The pitter is held in one hand over a dish;
the other hand sets the olive in the cup, end up; the handles are squeezed all the way so the plunger goes through the
olive and pushes the stone out of the hole in the bottom of the cup; then the pitted olive is tipped out.

## The hand skill

- The pitter lies on the table. It must be picked up by its handles and turned in the hand so that the cup faces up.
  It is held by its handles alone: the spring pushes them apart against the fingers.
- Each olive (22 x 18 mm, slick: friction 0.25) is taken from the saucer, turned end up and set into the cup (a
  19.6 mm bore with a funnel at its mouth) under the raised plunger, without knocking the pitter. To be pitted it must
  sit in the seat, centred within 3 mm, its long axis within 25 deg of the plunger; otherwise the plunger tears its
  side and the stone stays in.
- The handles are squeezed fully shut against the spring (1.9 N m at full stroke: about 20-30 N across the handles,
  within the YAM fingers' roughly 40 N), while the pitter stays over the ramekin so the stone falls into it.
- The pitted olive is tipped out over the bowl, and the pitter is turned back cup up for the next olive.

One hand holds and works the pitter while the other loads the olives, as a person does.

The task is the task file [`pit_three_olives_with_a_hand_pitter.py`](pit_three_olives_with_a_hand_pitter.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: everything is built in these files from boxes, cylinders and capsules.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a joint. The
pitter (a loose base with the upper handle on a sprung revolute joint), the olives, their stones (each held to its
olive by a breakable fixed joint), the saucer, the ramekin and the bowl are built with `bodykit.py`, which adds such
bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and
gives the task file each body's pose and each joint's position. This copy of `bodykit.py` adds two options to its
bodies: a friction combine mode (a wet olive is slick against anything) and a list of bodies a body never collides
with (a stone inside its olive).

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Pitter: loose, 100 g, 14.6 cm long. Handles 2.8 cm wide and 8 mm thick, hinged at one end and held 30 deg open by a
  return spring (3.6 N m/rad). The lower handle ends in the cup ring, 2.8 cm across: a cone seat over a 12 mm hole,
  a 19.6 mm bore 8 mm deep above it and a funnel opening to 25.6 mm at the rim; a clear splatter shield below it
  (drawn only). The upper handle carries a 4 mm plunger over the cup; shut, the plunger's end is about 5 mm below the hole.
  It lies on its side (hinge axis vertical), the hinge at x 12, y -20, the cup end toward +y at y -7.5, the upper
  handle on the far side (+x).
- Saucer: white, 14 cm across, 4 mm thick, at x 20, y 14, loose.
- Olives: three large green olives with stones, each 22 x 18 mm (a capsule), 5.5 g of flesh and a 14 x 8 mm, 1.5 g
  stone inside, lying on the saucer along y at x 16, 20 and 24, y 14.
- Ramekin for the stones: white ceramic, 9 cm across inside, 3.5 cm deep, 250 g, at x 28, y 0, loose.
- Bowl for the pitted olives: 9.6 cm across inside, 4.5 cm deep, 300 g, at x 16.5, y 0, loose.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

What is modelled, and the pitting rule (a stand-in for the plunger cutting through the olive):

- The handles' return spring and their limits (30 deg open, shut).
- The plunger and the upper handle's front arm do not collide with the olive. When the handles are squeezed to within
  1.5 deg of shut (the plunger at full stroke, which takes 1.9 N m against the spring), an olive in the cup is pitted
  if its centre is within 3 mm of the cup's axis and within 4 mm of the seated height and its long axis within 25 deg
  of the plunger. Its stone's fixed joint is then let go (its break force is set to almost nothing, so the stone's
  own weight breaks it) and the stone falls out of the cup's hole. An olive in the cup that is not seated like that is
  torn: its stone stays in for good.
- The olive's flesh is not cut and is not deformed.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the handles' angle, the pose of each olive and each stone and the points of each
finger's collision hulls.

Per olive, 1/6 for each of:

- its stone has been pushed out and lies in the ramekin (inside its rim, still);
- it was pitted (not torn) and lies in the bowl (its lowest point inside the bowl's rim), released (every finger point
  1 cm or more from it) and still.

The score is the sum: 1.0 (success, held 1 s) when all six parts hold. Doing nothing scores 0, and so does putting
unpitted olives in the bowl. A real trial counts the stones in the ramekin and checks each olive in the bowl by eye: a
hole through its ends and no split side (a torn olive keeps its stone or is split down the side).

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pit_three_olives_with_a_hand_pitter.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pit the three olives with the pitter: the pitted olives go in the bowl and the stones in the ramekin." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pitter-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The right gripper comes straight down
over the lying pitter, closes on its handles to 12 deg short of open (the spring then presses them out against the
fingers, about 8 N), lifts it and turns it cup up over the ramekin, its hand level and pitched 8 deg nose down. For
each olive the left gripper pinches it on the saucer from above, lifts it and turns it end up, carries it in over the
cup under the raised plunger and lets it drop 24 mm into the cup's funnel, and backs out. The right gripper squeezes
the handles shut (the stone drops into the ramekin), eases off, carries the pitter over the bowl, tips it nose down
to 118 deg so the pitted olive slides out into the bowl, and turns back. It reaches success at about step 1,110 (56 s).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pit_three_olives_with_a_hand_pitter.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pit the three olives with the pitter: the pitted olives go in the bowl and the stones in the ramekin." \
  --scorer task_score --max-steps 1500 --store-frames --log-dir ~/yam-eval/pitter-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/pitter-dream/*.json; do inspect-robots video "$log"; done
```

`PIT_DEBUG=1` prints, every 5 steps (`PIT_DEBUG_EVERY=1` for every step), the handles' angle, the number of full
squeezes, the cup's axis and position, and for each olive its position, its stone's position, how it sits in the cup
(distance from the axis, height from the seated height, tilt), whether it is pitted, torn, in the bowl and its stone
in the ramekin, how close the nearest finger is, and where each gripper is and how far it is open.

`PIT_SOFT=1` runs the demo the wrong way: the same, but each squeeze is light (the gripper closes to 50 mm instead of
all the way): the handles close only to 11 deg of their 30, the plunger never reaches full stroke, no stone comes out,
and the three unpitted olives are tipped into the bowl. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell Server Edition shared with other simulation jobs: debug runs of the scripted demo
without video (about 1,110 steps, 56 s of simulated time, ending on success) took 8-15 minutes from start to finish,
and the wrong-way run (1,500 steps) about 15 minutes. A Dream-1 trial of 2,400 steps at 320 x 176 took about 7 minutes, and the demo rendered at 1280 x 720 (1,300 steps) about 9 minutes. Isaac Sim's first start on a new machine adds a
few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A plier-style cherry and olive pitter with its cup at the end of the lower handle and the plunger above it (for
example the OXO Good Grips Cherry & Olive Pitter), a saucer, a ramekin about 9 cm across inside and a small bowl about
10 cm across inside, all set out as in the screenshot (the pitter lying on its side, handles open, its cup end toward
the middle). Large green olives with stones from a jar (for example Manzanilla, about 22 x 18 mm), drained and patted
dry at room temperature, three new ones per trial; a jar holds enough for many trials. Empty the ramekin and the bowl
and rinse the pitter between trials.

[harrod]: https://www.harrodhorticultural.com/oxo-good-grips-cherry-olive-pitter-pid8513.html
[jl]: https://www.johnlewis.com/oxo-good-grips-cherry-olive-pitter/p6270521
[lakeland]: https://www.lakeland.co.uk/15379/oxo-good-grips-cherry-and-olive-pitter
