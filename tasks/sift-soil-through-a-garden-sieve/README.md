# Sift potting soil through a two-handled garden sieve over a tub, keeping the pebbles in the sieve

A square garden sieve full of potting soil, with six pebbles in it, rests on two battens laid across a garden tub. The
instruction asks for the soil to be sifted into the tub with the pebbles kept in the sieve, and for the sieve to be left
resting on the battens; it does not say how. Gardeners do it by lifting the sieve off the battens by its two handles,
shaking it level over the tub until the soil has gone through the mesh, and setting it back. The sieve is wider than a
gripper can span (24.8 x 21.6 cm outside, against a 9.4 cm gripper opening), with a flat handle on each side, their
middles 24 cm apart. Only with both handles held can it be lifted off the battens, shaken level and set back. Lifted
by one handle it stays propped on its other side: rocked front to back that way it walks off the battens into the
tub, and rocked side to side it stays on them but sifts less than half the soil (see `SIEVE_ONE` below). The task is
the task file
[`sift_soil_through_a_garden_sieve.py`](sift_soil_through_a_garden_sieve.py) for the `yam_arena` embodiment (scene,
scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. No third-party assets are used: the tub, the battens,
the sieve, the soil crumbs and the pebbles are built from boxes and spheres in the task file, and the task needs no
models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres. The sieve (a frame, a mesh of bars and two handles)
and the soil crumbs are loose rigid bodies built from several shapes, so `bodykit.py` adds such bodies: importing it
hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each body's own pose.
It can also give a body gentler contact settings, which the crumbs and the pebbles use (below).

## What is modelled and what is not

- The sieve: one rigid body, 0.6 kg; a frame 20 cm square inside and 7 cm tall (front and back walls 2.4 cm thick,
  resting on the battens; side walls 8 mm) over a mesh of 4 mm bars with 14 mm square holes, and a flat handle
  6 x 2.5 x 1 cm out of the middle of each side wall, 2.5 cm above the mesh.
- The soil: 72 rigid crumbs shaped as 2 x 1 x 1 cm sticks, 3 g each (216 g), in two layers on the mesh. Round crumbs
  smaller than the holes would drop straight through, while damp real soil only passes when shaken; the sticks lie
  across the bars until shaking tips them end-first through a hole. Four of them drop through while the scene settles,
  as some real soil does while it is spread on the mesh.
- The pebbles: six rigid spheres 2.6 cm across, 30 g each, laid on the soil; they roll down onto the mesh while the
  scene settles.
- Crumbs and pebbles have gentler contact settings: PhysX's maximum linear velocity set to 0.8 m/s, overlapping
  pieces pushed apart at most at 0.3 m/s, and 32 solver iterations. Without them the physics engine threw crumbs and
  pebbles jammed between the bars, a pebble and a wall out of the sieve at up to 11 m/s, which no shaken soil does
  (seen with `SIEVE_DEBUG_FAST=1`). With them, kicks of a few m/s still happen now and then, but in the eight runs that
  shook the sieve (the demo, its variants and the one-arm runs) no crumb and no pebble was thrown over the walls.
- The tub and the battens are fixed.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 2 s, otherwise the score at the end of the run. The task
file follows the run itself, so a broken rule sets the score to 0 whatever was earned before it.

Soil counts only once it has gone down through the mesh. The task file follows every crumb in the sieve's own frame: a
crumb is in the sieve while its centre is above the mesh and inside the walls; going below the mesh from there means it
went through the mesh; going outside the walls means it left over a wall, and it is never credited. The crumbs that
drop through while the scene settles count as through the mesh. The measure is the mass of the soil that went through
the mesh and lies in the tub, as a share of the 216 g on the sieve at the start (each crumb is 3 g).

- 0 for the rest of the run once any pebble has left the sieve (its centre outside the walls: lifted out, onto the
  table or into the tub), or once any soil has left the sieve over its walls.
- Otherwise, at the end of the run: 0.25 for 25 % or more of the soil through the mesh and in the tub (54 g; 18 of the
  72 crumbs); 0.5 for 50 % or more (108 g; 36 crumbs).
- 1.0: 70 % or more (151.2 g; 51 crumbs), all six pebbles in the sieve, and the sieve resting back on the battens (its
  middle within 2 cm of where it started and within 1.5 cm of its starting height, tilted less than 6 degrees), let go
  (both grippers' centres 3 cm or more from every part of the sieve) and still, for 2 s.

Doing nothing scores 0 (the 4 crumbs that drop through while the scene settles are 12 g, 6 %). `-E object_jitter=0`
keeps every object exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=sift_soil_through_a_garden_sieve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sift the soil into the tub, keeping the pebbles in the sieve, and leave the sieve resting on the battens." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/sift-dream
```

The scripted demo, which reaches full score (each gripper comes in level from its own side with its fingers upright,
slides onto its handle from the front and clamps it from above and below; together they lift the sieve about 2 cm off
the battens, shake it round in small circles (3 cm radius commanded) three times a second for 20 s, reversing every
2 s, set it back on the battens, let go, slide off the handles to the front and rise clear):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=sift_soil_through_a_garden_sieve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sift the soil into the tub, keeping the pebbles in the sieve, and leave the sieve resting on the battens." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/sift-demo
```

It sifts 67 of the 72 crumbs (201 g, 93 %) through the mesh into the tub, with no soil and no pebble out over the walls,
and sets the sieve down about 3 mm from where it started; success holds from step 727 and the run ends at step 767.

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/sift-dream/*.json; do inspect-robots video "$log"; done
```

Debug and test switches (environment variables, read by the task file):

- `SIEVE_DEBUG=1` prints, every 10 steps (`SIEVE_DEBUG_EVERY=5` for every 5), where the sieve is and how far it is
  tilted, how many crumbs went through the mesh and how many of those lie in the tub, how many crumbs are in the tub by
  any way, how many left over the walls, whether a pebble has left the sieve, whether the sieve is back on the battens,
  how far each gripper is from the sieve, and the score so far. `SIEVE_DEBUG_FAST=1` also prints, every step, any crumb
  or pebble thrown upwards inside the sieve.
- `SIEVE_ONE=1` runs the demo the wrong way, with the left arm alone: lifted by its left handle, the sieve stays propped
  on its right side on the battens. Rocked round in circles that way it sifts 63 crumbs, but it walks off the front
  batten and drops into the tub, tilted 21 degrees, and is not set back: 0.5. With `SIEVE_AMP=0.001 SIEVE_AMP_Y=0.03`
  added (rocked side to side only) it stays on the battens but sifts 28 crumbs (84 g, 39 %): 0.25.
- `SIEVE_PEBBLE_OUT=1` replays the first loophole: before the sift, the left gripper lifts a pebble out of the sieve
  and lays it on the table; then both arms sift the soil (68 crumbs through the mesh) and set the sieve back. It scores
  0, as the pebble left the sieve; by its rules the previous scorer, which only looked for pebbles in the tub, would
  have given it 1.0.
- `SIEVE_POUR=1` replays the second loophole: instead of shaking, the arms lift the sieve and turn it 120 degrees about
  its handles, back wall down, so that the soil pours out over the wall into the tub, then set it back. With
  `SIEVE_NO_PEBBLES=1` (the pebbles left out of the scene, as if they had all been taken out to the table first) it
  puts 46 crumbs in the tub, 39 over the wall and 7 through the mesh, and scores 0; the previous scorer counted every
  crumb in the tub and would have given it 0.5.

## Scoring a real trial

The soil is prepared the same way for every trial: multipurpose potting compost from one bag, rubbed through a 10 mm
garden sieve so that no piece is larger than the mesh holes, spread out to air-dry for two days, then mixed with half
its mass of water (for example 600 g of compost and 300 g of water) and kept in a closed box. It is then damp and
crumbly: a squeezed handful holds together and breaks up when poked. For each trial the tub is empty, 216 g of this soil
is weighed onto the mesh on a kitchen scale and spread evenly over its middle, about 1.5 cm deep (whatever drops through
meanwhile stays in the tub), and the six pebbles (2.5 to 3 cm across, about 30 g each) are laid on it where the
screenshot shows them. The sieve sits on the battens at marks drawn on them.

During the run, the overhead and wrist videos show whether any pebble or any soil left the sieve over its walls: either
scores 0 for the trial, since soil poured over the rim cannot be told apart from sifted soil once it is in the tub.
After the run, all the soil inside the tub is brushed into a bowl and weighed on the kitchen scale (1 g steps, bowl
tared). Its share of the 216 g gives the score: 0.25 for 54 g or more (25 %), 0.5 for 108 g or more (50 %), and 1.0 for
151.2 g or more (70 %) if all six pebbles are in the sieve and the sieve rests back on the battens, let go, its middle
within 2 cm of its marks, its walls within 1.5 cm of the battens' tops and tilted less than 6 degrees (a phone level on
its frame). Afterwards all the soil, from the tub, the sieve, the battens and the table, goes back into the box.

## Runtime per trial

On one RTX PRO 6000 (Blackwell) that other jobs kept at full load: about 13 minutes for the scripted demo while
rendering 1280 x 720 video (767 steps, 38 s of simulated time; 803 s), and about 30 minutes for a Dream-1 trial of
2,400 steps (1,820 s). On the idle GPU the first version of the task took about 4 minutes for the demo and 6 minutes
for Dream-1. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A square garden sieve (riddle) 20 cm across inside, with walls 7 cm tall, a mesh of 4 mm wire with 14 mm square holes
and a flat handle on two opposite sides (or a wooden frame built to these sizes), a garden tub or tray 28 cm across and
6 cm deep inside, two battens 2 x 32 x 1.2 cm laid across it, six pebbles 2.5 to 3 cm across, the soil prepared as
above, and a kitchen scale.
