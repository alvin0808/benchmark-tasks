# Refit a plug-in coil burner after cleaning

The table carries a true-size section of an electric coil range's cooktop round one burner well: a white enamelled steel top 7.5 cm
above the table with an 8-inch burner well. The well's chrome drip bowl has just been washed and dropped back into its
recess a quarter turn wrong, so the opening in its side faces the back of the cooktop instead of the plug-in receptacle,
which is hidden under the cooktop's edge on the robot's right of the well. The 8-inch plug-in coil element lies on the
cooktop to the left of the well, its terminals pointing toward the well. The instruction asks for the coil burner to be
put back so that it is plugged in and lies flat in its drip bowl; it does not say how.

The procedure is the range maker's. GE electric range owner's manuals, "Plug-in surface units" (for example the
[JSC57 manual, page 26][jsc57], the [JBP15 manual, page 22][jbp15] and the [JBC16GK manual, page 20][jbc16]):

> "Replace the drip pan into the recess in the cooktop. Make sure opening in the pan lines up with the receptacle."
> "Insert the terminals of the surface unit through the opening in the drip pan and into the receptacle."
> "Guide the surface unit into place so it rests evenly."

and, on taking a unit out: "Lift the surface unit about 1 inch above the drip pan and pull it out. Do not lift the
surface unit more than 1 inch. If you do, it may not lie flat on the drip pan when you plug it back in." and "Do not bend
the surface unit plug terminals."

What makes it hard, from the objects themselves:
- The receptacle cannot be seen: it sits under the cooktop's edge, 12.2 cm from the well's centre, its two slots 1.3 cm
  below the cooktop's top. The terminals can only reach it sideways, through the bowl's opening and under the edge.
- So the coil cannot simply be set down: lowered straight into the well, its terminals land on the cooktop's edge and it
  lies 6 mm high, tilted 6 deg and unplugged. It has to come in tipped up a few degrees at the far side (about 1.5 cm, so its
  far edge clears the recess while it is still 3 cm back from home), be pushed in along the terminals' line until they
  are home (each 4.8 mm terminal 2 cm or more into a 6.5 mm slot, past a spring contact that it pushes aside), and then
  be let down about its terminals until it lies flat on its legs.
- With the bowl's opening turned away, the leads and terminal bracket hit the bowl's side and never reach the
  receptacle, so the bowl has to be turned first, by sight, from the opening in its side.

One arm can do it all; the scripted demo uses the right arm to turn the bowl (its closed fingertips in the bowl's
opening, carried round a quarter turn) and the left arm for the coil (its closed fingers down into the coil's 52 mm
middle, opened against the inner turn, so the coil can be tipped and pushed from the middle). The second arm is not
needed: the bowl lies still in its recess while the coil goes in.

The task is the task file
[`refit_an_electric_coil_burner_after_cleaning.py`](refit_an_electric_coil_burner_after_cleaning.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task uses no third-party assets: every object is built in the task file
from boxes, cylinders and capsules.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a spring. The
cooktop, the drip bowl and the coil element are bodies built from many shapes, and the receptacle is an articulation
whose two spring contacts are leaves on sprung hinges, so `bodykit.py` adds such bodies and linkages: importing it
hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each body's pose and
each hinge's angle.

## What is modelled and what is not

- Cooktop section (fixed): a 3 mm steel top, its top 7.5 cm above the table, x 6 to 40 cm, y -25 to 33 cm (from the
  midpoint between the arm bases), with a round recess hole 22.6 cm across centred at x 20.5, y -5.0, and its housing
  down to the table.
- Drip bowl (loose, 0.20 kg): an 8-inch chrome drip bowl (like GE WB32X10014): flange 24.8 cm across resting on the
  top round the hole; a sloped side from 22.2 cm across at the top to 17.0 cm at the floor, 4.0 cm deep; a 5.0 cm
  centre hole with a 6 mm collar; the opening in its side 4.3 cm wide and 2.6 cm deep. Sheet steel 1.2 mm in
  collision; 48 flat segments round.
- Receptacle (fixed, under the cooktop's edge on the well's right, its face 12.2 cm from the well's centre, the
  terminals' line 1.3 cm below the cooktop's top): two slots 6.5 mm wide, 10 mm tall and 3.2 cm deep, 22 mm apart,
  their floors 1 mm under a seated coil's terminals, each with a mouth flaring 3 mm all round over its first 5 mm. In
  each slot a spring contact: a brass leaf along the slot's side, hinged 8 mm in on a 0.4 N m/rad torsion spring and
  slanting into the slot so that its deep end (20 mm in) stands 4.2 mm from the far side; a 4.8 mm terminal pushes it
  about 0.9 mm aside (about 2.5 N). The receptacle is a PhysX articulation (a fixed base and two sprung leaves). These
  sizes are estimates for a GE plug-in receptacle (GE publishes no drawing): measure the real one.
- Coil element (loose, 0.55 kg): an 8-inch plug-in coil (like GE WB30M2: 7 7/8 in across, 9.75 to 10 in overall with
  its terminals): five turns of 8 mm sheathed tube, the outer turn 20.8 cm across, a 52 mm opening in the middle (the
  spiral is modelled as five rings); a three-legged support under the turns standing 4.3 cm; two leads dropping 17 mm to
  a terminal bracket (32 x 8 x 12 mm) with two terminals 4.8 mm across, 22 mm apart, 29 mm long with rounded tips;
  25.2 cm from the far side to the terminals' tips. Friction 0.8 (a rubber-padded finger on the sheath; estimate);
  0.2 in the receptacle (estimate).
- Not modelled: electricity and heat (the range is off and cold), the trim ring some ranges have, bending of the
  terminals, the receptacle's own float in its bracket.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run, which is
the stage reached:

- 0: the bowl's opening more than 15 deg from the receptacle's direction, or the bowl not resting flat in its recess
  (tilt 2 deg or more, flange 2 mm or more off the top, or off centre by 6 mm or more).
- 0.25: the bowl right, the coil not plugged in.
- 0.5: the bowl right and both terminals in the receptacle (each tip 20 mm or more past the receptacle's face, and each
  spring contact's deep end pushed aside 0.4 mm or more by its terminal), but the coil not flat on its legs in the
  bowl, still held, or not yet still.
- 1.0 (success): the bowl right, plugged in, the coil flat (tilt under 2 deg) and resting on its legs in the bowl (its
  centre within 2 mm of its seated height), every point of both grippers' fingers 2 cm or more from the coil, and the
  coil still (moved under 1 mm in 1 s), held for 1 s.

Nothing is latched: the manual's warning that lifting a plugged-in unit more than 1 inch can damage the receptacle is
about repeated lifting, so it is not scored. In a real trial: the bowl's opening is checked against a mark at the
receptacle's direction (15 deg either side); plugged in is checked by eye under the
edge with a torch and by the coil's position, its centre within 6 mm of the well's centre toward the receptacle (a tape
measure from the bowl's rim on both sides); flat with a small spirit level laid across the coil both ways; let go and
still on the video.

`-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=refit_an_electric_coil_burner_after_cleaning.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put the coil burner back in its burner well so that it is plugged in and lies flat in its drip bowl, then let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/coil-dream
```

Dream-1 scored 0.0: its left gripper went down onto the coil on the cooktop and pushed at it, but never lifted it; its
right gripper stayed by the cooktop's right edge; the bowl was never turned.

The scripted demo reaches full score (open loop, from the known layout): the right gripper, fingers closed, goes down
into the bowl's opening and carries it round a quarter turn, then goes back to the ready pose; the left gripper puts
its closed fingers into the coil's middle, opens them against the inner turn, lifts the coil, carries it over the well
tipped up 3 deg at the far side, lowers it with the terminals in line with the bowl's opening, pushes it along the
terminals' line until they are home, lets the far side down about the terminals, closes its fingers off the inner turn
and moves away (success at step 748):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=refit_an_electric_coil_burner_after_cleaning.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "scripted demo" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/coil-demo
```

Add `-E record_video=true` to either command for a video. `COIL_DEBUG=1` prints a `DBG` line every 5 control steps
(the bowl's angle, the coil's offset, tilt and height, the terminals' depth, the spring contacts' travel, the grippers).

Wrong ways, for checking the scorer: `COIL_MODE=noturn` leaves the bowl as it is and does the coil exactly as the demo
does (score 0: the leads and terminal bracket hit the bowl's side, the coil rides up on the bowl and the cooktop's
edge and never reaches the receptacle, and the bowl is still a quarter turn wrong). `COIL_MODE=drop` turns the bowl,
then carries the coil level over the well and sets it straight down in its seated place (score 0.25: the bowl is
right, but the terminals land on the cooktop's edge above the receptacle; the coil lies 6 mm high and tilted 6 deg,
unplugged; its terminals reach past the receptacle's face but not into the slots, so the spring contacts are not
touched).

## Runtime per trial

Measured on one NVIDIA RTX PRO 6000 Blackwell (another simulator run sharing the GPU): the scripted demo reaches full
score at step 748 (37 s of simulated time) and its run takes about 3 minutes including the simulator's start; the
Dream-1 run of 2,400 steps (the run limit) takes about 7 minutes. The run limit is about 3 times what the demo needs.

## Seeds

The layout is fixed (`object_jitter=0`); the simulation is deterministic up to the physics engine's GPU scheduling.

## Building it for real

A whole range is not needed: a 3 mm steel (or plywood) plate 34 x 58 cm with a 22.6 cm hole, on a 7.5 cm frame, with a
real GE plug-in receptacle (the surface burner terminal block kit WB17T10006) screwed under the edge 12.2 cm from the
hole's centre so that the terminals' line is 1.3 cm below the top; a real 8-inch plug-in coil (GE WB30M2) and an 8-inch
chrome drip bowl (GE WB32X10014). These are ordinary spare parts. Nothing is wired: the rig is never powered. Mark the
bowl's starting angle (opening toward the back) and the coil's place on the top with tape.

[jsc57]: https://www.manualslib.com/manual/2748629/Ge-Jsc57.html?page=26
[jbp15]: https://www.manualslib.com/manual/1113869/Ge-Jbp15.html?page=22
[jbc16]: https://www.manualslib.com/manual/1186548/Ge-Jbc16gk.html?page=20
