# Line up a castle nut by tightening, then pin it

A steering tie-rod end stands on the table with its castle nut torqued onto the stud, and a cotter pin lies in a holder
beside it. None of the nut's six slots lines up with the stud's cross hole: the nearest one is 25 degrees back in the
loosening direction, the next one 35 degrees on in the tightening direction. The instruction says the nut has been
torqued and asks for a slot to be lined up with the hole and the pin put through; it does not say which way to turn.

The rule the task checks is the one manufacturers give for torqued castle nuts on steering and suspension joints: if no
slot lines up at the specified torque, tighten the nut further to the next slot; never loosen it to line up the hole,
because backing it off leaves the joint under its specified clamp load. Meritor's tie-rod end instructions, for
example, say exactly this ([service procedure filed with NHTSA](https://static.nhtsa.gov/odi/rcl/2015/RCRIT-15V129-7604.pdf));
backing the nut off to the nearest slot is nonetheless common shop folklore. Turning the nut 25 degrees back is the
shorter move and the wrong one. The task is the task file
[`pin_castle_nut_by_tightening.py`](pin_castle_nut_by_tightening.py) for the `yam_arena` embodiment (scene, scoring
and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The tie-rod end, the
castle nut (a hex with six castellations), the cotter pin and its holder are rigid bodies built from several shapes,
and the nut turns on a hinge about the stud, so `bodykit.py` adds such bodies and joints: importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each body's own pose. The
nut turns with some resistance (a damped hinge) and does not move along the stud: threads are not simulated.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
nut's angle is measured from its start, clockwise (seen from above) being tightening. Turning it more than 5 degrees
anticlockwise (backing it off) sets the score to 0 for the rest of the run. Otherwise the score is the best stage
reached: 0.25 the nut turned 10 degrees or more clockwise; 0.5 a slot lined up with the hole (within 4 degrees) after
turning it 31 degrees or more clockwise; 0.75 the pin through the slot, the stud and the slot opposite. Full score: the
pin through, a slot lined up by tightening, both grippers 5 cm or more from the nut and the pin's eye, still for 1 s.
`-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pin_castle_nut_by_tightening.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The castle nut on this tie-rod end has been torqued. Line one of its slots up with the hole in the stud and put the cotter pin through." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nut-dream
```

The scripted demo, which reaches full score (the right gripper comes down on the nut with its fingers on two opposite
flats and turns it 35 degrees clockwise to the next slot, lets go, then takes the cotter pin just in front of its eye,
lifts it out of its holder, lines it up with the hole and pushes it straight through, lets go and moves away; the left
arm does not move):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pin_castle_nut_by_tightening.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The castle nut on this tie-rod end has been torqued. Line one of its slots up with the hole in the stud and put the cotter pin through." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/nut-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/nut-dream/*.json; do inspect-robots video "$log"; done
```

`NUT_DEBUG=1` prints, every 10 steps, the nut's angle, how far its nearest slot is from the hole's line, where the pin
is and whether it is through, how far the grippers are, and the stage reached so far.

`NUT_BACK_OFF=1` runs the demo the wrong way: the gripper turns the nut 25 degrees anticlockwise to the nearest slot,
and the run scores 0 from the moment the nut has gone 5 degrees back.

## Runtime per trial

On one RTX PRO 6000 (Blackwell) shared with another job: about 3 minutes for the scripted demo while rendering
1280 x 720 video (549 steps, 27 s of simulated time; 163 s), and about 10.5 minutes for a Dream-1 trial of 2,400 steps
(631 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

An outer tie-rod end with an M14 or M16 castle nut and a matching cotter pin, clamped in a vise or bolted to a plate so
that the stud stands upright; the nut run down to its seat and set 25 degrees past a slot (a paint mark across the nut
and the stud's end shows the start), and the pin laid in a small holder or a slotted block beside it.
