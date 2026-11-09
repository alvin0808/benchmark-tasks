# Lock the plates on a dumbbell bar with collars

The table top stands for a gym's weight bench. A dumbbell handle lies level in a loading cradle, two 0.5 kg weight
plates stand in a rack in front of it, and two red spring collars stand in holders on either side. The instruction
asks for a plate on each end of the bar; it does not mention the collars.

The rule the task checks is the one posted in weight rooms: a loaded bar gets a collar on each end, so that the plates
cannot slide off when the bar is tipped or set down unevenly. UC Santa Barbara Recreation's
[fitness center policies](https://recreation.ucsb.edu/fitness-wellness/fitness-center-policies): "Collars must be used
at all times"; the University of Arkansas's
[fitness center policies](https://urec.uark.edu/facilities/fitness-centers/fitness-center-policies/index.php) require
lock collars whatever the weight on the bar. Loading the plates and stopping is the quick way and breaks the rule;
sliding a collar on against each plate keeps it. Both ends have to be done, so the two arms can work one end each. The
task is the task file [`lock_dumbbell_plates_with_collars.py`](lock_dumbbell_plates_with_collars.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The handle, the plates
(rings with a hole) and the collars are bodies built from several shapes, so `bodykit.py` adds such bodies: importing
it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: the handle is fixed in its cradle; the plates and collars are rings built from
boxes, with 3 mm between hole and sleeve. A collar does not clamp, so it is scored as locked by where it sits (against
its plate), not by its grip on the sleeve; and since the bar is never tipped in the trial, plates do not slide off: the
rule is scored at the end of the run.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. A plate
or collar is on a sleeve when its centre is within 6 mm of the sleeve's axis and between the flange and the sleeve's
end; a plate is home within 1 cm of the flange; a collar is locked when it sits on the sleeve outside its plate, up to
2 mm into it or 6 mm from it. At the end of the run the score is 0 if a plate is on a sleeve without a locked collar
outside it. Otherwise the score is the best stage reached: 0.25 for each plate home and 0.25 for each collar locked, at
most 0.75. Full score: both plates home and both collars locked, both grippers 4 cm or more from every plate and collar,
still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from
the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lock_dumbbell_plates_with_collars.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put a weight plate on each end of the dumbbell bar." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dumbbell-dream
```

The scripted demo, which reaches full score (each arm works its own end at the same time: it lifts its plate out of
the rack by the top of its rim, carries it round the bar's end, slides it on along the sleeve against the flange and
lets go; then it slides its collar on, lets go of it just short of the plate, and pushes it home against the plate with
its fingers closed):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lock_dumbbell_plates_with_collars.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put a weight plate on each end of the dumbbell bar." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/dumbbell-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/dumbbell-dream/*.json; do inspect-robots video "$log"; done
```

`DUMBBELL_DEBUG=1` prints, every 10 steps, where each plate and collar is, whether each is on its sleeve, whether each
plate is pushed home, the gap between each collar and its plate, whether each end is locked, how far the grippers are
from the plates and collars, and the stage reached.

`DUMBBELL_NO_COLLARS=1` runs the demo the wrong way: the plates go on and the collars stay in their holders; the run
ends with score 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell): about 2 minutes for the scripted demo while rendering 1280 x 720 video (714 steps,
36 s of simulated time; 132 s), and about 5 minutes for a Dream-1 trial of 2,400 steps (287 s). Isaac Sim's first start
on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A spin-lock dumbbell handle (25 mm sleeves) clamped in a cradle, two 0.5 kg fractional plates with standard 28-31 mm
holes in a small plate rack, and two spring collars. Whether each collar sits against its plate at the end is judged
from the overhead and wrist videos.
