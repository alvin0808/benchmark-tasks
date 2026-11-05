# Turn a long parcel a quarter turn in place between two neighbouring boxes

A long cardboard parcel lies across the table between two other boxes, with only about 2 cm to spare at each end.
It has to be turned 90 degrees so that its long sides run front to back, without bumping either neighbour. Pushed at
one end, a long box swings about its other end and slides sideways into a neighbour; turned about its centre, its
corners sweep a circle that just clears them. That takes both arms pushing the two ends in opposite directions at
the same time. The whole task (scene, scoring and the scripted demo) is one task file,
[`turn_parcel_in_place.py`](turn_parcel_in_place.py), for the `yam_arena` embodiment.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. The task uses only boxes, so it needs no models from
Arena's object library.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` holds for 2 s, otherwise `score()` at the end of the episode
(0.5 or 0). Two rules hold for the whole trial and are latched in the task file's module state: if the parcel's
centre ever moves more than 4 cm from where it started, or either white box ever moves more than 1 cm, the trial
scores 0. Run one trial per process (as below) so this state starts fresh. `-E object_jitter=0` keeps every object
exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=turn_parcel_in_place.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn the long brown parcel a quarter turn in place, so that its long sides run front to back, without bumping the white boxes on either side of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/parcel-dream
```

The scripted demo, which reaches full score (both grippers, closed, push the parcel near its two ends in opposite
directions and follow its faces round in small steps; halfway the right gripper turns its fingers over while the
left one eases off, then both finish the turn together):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=turn_parcel_in_place.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn the long brown parcel a quarter turn in place, so that its long sides run front to back, without bumping the white boxes on either side of it." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/parcel-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/parcel-dream/*.json; do inspect-robots video "$log"; done
```

`TP_DEBUG=1` prints the parcel's angle, its drift from the start and how far the neighbours have moved every 10
steps.

## Runtime per trial

On one L40S: about 5 minutes for a Dream-1 trial of 2,400 steps (296 s on the clock, with Dream answering in
0.84 s on average), and about 1 minute for the scripted demo (414 steps, 20.7 s of simulated time). Isaac Sim's
first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic. With a nonzero
`object_jitter`, the trial's seed sets how far the objects shift.

## Building it for real

Any closed carton of about 42 x 14 x 10 cm (800 g) and two boxes of about 20 x 14 x 10 cm (1.2 kg) on a smooth
table, with their start positions marked in tape. A cross drawn on the middle of the long parcel's top makes the
"in place" rule easy to check in the overhead video against a tape mark beside its start position.
