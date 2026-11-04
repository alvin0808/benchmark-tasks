# Squeeze a handle-less box between both grippers and lift it onto a riser

A closed cardboard box with no handles is wider and deeper than a gripper can open, so no single gripper can pick
it up. The two arms have to squeeze it between them, lift it together and set it down upright in the middle of a
5 cm riser just beyond it. The whole task (scene, scoring and the scripted demo) is one task file,
[`two_hand_box_lift.py`](two_hand_box_lift.py), for the `yam_arena` embodiment.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. The task uses only boxes, so it needs no models from
Arena's object library.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` holds for 3 s, otherwise `score()` at the end of the episode
(0.5 or 0). `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts
from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=two_hand_box_lift.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the cardboard box onto the wooden riser and leave it standing upright in the middle of the riser. The box is too wide for one gripper to grasp, so hold it between both grippers." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/box-dream
```

The scripted demo, which reaches full score (both arms press their closed fingers against the box's left and
right faces, squeeze, lift it about 10 cm, carry it forward over the riser, lower it onto the riser and let go
sideways):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=two_hand_box_lift.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the cardboard box onto the wooden riser and leave it standing upright in the middle of the riser. The box is too wide for one gripper to grasp, so hold it between both grippers." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/box-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/box-dream/*.json; do inspect-robots video "$log"; done
```

## Runtime per trial

On one L40S: about 4.5 minutes for a Dream-1 trial of 2,400 steps (261 s on the clock, with Dream answering in
0.65 s on average), and under 1 minute for the scripted demo (327 steps, 16.4 s of simulated time, 51 s on the
clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic. With a nonzero
`object_jitter`, the trial's seed sets how far the box shifts.
