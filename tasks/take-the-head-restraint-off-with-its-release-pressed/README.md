# Take a car seat's head restraint off, one arm holding its release in

The table top stands for the floor of a car's rear seat. A rear seat, scaled down, has its head restraint in place: the
head restraint's two poles stand in holes in the top of the seatback, and a lock plate in the seatback's top catches a
stop at the foot of the left pole when the head restraint is pulled up as far as it goes. The plate lets the stop pass
only while its red release tab, standing out of the seatback's front on the left, is pushed in. An open box stands to
the right of the seat. The instruction asks for the head restraint to be taken off and put in the box; it does not say
how.

The procedure the task checks is the car maker's. Hyundai owner's manual, [Head restraint][manual] (removal from a rear
seat): "Raise the head restraint as far as it can go." "Press the head restraint release button (1) while pulling up
the head restraint (2)." Pulled up on its own, the head restraint stops at the top; pulled harder, it forces the lock
and breaks it. One hand cannot hold the release in and pull the head restraint at once, so taking it off takes both arms
working together. The task is the task file
[`take_the_head_restraint_off_with_its_release_pressed.py`](take_the_head_restraint_off_with_its_release_pressed.py) for
the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or put one on a spring.
The seatback (with its two holes), the seat cushion, the box, the head restraint (cushion, two poles and the stop) and
the lock plate (tongue, arm and tab) are bodies built from several shapes, and the lock plate slides on a spring that
breaks when forced, so `bodykit.py` adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator
starts, without changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: the seat is scaled down so that the arms reach it (seatback 5 cm tall, head restraint
10 x 4 x 4 cm on 5 cm poles, 0.15 kg). The lock is a plate on a 400 N/m spring sliding front to back in the seatback's
top: its tongue covers the stop's way up, and pushing its tab in 6 mm or more lets the stop pass; its joint breaks under
15 N, which stands for a forced lock. The release is a tab at the seatback's front rather than a button on the pole's
collar, so that a gripper can reach it beside the head restraint. The foam is rigid.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The lock
is forced once its plate leaves its slide (its joint broke): that sets the score to 0 for the rest of the run.
Otherwise the score is the best stage reached: 0.25 the head restraint raised as far as it goes (2 cm or more); 0.5
taken off (its poles out of the holes); 0.75 in the box (its middle inside the walls and its lowest point 1 cm or more
below their tops, at rest, no gripper holding it). Full score: that, with both grippers 6 cm or more from it and from
the release tab, still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every
trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=take_the_head_restraint_off_with_its_release_pressed.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Take the head restraint off and put it in the box." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/headrest-dream
```

The scripted demo, which reaches full score (the right gripper takes the head restraint's cushion and raises it nearly
as far as it goes; the left gripper, level and fingers closed, pushes the release tab in with its fingertips; with the
tab held in, the right gripper pulls the head restraint up out of the holes; the left lets the tab go and moves away;
the right carries the head restraint over the box, lowers it in and lets go):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=take_the_head_restraint_off_with_its_release_pressed.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Take the head restraint off and put it in the box." \
  --scorer task_score --max-steps 440 --store-frames --log-dir ~/yam-eval/headrest-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/headrest-dream/*.json; do inspect-robots video "$log"; done
```

`HEADREST_DEBUG=1` prints, every 10 steps (`HEADREST_DEBUG_EVERY=5` for every 5), how far the head restraint has risen
and where its middle is, how far the tab is pushed in, whether the lock plate has left its slide, whether a gripper
holds the head restraint, whether it is in the box, how close the nearest gripper is, where each gripper is, and the
stage reached.

`HEADREST_YANK=1` runs the demo the wrong way: nobody pushes the tab and the right gripper pulls the head restraint
straight up; it stops at the lock, the pull forces the lock and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 1.5 minutes for the scripted demo while rendering 1280 x 720
video (345 steps, 17 s of simulated time; 82 s), and about 6 minutes for a Dream-1 trial of 2,400 steps (357 s). Isaac
Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A car seat (or a seatback mock-up) with a removable head restraint and its release button, and a box. Whether the
release was held in while the head restraint came out, and whether the lock was forced, is judged from the overhead and
wrist videos.

[manual]: https://ownersmanual.hyundai.com/full_webhelp/dn8/2025/en_us/id9cf5c6117ee.html
