# Hold a grand piano's lid up with one hand while the other sets its prop

The table top stands for a stage or a showroom floor. A grand piano, scaled down, stands with its lid closed; its
keyboard is along its right end, the lid is hinged along its far side and opens by lifting its front edge, and the lid
prop lies folded down on its bracket at the front of the case. The instruction asks for the lid to be opened; it does
not say how.

The procedure the task checks is the piano maker's. Yamaha, [Grand Piano / Upright Piano Owner's Manual][manual]
("Opening/closing the lid of the Grand Piano"): "While supporting the lifted two stacked Lids with your right hand, lift
the lid prop with your left hand and then insert the end into the hole on the main lid." The same manual warns: "Open
and close the lid carefully as described in the operation procedure. Otherwise, the lid could fall down and injure your
hands and fingers." A lid lifted and let go, or lifted with nothing to hold it up, falls; one arm cannot both hold the
lid and set the prop, so opening it takes both arms working together. The task is the task file
[`hold_a_piano_lid_up_while_setting_its_prop.py`](hold_a_piano_lid_up_while_setting_its_prop.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or hinge them. The
piano's case (rim, iron frame, strings and keyboard), its lid (with its grip block and prop cup) and the prop are bodies
built from several shapes, and the lid and the prop each turn on a hinge, so `bodykit.py` adds such bodies and joints:
importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each
body's pose.

What is modelled and what is not: the piano is scaled down so that the arms reach it (its case 21 x 26 cm and 4 cm
tall, the lid 0.35 kg on a hinge along the far side), and a small grip block on the lid stands for the front edge a
hand lifts. The prop turns on its own hinge at the front of the case and has a round knob where a hand takes it. The
strings, action and keys do not move.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The lid
is held while its grip block is within 3.5 cm of a gripper's centre and keeps the same offset from it; the prop's end is
in its cup when it is within 7 mm of the cup's centre. The lid falling (after it has been raised 8 degrees or more, it
drops 6 degrees or more within 0.5 s while no gripper holds it) sets the score to 0 for the rest of the run. Otherwise
the score is the best stage reached: 0.25 the lid raised 15 degrees or more; 0.5 the prop's end in the cup; 0.75 the
lid resting on its prop (the prop's end in the cup, no gripper holding the lid, the lid at rest). Full score: that, with
both grippers 6 cm or more from the grip block, the cup and the prop, still for 1 s. `-E object_jitter=0` keeps every
object exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=hold_a_piano_lid_up_while_setting_its_prop.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Open the piano lid." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/piano-dream
```

The scripted demo, which reaches full score (the right gripper takes the grip block and raises the lid to 30 degrees,
the block turning between its fingers, and holds it there; meanwhile the left gripper takes the folded prop by its knob
and swings it up until its end stands under the cup; the right gripper lowers the lid onto the prop's end; the left
gripper lets go of the prop and backs out, and only then does the right gripper let go of the lid):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=hold_a_piano_lid_up_while_setting_its_prop.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Open the piano lid." \
  --scorer task_score --max-steps 560 --store-frames --log-dir ~/yam-eval/piano-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/piano-dream/*.json; do inspect-robots video "$log"; done
```

`PIANO_DEBUG=1` prints, every 10 steps (`PIANO_DEBUG_EVERY=5` for every 5), the lid's and the prop's angles, where the
prop's end and the cup are and how far apart, whether the lid is held, whether it rests on its prop, how close the
nearest gripper is, where each gripper is, and the stage reached.

`PIANO_DROP=1` runs the demo the wrong way: the right gripper raises the lid and lets go of it without the prop; the
lid falls and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine whose GPU was also running another user's training job: about 8
minutes for the scripted demo while rendering 1280 x 720 video (437 steps, 22 s of simulated time; 462 s), and about 5
minutes for a Dream-1 trial of 2,400 steps (286 s). Isaac Sim's first start on a new machine adds a few minutes while it
compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A box with a lid on a hinge along its far side (a piano lid, or a lid of similar weight), a stick hinged at the front
as its prop and a cup or hole under the lid for the prop's end. Whether the lid ever dropped, and whether the prop's end
sits in the cup, is judged from the overhead and wrist videos.

[manual]: https://data.yamaha.com/files/download/other_assets/2/2925252/GP_UP_owners_manual_En_A0.pdf
