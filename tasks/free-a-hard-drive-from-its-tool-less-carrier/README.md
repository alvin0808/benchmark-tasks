# Free a hard drive from its tool-less carrier while holding one sprung rail open

The table top stands for a workbench. A 2.5-inch hard drive lies label up in a tool-less drive carrier of the kind used
in desktop computers. The carrier's base is fixed to the table, and along each side of the drive runs one of the
carrier's blue plastic rails. Each rail carries two pins that sit in the drive's two side screw holes, and each rail is
sprung inward, so that it springs back into the holes as soon as it is let go; a tab stands up at the front end of each
rail, in front of the drive. An anti-static mat lies to the right of the carrier. The instruction asks for the drive to
be taken out of its carrier and put on the mat; it does not say how.

The procedure the task checks is the computer maker's. Dell's [OptiPlex 5060 Tower Service Manual][manual], "Removing
the 2.5-inch drive from the drive bracket": "Pull one side of the drive bracket to disengage the pins", then lift the
drive out of the bracket. Lenovo's [requirements for handling drives][handling] say "Do not bump drives" and to put them
down on cushioned surfaces. With one rail pulled out and held, the drive's side next to it is lifted, so that the drive
pivots on the other rail's pins; the drive is then slid sideways off those pins and lifted clear. One hand cannot hold a
rail out and lift the drive at once: let go, the rail springs back into the holes. So taking the drive out takes both
arms working together. Pulled straight up with both rails in place, the drive is held by the pins, and pulled hard, a
rail breaks. The task is the task file
[`free_a_hard_drive_from_its_tool_less_carrier.py`](free_a_hard_drive_from_its_tool_less_carrier.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task uses no third-party assets: every object is built from boxes and
cylinders in the task file, and it needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or put one on a spring.
The carrier's base, the two rails (each a bar with two posts, two pins and a tab), the drive (with its four screw holes)
and the mat are bodies built from several shapes, and each rail turns on a sprung hinge that breaks when forced, so
`bodykit.py` adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without
changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: the drive is a 2.5-inch drive at its real size (100.2 x 69.85 x 9.5 mm, SFF-8201) and
weight (115 g), with two side screw holes on each side, 3.0 mm above its bottom and 76.6 mm apart, as SFF-8201 places
them; the holes are 4 mm square (real M3 holes are 3 mm across). The flex of each plastic rail is a hinge along the
drive's side, 7 mm below the drive, on a spring (0.2 N m per radian, resting on its stop). Its two pins (2.5 mm across,
3 mm long) stand 2.7 mm into the holes; opening the rail about 17 degrees draws them out, and it opens 40 degrees at
most. A rail's hinge breaks under 10 N or 0.5 N m, standing for a cracked rail: pulling the drive up against the pins,
or forcing the rail past 40 degrees, breaks it. The rails' tabs stand 8 cm tall so that a gripper can push one out. The
carrier's base is fixed to the table, as if held in a fixture. The drive has no electronics and no cables are attached;
dropping or knocking it is judged from its speed.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The drive
counts as dropped or knocked once its middle moves down faster than 0.5 m/s (as after a free fall of 1.3 cm) or its
speed changes by more than 0.6 m/s within one 50 ms control step, and a rail counts as broken once its hinge has given
way: either sets the score to 0 for the rest of the run. Otherwise the score is the best stage reached: 0.25 one side
of the drive lifted off its rail's pins (that side's two holes off the pins and risen 4 mm or more); 0.5 the drive out
of the carrier (all four holes off the pins, and its lowest point 1 cm or more above the carrier's base or its middle
6 cm or more from where it started); 0.75 the drive lying on the mat (label up, tilted 10 degrees or less, its middle
over the mat at least 5 mm in from its edges, its lowest point within 4 mm of the mat's top), at rest, not held (no
gripper's centre within 7 cm of the drive's middle and keeping the same offset to it). Full score: that, with both rails
back home (within 2 degrees of shut) and both grippers 6 cm or more from the drive and from the rails' tabs, still for
1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=free_a_hard_drive_from_its_tool_less_carrier.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Take the hard drive out of its carrier and put it on the anti-static mat." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/hdd-dream
```

The scripted demo, which reaches full score (the right gripper takes the drive by its two sides; the left gripper,
level and fingers closed, pushes the left rail's tab out, drawing that rail's pins out of the drive's holes, and holds
it there; the right gripper lifts the drive's left side, so that it pivots on the right rail's pins, slides it sideways
off those pins and lifts it clear; the left gripper lets the rail spring back and moves away; the right carries the
drive over the mat, lowers it until it nearly touches, lets go and moves away). It is open-loop: it works from the known
layout and never reads the objects' poses.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=free_a_hard_drive_from_its_tool_less_carrier.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Take the hard drive out of its carrier and put it on the anti-static mat." \
  --scorer task_score --max-steps 480 --store-frames --log-dir ~/yam-eval/hdd-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/hdd-dream/*.json; do inspect-robots video "$log"; done
```

`HDD_DEBUG=1` prints, every 10 steps (`HDD_DEBUG_EVERY=5` for every 5), where the drive's middle is, its lowest point and
tilt, which pins stand in their holes, how far each side's holes have risen, how far each rail is open, whether a
rail's hinge has given way, whether a gripper holds the drive, whether it is lifted, out or on the mat, whether the
rails are home, how close the nearest gripper is, the drive's fastest fall and largest speed change, where each gripper
is, and the stage reached. `HDD_PUSH`, `HDD_TILT`, `HDD_SLIDE` and `HDD_A_Z` change how far the demo's left gripper
pushes the tab, how far the right one tilts and slides the drive, and the left gripper's height.

`HDD_YANK=1` runs the demo the wrong way: nobody pulls a rail and the right gripper lifts the drive straight up. The pins
hold it (it rises 0.7 mm), the pull breaks both rails' hinges about 1.3 s into the pull, and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 2.5 minutes for the scripted demo while rendering 1280 x 720
video (421 steps, 21 s of simulated time; 144 s, with two other runs on the same GPU), and about 7.5 minutes for a
Dream-1 trial of 2,400 steps (441 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A 2.5-inch drive in the tool-less 2.5-inch drive bracket of a desktop computer (for instance a Dell OptiPlex tower's),
the bracket clamped flat to the table with a tab fitted to the front end of each side, and an anti-static mat. Whether
a rail was held open while the drive came out, whether a rail cracked, and whether the drive was dropped or knocked are
judged from the overhead and wrist videos.

[manual]: https://www.dell.com/support/manuals/en-us/optiplex-5060-desktop/optiplex_5060_mt_service_manual/removing-the-25inch-drive-from-the-drive-bracket?guid=guid-4aabdcf4-a7c7-494a-8e0d-cad2b4b0828a&lang=en-us
[handling]: https://pubs.lenovo.com/thinksystem_storage_de_himg_11.50.3/C0D0D28C-E45A-4610-8FAC-D282F32F3CFE_
