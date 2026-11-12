# Dock a laptop: lift it level by both sides onto its docking station, then press it down until it clicks

A closed 13-inch business laptop (21.5 cm deep, 31 cm wide, 1.9 cm thick, 1.3 kg) lies on a small wooden laptop stand
near the robot, its left and right sides overhanging the stand. Behind it stands a mechanical docking station (a port
replicator that the laptop's back sits on), its top 1 cm higher than the stand's. On the docking station's top are the
docking connector, a spring latch hook on each side of it, and two guide posts at its back corners; in the laptop's
underside, near its back edge, are the docking socket and a latch slot on each side of it. The instruction is only
"Dock the laptop."; it does not say how.

The procedure the task checks is the docking station maker's. Dell's [E-Port User's Guide][guide] (model PR03X),
"Docking Your Laptop", step 6: "Center the laptop with the E-Port, then push down on the laptop until it clicks" (into
place in the docking connector); the same guide warns not to pick the laptop up while it is docked, as that can damage
the connectors. The laptop is more than three times as wide as a gripper opens and has no handle, so it is lifted by
both sides together, as people pick a laptop up, and kept level and square: carried onto the docking station, set back
against the guide posts and lowered straight down so that the connector enters the socket; then both sides are pressed
down until both latches click. Lifted by one side, it tips about the other and cannot be carried level, and the
docking station stands higher than the stand, so the laptop cannot be pushed across onto it. The task is the task file
[`dock_a_laptop_level_onto_its_docking_station.py`](dock_a_laptop_level_onto_its_docking_station.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. No third-party assets are used: every object is built from boxes and
cylinders in the task file, and the task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or put one on a spring.
The laptop (with its socket and latch slots), the stand, the docking station (with its guide posts and connector) and
the two latch hooks are bodies built from several shapes, and each hook slides on a spring, so `bodykit.py` adds such
bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives
`score()` each body's pose.

What is modelled and what is not: the laptop has the size and mass of a 13-inch business laptop; the docking station is
21 cm wide, like Dell's E-Port, and 14.75 cm deep, so the docked laptop's front overhangs it by 8 cm instead of resting
on the desk in front of it. The laptop is one rigid block with a 22 x 50 mm socket, 12 mm deep, centred 3.25 cm in
front of its back edge, and a 16 mm wide latch slot on each side of it (at y -6.5 and 6.5 cm), whose catch is the
bottom cover's 3 mm lip at the slot's back. The connector is a plain 16 x 44 mm block, 7 mm tall, 3 mm clear of the
socket all round (no contacts). Each latch hook is a steel stem with a rounded nose (a 3.5 mm radius cylinder) on a
5000 N/m spring slide; docked, the nose overlaps the catch by 2 mm. Pressed down onto the hooks, the catches push the
noses forward (and the noses push the laptop back against the guide posts) until they pass below them, and the noses
spring back over the catches: the click. The laptop's weight alone does not get past the noses: let go undocked, it
rests on them with its back 8 mm up; pressing both sides down does. Electrical docking (power, ports, the docking
light) is not modelled.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
bodies' poses are read from the simulation, and each gripper's centre from the arm's joints. The score is 0 for the
rest of the run once the laptop has been dropped (its middle falling faster than 0.6 m/s, as from about 2 cm), turned
over, knocked off the table, or pulled off the docking station after it clicked in (its socket's mouth 3 mm or more
above the connector's base after both latches clicked). Otherwise it is the best stage reached: 0.25 lifted clear of
the stand (its lowest corner 1.5 cm or more above the stand's top, tilted under 10 degrees); 0.5 put down on the
docking station against the guide posts and let go (its back edge within 5 mm of both posts, its middle within 1 cm of
the station's centre line, the middle of its underside within 1 cm above the station's top, tilted under 6 degrees, at
rest, both grippers 6 cm or more from it); 0.75 docked (within 3 mm of its docked place, its socket's mouth within
1.5 mm of the connector's base, level within 2 degrees, both latch hooks back over their catches). Full score: docked,
both grippers 6 cm or more from the laptop, still for 1 s. `-E object_jitter=0` keeps every object exactly where the
task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=dock_a_laptop_level_onto_its_docking_station.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Dock the laptop." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dock-dream
```

The scripted demo, which reaches full score (each gripper, tilted 75 degrees from straight down so that it points in
over the laptop's side edge, comes in with its fingers open, one above the lid and one below the edge, and pinches the
edge; together they lift the laptop off the stand, carry it back over the docking station, lower it to just above the
connector, set its back edge against the guide posts and lower it straight down onto the connector, where it rests on
the latch noses; they let go and draw out sideways, then, straight down with their fingers closed, press the lid down
near its left and right edges together until both latches click, and move away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=dock_a_laptop_level_onto_its_docking_station.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Dock the laptop." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/dock-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/dock-dream/*.json; do inspect-robots video "$log"; done
```

`DOCK_DEBUG=1` prints, every 10 steps (`DOCK_DEBUG_EVERY=5` for every 5), where the laptop is, its lowest corner, tilt
and fall speed, how high its socket's mouth is above the connector's base, how far its back edge is from each guide
post, how far each latch hook is pushed forward and how far its nose overlaps its catch, whether each latch is engaged,
the stages reached, how close the nearest gripper is, and where each gripper is.

`DOCK_ONE=1` runs the demo the wrong way: the right arm alone pinches the laptop's right side edge and does what it
does in the demo. The laptop tips about its left side, which stays on the stand, so it is never lifted level; dragged
back by one side, it swings round and ends crooked against the docking station's front, tilted, with its socket
nowhere near the connector: score 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 2 minutes for the scripted demo while rendering 1280 x 720
video (486 steps, 24.3 s of simulated time; 124 s), and about 7 minutes for a Dream-1 trial of 2,400 steps (431 s),
with both runs sharing the GPU. Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A 13-inch laptop with a mechanical docking station (a port replicator with a docking connector, latches and an eject
button, such as Dell's E-Port with a Latitude E-series laptop), and a small stand 3 cm tall and 20 cm wide for the
laptop to start on. A powered docking station shows the docking light when the laptop is docked; the click, the flush
seat against the guide posts and the light are checked by eye, the rest with a ruler and a protractor from the overhead
and wrist videos, and whether the laptop was pulled off after it clicked in is judged from the videos.

[guide]: https://dl.dell.com/manuals/all-products/esuprt_electronics/esuprt_docking_stations/e-prt_user%27s%20guide_en-us.pdf
