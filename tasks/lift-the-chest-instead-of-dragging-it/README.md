# Lift a heavy chest onto the rug, never drag it across the wood floor

The table's wooden top stands for a room's hardwood floor. A heavy wooden storage chest, with a brass loop handle at
each end, stands on the bare floor in front of the arms, and a rug lies beyond it. The instruction asks for the chest to
be moved onto the rug; it does not say how.

The rule the task checks is the one flooring makers give everyone who moves furniture over a wood floor. Shaw Floors,
[Hardwood Floor Care & Maintenance](https://shawfloors.com/flooring/how-to/hardwood/care-maintenance): "Don't drag heavy
furniture across the floor." Pushing the chest along the floor, or lifting one end and dragging the other, is the quick
way; the chest has a handle at each end, so lifting it clear of the floor with both hands and carrying it keeps the
rule. The task is the task file
[`lift_the_chest_instead_of_dragging_it.py`](lift_the_chest_instead_of_dragging_it.py) for the `yam_arena` embodiment
(scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The chest (its body,
lid line, plinth, lock plate and two loop handles) and the rug (with its border and fringes) are bodies built from
several shapes, so `bodykit.py` adds such bodies: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: the chest is a rigid box (10 cm deep, 20 cm long, 8 cm tall, 1.0 kg), scaled down so
that two grippers can lift it by its handles, and the rug is a fixed pad 3 mm thick. Scratches are not simulated: the
rule is scored from how far the chest slides while touching the floor.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The chest
is in contact with the floor (or the rug) while its lowest corner is less than 3 mm above the surface under it; every
step it is in contact, the distance its centre moves sideways is added up as the distance it was dragged. Dragging it 2
cm or more sets the score to 0 for the rest of the run. Otherwise the score is the best stage reached: 0.25 the chest
lifted clear of the floor (its lowest corner 1 cm or more above it); 0.5 carried, clear of the floor, over the rug; 0.75
set down on the rug (its whole base on it, upright to 10 degrees). Full score: that, with both grippers 6 cm or more
from both handles, still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every
trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lift_the_chest_instead_of_dragging_it.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Move the chest onto the rug." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/chest-dream
```

The scripted demo, which reaches full score (both grippers come down on the handle bars at the same time, take them,
lift the chest level 4 cm clear of the floor, carry it over the rug, lower it until it stands on the rug, let go and
rise):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lift_the_chest_instead_of_dragging_it.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Move the chest onto the rug." \
  --scorer task_score --max-steps 420 --store-frames --log-dir ~/yam-eval/chest-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/chest-dream/*.json; do inspect-robots video "$log"; done
```

`CHEST_DEBUG=1` prints, every 10 steps (`CHEST_DEBUG_EVERY=5` for every 5), where the chest is, how far its lowest
corner is above the surface under it, how far it tilts and turns, how far it has been dragged, whether a gripper holds a
handle, whether it stands on the rug, how close the nearest gripper is to a handle, where each gripper is, and the stage
reached.

`CHEST_DRAG=1` runs the demo the wrong way: the grippers take the handles but slide the chest along the floor onto the
rug without lifting it; the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 1 minute for the scripted demo while rendering 1280 x 720
video (279 steps, 14 s of simulated time; 50 s), and about 4 minutes for a Dream-1 trial of 2,400 steps (245 s).
Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A small wooden box with a handle at each end, weighted to about 1 kg, a wood board or floor, and a rug or mat. Whether
the box slid along the floor is judged from the overhead and wrist videos.
