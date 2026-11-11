# Take a steak's temperature in its thickest part, off the bone

The table top stands for a restaurant kitchen's pass. A T-bone steak just off the grill lies on a wooden board, and an
instant-read probe thermometer stands in its holder beside it. The instruction asks for the steak's temperature to be
checked with the thermometer; it does not say where the probe goes.

The rule the task checks is the one every food-safety course teaches about probe thermometers. Clemson Cooperative
Extension, Home & Garden Information Center, HGIC 3587, [Food Thermometers: A Key to Food
Safety](https://hgic.clemson.edu/factsheet/food-thermometers-a-key-to-food-safety/): "The food thermometer should be
placed in the thickest part of the food and should not be touching bone, fat or gristle." Bone and fat heat
differently from the meat, so a probe against them reads wrong, and the thickest part is the last to cook. A T-bone's
bone runs through the middle of the steak, with the thick strip loin on one side and the thin tenderloin on the other:
the probe pushed in near the middle meets the bone; its tip in the middle of the strip, clear of bone and fat, keeps
the rule. The task is the task file
[`take_the_steak_temperature_off_the_bone.py`](take_the_steak_temperature_off_the_bone.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The board with the
steak on it (meat, bone, fat cap and grill marks), the thermometer (a probe under a head with a dial) and its holder (a
tube of panels on a base) are bodies built from several shapes, and the green reading light is a body the task file
moves with the thermometer, so `bodykit.py` adds such bodies: importing it hooks `yam_arena.scene` when the simulator
starts, without changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: there is no heat in the simulation, and the steak is shown only (the probe passes
into it without resistance). The rule is scored from where the probe's tip and shaft are against the steak's bone, fat
and thick middle while the thermometer is held still. The steak's shape is simplified to boxes.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
probe's tip is in the meat when it is inside the steak's outline and below its top. A reading is taken when the tip
has stayed in the meat, moving less than 0.5 mm a step, for 1.5 s. A reading taken with the probe within 6 mm of the
bone or the fat, or with its tip outside the strip's thick middle (within 1.8 cm of the strip's centre and 1 to 2.6 cm
above the board), sets the score to 0 for the rest of the run. Otherwise the score is the best stage reached: 0.25 the
thermometer lifted out of its holder; 0.5 its tip in the meat, clear of bone and fat; 0.75 a reading held for 3 s in
the strip's thick middle (a green light then shows on the head). Full score: that, and the thermometer standing back in
its holder, both grippers 6 cm or more from its head, still for 1 s. `-E object_jitter=0` keeps every object exactly
where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=take_the_steak_temperature_off_the_bone.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Check the steak's temperature with the thermometer." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/steak-dream
```

The scripted demo, which reaches full score (the right gripper takes the thermometer by its head, lifts it out of its
holder, pushes the probe straight down into the middle of the strip loin until the tip is halfway down, holds it there
for the reading, draws it out, stands it back in its holder and moves away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=take_the_steak_temperature_off_the_bone.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Check the steak's temperature with the thermometer." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/steak-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/steak-dream/*.json; do inspect-robots video "$log"; done
```

`STEAK_DEBUG=1` prints, every 10 steps (`STEAK_DEBUG_EVERY=5` for every 5), where the probe's tip is (also on the
board), whether it is in the meat, whether the probe touches bone or fat, whether the tip is in the strip's thick
middle, how long it has been held still, whether a reading was taken, whether the thermometer is out of or back in its
holder, how far the grippers are from its head, and the stage reached.

`STEAK_BONE=1` runs the demo the wrong way: the probe goes in beside the bone, near the steak's middle; the run scores 0
when the reading is taken.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 2.5 minutes for the scripted demo while rendering 1280 x
720 video (429 steps, 21 s of simulated time; 145 s), and about 6 minutes for a Dream-1 trial of 2,400 steps (370 s).
Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A cooked T-bone steak (or a shaped stand-in of foam with a bone marked through it) on a board, and an instant-read
thermometer in a holder. Where the probe went in, and whether it touched the bone, is judged from the overhead and
wrist videos.
