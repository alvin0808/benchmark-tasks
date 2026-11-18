# Turn a cake out onto a plate: clamp plate and pan together with both hands, turn them over and lift the pan off

A round cake pan with a plain sponge cake in it sits on a cake stand, and a flat plate lies upside down on the pan's
narrow rim and its two tab handles. The task is to turn the cake out onto the plate without damaging it, as every home
baker does: hold the plate and the pan together, turn the two over as one so the cake drops onto the plate, set them
down on the stand plate side down and lift the pan off. The instruction states only the result (the cake undamaged and
upside down on the plate on the stand, the pan off). The plate is 21 cm across, far wider than one gripper opens, and
held from one side only the stack hangs off that grip and comes apart, so the two arms clamp it from opposite sides and
turn it together. A sponge cake is soft: squeezed it dents, and bent or dropped it cracks, which is why the hands hold
the plate and the pan and never the cake. The task is the task file
[`turn_cake_out_onto_plate.py`](turn_cake_out_onto_plate.py) for the `yam_arena` embodiment (scene, scoring and the
scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. No third-party assets are used: every object is built
from boxes and cylinders in the task file, and the task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres. The pan (a floor, 24 wall panels, 24 rim panels and
two tab handles), the two halves of the cake (one with its crust) and the plate are loose rigid bodies built from
several shapes, and the cake's halves are held together by a joint, so `bodykit.py` adds such bodies and joints:
importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each
body's own pose. It also lets the demo's waypoints carry a full gripper rotation, for the side grips.

## The cake and how damage is judged

The cake is a sponge disc 12.7 cm across and 2.8 cm tall (100 g). The simulator cannot dent, crack or crumble a
sponge, so the simulated cake is two rigid halves, each 1.4 cm thick, held together by a sprung slide joint along the
cake's axis: 6,000 N/m, so the whole cake squashes 3 mm under 18 N spread over its face (a sponge about 13 kPa stiff),
well damped, free to squash 10 mm and to stretch 4 mm. The joint breaks for good when the halves are sheared, bent or
twisted with more than 40 N or 0.3 N m, or pushed against its travel limits that hard, as when the cake lands on its
edge from a height: the cake has split. The scorer counts the cake as damaged, for the rest of the run, once:

- any part of either arm presses on it with more than 3 N (read every control step from PhysX's contact forces
  between the arms' links and the two halves): a gripper denting or pinching the sponge;
- its two halves are pressed together by more than 3 mm: the cake squashed;
- it has broken: the halves sheared more than 2 mm out of line, bent or twisted more than 3 degrees, or pulled more than
  3 mm apart.

In a real trial the same damage is judged after the run with a ruler and a photo (see "Building it for real").

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 2 s, otherwise the score at the end of the episode,
which the task file keeps itself: 0 for the rest of the run once the cake is damaged, otherwise the best stage reached
(`SCORE_OVER = "end"`):

- 0.25: the cake has lain upside down on the plate (crust down, turned more than 160 degrees, its crust face within
  2.5 cm of the plate's middle and touching the plate, within 3 mm);
- 0.5: it lies like that on the plate with the plate flat on the stand (within 10 degrees of level, its middle within
  3 cm of the stand's, resting on it within 3 mm) and the pan, the cake and the plate still;
- 1.0 (`success()`): that, with the pan lifted off and away (its middle 15 cm or more from the cake's, so no part of it
  is over the cake), both grippers 4 cm or more from the cake and the plate, the cake undamaged and everything still,
  held for 2 s.

`-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=turn_cake_out_onto_plate.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn the cake out of its pan onto the plate without damaging it, so that it ends up upside down on the plate on the stand with the pan off it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/cake-dream
```

The scripted demo, which reaches full score in 650 steps (both grippers come in level from the sides with their
fingers upright, clamp the plate and the pan's handle under it together, as a person holds plate and tin by the tin's
handles, lift the stack, turn it over about the line between them, set it down on the stand plate side down and let go;
then each takes one of the pan's handles from above, and they lift the pan off the cake, carry it towards the arms and
set it down; neither gripper touches the cake):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=turn_cake_out_onto_plate.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn the cake out of its pan onto the plate without damaging it, so that it ends up upside down on the plate on the stand with the pan off it." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/cake-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/cake-dream/*.json; do inspect-robots video "$log"; done
```

Environment variables for the scripted demo and the scorer:

- `CAKE_DEBUG=1` prints, every 10 steps, where the pan, the two cake halves and the plate are and how far each is turned
  over, each scoring condition, the cake's squash, shear, bend and twist, the arms' press on it (now and the peak), and
  where the grippers are.
- `CAKE_ONE=1` (the wrong way): only the left arm works and the right arm stays at its ready pose. The left gripper
  clamps the plate and the pan's handle on its side alone and lifts and turns the stack: held on one side, the stack
  comes apart part-way through the turn, the plate slides off the stand and the cake falls out of the pan onto the bare
  stand, hard enough to count as broken (its halves pulled 3.8 mm apart). The run scores 0 (706 steps).
- `CAKE_SQUASH=1` (the damage check): after the demo, the right gripper comes back over the turned-out cake with its
  fingers closed, presses down on it 2.5 cm past its top for 1 s and rises, while the left gripper keeps the pan
  sliding to and fro on the table (so the run does not end on success first) and then lets go. The cake ends upside
  down on the plate on the stand with the pan away, as full score needs, but the gripper pressed on it with up to
  9.4 N (more than 3 N), so the run scores 0 (919 steps). `CAKE_SQUASH_DEPTH` (m, default 0.025) sets how far past
  the cake's top it presses.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell: about 5.5 minutes for a Dream-1 trial of 2,400 steps (332 s on the clock), and
about 3 minutes for the scripted demo while rendering 1280 x 720 video (650 steps, 32.5 s of simulated time; 179 s).
Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

- Pan: a 14 cm round anodised-aluminium cake tin (about 2 mm sheet, 200 g), 3.6 cm deep, with an ordinary narrow rim
  (about 7 mm) and two flat tab handles 3 cm wide reaching 13 cm from its centre (tabs riveted to an ordinary tin's rim
  do the same).
- Plate: a flat white ABS plastic plate 21 cm across and 6 mm thick (220 g).
- Stand: a cake stand with a non-slip silicone top, or any 10 cm pedestal about 13 cm across with a silicone mat on it.
- Cake: a plain sponge cake disc 12.7 cm across and 2.8 cm tall, cut from a store-bought un-iced, unfilled round sponge
  cake layer at least 3 cm tall: for example Rich's Allen un-iced round sponge cake (7-8 inch layers of 11.22 oz,
  sold frozen to bakeries and restaurants in cases of 24, e.g. by WebstaurantStore, item 108RICH63784). Cut every cake
  for a series of trials from the same product, so they are alike. A disc weighs about 100 g, as in the simulation.
- Storage and temperature: keep the layers frozen (-18 °C or colder) until the day of the trials; thaw a layer in its
  wrapping at room temperature (20 to 25 °C) before cutting. Keep cut cakes covered (a lidded box) at room temperature
  and use them the same day; run the trials at room temperature.
- Preparing a cake (about 3 minutes): on a board, cut a disc with a 12.7 cm (5-inch) round cutter, then level it to
  2.8 cm by running a serrated knife along two 2.8 cm guide sticks laid either side, keeping the browned top as its
  crust; weigh it and use discs within 10 g of the first one used. Handle cakes with food-safe gloves.
- Before each trial: the cake goes crust up in the middle of the pan, the pan on the stand, the plate centred on the
  pan's rim, as in the screenshot.
- Damage check after the trial, with a steel ruler and a photo: the cake is damaged if its height, measured with the
  ruler held upright against its side at four points (front, back, left, right), is under 2.5 cm at any of them (it was
  cut to 2.8 cm: squashed by more than 3 mm); if a ruler laid across it shows a dent 3 mm or deeper under the ruler's
  edge; or if a photo from above, with the ruler beside the cake, shows a crack longer than 2 cm or a piece larger than
  1 cm broken off. A damaged cake scores 0, whatever else the run did.
- Reset (about 1 minute): brush crumbs off the pan, the plate and the grippers (the cake has no icing, filling or
  syrup, so it leaves only dry crumbs); if the cake passed the damage check, slide it back into the pan crust up on a
  flat spatula, otherwise use a fresh disc; put the pan on the stand and the plate on its rim.

In the simulation the friction is set per material: pan and plate 0.6 static / 0.5 dynamic, cake 0.7 / 0.6, the
silicone top and the gripper pads 1.2 / 1.0.
