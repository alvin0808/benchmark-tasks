# Close a deli slicer's gauge plate over the knife edge, then lift its product tray off for cleaning

A 12-inch manual deli slicer (305 mm knife), switched off and unplugged, stands on the bench with its product tray (the
carriage tray the meat rides on) parked in the home position, pulled toward the operator (the robot). Its gauge plate is
still open at 9 mm, so a strip of the knife's edge stands bare beside it. A quarter-size sheet pan stands on the bench
to the robot's right, to carry the removed parts to the sink. The instruction asks for the gauge plate to be closed over
the knife edge and for the product tray to be taken off the slicer into the sheet pan.

## The procedure and its source

The task follows the slicer makers' cleaning instructions:

- Hobart HS7 instruction manual ([PDF][hobart], "Cleaning"): "Unplug the machine power cord and turn the index knob
  fully clockwise to close the gauge plate before cleaning the slicer." "Product Tray: To clean the Product Tray, make
  sure the Gauge Plate is closed. Pull the Product Tray toward the operator until it stops. Loosen the Knob on the
  Support Arm. ... it can be removed by grasping with both hands and lifting straight up." Its features list: "Gauge
  Plate Interlock - prevents the Product Tray from being tilted or removed unless the Product Tray is in the home
  position (pulled all the way forward) and the Gauge Plate is closed."
- Globe G-series manual for the G10, G12 and G14 ([PDF][globe], "Disassembly & cleaning"): "Turn slice thickness dial
  clockwise past zero (0) until it stops so table covers knife edge." "BE ALERT. Slicer knife edge is exposed. Always be
  aware of the placement of your hands/fingers in relationship to knife."
- FDA Food Code 2022, 4-602.11: food-contact surfaces of equipment in continuous use with time/temperature-control
  foods are cleaned at least every 4 hours, so a deli slicer is taken apart several times a shift.

Food preparation workers (O*NET 35-2021.00, about 0.9 million US jobs; task statement "Clean and sanitize work areas,
equipment, utensils, dishes, or silverware") and restaurant cooks (35-2014.00, about 1.4 million; "Inspect and clean
food preparation areas, such as equipment, work surfaces, and serving areas") do this as part of their regular work.

## What the robot has to do, and why it takes two arms

1. Turn the index knob clockwise until it stops past zero: the gauge plate slides over the knife's edge in front of
   the tray. The knob sits on the base's sloping front, so the gripper turns it along its tilted axis, with a regrasp.
2. Loosen the support-arm knob under the front of the tray (a counter-clockwise turn): its lug clears the support arm.
3. Lift the tray straight up off its two hinge pins and carry it into the sheet pan.

The order is forced by the slicer itself: the gauge plate interlock pawl sits over a tab on the tray's support arm and
swings clear only while the gauge plate is closed, and the support-arm knob's lug holds the tray down until it is
loosened. The tray (2.2 kg, 19.5 x 21 cm) is heavier than one YAM arm's rated 2 kg payload; one gripper on its side
fence cannot lift it (in the simulation it slips out of a single gripper's pinch, see below), and the maker's manual
itself says "grasping with both hands". One gripper takes the front lip and the other the side fence, 20 cm apart.
The knife's slicing arc (its front-lower quarter) is bare all the time, the ring guard covering the rest of the rim, so
the hands have to stay clear of it: taking the tray by its back end puts a finger at the bare edge.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) on Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on
Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs everything, including the
Dream-1 policy. The task needs no models from Arena's object library and uses no third-party assets: every body is built
from boxes and cylinders in the task file.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a joint. The slicer
(base, knife, knife cover, ring guard, gauge plate, index knob, support-arm knob and interlock pawl), the product tray
and the sheet pan are bodies built from several shapes, and the slicer's moving parts sit on revolute joints, so
[`bodykit.py`](bodykit.py) adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives `score()` each body's pose and each joint's angle. This copy also lets a joint
turn about any direction (the index knob's tilted axis).

## What is modelled and what is not

- Knife: a 305 mm steel disc, 3 mm thick, leaning back 12 degrees (away from the tray), its hub at x 0.46, about 20 cm
  up, with an aluminium knife cover (230 mm) over its middle on the product side and a ring guard round its rim except
  the slicing arc (165 to 275 degrees, measured in the knife's plane from forward-horizontal toward up: the front-lower
  quarter). It cannot cut or spin.
- Gauge plate: 220 x 140 mm aluminium plate in the knife's plane in front of it, closed 3 mm proud of the edge and open
  9 mm at the start. The index knob's screw is not modelled: a stiff spring sets the plate to the knob's reading (12
  degrees of knob per mm; the knob stops 6 degrees past zero, as on the Globe) and the index knob turns against 0.10 N m.
- Interlock: a pawl on the carriage over a tab on the tray's support arm. The HS's cam linkage from the gauge plate is
  not modelled: the pawl is driven from the knob's reading, clear only within 0.5 mm of zero.
- Support-arm knob: on the HS it is a screw knob; here it is a knob with a cam lug that holds the support arm's front tab
  down and clears it after 90 degrees counter-clockwise (it stops at 110), against 0.15 N m.
- Product tray: aluminium, 2.2 kg (an estimate for a 12-inch slicer's tray with its support arm; weigh yours), floor 195
  x 210 mm with a 35 mm front lip and a 55 mm side fence, on a support arm with two 22 mm bores over 16 mm hinge pins 12
  mm tall. The HS lets the tray tilt to the right on its hinge before lifting; here it lifts straight off (no tilt).
- The meat grip (end weight) has already been taken off, as the manuals do first. The carriage stays in its home
  position (it does not slide).

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run.

- 0 for the rest of the run once any part of either arm enters the knife's bare edge band: every vertex of every
  collision mesh of every link of both arms (grippers and finger pads included), posed every control step, is tested
  against the band: within the slicing arc (165 to 275 degrees), 2 cm inside to 1.5 cm outside the edge's radius, from
  3 cm in front of the knife's face to 1.2 cm behind it. Where the closed gauge plate covers the edge (x 0.35 or less)
  it counts only while the plate is open. Also 0 if the tray falls off the bench.
- Otherwise the best stage reached, each needing the ones before: 0.25 the gauge plate closed (index knob within 0.5 mm
  of zero); 0.5 also the support-arm knob loosened 90 degrees or more; 0.75 also the tray off its hinge pins (lifted
  1.5 cm or more, or moved 3 cm or more from its seat).
- 1.0: the tray standing in the sheet pan (all four corners of its support arm inside the pan's rim, the support arm's
  underside within 8 mm of the pan's floor, tilted 10 degrees or less), the gauge plate still closed, both grippers 3 cm
  or more clear of the tray, the tray and the pan still for 1 s.

`-E object_jitter=0` keeps every object exactly where the task file puts it.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=close_a_deli_slicers_gauge_plate_and_lift_off_its_carriage.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The deli slicer is unplugged for cleaning. Close its gauge plate over the knife edge and take the product tray off the slicer into the sheet pan." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/slicer-dream
```

The scripted demo, which reaches full score (the left gripper, pointing along the index knob's tilted axis, turns it
clockwise in two strokes until it stops past zero; meanwhile the right gripper turns the support-arm knob
counter-clockwise in two strokes; the left gripper goes round the front of the gauge plate and takes the tray's front
lip, the right takes its side fence; together they lift the tray off its hinge pins, carry it to the right and set it
down in the sheet pan, let go and rise):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=close_a_deli_slicers_gauge_plate_and_lift_off_its_carriage.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The deli slicer is unplugged for cleaning. Close its gauge plate over the knife edge and take the product tray off the slicer into the sheet pan." \
  --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/slicer-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/slicer-dream/*.json; do inspect-robots video "$log"; done
```

`SLICER_DEBUG=1` prints, every 10 steps (`SLICER_DEBUG_EVERY=5` for every 5), the gauge reading, the plate's face
position, the support-arm knob's and the pawl's angles, the tray's position and tilt, whether it is off its pins and in
the pan, how close the nearest gripper is, how many arm vertices are in the bare edge band, each gripper's position and
the stage reached.

Wrong ways (all with the scripted demo):

- `SLICER_WRONG=1`: the gauge plate is left open and the arms go straight for the tray. The support-arm knob is
  loosened, but the interlock pawl stays over the support arm's tab and the tray does not come off: score 0.
- `SLICER_BACK=1`: the left gripper takes the tray by its back end, beside the knife: a finger goes behind the tray, into
  the bare edge band: score 0 (latched).
- `SLICER_ONE=1`: the right arm alone lifts the tray by its side fence: the tray slips out of the single pinch and stays
  on its pins: score 0.5 (gauge closed and knob loosened).

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine with two other runs on the same GPU: about 10 minutes for the
scripted demo while rendering 1280 x 720 video (801 steps, 40 s of simulated time), and about 15 minutes for a Dream-1
trial of 2,400 steps at 320 x 176. Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

Scores: scripted demo 1.0 (success at step 791); Dream-1 0.0 (the right gripper reached over the tray's side fence and the sheet pan without taking hold; the left gripper went down beside the index knob but never turned it).

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A used 12-inch manual slicer with a removable product tray and a gauge plate interlock (Hobart HS series, Globe or
Berkel; used units are widely sold, and a budget 10-12 inch commercial slicer also has a gauge plate and a
removable product tray, though not always the interlock), unplugged, and a quarter-size aluminium sheet pan. For the rig's
safety the knife may be a worn one, with the edge dulled with a stone; the bare slicing arc is outlined with tape 2 cm
round the edge, and a finger or gripper inside the outline (from the side and overhead videos) scores 0. Whether the
gauge is closed is read from the index knob's pointer (0), and the tray in the pan from a photo.

[hobart]: https://cdn.restaurantsupply.com/pdf/H/O/B/HOB_HS7-1R_DOC.pdf
[globe]: https://www.webstaurantstore.com/documents/pdf/g_series_manual.pdf
