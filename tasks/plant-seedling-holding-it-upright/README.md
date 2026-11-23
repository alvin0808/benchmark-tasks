# Plant a seedling: hold it upright in the hole with one hand while tipping the soil in with the other, then let it stand

A seedling stands in a small nursery cell on a potting bench, next to a square planting hole, and an open-fronted tray
of soil lies at the hole's edge. The instruction asks for the seedling to be planted in the middle of the hole, upright,
its root plug on the bottom of the hole, with the hole filled with the tray's soil up to the top of the plug all round,
and to be left standing on its own. The scene, scorer and scripted demo are in
[`plant_seedling_holding_it_upright.py`](plant_seedling_holding_it_upright.py) for the `yam_arena` embodiment, with
[`bodykit.py`](bodykit.py) next to it.

Transplants are placed at their previous depth and backfilled around the root plug. Here the hole is twice as wide as the
plug and as deep as the plug is tall. A plug on the floor sits at its prior growing depth, and the rim provides a
straightedge reference for the final surface. [UMass Extension][umass] · [Royal Horticultural Society][rhs] ·
[New Zealand Forest Service][nz]

The seedling's round-bottomed plug tips when released onto the empty hole's flat floor. One hand holds the plant
upright in the middle of the hole while the other tips the tray. The plant stands on its own once the medium settles
around the plug.

The two-arm coordination occurs during the pour, while the rounded plug is still unstable. Related plant-manipulation
evaluations measure forceful insertion into a prefilled pot or count two scoops before planting the flower.
[Plant Insertion Task, §5.2](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/ErikHelmut/msc_erik_forceful_umi.pdf) ·
[ReCAT PLANT](https://corl2026-memory.github.io/papers/011.pdf)

## The soil: how much there is and how its level is measured

The 8 × 8 × 4.5 cm hole holds 288 mL. The rounded 4 × 4.5 cm plug displaces about 48 mL, leaving about 240 mL
at rim height. The tray has a nominal internal capacity of about 373 mL below the side walls. Each physical run starts
with 275 ± 5 mL of loose, screened growing medium measured in a graduated cup without tapping or compression. This
leaves about 35 mL of material beyond the nominal volume needed around a centered plug.

The simulated load is 140 particles, each 12 mm across, with about 127 mL of solid material in total. The voids between
particles add bulk volume. Score the actual settled surface shown by the demo, since the packing and spilled volume
vary with the pour. In the full-score run, 130 particles settled in the hole and the four probes read 52, 51, 48
and 48 mm above the floor. The rim is 45 mm above the floor. The highest crown reading was 12 mm above the plug.

Tape four probe indicators on the fixture, 10 mm in from the midpoint of each side. Place a straightedge across the
rim and gently lower a flat-ended 10 mm dowel at each point until it contacts the settled medium. Read each depth
against the straightedge with a ruler. A probe that rests on the plug fails that point. For full credit, all four
spots must be within 10 mm of the rim. Mark the plug's crown height on the stem before the trial. Measure the
plug's height and tilt and put a ruler beside the hole in the wrist or side view. Before the pour, subtract the
plug height times the cosine of its tilt from the crown height to confirm that the plug reaches within 5 mm of the
floor. With the same dowel, measure the maximum surface height above the crown mark at
four points 12 mm from the stem, one on each side. Full credit allows up to 15 mm over the crown. The simulator computes
where the circular probe tip first touches the surface of a stationary soil particle. It excludes particles in the
tray or within 30 mm of a gripper. [Royal Horticultural Society planting-depth check][rhs]

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task uses no third-party assets and no models from Arena's object library:
every object is built in the task file from boxes, cylinders and spheres.

`yam_arena` task files place single boxes, cylinders and spheres. The seedling (a sphere and a cylinder for the root
plug, a stem and three leaves) and the tray (a floor with ribs, three walls and a handle) are loose rigid bodies built
from several shapes, so `bodykit.py` adds such bodies: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives `score()` each body's own pose and velocity. Its step hook also lets the task file
compute both grippers' positions from the arms' joint positions (yam_arena's own URDF kinematics), which the scoring
uses to check that the seedling stands on its own; and it lets the demo's waypoints carry a full gripper rotation, for
the side grip on the stem.

The loose backfill consists of 140 rigid 12 mm particles with friction 0.3 and rolling damping. The spheres flow
around the plug as a coarse granular medium. In physical soil, moisture can add cohesion and roots and stems can flex. The tray's floor has ribs 1 mm high across it, which hold the crumbs while
the tray lies flat and let them all roll out once it is tipped. The seedling is one rigid body whose mass follows its
shapes' volumes; the plug's round bottom stands for a root ball that does not stand by itself. The planting hole is
square, in a fixed bench.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run, which the
task file keeps as the best stage reached (held or not): 0.25 when the seedling is planted (upright within 10 degrees,
its plug's axis within 1 cm of the hole's centre, its bottom within 5 mm of the hole's floor) with the soil at all four
spots at least halfway up the hole (2.25 cm); 0.5 when it is planted with the soil at all four spots at most 1 cm below
the rim and at most 1.5 cm of soil on the plug's top. Full score: all of that with both grippers 8 cm or more from the
seedling's axis (open fingers reach about 5.5 cm from a gripper's centre) and the seedling still (moved under 1.5 mm and
turned under 1 degree over 1 s), for 1 s. Counting crumbs earns nothing: only the soil's level around the plug does.
`-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=plant_seedling_holding_it_upright.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Plant the seedling in the middle of the planting hole, upright, with its root plug on the bottom of the hole; fill the hole with the soil from the tray up to the top of the plug all round; and leave the seedling standing on its own." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/plant-dream
```

The scripted demo, which reaches full score (the left gripper takes the seedling's stem from its own side, lifts it
out of the cell, carries it over the hole and lowers it until the plug sits on the floor, then holds it upright; the
right gripper takes the tray's handle from above and lifts the tray's back, so that the tray turns about its front edge
at the hole's rim and the crumbs roll into the hole all round the plug; it lowers the empty tray and lets go; then the
left lets go and draws back 10 cm, and the seedling stands in the soil):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=plant_seedling_holding_it_upright.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Plant the seedling in the middle of the planting hole, upright, with its root plug on the bottom of the hole; fill the hole with the soil from the tray up to the top of the plug all round; and leave the seedling standing on its own." \
  --scorer task_score --max-steps 640 --store-frames --log-dir ~/yam-eval/plant-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/plant-dream/*.json; do inspect-robots video "$log"; done
```

`PLANT_DEBUG=1` prints, every 10 steps (`PLANT_DEBUG_EVERY=5` for every 5), how far the seedling leans, how high its
plug's bottom is and how far from the hole's centre, the soil level at the four spots and on the plug's top, how many
crumbs lie in the hole and in the tray, how far the tray is tipped, how close the nearest gripper is, whether the
seedling is still, the stage reached and where each gripper is; every 50 steps also a map of the soil surface over the
hole.

`PLANT_LET_GO=1` runs the demo the wrong way: the left gripper lets the seedling go as soon as its plug is on the floor,
before the soil goes in. The seedling falls over in the empty hole (to about 48 degrees), the soil fills in around it
lying there, and the run scores 0.

Controlled pours verify the scoring conditions.

- `PLANT_CRUMBS=45` starts with the original charge size. A total of 34 settled particles reached the hole, the
  final four readings were 12 mm and the released seedling leaned about 67 degrees. The score was 0.0.
- `PLANT_TILT=25` reduces the commanded tray tilt. A total of 116 settled particles reached the hole and the
  released seedling stood upright. The underfilled side measured 21 mm, below the 22.5 mm partial-credit threshold.
  The score was 0.0.
- `PLANT_HOVER=1` holds the plug 1.5 cm above the floor while the soil goes in, so soil runs under it: the plug ends
  11 mm above the floor and the run scores 0.

## Runtime per trial

On one RTX PRO 6000, the scripted demonstration reached 1.0 in 488 steps, or 24.4 seconds of simulated time.
Rendering the demo at 1280 × 720 took about 5 minutes. Dream-1 reached the 2,400-step limit with a
score of 0.0 in about 22 minutes on the same GPU. Isaac Sim may compile shaders on its first startup.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

Use a basil seedling with two to four true leaf sets, a 10–15 cm tall shoot, and a coherent rooted plug. Grow it in a
commercial Jiffy-7 propagation pellet or obtain a nursery starter of the same size. Water the pellet until it has
expanded, then drain it for 20 min and choose an intact, moist plug that is 38–42 mm across and 43–47 mm tall. The
rounded base should fit a 40 mm hemispherical measure without crumbling. Before the trial, check that the cell supports
the seedling and that it tilts past 10 degrees within 1 s when placed unsupported on the flat hole floor. Return the
plug to the cell without packing in loose substrate. Use a fresh, similarly sized seedling if the plug breaks or
the shoot is damaged. Use soft silicone pad covers to isolate the shared grippers from the damp plug. Validate the closing force with a
sacrificial rooted starter from the same lot. The starter should lift without slipping or visible bruising.
[Jiffy-7 propagation pellets][jiffy] · [Kansas State University Extension transplanting procedure][kstate]

Make the fixture from a 280 × 360 × 45 mm plywood or HDPE block with an 80 × 80 mm square opening. Attach a flat
5 mm base under the block to form the hole floor, then clamp the assembled fixture to the bench. Transfer the cell,
hole and tray positions in the setup. Use a 90 × 130 mm plastic tray with 35 mm walls on three sides, an open fourth
side, and a 20 × 40 mm grasp tab centered on the opposite wall. Use exterior tape to mark the tray's home position
and a shallow catch mat under the fixture to collect any spills.

For repeatable backfill on shared equipment, use Plagron Euro Pebbles expanded clay, screened to 10–14 mm and rinsed
and air-dried before use. At reset, pour 275 ± 5 mL into the rear of the tray, leaving the first 20 mm behind the lip clear. Record the first charge's mass and use the same batch for resets. Recover spills and clear all material
from the hole before the next trial. Check the first batch with a manual trial: after the plant is upright in the
hole and the measured charge is tipped from the tray, the four level readings must be within 10 mm of the rim, the
plant must stay upright after release and the crown must stay visible. The bench protocol still needs hardware
verification. [Plagron Euro Pebbles technical data sheet][pebbles]

[umass]: https://www.umass.edu/agriculture-food-environment/home-lawn-garden/fact-sheets/planting-vegetable-garden
[rhs]: https://www.rhs.org.uk/plants/types/shrubs/how-to-plant-a-shrub
[nz]: https://www.canopy.govt.nz/establish-forest/plant-seedlings
[jiffy]: https://jiffygroup.com/products/jiffy-pellets/?noredirect=en-US
[kstate]: https://hnr.k-state.edu/extension/horticulture-resource-center/horticulture-newsletter/documents/2024/april2024/transplanting%20success.pdf
[pebbles]: https://plagron.com/downloads/en/euro-pebbles-tds
