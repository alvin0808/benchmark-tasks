# Fit two new cartridges to a half-mask respirator, holding the mask with the other hand

A reusable half-mask respirator (a 3M 6000-series half facepiece, 82 g) lies face-down on the bench on its soft seal,
chin toward the robot, both cartridge ports empty. Two new organic-vapour cartridges (3M 6001 type, 7.5 cm across,
2.5 cm thick, 50 g) lie face-up on two foam pads in front of it, one on each side, their bayonets down in the pads'
holes. The instruction asks for both cartridges to be fitted, locked on, with the mask left lying on the bench; it does
not say how.

The procedure is the maker's. 3M Half Facepiece Respirator 7500 Series user instructions (the same 6000-series
cartridges and bayonet as the 6000 facepiece), [Assembly Instructions][manual], "3M Cartridge Assembly 6000 Series":
"Align cartridge notch with arrow on facepiece, as shown, and push together (Fig. 1). Turn cartridge clockwise to stop
(1/4 turn) (Fig. 2)." The figures show the facepiece held in one hand while the other fits the cartridge. Workers who
wear these masks change cartridges on a schedule: OSHA's respiratory protection standard, [29 CFR 1910.134][osha]
(d)(3)(iii)(B)(2), has the employer implement "a change schedule for canisters and cartridges that is based on
objective information or data", and OSHA's [1998 final rule][rule] "covers an estimated 5 million respirator wearers
working in an estimated 1.3 million workplaces".

Why two hands: the mask is light and rests on a soft seal, and its ports face up and out (42 degrees from vertical).
Pushing a cartridge onto a port shoves the mask across the bench, and turning it against the squeezed gasket turns
and tips the mask with it. One hand holds the mask while the other lines the notch up with the arrow, pushes the
cartridge on and turns it to the stop; then the hands swap roles. In the simulation, the right hand fitting the right
cartridge alone (`RESP_ONE=1`) pushes the mask 14 mm across the bench before the lugs find the slots, then, as it turns
the cartridge, tips the mask 19 degrees on its seal and shoves it 5.6 cm away: the cartridge never goes on and the run
scores 0.

The task is the task file
[`fit_cartridges_to_a_half_mask_respirator.py`](fit_cartridges_to_a_half_mask_respirator.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task uses no third-party assets: every object is built in the task file
from boxes, cylinders and one convex hull.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them. The mask (shell, rigid frame,
two bayonet ports with their lips, slots, stops, pockets and foam gaskets, the exhalation valve cover) and each
cartridge (body, spigot, two lugs) are bodies built from several shapes by `bodykit.py`: importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each body's pose. This copy
of `bodykit.py` also lets some shapes of a body (the gaskets) have compliant contacts.

## The bayonet, as modelled

Each port is a boss on the mask's frame, its axis tilted 42 degrees out from vertical. In the port's own frame (z out
along the axis, 0 on the gasket's face):

- a foam gasket ring (radius 15.5-21 mm, 2 mm thick) with compliant contacts (10 kN/m per contact patch; the demo's
  push squeezes it 0.6-0.9 mm), lying on a rigid lip (radius 15-23 mm, 3 mm thick);
- two slots, 10 mm wide, through the gasket and the lip, one at the yellow arrow on the boss's rim and one opposite it;
- under the lip, a groove 7 mm deep (to radius 19 mm) with stops: one beside each slot, so the lugs cannot go
  counter-clockwise, and one a quarter turn clockwise, where they stop;
- over the last lug-width before the stop the lip's underside is 0.5 mm higher: a pocket the lug springs into (the
  click), which keeps the cartridge from turning back unless it is pushed in again.

Each cartridge has a spigot (radius 13 mm, 9 mm long) with two lugs (reaching 18 mm from its axis, 6 mm wide, 3 mm
tall); its notch (a white mark on the rim) marks one lug. With the cartridge's face on the uncompressed gasket the
lugs' tops are 0.7 mm above the lip's underside, so they pass under the lip only while the cartridge is pushed in,
squeezing the gasket. Let go, the gasket pushes the cartridge back out until the lugs bear on the lip (or sit in the
pockets): it is locked and cannot be pulled off.

What is modelled and what is not: the facepiece is one rigid convex shell (a real one is soft silicone or elastomer on a
rigid frame; here only the frame's ports matter, and they are rigid on the real mask too). The cartridge is a round
disc of the real cartridge's size; the real 6001 is a rounded, slightly oval body. The head harness's elastic straps are
drawn lying slack behind the mask but do not collide. The gasket's give is a compliant contact rather than a foam
ring.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run (the end
state, so a cartridge pushed on and later knocked off no longer counts). For each cartridge it reads, from the poses of
the mask and the cartridge, where the cartridge sits on its own port:

- 0.2: pushed onto the port: its axis within 8 degrees of the port's, its centre within 3 mm of the port's axis, and its
  face within 0.5 mm of the gasket's face or squeezing it (only possible with the lugs down through the slots);
- 0.45 instead: also turned clockwise 80 degrees or more from the slot (the stop is a quarter turn on; with the lugs'
  play they settle at 95-100 degrees in the simulation): lugs under the lip, locked.

The two add up (0.9 with both locked). Full score, 1.0: both cartridges locked, the mask lying face-down on the bench
(tilted 15 degrees or less, its seal within 5 mm of the bench), every finger 15 mm or more from the mask and both
cartridges, and the mask still (under 1 mm of movement in 1 s), held for 1 s. Doing nothing scores 0. `-E
object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same layout.

On a real mask the same checks are made by eye and hand: each cartridge cannot be pulled off and its notch sits a
quarter turn clockwise from the arrow (at the stop), the mask lies on the bench, and the robot is not touching it.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=fit_cartridges_to_a_half_mask_respirator.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit both new cartridges onto the respirator so that each is locked on, and leave the respirator lying face-down on the bench." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/respirator-dream
```

The scripted demo, which reaches full score (the left gripper comes down the left port's axis and pinches the empty
port's boss, holding the mask; the right gripper takes the right cartridge off its pad, turns it to point down the right
port's axis with its notch at the arrow, pushes it on past the gasket's face and turns it clockwise to the stop, then
keeps holding it, which now holds the mask; the left lets go of the boss, takes the left cartridge and fits it the same
way; both let go and draw back along the ports' axes):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_cartridges_to_a_half_mask_respirator.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit both new cartridges onto the respirator so that each is locked on, and leave the respirator lying face-down on the bench." \
  --scorer task_score --max-steps 1040 --store-frames --log-dir ~/yam-eval/respirator-demo
```

The demo is open loop from the known layout. Where the mask lies while each cartridge goes on was read from the
simulation, which is deterministic, and is written in the task file (`RESP_MASK_AT_RIGHT`, `RESP_MASK_AT_LEFT`): the
left hand's pinch on the boss shifts the mask about 3 mm before the right cartridge goes on, and the mask settles a few
millimetres from there, held by the right hand through the fitted cartridge, before the left one goes on.

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/respirator-dream/*.json; do inspect-robots video "$log"; done
```

`RESP_DEBUG=1` prints, every 5 steps (`RESP_DEBUG_EVERY=10` for every 10), for each cartridge its axis angle to its
port, how far off-centre it is, how far its face is into the gasket, how far it is turned and its stage; the mask's
pose, tilt and lift; the nearest finger; and each gripper's centre, opening and turn.

Wrong ways: `RESP_ONE=1` fits the right cartridge with the right hand alone, the left arm staying at the ready pose:
the push shoves the mask 14 mm across the bench, the turn tips it 19 degrees and slides it 5.6 cm away, the cartridge
never goes on, and the run scores 0. `RESP_MISALIGN=1` holds each cartridge with its notch 45 degrees clockwise of the
arrow when pushing it on: the lugs land on the gasket, not in the slots, turning only takes them further from the slots,
nothing goes on, and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 4 minutes for the scripted demo while rendering 1280 x 720
video (1,019 steps, 51 s of simulated time; 243 s, with two other runs on the same GPU), and about 7.5 minutes (449 s) for a
Dream-1 trial of 2,400 steps. Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A 3M 6000-series half facepiece (6100, 6200 or 6300) and a pair of 6001 organic vapour cartridges, both sold in
hardware stores, and two 9 x 9 cm squares of 12 mm foam with a 4.6 cm square hole each, taped to the table.
Reset: turn both cartridges back a quarter turn counter-clockwise, pull them off and lay them face-up on the pads with
their notches as in the screenshot, and lay the mask face-down in its place.

[manual]: https://images.toolsid.com/3m/items/pdf/37083-manual.pdf
[osha]: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.134
[rule]: https://www.osha.gov/laws-regs/federalregister/1998-01-08
