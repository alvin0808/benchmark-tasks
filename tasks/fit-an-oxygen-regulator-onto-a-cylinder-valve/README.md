# Fit a yoke oxygen regulator onto a cylinder's pin-index valve: hold it on its hidden pins and tighten the T-handle

A full size D aluminium medical oxygen cylinder lies in a wooden cradle, its brass CGA 870 pin-index post valve
pointing at the robot with the valve's outlet facing down. A yoke oxygen regulator stands upright on the table to the
right, its sealing washer lying on its inlet. The instruction asks only for the result: the regulator fitted onto the
valve, hand-tight and ready to use, and let go of.

The procedure is the regulator maker's (Graham-Field, *Oxygen Regulator with Yoke Connector*, user manual for
JB0150-081/082/083, "Setup",
[PDF](https://shop.grahamfield.com/nosync/productimagesV2/ProductAdditionalInfoItemOriginal9529.PDF)):

> Ensure the regulator inlet sealing washer is in place. [...] Slip the yoke over the cylinder post and fit the pins
> into the holes on the cylinder valve. The pins should fit easily into these holes. Tighten the T-handle firmly by
> hand; turn it clockwise until the screw is seated in the dimple on the cylinder valve. Ensure the regulator is
> tightened securely into the cylinder. [...] If you can hear a hissing sound, there is a leak in the system.

The cylinder lies in a cradle because a D cylinder standing upright puts its valve 50 cm up, beyond the arms' reach;
it is turned so that the outlet faces down, which keeps the difficulty a person meets with an upright cylinder (there
the yoke slides down the post off its pins unless it is held; here it drops off them).

The valve and the yoke follow ISO 407 (*Small medical gas cylinders - Pin-index yoke-type valve connections*, Table 2
basic dimensions; oxygen uses pin positions 2 and 5, on a 14.3 mm radius round the outlet). What makes it hard:

- **Blind alignment.** Once the yoke is over the post, its two index pins and the valve's two pin holes are hidden
  inside it. The pins (4 mm) enter their holes (4.75 mm with a 0.55 mm countersink) only within about 0.9 mm along the
  post, so the yoke has to be pressed up and moved along the post by feel until the pins drop in and the washer meets
  the valve's face. A yoke whose pins miss stands about 4 mm proud of the face and cannot be tightened flush.
- **Held while tightening.** With the outlet facing down the regulator hangs under the valve: it stays on its pins only
  while it is held up. Let go before the T-screw is tight and it drops 5 mm off its pins. One hand therefore holds it up
  on its pins while the other turns the T-handle, half a turn at a time with regrips, about five and a half turns
  until the screw's point is seated in the recess and the handle is tight. This is not just "hold one part and turn
  another": what is held is a blind alignment that is lost the moment the hand lets go.
- **The washer.** The sealing washer lies loose on the inlet round its spigot. Tipping or knocking the regulator
  while carrying it loses the washer (and without it, or with two, the connection leaks); that scores 0.

The task is the task file [`fit_an_oxygen_regulator_onto_a_cylinder_valve.py`](fit_an_oxygen_regulator_onto_a_cylinder_valve.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) and
[`linkkit.py`](linkkit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. No third-party assets are used: every object is built
from boxes, cylinders, spheres and capsules in the task file, and the task needs no models from Arena's object
library.

`yam_arena` task files place single boxes, cylinders and spheres. The cylinder with its valve, the cradle and the
washer are rigid bodies made of many shapes (`bodykit.py`); the regulator is one loose PhysX articulation
(`linkkit.py`): the yoke with its body, gauge and pins, the T-screw on a slide joint along the clamp axis, and the
T-handle on a turn joint on top of it. Importing them hooks `yam_arena.scene` when the simulator starts, without
changing yam_arena, and gives `score()` each body's pose and each joint's position.

## What is modelled

All sizes are true size; the post and yoke follow ISO 407 Table 2.

- **Cylinder:** Catalina MD (size D) aluminium cylinder, 111.3 mm across, 424 mm from neck to base, full (3.4 kg with
  0.6 kg of oxygen and the valve), green shoulder and white label, loose in a fixed cradle of two wooden V saddles and
  an end board, its axis 9.2 cm above the table.
- **Post valve (CGA 870):** the post 25 mm thick (outlet face to recess face) and 22.2 mm wide, its free end 16 mm in
  front of the outlet's centre with its four edges chamfered 2.5 mm; the outlet port (7.2 mm) in the bottom face; the
  two oxygen pin holes (4.75 mm, 6 mm deep, 0.55 mm 45-degree countersinks to 5.85 mm) 13.6 mm from the outlet toward
  the cylinder and 4.42 mm either side (14.3 mm radius, 18 degrees either side); the conical recess (6 mm, 2.1 mm
  deep) in the top face opposite the outlet; the spindle's square end and gland nut on the free end, a safety relief
  plug on the left side, the round valve body into the cylinder's neck. The valve stays closed.
- **Regulator (0.34 kg, 14.6 cm from the dial's bottom to the T-handle's top):** a chrome yoke with a window 24 mm
  wide and 36 mm from the inlet face to the top bar (ISO 407 allows the valve at most 6 degrees of turn in it before
  the pins engage; here about 5.7); on the inlet face a 6.3 mm brass spigot 3.3 mm proud and two 4 mm steel pins
  5.5 mm proud with flat ends; the M12 T-screw (1.75 mm per turn) through the top bar, its point a 2.5 mm ball, backed
  off so the point is 33.8 mm above the inlet face; a black T-handle 60 x 14 x 22 mm; under the yoke the green
  regulator body (38 mm across), the black flow dial (40 mm), the contents gauge facing the cylinder and a hose barb.
  It stands on its dial with the inlet's centre at x 20.5 cm, y -10 cm, 6.5 cm above the table.
- **Sealing washer:** 16 mm across, 6.6 mm hole, 1.6 mm thick (ISO 407 g), a loose rigid ring lying on the inlet face
  round the spigot. It weighs 20 g in the simulation (a real one 2 to 3 g): squeezed between the yoke and the valve, a
  lighter body sank into the inlet face under the clamp force.
- **The T-screw (PhysX has no screw joint):** the screw part slides along the yoke's clamp axis on a drive (100 kN/m)
  whose target follows the T-handle's turn at 1.75 mm per turn, clockwise seen from above tightening it. Until the
  point is stopped it simply follows; once the point sits in the recess and the yoke is drawn against the washer, the
  drive's lead times its stiffness is the clamp force (capped at 120 N, the thread then slips). The T-handle turns
  against Coulomb friction of 0.05 N m plus 0.019 m times the clamp force, so turning it further takes 0.75 N m at
  37 N (hand-tight) and it stays where it is left (the thread does not unwind).

Not modelled: the washer's rubber (it is rigid), gas and pressure, the regulator's insides, round holes (the pin
holes, port and recess are square pockets with 45-degree mouths). A real hand-tight clamp is several hundred newtons;
here it is capped at 120 N.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the episode, which
the task file judges itself from the end state (`SCORE_OVER = "end"`):

- **0 for the rest of the run** once the sealing washer has come off the regulator's inlet (its centre more than 4 mm
  from the spigot's axis, or more than 4.5 mm above or 1.5 mm below the inlet face: it is no longer round the spigot),
  the cylinder has left its cradle (moved 3 cm or tilted 10 degrees), or the regulator has fallen off the table.
- **0.25 seated:** the yoke's inlet face 1.1 to 2.1 mm below the valve's outlet face (the 1.6 mm washer between them),
  within 2 degrees of square and within 2 mm of the outlet's axis, both pins 3 mm or more into their holes, and the
  washer centred on the spigot (within 1.5 mm) between the two faces. A yoke whose pins rest on the face is 5.5 mm
  below and scores 0.
- **0.5:** seated, and the screw's point down in the recess (within 1.5 mm of its axis, 0.8 mm or more below the top
  face).
- **0.75:** that, and hand-tight: turning the T-handle further takes 0.75 N m or more (a clamp force of 37 N).
- **1.0 (`success()`):** that, with both grippers' centres 6 cm or more from the T-handle and from the regulator's
  body, and the regulator and the T-handle still (moved under 1 mm and turned under 2 degrees) for 1 s.
- Doing nothing scores 0.

`-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=fit_an_oxygen_regulator_onto_a_cylinder_valve.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the oxygen regulator onto the cylinder's valve, hand-tight and ready to use, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/o2-dream
```

The scripted demo reaches full score in 1,800 steps (90 s of simulated time):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_an_oxygen_regulator_onto_a_cylinder_valve.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the oxygen regulator onto the cylinder's valve, hand-tight and ready to use, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/o2-demo
```

What the demo does (open loop, from the known layout only; it reads no object state):

1. The right gripper, level and pointing forward, takes the regulator's body from the robot's side, lifts it upright
   (the washer stays on the inlet) and carries it in front of the valve, the yoke's window level with the post.
2. It slides the yoke over the post, lifts it until the pins nearly meet the valve's face, then presses up lightly and
   moves it slowly back and forth along the post until the pins drop into their holes and the washer meets the face,
   and holds it up there.
3. The left gripper, pointing down, takes the T-handle and turns it clockwise half a turn at a time: it lets go,
   rises clear of the handle, turns back, comes down and takes it again. The eleventh turn runs the point into the
   recess and goes on 35 degrees further until the handle is tight (in our run: clamp force 57 N, 1.14 N m to turn
   further).
4. The right gripper lets go and moves away, then the left.

The demo's few calibration constants (the right arm's sag and drift while it holds the regulator at the valve) come
from our own runs; they are named at the top of its section in the task file.

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/o2-dream/*.json; do inspect-robots video "$log"; done
```

Environment variables for the scripted demo and the scorer:

- `O2_DEBUG=1` prints, every 10 steps (`O2_DEBUG_EVERY` to change), where the yoke is in the valve's frame (offset,
  tilt, roll/pitch/yaw), both pins' depth, the gap between the inlet and the outlet face, where the washer is on the
  inlet, where the screw's point is, the T-handle's turns, the screw's slide and its target, the clamp force and the
  T-handle's friction, each scoring condition, and where both grippers are.
- `O2_ONE=1` (the wrong way: one hand at a time): the right gripper seats the regulator on its pins exactly as in the
  demo, then lets go of it and moves away before the left gripper turns the T-handle, as a single hand would have to.
  Let go, the regulator drops 7.5 mm off its pins at once and settles 4 mm back along the post, tipped 8 degrees.
  The left gripper's turns then run the screw's point into the recess and clamp the yoke there with its pins resting
  on the valve's face beside their holes: the inlet stands 3.7 mm from the face instead of 1.6 mm, so it is not
  seated, and the run scores 0 (stopped at 2,300 steps). A real regulator fitted like this leaks.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell: about 13.5 minutes for a Dream-1 trial of 2,400 steps (803 s on the clock,
Dream answering in 0.65 s on average), and about 10.5 minutes for the scripted demo while rendering 1280 x 720 video
(1,800 steps, 90 s of simulated time; 626 s). Isaac Sim's first start on a new machine adds a few minutes while it
compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

- Cylinder: a size D aluminium medical oxygen cylinder (for example Catalina MD, 111 mm across, 424 mm long) with a
  CGA 870 post valve, full, valve closed during the trials. Turn it in the cradle so the outlet faces down.
- Cradle: two plywood V saddles (90-degree V, 3 cm thick) on a base screwed or clamped to the table, holding the
  cylinder's axis 9.2 cm above the table, 14 cm and 40 cm behind the outlet; an upright end board behind the base.
- Regulator: any CGA 870 yoke flow regulator with a T-handle, for example the Graham-Field John Bunn JB0150-082
  (0-15 L/min, about 0.3 to 0.4 kg), and one new sealing washer (Bodok-type or the plastic crush washer supplied with
  the cylinder). Back the T-handle off until its point is about 34 mm above the inlet face.
- Scoring with simple tools: a photo and a feeler gauge or ruler for the gap between the inlet and the valve's face
  (seated: only the washer between them, under 2.1 mm; a yoke whose pins miss stands about 4 mm proud); a small
  torque gauge on the T-handle (hand-tight: 0.75 N m or more to turn it further; a spring scale hooked 3 cm out on the
  handle reads 25 N); and the maker's own check after the trial: open the cylinder valve one turn with the cylinder
  key, the regulator's gauge shows the cylinder's pressure and there is no hiss; close the valve and bleed the
  regulator before taking it off.
- Reset (about 1 minute): back the T-handle off to its start, slide the regulator off, put the washer back round the
  spigot (a new one if it was lost) and stand the regulator on its dial at its start; check the cylinder lies in both
  saddles against the end board.
