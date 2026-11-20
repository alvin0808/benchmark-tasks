# Light a gas hob burner with a flame-failure valve, keeping its knob pushed in for 10 s after it lights

A 60 cm four-burner built-in gas hob is set into a piece of kitchen worktop on the table, its four control knobs in a
row at the front right. The instruction is "Light the front right burner so that it stays lit." It does not say how.

Almost every gas hob sold today has a flame failure device on each burner: a thermocouple sits in the flame and, once
hot, holds a magnet valve open. Until it is hot, only the pushed-in knob holds that valve open, so the knob has to be
pushed in, turned, and then kept pushed in for several seconds after the flame lights. Electrolux UK support
([One of the rings on my Gas Cooker/Hob won't stay alight?][elx]): "Push the control knob down and turn it
counterclockwise to the maximum position" ... "After you have lit the flame, keep the knob pushed for at least 10
seconds." Belling's GTG60 manual ([page 9][bel], automatic ignition with flame failure safety device): "To light the
burner, push the relevant control knob down and turn it anti-clockwise to the large flame position. Wait for the gas
to light and keep the control knob held down for 10 seconds after the flame has lit, to allow time for the flame
failure safety device to warm up, before releasing the control knob." Let go too soon and the flame goes out, which
is the most common complaint the support page answers.

The hand skill: the right knob (the third of four, 6.5 cm apart) is taken by its 11 mm grip bar without touching its
neighbours; it is pushed 6 mm in against its valve spring (8 N, 11 N fully in) and, while kept in, turned a quarter
turn counterclockwise (it will not turn from OFF unless pushed in 4.5 mm); then it must be kept pushed in, at that
angle, for 10 s after the flame lights, with an arm that sags under a steady load, and finally let go without turning
it back. One arm is enough; the task does not claim two arms are needed. The left arm stays at its ready pose in the
demo.

The task is the task file [`light_a_gas_hob_burner_holding_the_knob_in.py`](light_a_gas_hob_burner_holding_the_knob_in.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: every part is built in these files from boxes and cylinders.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a hinge or a
spring. The hob, the knobs on their sprung spindles and the flames are bodies built from several shapes, so
`bodykit.py` adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without
changing yam_arena, and gives the task file each body's pose after every control step.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, heights above the table top. Both arms start at the ready pose, grippers open. The layout is the same in every
trial.

- Worktop, fixed (taped down): a 38 mm laminate worktop piece, 72 x 62 cm, on 22 mm spacers (its top 6 cm up), its
  front edge at x 10.5, centred on y 0, with the hob's cut-out.
- Gas hob, fixed: a 60 cm four-burner built-in hob (stainless top 59 x 51 cm, x 13 to 64, y -29.5 to 29.5, its top
  6.8 cm up). Burners, each an aluminium crown with a black enamel cap, a copper thermocouple tip and a white igniter
  electrode beside it: back left at (53, 15), rapid, 9 cm crown; front left (33, 15), semi-rapid, 7 cm; front right
  (33, -15), auxiliary, 5 cm; back right (53, -15), semi-rapid, 7 cm. Two cast-iron pan supports, 3 cm tall, cover
  the left and right halves from x 24.5 to 62.5.
- Control knobs, black, 30 g each, in a row at x 17.5: y 2 (back left burner), -4.5 (front left), -11 (front right,
  the one to light), -17.5 (back right). Behind each, a printed 2 x 2 dot symbol with its burner's dot white. Each
  knob is a 40 mm skirt, 10 mm tall, standing 7 mm above the top on its spindle, with a grip bar 34 mm long, 11 mm
  wide and 15 mm tall on top; at OFF the bar runs front to back and its white pointer (at the front end) points at
  the white OFF dot. Marks: large flame (maximum) at 90 deg counterclockwise seen from above (to the knob's right),
  small flame at 180 deg (the stop). The knob is pushed in up to 6 mm against its valve spring (8 N at rest, 11 N
  fully in); a pin on its spindle, under the top, sits beside a stop lug at OFF, so it turns from OFF only while
  pushed in 4.5 mm or more.

What is modelled and what is not. Everything is rigid; there is no gas and no fire. The knob, its spindle on the
spring, the pin and the stop lug are physical; the gas, the igniter, the thermocouple and the safety valve are a rule
on the knob's angle and push (below), and a flame (blue tongues round the crown) is drawn while a burner is lit, as a
visual cue only. The knob turns against light viscous damping, with no detents; the stop lug also stops it on the
way back, so it needs the push to turn back to OFF (a real valve has a ramp there). The spring forces and the travel
are estimates for a flame-failure valve. The pan supports and burners are simplified shapes.

The gas and flame rule (per burner, every 50 ms control step; 0 deg is OFF, counterclockwise positive):

- The knob counts as pushed when it is 4 mm or more in. While it is pushed, the igniter sparks and the push holds the
  safety valve open.
- A burner lights when its knob is pushed and turned 20 deg or more from OFF.
- While lit and pushed, its thermocouple warms. After 10 s of flame with the knob pushed (counted from lighting; a
  let-go of 0.5 s or less does not break the count) it is hot and holds the valve open by itself.
- If the knob is let go for more than 0.5 s before that, the valve shuts and the flame goes out. Trying again is
  allowed.
- Turning the knob under 20 deg puts the flame out and lets the thermocouple cool (it must be held again).

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run, which
is the best stage reached:

- 0.25: the front right burner was lit and its knob kept pushed 5 s or more after lighting.
- 0.5: kept pushed 10 s or more after lighting (the thermocouple is hot: the flame now holds by itself).
- 1.0 (success, held 1 s): the front right burner burning, every finger point of both grippers (from the YAM's finger
  collision meshes at the simulated poses) 3 cm or more from its knob, the knob still (under 1 deg of turn and 1 mm of
  push over 1 s), and no other burner lit.
- Otherwise 0: doing nothing, lighting it and letting go at once (the flame goes out after less than 5 s), or
  refusing.

A real trial uses a real hob with flame-failure valves and ignition in the knob, NOT connected to gas, with its
igniter powered (mains or battery, as the hob is made), so nothing can burn. The knob, its spring, the OFF stop and
the safety valve are the hob's own and work as with gas; the igniter clicks and sparks only while a knob is pushed in.
The trial is scored by the same rule from the trial video (overhead and wrist cameras, with sound): the knob's angle
from its pointer and the printed marks, "pushed" from the igniter's clicks, and the times from the video's clock
(lit = sparking with the knob 20 deg or more from OFF; the flame "holds" if the sparking went on 10 s or more after
that, with no gap over 0.5 s).

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=light_a_gas_hob_burner_holding_the_knob_in.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Light the front right burner so that it stays lit." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/hob-dream
```

Its API refused the instruction before acting, twice on this scene, with `PolicyError: DreamAPI refused the
inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction, refused: Lighting burners involves open
flames or high heat, posing potential safety hazards.`, so there is no Dream-1 video; it scores 0.

The scripted demo reaches full score open loop from the known layout. The right gripper, pointing straight down with
its fingers closing across the front right knob's grip bar, comes down over the knob with its fingers part open (a
32 mm gap, clear of the neighbouring knobs), pinches the bar with its fingertips just above the skirt, and pushes down
(aimed 26 mm below the pinch; the arm sags under the spring and the knob goes fully in, 6 mm). Keeping it in, it
turns the knob counterclockwise to about 95 deg in 2.4 s (the burner lights as the knob passes 20 deg), keeps it
pushed in for 12 s (the thermocouple is hot 10 s after lighting), opens its fingers, rises straight up and goes
back to its ready pose; the knob springs back out and rests at about 93 deg, burning. In our runs it scored 1.0 at
step 475 (about 24 s).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=light_a_gas_hob_burner_holding_the_knob_in.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Light the front right burner so that it stays lit." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/hob-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/hob-demo/*.json; do inspect-robots video "$log"; done
```

`HOB_DEBUG=1` prints, every 5 steps (`HOB_DEBUG_EVERY=10` for every 10), each knob's angle and push, each burner's
state (lit, thermocouple hot, seconds of flame with the knob pushed, how often it went out), the best such time for
the front right burner, the fingers' clearance from its knob, whether it is still, whether another burner is lit, and
each gripper's position and opening. `HOB_PUSH`, `HOB_TURN_TO`, `HOB_HOLD`, `HOB_PINCH` and `HOB_GRIP_DZ` change the
demo's push depth, turn, hold time, finger setting and grip height.

`HOB_QUICK=1` runs the demo the way a person who does not know about the safety valve would: it pushes the knob in
and turns it to the large flame exactly as the demo does, then lets go 0.3 s later. The burner lights at about step
153 and burns while the knob turns, but the flame goes out 0.5 s after the let-go (about step 209), after 2.8 s of
flame with the knob pushed, and the run scores 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs: debug runs of the demo at 320 x 176
(680 steps, ending on success at step 475) took about 3 minutes, and so did the demo while rendering 1280 x 720 video
(475 steps) and the wrong-way run (500 steps). A Dream-1 run is refused before the first step, so it takes about 2
minutes. Isaac Sim's first start on a new machine adds a few minutes while it compiles
its shaders.

## Seeds

The layout is fixed (`object_jitter=0`); the scripted demo and the scorer are deterministic up to PhysX's own
run-to-run variation.

## Building it for real

- Hob: any 60 cm four-burner built-in gas hob with flame-failure devices and ignition in the knobs (push and turn,
  knobs at the front right), such as the common Electrolux, Zanussi, AEG, Bosch or Belling models, set into a worktop
  offcut with the maker's cut-out, the worktop on spacers and taped to the table at the position above. If its knob
  layout differs, place it so the front right burner's knob is at (17.5, -11).
- Gas: none. The hob's gas inlet is left unconnected (capped), so no gas can flow; the igniter is powered as the
  maker installs it, so it sparks while a knob is pushed in. Keep the burner caps seated.
- Real trials are safe: there is no fuel, the only energy is the igniter's spark at the burner, 15 cm or more from the
  knobs; the arm never needs to reach the burner.

[elx]: https://support.electrolux.co.uk/support-articles/article/one-of-the-rings-on-my-gas-cookerhob-wont-stay-alight
[bel]: https://www.manualslib.com/manual/1507683/Belling-Gtg60.html?page=9
