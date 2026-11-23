# Center a benchtop balance's bubble with its two leveling feet

A precision balance must sit level before weighing. Turn the two front wheels to center the bubble and leave the
balance standing on its own. The round vial shows both axes at once. Extending both screws raises the front, and
extending one relative to the other rolls the balance sideways. This is the [procedure for a precision balance
with a fixed rear foot and two adjustable front feet][procedure].

## Setup

The bench is level. The balance sits loose on it with its control panel facing the robot.
[Build the physical fixture from the dimensioned drawing and parts list](FIXTURE.md). It is 2.2 kg with a
polymer housing 221 × 179 mm at the main body, approximately 285 mm across both corner thumbscrews. Its stainless
pan is 144 × 153 mm, 75 mm above the body origin, and empty. The analog level is beside the display, with a green
27 mm field, an 8.7 mm inner target ring, and a 5.8 mm diameter bubble. The three rubber-bearing feet carry the
case. The two steel thumbscrew wheels are 45 mm across, with dark knurled edges. The printable wheel blank, lead screw and stationary pad are dimensioned in the fixture drawing. The fixed rear foot sits 203 mm
behind their line. The left and right front pads are 118 mm either side of the midpoint, initially 3.1 and 5.4 mm
higher than the rear pad in the balance's frame. Both touch the bench as the case settles off level.

Coordinates in metres from the arm bases' midpoint, with x pointing away from the robot and y positive toward the
left arm. The balance origin is (0.352, 0.000) at 1 mm above the bench on reset. Front pads are at x = 0.236 m, the
rear pad is at x = 0.439 m and y = 0. No tool is needed. Both grippers start open at the station's ready pose.
For a repeat trial restore the indexed offsets using the rear pad sole as the datum and let the case and bubble settle.

## Real trial and scoring

Film the two adjustments, then take an overhead view of the vial and a side view of the feet after both hands
withdraw. A 1.6 mm feeler and side view suffice to verify all three feet are bearing. Both hands are at least
38 mm clear and the balance is stationary. The full score needs the entire bubble within the inner target ring
for 2 seconds. A stable finish with the bubble centre within 3 mm of the target centre gets 0.5. Other outcomes
get 0. A side view over the hold also shows that the case is resting without any hand supporting it.

In simulation, the full score checks total tilt from horizontal (0.16° maximum), bubble centre offset (1.45 mm
maximum), proximity of all three pads to the bench plane (1.6 mm), and a 38 mm clearance between sampled finger
points and the balance envelope. The largest pose change over 0.25 s is at most 0.6 mm and 0.13°. Full score
must hold for two seconds. Partial credit applies to a stationary, released finish within 0.40° and with bubble
centre offset at most 3 mm. At the end of a run, the scorer checks the state then visible.

The scale is a floating articulation on its three colliders. Each front wheel turns via gripper contact. Its measured
rotation drives a load-bearing pad through the relation for a 3 mm-pitch screw. Pad motion acts against the bench
and supports the full case. The bubble's two slide joints follow the projection of gravity on the vial and settle
with damping. The display is decorative. The screw threads and shape of the fluid surface are approximated.

## Reproduce

Install [yam-isaac-lab-arena][arena] with Isaac Lab-Arena and use a GPU. The task assets are generated from its
Python files. Use a fixed reset (`object_jitter=0`), with no random seed required. From this directory, run:

```bash
task=center_a_benchtop_balances_bubble_with_its_leveling_feet.py
instruction="Use the two front leveling wheels to center the bubble on the balance. Then move both grippers away and leave the balance resting on its feet."

inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./balance-demo

inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./balance-dream
```

Run the scripted command again with `BALANCE_RIGHT=0` to turn just the left wheel. The case stays visibly off level.
`BALANCE_DEBUG=1` prints the foot heights, tilt and bubble offsets every ten control steps. To render recordings, run
`inspect-robots video` on the emitted trial JSON in the log directory. Wall-clock runtime is a few minutes per trial
on one RTX PRO 6000. Dream-1 uses the configured Dream credentials and a maximum of 60 simulated seconds.

[procedure]: https://www.mt.com/dam/P5/labtec/17_Miscellaneous/QG_ML-T_Precision_and_Analytical_Balances_EN.pdf
[arena]: https://github.com/kairosiann/yam-isaac-lab-arena
