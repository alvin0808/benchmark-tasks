# Set a turntable's stylus tracking force to 2 grams on a gauge

Adjust the threaded counterweight until the stylus bears on its gauge at 2.0 grams. Use the tonearm's finger lift
to clear the gauge during adjustment and lower the stylus afterwards. Read the gauge with both hands away from the
arm. A manual turntable needs this adjustment to track a record without excessive pressure. The procedure follows
the turntable manufacturer's guidance for protecting the stylus and the external gauge method described by Shure.
[Audio-Technica manual][turntable] · [Shure stylus gauge instructions][gauge]

## Setup

The unplugged turntable is a manual deck with a 352 × 452 mm plinth, a 300 mm platter with a rubber mat, and an
installed S-shaped arm. The horizontal pivot-to-stylus distance is 230 mm. The rear counterweight is a steel
cylinder 37 mm across, 31 mm long and 92 g in mass, fitted to the threaded rear stub. The headshell has its original
metal finger lift. The platter is stationary and bare. A tared stylus gauge on the mat has a 16 × 16 mm contact pad
and a green readout. The stylus starts on the pad and reads about 2.9 g with both grippers at their ready poses.

The tonearm pivot is at (0.372, −0.086) m from the midpoint between the gripper bases, 110 mm above the worktop.
The coordinate x points away from the arms, and y increases to the left. The gauge pad is approximately at
(0.148, −0.034) m. The counterweight starts centred 54.5 mm behind the pitch bearing. To reset, move the weight
to the start, lift the stylus to tare the gauge, then lower it back onto the pad and check the starting reading.

## Scoring and simulation

Record the hands, the pad and the gauge readout. With the stylus on the central 13 × 13 mm area of the gauge pad,
both hands at least 63 mm from the stylus and counterweight, and the reading stable, 1.80–2.20 g for 1.5 seconds
earns 1.0. A released and stable reading within 1.60–2.40 g at the end earns 0.5 if full credit did not hold.
Other outcomes earn zero. Lowering the tip onto the bare platter or pressing the stylus onto the gauge above 15 g for 0.10 s
gives zero. A view from above checks pad placement and a side view checks the gap to the mat.

The tonearm swings independently in yaw and pitch. The counterweight's measured rotation moves its actual mass along
the rear shaft at 10 mm per revolution. The normal contact force between the stylus and the pad drives the displayed
reading. The scorer uses a ten-frame rolling average and requires the pitch velocity below 0.014 rad/s at the
reading. The gauge is fixed in its initial position for repeatability. The tip uses a 0.42 mm collision radius for
the simulator's contact solver. The model represents static force calibration. The motor and audio are inactive.

## Reproduce

Install [yam-isaac-lab-arena][arena] and use a GPU. The objects are generated from the task files. Use the fixed
reset below, with no random seed required.

```bash
task=set_a_turntables_stylus_tracking_force.py
instruction="Adjust the tonearm counterweight so the stylus reads 2.0 g on the gauge. Use the finger lift to move the stylus clear while adjusting, then lower the stylus onto the pad and withdraw both hands."

inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./tracking-demo

inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./tracking-dream
```

`TONE_NO_ADJUST=1` repeats lifting and lowering the stylus without adjusting the weight. `TONE_DEBUG=1` logs the gauge
load, position and joint angles. To render the trial, call `inspect-robots video` with its result JSON.
The scripted trial takes about a minute on an RTX PRO 6000. Dream-1 requires configured Dream credentials and
runs up to 60 simulated seconds.

[turntable]: https://docs.audio-technica.com/eu/AT-LP120XUSB_UM_162501380_V1_EN_web_190312.pdf
[gauge]: https://www.shure.com/en-US/docs/guide/SFG-2
[arena]: https://github.com/kairosiann/yam-isaac-lab-arena
