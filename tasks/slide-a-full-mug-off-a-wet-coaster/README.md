# Slide a filled mug free of a wet coaster without lifting the coaster or spilling water

Condensation can collect at the contact ring of a glass or mug and make a smooth coaster stick when the drink is lifted. The [description of a condensation-resistant coaster](https://patents.google.com/patent/US20200146485A1/en) documents the problem. This task uses a smooth glazed coaster. Steady the coaster at an exposed corner, slide the mug sideways until the wet film opens, and move the drink to a neighboring pad while keeping the surface level.

## Setup

Use the midpoint between two YAM arm bases 61 cm apart. X points forward and positive Y points toward the left arm. The table is level and dry apart from the specified area.

- Place a smooth 12.8 cm square ceramic coaster on cork feet centered at (37.6, −4.8) cm. Its finished upper face is about 8 mm above the table.
- Use an 8 cm diameter, 8.6 cm high handled mug filled with room-temperature water to 32 mm below the rim, about 190 mL with this taper. The total mass in simulation is 290 g. Center it at (37.6, −3.7) cm so an opposite corner remains exposed. Point the loop toward the left and 30° forward.
- Center a 14.5 × 11.6 cm flat wooden pad at (37.6, 7.5) cm. Its top is approximately level with the coaster and its closest edge is 1 mm from the coaster. The pad has a nonslip backing.
- Start with both arms ready. Use the frame at step zero of the scripted run as the setup image.

For hardware, use a smooth, chip-free base and a smooth glazed tile coaster. Apply approximately 0.2 mL of tap water at the mug's bottom contact ring using a marked dropper. Press the mug down under its own weight for five seconds. Qualify the pair before each batch by confirming on a spare setup that a gentle vertical lift carries the coaster and a gentle level slide separates the pair. Record the cup mass before the test. Dry the landing pad and desk, renew the film and repeat the setup for every trial. Measure spill by looking for water outside the mug and weighing the mug again to the nearest gram.

## Scoring

| Score | Observable result |
| --- | --- |
| 0 | The shift is under 7 mm, the coaster rises more than 4 mm or shifts more than 9 mm, the mug tilts more than 10° from upright, or any water spills |
| 0.25 | The mug shifts at least 7 mm horizontally relative to its starting contact point on the coaster |
| 0.75 | The mug is separate and its center is at least 10.7 cm from the center of the coaster |
| 1.0 | The mug is resting within 1.2 cm of the pad's center for one second, with each gripper center at least 4.8 cm from the handle and 9.5 cm from the mug base and coaster center |

Use overhead and side views with a ruler to judge the motion and final state. The model uses a compliant planar joint for the wet film with a 15 mm nominal shear release. Gripper and table contact determine motion. The cup's orientation and filtered linear acceleration estimate spill through the free-surface angle. Film adhesion and transient sloshing are approximated. Confirm those effects with the repeatable hardware test above.

## Reproduce the trial

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02`, Isaac Lab-Arena 0.3.1 and Isaac Sim 6.1. Run this command from the task folder.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=slide_a_full_mug_off_a_wet_coaster.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 1000 --store-frames
```

Use `--policy dream` and camera dimensions 320 × 176 for Dream-1. Set `WET_LIFT_ONLY=1` before the scripted command for the control. It holds the coaster and attempts a vertical lift without applying shear. Each trial uses a new process, zero jitter and default seed 0. The run limit is 50 seconds of simulated time.
