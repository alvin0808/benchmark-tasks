# Press a flat parchment sling into a loaf pan and leave both lift tabs standing

A parchment sling keeps baked goods from sticking to a loaf pan and gives the baker two handles for lifting the cooled loaf. [King Arthur Baking demonstrates the technique](https://www.kingarthurbaking.com/blog/2017/03/27/parchment-paper-sling). This trial stops after forming the liner, while the pan is still empty. Press the flat parchment near both short ends so it folds down the end walls and lies flat across the floor. Both tabs should remain accessible above the rim after hands leave.

## Setup

Use the midpoint between the two YAM arm bases as the origin. The bases are 61 cm apart. X points forward and Y points toward the left arm.

- Put an empty metal loaf pan at (36.9, 0.0) cm with its long direction spanning the arms. The pan is 22.5 cm long, 12.4 cm wide and 5.9 cm high at the outside rim. The floor is approximately 20.1 × 10.1 cm and 6 mm above the table. The two short walls flare out toward the rim.
- Cut a 36.8 × 9.4 cm strip of food-grade baking parchment. Lay the strip flat across the two short rims with its midpoint over the pan center, leaving about 7.3 cm beyond each end. Both edges remain free. The floor section between the two simulated folds is 19.2 cm long.
- Start at room temperature with a dry pan, fresh parchment, and both arms ready. The scene's setup image is the beginning of the scripted trial.

For a hardware trial, cut the strip to within 2 mm of the specified dimensions. Verify that the sheet spans the pan and fits between the long walls when pressed to the floor. Use a clean, cool, dry pan. Light pre-creasing during the attempt is allowed. Begin each trial with uncreased paper. Replace the strip after every attempt.

## Scoring

| Score | Observable result |
| --- | --- |
| 0 | The center misses the floor or a tab is folded underneath |
| 0.25 | The center is flat within 5 mm of the floor, within 6 mm on each horizontal axis, with its plane and long axis within 3.7° of the floor and pan axis |
| 0.50 | The center qualifies and one end follows its end wall with a tab that extends 18–40 mm above the rim |
| 0.75 | The center and both end tabs qualify |
| 1.0 | Both tabs remain correctly formed for one second, with the paper still and both gripper centers at least 52 mm above the rim |

Score alignment from the overhead image and confirm the center and tabs using a side view and ruler. Each tab should be over its corresponding end wall. In the simulation the end panel points at least 64° upward, its lower midsection is 95–116 mm from the pan center along the long axis, and the tab's tip is 100–137 mm from center along that axis. Confirm the bottom can receive batter freely.

The simulation represents the sheet as three thin rigid panels with soft joints at the two dominant folds. Gripper and pan contact drives the folding. The model approximates flex at the two folds and leaves small wrinkles, fibrous tearing and adhesion out of the model.

## Reproduce the trial

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02`, Isaac Lab-Arena 0.3.1, and Isaac Sim 6.1. Run the command below in the task folder.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_a_parchment_sling_into_a_loaf_pan.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 800 --store-frames
```

For the Dream-1 run select `--policy dream` and camera dimensions 320 × 176. For the under-travel control set `PARCHMENT_SHALLOW=1` on the scripted command. The control stops the grippers near the pan rim. Each trial uses a fresh process, fixed layout and default seed 0. The run limit is 40 simulated seconds. The full-score demo lasted 152 steps, or 7.6 simulated seconds.
