# Stretch a reusable silicone lid over the four corners of a square food pan

An 8-inch-class square stretch lid covers a pan of leftovers by catching its elastic skirt below the rolled rim. A [Stasher square stretch lid](https://www.stasherbag.com/products/square-stretch-lid) is an example of this kind of cover. Select a matching pan and lid whose two pull tabs can be pinched across opposite ends. The two ends should need approximately 16–22 mm of extension each before they pass the rim. Pull them apart, carry the lid level, lower the skirt past the flange, and let the material tighten around the wall.

## Setup

Measure relative to the midpoint between the two arm bases. The bases are 61 cm apart. Positive X is forward and positive Y is toward the left arm.

- A square tempered-glass pan has 20.4 cm outside walls, a 21.6 cm continuous flange, and a rim 6.5 cm above the table. Its center is at (44.3, 0.0) cm on a 23.1 cm square nonslip silicone mat. Keep the pan empty and dry.
- A reusable silicone lid, skirt-down at (21.1, 0.0) cm, has a membrane approximately 2 mm thick. The 1.8 cm deep skirt is 23.0 cm across X and 18.4 cm across the tab axis Y at rest. The molded pull tabs are 22.8 cm apart at rest. Place the skirt with the tabs pointing toward the two arm bases.
- Start both arms in their ready positions. The screenshot is the first frame of the demo with zero object jitter.

For hardware, first measure the outer rim and the inside of the two unstrained skirt ends. Choose an 8-inch-class pan with a continuous flange rather than a square pan with protruding carrying handles. Verify by hand that the skirt can clear the flange without excessive pull and stays on all four corners after release. A ruler on the four corners can check the skirt sits below the rim. Dry and reset the lid and pan for each trial. No food, heat or pressurized containers are needed.

## Scoring

| Score | Observable result |
| --- | --- |
| 0 | The lid is outside the pan opening or above the rim by more than 9 mm |
| 0.25 | The lid covers the opening, within 8 mm of center, 2 mm below to 9 mm above rim height, and within about 4.5° of square and level |
| 0.50 | The lid covers the opening and one end is seated outside the wall and 6–18 mm below the rim |
| 0.75 | Coverage and both ends are seated outside the wall and 6–18 mm below the rim |
| 1.0 | Both ends and all four skirt corners are seated outside the pan wall with the corners 6–20 mm below the rim. The lid stays still and both grippers remain at least 4.2 cm from the tabs for one second |

Use an overhead view for alignment and release. Inspect both ends and all four corners against the rim with a ruler. The model approximates the membrane with overlapping panels, with two sprung end sections providing the elastic extension. Glass-rim contact retains the skirt after release. The task scores physical seating. Vacuum pressure and liquid retention are outside the scope of the physical seating trial.

## Reproduce the trial

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02` with Isaac Lab-Arena 0.3.1 and Isaac Sim 6.1. From the task folder use the following command.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=stretch_a_silicone_lid_onto_a_square_food_pan.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames
```

Use `--policy dream -E cam_width=320 -E cam_height=176` for the Dream-1 run. For the no-stretch control set `LID_NO_STRETCH=1` on the scripted command. The arms grasp and carry the two tabs at their resting separation in this control. Each run starts a new process with zero object jitter and default seed 0. The run limit is 60 seconds of simulated time.
