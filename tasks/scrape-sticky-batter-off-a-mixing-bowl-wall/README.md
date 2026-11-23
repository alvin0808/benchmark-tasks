# Scrape sticky batter down the curved wall of a mixing bowl with a silicone spatula

A flexible silicone spatula recovers batter that clings to the side of a bowl. The bowl is supported with one hand while the other keeps the edge in contact as the wall curves toward the floor. This is the scrape performed while preparing thick batters. [King Arthur Baking describes which batters remain on the wall and how scraping is used](https://www.kingarthurbaking.com/blog/2020/06/24/do-i-really-have-to-scrape-my-mixing-bowl).

## Setup

Use a level table with two YAM arms 61 cm apart. Measure from the midpoint, with X forward, positive Y left and heights above the table.

- A 1-quart glass mixing bowl, 17.8 cm in diameter and 7.6 cm tall, rests loose at (34.9, −4.2) cm. The [Pyrex 322](https://pyrexhome.com/collections/pyrex-all/products/pyrex-1-quart-mixing-bowl) gives the nominal shape and scale. Keep the bowl empty apart from the batter.
- Apply one strip on the curved wall about 25 mm wide and 4–7 mm thick along the left-facing inner wall. Its 2.6 g lower, middle and upper sections are centered at heights of 3.4, 5.2 and 6.5 cm. The lowest portion sits about 46° toward the far side of the bowl, following the curve. The three adjacent portions make movement from each part of the wall observable.
- A small silicone spatula about 21 cm long stands blade-down in a 6.5 × 6.8 cm pickup rest at (23.6, 18.4) cm. The working edge is about 3.3 cm across and the rounded handle is about 2.2 cm wide. The [OXO small silicone spatula](https://www.oxo.com/oxo-gg-silicone-cookie-scoop-and-spatula-set.html) illustrates the shape and use. Match the handle and blade measurements for repeatability.
- Start with both arms ready. Place a dry bowl and tool at their marks, clean the utensil rest, and use thick batter at 22 ± 2 °C. A gram scale and reusable 25 mm spreader can reset the portions. Verify before each batch that the batter stays where placed for at least two minutes under gravity and slides when worked with the spatula. Use the same preparation for each run.

## Scoring

Award 0.25 for each of the three sections when it reaches the bowl's bottom. Check the empty stripe and the collected material in overhead and side views. A collected section has its center within 6.3 cm of the bowl center horizontally and between 0.5 and 3.1 cm above the table. The possible intermediate scores are 0, 0.25, 0.50 and 0.75.

Award 1.0 when all three portions are collected and the following state holds for one second.

- The bowl remains released, still, within 3 cm of its original position and within about 8° of upright.
- The silicone working edge is still and at least 3.5 cm above the table. It is at least 4 cm outside the bowl wall or at least 3.5 cm above the rim.
- Each gripper center is at least 4 cm outside the bowl wall or at least 4.3 cm above the rim. The tool can remain in the working gripper.

The bowl has faceted convex contact sections. A small sprung joint approximates the deflection of the silicone blade. Pressing down against a segment with the spatula detaches its mount when loaded contact is at least 0.42 N, a simulation estimate. Gravity, viscous damping and contact with the bowl determine its final location. The model represents recoverable clinging portions of thick batter. Thin-film transfer and bulk fluid mixing are outside the task.

## Reproduce the trial

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02`, Isaac Lab-Arena 0.3.1 and Isaac Sim 6.1. Run this from the task folder.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scrape_sticky_batter_off_a_mixing_bowl_wall.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames
```

The scripted reference follows the positions of the working edge and gripper during the scraping stroke. For Dream-1 select `--policy dream` and camera dimensions 320 × 176. For the inward-offset control set `BATTER_INWARD_OFFSET=1` on the scripted command to move the edge inward from the wall. Use a fresh process and fixed layout per trial, default seed 0 and 1,200 steps. The simulation limit is 60 seconds at 20 Hz.
