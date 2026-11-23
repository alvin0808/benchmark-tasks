# Draw the air out of a reusable food bag with a manual pump while holding its valve seal

A hand vacuum pump removes air from a reusable zipper food bag without opening the zipper. One hand keeps the rubber foot square and pressed against the valve. The other moves the plunger through several cycles. Lifting the foot on the pull stroke lets room air leak into the seal. The bag remains collapsed after the pump is removed. The [QuickSeal user manual](https://images.hs-plus.com/assets-dashboard/product-manuals-v2/QUICKSEAL_prirocnik_SI.pdf) illustrates positioning the pump over a bag valve, evacuating the bag and removing the pump.

## Setup

Two YAM arms are 61 cm apart. Positions are measured from the midpoint with X forward and positive Y left.

- A 21.8 × 22 cm reusable food bag lies flat at (37.4, 2.2) cm with the zipper fully closed. Dry crackers occupy the end away from the valve. The free-air pocket initially rises about 44 mm from the table. Its measurement area is approximately 60 × 60 mm.
- The 40 mm one-way valve is at (37.4, −6.1) cm and is about 11 mm above the table at its sealing floor. The matching hand pump sits upright on it. A rubber ring fits concentrically over the valve's boss. The contracted pump is about 14 cm high and the piston has 57 mm of usable travel.
- Clear a parking area centered at (21.7, −16.0) cm for the pump. Use a removable outline on a physical table if the boundary is hard to judge. Both arms start in the ready pose.

For a hardware run use a matched pump and bag whose gasket seats around its valve boss. Verify the valve stays shut after removing the pump. Place the same quantity of dry food in the bag for each trial. Reset by opening the valve and zipper, reintroducing room air, and closing the zipper. Check the marked side pocket is 44 ± 2 mm high before reseating the pump. Use a dry table and room-temperature props.

## Scoring

Count complete strokes over a continuously seated valve from the side video. A complete stroke extends the piston at least 45 mm and returns it to within 13 mm of the contracted position. A ruler alongside the pocket measures height. Intermediate points require both the stroke count and the visible reduction.

| Score | Observable result |
| --- | --- |
| 0 | No completed sealed stroke or pocket above 38 mm |
| 0.25 | At least one complete sealed stroke and pocket at or below 38 mm |
| 0.50 | At least two complete sealed strokes and pocket at or below 29 mm |
| 0.75 | At least three complete sealed strokes and pocket at or below 18 mm |
| 1.0 | The 0.75 state persists for one second after the pump is upright within 37 mm of its parking center. The bag stays flat, the pump is still, and each gripper center is at least 36 mm from the pump shaft, 40 mm from the plunger handle, and 80 mm from the pocket |

The piston slides in a prismatic joint with estimated sealing drag. The rubber foot and valve physically contact. The simulation counts a stroke only when the foot is within 6 mm laterally and 5 mm vertically of the mating point, within about 13° of upright, and carries at least 0.38 N of contact load. A completed cycle changes the target of a spring on the plastic pocket. The pocket's motion gives an observable height and approximate swept volume. It excludes gas-flow transients and long-term leakage.

## Reproduce the trial

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02`, Isaac Lab-Arena 0.3.1 and Isaac Sim 6.1. Run this from the task folder.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=evacuate_a_reusable_food_bag_with_a_hand_pump.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames
```

For Dream-1 select `--policy dream` and camera dimensions 320 × 176. For the unbraced control set `VAC_NO_BRACE=1` on the scripted command. This withdraws the right hand until the attempted strokes finish. Use fixed layout, default seed 0 and a fresh process for each run. The evaluation limit is 60 simulated seconds.
