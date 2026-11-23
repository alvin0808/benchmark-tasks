# Trim a marked acrylic practice-nail tip with a tabletop nail clipper

An adapted tabletop nail clipper replaces the small flip lever on an ordinary clipper with a palm-size pad. The [Maddak 74140-0000](https://www.wholesalepoint.com/mm5/graphics/00000001/documents/ableware-741400000.pdf) is one example. Place a nail-art practice tip in a reusable swatch handle. Cut a marked 4 mm strip all the way across the tip. One hand controls the curve and feed depth at the jaw while the other depresses the lever. The cut and clipping are measured directly. Use an inert curved nail-art tip for all trials.

## Setup

Positions are relative to the midpoint between the two arm bases, 61 cm apart. Positive X is forward and positive Y is toward the left arm. Heights are measured from the table.

- Tabletop lever clipper on a 13.3 × 6.7 cm nonslip base. The open jaw center is at (34.0, 2.6, 3.2) cm. The steel blade is 17 mm wide, the resting jaw opening is about 3.1 mm and the two jaw faces meet when the 88 mm lever reaches the end of its stroke. The thumb pad is 42 × 37 mm and points rightward from the jaw.
- A curved 11 mm acrylic or ABS nail tip is attached to a reusable beige swatch handle. Together they are about 12.2 × 2.3 cm. A contrasting fine stripe marks a line 4 mm from the free end. The nail points rightward, toward the clipper. Set the handle in the 5.5 × 8.8 cm ABS rest centered at (23.0, 19.4) cm.
- Start both grippers open in the ready pose. There are no randomized props. The supplied screenshot shows the first frame of the scripted run.

## Scoring

| Score | Observable result |
| --- | --- |
| 0 | No alignment or clipping, or the clipper cuts past the marked range at any point |
| 0.25 | With the jaws open, the horizontal cut line lies within 1.5 mm of the blade in depth and height, within 2.8 mm laterally, and approximately square to within 10.5° of yaw and 11.5° of level |
| 0.75 | A full-width clipping between 2.5 and 5.5 mm long separates at the marked line |
| 1.0 | The cut qualifies for 0.75 and the nail is resting within 2 cm of its start on the support, with both gripper centers at least 6.8 cm from the handle, continuously for one second |

Score the cut by measuring the strip down its center and confirming that it separates across the whole width. Use side and overhead video of the blade and a calibrated scale in the image for partial alignment. Verify the return and hand clearance from the overhead video. A visibly off-line cut invalidates later states.

The simulator uses a sprung steel jaw driven through actual pad-to-cam and cam-to-jaw contact. The offcut is initially fixed to the practice tip. The joint breaks only when blade contact exceeds 0.24 N, the remaining blade gap is no more than 1.4 mm, and the printed stripe meets the alignment tolerances. The force threshold detects loaded contact in the model. The hardware procedure below measures the actual fracture load.

For a hardware trial use a tabletop nail clipper sized for the same stock. Mark the strip with a ruler or gauge, verify the entire free edge fits between the jaws, and test spare tips by hand before each batch. A push gauge at the pad center can record peak force and travel for that actual stock. Confirm a smooth full-width cut with no longitudinal cracking. Use a fresh tip in the same position every run, remove the clipping from the workspace, and align the two bases with their desk marks. The task does not require powered equipment or a person in the workspace.

## Run the simulation

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at `0059d02` with Isaac Lab-Arena 0.3.1 and Isaac Sim 6.1. Run both commands from this folder.

```bash
instruction=$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=trim_a_practice_nail_with_a_tabletop_clipper.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=720 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames

inspect-robots run --embodiment yam_arena --policy dream \
  -E task=trim_a_practice_nail_with_a_tabletop_clipper.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames
```

For the feed-depth control set `CLIP_OFF_LINE=1` on the scripted command. It advances the tip 6 mm beyond the nominal feed target while leaving the pad motion and scoring intact. The demo reaches full score in 304 steps (15.2 seconds of simulated time). The run limit is 1,200 steps (60 seconds). The layout and script are deterministic at zero object jitter and the default seed of 0. Start a fresh process for each trial.
