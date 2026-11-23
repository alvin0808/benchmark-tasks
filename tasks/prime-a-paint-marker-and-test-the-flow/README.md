# Prime a dry paint marker and test the flow

This trial starts with a pre-shaken, uncapped blue water-based paint marker with a new dry nib. Prime the felt nib by pressing it against rough scrap paper and letting it return several times. Once paint reaches the tip, draw an unbroken test stroke and set the marker down on the bench.

A valve pen gives tactile feedback as its nib retracts under load. Each compression and return admits ink. The following test stroke needs continuous surface contact while moving the longer barrel sideways at a usable writing angle. The maker describes both steps in the [POSCA user guide](https://www.posca.com/en-uk/hints-and-tips/instructions/). The real setup uses a PC-5M or comparable 3 mm valve marker, a pen tray and rough white scrap paper. To repeat, fit a dry replacement nib, shake the marker while capped and remove the cap before placing it in the tray. Draw two short ticks 75 mm apart and a longer priming tick as shown in `task.json`.

## Simulation and scoring

`prime_a_paint_marker_and_test_the_flow.py` defines the scene, scoring and scripted demo for yam-isaac-lab-arena. The barrel and spring nib are collision geometry, as is the sheet. No third-party assets are used. A passive 3.5 mm slide and a spring approximate the valve. Priming requires load against the paper and 2.65 mm or more of stroke, separated by spring return. After four pumps the point can leave paint on the sheet. A normal writing angle allows marking with much less nib travel. The renderer draws dots at the real contact points, including spring compression in the surface contact band between reported force samples. A visible line is the combined ink trail. Reservoir mixing and absorption are omitted.

Read the end score. Unprimed is 0. Blue paint first transferred to paper is 0.25. An unbroken stretch of at least 35 mm along the test interval is 0.5. At least 70 mm between the ticks within an 18 mm lateral distance, with the marker laid still on the bench and hands clear for 1.1 s, is 1.0. Use the marked paper and a ruler in a physical trial.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=prime_a_paint_marker_and_test_the_flow.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Prime the blue paint marker on the scrap paper. Draw one unbroken test line between the two short gray ticks, then set the marker down beside the paper." \
  --scorer task_score --max-steps 1200 --store-frames
```

Use `--policy dream` for the Dream-1 run. Both policies use the fixed layout, with a 1,200-step limit (60 s simulated). The demo normally runs about 23 s of simulated time. With RTX rendering, allow a few wall-clock minutes for each run. `MARKER_ONCE=1` performs a single pump and then tries the same writing stroke.
