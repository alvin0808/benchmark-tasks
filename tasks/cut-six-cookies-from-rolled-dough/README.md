# Cut six complete cookies from one sheet of rolled dough

The task uses a 52 mm round stainless steel cookie cutter on one rolled sheet. Every cut must go through the dough, stay inside the sheet and leave at least 1 mm between it and every earlier outline. The six shapes remain in the sheet for judging. The cutter goes back on the bench.

Cut-out cookies are made by stamping rolled, chilled dough. King Arthur Baking gives a [standard cut-out recipe](https://www.kingarthurbaking.com/recipes/classic-cut-out-cookies-recipe). Use supermarket sugar cookie dough, a regular 52 mm round cutter and a flat pastry board for the physical trial. Roll the dough to 4.2 mm between guides. Trim an outside rectangle of 19.8 × 13.7 cm with a ruler and remove the trim before the trial. Set up at the positions in `task.json`. Use a fresh sheet for each trial. A comparable food-safe modelling dough can stand in on shared hardware.

## Simulation

`cut_six_cookies_from_rolled_dough.py` contains the scene, scorer and scripted demonstration. It needs yam-isaac-lab-arena with Isaac Lab, real-time RTX rendering and the `bodykit.py` included here. All geometry is generated from the source. There are no third-party assets.

The model supplies a rigid open steel cutter with a thin lower rim and thicker rolled lip. The visible dough is 4.2 mm thick. Contact at its lower surface represents the resistance near the end of the cut. The farthest point of the cutting edge must come within 2.2 mm of that surface and the rim must bear on the collision layer with at least 0.8 N. The simulated cut appears as a dark circular seam where the tool made contact. The approximation skips dough flow, adhesion and lifting the cutouts.

Scoring is end-of-trial. Each intact cut is worth 0.125 to a maximum of 0.75. A cut crossing an earlier cut or the slab edge zeros the trial. After six, the cutter must be lying still, outside the slab and at least 7 mm from it with the grippers clear for 1 s. In a physical trial, count from overhead, measure the gaps with a ruler and verify the dough is cut through. An outlined shape which remains connected to the sheet after gently lifting with a thin turner does not count.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=cut_six_cookies_from_rolled_dough.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Cut six complete cookies from the rolled dough with the round cutter. Keep each cut inside the dough and separate from the others. Set the cutter down beside the dough." \
  --scorer task_score --max-steps 1200 --store-frames
```

For an independent policy run, use `--policy dream` and the same instruction. The 1,200-step limit is 60 s of simulated time. The scripted demo uses the fixed scene geometry and takes about 24 s of simulated time. Isaac rendering takes a few minutes per trial on an RTX PRO 6000 GPU. `COOKIE_OVERLAP=1` changes the third stamp so it crosses the preceding circle and gives the overlap control.
