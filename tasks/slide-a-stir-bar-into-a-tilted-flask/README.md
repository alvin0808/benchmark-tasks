# Slide a stir bar into a tilted round-bottom flask instead of dropping it in

An empty 250 mL round-bottom flask stands in a cork ring on the table, and a PTFE magnetic stir bar lies on the table
to its left. The instruction asks for the stir bar to be put into the flask and the flask left standing in its cork
ring; it does not say how.

A stir bar dropped down the neck of the upright flask falls about 10 cm onto the flask's thin round bottom.
Chemistry teaching labs say to slide it in instead. The University of York's Chemistry Teaching Labs page
[Stirring & heating][york] says "Gently slide the magnetic stirrer bar into the glassware", because "dropping it in may
cause damage"; its photo shows a round-bottom flask held on its side while the bar is guided into its neck. With the
flask turned well over, a bar laid in the mouth slides down the neck and along the inside wall instead of falling onto
the bottom.

Why two arms: a round-bottom flask cannot stand outside its cork ring and rolls if laid on the table, so it has to be
held, by its neck, while it is turned over; the bar has to be laid in the mouth by another hand at the same time. One
hand cannot hold the flask turned over and lay the bar in. The task is the task file
[`slide_a_stir_bar_into_a_tilted_flask.py`](slide_a_stir_bar_into_a_tilted_flask.py) for the `yam_arena` embodiment
(scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. No third-party assets are used: every object is built in the task file from
boxes, cylinders and capsules, and the glass look is Isaac Sim's built-in OmniGlass material.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The flask (a bulb made
of 148 flat glass plates and a bottom disc, and a neck made of 24 plates), the cork ring (48 blocks) and the stir bar (a
capsule) are bodies built from several shapes or from shapes yam_arena lacks, so `bodykit.py` adds such bodies:
importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives `score()` each
body's pose. This copy of `bodykit.py` also has capsule shapes, a glass look and a friction-combine setting per body.

What is modelled and what is not: the flask is at its real size (a 250 mL flask: an 8.5 cm bulb of 2 mm glass, 14 cm
tall, a neck 2.8 cm across with a 2.2 cm bore whose top 4 cm is the frosted 24/40 ground-glass joint, 0.12 kg). The
bulb's inside is a polyhedron of flat plates touching an 8.1 cm sphere, smooth enough for the bar to slide on. The bore
is a straight 22 mm (a 24/40 joint's bore narrows from 24 to 20 mm). The stir bar is a 25 x 8 mm capsule of 5 g; its
magnet does nothing here. The cork ring (10 cm across, 3.4 cm tall) has a sloping seat the bulb rests on, and is fixed
in place. Glass slides on everything with friction 0.12 (PTFE on glass is about 0.1); the gripper pads hold the bar
with 0.55. The glass cannot break in the simulator: the impact rule below stands for it.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
bodies' poses come from the simulation, each gripper's centre from its arm's joints and its finger opening from its
finger joints. Every physics step (240 Hz) the scorer measures how fast each end of the bar moves toward the bulb's
inside wall (relative to the flask) while it is within 12 mm of the wall.

The score is 0 for the rest of the run once:
- the bar meets the bulb's inside wall at 1 m/s or faster (dropped in rather than slid in: as fast as a free fall of
  5 cm). A bar let go with its end 8 mm down the neck of the upright flask hits the bottom at 1.3 m/s (in a separate
  test, one released centred in the mouth hit it at 1.6 m/s); laid in the mouth of the flask turned 70 degrees over, it
  slides down the neck and meets the wall at 0.8 m/s;
- the flask touches the table (its lowest point within 4 mm of it: it can only stand in its ring), is dropped (moves
  down faster than 0.6 m/s while not held) or tips over (tilted more than 45 degrees while not held);
- the bar, once wholly inside the flask, comes back out (its middle leaves the flask).

Held means a gripper's centre within 1.5 cm of the neck's axis (or of the bulb's middle) with its fingers closed on it
(their gap from 4 mm less to 6 mm more than the neck's or the bulb's width), the flask moving with it (its offset from
the gripper changing by less than 5 mm a step). Otherwise the score is the best stage reached: 0.25 the flask lifted
out of its ring (its lowest point 3.9 cm or more above the table) and tilted 60 degrees or more; 0.5 the bar wholly
inside the flask; 0.75 the flask upright (tilted less than 10 degrees) in its cork ring (its bulb's middle within 1 cm
of the ring's axis and within 8 mm of the height at which it rests there) with the bar in its bulb, at rest (moved
less than 2 mm over 3 steps), not held. Full score: that, with both grippers' centres 6 cm or more from the flask's
glass, still for 1 s (moved less than 1.5 mm over 20 steps). `-E object_jitter=0` keeps every object exactly where the
task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=slide_a_stir_bar_into_a_tilted_flask.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put the stir bar into the round-bottom flask and leave the flask standing in its cork ring." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/stirbar-dream
```

The scripted demo, which reaches full score (the right gripper, level and pointing forward, takes the flask by its
neck, lifts it out of the cork ring and turns it 70 degrees over, the neck toward the left arm; the left gripper takes
the stir bar by one end, pinched slantwise near its fingertips so that most of the bar sticks out, lays it in the
mouth along the neck and lets go, and the bar slides down the neck into the bulb; the left gripper moves away; the
right gripper turns the flask upright slowly, sets it back in the cork ring, lets go and moves away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=slide_a_stir_bar_into_a_tilted_flask.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put the stir bar into the round-bottom flask and leave the flask standing in its cork ring." \
  --scorer task_score --max-steps 820 --store-frames --log-dir ~/yam-eval/stirbar-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/stirbar-dream/*.json; do inspect-robots video "$log"; done
```

`STIRBAR_DEBUG=1` prints, every 10 steps (`STIRBAR_DEBUG_EVERY=5` for every 5), where the flask is, how far it is
tilted, its lowest point, whether it is lifted, held or in its ring, where the bar is (outside, in the neck, in the
bulb), the fastest the bar has met the bulb's wall and where, how close the nearest gripper is, where each gripper is,
where the bar sits in the left gripper, each gripper's finger opening, and the stage reached.

`STIRBAR_DROP=1` runs the demo the wrong way: the flask stays upright in its ring; the left gripper takes the bar by one
end from above, turns level so that the bar hangs down, lowers its other end 8 mm into the mouth and lets go. The bar
falls down the neck and hits the bottom at 1.3 m/s, and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 2.5 minutes for the scripted demo (675 steps, 34 s of
simulated time; 150 s with 320 x 176 cameras, 156 s while rendering 1280 x 720 video), and about 6 minutes for a Dream-1
trial of 2,400 steps (375 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A 250 mL round-bottom flask with a 24/40 joint, a cork ring for it and a 25 x 8 mm PTFE stir bar. Whether the bar was
slid in or dropped is judged from the overhead and wrist videos: it must reach the bulb by sliding along the glass; a
bar seen falling freely 5 cm or more onto the glass (1 m/s) counts as dropped in. Whether the flask was set down
anywhere but in its ring, dropped or tipped over is seen on the same videos.

[york]: https://chemtl.york.ac.uk/techniques/basic-techniques/glassware/stirring-heating
