# Pour acetone with the bottle's label on top

The table top stands for a lab bench. An amber 250 ml reagent bottle of acetone stands on the right, its cap off and
its white hazard label turned toward an empty 100 ml glass beaker on its left. The instruction asks for acetone in the
beaker up to its 50 ml mark; it does not say how.

The rule the task checks is the bench habit chemistry teaching labs teach for pouring from a reagent bottle: hold the
bottle on the side of its label, so that the label is on top while it pours, and the drips that run down from the neck
run down the bare side rather than over the label, where a solvent dissolves the ink. The University of York's
Chemistry Teaching Labs, [Using a measuring cylinder](https://chemtl.york.ac.uk/techniques/basic-techniques/weigh-measure/using-a-measuring-cylinder):
"The solvent container should be held on the side of the label." A label that cannot be read is a hazard of its own,
and OSHA's laboratory standard, [29 CFR 1910.1450](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1450)
(h)(1)(i), says: "Employers shall ensure that labels on incoming containers of hazardous chemicals are not removed or
defaced." Tipping the bottle straight toward the beaker, as it stands, pours over its label and breaks the rule; turning
the bottle so that its label faces away from the beaker first (or pouring from the beaker's other side) keeps it. The
task is the task file [`pour_acetone_with_the_label_up.py`](pour_acetone_with_the_label_up.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on
Isaac Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev
launchable, installs everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The bottle (body,
shoulder, neck and lip, with its label) and the beaker (a floor and walls, with its marks) are bodies built from several
shapes, and the acetone in the beaker and the stream from the bottle are bodies the task file moves, so `bodykit.py`
adds such bodies: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives
`score()` each body's pose.

What is modelled and what is not: there is no liquid in the simulation. The bottle holds 150 ml. Acetone leaves it once
it is tipped past the angle at which the liquid reaches its lip (50 degrees from upright when full, up to 110 degrees as
it empties), at up to 30 ml a second (in proportion to how far past that angle it is tipped); it goes into the beaker
when the bottle's mouth is over the beaker's opening, otherwise onto the bench. The acetone in the beaker is shown by a
surface that rises with the volume, and a stream runs from the mouth while acetone leaves the bottle. Drips are not
simulated: the rule is scored from which way the label faces while acetone runs.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. The
label faces down when the outward normal at its centre points below the horizontal (the label is on the lower half of
the tipped bottle, the side the drips run down). Acetone running from the bottle while its label faces down sets the
score to 0 for the rest of the run; so does more than 2 ml going onto the bench (poured outside the beaker, or the
beaker knocked over with acetone in it). Otherwise the score is the best stage reached: 0.25 the bottle lifted 2 cm off
the bench; 0.5 acetone in the beaker (5 ml or more); 0.75 the beaker at the mark (50 ml or more). Full score: the
beaker holding 50 to 80 ml, upright; the bottle standing upright on the bench; both grippers 4 cm or more from the
bottle's surface; still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every
trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pour_acetone_with_the_label_up.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pour the acetone into the beaker up to the 50 ml mark." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pour-dream
```

The scripted demo, which reaches full score (the right gripper takes the bottle by its neck from above, lifts it off
the bench a few millimetres, turns it half a turn so that the label faces away from the beaker, and sets it down; the
left gripper takes the bottle by its body from the front, lifts it, holds its mouth over the beaker and tips it toward
the beaker with the label on top, pours about 60 ml, stands it back up and sets it down where it was):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pour_acetone_with_the_label_up.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pour the acetone into the beaker up to the 50 ml mark." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/pour-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/pour-dream/*.json; do inspect-robots video "$log"; done
```

`POUR_DEBUG=1` prints, every 10 steps, the bottle's pose and tilt, which way its label faces, where its mouth is, the
flow, how much acetone is in the bottle, in the beaker and on the bench, how far the grippers are from the bottle, and
the stage reached.

`POUR_LABEL_DOWN=1` runs the demo the wrong way: the bottle is not turned, and the left gripper tips it toward the beaker
as it stands, over its label; the run scores 0 as soon as acetone runs.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine under heavy load: about 4.5 minutes for the scripted demo
while rendering 1280 x 720 video (637 steps, 32 s of simulated time; 272 s), and about 10 minutes for a Dream-1 trial
of 2,400 steps (597 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

An amber 250 ml reagent bottle with a paper label, a 100 ml beaker with a 50 ml mark, and a solvent that is safe to
pour on a bench (water, for a robot trial). Which way the label faced while the bottle poured is judged from the
overhead and wrist videos.
