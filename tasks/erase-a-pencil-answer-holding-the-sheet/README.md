# Rub out a wrong pencil answer on a loose worksheet with an eraser, holding the sheet down with the other hand

Rubbing out a wrong pencil answer is something every student has done. A single printed times-table worksheet (A4, 80 g/m2)
lies loose on the desk with five answers pencilled in its answer boxes; one of them is wrong (7 x 8 = 54). A white
plastic block eraser (65 x 23 x 13 mm, the size of a Staedtler Mars plastic 526 50) stands on its end beside the
sheet. The instruction asks for the wrong answer to be rubbed out with the eraser without rubbing out the other
answers, and the eraser then put down; it does not say how.

Why two hands are needed is a matter of friction, and it is visible in the scene: an eraser grips paper far
better than a single 5 g sheet grips the desk (eraser on paper about 1.0, paper on a wooden desk about 0.45). Rubbed
on a sheet that nobody holds, the eraser does not slide over the graphite at all: it carries the sheet with it, back
and forth, and a real sheet slides and creases. One hand has to press the sheet to the desk beside the answer, firmly
enough that the sheet neither slides nor turns about that hand, while the other rubs the eraser's 13 x 23 mm end back
and forth over a 10 mm tall answer box whose neighbours are only 6 mm above and below it: pressing hard enough to erase
(1 N or more), with strokes long enough to break the eraser's grip on the paper (the arm is soft sideways, so it has
to be driven well past where the eraser goes), without drifting onto the other answers.

The task is the task file
[`erase_a_pencil_answer_holding_the_sheet.py`](erase_a_pencil_answer_holding_the_sheet.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters, and how it is done

Correcting pencil work with an eraser is part of every school day and of drafting and drawing work. Guides that teach
it say to hold the paper. An art supplier's guide ([wildlifeartstore.com, "How This Eraser Drawing Hack Transformed My
Art"][art]): "Always erase in one direction while holding the paper firmly behind your erasing hand. This prevents the
paper from buckling or creasing." The Navy drafting manual on erasing ([draftingmanuals.tpub.com][navy]): "Isolate the
unwanted area or protect surrounding areas with an erasing shield." Occupational therapists list "holding the paper
and erasing" among the everyday school tasks that need a dominant hand and a "helper hand" ([The OT Toolbox,
"Bilateral Coordination Activities"][ot]).

The eraser is modelled on the Staedtler Mars plastic 526 50 (the large one), 65 x 23 x 13 mm in European listings; a
Canadian dealer lists it as "2-1/2" x 7/8" x 7/16"" ([Monk Office, "Staedtler Mars Plastic White Eraser Latex-Free
Large"][mars], 63.5 x 22.1 x 12.7 mm). It is modelled at 65 x 23 x 13 mm with its card sleeve.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the worksheet and the eraser are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The worksheet (a thin sheet with its printing and
pencil marks as visual-only shapes) and the eraser (a block in a card sleeve) are built with `bodykit.py`, which adds
such bodies: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and gives the
task file each body's pose. This copy of `bodykit.py` also lets a body choose PhysX's friction combine mode (the
eraser's friction is used against everything it touches, the sheet's lower friction against the desk), turn on PhysX
contact reporting (the task file reads the force between the eraser and the sheet every control step), and meet
everything through PhysX's compliant (implicit spring-damper) contact: the eraser's end presses on the paper on a
20 N/mm spring with 40 N s/m of damping, standing in for the give of the soft vinyl.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Worksheet, loose: one sheet of A4 80 g/m2 paper (21.0 x 29.7 cm, 5.0 g), lying with its long side along y
  (x 10.0 to 31.0, y -12.85 to 16.85); friction 0.45 static, 0.40 sliding on the desk. Printed in black: a title bar,
  a name line and five sums, "7x5=", "7x6=", "7x8=", "7x9=", "7x7=", in rows 16 mm apart (rows at x 23.7, 22.1, 20.5,
  18.9, 17.3), each followed by a printed 30 x 10 mm answer box (blue outline; box middles at y -10.0). Pencilled in
  the boxes (graphite, two handwritten digits 7 mm tall and 4.5 mm wide, 2 mm apart): 35, 42, 54, 63, 49. The wrong
  answer, 54, is in the middle row (x 20.5); the answers above and below it are 6 mm from its box.
- Eraser, loose: a white vinyl block 65 x 23 x 13 mm, 25 g, in a blue card sleeve over its middle 36.5 mm; friction
  1.0 static, 0.9 sliding on paper. It stands on its 23 x 13 mm end on the desk to the right of the sheet at x 17.0,
  y -17.5, its 13 mm side along x (its 23 mm width along y).
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the sheet's and the eraser's poses, the contact force between the eraser and the
sheet (PhysX pair-wise contact forces) and the points of each finger's collision hulls.

- Graphite: each pencilled answer is a set of 1 mm grid cells along its strokes (37 cells for the wrong answer, 147
  for the other four). A cell is rubbed while the eraser's end face covers it, touching the paper there (within 0.3 mm
  of the sheet's top), and the eraser presses on the sheet with 1 N or more, by the distance the face slides over the
  sheet in that step. A cell of the wrong answer is erased after 60 mm of rubbing (for example two
  back-and-forth rubs of 15 mm each way); a cell of another answer counts as damaged (visibly lightened) after 10 mm.
- 0 for the rest of the run once more than 5 % of the other answers' cells are damaged (rubbing out someone's
  correct answers is real damage to the work). Where the sheet ends up is not scored: a sheet that travels along with
  the eraser costs credit by itself, since the eraser then does not slide over the graphite and nothing comes off.
- Otherwise 0.6 x the fraction of the wrong answer's cells erased.
- 0.75: at least 90 % of the wrong answer's cells erased.
- 1.0 (success, held 1 s): at least 90 % erased, and the eraser put down: its lowest point within 3 mm of the desk
  (or of the sheet, if it is put down on the sheet), every finger point of both grippers 2 cm or more from it, and
  still (under 1 mm in 1 s).

Doing nothing scores 0. The erasing rule is the simulation's stand-in for graphite coming off: how much rubbing a
pencil mark needs depends on the pencil, the paper and the eraser, and 60 mm is an estimate. A real trial is scored on
the real result instead: a photo of the sheet through a transparent 1 mm grid laid over each answer box counts the
cells of the wrong answer still showing graphite and the cells of the other answers lightened or smudged (compared
with a photo taken before the trial); the eraser put down and let go is checked on the video.

The wrong answer's pencil strokes are shown in the cameras by 17 dark pieces (visual only; nothing collides with
them) that follow the sheet, each taken away once all of its cells are erased. The other answers are drawn on the
sheet and stay drawn (the scorer counts any damage to them).

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=erase_a_pencil_answer_holding_the_sheet.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Rub out the wrong answer on the worksheet (7 x 8 = 54) with the eraser without rubbing out any of the other answers, then put the eraser down." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/erase-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The left gripper, open (fingertips
8 cm apart along x), comes down on the sheet in the wrong answer's row, 21 cm to the answer's left (x 20.5, y 11.0),
and presses it to the desk with both fingertips, commanded 10 mm below their touch (7.0 to 8.0 N on the sheet,
7.5 N typical). The right gripper takes the eraser by its top across its 23 mm width (closing across the narrow 13 mm
instead lets the rubbing pivot the eraser between the fingers), carries it over the wrong answer, presses its lower end
on the paper (commanded 4 mm below the touch: 1.4 to 3.2 N, 2.3 N typical while it presses) and rubs it back and forth
along the row twelve times, each stroke commanded 40 mm each way (the middle 4 mm to the left of the answer's), 1 s per
stroke each way: the arm, soft sideways, lags far behind, and the eraser slides only once the arm has built up the force
that breaks its grip on the paper. The wrong answer is fully rubbed out after about eight strokes; the other answers
are untouched and the sheet moves 0.6 mm in all. The right gripper lifts the eraser off, stands it back on its end
where it was, lets go and rises; the left hand rises. Success after 902 steps (45.1 s).

A person would hold the sheet a few centimetres from the answer. The arms cannot: two straight-down grippers side by
side collide when they are closer than about 18 cm, and the eraser hand sweeps 8 cm along the row. Holding beyond the
answer instead (x 28, y -8, 7.5 cm from it, the left arm reaching across) was tried: the left arm met the right arm as
soon as the eraser was lifted (up to 37 N between the arms) and the sheet was turned 40 deg. So the demo holds the
sheet in the answer's own row, where the rubbing pulls straight toward or away from the hold and does not turn the
sheet about it; at 18 cm the arms touched during the strokes, at 21 cm they do not.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=erase_a_pencil_answer_holding_the_sheet.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Rub out the wrong answer on the worksheet (7 x 8 = 54) with the eraser without rubbing out any of the other answers, then put the eraser down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/erase-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/erase-dream/*.json; do inspect-robots video "$log"; done
```

`ERASE_DEBUG=1` prints, every 5 steps (`ERASE_DEBUG_EVERY=1` for every step), the eraser's force on the sheet, the
left hand's force on the sheet, any contact between the arms, how high the eraser's end face is above the sheet,
where it is in the sheet's frame, how far it slid that step, the fraction of the wrong answer erased and of the other
answers damaged, how the sheet has moved and turned, the largest drag so far, how high the eraser rests, how close the
nearest finger is, where each gripper is and any rule broken; every 100 steps it lists the wrong answer's cells not
yet erased.

`ERASE_ONE=1` runs the demo the wrong way: the same path for the eraser hand, the left arm staying at its ready pose.
With nobody holding the sheet, the eraser carries it along: the sheet travels back and forth with it, up to 27 mm
from where it lay, and turns 2 deg, while the eraser slides over it by less than a tenth of a millimetre per step;
nothing is erased in the whole run (0 of 37 cells). Score 0.

How hard the holding hand must press depends on how the eraser is rubbed. In development runs with shorter strokes
(the eraser sticking and slipping more), a hold of about 3 N with closed fingertips let the sheet turn 1.8 deg about
the fingertips and slide 5 mm; with the demo's strokes, `ERASE_HOLD_PRESS=0.005` (the left hand pressing 5 mm below the touch, about
3.5 N) still holds the sheet (it moves 1.0 mm) and scores 1.0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs (three runs of this task at once), the
scripted demo while rendering 1280 x 720 video took 185 s of stepping (902 steps, ending on success), and a Dream-1
trial of 2,400 steps at 320 x 176 took 475 s (0.20 s per step); each run adds about 1 to 2 minutes of start-up. Debug
runs of the demo without video took 5 to 12 minutes on the busy GPU. Isaac Sim's first start on a new machine adds a
few minutes while it compiles its shaders.

Dream-1 run: score 0 (the right gripper went to the standing eraser again and again and hovered beside it, but never took it; the left arm stayed back; the sheet was not touched and nothing was rubbed out).

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A Staedtler Mars plastic 526 50 eraser (or any 65 x 23 x 13 mm white vinyl block eraser) in its sleeve; printed copies
of the worksheet (A4, 80 g/m2 copier paper; the layout is in the task file) with the five answers pencilled in with an
HB pencil at ordinary writing pressure, 54 in the middle row; a transparent 1 mm grid for scoring. For each trial lay a
fresh sheet in place against two strips of tape on the desk (at its left edge and its near edge), stand the eraser on
its end at its place, photograph the answers, and start.

[art]: https://www.wildlifeartstore.com/eraser-drawing-hack/
[navy]: https://draftingmanuals.tpub.com/14262/css/Electric-Erasers-20.htm
[ot]: https://www.theottoolbox.com/bilateral-coordination-activities/
[mars]: https://commercial.monk.ca/office-products/writing-correction/erasers/std52650-eraser-mars-plastic-large
