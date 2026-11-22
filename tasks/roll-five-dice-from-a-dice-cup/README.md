# Roll five dice from a dice cup into the game-box lid

Rolling dice out of a dice cup is how every turn of Yahtzee, Farkle or Liar's dice starts, and most people have done
it. The lid of a square board-game box lies open side up on the table as the dice tray, and beside it stands a
felt-lined leather dice cup with five dice in it. The instruction calls for an airborne cast into the lid, with
the dice bouncing and then settling flat. This is a repeatable tabletop version of a dice-cup roll.

What makes it hard is the hand skill. The cup is round and the dice are loose inside it: one hand takes the cup by its
side (the fingers must stay off the dice), lifts it over the lid and turns it mouth-down in one quick turn of the
wrist, so the dice fall out together and tumble as they land. Tipped slowly and low, with the lip just over the lid
floor (the way one pours from a cup), the dice only slide out and lie as they were: that is not a roll. Turned less
than about 120 deg, some dice stay in the cup (the felt lining holds them). Dropped in a heap, a die can come to rest
cocked, leaning on another die or on a wall, and does not count. One arm is enough; the difficulty is in how the
cup is held and turned.

The task is the task file [`roll_five_dice_from_a_dice_cup.py`](roll_five_dice_from_a_dice_cup.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters

Dice games are among the most familiar table games, and the dice cup is how they are played: Yahtzee lists "1 Dice cup
or shaker" in its equipment ([Yahtzee Manifesto, rules][rules]) and each turn starts by rolling all five dice with it.
For this station, the dice are cast from clear of the lid and must settle flat on one face. A die left propped
against a wall or balanced on an edge must be re-rolled
([backgammon federation rule FAQ on valid rolls][cocked]), and a tray "is a good way to keep your cubes in check"
([dice etiquette guide][tray]). The same hand skill, emptying loose small parts out of a held container with one
controlled turn so they spread rather than slide out in a heap, is used whenever a cup or scoop of small things is
emptied onto a tray.

The cup is a common felt-lined bicast-leather dice cup sold with five dice: 3 3/4 in (95 mm) tall and 3 1/4 in
(83 mm) across. The dice are standard 16 mm (5/8 in) acrylic dice with rounded corners, about 4.5 g each.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the lid, the cup and the dice are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The lid (a floor and four walls), the cup (a base
and a wall of 24 panels, felt lining drawn inside) and the dice (rounded cubes with their pips drawn on) are built with
`bodykit.py`, which adds such multi-shape bodies: importing it hooks `yam_arena.scene` when the simulator starts,
without changing yam_arena, and gives the task file each body's pose every control step. This copy also gives a body a
restitution (bounciness).

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Box lid, fixed (taped down), open side up: cardboard 2 mm thick, 25.4 x 25.4 cm outside, 3.0 cm tall, 25.0 x 25.0 cm
  inside, centred at x 25.0, y 7.5; its inner floor 2 mm above the table.
- Dice cup, loose, standing upright at x 27.0, y -16.5 (to the right of the lid, 7 cm from its wall): 83 mm across,
  95 mm tall, a 3 mm wall and a 4 mm base, felt-lined, 75 g.
- Five dice inside the cup, in one layer on its floor: one in the middle and four around it, 19.5 mm apart (centre to
  centre) along x and y, each standing square (faces along x and y), 1 face up. 16 mm, edges and corners rounded to
  2 mm, 4.5 g each.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the pose of the cup and of each die, and the points of each finger's collision hulls.
Both simulation and real trials use the same cast and final-position checks. First check the cast as a whole:

- When dice pass through the cup mouth, the lowest edge of the rim is at least 50 mm above the lid floor.
- At least one untouched die rebounds, with its whole body at least 5 mm clear of the lid floor within 0.15 s of
  first touching that floor.

The simulation samples these events at 240 Hz. If the cast passes, a die counts (0.2 each) when all of these hold:

- It passed out through the cup mouth.
- No gripper finger has touched it at any time in the trial (no finger point within 1.5 mm of it; once touched, the
  die never counts again in that trial).
- It rests flat on the lid floor: its lowest point within 1.5 mm of the floor, its faces within 5 deg of level (not
  cocked on another die or against a wall), its centre inside the lid's walls, at least 14 mm from the cup (not
  touching it), and still (moving under 1 mm in 1 s).

Score: 0.2 times the number of qualifying dice. Full score requires all five to qualify for 1 s. Doing nothing
scores 0.

For a real trial, record the cup and full lid in a close oblique view at 240 captured fps, high enough to see the
entire floor over the near rim, and record a second view from above to confirm that the grippers stay off the dice.
Before the trial, film a millimetre ruler standing on the floor at the near and far ends of the lid to establish
the floor plane and the 5 mm and 50 mm height references. Remove the ruler before starting without moving the
cameras. Step through the original high-speed frames. Check the lower rim height during the interval when dice
leave the cup. Find any die that touches the lid floor and then has a clear gap of at least 5 mm beneath it
within the next 36 frame intervals. Finally use a photo from above and the last second of video to count the dice
that are flat on the lid floor, inside the walls, clear of the cup, stationary and untouched. The frame with the
rebound can be from any one die in the cast.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=roll_five_dice_from_a_dice_cup.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Roll the five dice from the cup into the box lid so they bounce and settle flat. Keep the cup rim at least 5 cm above the lid floor as they leave." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dice-dream
```

The scripted demo, which reaches full score (open loop from the known layout; the left arm stays at its ready pose).
The right hand comes from behind the cup, pointing forward and 20 deg down, its fingers closing across the cup 60 mm
above the table, closes 4 mm inside the cup's sides, lifts it and gives it a gentle rattle (two 1 cm sideways moves).
It brings the cup, upright, over the right part of the lid with the grip 25 cm above the table. Then, in one turn of
0.6 s, it turns the wrist 150 deg about its pointing direction, so the cup's mouth turns down toward the open part of
the lid (the cup ends tilted 130 deg), while sweeping the cup 6 cm left: the dice fall 15 to 20 cm, tumble and spread
over the lid floor. In the revised scorer's debug run the rim was 187 to 188 mm above the floor at release, and
the largest clear rebound was 16.1 mm. After 0.6 s the gripper turns the cup back upright while moving it 10 cm
right, clear of the lid. The trial ends on success at step 332 (16.6 s), while the cup is carried back to its place.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=roll_five_dice_from_a_dice_cup.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Roll the five dice from the cup into the box lid so they bounce and settle flat. Keep the cup rim at least 5 cm above the lid floor as they leave." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/dice-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/dice-dream/*.json; do inspect-robots video "$log"; done
```

`DICE_DEBUG=1` prints, every 5 steps (`DICE_DEBUG_EVERY=1` for every step), the number of dice that count, the cup's
position, height and tilt, the right gripper's position and opening, and for each die its position, the height of its
lowest point over the lid floor, tilt from flat, the rim height at release (`h`, in mm), the rebound height
(`b`, in mm), and flags: X it has left the cup, T a finger touched it, OUT outside the lid, G it counts.

`DICE_MODE=slide` runs the demo the wrong way: the same grip and turn, but the cup is tipped slowly (over 1.6 s)
and low, without the sideways sweep. The debug run measured an 18 mm rim height at release and a largest rebound
of 3.4 mm. All five dice came to rest in the lid, and this low slide scored 0.

## What is modelled and what is not

- Rigid bodies. The dice are 16 mm cubes with edges and corners rounded to 2 mm (crossed boxes, a capsule on each edge
  and a ball at each corner); the pips are drawn on (they do not collide). The cup's wall is 24 flat panels (a
  24-sided wall, 83 mm across its flats); its felt lining is drawn on.
- Restitution (bounciness, estimates): dice and lid cardboard 0.30, the cup's felt 0.05; PhysX averages the two
  touching materials. Friction (static / sliding, estimates): dice 0.40 / 0.35, lid 0.40 / 0.35, cup 0.60 / 0.50.
- The rattle in the cup is gentle and only for show; the dice do not need shaking to score. Dice falling on each other
  and on the lid are simulated as they happen; a real lid's cardboard flexes a little, which is not modelled.
- The lid is fixed to the table (a real one would be taped down for the trial).

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell Server Edition shared with other simulation jobs, the scripted demo with
1280 x 720 cameras (frames stored, videos rendered) took about 2 minutes and ended on success at step 332 (16.6 s of
simulated time); the Dream-1 trial of 2,400 steps at 320 x 176 took about 7 minutes. Debug runs of the demo at
320 x 176 (400 steps) took about 5.5 minutes each with six running at once. Isaac Sim's first start on a new machine
adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic (the same run
repeated gave the same final dice positions to the millimetre).

## Building it for real

A felt-lined dice cup with five 16 mm dice (sold together, about 10 USD); the lid of a square board-game box about
25 x 25 cm inside and 3 cm deep (or any cardboard box lid of that size), taped to the table, open side up; tape marks
for the lid's corners and a circle for the cup's base. For each trial put the five dice back in the cup, one face up,
one in the middle and four around it, and stand the cup on its circle.

[rules]: https://www.yahtzeemanifesto.com/rules.php
[cocked]: https://usbgf.org/rule-faq-understanding-dice-on-checkers/
[tray]: https://www.vaultgames.com.au/blogs/news/the-tabletop-gamers-essential-guide-to-dice-etiquette
