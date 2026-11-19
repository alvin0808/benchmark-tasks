# Open a children's bike's V-brake to free the front wheel

A 14-inch children's bike stands upright on the table, facing the robot, its rear wheel in a slot-type floor stand. Its
front wheel is held by a V-brake (a linear-pull rim brake): two arms on pivots on the fork legs, each sprung outward,
joined across the top of the tyre by the brake cable. On the bike's left arm a link holds the end of the cable's curved
metal tube, the "noodle"; the bare cable runs from the noodle's end across to its anchor bolt on the right arm. The
springs keep the cable tight, so the noodle's end is pressed into the link's cup and the pads sit 3 mm off the rim, too
close for the 44 mm tyre to pass. The instruction asks for the front brake to be opened so that the wheel can be taken
out; it does not say how.

Opening a V-brake is the first step of fixing a flat on any bike with rim brakes. The procedure the task checks is in
Park Tool's [wheel removal guide][parktool]: "Squeeze linear-pull calipers together and disconnect cable 'noodle' from
linkage", and in a bike maker's [help page][maker]: "If you squeeze the brake arms together, you can pull the noodle out
of the black or silver tab that it is hooked into." (and, to close it again, "squeeze the brake arms toward each other
again and slip the noodle into its tab").

Why it takes two hands, from the brake itself: while the cable is tight the noodle's end sits in its cup and cannot be
lifted out. Pulling the noodle on its own only swings the whole brake (one arm in, the other out, the cable still
tight). Only squeezing both arms together, until the pads touch the rim, slackens the cable; then the noodle can be
drawn back off its seat and lifted out of the link. Let go of the arms first and they spring back onto the noodle. So
one gripper must hold both arms squeezed (about 9 N at each arm's top) while the other takes the noodle out. Once the
noodle is out and the arms are let go, they spring wide open (pads 54 mm apart) and the wheel can drop out.

The task is the task file [`open_a_v_brake_to_free_the_wheel.py`](open_a_v_brake_to_free_the_wheel.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it. No
third-party assets are used: every part is built from boxes, cylinders and capsules in the task file.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a spring. The bike,
the two brake arms and the noodle are bodies built from several shapes, and the arms and the noodle hang on joints, so
`bodykit.py` adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without
changing yam_arena, and gives `score()` each body's pose. This copy of `bodykit.py` adds a joint with chosen free
directions and limits (`kind="d6"`), used for the cable.

## The scene

Positions in cm from the midpoint between the two arm bases: x forward, y to the robot's left, heights above the table.

- Bike (fixed in place): wheelbase 53 cm; front axle at x 21.7, y 0, 17.2 up; rear axle at x 74.7, in the stand at the
  table's far edge; 14 x 1.75 tyres (34.3 cm across, 4.4 cm wide) on ISO 254 rims (25 mm wide); fork leaning back 20
  degrees; handlebar 50 cm wide, grips 59 cm up at x 37; saddle top 53 cm up at x 60.
- Front V-brake: two black aluminium arms (60 g each), pivots on the fork 8 cm apart and 8.8 cm below the cable along
  the fork; arm tops 6 mm below the cable; pads 3.7 cm up the arms, 3 mm off the rim each side. Each arm has a spring
  (0.8 N m/rad, preloaded: 0.63 N m, 7.1 N at the cable, when closed) and two stops: 4.6 degrees inward (pad on the rim)
  and 18 degrees outward (pads 54 mm apart).
- Cable: crosses 1.1 cm above the tyre, its middle at x 26, y 0, 34.8 up; anchor bolt on the right arm (y +3.6).
- Noodle (20 g): a 90-degree aluminium tube 6 mm across, with an 8 mm ferrule at its end seated 4 mm deep in the cup of
  the link on the left arm (end at y -1.0); it bends up at y -2.8 to 41.9 up, with a rubber boot and a piece of
  housing; the rest of the housing runs to the left lever.

## What is modelled and what is not

The bike and its wheels are one fixed body; the tyres and frame collide, the rim, pads, fork legs and pivots do not (the
arms turn on them, and the pad touching the rim is the arm's inward stop). Each arm is a rigid body on a revolute joint
with a spring and the two stops; the spring stiffness is an estimate (a firm two-finger squeeze, about 9 N at each arm's
top). The link's cup is real collision geometry (8 mm ferrule, 0.5 mm clearance, 4 mm deep, 3 mm walls).

The bare inner cable, which can only pull, is a joint between the noodle's end and the right arm's anchor bolt: the
noodle may come toward the anchor (the cable goes slack, up to 25 mm), never farther than its seated distance. The
housing is not modelled: gravity does not act on the noodle (the housing holds it up) and it stays where it is put,
damped; it can move up and down and turn about the cable's cross axis. When the arms spring open with the noodle out,
the cable drags it toward the right arm, as the real cable does.

The arms' angles, the pads' spacing and the noodle's place in the link are read from the bodies' poses.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run, read
every control step:

- 0 for the rest of the run if the noodle is forced out of its cup: risen 2.5 mm while its end is still 3 mm or more
  inside it (the link bends).
- Otherwise the best stage reached: 0.25 the noodle's end 4.5 mm or more back from its seat (clear of the cup);
  0.5 the noodle lifted out of the link (its end 12 mm or more above the cup's middle).
- At the moment: 0.75 the noodle out of the link and the pads 46 mm or more apart (the 44 mm tyre passes between them).
- 1.0 (success, held 1 s): that, with both grippers' centres 5 cm or more from the noodle and from the arms' tops, and
  the noodle and arms still (under 2 mm and 0.5 degrees over 1 s).

Doing nothing scores 0. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial
starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=open_a_v_brake_to_free_the_wheel.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Open the bike's front brake so that the front wheel can be taken out." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/vbrake-dream
```

The scripted demo, which reaches full score: the left gripper, pointing forward and tipped down 30 degrees, comes over
the front tyre with its fingers either side of the arms' tops and squeezes them until the pads touch the rim; the right
gripper, coming from the bike's left side, takes the noodle's upright leg, draws it back 7 mm, lifts it 3 cm out of the
link, lets go and moves away; the left gripper opens and backs away, and the arms spring wide open:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=open_a_v_brake_to_free_the_wheel.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Open the bike's front brake so that the front wheel can be taken out." \
  --scorer task_score --max-steps 480 --store-frames --log-dir ~/yam-eval/vbrake-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/vbrake-dream/*.json; do inspect-robots video "$log"; done
```

`VBRAKE_DEBUG=1` prints, every 10 steps (`VBRAKE_DEBUG_EVERY=2` for every 2), both arms' angles (positive is open), the
pads' spacing, how far the noodle's end is back from its seat and how far up, whether it is out and the brake open, the
noodle's position, how close the nearest gripper is, whether everything is still, the stage reached, and each gripper's
position.

`VBRAKE_ONE=1` runs the demo the wrong way, with one arm: nobody squeezes the brake; the right gripper takes the noodle,
pulls it back 6 mm and lifts. The brake swings as one (left arm 4.8 degrees out, right 4.8 degrees in until its pad meets the rim, the cable still
tight), the noodle's end stays in its cup, and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: the scripted demo succeeds after about 330 steps (16.5 s of
simulated time; about 70 s while rendering 1280 x 720 video), and about 5.5 minutes for a Dream-1 trial of
2,400 steps (316 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

Any 14- or 16-inch children's bike with a front V-brake (they are sold widely for about 100 to 250 US dollars), in a
slot-type floor stand, with the bike strapped so that it cannot roll or tip; the robot works on the real brake. Before
each trial, seat the noodle in its link and check the pads sit about 3 mm off the rim. After the trial: the noodle is
out of its holder (video), the pads' spacing is measured with a ruler (46 mm or more), or the wheel is let out of the
fork to check that the tyre passes between the pads; a bent link or noodle scores 0.

[parktool]: https://www.parktool.com/en-us/blog/repair-help/wheel-removal-and-installation
[maker]: https://sixthreezero.zendesk.com/hc/en-us/articles/360034796473-Front-Wheel-Can-t-Fit-Through-the-Brakes
