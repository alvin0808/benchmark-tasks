# Replace a self-inking stamp's ink pad: lock the stamp down, swap the pad, unlock it

An office self-inking stamp (Trodat Printy 4913 size) stands on the desk with its dried-out ink pad in it. A new pad
lies on the desk beside it, ink side down, and an empty parts tray stands in front of that. The instruction asks for
the dry pad to be replaced with the new one, the dry pad put in the tray and the stamp left standing ready to use; it
does not say how.

The procedure is the stamp makers'. thestampmaker.com, [How to Re-Ink a Self-Inking Stamp][reink] (Trodat
self-inking date stamps): "Push stamp down slightly and push buttons on side to lock into position. Pull out ink
pad." For 2000 Plus stamps: "Push down slightly on the stamp - about 1/8" ... Lock the stamp in position using the
locks on both sides of the stamp." stempel.eu.com, [How to change your ink pad][change]: "Press down the stamp
slightly (approx. 3mm)", "Lock stamping mechanism", "Pull or push out old ink pad", "Insert new ink pad", "Release
'Lock-mode'" (by pressing the stamp down onto a piece of paper). Sizes: the Trodat 6/4913 replacement pad is 68 x 31 x
10 mm; the Printy 4913 is listed at "7.4 (W) x 3.8 (D) x 8.5 (H) cm" ([Viking][viking]).

Why the lock matters: at rest the stamp's spring holds the rubber text plate (face up, under the pad) up in the pad's
foam, so a pad pulled without the lock catches on it and the light stamp just slides along the desk. Pressed down
about 3 mm, the text plate drops clear; the two small lock buttons on the stamp's ends, pushed in, then catch the
body there. Pressing the body down again frees the buttons, which spring out, and the body returns up with the text
plate in the new pad.

The hand skill, stated plainly: holding the body pressed a few millimetres against its spring while the two small
buttons at its ends are squeezed in (the scripted demo uses one hand for each: one gripper presses the cap, the other,
reaching in from the front, squeezes the buttons; a person often does it the same way); then pinching a 2.5 mm thick
grip tab and drawing the pad straight out of a slot with 0.5 mm play while the other hand steadies the light stamp;
carrying the new pad by its tab and sliding it in level until its detent clicks; then pressing the stamp down once
more so that the locks let go. The difficulties are small parts and precise force and position control, not strength.

The task is the task file [`replace_a_self_inking_stamps_ink_pad.py`](replace_a_self_inking_stamps_ink_pad.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it. No
third-party assets are used: every object is built from boxes and spheres in the task file.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a spring. The
stamp's frame, its sprung body, the two lock buttons, the pad detent, the pads and the tray are bodies built from
several shapes, joined by sprung slide joints, so `bodykit.py` adds such bodies and joints: importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena.

## What is modelled

Positions in the world frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z
0.7475. The stamp stands turned so that its front (the pad slot's opening, with the pad's grip tab) faces the robot's
right (-y) and its length runs along x.

- Stamp frame (loose, 40 g): 38 mm deep x 74 mm long x 68 mm tall, on two foot rails, its middle at (0.240, 0.070).
  The pad slot runs front to back between its end walls, 43-53.5 mm above the desk; grooves in the end walls hold the
  pad's ends with 0.5 mm play; a back stop. The slot's first 5 mm flare 2 mm up, down and to each side (lead-in
  faces). Each end wall has a lock notch 52-59 mm up.
- Stamp body (cap, two end skirts and the text-plate mount, 45 g) on a slide joint along the frame's vertical: 0-6 mm
  down, on a 1500 N/m spring (6 N at rest, 9.75 N at 2.5 mm, 15 N at 6 mm). Cap 40 x 76 x 13 mm, its top 85 mm above
  the desk. Text plate 58 x 22 mm; at rest its top is 44.4 mm up, 1.4 mm up in the pad's 1.5 mm pocket.
- Two lock buttons (20 g each, 10 x 8 mm, standing out 2.5 mm), in the skirts at the stamp's two ends, 60 mm up, on
  slide joints with 2.5 mm inward travel and light outward springs (0.4 N out, 0.8 N in). Their inner tips stop on the
  frame's end walls unless the body is down 2.5 mm or more, when they line up with the notches; pushed in, the
  notch's top edge holds the body 2.5 mm down, and the spring pressing the tip against that edge holds the button in
  by friction. Pressed down past 3.5 mm, the tips come free and the buttons spring out.
- Ink pads (15 g each): 68 x 31 x 10 mm, a plastic case on top, foam below with a 60 x 24 x 1.5 mm pocket where the
  text plate sits, and a grip tab on the front edge (15 mm long, 18 mm tall, 2.5 mm thick). The dry pad (grey foam,
  white tab) is home in the slot; the new one (black foam, blue tab) lies ink side down on the desk at (0.270,
  -0.100), its tab toward -y. A spring detent ball (2 mm radius, 800 N/m, 0.8 N at rest) in the slot's ceiling
  clicks into a 3.2 mm groove across the pad's top when the pad is home.
- Parts tray (fixed): open, 90 x 120 mm, 12 mm rims, its middle at (0.140, -0.110).

What is not modelled: the stamp's full stroke (on a real stamp the text plate turns over and prints when the body is
pressed all the way; here the body travels only 6 mm and the plate stays face up); the foam is rigid, and the text
plate pressed into the soft foam is a rigid 1.5 mm boss in a 1.5 mm pocket; the ink itself.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the best stage reached (the stages
build on each other):

- 0.1: the stamp locked: its body 2.0 mm or more down and both lock buttons pushed in 1.5 mm or more.
- 0.3: the dry pad out of the slot (its rear edge more than 1 mm in front of the frame's front face).
- 0.45: the dry pad in the tray (all of it inside the rims, its lowest point within 15 mm of the floor), no finger
  within 10 mm of it.
- 0.6: the new pad fully home (its rear edge within 1.0 mm of the back stop, where the detent clicks into its groove),
  ink side down (case up, within 15 deg), with the dry pad in the tray.
- 0.8: the stamp unlocked with the new pad home: both buttons out (less than 0.5 mm in) and the body back up (less
  than 0.5 mm down), the text plate standing in the new pad's pocket.
- 1.0: all of that at once, the stamp standing upright on its foot rails (within 10 deg of vertical, on the desk),
  every finger 10 mm or more from the stamp and both pads, and everything still (moving less than 1 mm in 1 s).

Everything is measured from the bodies' poses and the sliding parts' travel, read every control step; finger
distances use the fingers' collision meshes. On a real stamp each stage is seen in the overhead and wrist videos and
a final photo; the last check is a test impression on paper (an inked print shows the new pad home, ink side down,
and the stamp unlocked).

`-E object_jitter=0` keeps every object exactly where the task file puts it. Dream-1 (needs a key in
`~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=replace_a_self_inking_stamps_ink_pad.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the stamp's dry ink pad with the new one, put the dry pad in the tray, and leave the stamp standing ready to use." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/stamp-dream
```

The scripted demo, which reaches full score:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=replace_a_self_inking_stamps_ink_pad.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the stamp's dry ink pad with the new one, put the dry pad in the tray, and leave the stamp standing ready to use." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stamp-demo
```

In it, the left gripper, straight down with its fingers closed, presses the cap down near its back edge until the body
stops (6 mm); the right gripper, tipped 25 deg down and reaching in from the stamp's front with its fingers either
side of the stamp's ends, squeezes both lock buttons in. The left hand lets the cap up and the right hand opens: the
body rises onto the buttons' tips and stays locked 2.5 mm down. The left hand then holds the stamp by its two ends.
The right gripper, tipped 35 deg, pinches the dry pad's tab, draws the pad straight out and lays it in the tray, takes
the new pad by its tab, carries it to the slot and slides it home past the detent (told 20 mm past the stop, since
the arm is soft sideways). The left hand lets go, closes and presses the cap
down past the lock: the buttons spring out and the body comes back up.

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/stamp-dream/*.json; do inspect-robots video "$log"; done
```

`STAMP_DEBUG=1` prints, every 5 steps (`STAMP_DEBUG_EVERY=n` to change), how far the body is down, how far each
button is in, the stamp's tilt and position, each pad's place relative to its seat (and whether it is in the slot,
out, home or in the tray, and the nearest finger), each gripper's position and opening, and the stage reached.

`STAMP_MODE=nolock` runs the demo the wrong way: nobody presses the stamp or its buttons; the left hand only steadies
the stamp by its ends while the right hand pinches the dry pad's tab and pulls. The pad moves 1 mm and catches on the
text plate, the tab slips out of the fingers and the stamp is dragged and tips a little; the pad never leaves the slot
and the run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 4 minutes for the scripted demo while rendering
1280 x 720 video (about 1,160 steps, 58 s of simulated time; it ends once success has held 1 s), and about 6 minutes
for a Dream-1 trial of 2,400 steps. Isaac Sim's first start on a new machine adds a few minutes while it compiles its
shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A Trodat Printy 4913 (or any self-inking stamp of its size with side lock buttons), a dry pad in it and a new 6/4913
pad, a small open tray, and tape marks for the stamp, the new pad and the tray. After the run, stamp once on paper:
an even impression shows the new pad home, ink side down, and the stamp unlocked. Each stage is judged from the
overhead and wrist videos and a final photo.

[reink]: https://www.thestampmaker.com/re-inking.aspx
[change]: https://stempel.eu.com/en/pages/change-inkpad
[viking]: https://www.viking-direct.co.uk/trodat-self-inking-57-x-20mm-stamp-up-to-6-lines-p-n4913
