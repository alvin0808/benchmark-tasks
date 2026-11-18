# Shave a care manikin's left cheek and jaw with an electric shaver, the other hand holding the cheek skin taut

Shaving residents is part of the daily personal care that nursing assistants and home health aides give. The table
top stands for a resident's bedside: a life-size adult male care-training head and neck stands on a weighted base,
seated upright, its face turned to the robot's left so that its left cheek faces the robot. The left side of the face
has a stubble-shaded beard area from the sideburn to the chin and from the cheek line down to the jaw line. The skin
over the left cheekbone is a soft insert that slides up toward the temple when a fingertip drags it, and springs back
when let go. A three-head rotary electric shaver lies on its back in its tray in front of the face. The instruction
asks for the beard on the left cheek and jaw to be shaved with the shaver without pressing hard or touching the lips,
nose, eye or ear, and for the shaver to be put back on its tray; it does not say how.

The procedure the task checks is the nurse-aide skills checklist's. Reuter, *Nursing Assistant*, 5.30 "Skills
Checklist: Shaving With an Electric Razor" ([LibreTexts, CC BY 4.0][reuter]): "Sit the resident upright and place a
clothing protector over their chest. Hold the shaver at a right angle to the resident's face, using your free hand to
pull their skin taught as you shave. This will minimize snagging and the risk of cutting the resident." and, at the
end, "Clean the razor per facility guidelines and charge or plug it in." The shaver maker says the same for its
rotary shavers. Philips's Series 3000 shaver user manual ([page 12][philips]), in its shaving steps (before the
wet-shaving ones): "Stretch your skin with your free hand so the blades can lift and cut hair. Gently press the shaver
onto you skin so the floating heads can follow the contours of your face." and "Note: Do not press too hard as too
much pressure can cause shaving heads to pinch the skin." ("taught" and "you skin" are as written in the sources.)

Why two arms are needed, as seen in the scene: the cheek skin is soft. Its insert over the cheekbone has to be held
dragged up toward the temple, or it springs back and the cheek below bunches under the shaver; nothing holds it but a
hand. One gripper's closed fingertips press the insert and drag it about 5 mm up and keep it there, while the other
gripper holds the shaver flat on the cheek and moves it over the beard with a light, steady pressure. The jaw and the
chin, over bone, can be shaved one-handed; the soft cheek, 63 % of the beard area, cannot.

The task is the task file [`shave_a_care_manikins_cheek_with_an_electric_shaver.py`](shave_a_care_manikins_cheek_with_an_electric_shaver.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.
The real build's custom parts (the sprung cheek insert and the force-sensing shaver) are in
[`fabrication/`](fabrication/README.md): STL files, a dimensioned drawing, the parts list and the calibration.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. No third-party models or assets are used: every object is built in the task
file, the head's surface included (it is computed in the file from an implicit model of the head written there, see
below), from boxes, cylinders and triangle meshes the file generates with NumPy.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them, put one on a spring or give it
a curved surface. `bodykit.py` adds bodies made of several shapes and joints with springs; for this task it also has a
triangle-mesh shape (shown with smooth normals and, if given, per-point colours, or a convex collider that PhysX cooks
from its points) and a joint whose axes are each locked, limited or free on a spring. Importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena.

What is modelled and what is not:

- The head is an adult male head and neck at life size, built from its measurements (not a scan): 22.4 cm from the
  back of the head to the nose tip, 15.2 cm wide, 23.6 cm from chin to crown, the jaw 11.6 cm wide at its angles, the
  neck 11.4 cm across, ears 6.1 cm long. Its surface is a smooth blend of ellipsoids and capsules (cranium, forehead,
  cheekbones, jaw cut along the jaw line, chin, lips, nose with open nostrils, brow ridge, eye sockets with eyeballs,
  neck) written in the task file; the shown skin is that surface on a 1.5-degree grid of rays from inside the head,
  its colour set per point and blended between points (skin, the stubble with a soft edge, lips, nostrils), the eyes
  coloured apart (white, iris, pupil). It collides through about 130 convex pieces that follow the surface (small ones
  over the left side of the face): over the beard area the collision surface is within 1.2 mm of the shown skin. The
  head is rigid vinyl, 2.0 kg, on a sprung neck: a spring hinge at the base lets it tip away when pushed (1 N pushed
  into the middle of the beard area moves it about 0.8 mm along the push). It stands seated upright on a base 7 cm in
  radius, its face toward the robot's left (+y), the midpoint of its ear canals 30 cm in front of the arms' midpoint
  and 19 cm above the table.
- The beard area is the stubble-shaded skin on the left side: from 1.8 cm in front of the ear canals (the sideburn)
  forward to 2 cm left of the chin's midline, from the cheek line (9 mm below the ear canals' level at the sideburn,
  sloping down toward the corner of the mouth) down to where the skin under the jaw turns to face the table at 30
  degrees; it keeps 2.4 cm from the lips, 2.2 cm from the nostril and 1.6 cm from the eye and the ear (the moustache,
  the skin round the mouth, which needs the shaver tilted onto one head, and the skin under the jaw are not part of
  it). Its 38.0 cm2 are sampled at 593 points about 3 mm apart (the first point of the skin in each 3 mm cube), with no
  gaps.
- The soft cheek: the 372 points (63 %) of the beard area behind a line 7.2 cm in front of the ear canals and above a
  line 6.2 cm below them. The skin insert over the cheekbone (30 x 22 mm, its lower edge 1.0 cm above the ear canals'
  level, 2.4 cm above the beard's top edge) slides up toward the temple from 0 to 8 mm on a 0.12 N/mm spring; it is
  taut when pulled 3 mm or more. The rest of the face is rigid; the skin does not bunch or stretch in the simulation,
  the scorer counts a soft-cheek point as shaved only while the insert is held pulled (see the scoring).
- The shaver is a three-head rotary electric shaver: 16.1 cm long, 5.8 cm wide, 4.5 cm thick, 170 g (a 145 g body
  and a 25 g head unit). Its three heads are 20 mm across, their centres 23 mm apart, each with a 1.2 mm domed face. The
  head unit pivots up to 20 degrees either way on a soft centring spring and presses in up to 6 mm on a 1 N/mm spring,
  so the force on the skin is read from how far it is pressed in. It lies on its back in a white tray (20 x 9 cm, 1 cm
  walls) in front of the face, its heads toward the face.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
50 ms step it reads the head's, the skin insert's, the shaver's and its head unit's poses and the grippers' joints.

- A point of the beard area is shaved when one of the three heads passes over it: it lies inside the head's 20 mm
  circle and within 1.5 mm of the head's face, the head unit is within 20 degrees of flat to the skin there, and the
  head unit's spring reads 0.3 to 3 N (force = 1 N/mm x how far it is pressed in, less its own weight). Moves between
  steps are swept in 3 mm steps, so a fast pass still counts. On the soft cheek a point counts only while the skin
  insert is pulled 3 mm or more toward the temple. Hair and skin are not simulated: this is the cutting rule, after
  the two sources (unstretched skin bunches and its hairs lie flat; a light, flat press lets the heads lift and cut
  them).
- 0 for the rest of the run if: the spring reads over 5 N (pressed hard enough to pinch the skin); a head comes within
  1 mm of the lips (and 2 mm round them), the left nostril (and the nose wing round it), the left eye (the eye opening
  and its lids) or the left ear; a gripper's finger comes within 1 mm of the left eye; the head is pushed 8 mm from
  where it stood (measured at the middle of the beard area), a push that moves a person's head; or the shaver falls 3
  cm or more with no gripper holding it (a gripper holds it when its centre is inside the shaver's body, 6 mm grace,
  and its fingers are closed to less than 6.4 cm apart).
- Otherwise, at the end of the run: 1.0 when at least 95 % of the beard area is shaved and the shaver rests on its
  tray (its lowest point within 4 mm of the tray's floor, its middle over the tray), let go (both gripper centres 5 cm
  or more from it) and still (within 1.5 mm) for 1 s; 0.75 when at least 95 % is shaved but the shaver is not back on
  its tray; 0.5 at least 70 %; 0.25 at least 30 %; 0 otherwise, also when nothing is done. `-E object_jitter=0` keeps
  every object where the task file puts it.

Real trial (scored with the shaver's own force log, the trial video and a photo; the parts, drawings, STL files and
calibration are in [`fabrication/`](fabrication/README.md)). The real shaver is rebuilt as a force-sensing shaver: its
head unit sits on a 1 N/mm spring over a 5 N force sensor, and a logger inside it writes the pressing force every 50 ms,
less the head unit's own weight along its axis, as the simulation computes it. The tracer is a washable UV-fluorescent
training gel (Glo Germ Gel), not a powder: a thin layer is rubbed on the beard area only, a 365 nm UV torch on a stand
lights it during the trial, and it goes dark where a head wipes it. A phone or webcam on a stand films the left side of
the face, with a millimetre scale beside the insert's dot; a tap on the head unit at the start puts one spike in the log
at a known video frame, so the two clocks line up. Nothing has to be seen under the shaver: for each 3 mm square, the
video gives the time between the last frame in which it still glows and the first in which it is dark (the shaver hides
it in between; the squares one pass wipes share that time). The square counts as shaved when, over that time, the log
reaches 0.3 N and never goes over 3 N, and, for a soft-cheek square, the dot stands 3 mm or more up on the scale
whenever the log reads 0.3 N or more. A UV photo after the trial with a transparent 3 mm grid card of the beard area
(printed from the drawing) gives the count of wiped squares and checks the video. Matching squares to the log by time
is stricter than the simulated rule, which reads each step's force for the squares under the heads at that step. Each
of these sets 0 for the rest of the run: the log reading 5 N (the sensor's full scale) or more; lipstick from the lips, or eyeliner from the nostril rims,
eyelids or ear, found on a head, or eyeliner from the eyelids on a gripper's finger cot; a pin taped to the back of the
neck at the beard middle's height (11.6 cm above the neck's hinge) moving 7 mm along the push over a millimetre card
(the beard middle's 8 mm); the shaver falling in the video. The end state (on its tray, let go, still for 1 s) is read
from the video. The gripper fingertips wear nitrile finger cots, so neither gel nor lipstick reaches the robot. The
gel's faint glow under the torch is the one change the robot's cameras see from the simulated scene's look.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=shave_a_care_manikins_cheek_with_an_electric_shaver.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Shave the beard on the manikin's left cheek and jaw with the electric shaver without pressing hard or touching its lips, nose, eye or ear, then put the shaver back on its tray." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shave-dream
```

The scripted demo, which reaches full score (success at step 1,790 of 2,400): the left gripper takes the shaver
straight down across its body and turns it to face the left side of the face; one-handed, it shaves the firm skin in
one continuous pass along the chin's left side, the jaw line and the jaw, and a short pass down beside the mouth; it
moves clear while the right gripper's closed fingertips press the skin insert over the cheekbone and drag it up toward
the temple (the insert ends up 4.7 to 5.1 mm up); with the skin held there the left gripper shaves the cheek in one
continuous zig-zag of five rows 6 degrees apart, from the top of the cheek down; the right gripper lets the skin go
and moves away, and the left lays the shaver back in its tray, lets go and moves away. Each pass follows the face: the
head unit flat on the skin (the shaver's body tilted up at most 12 degrees so that its tail stays clear of the table;
the head unit's pivot takes the rest), the heads aimed 12 to 13 mm past the skin (the head unit's spring, the arm and
the neck give, so the force on the skin comes out at 0.3 to 3.4 N, mostly 0.5 to 2 N). In that run 98.1 % of the
beard area is shaved (the soft cheek 100 %, the firm skin 95 %); the heads come no nearer than 7.0 mm to the ear, 7.2
mm to the lips, 8.7 mm to the eye and 21 mm to the nostril; the head is pushed 5.7 mm at most. It is open-loop from
the fixed layout and the head's known shape and reads nothing from the simulation while it runs; its corrections for
the arms' give (the head unit lands 9 to 21 mm lower and a few millimetres further forward than commanded, more with
the gripper turned for the cheek) were measured once in debug runs and are constants in the file:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=shave_a_care_manikins_cheek_with_an_electric_shaver.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Shave the beard on the manikin's left cheek and jaw with the electric shaver without pressing hard or touching its lips, nose, eye or ear, then put the shaver back on its tray." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shave-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/shave-dream/*.json; do inspect-robots video "$log"; done
```

`SHAVE_DEBUG=1` prints, every 10 steps (`SHAVE_DEBUG_EVERY=5` for every 5), the fraction shaved (soft cheek and firm
skin apart), the spring's force and the most so far, how far the head unit is pressed in and tilted, the skin
insert's pull, how far the head has been pushed, where the head unit is on the face, whether the shaver is held, its
lowest point, whether it is on its tray, the grippers' places and openings, the closest each zone has come to a head,
and how many points lie under a head and on its face. With it, `SHAVE_DUMP=<file.npy>` saves which points are shaved
every 50 steps.

`SHAVE_ONE=1` runs the demo the wrong way, with one arm: the right arm never pulls the skin, and the left does every
pass as before and puts the shaver back. With the cheek skin never held taut, none of the 372 soft-cheek points
counts; the jaw and chin (221 points) come out 95 % shaved, 35.4 % of the beard area: score 0.25 (not 0.5, which
needs 70 %). The soft cheek is 63 % of the area, so a one-handed shave cannot pass 37 %.

`SHAVE_HARD=1` runs the demo pressing far too hard on the jaw and chin (the heads aimed 40 mm deeper): the head
unit's spring reads 5.2 N at step 213, as the heads land on the chin, and the run scores 0 (the head is pushed past
8 mm a moment later too).

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell (shared with another job): about 3 minutes to start Isaac Sim and about 8
minutes for the scripted demo at 320 x 176 (1,790 steps, 90 s of simulated time); a Dream-1 trial of 2,400 steps
takes about 15 minutes (with two other simulations sharing the GPU). Importing the task file takes about 5 s (it
builds the head's surface and colliders with NumPy). Isaac Sim's first start on a new machine adds a few minutes while
it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

Parts (drawings, STL files, catalogue part numbers, assembly and calibration in
[`fabrication/`](fabrication/README.md)): a life-size adult male care-training head and neck in rigid vinyl on a
weighted base, clamped to the bench, seated upright with its face toward the robot's left and the midpoint of its ear
canals 30 cm in front of the arms' midpoint and 19 cm above the bench; its neck on a spring hinge at the base, about 20
N m/rad (check before each session: 5 N pushed level into the middle of the beard area, with a force gauge, moves it
4.0 +- 0.5 mm along the push, the 0.8 mm per newton of the simulation). The beard area on its left side painted
stubble-grey to the outline above. Over the left cheekbone the 30 x 22 mm silicone skin insert in its window tray, on a
HIWIN MGN5C block and MGNR5R rail with a Lee Spring LP 018E 03 S316 return spring (0.12 N/mm, 0 to 8 mm), a 2 mm dot
on it and a millimetre scale beside it. A three-head rotary electric shaver (about 16 x 6 x 4.5 cm, 170 g) rebuilt as
the force-sensing shaver: its head unit on a 1 N/mm spring over a Honeywell FMAMSDXX005WC2C3 force sensor, a Seeed
XIAO nRF52840 Sense logger inside in the motor's place (the motor is removed, so the cutters do not turn; the rebuilt
shaver is about 1.5 cm longer than the simulated one); its white tray. Glo Germ Gel, a 365 nm UV torch and a phone or
webcam, each on a stand; a transparent 3 mm grid card of the beard area; a red lipstick and a dark eyeliner pencil;
nitrile finger cots for the gripper fingertips; a pin and a millimetre card for the head's push.

Before every trial: put fresh finger cots on the gripper fingertips if the last ones are marked; wipe the face and the
shaver's heads with a damp cloth; rub a thin, even layer of gel on the beard area only (about 0.5 ml, with a gloved
finger); redraw the lipstick on the lips and the eyeliner on the nostril rims, eyelids and ear; check that the skin
insert springs back to 0 and the head stands upright; lay the shaver on its back in its tray, heads toward the head;
start the log, the video and the UV torch, and tap the head unit once.

After the trial: score from the log, the video and the UV photo as in "Real trial" above, and look for lipstick or
eyeliner on the heads and the finger cots; then wash the gel off the face and the heads with soap and water.

[reuter]: https://med.libretexts.org/Bookshelves/Nursing/Nursing_Assistant_(Reuter)/05%3A_Provide_for_Personal_Care_Needs_of_Clients/5.30%3A_Skills_Checklist-_Shaving_With_an_Electric_Razor
[philips]: https://www.manualslib.com/manual/3329817/Philips-S3000.html?page=12
