# Give a care manikin's ankle three passive range-of-motion repetitions, leg held on the bed, without forcing it

Passive range-of-motion exercise keeps the joints of a bed-bound patient from stiffening; for the ankle it prevents
the foot from setting in a pointed position (foot drop). An adult care-training lower leg (knee to foot), moulded at
true size, lies on its back on a thin foam bed pad, its calf cradled in a foam leg trough and its heel on the pad. Its
ankle is a hinge that, like a relaxed ankle, gives easily in mid-range and stiffens toward each end. The instruction
asks for the ankle to be worked through its range three times, the foot bent up toward the shin and then the toes
pointed down, gently and without forcing it, with the leg kept on the bed, and for the foot to be left back at rest;
it does not say how.

Why it takes both arms follows from the leg. The training leg is loose and light (a hollow moulded shell, 0.30 kg),
and pointing the toes to 30 deg takes 0.94 N m at the ankle, more than the 0.71 N m by which the lower leg's own
weight holds its knee end down. A hand that only moves the foot therefore moves the leg as well: pointing the toes
levers the leg up about the ankle (its knee end rises off the bed) and pulls it down the bed, and the ankle never
reaches the range. So one hand holds the leg just above the ankle, the way the nurse aide skills exam asks ("While
supporting the foot and ankle close to the bed ..."), and the other turns the foot.

The task is the task file
[`move_a_manikins_ankle_through_its_range.py`](move_a_manikins_ankle_through_its_range.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`leg_mesh.py`](leg_mesh.py) (the leg's two moulded surfaces
and their measurements) and [`bodykit.py`](bodykit.py) next to it. [`fabrication/`](fabrication/) has the STL files
and the drawing for building the leg.

## Why it matters

Nursing assistants ([O*NET 31-1131.00][onet-na]: "Exercise patients who are comatose, paralyzed, or have restricted
mobility"; about 1.4 million US jobs, [BLS OEWS][bls]) do this on every shift for residents who cannot move their own
limbs. It is one of the skills on the national nurse aide competency exam (NNAAP, administered by Credentia), Skill 15,
"Performs modified passive range of motion (PROM) for one knee and one ankle". The candidate handbook
([North Carolina Nurse Aide I handbook, Credentia, February 2022][nnaap], step 5) reads: "While supporting the foot and
ankle close to the bed, pushes/pulls foot toward head (dorsiflexion), and pushes/pulls foot down, toes point down
(plantar flexion) (AT LEAST 3 TIMES unless pain is verbalized). Moves joints gently, slowly, and smoothly through the
range of motion, discontinuing exercise if client verbalizes pain." Goniometry norms put an adult's passive ankle range
at about 10 to 20 deg of dorsiflexion (10 deg with the knee straight, 20 with it bent: [Physiopedia][pp], after
StatPearls) and about 40 to 50 deg of plantarflexion; the training leg is built with a range of 20 deg up and 45 deg
down. The passive moment a relaxed ankle gives rises steeply toward each end of its range ([Silder et al. 2007][silder]
fit it with exponentials), and that is where forcing a joint injures it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the leg, the pad and the trough are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them, put one on a hinge or give one
a curved surface. The leg (two smooth triangle meshes on a sprung hinge) is built with `bodykit.py`, which adds such
bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and
gives the task file each body's pose and the hinge's angle. This copy of `bodykit.py` lets a moving part collide as
its mesh's convex hull and gives a sprung hinge a drive (a USD angular drive with a force limit), without which its
spring does nothing. The task file sets the ankle's spring every control step to the tangent of its moment curve at
the present angle, so the moment follows the curve below (it stiffens toward each end).

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Bed pad, fixed: firm foam covered in light blue vinyl, 24 x 64 x 1.2 cm, x 9.5 to 33.5, y -21.5 to 42.5. On it a
  foam leg trough (darker blue): two walls 3 cm thick, 7 cm tall and 11.9 cm long, their inner faces 5.53 cm either
  side of the leg's midline (x 21.5), from y 13.8 to 25.7, cradling the calf (a leg lying on its back rolls outward
  unless something holds it).
- Training leg, loose: an adult woman's lower leg and foot at true size, moulded skin colour (soft vinyl, friction
  1.2), its surfaces built from ANSUR II women's means ([public data][ansur], 1,986 women): foot length 246 mm, foot
  breadth 93 mm, heel breadth 67 mm, ball-of-foot circumference 228 mm, bimalleolar breadth 67 mm, ankle
  circumference 216 mm, calf circumference 373 mm (widest 13 cm below the knee joint line), tibial height 438 mm. The
  lower leg is 0.30 kg (a hollow shell; its centre of mass 24 cm up from the ankle), the foot 0.15 kg. The ankle's
  axis runs through the malleoli 50 mm above the sole and 49 mm in front of the back of the heel. The leg lies on its
  back along +y, the knee end toward the robot's left (y 33.2), resting on its heel and its calf, so it tilts 7.2 deg
  (knee end up); the ankle axis is at x 21.5, y -8.0, z 6.4, along x; the foot points up (top of the toes z 25.7),
  the heel's lowest point at y -8.9 on the pad. A black tape strip runs along the outer side (toward the robot) of the
  shin and one along the outer side of the foot, at right angles with the ankle in neutral.
- The ankle: a hinge whose restoring moment rises in steps toward each end (N m/rad): toward the shin
  (dorsiflexion) 1.8 up to 10 deg, 5.4 up to 20 deg, 18 beyond; toes down (plantarflexion) 1.2 up to 20 deg, 3.0 up
  to 45 deg, 15 beyond; hard stops at 30 deg up and 55 deg down. That is 0.79 N m at 15 deg up, 1.26 N m at 20 deg
  up, 0.94 N m at 30 deg down and 1.73 N m at 45 deg down. Resting on its heel, the foot sits 3 deg toward
  plantarflexion.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the ankle's angle (from neutral, dorsiflexion positive), the lowest points of the
heel and of the leg's knee end, where the knee end is, and the points of each finger's collision hulls.

- 0 for the rest of the run once a rule is broken: the foot pushed past the end of its range (dorsiflexion over 20 deg
  or plantarflexion over 45 deg); the ankle jerked (its angle changing faster than 90 deg/s between two control steps,
  as when the foot is let go at the end of its range and snaps back); the heel lifted more than 1 cm above where it
  lay (the foot and ankle not kept close to the bed); the knee end lifted more than 1 cm (the leg levered up off the
  bed); the leg dragged more than 2 cm along the bed (its knee end).
- 0.25 for each repetition, at most 0.75: a repetition is the foot bent up to 15 deg or more and pointed down to 30 deg
  or more, one after the other (each counted only after the other has been reached since).
- 1.0 (success, held 1 s): three repetitions; at the end the foot within 5 deg of its resting angle, the heel and the
  knee end back down (within 5 mm of where they lay), every finger point 3 cm or more from the leg and foot, and the
  leg and foot still (under 0.5 deg and 1 mm over 1 s).

Doing nothing scores 0. In a real trial a phone on a stand beside the table films the leg's outer side with a
protractor card and a ruler standing behind the ankle: the angle between the two tape strips, read frame by frame,
gives the angles, the repetitions and the speed (a change of more than 3 deg between frames 1/30 s apart); the ruler
gives the heel's and the knee end's lift, and a tape mark on the pad beside the knee end shows any drag.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=move_a_manikins_ankle_through_its_range.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Work the ankle joint of the plastic training-manikin leg on the pad through its range three times, bending its foot up toward its shin and then pointing its toes down, gently and without forcing it and with the leg kept on the pad; finish with the foot back at rest and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ankle-dream
```

The scripted demo, which reaches full score (the left gripper takes the lower leg 10 cm above the ankle from above,
fingers closing across it at its widest, its wrist leaning 20 deg back toward the knee, and holds it where it lies;
the right gripper comes round from the robot's right, clear of the toes, and slides in from the robot's side at the
forefoot's height, pointing 35 deg down, one finger against the sole and one over the top of the foot, 17 cm along the
foot from the heel; it closes and turns the foot about the ankle's axis toward the shin and then down, three times,
2.2 s each way, then back to where the foot rests; it lets go and backs out, and then the left gripper lets go):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=move_a_manikins_ankle_through_its_range.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Work the ankle joint of the plastic training-manikin leg on the pad through its range three times, bending its foot up toward its shin and then pointing its toes down, gently and without forcing it and with the leg kept on the pad; finish with the foot back at rest and let go." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/ankle-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/ankle-dream/*.json; do inspect-robots video "$log"; done
```

`ANKLE_DEBUG=1` prints, every 5 steps (`ANKLE_DEBUG_EVERY=1` for every step), the ankle's angle and its moment, the
speed, the heel's and the knee end's lift, the drag, the repetitions counted, the peaks reached, how close the
nearest finger is, where each gripper is and any rule broken; and on a second line how far the knee end and the ankle
have moved and how the lower leg has turned.

`ANKLE_ONE=1` runs the demo the wrong way: the right gripper alone, the left arm staying where it is. With nothing holding the leg, turning the foot toward the shin tips the leg on its calf and lifts the heel off the bed; at about step 300 the heel is 1 cm up and the run scores 0 for the rest of the run (the foot reaches 15 deg only with the leg tipping, peak 17 deg). Pointing the toes then levers the leg up about the ankle, its knee end rising 1.3 cm, and drags it 2.9 cm down the bed, and the ankle stops at 29 deg, short of 30: the moment the ankle needs is more than the leg's weight holds down. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with several other simulation jobs, the scripted demo while rendering
1280 x 720 video took 192 s (795 steps, 39.8 s of simulated time, ending on success), running beside a Dream-1 trial
of 2,400 steps at 320 x 176 that took 412 s (0.17 s per step). Debug runs of the demo without video took 4 to 8
minutes on the busy GPU. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

The training leg is 3D-printed from [`fabrication/`](fabrication/README.md): the two shells (STL files, the same
surfaces as in the simulation), an 8 mm steel axle through the ankle in two bushings, two torsion springs (one each
way) and two urethane bump stops that give the stiffening moment curve, with a calibration by hanging weights; about
110 USD of parts and a day of printing. The bed pad is any firm 1 to 2 cm foam mat with a wipe-clean cover, the
trough two foam blocks (7 x 3 x 12 cm) glued to it as in the drawing. For each trial lay the leg in the trough with
its heel on a taped mark, film the outer side of the leg, and score from the video as above.

[onet-na]: https://www.onetonline.org/link/summary/31-1131.00
[bls]: https://www.bls.gov/oes/current/oes311131.htm
[nnaap]: https://ncnar.ncdhhs.gov/pdf/NC-NurseAide-I-Handbook.pdf
[pp]: https://www.physio-pedia.com/Goniometry:_Ankle_(Talocrural)_Dorsiflexion
[silder]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2020832/
[ansur]: https://www.openlab.psu.edu/ansur2/
