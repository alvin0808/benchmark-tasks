# Give a toddler manikin a 5 mL dose with an oral syringe: tip between the gums and cheek, given in small squirts

Giving a small child a dose of liquid medicine with an oral syringe is something most parents and carers have done.
A toddler care manikin's head (2 to 3 years old) sits upright on a stand facing the robot, its mouth open. A 5 mL oral
syringe, filled to the 5 mL line, lies on two rests in a tray beside it. The instruction asks for the 5 mL dose to be
given with the syringe and the syringe put back on its rests; it does not say how.

What the task tests is where the tip goes and how the plunger is pushed, both of which a person does without
thinking. The tip has to go in at the corner of the open mouth and sit in the cheek pocket, the 4 to 7 mm space
between the side teeth and the inside of the cheek, pressing the soft cheek out a little, with the barrel slanted
toward the cheek rather than pointing down the middle of the mouth toward the throat; and the dose has to go in as
small squirts with pauses (the child swallows in between), not in one push. Inside the mouth the primary teeth and
gum ridges, the tongue, the throat and the two soft inner cheeks are all separate colliders, so the pocket is a real
place the tip has to reach, past the corner of the lips.

Why two arms, visible in the scene: a parallel gripper that holds the barrel cannot also push the plunger (a person
uses the thumb of the holding hand). One hand holds the barrel and keeps the tip in the cheek pocket while the other
pushes the plunger in 1 mL steps; the plunger's friction (0.8 N) is taken by the holding hand, not by the child's
cheek. With no hand free, the head rests on its stand (as a child leans on a pillow or a carer's arm).

The task is the task file
[`give_a_child_liquid_medicine_with_an_oral_syringe.py`](give_a_child_liquid_medicine_with_an_oral_syringe.py) for
the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py),
[`child_head.py`](child_head.py) (the head's surface) and [`manikin_face.py`](manikin_face.py) (its nose and ears)
next to it.

## Why it matters

Liquid medicine is how small children take most of their medicines, and the oral syringe is the measuring device
pharmacists hand out with it. Great Ormond Street Hospital's leaflet "How to give your child liquid medicines"
([GOSH][gosh]) says: "Put the tip of the oral syringe inside your child's mouth between the gums and the inside surface
of their cheek."; "Gently push the plunger to squirt small amounts of medicine into the side of your child's mouth.";
"Allow your child to swallow before you carry on pushing the plunger."; and "Do not squirt the entire dose into your
child's mouth in one go" because "they may choke". The American Society of Health-System Pharmacists' patient page
([SafeMedication][ashp]) adds: "Never squirt the liquid from the dropper or oral syringe directly onto the back of a
child's throat." and "Be sure to take breaks to allow your child to swallow." Anyone watching a clip knows at once
what the robot is doing.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the head, the syringe and the tray are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or put one on a slide. The head
(one triangle mesh, meshed from a signed-distance model of a toddler's head in `child_head.py`, plus the teeth, gum
ridges and tongue as boxes), the sprung inner cheeks and the syringe are built with `bodykit.py`, which adds such
bodies and joints: importing it hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, and
gives the task file each body's pose. This copy of `bodykit.py` lets a part of an articulation slide (a PhysX
prismatic joint with Coulomb friction): the syringe is a loose articulation whose plunger slides in the barrel with
0.8 N of friction. The head's mesh is built once (VTK, about 20 s) and cached in `child_head_mesh.npz`.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Toddler care manikin head, fixed, upright on a round stand 10 cm across and 3 cm high, face toward the robot (-x):
  head 16.8 cm long and 13.4 cm wide, face 8.2 cm from the nasal root to the chin (2 to 3 years). Its nasal root is
  at x 31.0, y 0, z 20.0. The mouth is open 4.6 cm wide and 2.6 cm high, its middle 15 cm above the table.
- Inside the mouth: 20 primary teeth (incisors, canines, first and second molars) on upper and lower gum ridges, 14 mm
  between the upper and lower front teeth; the outer (cheek-side) faces of the side teeth 14 to 19.5 mm from the
  midline; the tongue; the throat behind it.
- The inner cheeks: on each side a soft plate 23.5 x 22 mm (1.0 to 3.4 cm behind the lips' corner region, x 3 to
  26.5 mm behind the nasal root), its face 21.5 mm from the midline, which gives outward on a spring (0.06 N/mm, 0.08
  N preload, 4 mm travel): the cheek pocket is the 2 to 7.5 mm space between it and the side teeth.
- 5 mL oral syringe (like BD 305218), loose, filled to the 5 mL line: barrel 14 mm across (12.0 mm bore), 58 mm long,
  with a 3 mm cone and a 4.5 mm tip 10 mm long; finger flange 30 x 12 x 2.5 mm; the plunger stands 52 mm out of the
  barrel (44.2 mm of travel is 5 mL, 0.113 mL per mm), its thumb disc 14 mm across; the plunger slides with 0.8 N of
  friction; 17 g barrel with the dose, 4 g plunger. It lies along x, tip forward at x 25.5, y 12.0, on two rests
  (2.2 and 2.3 cm high, one under the barrel's front end and one under the flange, each with a notch) on a blue tray
  11 x 4.4 cm at x 20.0, y 12.0 (fixed).
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the syringe's pose and its plunger's travel, and how far each inner cheek has been
pushed out.

- The dose is the plunger's travel (0.113 mL per mm; travel counts once, beyond the farthest it has been). Each bit of
  travel is booked to where the tip is at that moment: "in the cheek pocket" when the tip is beside the side teeth (2
  to 24 mm behind the nasal root, at least 0.5 mm farther out than the side teeth's outer faces, within 12 mm of the
  mouth's mid-height) and presses that inner cheek (pushed out at least 0.5 mm, 0.11 N); "toward the throat" when
  the tip is in the mouth inside the row of teeth (more than 2 mm in from their outer faces) or more than 27 mm back
  (behind the last molar); anything else is spilled (on the lips or outside the mouth).
- A squirt is the travel between two pauses; a pause is 1 s in which the plunger moves less than 0.5 mm (0.06 mL).
- A fault, which caps the score at 0.5 for the rest of the run: any squirt of more than 1.5 mL, more than 0.2 mL given
  toward the throat, or more than 0.3 mL spilled.
- A hazard, which sets 0 for the rest of the run: more than 1.5 mL given toward the throat (choking).
- Score: 0.15 for each whole mL given into the cheek pocket (0.15 to 0.6); 0.8 when at least 4.7 mL went into the
  pocket; 1.0 (success, held 1 s) when, in addition, the syringe lies released and still: its axis within 20 deg of
  level, its middle less than 5 cm above the table (on its rests or flat on the table), no gripper within 4 cm of its
  middle, and still for 1 s (under 1.5 mm). With a fault the score is at most 0.5, and 0.5 once the whole dose (4.7
  mL or more) has been pushed out.

Doing nothing scores 0. A real trial is scored the same way with water: the printed head drains each cheek pocket and
the throat through holes into two cups, weighed afterwards (1 g = 1 mL; the pocket cup needs at least 4.7 g, the
throat cup at most 0.2 g, and at most 0.3 g may be on the absorbent bib below the chin); the squirts and pauses are
read from a side video against the syringe's 0.2 mL marks (a pause is 1 s with the plunger still).

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=give_a_child_liquid_medicine_with_an_oral_syringe.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Give the toddler care manikin its 5 mL practice dose (water) with the oral syringe, then put the syringe back on its rests in the tray." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/osyr-dream
```

Dream-1 refused this task before acting, on both instructions tried (the first, "Give the child its 5 mL dose of
medicine with the oral syringe, then put the syringe back on its rests in the tray.", and the one above), so there is
no Dream-1 recording: "DreamAPI refused the inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction,
refused: Administering medication directly to a person or child poses risk of injury." and "DreamAPI refused the
inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene, refused: The instruction acts on
a lifelike baby manikin with a syringe."

The scripted demo, which reaches full score (open loop from the known layout, reading the simulator's syringe pose
twice as v24 allows): the left gripper comes down over the barrel, its fingers closing across it, the hand leaning
40 deg back from straight down so that it stays clear of the child's nose; it lifts the syringe and measures how it
sits in the hand, then turns it toward the mouth's left corner (the child's right cheek) with the barrel slanted 36
deg toward the cheek and 8 deg down, and slides the tip in along its axis between the side teeth and the cheek,
pressing the cheek out about 1 mm. The right gripper, closed, comes round behind the syringe level with it and puts
its fingertips on the thumb disc (re-aimed every step at where the disc really is), then pushes the plunger in five
1 mL squirts of 2 s each with 2.5 s pauses. It backs off; the left hand draws the tip out along its axis, carries
the syringe back over the tray, lays it on its rests and lets go.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=give_a_child_liquid_medicine_with_an_oral_syringe.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Give the toddler care manikin its 5 mL practice dose (water) with the oral syringe, then put the syringe back on its rests in the tray." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/osyr-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/osyr-*/*.json; do inspect-robots video "$log"; done
```

`OSYR_DEBUG=1` prints, every 5 steps (`OSYR_DEBUG_EVERY=1` for every step), the plunger's travel, where the tip is
(head frame, mm) and how it is booked, how far each cheek is pushed out, the mL booked to the pocket, the throat and
spilled, the current and past squirts, any fault, whether the syringe lies flat, low and still, how close the
grippers are and where each gripper is.

Wrong ways (each scores below full for a physical reason):

- `OSYR_ONE=1`: one arm only. The left hand picks the syringe up and holds its tip in the cheek pocket for 20 s;
  no hand can push the plunger, so nothing is given, and it lays the syringe back. Score 0.
- `OSYR_SQUIRT=5`: the whole dose in one push. The tip is in the pocket, but the right hand pushes the plunger all
  the way in 2 s without a pause: one 4.94 mL squirt, a fault. Score 0.5.
- `OSYR_THROAT=1`: the syringe aimed down the middle of the mouth, its tip inside the row of teeth over the tongue
  (16 mm behind the nasal root, 6 mm from the midline), in five 1 mL squirts: 5.0 mL toward the throat, more than
  1.5 mL, the choking hazard. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs, the scripted demo while rendering 1280 x 720
video took about 9 minutes (974 steps, 48.7 s of simulated time, ending on success), running beside two debug runs
of the task; debug runs of the demo without video (320 x 176) took 6 to 8 minutes for 1,000 to 1,200 steps.
Dream-1 refused at its first request, so there is no Dream-1 timing.Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders. The head's
mesh is built on the first run (about 20 s) and cached next to the task file.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## What is modelled and what is not

Rigid bodies at 20 Hz control. The head's skin, lips and lining are one fixed triangle mesh; the teeth, gum ridges
and tongue are boxes; each inner cheek is a plate on a sprung slide (0.06 N/mm, an estimate) standing in for the soft
cheek, which in a real child wraps round the tip. No liquid is simulated: the dose is the plunger's travel, booked by
where the tip is when it moves, and the plunger slides in the barrel with 0.8 N of Coulomb friction (an estimate for
a smooth-running 5 mL oral syringe, on the low side: real ones need about 1 to 3 N). The child does not move, swallow, cough or close
its mouth, and the head is fixed on its stand.

## Building it for real

- Head: print `child_head_mesh.npz` (export it to STL with a few lines of numpy-stl or trimesh) at full size in PLA
  (about 0.4 kg, 15 h on a desktop printer), with the teeth, gums and tongue printed in as in the task file; glue a 3
  mm silicone sheet (Shore 10A) over each cheek recess so the inner cheek gives; drill a 4 mm drain hole at the floor
  of each cheek pocket and one at the bottom of the throat, each with a silicone tube to its own cup below the stand.
  An off-the-shelf alternative is a pediatric care manikin with an articulating jaw, teeth and tongue (Gaumard Mike
  and Michelle S110, a 1-year-old with a "fully articulating head and jaw with teeth and tongue", about 725 USD
  ([3B Scientific listing][s110])), its jaw propped open 26 mm and its cheek pockets and throat drained the same way.
- Syringe: a 5 mL oral syringe (BD 305218 or any 5 mL oral dispenser with 0.2 mL marks), filled with water to 5 mL.
- An absorbent bib under the chin and a cloth over the stand keep the rig dry; the grippers touch only the syringe.

[gosh]: https://www.gosh.nhs.uk/medical-information/medicines-information/how-give-your-child-liquid-medicines
[ashp]: https://www.safemedication.com/How-To-Use-Medication/Droppers-or-Oral-Syringes
[s110]: https://3bscientific.com/mike-michelle-infant-patient-care-manikin-1005804-w45062-s110m,p_155_5285.html
