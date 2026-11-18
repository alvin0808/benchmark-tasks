# Fit a palm-protector cone into a care manikin's contracted hand, its curled fingers held open and eased down onto it

A resident whose hand has contracted cannot open it: the fingers stay curled into the palm and the nails can open
the skin there. Care staff fit a palm protector, often a firm padded cone, into such a hand every morning, after
washing and drying it. Here a true-size adult care-training forearm and right hand lies palm up
on a firm foam pad, its fingers curled into a fist by sprung joints, and a firm palm cone lies beside it. The
instruction asks for the cone to be fitted across the palm under the fingers, narrow end at the thumb, with all four
fingertips resting on it, without forcing the fingers and with the hand kept on the pad; it does not say how.

Why it takes both arms follows from the hand. The fingers spring shut the moment they are let go, and the fist
leaves no room for the cone: under the curled fingers there is a space about 2 cm across, closed off by the
fingertips, which hang 0.6 to 1 cm above the palm, and the cone is 2.5 to 4.2 cm across. So one hand holds the
fingers open while the other lays the cone in the palm, and the fingers are then eased back down onto it. One gripper
alone can do only one of these at a time: if it opens the fingers and lets go to fetch the cone, they snap shut; if it
brings the cone, it meets a closed fist.

The task is the task file
[`fit_a_palm_protector_in_a_contracted_hand.py`](fit_a_palm_protector_in_a_contracted_hand.py) for the `yam_arena`
embodiment (scene, scoring and the scripted demo), with [`hand_mesh.py`](hand_mesh.py) (the hand's moulded surfaces,
finger sizes and resting curl, from measurements) and [`bodykit.py`](bodykit.py) next to it.
[`fabrication/`](fabrication/README.md) has the STL files, a drawing and the parts list for building the hand.

## Why it matters and how it is done

Nursing assistants ([O*NET 31-1131.00][onet-na], about 1.4 million US jobs; tasks "Turn or reposition bedridden
patients" and "Apply clean dressings, slings, stockings, or support bandages, under direction of nurse or physician")
and home health and personal care aides apply positioning devices to residents' contracted hands as part of daily
care. In 2005 US survey data (OSCAR) 28.9 % of about 386,000 nursing-home residents had contractures
([Annals of Long-Term Care review][altc]).

- The open textbook *Nursing Assistant* (Chippewa Valley Technical College, [9.4 Complications of Immobility][wtcs])
  lists "Applying splints or positioning devices as prescribed" against contracture and says: "The first type of hand
  device is a cone that slides into the palm of the hand and is kept in place with a soft elastic band. The cone
  should not be forced into the fingers but placed gently." Its photograph shows the cone across the palm, the fingers
  wrapped round it.
- Oxford Health NHS Foundation Trust, Care Home Support Service, [Palm protectors: advice for care home staff][oxf]:
  "Fit the palm protector by gently prising open the fingers and thumb. Place the thumb through first and then slide
  the sheepskin pad under the fingers as shown."
- The Royal Wolverhampton NHS Trust, [Palm Cone][rwt] (physiotherapy and occupational therapy leaflet): "Place the cone
  inside the palm of the hand, the narrower end of the cone should be towards the thumb and the wider end towards the
  little finger".

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the hand, the cone and the pad are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them, put one on a hinge or give one
a curved surface. The hand (smooth triangle meshes and rounded finger segments on sprung joints) and the cone are
built with `bodykit.py`, which adds such bodies and joints: importing it hooks `yam_arena.scene` when the simulator
starts, without changing yam_arena, and gives the task file each body's pose and each joint's angle. This copy lets a
moving part collide as its mesh's convex hull and gives a sprung joint a drive (a USD angular drive with a force
limit), without which its spring does nothing.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Pad, fixed: firm foam with a wipe-clean light blue cover, 46 x 39 x 2 cm, x 10 to 56, y -25 to 14.
- Training hand, loose, lying palm up on the pad: an adult woman's right forearm and hand at true size, moulded skin
  colour (soft vinyl, friction 1.0), its surfaces built from ANSUR II women's means ([public data][ansur], 1,986
  women): hand length 181 mm, hand breadth 78 mm, palm length 109 mm (so the middle finger is 90 mm from its knuckle
  joint to its tip), wrist 155 mm round. The forearm is cut 20 cm above the wrist. The fingers point toward the robot
  (-x), the thumb to the robot's left (+y). The knuckle (MCP) joints' axis runs along y at x 24.0, z 3.5 (1.4 cm
  above the back of the hand); the middle finger's knuckle is at y 4.8, the palm spans y 0.1 to 7.9 and x 22.6 to
  33.1 (wrist crease), the forearm runs to x 53.1. Forearm and palm are one hollow shell, 0.50 kg; the whole hand
  0.58 kg.
- The hand's joints, all sprung toward the resting fist (spring rates in N m per radian):
  - Knuckles: the four fingers turn together on one sprung joint, 0.08 (as the fingers of a contracted hand flex
    together), each finger with its own sprung play of 8 deg open and 6 deg closed about it, 0.15. The knuckles rest
    at 90 deg and open to 10 deg past straight at their stop. Damping 0.03 N m s/rad: let go from wide open, the
    fingers close at about 200 deg/s.
  - Middle joints (PIP): one spring each, 0.02, resting at 106, 102, 98 and 88 deg (index to little finger); the end
    joints are fixed at 30 deg. At rest each fingertip is curled down to 0.6 to 1.0 cm above the palm.
  - Thumb: lying on its side along the index finger's outer side, at the palm's mid-thickness, on one joint about
    the vertical through its knuckle, 0.03, resting against a stop beside the index finger and swinging out against
    its spring.
- Palm cone, loose: a firm cone covered in tan terry cloth with a white elastic band sewn along it, 13 cm long, 2.5 cm
  across at its narrow end and 4.2 cm at its wide end, 0.07 kg (the Posey Palm Cone, REF 5608, is such a cone, 13 cm
  long; its end diameters are not published, so measure the one used). It lies on its side on the pad with its axis
  along y, wide end toward -y, centre at x 26.0, y -15.5. Laid in the palm as the leaflet says, its narrow end
  reaches about 1.5 cm past the index finger's outer side, over the thumb, and its wide end 4.5 cm past the little
  finger's side.
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads each finger's knuckle and middle-joint angles, the thumb's, where the hand lies,
where the cone lies in the hand, how far each fingertip is from the cone's surface, and the points of each gripper
finger's collision hulls.

- 0 for the rest of the run once a rule is broken: a finger jerked open (a knuckle opening faster than 90 deg/s
  between two control steps); a finger let snap shut (any finger joint closing faster than 150 deg/s, as when the
  fingers are let go before the cone is in); a finger opened past straight at its knuckle, or its middle joint
  straightened to under 30 deg of bend; the thumb pushed out more than 35 deg; a finger pushed open more than 10 deg at
  the knuckle by the cone pressing on it (the cone forced into the fingers); the hand lifted more than 1 cm off the
  pad or moved more than 2 cm along it (at its knuckles or at its forearm's end).
- 0.25: all four fingers held open at once (each knuckle bent 45 deg or less).
- 0.5: the cone seated in the palm: lying across it (its axis within 25 deg of the knuckle line), its narrow end at
  the thumb side (from 3 cm inside to 2.5 cm beyond the index finger's outer edge), its middle no more than 4 cm
  behind the knuckle line, and its lowest point over the palm (within 3.5 cm of the palm's middle line, 0 to 8 cm
  behind the knuckles) within 1 cm of the palm's surface.
- 0.75: the cone seated and all four fingertips resting on it (each within 2.5 mm of its surface).
- 1.0 (success, held 1 s): that, with every gripper finger point 3 cm or more from the hand and the cone, and the
  cone and the fingers still (under 1 mm and 0.5 deg over 1 s).

Doing nothing scores 0. In a real trial a phone on a stand films the hand from the thumb side at knuckle height, with
a protractor card standing behind the knuckles, and the robot's top camera (or a phone above) photographs the palm
with a ruler laid along the knuckles: the knuckle and middle-joint angles and their speeds come from the side video
frame by frame (more than 3 deg of opening, or 5 deg of closing, between frames 1/30 s apart breaks the rules); where
the cone lies and whether each fingertip touches it, from the photo; the hand's lift and drift, from the side video
and from tape marks on the pad beside the knuckles and the forearm's end.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=fit_a_palm_protector_in_a_contracted_hand.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the padded palm cone into the curled hand of the plastic training-manikin arm on the pad: the cone lying across the palm under the fingers, its narrow end at the thumb, with all four fingertips resting on it. Do not force the fingers, keep the hand on the pad, and let go when it is in." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/palm-dream
```

The scripted demo, which reaches full score:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_a_palm_protector_in_a_contracted_hand.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the padded palm cone into the curled hand of the plastic training-manikin arm on the pad: the cone lying across the palm under the fingers, its narrow end at the thumb, with all four fingertips resting on it. Do not force the fingers, keep the hand on the pad, and let go when it is in." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/palm-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/palm-dream/*.json; do inspect-robots video "$log"; done
```

The scripted demo (open loop, from the known layout) reaches full score. The left gripper comes down over the fist,
jaws across it front to back, and closes on the curl of the middle and ring fingers (front jaw on the backs of their
first segments, rear jaw on their end joints); it turns the curled fingers open about the knuckle axis, 80 deg in
6 s (the knuckles go from 90 to about 23 deg; the index and little fingers follow on the common knuckle spring), and
holds them there. The right gripper, its wrist leaning 30 deg to the robot's right, picks the cone up by its wide end,
carries it over and lays it across the palm, narrow end at the thumb, 2.2 cm behind the knuckle line. The left
gripper turns the fingers back down until their tips are just over the cone (knuckles 44 deg), loosens its grip so
that the fingers lean on its rear jaw, and eases them on down until all four fingertips rest on the cone (knuckles
about 50 deg). The right gripper lets go of the cone and rises. The left gripper then opens slowly over 3 s, so that
the fingers follow its rear jaw down and settle on the cone (knuckles about 60 deg; the cone rolls 0.5 cm toward the
wrist), turns its open jaws on, away from the backs of the fingers, and lifts clear. The run succeeds at step 741
(37 s).

`PALM_DEBUG=1` prints, every 5 steps (`PALM_DEBUG_EVERY=1` for every step), the stage reached, each finger's knuckle
and middle-joint angles, the thumb's, the fastest joint speed, the hand's lift and drift, each fingertip's distance
from the cone, where the cone lies in the hand, how close the nearest gripper finger is, where each gripper is and
any rule broken; and a second line with every joint's raw angle.

`PALM_ONE=1` runs the demo the wrong way: the right gripper alone, the left arm staying where it is. The cone,
brought down over the palm, lands on the curled fingers and presses them onto their stops (knuckles 98 deg): its
lowest point over the palm stays 2 cm up, on the fingers' middle segments, and it lies turned 22 deg. When it is let
go it slides off the fingers and off the hand, 12 cm behind the knuckle line. It is never seated: score 0.

`PALM_LETGO=1` runs the other one-gripper attempt, for comparison: the left gripper opens the fingers as in the demo
and then lets go of them (as it would to fetch the cone), the right arm staying where it is. The fingers spring shut,
the middle finger's knuckle closing at 196 deg/s, which the scorer counts as letting them snap shut: score 0 (after
0.25 for holding them open).

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs, the scripted demo while rendering
1280 x 720 video took 228 s from start to finish (741 control steps, 37 s of simulated time, ending on success),
sharing the GPU with a second simulation. Debug runs of the demo without video took 6 to 9 minutes with two
simulations on the GPU. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.
A Dream-1 trial of 2,400 steps at 320 x 176 took 354 s on the same GPU, beside a second simulation.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

The training hand is 3D-printed from [`fabrication/`](fabrication/README.md): the forearm-and-palm shell and the
finger segments (STL files, the same surfaces as in the simulation), a 3 mm steel knuckle axle with a printed knuckle
bar, pinned middle joints and small torsion springs at the rates above, with a check by hanging weights; about
125 USD of parts with the pad and the cone, and two days of printing. The pad is any firm 2 cm foam mat with a
wipe-clean cover; the cone is a stock firm palm cone (about 30 USD; one UK seller lists a 13 cm terry-covered
anti-contracture palm cone at 28 GBP). For each trial lay the hand palm up on its taped marks on the pad with the
fingers at rest, lay the cone on its mark, film as above, and score from the video and the photo.

[onet-na]: https://www.onetonline.org/link/summary/31-1131.00
[altc]: https://www.hmpgloballearningnetwork.com/site/altc/article/are-joint-contractures-patients-alzheimer-s-disease-preventable
[wtcs]: https://wtcs.pressbooks.pub/nurseassist/chapter/9-4-complications-of-immobility/
[oxf]: https://www.oxfordhealth.nhs.uk/wp-content/uploads/sites/24/2023/06/1.6.4-Palm-protector-advise-for-care-staff.pdf
[rwt]: https://royalwolverhampton.nhs.uk/wp-content/uploads/2025/06/Palm-Cone.pdf
[ansur]: https://www.openlab.psu.edu/ansur2/
