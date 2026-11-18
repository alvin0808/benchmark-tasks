# Fabrication: the training hand with its sprung, contracted fingers

The task's mechanism is the training hand: a true-size adult woman's right forearm and hand whose fingers are held in
a fist by springs, so that they can be opened gently but close again the moment they are let go. This folder has
what is needed to build it: the printed parts as STL files (millimetres, closed surfaces, the same shapes the
simulation uses), a drawing sheet (`drawing.png`), the parts list, the assembly steps and a check of the springs by
hanging weights. `make_parts.py` writes the STL files and the drawing from `hand_mesh.py` (it needs `numpy` and
`matplotlib`). The palm cone is bought, not made.

## Files

| File | Part | Notes |
|---|---|---|
| `hand_base.stl` | forearm and palm shell with the thenar and hypothenar pads | frame: origin on the knuckle axle under the middle finger, x toward the forearm, y toward the thumb, z out of the palm (palm up); the back of the hand at z = -14 mm |
| `finger_<index, middle, ring, little>_first.stl` | each finger's first segment (knuckle to middle joint) | drawn in the resting fist; drill the 3 mm knuckle bore and the 2 mm middle-joint bore at the segment's ends |
| `finger_<index, middle, ring, little>_end.stl` | each finger's middle and end segments, printed as one piece bent 30 deg at the end joint | same frame |
| `thumb.stl` | the thumb (two segments, one piece) | its knuckle at x 36, y 50, z -1 mm |
| `drawing.png` | A: top view with the main dimensions; B: the middle finger's joints at rest, held open and on the cone; C: the moments the springs must give; D: the cone | |

## Bought parts (about 125 USD, pad and cone included)

| Qty | Part | About (USD) |
|---|---|---|
| 1 kg | PLA filament, skin colour | 20 |
| 1 | 3 mm ground steel rod, 100 mm (the knuckle axle), with two shaft collars | 6 |
| 5 | 2 mm steel pins, 20 mm (four middle joints and the thumb) | 3 |
| 1 | torsion spring K (the knuckle bar, all four fingers): 1.40 N mm per degree (0.080 N m/rad), at least 100 deg of working deflection, bore over 3.5 mm | 6 |
| 4 | torsion springs F (each finger on the bar): 2.6 N mm per degree (0.15 N m/rad), at least 15 deg of working deflection, bore over 3.5 mm | 12 |
| 4 | torsion springs P (middle joints): 0.35 N mm per degree (0.020 N m/rad), at least 90 deg of working deflection, bore over 2.5 mm | 10 |
| 1 | torsion spring T (thumb): 0.52 N mm per degree (0.030 N m/rad), at least 40 deg of working deflection, bore over 2.5 mm | 3 |
| 0.3 kg | steel shot and epoxy (ballast in the forearm) | 8 |
| 1 | firm foam mat 46 x 39 x 2 cm with a wipe-clean vinyl cover (the pad) | 20 |
| 1 | firm palm cone, 13 cm long, about 2.5 cm across at its narrow end and 4.2 cm at its wide end, terry-covered (for example the Posey Palm Cone, REF 5608, 13 cm long, one size) | 30 |
| 1 | protractor card, 30 cm ruler, black PVC tape | 5 |

Stock torsion springs are listed by their torque at full deflection (McMaster-Carr, Lee Spring, Century Spring);
divide by the deflection in degrees and pick the nearest to the rates above; the check below says how far off is
acceptable.

## Building

1. Print the shell in PLA, 0.2 mm layers, 3 perimeters, 5 % infill, split along the y = 0 plane so each half prints
   on its cut face; glue the halves with cyanoacrylate. Leave the knuckle end open between the knuckle pads.
2. Knuckle axle: drill a 3 mm hole across the shell's knuckle end along the axle line (14 mm above the back of the
   hand, under the knuckle line in the drawing) and fit the rod with a collar at each side. On it, inside the shell,
   print a knuckle bar 66 mm long with four slots; each finger's first segment sits in its slot on the same axle.
3. Springs at the knuckles: spring K between the shell and the bar, so that the bar is pulled toward the fist; a
   printed stop on the shell where the bar is at 90 deg (the resting fist) and one where it is 10 deg past straight.
   Each first segment has its own spring F between it and the bar, and two printed stops on the bar that let it turn
   8 deg open and 6 deg closed about the bar. All springs are unloaded in the resting fist.
4. Middle joints: pin each end piece to its first segment with a 2 mm pin and a spring P, unloaded at the resting
   bend (106, 102, 98 and 88 deg from the index to the little finger, drawing panel B), with a printed stop there; the
   end pieces open freely to well under 30 deg of bend.
5. Thumb: pin it to the shell's side at its knuckle, upright, with spring T holding it against a stop beside the
   index finger; it swings out up to 40 deg.
6. Ballast: steel shot in epoxy inside the forearm, 100 to 200 mm up from the wrist crease, until the shell with the
   axle weighs 0.50 kg; each first segment should weigh about 8 g, each end piece 8 g, the thumb 12 g.
7. Tape a 6 mm black strip along the back of the index finger's first segment and one along the back of the hand
   beside it, for reading the knuckle angle on the side video.

## Checking the springs (drawing panel C)

Lay the hand on its side on the table, thumb up, so that the finger joints turn in a horizontal plane (their weight
then does not act). Tie a thread round all four first segments together 40 mm from the axle, run it at right angles
to them over a pulley, and hang masses from it; then do the same for one middle joint, the thread round the end piece
20 mm from its pin. The angle (from the tape strips, read with the protractor card) should be:

| Joint | Mass on the thread | Moment | Opened from rest |
|---|---|---|---|
| knuckles, all four fingers | 0.16 kg at 40 mm | 0.063 N m | 51 deg (45 at the bar, 6 more on the four F springs) |
| knuckles, all four fingers | 0.28 kg at 40 mm | 0.112 N m | 88 deg (80 at the bar, 8 more, to the F stops) |
| one middle joint | 0.08 kg at 20 mm | 0.016 N m | 45 deg |

Within 5 deg at each row is good; otherwise swap the spring for the next stock rate.
