# Fabrication: the training leg with its sprung ankle

The task's mechanism is the training leg: a true-size adult woman's lower leg and foot on a hinge at the ankle whose
passive moment stiffens toward each end of the range. This folder has what is needed to build it: the two shells as
STL files (millimetres, closed surfaces, the same meshes the simulation uses), a drawing sheet (`drawing.png`: side
view with the main dimensions, the spring layout, and the moment curve the ankle must give), the parts list, the
assembly steps and the calibration. `make_parts.py` writes the STL files and the drawing from `leg_mesh.py` and the
task file (it needs `numpy` and `matplotlib`).

## Files

| File | Part | Notes |
|---|---|---|
| `leg_shank.stl` | lower leg, from 15 mm below the ankle axis to the knee end 415 mm above it | frame: origin on the ankle axis between the malleoli, y along the leg, z out of the shin, x across (the outer side is -x) |
| `leg_foot.stl` | foot, 246 mm long, toes along +z with the ankle in neutral | same frame |
| `drawing.png` | A: side view and dimensions; B: springs and stops; C: the moment-angle curve | |

## Bought parts (about 110 USD)

| Qty | Part | About (USD) |
|---|---|---|
| 1 kg | PLA filament, skin colour | 20 |
| 1 | 8 mm ground steel shaft, 80 mm long (the ankle axle), with two shaft collars | 8 |
| 2 | 8 mm bore flanged sleeve bushings (oil-impregnated bronze), 12 mm OD, 8 mm long | 4 |
| 1 | torsion spring D (resists dorsiflexion): 1.8 N m/rad (0.031 N m per degree), at least 35 deg of working deflection, bore over 9 mm (music wire about 2.3 to 2.6 mm) | 10 |
| 1 | torsion spring P (resists plantarflexion): 1.2 N m/rad (0.021 N m per degree), at least 60 deg of working deflection | 10 |
| 2 | polyurethane bump stops, Shore 70A, 12 mm diameter, 10 mm thick (cut from rod) | 6 |
| 0.15 kg | steel shot and epoxy (ballast in the calf) | 8 |
| 1 | firm foam mat 25 x 65 cm, 1 to 2 cm thick, wipe-clean vinyl cover (bed pad) | 25 |
| 2 | firm foam blocks 12 x 7 x 3 cm (the leg trough's walls), glued to the pad | 6 |
| 1 | black PVC tape, 6 mm | 2 |
| 1 | protractor card and 30 cm ruler, stood behind the ankle for the trial video | 5 |

Stock torsion springs are listed by their torque at full deflection (McMaster-Carr, Lee Spring, Century Spring);
divide by the deflection in degrees and pick the nearest to the two rates above; the calibration below sets the
final moment with the bump stops' thickness.

## Building

1. Print both shells in PLA, 0.2 mm layers, 3 perimeters, 5 % infill, split along the x = 0 plane (inner and outer
   halves) so each half prints on its cut face. Glue the halves with cyanoacrylate.
2. Ankle: in each lower-leg half, at the axis (the malleoli), glue a printed block 25 mm across holding one bushing on
   the axis. Inside the foot, at its top, glue a printed yoke whose two cheeks fit between the bushings and clamp the
   axle (an M4 set screw onto a flat on the axle). Cut the shells where they meet at the ankle so that the foot turns
   30 deg toward the shin and 55 deg down before shell meets shell (these are the hard stops).
3. Springs: put both torsion springs on the axle inside the ankle block. Each has one leg against the lower leg's
   block and the other against a pin on the yoke on its own side only, so spring D is loaded only when the foot turns
   toward the shin and spring P only when it turns down, both from neutral (no preload).
4. Bump stops: glue one polyurethane stop on the lower leg's block where the yoke's stop finger meets it at 10 deg
   toward the shin, the other where the finger meets it at 20 deg down. They add the stiffer second part of each side
   of the curve.
5. Ballast: add steel shot in epoxy inside the calf, 240 mm up from the axis, until the lower leg weighs 0.30 kg and
   balances on a knife edge 240 mm above the axis; the foot (with its yoke) should weigh 0.15 kg.
6. Tape a 6 mm black strip along the outer side of the shin (20 to 140 mm above the axis, along the leg) and one along
   the outer side of the foot (30 to 130 mm along the foot from the axis), at right angles with the ankle in neutral.

## Calibration (moment curve, drawing panel C)

Clamp the lower leg on its side on the table, so that the ankle axis is vertical and the foot turns in a horizontal
plane (its weight then does not act), the foot past the table edge. Tie a cord to the forefoot 150 mm from the axis,
run it at right angles to the foot over a pulley at the table edge, and hang masses from it; turn the leg over for the
other direction. The angle (between the tape strips, read with the protractor card) should be:

| Direction | Mass on the cord at 150 mm | Moment | Angle |
|---|---|---|---|
| toward the shin | 0.21 kg | 0.31 N m | 10 deg |
| toward the shin | 0.53 kg | 0.79 N m | 15 deg |
| toward the shin | 0.85 kg | 1.26 N m | 20 deg |
| toes down | 0.29 kg | 0.42 N m | 20 deg |
| toes down | 0.64 kg | 0.94 N m | 30 deg |
| toes down | 1.17 kg | 1.73 N m | 45 deg |

Within 2 deg at each row is good; otherwise change the bump stops' thickness or position (the rows past 10 deg toward
the shin and past 20 deg down) or the springs (the others).
