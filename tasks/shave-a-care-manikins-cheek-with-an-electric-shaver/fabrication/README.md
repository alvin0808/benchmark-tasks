# Fabrication: the sprung cheek insert and the force-sensing shaver

Two custom parts carry this task's mechanism: the soft cheek insert that slides up toward the temple on a spring, and
the shaver whose head unit sits on a spring over a force sensor, so the shaver itself logs how hard it presses. This
folder has everything needed to build them: STL files (millimetres), a dimensioned drawing (`drawing.svg`,
`drawing.png`), the parts list with catalogue numbers, the assembly steps and the calibration. The STL files are made by
`make_parts.py` from the task file's own head model (the insert's curved faces are the simulated skin surface), and
`make_drawing.py` draws the sheet, cutting its sections from the STL files (both need `numpy`, `shapely`, `trimesh`;
the drawing also `matplotlib`).

## Files

| File | Part | Material |
|---|---|---|
| `cheek_insert_skin.stl` | the insert's 2 mm skin, 30 x 22 mm, its outer face the head model's surface (shape reference) | Ecoflex 00-30 sheet, see below |
| `cheek_insert_carrier.stl` | 1 mm curved plate under the skin; a 4 x 4 mm post through the tray's slot; a 12 x 16 x 2 mm tab with 2 x 2.2 mm holes 8 mm apart for the block; an end wall the return spring pushes on | PETG |
| `cheek_insert_window_tray.stl` | the tray glued into a 40 x 48 mm window cut in the face: its rim continues the face's surface, a 32 x 32 x 3 mm pocket holds the insert flush (0.3 mm clear behind the carrier), its 3 mm floor has the 4.4 x 12 mm slot whose ends stop the post at 0 and 8 mm, and two standoffs (2.5 mm pilot holes) for the frame's M3 screws | PETG |
| `cheek_insert_guide_frame.stl` | 16 x 62 x 3 mm plate for the rail (3 x 2.2 mm holes at 15 mm pitch), two 3.4 mm holes for M3 screws into the tray's standoffs, and the cup that holds the return spring's far end (9.7 mm bore, 8 mm deep) | PETG |
| `shaver_neck_housing.stl` | sleeve over the shaver body's top end (58.6 x 45.6 mm inside, 10 mm deep), a floor with a 12.4 mm square pocket for the sensor board and a wire hole, the spring's bore (12.4 mm), the flange's chamber whose floor is the 6 mm travel stop, and the cap that guides the plunger | PETG |
| `shaver_neck_plunger.stl` | 16.4 mm flange and 12 mm shaft with a 2.6 mm bore for an M3 screw that holds the head unit | PETG |
| `shaver_neck_spring_seat.stl` | 12 mm disc under the neck spring, with a 6.2 mm boss inside the spring and a 2 mm nub that rests on the sensor's steel sphere | PETG |

Print in PETG with 0.12 mm layers and 100 % infill (the curved parts on their backs, with supports). Each STL is a
set of closed shells that overlap by 0.2 mm where they join; slicers merge them. The insert frame
of the drawing (panel B) has x along the face, y up the slide, z out of the cheek, its origin on the skin at the insert's
middle; the neck's z is along the shaver's axis from its body's top end (panel C).

## Bought parts

| Qty | Part | Catalogue part | About (USD) |
|---|---|---|---|
| 1 | miniature linear rail, cut to 40 mm | HIWIN MGNR5R (5 mm wide, 3.6 mm high, M2 holes at 15 mm pitch) | 15 |
| 1 | carriage block for it | HIWIN MGN5C (12 mm wide, 16 mm long, 6 mm assembly height, 2 x M2 8 mm apart) | 20 |
| 1 | insert return spring | Lee Spring LP 018E 03 S316: 0.12 N/mm, free length 19.05 mm, OD 9.14 mm, solid 3.58 mm | 5 |
| 1 | skin | Smooth-On Ecoflex 00-30 (trial unit) and Sil-Poxy silicone adhesive | 45 |
| 1 | force sensor | Honeywell FMAMSDXX005WC2C3 (MicroForce FMA, 5 N compression, I2C at 0x28, 3.3 V, accuracy +-2 % of span = +-0.1 N, 3x overforce) | 40 |
| 1 | neck spring | stainless compression spring, OD 8 mm, wire 0.70 mm, free length 12 mm, about 1.0 N/mm, solid under 6 mm (any spring maker's stock spring near these values; the calibration below sets the check) | 3 |
| 1 | logger | Seeed Studio XIAO nRF52840 Sense (I2C, built-in 6-axis IMU, Bluetooth LE, LiPo charging) | 16 |
| 1 | battery | 3.7 V LiPo cell, 250 to 400 mAh, JST lead | 8 |
| 1 | shaver | a three-head rotary shaver of about 16 x 6 x 4.5 cm and 170 g (for example from the Philips Series 3000 range whose manual the task cites) | 50 |
| 1 | tracer | Glo Germ Gel (washable UV-fluorescent training lotion) | 25 |
| 2 | UV torch | 365 nm LED torch, 3 to 5 W, each on a small stand | 30 |
| 3 | camera | Logitech C920 HD Pro webcam (1920 x 1080 at 30 fps; focus and exposure locked in its settings), or any phone recording 1080p at 30 fps, each on a small stand or clamp | 3 x 60 |
| 1 | force gauge | 5 N spring scale or digital push-pull gauge with 0.05 N divisions or finer | 20 to 80 |

Ordinary items: M2 x 6 screws (3 for the rail, 2 for the block), M3 x 8 screws (2) and one M3 x 10 for the head unit,
two-part epoxy, 4.7 kOhm resistors (2, I2C pull-ups), thin four-core wire, a small piece of 1.6 mm FR-4 board for the
sensor (any PCB service, with the land pattern from the sensor's datasheet; cut to 12 x 12 mm), a printed millimetre
scale, a transparent 3 mm grid card printed from panel A of the drawing, nitrile finger cots (or the cut-off fingers of
nitrile gloves), a red lipstick and a dark eyeliner pencil, a laptop that records the three cameras and the shaver's
log.

## Building the cheek insert

1. Cut a 40 x 48 mm window in the manikin's vinyl over its left cheekbone where panel A shows: its front edge 20 mm in
   front of the ear canals' midpoint, its lower edge 3 mm below their level (along the face). Glue the window tray in
   with epoxy so its rim continues the face; sand the seam flush and paint the rim skin colour.
2. Cast a flat 2 mm sheet of Ecoflex 00-30 (pour it into any flat tray to 2 mm), cut a 30 x 22 mm piece and bond it to
   the carrier's curved face with Sil-Poxy; it takes the carrier's curve (`cheek_insert_skin.stl` shows the result).
   Paint a 1 mm dark line along its lower edge, and glue a printed millimetre scale on the tray's rim just in front of
   the pocket, beside the insert's lower edge, its 0 level with that edge at rest. The gripper's fingertips press the
   insert's middle and upper part, so its lower edge and the scale stay in view while the skin is held.
3. Screw the rail to the guide frame (3 x M2), slide the block on, put the return spring in the frame's cup.
4. From inside the head, pass the carrier's post through the tray's slot, screw its tab to the block (2 x M2), and screw
   the frame to the tray's standoffs (2 x M3), so the spring sits between the cup and the block's top end and the
   carrier's end wall. At rest the post rests on the slot's lower end with the spring just free (no preload), the skin
   flush with the face; pulled up, the post stops at the slot's upper end, 8 mm on.

## Building the force-sensing shaver

1. Take off the shaver's head unit and remove its head holder (the neck with the drive shafts) and the motor; the
   cutters no longer turn. Keep the head unit with its own pivoting frame (it tilts +-20 degrees on its centring spring).
2. Solder the force sensor on its 12 x 12 mm board with four wires (VS, GND, SCL, SDA, pins 1, 3, 4 and 5); put the
   pull-ups at the logger. Lay the board in the housing's pocket, sensor up, the wires down through the floor's hole into
   the body, where the logger and the battery go in the motor's place.
3. Print the spring seat so that its top stands 7.3 mm above the floor's underside: its disc is 4.8 mm minus the
   sensor's height above its board (cover plus sphere, measured with calipers); the file's disc is 1.8 mm, so scale it
   in height in the slicer. Put the seat on the sensor (the nub on the sphere), the neck spring on its boss, the plunger
   in, and press the housing onto the body's top end (tape round the body if the sleeve is loose).
4. Screw the head unit's pivot frame to the plunger with an M3 x 10 screw. Pressed, the head unit goes in up to 6 mm,
   then the flange rests on the stop and the sensor never sees more than about 6 N (its limit is 15 N). The finished
   shaver is about 1.5 cm longer than the simulated one; weigh it and add or remove ballast inside to 170 +- 10 g.
5. The logger reads the sensor over I2C (force = (counts - 3277) x 5 N / 9830 before calibration; the calibration below
   replaces this), averages the readings in each 50 ms, and writes one line per 50 ms, time and force, over Bluetooth
   to a laptop and to its own flash. It subtracts the head unit's own weight along the shaver's axis, 0.245 N times the
   cosine of the axis's angle from straight down (from its IMU), as the simulation does. It tares itself when the shaver
   lies in its tray before each trial.

## The manikin's sprung neck

The head's base plate sits on a 50 mm steel butt hinge screwed to a bench plate, its axis along the robot's left-right
(y) line, so the head can tip toward and away from the robot. Two compression springs of about 6.25 N/mm, 40 mm either
side of the hinge between the two plates and each pressed a few millimetres, hold it upright: 2 x 6.25 N/mm x (40 mm)^2
is 20 N m/rad, the simulated neck's stiffness. Any stock springs near that rate will do; the check below sets them.

## Calibration (before each session)

- Insert spring: push the insert up by its lower edge with the force gauge: 0.36 N at 3 mm and 0.96 N at 8 mm, each
  within 15 %, read on the scale beside its lower edge; let go: it returns to 0.
- Force sensor: clamp the shaver upright, heads up; tare; put 20, 50, 100, 200 and 400 g on the head unit's middle,
  read each, and fit a straight line; every reading must be within 0.1 N of the mass times 9.81 m/s2.
- Neck spring: 1 N on the head unit (102 g, by the sensor's reading) presses it in 1.0 +- 0.15 mm (feeler gauge
  between the flange and the stop).
- Manikin's neck (a spring hinge at its base, about 20 N m/rad): 5 N pushed level into the middle of the beard area
  (11.6 cm above the hinge) moves it 4.0 +- 0.5 mm along the push (0.8 mm per newton), read on a ruler on a stand.
- Gel: on a vinyl test tile with the same gel layer, one pass of the shaver at 0.5 N leaves a dark track as wide as a
  head (20 mm) under the UV torch.

## Cameras for the real-trial scoring

Three cameras film the trial, placed so that the shaver and the hand on the insert do not hide what each must see.
Positions are in cm in the task's frame (x forward from the midpoint between the arm bases, y to the robot's left, z
above the table); the beard area spans x 22.9 to 27.9, y 1.8 to 8.3, z 9.8 to 18.1, and the insert sits at x 22.7,
y 4.0, z 21.1.

| Camera | Where | Aimed at | Sees |
|---|---|---|---|
| A, low in front of the jaw | (13, 28, 3), beside the tray, 3 cm up | the beard's middle (25, 5, 14), 28 cm away | the chin, the jaw and the front of the cheek from in front and below |
| B, low behind the jaw line | (8, -10, 3), 3 cm up | (25, 4, 12), 24 cm away | the jaw angle, the sideburn and the back of the cheek from behind and below |
| C, level with the insert | (12, 22, 20) | the insert's lower front corner (22.7, 5.5, 20), 20 cm away | the insert's lower edge against the rim's scale, under the fingertips and above the shaver |

At these distances a C920 image spans about 30 to 40 cm, so a 3 mm square of the beard is about 10 to 15 pixels
across in A and B (seen at 40 to 60 degrees), and a millimetre on the scale about 5 pixels in C. The two UV torches
stand beside A and B. Before each session, put the shaver's heads on the middle of the beard and the right gripper's
fingertips on the insert (by hand or by moving the arms there slowly), and check each view: every beard square outside
the heads must be visible in A or B, and C must show the insert's lower edge and the scale; move a camera until they
are. Record all three with the shaver's log on one laptop; the tap on the head unit at the start marks the same moment
in the log and the videos.

## Keeping the shared robot clean

The tracer is a washable lotion, not a powder: it is rubbed thinly on the manikin's beard area only, and it comes off
the shaver's heads and the face with soap and water. The gripper fingertips wear nitrile finger cots (taped at their
base), changed when marked; the right gripper drags the insert, which carries no gel. Lipstick and eyeliner mark the lips,
nostril rims, eyelids and ear for the touch checks; they too stay on the manikin, the shaver's heads and the cots.
