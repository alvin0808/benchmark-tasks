# Scan four groceries across a checkout scanner, one read each, and put them on the bagging shelf

A checkout counter on the station table. In the middle, set into the counter, is an in-counter bioptic barcode scanner
at true size (a Zebra MP7000 short: a horizontal window in a platter flush with the counter, a tower with a vertical
window facing the robot, and a two-colour LED). On the robot's left is the end of the conveyor belt with four
groceries on it: a 1 lb spaghetti box, a 10.75 oz can of condensed soup, a 16 oz glass jar of salsa and a 16.9 fl oz
bottle of water. Each carries a real 12-digit UPC-A barcode on one side, and none starts with its barcode facing either
window. On the robot's right is the bagging shelf. The instruction asks for the four groceries to be rung up on the
scanner, each item read once, and put on the bagging shelf; it does not say how.

## Why it matters, and the procedure

Cashiers ([O*NET 41-2011.00][onet]; 3,106,300 US jobs in 2025) "Establish or identify prices of goods, services, or
admission, and tabulate bills, using calculators, cash registers, or optical price scanners", and "Bag, box, wrap, or
gift-wrap merchandise". The same hand skill is asked of every shopper at a self-checkout. At a bioptic scanner the
cashier passes each item so that its barcode faces one of the two windows, with no finger over the code, and listens
for the beep. Zebra's documents for the MP7000 give what the task checks:

- [MP7000 Integrator Guide][ig] (MN-002914), "Scanning": "Position the barcode anywhere within the MP70XX's field of
  view. [...] The scanner beeps to indicate a successful decode, and the green LED flashes momentarily"; Table 5-1:
  "Barcode decoded: [...] Bright green [...] One bright green LED flash"; "System is idle: [...] Dim green [...] LED
  constant on; ready for decode".
- [MP7000 Bar Code Programming Guide][prg] (MN-002912-12EN), p. 6-32, "Timeout Between Decodes, Same Symbol": "Use this
  option in presentation mode to prevent the scanner from continuously decoding the same barcode when it is left in the
  scanner's field of view. The barcode must be out of the field of view for the timeout period before the scanner reads
  the same consecutive symbol. [...] The default interval is 0.5 seconds." And p. 14-4, among the reasons "a
  label/barcode may not be decoded while in the field of view": "The label is presented at extreme angle to scanner."
- [MP7000 specification sheet][spec]: short scanner 351 mm L x 292 mm W x 102 mm D x 128 mm H; "Large two-color LED";
  "Beeper: adjustable volume and beep tones"; horizontal glass "Sapphire".

So an item must be read exactly once: an item put in the bag unread leaves the store unpaid, and an item passed back
across the windows after its barcode has been out of view for 0.5 s is rung up twice and overcharges the customer.

How this differs from turning parcels' labels toward a fixed scanner on a moving line: there the cartons stay on their
carriers and are only yawed, without being picked up. Here each item is picked up and turned in the hand (by a quarter
or half turn), the grip has to be chosen so that no finger covers the code (the box is held 2.5 cm from its middle, on
the side away from its code), the code is passed across the windows within reading range and the reads are counted
(none missed, none doubled), and the item is handed on and set down; the four items are different real shapes (a long
box, a squat can, a heavy glass jar, a tall bottle).

## Why it takes both arms, as seen in the scene

The belt's end and the bagging shelf are on opposite sides of the counter, farther apart than one arm reaches. The
right arm's gripper centre reaches no further left than 45 cm from the middle between the arm bases, with the gripper
in any orientation (computed from the YAM's kinematics and joint limits; less than that farther forward), and the
groceries' middles are 49 to 60.5 cm to the left, so it cannot take any of them. The left arm's gripper centre reaches
no further right than 45 cm, and the shelf's inner edge is 46 cm to the right, so it cannot set any of them down on the
shelf (and with its gripper pointing down, as setting a glass jar down needs, it reaches only 15 cm to the right). So
the items go from the belt to the left hand, across the scanner, and on to the right hand, which puts them on the
shelf, as a cashier passes items from one hand to the other across the scanner. `SCAN_ONE=1` (below) shows what the
left hand manages alone.

The task is the task file [`scan_four_groceries_across_a_checkout_scanner.py`](scan_four_groceries_across_a_checkout_scanner.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. No third-party models or assets are used: every object is built in the task
file from boxes and cylinders.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them. `bodykit.py` adds bodies made of
several shapes (the groceries with their labels and printed bars, the scanner, the belt, the shelf): importing it hooks
`yam_arena.scene` when the simulator starts, without changing yam_arena, and lets the task file follow the groceries
and the arms after every control step and move the LED's lit lens into place.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left; heights above the counter (the table top).

- Scanner, fixed: a Zebra MP7000 short, 35.1 cm along y by 29.2 cm along x (x 2.0 to 31.2, y -2.3 to 32.9), its
  platter flush with the counter. The horizontal window (sapphire, 15 x 10 cm) is at x 9 to 19, y 7.8 to 22.8. The
  tower, 12.8 cm tall, is at x 23.7 to 31.2; its vertical window (22 x 9.2 cm) is on its front face at x 23.7, y 4.3 to
  26.3, 1.8 to 11.0 cm above the counter, facing the robot. The LED lens is on the tower's right end (y -0.7), dim green
  when the scanner is ready; on each read it flashes bright green for 0.5 s. (Zebra's sheet gives only the outside
  dimensions; the windows' sizes are estimated from Zebra's photos.)
- Belt end, fixed: black rubber flush with the counter, x 2 to 40, y 45.5 to 64.5.
- Groceries, loose, each with a UPC-A barcode at 100 % size (0.33 mm modules, 22.85 mm bars, 37.3 mm wide with its
  quiet zones), bars upright, printed on a label (the box: on its carton). The 95-module patterns are those of the
  12-digit numbers below, check digits included (made-up numbers in the in-store range); drawn flat from the task
  file's bar list, all four decode with OpenCV's `barcode` decoder.
  - spaghetti box (1 lb): 26.4 x 7.1 x 4.1 cm, 0.48 kg, lying on its broad face along x at (20, 49); code 400454264025
    on its left (+y) face, 7 cm from its middle away from the robot.
  - soup can (10.75 oz condensed, 211 x 400 can): 6.8 cm across the seams, 10.2 cm tall, 0.35 kg, standing at
    (4.5, 60.5); code 400107503051 on the white part of its label, facing back toward the robot (160 degrees from +x).
  - glass jar (16 oz salsa): 7.5 cm across, 12.7 cm tall with its 7 cm lug lid, 0.65 kg, standing at (15, 60.5);
    code 400454160013 facing left (+y).
  - water bottle (16.9 fl oz, 500 ml): 6.5 cm across, 20.4 cm tall with its cap, 0.52 kg, standing at (26, 60.5);
    code 400500169007 facing left (+y).
- Bagging shelf, fixed: brushed stainless flush with the counter, x -5 to 40, y -64.5 to -46; its inner edge (y -46)
  is marked with a 1.6 cm yellow line.

## The scanner model

The simulation does not render what the scanner's imagers see. A read is computed every control step (20 Hz) from the
code's pose and the arms' joints. A code is read when:

1. its centre is within 15 cm of a window: over the horizontal window (up to 2 cm past its edge) and at most 15 cm
   above it, or in front of the vertical window (up to 2 cm past its edges) and at most 15 cm from it;
2. it faces that window within 45 degrees (the code's outward normal within 45 degrees of straight down, or of the
   direction toward the tower);
3. it moves slower than 1.5 m/s (Zebra rates its single-window in-counter [SP72][sp72] for swipes up to 3.05 m/s);
4. no part of either arm (links, gripper body and fingers, as boxes fitted to the YAM's collision meshes and placed by
   the arm's joints) lies on the straight line from any of nine points spread over the code's bars to the window.

After a read, the same code reads again only once it has been out of view (failed the four tests) for 0.5 s, the
MP7000's default. The scanner's LED flashes bright green for 0.5 s on every read (the lit lens, a visual-only body, is
moved into place over the dim one). The reach of 15 cm and the 45 degrees are the task's numbers (the MP7000's sheet
gives none; Zebra lists 0 to 18.2 cm for a 100 % UPC-A on its single-window horizontal SP72); the real MP7000 also reads
some codes at steeper angles through its angled views, so the model asks for the barcode to face a window, as a cashier
presents it.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run.

- 0 for the rest of the run once the glass jar falls faster than 1 m/s (dropped from about 5 cm or more) or tumbles
  faster than 5 rad/s (knocked over), or any grocery falls off the counter.
- Otherwise 0.25 for each item checked out: read exactly once, and ending on the bagging shelf, its whole footprint
  past the yellow line and on the shelf, resting on it (lowest point within 1 cm of the counter), moved less than 3 mm
  in the last second, every finger 2 cm or more from it.
- An item read but left on the counter earns nothing: it has not reached the customer. An item read twice or more
  earns nothing: it was charged twice. An item that reaches the shelf before its first read scores 0 for the rest of
  the run: it left unpaid.
- Full score (1.0, success): all four checked out, held for 1 s.

Doing nothing scores 0: no code starts in view of a window and nothing starts on the shelf. In a real trial the reads
are the scanner's own: its beeps, and the decoded items on the POS (or in Zebra's 123Scan log), counted per item; a
photo of the shelf after the robot lets go gives the bagged items; a dropped jar is judged from the videos.
`-E object_jitter=0` keeps every object exactly where the task file puts it.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=scan_four_groceries_across_a_checkout_scanner.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Ring up the four groceries on the scanner, each item read once, and put them on the bagging shelf." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/scan-dream
```

The scripted demo, which reaches full score (the left gripper takes each grocery from the belt with its fingers on
two sides away from its code: the box 2.5 cm from its middle, on the side away from its code, and the 20 cm bottle from
the side, since the arms cannot reach over it; it draws the item clear of the others and turns it by a quarter or half
turn so that its code faces the vertical window, the box lifted above the tower to turn; it passes the item across 4
to 7 cm in front of the window at about 0.15 m/s, one read each with the LED flashing, and sets it on the counter just
past the scanner; the right gripper takes each from there round to the bagging shelf: the box along its outer edge,
then the jar, the can and the bottle in a row along its inner side):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scan_four_groceries_across_a_checkout_scanner.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Ring up the four groceries on the scanner, each item read once, and put them on the bagging shelf." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/scan-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/scan-dream/*.json; do inspect-robots video "$log"; done
```

`SCAN_DEBUG=1` prints, every 10 steps (`SCAN_DEBUG_EVERY=5` for every 5), each grocery's position, tilt, reads, which
window it is in view of and why it is or is not read (out of view, covered, too fast), how near the nearest finger
is, whether it is on the shelf and still, and whether it left unpaid; whether the LED is lit, the score and any broken
rule; and each gripper's centre and finger opening. It also prints a line for every read and every broken rule.

`SCAN_ONE=1` runs the demo the wrong way: the left hand alone, with the right arm left where it starts. It rings up
all four items exactly as the demo does (four reads, four LED flashes), but with nobody to pass them to it can only set
them down on the counter just past the scanner, their middles 3 to 17 cm to the right of the middle (pointing down,
its gripper reaches no further); the bagging shelf starts at 46 cm. The run scores 0: every item was read once, but
none reached the customer's side, and an item earns its 0.25 only once it is read and on the shelf. The right arm alone
also scores 0: its gripper centre stops at 45 cm to the left, short of every grocery's middle (49 to 60.5 cm), so it
cannot even take one off the belt.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell: the scripted demo is 1,751 steps (87.6 s of simulated time; the run ends once
success has held for 1 s). It took about 6 minutes at 320 x 176 without video with the GPU to itself (363 s), and about
17 minutes while rendering 1280 x 720 video with the GPU shared with another user's four simulations (996 s). A Dream-1
trial of 2,400 steps at 320 x 176 took about 5 minutes (295 s: 50 policy calls of 48 steps each, 0.79 s per call on
average). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, the scripted demo is deterministic and the scorer draws no
random numbers.

## Building it for real

A Zebra MP7000 short (or another bioptic scanner) set into a counter, connected to a POS or to a PC running Zebra's
123Scan, with its default same-symbol timeout (0.5 s); a strip of belt or a black rubber mat as the belt's end; a
stainless sheet or a marked area as the bagging shelf; and the four groceries (a 1 lb spaghetti box, a 10.75 oz can of
condensed soup, a 16 oz glass jar of salsa, a 16.9 fl oz bottle of water) with their own barcodes, set down on taped
marks with their barcodes turned away from the windows as in the screenshot. The decode log gives each item's reads;
a photo of the shelf, the bagged items; the videos, a drop.

[onet]: https://www.onetonline.org/link/summary/41-2011.00
[ig]: https://www.zebra.com/content/dam/zebra_new_ia/en-us/manuals/barcode-scanners/mp7x00-ig-en.pdf
[prg]: https://www.zebra.com/content/dam/support-dam/en/documentation/unrestricted/guide/product/mp7x00-prg-en.pdf
[spec]: https://www.barcodes.com.au/content/Zebra-MP7000-Specification-Sheet.pdf
[sp72]: https://www.zebra.com/gb/en/products/spec-sheets/scanners/multi-plane-scanners/sp72-series.html
