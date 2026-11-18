# Scan four groceries across a checkout scanner, one read each, and put them on the bagging shelf

A checkout counter on the station table. In the middle, standing on the counter with its window facing the robot, is
a hands-free presentation barcode scanner at true size: a Zebra DS9308-SR, the kind of scanner used at convenience
store, pharmacy and small-store checkouts (USD 410.19 with its USB cable; where to buy it is below). On the robot's
left is the end of the conveyor belt with four groceries on it: a 1 lb spaghetti box, a 10.75 oz can of condensed soup,
a 16 oz glass jar of salsa and a 16.9 fl oz bottle of water. Each carries a real 12-digit UPC-A barcode on one side, and
none starts with its barcode facing the scanner. On the robot's right is the bagging shelf. The instruction asks for
the four groceries to be rung up on the scanner, each item read once, and put on the bagging shelf; it does not say
how.

## Why it matters, and the procedure

Cashiers ([O*NET 41-2011.00][onet]; 3,106,300 US jobs in 2025) "Establish or identify prices of goods, services, or
admission, and tabulate bills, using calculators, cash registers, or optical price scanners", and "Bag, box, wrap, or
gift-wrap merchandise". The same hand skill is asked of every shopper at a self-checkout. At a presentation scanner
the cashier presents each item with its barcode toward the window, with no finger over the code, and listens for the
beep. Zebra's documents for the DS9308 give what the task checks:

- [DS9308 Product Reference Guide][prg] (MN-003532-08EN), p. 45, "Hands-free Scanning": "The scanner is in hands-free
  (presentation) mode when it sits on a counter top [...] In this mode, the scanner operates in continuous (constant-on)
  mode, where it automatically decodes a barcode presented in its field of view. In hands-free mode the LED indicator
  turns solid green." Table 3 (p. 42), "A barcode was successfully decoded": "Medium beep (or as configured)",
  "Hands-free - solid green LED; turns off after decode."
- The same guide, p. 80, "Timeout Between Decodes, Same Symbol": "Use this option in presentation mode or Continuous
  Barcode Read mode to prevent the scanner from continuously decoding the same barcode when it is left in the
  scanner's field of view. The barcode must be out of the field of view for the timeout period before the scanner reads
  the same consecutive symbol. [...] The default interval is 0.5 seconds." Table 4 (p. 47), decode ranges: "13 mil
  (100%) UPC 0 in. / 0 cm 8.8 in. / 22.4 cm"; Table 6 (p. 55-56) lists the size, weight, field of view and
  tolerances quoted from the specification sheet below.
- [DS9300 Series specification sheet][spec]: 14.5 cm H x 8.6 cm W x 8.3 cm D, 318 g; imager field of view "52° H x
  33° V nominal"; skew and pitch tolerance "+/-60°", roll "360°"; swipe speed "Up to 120 in./305 cm. per second for 13
  mil UPC in optimized mode"; user indicators "Good decode LEDs, rear view LEDs, beeper (Adjustable tone and volume)";
  "The flexible ratcheting base offers four positions, from 15° back to 45° forward."

So an item must be read exactly once: an item put in the bag unread leaves the store unpaid, and an item presented
again after its barcode has been out of view for 0.5 s is rung up twice and overcharges the customer.

How this differs from turning parcels' labels toward a fixed scanner on a moving line: there the cartons stay on their
carriers and are only yawed, without being picked up. Here each item is picked up and turned in the hand (by a quarter
or half turn), the grip has to be chosen so that no finger covers the code (the box is held 2.5 cm from its middle, on
the side away from its code), the code is passed in front of the window within reading range and the reads are counted
(none missed, none doubled), and the item is handed on and set down; the four items are different real shapes (a long
box, a squat can, a heavy glass jar, a tall bottle).

## The scanner, and where to get it

The scanner is an off-the-shelf Zebra DS9308-SR (standard range). Its USB kit, the scanner with its shielded USB cable
(part DS9308-SR4U2100AZW), costs USD 410.19 at [Barcode Factory][buy], a US reseller (seen on 17 November 2026; MSRP
USD 569.71; the scanner alone, part DS9308-SR00004ZZWW, USD 384.97). Zebra's locking-mount bracket that holds
it on the counter (BRKT-LM0093C-04) costs USD 14.09 there, so the scanner set-up costs about USD 425. It plugs into any
PC by USB as a keyboard and needs nothing else.

It is a checkout scanner, used the way cashiers use it. Zebra's specification sheet lists retail point of sale first
among the DS9300 Series' uses, "From convenience and drug stores to quick serve restaurants and other retail
environments", and its locking mount for "self checkout". Supermarket lanes also use larger in-counter scanners with two
windows (bioptic scanners), which cost much more. The presentation scanner keeps what this task tests and how it is
scored:

- The hand skill is the same. Each item is picked up and turned in the hand until its code faces the window, held with
  no finger over the code, presented once within reading range, handed on and set down on the bagging shelf. With one
  window, a code has to face that window within 60 degrees, and inside its field of view no code facing left (as three
  of them start) can: every item has to be turned to be read.
- The score is the same. An item counts once it is read exactly once and rests, released, on the bagging shelf; a
  second read after the 0.5 s same-symbol timeout is a double charge, and an item that reaches the shelf unread has
  left unpaid. In a real trial every read is the scanner's own: one beep, the LED going out, one decode in the log.
- The read model uses the DS9308's own numbers (field of view, range, skew and pitch tolerance, swipe speed,
  same-symbol timeout, LED), from its specification sheet and guide, so the simulated reads follow the scanner that a
  real trial uses.

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

- Scanner, fixed to the counter (in Zebra's locking-mount bracket, or its multi-mount bracket on the bracket's adhesive
  pad, both in Zebra's guide): a Zebra DS9308-SR, 14.5 cm tall, 8.6 cm wide and 8.3 cm deep, 318 g, standing upright
  (its base set level, within its 15 degrees back to 45 degrees forward), its oval base centred at x 37.3, y 15. Its
  head's front face, at x 33.5, carries the window (about 6.4 x 5.0 cm, its middle 10 cm above the counter, at y 15),
  facing the robot. The LED lens is on the head's top, at its front; it is solid green while the scanner is ready and
  goes out for 0.5 s after each read.
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

The simulation does not render what the scanner's imager sees. A read is computed every control step (20 Hz) from the
code's pose and the arms' joints, with the DS9308's own numbers. A code is read when:

1. its centre is in the imager's field of view: inside the 52 x 33 degree pyramid that opens toward the robot from the
   window's middle (x 33.5, y 15, 10 cm up), and 0 to 22.4 cm in front of the window (its range for a 100 % UPC-A);
2. it faces the window within 60 degrees (the angle between the code's outward normal and its line to the window's
   middle; the scanner's skew and pitch tolerance);
3. it moves slower than 3.05 m/s (the scanner's swipe speed for a 13 mil UPC);
4. no part of either arm (links, gripper body and fingers, as boxes fitted to the YAM's collision meshes and placed by
   the arm's joints) lies on the straight line from any of nine points spread over the code's bars to the window's
   middle.

After a read, the same code reads again only once it has been out of view (failed the four tests) for 0.5 s, the
DS9308's default. The scanner's solid green LED goes out for 0.5 s on every read, as in hands-free mode (Zebra gives
no duration; the lit lens, a visual-only body, is moved away and back; the score does not use it). The beep is not
modelled.

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

Doing nothing scores 0: no code starts in the scanner's view and nothing starts on the shelf. In a real trial the reads
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
turn so that its code faces the scanner, the box lifted to turn; it raises the item until its code is level with the
window and passes it across 14 to 17 cm in front of the window at about 0.2 m/s, one read each with the LED going out,
and sets it on the counter past the scanner; the right gripper takes each from there round to the bagging shelf: the
box along its outer edge, then the jar, the can and the bottle in a row along its inner side):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scan_four_groceries_across_a_checkout_scanner.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Ring up the four groceries on the scanner, each item read once, and put them on the bagging shelf." \
  --scorer task_score --max-steps 1900 --store-frames --log-dir ~/yam-eval/scan-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/scan-dream/*.json; do inspect-robots video "$log"; done
```

`SCAN_DEBUG=1` prints, every 10 steps (`SCAN_DEBUG_EVERY=5` for every 5), each grocery's position, tilt, reads,
whether it is in the scanner's view and why it is or is not read (out of view, covered, too fast), how near the
nearest finger is, whether it is on the shelf and still, and whether it left unpaid; whether the LED is lit, the score
and any broken rule; and each gripper's centre and finger opening. It also prints a line for every read and every
broken rule.

`SCAN_ONE=1` runs the demo the wrong way: the left hand alone, with the right arm left where it starts. It rings up
all four items exactly as the demo does (four reads, the LED going out four times), but with nobody to pass them to it
can only set them down on the counter just past the scanner, their middles 3 to 15 cm to the right of the middle
(pointing down, its gripper reaches no further); the bagging shelf starts at 46 cm. The run scores 0: every item was
read once, but none reached the customer's side, and an item earns its 0.25 only once it is read and on the shelf. The
right arm alone also scores 0: its gripper centre stops at 45 cm to the left, short of every grocery's middle (49 to
60.5 cm), so it cannot even take one off the belt.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell: the scripted demo is 1,784 steps (89.2 s of simulated time; the run ends once
success has held for 1 s). It took about 4.5 minutes while rendering 1280 x 720 video (258 s), and about 7 minutes at
320 x 176 without video with two other simulations on the same GPU (397 s). A Dream-1 trial of 2,400 steps at
320 x 176 took about 6 minutes (356 s: 50 policy calls of 48 steps each, 0.67 s per call on average). Isaac Sim's
first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, the scripted demo is deterministic and the scorer draws no
random numbers.

## Building it for real

A Zebra DS9308-SR USB kit (DS9308-SR4U2100AZW, about USD 410) on the counter in Zebra's locking-mount bracket
(BRKT-LM0093C-04, or the multi-mount bracket BRKT-MM0093C-04 on its adhesive pad), upright, in its default hands-free
mode with the default same-symbol timeout (0.5 s), plugged into a PC by USB, where each decode types the item's number
into a text file (or Zebra's free 123Scan shows the decodes); a strip of belt or a black rubber mat as the belt's end; a
stainless sheet or a marked area as the bagging shelf; and the four groceries (a 1 lb spaghetti box, a 10.75 oz can of
condensed soup, a 16 oz glass jar of salsa, a 16.9 fl oz bottle of water) with their own barcodes, set down on taped
marks with their barcodes turned away from the scanner as in the screenshot. The decode log gives each item's reads; a
photo of the shelf, the bagged items; the videos, a drop.

[onet]: https://www.onetonline.org/link/summary/41-2011.00
[prg]: https://www.zebra.com/content/dam/support-dam/en/documentation/unrestricted/guide/product/ds9308-prg-en.pdf
[spec]: https://www.barcodes.com.au/content/Zebra-DS9300-series-spec-sheet-barcodes.com.au.pdf
[buy]: https://www.barcodefactory.com/zebra/scanners/ds9308/ds9308-sr4u2100azw
