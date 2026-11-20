# Deal four cards face down from a held deck: one hand holds it, the other slides only the top card off onto each place

Dealing cards is one of the most familiar table skills: anyone who has played a card game has dealt or watched it. A
standard 52-card deck lies face down on a green felt card mat with four card places marked on it in white. The
instruction asks for one card to be dealt face down onto each place and the deck then put down on the mat; it does not
say how.

What makes it hard is the hand skill, and it is visible in the scene. A playing card is 0.3 mm thick: a gripper cannot
pinch the top card off a deck. It can only be moved by friction, a fingertip pressed on its back and drawn sideways,
and it must come off alone (not with the card below), cross the deck's edge, drop to the felt and lie flat inside its
place without turning over. The card must be laid at four places in turn, so one hand picks the deck up by its long edges
and holds it level and low beside each place while the other hand deals. The two hands also work together at the end
of each card: once the pushed card has tipped over the deck's edge onto the felt, the holding hand draws the deck back
from under its trailing end, so the card drops flat where it is.

The task is the task file [`deal_four_cards_from_a_held_deck.py`](deal_four_cards_from_a_held_deck.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters

Every card game starts with a deal. The way it is done is the same everywhere: the dealer holds the pack face down in
one hand and takes cards off its top with the other ([pagat.com, Card Game Mechanics][pagat]: "This is done by dealer
holding the pack, face-down, in one hand"; "removing cards from the top of it with her other hand to distribute to the
players"; "placing them face-down on the table in front of the players to whom they are dealt"; "Unless the rules
specify otherwise, assume that the cards are dealt one at a time."). Someone watching a ten-second clip knows at once
what the robot is doing, and why it is hard: a thin, slippery card has to be separated from the one under it by touch
alone. The same skill (sliding one sheet off a stack by friction) is what a person uses to take one banknote, one
sheet of paper or one card off a pile.

The cards are Bicycle Standard (poker size) playing cards: 2.5 x 3.5 in (63.5 x 88.9 mm), about 0.30 mm thick, 52 to
a deck (15.6 mm), about 1.8 g each.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the mat, the deck and the cards are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The deck (a block for the lower cards with a red back
on top), the six separate top cards (a white card with a red back printed on it) and the mat with its printed places
are built with `bodykit.py`, which adds such multi-shape bodies: importing it hooks `yam_arena.scene` when the
simulator starts, without changing yam_arena, and gives the task file each body's pose every control step.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top.

- Card mat, fixed (taped down): green felt (a smooth card-table cloth), 26 x 34 cm, 3 mm thick, centred at x 21.0,
  y -1.5. Four card places printed in white, each the outline of a card (63.5 x 88.9 mm, long side along y, 2 mm
  lines just outside it), centred at A (16.5, -9.0), B (25.5, -9.0), C (16.5, 1.0) and D (25.5, 1.0).
- Deck of 52 Bicycle Standard cards, loose, face down (red backs up), long side along y, centred at (21.0, 11.5) on the
  mat, square (its left end 4 mm past the mat's edge). In the simulation the lower 46 cards are one rigid block (63.5 x 88.9 x 9.6 mm, 82.8 g) and the top six
  are separate cards (1.8 g each); see "What is modelled".
- Both arms start at the ready pose, grippers open. The layout is the same in every trial.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the pose of the deck and of each separate card, and the points of each finger's
collision hulls.

- A card is in a place when half or more of its area lies inside the place's outline and it is down on the mat (its
  lowest point within 8 mm of the felt, not on the deck held above it). A place is dealt when exactly one card is in
  it, and that card has at least 80 % of its area inside the outline, lies face down (back up, tilted under 15 deg)
  and lies on the felt (its underside within 2 mm of it). Two or more cards in a place: that place scores 0.
- Score: 0.25 for each dealt place; while the deck is not set down and let go, at most 0.9.
- 1.0 (success, held 1 s): all four places dealt; the deck's lower block resting flat on the mat (underside within
  3 mm of it, tilted under 10 deg); every fingertip 15 mm or more from the deck and from the dealt cards; and the deck
  and the dealt cards still (each moving under 1 mm in 1 s).

Doing nothing scores 0. A real trial is scored from a photo taken from above at the end and the trial video: for each
place, the cards lying in it are counted (a card overlapping a place by half or more counts), and the one card is
checked against the outline with a transparent overlay marked at 80 % of the card's area (a card shifted up to 12 mm
across or 17 mm along the place still passes), face down; the deck on the mat, let go, is checked on the video. The
deck is counted afterwards: 48 cards left.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=deal_four_cards_from_a_held_deck.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Deal one card face down from the deck onto each of the four card places on the felt mat, then put the deck down on the mat." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/deal-dream
```

The scripted demo, which reaches full score (open loop from the known layout). The left hand comes down straight over
the deck's left end (fingers closing across the deck), its pads overlapping the deck's last 50 mm, closes on the long
edges, lifts the deck 6 mm off the mat and carries it, level, to beside each place in turn (A and B first, the deck
held over the still empty places C and D; then C and D), its right edge about 39 mm left of the place's middle (a
tipped card's middle lands about that far past the edge; each place has a small correction found by trial, up to 14 mm
along y and 9 mm along x, since the arms' give and the card's start on the deck differ a little between places). At each, the
right hand, tilted 55 deg so its wrist stays out to the right, away from the holding hand, deals the top card in three
strokes: each comes down on the card's back 22 mm inside the deck's edge, already moving right, presses lightly
(commanded 2.5 mm into it) and pushes it 20 mm to the right, then lifts. The fingertip never presses on the part of the
card that already overhangs the edge: pressed there, the card levers about the edge and its far end flips up. During
the third stroke the card, more than half out, tips over the edge, its leading end onto the felt and its trailing end
still on the deck. The fingertip rises clear and the left hand draws the deck 50 mm back (left) and 2 mm down from
under the card's trailing end, so the card drops flat into its place. A fingertip holding the tipped card down while
the deck is drawn back drags the next card off the deck with it (tried; it then lands too far on the next deal). After
the four cards, the left hand sets the deck down where it was, lets go and rises clear.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=deal_four_cards_from_a_held_deck.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Deal one card face down from the deck onto each of the four card places on the felt mat, then put the deck down on the mat." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/deal-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/deal-dream/*.json; do inspect-robots video "$log"; done
```

`DEAL_DEBUG=1` prints, every 10 steps (`DEAL_DEBUG_EVERY=1` for every step), where the deck is (and its height and
tilt), whether it is down on the mat, the nearest fingertip's distance from the deck and the dealt cards, whether all
is still, for each place the number of cards in it (G when it is dealt, with the dealt card's fraction inside), each
separate card's position, height and tilt, and where each gripper is.

`DEAL_MODE=press` runs the demo the wrong way: the same motions, pressing the dealing fingertip 14 mm into the card
instead of 2.5 mm. Pressed that hard, the fingertip pushes the held deck down onto the felt and tilts it 2 to 4 deg in
the holding hand, and the strokes drag the top five cards out together, fanned out, instead of one; none tips off at
place A, and they come back with the deck when it is drawn back. At place B three cards slide off together, two of
them in the place; later strokes scatter the rest. No place ends with exactly one card in it: score 0.

## What is modelled and what is not

- The cards are rigid: they do not bend, so the small bow a real card takes under a thumb is not there. A real card is
  0.30 mm thick; the separate top cards are 1.0 mm thick in the simulation (thinner cards jitter in a stack) at their
  true mass (1.8 g), and the lower 46 cards are one rigid block (9.6 mm, 82.8 g), so the deck is 15.6 mm tall as a real
  one. Only the top six cards can slide; four are dealt.
- The separate top cards are 4 mm narrower (59.5 mm) than the block (63.5 mm), so the holding jaws squeeze only the
  block: with the cards as wide as the block, the jaws' contact with the card edges made the top cards creep along the
  deck while it was held. A hand holding a deck squeezes mostly the lower cards in the same way.
- Friction (static / sliding, estimates; PhysX averages the two materials): card on card 0.20 / 0.16 (coated playing
  cards slide easily), card on the felt cloth 0.25 / 0.21, the robot's fingertip on a card 0.35 / 0.33 (its pads use
  PhysX's default 0.5), the holding jaws on the deck's edges 0.55 / 0.53.
- The felt mat is a rigid 3 mm sheet fixed to the table, its places printed outlines.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with several other simulation jobs, the scripted demo with 640 x 360
cameras (frames stored, videos rendered) took 13 minutes and ended on success at step 2,207 (110 s of simulated time;
the deck is let go at 109 s). Debug runs of the demo at 320 x 176 took 15 to 20 minutes beside two or three other runs,
and the Dream-1 trial of 3,000 steps at 320 x 176 about 15 minutes. Isaac Sim's first start on a new machine adds a few
minutes while it compiles its shaders. Stored frames take about 1.2 GB per 2,400-step run at 320 x 176.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A deck of Bicycle Standard playing cards (poker size, about 4 USD); a green felt or speed-cloth card mat at least
26 x 34 cm (or a sheet of felt), taped to the table; the four card places drawn on it in white (a paint pen, or white
tape 2 mm wide) as 63.5 x 88.9 mm outlines at the positions above; tape marks for the deck's corners. For each trial
put the four dealt cards back on top of the deck face down, square it and lay it at its marks.

[pagat]: https://www.pagat.com/mech.html
