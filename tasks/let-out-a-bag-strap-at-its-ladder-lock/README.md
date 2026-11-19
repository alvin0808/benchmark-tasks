# Let a backpack's shoulder strap out at its ladder lock, one arm holding the buckle tipped up

An empty school daypack lies face down on the table, its back panel and shoulder straps up. Each shoulder strap is a
padded strap sewn to the top of the back panel; at its lower end a short webbing tab is sewn round the end bar of a
black 25 mm ladder-lock buckle. A 25 mm nylon webbing sewn to the panel's bottom corner runs up the panel to that
buckle, goes through it and over its centre bar, and comes back out under the buckle's lift tab; its loose end (the
tail, its tip folded and stitched) lies back down along the webbing, 15 cm of it past the buckle. The instruction
asks for the near shoulder strap (the one nearer the robot) to be lengthened until only 7 cm of its loose end sticks
out past the buckle, with the buckle left holding; it does not say how.

A ladder lock lets webbing run through it in the lengthening direction only while its tab is lifted. A buckle
maker, Country Brook Design ([1 1/2 inch ladderlock buckles][cbd]): "Created to hold under tension, then release by
lifting the tab." Trivantage's guide to webbing hardware ([Fastener Physics 101][triv]) gives the threading the
buckle has: "Thread the webbing up through the bottom slot. Bring it over the center bar. Feed it back down through
the top slot." and "Pull the free end to tighten." So the strap is lengthened with two hands: one holds the buckle
tipped up off the webbing (let go, it lies back down flat and grips), the other draws the webbing through it from the
bag side until the loose end is as short as wanted; then the buckle is laid flat again so that it holds. Pulled
without the buckle lifted, the strap does not give and the backpack is dragged along the table. Both reasons are in
the scene: the buckle is a small part on the strap that springs back flat, and the webbing to be drawn runs away from
it toward the bag's corner, 15 to 25 cm from it.

The task is the task file [`let_out_a_bag_strap_at_its_ladder_lock.py`](let_out_a_bag_strap_at_its_ladder_lock.py)
for the `yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) and
[`strapkit.py`](strapkit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac
Lab-Arena 0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable,
installs everything, including the Dream-1 policy. No third-party models or assets are used: every object is built in
the task file from boxes and cylinders.

`yam_arena` task files place single boxes, cylinders and spheres and cannot join them or make anything flexible.
`strapkit.py` builds the backpack, its near strap's buckle and webbing as one PhysX articulation: rigid parts (each
made of several shapes, with `bodykit.py`'s shape classes) on revolute, prismatic and fixed joints, with joint springs
and Coulomb friction the task file changes during the run, a loop-closing joint (the webbing is sewn to the bag at its
end) and collision filters. Importing it hooks `yam_arena.scene` when the simulator starts, without changing
yam_arena, and gives `score()` every joint's position and every part's pose.

What is modelled and what is not: the backpack (30 x 44 x 8.5 cm lying face down, 0.45 kg, rigid, loose on the table),
the buckle (35 x 26 mm, 6.5 mm tall, 5 g) and the webbing (25 mm wide, 1.6 mm thick, 18 g/m) are at true size. The
webbing is a chain of rigid 15 mm links on springy hinges across its width (0.006 N m per radian each): it bends and
drapes, but cannot twist or bend sideways. Its wrap over the buckle's centre bar is not simulated: under the buckle
the strap's webbing runs on into the padded strap (out of sight) to a hidden carrier on a slide along the strap, and
the tail's webbing does the same to a second carrier, which is sprung (2,000 N/m) to follow the first: every control
step its rest point is set to go in as far as the strap's carrier has come out, so webbing drawn out on the bag side
comes off the tail. The buckle tips about its centre bar, and its grip is a rule on the strap's slide, set every
control step from the buckle's angle: within 30 degrees of the webbing it grips (300 N of friction on the slide, more
than the arms can pull, and a 2,000 N/m spring holding the slide where it was gripped, since PhysX lets a slide on
friction alone creep); past 50 degrees the webbing runs with 1.5 N of friction; in between, the friction falls from
300 N to 1.5 N as the tab rises. A real ladder lock is shortened by pulling the loose end; here pulling the loose end
does not shorten the strap (the tail only follows the strap's side), which lengthening it does not need. The buckle
returns flat on a spring (0.03 N m per radian), standing for the bent webbing that pushes a real buckle back down. The
backpack's fabric, foam and contents are rigid.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. How far
the strap has been let out is how far the strap's slide has moved (the tail is 15 cm minus that past the buckle's tab
edge). The backpack pulled off the table (its middle 3 cm or more below where it lay) sets the score to 0 for the rest
of the run; sliding it on the table costs nothing. Otherwise the score is the best stage reached: 0.25 the strap let out by 2 cm or more; 0.5 let out to the target, 7 to 9 cm (its
loose end 6 to 8 cm past the buckle); 0.75 at the target with the buckle laid back flat (within 15 degrees of the
webbing), so that it grips. Full score: at the target, the buckle flat, both grippers 5 cm or more from the buckle and
from every link of the webbing in sight, everything still for 1 s. `-E object_jitter=0` keeps every object exactly
where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=let_out_a_bag_strap_at_its_ladder_lock.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Lengthen the near shoulder strap of the backpack until its loose end sticks out only 7 cm past the buckle, and leave the buckle holding." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/strap-dream
```

The scripted demo, which reaches full score (the left gripper takes the buckle by its two side rails at the tab end
and tips it up about its centre bar, turning with it; the right gripper pinches the webbing by its edges 6 cm from the
sewn corner and draws it toward the corner and up, so that 8 cm of webbing runs through the buckle and the tail
shortens by as much; the left lays the buckle flat again; both let go and move away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=let_out_a_bag_strap_at_its_ladder_lock.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Lengthen the near shoulder strap of the backpack until its loose end sticks out only 7 cm past the buckle, and leave the buckle holding." \
  --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/strap-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/strap-dream/*.json; do inspect-robots video "$log"; done
```

`STRAP_DEBUG=1` prints, every 10 steps (`STRAP_DEBUG_EVERY=5` for every 5), how far the strap has been let out and
the tail's length, the buckle's angle, the slide's friction, how far the backpack has moved, how close the nearest
gripper is to the buckle and the webbing, whether everything is still, the stage reached, a few webbing links'
positions and where each gripper is.

`STRAP_ONE=1` runs the demo the wrong way, with one arm: the left arm stays put and the right gripper pinches the
webbing and pulls it as in the demo, with the buckle lying flat. The buckle grips, the strap does not give, and the
pull drags the backpack across the table without letting the strap out, so no stage is reached and the score is 0.
Dragged on past about 5 cm this way, the strap's simulation can break down (its poses become NaN); a broken state
never earns a stage, so the score stays at the best stage reached before it.

## Runtime per trial

On one RTX PRO 6000 Blackwell, on a shared machine: about 8 minutes for the scripted demo while rendering 1280 x 720
video (325 steps, 16 s of simulated time; 483 s, with a Dream-1 run on the same GPU), and about 40 minutes for a
Dream-1 trial of 2,400 steps at 320 x 176 (2,368 s, the GPU shared with other runs). Isaac Sim's first start on a new
machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

An empty school daypack whose shoulder straps adjust at 25 mm (1 inch) ladder locks at the lower ends of the padded
straps, laid face down with its bottom toward the robot's right as in the screenshot, and four
strips of tape on the table at its corners. Before each trial, pull the near strap's loose end until 15 cm of it
sticks out past the buckle's tab edge and lay the webbing straight along the panel with the tail on top of it. Score
with a tape measure (the loose end from the buckle's tab edge to its tip), a photo (the buckle lying flat on the
webbing; the backpack still on the table) and the trial's video.

[cbd]: https://countrybrookdesign.com/triglides-tension-lock-1-1-2-inch.html
[triv]: https://www.trivantage.com/blog/products/fastener-physics-101-cam-buckles-ladderloc-webbing
